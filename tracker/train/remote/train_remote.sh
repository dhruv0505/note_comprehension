#!/usr/bin/env bash
# train_remote.sh — the training run on a Jarvis Labs GPU VM, from the trainer bundle
# unpacked into ~/train/code. Each stage logs to logs/<stage>.log and records its start
# and end in stages/<stage>.json (train_status.py reads both). A stage already done is
# skipped, so the script can be started again after an interruption.
#
#   DATA_HOST=10.0.0.2 DATA_USER=ubuntu NPROC=2 bash ~/train/code/remote/train_remote.sh
#
# Stages: setup_env -> fetch_dataset -> fetch_model -> smoke -> train
#   setup_env      a venv with torch for the driver's CUDA, transformers 4.49, peft 0.14
#   fetch_dataset  train/val samples, val and test pages, and the train textline and formula recognition
#                  samples when the dataset has them, from the dataset VM over its private IP (marks, usage
#                  and val/test recognition stay behind: training does not read them). Recognition is checked:
#                  rows as reported, its prompts, newlines between rows
#   train/smoke    train.py reads the train samples and train recognition as one dataset, shuffled together
#   fetch_model    Florence-2-large into ~/train/hf_cache, loaded once on the GPU
#   smoke          6 steps on every GPU with a tiny validation, then thrown away (same LoRA rank and
#                  starting adapter as the run)
#   train          torchrun train.py with the arguments in ~/train/train_args.json; a run trained further
#                  from a saved model has its adapter in ~/train/init (--init-adapter)
# Local test mode (LOCAL_TEST=1): no setup, dataset copied from DATA_DIR, python in $PY.
set -uo pipefail
ROOT="$HOME/train"
CODE="$ROOT/code"
mkdir -p "$ROOT/logs" "$ROOT/stages" "$ROOT/data" "$ROOT/run"
# one runner at a time: orchestrator.py starts this again when its call is cut off mid-launch, and two of them would
# fight over the same stages and checkpoints
exec 9>"$ROOT/.runner.lock"
if ! flock -n 9; then
  echo "another train_remote.sh is already running: leaving it to it"
  exit 0
fi
NPROC=${NPROC:-1}
LOCAL_TEST=${LOCAL_TEST:-0}
export HF_HOME="${HF_HOME:-$ROOT/hf_cache}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
if [ "$LOCAL_TEST" = 1 ]; then
  PY=${PY:?LOCAL_TEST needs PY}
else
  PY="$ROOT/venv/bin/python"
fi

mark() {    # mark start|done|failed STAGE
  "${PY_MARK:-python3}" - "$ROOT/stages/$2.json" "$1" <<'PYEOF'
import json, os, sys, time
path, action = sys.argv[1], sys.argv[2]
try:
    state = json.load(open(path))
except (OSError, ValueError):
    state = {}
if action == "start":
    state = {"start": time.time(), "state": "running"}
else:
    state.update(end=time.time(), state=action)
with open(path + ".tmp", "w") as f:
    json.dump(state, f)
os.replace(path + ".tmp", path)
PYEOF
}
[ "$LOCAL_TEST" = 1 ] && PY_MARK="$PY"

run() {     # run STAGE COMMAND...
  local name=$1; shift
  if [ -f "$ROOT/stages/$name.json" ] && grep -q '"state": "done"' "$ROOT/stages/$name.json"; then
    return 0
  fi
  mark start "$name"
  if "$@" > "$ROOT/logs/$name.log" 2>&1; then
    mark done "$name"
  else
    mark failed "$name"
    return 1
  fi
}

launcher() {  # the command that starts one process per GPU
  if [ "$NPROC" -gt 1 ]; then
    echo "$ROOT/venv/bin/torchrun --standalone --nproc_per_node $NPROC"
  else
    echo "$PY"
  fi
}

setup_env() {
  nvidia-smi || { echo "no NVIDIA driver on this VM"; return 1; }
  local cuda
  cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1\2/p' | head -1)
  local index=cu118
  if [ "${cuda:-0}" -ge 124 ]; then index=cu124; elif [ "${cuda:-0}" -ge 121 ]; then index=cu121; fi
  echo "driver CUDA ${cuda:-unknown} -> torch wheels $index"
  if ! python3 -m venv "$ROOT/venv" 2>/dev/null; then
    sudo apt-get update && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-pip
    python3 -m venv "$ROOT/venv" || return 1
  fi
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q "torch==2.5.1" --index-url "https://download.pytorch.org/whl/$index" || return 1
  "$PY" -m pip install -q "transformers==4.49.0" "peft==0.14.0" timm einops numpy pillow safetensors || return 1
  "$PY" - "$NPROC" <<'PYEOF'
import sys, torch, transformers, peft
need = int(sys.argv[1])
count = torch.cuda.device_count()
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", count,
      [torch.cuda.get_device_name(i) for i in range(count)], "transformers", transformers.__version__, "peft", peft.__version__)
if count < need:
    sys.exit(f"{count} GPUs visible, {need} needed")
PYEOF
}

fetch_dataset() {
  local parts="train/samples val/samples val/pages test/pages train/report.json val/report.json test/report.json"
  # train/recognition joins them when the dataset has it: trained on with the samples, as one shuffled dataset
  # ... and train/recognition_pool/textline, the textline samples from the whole train pool
  # ... and train/geometry, the geometry figures drawn without a page (an additions dataset run)
  local extra='$(for task in textline formula; do [ -f train/recognition/$task/manifest.jsonl ] && echo train/recognition/$task; done; [ -f train/recognition_pool/textline/manifest.jsonl ] && echo train/recognition_pool/textline; [ -f train/geometry/manifest.jsonl ] && echo train/geometry)'
  if [ "$LOCAL_TEST" = 1 ]; then
    (cd "${DATA_DIR:?}" && eval "tar cf - --exclude=shards $parts $extra") | tar xf - -C "$ROOT/data" || return 1
  else
    ssh -i "$HOME/.ssh/datavm" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 \
      "${DATA_USER:-ubuntu}@${DATA_HOST:?}" "cd ~/run/dataset && tar cf - --exclude=shards $parts $extra" \
      | tar xf - -C "$ROOT/data" || return 1
  fi
  # the checks fail the stage: a dataset they refuse is not trained on
  "$PY" - "$ROOT/data" "$CODE" <<'PYEOF' || return 1
import json, os, sys
from collections import Counter
sys.path.insert(0, sys.argv[2])
import recog_layout as rl
from seg_eval import ARROW_PROMPT, LEVEL1_PROMPTS, LEVEL2_PROMPT
root = sys.argv[1]
# recognition samples, when copied: every row as its report counts them, with its task's prompt, newline-separated
# geometry figures without a page, when copied: one level-1 geometry prompt per row, every image present
geometry = os.path.join(root, "train", "geometry")
if os.path.exists(os.path.join(geometry, "manifest.jsonl")):
    report = json.load(open(os.path.join(geometry, "report.json")))
    rows = [json.loads(line) for line in open(os.path.join(geometry, "manifest.jsonl"), encoding="utf-8") if line.strip()]
    if len(rows) != report["samples"]:
        sys.exit(f"train geometry: {len(rows)} manifest rows, report says {report['samples']}")
    prompts = {r["prompt"] for r in rows}
    if prompts != {report["prompt"]} or any(r.get("task") for r in rows):
        sys.exit(f"train geometry: rows with another prompt than {report['prompt']!r}: {sorted(prompts)[:3]}")
    missing = [r["image"] for r in rows if not os.path.exists(os.path.join(geometry, r["image"]))]
    if missing:
        sys.exit(f"train geometry: {len(missing)} images missing, e.g. {missing[:3]}")
    print(f"train geometry: {len(rows):,} samples without a page")
for sub, task in (("recognition", "textline"), ("recognition", "formula"), ("recognition_pool", "textline")):
    folder = os.path.join(root, "train", sub, task)
    name = f"train {task} {sub.replace('_', ' ')}"
    if not os.path.exists(os.path.join(folder, "manifest.jsonl")):
        continue
    report = json.load(open(os.path.join(folder, "report.json")))
    rows = [json.loads(line) for line in open(os.path.join(folder, "manifest.jsonl"), encoding="utf-8") if line.strip()]
    if len(rows) != report["samples"]:
        sys.exit(f"{name}: {len(rows)} manifest rows, report says {report['samples']}")
    if report.get("target_separator") != rl.SEP or any("<sep>" in r["target"] for r in rows):
        sys.exit(f"{name} was made with <sep> between rows: regenerate the dataset with Recognition on")
    if {r["prompt"] for r in rows} != {rl.PROMPTS[task]} or any(r.get("task") != task for r in rows):
        sys.exit(f"{name}: rows with another prompt or task than {rl.PROMPTS[task]!r}")
    missing = [r["image"] for r in rows if not os.path.exists(os.path.join(folder, r["image"]))]
    if missing:
        sys.exit(f"{name}: {len(missing)} images missing, e.g. {missing[:3]}")
    sizes = Counter(len(r["target"].split(rl.SEP)) for r in rows)
    print(f"{name}: {len(rows):,} samples, {sum(k * v for k, v in sizes.items()):,} rows, "
          f"{sum(v for k, v in sizes.items() if k < rl.ROWS):,} with fewer than {rl.ROWS}")
PROMPTS = [LEVEL2_PROMPT, ARROW_PROMPT, *LEVEL1_PROMPTS.values()]
root = sys.argv[1]
for split in ("train", "val"):
    report = json.load(open(os.path.join(root, split, "report.json")))
    prompts = Counter()
    with open(os.path.join(root, split, "samples", "manifest.jsonl")) as f:
        for line in f:
            if line.strip():
                prompts[json.loads(line)["prompt"]] += 1
    rows = sum(prompts.values())
    if rows != report["rows"]:
        sys.exit(f"{split}: {rows} manifest rows, report says {report['rows']}")
    print(f"{split}: {rows:,} rows, {report['samples']:,} images")
    for prompt in PROMPTS:
        print(f"  {prompts[prompt]:>8,}  {prompt}")
    missing = [p for p in PROMPTS if not prompts[p]]
    unknown = sorted(set(prompts) - set(PROMPTS))
    if missing or unknown:
        sys.exit(f"{split}: prompts missing {missing}, prompts the cascade never asks {unknown}")
for split in ("val", "test"):
    pages = [n for n in os.listdir(os.path.join(root, split, "pages")) if n.endswith(".json")]
    report = json.load(open(os.path.join(root, split, "report.json")))
    if len(pages) != report["pages"]:
        sys.exit(f"{split}: {len(pages)} page files, report says {report['pages']}")
    print(f"{split}: {len(pages):,} pages")
PYEOF
  du -sh "$ROOT/data"
}

fetch_model() {
  cd "$CODE" && "$PY" - <<'PYEOF'
import torch
from model import build_model, report_parameters
from seg_dataset import load_processor
load_processor()
model, _, _ = build_model()
report_parameters(model)
print("loaded on", next(model.parameters()).device)
PYEOF
}

common_args() {   # with the mark separation model the run was given, recognition in page validation takes marks out
  # the train samples and train recognition are one dataset, shuffled together by train.py; val rows stay segmentation
  local recog="" task
  for task in textline formula; do
    [ -f "$ROOT/data/train/recognition/$task/manifest.jsonl" ] && recog="$recog $ROOT/data/train/recognition/$task/manifest.jsonl"
  done
  [ -f "$ROOT/data/train/recognition_pool/textline/manifest.jsonl" ] && recog="$recog $ROOT/data/train/recognition_pool/textline/manifest.jsonl"
  [ -f "$ROOT/data/train/geometry/manifest.jsonl" ] && recog="$recog $ROOT/data/train/geometry/manifest.jsonl"
  echo "--manifest $ROOT/data/train/samples/manifest.jsonl$recog --val-manifest $ROOT/data/val/samples/manifest.jsonl --page-val $ROOT/data/val/pages"
  [ -f "$ROOT/marks/model.pt" ] && echo "--mark-model $ROOT/marks/model.pt"
  return 0
}

model_args() {  # the run's LoRA rank and starting adapter, from train_args.json
  "$PY" - "$ROOT/train_args.json" <<'PYEOF'
import json, sys
args = json.load(open(sys.argv[1]))
keep = ("--lora-rank", "--lora-alpha", "--init-adapter")
print(" ".join(f"{a} {args[i + 1]}" for i, a in enumerate(args) if a in keep and i + 1 < len(args)))
PYEOF
}

smoke() {
  rm -rf "$ROOT/smoke"
  local extra="" model
  [ "$LOCAL_TEST" = 1 ] && extra="--allow-cpu --batch-size 1 --eval-batch-size 2 --max-new-tokens 24 --num-workers 0 --page-batch 1 --max-steps 2 --page-val-max 1"
  model=$(model_args) || return 1
  cd "$CODE" && $(launcher) train.py $(common_args) --out "$ROOT/smoke" --max-steps 6 --log-every 2 \
    --eval-batches 1 --page-val-max 4 --samples 1 --save-every 6 --warmup-steps 1 $model $extra || return 1
  test -f "$ROOT/smoke/last/adapter_model.safetensors" || { echo "smoke run saved no checkpoint"; return 1; }
  rm -rf "$ROOT/smoke"
}

train() {
  local args
  args=$("$PY" -c "import json, sys; print(' '.join(json.load(open(sys.argv[1]))))" "$ROOT/train_args.json") || return 1
  rm -f "$ROOT/run/STOP"
  local resume=""
  [ -f "$ROOT/run/last/adapter_model.safetensors" ] && resume="--resume $ROOT/run/last"
  cd "$CODE" && $(launcher) train.py $(common_args) --out "$ROOT/run" $args $resume
}

if [ "$LOCAL_TEST" = 1 ]; then
  run setup_env echo "local test: the python in PY is used as is"
else
  run setup_env setup_env || exit 1
fi
run fetch_dataset fetch_dataset || exit 1
run fetch_model fetch_model || exit 1
run smoke smoke || exit 1
run train train || exit 1
