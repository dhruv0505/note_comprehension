#!/usr/bin/env bash
# train_remote.sh (mark separation) — the mark separation CNN on a one-GPU Jarvis Labs VM, from
# the trainer bundle unpacked into ~/train/code. Each stage logs to logs/<stage>.log and records
# its start and end in stages/<stage>.json (train_status.py reads both). A stage already done is
# skipped, so the script can be started again after an interruption.
#
#   DATA_HOST=10.0.0.2 DATA_USER=ubuntu bash ~/train/code/remote/train_remote.sh
#
# Stages: setup_env -> fetch_dataset -> prepare -> smoke -> train
#   setup_env      a venv with torch for the driver's CUDA, numpy, pillow
#   fetch_dataset  train, val and test marks/ from the dataset VM over its private IP
#   prepare        marks_data.py: every crop scaled to fit 600 x 100 and padded, one array per split
#   smoke          6 steps and a small validation, then thrown away
#   train          train_marks.py with the arguments in ~/train/train_args.json
# The test arrays are prepared with the rest but only test_remote.sh reads them.
# Local test mode (LOCAL_TEST=1): no setup, dataset copied from DATA_DIR, python in $PY.
set -uo pipefail
ROOT="$HOME/train"
CODE="$ROOT/code"
mkdir -p "$ROOT/logs" "$ROOT/stages" "$ROOT/data" "$ROOT/run"
LOCAL_TEST=${LOCAL_TEST:-0}
export PYTHONUNBUFFERED=1
# memory freed after training steps is reused for validation's differently sized tensors rather than stranded
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
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
  "$PY" -m pip install -q numpy pillow || return 1
  "$PY" - <<'PYEOF'
import sys, torch
count = torch.cuda.device_count()
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", count,
      [torch.cuda.get_device_name(i) for i in range(count)])
if count < 1:
    sys.exit("no GPU visible")
PYEOF
}

fetch_dataset() {
  local parts="train/marks val/marks test/marks train/report.json val/report.json test/report.json"
  if [ "$LOCAL_TEST" = 1 ]; then
    (cd "${DATA_DIR:?}" && tar cf - --exclude=shards $parts) | tar xf - -C "$ROOT/data" || return 1
  else
    ssh -i "$HOME/.ssh/datavm" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 \
      "${DATA_USER:-ubuntu}@${DATA_HOST:?}" "cd ~/run/dataset && tar cf - --exclude=shards $parts" \
      | tar xf - -C "$ROOT/data" || return 1
  fi
  # the checks fail the stage: crops marks_data.py can't read are not prepared or trained on
  "$PY" - "$ROOT/data" "$CODE" <<'PYEOF' || return 1
import json, os, sys
from collections import Counter
root = sys.argv[1]
sys.path.insert(0, sys.argv[2])
from marks_data import MARK_TYPES, PARTS, row_problem
for split in ("train", "val", "test"):
    folder = os.path.join(root, split, "marks")
    report = json.load(open(os.path.join(root, split, "report.json")))
    rows = [json.loads(line) for line in open(os.path.join(folder, "manifest.jsonl")) if line.strip()]
    if len(rows) != report["marks"]:
        sys.exit(f"{split}: {len(rows)} manifest rows, report says {report['marks']} marks")
    bad = [(row.get("input"), row_problem(row)) for row in rows if row_problem(row)]
    if bad:
        sys.exit(f"{split}: {len(bad)} rows marks_data.py can't read, e.g. {bad[:3]}")
    missing = [row[p] for row in rows for p in PARTS if not os.path.exists(os.path.join(folder, row[p]))]
    if missing:
        sys.exit(f"{split}: {len(missing)} crop files missing, e.g. {missing[:3]}")
    types = {t: sum(row["labels"][k] for row in rows) for k, t in enumerate(MARK_TYPES)}
    print(f"{split}: {len(rows):,} crops; crops with each type {types}; "
          f"{sum(1 for row in rows if sum(row['labels']) > 1):,} with several kinds; {dict(Counter(r.get('crop') for r in rows))}")
PYEOF
  du -sh "$ROOT/data"
}

prepare() {
  cd "$CODE" && "$PY" marks_data.py --root "$ROOT/data" --out "$ROOT/prepared" --splits train val test \
    --workers "${PREP_WORKERS:-$(nproc 2>/dev/null || echo 4)}"
}

smoke() {
  rm -rf "$ROOT/smoke"
  local extra="" batch
  # the run's own batch size, and one full validation batch (64 crops): a batch this GPU can't hold fails here,
  # minutes in, rather than at the start of training
  batch=$("$PY" -c "import json, sys; a = json.load(open(sys.argv[1])); print(a[a.index('--batch-size') + 1] if '--batch-size' in a else 64)" \
    "$ROOT/train_args.json") || return 1
  [ "$LOCAL_TEST" = 1 ] && extra="--allow-cpu --batch-size 4 --eval-batch-size 8"
  cd "$CODE" && "$PY" train_marks.py --data "$ROOT/prepared" --out "$ROOT/smoke" --max-steps 6 --log-every 2 \
    --val-every 6 --val-max 64 --samples 2 --save-every 6 --warmup-steps 1 --batch-size "$batch" $extra || return 1
  test -f "$ROOT/smoke/last/model.pt" || { echo "smoke run saved no checkpoint"; return 1; }
  rm -rf "$ROOT/smoke"
}

train() {
  local args
  args=$("$PY" -c "import json, sys; print(' '.join(json.load(open(sys.argv[1]))))" "$ROOT/train_args.json") || return 1
  rm -f "$ROOT/run/STOP"
  local resume=""
  [ -f "$ROOT/run/last/model.pt" ] && resume="--resume $ROOT/run/last"
  cd "$CODE" && "$PY" train_marks.py --data "$ROOT/prepared" --out "$ROOT/run" $args $resume
}

if [ "$LOCAL_TEST" = 1 ]; then
  run setup_env echo "local test: the python in PY is used as is"
else
  run setup_env setup_env || exit 1
fi
run fetch_dataset fetch_dataset || exit 1
run prepare prepare || exit 1
run smoke smoke || exit 1
run train train || exit 1
