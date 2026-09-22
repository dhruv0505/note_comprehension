#!/usr/bin/env bash
# test_remote.sh — the one evaluation of the test pages, with the best checkpoint
# (last/ if no validation ever chose one). It runs once: ~/train/test/STARTED is
# created exclusively before anything is read, and a second start finds it and stops.
#
#   NPROC=2 bash ~/train/code/remote/test_remote.sh
#
# Writes ~/train/test/: test_metrics.json, test_predictions.jsonl (every page),
# test_samples.jsonl (the first pages, as they are scored), test_progress.json,
# test.log, done.json {"exit_code", "end"}.
set -uo pipefail
ROOT="$HOME/train"
T="$ROOT/test"
mkdir -p "$T"
NPROC=${NPROC:-1}
LOCAL_TEST=${LOCAL_TEST:-0}
export HF_HOME="${HF_HOME:-$ROOT/hf_cache}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
if [ "$LOCAL_TEST" = 1 ]; then PY=${PY:?LOCAL_TEST needs PY}; else PY="$ROOT/venv/bin/python"; fi

if ! (set -o noclobber; date +%s > "$T/STARTED") 2>/dev/null; then
  echo "$(date +%s) refused: the test set was already evaluated or started" >> "$T/refused.log"
  exit 3
fi

CKPT="$ROOT/run/best"
[ -f "$CKPT/adapter_model.safetensors" ] || CKPT="$ROOT/run/last"
echo "$CKPT" > "$T/checkpoint"
if [ "$NPROC" -gt 1 ]; then LAUNCH="$ROOT/venv/bin/torchrun --standalone --nproc_per_node $NPROC"; else LAUNCH="$PY"; fi
MARKS=""
[ -f "$ROOT/marks/model.pt" ] && MARKS="--mark-model $ROOT/marks/model.pt"     # recognition takes marks out, as in validation

cd "$ROOT/code" && $LAUNCH seg_eval.py --checkpoint "$CKPT" --pages "$ROOT/data/test/pages" $MARKS \
  --out "$T/test_metrics.json" --predictions "$T/test_predictions.jsonl" --progress "$T/test_progress.json" \
  --samples-out "$T/test_samples.jsonl" --samples 12 --num-workers 4 ${TEST_EXTRA:-} > "$T/test.log" 2>&1
code=$?
"$PY" -c "import json, sys, time; json.dump({'exit_code': int(sys.argv[1]), 'end': time.time()}, open(sys.argv[2], 'w'))" \
  "$code" "$T/done.json"
exit $code
