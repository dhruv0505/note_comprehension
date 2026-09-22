#!/usr/bin/env bash
# test_remote.sh (mark separation) — the one evaluation of the test mark crops, with the best
# checkpoint (last/ if no validation ever chose one). It runs once: ~/train/test/STARTED is
# created exclusively before anything is read, and a second start finds it and stops.
#
#   bash ~/train/code/remote/test_remote.sh
#
# Writes ~/train/test/: test_metrics.json (aggregate scores only), test_progress.json,
# test.log, done.json {"exit_code", "end"}.
set -uo pipefail
ROOT="$HOME/train"
T="$ROOT/test"
mkdir -p "$T"
LOCAL_TEST=${LOCAL_TEST:-0}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
if [ "$LOCAL_TEST" = 1 ]; then PY=${PY:?LOCAL_TEST needs PY}; else PY="$ROOT/venv/bin/python"; fi

if ! (set -o noclobber; date +%s > "$T/STARTED") 2>/dev/null; then
  echo "$(date +%s) refused: the test set was already evaluated or started" >> "$T/refused.log"
  exit 3
fi

CKPT="$ROOT/run/best"
[ -f "$CKPT/model.pt" ] || CKPT="$ROOT/run/last"
echo "$CKPT" > "$T/checkpoint"

cd "$ROOT/code" && "$PY" marks_eval.py --checkpoint "$CKPT" --data "$ROOT/prepared" --split test \
  --out "$T/test_metrics.json" --progress "$T/test_progress.json" ${TEST_EXTRA:-} > "$T/test.log" 2>&1
code=$?
"$PY" -c "import json, sys, time; json.dump({'exit_code': int(sys.argv[1]), 'end': time.time()}, open(sys.argv[2], 'w'))" \
  "$code" "$T/done.json"
exit $code
