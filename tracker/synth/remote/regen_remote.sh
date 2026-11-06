#!/usr/bin/env bash
# regen_remote.sh — regenerate the dataset, or a sample of it, on the dataset VM with the generator
# code unpacked beside this script (note_synth/), from the split caches the first run built
# (~/run/cache). synth_orchestrator.py uploads this folder to ~/regen/<run id>/ and starts it.
# Each stage logs to logs/<stage>.log and records its start and end in stages/<stage>.json
# (status.py reads both); a stage already done is skipped, so a rerun carries on.
#
# targets.json (written by synth_orchestrator.py): mode (sample | full), splits, pages per split,
# shard_pages, stages, cache, out and final (paths under $HOME), and the run's settings: tuning
# (class_tuning.py overrides, written to tuning.json for synthesize.py --tuning), mark_crop, and
# recognition {tasks}: when set, synthesize.py --recognition also makes the train and val pages'
# recognition samples as it builds them, with recog_synth's recog_pages.py beside note_synth.
#   sample   every split into ~/regen/<run id>/dataset; ~/run/dataset is not touched
#   full     ~/run/dataset is deleted first (the disk can't hold two), then every split into
#            ~/run/dataset_new, moved to ~/run/dataset once all are done
#   additions  train only, only what the dataset lacks: geometry figures without a page (synth_geometry,
#            synthesize.py --only-geometry) and the textline pool passes (recog_pool_train), written to
#            ~/run/additions_new and moved into ~/run/dataset/train/{geometry,recognition_pool} (swap_additions).
#            No page, sample, mark or recognition sample is made, and val and test are never read or written
#
# Stages: check_cache -> cache_rebuild -> prepare_output -> synth_<split>... -> [recog_pool_train]
#         -> [swap_dataset] -> count_classes -> all_done
# targets.json's geometry_per_page gives synth_train geometry figures without a page (synthesize.py
# --geometry-per-page), and recognition's pool_epochs the train textline pool pass (recog_pool_train).
# cache_rebuild rebuilds the split caches, in place, from the archives the first run downloaded
# (~/run/note_synth/data) with the splits.json uploaded with this code, when they are stale for it: built
# with another splits.json or by an older split_pools.py (its POOLS_VERSION), or, with recognition on,
# before strokes kept their times (recognition colours ink by each point's time). A recognition-only run
# rebuilds pages its dataset was already made from, so it keeps their caches and rebuilds only for times.
set -uo pipefail
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS="$RUN/note_synth"
mkdir -p "$RUN/logs" "$RUN/stages"

eval "$(python3 - "$RUN/targets.json" "$HOME" <<'PY'
import json, os, shlex, sys
t = json.load(open(sys.argv[1]))
home = sys.argv[2].rstrip("/")          # the shell's own spelling of $HOME, so the swap guard below can compare

values = {"MODE": t["mode"], "SPLITS": " ".join(t["splits"]), "SHARD_PAGES": t["shard_pages"],
          "CACHE": f"{home}/{t['cache']}", "OUT": f"{home}/{t['out']}",
          "FINAL": f"{home}/{t['final']}", "TOTAL_PAGES": sum(t.get(s, 0) for s in t["splits"])}
for split in ("train", "val", "test"):
    values[f"PAGES_{split}"] = t.get(split, 0)
values["TUNING_FILE"] = ""
if t.get("tuning"):
    values["TUNING_FILE"] = os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), "tuning.json")
    with open(values["TUNING_FILE"], "w") as f:
        json.dump(t["tuning"], f, indent=1)
values["MARK_CROP"] = t.get("mark_crop") or "context"
rec = t.get("recognition") or {}
values["RECOG_TASKS"] = " ".join(rec.get("tasks") or [])
values["POOL_EPOCHS"] = int(rec.get("pool_epochs") or 0)
# the train textline pool pass: with recognition on, in a full or recognition run that makes train
values["POOL_STAGE"] = ("1" if values["RECOG_TASKS"] and values["POOL_EPOCHS"] > 0 and t["mode"] != "sample"
                        and "train" in t["splits"] else "")
geometry = float(t.get("geometry_per_page") or 0)
values["GEOMETRY_ARGS"] = f"--geometry-per-page {geometry}" if geometry > 0 else ""
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)" || { echo "could not read targets.json"; exit 1; }
WORKERS=${WORKERS:-$(nproc)}

mark() {    # mark start|done|failed STAGE
  python3 - "$RUN/stages/$2.json" "$1" <<'PY'
import json, os, sys, time
path, action = sys.argv[1], sys.argv[2]
state = json.load(open(path)) if os.path.exists(path) else {}
if action == "start":
    state = {"start": time.time(), "state": "running"}
else:
    state.update(end=time.time(), state=action)
with open(path + ".tmp", "w") as f:
    json.dump(state, f)
os.replace(path + ".tmp", path)
PY
}

run() {     # run STAGE COMMAND...
  local name=$1; shift
  if [ -f "$RUN/stages/$name.json" ] && grep -q '"state": "done"' "$RUN/stages/$name.json"; then
    return 0
  fi
  mark start "$name"
  if "$@" > "$RUN/logs/$name.log" 2>&1; then
    mark done "$name"
  else
    mark failed "$name"
    return 1
  fi
}

check_cache() {
  for split in $SPLITS; do
    [ -f "$CACHE/$split/pools.pkl" ] || { echo "no split cache for $split in $CACHE"; return 1; }
    echo "cache $split: $(du -sh "$CACHE/$split" | cut -f1)"
    case " $RECOG_TASKS " in *" formula "*)
      [ -f "$CACHE/$split/mathwriting/meta.json" ] || { echo "no MathWriting store for $split in $CACHE"; return 1; } ;;
    esac
  done
  if [ -n "$RECOG_TASKS" ]; then
    [ -f "$RUN/recog_synth/recog_pages.py" ] || { echo "recognition is on but recog_synth was not uploaded"; return 1; }
  fi
  [ -z "$TUNING_FILE" ] || { echo "class tuning:"; cat "$TUNING_FILE"; }
  echo "mark crops: $MARK_CROP"
  df -h "$HOME"
  if [ "$MODE" = full ] && [ -d "$FINAL" ]; then
    # prepare_output deletes the old dataset before the new one is written: the new one must fit in what is free
    # now plus what the old one takes
    python3 - "$FINAL" "$HOME" "$TOTAL_PAGES" <<'PY'
import json, os, shutil, subprocess, sys
final, home, total = sys.argv[1], sys.argv[2], int(sys.argv[3])
used = int(subprocess.run(["du", "-sb", final], capture_output=True, text=True).stdout.split()[0])
pages = 0
for split in ("train", "val", "test"):
    try:
        pages += json.load(open(os.path.join(final, split, "report.json")))["pages"]
    except (OSError, ValueError, KeyError):
        pass
need = used / max(pages, 1) * total * 1.3
free = shutil.disk_usage(home).free
print(f"current dataset {used / 1e9:.1f} GB over {pages:,} pages, deleted before the new one is written; the new one "
      f"needs about {need / 1e9:.1f} GB; {free / 1e9:.1f} GB free now, {(free + used) / 1e9:.1f} GB once it is deleted")
if pages and free + used < need:
    sys.exit(f"not enough disk: about {need / 1e9:.0f} GB needed, {(free + used) / 1e9:.0f} GB free even with the "
             f"current dataset deleted")
PY
  fi
}

cache_rebuild() {
  if python3 - "$CACHE" "$NS" "$MODE" "$RECOG_TASKS" <<'PY'
import hashlib, json, os, re, sys
cache, ns, mode, recognition = sys.argv[1:5]
with open(os.path.join(ns, "split_pools.py")) as f:
    version = int(re.search(r"^POOLS_VERSION = (\d+)", f.read(), re.M).group(1))
with open(os.path.join(ns, "splits.json"), "rb") as f:
    splits = hashlib.sha1(f.read()).hexdigest()
def load(*parts):
    try:
        with open(os.path.join(cache, *parts)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}
stale = []
for split in ("train", "val", "test"):
    summary, reasons = load(split, "summary.json"), []
    if recognition and not (summary.get("stroke_times") and load(split, "mathwriting", "meta.json").get("times")):
        reasons.append("no stroke times")
    if mode not in ("recognition", "additions"):     # adding to a dataset keeps the caches it was made from
        if summary.get("pools_version", 1) < version:
            reasons.append(f"built by split_pools version {summary.get('pools_version', 1)}, this code's is {version}")
        if summary.get("splits_sha1") != splits:
            reasons.append("built with another splits.json")
    if reasons:
        stale.append(f"{split} ({', '.join(reasons)})")
print("stale split caches: " + ("; ".join(stale) or "none"))
sys.exit(1 if stale else 0)
PY
  then
    return 0
  fi
  local src="$HOME/run/note_synth" name
  for name in iamondb_linestrokes.tar.gz iamondb_ascii.tar fc.zip emnist.zip mathwriting.tgz quickdraw.zip; do
    [ -f "$src/data/$name" ] || { echo "no $src/data/$name to rebuild the split caches from"; return 1; }
  done
  # this run's code and splits.json, reading the first run's archives: linked into its data/, where the loaders look
  mkdir -p "$NS/data"
  for name in "$src"/data/*; do
    [ -e "$NS/data/$(basename "$name")" ] || ln -s "$name" "$NS/data/"
  done
  (cd "$NS" && python3 -u -W ignore split_pools.py --cache "$CACHE" --splits-file "$NS/splits.json" \
    --mathwriting "$src/data/mathwriting.tgz" --quickdraw "$src/data/quickdraw.zip")
}

recog_pool() {
  # every IAM-OnDB line of train's pool, POOL_EPOCHS passes, ten a sample (recog_synth/recog_synthesize.py), into
  # train/recognition_pool/textline beside the page samples' train/recognition: lines no page placed are read too
  [ -f "$RUN/recog_synth/recog_synthesize.py" ] || { echo "recognition is on but recog_synth was not uploaded"; return 1; }
  (cd "$RUN/recog_synth" && python3 -u -W ignore recog_synthesize.py --task textline --split train --cache "$CACHE" \
    --out "$OUT" --subdir recognition_pool --epochs "$POOL_EPOCHS" --workers "$WORKERS")
}

prepare_output() {
  if [ "$MODE" = additions ]; then
    # kept between attempts: a shard with a usage record is skipped, so a run that died carries on where it stopped
    mkdir -p "$OUT" && echo "adding to $OUT ($(du -sh "$OUT" 2>/dev/null | cut -f1) already there)"
    return 0
  fi
  if [ "$MODE" = full ] && [ "$OUT" = "$FINAL" ]; then
    echo "refusing: a full run must write beside the dataset, not over it"; return 1
  fi
  if [ "$MODE" = full ] && [ -d "$FINAL" ]; then
    # the old dataset goes before the new one is written, to make room (check_cache measured it). Only ever
    # ~/run/dataset, compared as resolved folders, as swap_dataset checks it
    local parent
    parent=$(cd "$(dirname "$FINAL")" 2>/dev/null && pwd -P)
    if [ "$(basename "$FINAL")" != dataset ] || [ -z "$parent" ] || [ "$parent" != "$(cd "$HOME/run" && pwd -P)" ]; then
      echo "refusing to delete $FINAL"; return 1
    fi
    echo "deleting the old dataset $FINAL ($(du -sh "$FINAL" | cut -f1)) to make room for the new one"
    rm -rf "$FINAL" || return 1
  fi
  rm -rf "$OUT" && mkdir -p "$OUT" && echo "writing to $OUT"
}

swap_dataset() {
  # only ever ~/run/dataset: compared as resolved folders, however the path is spelled. Its parent is resolved
  # without it, since prepare_output has deleted it by now
  local parent
  parent=$(cd "$(dirname "$FINAL")" 2>/dev/null && pwd -P)
  if [ "$(basename "$FINAL")" != dataset ] || [ -z "$parent" ] || [ "$parent" != "$(cd "$HOME/run" && pwd -P)" ]; then
    echo "refusing to replace $FINAL"; return 1
  fi
  if [ -d "$OUT" ]; then          # absent on a rerun after the move already happened
    rm -rf "$FINAL" && mv "$OUT" "$FINAL" || return 1
  fi
  ls "$FINAL"
}

swap_recognition() {
  # only ever ~/run/dataset/<split>/recognition, each split's replaced by the new one; nothing else is touched
  local parent split
  parent=$(cd "$FINAL/.." 2>/dev/null && pwd -P)
  if [ "$(basename "$FINAL")" != dataset ] || [ -z "$parent" ] || [ "$parent" != "$(cd "$HOME/run" && pwd -P)" ]; then
    echo "refusing to write into $FINAL"; return 1
  fi
  for split in $SPLITS; do
    [ -d "$FINAL/$split" ] || { echo "no $FINAL/$split to put the recognition samples in"; return 1; }
    if [ -d "$OUT/$split/recognition" ]; then     # absent on a rerun after the move already happened
      rm -rf "$FINAL/$split/recognition" && mv "$OUT/$split/recognition" "$FINAL/$split/recognition" || return 1
    fi
    if [ -d "$OUT/$split/recognition_pool" ]; then
      rm -rf "$FINAL/$split/recognition_pool" && mv "$OUT/$split/recognition_pool" "$FINAL/$split/recognition_pool" || return 1
    fi
    echo "$split: $(ls "$FINAL/$split/recognition")"
  done
}

swap_additions() {
  # only ever ~/run/dataset/train/geometry and ~/run/dataset/train/recognition_pool, each replaced by this run's
  local parent name
  parent=$(cd "$FINAL/.." 2>/dev/null && pwd -P)
  if [ "$(basename "$FINAL")" != dataset ] || [ -z "$parent" ] || [ "$parent" != "$(cd "$HOME/run" && pwd -P)" ]; then
    echo "refusing to write into $FINAL"; return 1
  fi
  [ -f "$FINAL/train/samples/manifest.jsonl" ] || { echo "no $FINAL/train/samples: nothing to add to"; return 1; }
  for name in geometry recognition_pool; do
    if [ -d "$OUT/train/$name" ]; then       # absent on a rerun after the move already happened
      rm -rf "$FINAL/train/$name" && mv "$OUT/train/$name" "$FINAL/train/$name" || return 1
    fi
    if [ -d "$FINAL/train/$name" ]; then
      echo "train/$name: $(du -sh "$FINAL/train/$name" | cut -f1)"
    fi
  done
  return 0
}

run check_cache check_cache || exit 1
run cache_rebuild cache_rebuild || exit 1
run prepare_output prepare_output || exit 1
cd "$NS" || exit 1
if [ "$MODE" = additions ]; then
  [ -n "$GEOMETRY_ARGS" ] || [ -n "$POOL_STAGE" ] || { echo "an additions run with nothing to add"; exit 1; }
  if [ -n "$GEOMETRY_ARGS" ]; then
    run synth_geometry python3 -u -W ignore synthesize.py --cache "$CACHE" --split train --pages "$PAGES_train" \
      --workers "$WORKERS" --shard-pages "$SHARD_PAGES" --out "$OUT" ${TUNING_FILE:+--tuning "$TUNING_FILE"} \
      $GEOMETRY_ARGS --only-geometry || exit 1
  fi
  if [ -n "$POOL_STAGE" ]; then
    run recog_pool_train recog_pool || exit 1
  fi
  run swap_additions swap_additions || exit 1
elif [ "$MODE" = recognition ]; then
  # only the train and val pages' recognition samples, from the pages rebuilt in memory: the dataset's pages, samples
  # and mark crops stay as they are, and each split's recognition/ replaces the old one once every split is done
  for split in $SPLITS; do
    pages_var="PAGES_$split"
    run "recog_$split" python3 -u -W ignore synthesize.py --cache "$CACHE" --split "$split" --pages "${!pages_var}" \
      --workers "$WORKERS" --shard-pages "$SHARD_PAGES" --out "$OUT" ${TUNING_FILE:+--tuning "$TUNING_FILE"} \
      --only-recognition || exit 1
  done
  if [ -n "$POOL_STAGE" ]; then
    run recog_pool_train recog_pool || exit 1
  fi
  run swap_recognition swap_recognition || exit 1
else
  for split in $SPLITS; do
    pages_var="PAGES_$split"
    geo=""
    [ "$split" = train ] && geo="$GEOMETRY_ARGS"        # geometry figures without a page: train only
    run "synth_$split" python3 -u -W ignore synthesize.py --cache "$CACHE" --split "$split" --pages "${!pages_var}" \
      --workers "$WORKERS" --shard-pages "$SHARD_PAGES" --out "$OUT" --mark-crop "$MARK_CROP" \
      ${TUNING_FILE:+--tuning "$TUNING_FILE"} ${RECOG_TASKS:+--recognition} $geo || exit 1
  done
  if [ -n "$POOL_STAGE" ]; then
    run recog_pool_train recog_pool || exit 1
  fi
  if [ "$MODE" = full ]; then
    run swap_dataset swap_dataset || exit 1
  fi
  run count_classes sh -c "python3 '$RUN/class_counts.py' '$FINAL' > '$RUN/class_counts.json'" || exit 1
fi
mark start all_done
mark done all_done
