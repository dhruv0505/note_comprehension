#!/usr/bin/env bash
# bootstrap_remote.sh — set up a brand-new dataset VM: what ../bootstrap_dataset_vm.py uploads it with, and runs.
#
# Expects ~/run/note_synth/ (the generator's code and splits.json) and ~/run/note_synth/data/ holding the archives
# that can't be fetched without a login or are only here: iamondb_linestrokes.tar.gz, iamondb_ascii.tar,
# iamondb_writers.json, deepwriting.zip, fc.zip. Fetches the rest straight onto the VM:
#   mathwriting.tgz   the official MathWriting 2024 release (~3 GB)
#   emnist.zip        NIST SD19 1st Edition (1995), the .mis rasters loaders.load_emnist reads
#   quickdraw.zip     the first QUICKDRAW_BYTES of every one of QuickDraw's 345 full/raw .ndjson files (HTTP range),
#                     cut back to whole lines: the split caches read at most 120 + 30 + 30 drawings a category,
#                     each from its own hash bucket, so a few thousand per category is plenty
# then builds the split caches (~/run/cache) with split_pools.py, which every dataset run synthesizes from.
#
# Each stage logs to ~/bootstrap/logs/<stage>.log and leaves ~/bootstrap/stages/<stage>.done when it succeeds; a
# rerun skips what is done. ~/bootstrap/ALL_DONE or ~/bootstrap/FAILED (with the stage's name) ends it.
set -uo pipefail
B="$HOME/bootstrap"
NS="$HOME/run/note_synth"
DATA="$NS/data"
QUICKDRAW_BYTES=${QUICKDRAW_BYTES:-6000000}
mkdir -p "$B/logs" "$B/stages" "$DATA"
rm -f "$B/FAILED"

stage() {     # stage NAME FUNCTION
  local name=$1; shift
  [ -f "$B/stages/$name.done" ] && return 0
  echo "$(date +%s) $name start" >> "$B/progress.log"
  if "$@" > "$B/logs/$name.log" 2>&1; then
    touch "$B/stages/$name.done"
    echo "$(date +%s) $name done" >> "$B/progress.log"
  else
    echo "$name" > "$B/FAILED"
    echo "$(date +%s) $name failed" >> "$B/progress.log"
    exit 1
  fi
}

setup_python() {
  python3 -c "import numpy, PIL, matplotlib" 2>/dev/null && { echo "numpy, Pillow and matplotlib already there"; return 0; }
  python3 -m pip install --quiet numpy pillow matplotlib \
    || python3 -m pip install --quiet --break-system-packages numpy pillow matplotlib \
    || { sudo apt-get update -qq && sudo apt-get install -y -qq python3-pip python3-numpy python3-pil python3-matplotlib; }
  python3 -c "import numpy, PIL, matplotlib; print('numpy', numpy.__version__, 'Pillow', PIL.__version__, 'matplotlib', matplotlib.__version__)"
  command -v zip >/dev/null || sudo apt-get install -y -qq zip unzip
}

check_uploads() {
  for name in iamondb_linestrokes.tar.gz iamondb_ascii.tar iamondb_writers.json deepwriting.zip fc.zip; do
    [ -s "$DATA/$name" ] || { echo "missing $DATA/$name"; return 1; }
    echo "$name $(du -h "$DATA/$name" | cut -f1)"
  done
  [ -f "$NS/splits.json" ] || { echo "missing $NS/splits.json"; return 1; }
}

fetch_mathwriting() {
  curl -fsSL --retry 5 -o "$DATA/mathwriting.tgz.part" https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz \
    && mv "$DATA/mathwriting.tgz.part" "$DATA/mathwriting.tgz" && ls -la "$DATA/mathwriting.tgz"
}

fetch_emnist() {
  curl -fsSL --retry 5 -o "$DATA/emnist.zip.part" https://s3.amazonaws.com/nist-srd/SD19/1stEdition1995.zip \
    && unzip -tq "$DATA/emnist.zip.part" && mv "$DATA/emnist.zip.part" "$DATA/emnist.zip" && ls -la "$DATA/emnist.zip"
}

fetch_quickdraw() {
  local work="$HOME/quickdraw_raw"
  rm -rf "$work" && mkdir -p "$work/raw"
  python3 - "$work/raw" "$QUICKDRAW_BYTES" <<'PY'
import concurrent.futures, json, os, sys, urllib.parse, urllib.request
out, limit = sys.argv[1], int(sys.argv[2])
listing = json.load(urllib.request.urlopen(
    "https://storage.googleapis.com/storage/v1/b/quickdraw_dataset/o?prefix=full/raw/&fields=items(name)&maxResults=1000"))
names = [i["name"] for i in listing["items"] if i["name"].endswith(".ndjson")]
assert len(names) == 345, f"expected 345 QuickDraw files, the listing has {len(names)}"

def fetch(name):
    url = "https://storage.googleapis.com/quickdraw_dataset/" + urllib.parse.quote(name)
    for attempt in range(5):
        try:
            data = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes=0-{limit - 1}"}),
                                          timeout=120).read()
            break
        except OSError:
            if attempt == 4:
                raise
    data = data[:data.rfind(b"\n") + 1]           # whole lines only: the range cuts the last drawing short
    with open(os.path.join(out, os.path.basename(name)), "wb") as f:
        f.write(data)
    return data.count(b"\n")

with concurrent.futures.ThreadPoolExecutor(16) as pool:
    counts = list(pool.map(fetch, names))
print(f"{len(counts)} files, {min(counts)}-{max(counts)} drawings each, {sum(counts):,} in all")
PY
  [ $? -eq 0 ] || return 1
  (cd "$work" && rm -f "$DATA/quickdraw.zip" && zip -q -r -0 "$DATA/quickdraw.zip" raw/) || return 1
  [ "$(unzip -l "$DATA/quickdraw.zip" | grep -c '\.ndjson$')" -eq 345 ] || { echo "quickdraw.zip is short of files"; return 1; }
  rm -rf "$work"
  ls -la "$DATA/quickdraw.zip"
}

build_caches() {
  cd "$NS" && python3 -u -W ignore split_pools.py --cache "$HOME/run/cache" --splits-file "$NS/splits.json" \
    --mathwriting "$DATA/mathwriting.tgz" --quickdraw "$DATA/quickdraw.zip"
  local split
  for split in train val test; do
    [ -f "$HOME/run/cache/$split/pools.pkl" ] || { echo "no cache for $split"; return 1; }
    cat "$HOME/run/cache/$split/summary.json"; echo
  done
  df -h "$HOME"
}

stage setup_python setup_python
stage check_uploads check_uploads
stage fetch_mathwriting fetch_mathwriting
stage fetch_emnist fetch_emnist
stage fetch_quickdraw fetch_quickdraw
stage build_caches build_caches
touch "$B/ALL_DONE"
