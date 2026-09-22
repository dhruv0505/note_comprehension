"""
marks_data.py — the mark separation crops (note_synth's mark_crops.py) as fixed-size arrays.

    python marks_data.py --root data --out prepared --splits train val test --workers 16

A row of <root>/<split>/marks/manifest.jsonl names two greyscale crops of one region of
a page, input (the writing and its marks drawn together) and text (the writing alone),
and gives the crop's labels: one 0/1 per MARK_TYPES, whether that kind of mark is in it
(several can be). Both crops are scaled by the one factor that fits them inside
WIDTH x HEIGHT — up or down, aspect kept — and centred on white. Under --out, per split:

    <split>.npy     uint8 (N, 2, HEIGHT, WIDTH): input, text; 255 is paper
    <split>.jsonl   one row per crop, in the same order: name, labels, shapes, crop,
                    text_label, page, the source size, the scale and the content box
                    [x0, y0, x1, y1] inside the padded image
"""

from __future__ import annotations
import argparse
import json
import os
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
from PIL import Image

HEIGHT, WIDTH = 100, 600
PARTS = ("input", "text")
MARK_TYPES = ("underline", "strikethrough", "oval", "rectangle")    # the order of a row's labels, as mark_crops.MARK_TYPES


def fit(width, height):
    """(scale, new width, new height, x offset, y offset) placing a width x height crop in WIDTH x HEIGHT."""
    scale = min(WIDTH / width, HEIGHT / height)
    w = min(WIDTH, max(1, round(width * scale)))
    h = min(HEIGHT, max(1, round(height * scale)))
    return scale, w, h, (WIDTH - w) // 2, (HEIGHT - h) // 2


def row_problem(row):
    """Why a manifest row can't be used, or None."""
    missing = [part for part in PARTS if not row.get(part)]
    if missing:
        return f"no {' or '.join(missing)} crop"
    labels = row.get("labels")
    if not (isinstance(labels, list) and len(labels) == len(MARK_TYPES) and all(v in (0, 1) for v in labels)):
        return f"labels {labels!r} are not one 0/1 per {', '.join(MARK_TYPES)}"
    return None


def load_row(task):
    folder, row = task
    problem = row_problem(row)
    if problem:
        raise ValueError(f"{row.get('input')}: {problem}")
    images = [Image.open(os.path.join(folder, row[part])).convert("L") for part in PARTS]
    width, height = images[0].size
    if any(im.size != (width, height) for im in images):
        raise ValueError(f"{row['input']}: input and text differ in size")
    scale, w, h, x0, y0 = fit(width, height)
    out = np.full((len(PARTS), HEIGHT, WIDTH), 255, dtype=np.uint8)
    for k, im in enumerate(images):
        out[k, y0:y0 + h, x0:x0 + w] = np.asarray(im.resize((w, h), Image.BILINEAR))
    name = os.path.basename(row["input"])
    meta = {"name": name[:-len("_input.png")] if name.endswith("_input.png") else os.path.splitext(name)[0],
            "labels": [int(v) for v in row["labels"]], "shapes": row.get("shapes"), "crop": row.get("crop"),
            "text_label": row.get("text_label"), "page": row.get("page"), "source_size": [width, height],
            "scale": round(scale, 5), "content": [x0, y0, x0 + w, y0 + h]}
    return out, meta


def prepare(root, out, split, workers):
    folder = os.path.join(root, split, "marks")
    with open(os.path.join(folder, "manifest.jsonl")) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    os.makedirs(out, exist_ok=True)
    target = os.path.join(out, f"{split}.npy")
    started = time.time()
    metas = []
    if not rows:
        np.save(target, np.zeros((0, len(PARTS), HEIGHT, WIDTH), dtype=np.uint8))
    else:
        tmp = target + ".tmp"
        array = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8,
                                          shape=(len(rows), len(PARTS), HEIGHT, WIDTH))
        tasks = [(folder, row) for row in rows]
        pool = Pool(workers) if workers > 1 else None
        try:
            results = pool.imap(load_row, tasks, chunksize=64) if pool else map(load_row, tasks)
            for i, (crop, meta) in enumerate(results):
                array[i] = crop
                metas.append(meta)
                if (i + 1) % 2000 == 0:
                    print(f"  {split}: {i + 1:,} of {len(rows):,} crops", flush=True)
        finally:
            if pool:
                pool.close()
                pool.join()
        array.flush()
        del array
        os.replace(tmp, target)
    with open(target[:-4] + ".jsonl.tmp", "w", newline="\n") as f:
        for meta in metas:
            f.write(json.dumps(meta) + "\n")
    os.replace(target[:-4] + ".jsonl.tmp", target[:-4] + ".jsonl")

    types = {t: sum(m["labels"][k] for m in metas) for k, t in enumerate(MARK_TYPES)}
    several = sum(1 for m in metas if sum(m["labels"]) > 1)
    upscaled = sum(1 for m in metas if m["scale"] > 1)
    print(f"{split}: {len(metas):,} crops -> {target} ({os.path.getsize(target) / 1e9:.2f} GB, "
          f"{time.time() - started:.0f}s); crops with each type {types}; {several:,} with several kinds; "
          f"{dict(Counter(m['crop'] for m in metas))} crops; {upscaled:,} scaled up", flush=True)
    return len(metas)


def load_split(folder, split, mmap=True):
    """(uint8 array (N, 2, HEIGHT, WIDTH), per-crop metadata) as prepare() wrote them."""
    array = np.load(os.path.join(folder, f"{split}.npy"), mmap_mode="r" if mmap else None)
    with open(os.path.join(folder, f"{split}.jsonl")) as f:
        metas = [json.loads(line) for line in f if line.strip()]
    if array.ndim != 4 or array.shape[1] != len(PARTS) or any("labels" not in m for m in metas):
        raise ValueError(f"{split}: prepared by an older marks_data.py ({array.shape}); prepare it again")
    if len(metas) != len(array):
        raise ValueError(f"{split}: {len(array)} crops but {len(metas)} metadata rows")
    return array, metas


def label_array(metas):
    """float32 (N, len(MARK_TYPES)): each crop's labels."""
    return np.array([m["labels"] for m in metas], dtype=np.float32).reshape(len(metas), len(MARK_TYPES))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="the dataset: <root>/<split>/marks/manifest.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()
    for split in args.splits:
        prepare(args.root, args.out, split, args.workers)


if __name__ == "__main__":
    main()
