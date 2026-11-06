"""
split_pools.py — the pools one split's pages are built from, prepared once and cached.

    python split_pools.py --cache cache --mathwriting /data/mathwriting-2024.tgz
    python split_pools.py --cache cache --splits val test          # MathWriting stores already built

Needs splits.json (dataset_splits.py). For each split it writes cache/<split>/:

    pools.pkl          the split's IAM-OnDB lines, FC charts, QuickDraw drawings and
                       EMNIST glyphs, chosen by dataset_splits.py, pickled
    mathwriting/       the split's MathWriting store (mathwriting_store.py)
    shapes_bank.json   a shape bank mined from this split's pools alone
    summary.json       what went in, the POOLS_VERSION that built it and the splits.json's sha1
                       (a dataset run rebuilds a cache whose either differs from its code's:
                       tracker/synth/remote/regen_remote.sh)

A synthesis worker loads these in seconds (load_split) rather than parsing the
archives for minutes, and gets exactly its split's content and nothing else.

Pool sizes. IAM-OnDB, FC and MathWriting are taken whole. QuickDraw is 50M
drawings, far more than any run uses, so each split reads QUICKDRAW_PER_CATEGORY
drawings from each category (a page uses about 1.4). EMNIST glyphs are
vectorized on load, so each split takes EMNIST_PER_CLASS per character.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import pickle
import random
import time

import dataset_splits as ds
import geometry_diagrams as gd
import loaders
import mathwriting_store as ms

QUICKDRAW_PER_CATEGORY = {"train": 120, "val": 30, "test": 30}
EMNIST_PER_CLASS = {"train": 200, "val": 100, "test": 100}
# Items per pool the shape bank is mined from, drawn at random. The bank keeps
# the first 1,800 lines and 600 ovals it finds, so mining everything buys nothing
# but time, and a random draw keeps those from all coming from the first items.
MINE_PER_POOL = 4000
# Raised whenever what goes into the pools changes, so caches built by older code are rebuilt.
# 2: OHFD's text with no transcription that captions no shape is taken as what it is.
# 3: the flowcharts are the FC database (loaders.load_fc), not OHFD.
POOLS_VERSION = 3


def split_dir(cache: str, split: str) -> str:
    return os.path.join(cache, split)


def build_split(cache: str, split: str, splits: dict, iam, fc, quickdraw_path=None,
                qd_per_category: int = None, emnist_per_class: int = None,
                mine_per_pool: int = MINE_PER_POOL, seed: int = 0, splits_sha1: str = None) -> dict:
    started = time.time()
    out = split_dir(cache, split)
    os.makedirs(out, exist_ok=True)
    qd_per_category = qd_per_category or QUICKDRAW_PER_CATEGORY[split]
    emnist_per_class = emnist_per_class or EMNIST_PER_CLASS[split]

    text = [line for line in iam
            if splits["iam_forms"].get((line.meta or {}).get("form")) == split]
    charts = []
    for chart in fc:
        if splits["fc_charts"].get(chart.label) == split:
            chart.meta = dict(chart.meta or {}, item=f"fc:{chart.label}")
            charts.append(chart)
    drawings = loaders.load_quickdraw(quickdraw_path, max_per_category=qd_per_category,
                                      keep_key=lambda key: ds.quickdraw_split(key) == split)
    glyph_pool = {}
    for glyph in loaders.load_emnist(max_per_class=emnist_per_class, sets=ds.EMNIST_SETS[split]):
        glyph_pool.setdefault(glyph.label, []).append(glyph)

    with open(os.path.join(out, "pools.pkl"), "wb") as f:
        pickle.dump({"split": split, "text_pool": text, "flowchart_pool": charts,
                     "quickdraw_pool": drawings, "glyph_pool": glyph_pool},
                    f, protocol=pickle.HIGHEST_PROTOCOL)

    # the bank is mined from this split's own ink only
    rng = random.Random(seed)
    math = ms.open_pool(cache, split)
    mine = {"text_pool": rng.sample(text, min(mine_per_pool, len(text))),
            "flowchart_pool": list(charts),
            "quickdraw_pool": rng.sample(drawings, min(mine_per_pool, len(drawings)))}
    if math is not None and len(math):
        mine["math_pool"] = [math.ink(i) for i in rng.sample(range(len(math)), min(mine_per_pool, len(math)))]
    bank_path = os.path.join(out, "shapes_bank.json")
    if os.path.exists(bank_path):
        os.remove(bank_path)
    bank = gd.build_bank(cache=bank_path, pools=mine, split=split)

    summary = {
        "split": split,
        "iam_lines": len(text), "iam_writers": len({line.writer for line in text}),
        "fc_charts": len(charts), "quickdraw_drawings": len(drawings),
        "quickdraw_categories": len({d.label for d in drawings}),
        "emnist_glyphs": sum(len(v) for v in glyph_pool.values()),
        "mathwriting_expressions": len(math) if math is not None else 0,
        "bank_lines": len(bank.lines), "bank_ovals": len(bank.ovals), "bank_sources": bank.sources,
        # the lines and drawings were loaded with each point's time (Stroke.t), which
        # recog_synth colours ink by; a cache without this was built before they were
        "stroke_times": True,
        "pools_version": POOLS_VERSION, "splits_sha1": splits_sha1,
        "seconds": round(time.time() - started, 1),
    }
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def load_split(cache: str, split: str) -> dict:
    """generate_page's keyword arguments for one split, from its cache."""
    folder = split_dir(cache, split)
    with open(os.path.join(folder, "pools.pkl"), "rb") as f:
        pools = pickle.load(f)
    if pools.pop("split") != split:
        raise ValueError(f"{folder}/pools.pkl is not the {split} split's")
    bank_path = os.path.join(folder, "shapes_bank.json")
    if not os.path.exists(bank_path):
        # build_bank would mine every dataset in its place, which is the leak the
        # per-split bank exists to prevent
        raise FileNotFoundError(f"{bank_path} missing; run split_pools.py for {split}")
    pools["shape_bank"] = gd.build_bank(cache=bank_path, split=split)
    pools["math_pool"] = ms.open_pool(cache, split)
    return pools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--splits", nargs="+", default=list(ds.SPLITS), choices=ds.SPLITS)
    ap.add_argument("--splits-file", default=ds.DEFAULT_PATH)
    ap.add_argument("--mathwriting", default=None,
                    help="MathWriting release (.tgz or directory); built into <cache>/<split>/mathwriting "
                         "for every split in one pass, unless those stores already exist")
    ap.add_argument("--quickdraw", default=None, help="QuickDraw raw zip or directory (default: data/)")
    ap.add_argument("--quickdraw-per-category", type=int, default=None)
    ap.add_argument("--emnist-per-class", type=int, default=None)
    ap.add_argument("--mine-per-pool", type=int, default=MINE_PER_POOL)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    splits = ds.load(args.splits_file)
    with open(args.splits_file, "rb") as f:
        splits_sha1 = hashlib.sha1(f.read()).hexdigest()
    # a store built before times were kept (ms.has_times) is built again
    if args.mathwriting and not all(ms.has_times(args.cache, s) for s in ds.SPLITS):
        print(f"MathWriting: building stores from {args.mathwriting}", flush=True)
        print(json.dumps(ms.build(args.mathwriting, args.cache), indent=1), flush=True)

    t = time.time()
    iam = loaders.load_iam_ondb()
    fc = loaders.load_fc()
    print(f"loaded IAM-OnDB {len(iam)} lines, FC {len(fc)} charts in {time.time() - t:.0f}s", flush=True)
    for split in args.splits:
        summary = build_split(args.cache, split, splits, iam, fc, args.quickdraw,
                              args.quickdraw_per_category, args.emnist_per_class,
                              args.mine_per_pool, args.seed, splits_sha1)
        print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
