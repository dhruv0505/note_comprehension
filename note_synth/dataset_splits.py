"""
dataset_splits.py — which split every piece of source content belongs to,
decided before any page is generated.

    python dataset_splits.py                    # -> splits.json, and a summary
    python dataset_splits.py --seed 1 --out other_splits.json

Nothing a page is built from may sit in two splits: validation and test must
measure generalisation to unseen writers, texts and layouts, not memorisation of
content training already saw. Each source is split on the unit that carries its
identity:

  IAM-OnDB     forms, grouped so no writer and no text is in two splits. A form's
               writer (data/iamondb_writers.json, see iam_writers.py) and its text
               (the form id without its trailing letter) join it to every other
               form sharing either, and whole groups are dealt to val and test
               until each holds VAL_SHARE / TEST_SHARE of the lines; the rest is
               train. A group holding any form with no known writer stays in
               train, since that form's writer could be anyone.
  OHFD         templates (writer10_1b -> 1b): one flowchart layout copied by many
               writers, so a layout is in exactly one split. Val and test get
               OHFD_EVAL_TEMPLATES layouts each, the rest are train.
  MathWriting  the release's own directories (MATHWRITING_SPLITS): train and
               synthetic to train, valid to val, test to test.
  EMNIST       SD19 writer sets (EMNIST_SETS); the hsf_N sets are disjoint writers.
               The archive's train_*.mis files repeat hsf_0-3,6,7 glyph for glyph
               and are never read.
  QuickDraw    a hash of each drawing's key_id (quickdraw_split).

The IAM-OnDB and OHFD assignments are written out, since they come from a seeded
shuffle; the other three are fixed rules and are recorded for reference.
"""

from __future__ import annotations
import argparse
import collections
import hashlib
import json
import os
import random
import re
from typing import Dict, Tuple

import loaders

SPLITS = ("train", "val", "test")
VAL_SHARE = 0.10
TEST_SHARE = 0.10
# OHFD layouts each of val and test holds. There are only 28, and VAL_SHARE / TEST_SHARE of them left val and
# test three each, so their flowcharts were nearly all one layout: the only small one a small cell fits.
OHFD_EVAL_TEMPLATES = 5
# Layouts that are one chart drawn out two ways, and so share a split: by their captions (every drawing's, compared),
# 5/5b share 9 of 13-17 and 14/14b 7 of 12; every other N/Nb pair shares at most 5, most of them none.
OHFD_SAME_CHART = (("5", "5b"), ("14", "14b"))

MATHWRITING_SPLITS = {"train": "train", "synthetic": "train", "valid": "val", "test": "test"}
EMNIST_SETS = {"train": ("hsf_0", "hsf_1", "hsf_2", "hsf_3", "hsf_6"),
               "val": ("hsf_4",), "test": ("hsf_7",)}
QUICKDRAW_BUCKETS = 100
QUICKDRAW_VAL = range(80, 90)
QUICKDRAW_TEST = range(90, 100)

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits.json")


def quickdraw_split(key_id) -> str:
    """A drawing's split from its key_id, stable across runs and machines."""
    bucket = int(hashlib.sha1(str(key_id).encode()).hexdigest()[:8], 16) % QUICKDRAW_BUCKETS
    if bucket in QUICKDRAW_VAL:
        return "val"
    if bucket in QUICKDRAW_TEST:
        return "test"
    return "train"


def ohfd_template(chart: str) -> str:
    """'writer10_1b' -> '1b': the flowchart layout the writer copied.

    OHFD has 28 layouts, 1-14 and 1b-14b. Some files carry a trailing underscore
    ('writer3_5_', 'writer7_9b_') — another copy of the same layout, so it is
    stripped rather than taken for a layout of its own.
    """
    stem = os.path.basename(chart)
    if stem.endswith(".inkml"):
        stem = stem[:-len(".inkml")]
    return re.sub(r"_+$", "", re.sub(r"^writer\d+_", "", stem))


def iam_line_counts() -> Dict[str, int]:
    """IAM-OnDB form -> how many line files with ink it has."""
    names = loaders._Archive(loaders._resolve(None, "iamondb_strokes")).names(".xml")
    return dict(collections.Counter(loaders._iam_doc_and_line(name)[0] for name in names))


def iam_form_splits(rng: random.Random, lines: Dict[str, int]) -> Dict[str, str]:
    writers = loaders._iam_writer_map()
    parent: Dict[tuple, tuple] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for form in lines:
        text = ("text", re.sub(r"[a-z]$", "", form))
        union(("form", form), text)
        if form in writers:
            union(("writer", writers[form]), text)

    groups = collections.defaultdict(list)
    for form in lines:
        groups[find(("form", form))].append(form)

    total = sum(lines.values())
    target = {"val": VAL_SHARE * total, "test": TEST_SHARE * total}
    held = {"val": 0, "test": 0}
    order = sorted(groups, key=str)
    rng.shuffle(order)
    assignment = {}
    for key in order:
        forms = groups[key]
        count = sum(lines[f] for f in forms)
        split = "train"
        if all(f in writers for f in forms):
            for name in ("val", "test"):
                if held[name] + count <= target[name]:
                    split = name
                    held[name] += count
                    break
        for form in forms:
            assignment[form] = split
    return assignment


def ohfd_chart_splits(rng: random.Random) -> Tuple[Dict[str, str], Dict[str, str]]:
    """(chart -> split, template -> split)."""
    archive = loaders._Archive(loaders._resolve(None, "ohfd"))
    charts = [os.path.basename(n)[:-len(".inkml")] for n in archive.names(".inkml")]
    templates = sorted({ohfd_template(c) for c in charts})
    rng.shuffle(templates)
    # a layout goes with the others of its chart (OHFD_SAME_CHART), dealt in the order the first of them is shuffled to
    together = {t: group for group in OHFD_SAME_CHART for t in group}
    units, dealt = [], set()
    for t in templates:
        if t not in dealt:
            unit = [u for u in together.get(t, (t,)) if u in templates]
            units.append(unit)
            dealt.update(unit)
    by_template = {t: "train" for t in templates}
    held = {"val": 0, "test": 0}
    for unit in units:
        for split in ("val", "test"):
            if held[split] + len(unit) <= OHFD_EVAL_TEMPLATES:
                by_template.update({t: split for t in unit})
                held[split] += len(unit)
                break
    return {c: by_template[ohfd_template(c)] for c in charts}, by_template


def build(path: str = DEFAULT_PATH, seed: int = 0) -> dict:
    rng = random.Random(seed)
    lines = iam_line_counts()
    iam = iam_form_splits(rng, lines)
    ohfd, templates = ohfd_chart_splits(rng)
    summary = {
        "iam_lines": {s: sum(lines[f] for f, sp in iam.items() if sp == s) for s in SPLITS},
        "iam_forms": {s: sum(1 for sp in iam.values() if sp == s) for s in SPLITS},
        "ohfd_charts": {s: sum(1 for sp in ohfd.values() if sp == s) for s in SPLITS},
        "ohfd_templates": {s: sum(1 for sp in templates.values() if sp == s) for s in SPLITS},
    }
    data = {
        "seed": seed, "val_share": VAL_SHARE, "test_share": TEST_SHARE,
        "iam_forms": iam, "ohfd_charts": ohfd, "ohfd_templates": templates,
        "mathwriting_dirs": MATHWRITING_SPLITS, "emnist_sets": EMNIST_SETS,
        "quickdraw": {"rule": "sha1(key_id) % 100", "val": [QUICKDRAW_VAL.start, QUICKDRAW_VAL.stop],
                      "test": [QUICKDRAW_TEST.start, QUICKDRAW_TEST.stop]},
        "summary": summary,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=1)
    return data


def load(path: str = DEFAULT_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; run `python dataset_splits.py` first")
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_PATH)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    data = build(args.out, args.seed)
    print(json.dumps(data["summary"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
