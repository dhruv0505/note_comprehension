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
               writer (data/iamondb_writers.json, see iam_writers.py), its text
               (the form id without its trailing letter) and its passage (any
               IAM_SHARED_RUNS runs of IAM_RUN_WORDS words, or any line of at least
               IAM_LINE_WORDS words, in common with another form's transcription:
               IAM-OnDB gives some passages to forms with unrelated ids, d05-008 and
               d05-315z for one) join it to every other form sharing any, and whole groups are dealt to val and test
               until each holds VAL_SHARE / TEST_SHARE of the lines; the rest is
               train. A group holding any form with no known writer stays in
               train, since that form's writer could be anyone.
  FC           layouts AND writers: every one of the 28 flowchart layouts is drawn
               by all 24 writers, so a chart goes to a split only when both its
               layout and its writer belong there, and the rest
               are used nowhere (63%, 422 of 672). Layouts that are one flowchart drawn twice
               (FC_SAME_CHART: an N/Nb pair is the same graph with its variables
               renamed) go together, FC_EVAL_TEMPLATES to each of val and test,
               chosen for room to write in (FC_EVAL_LAYOUTS); writers follow the release's own FC_Train/Validation/Test
               lists (FC_WRITER_SPLITS), which on their own put every layout in
               every split.
  MathWriting  the release's own directories (MATHWRITING_SPLITS): train and
               synthetic to train, valid to val, test to test.
  EMNIST       SD19 writer sets (EMNIST_SETS); the hsf_N sets are disjoint writers.
               The archive's train_*.mis files repeat hsf_0-3,6,7 glyph for glyph
               and are never read.
  QuickDraw    a hash of each drawing's key_id (quickdraw_split).

The IAM-OnDB and FC assignments are written out, since they come from a seeded
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
# FC layouts each of val and test holds. There are only 28, and VAL_SHARE / TEST_SHARE of them would leave val and
# test three each, so their flowcharts would be nearly all one layout: the only small one a small cell fits.
FC_EVAL_TEMPLATES = 5
# Which FC_EVAL_TEMPLATES layouts val and test get, chosen rather than dealt at random. A flowchart cell takes only a
# chart whose captions stay legible at its size, and a dealt split (001/001b/006/006b + 019 to val) left val one
# layout's 7 drawings roomy enough: they reached their reuse cap within a shard, and the val pages came out with 0.03
# flowcharts a page against train's 0.76 and test's 1.2. These are the assignment, of whole FC_SAME_CHART groups,
# with the most charts in the smaller of val and test whose median room inside a captioned shape is at least 6% of
# the chart's size (32 val, 30 test; the dealt split had 15 and 20). The tightest group, 001/001b/006/006b, is train's.
FC_EVAL_LAYOUTS = {"val": ("002", "002b", "014", "014b", "019"), "test": ("011", "011b", "012", "012b", "018")}
# Layouts that are one flowchart drawn out more than once, and so share a split. Every N/Nb pair is the same graph
# (node types and edges) for 79-100% of writers, with its variables renamed (008 r=a%b, 008b r=n%m) and its shapes
# moved about; 001 and 006 are the same graph for 92% of writers, drawn in nearly the same places.
FC_SAME_CHART = (("001", "001b", "006", "006b"), ("002", "002b"), ("003", "003b"), ("008", "008b"), ("009", "009b"),
                 ("010", "010b"), ("011", "011b"), ("012", "012b"), ("014", "014b"), ("015", "015b"))
# FC writers by split: the release's FC_Train.txt / FC_Validation.txt / FC_Test.txt lists (writers 000-009, 010-016,
# 017-023), so no val or test chart is in a hand training saw.
FC_WRITER_SPLITS = {**{f"{w:03d}": "train" for w in range(0, 10)}, **{f"{w:03d}": "val" for w in range(10, 17)},
                    **{f"{w:03d}": "test" for w in range(17, 24)}}
# IAM-OnDB passages: forms sharing this many runs of IAM_RUN_WORDS words, or one line of IAM_LINE_WORDS words or more,
# are one text
IAM_RUN_WORDS = 6
IAM_SHARED_RUNS = 3
IAM_LINE_WORDS = 5

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


def fc_writer(chart: str) -> str:
    """'writer003_fc_011b' -> '003'."""
    return re.match(r"writer(\d+)", os.path.basename(chart)).group(1)


def fc_template(chart: str) -> str:
    """'writer003_fc_011b' -> '011b': the flowchart layout the writer drew."""
    stem = os.path.basename(chart)
    if stem.endswith(".inkml"):
        stem = stem[:-len(".inkml")]
    return re.sub(r"_+$", "", re.sub(r"^writer\d+_(fc_)?", "", stem))


def iam_line_counts() -> Dict[str, int]:
    """IAM-OnDB form -> how many line files with ink it has."""
    names = loaders._Archive(loaders._resolve(None, "iamondb_strokes")).names(".xml")
    return dict(collections.Counter(loaders._iam_doc_and_line(name)[0] for name in names))


def _words(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


def iam_shared_passages(forms) -> set:
    """Pairs of forms whose transcriptions are one passage: IAM_SHARED_RUNS runs of IAM_RUN_WORDS words in common, or a
    line of IAM_LINE_WORDS words or more."""
    texts = loaders._iam_parse_transcriptions(loaders._Archive(loaders._resolve(None, "iamondb_ascii")))
    runs = collections.defaultdict(set)
    whole = collections.defaultdict(set)
    for form, lns in texts.items():
        if form not in forms:
            continue
        words = [w for ln in lns for w in _words(ln)]
        for i in range(len(words) - IAM_RUN_WORDS + 1):
            runs[tuple(words[i:i + IAM_RUN_WORDS])].add(form)
        for ln in lns:
            w = _words(ln)
            if len(w) >= IAM_LINE_WORDS:
                whole[tuple(w)].add(form)
    shared = collections.Counter()
    for owners in runs.values():
        if 1 < len(owners) <= 20:          # a stock phrase many forms share is not a passage
            owners = sorted(owners)
            for i, a in enumerate(owners):
                for b in owners[i + 1:]:
                    shared[(a, b)] += 1
    pairs = {pair for pair, n in shared.items() if n >= IAM_SHARED_RUNS}
    for owners in whole.values():
        owners = sorted(owners)
        pairs.update((a, b) for i, a in enumerate(owners) for b in owners[i + 1:])
    return pairs


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
    for a, b in sorted(iam_shared_passages(set(lines))):      # sorted: a set's order changes from run to run
        union(("form", a), ("form", b))

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


def fc_chart_splits(rng: random.Random) -> Tuple[Dict[str, str], Dict[str, str]]:
    """(chart -> split, template -> split). A chart whose writer belongs to another split than its layout is left out."""
    archive = loaders._Archive(loaders._resolve(None, "fc"))
    charts = [os.path.basename(n)[:-len(".inkml")] for n in archive.names(".inkml")]
    templates = sorted({fc_template(c) for c in charts})
    by_template = {t: "train" for t in templates}
    for split, layouts in FC_EVAL_LAYOUTS.items():
        by_template.update({t: split for t in layouts if t in by_template})
    # a layout is in the split of every other layout of its chart (FC_SAME_CHART)
    for group in FC_SAME_CHART:
        if len({by_template.get(t) for t in group if t in by_template}) > 1:
            raise ValueError(f"FC layouts {group} are one chart but FC_EVAL_LAYOUTS splits them")
    # a chart is used only where its layout and its writer agree
    return ({c: by_template[fc_template(c)] for c in charts
             if FC_WRITER_SPLITS.get(fc_writer(c)) == by_template[fc_template(c)]}, by_template)


def build(path: str = DEFAULT_PATH, seed: int = 0) -> dict:
    rng = random.Random(seed)
    lines = iam_line_counts()
    iam = iam_form_splits(rng, lines)
    fc, templates = fc_chart_splits(rng)
    summary = {
        "iam_lines": {s: sum(lines[f] for f, sp in iam.items() if sp == s) for s in SPLITS},
        "iam_forms": {s: sum(1 for sp in iam.values() if sp == s) for s in SPLITS},
        "fc_charts": {s: sum(1 for sp in fc.values() if sp == s) for s in SPLITS},
        "fc_charts_unused": len(loaders._Archive(loaders._resolve(None, "fc")).names(".inkml")) - len(fc),
        "fc_templates": {s: sum(1 for sp in templates.values() if sp == s) for s in SPLITS},
    }
    data = {
        "seed": seed, "val_share": VAL_SHARE, "test_share": TEST_SHARE,
        "iam_forms": iam, "fc_charts": fc, "fc_templates": templates, "fc_writers": FC_WRITER_SPLITS,
        "fc_same_chart": [list(g) for g in FC_SAME_CHART],
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
