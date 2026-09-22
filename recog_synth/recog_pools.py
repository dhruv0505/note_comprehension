"""
recog_pools.py — one split's items for one recognition task.

    textline   IAM-OnDB lines with a transcription, split by splits.json's form assignment
    formula    MathWriting expressions, split by the release's own directories
    drawing    QuickDraw sketches, split by the hash of their key_id

Given note_synth's split_pools.py cache, items come out of it: the same lines,
MathWriting store and drawings that split's pages are built from, so
recognition and segmentation never disagree about which split an item is in.
Without one, they are read from the archives with note_synth's loaders and its
splits.json. `--mathwriting` / `--quickdraw` force the archive for that source
even with a cache (the cache holds only 120 drawings a category for train).

note_synth is imported read-only: bytecode writing is switched off before it is
put on the path, and nothing is written under it. This module's own top level
imports none of it, so the pools below can be built from any Ink-like objects.

A pool is len(pool) items, each with label(i), ink(i) and item_id(i) — the ids
note_synth gives its source items ("iam:<form>:<line>", "mw:<split>:<index>",
"qd:<key_id>"). Items whose label cannot go into a target
(recog_layout.clean_label) or that carry no ink are left out.
"""

from __future__ import annotations
import sys

sys.dont_write_bytecode = True          # before note_synth is ever imported

import os
import pickle

from recog_layout import clean_label

HERE = os.path.dirname(os.path.abspath(__file__))
NOTE_SYNTH = os.path.abspath(os.path.join(HERE, "..", "note_synth"))
SPLITS = ("train", "val", "test")

# QuickDraw read from the archive: drawings per category, per split.
QUICKDRAW_PER_CATEGORY = {"train": 1000, "val": 100, "test": 100}
# The loaders drop items under three strokes to keep stray marks off a page
# (loaders.MIN_STROKES), which costs QuickDraw about a third of its sketches. A
# classifier has no such reason, so reading from the archive keeps them.
QUICKDRAW_MIN_STROKES = 1


def use_note_synth() -> None:
    """note_synth on the import path, with bytecode writing off."""
    sys.dont_write_bytecode = True
    if NOTE_SYNTH not in sys.path:
        sys.path.insert(0, NOTE_SYNTH)


class ListPool:
    """Items held as Ink objects."""

    def __init__(self, inks, task: str, split: str, prefix: str):
        self.task, self.split = task, split
        self._inks, self._labels, self._ids = [], [], []
        for position, ink in enumerate(inks):
            label = clean_label(ink.label)
            if label is None or not any(len(s.x) for s in ink.all_strokes()):
                continue
            self._inks.append(ink)
            self._labels.append(label)
            # an item with no id of its own is named by its place in what was loaded
            self._ids.append((ink.meta or {}).get("item") or f"{prefix}:{position}")

    def __len__(self) -> int:
        return len(self._inks)

    def label(self, index: int) -> str:
        return self._labels[index]

    def ink(self, index: int):
        return self._inks[index]

    def item_id(self, index: int) -> str:
        return self._ids[index]


class MathItems:
    """A mathwriting_store.MathPool's labelled expressions, each built when asked
    for. Reads through MathPool.ink, not indexing, so the synthesis runner's
    `redirect` (which is not set here anyway) never swaps an item."""

    def __init__(self, math_pool, split: str):
        self.task, self.split = "formula", split
        self._math = math_pool
        self._index = [i for i, label in enumerate(math_pool.labels) if clean_label(label) is not None]

    def __len__(self) -> int:
        return len(self._index)

    def label(self, index: int) -> str:
        return clean_label(self._math.labels[self._index[index]])

    def ink(self, index: int):
        return self._math.ink(self._index[index])

    def item_id(self, index: int) -> str:
        return self._math.item_id(self._index[index])


def _cached_pools(cache: str, split: str) -> dict:
    """split_pools.py's pools.pkl for `split` (Ink objects, so core must import)."""
    use_note_synth()
    path = os.path.join(cache, split, "pools.pkl")
    with open(path, "rb") as f:
        pools = pickle.load(f)
    if pools.get("split") != split:
        raise ValueError(f"{path} is the {pools.get('split')!r} split's, not {split!r}")
    return pools


def load_pool(task: str, split: str, cache: str = None, splits_file: str = None,
              mathwriting: str = None, quickdraw: str = None, qd_per_category: int = None):
    if split not in SPLITS:
        raise ValueError(f"split {split!r} is not one of {SPLITS}")
    use_note_synth()

    if task == "textline":
        if cache:
            inks = _cached_pools(cache, split)["text_pool"]
        else:
            import dataset_splits as ds
            import loaders
            forms = ds.load(splits_file or ds.DEFAULT_PATH)["iam_forms"]
            inks = [line for line in loaders.load_iam_ondb()
                    if forms.get((line.meta or {}).get("form")) == split]
        return ListPool(inks, task, split, "iam")

    if task == "formula":
        if cache and not mathwriting:
            import mathwriting_store as ms
            math = ms.open_pool(cache, split)
            if math is None:
                raise FileNotFoundError(f"no MathWriting store at {ms.store_dir(cache, split)}; "
                                        "build it with split_pools.py --mathwriting, or pass --mathwriting")
            return MathItems(math, split)
        import dataset_splits as ds
        import loaders
        dirs = tuple(d for d, s in ds.MATHWRITING_SPLITS.items() if s == split)
        return ListPool(loaders.load_mathwriting(mathwriting, splits=dirs), task, split, f"mw:{split}")

    if task == "drawing":
        if cache and not quickdraw:
            inks = _cached_pools(cache, split)["quickdraw_pool"]
        else:
            import dataset_splits as ds
            import loaders
            inks = loaders.load_quickdraw(quickdraw,
                                          max_per_category=qd_per_category or QUICKDRAW_PER_CATEGORY[split],
                                          min_strokes=QUICKDRAW_MIN_STROKES,
                                          keep_key=lambda key: ds.quickdraw_split(key) == split)
        return ListPool(inks, task, split, "qd")

    raise ValueError(f"unknown task {task!r}")
