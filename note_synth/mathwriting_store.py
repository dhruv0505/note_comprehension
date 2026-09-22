"""
mathwriting_store.py — MathWriting per split as memory-mapped arrays, and a
lazy pool over them.

    python mathwriting_store.py --src /path/to/mathwriting-2024.tgz --out cache
        -> cache/<split>/mathwriting/{points.f32, times.f32, stroke_ends.npy, expr_ends.npy, labels.json, meta.json}

times.f32 holds each point's time (InkML's T channel, ms) from its expression's first
point, NaN where a file has none; a store built before it existed has no times
(has_times), and split_pools.py rebuilds it.

The full release is over 600k InkML expressions. Parsed into Ink objects that is
many GB, and a synthesis run holds its pools in every worker process; read
through loaders._Archive it is slower still, since a member read out of archive
order makes a .tgz decompress from the start again. So the release is streamed
once, in archive order, into three flat arrays per split, and pages draw from a
MathPool that builds an expression's Ink only when it is asked for. The arrays
are memory-mapped, so the workers on one machine share a single copy.

Which directory goes to which split is dataset_splits.MATHWRITING_SPLITS.
`symbols` is left out, as loaders.load_mathwriting leaves it out: most of its
items are one stroke.
"""

from __future__ import annotations
import argparse
import json
import os
import tarfile
import time
import xml.etree.ElementTree as ET
from array import array
from collections.abc import Sequence
from typing import Callable, Optional

import numpy as np

import loaders
from core import Ink, Stroke
from dataset_splits import MATHWRITING_SPLITS


def store_dir(cache: str, split: str) -> str:
    return os.path.join(cache, split, "mathwriting")


class _Writer:
    """Appends expressions for one split to its arrays."""

    def __init__(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        self.folder = folder
        self.points = open(os.path.join(folder, "points.f32"), "wb")
        self.times = open(os.path.join(folder, "times.f32"), "wb")
        self.n_points = 0
        self.stroke_ends = array("q")
        self.expr_ends = array("q")
        self.labels = []
        self.names = []

    def add(self, name: str, label: Optional[str], strokes) -> None:
        """`strokes`: one (points, 3) array of x, y, t per stroke (t NaN when the file has none)."""
        start = None
        for xyt in strokes:
            xyt = np.asarray(xyt, dtype=np.float64)
            if start is None:
                start = xyt[0, 2]
            self.points.write(np.ascontiguousarray(xyt[:, :2], dtype=np.float32).tobytes())
            self.times.write(np.ascontiguousarray(xyt[:, 2] - start, dtype=np.float32).tobytes())
            self.n_points += len(xyt)
            self.stroke_ends.append(self.n_points)
        self.expr_ends.append(len(self.stroke_ends))
        self.labels.append(label or "")
        self.names.append(name)

    def close(self) -> dict:
        self.points.close()
        self.times.close()
        np.save(os.path.join(self.folder, "stroke_ends.npy"), np.frombuffer(self.stroke_ends, dtype=np.int64))
        np.save(os.path.join(self.folder, "expr_ends.npy"), np.frombuffer(self.expr_ends, dtype=np.int64))
        with open(os.path.join(self.folder, "labels.json"), "w") as f:
            json.dump(self.labels, f)
        with open(os.path.join(self.folder, "names.json"), "w") as f:
            json.dump(self.names, f)
        meta = {"expressions": len(self.labels), "strokes": len(self.stroke_ends),
                "points": self.n_points, "times": True}
        with open(os.path.join(self.folder, "meta.json"), "w") as f:
            json.dump(meta, f)
        return meta


def _split_of(name: str) -> Optional[str]:
    return next((MATHWRITING_SPLITS[p] for p in name.split("/")[:-1] if p in MATHWRITING_SPLITS), None)


def _parse_member(item):
    """(split, name, label, strokes as (points, 3) arrays of x, y, t) for one InkML file,
    or None when it has too few strokes to keep. t is NaN for a trace without times."""
    name, data, min_strokes = item
    root = ET.fromstring(data)
    strokes, _ = loaders._parse_inkml_traces(root)
    if len(strokes) < min_strokes:
        return None
    annotations = loaders._inkml_annotations(root)
    arrays = [np.column_stack([np.asarray(s.x, dtype=np.float64), np.asarray(s.y, dtype=np.float64),
                               np.asarray(s.t if s.t is not None else [np.nan] * len(s.x), dtype=np.float64)])
              for s in strokes]
    return (_split_of(name), "/".join(name.split("/")[-2:]),
            annotations.get("normalizedLabel") or annotations.get("label"), arrays)


def build(src: str, cache: str, min_strokes: int = loaders.MIN_STROKES,
          limit_per_split: Optional[int] = None, log_every: int = 20000,
          workers: Optional[int] = None, batch: int = 4000) -> dict:
    """Stream the release (a .tgz or extracted directory) into per-split stores.

    Parsing the InkML is nearly all of the time — over 600k files — so one
    process reads the archive in order and `workers` processes (default: every
    core) parse it in batches of `batch` files. Results are written back in
    archive order, so the store is the same whatever the worker count.
    """
    workers = workers or os.cpu_count() or 1
    writers = {}
    seen = 0
    started = time.time()
    next_log = log_every

    def members():
        if os.path.isdir(src):
            for dirpath, _, files in os.walk(src):
                for fn in sorted(files):
                    if fn.endswith(".inkml"):
                        path = os.path.join(dirpath, fn)
                        rel = os.path.relpath(path, src).replace(os.sep, "/")
                        if _split_of(rel) is not None:
                            with open(path, "rb") as f:
                                yield rel, f.read()
            return
        with tarfile.open(src, "r:*") as tar:
            for member in tar:
                if member.isfile() and member.name.endswith(".inkml") and _split_of(member.name) is not None:
                    yield member.name, tar.extractfile(member).read()

    def flush(items, pool):
        chunk = max(1, len(items) // (workers * 4))
        results = pool.map(_parse_member, items, chunksize=chunk) if pool else map(_parse_member, items)
        for result in results:
            if result is None:
                continue
            split, name, label, arrays = result
            writer = writers.get(split)
            if writer is None:
                writer = writers[split] = _Writer(store_dir(cache, split))
            if limit_per_split is not None and len(writer.labels) >= limit_per_split:
                continue
            writer.add(name, label, arrays)

    pool = None
    if workers > 1:
        import multiprocessing
        pool = multiprocessing.get_context("spawn" if os.name == "nt" else "fork").Pool(workers)
    try:
        items = []
        for name, data in members():
            items.append((name, data, min_strokes))
            seen += 1
            if len(items) >= batch:
                flush(items, pool)
                items = []
                if log_every and seen >= next_log:
                    print(f"  mathwriting: {seen} read, {time.time() - started:.0f}s", flush=True)
                    next_log += log_every
        if items:
            flush(items, pool)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return {split: writer.close() for split, writer in writers.items()}


class MathPool(Sequence):
    """One split's expressions, each built as an Ink when indexed.

    Behaves as the list of Inks the generators expect (random.choice,
    random.sample, len, truthiness). `labels` lets a caller read every label
    without building the ink (geometry_diagrams.radian_expressions uses it), and
    `ink(i)` builds one directly. `redirect`, when set, maps each index asked for
    to the index actually served — how the synthesis runner steers picks away
    from items at their usage cap without touching the generators.
    """

    def __init__(self, folder: str, split: str):
        with open(os.path.join(folder, "meta.json")) as f:
            meta = json.load(f)
        self.split = split
        self.points = (np.memmap(os.path.join(folder, "points.f32"), dtype=np.float32, mode="r",
                                 shape=(meta["points"], 2))
                       if meta["points"] else np.zeros((0, 2), dtype=np.float32))
        # None for a store built before times were kept: its strokes come out with t None
        self.times = (np.memmap(os.path.join(folder, "times.f32"), dtype=np.float32, mode="r",
                                shape=(meta["points"],))
                      if meta.get("times") and meta["points"] else None)
        self.stroke_ends = np.load(os.path.join(folder, "stroke_ends.npy"), mmap_mode="r")
        self.expr_ends = np.load(os.path.join(folder, "expr_ends.npy"), mmap_mode="r")
        with open(os.path.join(folder, "labels.json")) as f:
            self.labels = json.load(f)
        self.redirect: Optional[Callable[[int], int]] = None

    def __len__(self) -> int:
        return len(self.labels)

    def item_id(self, index: int) -> str:
        return f"mw:{self.split}:{index}"

    def ink(self, index: int) -> Ink:
        first = int(self.expr_ends[index - 1]) if index else 0
        last = int(self.expr_ends[index])
        strokes = []
        for k in range(first, last):
            p0 = int(self.stroke_ends[k - 1]) if k else 0
            p1 = int(self.stroke_ends[k])
            xy = np.asarray(self.points[p0:p1], dtype=np.float64)
            t = None
            if self.times is not None:
                t = np.asarray(self.times[p0:p1], dtype=np.float64)
                t = None if np.isnan(t).any() else t.tolist()
            strokes.append(Stroke(x=xy[:, 0].tolist(), y=xy[:, 1].tolist(), t=t))
        return Ink(strokes=strokes, label=self.labels[index] or None, source="mathwriting",
                   meta={"item": self.item_id(index)})

    def __iter__(self):
        # Reading the whole pool is not drawing from it: going through
        # __getitem__ would send every index through `redirect`, which counts each
        # as served and swaps capped ones, so a filtered copy came back with repeats.
        for index in range(len(self)):
            yield self.ink(index)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.redirect is not None:
            index = self.redirect(index)
        return self.ink(index)


def open_pool(cache: str, split: str) -> Optional[MathPool]:
    folder = store_dir(cache, split)
    if not os.path.exists(os.path.join(folder, "meta.json")):
        return None
    return MathPool(folder, split)


def has_times(cache: str, split: str) -> bool:
    """Whether a split's store keeps each point's time; one built before times.f32 does not."""
    try:
        with open(os.path.join(store_dir(cache, split), "meta.json")) as f:
            return bool(json.load(f).get("times"))
    except (OSError, ValueError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="mathwriting-2024.tgz, or the extracted release")
    ap.add_argument("--out", required=True, help="cache root; stores go in <out>/<split>/mathwriting")
    ap.add_argument("--limit-per-split", type=int, default=None, help="for a quick test")
    ap.add_argument("--workers", type=int, default=None, help="parsing processes (default: every core)")
    args = ap.parse_args()
    started = time.time()
    result = build(args.src, args.out, limit_per_split=args.limit_per_split, workers=args.workers)
    print(json.dumps(result, indent=2))
    print(f"done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
