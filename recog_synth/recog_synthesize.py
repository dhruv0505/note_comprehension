"""
recog_synthesize.py — recognition samples from the source pools, for one task and one split,
generated in parallel.

    python recog_synthesize.py --task drawing  --split train --quickdraw ../note_synth/data/quickdraw.zip --out dataset
    python recog_synthesize.py --task textline --split train --cache ../note_synth/cache --out dataset
    python recog_synthesize.py --task formula  --split val   --cache ../note_synth/cache --out dataset

The dataset the tracker makes takes its textline and formula samples straight from the synthesized
pages instead (recog_pages.py, note_synth's synthesize.py --recognition); this makes them from the
pools, items dealt at random.

Tasks, prompts and images are recog_layout's; items and splits recog_pools'.
Under <out>/<split>/<subdir>/<task>/, subdir "recognition" unless --subdir names another (the dataset's train
textline pool pass writes to recognition_pool, beside the page samples' recognition/):

    images/<shard>/<task>_0000123.png   one sample, CANVAS square, RGB: ink coloured by time
                                        and pen movement (recog_layout)
    manifest.jsonl                      one row per sample, {"image", "prompt", "target",
                                        "task", "items"}: image relative to this folder,
                                        items the source ids top to bottom; a text sample
                                        also "marks", each row's mark kind or null
                                        (recog_marks, --mark-prob)
    shards/, done/                      per shard; a shard whose done file exists is skipped
    config.json                         the settings; a rerun with other settings is refused
    report.json                         pool size, samples, rows shrunk to fit, uses per item

Which items a sample holds. The pool is walked through a fresh shuffle each pass,
ROWS items a sample (one for a drawing) — or, for SHORT_PROB of text samples
(--short-prob), a number from 1 to ROWS drawn uniformly, filling the top slots of
the layout as the cascade draws a page's last group of lines — so every item is
used once a pass and equally often overall, and sample N holds the same items
whichever worker makes it (Dealer). --epochs sets how many passes (default one);
--samples overrides it. Rows are joined by recog_layout.SEP, a newline.
"""

from __future__ import annotations
import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import math
import os
import random
import time
import zlib
from collections import Counter

import numpy as np

import recog_layout as rl
import recog_marks as rm

SHARD_SAMPLES = 1000


def sample_seed(seed: int, split: str, task: str, index: int) -> int:
    digest = hashlib.sha1(f"recog:{seed}:{split}:{task}:{index}".encode()).hexdigest()
    return int(digest[:12], 16)


SHORT_PROB = 0.1       # share of text samples holding 1 to ROWS rows, drawn uniformly, instead of ROWS
COUNT_BLOCK = 4096


class Dealer:
    """The pool indices each sample holds: pass after pass through a shuffle of
    the pool, each sample taking the next count(index) of them — per_sample, or
    with chance short_prob a number from 1 to per_sample drawn uniformly. The
    counts come from a stream of their own, keyed like the shuffles and drawn in
    order, so sample N holds the same items whichever worker makes it, and every
    item is still used once a pass. A sample that straddles two passes and would
    hold one item twice takes a random other item for the repeat."""

    def __init__(self, n: int, per_sample: int, seed: int, split: str, task: str, short_prob: float = 0.0):
        if n < per_sample:
            raise ValueError(f"a pool of {n} items cannot fill {per_sample} rows")
        self.n, self.per_sample, self.short_prob = n, per_sample, float(short_prob)
        self._key = [seed, zlib.crc32(split.encode()), zlib.crc32(task.encode())]
        self._perms = {}
        self._count_stream = np.random.default_rng(self._key + [0x5EED, 1])     # a longer key than any shuffle's
        self._counts = np.zeros(0, dtype=np.int64)
        self._starts = np.zeros(1, dtype=np.int64)          # where each sample's items begin; the last, where they end

    def _extend(self, size: int) -> None:
        while len(self._counts) < size:
            draws = self._count_stream.random((COUNT_BLOCK, 2))
            short = 1 + np.minimum(self.per_sample - 1, np.floor(draws[:, 1] * self.per_sample))
            counts = np.where(draws[:, 0] < self.short_prob, short, self.per_sample).astype(np.int64)
            self._counts = np.concatenate([self._counts, counts])
            self._starts = np.concatenate([[0], np.cumsum(self._counts)])

    def count(self, index: int) -> int:
        self._extend(index + 1)
        return int(self._counts[index])

    def samples_for(self, positions: float) -> int:
        """How many samples it takes to deal `positions` items: one pass is the pool's size."""
        while self._starts[-1] < positions:
            self._extend(len(self._counts) + 1)
        return max(1, int(np.searchsorted(self._starts, positions, side="left")))

    def _perm(self, epoch: int):
        perm = self._perms.get(epoch)
        if perm is None:
            if len(self._perms) > 2:
                self._perms.clear()
            perm = self._perms[epoch] = np.random.default_rng(self._key + [epoch]).permutation(self.n)
        return perm

    def items(self, index: int):
        self._extend(index + 1)
        start, count = int(self._starts[index]), int(self._counts[index])
        picked, rng = [], None
        for position in range(start, start + count):
            item = int(self._perm(position // self.n)[position % self.n])
            if item in picked:
                rng = rng or random.Random(f"{self._key}:{index}")
                while item in picked:
                    item = rng.randrange(self.n)
            picked.append(item)
        return picked


def base_config(task: str, split: str, out: str, **given) -> dict:
    # timing "sections": a mark's points take one time after the row's writing (recog_layout.SECTION_PAUSE)
    config = {"task": task, "split": split, "out": out, "subdir": "recognition", "cache": None, "splits_file": None,
              "mathwriting": None, "quickdraw": None, "qd_per_category": None, "seed": 0,
              "canvas": rl.CANVAS, "rows": rl.ROWS, "row_height": rl.ROW_HEIGHT,
              "row_gap": rl.ROW_GAP, "pen": None, "ink": rl.INK, "mark_prob": None, "short_prob": None,
              "separator": rl.SEP, "timing": "sections"}
    unknown = set(given) - set(config)
    if unknown:
        raise TypeError(f"unknown settings {sorted(unknown)}")
    if task not in rl.TASKS:
        raise ValueError(f"unknown task {task!r}")
    config.update({k: v for k, v in given.items() if v is not None})
    if config["seed"] < 0:
        raise ValueError("--seed must be 0 or more")
    # text rows only: how often a row carries a mark as mark separation leaves it (recog_marks)
    if task in rl.TEXT_TASKS:
        config["mark_prob"] = float(rm.MARK_PROB if config["mark_prob"] is None else config["mark_prob"])
        if not 0.0 <= config["mark_prob"] <= 1.0:
            raise ValueError("--mark-prob must be between 0 and 1")
        # ... and how often a sample holds 1 to rows rows instead of all of them (Dealer)
        config["short_prob"] = float(SHORT_PROB if config["short_prob"] is None else config["short_prob"])
        if not 0.0 <= config["short_prob"] <= 1.0:
            raise ValueError("--short-prob must be between 0 and 1")
    else:
        config["mark_prob"] = config["short_prob"] = config["separator"] = None
    config["pen"] = [float(v) for v in (config["pen"] or
                                        (rl.TEXT_PEN if task in rl.TEXT_TASKS else rl.DRAWING_PEN))]
    if task in rl.TEXT_TASKS:
        rl.row_tops(config["rows"], config["row_height"], config["row_gap"], config["canvas"])  # raises if they do not fit
    return config


def per_sample(config: dict) -> int:
    return config["rows"] if config["task"] in rl.TEXT_TASKS else 1


def task_dir(out: str, split: str, task: str, subdir: str = "recognition") -> str:
    return os.path.join(out, split, subdir, task)


def load(config: dict):
    import recog_pools
    return recog_pools.load_pool(config["task"], config["split"], cache=config["cache"],
                                 splits_file=config["splits_file"], mathwriting=config["mathwriting"],
                                 quickdraw=config["quickdraw"], qd_per_category=config["qd_per_category"])


# ----------------------------------------------------------------------------
# Workers
# ----------------------------------------------------------------------------

_W: dict = {}


def _setup(pool, config: dict) -> None:
    _W.update(pool=pool, config=config,
              dealer=Dealer(len(pool), per_sample(config), config["seed"], config["split"], config["task"],
                            config.get("short_prob") or 0.0))


def _init(config: dict) -> None:
    sys.dont_write_bytecode = True
    _setup(load(config), config)


def _shard_manifest(folder: str, shard: int) -> str:
    return os.path.join(folder, "shards", f"manifest_{shard:05d}.jsonl")


def _done_path(folder: str, shard: int) -> str:
    return os.path.join(folder, "done", f"shard_{shard:05d}.json")


def _write_rows(path: str, rows) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="\n", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _run_shard(shard_task):
    shard, first, count = shard_task
    pool, config, dealer = _W["pool"], _W["config"], _W["dealer"]
    task, split = config["task"], config["split"]
    folder = task_dir(config["out"], split, task, config.get("subdir") or "recognition")
    rel_dir = f"images/{shard:05d}"
    os.makedirs(os.path.join(folder, rel_dir), exist_ok=True)
    lo, hi = config["pen"]
    rows, stats, min_height = [], Counter(), None
    started = time.time()

    for index in range(first, first + count):
        rng = random.Random(sample_seed(config["seed"], split, task, index))
        picked = dealer.items(index)
        inks = [pool.ink(i) for i in picked]
        labels = [pool.label(i) for i in picked]
        extra = {}
        if task in rl.TEXT_TASKS:
            pens = [rng.uniform(lo, hi) for _ in picked]
            # marks from a generator of their own, so a sample without one is drawn as it was before marks
            marker = random.Random(f"marks:{sample_seed(config['seed'], split, task, index)}")
            marks = []
            for k, pen in enumerate(pens):
                # one pen width at the size the row is drawn, in the item's units: row_fit's scale, pen for pen
                reach = lambda lines, pen=pen: pen / rl.row_fit(lines, pen, config["row_height"], config["canvas"])["scale"]
                inks[k], mark = rm.maybe_mark(inks[k], task, reach, marker, config["mark_prob"])
                marks.append(mark["kind"] if mark and mark["points"] else None)
                stats["mark_points"] += mark["points"] if mark else 0
            # a short sample's rows fill the top slots of the full layout, as the cascade draws a group of fewer lines
            image, placed = rl.text_sample(inks, pens, config["row_height"], config["row_gap"], config["canvas"],
                                           slots=config["rows"])
            target = rl.text_target(labels)
            stats["rows"] += len(placed)
            stats["short_samples"] += len(placed) < config["rows"]
            stats["rows_shrunk"] += sum(row["shrunk"] for row in placed)
            stats["rows_marked"] += sum(mark is not None for mark in marks)
            lowest = min(row["box"][3] - row["box"][1] for row in placed)
            min_height = lowest if min_height is None else min(min_height, lowest)
            extra["marks"] = marks
        else:
            image, _ = rl.drawing_sample(inks[0], rng.uniform(lo, hi), config["canvas"])
            target = labels[0]
            stats["rows"] += 1
        assert image.size == (config["canvas"], config["canvas"]) and image.mode == "RGB"
        rel = f"{rel_dir}/{task}_{index:07d}.png"
        image.save(os.path.join(folder, rel))
        rows.append({"image": rel, "prompt": rl.PROMPTS[task], "target": target, "task": task,
                     "items": [pool.item_id(i) for i in picked], **extra})
        stats["samples"] += 1

    _write_rows(_shard_manifest(folder, shard), rows)
    seconds = round(time.time() - started, 2)
    record = {"shard": shard, "first": first, "count": count, **stats,
              "min_row_height": min_height, "seconds": seconds}
    done = _done_path(folder, shard)
    os.makedirs(os.path.dirname(done), exist_ok=True)
    with open(done + ".tmp", "w") as f:
        json.dump(record, f)
    os.replace(done + ".tmp", done)             # written last: a shard with this file is done
    return shard, dict(stats), seconds


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

def join_manifests(folder: str, shards) -> int:
    """Every shard's rows, in shard order, into `folder`/manifest.jsonl."""
    target = os.path.join(folder, "manifest.jsonl")
    rows = 0
    with open(target + ".tmp", "w", newline="\n", encoding="utf-8") as joined:
        for shard, _, _ in shards:
            with open(_shard_manifest(folder, shard), encoding="utf-8") as part:
                for line in part:
                    if line.strip():
                        joined.write(line if line.endswith("\n") else line + "\n")
                        rows += 1
    os.replace(target + ".tmp", target)
    return rows


def write_report(folder: str, config: dict, pool, shards) -> dict:
    totals, min_height = Counter(), None
    for shard, _, _ in shards:
        with open(_done_path(folder, shard)) as f:
            record = json.load(f)
        for key in ("samples", "rows", "rows_shrunk", "rows_marked", "mark_points", "short_samples", "seconds"):
            totals[key] += record.get(key, 0)
        if record.get("min_row_height") is not None:
            min_height = record["min_row_height"] if min_height is None else min(min_height, record["min_row_height"])

    text = config["task"] in rl.TEXT_TASKS
    report = {"task": config["task"], "split": config["split"], "prompt": rl.PROMPTS[config["task"]],
              "target_separator": rl.SEP if text else None,
              "pool_items": len(pool), "samples": totals["samples"], "rows": totals["rows"],
              "uses_per_item": round(totals["rows"] / len(pool), 3),
              "canvas": config["canvas"], "pen": config["pen"],
              "worker_seconds": round(totals["seconds"], 1),
              "seconds_per_sample": round(totals["seconds"] / totals["samples"], 4) if totals["samples"] else None}
    if text:
        report.update(rows_per_sample=config["rows"], row_height=config["row_height"], row_gap=config["row_gap"],
                      row_tops=rl.row_tops(config["rows"], config["row_height"], config["row_gap"], config["canvas"]),
                      rows_shrunk=totals["rows_shrunk"], min_row_height=min_height,
                      mark_prob=config["mark_prob"], rows_marked=totals["rows_marked"], mark_points=totals["mark_points"],
                      short_prob=config["short_prob"], short_samples=totals["short_samples"])
    else:
        counts = Counter(pool.label(i) for i in range(len(pool)))
        report.update(categories=len(counts), items_per_category={"min": min(counts.values()),
                                                                  "max": max(counts.values())})
    with open(os.path.join(folder, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    return report


def _check_config(folder: str, settings: dict) -> None:
    path = os.path.join(folder, "config.json")
    if os.path.exists(path):
        with open(path) as f:
            previous = json.load(f)
        if previous != settings:
            changed = sorted(k for k in set(previous) | set(settings) if previous.get(k) != settings.get(k))
            raise SystemExit(f"{folder} was started with other settings ({', '.join(changed)}); "
                             "finish it with the same ones or use another --out")
        return
    with open(path, "w") as f:
        json.dump(settings, f, indent=1)


def run(config: dict, samples: int = None, epochs: float = 1.0, workers: int = 1,
        shard_samples: int = SHARD_SAMPLES, pool=None, log=print) -> dict:
    """Generate, join and report. `pool` skips loading, and only works with one
    worker: other workers load their own from `config`."""
    if pool is None:
        pool = load(config)
    elif workers > 1:
        raise ValueError("a pool passed in is not shared with worker processes; use workers=1")
    size = per_sample(config)
    if len(pool) < size:
        raise SystemExit(f"{config['task']} {config['split']}: {len(pool)} usable items, {size} needed a sample")
    try:        # the ink's colour needs every point's time: a pool without them fails here, not in a worker
        for index in range(min(len(pool), 5)):
            rl.polylines(pool.ink(index))
    except ValueError as error:
        raise SystemExit(f"{config['task']} {config['split']}: {error}. Rebuild note_synth's split caches "
                         "(split_pools.py), or read the archives, so strokes keep their times")
    # --epochs passes: as many samples as it takes to deal that many items, short samples included
    dealer = Dealer(len(pool), size, config["seed"], config["split"], config["task"], config.get("short_prob") or 0.0)
    total = samples if samples is not None else dealer.samples_for(epochs * len(pool))
    folder = task_dir(config["out"], config["split"], config["task"], config.get("subdir") or "recognition")
    os.makedirs(folder, exist_ok=True)
    _check_config(folder, dict(config, pool_items=len(pool), samples=total, shard_samples=shard_samples))

    shards = [(n, first, min(shard_samples, total - first))
              for n, first in enumerate(range(0, total, shard_samples))]
    todo = [s for s in shards if not os.path.exists(_done_path(folder, s[0]))]
    log(f"{config['task']} {config['split']}: {len(pool)} items, {total} samples in {len(shards)} shards, "
        f"{len(todo)} to do, {workers} workers -> {folder}")

    started, made = time.time(), 0

    def progress(shard, stats, seconds):
        nonlocal made
        made += stats.get("samples", 0)
        elapsed = max(time.time() - started, 1e-6)
        left = sum(s[2] for s in todo) - made
        log(f"  shard {shard}: {stats.get('samples', 0)} samples in {seconds:.0f}s | "
            f"{made / elapsed:.1f} samples/s, ~{left / max(made / elapsed, 1e-6) / 60:.0f} min left")

    if todo and workers <= 1:
        _setup(pool, config)
        for task in todo:
            progress(*_run_shard(task))
    elif todo:
        import multiprocessing
        context = multiprocessing.get_context("spawn" if os.name == "nt" else "fork")
        with context.Pool(workers, initializer=_init, initargs=(config,)) as processes:
            for result in processes.imap_unordered(_run_shard, todo):
                progress(*result)

    rows = join_manifests(folder, shards)
    log(f"joined {rows} rows -> {os.path.join(folder, 'manifest.jsonl')}")
    return write_report(folder, config, pool, shards)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=rl.TASKS)
    ap.add_argument("--split", required=True, choices=("train", "val", "test"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--subdir", default="recognition", help="the folder under <out>/<split>/ the task's samples go in")
    ap.add_argument("--cache", default=None, help="note_synth's split_pools.py cache root")
    ap.add_argument("--splits-file", default=None, help="note_synth's splits.json (archive mode, textline)")
    ap.add_argument("--mathwriting", default=None, help="MathWriting release; read instead of the cache's store")
    ap.add_argument("--quickdraw", default=None, help="QuickDraw raw zip or directory; read instead of the cache")
    ap.add_argument("--qd-per-category", type=int, default=None,
                    help="QuickDraw drawings a category from the archive (default 1000 train, 100 val/test)")
    ap.add_argument("--samples", type=int, default=None, help="samples to make (default: --epochs passes)")
    ap.add_argument("--epochs", type=float, default=1.0, help="passes over the pool")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-samples", type=int, default=SHARD_SAMPLES)
    ap.add_argument("--canvas", type=int, default=rl.CANVAS)
    ap.add_argument("--rows", type=int, default=rl.ROWS)
    ap.add_argument("--row-height", type=int, default=rl.ROW_HEIGHT)
    ap.add_argument("--row-gap", type=int, default=rl.ROW_GAP)
    ap.add_argument("--pen", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help=f"pen width range, pixels (default {rl.TEXT_PEN} text, {rl.DRAWING_PEN} drawing)")
    ap.add_argument("--mark-prob", type=float, default=None,
                    help=f"text: share of rows given a mark as mark separation leaves it (default {rm.MARK_PROB})")
    ap.add_argument("--short-prob", type=float, default=None,
                    help=f"text: share of samples holding 1 to --rows rows instead of all of them (default {SHORT_PROB})")
    args = ap.parse_args()

    config = base_config(args.task, args.split, args.out, subdir=args.subdir, cache=args.cache, splits_file=args.splits_file,
                         mathwriting=args.mathwriting, quickdraw=args.quickdraw,
                         qd_per_category=args.qd_per_category, seed=args.seed, canvas=args.canvas,
                         rows=args.rows, row_height=args.row_height, row_gap=args.row_gap, pen=args.pen,
                         mark_prob=args.mark_prob, short_prob=args.short_prob)
    report = run(config, samples=args.samples, epochs=args.epochs, workers=args.workers,
                 shard_samples=args.shard_samples)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
