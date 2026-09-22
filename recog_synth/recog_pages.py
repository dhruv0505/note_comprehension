"""
recog_pages.py — recognition samples straight from a synthesized page, read the way the cascade reads a page.

    groups = page_groups(objects)          # a page's objects, as note_synth's page_export lists them
    rows, stats = write_samples(objects, "dataset/train/recognition", "images/00003", "p0000123")

The cascade (training/recog_eval.py) reads a page's textlines ten at a time, then its formulas. A row is every stroke
of the page with at least MIN_INSIDE of its points inside the line's box, taken whole: the line's own, and any of the
lines above and below that reach far enough in. A group is drawn by group_image. A sample here is that same read with
the ground truth's boxes: the lines of each task in the order the page lists them (note_synth lists a block's lines
top to bottom), ten a sample, the page's last sample holding what is left. Marks are left out altogether, as if they
were never made: the cascade takes them out with the mark separation network, and the ground truth knows which strokes
are marks. A line whose transcription can't go into a target (recog_layout.clean_label) is not a row of its own,
though its strokes are still read in the boxes of the lines beside it.

Times are never made up. Every line keeps its recorded times and moves only as a whole, into the order a page is
written (write_order), since the page JSON times each line from its own first point. A line without recorded times
(a geometry figure's letters, written from NIST character images) is left out altogether.

Pen. The sources record x, y and time, not width. Every row is drawn with the same pen, PEN pixels in the drawn row
whatever the line's size (row_fit thins it only for a row shrunk to fit the canvas's width): the page's own pen scaled
with its line blew small lines up into unreadable blots.

training/recog_eval.py draws what the cascade reads with this module, and note_synth's synthesize.py draws the samples
with it, so both are drawn by the same code. Uses numpy, PIL and recog_layout only.
"""

from __future__ import annotations
import os
from collections import Counter

import numpy as np
from PIL import Image

import recog_layout as rl

PEN = 2.0                                   # every row's pen, in the drawn row's pixels
MIN_INSIDE = 0.5                            # share of a stroke's points inside a box for the box to take it
TEXT_CLASSES = rl.TEXT_TASKS                # textline, formula: the page JSON's class names too
MARK_CLASSES = ("enclosure", "line_mark")


def page_text(objects):
    """A page's textlines, formulas and marks as [{class, bbox, text, strokes, parent}], from its objects as note_synth's
    page_export lists them: each stroke an (n, 3) float array of x, y and time, NaN for points without one (a mark's
    always). Coordinates are kept as they are."""
    text = []
    for obj in objects:
        if obj["class"] not in TEXT_CLASSES + MARK_CLASSES or "ink" not in obj:
            continue
        strokes = []
        for stroke in obj["ink"]:
            if stroke:
                points = np.asarray(stroke, dtype=np.float64)
                t = points[:, 2] if points.shape[1] > 2 else np.full(len(points), np.nan)
                strokes.append(np.column_stack([points[:, 0], points[:, 1], t]))
        text.append({"class": obj["class"], "bbox": tuple(obj["bbox"]), "text": obj.get("text"), "strokes": strokes,
                     "parent": obj.get("parent")})
    return text


def write_order(text):
    """The page's ink (page_text's) with each line's recorded times moved as one section into the order a page is
    written: the lines of a block top to bottom, a pause (recog_layout.section_pause, seeded by the page's ink) between
    one and the next, blocks one after another as the page lists them. The page JSON times every line from its own first
    point (note_synth page_export), so read as it is, the lines above and below a row would be written at the same
    moments as the row. Nothing inside a line changes, and ink without times (marks) is left as it is."""
    timed = [o for o in text if o["strokes"] and not any(np.isnan(s[:, 2]).any() for s in o["strokes"])]
    if not timed:
        return text
    blocks = {}
    for o in timed:
        blocks.setdefault(o.get("parent") if o.get("parent") is not None else ("alone", id(o)), []).append(o)
    seed = int(round(sum(float(np.abs(s[:, :2]).sum()) for o in timed for s in o["strokes"]) * 10)) % 2 ** 32
    rng = np.random.default_rng(seed)
    moved, clock, previous = {}, None, 0.0
    for block in blocks.values():
        for o in sorted(block, key=lambda o: (o["bbox"][1], o["bbox"][0])):
            times = np.concatenate([s[:, 2] for s in o["strokes"]])
            lo, hi = float(times.min()), float(times.max())
            start = 0.0 if clock is None else clock + rl.section_pause(float(rng.random()), hi - lo, previous)
            shift = start - lo
            moved[id(o)] = dict(o, strokes=[np.column_stack([s[:, :2], s[:, 2] + shift]) for s in o["strokes"]])
            clock, previous = hi + shift, hi - lo
    return [moved.get(id(o), o) for o in text]


class PageInk:
    """Every stroke of a page's lines (and marks, when given), to pick out what lies in a box."""

    def __init__(self, text):
        self.strokes = [stroke for obj in text for stroke in obj["strokes"]]
        self.objects = [k for k, obj in enumerate(text) for _ in obj["strokes"]]      # which of `text` each stroke is
        if self.strokes:
            self.sizes = np.array([len(s) for s in self.strokes])
            self.xy = np.concatenate(self.strokes)[:, :2]
            self.owner = np.repeat(np.arange(len(self.strokes)), self.sizes)

    def taken(self, box):
        """The indices of the strokes with at least MIN_INSIDE of their points inside `box`."""
        if not self.strokes:
            return []
        x0, y0, x1, y1 = box
        hit = (self.xy[:, 0] >= x0) & (self.xy[:, 0] <= x1) & (self.xy[:, 1] >= y0) & (self.xy[:, 1] <= y1)
        share = np.bincount(self.owner[hit], minlength=len(self.strokes)) / self.sizes
        return np.flatnonzero(share >= MIN_INSIDE).tolist()

    def inside(self, box):
        return [self.strokes[k] for k in self.taken(box)]


def row_lines(strokes):
    """A row's (n, 3) strokes as recog_layout draws them. Times are never made up: points without one (a mark's, which
    note_synth builds rather than records) are one section made after the row's writing, all at a single time a pause
    (recog_layout.section_pause, seeded by the row's ink) after its latest recorded point; a row with no times at all
    takes its points' drawing order."""
    if not any(np.isnan(s[:, 2]).any() for s in strokes):
        return strokes
    times = np.concatenate([s[:, 2] for s in strokes])
    times = times[~np.isnan(times)]
    if not len(times):
        lines, order = [], 0
        for s in strokes:
            lines.append(np.column_stack([s[:, :2], np.arange(order, order + len(s), dtype=np.float64)]))
            order += len(s)
        return lines
    seed = int(round(float(np.abs(np.concatenate(strokes)[:, :2]).sum()) * 10)) % 2 ** 32
    lo, hi = float(times.min()), float(times.max())
    when = hi + rl.section_pause(float(np.random.default_rng(seed).random()), hi - lo)
    lines = []
    for s in strokes:
        t = s[:, 2].copy()
        t[np.isnan(t)] = when
        lines.append(np.column_stack([s[:, :2], t]))
    return lines


def group_image(rows):
    """One recognition image: each row's strokes (maybe none) in the ten-row layout's slots, from the top, each drawn
    ROW_HEIGHT tall (recog_layout.row_fit) with the pen PEN, left-aligned; slots past the last row stay white."""
    image = Image.new("RGB", (rl.CANVAS, rl.CANVAS), rl.BACKGROUND)
    for strokes, top in zip(rows, rl.row_tops()):
        if not strokes:
            continue
        lines = row_lines(strokes)
        placement = rl.row_fit(lines, PEN)
        image.paste(rl.draw(lines, placement), (0, top + (rl.ROW_HEIGHT - placement["size"][1]) // 2))
    return image


def page_groups(objects, tasks=TEXT_CLASSES, untimed=None):
    """{task: [{"targets", "rows", "boxes", "others"}]}: a page's samples read with the ground truth's boxes (module
    docstring). "rows" holds each row's strokes as the cascade reads them, "others" how many of them are other lines'.
    A line whose ink has no recorded times (a geometry figure's letters, written from NIST character images) can't be
    coloured, and times are never made up, so it is left out altogether: not a row, and not in the rows beside it.
    `untimed`, a dict, counts them by class."""
    text = [obj for obj in page_text(objects) if obj["class"] in TEXT_CLASSES]        # the marks never happened
    timed = [bool(obj["strokes"]) and not any(np.isnan(s[:, 2]).any() for s in obj["strokes"]) for obj in text]
    if untimed is not None:
        for obj, ok in zip(text, timed):
            if not ok and obj["strokes"]:
                untimed[obj["class"]] = untimed.get(obj["class"], 0) + 1
    text = [obj for obj, ok in zip(text, timed) if ok]
    ordered = write_order(text)
    ink = PageInk(ordered)
    groups = {}
    for task in tasks:
        lines = [(k, rl.clean_label(obj["text"])) for k, obj in enumerate(ordered)
                 if obj["class"] == task and obj["strokes"]]
        lines = [(k, label) for k, label in lines if label is not None]
        groups[task] = []
        for first in range(0, len(lines), rl.ROWS):
            chunk = lines[first:first + rl.ROWS]
            taken = [ink.taken(ordered[k]["bbox"]) for k, _ in chunk]
            groups[task].append({"targets": [label for _, label in chunk],
                                 "rows": [[ink.strokes[i] for i in t] for t in taken],
                                 "boxes": [[round(float(v), 1) for v in ordered[k]["bbox"]] for k, _ in chunk],
                                 "others": [sum(ink.objects[i] != k for i in t) for (k, _), t in zip(chunk, taken)]})
    return groups


def write_samples(objects, folder, rel_dir, name, tasks=TEXT_CLASSES):
    """(rows, stats): a page's samples (page_groups) drawn and saved as <folder>/<task>/<rel_dir>/<name>_NN.png, with
    {task: [manifest rows]} — {"image" (relative to <folder>/<task>), "prompt", "target", "task", "page": name,
    "boxes"} — and {task: Counter of samples, rows, short_samples, rows_with_other_lines, lines_without_times}."""
    rows, stats, untimed = {}, {}, {}
    for task, groups in page_groups(objects, tasks, untimed).items():
        rows[task], stats[task] = [], Counter({"lines_without_times": untimed.get(task, 0)})
        if groups:
            os.makedirs(os.path.join(folder, task, rel_dir), exist_ok=True)
        for k, group in enumerate(groups):
            rel = f"{rel_dir}/{name}_{k:02d}.png"
            group_image(group["rows"]).save(os.path.join(folder, task, rel))
            rows[task].append({"image": rel, "prompt": rl.PROMPTS[task], "target": rl.SEP.join(group["targets"]),
                               "task": task, "page": name, "boxes": group["boxes"]})
            count = stats[task]
            count["samples"] += 1
            count["rows"] += len(group["targets"])
            count["short_samples"] += len(group["targets"]) < rl.ROWS
            count["rows_with_other_lines"] += sum(n > 0 for n in group["others"])
    return rows, stats
