"""
seg_eval.py — validation metrics on generated segmentation outputs, separate from the training loss.

    python seg_eval.py --checkpoint runs/seg/best --pages val_pages/             # the cascade on whole pages
    python seg_eval.py --checkpoint runs/seg/best --pages test_pages/ --out test_metrics.json
    python seg_eval.py --checkpoint runs/seg/best                                # manifest rows (tail 1%)
    python seg_eval.py --checkpoint none --pages val_pages/ --max-pages 20       # the base model

Two ways to score a checkpoint; train.py runs both at the end of every epoch.

Rows (teacher-forced): each manifest row's prompt on its own image, as in
training, so a level-1 crop is cut from its ground-truth level-2 box. Scored in
the row image's pixels.

Pages (the cascade, as the model is used): each page image, any size (e.g.
1600 x 1600), is resized to 768 x 768 and asked
    LEVEL2_PROMPT   -> textblock, drawing, geometry, table, flowchart boxes
    ARROW_PROMPT    -> arrows
Every *predicted* textblock, table, geometry and flowchart box (not the ground
truth's) is then cut out of the full-resolution page, scaled so its longer side
is 768 and centred on white (seg_prompts.crop_transform / sample_image), and
asked its LEVEL1_PROMPTS prompt -> textlines, formulas, arcs, segments,
vertices, enclosures, line marks. Every prediction is mapped back to page pixels and scored against
every object on the page at once, so a missed or misplaced level-2 box costs its
level-1 objects too. A drawing has no level-1 prompt.

Recognition (recog_eval.py), when validate_pages is given it: the predicted textlines, then formulas, ten
at a time, drawn from the page's own strokes as the recognition data draws its samples and read with its
prompts. recog_loss/textline and recog_loss/formula are the token cross-entropy of the matched ground-truth
transcriptions; recog_cer/... is the character error rate of the text generated for them, every ground-truth
line no detected box matched counted as read as nothing (recog_missed/...), and recog_acc/... the share of its
characters read correctly (correct characters over total characters, recog_correct/... and recog_total/...;
recog_eval.score_rows says how they are counted). A kept page sample carries its reads
as "recognition" and those lines as "recognition_missed". It needs "text" and "ink" on the page JSON's textlines and formulas; a page without them
is not read. The ink of enclosures and line marks, where the JSON has it, is read with the lines it lies in;
--mark-model (the mark separation network's model.pt) takes it out of each detected line first, as the
cascade is meant to run (recog_eval.MarkRemover).

Page folder: each page image beside a JSON of the same stem, as render_page_val.py
writes them from note_synth (data/val_pages/), objects in any order:
    {"image_size": [1600, 1600], "page_size": [1600, 1600],   page_size: the coordinates' units,
                                                              scaled onto the image (absent: pixels)
     "objects": [
       {"id": 0, "level": 2, "class": "textblock", "bbox": [x0, y0, x1, y1]},
       {"id": 1, "level": 1, "class": "textline", "parent": 0, "bbox": [x0, y0, x1, y1],
        "text": "its transcription", "ink": [[[x, y, t], ...], ...]},       a formula's text: its LaTeX
       {"id": 2, "level": 1, "class": "vertex", "parent": 5, "position": [x, y]},
       {"id": 4, "level": 1, "class": "arc", "parent": 5, "points": [x1, y1, x2, y2, x3, y3]},
       {"id": 6, "level": 1, "class": "segment", "parent": 5, "points": [x1, y1, x2, y2]},
       {"id": 3, "level": 0, "class": "arrow", "start": [x, y], "end": [x, y], "bbox": [...]}]}
Scoring reads class and bbox / position / points / start + end. A vertex may give a zero-area
bbox instead of a position; an arrow must give start (its tail) and end (its head),
and a page with one that does not is an error. Other keys (id, level, parent, ...)
and classes no prompt asks for (word, unknown) are ignored; "line_mark" is "line mark". The prompts below are copied from
note_synth/seg_prompts.py and must match the training manifest word for word
(train.py warns when they do not).

Outputs are generated greedily and parsed back into objects:
    box      <loc_x0><loc_y0><loc_x1><loc_y1>class    textblock drawing geometry table flowchart
                                                      textline formula enclosure "line mark"
    vertex   <loc_x><loc_y>vertex
    arc      <loc_x1><loc_y1><loc_x2><loc_y2><loc_x3><loc_y3>arc
                                                      start, middle, end, counterclockwise
    segment  <loc_x1><loc_y1><loc_x2><loc_y2>segment  its two ends
    arrow    <loc_xs><loc_ys><loc_xe><loc_ye>arrow    start, then end

Parse validity: an output is valid when it is "</s><s>", then whole objects only,
then "</s>" (padding after it is fine). Each object is exactly its class's number
of loc tokens followed by a class its prompt asks for, a box with x0 <= x1 and
y0 <= y1. Anything else makes it invalid: stray text or tokens, coordinates with
no class, an unknown class, running out of --max-new-tokens. Malformed pieces are
counted by reason (and by class, where the class is known) and left out of
matching. The well-formed objects of an invalid output still count. Pages report
the rate for each stage (level2, arrow, level1) as well as overall.

Matching, per image and per class, greedy best-first and one-to-one, in pixels:
  box      IoU >= --iou (0.5). Rows: a coordinate k covers grid cell [k, k + 1).
           Pages: a predicted coordinate sits at its cell's centre, and a box thinner
           than --page-min-box-px (an underline can be a pixel or two tall) is widened
           to it about its centre, prediction and ground truth alike
  vertex   distance <= --vertex-px (rows) / --page-vertex-px (pages)
  arc      start and end each within that same vertex distance, in order, and the middle
           within it of the true arc anywhere between them (arc_distance): any point on the
           arc describes it, the labelled one being only its angular midpoint. A middle
           closer to an end than half the true middle's distance to it does not match
  segment  both ends within the vertex distance, either way round (a segment
           has no direction)
  arrow    start-to-start AND end-to-end each within --arrow-px / --page-arrow-px;
           an arrow predicted end-to-start does not match
TP / FP / FN summed over the whole set give each class's precision, recall and F1.
mean_f1 is the unweighted mean of F1 over the classes present in the ground truth,
a fixed set for a fixed validation set, so checkpoints compare on the same classes;
mean_f1_by_group splits it into level2, level1 and arrow. A class only ever
predicted still reports its FP but is left out of the means.
"""

from __future__ import annotations
import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from seg_dataset import DEFAULT_MANIFEST, GRID, Collator

LEVEL2_PROMPT = ("Where are the level 2 objects? "
                 "Detect multiple textblock, drawing, geometry, table, flowchart.")
ARROW_PROMPT = "Where are the arrows? Detect multiple arrow."
LEVEL1_PROMPTS = {
    "textblock": "Where are the level 1 objects in textblock? Detect multiple textline, formula, enclosure, line mark.",
    "table": "Where are the level 1 objects in table? Detect multiple textline, formula.",
    "geometry": "Where are the level 1 objects in geometry? Detect multiple textline, arc, segment, vertex.",
    "flowchart": "Where are the level 1 objects in flowchart? Detect multiple textline, formula, enclosure.",
}

LEVEL2_CLASSES = ("textblock", "drawing", "geometry", "table", "flowchart")
LEVEL1_CLASSES = ("textline", "formula", "arc", "segment", "vertex", "enclosure", "line mark")
GROUPS = {"level2": LEVEL2_CLASSES, "level1": LEVEL1_CLASSES, "arrow": ("arrow",)}
KIND = {**{name: "box" for name in LEVEL2_CLASSES + LEVEL1_CLASSES},
        "arc": "arc", "segment": "segment", "vertex": "point", "arrow": "arrow"}
ARITY = {"box": 4, "point": 2, "arrow": 4, "arc": 6, "segment": 4}
TRUTH_CLASS = {"line_mark": "line mark"}
TOKEN = re.compile(r"<loc_(\d+)>|(<[^<>]*>)|([^<]+)|(<)")
PROMPT_CLASSES = re.compile(r"Detect multiple (.+?)\.?\s*$")
START, EOS, PAD = "</s><s>", "</s>", "<pad>"      # decoder start token, forced <s>; end; padding
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def arc_distance(point, arc) -> float:
    """How far `point` is from the arc that `arc`'s start, middle and end ([x1, y1, x2, y2, x3, y3]) describe: from
    the start through the middle to the end. A full circle starts and ends at one point with its middle opposite;
    three points in a row are a straight stroke from start to end."""
    (sx, sy), (mx, my), (ex, ey) = arc[0:2], arc[2:4], arc[4:6]
    px, py = point
    if math.hypot(sx - ex, sy - ey) < 1e-6:
        cx, cy = (sx + mx) / 2.0, (sy + my) / 2.0
        return abs(math.hypot(px - cx, py - cy) - math.hypot(sx - cx, sy - cy))
    det = 2.0 * (sx * (my - ey) + mx * (ey - sy) + ex * (sy - my))
    if abs(det) < 1e-9:
        vx, vy = ex - sx, ey - sy
        t = max(0.0, min(1.0, ((px - sx) * vx + (py - sy) * vy) / (vx * vx + vy * vy)))
        return math.hypot(px - sx - t * vx, py - sy - t * vy)
    s2, m2, e2 = sx * sx + sy * sy, mx * mx + my * my, ex * ex + ey * ey
    cx = (s2 * (my - ey) + m2 * (ey - sy) + e2 * (sy - my)) / det
    cy = (s2 * (ex - mx) + m2 * (sx - ex) + e2 * (mx - sx)) / det
    angle = lambda x, y: math.atan2(y - cy, x - cx)
    start, turn = angle(sx, sy), 2.0 * math.pi
    to_middle, to_end, to_point = ((angle(x, y) - start) % turn for x, y in ((mx, my), (ex, ey), (px, py)))
    # the sweep that passes the middle: angles increasing from the start, or else decreasing
    on = to_point <= to_end if to_middle <= to_end else (turn - to_point) % turn <= (turn - to_end) % turn
    if on:
        return abs(math.hypot(px - cx, py - cy) - math.hypot(sx - cx, sy - cy))
    return min(math.hypot(px - sx, py - sy), math.hypot(px - ex, py - ey))


def prompt_classes(prompt: str):
    """The classes a prompt asks for ("... Detect multiple a, b, c."), or None."""
    match = PROMPT_CLASSES.search(prompt)
    return tuple(name.strip() for name in match.group(1).split(",")) if match else None


# ----------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------

class Parsed:
    def __init__(self):
        self.objects = []       # (class, grid coordinates), well-formed only
        self.errors = []        # (reason, class or None)

    @property
    def valid(self):
        return not self.errors


def parse_output(text: str, allowed=None, generated=True, truncated=False) -> Parsed:
    """Objects in a loc-token string. `generated` expects the decoder's framing
    (START ... EOS, then padding); a manifest target converted by to_loc_target has none."""
    out = Parsed()
    if generated:
        if truncated:
            out.errors.append(("truncated", None))
        if text.startswith(START):
            text = text[len(START):]
        else:
            out.errors.append(("bad_start", None))
        end = text.find(EOS)
        if end < 0:
            if not truncated:
                out.errors.append(("no_eos", None))
        else:
            if text[end + len(EOS):].replace(PAD, "").strip():
                out.errors.append(("after_eos", None))
            text = text[:end]

    locs = []
    for match in TOKEN.finditer(text):
        loc, _, words, _ = match.groups()
        if loc is not None and int(loc) < GRID:
            locs.append(int(loc))
            continue
        if words is not None:
            name = " ".join(words.split())
            if not name:
                continue
            kind = KIND.get(name)
            if kind is None:
                out.errors.append(("unknown_class", None))
            elif allowed is not None and name not in allowed:
                out.errors.append(("class_not_in_prompt", name))
            elif len(locs) != ARITY[kind]:
                out.errors.append(("wrong_coord_count", name))
            elif kind == "box" and (locs[0] > locs[2] or locs[1] > locs[3]):
                out.errors.append(("inverted_box", name))
            else:
                out.objects.append((name, tuple(locs)))
            locs = []
            continue
        out.errors.append(("stray_token", None))
        if locs:
            out.errors.append(("orphan_coords", None))
            locs = []
    if locs:
        out.errors.append(("trailing_coords", None))
    return out


# ----------------------------------------------------------------------------
# Matching and metrics
# ----------------------------------------------------------------------------

def _widen(box, minimum):
    x0, y0, x1, y1 = box
    if x1 - x0 < minimum:
        x0, x1 = (x0 + x1 - minimum) / 2, (x0 + x1 + minimum) / 2
    if y1 - y0 < minimum:
        y0, y1 = (y0 + y1 - minimum) / 2, (y0 + y1 + minimum) / 2
    return x0, y0, x1, y1


def box_iou(a, b, min_extent=0.0) -> float:
    a, b = _widen(a, min_extent), _widen(b, min_extent)
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def grid_to_pixels(objects, size):
    """Row objects in the image's pixels: a box as the cells it covers, a point,
    arc point, segment end or arrow end at its cell's centre."""
    sx, sy = size[0] / GRID, size[1] / GRID
    out = []
    for name, c in objects:
        if KIND[name] == "box":
            out.append((name, (c[0] * sx, c[1] * sy, (c[2] + 1) * sx, (c[3] + 1) * sy)))
        else:
            out.append((name, tuple((k + 0.5) * (sx if i % 2 == 0 else sy) for i, k in enumerate(c))))
    return out


class SegMetrics:
    """Accumulates parse validity and per-class TP / FP / FN over rows or pages."""

    def __init__(self, iou=0.5, vertex_px=10.0, arrow_px=15.0, min_box_px=0.0, unit="rows"):
        self.iou, self.vertex_px, self.arrow_px, self.min_box_px = iou, vertex_px, arrow_px, min_box_px
        self.unit, self.units = unit, 0
        self.counts = defaultdict(lambda: [0, 0, 0])        # class -> tp, fp, fn
        self.truth_classes = set()
        self.outputs, self.valid = Counter(), Counter()     # by stage
        self.errors, self.malformed = Counter(), Counter()
        self.truth_invalid = 0
        # task -> loss sum, target tokens, images, rows, matched, edits, target characters, missed lines,
        # correct characters, total characters
        self.recog = defaultdict(lambda: [0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def empty_like(self):
        return SegMetrics(self.iou, self.vertex_px, self.arrow_px, self.min_box_px, self.unit)

    def state(self) -> dict:
        """Every count, as plain data (for gathering across processes)."""
        return {"units": self.units, "counts": {k: list(v) for k, v in self.counts.items()},
                "truth_classes": sorted(self.truth_classes), "outputs": dict(self.outputs),
                "valid": dict(self.valid), "errors": dict(self.errors), "malformed": dict(self.malformed),
                "truth_invalid": self.truth_invalid, "recog": {k: list(v) for k, v in self.recog.items()}}

    def merge(self, state: dict):
        """Adds another SegMetrics' state() to these counts."""
        self.units += state["units"]
        for name, (tp, fp, fn) in state["counts"].items():
            counts = self.counts[name]
            counts[0] += tp
            counts[1] += fp
            counts[2] += fn
        self.truth_classes.update(state["truth_classes"])
        self.outputs.update(state["outputs"])
        self.valid.update(state["valid"])
        self.errors.update(state["errors"])
        self.malformed.update(state["malformed"])
        self.truth_invalid += state["truth_invalid"]
        for task, values in state.get("recog", {}).items():
            self.recog[task] = [a + b for a, b in zip(self.recog[task], values)]

    def _matches(self, kind, preds, truths) -> int:
        dist = lambda p, q: math.hypot(p[0] - q[0], p[1] - q[1])
        pairs = []
        for i, p in enumerate(preds):
            for j, g in enumerate(truths):
                if kind == "box":
                    score = box_iou(p, g, self.min_box_px)
                    ok = score >= self.iou - 1e-9
                elif kind == "point":
                    d = dist(p, g)
                    score, ok = -d, d <= self.vertex_px
                elif kind == "arc":
                    # the ends as points; the middle anywhere on the true arc, since any point on it between the
                    # ends describes the same arc, but not collapsed onto an end, which describes none
                    d = max(dist(p[:2], g[:2]), dist(p[4:6], g[4:6]), arc_distance(p[2:4], g))
                    collapsed = (min(dist(p[2:4], p[:2]), dist(p[2:4], p[4:6]))
                                 < 0.5 * min(dist(g[2:4], g[:2]), dist(g[2:4], g[4:6])))
                    score, ok = -d, d <= self.vertex_px and not collapsed
                elif kind == "segment":
                    d = min(max(dist(p[:2], g[:2]), dist(p[2:4], g[2:4])),
                            max(dist(p[:2], g[2:4]), dist(p[2:4], g[:2])))
                    score, ok = -d, d <= self.vertex_px
                else:
                    start, end = dist(p[:2], g[:2]), dist(p[2:], g[2:])
                    score, ok = -max(start, end), start <= self.arrow_px and end <= self.arrow_px
                if ok:
                    pairs.append((score, i, j))
        pairs.sort(key=lambda pair: -pair[0])
        used_p, used_g = set(), set()
        for _, i, j in pairs:
            if i not in used_p and j not in used_g:
                used_p.add(i)
                used_g.add(j)
        return len(used_p)

    def parsed(self, parsed: Parsed, stage: str):
        """Counts one generated output toward parse validity."""
        self.outputs[stage] += 1
        self.valid[stage] += parsed.valid
        for reason, name in parsed.errors:
            self.errors[reason] += 1
            if name is not None:
                self.malformed[name] += 1

    def score(self, preds, truths):
        """One image's predictions against its ground truth, both (class, pixel coordinates)."""
        for name in {o[0] for o in truths} | {o[0] for o in preds}:
            p = [o[1] for o in preds if o[0] == name]
            g = [o[1] for o in truths if o[0] == name]
            tp = self._matches(KIND[name], p, g) if p and g else 0
            counts = self.counts[name]
            counts[0] += tp
            counts[1] += len(p) - tp
            counts[2] += len(g) - tp
            if g:
                self.truth_classes.add(name)

    def add(self, prompt, target, output, size, truncated=False) -> Parsed:
        """One manifest row. target: its loc-token target; output: the generated
        text with special tokens; size: the stored image's (width, height)."""
        allowed = prompt_classes(prompt)
        truth = parse_output(target, allowed, generated=False)
        self.truth_invalid += not truth.valid
        pred = parse_output(output, allowed, generated=True, truncated=truncated)
        self.parsed(pred, "row")
        self.units += 1
        self.score(grid_to_pixels(pred.objects, size), grid_to_pixels(truth.objects, size))
        return pred

    def recognized(self, task, loss_sum, tokens, rows, matched):
        """One recognition image (recog_eval.py): its summed token loss over `tokens` target tokens, read from
        `rows` detected lines, `matched` of them to a ground-truth line."""
        counts = self.recog[task]
        for i, value in enumerate((loss_sum, tokens, 1, rows, matched)):
            counts[i] += value

    def transcribed(self, task, edits, chars, missed=0, correct=0, total=0):
        """Generated recognition text (recog_eval.py): `edits` character edits against targets of `chars` characters,
        `missed` of them ground-truth lines no detected box matched, counted as read as nothing; `correct` characters
        read correctly out of `total` (recog_eval.score_rows)."""
        counts = self.recog[task]
        for i, value in enumerate((edits, chars, missed, correct, total), 5):
            counts[i] += value

    def summary(self) -> dict:
        order = list(KIND)
        per_class = {}
        for name in sorted(set(self.counts) | set(self.malformed), key=order.index):
            tp, fp, fn = self.counts[name]
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp,
                               "fn": fn, "support": tp + fn, "malformed": self.malformed[name]}
        mean = lambda names: sum(per_class[n]["f1"] for n in names) / len(names) if names else 0.0
        in_mean = [name for name in per_class if name in self.truth_classes]
        outputs = sum(self.outputs.values())
        return {"unit": self.unit, "units": self.units, "outputs": outputs,
                "parse_valid_rate": sum(self.valid.values()) / outputs if outputs else 0.0,
                "parse_valid_by_stage": {s: self.valid[s] / n for s, n in self.outputs.items()},
                "mean_f1": mean(in_mean), "mean_f1_classes": in_mean,
                "mean_f1_by_group": {group: mean([n for n in in_mean if n in names])
                                     for group, names in GROUPS.items() if any(n in names for n in in_mean)},
                "per_class": per_class, "parse_errors": dict(self.errors), "truth_invalid": self.truth_invalid,
                "recognition": {task: {"loss": loss / tokens if tokens else None, "tokens": tokens, "images": images,
                                       "rows": rows, "matched": matched, "cer": edits / chars if chars else None,
                                       "edits": edits, "chars": chars, "missed": missed,
                                       "accuracy": correct / total if total else None, "correct": correct,
                                       "total": total}
                                for task, (loss, tokens, images, rows, matched, edits, chars, missed, correct, total)
                                in sorted(self.recog.items())}}


def flat_record(summary, prefix="val/") -> dict:
    """Every number as its own scalar key, per class kept separate, for log.jsonl."""
    record = {f"{prefix}{summary['unit']}": summary["units"], f"{prefix}outputs": summary["outputs"],
              f"{prefix}parse_valid_rate": summary["parse_valid_rate"], f"{prefix}mean_f1": summary["mean_f1"]}
    if "val_loss" in summary:
        record[f"{prefix}loss"] = summary["val_loss"]
    for stage, rate in summary["parse_valid_by_stage"].items():
        record[f"{prefix}parse_valid_rate/{stage}"] = rate
    for group, value in summary["mean_f1_by_group"].items():
        record[f"{prefix}mean_f1/{group}"] = value
    for name, stats in summary["per_class"].items():
        key = name.replace(" ", "_")
        for stat in ("f1", "precision", "recall", "tp", "fp", "fn", "malformed"):
            record[f"{prefix}{stat}/{key}"] = stats[stat]
    for reason, n in summary["parse_errors"].items():
        record[f"{prefix}parse_error/{reason}"] = n
    for task, stats in summary.get("recognition", {}).items():
        record[f"{prefix}recog_loss/{task}"] = stats["loss"]
        record[f"{prefix}recog_rows/{task}"] = stats["rows"]
        record[f"{prefix}recog_matched/{task}"] = stats["matched"]
        record[f"{prefix}recog_cer/{task}"] = stats["cer"]
        record[f"{prefix}recog_missed/{task}"] = stats["missed"]
        record[f"{prefix}recog_acc/{task}"] = stats["accuracy"]
        record[f"{prefix}recog_correct/{task}"] = stats["correct"]
        record[f"{prefix}recog_total/{task}"] = stats["total"]
    return record


def format_report(summary) -> str:
    errors = ", ".join(f"{reason} {n}" for reason, n in
                       sorted(summary["parse_errors"].items(), key=lambda kv: -kv[1])) or "none"
    stages = summary["parse_valid_by_stage"]
    by_stage = f" [{', '.join(f'{s} {100 * r:.1f}%' for s, r in stages.items())}]" if len(stages) > 1 else ""
    groups = ", ".join(f"{g} {v:.4f}" for g, v in summary["mean_f1_by_group"].items())
    lines = []
    if "val_loss" in summary:
        lines.append(f"  val loss      {summary['val_loss']:.4f}")
    lines += [f"  parse valid   {100 * summary['parse_valid_rate']:.1f}% of {summary['outputs']} outputs "
              f"over {summary['units']} {summary['unit']}{by_stage}  (errors: {errors})",
              f"  mean F1       {summary['mean_f1']:.4f} over {len(summary['mean_f1_classes'])} "
              f"ground-truth classes ({groups})",
              f"  {'class':<10} {'P':>6} {'R':>6} {'F1':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'malformed':>9}"]
    for name, s in summary["per_class"].items():
        note = "" if name in summary["mean_f1_classes"] else "  (no ground truth; not in mean)"
        lines.append(f"  {name:<10} {s['precision']:6.3f} {s['recall']:6.3f} {s['f1']:6.3f} "
                     f"{s['tp']:6d} {s['fp']:6d} {s['fn']:6d} {s['malformed']:9d}{note}")
    for task, r in summary.get("recognition", {}).items():
        loss = f"{r['loss']:.4f}" if r["loss"] is not None else "–"
        cer = f"{100 * r['cer']:.1f}%" if r.get("cer") is not None else "–"
        accuracy = f"{100 * r['accuracy']:.1f}%" if r.get("accuracy") is not None else "–"
        lines.append(f"  recognition {task:<8} loss {loss} over {r['tokens']} target tokens: {r['rows']} detected "
                     f"lines ({r['matched']} matched) in {r['images']} images; generated text accuracy {accuracy} "
                     f"({r.get('correct', 0)} of {r.get('total', 0)} characters), CER {cer} "
                     f"({r.get('edits', 0)} edits over {r.get('chars', 0)} characters, {r.get('missed', 0)} lines missed)")
    if summary["truth_invalid"]:
        lines.append(f"  WARNING: {summary['truth_invalid']} ground-truth targets did not parse cleanly")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------

class EvalCollator(Collator):
    """Collator's batch plus what scoring needs: prompts, loc-token targets,
    image paths, and each stored image's (width, height)."""

    def __call__(self, batch):
        out = super().__call__(batch)
        out["prompts"] = [item[1] for item in batch]
        out["targets"] = [item[2] for item in batch]
        out["paths"] = [item[3] for item in batch]
        out["sizes"] = [(item[0].shape[1], item[0].shape[0]) for item in batch]
        return out


def encoder_inputs(base, batch, device, dtype):
    """Image tokens then prompt tokens, with the prompt's own padding mask. The
    vision tower, projection and embeddings are all frozen, so no graph is kept."""
    pixel_values = batch["pixel_values"].to(device, dtype, non_blocking=True)
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    prompt_mask = batch["attention_mask"].to(device, non_blocking=True)
    with torch.no_grad():
        image = base._encode_image(pixel_values)
        text = base.get_input_embeddings()(input_ids)
    embeds = torch.cat([image, text.to(image.dtype)], dim=1)
    mask = torch.cat([torch.ones(image.shape[:2], device=device, dtype=prompt_mask.dtype), prompt_mask], dim=1)
    return embeds, mask


def _no_bos_after_start(tokenizer):
    """A LogitsProcessorList that forbids <s> once the decoder start and the forced <s> are in place."""
    from transformers import LogitsProcessor, LogitsProcessorList

    class NoBosAfterStart(LogitsProcessor):
        def __call__(self, input_ids, scores):
            if input_ids.shape[1] >= 2:
                scores[:, tokenizer.bos_token_id] = -float("inf")
            return scores

    return LogitsProcessorList([NoBosAfterStart()])


@torch.no_grad()
def generate(base, batch, tokenizer, device, dtype, max_new_tokens):
    """Greedy outputs as text with special tokens kept, and whether each row was
    still going when max_new_tokens ran out. forced_eos_token_id closes such a
    row with </s>, so the text alone cannot tell."""
    embeds, mask = encoder_inputs(base, batch, device, dtype)
    with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
        # no_repeat_ngram_size=0: Florence-2's config sets 3, which forbids any 3-token run from appearing twice. These
        # outputs repeat runs by design (a corner is a segment's end and a vertex; "<loc_a><loc_b>segment" comes back;
        # LaTeX repeats "}^{"), so it nudged repeated coordinates a step or two and respelled class words ("SEgment").
        # It was also all that ended the model's run of <s>: the tuned decoder's next token after </s>, </s><s> and
        # </s><s><s><s> is the same (<s> at 0.95), so without it every output is <s> to max_new_tokens. <s> after the
        # forced one is suppressed instead (NoBosAfterStart); what follows is what followed the three <s> before.
        ids = base.language_model.generate(inputs_embeds=embeds, attention_mask=mask, num_beams=1,
                                           early_stopping=False, do_sample=False, no_repeat_ngram_size=0,
                                           logits_processor=_no_bos_after_start(tokenizer),
                                           max_new_tokens=max_new_tokens)
    if ids.shape[1] - 1 < max_new_tokens:           # every row stopped on its own
        truncated = [False] * len(ids)
    else:
        truncated = (ids[:, -1] != tokenizer.pad_token_id).tolist()
    texts = tokenizer.batch_decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return texts, truncated


def from_loc_target(text: str) -> str:
    """Loc-token text back into the manifest's "x y x y class" form, for reading."""
    for special in ("<s>", "</s>", "<pad>"):
        text = text.replace(special, "")
    return " ".join(re.sub(r"<loc_(\d+)>", r" \1 ", text).split())


def to_ink(image):
    """A greyscale PIL image as the dataset gives it: 3 equal channels, background 0, ink 1."""
    grey = np.asarray(image.convert("L"))
    return 1.0 - np.repeat(grey[:, :, None], 3, axis=2).astype(np.float32) / 255.0


def model_generate_fn(base, processor, device, dtype, max_new_tokens, batch_size):
    """generate_fn for validate_pages: requests -> (texts, truncated), batch_size at a time."""
    collator = Collator()
    collator.processor = processor

    def run(requests):
        texts, truncated = [], []
        for i in range(0, len(requests), batch_size):
            chunk = requests[i:i + batch_size]
            batch = collator([(to_ink(r["image"]), r["prompt"], "", "") for r in chunk])
            t, c = generate(base, batch, processor.tokenizer, device, dtype, max_new_tokens)
            texts += t
            truncated += c
        return texts, truncated
    return run


# ----------------------------------------------------------------------------
# Rows
# ----------------------------------------------------------------------------

@torch.no_grad()
def validate(base, loader, tokenizer, device, dtype, max_new_tokens, metrics, loss_fn=None, keep_samples=0,
             progress=None, on_sample=None, summarize=True):
    """Manifest rows, one pass over `loader` (EvalCollator batches). The caller
    puts the model in eval mode. loss_fn(batch) -> (mean loss, target tokens) adds
    val_loss. progress(rows done) is called after each batch and on_sample(sample)
    as each kept sample is made. Returns (metrics summary, the first keep_samples
    rows' outputs); with summarize=False, (metrics, (loss sum, target tokens), samples)
    so counts from several processes can be combined first."""
    loss_sum, tokens, samples, done = 0.0, 0, [], 0
    for batch in loader:
        if loss_fn is not None:
            loss, n = loss_fn(batch)
            loss_sum += loss.item() * n
            tokens += n
        texts, truncated = generate(base, batch, tokenizer, device, dtype, max_new_tokens)
        for i, text in enumerate(texts):
            parsed = metrics.add(batch["prompts"][i], batch["targets"][i], text, batch["sizes"][i], truncated[i])
            if len(samples) < keep_samples:
                samples.append({"image": batch["paths"][i], "prompt": batch["prompts"][i],
                                "target": from_loc_target(batch["targets"][i]),
                                "prediction": from_loc_target(text), "valid": parsed.valid,
                                "errors": [reason for reason, _ in parsed.errors], "raw": text})
                if on_sample is not None:
                    on_sample(samples[-1])
        done += len(texts)
        if progress is not None:
            progress(done)
    if not summarize:
        return metrics, (loss_sum, tokens), samples
    summary = metrics.summary()
    if loss_fn is not None:
        summary["val_loss"] = loss_sum / max(tokens, 1)
    return summary, samples


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------

def find_pages(folders):
    """(image path, JSON path) for every JSON in the folders with an image of its stem."""
    pages = []
    for folder in folders:
        for name in sorted(os.listdir(folder)):
            stem, ext = os.path.splitext(name)
            if ext.lower() != ".json":
                continue
            candidates = [os.path.join(folder, stem + e) for e in IMAGE_EXTENSIONS + tuple(x.upper() for x in IMAGE_EXTENSIONS)]
            image = next((path for path in candidates if os.path.exists(path)), None)
            if image is None:
                raise FileNotFoundError(f"{os.path.join(folder, name)}: no image {stem}.png / .jpg / ... beside it")
            pages.append((image, os.path.join(folder, name)))
    if not pages:
        raise FileNotFoundError(f"no page JSONs in {folders}")
    return pages


TEXT_CLASSES = ("textline", "formula")
MARK_CLASSES = ("enclosure", "line_mark")          # their ink goes to recognition with the lines it crosses


def page_truth(path, width, height):
    """A page's ground truth as (class, page-pixel coordinates)."""
    return read_page(path, width, height)[0]


def read_page(path, width, height):
    """(page_truth's ground truth, the page's ink for recognition). The second is [{class, bbox, text, strokes}] in
    page pixels for its textlines and formulas, and its enclosures and line marks where the JSON gives their "ink",
    each stroke an (n, 3) array of x, y and the source's time (NaN for ink without times: a mark's always), or None
    when the JSON gives the textlines and formulas no "ink" (written before page_export did)."""
    with open(path) as f:
        data = json.load(f)
    objects = data["objects"] if isinstance(data, dict) else data
    page_w, page_h = data.get("page_size", (width, height)) if isinstance(data, dict) else (width, height)
    sx, sy = width / page_w, height / page_h
    scale = lambda values: tuple(v * (sx if i % 2 == 0 else sy) for i, v in enumerate(values))
    truth = []
    for obj in objects:
        name = TRUTH_CLASS.get(obj["class"], obj["class"])
        kind = KIND.get(name)
        if kind is None:
            continue
        if kind == "point":
            truth.append((name, scale(obj.get("position") or obj["bbox"][:2])))
        elif kind in ("arc", "segment"):
            truth.append((name, scale(obj["points"])))
        elif kind == "arrow":
            if obj.get("start") is None or obj.get("end") is None:
                raise ValueError(f"{path}: an arrow without start and end: {obj}")
            truth.append((name, scale([*obj["start"], *obj["end"]])))
        else:
            truth.append((name, scale(obj["bbox"])))
    text = []
    for obj in objects:
        if obj["class"] in TEXT_CLASSES and "ink" not in obj:
            text = None
            break
        if obj["class"] not in TEXT_CLASSES + MARK_CLASSES or "ink" not in obj:
            continue
        strokes = []
        for stroke in obj["ink"]:
            if stroke:
                points = np.asarray(stroke, dtype=np.float64)
                t = points[:, 2] if points.shape[1] > 2 else np.full(len(points), np.nan)
                strokes.append(np.column_stack([points[:, 0] * sx, points[:, 1] * sy, t]))
        text.append({"class": obj["class"], "bbox": scale(obj["bbox"]), "text": obj.get("text"), "strokes": strokes,
                     "parent": obj.get("parent")})
    return truth, text


class PageSet(Dataset):
    """Page images with their ground truth (find_pages, page_truth)."""

    def __init__(self, folders, limit=0):
        self.pages = find_pages(folders)
        if limit:
            self.pages = self.pages[:limit]

    def __len__(self):
        return len(self.pages)

    def __getitem__(self, index):
        image_path, truth_path = self.pages[index]
        with Image.open(image_path) as image:
            grey = np.asarray(image.convert("L"))
        height, width = grey.shape
        truth, text = read_page(truth_path, width, height)
        return {"path": image_path, "image": grey, "truth": truth, "text": text}


def collate_pages(batch):
    return batch


def crop_transform(box, width, height, size):
    """seg_prompts.crop_transform with the page image as the board: the box scaled
    so its longer side fills `size`, centred. image = page * scale + offset."""
    x0, y0 = max(0.0, box[0]), max(0.0, box[1])
    w = max(min(float(width), box[2]) - x0, 1e-3)
    h = max(min(float(height), box[3]) - y0, 1e-3)
    fit = size / max(w, h)
    out_w = int(min(size, max(1, round(w * fit))))
    out_h = int(min(size, max(1, round(h * fit))))
    pad = ((size - out_w) // 2, (size - out_h) // 2)
    scale = (out_w / w, out_h / h)
    return {"box": (x0, y0, x0 + w, y0 + h), "size": (out_w, out_h), "pad": pad, "scale": scale,
            "offset": (pad[0] - x0 * scale[0], pad[1] - y0 * scale[1])}


def crop_image(page, t, size):
    """seg_prompts.sample_image: the box resampled, then padded onto white."""
    region = page.resize(t["size"], resample=Image.LANCZOS, box=t["box"])
    image = Image.new("L", (size, size), 255)
    image.paste(region, t["pad"])
    return image


def page_grid_to_pixels(coords, width, height):
    """A whole-page output's coordinates, each at its cell's centre, in page pixels."""
    return tuple((k + 0.5) / GRID * (width if i % 2 == 0 else height) for i, k in enumerate(coords))


def crop_grid_to_pixels(coords, t, size):
    """A crop output's coordinates, each at its cell's centre, back in page pixels."""
    return tuple(((k + 0.5) / GRID * size - t["offset"][i % 2]) / t["scale"][i % 2] for i, k in enumerate(coords))


@torch.no_grad()
def validate_pages(loader, generate_fn, metrics, size, keep_samples=0, progress=None, on_sample=None,
                   summarize=True, recognize=None):
    """The cascade over `loader` (PageSet batches, collate_pages). generate_fn
    takes requests {image (PIL, size x size), prompt, stage, path, transform, parent}
    and returns (texts, truncated). recognize(pages, predictions, metrics)
    (recog_eval.Recognizer) then reads each page's detected textlines and formulas;
    a kept sample carries what it returns for its page as "recognition". The caller
    puts the model in eval mode. progress(pages done) is called after each batch and
    on_sample(sample) as each kept sample is made. Returns (metrics summary, the first
    keep_samples pages' predictions); with summarize=False, (metrics, samples)."""
    samples, done = [], 0
    for pages in loader:
        full = [Image.fromarray(page["image"], "L") for page in pages]
        requests = []
        for p, image in enumerate(full):
            small = image.resize((size, size), resample=Image.LANCZOS)
            for stage, prompt in (("level2", LEVEL2_PROMPT), ("arrow", ARROW_PROMPT)):
                requests.append({"image": small, "prompt": prompt, "stage": stage, "page": p,
                                 "path": pages[p]["path"], "transform": None, "parent": None})

        preds = [[] for _ in pages]
        outputs = [[] for _ in pages]
        crops = []
        for request, text, cut in zip(requests, *generate_fn(requests)):
            p = request["page"]
            width, height = full[p].size
            parsed = parse_output(text, prompt_classes(request["prompt"]), truncated=cut)
            metrics.parsed(parsed, request["stage"])
            outputs[p].append({"stage": request["stage"], "valid": parsed.valid, "output": text})
            for name, coords in parsed.objects:
                pixels = page_grid_to_pixels(coords, width, height)
                preds[p].append((name, pixels))
                if name in LEVEL1_PROMPTS:
                    t = crop_transform(pixels, width, height, size)
                    crops.append({"image": crop_image(full[p], t, size), "prompt": LEVEL1_PROMPTS[name],
                                  "stage": "level1", "page": p, "path": pages[p]["path"],
                                  "transform": t, "parent": (name, pixels)})

        if crops:
            for request, text, cut in zip(crops, *generate_fn(crops)):
                p = request["page"]
                parsed = parse_output(text, prompt_classes(request["prompt"]), truncated=cut)
                metrics.parsed(parsed, "level1")
                outputs[p].append({"stage": "level1", "parent": request["parent"][0], "valid": parsed.valid,
                                   "output": text})
                for name, coords in parsed.objects:
                    preds[p].append((name, crop_grid_to_pixels(coords, request["transform"], size)))

        read = recognize(pages, preds, metrics) if recognize is not None else None
        for p, page in enumerate(pages):
            metrics.score(preds[p], page["truth"])
            metrics.units += 1
            if len(samples) < keep_samples:
                samples.append({"image": page["path"], "truth_objects": len(page["truth"]),
                                "predicted": [[name, [round(v, 1) for v in c]] for name, c in preds[p]],
                                "outputs": outputs[p]})
                if read is not None:        # None: the page's JSON has no ink to read
                    samples[-1]["recognition"] = read[p] and read[p]["groups"]
                    samples[-1]["recognition_missed"] = read[p] and read[p]["missed"]
                if on_sample is not None:
                    on_sample(samples[-1])
        done += len(pages)
        if progress is not None:
            progress(done)
    if not summarize:
        return metrics, samples
    return metrics.summary(), samples


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------

def add_metric_args(ap):
    ap.add_argument("--eval-batch-size", type=int, default=12, help="outputs generated at once")
    ap.add_argument("--max-new-tokens", type=int, default=1024, help="an output that reaches it is truncated")
    ap.add_argument("--iou", type=float, default=0.5, help="box match threshold")
    ap.add_argument("--vertex-px", type=float, default=10.0, help="rows: vertex match distance, image pixels")
    ap.add_argument("--arrow-px", type=float, default=15.0,
                    help="rows: arrow start-to-start and end-to-end match distance, image pixels")
    ap.add_argument("--page-vertex-px", type=float, default=20.0, help="pages: vertex match distance, page pixels")
    ap.add_argument("--page-arrow-px", type=float, default=30.0,
                    help="pages: arrow start-to-start and end-to-end match distance, page pixels")
    ap.add_argument("--page-min-box-px", type=float, default=4.0,
                    help="pages: boxes thinner than this are widened to it before IoU, page pixels")
    ap.add_argument("--page-batch", type=int, default=4, help="pages loaded and run through the cascade at once")
    ap.add_argument("--mark-model", default=None,
                    help="pages: the mark separation network's model.pt (training/marks_cnn); recognition takes marks out "
                         "of each detected line's ink with it before reading (recog_eval.MarkRemover). Default: none")


def main():
    """Under torchrun (torchrun --nproc_per_node 2 seg_eval.py ...) each process scores
    its share of the pages or rows and the counts are combined before reporting."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="adapter folder (e.g. runs/seg/best), or 'none' for the base model")
    ap.add_argument("--pages", nargs="+", default=None, help="page folder(s): run the cascade instead of manifest rows")
    ap.add_argument("--max-pages", type=int, default=0, help="first N pages (0 = all)")
    ap.add_argument("--manifest", nargs="+", default=[DEFAULT_MANIFEST])
    ap.add_argument("--val-manifest", nargs="+", default=None)
    ap.add_argument("--val-fraction", type=float, default=0.01, help="tail rows, as train.py holds out")
    ap.add_argument("--max-rows", type=int, default=0, help="first N validation rows (0 = all)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default=None, help="summary JSON (default: <checkpoint>/page_metrics.json or val_metrics.json)")
    ap.add_argument("--predictions", default=None, help="also write every row's / page's outputs to this JSONL")
    ap.add_argument("--progress", default=None, help="keep this progress JSON current while running")
    ap.add_argument("--samples-out", default=None, help="append the first --samples outputs here as they are made")
    ap.add_argument("--samples", type=int, default=8)
    add_metric_args(ap)
    args = ap.parse_args()

    from torch.utils.data import DataLoader, Subset
    from distributed import cleanup, gather_metrics, gather_objects, is_main, setup, shard
    from model import load_base, pick_dtype
    from progress import ProgressFile, SampleStream

    _, world, device = setup()
    dtype = pick_dtype()
    progress = ProgressFile(args.progress, phase="loading", checkpoint=args.checkpoint, world=world)
    progress.update(force=True)
    base, processor = load_base(dtype)
    if args.checkpoint.lower() != "none":
        from peft import PeftModel
        base = PeftModel.from_pretrained(base, args.checkpoint).get_base_model()
    base = base.to(device, dtype).eval()
    stream = SampleStream(args.samples_out, limit=args.samples) if args.samples_out else None

    if args.pages:
        pages = PageSet(args.pages, args.max_pages)
        mine = Subset(pages, shard(range(len(pages))))
        loader = DataLoader(mine, batch_size=args.page_batch, collate_fn=collate_pages, num_workers=args.num_workers)
        metrics = SegMetrics(args.iou, args.page_vertex_px, args.page_arrow_px, args.page_min_box_px, unit="pages")
        total, unit = len(pages), "pages"
        if is_main():
            print(f"cascade on {len(pages):,} pages with {args.checkpoint} on {world} x {device.type} ({dtype})")
        progress.update(force=True, phase="evaluating", unit=unit, done=0, total=total)
        generate_fn = model_generate_fn(base, processor, device, dtype, args.max_new_tokens, args.eval_batch_size)
        from recog_eval import MarkRemover, Recognizer, model_read_fn
        remove_marks = MarkRemover(args.mark_model, device) if args.mark_model else None
        recognize = Recognizer(model_read_fn(base, processor, device, dtype, args.max_new_tokens, args.eval_batch_size),
                               args.iou, args.page_min_box_px, remove_marks=remove_marks)
        metrics, rows = validate_pages(loader, generate_fn, metrics, processor.image_processor.size["height"],
                                       keep_samples=len(mine) if args.predictions else args.samples,
                                       progress=lambda n: progress.update(done=min(total, n * world)),
                                       on_sample=stream.write if stream else None, summarize=False,
                                       recognize=recognize)
        thresholds = {"iou": args.iou, "vertex_px": args.page_vertex_px, "arrow_px": args.page_arrow_px,
                      "min_box_px": args.page_min_box_px}
        default_name = "page_metrics.json"
    else:
        from train import split
        _, _, val, indices = split(args)
        if args.max_rows:
            indices = indices[:args.max_rows]
        if not len(indices):
            raise SystemExit("no validation rows")
        total, unit = len(indices), "rows"
        loader = DataLoader(val, sampler=shard(indices).tolist(), batch_size=args.eval_batch_size,
                            collate_fn=EvalCollator(), num_workers=args.num_workers,
                            pin_memory=device.type == "cuda")
        metrics = SegMetrics(args.iou, args.vertex_px, args.arrow_px)
        if is_main():
            print(f"validating {len(indices):,} rows with {args.checkpoint} on {world} x {device.type} ({dtype})")
        progress.update(force=True, phase="evaluating", unit=unit, done=0, total=total)
        metrics, _, rows = validate(base, loader, processor.tokenizer, device, dtype, args.max_new_tokens, metrics,
                                    keep_samples=len(indices) if args.predictions else args.samples,
                                    progress=lambda n: progress.update(done=min(total, n * world)),
                                    on_sample=stream.write if stream else None, summarize=False)
        thresholds = {"iou": args.iou, "vertex_px": args.vertex_px, "arrow_px": args.arrow_px}
        default_name = "val_metrics.json"

    summary = gather_metrics(metrics).summary()
    rows = [row for part in gather_objects(rows) for row in part] if args.predictions else rows
    if is_main():
        print(format_report(summary))
        out = args.out or os.path.join(args.checkpoint if args.checkpoint.lower() != "none" else ".", default_name)
        with open(out, "w") as f:
            json.dump({"checkpoint": args.checkpoint, "thresholds": thresholds, **summary}, f, indent=2)
        print(f"wrote {out}")
        if args.predictions:
            with open(args.predictions, "w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            print(f"wrote {args.predictions}")
    progress.update(force=True, phase="finished", done=total, mean_f1=summary["mean_f1"],
                    parse_valid_rate=summary["parse_valid_rate"])
    cleanup()


if __name__ == "__main__":
    main()
