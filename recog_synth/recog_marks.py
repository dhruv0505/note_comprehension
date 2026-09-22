"""
recog_marks.py — a mark on a recognition row, as mark separation leaves it.

In the page cascade (training/recog_eval.py) a detected line's ink goes through the mark
separation network before it is read, and a point is kept only where the network sees
writing under it. An enclosure or line mark that crosses the writing keeps its points
there and loses the rest. Those points have no times of their own (marks are built, not
loaded), so the cascade gives each one a time drawn at random from the row's span.

A text sample is made to look the same (maybe_mark). Now and then, MARK_PROB of textline
rows, a mark is drawn on the row the way note_synth marks its pages: an underline or a
strikethrough under or through a stretch of it, or a box or oval round a stretch of it. A
formula row is never marked: note_synth never underlines or strikes through a formula, and
an enclosure round one stays outside its box, so the cascade never reads a mark with one.
Only the mark's points that overlap the writing are added to the row, cut into a stroke
for each run of them, and each takes a time drawn uniformly between the row's earliest
and latest. A mark that overlaps nothing adds nothing.

Overlap is in the item's own units: a mark point overlaps when it lies within `reach` of
the writing's centreline, which recog_synthesize sets to one pen width at the size the
row is drawn, so the two inks would touch in the image. The mark is sampled at the
writing's own point spacing, so what is added is as dense as the ink it lies on.

Nothing here imports note_synth; the shapes follow block_generators' (UNDERLINE_DROP,
ENCLOSURE_PAD and the rest are its values).
"""

from __future__ import annotations
import math
import random
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

import recog_layout as rl

MARK_PROB = 0.15            # share of text rows a mark is drawn on
# Only textlines: note_synth never underlines or strikes through a formula, and an enclosure round one stays outside
# its box, so the cascade reads no mark with a formula (none in 254 formula rows of the val pages).
KINDS = {"textline": (("underline", 0.55), ("strikethrough", 0.15), ("oval", 0.15), ("rectangle", 0.15))}
UNDERLINE_DROP = 0.40       # below the writing band's foot, in band heights
UNDERLINE_DOUBLE_PROB = 0.08
UNDERLINE_DOUBLE_GAP = 0.18
MAX_SLOPE = 0.03            # a line mark's rise over run
ENCLOSURE_PAD = 0.18        # clearance round what is enclosed, in band heights
LINE_MARK_LEAST = 0.3       # a line mark runs under at least this share of the row's width
ENCLOSURE_LEAST = 0.15      # an enclosure holds at least this share of it


def _band(points: np.ndarray) -> Tuple[float, float, float]:
    """(top, foot, height) of the writing's core band: the 15th to 85th percentile of its y."""
    top, foot = np.percentile(points[:, 1], [15, 85])
    return float(top), float(foot), max(float(foot - top), 1e-6)


def _spacing(lines) -> float:
    """The writing's median distance between neighbouring points."""
    steps = [np.hypot(*np.diff(line[:, :2], axis=0).T) for line in lines if len(line) > 1]
    steps = np.concatenate(steps) if steps else np.zeros(0)
    steps = steps[steps > 0]
    return float(np.median(steps)) if len(steps) else 1.0


def _resample(path: np.ndarray, spacing: float) -> np.ndarray:
    """Points along the polyline `path` (k, 2) at most `spacing` apart, both ends included."""
    lengths = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
    if lengths[-1] <= 0:
        return path[:1].copy()
    along = np.linspace(0.0, lengths[-1], max(2, int(math.ceil(lengths[-1] / max(spacing, 1e-9))) + 1))
    return np.column_stack([np.interp(along, lengths, path[:, 0]), np.interp(along, lengths, path[:, 1])])


def _stretch(points: np.ndarray, rng: random.Random, least: float) -> Tuple[float, float]:
    """A random stretch [x0, x1] of the row's width, at least `least` of it."""
    left, right = float(points[:, 0].min()), float(points[:, 0].max())
    width = right - left
    span = width * rng.uniform(least, 1.0)
    start = left + rng.uniform(0.0, width - span)
    return start, start + span


def mark_paths(kind: str, lines, rng: random.Random) -> List[np.ndarray]:
    """A mark's strokes as (k, 2) polylines in the item's units: `kind` of MARK_KINDS on `lines`
    (recog_layout.polylines of the row)."""
    points = np.concatenate(lines)[:, :2]
    top, foot, band = _band(points)
    if kind in ("underline", "strikethrough"):
        x0, x1 = _stretch(points, rng, LINE_MARK_LEAST)
        x0, x1 = x0 - rng.uniform(0.0, 0.2) * band, x1 + rng.uniform(0.0, 0.35) * band     # runs a touch past the words
        if kind == "underline":
            base = foot + UNDERLINE_DROP * band * rng.uniform(0.85, 1.3)
        else:
            base = (top + foot) / 2 + rng.uniform(-0.15, 0.15) * band
        xs = np.linspace(x0, x1, 32)
        slope, wobble = rng.uniform(-MAX_SLOPE, MAX_SLOPE), rng.uniform(0.0, 0.06) * band
        waves, phase = rng.uniform(0.5, 2.0), rng.uniform(0.0, 2 * math.pi)
        ys = base + slope * (xs - x0) + wobble * np.sin(phase + 2 * math.pi * waves * (xs - x0) / max(x1 - x0, 1e-9))
        paths = [np.column_stack([xs, ys])]
        if kind == "underline" and rng.random() < UNDERLINE_DOUBLE_PROB:
            gap = UNDERLINE_DOUBLE_GAP * band * rng.uniform(0.85, 1.15)
            paths.append(np.column_stack([xs + rng.uniform(-0.1, 0.1) * band, ys + gap]))
        return paths

    x0, x1 = _stretch(points, rng, ENCLOSURE_LEAST)
    held = points[(points[:, 0] >= x0) & (points[:, 0] <= x1)]
    held = held if len(held) else points
    x0, y0 = held.min(axis=0)
    x1, y1 = held.max(axis=0)
    pad = ENCLOSURE_PAD * band
    if kind == "oval":
        # the smallest ellipse through the padded box's corners is its half-extents times sqrt 2
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rx = ((x1 - x0) / 2 + pad) * math.sqrt(2) * rng.uniform(0.9, 1.05)
        ry = ((y1 - y0) / 2 + pad) * math.sqrt(2) * rng.uniform(0.9, 1.05)
        start = rng.uniform(0.0, 2 * math.pi)
        angles = np.linspace(start, start + 2 * math.pi + rng.uniform(0.0, 0.5), 96)      # closes with an overshoot
        swell = 1.0 + rng.uniform(0.0, 0.05) * np.sin(angles * rng.uniform(1.0, 3.0) + rng.uniform(0.0, 2 * math.pi))
        return [np.column_stack([cx + rx * swell * np.cos(angles), cy + ry * swell * np.sin(angles)])]
    corners = [(x0 - pad, y0 - pad), (x1 + pad, y0 - pad), (x1 + pad, y1 + pad), (x0 - pad, y1 + pad)]
    paths = []
    for k in range(4):         # each side its own stroke, its ends overshooting or falling short of the corners
        (ax, ay), (bx, by) = corners[k], corners[(k + 1) % 4]
        length = math.hypot(bx - ax, by - ay) or 1.0
        ux, uy = (bx - ax) / length, (by - ay) / length
        a, b = rng.uniform(-0.6, 1.0) * pad, rng.uniform(-0.6, 1.0) * pad
        nudge = [rng.uniform(-0.3, 0.3) * pad for _ in range(2)]
        paths.append(np.array([[ax - ux * a + uy * nudge[0], ay - uy * a - ux * nudge[0]],
                               [bx + ux * b + uy * nudge[1], by + uy * b - ux * nudge[1]]]))
    return paths


def overlapping(points: np.ndarray, lines, reach: float) -> np.ndarray:
    """Whether each of `points` (m, 2) lies within `reach` of the writing's centreline."""
    dense = np.concatenate([_resample(line[:, :2], reach / 2) if len(line) > 1 else line[:, :2] for line in lines])
    near = np.zeros(len(points), dtype=bool)
    for start in range(0, len(points), 128):
        chunk = points[start:start + 128]
        closest = ((chunk[:, None, :] - dense[None, :, :]) ** 2).sum(-1).min(axis=1)
        near[start:start + 128] = closest <= reach * reach
    return near


def maybe_mark(item, task: str, reach: Union[float, Callable], rng: random.Random, prob: float = MARK_PROB):
    """(the row to draw, the mark drawn on it or None). With chance `prob` a mark of the task's KINDS is drawn on
    `item` and the row comes back as its strokes (objects with x, y, t) with the mark's overlapping points added:
    the mark is {"kind", "points": how many were added}. `reach` is the overlap distance in the item's units, or a
    function of the row's recog_layout.polylines giving it. A row with no mark, or a mark that overlaps nothing,
    comes back as `item` itself."""
    if prob <= 0 or task not in KINDS or rng.random() >= prob:
        return item, None
    names, weights = zip(*KINDS[task])
    kind = rng.choices(names, weights)[0]
    lines = rl.polylines(item)
    if not lines:
        return item, {"kind": kind, "points": 0}
    distance = reach(lines) if callable(reach) else float(reach)
    spacing = _spacing(lines)
    times = np.concatenate([line[:, 2] for line in lines])
    earliest, latest = float(times.min()), float(times.max())
    # a mark has no recorded times: it is one section made after the writing it is on, every point at one time
    # (recog_layout.SECTION_PAUSE), as training/recog_eval.row_lines times the marks the cascade reads
    when = latest + rl.section_pause(rng.random(), latest - earliest)
    added = []
    for path in mark_paths(kind, lines, rng):
        points = _resample(path, spacing)
        near = overlapping(points, lines, distance)
        edges = np.flatnonzero(np.diff(np.concatenate([[0], near.astype(np.int8), [0]])))
        for a, b in zip(edges[::2], edges[1::2]):
            piece = points[a:b]
            added.append(SimpleNamespace(x=piece[:, 0].tolist(), y=piece[:, 1].tolist(),
                                         t=[when] * len(piece)))
    count = sum(len(s.x) for s in added)
    if not count:
        return item, {"kind": kind, "points": 0}
    strokes = [SimpleNamespace(x=line[:, 0].tolist(), y=line[:, 1].tolist(), t=line[:, 2].tolist()) for line in lines]
    return strokes + added, {"kind": kind, "points": count}
