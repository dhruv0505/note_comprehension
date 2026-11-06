"""
core.py — shared data structures + geometry utilities for synthetic note-page generation.

Everything downstream (loaders, block generators, layout engine) works with these types,
so every source dataset (IAMonDB, DeepWriting, MathWriting, QuickDraw, FC) just needs
a loader function that returns Ink objects in this format. See loaders.py.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import copy
import math
import random


# ----------------------------------------------------------------------------
# Core stroke types
# ----------------------------------------------------------------------------

@dataclass
class Stroke:
    """A single pen-down-to-pen-up stroke. x and y are parallel arrays."""
    x: List[float]
    y: List[float]
    # Optional per-point pen width, parallel to the others. None means "render at
    # whatever width the renderer likes" — the datasets carry no width, so only
    # synthesised ink sets it.
    w: Optional[List[float]] = None
    # Optional per-point time, parallel to the others, in the source's own units
    # (IAM-OnDB seconds; MathWriting and QuickDraw milliseconds). None when the source
    # has none, or the ink was built rather than loaded. recog_synth colours ink by it.
    t: Optional[List[float]] = None

    def bbox(self) -> Tuple[float, float, float, float]:
        return (min(self.x), min(self.y), max(self.x), max(self.y))

    def n_points(self) -> int:
        return len(self.x)


@dataclass
class Ink:
    """
    A piece of ink content at any granularity: a single word, a text line, a whole
    diagram, a sketch, etc. `children` lets you carry hierarchy from the source
    dataset (e.g. a "line" Ink whose children are "word" Inks) straight through
    the pipeline, so source-level annotations become level-0 labels for free.
    """
    strokes: List[Stroke]
    label: Optional[str] = None          # semantic label: "word", "textline", "diagram", quickdraw category, etc.
    class_name: Optional[str] = None     # taxonomy class if already known ("rectangle","oval","arrow",...)
    children: List["Ink"] = field(default_factory=list)
    source: Optional[str] = None         # which dataset this came from, for debugging/balancing
    writer: Optional[str] = None         # who wrote it, so one page can be one hand
    meta: Optional[dict] = None          # source-specific extras (e.g. a flowchart edge's endpoints)

    def bbox(self) -> Tuple[float, float, float, float]:
        xs0, ys0, xs1, ys1 = [], [], [], []
        for s in self.strokes:
            a, b, c, d = s.bbox()
            xs0.append(a); ys0.append(b); xs1.append(c); ys1.append(d)
        for ch in self.children:
            a, b, c, d = ch.bbox()
            xs0.append(a); ys0.append(b); xs1.append(c); ys1.append(d)
        if not xs0:
            return (0, 0, 0, 0)
        return (min(xs0), min(ys0), max(xs1), max(ys1))

    def n_points(self) -> int:
        return sum(s.n_points() for s in self.strokes) + sum(c.n_points() for c in self.children)

    def all_strokes(self) -> List[Stroke]:
        """Flattened stroke list including children, in a stable order."""
        out = list(self.strokes)
        for ch in self.children:
            out.extend(ch.all_strokes())
        return out


# ----------------------------------------------------------------------------
# Geometric transforms
# ----------------------------------------------------------------------------

def deep_copy_ink(ink: Ink) -> Ink:
    return copy.deepcopy(ink)


# Some Ink carries, in meta["exact"], the construction its strokes were drawn
# from — the true segment, the true circle (see geometry_diagrams). That travels
# in figure-local coordinates, so it has to be moved with the ink or the
# beautified render lands somewhere the ink is not.
#
# Only a uniform scale can be applied to it: an arc is stored as centre + radius
# + angles, and a circle squashed along one axis is not a circle any more. An
# anisotropic fit therefore drops the construction rather than emitting
# geometry that no longer matches the ink.

def _scale_exact(prims, scale_x: float, scale_y: float,
                  origin: Tuple[float, float], offset: Tuple[float, float]):
    if abs(scale_x - scale_y) > 1e-9:
        return None

    def move(point):
        return [offset[0] + (point[0] - origin[0]) * scale_x,
                offset[1] + (point[1] - origin[1]) * scale_y]

    out = []
    for prim in prims:
        moved = dict(prim)
        if prim.get("kind") == "segment":
            moved["a"], moved["b"] = move(prim["a"]), move(prim["b"])
        elif prim.get("kind") == "arc":
            moved["centre"] = move(prim["centre"])
            moved["radius"] = prim["radius"] * scale_x
        elif prim.get("kind") == "polygon":
            moved["points"] = [move(p) for p in prim["points"]]
        else:
            continue
        out.append(moved)
    return out


def _move_exact_meta(node: Ink, scale_x: float, scale_y: float,
                      origin: Tuple[float, float],
                      offset: Tuple[float, float]) -> None:
    """Bring a node's meta["exact"] along with its strokes, in place."""
    if not node.meta or "exact" not in node.meta:
        return
    moved = _scale_exact(node.meta["exact"], scale_x, scale_y, origin, offset)
    node.meta = dict(node.meta)
    if moved is None:
        node.meta.pop("exact")
    else:
        node.meta["exact"] = moved


def fit_ink_to_box(ink: Ink, box: Tuple[float, float, float, float],
                    preserve_aspect: bool = True, jitter: float = 0.0) -> Ink:
    """
    Return a NEW Ink (deep copy) rescaled + translated so its bbox fits inside `box`
    = (x0, y0, x1, y1). If preserve_aspect, scales uniformly and centers within box
    (never distorts letterforms). `jitter` adds small random scale/offset noise
    for visual variety across synthetic pages.
    """
    out = deep_copy_ink(ink)
    sx0, sy0, sx1, sy1 = out.bbox()
    sw, sh = max(sx1 - sx0, 1e-6), max(sy1 - sy0, 1e-6)
    bx0, by0, bx1, by1 = box
    bw, bh = bx1 - bx0, by1 - by0

    if preserve_aspect:
        scale = min(bw / sw, bh / sh)
        if jitter:
            scale *= 1.0 + random.uniform(-jitter, jitter)
        new_w, new_h = sw * scale, sh * scale
        # center within box
        off_x = bx0 + (bw - new_w) / 2.0
        off_y = by0 + (bh - new_h) / 2.0
        scale_x = scale_y = scale
    else:
        scale_x, scale_y = bw / sw, bh / sh
        off_x, off_y = bx0, by0

    def transform(s: Stroke) -> Stroke:
        return Stroke(
            x=[off_x + (px - sx0) * scale_x for px in s.x],
            y=[off_y + (py - sy0) * scale_y for py in s.y],
            t=s.t,
        )

    def walk(node: Ink) -> Ink:
        node.strokes = [transform(s) for s in node.strokes]
        _move_exact_meta(node, scale_x, scale_y, (sx0, sy0), (off_x, off_y))
        node.children = [walk(c) for c in node.children]
        return node

    return walk(out)


def transform_ink(ink: Ink, scale: float, dx: float, dy: float) -> Ink:
    """A NEW Ink scaled uniformly by `scale` then translated by (dx, dy).

    Unlike fit_ink_to_box this takes the scale as given rather than deriving it
    from a target box, which is what lets a whole page share one handwriting
    size: the caller picks the scale once and anchors each block wherever it
    belongs, instead of every block being stretched to fill its own slot.
    """
    out = deep_copy_ink(ink)

    def walk(node: Ink) -> Ink:
        node.strokes = [Stroke(x=[dx + px * scale for px in s.x],
                                y=[dy + py * scale for py in s.y], t=s.t)
                        for s in node.strokes]
        _move_exact_meta(node, scale, scale, (0.0, 0.0), (dx, dy))
        node.children = [walk(c) for c in node.children]
        return node

    return walk(out)


def union_bbox(boxes: List[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    xs0 = [b[0] for b in boxes]; ys0 = [b[1] for b in boxes]
    xs1 = [b[2] for b in boxes]; ys1 = [b[3] for b in boxes]
    return (min(xs0), min(ys0), max(xs1), max(ys1))
