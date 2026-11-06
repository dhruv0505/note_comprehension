"""
geometry_diagrams.py — build textbook geometry figures out of mined ink.

    python geometry_diagrams.py --demo geometry_demo.png
    python geometry_diagrams.py --cache shapes_bank.json --n 24

Nothing here draws a shape from an equation. Every stroke on the page is real
pen ink that shape_mining.py found in the pools — a mined line becomes any
segment, a mined oval becomes any circle or arc of one — re-warped to the size
the construction calls for. There are no ellipses: every curve on a figure is
round. So a synthesised right-angle box carries the same
tremor as the hand that drew the line it came from, and the figure as a whole
is annotated for free: each part knows what it is.

The pipeline is

    bank        mined line / oval ink, normalised to a unit shape
    figures     circle, polygons of any number of sides, semicircle, sector
    operations  inscribe, circumscribe, chord, diameter, altitude, angle marks,
                right-angle boxes, equal-side ticks, concatenate
    assembly    a few of the above, glued or stacked, then rotated as a whole

A Figure keeps its straight edges as `segments`, which is what makes gluing
possible: two figures are joined by mapping one's edge onto the other's, and
since both edges are then the same line, the copy is deleted.
"""

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple
from dataclasses import dataclass, field
import argparse
import itertools
import json
import math
import random
import re

import numpy as np

from core import Ink, Stroke, transform_ink
import shape_mining as S


UNIT = 100.0             # nominal figure size; the page rescales anyway

# A shape whose two dimensions land within this of each other is drawn as the
# equal-sided version — a true square — rather than by chance being very nearly
# one, which reads as a mistake rather than as a square.
EQUAL_SNAP = 0.12
MARK_EQUAL_PROB = 0.70   # chance equal angles get their arc marking
TICK_SIDES_PROB = 0.45   # chance equal sides get their tick marks (mark_equal_sides)
RIGHT_ANGLE_PROB = 0.55  # chance a square's corners get their little boxes
ALTITUDE_RIGHT_PROB = 0.65   # chance an altitude's foot is marked square


# ----------------------------------------------------------------------------
# The bank: mined ink, normalised so it can be redrawn at any size
# ----------------------------------------------------------------------------

@dataclass
class LineInk:
    """One mined line, as an offset from the chord it spans.

    `u` runs 0 to 1 along the chord and `v` is the sideways wander, in the same
    units — so scaling a segment scales its wobble with it, and a long line
    drawn from this ink wavers proportionally rather than looking ruled.
    """
    u: np.ndarray
    v: np.ndarray


@dataclass
class OvalInk:
    """One mined oval, as a radius profile around its own fitted ellipse.

    `radius` is sampled at `theta` in the frame where the fitted ellipse is the
    unit circle, so it is a little over and under 1 — the departure of a real
    hand from the ellipse it was aiming at. Redrawing at any size, aspect or
    angular span replays that departure instead of inventing jitter.
    """
    theta: np.ndarray
    radius: np.ndarray

    def profile(self, angles: np.ndarray, offset: float = 0.0) -> np.ndarray:
        """The radius at each angle, reading the loop as periodic."""
        wanted = (angles + offset) % (2 * math.pi)
        return np.interp(wanted, self.theta, self.radius, period=2 * math.pi)


@dataclass
class Bank:
    """Only lines and ovals: every straight thing on the page, a polygon's side
    included, is built out of a mined line rather than borrowed whole from a
    mined outline. See the polygon section for why."""
    lines: List[LineInk] = field(default_factory=list)
    ovals: List[OvalInk] = field(default_factory=list)
    # The dataset split the ink was mined from (None: not recorded, as in a bank
    # mined from every dataset), and how many lines and ovals each source gave.
    split: Optional[str] = None
    sources: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def ok(self) -> bool:
        return bool(self.lines and self.ovals)


def _line_ink(points: np.ndarray) -> Optional[LineInk]:
    start, end = points[0], points[-1]
    chord = end - start
    length = float(np.hypot(*chord))
    if length < 1e-9:
        return None
    angle = math.atan2(chord[1], chord[0])
    rotation = np.array([[math.cos(-angle), -math.sin(-angle)],
                         [math.sin(-angle), math.cos(-angle)]])
    local = (points - start) @ rotation.T / length
    return LineInk(u=local[:, 0], v=local[:, 1])


# Detecting an oval and borrowing one are different questions. The detector is
# set where it is so that a lopsided loop in someone's flowchart still counts as
# the oval it plainly is; a figure, though, redraws the ink it borrows at a size
# and an aspect the writer never drew it at, and every flaw is magnified by
# that. These are stricter than the detector on purpose — they are asking "was
# this drawn well", not "is this an oval".
OVAL_INK_CV = 0.038          # spread about the fitted ellipse
OVAL_INK_RADIAL_MAX = 0.10   # the worst single point, so no dents
OVAL_INK_CORNER = 40.0       # degrees; nothing with a visible corner in it

# Sampling, which the detector has no reason to care about but redrawing does.
# What is interpolated between samples is the RADIUS against the angle, not the
# position, so a gap is replayed as an arc at a drifting radius rather than as a
# chord across it — a wide step costs far less than it looks like it should, and
# this only has to catch a hole big enough to swing the radius across it.
#
# Note what is NOT here: a limit on the radius jump between neighbouring
# samples. It reads as a smoothness test and is really a sampling-density test,
# since the same shape sampled twice as finely halves every jump; the rate,
# dr/dtheta, is no better, because two samples that land at nearly the same
# angle send it to infinity whatever the shape. Smoothness is what `corner`
# already measures, and it bins the ink by angle before measuring, which is what
# makes it independent of how the pen happened to sample.
OVAL_INK_MIN_SAMPLES = 28
OVAL_INK_MAX_STEP = 0.60     # radians between neighbouring samples


def _oval_ink(hit) -> Optional[OvalInk]:
    metrics = hit.metrics
    if (metrics.get("radial_cv", 1.0) > OVAL_INK_CV
            or metrics.get("radial_max", 1.0) > OVAL_INK_RADIAL_MAX
            or metrics.get("corner", 1e9) > OVAL_INK_CORNER):
        return None

    fitted = S.fit_ellipse(hit.points)
    if fitted is None:
        return None
    centre, (a, b), angle = fitted
    if a < 1e-9 or b < 1e-9:
        return None
    rotation = np.array([[math.cos(-angle), -math.sin(-angle)],
                         [math.sin(-angle), math.cos(-angle)]])
    local = (hit.points - centre) @ rotation.T
    norm = np.column_stack([local[:, 0] / a, local[:, 1] / b])

    theta = np.arctan2(norm[:, 1], norm[:, 0]) % (2 * math.pi)
    radius = np.hypot(norm[:, 0], norm[:, 1])
    # sorting by angle puts the samples in loop order however many strokes they
    # arrived in and whichever way round each was drawn — the same trick the
    # corner test uses, and what lets an arc be cut out of the result
    order = np.argsort(theta)
    theta, radius = theta[order], radius[order]
    keep = np.concatenate([[True], np.diff(theta) > 1e-9])
    theta, radius = theta[keep], radius[keep]

    if len(theta) < OVAL_INK_MIN_SAMPLES:
        return None
    # the step across the wrap from the last sample back round to the first
    # counts too, or a loop with its gap at 0 degrees slips through
    steps = np.diff(theta, append=theta[0] + 2 * math.pi)
    if steps.max() > OVAL_INK_MAX_STEP:
        return None
    return OvalInk(theta=theta, radius=radius)


def build_bank(per_pool: Optional[int] = None, cache: Optional[str] = None,
               keep: int = 600, pools: Optional[Dict[str, List[Ink]]] = None,
               split: Optional[str] = None) -> Bank:
    """Mine the pools for primitives, or reload a cache of a previous mining.

    Every pool is mined by default rather than a sample of each. The quality
    bar for oval ink below only about one in ten mined ovals clears it, so a
    sample large enough to be worth drawing from has to start much larger than
    the number wanted. It takes minutes, which is why the result is kept on
    disk between runs.

    `pools` is what gets mined, defaulting to every dataset make_pdf loads. That
    suits the demo and not a dataset: the bank draws every underline, box, table
    rule, arrow and geometry side, so a split has to be mined from its own pools
    alone or held-out ink ends up on every training page. `split` names the
    split; it is written into the cache along with each primitive's source, and
    a cache is only reloaded for the split it was mined for — one from another
    split, or from before splits were recorded, raises instead of leaking in.
    """
    if cache:
        try:
            with open(cache) as f:
                raw = json.load(f)
        except (OSError, ValueError):
            raw = None                              # no cache yet, so mine one
        if raw is not None and split is not None and raw.get("split") != split:
            raise ValueError(f"{cache} was mined for split {raw.get('split')!r}, not "
                             f"{split!r}; mine a bank from this split's own pools")
        if raw is not None:
            try:
                return Bank(
                    lines=[LineInk(np.array(d["u"]), np.array(d["v"])) for d in raw["lines"]],
                    ovals=[OvalInk(np.array(d["theta"]), np.array(d["radius"]))
                           for d in raw["ovals"]],
                    split=raw.get("split"), sources=raw.get("sources", {}))
            except KeyError:
                pass                                # not a bank; mine over it

    if pools is None and split is not None:
        # a split's bank is mined from that split's pools (split_pools.py) and nothing else: one that is unreadable
        # must not be replaced with a bank of every dataset, held-out ink and all
        raise ValueError(f"no usable bank for split {split!r} at {cache}; rebuild it with split_pools.py")
    if pools is None:
        import make_pdf
        pools = make_pdf.build_pools()
    hits = S.mine(pools, per_pool=per_pool)
    # Only the first `keep` lines and ovals found are kept, and hits arrive pool by
    # pool, so unshuffled every line came from whichever pool was mined first.
    random.Random(0).shuffle(hits)

    bank = Bank(split=split)
    line_sources: List[str] = []
    oval_sources: List[str] = []
    for hit in hits:
        if hit.kind == "line" and len(bank.lines) < keep * 3:
            ink = _line_ink(hit.points)
            if ink is not None:
                bank.lines.append(ink)
                line_sources.append(hit.source)
        elif hit.kind == "oval" and len(bank.ovals) < keep:
            ink = _oval_ink(hit)
            if ink is not None:
                bank.ovals.append(ink)
                oval_sources.append(hit.source)
    from collections import Counter
    bank.sources = {"lines": dict(Counter(line_sources)),
                    "ovals": dict(Counter(oval_sources))}

    if cache:
        with open(cache, "w") as f:
            json.dump({"split": split, "sources": bank.sources,
                       "lines": [{"u": l.u.round(5).tolist(), "v": l.v.round(5).tolist(),
                                  "source": s}
                                 for l, s in zip(bank.lines, line_sources)],
                       "ovals": [{"theta": o.theta.round(5).tolist(),
                                  "radius": o.radius.round(5).tolist(), "source": s}
                                 for o, s in zip(bank.ovals, oval_sources)]}, f)
    return bank


# ----------------------------------------------------------------------------
# Emitting ink
# ----------------------------------------------------------------------------

def _cross(u: np.ndarray, v: np.ndarray) -> float:
    """The z of the cross product of two plane vectors.

    Written out rather than np.cross, which stopped accepting 2-D vectors in
    numpy 2.0.
    """
    return float(u[0] * v[1] - u[1] * v[0])


def _stroke(points: np.ndarray) -> Stroke:
    return Stroke(x=[float(v) for v in points[:, 0]],
                  y=[float(v) for v in points[:, 1]])


def segment(bank: Bank, rng: random.Random, start, end,
            wobble: float = 1.0) -> Stroke:
    """Draw a straight run from `start` to `end` with a mined line's hand."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    along = end - start
    length = float(np.hypot(*along))
    if length < 1e-9:
        return _stroke(np.vstack([start, end]))

    ink = rng.choice(bank.lines)
    direction = along / length
    normal = np.array([-direction[1], direction[0]])
    u, v = ink.u, ink.v
    if rng.random() < 0.5:              # either end could have been the start
        u, v = 1.0 - u[::-1], v[::-1]
    points = (start
              + np.outer(u * length, direction)
              + np.outer(v * length * wobble, normal))
    return _stroke(points)


def arc(bank: Bank, rng: random.Random, centre, rx: float, ry: float,
        start_angle: float = 0.0, end_angle: float = 2 * math.pi,
        rotation: float = 0.0, samples: int = 0) -> Stroke:
    """Draw an ellipse, or the piece of one between two angles.

    A full loop overshoots its start a little, the way a hand-drawn one does;
    an arc stops where it is told, since an angle marking that ran past itself
    would read as a different mark.
    """
    centre = np.asarray(centre, dtype=float)
    ink = rng.choice(bank.ovals)
    sweep = end_angle - start_angle
    closed = abs(abs(sweep) - 2 * math.pi) < 1e-6
    if closed:
        sweep += math.copysign(rng.uniform(0.04, 0.20), sweep)
    if not samples:
        samples = max(10, int(abs(sweep) / (2 * math.pi) * 64))

    angles = np.linspace(start_angle, start_angle + sweep, samples)
    radius = ink.profile(angles, offset=rng.uniform(0, 2 * math.pi))
    local = np.column_stack([radius * np.cos(angles) * rx,
                             radius * np.sin(angles) * ry])
    turn = np.array([[math.cos(rotation), -math.sin(rotation)],
                     [math.sin(rotation), math.cos(rotation)]])
    return _stroke(local @ turn.T + centre)


@dataclass(frozen=True)
class CornerJoin:
    """How loosely a hand joins one straight side of a closed shape to the next.

    Nobody drawing a polygon lands each line exactly on the end of the last: a
    line runs on a little past the corner, stops just short of it, or meets it a
    touch to one side. Each end of each side is decided on its own, in fractions
    of the shape's shortest side: an overshoot (`overshoot_prob`, up to
    `overshoot`), a gap (`gap_prob`, up to `gap`), or neither — and then nudged
    sideways by up to `drift`.

    Not every shape is drawn that way, though: `clean_prob` of them are drawn
    neatly, every side meeting its corners. Loose joins on every shape read as a
    tic rather than as a hand.
    """
    clean_prob: float
    overshoot_prob: float
    overshoot: float
    gap_prob: float
    gap: float
    drift: float


# A geometry figure is drawn with care: half of them close cleanly, and the rest
# run on slightly past a corner now and then, with hardly ever a gap.
FIGURE_JOIN = CornerJoin(clean_prob=0.50, overshoot_prob=0.20, overshoot=0.04,
                         gap_prob=0.04, gap=0.02, drift=0.0)
# A box round some writing (or a table's border) is often dashed off: its corners
# overshoot, open up and miss — though not on every box. Overshoot was on 55% of
# ends, up to 0.30 of the shortest side, which read as sloppy every time.
ENCLOSURE_JOIN = CornerJoin(clean_prob=0.30, overshoot_prob=0.30, overshoot=0.20,
                            gap_prob=0.30, gap=0.15, drift=0.06)


def loose_sides(corners, rng: random.Random,
                join: CornerJoin) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Where each side of the closed shape through `corners` is actually drawn:
    (start, end) from corner i to corner i+1, joined as loosely as `join` says.

    Only the ink moves. A figure keeps the true corners in its construction
    (Figure.add_segment's `drawn`), so gluing, vertices and the beautified
    drawing all still meet exactly.
    """
    points = [np.asarray(c, dtype=float) for c in corners]
    n = len(points)
    lengths = [float(np.hypot(*(points[(i + 1) % n] - points[i]))) for i in range(n)]
    scale = min(lengths) if lengths else 0.0
    if rng.random() < join.clean_prob:
        # drawn neatly this time: every side meets its corners
        return [(points[i], points[(i + 1) % n]) for i in range(n)]

    def reach(length: float) -> float:
        """How far past its corner one end runs: positive overshoots, negative
        falls short — never so short that most of the side is missing."""
        roll = rng.random()
        if roll < join.overshoot_prob:
            return rng.uniform(0.3, 1.0) * join.overshoot * scale
        if roll < join.overshoot_prob + join.gap_prob:
            return -min(rng.uniform(0.3, 1.0) * join.gap * scale, 0.25 * length)
        return 0.0

    sides = []
    for i in range(n):
        start, end = points[i], points[(i + 1) % n]
        if lengths[i] < 1e-9:
            sides.append((start, end))
            continue
        along = (end - start) / lengths[i]
        across = np.array([-along[1], along[0]])
        start = (start - along * reach(lengths[i])
                 + across * rng.uniform(-join.drift, join.drift) * scale)
        end = (end + along * reach(lengths[i])
               + across * rng.uniform(-join.drift, join.drift) * scale)
        sides.append((start, end))
    return sides


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

@dataclass
class Edge:
    """A straight side on the figure's outline, and which stroke draws it.

    Only the outline belongs here, because this list is what gluing reads: a
    chord, a diameter or the side of an inscribed polygon is straight too, but
    joining another figure along one would put it through the middle of this
    one rather than against it.

    `stroke` is the index into Figure.strokes, or -1 when the side is part of a
    stroke that draws more than just this side — a mined rectangle outline drawn
    in one go. Such a side can still be measured and glued along, but it cannot
    be deleted on its own, so gluing prefers the sides that can.
    """
    a: np.ndarray
    b: np.ndarray
    stroke: int = -1

    def length(self) -> float:
        return float(np.hypot(*(self.b - self.a)))


def exact_segment(start, end, role: str, mark: bool = False) -> Dict:
    return {"kind": "segment", "role": role, "mark": mark,
            "a": [float(start[0]), float(start[1])],
            "b": [float(end[0]), float(end[1])]}


def exact_arc(centre, radius: float, role: str, start: float = 0.0,
              end: float = 2 * math.pi, mark: bool = False) -> Dict:
    """A circle, or the arc of one from angle `start` to `end` (radians, in the
    page's own frame, y down), which is a full circle when they are 2 pi apart."""
    return {"kind": "arc", "role": role, "mark": mark,
            "centre": [float(centre[0]), float(centre[1])],
            "radius": float(radius), "start": float(start), "end": float(end)}


def _move_exact(prim: Optional[Dict], move, scale: float,
                angle: float) -> Optional[Dict]:
    """The same primitive after the figure it belongs to has been moved.

    An arc's angles are in the page's frame, so turning the figure turns them.
    """
    if prim is None:
        return None
    out = dict(prim)
    if prim["kind"] == "segment":
        out["a"] = [float(v) for v in move(prim["a"])]
        out["b"] = [float(v) for v in move(prim["b"])]
    elif prim["kind"] == "arc":
        out["centre"] = [float(v) for v in move(prim["centre"])]
        out["radius"] = prim["radius"] * scale
        out["start"] = prim["start"] + angle
        out["end"] = prim["end"] + angle
    return out


@dataclass
class Note:
    """A label written on a figure: a corner's name, a side's length, an angle's size.

    `kind` says which. On the page the label is a textline (a formula, in
    radians) of the figure (_label_line), and nothing ties it to the corner, side
    or angle it is beside.
    """
    text: str
    kind: str                       # "vertex" | "edge" | "angle"
    strokes: List[Stroke] = field(default_factory=list)
    # `word` for anything written out of character ink, `formula` for an angle
    # given in radians, which is a piece of maths and is labelled as one
    written: str = "word"
    source: str = "emnist"


def _label_line(note: Note) -> Ink:
    """A label as a line of the figure, the way a table holds its cells' lines.

    Written out of character ink it is a textline holding one word; in radians
    it is a formula.
    """
    if note.written == "formula":
        return Ink(strokes=note.strokes, class_name="formula", label=note.text,
                   source=note.source, meta={"geometry_label": note.kind})
    word = Ink(strokes=note.strokes, class_name="word", label=note.text,
               source=note.source)
    return Ink(strokes=[], children=[word], class_name="textline", label=note.text,
               source=note.source, meta={"geometry_label": note.kind})


@dataclass
class Figure:
    strokes: List[Stroke] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    # centre and radius of the circle this figure is (or is part of), so a later
    # operation can hang a chord or a diameter on it
    circle: Optional[Tuple[np.ndarray, float]] = None
    parts: List[str] = field(default_factory=list)      # names of what is drawn
    # The construction the ink was drawn from — the true segment, the true
    # circle — kept alongside it, one entry per stroke so the two stay in step
    # when a stroke is deleted. This is what the beautified drawing is rendered
    # from: not a smoothed trace of the ink, but the geometry the ink was an
    # attempt at, which is the only version where a tangent really is tangent.
    exact: List[Optional[Dict]] = field(default_factory=list)
    # The angles that may be arced as equal, each (vertex, toward_a, toward_b,
    # forced). They are arced only once the whole figure is assembled
    # (mark_equal_angles), so an arc count means one size across every shape
    # glued into it. `forced` marks an angle even with nothing equal to it: a
    # sector's.
    angle_candidates: List[Tuple[np.ndarray, np.ndarray, np.ndarray, bool]] = field(
        default_factory=list)
    # The labels written on it, filled in by annotate_figure at the very end —
    # after every scale, glue and rotation — so the writing sits upright however
    # the figure has been turned.
    notes: List[Note] = field(default_factory=list)

    def points(self) -> np.ndarray:
        return np.array([(x, y) for s in self.strokes for x, y in zip(s.x, s.y)])

    def centroid(self) -> np.ndarray:
        points = self.points()
        return points.mean(axis=0) if len(points) else np.zeros(2)

    def add(self, stroke: Stroke, name: str,
            exact: Optional[Dict] = None) -> int:
        self.strokes.append(stroke)
        self.parts.append(name)
        self.exact.append(exact)
        return len(self.strokes) - 1

    def add_segment(self, bank: "Bank", rng: random.Random, start, end,
                    role: str, wobble: float = 1.0, mark: bool = False,
                    drawn: Optional[Tuple] = None) -> int:
        """Draw a straight run, and remember the straight run it was.

        `mark` says this line annotates the figure rather than being part of its
        outline, which is how the beautified drawing knows to draw it lightly.
        It cannot be read off the role: a diameter inside a circle is an
        annotation, and the identical diameter closing a semicircle is that
        figure's own flat side.

        `drawn` is where the ink actually runs, when a hand's join at the corners
        (loose_sides) puts it somewhere other than the true segment; the true
        segment is still what is remembered.
        """
        ink_start, ink_end = drawn if drawn is not None else (start, end)
        return self.add(segment(bank, rng, ink_start, ink_end, wobble), role,
                        exact_segment(start, end, role, mark))

    def add_arc(self, bank: "Bank", rng: random.Random, centre, radius: float,
                role: str, start: float = 0.0, end: float = 2 * math.pi,
                mark: bool = False) -> int:
        """Draw a circle or an arc of one, and remember the true curve."""
        return self.add(arc(bank, rng, centre, radius, radius, start, end),
                        role, exact_arc(centre, radius, role, start, end, mark))

    def to_ink(self, source: str = "geometry") -> Ink:
        children = [Ink(strokes=[s], class_name=name, label=name, source=source)
                    for s, name in zip(self.strokes, self.parts)]

        # Each circle, arc and straight line of the figure is an object too, and
        # like a vertex it is made of single-point strokes that draw nothing: a
        # circle its start, middle and end, a segment its two ends. Points rather
        # than numbers in meta, so they move with the ink when the page places it.
        for part in figure_parts(self):
            children.append(Ink(
                strokes=[Stroke(x=[float(x)], y=[float(y)]) for x, y in part["points"]],
                class_name=part["class"], label=part["class"], source=source))

        # A vertex is a point, so that is what its object is: a single-point
        # stroke at the vertex itself. It draws nothing — a one-point polyline has
        # no length — which is the intent, since the vertex is already there in
        # the ink where its lines meet. What it adds is the vertex as a thing in
        # its own right: its angles, how they are marked, and its neighbours.
        figure_id = next(_FIGURE_IDS)
        vertices = figure_vertices(self)
        ids = [f"g{figure_id}v{i}" for i in range(len(vertices))]
        for vertex_id, vertex in zip(ids, vertices):
            x, y = (float(v) for v in vertex["point"])
            children.append(Ink(
                strokes=[Stroke(x=[x], y=[y])], class_name="vertex", label="vertex",
                source=source,
                meta={"id": vertex_id, "angles": vertex["angles"],
                      "angle_mark": vertex["angle_mark"],
                      "angle_mark_count": vertex["angle_mark_count"],
                      "connected": [ids[j] for j in vertex["connected"]]}))

        # each label written on the figure is a line of the figure itself
        children.extend(_label_line(note) for note in self.notes)

        return Ink(strokes=[], children=children,
                   class_name="geometry", source=source,
                   meta={"exact": [p for p in self.exact if p is not None]})


# ----------------------------------------------------------------------------
# Vertices
# ----------------------------------------------------------------------------
# Every point where a drawn line or arc of the figure ends — on its own, at
# another's end, or partway along another — is a vertex, named or not. Found on
# the exact construction rather than the ink, so two ends that are the same
# point in the construction are one vertex however the two strokes wobble.

_FIGURE_IDS = itertools.count()
VERTEX_TOL = 0.002      # two points this close are one vertex, as a fraction of the figure's size
VERTEX_PROBE = 0.01     # how far into an angle its inside is tested, likewise
# Drawn parts that are marks on the figure rather than lines of it — an angle's
# box or arcs, a side's ticks: they neither make a vertex nor join one to another.
ANGLE_MARK_ROLES = ("right angle", "angle mark")
TICK_ROLE = "tick mark"
MARK_ROLES = ANGLE_MARK_ROLES + (TICK_ROLE,)


def _rim(prim: Dict, angle: float) -> np.ndarray:
    """The point of an arc's circle at `angle`."""
    return (np.asarray(prim["centre"], dtype=float)
            + prim["radius"] * np.array([math.cos(angle), math.sin(angle)]))


def _is_closed(prim: Dict) -> bool:
    return (prim["kind"] == "arc"
            and abs(abs(prim["end"] - prim["start"]) - 2 * math.pi) < 1e-6)


def _part_ends(prim: Dict) -> List[np.ndarray]:
    """Where a drawn part ends: a segment's two ends, a partial arc's, none for a
    closed loop."""
    if prim["kind"] == "segment":
        return [np.asarray(prim["a"], dtype=float), np.asarray(prim["b"], dtype=float)]
    if _is_closed(prim):
        return []
    return [_rim(prim, t) for t in (prim["start"], prim["end"])]


def figure_parts(figure: Figure) -> List[Dict]:
    """Every circle, arc and straight line of the figure, as it is labelled.

    Marks on the figure are left out — a right-angle box, an angle's arcs, the
    ticks across a side — since they are not lines of it (a vertex's angle
    marking says the first two). Each comes back as

        class    "arc" for a circle or an arc of one, "segment" for a line
        points   an arc: its start, middle and end, going counterclockwise as
                 the page is seen (y down); a full circle starts and ends at its
                 0-degree point, the right-hand end of its horizontal diameter,
                 and its middle is the 180-degree point. A segment: its two
                 ends, the upper one (then the left one) first.
    """
    parts = []
    for prim in figure.exact:
        if not prim or prim.get("role") in MARK_ROLES:
            continue
        if prim["kind"] == "segment":
            ends = sorted((tuple(float(v) for v in prim["a"]),
                           tuple(float(v) for v in prim["b"])),
                          key=lambda p: (p[1], p[0]))
            parts.append({"class": "segment", "points": ends})
        elif prim["kind"] == "arc":
            if _is_closed(prim):
                start = end = _rim(prim, 0.0)
                middle = _rim(prim, math.pi)
            else:
                # with y down, a growing angle turns clockwise on the page, so
                # counterclockwise runs from the larger angle to the smaller
                low, high = sorted((prim["start"], prim["end"]))
                start, middle, end = _rim(prim, high), _rim(prim, (low + high) / 2), _rim(prim, low)
            parts.append({"class": "arc",
                          "points": [tuple(float(v) for v in p) for p in (start, middle, end)]})
    return parts


def _on_part(prim: Dict, point: np.ndarray, tol: float) -> bool:
    """Whether a point lies on a drawn part, to within `tol`.

    Measured on the exact curve, not its drawn polyline: a circle's polyline
    samples sit about a unit and a half apart, far coarser than the tolerance
    two constructed points are held to, so a chord's end placed exactly on its
    circle read as off it — 101 of 1,947 vertices over 400 figures (a chord's,
    diameter's or radius's end on a circle) lost the arc through them, and with
    it their angles and their neighbours along it.
    """
    if prim["kind"] == "segment":
        return _distance_to_primitive(point, prim) <= tol
    radius = prim["radius"]
    local = np.asarray(point, dtype=float) - np.asarray(prim["centre"], dtype=float)
    if abs(float(np.hypot(*local)) - radius) > tol:
        return False
    if _is_closed(prim):
        return True
    theta = math.atan2(local[1], local[0])
    sweep = prim["end"] - prim["start"]
    sign = 1.0 if sweep >= 0 else -1.0
    position = ((theta - prim["start"]) * sign) % (2 * math.pi)
    step = tol / radius
    return position <= abs(sweep) + step or position >= 2 * math.pi - step


def _along_part(prim: Dict, point: np.ndarray, tol: float) -> Tuple[float, List[np.ndarray]]:
    """Where `point` lies along a drawn part it is on, and the directions the
    part leaves it in: one at an end, two partway along or on a closed loop.

    The position is a distance along a segment or an angle round an arc, from
    the part's start, which is what orders the vertices on it."""
    if prim["kind"] == "segment":
        a = np.asarray(prim["a"], dtype=float)
        b = np.asarray(prim["b"], dtype=float)
        length = float(np.hypot(*(b - a)))
        if length < 1e-9:
            return 0.0, []
        unit = (b - a) / length
        t = float((point - a) @ unit)
        rays = ([-unit] if t > tol else []) + ([unit] if t < length - tol else [])
        return t, rays

    local = point - np.asarray(prim["centre"], dtype=float)
    theta = math.atan2(local[1], local[0])
    sweep = prim["end"] - prim["start"]
    sign = 1.0 if sweep >= 0 else -1.0
    forward = sign * np.array([-math.sin(theta), math.cos(theta)])
    position = ((theta - prim["start"]) * sign) % (2 * math.pi)
    if _is_closed(prim):
        return position, [forward, -forward]
    step = tol / prim["radius"]
    if position > abs(sweep) + step:
        position -= 2 * math.pi       # a hair before the start, wrapped by the modulo
    rays = (([-forward] if position > step else [])
            + ([forward] if position < abs(sweep) - step else []))
    return position, rays


def _inside_outline(outline: List[Dict], point: np.ndarray) -> bool:
    """Even-odd: whether a point is inside the figure's outline."""
    x, y = float(point[0]), float(point[1])
    inside = False
    for prim in outline:
        pts = exact_polyline(prim)
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
                inside = not inside
    return inside


def figure_vertices(figure: Figure) -> List[Dict]:
    """Every vertex of the figure, named or not.

    A vertex is any point where a drawn line or arc ends, merged with any other
    end or part it lands on — so an altitude's foot partway along its base is a
    vertex, and so is a chord's end on its circle. A point where two lines only
    cross is not; nor are right-angle boxes and angle arcs, which are marks on
    an angle, not lines of the figure. Each comes back as

        point             the vertex
        angles            degrees, one per gap between the lines meeting here
                          that lies inside the figure: one at a polygon's corner
                          (over 180 at a dent), more where more lines meet
        angle_mark        "right" for a right-angle box, "arc" for angle arcs,
                          "none"
        angle_mark_count  how many concentric arcs, or 1 for a box
        connected         indices of the vertices one drawn line or arc runs to

    "Inside" is an even-odd test, just into each gap, against the figure's
    outline: its sides, a sector's radii, a semicircle's diameter, a circle. A
    construction drawn on it (chord, altitude, inscribed polygon, incircle) joins
    vertices but is not outline, and neither is a side two glued shapes share,
    which concatenate flags as it glues.
    """
    size = _figure_size(figure)
    tol = VERTEX_TOL * size
    parts = [p for p in figure.exact if p and p.get("role") not in MARK_ROLES]

    points: List[np.ndarray] = []
    for prim in parts:
        for end in _part_ends(prim):
            if all(float(np.hypot(*(end - p))) > tol for p in points):
                points.append(end)

    rays: List[List[np.ndarray]] = [[] for _ in points]
    joined: List[Set[int]] = [set() for _ in points]
    for prim in parts:
        on = []
        for index, point in enumerate(points):
            if _on_part(prim, point, tol):
                position, leaving = _along_part(prim, point, tol)
                rays[index].extend(leaving)
                on.append((position, index))
        on.sort()
        pairs = list(zip(on, on[1:]))
        if _is_closed(prim) and len(on) > 2:
            pairs.append((on[-1], on[0]))
        for (_, i), (_, j) in pairs:
            if i != j:
                joined[i].add(j)
                joined[j].add(i)

    outline = [p for p in parts if not p.get("mark") and not p.get("shared")
               and not str(p.get("role", "")).startswith("inscribed")]
    boxed = _right_angle_corners(figure)
    arcs = [p for p in figure.exact if p and p.get("role") == "angle mark"]

    vertices = []
    for index, point in enumerate(points):
        directions = sorted({round(math.atan2(r[1], r[0]), 6) for r in rays[index]})
        angles = []
        if len(directions) >= 2:
            for k, start in enumerate(directions):
                end = directions[(k + 1) % len(directions)]
                if k == len(directions) - 1:
                    end += 2 * math.pi
                width = end - start
                if width < 1e-6:
                    continue
                middle = start + width / 2.0
                probe = point + VERTEX_PROBE * size * np.array([math.cos(middle), math.sin(middle)])
                if _inside_outline(outline, probe):
                    angles.append(round(math.degrees(width), 1))
        arc_count = sum(1 for prim in arcs
                        if float(np.hypot(*(np.asarray(prim["centre"], dtype=float) - point))) <= tol)
        right = any(float(np.hypot(*(corner - point))) <= tol for corner in boxed)
        vertices.append({"point": point, "angles": angles,
                         "angle_mark": "right" if right else "arc" if arc_count else "none",
                         "angle_mark_count": 1 if right else arc_count,
                         "connected": sorted(joined[index])})
    return vertices


def _apply(figure: Figure, scale: float, angle: float,
           offset: np.ndarray) -> Figure:
    """A new Figure scaled about the origin, turned, and moved."""
    turn = np.array([[math.cos(angle), -math.sin(angle)],
                     [math.sin(angle), math.cos(angle)]]) * scale

    def move(p):
        return np.asarray(p, dtype=float) @ turn.T + offset

    strokes = []
    for s in figure.strokes:
        moved = move(np.column_stack([s.x, s.y]))
        strokes.append(Stroke(x=moved[:, 0].tolist(), y=moved[:, 1].tolist(), t=s.t))
    edges = [Edge(move(e.a), move(e.b), e.stroke) for e in figure.edges]
    circle = None
    if figure.circle is not None:
        centre, radius = figure.circle
        circle = (move(centre), radius * scale)
    exact = [_move_exact(p, move, scale, angle) for p in figure.exact]
    candidates = [(move(v), move(a), move(b), forced)
                  for v, a, b, forced in figure.angle_candidates]
    return Figure(strokes=strokes, edges=edges, circle=circle,
                  parts=list(figure.parts), exact=exact,
                  angle_candidates=candidates)


def rotated(figure: Figure, rng: random.Random,
            spread: float = math.pi) -> Figure:
    """Turn the whole figure about its own centre by a random angle."""
    centre = figure.centroid()
    angle = rng.uniform(-spread, spread)
    moved = _apply(figure, 1.0, angle, np.zeros(2))
    return _apply(moved, 1.0, 0.0, centre - moved.centroid())


# --- the shapes themselves ---------------------------------------------------

def make_circle(bank: Bank, rng: random.Random, size: float = UNIT) -> Figure:
    """A circle. Never an ellipse: every curve a figure draws is round.

    Its part is named "circle outline" rather than "arc", since a drawn part's
    name becomes its child's class_name and `arc` is a label class.
    """
    radius = size / 2.0
    figure = Figure(circle=(np.zeros(2), radius))
    figure.add_arc(bank, rng, (0, 0), radius, "circle outline")
    return figure


# ----------------------------------------------------------------------------
# Polygons
# ----------------------------------------------------------------------------
#
# Every closed straight-sided figure is built the way the triangle always was:
# n mined lines, laid corner to corner. Nothing replays a mined outline whole.
# Drawing side by side is what makes a side a thing in its own right — it can be
# measured, marked, glued onto and, when something is glued there, deleted — and
# it is the only construction that generalises past four sides, since nobody's
# notes contain a mined heptagon to borrow.
#
# Two distributions shape which polygon turns up, and both are meant to match
# what is actually on a page rather than to spread evenly over what is possible:
#
#   sides       falls away geometrically. Triangles and quadrilaterals are most
#               of it, a pentagon or hexagon happens, a decagon almost never.
#   symmetry    within a number of sides, the symmetric forms dominate. A hexagon
#               someone draws is nearly always the regular one; six sides at
#               unrelated angles reads as a blob, not as a figure. Most come out
#               convex, and a minority are dented inward — a star, an arrowhead,
#               a chevron — which are real figures too.

POLYGON_NAMES = {3: "triangle", 4: "quadrilateral", 5: "pentagon", 6: "hexagon",
                 7: "heptagon", 8: "octagon", 9: "nonagon", 10: "decagon"}
MAX_SIDES = max(POLYGON_NAMES)
SIDE_DECAY = 0.45        # each extra side is this much as likely as the last
SYMMETRIC_PROB = 0.72    # of polygons, how many are built symmetric
CONCAVE_PROB = 0.22      # how many are dented inward rather than convex


def polygon_sides(rng: random.Random, minimum: int = 3,
                  maximum: int = MAX_SIDES) -> int:
    """How many sides this polygon has, fewer being far likelier than more."""
    sides = list(range(minimum, maximum + 1))
    weights = [SIDE_DECAY ** (n - minimum) for n in sides]
    return rng.choices(sides, weights=weights)[0]


def _radial(angles: Sequence[float], radii: Sequence[float],
            aspect: float = 1.0) -> List[np.ndarray]:
    """Corners at the given angles and radii, in the order they go round.

    Sorting by angle is what keeps the outline simple whatever the radii are: a
    polygon whose corners are in angular order about an interior point cannot
    have one side cross another, so a dented one is still a figure and not a
    knot. It is also why every dent here is star-shaped about the centre, which
    is the kind of concavity a drawn figure has.
    """
    order = sorted(range(len(angles)), key=lambda i: angles[i] % (2 * math.pi))
    return [np.array([radii[i] * math.cos(angles[i]) * aspect,
                      radii[i] * math.sin(angles[i])]) for i in order]


NEEDLE = 0.38            # thinnest a polygon may be, across against along


def _placed(corners: Sequence[np.ndarray], size: float) -> List[np.ndarray]:
    """Corners centred on the origin and scaled to `size` across the widest way.

    A polygon thinner than NEEDLE is widened first. Corners drawn at random
    angles occasionally all land on one side, and what comes back is a splinter
    that happens to have seven sides rather than a figure with seven sides —
    stretching it across is what makes it the second thing. The stretch is along
    an axis, so a shape symmetric about one keeps its mirror.
    """
    points = np.asarray(corners, dtype=float)
    points = points - points.mean(axis=0)
    span = points.max(axis=0) - points.min(axis=0)
    thin, wide = (0, 1) if span[0] < span[1] else (1, 0)
    if span[wide] > 1e-9 and span[thin] < NEEDLE * span[wide]:
        points[:, thin] *= NEEDLE * span[wide] / max(span[thin], 1e-9)
        span = points.max(axis=0) - points.min(axis=0)
    if span.max() > 1e-9:
        points = points * (size / span.max())
    return [p for p in points]


def _spread_angles(count: int, rng: random.Random, span: float = 2 * math.pi,
                   start: float = 0.0, floor: float = 0.55) -> List[float]:
    """`count` increasing angles across `span`, none crowding its neighbour.

    Angles drawn independently clump, and two corners a few degrees apart give a
    side too short to see and two others nearly in line — which is a polygon
    with fewer sides, drawn badly. So each gap gets `floor` of its even share
    outright and the rest of the span is shared out at random.
    """
    even = span / count
    slack = span - count * even * floor
    cuts = sorted(rng.random() for _ in range(count - 1))
    shares = np.diff([0.0] + cuts + [1.0]) * slack
    angles, here = [], start
    for share in shares:
        angles.append(here)
        here += even * floor + float(share)
    return angles


def _dent(corners: List[np.ndarray], i: int, rng: random.Random) -> bool:
    """Push one corner in past the chord joining its two neighbours.

    That chord is exactly where a corner stops pointing outward and starts
    pointing inward, so the dent is measured against it rather than by shrinking
    a radius and hoping it is enough: how far a corner has to come in before it
    is a dent depends on how far apart its neighbours are, and on a pentagon it
    is most of the way to the centre.

    The corner only moves along its own ray from the centre, which is what the
    angular order that keeps the outline simple rests on. Says whether it moved.
    """
    n = len(corners)
    here, behind = corners[i], corners[(i - 1) % n]
    chord = corners[(i + 1) % n] - behind
    across = _cross(here, chord)
    if abs(across) < 1e-9:
        return False
    touch = _cross(behind, chord) / across   # where the ray meets that chord
    if touch <= 0.05:
        return False                         # the chord is not out this way
    corners[i] = here * touch * rng.uniform(0.55, 0.90)
    return True


def _dented(corners: List[np.ndarray], rng: random.Random) -> List[np.ndarray]:
    """Take a bite or two out of a polygon.

    Never more than two: a polygon dented at every other corner is a zigzag, and
    the figures worth drawing have one bite out of them.
    """
    for i in rng.sample(range(len(corners)),
                        rng.randint(1, min(2, max(1, len(corners) // 3)))):
        _dent(corners, i, rng)
    return corners


def _mirror_dented(corners: List[np.ndarray], rng: random.Random) -> List[np.ndarray]:
    """Dent a polygon built about the vertical axis without losing the axis.

    A corner on the axis is its own reflection, so denting it costs the figure
    nothing and gives the notch of an arrowhead or a dart — which is why one is
    picked whenever the polygon has one. Failing that a corner off the axis is
    dented and its reflection replaced by the reflection of the result, which
    keeps the mirror line by pairing the bite with a matching one.
    """
    on_axis = [i for i, c in enumerate(corners) if abs(c[0]) <= 1e-6]
    off_axis = [i for i, c in enumerate(corners) if abs(c[0]) > 1e-6]
    if not (on_axis or off_axis):
        return corners
    i = rng.choice(on_axis or off_axis)
    twin = [j for j, c in enumerate(corners)
            if np.allclose(c, (-corners[i][0], corners[i][1]), atol=1e-6)]
    if _dent(corners, i, rng):
        for j in twin:
            corners[j] = np.array([-corners[i][0], corners[i][1]])
    return corners


QUAD_KINDS = ("rectangle", "square", "trapezium", "parallelogram", "rhombus",
              "kite")
QUAD_WEIGHTS = (0.32, 0.26, 0.16, 0.12, 0.08, 0.06)


def _quadrilateral(rng: random.Random, size: float,
                   kind: Optional[str] = None) -> Tuple[List[np.ndarray], str]:
    """One of the four-sided figures that has a name, since those are the ones
    that get drawn. All but the parallelogram have a mirror line; the
    parallelogram has the half-turn instead, which is symmetry enough to read as
    a deliberate shape rather than as four lines that happened to close."""
    kind = kind or rng.choices(QUAD_KINDS, weights=QUAD_WEIGHTS)[0]

    if kind in ("rectangle", "square"):
        width = size
        height = width if kind == "square" else width * rng.uniform(0.42, 1.0)
        if abs(width - height) / width < EQUAL_SNAP:
            height = width          # near-square reads as a mistake, so snap it
        if rng.random() < 0.5:
            width, height = height, width
        kind = "square" if abs(width - height) < 1e-9 else "rectangle"
        a, b = width / 2.0, height / 2.0
        corners = [(-a, -b), (a, -b), (a, b), (-a, b)]
    elif kind == "parallelogram":
        height = size * rng.uniform(0.45, 0.85)
        lean = size * rng.uniform(0.20, 0.50)
        corners = [(0.0, 0.0), (size, 0.0), (size + lean, -height), (lean, -height)]
    elif kind == "trapezium":
        top = size * rng.uniform(0.30, 0.72)
        height = size * rng.uniform(0.40, 0.85)
        corners = [(0.0, 0.0), (size, 0.0),
                   ((size + top) / 2.0, -height), ((size - top) / 2.0, -height)]
    else:                            # rhombus and kite: the two diagonals
        across = size / 2.0
        upper = size * rng.uniform(0.35, 0.60)
        lower = upper if kind == "rhombus" else upper * rng.uniform(1.35, 2.2)
        corners = [(0.0, -upper), (across, 0.0), (0.0, lower), (-across, 0.0)]
    return _placed([np.array(c, dtype=float) for c in corners], size), kind


def polygon_corners(n: int, size: float, rng: random.Random,
                    kind: Optional[str] = None) -> Tuple[List[np.ndarray], str]:
    """`n` corners of a polygon, centred on the origin, and what to call it.

    The name is settled here rather than by the caller because it is decided by
    the construction: a rectangle that came out square is a square, and a
    symmetric hexagon whose corners were pulled inward is a star.
    """
    if n == 3:
        kind = kind or rng.choice(TRIANGLE_KINDS)
        return triangle_corners(kind, size, rng), f"{kind} triangle"

    name = POLYGON_NAMES.get(n, f"{n}-gon")
    symmetric = rng.random() < SYMMETRIC_PROB
    concave = rng.random() < CONCAVE_PROB
    turn = rng.uniform(0, 2 * math.pi)

    if n == 4 and symmetric and not concave:
        return _quadrilateral(rng, size, kind)

    if symmetric and concave and n >= 6 and n % 2 == 0:
        # alternate long and short radii: the star, and the only dent that keeps
        # the polygon symmetric — pulling one corner of a regular one inward
        # would break the very thing the shape was chosen for. Six corners is
        # the fewest that can be starred; four alternating radii are a kite,
        # which the named quadrilaterals already draw and draw better.
        inner = rng.uniform(0.30, 0.55)
        angles = [turn + 2 * math.pi * k / n for k in range(n)]
        radii = [1.0 if k % 2 == 0 else inner for k in range(n)]
        corners = _placed(_radial(angles, radii), size)
        return corners, f"{'star' if is_concave(corners) else 'symmetric'} {name}"

    if symmetric and not concave and rng.random() < 0.55:
        angles = [turn + 2 * math.pi * k / n for k in range(n)]
        corners = _placed(_radial(angles, [1.0] * n), size)
        # a regular quadrilateral is a square, and gets called one
        return corners, "square" if n == 4 else f"regular {name}"

    if symmetric:
        # a mirror line: half the corners are chosen and the rest are their
        # reflections, so the figure is symmetric without being regular — the
        # arrowheads, chevrons and lopsided kites that a book is full of
        pairs, spare = divmod(n, 2)
        axis = [-math.pi / 2] if spare else []   # odd: one corner sits on the line
        if concave and not spare:
            # a notch that keeps the mirror line has to sit on it, and an even
            # polygon of pairs alone has no corner there — so it is given both
            # ends of the line instead, which is the arrowhead and the dart
            pairs, axis = pairs - 1, [-math.pi / 2, math.pi / 2]
        offsets = _spread_angles(pairs, rng, span=math.pi - 0.30,
                                 start=0.35 if len(axis) == 2 else 0.15)
        radii = [rng.uniform(0.62, 1.0) for _ in range(pairs)]
        angles = ([math.pi / 2 - o for o in offsets]
                  + [math.pi / 2 + o for o in offsets] + axis)
        # the reflection shares its radius; a corner on the line is its own
        both = radii + radii + [rng.uniform(0.62, 1.0) for _ in axis]
        corners = _radial(angles, both)
        if concave:
            corners = _mirror_dented(corners, rng)
        corners = _placed(corners, size)
        return corners, f"{'concave' if is_concave(corners) else 'symmetric'} {name}"

    # no symmetry: corners at unrelated angles round the centre. Corners on a
    # circle are convex whatever the angles are, so a dent has to be asked for.
    angles = _spread_angles(n, rng, start=turn)
    aspect = 1.0 if rng.random() < 0.4 else rng.uniform(0.55, 1.0)
    corners = _radial(angles, [1.0] * n, aspect)
    if concave:
        corners = _dented(corners, rng)
    corners = _placed(corners, size)
    return corners, f"{'concave' if is_concave(corners) else 'irregular'} {name}"


def make_polygon(bank: Bank, rng: random.Random, size: float = UNIT,
                 n: Optional[int] = None,
                 kind: Optional[str] = None) -> Figure:
    """A closed polygon of mined lines, plus the marks its shape earns.

    Every side is its own stroke and its own Edge, so anything can be glued to
    any of them and the beautified drawing has the true segment behind each.
    """
    n = n or polygon_sides(rng)
    corners, name = polygon_corners(n, size, rng, kind)
    figure = Figure()
    drawn = loose_sides(corners, rng, FIGURE_JOIN)
    for i in range(len(corners)):
        start, end = corners[i], corners[(i + 1) % len(corners)]
        index = figure.add_segment(bank, rng, start, end, "side", drawn=drawn[i])
        figure.edges.append(Edge(start, end, index))

    mark_equal = rng.random() < MARK_EQUAL_PROB

    # Which corners get the little square is settled here, so the arcs can be
    # kept off them. A square is the case that needs it: all
    # four of its angles are equal to each other, so marking equal angles would
    # arc exactly the four corners the boxes are on.
    right_triangle = name.endswith("right triangle")
    boxed: Set[int] = set()
    if right_triangle and rng.random() < RIGHT_ANGLE_PROB:
        # the right angle is at the corner the two axis-aligned sides meet
        boxed = {0}
    elif name in ("square", "rectangle") and rng.random() < RIGHT_ANGLE_PROB:
        boxed = set(range(4))

    if mark_equal:
        # only offered here; which of them are equal, and to what, is decided
        # once the whole figure is assembled (mark_equal_angles). A corner that
        # points inward is left out: an arc across a reflex angle sits outside
        # the figure and reads as a different mark entirely.
        angles = interior_angles(corners)
        count = len(corners)
        for i in range(count):
            if angles[i] < math.pi - 1e-6 and i not in boxed:
                figure.angle_candidates.append(
                    (corners[i], corners[(i + 1) % count], corners[(i - 1) % count], False))
    if boxed and right_triangle:
        right_angle_box(figure, bank, rng, corners[0], corners[1], corners[2])
    elif boxed:
        add_right_angle_boxes(figure, bank, rng)
    return figure


TRIANGLE_KINDS = ("equilateral", "isosceles", "right", "scalene")


def triangle_corners(kind: str, size: float, rng: random.Random) -> List[np.ndarray]:
    """Three corners of a triangle of the named kind, centred on the origin."""
    if kind == "equilateral":
        pts = [(0.0, 0.0), (size, 0.0), (size / 2.0, -size * math.sqrt(3) / 2.0)]
    elif kind == "isosceles":
        # apex over the midpoint of the base: the two slanted sides are equal
        height = size * rng.uniform(0.55, 1.30)
        pts = [(0.0, 0.0), (size, 0.0), (size / 2.0, -height)]
    elif kind == "right":
        pts = [(0.0, 0.0), (size, 0.0), (0.0, -size * rng.uniform(0.5, 1.2))]
    else:
        pts = [(0.0, 0.0), (size, 0.0),
               (size * rng.uniform(0.10, 0.85), -size * rng.uniform(0.4, 1.1))]
    corners = [np.array(p, dtype=float) for p in pts]
    centre = sum(corners) / 3.0
    return [c - centre for c in corners]


def _side_lengths(corners: Sequence[np.ndarray]) -> List[float]:
    n = len(corners)
    return [float(np.hypot(*(corners[(i + 1) % n] - corners[i]))) for i in range(n)]


def make_triangle(bank: Bank, rng: random.Random, size: float = UNIT,
                  kind: Optional[str] = None) -> Figure:
    """Three mined lines closed into a triangle, plus its marks."""
    return make_polygon(bank, rng, size, n=3, kind=kind)


def make_semicircle(bank: Bank, rng: random.Random, size: float = UNIT) -> Figure:
    """Half a disc: the arc, and the diameter that closes it."""
    radius = size / 2.0
    figure = Figure(circle=(np.zeros(2), radius))
    figure.add_arc(bank, rng, (0, 0), radius, "arc outline", 0.0, math.pi)
    start, end = np.array([radius, 0.0]), np.array([-radius, 0.0])
    index = figure.add_segment(bank, rng, start, end, "diameter")
    figure.edges.append(Edge(start, end, index))
    return figure


def make_sector(bank: Bank, rng: random.Random, size: float = UNIT) -> Figure:
    """A slice cut from a disc: an arc, and a radius drawn to each of its ends.

    Both radii are straight, so either can be glued to — a sector is the piece
    that most often ends up joined to something else.
    """
    radius = size / 2.0
    span = rng.uniform(math.pi / 5, math.pi * 1.4)
    start_angle = rng.uniform(0, 2 * math.pi)
    figure = Figure(circle=(np.zeros(2), radius))
    figure.add_arc(bank, rng, (0, 0), radius, "arc outline", start_angle,
                   start_angle + span)
    centre = np.zeros(2)
    for angle in (start_angle, start_angle + span):
        rim = np.array([radius * math.cos(angle), radius * math.sin(angle)])
        # not a mark: these two radii are the sector's own straight sides
        index = figure.add_segment(bank, rng, centre, rim, "radius")
        figure.edges.append(Edge(centre.copy(), rim, index))
    return figure


# ----------------------------------------------------------------------------
# Operations on a figure
# ----------------------------------------------------------------------------

# When a construction line would be drawn along one already there. A right
# triangle is the case that matters: the perpendicular from either of its acute
# corners to the opposite side runs straight down a side that is already drawn,
# so the figure comes out with that side inked twice, in two different hands.
# Anything laid this closely on top of an existing line is not drawn at all.
DOUBLE_ANGLE = 5.0          # degrees between the two directions
DOUBLE_OFFSET = 0.02        # apart, as a fraction of the figure's own size
DOUBLE_OVERLAP = 0.5        # of the shorter line's length, lying along the other


def _figure_size(figure: Figure) -> float:
    """The extent of everything drawn so far, for scaling the tolerances above.

    Arcs included, not just segments: a plain circle has no segments at all, and
    measuring it as UNIT rather than as itself put every tolerance below on the
    wrong scale for exactly the figures the chord and diameter operations act on.
    """
    points = []
    for prim in figure.exact:
        if not prim:
            continue
        if prim.get("kind") == "segment":
            points.extend((prim["a"], prim["b"]))
        elif prim.get("kind") == "arc":
            points.extend(exact_polyline(prim))
    if not points:
        return UNIT
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return max(max(xs) - min(xs), max(ys) - min(ys)) or UNIT


# How close a new line's end has to come to something already drawn to count as
# meeting it, as a fraction of the figure's size.
ENDPOINT_TOUCH_TOL = 0.02


def _distance_to_primitive(point, prim) -> float:
    """Shortest distance from a point to one exact primitive."""
    point = np.asarray(point, dtype=float)
    if prim.get("kind") == "segment":
        start = np.asarray(prim["a"], dtype=float)
        end = np.asarray(prim["b"], dtype=float)
        along = end - start
        length_sq = float(along @ along)
        if length_sq < 1e-12:
            return float(np.hypot(*(point - start)))
        t = min(1.0, max(0.0, float((point - start) @ along) / length_sq))
        return float(np.hypot(*(point - (start + along * t))))
    points = exact_polyline(prim)
    return float(np.min(np.hypot(points[:, 0] - point[0], points[:, 1] - point[1])))


def _endpoint_is_supported(figure: Figure, point, size: float) -> bool:
    """Whether a line ending here would meet something already drawn.

    The invariant this protects: no line in a figure stops in free space. It is
    the *drawn* arc that has to be met, not the circle it came from — a
    semicircle and a sector both keep the whole circle on the figure so a chord
    or a diameter can be hung off them, but only part of that circle is inked,
    and a chord to a point on the half that was never drawn hangs in mid-air.
    """
    for prim in figure.exact:
        if prim and _distance_to_primitive(point, prim) <= ENDPOINT_TOUCH_TOL * size:
            return True
    return False


def _segments_coincide(a0, a1, b0, b1, size: float) -> bool:
    """Whether the segment a0-a1 lies along b0-b1 closely enough to double it."""
    a0, a1, b0, b1 = (np.asarray(p, dtype=float) for p in (a0, a1, b0, b1))
    along = a1 - a0
    length = float(np.hypot(*along))
    other = b1 - b0
    other_length = float(np.hypot(*other))
    if length < 1e-9 or other_length < 1e-9:
        return False

    unit = along / length
    turn = abs(math.degrees(math.atan2(abs(_cross(unit, other)),
                                       abs(float(unit @ other)))))
    if turn > DOUBLE_ANGLE:
        return False
    if max(abs(_cross(unit, b0 - a0)), abs(_cross(unit, b1 - a0))) > DOUBLE_OFFSET * size:
        return False

    # both projected onto a0-a1, so the shared run is measured on one axis
    starts = sorted((float((b0 - a0) @ unit), float((b1 - a0) @ unit)))
    shared = min(length, starts[1]) - max(0.0, starts[0])
    return shared >= DOUBLE_OVERLAP * min(length, other_length)


def _would_double(figure: Figure, start, end) -> bool:
    """Whether drawing start-end would lay a second line over an existing one."""
    size = _figure_size(figure)
    for prim in figure.exact:
        if not prim or prim.get("kind") != "segment":
            continue
        if _segments_coincide(start, end, prim["a"], prim["b"], size):
            return True
    return False


def right_angle_box(figure: Figure, bank: Bank, rng: random.Random,
                    corner, toward_a, toward_b, size: Optional[float] = None):
    """The little square that says this angle is ninety degrees.

    Two short segments, drawn with mined line ink like everything else, set a
    step in along each arm and closed across.
    """
    corner = np.asarray(corner, dtype=float)
    arm_a = np.asarray(toward_a, dtype=float) - corner
    arm_b = np.asarray(toward_b, dtype=float) - corner
    len_a, len_b = np.hypot(*arm_a), np.hypot(*arm_b)
    if len_a < 1e-9 or len_b < 1e-9:
        return
    step = size or min(len_a, len_b) * rng.uniform(0.10, 0.17)
    foot_a = corner + arm_a / len_a * step
    foot_b = corner + arm_b / len_b * step
    far = foot_a + (foot_b - corner)
    figure.add_segment(bank, rng, foot_a, far, "right angle", wobble=1.4, mark=True)
    figure.add_segment(bank, rng, far, foot_b, "right angle", wobble=1.4, mark=True)


def angle_arc(figure: Figure, bank: Bank, rng: random.Random, vertex,
              toward_a, toward_b, count: int = 1, size: Optional[float] = None):
    """One or more little arcs across an angle, the mark for "these are equal".

    The arc is cut out of a mined oval, so the mark is drawn by the same hand as
    the rest of the figure. Two or three concentric arcs distinguish a second
    and third group of equal angles in the same figure.
    """
    vertex = np.asarray(vertex, dtype=float)
    arm_a = np.asarray(toward_a, dtype=float) - vertex
    arm_b = np.asarray(toward_b, dtype=float) - vertex
    len_a, len_b = np.hypot(*arm_a), np.hypot(*arm_b)
    if len_a < 1e-9 or len_b < 1e-9:
        return
    start = math.atan2(arm_a[1], arm_a[0])
    end = math.atan2(arm_b[1], arm_b[0])
    # the mark belongs inside the angle, so take the way round that is shorter
    while end - start > math.pi:
        end -= 2 * math.pi
    while end - start < -math.pi:
        end += 2 * math.pi

    base = size or min(len_a, len_b) * rng.uniform(0.16, 0.24)
    for k in range(count):
        radius = base * (1.0 + 0.28 * k)
        figure.add_arc(bank, rng, vertex, radius, "angle mark", start, end, mark=True)


# Radians. Equal means equal in the construction, so this only absorbs float
# noise: marks and labels say what measuring the intended figure would find,
# and 2.5 degrees of slack let two angles a reader could see differ share an arc.
EQUAL_ANGLE_TOL = 1e-6
MAX_ANGLE_GROUPS = 3                    # three concentric arcs is the most a
                                        # reader can tell apart at this size


def interior_angles(corners: Sequence[np.ndarray]) -> List[float]:
    """The angle inside the polygon at each corner, reflex ones included.

    Measured against the polygon's own winding, so a dent comes out over pi
    rather than as its outside angle — which is the whole difference between
    knowing a corner points inward and not.
    """
    n = len(corners)
    area = sum(_cross(corners[i], corners[(i + 1) % n]) for i in range(n))
    winding = 1.0 if area >= 0 else -1.0
    angles = []
    for i in range(n):
        back = corners[(i - 1) % n] - corners[i]
        forth = corners[(i + 1) % n] - corners[i]
        if np.hypot(*back) < 1e-9 or np.hypot(*forth) < 1e-9:
            angles.append(math.pi)
            continue
        # the interior is on the left of a polygon wound one way and on the
        # right of one wound the other, which is what `winding` is for; the
        # angle itself is swept from the outgoing side back to the incoming one
        turn = math.atan2(_cross(forth, back), float(back @ forth)) * winding
        angles.append(turn % (2 * math.pi))
    return angles


def is_concave(corners: Sequence[np.ndarray]) -> bool:
    """Does any corner point inward?

    Asked of the corners after they are built rather than of the intention
    behind them, because a dent can be asked for and not arrive — radii pulled
    in by a little on a polygon with few enough sides leave every corner still
    pointing out. What the figure is called follows from what it is.
    """
    return any(a > math.pi + 1e-6 for a in interior_angles(corners))


def mark_equal_angles(figure: Figure, bank: Bank, rng: random.Random):
    """Arc whichever of the figure's candidate angles are equal to each other.

    Run once the figure is assembled, over every shape glued into it, so the
    number of arcs means one size across the whole figure: two shapes each
    marking their own equal pair with one arc would otherwise claim the pairs
    are equal to each other. The angles themselves are compared, exactly as the
    construction has them, so on a triangle it comes to the same thing as
    comparing the opposite sides, and on a regular hexagon it marks all six.

    A forced angle (a sector's) is arced even with nothing equal to it, but it
    still takes an arc count no other size has.

    Corners getting a right-angle box were never offered: the box already says
    ninety, and a lone arc on the corner's partner would say "equal" with
    nothing left to be equal to. One point carries at most one mark: where
    glued shapes bring two candidate corners together, only the first is arced
    (the other is an equal angle left unmarked, never a false one), and nothing
    is arced where a box already sits.
    """
    size = _figure_size(figure)
    tolerance = VERTEX_TOL * size
    measured = []
    for vertex, toward_a, toward_b, forced in figure.angle_candidates:
        arm_a, arm_b = toward_a - vertex, toward_b - vertex
        if np.hypot(*arm_a) < 1e-9 or np.hypot(*arm_b) < 1e-9:
            continue
        # the arc goes the shorter way round, so this is the angle it marks
        angle = math.atan2(abs(_cross(arm_a, arm_b)), float(arm_a @ arm_b))
        measured.append((angle, vertex, toward_a, toward_b, forced))

    groups: List[List[tuple]] = []
    for item in sorted(measured, key=lambda m: m[0]):
        if groups and item[0] - groups[-1][-1][0] <= EQUAL_ANGLE_TOL:
            groups[-1].append(item)
        else:
            groups.append([item])
    rng.shuffle(groups)

    taken = list(_right_angle_corners(figure))
    marked = 0
    for group in groups:
        members = []
        for item in group:
            here = item[1]
            if any(float(np.hypot(*(here - p))) <= tolerance
                   for p in taken + [m[1] for m in members]):
                continue
            members.append(item)
        if len(members) < 2 and not any(m[4] for m in members):
            continue                        # nothing to be equal to
        if marked == MAX_ANGLE_GROUPS:
            break
        marked += 1
        for _, vertex, toward_a, toward_b, _ in members:
            taken.append(vertex)
            angle_arc(figure, bank, rng, vertex, toward_a, toward_b, count=marked)


def add_right_angle_boxes(figure: Figure, bank: Bank, rng: random.Random):
    """Box every corner of a four-sided figure, the mark for a square."""
    if len(figure.edges) < 4:
        return
    for i in range(4):
        here = figure.edges[i]
        previous = figure.edges[(i - 1) % 4]
        right_angle_box(figure, bank, rng, here.a, here.b, previous.a)


# The side's counterpart to the arcs on equal angles: short strokes across the
# middle of each side that is equal to another — one for the first group of
# equal sides, two for the second, three for the third.
EQUAL_SIDE_TOL = 1e-6           # relative; float noise only, as with EQUAL_ANGLE_TOL
MAX_TICK_GROUPS = 3             # as with the arcs, the most a reader can tell apart
TICK_LENGTH = (0.045, 0.070)    # of the figure's size
TICK_GAP = (0.30, 0.45)         # between the strokes of one mark, of a stroke's length
TICK_LEAN = 12.0                # degrees a whole mark may lean off square to its side
# Only the outline's own straight sides get ticked: a sector's two radii are its
# sides, while a chord or a diameter drawn on a figure is a construction.
TICK_SIDE_ROLES = ("side", "radius")


def side_ticks(figure: Figure, bank: Bank, rng: random.Random, start, end,
               count: int, size: float):
    """`count` short strokes across the middle of the side from `start` to `end`.

    Each is a mined line like every other mark. The strokes of one mark lean
    together, the way a hand drawing two ticks in a row holds the pen the same
    for both, and each strays a few degrees from that on its own.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    along = end - start
    length = float(np.hypot(*along))
    if length < 1e-9 or count < 1:
        return
    unit = along / length
    tick = min(size * rng.uniform(*TICK_LENGTH), 0.35 * length)
    # the whole mark stays near the middle, even on a short side
    gap = min(tick * rng.uniform(*TICK_GAP), 0.4 * length / max(count - 1, 1))
    lean = math.radians(rng.uniform(-TICK_LEAN, TICK_LEAN))
    middle = start + along * rng.uniform(0.46, 0.54)
    for k in range(count):
        turn = math.pi / 2 + lean + math.radians(rng.uniform(-3.0, 3.0))
        across = np.array([unit[0] * math.cos(turn) - unit[1] * math.sin(turn),
                           unit[0] * math.sin(turn) + unit[1] * math.cos(turn)])
        centre = middle + unit * (k - (count - 1) / 2.0) * gap
        half = across * tick * rng.uniform(0.45, 0.55)
        figure.add_segment(bank, rng, centre - half, centre + half, TICK_ROLE,
                           wobble=1.4, mark=True)


def mark_equal_sides(figure: Figure, bank: Bank, rng: random.Random) -> int:
    """Tick whichever straight sides of the figure are equal to each other.

    Lengths are read off the finished construction, after any gluing, so the
    marks say what is true of the figure as drawn: a side of one glued shape
    that matches a side of the other is ticked as the equal it is, and two
    groups that merely got the same count in their own shapes cannot claim an
    equality that is not there. Every side of a group is ticked or none is — a
    lone tick has nothing to be equal to. Which group gets one stroke and which
    gets two is shuffled, and past MAX_TICK_GROUPS the rest go unmarked.

    Returns how many groups were marked.
    """
    size = _figure_size(figure)
    lengths = {index: float(np.hypot(*np.subtract(prim["b"], prim["a"])))
               for index, prim in enumerate(figure.exact)
               if prim and prim.get("kind") == "segment" and not prim.get("mark")
               and prim.get("role") in TICK_SIDE_ROLES}

    groups: List[List[int]] = []
    for index in sorted(lengths, key=lengths.get):
        here = lengths[index]
        if here < 1e-9:
            continue
        if groups and here - lengths[groups[-1][0]] <= EQUAL_SIDE_TOL * here:
            groups[-1].append(index)
        else:
            groups.append([index])
    groups = [group for group in groups if len(group) >= 2]
    rng.shuffle(groups)

    for count, group in enumerate(groups[:MAX_TICK_GROUPS], 1):
        for index in group:
            prim = figure.exact[index]
            side_ticks(figure, bank, rng, prim["a"], prim["b"], count, size)
    return min(len(groups), MAX_TICK_GROUPS)


def add_altitude(figure: Figure, bank: Bank, rng: random.Random,
                 corners: Sequence[np.ndarray]):
    """Drop a perpendicular from one corner to the opposite side.

    The foot can land outside the side it is dropped to, on an obtuse triangle;
    the line is then drawn to where the side would reach, which is how it is
    drawn in a book, and the side itself is extended to meet it.
    """
    # Every corner is tried, starting from a random one, because on a right
    # triangle the perpendicular from either acute corner lands on the corner
    # with the right angle — and the "altitude" is then just one of the sides
    # drawn over again. Only the apex opposite the hypotenuse gives a new line.
    first = rng.randrange(3)
    for step in range(3):
        apex_index = (first + step) % 3
        apex = corners[apex_index]
        base_start = corners[(apex_index + 1) % 3]
        base_end = corners[(apex_index + 2) % 3]

        along = base_end - base_start
        length = float(np.hypot(*along))
        if length < 1e-9:
            continue
        direction = along / length
        travel = float((apex - base_start) @ direction)
        foot = base_start + direction * travel
        if not _would_double(figure, apex, foot):
            break
    else:
        return              # nowhere to drop one that is not already drawn

    figure.add_segment(bank, rng, apex, foot, "altitude", mark=True)
    if not 0.0 <= travel <= length:
        # the foot is off the end of the base — carry the base out to meet it
        outside = base_start if travel < 0 else base_end
        figure.add_segment(bank, rng, outside, foot, "extension", mark=True)
    if rng.random() < ALTITUDE_RIGHT_PROB:
        toward = base_end if travel < length / 2 else base_start
        right_angle_box(figure, bank, rng, foot, apex, toward)


def add_chord(figure: Figure, bank: Bank, rng: random.Random):
    """Join two points of the figure's circle.

    Retried a few times because a chord drawn at the wrong angle lands on the
    flat side of a semicircle, or on a diameter already there, and one line over
    another is not a chord — it is the same line drawn twice.
    """
    if figure.circle is None:
        return
    centre, radius = figure.circle
    size = _figure_size(figure)
    for _ in range(12):
        first = rng.uniform(0, 2 * math.pi)
        second = first + rng.uniform(math.pi / 3, math.pi * 5 / 3)
        ends = [centre + radius * np.array([math.cos(a), math.sin(a)])
                for a in (first, second)]
        if _would_double(figure, ends[0], ends[1]):
            continue
        # both ends have to land on ink, not merely on the circle: a semicircle
        # keeps the whole circle so a chord can be hung off it, but only half of
        # it was ever drawn
        if not all(_endpoint_is_supported(figure, end, size) for end in ends):
            continue
        figure.add_segment(bank, rng, ends[0], ends[1], "chord", mark=True)
        return


def add_diameter(figure: Figure, bank: Bank, rng: random.Random):
    """Draw a diameter, and sometimes the radius mark that goes with it."""
    if figure.circle is None:
        return
    centre, radius = figure.circle
    size = _figure_size(figure)
    for _ in range(12):
        angle = rng.uniform(0, 2 * math.pi)
        step = radius * np.array([math.cos(angle), math.sin(angle)])
        ends = [centre - step, centre + step]
        if _would_double(figure, ends[0], ends[1]):
            continue        # a semicircle's own flat side is a diameter already
        if not all(_endpoint_is_supported(figure, end, size) for end in ends):
            continue        # only part of this circle is inked
        figure.add_segment(bank, rng, ends[0], ends[1], "diameter", mark=True)
        if rng.random() < 0.35:
            square = angle + math.pi / 2
            rim = centre + radius * np.array([math.cos(square), math.sin(square)])
            # the centre end lands on the diameter just drawn, so only the rim
            # end can be stranded
            if (not _would_double(figure, centre, rim)
                    and _endpoint_is_supported(figure, rim, size)):
                figure.add_segment(bank, rng, centre, rim, "radius", mark=True)
        return


def _incircle(corners: Sequence[np.ndarray]) -> Tuple[np.ndarray, float]:
    """Centre and radius of the circle touching all three sides."""
    lengths = [float(np.hypot(*(corners[(i + 2) % 3] - corners[(i + 1) % 3])))
               for i in range(3)]           # length opposite each corner
    total = sum(lengths)
    centre = sum(l * c for l, c in zip(lengths, corners)) / total
    half = total / 2.0
    area = abs(_cross(corners[1] - corners[0], corners[2] - corners[0])) / 2.0
    return centre, float(area / half)


def _circumcircle(corners: Sequence[np.ndarray]) -> Optional[Tuple[np.ndarray, float]]:
    """Centre and radius of the circle through all three corners."""
    (ax, ay), (bx, by), (cx, cy) = [tuple(c) for c in corners]
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        return None
    ua = ax * ax + ay * ay
    ub = bx * bx + by * by
    uc = cx * cx + cy * cy
    centre = np.array([(ua * (by - cy) + ub * (cy - ay) + uc * (ay - by)) / d,
                       (ua * (cx - bx) + ub * (ax - cx) + uc * (bx - ax)) / d])
    return centre, float(np.hypot(*(corners[0] - centre)))


# A circle drawn inside or around a polygon is only a construction when it
# really touches: on a triangle there is always such a circle and it is computed
# exactly, and on a regular polygon the centre and the two radii follow from the
# symmetry. Any other polygon has no incircle and no circumcircle, and drawing a
# circle that nearly touches some of the sides is worse than drawing none —
# it teaches the model a tangency that is not there.

def _regular_centre(corners: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    """The centre of a polygon whose corners are all the same distance from it,
    or None when there is no such point."""
    centre = np.mean(corners, axis=0)
    reach = [float(np.hypot(*(c - centre))) for c in corners]
    sides = _side_lengths(corners)
    # exactly, to float noise: a circle drawn as touching every side has to
    # touch every one, not nearly
    if (max(reach) - min(reach)) / max(reach) > EQUAL_SIDE_TOL:
        return None
    if (max(sides) - min(sides)) / max(sides) > EQUAL_SIDE_TOL:
        return None                          # equal reach but unequal sides
    return centre


def inscribe_circle(figure: Figure, bank: Bank, rng: random.Random,
                    corners: Optional[Sequence[np.ndarray]] = None):
    """Draw the circle that sits inside the figure, touching every side."""
    if not corners:
        return
    if len(corners) == 3:
        centre, radius = _incircle(corners)
    else:
        centre = _regular_centre(corners)
        if centre is None:
            return
        # the touching radius is the distance to a side, not to a corner
        n = len(corners)
        radius = min(float(np.hypot(*((corners[i] + corners[(i + 1) % n]) / 2.0
                                      - centre)))
                     for i in range(n))
    figure.add_arc(bank, rng, centre, radius, "incircle", mark=True)


def circumscribe_circle(figure: Figure, bank: Bank, rng: random.Random,
                        corners: Optional[Sequence[np.ndarray]] = None):
    """Draw the circle that passes through the figure's corners."""
    if not corners:
        return
    if len(corners) == 3:
        found = _circumcircle(corners)
        if found is None:
            return
        centre, radius = found
    else:
        centre = _regular_centre(corners)
        if centre is None:
            return
        radius = float(np.hypot(*(corners[0] - centre)))
    figure.add_arc(bank, rng, centre, radius, "circumcircle", mark=True)


def inscribe_polygon(figure: Figure, bank: Bank, rng: random.Random):
    """Put a polygon inside the figure's circle, every corner on the curve.

    The number of sides is drawn the same falling way as anywhere else, and the
    corners are either evenly spaced — the regular polygon — or at unrelated
    angles. Four corners get a third option: two opposite pairs (both
    diagonals diameters), which is what makes an inscribed quadrilateral a
    genuine rectangle rather than one that merely has its corners on the curve.
    """
    if figure.circle is None:
        return
    centre, radius = figure.circle

    def rim(angle):
        return centre + radius * np.array([math.cos(angle), math.sin(angle)])

    n = polygon_sides(rng)
    start = rng.uniform(0, 2 * math.pi)
    name = f"inscribed {POLYGON_NAMES.get(n, str(n) + '-gon')}"
    if n == 4 and rng.random() < 0.5:
        span = rng.uniform(math.pi / 5, math.pi / 2)
        angles = [start, start + span, start + math.pi, start + math.pi + span]
        name = "inscribed rectangle"
    elif rng.random() < SYMMETRIC_PROB:
        angles = [start + 2 * math.pi * k / n for k in range(n)]
        name = f"inscribed regular {POLYGON_NAMES.get(n, str(n) + '-gon')}"
    else:
        angles = _spread_angles(n, rng, start=start)

    corners = [rim(a) for a in angles]
    drawn = loose_sides(corners, rng, FIGURE_JOIN)
    for i in range(n):
        figure.add_segment(bank, rng, corners[i], corners[(i + 1) % n], name,
                           drawn=drawn[i])


# ----------------------------------------------------------------------------
# Putting figures together
# ----------------------------------------------------------------------------

# Two figures are put together by gluing them edge to edge, and only that way.
# Dropping one on top of another about a shared centre was tried and dropped:
# where a circle in a square means something — inscribed, circumscribed — the
# construction that means it already draws it, and everything else it produced
# was two figures piled on one spot rather than one figure.


def _merge(first: Figure, second: Figure) -> Figure:
    shift = len(first.strokes)
    edges = list(first.edges) + [
        Edge(e.a, e.b, e.stroke + shift if e.stroke >= 0 else -1)
        for e in second.edges]
    return Figure(strokes=first.strokes + second.strokes,
                  edges=edges,
                  circle=first.circle,
                  parts=first.parts + second.parts,
                  exact=first.exact + second.exact,
                  angle_candidates=first.angle_candidates + second.angle_candidates)


def concatenate(base: Figure, other: Figure, rng: random.Random) -> Figure:
    """Glue `other` onto `base` along a straight edge of each.

    The second figure is turned and scaled so its chosen edge lands exactly on
    the first one's — that is what "making the sizes match" comes to, one
    similarity transform fixed by two point pairs. Of the two ways to pair the
    endpoints, the one taken is whichever puts the second figure on the far side
    of the shared edge, so the two sit against each other rather than one on top
    of the other; the two differ by a half turn about the edge's midpoint, so
    neither mirrors the ink.

    Both figures then draw the same line, so the second's copy is deleted.
    """
    takers = [e for e in other.edges if e.stroke >= 0 and e.length() > 1e-9]
    givers = [e for e in base.edges if e.length() > 1e-9]
    if not takers or not givers:
        # nothing straight to glue along, and gluing is the only way two figures
        # go together, so the newcomer is dropped rather than piled on top
        return base

    # The newcomer is scaled by the ratio of the two edges, so gluing a whole
    # figure onto some stub of a side shrinks it to a speck against its host.
    # Both ends are therefore chosen from the longer edges each figure has.
    def substantial(edges: List[Edge]) -> List[Edge]:
        longest = max(e.length() for e in edges)
        return [e for e in edges if e.length() >= 0.55 * longest]

    target = rng.choice(substantial(givers))
    source = rng.choice(substantial(takers))
    base_centre = base.centroid()

    best = None
    for first, second in ((target.a, target.b), (target.b, target.a)):
        scale = target.length() / source.length()
        angle = (math.atan2(*(second - first)[::-1])
                 - math.atan2(*(source.b - source.a)[::-1]))
        turn = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]]) * scale
        offset = first - source.a @ turn.T
        moved = _apply(other, scale, angle, offset)

        edge = second - first
        side = _cross(edge, moved.centroid() - first)
        against = _cross(edge, base_centre - first)
        # want them on opposite sides; among the two pairings prefer that one
        score = -side * against
        if best is None or score > best[0]:
            best = (score, moved)

    moved = best[1]
    # the glued edge is now drawn twice — drop the copy the newcomer brought
    keep = [i for i in range(len(moved.strokes)) if i != source.stroke]
    remap = {old: new for new, old in enumerate(keep)}
    trimmed = Figure(strokes=[moved.strokes[i] for i in keep],
                     parts=[moved.parts[i] for i in keep],
                     exact=[moved.exact[i] for i in keep],
                     edges=[Edge(e.a, e.b, remap.get(e.stroke, -1))
                            for e in moved.edges if e.stroke != source.stroke],
                     circle=moved.circle,
                     angle_candidates=moved.angle_candidates)

    glued = _merge(base, trimmed)
    # The shared edge is inside the joined figure now. It is still drawn, once,
    # so it is flagged as no longer part of the outline (figure_vertices)...
    if target.stroke >= 0 and glued.exact[target.stroke]:
        glued.exact[target.stroke] = dict(glued.exact[target.stroke], shared=True)
    # ...and nothing else may be glued along it
    glued.edges = [e for e in glued.edges
                   if not (np.allclose(e.a, target.a) and np.allclose(e.b, target.b))]
    return glued


# ----------------------------------------------------------------------------
# Writing on the figure — naming corners, and giving lengths and angles
# ----------------------------------------------------------------------------
# A textbook figure is half drawing and half writing: the corners are named so
# the prose can refer to them, a side carries its length, an angle carries its
# size. The writing is real handwriting, one character at a time out of the
# EMNIST/SD19 pool (loaders.load_emnist), so it is the same kind of ink as
# everything else on the page rather than a typeset overlay.
#
# Each label is a line of the figure itself — a textline, or a formula in
# radians — the way a table holds its cells' lines (_label_line), and nothing
# ties it to the corner, side or angle it is beside.

# O and I are left out: handwritten, one reads as a zero and the other as a 1.
VERTEX_NAMES = "ABCDEFGHJKLMNPQRSTUVWXYZ"
# An unknown length or angle. These are the letters a textbook actually uses for
# one, and every one of them is a class EMNIST has.
UNKNOWNS = "xyzwabcmnpqrst"

LABEL_VERTICES_PROB = 0.55      # chance a figure names its corners at all
LABEL_EDGE_PROB = 0.45          # chance any one side is given a length
LABEL_ANGLE_PROB = 0.30         # chance any one corner is given an angle
UNKNOWN_PROB = 0.35             # a length or angle written as a letter, not a number

# All as fractions of the figure's overall size.
LABEL_HEIGHT = 0.105            # cap height of the written characters
VERTEX_CLEARANCE = 0.075        # how far outside its corner a name sits
EDGE_CLEARANCE = 0.070          # how far off its side a length sits
ANGLE_CLEARANCE = 0.185         # how far in from its corner an angle sits
GLYPH_GAP = 0.16                # between characters, as a fraction of their height
LABEL_CLEARANCE = 0.018         # blank kept around a label, as a fraction of size
# Multiples of the first choice of distance to try when that one is occupied.
LABEL_RETRIES = (1.0, 1.35, 1.75, 2.2)


def _write_text(text: str, glyphs: Dict[str, List[Ink]], rng: random.Random,
                height: float) -> List[Stroke]:
    """`text` written in character ink, `height` tall, centred on the origin.

    Each character is scaled to the same height rather than to its own natural
    one. For the one- and two-character labels here — a capital, a two-digit
    angle — that is what a writer does anyway, and it keeps a "12" from coming
    out with a 1 half the height of its 2.

    Centred on the origin rather than on a given point so the caller can try the
    same written label in several places (see _place_label) without rewriting it
    in a different hand each time.
    """
    # all of it or nothing: a "12" with no 1 in the pool is not written as "2"
    if not text or not all(glyphs.get(character) for character in text):
        return []
    placed: List[Ink] = []
    cursor = 0.0
    for character in text:
        glyph = rng.choice(glyphs[character])
        x0, y0, x1, y1 = glyph.bbox()
        scale = height / max(y1 - y0, 1e-6)
        placed.append(transform_ink(glyph, scale, cursor - x0 * scale, -y0 * scale))
        cursor += (x1 - x0) * scale + GLYPH_GAP * height
    if not placed:
        return []

    return _centred([s for ink in placed for s in ink.all_strokes()])


def _centred(strokes: Sequence[Stroke]) -> List[Stroke]:
    """The same strokes moved so their bounding box is centred on the origin."""
    xs = [x for s in strokes for x in s.x]
    ys = [y for s in strokes for y in s.y]
    return _shifted(strokes, -(min(xs) + max(xs)) / 2.0, -(min(ys) + max(ys)) / 2.0)


def _write_formula(expression: Ink, height: float) -> List[Stroke]:
    """A MathWriting expression scaled to `height`, centred on the origin."""
    x0, y0, x1, y1 = expression.bbox()
    if y1 - y0 <= 0:
        return []
    scale = height / (y1 - y0)
    moved = transform_ink(expression, scale, -x0 * scale, -y0 * scale)
    strokes = moved.all_strokes()
    return _centred(strokes) if strokes else []


def _shifted(strokes: Sequence[Stroke], dx: float, dy: float) -> List[Stroke]:
    """The same strokes moved, in plain floats.

    Plain floats deliberately: the anchor comes from the figure's own geometry,
    and a numpy scalar rides all the way through to json.dumps and stops it.
    """
    return [Stroke(x=[float(x + dx) for x in s.x], y=[float(y + dy) for y in s.y], t=s.t)
            for s in strokes]


def _strokes_bbox(strokes: Sequence[Stroke]) -> Tuple[float, float, float, float]:
    xs = [x for s in strokes for x in s.x]
    ys = [y for s in strokes for y in s.y]
    return min(xs), min(ys), max(xs), max(ys)


def _place_label(strokes: List[Stroke], anchor, direction, size: float,
                 ink: np.ndarray, taken: List[Tuple[float, float, float, float]],
                 distances: Sequence[float]) -> Optional[List[Stroke]]:
    """The written label put where it does not sit on anything already there.

    Tried at each distance out from its anchor in turn, and given up on if none
    of them is clear. Giving up is the right answer often enough to be worth
    saying: where two glued shapes bring their corners together there is
    genuinely nowhere to write, and a number laid across the lines is worse than
    no number at all.
    """
    margin = LABEL_CLEARANCE * size
    for distance in distances:
        centre = np.asarray(anchor, dtype=float) + np.asarray(direction, dtype=float) * distance
        moved = _shifted(strokes, float(centre[0]), float(centre[1]))
        x0, y0, x1, y1 = _strokes_bbox(moved)
        x0 -= margin; y0 -= margin; x1 += margin; y1 += margin
        if len(ink) and bool(np.any((ink[:, 0] >= x0) & (ink[:, 0] <= x1)
                                    & (ink[:, 1] >= y0) & (ink[:, 1] <= y1))):
            continue                    # over a line the figure already drew
        if any(not (x1 < tx0 or x0 > tx1 or y1 < ty0 or y0 > ty1)
               for tx0, ty0, tx1, ty1 in taken):
            continue                    # over a label already written
        taken.append((x0, y0, x1, y1))
        return moved
    return None


# How near ninety a corner has to measure to count as a right angle.
RIGHT_ANGLE_TOL = math.radians(4.0)


def _right_angle_corners(figure: Figure) -> List[np.ndarray]:
    """The corners carrying the little square that says "ninety degrees".

    right_angle_box draws the square as two segments, foot_a->far and
    far->foot_b, stepped in along the two arms — so the corner they were stepped
    in from is foot_a + foot_b - far. The pair is always added back to back and
    nothing later deletes one half of it (concatenate only ever drops a side),
    so taking them two at a time recovers each box.
    """
    arms = [prim for prim in figure.exact
            if prim and prim.get("kind") == "segment"
            and prim.get("role") == "right angle"]
    corners = []
    for index in range(0, len(arms) - 1, 2):
        foot_a = np.asarray(arms[index]["a"], dtype=float)
        far = np.asarray(arms[index]["b"], dtype=float)
        foot_b = np.asarray(arms[index + 1]["b"], dtype=float)
        corners.append(foot_a + foot_b - far)
    return corners


def _is_right_angle(arms: Sequence[np.ndarray], corner, boxed: Sequence[np.ndarray],
                    tolerance: float) -> bool:
    """Whether this corner is a right angle — drawn as one, or measuring as one.

    Either way it goes unlabelled: the little square already says ninety, and a
    number written next to it says the same thing again in another notation. A
    corner that measures ninety without carrying the square is left alone too —
    a "60" written into what is visibly a right angle reads as a mistake in the
    figure, whatever the number was going to be worth.
    """
    if any(float(np.hypot(*(np.asarray(corner, dtype=float) - box))) <= tolerance
           for box in boxed):
        return True
    turn = math.atan2(abs(_cross(arms[0], arms[1])), float(arms[0] @ arms[1]))
    return abs(turn - math.pi / 2) <= RIGHT_ANGLE_TOL


def _corners_of(figure: Figure, tolerance: float):
    """The figure's corners, each with the sides that meet there.

    Read back off the drawn sides rather than remembered from when the shape was
    built, because by this point the figure may be two shapes glued together and
    turned — and the corners that matter are the ones in the drawing.
    """
    found: List[Tuple[np.ndarray, set]] = []
    for index, prim in enumerate(figure.exact):
        if not prim or prim.get("kind") != "segment" or prim.get("role") != "side":
            continue
        for end in (prim["a"], prim["b"]):
            point = np.asarray(end, dtype=float)
            for corner, owners in found:
                if float(np.hypot(*(corner - point))) <= tolerance:
                    owners.add(index)
                    break
            else:
                found.append((point, {index}))
    return found


def _angle_is_split(figure: Figure, corner, arms: Sequence[np.ndarray],
                    owners: Set[int], tolerance: float) -> bool:
    """Whether another line of the figure leaves `corner` inside the angle
    between `arms` — an altitude from it, a glued shape's side, a radius.

    The corner is then two angles, and a size written in the gap would not say
    which of the three it is the size of.
    """
    whole = math.atan2(abs(_cross(arms[0], arms[1])), float(arms[0] @ arms[1]))
    point = np.asarray(corner, dtype=float)
    for index, prim in enumerate(figure.exact):
        if not prim or index in owners or prim.get("role") in MARK_ROLES:
            continue
        if not _on_part(prim, point, tolerance):
            continue
        for ray in _along_part(prim, point, tolerance)[1]:
            ray = ray / max(float(np.hypot(*ray)), 1e-12)
            parts = [math.atan2(abs(_cross(ray, arm)), float(ray @ arm)) for arm in arms]
            if min(parts) > 1e-3 and abs(sum(parts) - whole) < 1e-3:
                return True
    return False


# A length is written as a whole number, on one scale for the whole figure.
LENGTH_LONGEST = (5, 36)        # what the longest side may be written as
LENGTH_ROUNDING = 0.03          # the most a written length may be off, relative


def _rounds_true(value: float) -> bool:
    """Whether a side `value` long on the figure's scale can be written as the
    nearest whole number and still be its length."""
    return round(value) >= 1 and abs(value - round(value)) <= LENGTH_ROUNDING * value


def _length_scale(lengths: Sequence[float], rng: random.Random) -> Optional[float]:
    """The one factor every written length is its side's true length times.

    One factor, so the numbers keep the construction's proportions: a side
    written 12 is twice one written 6. It is the longest side written as a
    whole number, picked at random but weighted by how many sides that lets be
    written within LENGTH_ROUNDING of the truth — never counting a number that
    two unequal sides would both round to. None if no side can be written.
    """
    lengths = [length for length in lengths if length > 1e-9]
    if not lengths:
        return None
    longest = max(lengths)
    scales, weights = [], []
    for top in range(LENGTH_LONGEST[0], LENGTH_LONGEST[1] + 1):
        scale = top / longest
        by_number: Dict[int, List[float]] = {}
        for length in lengths:
            if _rounds_true(length * scale):
                by_number.setdefault(round(length * scale), []).append(length)
        count = sum(len(group) for group in by_number.values()
                    if max(group) - min(group) <= EQUAL_SIDE_TOL * max(group))
        if count:
            scales.append(scale)
            weights.append(count)
    return rng.choices(scales, weights=weights)[0] if scales else None


def _unknown_letter(rng: random.Random, used: Set[str]) -> Optional[str]:
    """A letter for an unknown length or angle that the figure has not used:
    the same letter on two things says they are the same size."""
    left = [letter for letter in UNKNOWNS if letter not in used]
    return rng.choice(left) if left else None


def _agrees(stated: Dict[float, float], says: float, measure: float,
            tolerance: float) -> bool:
    """Whether a label saying `says` may go on something measuring `measure`:
    not if a label saying the same is already on something of another size."""
    known = stated.get(round(says, 6))
    return known is None or abs(known - measure) <= tolerance


# An angle written in radians is a formula, not a number — "\frac{\pi}{3}" is a
# fraction with a Greek letter in it, and it is drawn and labelled as maths
# rather than assembled out of character ink. So these come from MathWriting,
# picked by what their LaTeX says, and are the one kind of label on a figure
# that carries class `formula`.
#
# Matched: a bare multiple of pi, a pi fraction, and pi over a number written
# with a slash — 2\pi, \frac{\pi}{4}, \frac{2\pi}{3}, \pi/6. Deliberately no
# relation or equals sign: an angle is labelled with its size, and "\pi = 180"
# is a statement about pi rather than the measure of the corner it sits in.
RADIAN_RE = re.compile(
    r"^\s*(?:"
    r"\\frac\s*\{\s*(\d*)\s*\\pi\s*\}\s*\{\s*(\d+)\s*\}"  # \frac{2\pi}{3}
    r"|(\d*)\s*\\pi(?:\s*/\s*(\d+))?"                     # \pi, 2\pi, 2\pi/3
    r")\s*$")

# How often an angle that gets labelled at all is labelled in radians, when
# the pool has an expression for its size.
RADIAN_PROB = 0.35
# The most, in degrees, a written angle may be off the angle it is written in:
# a whole number of degrees is never further than this, and a pi fraction has
# to be as close.
ANGLE_ROUNDING = 0.5


def _radian_degrees(label: Optional[str]) -> Optional[float]:
    """The size in degrees that an angle written as `label` in radians states."""
    match = RADIAN_RE.match(label or "")
    if not match:
        return None
    fraction = match.group(2) is not None
    top = match.group(1) if fraction else match.group(3)
    bottom = match.group(2) if fraction else match.group(4)
    denominator = int(bottom) if bottom else 1
    if denominator == 0:
        return None
    return 180.0 * (int(top) if top else 1) / denominator


# radian_expressions' answer per pool: (the pool, its length, the expressions).
# make_geometry asks once per figure, and scanning every label with RADIAN_RE is
# nothing on the 200-expression excerpt but most of a page's time on the full
# release's 600k. Keyed by the pool's id and checked by identity and length, so a
# different or grown pool is scanned afresh; holding the pool keeps its id from
# being reused by another list.
_RADIAN_CACHE: Dict[int, Tuple[Sequence[Ink], int, List[Ink]]] = {}


def radian_expressions(math_pool: Optional[Sequence[Ink]]) -> List[Ink]:
    """The expressions in a MathWriting pool that read as an angle in radians.

    Selected by their LaTeX rather than synthesised, so the ink really is
    somebody's handwritten pi over three. A pool with none in it — the shipped
    excerpt is small enough to have none — simply yields nothing, and the
    figures fall back to writing every angle in degrees.

    Scanned once per pool (_RADIAN_CACHE); the list returned is shared, so
    callers only read from it.
    """
    if not math_pool:
        return []
    hit = _RADIAN_CACHE.get(id(math_pool))
    if hit is not None and hit[0] is math_pool and hit[1] == len(math_pool):
        return hit[2]
    labels = getattr(math_pool, "labels", None)
    if labels is not None and hasattr(math_pool, "ink"):
        # a lazy pool (mathwriting_store.MathPool): read the labels, build only the matches
        found = [math_pool.ink(i) for i, label in enumerate(labels)
                 if label and RADIAN_RE.match(label)]
    else:
        found = [ink for ink in math_pool
                 if ink.label and RADIAN_RE.match(ink.label)]
    _RADIAN_CACHE[id(math_pool)] = (math_pool, len(math_pool), found)
    return found


def annotate_figure(figure: Figure, rng: random.Random,
                    glyphs: Optional[Dict[str, List[Ink]]],
                    radians: Optional[Sequence[Ink]] = None) -> None:
    """Name the figure's corners and write lengths and angles on it, in place.

    Called last, once the figure has been glued and turned, so the text is
    upright and lands where the finished drawing actually is.

    Every length and angle written is what the construction measures, not what
    the ink happens to show. An angle is its size to the nearest degree, or a
    pi fraction within ANGLE_ROUNDING of it; lengths share one scale
    (_length_scale), so their numbers keep the figure's proportions. Two labels
    that say the same thing are on two things of the same size: a number that
    would repeat one already written on another size is not written, and no
    unknown letter is used twice.
    """
    if not glyphs:
        return
    radians = radians or []
    size = _figure_size(figure)
    tolerance = VERTEX_TOL * size
    corners = _corners_of(figure, tolerance)
    # a circle has no corners and no sides; there is nothing here to name
    if len(corners) < 3:
        return

    middle = figure.centroid()
    height = LABEL_HEIGHT * size
    ink = figure.points()
    taken: List[Tuple[float, float, float, float]] = []
    boxed_corners = _right_angle_corners(figure)
    used_letters: Set[str] = set()
    # what each written number says -> the true size it was written on
    stated_angles: Dict[float, float] = {}
    stated_lengths: Dict[float, float] = {}

    def outward(point):
        """Away from the middle of the figure, so a label sits clear of the ink."""
        away = np.asarray(point, dtype=float) - middle
        length = float(np.hypot(*away))
        return away / length if length > 1e-9 else np.array([0.0, -1.0])

    # Corners are named going round the figure rather than in the order the
    # sides happened to be drawn, so the letters read A, B, C round the shape
    # the way they are meant to.
    ordered = sorted(corners, key=lambda c: math.atan2(*(c[0] - middle)[::-1]))

    # All of them or none: naming one corner of a triangle and leaving the other
    # two blank is not something a figure does, and it is worse where the
    # unnamed corners are the symmetric partners of the named one.
    naming = rng.random() < LABEL_VERTICES_PROB and len(ordered) <= len(VERTEX_NAMES)
    start = rng.randrange(len(VERTEX_NAMES))

    for position, (corner, owners) in enumerate(ordered):
        if naming:
            name = VERTEX_NAMES[(start + position) % len(VERTEX_NAMES)]
            written = _write_text(name, glyphs, rng, height)
            reach = VERTEX_CLEARANCE * size + height / 2
            placed = written and _place_label(
                written, corner, outward(corner), size, ink, taken,
                [reach * m for m in LABEL_RETRIES])
            if placed:
                figure.notes.append(Note(text=name, kind="vertex", strokes=placed))

        # An angle needs two sides to be an angle between; and it is written
        # inside the shape, along the line that splits the corner in two.
        if len(owners) == 2 and rng.random() < LABEL_ANGLE_PROB:
            arms = []
            for side in owners:
                prim = figure.exact[side]
                ends = [np.asarray(prim["a"], float), np.asarray(prim["b"], float)]
                far = max(ends, key=lambda e: float(np.hypot(*(e - corner))))
                span = far - corner
                length = float(np.hypot(*span))
                if length > 1e-9:
                    arms.append(span / length)
            bisector = arms[0] + arms[1] if len(arms) == 2 else np.zeros(2)
            reach = float(np.hypot(*bisector))
            if (len(arms) == 2 and reach > 1e-6
                    and not _is_right_angle(arms, corner, boxed_corners, tolerance)
                    and not _angle_is_split(figure, corner, arms, owners, tolerance)):
                measure = math.atan2(abs(_cross(arms[0], arms[1])), float(arms[0] @ arms[1]))
                degrees = math.degrees(measure)
                fitting = [e for e in radians
                           if abs((_radian_degrees(e.label) or -1e9) - degrees) <= ANGLE_ROUNDING]
                # in radians it is a fraction with a pi in it, so it is real
                # maths ink labelled `formula`; in degrees it is a number,
                # written out of character ink like every other label
                text, says, written = None, None, []
                kind, origin = "word", "emnist"
                if fitting and rng.random() < RADIAN_PROB:
                    expression = rng.choice(fitting)
                    text, says = expression.label, _radian_degrees(expression.label)
                    kind, origin = "formula", (expression.source or "mathwriting")
                elif rng.random() < UNKNOWN_PROB:
                    text = _unknown_letter(rng, used_letters)
                else:
                    says = float(round(degrees))
                    text = str(int(says))
                if says is not None and not _agrees(stated_angles, says, measure,
                                                    EQUAL_ANGLE_TOL):
                    text = None
                if text and kind == "formula":
                    written = _write_formula(expression, height * 1.15)
                elif text:
                    written = _write_text(text, glyphs, rng, height * 0.85)
                step = ANGLE_CLEARANCE * size
                placed = written and _place_label(
                    written, corner, bisector / reach, size, ink, taken,
                    [step * m for m in LABEL_RETRIES])
                if placed:
                    figure.notes.append(Note(text=text, kind="angle",
                                              strokes=placed,
                                              written=kind, source=origin))
                    if says is None:
                        used_letters.add(text)
                    else:
                        stated_angles[round(says, 6)] = measure

    sides = [prim for prim in figure.exact
             if prim and prim.get("kind") == "segment" and prim.get("role") == "side"]
    lengths = [float(np.hypot(*np.subtract(prim["b"], prim["a"]))) for prim in sides]
    scale = _length_scale(lengths, rng)
    for prim, length in zip(sides, lengths):
        if rng.random() >= LABEL_EDGE_PROB:
            continue
        start_point = np.asarray(prim["a"], dtype=float)
        end_point = np.asarray(prim["b"], dtype=float)
        midpoint = (start_point + end_point) / 2.0
        text, says = None, None
        if rng.random() < UNKNOWN_PROB:
            text = _unknown_letter(rng, used_letters)
        elif scale is not None and _rounds_true(length * scale):
            says = float(round(length * scale))
            if _agrees(stated_lengths, says, length, EQUAL_SIDE_TOL * length):
                text = str(int(says))
        if not text:
            continue
        written = _write_text(text, glyphs, rng, height)
        reach = EDGE_CLEARANCE * size + height / 2
        placed = written and _place_label(
            written, midpoint, outward(midpoint), size, ink, taken,
            [reach * m for m in LABEL_RETRIES])
        if placed:
            figure.notes.append(Note(text=text, kind="edge", strokes=placed))
            if says is None:
                used_letters.add(text)
            else:
                stated_lengths[round(says, 6)] = length


# ----------------------------------------------------------------------------
# One whole diagram
# ----------------------------------------------------------------------------

def make_figure(bank: Bank, rng: random.Random, size: float = UNIT,
                gluable: bool = False) -> Tuple[Figure, Optional[List[np.ndarray]]]:
    """One shape, decorated with the operations that suit it.

    Returns the figure and, for a polygon, its corners — the constructions that
    need them (altitude, incircle, circumcircle) cannot recover them from the
    ink once it has been drawn with a wobble.
    """
    # A figure meant to be glued has to have a straight side to glue along, so
    # the round shapes are left out of the draw when one is asked for —
    # otherwise the caller keeps drawing circles that have to be thrown away.
    choices = ["circle", "polygon", "sector", "semicircle"]
    weights = [0.34, 0.52, 0.08, 0.06]
    if gluable:
        choices, weights = choices[1:], weights[1:]
    pick = rng.choices(choices, weights=weights)[0]

    if pick == "circle":
        figure = make_circle(bank, rng, size)
        corners = None
        for operation, chance in ((add_diameter, 0.35), (add_chord, 0.35),
                                  (inscribe_polygon, 0.30)):
            if rng.random() < chance:
                operation(figure, bank, rng)
    elif pick == "polygon":
        figure = make_polygon(bank, rng, size)
        # every side is its own edge and they were added in order, so the
        # corners are recovered from the edges rather than passed alongside
        corners = [e.a for e in figure.edges]
        if len(corners) == 3 and rng.random() < 0.40:
            add_altitude(figure, bank, rng, corners)
        if rng.random() < 0.22:
            inscribe_circle(figure, bank, rng, corners)
        elif rng.random() < 0.22:
            circumscribe_circle(figure, bank, rng, corners)
    elif pick == "sector":
        figure = make_sector(bank, rng, size)
        corners = None
        outline = figure.exact[0]
        # an arc goes the shorter way round, which on a sector wider than a
        # half turn is outside it, so only a narrower one is marked
        if (rng.random() < 0.30
                and abs(outline["end"] - outline["start"]) < math.pi - 1e-6):
            figure.angle_candidates.append(
                (np.zeros(2), figure.edges[0].b, figure.edges[1].b, True))
    else:
        figure = make_semicircle(bank, rng, size)
        corners = None
        if rng.random() < 0.30:
            add_chord(figure, bank, rng)
    return figure, corners


def make_diagram(bank: Bank, rng: random.Random = random,
                 size: float = UNIT,
                 glyphs: Optional[Dict[str, List[Ink]]] = None,
                 radians: Optional[Sequence[Ink]] = None) -> Ink:
    """A whole geometry figure: a few shapes, operated on and put together.

    `glyphs` is character ink keyed by the character, from
    loaders.load_emnist_pool. Given one, the figure names its corners and writes
    lengths and angles on itself; without one it is drawn unlabelled.

    `radians` is MathWriting ink for angles written in radians, from
    radian_expressions(); without it every angle is written in degrees.
    """
    if not bank.ok():
        raise ValueError("the bank has no mined line or oval ink to draw with")

    count = rng.choices((1, 2, 3), weights=(0.45, 0.40, 0.15))[0]
    figure, _ = make_figure(bank, rng, size, gluable=count > 1)

    for _ in range(count - 1):
        other, _ = make_figure(bank, rng, size * rng.uniform(0.5, 1.0),
                               gluable=True)
        figure = concatenate(figure, other, rng)

    # after gluing, so an arc count means one size across every shape in it
    mark_equal_angles(figure, bank, rng)

    # after gluing, so the sides are compared as one figure; the probability is
    # tested first so that at 0 no random number is drawn and figures come out
    # exactly as they did before ticks existed
    if TICK_SIDES_PROB and rng.random() < TICK_SIDES_PROB:
        mark_equal_sides(figure, bank, rng)

    if rng.random() < 0.75:
        figure = rotated(figure, rng)

    # last, so the writing is upright and lands on the finished drawing
    annotate_figure(figure, rng, glyphs, radians)
    return figure.to_ink()


# ----------------------------------------------------------------------------
# The beautified drawing
# ----------------------------------------------------------------------------
#
# Every figure carries the construction it was drawn from, so the clean version
# is not a smoothed trace of the ink — it is the geometry the hand was aiming
# at, drawn exactly. That is the difference that matters: smoothing a hand-drawn
# incircle leaves a tidy curve that still misses the side it is supposed to
# touch, whereas the construction has it tangent because it was computed to be.

EXACT_ARC_STEPS = 200


def exact_polyline(prim: Dict) -> np.ndarray:
    """The points that draw one primitive exactly."""
    if prim["kind"] == "segment":
        return np.array([prim["a"], prim["b"]], dtype=float)

    span = prim["end"] - prim["start"]
    steps = max(8, int(EXACT_ARC_STEPS * abs(span) / (2 * math.pi)))
    angles = np.linspace(prim["start"], prim["end"], steps)
    return (prim["radius"] * np.column_stack([np.cos(angles), np.sin(angles)])
            + np.array(prim["centre"], dtype=float))


def exact_of(ink: Ink) -> List[Dict]:
    """The construction stored on a diagram made by make_diagram."""
    return list((ink.meta or {}).get("exact", []))


def draw_exact(ax, ink: Ink):
    for prim in exact_of(ink):
        points = exact_polyline(prim)
        mark = prim.get("mark", False)
        ax.plot(points[:, 0], points[:, 1], "-", color="#555" if mark else "#111",
                lw=0.8 if mark else 1.4, solid_capstyle="round")


def to_svg(ink: Ink, size: float = 320.0, pad: float = 16.0) -> str:
    """The diagram as standalone SVG — the drawing as code, not as pixels."""
    prims = exact_of(ink)
    if not prims:
        return "<svg xmlns='http://www.w3.org/2000/svg'/>"

    every = np.vstack([exact_polyline(p) for p in prims])
    low = every.min(axis=0)
    extent = float(max(every.max(axis=0) - low).max()) or 1.0
    scale = (size - 2 * pad) / extent

    def place(points):
        moved = (points - low) * scale + pad
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in moved)

    body = []
    for prim in prims:
        mark = prim.get("mark", False)
        body.append(
            f"  <polyline points='{place(exact_polyline(prim))}' fill='none' "
            f"stroke='{'#555' if mark else '#111'}' stroke-width='{1.0 if mark else 1.7}' "
            f"stroke-linecap='round' stroke-linejoin='round'>"
            f"<title>{prim.get('role', '')}</title></polyline>")
    return (f"<svg xmlns='http://www.w3.org/2000/svg' width='{size:.0f}' "
            f"height='{size:.0f}' viewBox='0 0 {size:.0f} {size:.0f}'>\n"
            + "\n".join(body) + "\n</svg>\n")


# ----------------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------------

def draw_sheet(inks: List[Ink], path: str, columns: int = 6):
    """Each figure twice: the ink, and the construction it was drawn from.

    The two are stacked in the same column rather than put on separate sheets,
    because the pair is the point — what is worth seeing is that the clean
    drawing is the same figure, and where the hand departed from it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bands = max(1, math.ceil(len(inks) / columns))
    fig, axes = plt.subplots(bands * 2, columns,
                             figsize=(columns * 2.2, bands * 4.4), squeeze=False)
    for index in range(bands * columns):
        band, column = divmod(index, columns)
        drawn = axes[band * 2][column]
        clean = axes[band * 2 + 1][column]
        for ax in (drawn, clean):
            ax.axis("off")
        if index >= len(inks):
            continue

        ink = inks[index]
        for stroke in ink.all_strokes():
            drawn.plot(stroke.x, stroke.y, "-", color="#111", lw=1.1,
                       solid_capstyle="round")
        draw_exact(clean, ink)
        for ax in (drawn, clean):
            ax.set_aspect("equal")
            ax.invert_yaxis()

    fig.suptitle("geometry figures: mined ink above, the construction it was "
                 "drawn from below", fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    plt.savefig(path, dpi=165)
    plt.close(fig)
    print(f"wrote {path}")


def draw_bank_ovals(bank: Bank, rng: random.Random, path: str,
                    columns: int = 12, count: int = 48):
    """Redraw a sample of the bank's oval ink, as a circle and as an ellipse.

    Each is shown at the two extremes it will be asked for — perfectly round,
    and squashed to half — because a profile that looks acceptable as a circle
    can still show its flaws once stretched, and stretching is what a figure
    does to it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ovals = bank.ovals[:count]
    rows = max(1, math.ceil(len(ovals) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.2, rows * 1.4),
                             squeeze=False)
    for index in range(rows * columns):
        ax = axes[index // columns][index % columns]
        ax.axis("off")
        if index >= len(ovals):
            continue
        one = Bank(lines=bank.lines, ovals=[ovals[index]])
        for centre, rx, ry in (((0, 0), 40, 40), ((0, 105), 40, 20)):
            stroke = arc(one, rng, centre, rx, ry)
            ax.plot(stroke.x, stroke.y, "-", color="#111", lw=1.0)
        ax.set_aspect("equal")
        ax.invert_yaxis()

    fig.suptitle("oval ink the bank accepted: as a circle, and squashed to 2:1",
                 fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--demo", default="geometry_demo.png")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--cache", default="shapes_bank.json",
                    help="where the mined primitives are kept between runs")
    ap.add_argument("--per-pool", type=int, default=0,
                    help="cap items per pool when mining; 0 mines everything")
    ap.add_argument("--bank-sheet", default=None,
                    help="draw the oval ink the bank accepted, to check it")
    ap.add_argument("--svg-dir", default=None,
                    help="write each beautified figure as standalone SVG")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-labels", action="store_true",
                    help="draw the figures unlabelled, without naming corners "
                         "or writing lengths and angles on them")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    bank = build_bank(per_pool=args.per_pool or None, cache=args.cache)
    print(f"bank: {len(bank.lines)} lines, {len(bank.ovals)} ovals")

    glyphs = None
    if not args.no_labels:
        import loaders
        try:
            glyphs = loaders.load_emnist_pool(max_per_class=12)
            print(f"glyphs: {sum(len(v) for v in glyphs.values())} characters "
                  f"across {len(glyphs)} classes")
        except (FileNotFoundError, ImportError) as exc:
            print(f"glyphs: none ({exc}) — figures will be unlabelled")

    if args.bank_sheet:
        draw_bank_ovals(bank, rng, args.bank_sheet)

    inks = [make_diagram(bank, rng, glyphs=glyphs) for _ in range(args.n)]

    if args.svg_dir:
        import os
        os.makedirs(args.svg_dir, exist_ok=True)
        for index, ink in enumerate(inks):
            with open(os.path.join(args.svg_dir, f"{index:03d}_figure.svg"), "w") as f:
                f.write(to_svg(ink))
        print(f"wrote {len(inks)} SVG figures -> {args.svg_dir}")

    draw_sheet(inks, args.demo)


if __name__ == "__main__":
    main()
