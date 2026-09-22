"""
shape_mining.py — find primitive shapes in the loaded ink.

    python shape_mining.py                    # mine every pool, print a report
    python shape_mining.py --out shapes.json  # ...and save what it found

Three categories, each defined by a scale-free measurement on the ink itself
rather than by any dataset's annotation, so a hit can come from anywhere: an
equals sign in MathWriting, a QuickDraw envelope, an OHFD process box, the
cross-bar of a hand-written 't'.

    line        every point lies close to the chord joining the first and last
    rectangle   a closed stroke hugging its own minimum-area rectangle
    oval        a closed stroke hugging its own least-squares ellipse

A square is a rectangle and a circle is an oval: both come back under the
general name with the proportions in `metrics["hw_ratio"]` — the height of the
shape over its width — for a caller to split on if it wants to. The detector
does not split on it, because where a hand-drawn box stops being square is a
question about what you are going to do with the answer, not about the ink.
Note that OHFD separately annotates flowchart nodes as `oval` and `circle`;
those are labels the dataset asserts, and this `oval` is a measurement.

Every threshold is a ratio — residual over chord length, radius spread over mean
radius — because the pools are in wildly different units (QuickDraw is 0-255,
IAM around 600 per line, DeepWriting rescaled to 3000). An absolute tolerance
would silently mine one dataset and ignore another.

A shape is looked for both in each stroke alone and in the strokes of one
annotated node taken together, since a box drawn as two L-shaped strokes is a
rectangle that neither stroke is on its own.
"""

from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from dataclasses import dataclass, field
import argparse
import json
import math

import numpy as np

from core import Ink, Stroke


# ----------------------------------------------------------------------------
# Thresholds — all dimensionless
# ----------------------------------------------------------------------------

MIN_POINTS = 6           # below this "lies close to the chord" is vacuous

# Distinct positions, after dropping repeated samples. A pen resting on the paper
# emits a run of identical points, and any three distinct points lie exactly on
# some circle — one such dot, 12 samples at 4 positions spanning half a unit
# inside a 2000-unit flowchart, fitted a circle with a radial spread of exactly
# zero. An ellipse fit has 5 free parameters and a minimum-area rectangle 5, so
# below this there is not enough independent evidence for the fit to mean
# anything, however good its residual looks.
MIN_DISTINCT = 8

# A shape also has to be a real feature of the ink it came from, not a speck: its
# extent must be at least this fraction of the whole item's extent.
MIN_SPAN_RATIO = 0.02

# Samples closer together than this, relative to the candidate's own extent, are
# treated as one position.
DUPLICATE_EPS = 1e-3

# line
LINE_RESIDUAL = 0.05     # max distance off the chord, over chord length
LINE_DETOUR_MAX = 1.08   # arc length over chord: rejects an out-and-back, which
                         # has a near-zero chord and so a perfect residual

# closure — shared by rectangle and oval
CLOSE_GAP = 0.20         # end-to-start gap, over perimeter

# oval
# What separates an oval from a polygon is not how far the ink strays from the
# fitted ellipse but WHERE: a polygon puts all of its deviation at a few
# corners, and a hand-drawn oval spreads a larger total deviation smoothly. The
# radial thresholds below cannot see that difference and measuring OHFD proves
# it — its annotated ovals have a median radial spread of 0.050 against an ideal
# pentagon's 0.067, so no threshold on spread alone separates the two
# populations. Cornering is measured directly instead, and these two are left
# loose, as guards against a blob rather than as the polygon test.
OVAL_RADIAL_CV = 0.10
OVAL_RADIAL_MAX = 0.30   # worst single radius, not the spread

# Sharpest corner anywhere on the loop, in degrees. The ink is binned by angle
# about the centre and averaged inside each bin, and the turn is taken across a
# window either side, so this measures the shape's own cornering and not the
# jitter between neighbouring samples. Measured over OHFD's annotated nodes,
# every quadrilateral class sits far above this — rectangle, diamond and
# parallelogram all have a median near 95 and a 10th percentile above 80 — while
# its ovals have a median of 50. An ideal square comes to 72 and a pentagon 64.
OVAL_CORNER_TURN = 70.0

OVAL_SWEEP = 1.80        # radians/pi swept about the centre: rejects an arc
OVAL_SWEEP_MAX = 2.40    # ...and rejects a spiral, which sweeps past one turn

# Largest stretch of the circumference carrying no ink, as a fraction of a turn.
# `sweep` counts how far the pen travelled about the centre but not where it
# went: a C closed by a straight chord sweeps a full turn about a centre it
# never encircles evenly, and only the gap catches it.
OVAL_ANGULAR_GAP = 0.13

# An oval has to close on itself more convincingly than a rectangle does, whose
# four corners give the eye somewhere to end. Only meaningful for a candidate
# that is one stroke — see the note on ordering in oval_metrics.
OVAL_CLOSE_GAP = 0.12

# Bins round the loop, and how many of them either side the corner window
# spans. 48 bins puts a window at 15 degrees of arc, wide enough to average out
# pen jitter and narrow enough that a corner is not diluted by the smooth curve
# on either side of it.
OVAL_BINS = 48
OVAL_CORNER_WINDOW = 2

# Not a roundness test — elongation is what an oval is allowed to have, and is
# reported as metadata instead. This is only a degeneracy guard: past it the
# "ellipse" is a sliver whose minor axis is within the pen's own wobble, and its
# fit means nothing.
OVAL_ASPECT_MAX = 4.0

# rectangle
RECT_RESIDUAL = 0.055    # distance to the min-area rectangle, over its diagonal
RECT_FILL_MIN = 0.72     # hull area over min-area-rect area: rejects a triangle
RECT_SIDE_COVER = 0.10   # every side must carry points along this much of it


@dataclass
class ShapeHit:
    """One found shape, with the measurements that qualified it."""
    kind: str                       # line | rectangle | oval
    source: str                     # which dataset it came from
    label: Optional[str]            # the ink's own label, where it has one
    class_name: Optional[str]       # its annotated class, where it has one
    points: np.ndarray = field(repr=False)
    score: float = 0.0              # 1 is a perfect fit, 0 is at the threshold
    metrics: Dict[str, float] = field(default_factory=dict)
    # The candidate's strokes kept apart. `points` is their concatenation, which
    # is what the fitting reads, but anything that DRAWS the hit has to use these
    # or it will join the end of one stroke to the start of the next with a line
    # that was never written.
    parts: List[np.ndarray] = field(default_factory=list, repr=False)

    def bbox(self):
        return (float(self.points[:, 0].min()), float(self.points[:, 1].min()),
                float(self.points[:, 0].max()), float(self.points[:, 1].max()))

    def to_dict(self):
        return {"kind": self.kind, "source": self.source, "label": self.label,
                "class": self.class_name, "score": round(self.score, 4),
                "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
                "points": [[round(float(x), 3), round(float(y), 3)]
                           for x, y in self.points]}


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------

def _arc_length(points: np.ndarray) -> float:
    return float(np.hypot(*np.diff(points, axis=0).T).sum())


def line_metrics(points: np.ndarray) -> Dict[str, float]:
    """How well a stroke stays on the chord from its first point to its last.

    `residual` is the largest perpendicular distance from that chord, over the
    chord's length. `detour` is arc length over chord length: a stroke that runs
    out and back retraces the chord exactly, scoring a perfect residual on a
    chord of near-zero length, and only the detour catches it.
    """
    start, end = points[0], points[-1]
    chord = end - start
    chord_length = float(np.hypot(*chord))
    if chord_length < 1e-9:
        return {"residual": float("inf"), "detour": float("inf"), "chord": 0.0}

    # perpendicular distance of every point from the infinite line through both
    normal = np.array([-chord[1], chord[0]]) / chord_length
    residual = float(np.abs((points - start) @ normal).max())
    return {"residual": residual / chord_length,
            "detour": _arc_length(points) / chord_length,
            "chord": chord_length}


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain, counter-clockwise, no repeated final point."""
    order = np.lexsort((points[:, 1], points[:, 0]))
    ordered = points[order]
    if len(ordered) < 3:
        return ordered

    def half(seq):
        out: List[np.ndarray] = []
        for p in seq:
            while len(out) >= 2:
                a, b = out[-2], out[-1]
                if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) > 0:
                    break
                out.pop()
            out.append(p)
        return out[:-1]

    return np.array(half(ordered) + half(ordered[::-1]))


def _polygon_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def min_area_rect(points: np.ndarray):
    """Smallest-area enclosing rectangle, by rotating calipers on the hull.

    Returns (centre, (long_side, short_side), angle, corners). The minimum-area
    rectangle always has a side flush with a hull edge, so trying each edge in
    turn is exact rather than a search.
    """
    hull = _convex_hull(points)
    if len(hull) < 3:
        return None

    best = None
    edges = np.roll(hull, -1, axis=0) - hull
    for edge in edges:
        length = float(np.hypot(*edge))
        if length < 1e-9:
            continue
        angle = math.atan2(edge[1], edge[0])
        rotation = np.array([[math.cos(-angle), -math.sin(-angle)],
                             [math.sin(-angle), math.cos(-angle)]])
        turned = hull @ rotation.T
        lo, hi = turned.min(axis=0), turned.max(axis=0)
        area = float((hi[0] - lo[0]) * (hi[1] - lo[1]))
        if best is None or area < best[0]:
            centre_turned = (lo + hi) / 2.0
            back = np.array([[math.cos(angle), -math.sin(angle)],
                             [math.sin(angle), math.cos(angle)]])
            best = (area, centre_turned @ back.T, (hi - lo), angle)

    if best is None:
        return None
    _, centre, size, angle = best
    width, height = float(size[0]), float(size[1])
    corners = np.array([[-width / 2, -height / 2], [width / 2, -height / 2],
                        [width / 2, height / 2], [-width / 2, height / 2]])
    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])
    return centre, (width, height), angle, corners @ rotation.T + centre


def rect_metrics(points: np.ndarray) -> Optional[Dict[str, float]]:
    """How closely a closed stroke follows its own minimum-area rectangle.

    Three separate things have to hold, and each rejects a different impostor:
    `residual` that the ink sits on the outline (a filled scribble fails),
    `fill` that the outline is really a rectangle and not a triangle or a blob
    inscribed in one, and `side_cover` that all four sides were actually drawn
    (an L-shape scores well on the first two).
    """
    fitted = min_area_rect(points)
    if fitted is None:
        return None
    centre, (width, height), angle, _ = fitted
    if width < 1e-9 or height < 1e-9:
        return None

    diagonal = math.hypot(width, height)
    rotation = np.array([[math.cos(-angle), -math.sin(-angle)],
                         [math.sin(-angle), math.cos(-angle)]])
    local = (points - centre) @ rotation.T
    a, b = width / 2.0, height / 2.0

    # distance from each point to the rectangle's outline
    dx = np.abs(local[:, 0]) - a
    dy = np.abs(local[:, 1]) - b
    outside = np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0))
    inside = np.minimum(-dx, -dy)
    distance = np.where((dx > 0) | (dy > 0), outside, np.abs(inside))

    # fraction of each side's length that has ink lying along it
    near = distance <= RECT_RESIDUAL * diagonal
    covers = []
    for axis, half, other, other_half in ((0, a, 1, b), (1, b, 0, a)):
        for sign in (-1.0, 1.0):
            on_side = near & (np.abs(local[:, axis] - sign * half)
                              <= RECT_RESIDUAL * diagonal)
            if not on_side.any():
                covers.append(0.0)
                continue
            along = local[on_side, other]
            covers.append(float(along.max() - along.min()) / (2.0 * other_half))

    long_side, short_side = max(width, height), min(width, height)
    span_x = float(np.ptp(points[:, 0]))
    span_y = float(np.ptp(points[:, 1]))
    return {"residual": float(distance.max()) / diagonal,
            "fill": _polygon_area(_convex_hull(points)) / (width * height),
            "side_cover": min(covers),
            "aspect": long_side / max(short_side, 1e-9),
            "hw_ratio": span_y / max(span_x, 1e-9),
            "width": width, "height": height}


def fit_ellipse(points: np.ndarray):
    """Least-squares ellipse through the points, as (centre, (a, b), angle).

    `a` and `b` are the semi-axes, `a` the longer, and `angle` is where `a`
    points. The fit is algebraic: the general conic Ax^2+Bxy+Cy^2+Dx+Ey+F=0 is
    linear in its six coefficients, so the best one is the smallest singular
    vector of the design matrix — no iteration, no starting guess. Coordinates
    are centred and scaled first, because x^2 terms on raw pen coordinates in
    the thousands square to millions and the fit loses its low-order terms to
    rounding.

    Returns None when the conic that comes back is a hyperbola or parabola,
    which is what an open or nearly straight stroke fits.
    """
    centroid = points.mean(axis=0)
    scale = float(np.abs(points - centroid).max())
    if scale < 1e-9:
        return None
    u, v = ((points - centroid) / scale).T

    design = np.column_stack([u * u, u * v, v * v, u, v, np.ones(len(u))])
    try:
        _, _, vt = np.linalg.svd(design, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    A, B, C, D, E, F = vt[-1]

    # B^2-4AC is the conic's discriminant; only a negative one is an ellipse
    discriminant = B * B - 4 * A * C
    if discriminant >= -1e-12:
        return None

    cx = (2 * C * D - B * E) / discriminant
    cy = (2 * A * E - B * D) / discriminant

    # axes from the eigenvalues of the quadratic part, evaluated at the centre
    constant = A * cx * cx + B * cx * cy + C * cy * cy - F
    if abs(constant) < 1e-15:
        return None
    quadratic = np.array([[A, B / 2.0], [B / 2.0, C]]) / constant
    eigenvalues, eigenvectors = np.linalg.eigh(quadratic)
    if np.any(eigenvalues <= 0):
        return None
    semi_axes = 1.0 / np.sqrt(eigenvalues)      # eigh sorts ascending, so this
    order = np.argsort(-semi_axes)              # comes back descending already
    semi_axes = semi_axes[order]
    major = eigenvectors[:, order[0]]

    centre = np.array([cx, cy]) * scale + centroid
    return (centre, (float(semi_axes[0] * scale), float(semi_axes[1] * scale)),
            float(math.atan2(major[1], major[0])))


def _corner_turn(normalised: np.ndarray, angles: np.ndarray) -> Optional[float]:
    """Sharpest corner on the loop, in degrees, or None if too little of it.

    `normalised` is the ink in the frame where the fitted ellipse is the unit
    circle, so a round shape turns at the same rate everywhere however elongated
    the original was, and one threshold covers every aspect ratio. The affine
    squash that gets there maps straight lines to straight lines, so a polygon's
    corners survive it as corners.

    The loop is rebuilt by binning the ink by angle about the centre and
    averaging inside each bin. That does three things at once: it puts the
    points in loop order however many strokes they came from and whichever way
    round each was drawn, it averages away the jitter between neighbouring
    samples, which is what a raw turning angle mostly measures, and it makes the
    result independent of how densely the pen happened to sample each part.
    """
    bins = np.minimum((angles / (2 * math.pi) * OVAL_BINS).astype(int), OVAL_BINS - 1)
    loop = np.full((OVAL_BINS, 2), np.nan)
    for b in np.unique(bins):
        loop[b] = normalised[bins == b].mean(axis=0)

    filled = ~np.isnan(loop[:, 0])
    if filled.sum() < max(8, OVAL_BINS // 3):
        return None                 # too sparse for the averaging to mean much
    if not filled.all():
        # bridge the empty bins around the loop; a candidate with a gap wide
        # enough for this to invent a shape fails the angular-gap test anyway
        position = np.flatnonzero(filled)
        extended = np.concatenate([position, position[:1] + OVAL_BINS])
        empty = np.flatnonzero(~filled)
        for axis in (0, 1):
            values = np.concatenate([loop[position, axis], loop[position[:1], axis]])
            loop[empty, axis] = np.interp(empty, extended, values, period=OVAL_BINS)

    behind = loop - np.roll(loop, OVAL_CORNER_WINDOW, axis=0)
    ahead = np.roll(loop, -OVAL_CORNER_WINDOW, axis=0) - loop
    cross = behind[:, 0] * ahead[:, 1] - behind[:, 1] * ahead[:, 0]
    turn = np.abs(np.arctan2(cross, (behind * ahead).sum(axis=1)))
    return float(np.degrees(turn.max()))


def oval_metrics(points: np.ndarray) -> Optional[Dict[str, float]]:
    """Fit an ellipse and report how closely the ink tracks it.

    Everything is measured in the ellipse's own frame, where the fitted curve is
    the unit circle: a point's normalised radius is sqrt((u/a)^2+(v/b)^2), which
    is 1 exactly on the outline. That makes `radial_cv` mean the same thing for a
    round loop and a long thin one, and leaves elongation to be reported rather
    than judged.

    The fit is algebraic rather than centroid-based on purpose: an unevenly
    sampled stroke — and pen data always is, since the hand slows into curves —
    drags its centroid toward wherever the points bunch.

    `sweep` is the one measurement here that reads the points in the order they
    arrive, and so the only one that means anything solely for a candidate that
    is a single stroke. A shape drawn as four arcs reaches this function as
    those four concatenated in whatever order the file lists them and whichever
    way round each was drawn, and the angle then doubles back at every join: a
    hand-drawn pill made of four strokes measures a sweep near 0.5 of a turn
    when it plainly goes all the way round. `angular_gap` is the order-free
    reading of the same thing, and classify() leans on that instead when the
    candidate has more than one stroke.
    """
    fitted = fit_ellipse(points)
    if fitted is None:
        return None
    centre, (a, b), angle = fitted
    if b < 1e-9:
        return None

    rotation = np.array([[math.cos(-angle), -math.sin(-angle)],
                         [math.sin(-angle), math.cos(-angle)]])
    local = (points - centre) @ rotation.T
    normalised = np.hypot(local[:, 0] / a, local[:, 1] / b)
    mean_radius = float(normalised.mean())
    if mean_radius < 1e-9:
        return None

    # angle swept about the centre, to tell a full loop from an arc. Taken in
    # the ellipse's frame after undoing the elongation, so an oval's angles
    # advance evenly the way a circle's do instead of racing past the ends.
    raw_angles = np.arctan2(local[:, 1] / b, local[:, 0] / a)
    unwrapped = np.unwrap(raw_angles)
    sweep = float(abs(unwrapped[-1] - unwrapped[0]))

    # widest stretch of the circumference with no ink on it: sort the directions
    # round the centre and take the largest gap between neighbours, including
    # the one that wraps past pi back to -pi
    turned = raw_angles % (2 * math.pi)
    sorted_angles = np.sort(turned)
    gaps = np.diff(sorted_angles, append=sorted_angles[0] + 2 * math.pi)

    corner = _corner_turn(np.column_stack([local[:, 0] / a, local[:, 1] / b]), turned)

    span_x = float(np.ptp(points[:, 0]))
    span_y = float(np.ptp(points[:, 1]))
    return {"radial_cv": float(normalised.std()) / mean_radius,
            "radial_max": float(np.abs(normalised - mean_radius).max()) / mean_radius,
            "sweep": sweep / math.pi,
            "angular_gap": float(gaps.max()) / (2 * math.pi),
            "corner": float("inf") if corner is None else corner,
            "aspect": a / max(b, 1e-9),
            "hw_ratio": span_y / max(span_x, 1e-9),
            "semi_major": a, "semi_minor": b, "angle": angle}


def closure(points: np.ndarray) -> float:
    """End-to-start gap as a fraction of the path's own length."""
    perimeter = _arc_length(points)
    if perimeter < 1e-9:
        return float("inf")
    return float(np.hypot(*(points[-1] - points[0]))) / perimeter


# ----------------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------------

def _drop_repeats(points: np.ndarray, span: float) -> np.ndarray:
    """Collapse consecutive samples that sit on the same spot.

    Kept in order rather than uniqued, because the sequence is what the closure
    and sweep tests read.
    """
    if len(points) < 2:
        return points
    step = np.hypot(*np.diff(points, axis=0).T)
    keep = np.concatenate([[True], step > DUPLICATE_EPS * max(span, 1e-9)])
    return points[keep]


def classify(points: np.ndarray,
             min_span: float = 0.0,
             contiguous: bool = True) -> Optional[Tuple[str, float, Dict[str, float]]]:
    """Name the shape a point sequence makes, or None.

    `min_span` rejects a candidate smaller than that extent, so a caller can pass
    a fraction of the whole item's size and keep specks out.

    `contiguous` says whether the points are one stroke, drawn in the order they
    appear. Pass False for several strokes concatenated: the tests that read the
    order — how far the path swept, and whether it ended where it began — are
    then measuring the joins rather than the shape, and are skipped.

    Returns (kind, score, metrics), where score is how far inside the threshold
    the worst-performing measurement sits: 1.0 is exact, 0.0 is right on the
    limit. It exists so the demo can show the best examples rather than an
    arbitrary handful of barely-passing ones.
    """
    if len(points) < MIN_POINTS:
        return None
    span = float(np.hypot(np.ptp(points[:, 0]), np.ptp(points[:, 1])))
    if span < 1e-9 or span < min_span:
        return None

    points = _drop_repeats(points, span)
    if len(points) < MIN_POINTS or len(np.unique(points, axis=0)) < MIN_DISTINCT:
        return None

    # A line is judged on the chord, so it need not be closed; test it first
    # because a closed shape can never satisfy it (its chord is ~zero).
    line = line_metrics(points)
    if (line["residual"] <= LINE_RESIDUAL
            and line["detour"] <= LINE_DETOUR_MAX):
        score = min(1.0 - line["residual"] / LINE_RESIDUAL,
                    1.0 - (line["detour"] - 1.0) / (LINE_DETOUR_MAX - 1.0))
        return "line", max(0.0, score), line

    gap = closure(points)
    if contiguous and gap > CLOSE_GAP:
        return None                 # everything below is a closed shape

    oval = oval_metrics(points)
    if (oval is not None
            and oval["radial_cv"] <= OVAL_RADIAL_CV
            and oval["radial_max"] <= OVAL_RADIAL_MAX
            and oval["angular_gap"] <= OVAL_ANGULAR_GAP
            and oval["aspect"] <= OVAL_ASPECT_MAX
            and oval["corner"] <= OVAL_CORNER_TURN
            # a single stroke has to arrive back where it started and go round
            # exactly once; across several strokes neither reading survives the
            # concatenation, and the angular gap above is what stands in
            and (not contiguous
                 or (gap <= OVAL_CLOSE_GAP
                     and OVAL_SWEEP <= oval["sweep"] <= OVAL_SWEEP_MAX))):
        # every test that can be scraped past, on the same 1-is-exact scale, so
        # the score reports whichever one the shape is closest to failing
        margins = [1.0 - oval["radial_cv"] / OVAL_RADIAL_CV,
                   1.0 - oval["radial_max"] / OVAL_RADIAL_MAX,
                   1.0 - oval["angular_gap"] / OVAL_ANGULAR_GAP,
                   1.0 - oval["corner"] / OVAL_CORNER_TURN]
        if contiguous:
            margins.append(1.0 - gap / OVAL_CLOSE_GAP)
        return "oval", max(0.0, min(margins)), oval

    rect = rect_metrics(points)
    if (rect is not None
            and rect["residual"] <= RECT_RESIDUAL
            and rect["fill"] >= RECT_FILL_MIN
            and rect["side_cover"] >= RECT_SIDE_COVER):
        score = min(1.0 - rect["residual"] / RECT_RESIDUAL,
                    (rect["fill"] - RECT_FILL_MIN) / (1.0 - RECT_FILL_MIN))
        return "rectangle", max(0.0, min(1.0, score)), rect

    return None


# ----------------------------------------------------------------------------
# Mining
# ----------------------------------------------------------------------------

def _points(strokes: Sequence[Stroke]) -> np.ndarray:
    return np.array([(x, y) for s in strokes for x, y in zip(s.x, s.y)],
                    dtype=float)


def _nodes(ink: Ink) -> Iterable[Ink]:
    yield ink
    for child in ink.children:
        yield from _nodes(child)


def mine_ink(ink: Ink, source: Optional[str] = None) -> List[ShapeHit]:
    """Every shape in one Ink, from single strokes and from whole nodes.

    A node's strokes are tried together only when there are a handful of them: a
    box drawn as two L-shaped strokes is a rectangle that neither stroke is on
    its own, whereas concatenating a whole word's strokes just makes a scribble
    that no threshold should accept anyway.
    """
    whole = ink.all_strokes()
    if not whole:
        return []
    x0, y0, x1, y1 = ink.bbox()
    min_span = math.hypot(x1 - x0, y1 - y0) * MIN_SPAN_RATIO

    hits: List[ShapeHit] = []
    for node in _nodes(ink):
        candidates: List[List[np.ndarray]] = [[_points([s])] for s in node.strokes]
        if 2 <= len(node.strokes) <= 4:
            candidates.append([_points([s]) for s in node.strokes])

        for parts in candidates:
            points = parts[0] if len(parts) == 1 else np.vstack(parts)
            if len(points) < MIN_POINTS:
                continue
            found = classify(points, min_span=min_span,
                             contiguous=len(parts) == 1)
            if found is None:
                continue
            kind, score, metrics = found
            hits.append(ShapeHit(kind=kind, source=source or node.source or "?",
                                  label=node.label, class_name=node.class_name,
                                  points=points, score=score, metrics=metrics,
                                  parts=parts))
    return hits


def mine(pools: Dict[str, List[Ink]], per_pool: Optional[int] = None) -> List[ShapeHit]:
    """Mine every Ink in every pool. `per_pool` caps items for a quick run.

    Only list pools are mined. make_pdf.build_pools also hands back the EMNIST
    glyphs keyed by character, whose strokes are traced off scanned rasters
    rather than recorded from a pen, so they are not ink to borrow a hand from.
    """
    hits: List[ShapeHit] = []
    for name, pool in pools.items():
        if not isinstance(pool, list):
            continue
        items = pool[:per_pool] if per_pool else pool
        found = 0
        for ink in items:
            new = mine_ink(ink)
            hits.extend(new)
            found += len(new)
        print(f"  {name:14s} {len(items):5d} items -> {found:6d} shapes")
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default=None, help="write the hits as JSON")
    ap.add_argument("--per-pool", type=int, default=None,
                    help="cap items per pool for a quick run")
    ap.add_argument("--top", type=int, default=5,
                    help="how many best-scoring examples to list per kind")
    args = ap.parse_args()

    import make_pdf
    from collections import Counter

    pools = make_pdf.build_pools()
    hits = mine(pools, per_pool=args.per_pool)

    print(f"\n{len(hits)} shapes found")
    by_kind = Counter(h.kind for h in hits)
    for kind, count in by_kind.most_common():
        sources = Counter(h.source for h in hits if h.kind == kind)
        print(f"\n{kind:10s} {count:6d}   by source: "
              + ", ".join(f"{s}={n}" for s, n in sources.most_common()))
        best = sorted((h for h in hits if h.kind == kind),
                      key=lambda h: -h.score)[:args.top]
        for hit in best:
            detail = " ".join(f"{k}={v:.3f}" for k, v in hit.metrics.items()
                              if k in ("residual", "detour", "radial_cv",
                                        "radial_max", "angular_gap", "fill",
                                        "side_cover", "hw_ratio"))
            print(f"    {hit.score:.3f}  {hit.source:12s} "
                  f"{str(hit.class_name or '-'):10s} {str(hit.label or '')[:24]:24s} {detail}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump([h.to_dict() for h in hits], f)
        print(f"\nWrote {len(hits)} shapes -> {args.out}")


if __name__ == "__main__":
    main()
