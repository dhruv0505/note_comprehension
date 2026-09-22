"""
block_generators.py — build one Ink tree per page-object (textblock, table,
flowchart, geometry, drawing), each properly labeled at the right taxonomy level:

    level 2: textblock, table, drawing, geometry, rectangle, oval, diamond, data
    level 1: textline, formula
    level 0: word, arrow, vertex

Each generator lays its content out in a *local* coordinate frame of arbitrary
size (whatever's natural for the content it pulled). The page layout engine
later rescales the whole returned Ink uniformly into its assigned page region
via core.fit_ink_to_box, so you don't need to worry about absolute page
coordinates here — just relative arrangement and readable proportions.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import itertools
import math
import random

from core import Stroke, Ink, transform_ink


# ----------------------------------------------------------------------------
# Procedural stroke primitives (hand-drawn-looking shapes with jitter)
# ----------------------------------------------------------------------------

def _jitter_pts(pts, noise=0.6):
    return [(x + random.uniform(-noise, noise), y + random.uniform(-noise, noise)) for x, y in pts]


def draw_rect_stroke(x0, y0, x1, y1, n_per_side=6) -> Stroke:
    pts = []
    for i in range(n_per_side): pts.append((x0 + (x1 - x0) * i / n_per_side, y0))
    for i in range(n_per_side): pts.append((x1, y0 + (y1 - y0) * i / n_per_side))
    for i in range(n_per_side): pts.append((x1 - (x1 - x0) * i / n_per_side, y1))
    for i in range(n_per_side): pts.append((x0, y1 - (y1 - y0) * i / n_per_side))
    pts.append((x0, y0))
    pts = _jitter_pts(pts)
    return Stroke(x=[p[0] for p in pts], y=[p[1] for p in pts])


def draw_oval_stroke(x0, y0, x1, y1, n_points=24) -> Stroke:
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rx, ry = (x1 - x0) / 2, (y1 - y0) / 2
    pts = [(cx + rx * math.cos(a), cy + ry * math.sin(a))
           for a in [2 * math.pi * i / n_points for i in range(n_points + 1)]]
    pts = _jitter_pts(pts, noise=0.4)
    return Stroke(x=[p[0] for p in pts], y=[p[1] for p in pts])


def draw_rounded_rect_stroke(x0, y0, x1, y1, rng: random.Random = random) -> Stroke:
    """A hand-drawn box: rounded corners and edges that bow rather than rule.

    A box drawn round something by hand is never square. The corners are cut
    with a quarter arc and each edge carries a slow bow plus fine tremor, so the
    result reads as pen-drawn instead of like a plotted rectangle. The path also
    overshoots slightly where it closes, the way a real one does.
    """
    w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    radius = min(w, h) * rng.uniform(0.10, 0.26)
    bow = min(w, h) * rng.uniform(0.010, 0.035)      # how much each edge bellies out
    tremor = min(w, h) * 0.006

    # centre of each corner arc, and the angle range it sweeps
    corners = [((x1 - radius, y0 + radius), -math.pi / 2, 0.0),
               ((x1 - radius, y1 - radius), 0.0, math.pi / 2),
               ((x0 + radius, y1 - radius), math.pi / 2, math.pi),
               ((x0 + radius, y0 + radius), math.pi, 3 * math.pi / 2)]

    pts = []
    for index, ((cx, cy), a0, a1) in enumerate(corners):
        # the straight run leading into this corner, bowed outward
        if pts:
            start = pts[-1]
            end = (cx + radius * math.cos(a0), cy + radius * math.sin(a0))
            for s in range(1, 7):
                t = s / 7.0
                px = start[0] + (end[0] - start[0]) * t
                py = start[1] + (end[1] - start[1]) * t
                swell = math.sin(t * math.pi) * bow * rng.choice((-1.0, 1.0)) * 0.5
                nx, ny = -(end[1] - start[1]), (end[0] - start[0])
                length = math.hypot(nx, ny) or 1.0
                pts.append((px + nx / length * swell, py + ny / length * swell))
        for s in range(5):
            a = a0 + (a1 - a0) * s / 4.0
            pts.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))

    # close the loop, overshooting a little past the start
    if pts:
        first = pts[0]
        pts.append((first[0] + rng.uniform(-1.0, 2.5), first[1] + rng.uniform(-1.5, 1.5)))

    pts = _jitter_pts(pts, noise=tremor)
    return Stroke(x=[p[0] for p in pts], y=[p[1] for p in pts])


# Raised from 0.08 and 0.01 (about 0.8 line marks a page) for more line marks to learn
# from, and because pages now carry fewer textblocks to put them on.
UNDERLINE_PROB = 0.20           # chance a text line carries an underline
STRIKETHROUGH_PROB = 0.04       # chance a text line is struck through instead
UNDERLINE_DROP = 0.40       # how far below the writing's foot it sits, in x-heights
UNDERLINE_WOBBLE = 0.2      # share of a mined line's sideways wander a mark keeps
UNDERLINE_MAX_SLOPE = 0.10  # the row's fitted slope is held to this (rise over run)
UNDERLINE_DOUBLE_PROB = 0.08    # of underlines, how many are drawn twice
UNDERLINE_DOUBLE_GAP = 0.18     # between a double underline's strokes, in x-heights

# A mark has to run under (or through) a real stretch of writing, counted as the
# row's strokes it passes: IAM's word boundaries are geometric, so a "word" is
# often a couple of letters or a full stop, and a mark under one of those reads
# as a stray tick under a single character.
UNDERLINE_MIN_STROKES = 10
UNDERLINE_MIN_WIDTH = 4.0   # ...and it must be at least this wide, in x-heights

# How far a mark may reach into a neighbouring row's writing, from either edge,
# as a share of that row's band. Clipping the top of the next line is what a
# hand does now and then; reaching its middle strikes it through.
MARK_NEIGHBOUR_REACH = 0.25


def _writing_band(node: Ink) -> Tuple[float, float]:
    """The vertical extent of a row's (or a word's) writing that a mark is placed
    against: prose's x-height band, or a formula's whole box, since its sub- and
    superscripts and fraction bars sit well outside any band."""
    band = None if node.class_name == "formula" else _core_band(node)
    if band is None:
        _, y0, _, y1 = node.bbox()
        return y0, y1
    return band


def _mark_parts(row: Ink) -> List[Ink]:
    """What a mark is measured against along a row: its words left to right, or
    a formula as one piece."""
    if row.class_name == "formula":
        return [row]
    return sorted((c for c in row.children if c.class_name == "word" and c.all_strokes()),
                  key=lambda w: w.bbox()[0])


def _mark_fits(strokes: List[Stroke], kind: str, chosen: List[Ink],
               neighbours: List[Ink]) -> bool:
    """Whether a drawn mark keeps to its own writing and off its neighbours'.

    An underline may not rise into the writing it is under; a strikethrough may
    not wander off the writing it strikes. Neither may come within the middle of
    the row above or below (MARK_NEIGHBOUR_REACH), which would strike that row.
    """
    points = [(x, y) for s in strokes for x, y in zip(s.x, s.y)]
    for part in chosen:
        px0, _, px1, _ = part.bbox()
        top, bottom = _writing_band(part)
        height = bottom - top
        for x, y in points:
            if not px0 <= x <= px1:
                continue
            if kind == "underline" and y < bottom - 0.1 * height:
                return False
            if kind == "strikethrough" and not top - 0.2 * height <= y <= bottom + 0.2 * height:
                return False
    for row in neighbours:
        for part in _mark_parts(row):
            px0, _, px1, _ = part.bbox()
            top, bottom = _writing_band(part)
            reach = MARK_NEIGHBOUR_REACH * (bottom - top)
            if any(px0 <= x <= px1 and top + reach < y < bottom - reach for x, y in points):
                return False
    return True


def make_line_mark(row: Ink, kind: str, shape_bank, neighbours: List[Ink] = (),
                   rng: random.Random = random) -> Optional[Ink]:
    """Underline or strike through a stretch of `row` (a textline or a formula),
    or None if it can't.

    The mark is a `line_mark` labelled `kind` ("underline" or
    "strikethrough") — one class for both, the label saying which — and an
    object on its own: it is not tied to the writing it runs under or through,
    and the row is not touched.
    UNDERLINE_DOUBLE_PROB of underlines are drawn as two strokes, one just under
    the other, and still labelled "underline": a double underline is emphasis
    the same way a single one is (meta["double"] records which was drawn).

    The stroke is a mined line (geometry_diagrams.segment), the way a table's
    rules and a geometry figure's sides are, so it carries a real hand's wobble.
    No bank to draw from, no mark.

    The stretch is grown from a random word until it passes under at least
    UNDERLINE_MIN_STROKES of the row's strokes and UNDERLINE_MIN_WIDTH of width,
    and abandoned if the row runs out first.

    The mark follows the row's own drift — handwriting rises or falls across a
    line, and a level underline under a sloping one cut into its words at one end
    — but only up to UNDERLINE_MAX_SLOPE: fitted freely (it was capped at 0.25),
    a slope over a long line put the underline diagonally through the writing.
    An underline is then set UNDERLINE_DROP below the lowest foot of the words
    it spans, so it cannot rise into any of them, and a strikethrough through
    their middle. What is drawn is checked (_mark_fits) against its own writing
    and against the rows above and below (`neighbours`); a mark that fails is
    redrawn level, then single, then closer under its row, and given up if none
    of those fit.
    """
    import geometry_diagrams as gd
    if shape_bank is None or not shape_bank.ok():
        return None
    parts = _mark_parts(row)
    if not parts:
        return None
    row_strokes = [s for s in row.all_strokes() if s.x]

    def substantial(run: List[Ink]) -> bool:
        x0 = min(p.bbox()[0] for p in run)
        x1 = max(p.bbox()[2] for p in run)
        crossed = sum(1 for s in row_strokes if min(s.x) < x1 and max(s.x) > x0)
        return crossed >= UNDERLINE_MIN_STROKES and x1 - x0 >= UNDERLINE_MIN_WIDTH * LINE_HEIGHT

    start = rng.randrange(len(parts))
    end = start + rng.randint(1, len(parts) - start)
    # grow to the right, then to the left, until it is substantial enough
    while not substantial(parts[start:end]) and (end < len(parts) or start > 0):
        if end < len(parts):
            end += 1
        else:
            start -= 1
    chosen = parts[start:end]
    if not substantial(chosen):
        return None

    under = kind == "underline"

    def guide(part: Ink) -> float:
        top, bottom = _writing_band(part)
        return bottom if under else (top + bottom) / 2.0

    # the row's drift, fitted through every word's guide
    feet = [((p.bbox()[0] + p.bbox()[2]) / 2.0, guide(p)) for p in parts]
    fitted = 0.0
    if len(feet) >= 2:
        fx = sum(x for x, _ in feet) / len(feet)
        fy = sum(y for _, y in feet) / len(feet)
        spread = sum((x - fx) ** 2 for x, _ in feet)
        if spread > 1e-9:
            fitted = sum((x - fx) * (y - fy) for x, y in feet) / spread
            fitted = max(-UNDERLINE_MAX_SLOPE, min(UNDERLINE_MAX_SLOPE, fitted))

    # both ends of every chosen word, which the mark is placed against
    guides = [(x, guide(p)) for p in chosen for x in (p.bbox()[0], p.bbox()[2])]
    mean_x = sum(x for x, _ in guides) / len(guides)
    x0 = min(p.bbox()[0] for p in chosen)
    x1 = max(p.bbox()[2] for p in chosen)
    # a hand-drawn mark usually runs a touch past the words it is on
    sx, ex = x0 - rng.uniform(0.0, 3.0), x1 + rng.uniform(0.0, 5.0)
    drop = LINE_HEIGHT * UNDERLINE_DROP * rng.uniform(0.85, 1.3) if under else 0.0
    double = under and rng.random() < UNDERLINE_DOUBLE_PROB
    gap = LINE_HEIGHT * UNDERLINE_DOUBLE_GAP * rng.uniform(0.85, 1.15)

    def draw(slope: float, drop: float, double: bool) -> List[Stroke]:
        offsets = [y - slope * (x - mean_x) for x, y in guides]
        # an underline clears the lowest foot it passes; a strikethrough runs
        # through the middle of the words
        base = max(offsets) if under else sum(offsets) / len(offsets)

        def at(x: float) -> float:
            return base + slope * (x - mean_x) + drop

        # Nearly straight: at a mined line's full wander an underline bowed up
        # into the words it was under. A fifth of it keeps the hand without the bow.
        strokes = [gd.segment(shape_bank, rng, (sx, at(sx)), (ex, at(ex)), UNDERLINE_WOBBLE)]
        if double:
            # the second pass is its own stroke, so its ends land a little off the first's
            strokes.append(gd.segment(shape_bank, rng,
                                      (sx + rng.uniform(-2.0, 2.0), at(sx) + gap),
                                      (ex + rng.uniform(-3.0, 1.0), at(ex) + gap),
                                      UNDERLINE_WOBBLE))
        return strokes

    attempts = [(fitted, drop, double), (0.0, drop, double)]
    if double:
        attempts += [(fitted, drop, False), (0.0, drop, False)]
    if under:
        attempts.append((0.0, drop * 0.6, False))
    for slope, attempt_drop, attempt_double in attempts:
        strokes = draw(slope, attempt_drop, attempt_double)
        if _mark_fits(strokes, kind, chosen, neighbours):
            return Ink(strokes=strokes, label=kind, class_name="line_mark", source=kind,
                       meta={"double": attempt_double} if under else {})
    return None


def _mark_rows(rows: List[Ink], shape_bank, rng: random.Random = random) -> List[Ink]:
    """Underlines and strikethroughs for a block's stacked rows.

    A text line is struck through (STRIKETHROUGH_PROB) or underlined
    (UNDERLINE_PROB). A formula is never marked: a line under maths is too
    easily read as a fraction bar. It still counts as a neighbour, though, so a
    text line's mark stays out of it. Each row's mark is checked against the rows
    directly above and below it, which is why this runs once a block's rows are
    in place rather than as each row is written.
    """
    marks: List[Ink] = []
    ordered = sorted((r for r in rows if r.class_name in ("textline", "formula")),
                     key=lambda r: r.bbox()[1])
    for i, row in enumerate(ordered):
        if row.class_name != "textline":
            continue
        roll = rng.random()
        kind = ("strikethrough" if roll < STRIKETHROUGH_PROB else
                "underline" if roll < STRIKETHROUGH_PROB + UNDERLINE_PROB else None)
        if kind is None:
            continue
        neighbours = ordered[max(i - 1, 0):i] + ordered[i + 1:i + 2]
        mark = make_line_mark(row, kind, shape_bank, neighbours, rng)
        if mark is not None:
            # the row it marks, counted down the block, for level-1 target order
            mark.meta = dict(mark.meta or {}, attends_row=i)
            marks.append(mark)
    return marks


def draw_arrow_stroke(x0, y0, x1, y1) -> Stroke:
    ang = math.atan2(y1 - y0, x1 - x0)
    head = 6.0
    left = (x1 - head * math.cos(ang - math.pi / 6), y1 - head * math.sin(ang - math.pi / 6))
    right = (x1 - head * math.cos(ang + math.pi / 6), y1 - head * math.sin(ang + math.pi / 6))
    pts = [(x0, y0), (x1, y1), left, (x1, y1), right]
    return Stroke(x=[p[0] for p in pts], y=[p[1] for p in pts])


# ----------------------------------------------------------------------------
# Connector arrows — hand-drawn arrows linking one text line to another
# ----------------------------------------------------------------------------
# Only the two endpoints are fixed. Between them the route is free, and it is
# sampled rather than constructed: a detour magnitude is drawn from a
# distribution with most of its mass near zero and a long tail, so a connector
# is usually fairly direct but will occasionally swing right around something.
# A route that crosses any text is thrown away and re-sampled, which is what
# makes arrows appear to go around paragraphs rather than through them.

# Kept small: at 0.15 / 0.22, with more waypoints and more weaving, arrows curved
# far more than they needed to get round anything, and 0.11 / 0.15 still did.
ARROW_DETOUR_SIGMA = 0.08   # detour size as a fraction of the span, before the tail
ARROW_DETOUR_TAIL = 0.10    # chance of a much wider swing
ARROW_WEAVE_PROB = 0.10     # chance a detour switches side between waypoints
# Hard cap on how far off the straight line a route may stray, as a multiple of
# the span. This is not just taste: make_connector_arrow prefilters obstacles to
# a corridor around the endpoints, and a route that could wander outside that
# corridor might cross a box that was filtered out and never tested.
ARROW_MAX_DETOUR = 0.9
ARROW_WAYPOINT_WEIGHTS = {1: 0.65, 2: 0.28, 3: 0.07}
ARROW_ROUTE_ATTEMPTS = 140  # random routes tried before giving up on a pair
ARROW_CURVE_STEPS = 9       # points sampled per spline segment
ARROW_HEAD_FRAC = 0.09      # head length as a fraction of the span
ARROW_CLEARANCE = 2.0       # how far a route must stay off any text box

# The ink of an arrow is mined, not invented. The shaft is several real
# hand-drawn lines from the shape bank laid end to end (_chain_ink): each starts
# where the last one stopped, heading the way that one was travelling at its end,
# so a long line is made of strokes a hand actually drew and its curvature is the
# joints between them. The route only steers. A head is short, and each of its
# strokes is one mined line laid along it (_trace). A mined line starts and ends
# on its own chord, so the arrow leaves its anchor and meets its target exactly.
TRACE_STEP = 3.0            # page units between samples of a traced path
# Pieces of a shaft (_chain_ink). A mined line wavers by up to ~5% of its own
# length, so a piece this long wavers by a unit or two — a hand's waver, where
# one mined line stretched over a whole connector swung by several.
ARROW_PIECE_LENGTH = (18.0, 40.0)       # page units per mined line in a shaft
ARROW_PIECE_MIN = 6.0                   # shortest a piece gets where the route turns harder
# Sharpest turn from one piece's end into the next. 14 degrees drew a visible
# corner at every join; a pen carrying on does not make one.
ARROW_MAX_JOINT = math.radians(6.0)
# Share of a mined line cut off each end before it is used as a piece: where the
# pen set down and lifted off it hooks, and those hooks made every join a spike.
ARROW_PIECE_TRIM = 0.08
ARROW_MAX_PIECES = 160
# The last stretch of a shaft — one mined line curving from the chain's heading
# into the arrival direction — starts at least this far out. Started within a
# piece and a half, the turn was crammed into the last few units and curled
# into the head (arrival 41 degrees off square at p90).
ARROW_FINAL_STRETCH = 60.0
# A chain that has drawn more than this many times its route's length has lost
# the route and is circling, so the attempt is abandoned (_chain_ink). Well
# clear of honest shafts, which reached 1.73 on a curvy route; the one that ran
# away drew 5.27. At 1.35 it threw out a third of good arrows as well.
ARROW_CHAIN_SLACK = 1.8
# easing passes over the route before it is traced; four only rounded a corner
# a few units across and left the hook where the lead-in met the route
ARROW_SMOOTH_PASSES = 10
ARROW_SQUARE_RUN = 12.0     # page units at each end held straight before any rounding
# How the head is drawn: two separate barbs, one V stroke in to the tip and back
# out, or a closed triangle of three separate strokes. There is no filled head:
# scribbled in, it never looked like a pen had made it.
ARROW_HEAD_STYLES = {"barbs": 0.40, "v": 0.40, "closed": 0.20}
# Some people put a little loop partway along an arrow's shaft — and someone who
# does, does it on most of their arrows. So it is decided per page, not per
# arrow: about one page in twenty is written by a looper, and on that page more
# than half its connectors get a loop; on every other page none do (see
# page_generator._add_connector_arrows). Loops are an uncommon habit — at one
# page in ten with 0.8 on it there were still too many, and even at one in twenty
# a looped shaft read as a squiggle more than an arrow, so they are off.
ARROW_LOOP_PAGE_PROB = 0.0      # chance a page's arrows are drawn with loops at all
ARROW_LOOP_PROB_ON_PAGE = 0.60  # ...and then, the chance each arrow on it gets one
# as a fraction of the span, held to a range in page units: much smaller read as
# a blip on the line, much larger dwarfed a short arrow
ARROW_LOOP_RADIUS = (0.04, 0.07)
ARROW_LOOP_RADIUS_RANGE = (7.0, 16.0)


def _catmull_rom(points, steps: int = ARROW_CURVE_STEPS):
    """Smooth polyline through `points` — a hand-drawn arrow is never straight."""
    if len(points) < 2:
        return list(points)
    padded = [points[0]] + list(points) + [points[-1]]
    out = []
    for i in range(len(padded) - 3):
        p0, p1, p2, p3 = padded[i:i + 4]
        # enough samples that a long segment is still a curve once traced, not a
        # run of straight chords with a corner at every sample
        count = max(steps, int(math.hypot(p2[0] - p1[0], p2[1] - p1[1]) / TRACE_STEP))
        for s in range(count):
            t = s / count
            t2, t3 = t * t, t * t * t
            out.append((
                0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t
                       + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                       + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3),
                0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t
                       + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                       + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3),
            ))
    out.append(points[-1])
    return out


def _trace(shape_bank, rng: random.Random, path,
           wobble_length: Optional[float] = None,
           ease_ends: float = 0.0) -> List[Tuple[float, float]]:
    """`path` redrawn with a mined line's hand.

    A mined line is stored as positions along its own chord (`u`, 0 to 1) and
    sideways offsets from it (`v`, in chord lengths). Here the chord is bent to
    follow `path` instead: `v` becomes an offset square to the path's direction,
    applied at evenly spaced points along it. So the path supplies the
    curvature and the ink supplies the hand's waver — the same ink a straight
    table rule would use, taken round a corner.

    The line's `v` is resampled to TRACE_STEP spacing first — a mined line has
    about eleven samples, far too few to hold a curve. Its `u` is not used at
    all: it only says how fast the pen moved along the chord, and about a third
    of mined lines step slightly backwards somewhere. Following it meant holding
    still and then jumping ahead, which drew a straight chord across whatever
    curve the path took there — loops came out as polygons. `wobble_length` caps the
    length the sideways offset scales with, so a long path wavers like a long
    line rather than swinging by a twentieth of its span. Both ends come back
    exactly where `path` starts and ends: a mined line's offset is zero there.

    `ease_ends` fades the sideways offset in and out over that distance at each
    end. A mined line's offset is zero at its ends but grows straight away, and
    over a connector's first few units that turned a square departure 10-35
    degrees off square; eased, the path's own direction holds at the ends and
    the hand's waver is left everywhere in between.
    """
    import numpy as np
    pts = np.asarray(path, dtype=float)
    along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    length = float(along[-1])
    if length < 1e-9:
        return [tuple(p) for p in pts.tolist()]

    v = rng.choice(shape_bank.lines).v
    if rng.random() < 0.5:              # either end could have been the start
        v = v[::-1]
    # never coarser than the path itself, or a finely drawn loop is flattened
    count = max(len(v), int(length / TRACE_STEP) + 2, 2 * len(pts))
    v = np.interp(np.linspace(0.0, len(v) - 1, count), np.arange(len(v)), v)
    # evenly along the path; see the docstring for why the ink's own `u` is not used
    s = np.linspace(0.0, length, count)

    x = np.interp(s, along, pts[:, 0])
    y = np.interp(s, along, pts[:, 1])
    # direction of travel from a short window either side of each sample
    reach = min(2.0 * TRACE_STEP, length / 4.0)
    ahead, behind = np.clip(s + reach, 0.0, length), np.clip(s - reach, 0.0, length)
    tx = np.interp(ahead, along, pts[:, 0]) - np.interp(behind, along, pts[:, 0])
    ty = np.interp(ahead, along, pts[:, 1]) - np.interp(behind, along, pts[:, 1])
    norm = np.hypot(tx, ty)
    norm[norm < 1e-9] = 1.0
    scale = length if wobble_length is None else min(length, wobble_length)
    offset = v * scale
    if ease_ends > 0.0:
        edge = np.clip(np.minimum(s, length - s) / ease_ends, 0.0, 1.0)
        offset = offset * edge * edge * (3.0 - 2.0 * edge)      # smoothstep
    x = x - ty / norm * offset
    y = y + tx / norm * offset
    return list(zip(x.tolist(), y.tolist()))


def _piece_shape(shape_bank, rng: random.Random):
    """One mined line cleaned up for use as a piece of a longer line.

    Returns (t, v): evenly spaced positions 0..1 along the piece and its
    sideways offsets, in piece lengths, with v zero at both ends. Three things
    are done to the raw line, all of which showed up as damage at the joins:
    the ends are trimmed off (ARROW_PIECE_TRIM), since a pen hooks as it sets
    down and lifts; the waver is re-measured from the trimmed part's own ends,
    so the piece still starts and finishes on its chord; and the line's own
    progress along its chord is replaced by even steps, since about a third of
    mined lines step backwards somewhere and that drew spikes.
    """
    import numpy as np
    ink = rng.choice(shape_bank.lines)
    u, v = np.asarray(ink.u, dtype=float), np.asarray(ink.v, dtype=float)
    if rng.random() < 0.5:                  # either end could have been the start
        u, v = 1.0 - u[::-1], v[::-1]
    keep = (u >= ARROW_PIECE_TRIM) & (u <= 1.0 - ARROW_PIECE_TRIM)
    if keep.sum() >= 4:
        first = int(np.argmax(keep))
        last = len(keep) - 1 - int(np.argmax(keep[::-1]))
        u, v = u[first:last + 1], v[first:last + 1]
    span = float(u[-1] - u[0])
    if span < 0.3 or len(v) < 2:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])
    t = np.linspace(0.0, 1.0, len(v))
    v = (v - (v[0] + (v[-1] - v[0]) * t)) / span
    return t, v


def _bezier(p0, p1, p2, p3, count: int):
    """`count` points along the cubic Bezier through the four control points."""
    out = []
    for k in range(count):
        s = k / (count - 1)
        a, b, c, d = (1 - s) ** 3, 3 * s * (1 - s) ** 2, 3 * s * s * (1 - s), s ** 3
        out.append((a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
                    a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]))
    return out


def _chain_ink(shape_bank, rng: random.Random, route,
               start_heading=None, end_heading=None,
               straight_end: float = 0.0) -> Optional[List[Tuple[float, float]]]:
    """A long line drawn the way a pen draws one: real mined lines end to end.

    Each piece is a mined line (_piece_shape) laid from where the previous one
    stopped, setting off along the direction that piece was travelling at its
    end, turned toward the point a piece-length further on along `route` by at
    most ARROW_MAX_JOINT. Where the route turns harder than that allows, the
    piece is shortened (down to ARROW_PIECE_MIN) rather than the line kinked.
    So the line's curvature is the joints between strokes a hand really drew
    plus their own gentle curve; the route only steers.

    The last stretch is different. Aiming a straight piece at the end from
    wherever the chain had got to landed it at an angle (24 degrees off square
    at the median), so the final stretch is one mined line laid along a curve
    that leaves in the chain's direction and arrives in `end_heading`, ending
    exactly on the end of `route`.

    `start_heading` / `end_heading` are directions of travel as (dx, dy): the
    way the line sets off, and the way it is moving as it arrives. Either
    defaults to the route's own direction there.
    """
    import bisect
    import numpy as np
    pts = [tuple(p) for p in route]
    along = [0.0]
    for a, b in zip(pts, pts[1:]):
        along.append(along[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    total, end = along[-1], pts[-1]
    if total < 1e-9 or len(pts) < 2:
        return pts

    def point_at(s: float):
        s = min(max(s, 0.0), total)
        i = min(max(bisect.bisect_left(along, s), 1), len(pts) - 1)
        a, b = pts[i - 1], pts[i]
        step = along[i] - along[i - 1]
        f = (s - along[i - 1]) / step if step > 1e-12 else 0.0
        return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)

    def wrap(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi

    p = pts[0]
    if start_heading is not None:
        heading = math.atan2(start_heading[1], start_heading[0])
    else:
        first = point_at(min(TRACE_STEP * 2, total))
        heading = math.atan2(first[1] - p[1], first[0] - p[0])

    out = [p]
    index = 0
    drawn = 0.0
    for _ in range(ARROW_MAX_PIECES):
        remaining = math.hypot(end[0] - p[0], end[1] - p[1])
        length = rng.uniform(*ARROW_PIECE_LENGTH)
        if remaining <= max(length * 1.5, ARROW_FINAL_STRETCH):
            break                           # near enough for the last stretch
        t, v = _piece_shape(shape_bank, rng)
        # the piece's own direction at each end, against its chord, read over a
        # fifth of it rather than a sample or two
        m = max(1, len(t) // 5)
        start_off = math.atan2(v[m] - v[0], t[m] - t[0])
        end_off = math.atan2(v[-1] - v[-1 - m], t[-1] - t[-1 - m])

        # where the pen is along the route: the nearest route point from here on
        reach = bisect.bisect_right(along, along[index] + 3 * ARROW_PIECE_LENGTH[1])
        index = min(range(index, max(index + 1, reach)),
                    key=lambda i: math.hypot(pts[i][0] - p[0], pts[i][1] - p[1]))
        while True:
            target = point_at(along[index] + length)
            chord = math.atan2(target[1] - p[1], target[0] - p[0])
            joint = wrap(chord + start_off - heading)
            if abs(joint) <= ARROW_MAX_JOINT or length <= ARROW_PIECE_MIN:
                break
            length = max(ARROW_PIECE_MIN, length * 0.5)
        joint = max(-ARROW_MAX_JOINT, min(ARROW_MAX_JOINT, joint))
        chord = heading + joint - start_off

        count = max(len(t), int(length / TRACE_STEP) + 2)
        tt = np.linspace(0.0, 1.0, count)
        vv = np.interp(tt, t, v)
        c, s = math.cos(chord), math.sin(chord)
        for a, b in zip(tt[1:].tolist(), vv[1:].tolist()):
            out.append((p[0] + (a * c - b * s) * length, p[1] + (a * s + b * c) * length))
        p = out[-1]
        heading = chord + end_off
        drawn += length
        if drawn > total * ARROW_CHAIN_SLACK:
            # The chain has lost the route: each piece may only turn a little,
            # and one that falls off a tight bend circles instead of catching
            # up. One did, drawing 618 units of shaft between anchors 117
            # apart. Give up and let the caller try another route.
            return None

    remaining = math.hypot(end[0] - p[0], end[1] - p[1])
    if remaining < 1e-6:
        return out
    if end_heading is not None:
        arrive = math.atan2(end_heading[1], end_heading[0])
    else:
        before = point_at(max(total - 2 * TRACE_STEP, 0.0))
        arrive = math.atan2(end[1] - before[1], end[0] - before[0])
    # The last `straight_end` runs straight into the end along the arrival
    # direction, as its own mined line, and the curve from the chain's heading
    # finishes where that begins. A curve all the way to the end did its
    # tightest turning in the last few units, which is exactly where a head sits:
    # the shaft curled out through the side of a closed head.
    straight = min(straight_end, remaining * 0.5)
    corner = (end[0] - math.cos(arrive) * straight, end[1] - math.sin(arrive) * straight)
    bend = math.hypot(corner[0] - p[0], corner[1] - p[1])
    handle = bend * 0.45
    curve = _bezier(p,
                    (p[0] + math.cos(heading) * handle, p[1] + math.sin(heading) * handle),
                    (corner[0] - math.cos(arrive) * handle, corner[1] - math.sin(arrive) * handle),
                    corner, max(8, int(bend / TRACE_STEP) + 2))
    out.extend(_trace(shape_bank, rng, curve, wobble_length=max(bend, 1e-6))[1:])
    if straight > 1e-6:
        out.extend(_trace(shape_bank, rng, [corner, end], wobble_length=straight)[1:])
    return out


def _smooth_route(points, span: float, passes: int = ARROW_SMOOTH_PASSES):
    """The route thinned to about TRACE_STEP spacing and eased, its lead-in and
    lead-out held fixed.

    The spline through the stubs and waypoints can turn sharply where a stub
    meets the first detour — the little hook an arrow set off with, or arrived
    on. A pen rounds that corner instead of making it. A few passes of a
    [1, 2, 1] filter round every such corner and leave gentle curves as they
    were.

    The stubs themselves are not eased. They are what make an arrow leave and
    land square to the edge it is attached to, and easing the whole route bent
    them too: the arrival drifted to 26 degrees off square at p90. So every
    point within half a stub's length of either end stays where it is. Half,
    not all of it: holding the whole stub left the corner where it meets the
    route unrounded, which is the hook this is here to remove.
    """
    pts = [tuple(p) for p in _resample(points, TRACE_STEP)]
    if len(pts) < 5:
        return pts
    # never less than ARROW_SQUARE_RUN either: on a short arrow half a stub is a
    # few units, and the rounding then started inside the stretch that reads as
    # leaving square
    hold = max(0.5 * min(max(span * ARROW_STUB_FRAC, ARROW_STUB_RANGE[0]),
                         ARROW_STUB_RANGE[1]),
               ARROW_SQUARE_RUN)
    lengths = [0.0]
    for a, b in zip(pts, pts[1:]):
        lengths.append(lengths[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    movable = [hold <= s <= lengths[-1] - hold for s in lengths]
    for _ in range(passes):
        eased = list(pts)
        for k in range(1, len(pts) - 1):
            if movable[k]:
                a, b, c = pts[k - 1], pts[k], pts[k + 1]
                eased[k] = ((a[0] + 2 * b[0] + c[0]) / 4.0, (a[1] + 2 * b[1] + c[1]) / 4.0)
        pts = eased
    return pts


def _polyline_stroke(points) -> Stroke:
    return Stroke(x=[p[0] for p in points], y=[p[1] for p in points])


def _segment_hits_box(p0, p1, box, pad: float = 0.0) -> bool:
    """Whether segment p0-p1 touches an axis-aligned box, via the slab method."""
    x0, y0, x1, y1 = box
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    t_lo, t_hi = 0.0, 1.0
    for origin, delta, lo, hi in ((p0[0], dx, x0, x1), (p0[1], dy, y0, y1)):
        if abs(delta) < 1e-12:
            if origin < lo or origin > hi:
                return False        # parallel and outside this slab
            continue
        a, b = (lo - origin) / delta, (hi - origin) / delta
        if a > b:
            a, b = b, a
        t_lo, t_hi = max(t_lo, a), min(t_hi, b)
        if t_lo > t_hi:
            return False
    return True


ARROW_MAX_TURN = 95.0       # degrees; a sharper corner than this is a doubling-back


ARROW_TURN_STEP = 12.0      # arc length the turn test looks over, in page units


def _resample(points, step: float = ARROW_TURN_STEP):
    """Points spaced roughly `step` apart along the path."""
    out = [points[0]]
    travelled = 0.0
    for a, b in zip(points, points[1:]):
        travelled += math.hypot(b[0] - a[0], b[1] - a[1])
        if travelled >= step:
            out.append(b)
            travelled = 0.0
    # Only keep the true endpoint if it is far enough to have a meaningful
    # direction; a 1px tail segment is pure tremor and would read as a huge
    # turn that isn't there.
    if out[-1] != points[-1]:
        tail = math.hypot(points[-1][0] - out[-1][0], points[-1][1] - out[-1][1])
        if tail >= step / 3.0:
            out.append(points[-1])
        elif len(out) > 1:
            out[-1] = points[-1]
    return out


def _turns_back(points, limit: float = ARROW_MAX_TURN) -> bool:
    """Whether the path ever swings back on itself.

    This is what produced arrows that ran one way and then hooked round to come
    in from the other side, so the head appeared to point backwards along the
    shaft (`-----<-`). It happens when a detour throws a waypoint past the
    target and the route has to double back to reach it.

    The test has to run on a path resampled at a real distance, not on the raw
    spline samples: those are spaced a pixel or two apart, so a full 180-degree
    reversal spread over nine of them is only twenty degrees per step and slips
    straight through a per-sample check. Twelve pixels is roughly the scale the
    reversal is visible at.
    """
    threshold = math.cos(math.radians(limit))
    coarse = _resample(points)
    previous = None
    for a, b in zip(coarse, coarse[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length < 1e-9:
            continue
        current = (dx / length, dy / length)
        if previous is not None:
            if previous[0] * current[0] + previous[1] * current[1] < threshold:
                return True
        previous = current
    return False


ARROW_MAX_OPPOSE = -0.15    # cosine; how far the ends may oppose the overall run


def _reverses(points, limit: float = ARROW_MAX_OPPOSE) -> bool:
    """Whether the arrow sets off or arrives against its own overall direction.

    This is the precise thing that looked wrong: an arrow that travels left to
    right and then hooks round to come back in from the right, so the head reads
    as pointing backwards along the shaft. Testing it directly — the first and
    last leg against the start-to-end run — catches it where a general
    corner-angle test does not, because the reversal is spread smoothly over a
    long curve rather than concentrated in one sharp bend.
    """
    coarse = _resample(points, ARROW_TURN_STEP * 1.25)
    if len(coarse) < 3:
        return False
    run = (coarse[-1][0] - coarse[0][0], coarse[-1][1] - coarse[0][1])
    run_length = math.hypot(*run)
    if run_length < 1e-6:
        return True

    for a, b in ((coarse[-2], coarse[-1]), (coarse[0], coarse[1])):
        leg = (b[0] - a[0], b[1] - a[1])
        leg_length = math.hypot(*leg)
        if leg_length < 1e-6:
            continue
        cosine = (leg[0] * run[0] + leg[1] * run[1]) / (leg_length * run_length)
        if cosine < limit:
            return True
    return False


def _path_is_clear(points, obstacles, pad: float = ARROW_CLEARANCE) -> bool:
    # A chained shaft is sampled every few units, and slab-testing every one of
    # its segments against every nearby box was half the time a page took. So
    # boxes the whole path cannot reach are dropped first, and each segment only
    # runs the exact test against boxes its own extent overlaps.
    if len(points) < 2:
        return True
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    lo_x, hi_x = min(xs) - pad, max(xs) + pad
    lo_y, hi_y = min(ys) - pad, max(ys) + pad
    boxes = [b for b in obstacles
             if b[0] <= hi_x and b[2] >= lo_x and b[1] <= hi_y and b[3] >= lo_y]
    if not boxes:
        return True
    for a, b in zip(points, points[1:]):
        sx0, sx1 = min(a[0], b[0]) - pad, max(a[0], b[0]) + pad
        sy0, sy1 = min(a[1], b[1]) - pad, max(a[1], b[1]) + pad
        for box in boxes:
            if (box[0] <= sx1 and box[2] >= sx0 and box[1] <= sy1 and box[3] >= sy0
                    and _segment_hits_box(a, b, box, pad)):
                return False
    return True


def _strokes_are_clear(strokes: List[Stroke], obstacles,
                        pad: float = ARROW_CLEARANCE) -> bool:
    """Whether every one of these strokes stays off every box."""
    return all(_path_is_clear(list(zip(stroke.x, stroke.y)), obstacles, pad)
               for stroke in strokes)


ARROW_STUB_FRAC = 0.14      # length of the perpendicular lead-in, as a span fraction
ARROW_STUB_RANGE = (9.0, 42.0)


def _stub(point, normal, span: float, rng: random.Random, fraction: float = 1.0):
    """The point a short way out from `point` along `normal`.

    Both ends of a connector get one of these. Because the spline is anchored to
    duplicated endpoints, its tangent at each end is exactly the direction of
    the neighbouring point — so putting that point straight out along the
    normal is what makes the arrow leave (and land) square to the edge it is
    attached to, instead of skimming across the text at a shallow angle.
    """
    length = min(max(span * ARROW_STUB_FRAC, ARROW_STUB_RANGE[0]), ARROW_STUB_RANGE[1])
    # kept close to the nominal length: this is what sets how far the arrow runs
    # straight before it starts to curve, and letting it vary widely made two
    # arrows off the same edge leave at visibly different rates
    length *= rng.uniform(0.9, 1.15) * fraction
    return (point[0] + normal[0] * length, point[1] + normal[1] * length)


def _sample_route(start, end, rng: random.Random, urgency: float = 0.0,
                   start_normal=None, end_normal=None):
    """One candidate path from start to end, with a randomly sized detour.

    `urgency` rises as earlier attempts get rejected, widening the detours the
    sampler is willing to draw. That keeps the *first* routes tried close to
    direct — so direct routes stay the most probable outcome — while still
    finding a way around when the straight line is blocked.

    The normals, when given, fix the direction the arrow leaves and arrives on.
    """
    sx, sy = start
    ex, ey = end
    span = math.hypot(ex - sx, ey - sy)
    if span < 1e-6:
        return [start, end]

    # unit vectors along the run and across it
    ux, uy = (ex - sx) / span, (ey - sy) / span
    px, py = -uy, ux

    sigma = ARROW_DETOUR_SIGMA * (1.0 + 1.8 * urgency)
    if rng.random() < ARROW_DETOUR_TAIL:
        sigma *= rng.uniform(1.8, 3.2)      # the occasional wide swing

    n = rng.choices(list(ARROW_WAYPOINT_WEIGHTS),
                    weights=list(ARROW_WAYPOINT_WEIGHTS.values()))[0]
    side = rng.choice((-1.0, 1.0))          # detours usually stay on one side
    limit = ARROW_MAX_DETOUR * span

    points = [start]
    if start_normal is not None:
        # two points along the normal, not one: a single stub sets the initial
        # tangent but the spline starts bending away immediately, so the arrow
        # only looked perpendicular for a pixel or two. A short straight run
        # holds the direction long enough to read as square to the edge.
        points.append(_stub(start, start_normal, span, rng, 0.45))
        points.append(_stub(start, start_normal, span, rng))
    for i in range(1, n + 1):
        t = i / (n + 1) + rng.uniform(-0.12, 0.12)
        t = min(max(t, 0.05), 0.95)
        across = side * min(abs(rng.gauss(0.0, sigma)) * span, limit)
        if rng.random() < ARROW_WEAVE_PROB:
            side = -side                    # ...but sometimes they weave
        along = rng.uniform(-0.05, 0.05) * span
        points.append((sx + ux * (t * span + along) + px * across,
                        sy + uy * (t * span + along) + py * across))
    if end_normal is not None:
        points.append(_stub(end, end_normal, span, rng))
        points.append(_stub(end, end_normal, span, rng, 0.45))
    points.append(end)
    return _catmull_rom(points)


def _arrow_head_strokes(points, span: float, rng: random.Random,
                        shape_bank) -> List[Stroke]:
    """The head at the end of the shaft, in mined ink.

    Its geometry is deliberately the tidiest part of the arrow. A head is made
    without lifting attention from the tip, so it comes out more consistent
    than the shaft that led to it: the size and the spread vary a little, and
    the two barbs share one spread rather than each drawing its own. Its ink is
    as real as the shaft's — every stroke of it is a mined line (_trace) — and
    it is drawn one of the ways people draw one (ARROW_HEAD_STYLES): two
    separate barbs, a single V stroke in to the tip and back out, or a closed
    triangle. The triangle is three separate strokes, each its own mined line —
    in along one side to the tip, out along the other, back across the base —
    since one stroke bent round all three corners read as traced. There is no
    filled head. The V's corner is pinned to the tip, and every other stroke
    that meets the tip starts or ends on it, so the head always sits exactly
    where the shaft ended.

    The direction the head points is the chord of the last head's length of
    shaft — the stretch the head actually covers — so the shaft runs up the
    middle of a closed head instead of cutting across its side. Aiming along
    only the last few units turned the head off the shaft wherever the shaft
    was still curving as it arrived.
    """
    tip = points[-1]
    size = max(span * ARROW_HEAD_FRAC, 4.0) * rng.uniform(0.92, 1.1)
    aim = size * 0.9
    back = points[0]
    for point in reversed(points[:-1]):
        back = point
        if math.hypot(tip[0] - point[0], tip[1] - point[1]) >= aim:
            break
    ang = math.atan2(tip[1] - back[1], tip[0] - back[0])
    spread = rng.uniform(math.pi / 7, math.pi / 5.2)

    outer = []
    for sign in (-1.0, 1.0):
        # a touch of asymmetry, far less than drawing each barb independently
        a = ang + math.pi + sign * spread * rng.uniform(0.96, 1.04)
        outer.append((tip[0] + size * math.cos(a), tip[1] + size * math.sin(a)))

    def pinned(points):
        """The stroke moved so its point nearest the tip lands on the tip."""
        corner = min(points, key=lambda p: math.hypot(p[0] - tip[0], p[1] - tip[1]))
        dx, dy = tip[0] - corner[0], tip[1] - corner[1]
        return _polyline_stroke([(x + dx, y + dy) for x, y in points])

    style = rng.choices(list(ARROW_HEAD_STYLES),
                        weights=list(ARROW_HEAD_STYLES.values()))[0]
    if style == "v":
        return [pinned(_trace(shape_bank, rng, [outer[0], tip, outer[1]]))]
    if style == "closed":
        # three separate mined lines — in along one side, out along the other,
        # back across the base — each wavering less than a line its length
        # would, since a head is small and drawn with care. They meet exactly at
        # the corners: a mined line ends on its chord.
        return [_polyline_stroke(_trace(shape_bank, rng, [a, b], wobble_length=size * 0.5))
                for a, b in ((outer[0], tip), (tip, outer[1]), (outer[1], outer[0]))]
    # in to the tip along one barb, out from it along the other
    return [_polyline_stroke(_trace(shape_bank, rng, [outer[0], tip])),
            _polyline_stroke(_trace(shape_bank, rng, [tip, outer[1]]))]


def _within(points, bounds) -> bool:
    """Whether every point stays inside `bounds` = (x0, y0, x1, y1)."""
    if bounds is None:
        return True
    x0, y0, x1, y1 = bounds
    return all(x0 <= px <= x1 and y0 <= py <= y1 for px, py in points)


def _add_loop(points, span: float, rng: random.Random):
    """The route with one small loop drawn into it partway along.

    Some people put a little curl in an arrow's shaft. The loop leaves the route
    tangent to it, goes once round, and rejoins it a short way further on — it
    drifts forward as it turns, the way a cursive "e" does, rather than
    retracing a closed circle on the spot. It sits in the middle stretch of the
    route, never near the anchor or the head.

    Added after the route has passed the doubling-back tests, which a loop fails
    by definition, and before the drawn arrow is tested against the obstacles,
    so a loop still cannot land on anything.
    """
    if len(points) < 8:
        return points, None
    lengths = [0.0]
    for a, b in zip(points, points[1:]):
        lengths.append(lengths[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    total = lengths[-1]
    radius = min(max(span * rng.uniform(*ARROW_LOOP_RADIUS), ARROW_LOOP_RADIUS_RANGE[0]),
                 ARROW_LOOP_RADIUS_RANGE[1])
    if total < radius * 8:
        return points, None         # too short to hold a loop and still read as an arrow

    target = rng.uniform(0.35, 0.65) * total
    i = next(k for k, s in enumerate(lengths) if s >= target)
    # the loop rejoins a little further on, which is the forward drift
    j = next((k for k, s in enumerate(lengths) if s >= lengths[i] + radius * 0.8),
             len(points) - 1)
    if i < 2 or j >= len(points) - 2:
        return points, None

    before, after = points[i - 2], points[i + 2]
    tx, ty = after[0] - before[0], after[1] - before[1]
    norm = math.hypot(tx, ty) or 1.0
    tx, ty = tx / norm, ty / norm
    side = rng.choice((-1.0, 1.0))
    start = points[i]
    centre = (start[0] - ty * side * radius, start[1] + tx * side * radius)
    a0 = math.atan2(start[1] - centre[1], start[0] - centre[0])
    # go round the way that leaves along the direction of travel
    turn = 1.0 if (-math.sin(a0) * tx + math.cos(a0) * ty) > 0 else -1.0
    drift = (points[j][0] - start[0], points[j][1] - start[1])
    # finely: a small loop at TRACE_STEP spacing is only a dozen points, which
    # draws a polygon
    steps = max(24, int(2 * math.pi * radius / (0.5 * TRACE_STEP)))
    loop = []
    for k in range(1, steps):
        f = k / steps
        a = a0 + turn * 2 * math.pi * f
        loop.append((centre[0] + radius * math.cos(a) + drift[0] * f,
                     centre[1] + radius * math.sin(a) + drift[1] * f))
    # also where the loop is in the new route — from the point it leaves to the
    # point it rejoins — and the circle it was laid out on, so the shaft can
    # draw it as one stroke of mined oval ink (_draw_shaft, _loop_ink)
    return (list(points[:i + 1]) + loop + list(points[j:]),
            {"leave": i, "rejoin": i + 1 + len(loop), "centre": centre,
             "radius": radius, "angle": a0, "turn": turn})


def _loop_ink(shape_bank, rng: random.Random, ring, loop) -> List[Tuple[float, float]]:
    """A loop drawn with a mined oval: real hand-drawn loop ink, once round.

    The shape comes from the bank's ovals (geometry_diagrams.arc), the same ink
    a geometry figure's ellipses use, so the loop carries a real hand's lean and
    unevenness rather than being a circle with a line's waver laid over it. It
    goes once round the circle _add_loop laid out, in the direction of travel,
    with none of the overshoot arc adds to a closed ellipse.

    A mined oval's radius wanders a little off its fitted ellipse, so its ends
    do not fall exactly where the loop leaves and rejoins the route — and the
    rejoin point is a little further on, since the loop drifts forward as it
    turns. Both are fixed by one offset blended along the stroke, from the gap
    at the start to the gap at the end, which moves the ends onto the route
    without changing the shape between them.
    """
    import geometry_diagrams as gd
    start_angle = loop["angle"]
    # just short of a full turn, so arc treats it as open and adds no overshoot
    end_angle = start_angle + loop["turn"] * (2 * math.pi - 1e-3)
    stroke = gd.arc(shape_bank, rng, loop["centre"], loop["radius"], loop["radius"],
                    start_angle=start_angle, end_angle=end_angle,
                    samples=max(len(ring), 24))
    points = list(zip(stroke.x, stroke.y))
    (sx, sy), (ex, ey) = ring[0], ring[-1]
    head = (sx - points[0][0], sy - points[0][1])
    tail = (ex - points[-1][0], ey - points[-1][1])
    last = len(points) - 1
    return [(x + head[0] * (1 - k / last) + tail[0] * (k / last),
             y + head[1] * (1 - k / last) + tail[1] * (k / last))
            for k, (x, y) in enumerate(points)]


def _draw_shaft(shape_bank, rng: random.Random, route, loop,
                start_normal=None, end_normal=None,
                straight_end: float = 0.0) -> List[Tuple[float, float]]:
    """The shaft along `route` in mined ink: chained pieces, with a loop, if
    the route has one, drawn as a single mined line bent round it.

    A loop is one quick motion of the pen, and a chain whose joins may each turn
    only ARROW_MAX_JOINT cannot follow a curl that tight. So the chain stops
    where the loop leaves the route, the loop is one mined oval drawn round it
    (_loop_ink), and a second chain carries on from where it rejoins. The headings agree at both joins, since
    the loop leaves and rejoins the route tangent to it.

    `start_normal` is the anchor's outward normal, which is the direction the
    shaft sets off in; `end_normal` points out of the target, so the shaft
    arrives travelling against it.
    """
    arrive = (-end_normal[0], -end_normal[1]) if end_normal is not None else None
    if loop is None:
        return _chain_ink(shape_bank, rng, route, start_normal, arrive, straight_end)
    leave_at, rejoin_at = loop["leave"], loop["rejoin"]
    ring = route[leave_at:rejoin_at + 1]
    leave = (ring[1][0] - ring[0][0], ring[1][1] - ring[0][1])
    rejoin = None
    if rejoin_at + 1 < len(route):
        rejoin = (route[rejoin_at + 1][0] - route[rejoin_at][0],
                  route[rejoin_at + 1][1] - route[rejoin_at][1])
    before = _chain_ink(shape_bank, rng, route[:leave_at + 1], start_normal, leave)
    after = _chain_ink(shape_bank, rng, route[rejoin_at:], rejoin, arrive, straight_end)
    if before is None or after is None:
        return None                 # a chain lost the route; see _chain_ink
    ring_ink = _loop_ink(shape_bank, rng, ring, loop)
    return before + ring_ink[1:] + after[1:]


def make_connector_arrow(start, end, obstacles, rng: random.Random = random,
                          start_normal=None, end_normal=None,
                          bounds=None, shape_bank=None,
                          loop_prob: float = 0.0) -> Optional[Ink]:
    """An arrow from `start` to `end` that doesn't cross anything in `obstacles`.

    Drawn in mined ink from `shape_bank`: the route is eased (_smooth_route),
    the shaft is several mined lines laid end to end along it (_chain_ink) and
    the head is mined too (_arrow_head_strokes). No bank, no arrow.

    Routes are sampled and rejected rather than planned: most draws are close to
    direct, a minority swing wide, and whichever first comes back clear of the
    obstacles is the one drawn. Returns None if no clear route was found, which
    is the honest outcome when the two points are boxed in.

    `start_normal` / `end_normal` fix the direction the arrow leaves and lands —
    square to the edge of the line or the stroke it is attached to.

    `bounds` is the rectangle the whole arrow must stay inside. Nothing else
    holds it in: a route may swing out by most of its own span (ARROW_MAX_DETOUR)
    and the obstacles are only the things on the page, so an arrow near an edge
    was free to loop out over it and come back.
    """
    if shape_bank is None or not shape_bank.ok():
        return None
    span = math.hypot(end[0] - start[0], end[1] - start[1])

    # Only boxes near the corridor between the endpoints can ever be hit, and
    # the clearance test runs on every segment of every candidate route, so
    # discarding the rest up front is worth it.
    reach = span * (ARROW_MAX_DETOUR + 0.6) + ARROW_STUB_RANGE[1] + ARROW_CLEARANCE
    lo_x, hi_x = min(start[0], end[0]) - reach, max(start[0], end[0]) + reach
    lo_y, hi_y = min(start[1], end[1]) - reach, max(start[1], end[1]) + reach
    nearby = [b for b in obstacles
              if b[0] <= hi_x and b[2] >= lo_x and b[1] <= hi_y and b[3] >= lo_y]

    for attempt in range(ARROW_ROUTE_ATTEMPTS):
        points = _smooth_route(_sample_route(start, end, rng,
                                             attempt / ARROW_ROUTE_ATTEMPTS,
                                             start_normal, end_normal), span)
        if _turns_back(points) or _reverses(points):
            continue
        # A route that already runs into something will not be cleared by ink
        # laid along it, so it is thrown out before a shaft is chained over it.
        if not _path_is_clear(points, nearby, pad=0.0):
            continue
        # When a loop is rolled, the looped route is tried first and the plain
        # one straight after if the loop lands on something. A loop sticks out
        # sideways, and throwing the whole attempt away for it kept a loop on 1
        # arrow in 25 — 156 loops drawn and rejected — instead of 1 in 8.
        routes = [(points, None)]
        if loop_prob > 0.0 and rng.random() < loop_prob:
            looped, loop = _add_loop(points, span, rng)
            if loop is not None:
                routes.insert(0, (looped, loop))

        for route, loop in routes:
            # Everything below is checked as *drawn*, not as routed. The shaft is
            # traced in mined ink before it is tested, because the hand's waver
            # pushes it sideways by about as much as the clearance allows —
            # clearing the route and then bending the line was how an arrow ended
            # up crossing a box it had already been checked against. The head is
            # tested for the same reason: it is drawn ink sticking out sideways
            # from the tip.
            # the head covers at most this much shaft (see _arrow_head_strokes),
            # and that stretch is drawn straight so the head sits on it
            straight_end = max(span * ARROW_HEAD_FRAC, 4.0) * 1.15
            shaft = _draw_shaft(shape_bank, rng, route, loop, start_normal, end_normal,
                                straight_end)
            if (shaft is None or not _within(shaft, bounds)
                    or not _path_is_clear(shaft, nearby)):
                continue
            head = _arrow_head_strokes(shaft, span, rng, shape_bank)
            if not _strokes_are_clear(head, nearby):
                continue
            if not all(_within(list(zip(s.x, s.y)), bounds) for s in head):
                continue

            strokes = [_polyline_stroke(shaft)]
            strokes.extend(head)
            return Ink(strokes=strokes, class_name="arrow", source="geometry")
    return None


def draw_dash_stroke(x, y, length=3.0) -> Stroke:
    pts = _jitter_pts([(x - length / 2, y), (x + length / 2, y)], noise=0.2)
    return Stroke(x=[p[0] for p in pts], y=[p[1] for p in pts])


def draw_bullet_stroke(x, y, r=1.5) -> List[Stroke]:
    """A list-item marker: either a dot or a dash, never an asterisk.

    Both come in a few variants, because a hand-drawn marker is never the same
    twice — a dot may be a tight scribble, a small open circle or a hard jab,
    and a dash varies in length, slope and straightness. Always returns a list
    of strokes so callers don't special-case the multi-stroke variants.
    """
    if random.random() < 0.5:
        return _draw_dot_marker(x, y, r)
    return _draw_dash_marker(x, y, r)


def _draw_dot_marker(x, y, r) -> List[Stroke]:
    style = random.choice(["scribble", "circle", "jab"])
    if style == "circle":
        # a small open circle, sometimes not quite closed
        rr = r * random.uniform(0.7, 1.2)
        stroke = draw_oval_stroke(x - rr, y - rr, x + rr, y + rr,
                                   n_points=random.randint(8, 12))
        if random.random() < 0.4:
            keep = len(stroke.x) - random.randint(1, 2)
            stroke = Stroke(x=stroke.x[:keep], y=stroke.y[:keep])
        return [stroke]

    if style == "scribble":
        # the usual filled dot: a couple of tight loops without lifting the pen
        rr = r * random.uniform(0.4, 0.8)
        turns = random.uniform(1.6, 2.6)
        n = 14
        pts = []
        for i in range(n + 1):
            a = 2 * math.pi * turns * i / n
            shrink = 1.0 - 0.45 * i / n
            pts.append((x + rr * shrink * math.cos(a), y + rr * shrink * math.sin(a)))
        pts = _jitter_pts(pts, noise=r * 0.12)
        return [Stroke(x=[p[0] for p in pts], y=[p[1] for p in pts])]

    # a hard jab: barely any travel, just a pen-down blob
    rr = r * random.uniform(0.15, 0.35)
    pts = _jitter_pts([(x, y)] * random.randint(2, 4), noise=rr)
    return [Stroke(x=[p[0] for p in pts], y=[p[1] for p in pts])]


def _draw_dash_marker(x, y, r) -> List[Stroke]:
    length = r * random.uniform(1.4, 3.2)
    slope = math.radians(random.uniform(-12, 12))     # hand-drawn dashes tilt
    dx, dy = length / 2 * math.cos(slope), length / 2 * math.sin(slope)
    if random.random() < 0.35:
        # a slightly bowed dash rather than a straight one
        mid = (x + random.uniform(-0.2, 0.2) * r, y + random.uniform(-0.35, 0.35) * r)
        pts = _jitter_pts([(x - dx, y - dy), mid, (x + dx, y + dy)], noise=r * 0.12)
    else:
        pts = _jitter_pts([(x - dx, y - dy), (x + dx, y + dy)], noise=r * 0.15)
    return [Stroke(x=[p[0] for p in pts], y=[p[1] for p in pts])]


# ----------------------------------------------------------------------------
# Scribbles: removed
# ----------------------------------------------------------------------------
# There used to be a family of procedural scribble generators here — crossings-
# out, loops, spirals, wandering squiggles, hatching — emitted as level-0
# `unknown` ink to fill a share of the drawing cells.
#
# They are gone, and nothing on a generated page carries `unknown` any more. See
# page_generator's note where SCRIBBLE_PROB used to be: they were invented ink
# rather than anything a source dataset contains, drawn by formula and parked in
# a cell of their own instead of over the thing they were supposedly crossing
# out, so they taught the `unknown` class a shape that no real page produces.


def draw_grid_strokes(x0, y0, x1, y1, n_rows, n_cols) -> List[Stroke]:
    strokes = [draw_rect_stroke(x0, y0, x1, y1)]
    for r in range(1, n_rows):
        y = y0 + (y1 - y0) * r / n_rows
        pts = _jitter_pts([(x0, y), (x1, y)], noise=0.3)
        strokes.append(Stroke(x=[p[0] for p in pts], y=[p[1] for p in pts]))
    for c in range(1, n_cols):
        x = x0 + (x1 - x0) * c / n_cols
        pts = _jitter_pts([(x, y0), (x, y1)], noise=0.3)
        strokes.append(Stroke(x=[p[0] for p in pts], y=[p[1] for p in pts]))
    return strokes


# ----------------------------------------------------------------------------
# Block generators
# ----------------------------------------------------------------------------

LINE_HEIGHT = 14.0      # target x-height band of a text line (NOT its full bbox)
LINE_GAP = 14.0         # room for ascenders/descenders to run into
MATH_CORE_MULT = 1.1    # expressions read slightly larger than prose
MATH_MAX_HEIGHT = 3.2   # ...but a tall one may not swallow its neighbours, in LINE_HEIGHTs
# The smallest writing inside a table or a flowchart may come out on the page, as a
# share of the page's own handwriting size (x-height), counting every shrink on the
# way: a formula fitted to its cell, a caption fitted to its shape, and the whole
# table or chart fitted to its page cell. Without it, a big table or a busy chart
# in a small cell had its writing squeezed until not even a person could read it.
# Content that would go below this is not used — a smaller table, another caption,
# a caption trimmed to fewer words, a blank shape or a textblock instead — rather
# than squeezed.
MIN_TEXT_RATIO = 0.9
# The floor for the writing in a flowchart, in place of MIN_TEXT_RATIO. A chart's
# 9-30 shapes are fitted into one cell together, so at 0.9 hardly a chart's shapes
# had room for a line of the page's handwriting, and flowcharts all but left the
# dataset (0.009 a val page, from 0.45).
FLOWCHART_MIN_TEXT_RATIO = 0.65

# Fraction of a line's ink, by point count, taken to be the x-height band.
CORE_BAND = (0.12, 0.88)

# No two lines — textlines and formulas alike, in any pairing — may share more than
# this of the smaller one's box area. A paragraph's rows sit on a fixed baseline grid,
# so a tall ascender, a deep descender or an inline expression's fraction reaches into
# the row beside it, as handwriting does; past this share it reads as two lines written
# over each other. A row is pushed down to keep to it (fill_text_cell), a flowchart
# caption landing on another is taken off (make_diagram), and generate_page builds the
# page again should any pair still break it. Table cells, geometry labels
# (geometry_diagrams._place_label) and a maths section's rows are kept apart
# altogether, which is stricter, and stay so.
LINE_CLASSES = ("textline", "formula")
LINE_MAX_OVERLAP = 0.30
_OVERLAP_EPS = 1e-9


def overlap_share(a, b) -> float:
    """The area boxes `a` and `b` share, as a share of the smaller one's area."""
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return width * height / smaller if smaller > 0 else 0.0


def line_overlaps(blocks, limit: float = LINE_MAX_OVERLAP) -> List[Tuple[Ink, Ink]]:
    """Every pair of lines (LINE_CLASSES: textline against textline, formula or formula
    against formula) anywhere under `blocks` sharing more than `limit` of the smaller
    one's box."""
    lines = []

    def walk(node: Ink):
        if node.class_name in LINE_CLASSES and node.all_strokes():
            lines.append((node, node.bbox()))
        for child in node.children:
            walk(child)

    for block in blocks:
        walk(block)
    return [(a, b) for i, (a, box_a) in enumerate(lines) for b, box_b in lines[i + 1:]
            if overlap_share(box_a, box_b) > limit + _OVERLAP_EPS]


def _clear_push(box, above, limit: float = LINE_MAX_OVERLAP) -> float:
    """How far down `box` must move to share no more than `limit` with any of the boxes
    `above` (rows already placed): the least of the pushes that work, trying for each box
    the one that leaves exactly `limit` shared and the one that clears it altogether."""
    x0, y0, x1, y1 = box
    area = (x1 - x0) * (y1 - y0)
    pushes = {0.0}
    for other in above:
        width = min(x1, other[2]) - max(x0, other[0])
        if width <= 0 or other[3] <= y0:
            continue            # beside it, or already below it: moving down cannot overlap it
        smaller = min(area, (other[2] - other[0]) * (other[3] - other[1]))
        pushes.add(other[3] - y0)
        pushes.add(max(0.0, other[3] - y0 - limit * smaller / width))
    for dy in sorted(pushes):
        moved = (x0, y0 + dy, x1, y1 + dy)
        if all(overlap_share(moved, other) <= limit + _OVERLAP_EPS for other in above):
            return dy
    return max(pushes)          # clear of every box above: nothing is shared


def _core_band(ink: Ink) -> Optional[Tuple[float, float]]:
    """The vertical band holding the bulk of a text line's ink.

    Sizing a line by its bbox height is wrong, and it is what made handwriting
    size jump around inside a single paragraph: "we ran on" has no ascender or
    descender, so its bbox is barely an x-height tall, while "of a proposed
    House of chiefs." spans nearly three times that. Normalising both to the
    same bbox height renders the first at triple the size of the second.

    Trimming the top and bottom few percent of the ink dodges that, leaving the
    x-height band, which is what the eye reads as text size. The trimming is
    weighted by **arc length**, not by point count: the datasets sample at a
    constant rate, so points bunch up wherever the pen moved slowly — around the
    baseline cusps of cursive especially. Counting points made DeepWriting's
    band collapse toward its baseline and the line get scaled up several times
    too large next to IAM's. Arc length is a property of the written shape, so
    it compares across sources.
    """
    segments = []
    for stroke in ink.all_strokes():
        xs, ys = stroke.x, stroke.y
        for i in range(len(xs) - 1):
            length = math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
            if length > 0.0:
                segments.append(((ys[i] + ys[i + 1]) / 2.0, length))

    if not segments:
        ys = sorted(y for stroke in ink.all_strokes() for y in stroke.y)
        if not ys:
            return None
        lo = ys[int(len(ys) * CORE_BAND[0])]
        hi = ys[min(len(ys) - 1, int(len(ys) * CORE_BAND[1]))]
        return (lo, hi) if hi > lo else None

    segments.sort()
    total = sum(length for _, length in segments)
    targets = [total * CORE_BAND[0], total * CORE_BAND[1]]
    bounds, running, at = [], 0.0, 0
    for y, length in segments:
        running += length
        while at < len(targets) and running >= targets[at]:
            bounds.append(y)
            at += 1
        if at >= len(targets):
            break
    while len(bounds) < 2:
        bounds.append(segments[-1][0])
    return (bounds[0], bounds[1]) if bounds[1] > bounds[0] else None


def _line_placement(line: Ink, cur_y: float) -> Tuple[float, float]:
    """(scale, y-offset) putting one line on the baseline grid at `cur_y`."""
    bx0, by0, bx1, by1 = line.bbox()

    if line.source == "mathwriting":
        # Same trap as prose, worse: one tall integral sign or fraction stack
        # blows up an expression's bbox, so sizing by bbox height shrinks every
        # glyph in it to a fraction of the surrounding text. Size by the band
        # holding the bulk of the ink instead, so the main row of symbols
        # matches the prose, and only cap the total so a really tall expression
        # cannot run over its neighbours.
        band = _core_band(line)
        height = by1 - by0
        if band is None:
            if height <= 1e-9:
                height = max(bx1 - bx0, 1e-9) * 0.2
            scale = MATH_CORE_MULT * LINE_HEIGHT / height
        else:
            scale = MATH_CORE_MULT * LINE_HEIGHT / (band[1] - band[0])
            if height * scale > MATH_MAX_HEIGHT * LINE_HEIGHT:
                scale = MATH_MAX_HEIGHT * LINE_HEIGHT / height
        return scale, cur_y - by0 * scale

    band = _core_band(line)
    if band is None:
        height = by1 - by0
        if height <= 1e-9:
            # no vertical extent at all (a single flat stroke): assume an aspect
            # rather than a constant, which would depend on the source's units
            height = max(bx1 - bx0, 1e-9) * 0.2
        scale = LINE_HEIGHT / height
        return scale, cur_y - by0 * scale

    core_lo, core_hi = band
    scale = LINE_HEIGHT / (core_hi - core_lo)
    return scale, cur_y - core_lo * scale     # align the x-height band, not the bbox


def _stack_lines_vertically(lines: List[Ink], x0=0.0, left_margin_jitter=4.0) -> Ink:
    """Place a list of line Inks (already labeled class_name='textline') vertically,
    left-aligned with slight indent jitter, returning a container Ink with them
    as children in a local coordinate frame starting at y=0.

    Lines sit on a fixed baseline grid and are scaled so their x-height matches,
    so ascenders and descenders overhang into LINE_GAP exactly as real
    handwriting does."""
    cur_y = 0.0
    placed = []
    for line in lines:
        bx0 = line.bbox()[0]
        scale, dy = _line_placement(line, cur_y)
        indent = random.uniform(0, left_margin_jitter)
        dx = x0 + indent - bx0 * scale

        def transform(s: Stroke, scale=scale, dx=dx, dy=dy) -> Stroke:
            # times ride along: recognition colours a line's ink by them (recog_synth/recog_pages.py)
            return Stroke(x=[dx + p * scale for p in s.x], y=[dy + p * scale for p in s.y], t=s.t)

        def walk(node: Ink, scale=scale, dx=dx, dy=dy) -> Ink:
            node.strokes = [transform(s, scale, dx, dy) for s in node.strokes]
            node.children = [walk(c, scale, dx, dy) for c in node.children]
            return node

        line = walk(line)
        if line.class_name in LINE_CLASSES:
            # no more than LINE_MAX_OVERLAP into a line above: this one goes down
            dy = _clear_push(line.bbox(), [p.bbox() for p in placed if p.class_name in LINE_CLASSES])
            if dy > 0.0:
                _shift_ink(line, 0.0, dy)
                cur_y += dy
        placed.append(line)
        cur_y += LINE_HEIGHT + LINE_GAP
    return Ink(strokes=[], class_name=None, children=placed)


MATH_LINE_PROB = 0.35   # chance any one line of a textblock is an expression


class LineDealer:
    """Hands out lines without repeating one until the pool is exhausted.

    Picking each line independently at random is fine against a pool of
    thousands, but a page draws all its prose from a single writer (see
    page_generator._single_writer_pool) and that is only a few dozen lines — so
    independent picks put the same sentence on a page two or three times.
    Dealing from a shuffled deck instead means a page repeats itself only if it
    genuinely needs more lines than that writer wrote.
    """

    def __init__(self, pool: List[Ink]):
        self.pool = list(pool)
        self._order = list(range(len(self.pool)))
        random.shuffle(self._order)
        self._next = 0
        # Every transcription already written on the page, by whatever wrote it
        # — a prose row, a table cell, a flowchart caption — so the same text is
        # not written twice. Dealing prose without replacement was not enough on
        # its own: a page's pool holds no two lines with the same text (0 in 40
        # pages), but captions and table cells were picked from it without being
        # dealt, so a line could be a caption and then come round again as a
        # prose row, and the deck reshuffles once a page has used it up (a
        # caption repeated text already on its page 23% of the time, a table
        # cell 5%, even with every IAM line loaded).
        self.used: set = set()
        # Source lines already on the page, by their loader-given item id
        # (meta["item"]), whether a row dealt them or a run of words was taken out
        # of them (_labelled_run). Text alone does not catch the second: two runs
        # out of one line differ in text but are the same writing.
        self.used_items: set = set()

    def is_used(self, text: Optional[str]) -> bool:
        return bool(text) and text.strip() in self.used

    def mark(self, text: Optional[str]) -> None:
        if text:
            self.used.add(text.strip())

    def item_used(self, line: Ink) -> bool:
        item = (line.meta or {}).get("item")
        return bool(item) and item in self.used_items

    def mark_item(self, line: Ink) -> None:
        item = (line.meta or {}).get("item")
        if item:
            self.used_items.add(item)

    def take(self, n: int) -> List[Ink]:
        out = []
        for _ in range(n):
            if not self.pool:
                break
            if self._next >= len(self._order):
                random.shuffle(self._order)
                self._next = 0
            line = self.pool[self._order[self._next]]
            out.append(line)
            self.mark(line.label)
            self.mark_item(line)
            self._next += 1
        return out

    def one(self) -> Optional[Ink]:
        got = self.take(1)
        return got[0] if got else None

    def take_best(self, cost, candidates: int = 8) -> Optional[Ink]:
        """Take the cheapest of the next few undealt lines, keeping the rest.

        Auditioning lines by drawing and discarding them would burn the deck
        `candidates` times faster and put a page back to repeating sentences
        after a handful of rows. This peeks at the upcoming entries instead and
        only consumes the one it keeps; the losers stay where they were and
        come up again later.
        """
        if not self.pool:
            return None
        if self._next >= len(self._order):
            random.shuffle(self._order)
            self._next = 0

        def fresh(line: Ink) -> bool:
            return not self.is_used(line.label) and not self.item_used(line)

        window = min(candidates, len(self._order) - self._next)
        best_at, best_cost = None, None
        for offset in range(window):
            at = self._next + offset
            if not fresh(self.pool[self._order[at]]):
                continue        # that text, or that line, is already on the page
            value = cost(self.pool[self._order[at]])
            if value is None:
                continue
            if best_cost is None or value < best_cost:
                best_at, best_cost = at, value
                if value <= 0.0:
                    break       # a perfect fit; no point looking further

        if best_at is None:
            # Every line in the window is already on the page. Taking the next one
            # anyway is what wrote one well-labelled line as row after row once a
            # page had dealt through its writer's deck ("It would", "would be",
            # "It would" ... thirteen times on one page); look further down the
            # deck, and give up on the row rather than repeat a line.
            for at in range(self._next + window, len(self._order)):
                line = self.pool[self._order[at]]
                if fresh(line) and cost(line) is not None:
                    best_at = at
                    break
        if best_at is None:
            return None

        # swap the winner to the front of the undealt run so the rest survive
        self._order[self._next], self._order[best_at] = \
            self._order[best_at], self._order[self._next]
        line = self.pool[self._order[self._next]]
        self._next += 1
        self.mark(line.label)
        self.mark_item(line)
        return line


def _as_dealer(pool) -> LineDealer:
    """Accept either a plain pool list or an already-built dealer."""
    return pool if isinstance(pool, LineDealer) else LineDealer(pool)


# ----------------------------------------------------------------------------
# Cell filling — build content to fit a grid cell at a FIXED scale
# ----------------------------------------------------------------------------
# The page is a grid of cells (see layout.build_grid) and every cell is filled at
# the same page-wide handwriting scale, so all of these work in "text units":
# one x-height is LINE_HEIGHT, and the caller converts to page units afterwards
# with core.transform_ink. Nothing here is stretched to fit, because stretching
# is what put three different handwriting sizes on one page.

WORD_SPACE = 0.42       # gap between words, in x-heights
WORD_SPACE_JITTER = 0.3
MATH_ROW_GAP = 0.42     # gap between stacked expressions, in x-heights
INLINE_MATH_HEIGHT = 1.9  # cap on an expression sitting inline in prose, in x-heights
MATH_CELL_TEXT_PROB = 0.12   # chance a row of a maths section is prose instead


def _baseline_of(line: Ink) -> float:
    """The y a line of text sits on: the bottom of its x-height band."""
    band = _core_band(line)
    if band is not None:
        return band[1]
    return line.bbox()[3]


# Lines auditioned per row. High because take_best only *peeks* — the losers
# stay in the deck — so a wide window is free. Measured over 30 pages, going
# 8 -> 64 cuts mid-line truncation from 16.3% to 0.4% and tightens the ragged
# right edge (longest/shortest line p90) from 1.91x to 1.30x, with no repeats.
LINE_FIT_CANDIDATES = 64
LINE_FIT_TARGET = (0.80, 1.0)   # of the cell width, the band a line aims to land in
MIN_ROW_STROKES = 3             # a shorter row reads as a stray mark, so it is dropped
ROW_LABEL_ATTEMPTS = 5          # lines auditioned for one that fits without being cut


def _line_width_in_units(line: Ink) -> Optional[float]:
    """A line's width once normalised to the standard x-height."""
    band = _core_band(line)
    if band is None:
        return None
    x0, _, x1, _ = line.bbox()
    return (x1 - x0) * (LINE_HEIGHT / (band[1] - band[0]))


ROW_UNTRANSCRIBED_COST = 1e6    # ranks a line that would lose its text behind every other


def _words_labelled(line: Ink) -> bool:
    """Whether every word of the line carries its own transcription."""
    words = [child for child in line.children if child.class_name == "word"]
    return bool(words) and all(word.label for word in words)


def _pick_line_for_width(dealer: "LineDealer", local_w: float) -> Optional[Ink]:
    """A source line that fills the cell width without needing to be cut.

    Auditioning several lines and keeping the best fit does two jobs at once: it
    stops lines being truncated mid-sentence, and it evens out the right-hand
    edge. Taking whatever line came next instead gave paragraphs where one line
    ran the full width and the next was three words long, because a source
    line's width is fixed once the page's handwriting size is.
    """
    lo, hi = LINE_FIT_TARGET[0] * local_w, LINE_FIT_TARGET[1] * local_w

    def cost(line: Ink) -> Optional[float]:
        width = _line_width_in_units(line)
        if width is None:
            return None
        # A line that would reach the page with no transcription ranks after
        # every line that would not: one with no label at all, or one that has
        # to be cut and whose words don't carry their own labels (a cut keeps
        # the text only word by word). Width alone chose those freely, and a
        # table cell, narrow enough that nearly every line is cut, came back
        # without a transcription 28% of the time.
        untranscribed = not line.label or (width > hi and not _words_labelled(line))
        penalty = ROW_UNTRANSCRIBED_COST if untranscribed else 0.0
        if lo <= width <= hi:
            return penalty               # lands in the target band
        # too wide is worse than too narrow: it is the case that forces a cut
        return penalty + ((width - hi) * 3.0 if width > hi else (lo - width))

    return dealer.take_best(cost, LINE_FIT_CANDIDATES)


def _row_from_line(source: Ink, local_w: float, baseline_y: float) -> Optional[Ink]:
    """One source line, normalised to the standard x-height and cut to the cell.

    The words keep the relative positions the writer gave them, and the line is
    simply cut where it runs past the cell edge. Re-flowing the words onto an
    even spacing instead looked far worse: IAM's word boundaries are geometric
    guesses, so its "words" are often two or three letters, and laying those out
    on a uniform gap turned sentences into evenly-spaced letter soup
    ("Whitewallfarceisitwhen"). The writer's own spacing is the real thing, so
    the only edit made to it is where the line stops.
    """
    band = _core_band(source)
    if band is None:
        return None
    scale = LINE_HEIGHT / (band[1] - band[0])

    words = sorted((c for c in source.children if c.class_name == "word"),
                   key=lambda w: w.bbox()[0])
    if not words:
        words = [source]

    origin = words[0].bbox()[0]
    kept = []
    for word in words:
        if kept and (word.bbox()[2] - origin) * scale > local_w:
            break       # runs past the cell edge — the line ends here
        kept.append(word)

    dx, dy = -origin * scale, baseline_y - band[1] * scale
    placed = []
    for word in kept:
        out = transform_ink(word, scale, dx, dy)
        out.class_name = "word"
        placed.append(out)

    # Prefer the source line's own transcription. Building the label out of the
    # word labels instead loses it entirely on most IAM lines: word boundaries
    # there are geometric guesses, so per-word labels are only assigned when the
    # group count happens to match the token count (~40% of lines), and the rest
    # came back with no label at all — which showed up on the ground-truth page
    # as a row of empty boxes.
    if len(kept) == len(words) and source.label:
        label = source.label
    else:
        label = " ".join(w.label for w in placed if w.label) or None

    return Ink(strokes=[], children=placed, class_name="textline",
                source=source.source, writer=source.writer, label=label)


def _pick_row(dealer: "LineDealer", local_w: float, baseline_y: float) -> Optional[Ink]:
    """One line of prose for a row `local_w` wide, or None when no row can be
    written with its transcription.

    A cut line loses its transcription: the source line's own label only
    describes the whole line, and IAM's per-word labels are usually absent, so
    the row ends up with no ground truth at all. Audition a few lines for one
    that keeps its text — whole, or cut with every word labelled. If none does,
    the row is written the way a table cell is, from a run of individually
    labelled words (_labelled_run), and if even that fails it is left out.

    Settling for the first cut line instead left 134 of 905 prose rows (15%)
    with no transcription over 60 pages, in formula-heavy and prose-heavy blocks
    alike: every one a line cut at a narrow cell's edge whose words carried no
    labels of their own.
    """
    for _ in range(ROW_LABEL_ATTEMPTS):
        source = _pick_line_for_width(dealer, local_w)
        if source is None:
            break
        candidate = _row_from_line(source, local_w, baseline_y)
        # A row cut down to one short word is a stray mark, not prose; the
        # loaders apply the same floor to whole items (loaders.MIN_STROKES).
        if candidate is None or len(candidate.all_strokes()) < MIN_ROW_STROKES:
            continue
        if candidate.label:
            return candidate
    run = _labelled_run(dealer, local_w)
    if run is not None:
        _shift_ink(run, 0.0, baseline_y - LINE_HEIGHT)
    return run


def fill_text_cell(text_pool, local_w: float, local_h: float,
                    math_pool: Optional[List[Ink]] = None,
                    math_prob: float = MATH_LINE_PROB,
                    shape_bank=None, marks: bool = True) -> Ink:
    """A textblock filling a local_w x local_h cell at the standard scale.

    A fixed scale means a line's width is whatever its word count makes it, so
    the cell is filled by choosing how many lines fit down it and cutting each
    line at its right-hand edge — never by scaling the writing, which is what
    put several handwriting sizes on one page.

    `shape_bank` is the mined ink underlines and strikethroughs are drawn with;
    without it the block has none. `marks=False` leaves them off altogether, for
    a caller that takes single rows out of the block and marks them itself once
    they sit among their real neighbours (fill_math_cell).
    """
    dealer = _as_dealer(text_pool)
    advance = LINE_HEIGHT + LINE_GAP
    n_lines = max(1, int((local_h + LINE_GAP) // advance))
    last_baseline = (n_lines - 1) * advance + LINE_HEIGHT
    pushed = 0.0        # how far rows have gone down to keep lines apart

    lines: List[Ink] = []

    def settle(ink: Ink, baseline_y: float) -> bool:
        """Push a row down off the lines above it (textlines and formulas alike) until it
        shares no more than LINE_MAX_OVERLAP of the smaller of any two, taking every row
        after it down too; False when that would take it past the cell's last row."""
        nonlocal pushed
        dy = _clear_push(ink.bbox(), [line.bbox() for line in lines if line.class_name in LINE_CLASSES])
        if dy > 0.0:
            if baseline_y + dy > last_baseline + _OVERLAP_EPS:
                return False
            _shift_ink(ink, 0.0, dy)
            pushed += dy
        return True

    for row in range(n_lines):
        baseline_y = row * advance + LINE_HEIGHT + pushed
        if baseline_y > last_baseline + _OVERLAP_EPS:
            break       # rows pushed down have used up the cell; the rest are left out, not squeezed

        if math_pool and random.random() < math_prob:
            # An expression sitting inline in a paragraph has to live inside its
            # row: at full maths height it would reach up over the line above.
            expression = _fit_expression(random.choice(math_pool), local_w,
                                          max_height=INLINE_MATH_HEIGHT * LINE_HEIGHT)
            if expression is not None:
                # _fit_expression copies the ink and drops its class, so it has to
                # be stated here — unset, an inline expression in a paragraph or a
                # list was unlabelled ink rather than a `formula`
                expression.class_name = "formula"
                ex0, ey0, _, ey1 = expression.bbox()
                centre = baseline_y - LINE_HEIGHT / 2.0
                _shift_ink(expression, 0.0, centre - (ey0 + ey1) / 2.0)
                if not settle(expression, baseline_y):
                    break
                lines.append(expression)
                continue

        row_ink = _pick_row(dealer, local_w, baseline_y)
        if row_ink is None:
            break
        # an ascender or descender may reach into the line above, but not past LINE_MAX_OVERLAP
        if not settle(row_ink, baseline_y):
            break
        lines.append(row_ink)

    if not lines:
        return Ink(strokes=[], class_name="textblock")

    if not marks:
        return Ink(strokes=[], children=lines, class_name="textblock")
    # An underline or strikethrough is a line_mark: it must not enlarge the
    # line's own box or change what the line holds, so each is a sibling of the
    # rows rather than a child of one.
    return Ink(strokes=[], children=lines + _mark_rows(lines, shape_bank),
               class_name="textblock")


BULLET_GUTTER = 2.2     # space reserved left of a list item, in x-heights


def fill_textblock_cell(text_pool, math_pool: Optional[List[Ink]],
                         local_w: float, local_h: float, formula_heavy: bool,
                         bulleted: bool = False,
                         minority_prob: float = MATH_CELL_TEXT_PROB,
                         shape_bank=None) -> Ink:
    """A textblock filling a cell: rows of prose and formulas, one kind the majority.

    Mostly formulas is a maths section (fill_math_cell) with the odd line of
    prose; mostly prose is a paragraph (fill_text_cell) with the odd expression
    inline. `minority_prob` is the chance any one row is the other kind.

    A bulleted block is the same rows with a marker in front of each. It is still
    labelled `textblock`, not a class of its own: a list is a block of rows with
    bullets in front, and the bullets carry no class.
    """
    gutter = LINE_HEIGHT * BULLET_GUTTER if bulleted else 0.0
    width = max(local_w - gutter, LINE_HEIGHT)
    if formula_heavy:
        body = fill_math_cell(math_pool, text_pool, width, local_h,
                              text_prob=minority_prob, shape_bank=shape_bank)
    else:
        body = fill_text_cell(text_pool, width, local_h, math_pool=math_pool,
                              math_prob=minority_prob, shape_bank=shape_bank)
    if not bulleted or not body.children:
        return body

    _shift_ink(body, gutter, 0.0)

    # A bullet is unlabelled ink beside its row, not part of it: it stays out of
    # the row's box and out of its transcription. Each marker is its own child of
    # the block rather than one lump. Only rows get one — underlines are children of the block too,
    # and bulleting every child put a bullet in front of each underline.
    bullets = []
    for row in body.children:
        if row.class_name not in ("textline", "formula"):
            continue
        y0, y1 = row.bbox()[1], row.bbox()[3]
        bullets.append(Ink(strokes=draw_bullet_stroke(gutter * 0.45, (y0 + y1) / 2,
                                                      r=LINE_HEIGHT * 0.18),
                           class_name=None, source="procedural"))
    return Ink(strokes=[], children=list(body.children) + bullets, class_name="textblock")


def _shift_ink(ink: Ink, dx: float, dy: float) -> None:
    """Translate an Ink tree in place."""
    def walk(node: Ink):
        for s in node.strokes:
            s.x = [v + dx for v in s.x]
            s.y = [v + dy for v in s.y]
        for c in node.children:
            walk(c)
    walk(ink)


def _fit_expression(expression: Ink, local_w: float,
                     max_height: Optional[float] = None,
                     min_ratio: Optional[float] = None) -> Optional[Ink]:
    """One MathWriting expression at the standard scale, shrunk if it won't fit.

    With `min_ratio`, an expression that would have to shrink below that share of
    its standard size is refused (None), and the share it kept is left in
    meta["text_ratio"] for the caller to take."""
    band = _core_band(expression)
    x0, y0, x1, y1 = expression.bbox()
    width, height = x1 - x0, y1 - y0
    if width <= 0 or height <= 0:
        return None

    ceiling = max_height if max_height is not None else MATH_MAX_HEIGHT * LINE_HEIGHT
    if band is not None:
        scale = MATH_CORE_MULT * LINE_HEIGHT / (band[1] - band[0])
    else:
        scale = MATH_CORE_MULT * LINE_HEIGHT / height
    standard = scale
    if height * scale > ceiling:
        scale = ceiling / height

    if width * scale > local_w:      # an expression may not run past the cell
        scale = local_w / width
    if min_ratio is not None and scale < min_ratio * standard:
        return None

    out = transform_ink(expression, scale, -x0 * scale, -y0 * scale)
    out.class_name = None
    out.source = "mathwriting"
    if min_ratio is not None:
        out.meta = dict(out.meta or {}, text_ratio=scale / standard)
    return out


def fill_math_cell(math_pool: List[Ink], text_pool, local_w: float, local_h: float,
                    text_prob: float = MATH_CELL_TEXT_PROB,
                    shape_bank=None) -> Ink:
    """A section given over to maths: expressions stacked down the whole cell.

    Labelled as a textblock whose rows are textlines. The taxonomy has no maths
    class, and leaving a whole section of the page unlabelled would train the
    model to see nothing there, which is worse than calling a row of handwritten
    symbols a line of handwriting. Expressions get no word children, so they
    contribute level-2 and level-1 boxes but nothing at level 0.
    """
    dealer = _as_dealer(text_pool) if text_pool else None
    rows: List[Ink] = []
    y = 0.0

    while y < local_h:
        row: Optional[Ink] = None

        if dealer is not None and random.random() < text_prob:
            # Take the line out of the container fill_text_cell wraps it in.
            # Nesting the container would put a textblock inside this cell's
            # textblock: prose and maths in one section are one block of several
            # lines, not a block within a block.
            filled = fill_text_cell(dealer, local_w, LINE_HEIGHT + LINE_GAP,
                                    shape_bank=shape_bank, marks=False)
            row = filled.children[0] if filled.children else None

        if row is None:
            if not math_pool:
                break
            row = _fit_expression(random.choice(math_pool), local_w)
            if row is None:
                continue
            row.class_name = "formula"

        x0, y0, x1, y1 = row.bbox()
        height = y1 - y0
        if y + height > local_h:
            if rows:
                break
            # nothing placed yet and it already overflows: shrink this one row
            # rather than hand back a block taller than its cell
            squeeze = local_h / height
            row = transform_ink(row, squeeze, -x0 * squeeze, -y0 * squeeze)
            # the squeeze copies the ink, so the class has to be restated; it is
            # whatever the row already was, prose or maths
            row.class_name = "formula" if row.source == "mathwriting" else "textline"
            x0, y0, x1, y1 = row.bbox()
            height = y1 - y0
        _shift_ink(row, random.uniform(0, max(0.0, local_w - (x1 - x0)) * 0.15) - x0,
                   y - y0)
        rows.append(row)
        y += height + LINE_HEIGHT * MATH_ROW_GAP

    if not rows:
        return Ink(strokes=[], class_name="textblock")
    # underlines and strikethroughs go on once the rows are stacked, so each is
    # checked against the rows really above and below it
    return Ink(strokes=[], children=rows + _mark_rows(rows, shape_bank),
               class_name="textblock")


def make_textblock(text_pool: List[Ink], n_lines: Optional[int] = None,
                    math_pool: Optional[List[Ink]] = None,
                    math_prob: float = MATH_LINE_PROB) -> Ink:
    """text_pool: list of line-level Ink objects (class_name should be 'textline',
    ideally with word children) pulled from IAMONDB/DeepWriting loaders.

    math_pool: optional MathWriting expressions. A line is occasionally replaced
    by one, the way a real page of notes has an equation sitting inline. It is
    labelled `formula` rather than `textline`: it is a row of maths wherever it
    turns up, and calling it one thing in a maths section and another in a
    paragraph would make the class mean nothing."""
    n_lines = n_lines or random.randint(2, 5)
    dealer = _as_dealer(text_pool)
    lines = []
    for _ in range(n_lines):
        if math_pool and random.random() < math_prob:
            expression = dict_deepcopy(random.choice(math_pool))
            expression.class_name = "formula"
            lines.append(expression)
            continue
        source_line = dealer.one()
        if source_line is None:
            continue
        line = dict_deepcopy(source_line)
        line.class_name = "textline"
        lines.append(line)
    if not lines:
        return Ink(strokes=[], class_name="textblock")
    container = _stack_lines_vertically(lines)
    container.class_name = "textblock"
    return container


def make_list(text_pool: List[Ink], n_items: Optional[int] = None) -> Ink:
    n_items = n_items or random.randint(2, 4)
    items = [dict_deepcopy(line) for line in _as_dealer(text_pool).take(n_items)]
    if not items:
        return Ink(strokes=[], class_name="textblock")
    for it in items:
        it.class_name = "textline"
    container = _stack_lines_vertically(items, x0=8.0)  # leave room for bullets
    # Bullet marks are visual only — no taxonomy class — but each is its own
    # child Ink rather than one lump on the container, so that page time
    # stitching sees it as a separate mark sitting just left of its line and
    # writes it immediately before that line, instead of drawing every bullet
    # in the list first.
    # beside each item where it ended up: a row pushed down off the one above is off the grid
    for item in list(container.children):
        _, iy0, _, iy1 = item.bbox()
        container.children.append(
            Ink(strokes=draw_bullet_stroke(3.0, (iy0 + iy1) / 2),
                class_name=None, source="procedural"))
    container.class_name = "textblock"   # a list is a textblock; see fill_textblock_cell
    return container


def table_grid_exact(x0, y0, x1, y1, n_rows, n_cols,
                     col_fracs: Optional[List[float]] = None,
                     row_fracs: Optional[List[float]] = None) -> List[dict]:
    """The table's full grid as exact segments, whatever its ruling style.

    Rows and columns are evenly divided unless `col_fracs` / `row_fracs` give
    where each inner divider falls, as a fraction of the table's width / height
    — make_table sizes its columns to what is written in them, so they are not.

    A table's structure is its grid, so the ground truth shows every divider
    even when the ink rules none of them: an unruled table still has rows and
    columns, and a label page that omitted them left the cells floating with
    nothing to say which cell belonged to which row. Stored under meta["exact"]
    so core's transforms carry it onto the page with the strokes (see
    core._move_exact_meta).
    """
    segments = [{"kind": "segment", "a": [x0, y0], "b": [x1, y0]},
                {"kind": "segment", "a": [x0, y1], "b": [x1, y1]},
                {"kind": "segment", "a": [x0, y0], "b": [x0, y1]},
                {"kind": "segment", "a": [x1, y0], "b": [x1, y1]}]
    row_fracs = row_fracs or [r / n_rows for r in range(1, n_rows)]
    col_fracs = col_fracs or [c / n_cols for c in range(1, n_cols)]
    for fraction in row_fracs:
        y = y0 + (y1 - y0) * fraction
        segments.append({"kind": "segment", "a": [x0, y], "b": [x1, y]})
    for fraction in col_fracs:
        x = x0 + (x1 - x0) * fraction
        segments.append({"kind": "segment", "a": [x, y0], "b": [x, y1]})
    return segments


TABLE_CELL_W = 7.0          # widest a cell's writing may run, in x-heights
TABLE_CELL_PAD = 0.8        # clearance between a cell's writing and its rules, in x-heights
TABLE_FORMULA_PROB = 0.30   # chance a cell holds an expression rather than prose
# Every column divider is always drawn — a vertical rule is what makes the
# columns — and the only choices are the horizontal ones: every row separated
# ("full"), or only the header row ("header_row"). The ratio is the one the two
# had before the rows-only and columns-only styles were dropped.
TABLE_RULE_STYLES = {"full": 0.79, "header_row": 0.21}
TABLE_BORDER_PROB = 0.75    # chance a table is also ruled round its outside


def _grid_segments(col_edges: List[float], row_edges: List[float], style: str,
                   border: bool = True) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """The rules a table draws, as (start, end) pairs: the outer border first
    when there is one, then the inner dividers.

    Every column divider is always drawn: the vertical rules are what make the
    columns, and a table without them is writing spaced out in a block that
    could not be told from a textblock. The horizontal dividers are the style's
    choice — every row ("full"), or only the one under the header row
    ("header_row") — and the border is chosen separately. Each rule is its own
    straight run, the way a geometry figure's sides are, so each can be drawn
    from its own mined line.
    """
    x0, x1 = col_edges[0], col_edges[-1]
    y0, y1 = row_edges[0], row_edges[-1]
    segments = []
    if border:
        segments += [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                     ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
    rows = [((x0, y), (x1, y)) for y in row_edges[1:-1]]
    columns = [((x, y0), (x, y1)) for x in col_edges[1:-1]]
    segments += columns
    segments += rows if style == "full" else rows[:1]
    return segments


TABLE_RUN_TRIES = 30        # random word runs tried before a cell falls back to _pick_row
TABLE_FORMULA_TRIES = 6     # expressions tried for a formula cell before it is written as prose


def _labelled_run(dealer: "LineDealer", width: float,
                  rng: random.Random = random, accept=None) -> Optional[Ink]:
    """A textline of consecutive words that each carry their own transcription,
    no wider than `width`, at the standard scale with its baseline at LINE_HEIGHT.
    `accept`, when given, may refuse a run: another is tried, and a refused run is
    not marked as written on the page.

    A table cell is narrow, so a whole source line almost never fits one and has
    to be cut — and a cut keeps its text only word by word. Choosing lines for
    width alone left 28% of cells with no transcription, and preferring
    word-labelled lines still left 19%: a page's writer has only a few of those,
    and dealing them out used them up. So a cell is written from a run of words
    out of any fully word-labelled line in the writer's pool, without dealing it
    — as many consecutive words, from a random start, as fit.
    """
    sources = [line for line in dealer.pool
               if _words_labelled(line) and _core_band(line) is not None]
    if not sources:
        return None
    # Only lines not yet on the page. Runs are taken without dealing, so one
    # well-labelled line otherwise filled cell after cell of a page — the same
    # writing repeated, however different each run's text.
    sources = [line for line in sources if not dealer.item_used(line)]
    if not sources:
        return None
    for _ in range(TABLE_RUN_TRIES):
        source = rng.choice(sources)
        band = _core_band(source)
        scale = LINE_HEIGHT / (band[1] - band[0])
        words = sorted((c for c in source.children if c.class_name == "word"),
                       key=lambda w: w.bbox()[0])
        start = rng.randrange(len(words))
        origin = words[start].bbox()[0]
        kept = []
        for word in words[start:]:
            if (word.bbox()[2] - origin) * scale > width:
                break
            kept.append(word)
        if not kept or sum(len(w.all_strokes()) for w in kept) < MIN_ROW_STROKES:
            continue                # nothing fits, or only a stray mark does
        label = " ".join(word.label for word in kept)
        if dealer.is_used(label):
            continue                # that text is already on the page
        dx, dy = -origin * scale, LINE_HEIGHT - band[1] * scale
        placed = []
        for word in kept:
            out = transform_ink(word, scale, dx, dy)
            out.class_name = "word"
            placed.append(out)
        run = Ink(strokes=[], children=placed, class_name="textline",
                  source=source.source, writer=source.writer, label=label)
        if accept is not None and not accept(run):
            continue
        dealer.mark(label)
        dealer.mark_item(source)
        return run
    return None


def _table_cell(dealer: "LineDealer", math_pool: Optional[List[Ink]],
                width: float) -> Optional[Ink]:
    """What one cell says — a formula or a line of prose — with its box at the origin.

    Written exactly as a row of a textblock is, at the page's standard scale, so
    `textline` and `formula` mean the same inside a table as anywhere else. Prose
    is cut to the width, never scaled; a formula scaled to fit may keep no less
    than MIN_TEXT_RATIO of its size, and the share it kept is in meta["text_ratio"]
    (make_table takes it off).
    """
    content = None
    if math_pool and random.random() < TABLE_FORMULA_PROB:
        for _ in range(TABLE_FORMULA_TRIES):
            content = _fit_expression(random.choice(math_pool), width,
                                      max_height=INLINE_MATH_HEIGHT * LINE_HEIGHT,
                                      min_ratio=MIN_TEXT_RATIO)
            if content is not None:
                content.class_name = "formula"
                break
    if content is None:
        content = _labelled_run(dealer, width) or _pick_row(dealer, width, LINE_HEIGHT)
    if content is None:
        return None
    x0, y0, _, _ = content.bbox()
    _shift_ink(content, -x0, -y0)
    return content


def make_table(text_pool, shape_bank, math_pool: Optional[List[Ink]] = None,
               rng: random.Random = random, n_rows: Optional[int] = None,
               n_cols: Optional[int] = None) -> Optional[Ink]:
    """A table whose cells each hold a textline or a formula, ruled in mined ink.

    The grid is sized to what was written in it — each column as wide as its
    widest cell, each row as tall as its tallest — rather than the writing being
    squeezed into a fixed grid. Its rules are drawn the way a geometry figure's
    sides are, one mined line each (geometry_diagrams.segment), so they carry a
    real hand's wobble.

    Every cell is written in. A blank cell is indistinguishable from a cell
    whose writing failed to place, so the ground truth could not say which it
    was looking at; if any cell comes up empty there is no table at all.

    Every table has all its column dividers, and either every row divider or
    only the header row's (TABLE_RULE_STYLES). Whether it also has an
    outer border is separate (TABLE_BORDER_PROB). A border is not labelled: its
    four strokes are the table's own unlabelled ink, like its inner rules.

    Returns None without a bank to rule with, so the caller writes prose instead.
    """
    import itertools
    import geometry_diagrams as gd
    if shape_bank is None or not shape_bank.ok():
        return None
    n_rows = n_rows or random.randint(2, 4)
    n_cols = n_cols or random.randint(2, 3)
    style = random.choices(list(TABLE_RULE_STYLES),
                           weights=list(TABLE_RULE_STYLES.values()))[0]
    bordered = random.random() < TABLE_BORDER_PROB

    dealer = _as_dealer(text_pool)
    pad = TABLE_CELL_PAD * LINE_HEIGHT
    cells = [[_table_cell(dealer, math_pool, TABLE_CELL_W * LINE_HEIGHT)
              for _ in range(n_cols)] for _ in range(n_rows)]
    if any(cell is None for row in cells for cell in row):
        return None
    # the smallest share of its standard size any cell's writing kept, which the
    # caller multiplies by the table's own fit to the page (MIN_TEXT_RATIO)
    text_ratio = min((cell.meta or {}).pop("text_ratio", 1.0) for row in cells for cell in row)

    def size(ink: Ink) -> Tuple[float, float]:
        x0, y0, x1, y1 = ink.bbox()
        return x1 - x0, y1 - y0

    col_w = [max(size(cells[r][c])[0] for r in range(n_rows)) + 2 * pad
             for c in range(n_cols)]
    row_h = [max(size(cells[r][c])[1] for c in range(n_cols)) + 2 * pad
             for r in range(n_rows)]
    col_edges = [0.0] + list(itertools.accumulate(col_w))
    row_edges = [0.0] + list(itertools.accumulate(row_h))

    children = []
    for r in range(n_rows):
        for c in range(n_cols):
            content = cells[r][c]
            _, height = size(content)
            # left-aligned in its column, centred down its row
            _shift_ink(content, col_edges[c] + pad,
                       row_edges[r] + (row_h[r] - height) / 2.0)
            # a cell's line sits in the table itself, not in a textblock of its own
            content.meta = dict(content.meta or {}, row=r, col=c)
            children.append(content)

    segments = _grid_segments(col_edges, row_edges, style, bordered)
    if bordered:
        # the border is a box drawn round the table, joined as loosely as any
        # box round writing is; the inner rules still run to the true border
        segments[:4] = gd.loose_sides([a for a, _ in segments[:4]], rng, gd.ENCLOSURE_JOIN)
    rules = [gd.segment(shape_bank, rng, a, b) for a, b in segments]
    # the exact grid is every divider whatever the style; see table_grid_exact
    col_fracs = [x / col_edges[-1] for x in col_edges[1:-1]]
    row_fracs = [y / row_edges[-1] for y in row_edges[1:-1]]
    exact = table_grid_exact(0.0, 0.0, col_edges[-1], row_edges[-1], n_rows, n_cols,
                             col_fracs=col_fracs, row_fracs=row_fracs)
    table = Ink(strokes=rules, class_name="table", children=children,
                meta={"n_rows": n_rows, "n_cols": n_cols, "style": style,
                      "bordered": bordered,
                      "col_fracs": col_fracs, "row_fracs": row_fracs,
                      "exact": exact, "text_ratio": text_ratio})
    return table


def make_diagram_procedural(text_pool: Optional[List[Ink]] = None) -> Ink:
    """Fallback diagram builder for when no OHFD sample is used: a couple of
    boxes/ovals connected by arrows, optionally with a text label inside."""
    n_nodes = random.randint(2, 4)
    node_w, node_h = 30.0, 16.0
    gap = 20.0
    children = []
    centers = []
    for i in range(n_nodes):
        x0 = i * (node_w + gap)
        y0 = 0.0
        x1, y1 = x0 + node_w, y0 + node_h
        # Boxes only. Procedurally drawn ovals looked obviously synthetic next to
        # real ink, so the `oval` class is now sourced solely from OHFD's real
        # hand-drawn terminators (see OHFD_CLASS_MAP in loaders.py).
        node = Ink(strokes=[draw_rect_stroke(x0, y0, x1, y1)], class_name="enclosure",
                   label="rectangle")
        if text_pool and random.random() < 0.7:
            from core import fit_ink_to_box
            label = fit_ink_to_box(random.choice(_as_dealer(text_pool).pool),
                                    (x0 + 3, y0 + 3, x1 - 3, y1 - 3))
            label.class_name = "textline"
            node.children.append(label)
        children.append(node)
        centers.append(((x0 + x1) / 2, (y0 + y1) / 2, x1, (y0 + y1) / 2))

    arrows = []
    for i in range(len(centers) - 1):
        _, _, x_from, y_from = centers[i]
        x_to = centers[i + 1][0] - node_w / 2
        y_to = centers[i + 1][1]
        arrows.append(Ink(strokes=[draw_arrow_stroke(x_from, y_from, x_to, y_to)], class_name="arrow"))

    return Ink(strokes=[], children=children + arrows, class_name="flowchart",
               meta={"flowchart": True})


# ----------------------------------------------------------------------------
# What goes inside a flowchart shape
# ----------------------------------------------------------------------------
# OHFD's own node captions are thrown away and rewritten. The reason is a
# labelling one. A caption like "i = i+1" or "x > y" is a formula written by
# hand, and "check next value" is a line of prose, and nothing in the strokes
# separates the two reliably — so the same marks would end up labelled
# `textline` in one chart and `formula` in another with no way to tell which was
# right. Replacing the caption with ink whose kind we chose means the label is
# known rather than guessed.
#
# It is also closer to the truth about flowcharts. Real ones hold far more than
# OHFD's short phrases, and drawing from the full text and maths pools gives a
# range of contents the 419 charts do not contain between them.
#
# Only shapes that already had a caption get one: a chart whose nodes are blank
# stays blank, which keeps each chart's own balance of labelled and unlabelled
# nodes rather than inventing captions the source never had.
SHAPE_FORMULA_PROB = 0.30

# The room inside a shape is measured against its own outline rather than
# derived from its class. A parallelogram's slant, a diamond's corners and an
# ellipse's curve each eat into the bounding box by a different amount, and the
# shapes are drawn by hand, so no per-class fraction is right for all of them —
# it has to assume the worst chart in the set and shrink every other one to
# match. Shrinking against the ink gets each shape its own answer.
SHAPE_SHRINK_STEPS = tuple(round(0.92 - 0.04 * i, 3) for i in range(16))
# Kept clear inside the outline, as a fraction of the shape's smaller side. The
# outline is a sampled polyline, so a box that merely misses every sample can
# still touch the stroke drawn between two of them.
SHAPE_MARGIN = 0.04

# Candidates drawn before picking the one whose proportions suit the space.
SHAPE_CONTENT_TRIES = 14

# The tallest a rewritten caption may be, in multiples of the chart's own
# (median) caption height. Fitting to the room inside the shape alone blew a
# short word or a one-line formula up to fill the whole shape: that room runs to
# 1.8x the chart's handwriting at p90 and 2.8x at p99, so a chart's captions came
# out in several sizes, some towering over everything else on the page. A
# formula is allowed more for its fractions and superscripts.
SHAPE_CAPTION_MAX_HEIGHT = {"textline": 1.0, "formula": 1.5}

# A caption's box height against its x-height band, for a run of a few words: a run
# is as tall as its tallest ascender and deepest descender, which over IAM's lines is
# 2.0 bands at the median (1.25 at p10, 2.95 at p90). A caption too long to keep the
# page's writing size in its shape is trimmed to fewer words rather than shrunk, so
# what limits a chart is its shapes' height, not its captions' length: a chart is
# chosen only if its shapes have room for a caption this many bands of the page's
# smallest allowed writing (make_diagram), and a caption's box is never cut below it,
# however small the chart's own handwriting was (shape_content).
CAPTION_RUN_BOX = 2.0

# Flowchart node classes something can be written inside.
SHAPE_CLASSES = ("enclosure",)


def _interior_box(shape: Ink) -> Optional[Tuple[float, float, float, float]]:
    """The largest centred box the shape's own outline does not cross.

    Found by shrinking rather than by formula: a box of the shape's proportions
    is tried at each step down from nearly the full bounding box, and the first
    one no outline point falls inside is the room there is to write in. That
    lands on about half-size for a diamond and most of the box for a rectangle
    without either being told, and it follows the slant of whichever
    parallelogram this particular chart happens to have drawn.
    """
    x0, y0, x1, y1 = shape.bbox()
    width, height = x1 - x0, y1 - y0
    if width <= 0 or height <= 0:
        return None

    outline = [(px, py) for stroke in shape.strokes
               for px, py in zip(stroke.x, stroke.y)]
    if not outline:
        return (x0, y0, x1, y1)

    mid_x, mid_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    margin = SHAPE_MARGIN * min(width, height)
    for step in SHAPE_SHRINK_STEPS:
        half_w, half_h = width * step / 2.0, height * step / 2.0
        lo_x, hi_x = mid_x - half_w - margin, mid_x + half_w + margin
        lo_y, hi_y = mid_y - half_h - margin, mid_y + half_h + margin
        if not any(lo_x <= px <= hi_x and lo_y <= py <= hi_y
                   for px, py in outline):
            return (mid_x - half_w, mid_y - half_h,
                    mid_x + half_w, mid_y + half_h)
    return None         # nothing legible fits; the shape is left empty


def _best_shaped(candidates: List[Ink], box: Tuple[float, float, float, float]
                  ) -> Optional[Ink]:
    """Whichever candidate's proportions best match the space available.

    fit_ink_to_box scales uniformly, so a candidate shaped unlike its box is not
    clipped — it is shrunk until its long side fits, and leaves the rest of the
    space empty. A wide line of prose dropped into a tall narrow box comes out
    unreadably small. Choosing on aspect ratio is what keeps the writing as
    large as the shape allows. The comparison is on the log of the ratio so that
    being twice too wide and half too wide cost the same.
    """
    x0, y0, x1, y1 = box
    wanted = (x1 - x0) / max(y1 - y0, 1e-6)
    best, best_cost = None, None
    for candidate in candidates:
        cx0, cy0, cx1, cy1 = candidate.bbox()
        width, height = cx1 - cx0, cy1 - cy0
        if width <= 0 or height <= 0:
            continue
        cost = abs(math.log((width / height) / wanted))
        if best_cost is None or cost < best_cost:
            best, best_cost = candidate, cost
    return best


# Up to this many items, shape_content filters the whole pool for text already on
# the page before picking; above it, it draws picks and skips used ones instead.
SHAPE_CONTENT_SCAN = 5000


def _unused_picks(pool, used: "LineDealer", k: int, rng: random.Random) -> List[Ink]:
    """Up to `k` distinct items of `pool` whose text is not on the page yet.

    Filtering the whole pool costs a pass over it per caption, about 0.3 s against
    the full MathWriting release; drawing random items and skipping used ones is
    the same uniform draw without replacement from the unused items, in O(k). A
    small pool — one writer's few dozen lines — is filtered outright as before,
    since there the used share can be large enough for misses to add up.
    """
    def fresh(ink: Ink) -> bool:
        # neither its text nor its source line is on the page yet: a line a table
        # cell took a run of words from would otherwise come back whole as a caption
        return not used.is_used(ink.label) and not used.item_used(ink)

    if len(pool) <= SHAPE_CONTENT_SCAN:
        unused = [ink for ink in pool if fresh(ink)] or pool
        return rng.sample(unused, min(k, len(unused)))
    picks, tried = [], set()
    for _ in range(k * 20):
        if len(picks) >= k:
            break
        index = rng.randrange(len(pool))
        if index in tried:
            continue
        tried.add(index)
        if fresh(pool[index]):
            picks.append(pool[index])
    return picks or rng.sample(pool, min(k, len(pool)))


def shape_content(box: Tuple[float, float, float, float], text_pool, math_pool,
                   rng: random.Random = random,
                   used: Optional["LineDealer"] = None,
                   caption_height: Optional[float] = None,
                   min_band: Optional[float] = None) -> Optional[Ink]:
    """One line of writing or one formula, sized to sit inside `box`.

    Containment is by construction rather than by checking: fit_ink_to_box
    scales uniformly and centres, so the result's bounding box is inside the one
    given, whatever shape the ink was. With `caption_height` (the chart's own
    caption height) the box is first cut down to SHAPE_CAPTION_MAX_HEIGHT of it,
    about its centre, so the writing is never scaled past the chart's handwriting.
    With `min_band` (chart units), a candidate whose x-height band would come out
    shorter than that once fitted is not used (a formula's, than MATH_CORE_MULT of
    it). Writing too long to keep that size is trimmed rather than shrunk: when no
    whole line (or, for a formula shape, no expression) keeps it, the shape gets a
    run of consecutive words cut to its width (_trimmed_caption), and for that the
    box is never cut below CAPTION_RUN_BOX bands of `min_band`. Only if no run keeps
    its size either is the shape left blank.
    """
    from core import fit_ink_to_box

    want_formula = bool(math_pool) and rng.random() < SHAPE_FORMULA_PROB
    pool = math_pool if want_formula else (text_pool or math_pool)
    if not pool:
        return None
    if caption_height:
        tallest = caption_height * SHAPE_CAPTION_MAX_HEIGHT["formula" if want_formula else "textline"]
        if min_band is not None:
            # room for writing at the page's size, however small the chart's own captions were
            tallest = max(tallest, min_band * CAPTION_RUN_BOX)
        x0, y0, x1, y1 = box
        if y1 - y0 > tallest:
            mid_y = (y0 + y1) / 2.0
            box = (x0, mid_y - tallest / 2.0, x1, mid_y + tallest / 2.0)
    # Nothing already written on the page (`used` is the page's dealer), and the
    # candidates drawn without replacement. Choosing the best-shaped of a few
    # picks from one writer's few dozen lines kept landing on the same line, and
    # a flowchart repeated its own captions, or the page's prose, 23% of the time.
    if used is not None:
        picks = _unused_picks(pool, used, SHAPE_CONTENT_TRIES, rng)
    else:
        picks = rng.sample(pool, min(SHAPE_CONTENT_TRIES, len(pool)))
    if min_band is not None:
        need = min_band * (MATH_CORE_MULT if want_formula else 1.0)
        bx0, by0, bx1, by1 = box

        def legible(candidate: Ink) -> bool:
            band = _core_band(candidate)
            cx0, cy0, cx1, cy1 = candidate.bbox()
            if band is None or cx1 <= cx0 or cy1 <= cy0:
                return False        # its size cannot be told, so it is not risked
            fit = min((bx1 - bx0) / (cx1 - cx0), (by1 - by0) / (cy1 - cy0))
            return (band[1] - band[0]) * fit >= need

        picks = [candidate for candidate in picks if legible(candidate)]
    chosen = _best_shaped(picks, box)
    if chosen is None:
        if min_band is not None and text_pool:
            return _trimmed_caption(box, text_pool, used, min_band, rng)
        return None
    if used is not None:
        used.mark(chosen.label)
        used.mark_item(chosen)

    out = fit_ink_to_box(chosen, box)
    out.class_name = "formula" if want_formula else "textline"
    out.label = chosen.label
    out.source = chosen.source
    return out


def _trimmed_caption(box: Tuple[float, float, float, float], text_pool,
                     used: Optional["LineDealer"], min_band: float,
                     rng: random.Random = random) -> Optional[Ink]:
    """A caption of as many consecutive words as fit `box` across at `min_band`, or None.

    The run (_labelled_run) is written as large as the box allows, up to the page's
    own handwriting (`min_band` / FLOWCHART_MIN_TEXT_RATIO), centred; a run whose x-height
    band would still come out below `min_band` — too tall for the box, or one word
    too wide on its own — is refused and another tried, and so is a run with no
    letter or digit in it: a narrow shape otherwise took IAM's "." or "," on its own,
    a stray mark for a caption. Tagged meta["trimmed"].
    """
    from core import fit_ink_to_box

    x0, y0, x1, y1 = box
    largest = min_band / FLOWCHART_MIN_TEXT_RATIO
    dealer = used if used is not None else _as_dealer(text_pool)
    sized = {}

    def accept(run: Ink) -> bool:
        if not any(ch.isalnum() for ch in run.label or ""):
            return False
        band = _core_band(run)
        rx0, ry0, rx1, ry1 = run.bbox()
        if band is None or rx1 <= rx0 or ry1 <= ry0:
            return False
        height = band[1] - band[0]
        scale = min(largest / height, (x1 - x0) / (rx1 - rx0), (y1 - y0) / (ry1 - ry0))
        sized["scale"] = scale
        return height * scale >= min_band

    run = _labelled_run(dealer, (x1 - x0) * LINE_HEIGHT / min_band, rng, accept=accept)
    if run is None:
        return None
    rx0, ry0, rx1, ry1 = run.bbox()
    half_w, half_h = (rx1 - rx0) * sized["scale"] / 2.0, (ry1 - ry0) * sized["scale"] / 2.0
    mid_x, mid_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    out = fit_ink_to_box(run, (mid_x - half_w, mid_y - half_h, mid_x + half_w, mid_y + half_h))
    out.class_name = "textline"
    out.label, out.source, out.writer = run.label, run.source, run.writer
    out.meta = dict(out.meta or {}, trimmed=True)
    return out


def restock_diagram_text(diagram: Ink, text_pool, math_pool,
                          rng: random.Random = random,
                          min_band: Optional[float] = None) -> None:
    """Swap every caption written inside a shape for ink of a known kind.

    In place, and only inside the shapes: a caption sitting on an *arrow* is
    left exactly as OHFD wrote it.
    """
    graph = (diagram.meta or {}).get("graph") or {}
    nodes = graph.get("nodes") or {}
    # callers hand this in as a LineDealer as often as a list, and a caption is
    # picked at random rather than dealt — a shape's contents have no continuity
    # with the prose elsewhere on the page to preserve
    # the page's dealer, when that is what was handed in, is also its record of
    # the text already written, which a caption must not repeat
    used = text_pool if isinstance(text_pool, LineDealer) else None
    if text_pool is not None and not isinstance(text_pool, list):
        text_pool = _as_dealer(text_pool).pool

    # the chart's own handwriting size, which a rewritten caption may not outgrow
    heights = sorted(child.bbox()[3] - child.bbox()[1]
                     for shape in diagram.children if shape.class_name in SHAPE_CLASSES
                     for child in shape.children if child.class_name == "textline")
    caption_height = heights[len(heights) // 2] if heights else None

    for shape in diagram.children:
        if shape.class_name not in SHAPE_CLASSES:
            continue
        had_caption = any(child.class_name == "textline"
                          for child in shape.children)
        if not had_caption:
            continue
        shape.children = [child for child in shape.children
                          if child.class_name != "textline"]
        box = _interior_box(shape)
        if box is None:
            continue
        content = shape_content(box, text_pool, math_pool, rng, used=used,
                                caption_height=caption_height, min_band=min_band)
        if content is None:
            continue
        shape.children.append(content)
        # keep the chart's own record of what its nodes say in step with the ink
        shape_id = (shape.meta or {}).get("id")
        if shape_id in nodes:
            nodes[shape_id]["text"] = content.label


DIAGRAM_FALLBACK_POOL = 12      # roomiest charts to choose among when none qualifies

# id of a pooled chart -> the median room inside its captioned shapes (None: it has none)
_CHART_ROOM: dict = {}


def _band_height(ink: Ink) -> float:
    band = _core_band(ink)
    return band[1] - band[0] if band is not None else 0.0


def _chart_room(diagram: Ink) -> Optional[float]:
    """How tall a caption a chart's shapes can take, in chart units: the median height
    of the room inside (_interior_box) each shape that carries a caption, measured
    without the caption, as restock_diagram_text measures it. No room counts as 0."""
    key = id(diagram)
    if key not in _CHART_ROOM:
        heights = []
        for shape in diagram.children:
            if shape.class_name not in SHAPE_CLASSES or not any(
                    c.class_name == "textline" for c in shape.children):
                continue
            box = _interior_box(Ink(strokes=shape.strokes, children=[
                c for c in shape.children if c.class_name != "textline"]))
            heights.append(box[3] - box[1] if box else 0.0)
        heights.sort()
        _CHART_ROOM[key] = heights[len(heights) // 2] if heights else None
    return _CHART_ROOM[key]


def _fit_scale(ink: Ink, cell: Tuple[float, float]) -> float:
    """The scale fit_ink_to_box gives `ink` in a `cell` (width, height), before jitter."""
    x0, y0, x1, y1 = ink.bbox()
    return min(cell[0] / max(x1 - x0, 1e-6), cell[1] / max(y1 - y0, 1e-6))


def make_diagram(ohfd_pool: Optional[List[Ink]], text_pool: Optional[List[Ink]] = None,
                  cell: Optional[Tuple[float, float]] = None,
                  min_area_per_symbol: Optional[float] = None,
                  usage: Optional[dict] = None,
                  math_pool: Optional[List[Ink]] = None,
                  text_scale: Optional[float] = None,
                  fit_jitter: float = 0.0,
                  page_charts: Optional[set] = None) -> Optional[Ink]:
    """A flowchart, chosen to stay legible in the space it is given.

    OHFD charts run from 9 to 30 symbols and the whole chart is scaled into its
    cell, so a complex one in a small cell came out as an illegible knot. The
    choice is scored on the area each symbol will *actually* occupy once fitted,
    which is not the cell's area: fitting preserves aspect, so a tall chart in a
    wide cell only uses a narrow column of it — sizing off the cell alone left
    charts at a third of the intended density.

    With `text_scale` (page units per text unit, the page's handwriting size) and
    `cell`, no writing in the chart may come out below FLOWCHART_MIN_TEXT_RATIO of the page's.
    A caption is trimmed to fewer words rather than shrunk to fit its shape
    (shape_content), so a chart is chosen by its shapes' height, not its captions'
    length or its writer's handwriting: only charts whose captioned shapes have room
    for CAPTION_RUN_BOX bands of that size once the chart is fitted to the cell
    (allowing for `fit_jitter`) are chosen. A shape no caption keeps its size in is
    left blank, and a caption beside an arrow that would be too small is taken off.
    With `page_charts` (the names of the OHFD charts already on the page), none of them is chosen again, and the
    chosen chart's name is added: a page may hold two drawings of one layout, by different writers, but not one
    drawing twice.
    Returns None when no chart qualifies, and the cell is written as something else.
    """
    if ohfd_pool:
        if page_charts:
            # the pool itself, not the candidates: every filter below starts again from the pool
            ohfd_pool = [d for d in ohfd_pool if d.label not in page_charts]
            if not ohfd_pool:
                return None
        candidates = ohfd_pool
        need = None             # smallest x-height band on the page, in page units
        if cell is not None and text_scale:
            need = FLOWCHART_MIN_TEXT_RATIO * LINE_HEIGHT * text_scale / (1.0 - fit_jitter)
            candidates = [d for d in ohfd_pool
                          if _chart_room(d) is None
                          or _chart_room(d) * _fit_scale(d, cell) >= need * CAPTION_RUN_BOX]
            if not candidates:
                return None
        if cell is not None and min_area_per_symbol:
            cell_w, cell_h = cell
            scored = []
            for diagram in candidates:
                x0, y0, x1, y1 = diagram.bbox()
                width, height = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
                scale = min(cell_w / width, cell_h / height)
                area = (width * scale) * (height * scale)
                scored.append((area / max(len(diagram.children), 1), diagram))
            roomy = [d for per, d in scored if per >= min_area_per_symbol]
            if roomy:
                candidates = roomy
            else:
                # Nothing is legible at this size. Falling back to the single
                # roomiest chart made every small cell on every page show the
                # same one — take the roomiest handful and choose among them.
                scored.sort(key=lambda item: -item[0])
                candidates = [d for _, d in scored[:DIAGRAM_FALLBACK_POOL]]

        # Prefer charts this run has used least. The candidate set is narrow —
        # a mid-sized cell fits only a few of the charts — so choosing uniformly
        # kept landing on the same ones: 62% of placements were repeats, with
        # one chart appearing ten times in a hundred pages.
        if usage is not None and candidates:
            fewest = min(usage.get(id(d), 0) for d in candidates)
            candidates = [d for d in candidates if usage.get(id(d), 0) == fewest]

        chosen = random.choice(candidates)
        if usage is not None:
            usage[id(chosen)] = usage.get(id(chosen), 0) + 1
        if page_charts is not None:
            page_charts.add(chosen.label)
        d = dict_deepcopy(chosen)
        # a flowchart is a level-2 object: its shapes and arrows are level 0,
        # and its captions are its lines (_unnest_flowchart)
        d.class_name = "flowchart"
        d.meta = dict(d.meta or {}, flowchart=True)
        # the copy is deep, so rewriting the captions cannot reach back into the
        # pooled chart this one was taken from
        min_band = need / _fit_scale(chosen, cell) if need is not None else None
        restock_diagram_text(d, text_pool, math_pool, min_band=min_band)
        if min_band is not None:
            # OHFD's own captions beside arrows ("yes", "no") are kept as written
            d.children = [c for c in d.children
                          if c.class_name != "textline" or _band_height(c) >= min_band]
        _drop_overlapping_captions(d)
        return _unnest_flowchart(d)
    d = make_diagram_procedural(text_pool)
    _drop_overlapping_captions(d)
    return _unnest_flowchart(d)


def _drop_overlapping_captions(diagram: Ink, limit: float = LINE_MAX_OVERLAP) -> None:
    """Take captions off until no two of the chart's lines (textlines and formulas, in any
    pairing) share more than `limit` of the smaller one's box. In place, before
    _unnest_flowchart.

    A caption beside an arrow goes first, since OHFD wrote it where it liked while a
    shape's caption was fitted inside its own shape; between two of a kind, the smaller.
    A shape whose caption goes is left blank, as one is when no caption keeps its size.
    """
    nodes = ((diagram.meta or {}).get("graph") or {}).get("nodes") or {}

    def area(line: Ink) -> float:
        x0, y0, x1, y1 = line.bbox()
        return (x1 - x0) * (y1 - y0)

    while True:
        # (line, what holds it, whether that is a shape)
        lines = [(c, diagram, False) for c in diagram.children
                 if c.class_name in LINE_CLASSES and c.all_strokes()]
        lines += [(c, shape, True) for shape in diagram.children if shape.class_name in SHAPE_CLASSES
                  for c in shape.children if c.class_name in LINE_CLASSES and c.all_strokes()]
        pair = next(((a, b) for i, a in enumerate(lines) for b in lines[i + 1:]
                     if overlap_share(a[0].bbox(), b[0].bbox()) > limit + _OVERLAP_EPS), None)
        if pair is None:
            return
        a, b = pair
        if a[2] != b[2]:
            line, holder, in_shape = a if not a[2] else b
        else:
            line, holder, in_shape = a if area(a[0]) <= area(b[0]) else b
        holder.children = [c for c in holder.children if c is not line]
        if in_shape:
            shape_id = (holder.meta or {}).get("id")
            if shape_id in nodes:
                nodes[shape_id]["text"] = None


def _unnest_flowchart(diagram: Ink) -> Ink:
    """The flowchart restructured so its captions are lines of the chart and no
    shape holds anything.

    A source chart nests each node's caption inside the node's shape. But a
    shape is an `enclosure`, and an enclosure holds nothing and stands on
    its own — so the caption's lines come out beside the shape as lines of the
    flowchart itself, the way a table holds its cells' lines, with nothing tying
    them to the shape. A caption sitting on an arrow is already beside it rather
    than inside it, and stays a line of the chart. Each caption line is tagged
    meta["caption"] "shape" or "arrow"; a shape and its caption's lines share a
    key (meta["caption_key"] on the shape, ["caption_of"] on each line), which is
    what orders the shapes among the chart's level-1 objects. In place; also
    returned.
    """
    children: List[Ink] = []
    for node in diagram.children:
        if node.class_name == "enclosure":
            lines = [c for c in node.children if c.class_name in ("textline", "formula")]
            node.children = [c for c in node.children
                             if c.class_name not in ("textline", "formula")]
            children.append(node)
            key = _new_id("caption") if lines else None
            if key is not None:
                node.meta = dict(node.meta or {}, caption_key=key)
            for line in lines:
                line.meta = dict(line.meta or {}, caption="shape", caption_of=key)
                children.append(line)
        elif node.class_name in ("textline", "formula"):
            node.meta = dict(node.meta or {}, caption="arrow")
            children.append(node)
        else:
            children.append(node)
    diagram.children = children
    return diagram


def make_drawing(quickdraw_pool: List[Ink]) -> Ink:
    d = dict_deepcopy(random.choice(quickdraw_pool))
    d.class_name = "drawing"
    return d


def make_geometry(shape_bank, rng: random.Random = random,
                  glyphs: Optional[dict] = None,
                  math_pool: Optional[List[Ink]] = None) -> Optional[Ink]:
    """A textbook geometry figure, class_name="geometry".

    A figure is boxed as a whole, as a flowchart is: it is one construction of
    segments and arcs whose parts mean nothing on their own, and the labels
    written on it are its lines.

    `glyphs` is character ink keyed by character (loaders.load_emnist_pool); with
    it the figure names its corners and writes its lengths and angles.
    `math_pool` supplies the ink for angles written in radians — whichever of
    its expressions read as one; see geometry_diagrams.radian_expressions.

    Returns None when there is no bank to draw with, so the caller can leave the
    cell to something else.
    """
    import geometry_diagrams as gd
    if shape_bank is None or not shape_bank.ok():
        return None
    figure = gd.make_diagram(shape_bank, rng, glyphs=glyphs,
                              radians=gd.radian_expressions(math_pool))
    figure.class_name = "geometry"
    return figure


ENCLOSURE_PAD = 0.18        # clearance round what is enclosed, in multiples of a line's height
ENCLOSURE_OVAL_PROB = 0.5   # of enclosures, how many are ovals rather than boxes
ENCLOSURE_TRIES = 8         # redraws, each a little larger, before giving up
# What gets enclosed, rarest first: a whole block is marked less often than one
# line of it, a line less often than a few words in it, and a few words less
# often than a single word.
ENCLOSURE_TARGET_WEIGHTS = {"textblock": 0.15, "line": 0.20, "words": 0.25, "word": 0.40}


def _inside_loop(loop_x, loop_y, px, py):
    """Whether each point lies inside the polygon a drawn loop traces (even-odd)."""
    import numpy as np
    xs, ys = np.asarray(loop_x, dtype=float), np.asarray(loop_y, dtype=float)
    inside = np.zeros(len(px), dtype=bool)
    for xa, ya, xb, yb in zip(xs, ys, np.roll(xs, 1), np.roll(ys, 1)):
        if ya == yb:
            continue
        crosses = (ya > py) != (yb > py)
        inside ^= crosses & (px < xa + (xb - xa) * (py - ya) / (yb - ya))
    return inside


def _row_height(node: Ink) -> float:
    """A typical line's height under `node` — what an enclosure's padding scales
    with, so a boxed paragraph is not padded by its whole height."""
    heights = []

    def walk(current: Ink):
        if current.class_name in ("textline", "formula"):
            box = current.bbox()
            heights.append(box[3] - box[1])
            return
        for child in current.children:
            walk(child)

    walk(node)
    if heights:
        return max(sorted(heights)[len(heights) // 2], 1e-6)
    box = node.bbox()
    return max(box[3] - box[1], 1e-6)


def _clear_of(strokes: List[Stroke], keep_clear, bounds) -> bool:
    """Whether the strokes' box stays inside `bounds` and off every `keep_clear` box."""
    xs = [x for s in strokes for x in s.x]
    ys = [y for s in strokes for y in s.y]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    if bounds is not None and (x0 < bounds[0] or y0 < bounds[1]
                               or x1 > bounds[2] or y1 > bounds[3]):
        return False
    return not any(min(x1, b[2]) > max(x0, b[0]) and min(y1, b[3]) > max(y0, b[1])
                   for b in keep_clear)


_GENERATED_IDS = itertools.count()


def _new_id(prefix: str) -> str:
    """An element id no source dataset uses, for tying generated objects together."""
    return "%s-%d" % (prefix, next(_GENERATED_IDS))


def _draw_enclosure(target: Ink, shape_bank, rng: random.Random,
                    keep_clear: List[Tuple[float, float, float, float]] = (),
                    bounds: Optional[Tuple[float, float, float, float]] = None,
                    unmarked_rows: List[Ink] = ()) -> Optional[Ink]:
    """A box or oval of mined ink round `target`, as an `enclosure` — one
    that holds nothing; maybe_enclose puts it in the block it marks.

    The shape is drawn the way a geometry figure is — mined lines for the box's
    sides, a mined oval for the loop — so it carries a real hand's wobble
    instead of a procedural one.

    The ink inside must not be cut off. A mined oval dips a little inside the
    ellipse it was fitted to, and a mined line wanders off its chord, so the
    drawn stroke is tested against what it encloses rather than trusted: an
    oval has to hold every point of the target's ink, a box's sides have to
    stay clear of the target's box. A shape that cuts in is redrawn a little
    larger.

    It also has to stay off the rest of the page. The smallest ellipse holding a
    box is sqrt 2 times the box's size, so an oval round a whole block reaches
    well past it, and before this check half the block enclosures ran into the
    block next door. A shape whose box crosses any `keep_clear` box or leaves
    `bounds` is refused; growing it would only reach further, so the other shape
    is tried instead — a box hugs its target far closer than an oval can. None
    comes back when neither fits.

    Nor may it cut through writing it does not mark: a shape whose strokes come
    within the middle of any of `unmarked_rows` is refused the same way, by the
    test an underline's neighbours get (_mark_fits). Checked only against other
    blocks, 3 of 11 ovals and 2 of 25 boxes round text over 60 pages ran through
    the middle of the line above or below their target.
    """
    import numpy as np
    import geometry_diagrams as gd
    points = target.all_strokes()
    if not points:
        return None
    x0, y0, x1, y1 = target.bbox()
    pad = _row_height(target) * ENCLOSURE_PAD
    px = np.array([x for s in points for x in s.x])
    py = np.array([y for s in points for y in s.y])

    first = "oval" if rng.random() < ENCLOSURE_OVAL_PROB else "rectangle"
    for shape in (first, "rectangle" if first == "oval" else "oval"):
        for attempt in range(ENCLOSURE_TRIES):
            grow = 1.15 ** attempt
            if shape == "oval":
                # the smallest ellipse through the padded box's corners is the
                # box's own half-extents times sqrt 2
                rx = ((x1 - x0) / 2 + pad) * math.sqrt(2) * grow
                ry = ((y1 - y0) / 2 + pad) * math.sqrt(2) * grow
                loop = gd.arc(shape_bank, rng, ((x0 + x1) / 2, (y0 + y1) / 2), rx, ry)
                strokes = [loop] if _inside_loop(loop.x, loop.y, px, py).all() else None
            else:
                p = pad * grow
                corners = [(x0 - p, y0 - p), (x1 + p, y0 - p),
                           (x1 + p, y1 + p), (x0 - p, y1 + p)]
                # a box round some writing is dashed off: its corners overshoot,
                # open up and miss far more than a figure's do
                sides = [gd.segment(shape_bank, rng, a, b)
                         for a, b in gd.loose_sides(corners, rng, gd.ENCLOSURE_JOIN)]
                strokes = None if any(x0 < x < x1 and y0 < y < y1
                                      for side in sides
                                      for x, y in zip(side.x, side.y)) else sides
            if strokes is None:
                continue            # cut into the target: redraw a little larger
            if (_clear_of(strokes, keep_clear, bounds)
                    and _mark_fits(strokes, "enclosure", [], list(unmarked_rows))):
                return Ink(strokes=strokes, label=shape,
                           class_name="enclosure", source="geometry")
            break                   # larger only reaches further; try the other shape
    return None


def maybe_enclose(block: Ink, shape_bank, probability: float,
                  rng: random.Random = random,
                  keep_clear: List[Tuple[float, float, float, float]] = (),
                  bounds: Optional[Tuple[float, float, float, float]] = None) -> Ink:
    """With the given probability, box or circle part of a text block: the whole
    block, one of its lines, a few words of a line, or one word.

    Someone marking up notes rings a whole paragraph now and then, a line more
    often, a few words more often still, and a single word most often of all
    (ENCLOSURE_TARGET_WEIGHTS). The shape is a level-0 `enclosure` — the
    class every flowchart shape and table border carries too — with the shape
    drawn kept as its label.

    A shape round the whole block is the exception: it is the block's border, kept
    as the block's own unlabelled strokes like a table's outer border, not an
    enclosure.

    An enclosure does not touch the structure of what it is drawn round. The
    block's lines and their words stay exactly as they were, and the shape goes
    into the block as a level-1 object beside them, recording the row it marks
    (meta["attends_row"]: that row counted down the block, or -1 for the whole
    block) so level-1 targets can order it by that row. Words are never split
    out of their line: a box round the middle of a line changes what those
    words mean, not which line they belong to.

    What comes back is the block, holding the shape when something was enclosed
    and untouched when nothing was.

    `keep_clear` is every other block on the page and `bounds` the page itself;
    the shape may touch neither (see _draw_enclosure). No bank to draw with, a
    level this block cannot offer, or no shape that fits clear of its target
    and its neighbours, and the block comes back untouched.
    """
    if shape_bank is None or not shape_bank.ok() or rng.random() >= probability:
        return block

    # Lines are prose only. A formula row is not a textline, and letting one be
    # enclosed made lines as common as single words: a maths section has rows
    # but no words, so it could only ever offer the line level.
    rows = [child for child in block.children
            if child.class_name == "textline" and child.all_strokes()]

    def words_of(line: Ink) -> List[Ink]:
        return sorted((child for child in line.children if child.class_name == "word"),
                      key=lambda word: word.bbox()[0])

    def word_lines(minimum: int) -> List[Ink]:
        """Prose rows with at least `minimum` words, all of them real ink. A list
        bullet in the row is not a word and is not counted."""
        return [line for line in rows
                if len(words_of(line)) >= minimum
                and all(word.all_strokes() for word in words_of(line))]

    levels = list(ENCLOSURE_TARGET_WEIGHTS)
    level = rng.choices(levels, weights=[ENCLOSURE_TARGET_WEIGHTS[k] for k in levels])[0]
    # A run of words that is not the whole line needs three words to choose from,
    # one word needs two. A level this block cannot offer is not swapped for a
    # different one: a maths section has no words at all, and sending its word
    # picks to a line instead made lines the commonest enclosure on the page, the
    # reverse of the weights. Only a run of words falls back, to a single word,
    # the nearest thing to it.
    if level == "words" and not word_lines(3):
        level = "word"
    if level == "word" and not word_lines(2):
        return block
    if level == "line" and not rows:
        return block

    attended = None                 # the row the shape marks; None for the whole block
    if level == "textblock":
        marked = [block]
        outline = block
    elif level == "line":
        marked = [rng.choice(rows)]
        outline = marked[0]
        attended = marked[0]
    else:
        attended = rng.choice(word_lines(3 if level == "words" else 2))
        words = words_of(attended)
        length = 1 if level == "word" else rng.randint(2, len(words) - 1)
        lo = rng.randrange(len(words) - length + 1)
        marked = words[lo:lo + length]
        # only to draw round: the words stay in their line
        outline = Ink(strokes=[], children=marked)

    # the block's other rows, whose writing the shape must not cut through; the
    # rest of a marked row is left out, since a shape round some of its words
    # cannot help passing close to the words beside them
    marked_ids = {id(node) for node in marked}
    unmarked_rows = [row for row in block.children
                     if row.class_name in ("textline", "formula")
                     and id(row) not in marked_ids
                     and not any(id(child) in marked_ids for child in row.children)]
    enclosure = _draw_enclosure(outline, shape_bank, rng, keep_clear, bounds, unmarked_rows)
    if enclosure is None:
        return block
    if attended is None:
        # A box or ring round a whole block is a border, the way a table's outer
        # border is: its strokes become the block's own unlabelled ink rather than
        # an `enclosure` of their own, so no target names it and no mark crop is made.
        block.strokes = list(block.strokes) + list(enclosure.strokes)
        block.meta = dict(block.meta or {}, border=enclosure.label)
        return block
    row_nodes = [row for row in block.children if row.class_name in ("textline", "formula")]
    rank = -1 if attended is None else next(i for i, row in enumerate(row_nodes)
                                            if row is attended)
    enclosure.meta = dict(enclosure.meta or {}, attends_row=rank)
    block.children.append(enclosure)
    return block


# small helper so we don't import copy everywhere
def dict_deepcopy(ink: Ink) -> Ink:
    from core import deep_copy_ink
    return deep_copy_ink(ink)