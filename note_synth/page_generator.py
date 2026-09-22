"""
page_generator.py — fill one page's grid of cells with content and emit the
hierarchical labels for it.

The page is a square tiled with cells (see layout.build_grid), grown from the
top-left corner with random widths and heights, about 3.5 across any line and
9-19 blocks in all. It is laid out at layout.PAGE_W and scaled to layout.OUTPUT_SIZE (224) at
the end, labels included.
Each cell is then given over to one kind of content — a flowchart, a geometry
figure, a drawing, a table, or a textblock (rows of prose and formulas, one of the
two the majority, sometimes bulleted) — and filled at the page's single
handwriting scale.

Which kind a cell gets depends on its size and shape. A flowchart needs room to
be legible at all, so its share rises with the cell's area; a sketch is the
opposite and prefers a small cell; a table is likeliest in a mid-sized cell of
roughly its own proportions and rare in extreme ones. A narrow textblock leans to
formulas and a wide one to prose.
"""

from __future__ import annotations
from collections import Counter
from typing import Dict, List, Optional, Tuple
import math
import os
import random

import numpy as np

from core import Ink, fit_ink_to_box, transform_ink
from labels import extract_labels, export_json, format_target_string
import block_generators as bg
import geometry_diagrams as gd
import layout as ly


# What a cell holds is drawn from these weights, each a function of the cell's
# size and shape (see _choose_kind). Width and height are fractions of the
# writing area, `area` is their product and `aspect` is width / height.
#
#   flowchart  FLOWCHART_WEIGHT * area                   (0 below MIN_FLOWCHART_AREA)
#   geometry   GEOMETRY_WEIGHT * area * fit(aspect, 1)   (0 below MIN_GEOMETRY_AREA)
#   drawing    DRAWING_WEIGHT * (1 - area) * fit(aspect, 1)
#   table      TABLE_WEIGHT * fit(area, TABLE_BEST_AREA) * fit(aspect, TABLE_BEST_ASPECT)
#   textblock  TEXTBLOCK_WEIGHT
#
# fit(x, best) is 1 at `best` and falls away on either side in log terms, so
# half as big and twice as big score the same (_fit). A flowchart or a figure
# grows likelier with the cell, a sketch prefers a small one, and a table is
# likeliest in a mid-sized cell of roughly its own proportions — rare in a
# thumbnail, in a cell spanning most of the page, or in a sliver. A figure, a
# sketch and a table are fitted into their cell at their own aspect ratio, so a
# cell of the wrong shape wastes most of itself; a textblock re-flows to its
# cell and does not care.
#
# Solved over 3000 simulated layouts (12.9 cells a page, 5.0 of them big enough
# for a flowchart) for about 2 flowcharts, 2 tables, 0.33 drawings and 1.14
# geometry figures a page, which leaves about 7.4 textblocks. The previous
# weights (flowchart 2.5, geometry 5.0, drawing 0.22, table 0.17) gave 0.43
# flowcharts, 1.16 drawings and 0.27 tables a page in the 10,000-page train set;
# flowcharts and tables were raised and sketches cut back. Geometry's weight rose
# only to hold its count while the other kinds took more of the page. With this
# FLOWCHART_WEIGHT about 9 pages in 10 carry at least one flowchart.
FLOWCHART_WEIGHT = 26.3
DRAWING_WEIGHT = 0.083
SHAPE_FIT_SPREAD = 1.0      # how fast geometry and drawings fall off away from square

# Below this share of the writing area no flowchart is placed. Lowered from 0.09
# to shift flowcharts toward smaller cells (see FLOWCHART_WEIGHT). Even the
# simplest chart in the set (9 symbols) needs about 13% of the writing area to
# meet AREA_PER_SYMBOL, so a smaller cell gets one of the roomiest few instead
# (make_diagram). That density is intended: a page stands in for a downsampled
# capture of a much larger screen, so small content is not meant to be legible.
MIN_FLOWCHART_AREA = 0.06
# Page area a flowchart symbol needs to stay legible. OHFD charts carry 9-30
# symbols and the whole chart is scaled into its cell, so without this a
# 30-symbol chart in a small cell rendered as a knot of overlapping boxes.
AREA_PER_SYMBOL = 12000.0

# How often each OHFD chart has been used, so the generator can spread itself
# over the set instead of returning to the same few. Module-level because it has
# to persist across generate_page calls; clear it to start a fresh run.
DIAGRAM_USAGE: Dict[int, int] = {}

# --- geometry ----------------------------------------------------------------
# A textbook figure — a triangle with an altitude, a circle with a chord, two
# shapes glued along a side — built by geometry_diagrams out of mined ink.
#
# Chosen per cell like every other kind, and a page may carry several.
GEOMETRY_WEIGHT = 9.06
# A figure is a single shape, not twenty labelled symbols, so it survives a much
# smaller cell than a flowchart — but below this it is a few strokes wide and the
# construction inside it (an inscribed circle, a right-angle box) is lost.
MIN_GEOMETRY_AREA = 0.045
# Fraction of the cell a figure is drawn into. Unlike a flowchart, which fills
# the space it is given, a geometry figure is drawn once on a page with room
# around it, and a triangle stretched to the full height of a tall cell reads as
# a mistake rather than as a triangle.
GEOMETRY_FILL = 0.82

# Where the mined ink for geometry lives. Built by
#   python geometry_diagrams.py --cache shapes_bank.json
# which takes minutes; if the cache is not there geometry is simply left off the
# page rather than mining the pools in the middle of a page-generation run.
SHAPES_BANK_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "shapes_bank.json")
_SHAPE_BANK: Optional[gd.Bank] = None
_SHAPE_BANK_TRIED = False

# A table is its own kind, not a kind of textblock, at about 2 a page. make_table
# refuses a cell that comes up empty and a textblock is written there instead, so
# the weight is set for somewhat more tables chosen than end up on the page: about
# a fifth are refused (150 pages from the split caches: 2.41 chosen, 1.89 placed at 2.2).
TABLE_WEIGHT = 2.4
# Sizes (rows, cols) a table is tried at before its cell is written as prose, when
# its writing would be squeezed below bg.MIN_TEXT_RATIO: first a random size, then
# the smallest. None is make_table's random choice.
TABLE_FIT_SIZES = ((None, None), (2, 2))
# fit_ink_to_box's random scale change for a table or flowchart; the legibility
# floor allows for it shrinking the content by up to this much.
CELL_FIT_JITTER = 0.02
TABLE_BEST_AREA = 0.10
TABLE_BEST_ASPECT = 1.6
TABLE_FIT_SPREAD = 0.8

# A textblock is rows of prose and formulas with one of the two the majority:
# mostly formulas with the odd line of prose (a worked problem), or mostly prose
# with the odd formula inline (notes). Paragraphs, bulleted lists and maths
# sections are all this one kind. Which way a block leans depends on its width:
# a prose line is cut where it meets the cell's right edge and a narrow cell cuts
# most of them into fragments, while a formula is scaled to fit. So the chance
# of leaning to prose rises linearly from TEXT_HEAVY_PROB[0] at
# TEXT_HEAVY_WIDTH_XH[0] x-heights wide to TEXT_HEAVY_PROB[1] at [1].
TEXTBLOCK_WEIGHT = 1.5
TEXT_HEAVY_WIDTH_XH = (8.0, 24.0)
TEXT_HEAVY_PROB = (0.1, 0.9)
MINORITY_ROW_PROB = 0.15    # chance any one row is the block's minority kind
BULLET_PROB = 0.22          # chance a textblock is written as a bulleted list
# chance a text block gets something boxed or circled — itself, a line or some
# words — as an `enclosure` (bg.maybe_enclose)
# Raised from 0.08 (about 0.4 textblock enclosures a page) along with the line marks.
ENCLOSURE_PROB = 0.30

# Scribbles — procedurally generated crossings-out, loops, spirals, squiggles and
# hatching, filed as level-0 `unknown` — are NOT generated. A `drawing` cell is
# always a real QuickDraw sketch.
#
# They were an invention, not something any source dataset carries: the argument
# for them was that a real page has marks belonging to none of the taxonomy's
# classes and the model needs somewhere to put them. But procedural squiggles are
# not those marks. They are drawn by formula, they sit in their own cell with
# space around them rather than on top of the thing they cross out, and they gave
# the page `unknown` boxes that no real page would produce in that shape or that
# place. Teaching a class off invented ink is worse than leaving the class empty.
#
# Nothing else emits `unknown`, so the class no longer appears on a generated
# page. It is deliberately left in the taxonomy (labels.LEVEL0): the right way to
# fill it is real unnameable ink, or scribbles drawn ON real text — a different
# generator with a different label relationship, not this one.

# A real page is written by one person, so all its prose should come from one
# hand. Without this, x-height normalisation makes every line the same size but
# a page still mixes a tight upright hand with a loose sprawling one, which is
# what actually reads as "inconsistent".
#
# Which sources may supply prose. DeepWriting is excluded by default for two
# reasons, both properties of the data rather than of anything we do to it
# (verified by rendering its lines straight out of the loader):
#   * it is loose cursive and simply much harder to read than IAM's print hand;
#   * its samples are fixed-length CHUNKS of a line, so they start and end
#     mid-word — "He smiled at his ow", "actually drippin", "k in". That is the
#     source of the fragments on the page, not the layout.
# IAM alone supplies 12k complete lines across ~200 writers, which is plenty.
# Pass sources explicitly or clear this to bring DeepWriting back.
PROSE_SOURCES = ("iam_ondb",)

# IAM-OnDB names its lines by form, not writer. data/iamondb_writers.json
# (iam_writers.py) recovers the real writer for 1,591 of its 1,730 forms — 200
# writers, a median of 55 lines each — and the other 139 forms stand in for
# their writer, about 7 lines apiece. The threshold is low for those.
MIN_LINES_PER_WRITER = 5
TARGET_LINES_PER_PAGE = 60
# Raised from 2 once the grid made pages dense, and again once DeepWriting was
# dropped from prose, when every "writer" was a single ~7-line form: a full page
# wants ~40 lines, so a tight cap meant the same sentence three times, and at 6
# hands repeated lines measured 0.0%. A real writer's median 55 lines now cover
# most of TARGET_LINES_PER_PAGE alone. A second hand on a page reads far less
# wrong than a repeated sentence does.
MAX_WRITERS_PER_PAGE = 6

# A page whose lines (textlines and formulas) still overlap past bg.LINE_MAX_OVERLAP is
# built again, up to this many times in all. The generators keep to that limit
# themselves, so this is only the guarantee.
LINE_OVERLAP_ATTEMPTS = 10


def _writer_weight(lines: List[Ink]) -> int:
    """How much usable prose a writer contributes: words, not lines.

    Weighting by line count favours DeepWriting, whose "lines" are often a
    single word, over IAM's full sentences. Counting words picks the hand with
    the most actual text behind it.
    """
    return sum(max(1, sum(1 for c in line.children if c.class_name == "word"))
               for line in lines)


def _single_writer_pool(text_pool: List[Ink]) -> List[Ink]:
    """The lines of one writer — or a few, if one can't fill a page."""
    if PROSE_SOURCES:
        preferred = [l for l in text_pool if l.source in PROSE_SOURCES]
        if preferred:
            text_pool = preferred      # fall through to everything if none match

    # Skip lines whose source ships no transcription. A handful of IAM documents
    # have none, and because a page draws its prose from only a few documents
    # they arrive in clumps — half a paragraph with no ground truth at all.
    # 12,178 of IAM's 12,195 lines are transcribed, so this costs nothing.
    transcribed = [l for l in text_pool if l.label]
    if transcribed:
        text_pool = transcribed

    by_writer: Dict[Optional[str], List[Ink]] = {}
    for line in text_pool:
        by_writer.setdefault(line.writer, []).append(line)

    usable = [lines for writer, lines in by_writer.items()
              if writer is not None and len(lines) >= MIN_LINES_PER_WRITER]
    if not usable:
        return text_pool     # unlabelled or thin pool: better mixed than empty

    weights = [_writer_weight(lines) for lines in usable]
    chosen: List[Ink] = []
    for _ in range(MAX_WRITERS_PER_PAGE):
        if not usable or len(chosen) >= TARGET_LINES_PER_PAGE:
            break
        pick = random.choices(range(len(usable)), weights=weights)[0]
        chosen.extend(usable.pop(pick))
        weights.pop(pick)
    return chosen


def shape_bank_default() -> Optional[gd.Bank]:
    """The mined line/oval ink geometry figures are drawn from.

    Loaded from the cache once per process and kept, since a page can want
    several figures and every figure re-reads the whole bank. Returns None when
    there is no cache to load, which just means no geometry on the page —
    building one mines every pool and takes minutes, so it is not something to
    trigger halfway through generating a page.
    """
    global _SHAPE_BANK, _SHAPE_BANK_TRIED
    if _SHAPE_BANK_TRIED:
        return _SHAPE_BANK
    _SHAPE_BANK_TRIED = True
    if os.path.exists(SHAPES_BANK_CACHE):
        bank = gd.build_bank(cache=SHAPES_BANK_CACHE)
        # build_bank falls through to mining when the cache is unreadable, but it
        # returns an empty bank rather than raising if the file is merely thin
        if bank.ok():
            _SHAPE_BANK = bank
    return _SHAPE_BANK


def _fit(value: float, best: float, spread: float) -> float:
    """1 at `best`, falling away on either side in log terms, so that half as
    big and twice as big score the same. `spread` is how far (in log units) the
    score takes to drop to 1/e."""
    return math.exp(-(math.log(value / best) / spread) ** 2)


def _choose_kind(width_fraction: float, height_fraction: float,
                  have: Dict[str, bool], rng: random.Random = random) -> str:
    """Pick what goes in a cell from its size and shape (see FLOWCHART_WEIGHT).

    `width_fraction` and `height_fraction` are the cell's size as a share of the
    writing area's width and height.
    """
    area = width_fraction * height_fraction
    aspect = width_fraction / height_fraction
    square = _fit(aspect, 1.0, SHAPE_FIT_SPREAD)
    weights = {
        "flowchart": (FLOWCHART_WEIGHT * area
                      if have["flowchart"] and area >= MIN_FLOWCHART_AREA else 0.0),
        "geometry": (GEOMETRY_WEIGHT * area * square
                     if have["geometry"] and area >= MIN_GEOMETRY_AREA else 0.0),
        "drawing": (DRAWING_WEIGHT * max(0.0, 1.0 - area) * square
                    if have["drawing"] else 0.0),
        "table": (TABLE_WEIGHT * _fit(area, TABLE_BEST_AREA, TABLE_FIT_SPREAD)
                  * _fit(aspect, TABLE_BEST_ASPECT, TABLE_FIT_SPREAD)
                  if have["table"] else 0.0),
        "textblock": TEXTBLOCK_WEIGHT if have["textblock"] else 0.0,
    }
    if not any(weights.values()):
        return "textblock"
    kinds = list(weights)
    return rng.choices(kinds, weights=[weights[k] for k in kinds])[0]


def _text_heavy_prob(local_w: float) -> float:
    """Chance a textblock `local_w` text units wide leans to prose (TEXT_HEAVY_PROB)."""
    lo, hi = TEXT_HEAVY_WIDTH_XH
    t = min(1.0, max(0.0, (local_w / bg.LINE_HEIGHT - lo) / (hi - lo)))
    return TEXT_HEAVY_PROB[0] + t * (TEXT_HEAVY_PROB[1] - TEXT_HEAVY_PROB[0])


def _inset(cell: Tuple[float, float, float, float], fill: float,
            rng: random.Random = random) -> Tuple[float, float, float, float]:
    """A smaller box inside the cell, placed randomly rather than centred."""
    x0, y0, x1, y1 = cell
    width, height = (x1 - x0) * fill, (y1 - y0) * fill
    left = x0 + rng.uniform(0.0, (x1 - x0) - width)
    top = y0 + rng.uniform(0.0, (y1 - y0) - height)
    return (left, top, left + width, top + height)


def _fill_cell(kind: str, cell: Tuple[float, float, float, float], scale: float,
                dealer, pools: Dict[str, list], page_charts: Optional[set] = None) -> Optional[Ink]:
    """Build the content for one cell, already in page coordinates. `page_charts`: the OHFD charts
    already on the page, which a flowchart may not repeat (bg.make_diagram); it is added to."""
    x0, y0, x1, y1 = cell
    local_w, local_h = ly.cell_capacity(cell, scale)

    if kind == "flowchart":
        diagram = bg.make_diagram(pools["ohfd"], dealer,
                                   cell=(x1 - x0, y1 - y0),
                                   min_area_per_symbol=AREA_PER_SYMBOL,
                                   usage=DIAGRAM_USAGE,
                                   math_pool=pools["math"],
                                   text_scale=scale, fit_jitter=CELL_FIT_JITTER,
                                   page_charts=page_charts)
        if diagram is not None:
            return fit_ink_to_box(diagram, cell, jitter=CELL_FIT_JITTER)
        # no chart keeps its writing legible in a cell this size, or every one that does is already on
        # the page; write a textblock

    if kind == "geometry":
        figure = bg.make_geometry(pools["shapes"], glyphs=pools["glyphs"],
                                   math_pool=pools["math"])
        if figure is None:
            return None
        return fit_ink_to_box(figure, _inset(cell, GEOMETRY_FILL), jitter=0.02)

    if kind == "drawing":
        # always a real sketch: `drawing` is never chosen without a pool to draw
        # from, and there is no scribble fallback any more (see above)
        return fit_ink_to_box(bg.make_drawing(pools["quickdraw"]), cell,
                              jitter=0.02)

    if kind == "table":
        for n_rows, n_cols in TABLE_FIT_SIZES:
            table = bg.make_table(dealer, pools["shapes"], math_pool=pools["math"],
                                  n_rows=n_rows, n_cols=n_cols)
            if table is None:
                continue
            # the table is squeezed or stretched whole into the cell; with the
            # smallest share any cell's writing kept, that may not go below
            # MIN_TEXT_RATIO of the page's handwriting
            text_ratio = table.meta.pop("text_ratio", 1.0)
            fit = bg._fit_scale(table, (x1 - x0, y1 - y0)) * (1.0 - CELL_FIT_JITTER)
            if fit * text_ratio >= bg.MIN_TEXT_RATIO * scale:
                return fit_ink_to_box(table, cell, jitter=CELL_FIT_JITTER)
        # make_table refuses when a cell of it comes up empty, and a table whose
        # writing would be squeezed is not used; write a textblock

    # the bank is what underlines are drawn with
    formula_heavy = bool(pools["math"]) and random.random() >= _text_heavy_prob(local_w)
    block = bg.fill_textblock_cell(dealer, pools["math"], local_w, local_h,
                                   formula_heavy=formula_heavy,
                                   bulleted=random.random() < BULLET_PROB,
                                   minority_prob=MINORITY_ROW_PROB,
                                   shape_bank=pools["shapes"])

    if not block.children and not block.strokes:
        return None
    # content is laid out from the cell's top-left corner in text units
    return _clamp_to_cell(transform_ink(block, scale, x0, y0), cell)


# Connector arrows drawn between text lines after the page is written, the way
# someone links two notes together afterwards.
ARROW_COUNT_WEIGHTS = {0: 0.24, 1: 0.30, 2: 0.22, 3: 0.15, 4: 0.09}
# Pair scoring: a link between two nearby lines is far more likely than one
# spanning the page, so candidate pairs are weighted by exp(-distance / scale)
# with the scale set as a fraction of the page diagonal.
ARROW_DISTANCE_SCALE = 0.22
ARROW_MIN_SPAN = 45.0       # below this the two lines are practically touching

# Connector arrows are kept apart. Left to themselves, the routes on a busy page
# bunch into one region — several long arrows crossing and running alongside each
# other, two or three landing on the same list — which reads as a tangle rather
# than as notes linked together. So an arrow is dropped if any of its ink comes
# within ARROW_MIN_SEPARATION of an arrow already on the page, and a block is the
# end of at most ARROW_MAX_ENDS_PER_BLOCK arrows. Arrows inside a flowchart are the
# flowchart's own and are not affected.
ARROW_MIN_SEPARATION = 45.0     # page units: about two lines of writing
ARROW_MAX_ENDS_PER_BLOCK = 1
ARROW_SPACING_STEP = 8.0        # an arrow's ink is compared at points this far apart
ARROW_TRIES_PER_ARROW = 8       # pairs tried per arrow wanted; routes fail and arrows get dropped


def _ink_points(ink: Ink, step: float = ARROW_SPACING_STEP) -> np.ndarray:
    """(N, 2) points along every stroke of `ink`, no more than `step` apart."""
    parts = []
    for stroke in ink.all_strokes():
        xy = np.column_stack([np.asarray(stroke.x, dtype=float), np.asarray(stroke.y, dtype=float)])
        if len(xy) > 1:
            seg = np.diff(xy, axis=0)
            counts = np.maximum(1, np.ceil(np.hypot(seg[:, 0], seg[:, 1]) / step).astype(int))
            starts = np.repeat(xy[:-1], counts, axis=0)
            fractions = np.concatenate([np.arange(n) / n for n in counts])[:, None]
            parts.append(starts + fractions * np.repeat(seg, counts, axis=0))
        parts.append(xy[-1:])
    return np.concatenate(parts) if parts else np.zeros((0, 2))


def _near_any(points: np.ndarray, others: List[np.ndarray], gap: float) -> bool:
    """Whether any of `points` lies within `gap` of any point in `others`."""
    for other in others:
        if not len(points) or not len(other):
            continue
        if (points[:, 0].min() > other[:, 0].max() + gap or other[:, 0].min() > points[:, 0].max() + gap
                or points[:, 1].min() > other[:, 1].max() + gap or other[:, 1].min() > points[:, 1].max() + gap):
            continue
        for chunk in np.array_split(points, max(1, len(points) // 256)):
            if ((chunk[:, None, :] - other[None, :, :]) ** 2).sum(-1).min() < gap * gap:
                return True
    return False


def _text_lines_and_obstacles(blocks: List[Ink]):
    """Text lines (tagged with the block they belong to) and the no-go boxes.

    Every block on the page is opaque: an arrow may not run through a paragraph,
    a list, a table, a sketch, a flowchart or a figure. Whole blocks, not just
    their lines — threading an arrow between two lines of a paragraph it has
    nothing to do with still crosses that paragraph, and reads as a line struck
    through the middle of someone else's writing.

    Each obstacle is tagged with the block it belongs to so an arrow can be
    routed clear of everything *except* the two blocks it joins; its own two
    ends obviously have to be reachable. The individual text lines are kept as
    obstacles as well, so those two blocks still shield their own other lines.
    """
    lines: List[Tuple[int, Ink]] = []
    # (box, owning block, whether it is the whole block rather than one line)
    obstacles: List[Tuple[tuple, int, bool]] = []

    def walk(node: Ink, index: int, inside_diagram: bool):
        # A formula is written ink like any other and an arrow may not be drawn
        # through it. It is an obstacle but not a connector target: an arrow
        # points from one piece of *prose* to another, and this keeps that as it
        # was before maths had a class of its own to be found by.
        if node.class_name == "formula":
            obstacles.append((node.bbox(), index, False))
            return
        if node.class_name == "textline":
            if not inside_diagram:
                lines.append((index, node))
            obstacles.append((node.bbox(), index, False))
            return
        for child in node.children:
            walk(child, index,
                 inside_diagram or bool((node.meta or {}).get("flowchart")))

    for index, block in enumerate(blocks):
        cls = block.class_name
        obstacles.append((block.bbox(), index, True))
        if cls in ("geometry", "drawing", "table") or (block.meta or {}).get("flowchart"):
            if cls != "table":
                # a sketch, a flowchart or a figure can be one end of a connector
                # as a whole; it anchors to its own ink rather than to a
                # bounding-box side (see _stroke_anchor)
                lines.append((index, block))
            continue
        walk(block, index, False)
    return lines, obstacles


ARROW_ANCHOR_GAP = (3.0, 9.0)   # how far off the ink an arrow starts and stops, drawn per end
# The same, where the end is a flowchart. A connector stopping as close to a
# chart as to a text line read as one of the chart's own arrows; about one and a
# half lines of writing clear of it, it reads as pointing at the chart as a whole.
ARROW_FLOWCHART_GAP = (18.0, 30.0)
# On a text line, how far from the middle of the side an arrow may land, along
# that side, as a share of its length: people aim at the middle of what they
# point at, not anywhere along it.
ARROW_ANCHOR_SPREAD = 0.20
# How far off square an arrow may leave or meet a sketch, flowchart or figure's
# stroke: square to it give or take, not exactly.
ARROW_STROKE_NORMAL_JITTER = math.radians(10.0)
# Where an arrow meets a sketch, flowchart or figure it must point into it: the
# stroke's outward normal has to face the arrow's other end by at least this
# cosine, and the object's outside has to be open there — no more of its ink
# within this share of its size just beyond the stroke.
ARROW_STROKE_MIN_FACING = 0.3
ARROW_EXPOSED_REACH = 0.25
ARROW_SIDE_PROBE = 2.2      # free space a side needs, in multiples of its own height


def _side_is_free(box, normal, obstacles, reach: float) -> bool:
    """Whether there is clear space off one side of a box.

    Used to keep an arrow from launching straight down into the next line of the
    same paragraph: that side is only offered if nothing is sitting in it.
    """
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    edge = (cx + normal[0] * (x1 - x0) / 2, cy + normal[1] * (y1 - y0) / 2)
    tip = (edge[0] + normal[0] * reach, edge[1] + normal[1] * reach)
    for other in obstacles:
        if other == box:
            continue
        if bg._segment_hits_box(edge, tip, other, 1.0):
            return False
    return True


def _box_anchor(box, towards, obstacles, rng: random.Random = random):
    """An anchor point on a box's edge, plus the outward normal of that edge.

    The arrow leaves square to whichever side it uses — vertical off the top or
    bottom, horizontal off the left or right — so it never sets off across the
    line it is attached to. Sides are ranked by how well they face the target,
    but a side with something parked against it is skipped, which is what stops
    an arrow diving into the next line of the same paragraph.

    Along the side, the anchor is within ARROW_ANCHOR_SPREAD of its centre, and
    it stands a random ARROW_ANCHOR_GAP off it.
    """
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    height = max(y1 - y0, 1.0)
    reach = height * ARROW_SIDE_PROBE

    sides = [
        ((0.0, -1.0), cy - towards[1]),     # up
        ((0.0, 1.0), towards[1] - cy),      # down
        ((-1.0, 0.0), cx - towards[0]),     # left
        ((1.0, 0.0), towards[0] - cx),      # right
    ]
    sides.sort(key=lambda s: -s[1])         # best-facing first

    for normal, facing in sides:
        if facing <= 0:
            continue                        # points away from the target
        if not _side_is_free(box, normal, obstacles, reach):
            continue
        # near the middle of the side — within ARROW_ANCHOR_SPREAD of its centre,
        # along it — and a small random distance off it
        gap = rng.uniform(*ARROW_ANCHOR_GAP)
        along = rng.uniform(-ARROW_ANCHOR_SPREAD, ARROW_ANCHOR_SPREAD)
        if normal[1]:
            point = (cx + along * (x1 - x0), y1 + gap if normal[1] > 0 else y0 - gap)
        else:
            point = (x1 + gap if normal[0] > 0 else x0 - gap, cy + along * (y1 - y0))
        return point, normal
    return None, None


def _stroke_anchor(ink: Ink, towards, rng: random.Random = random, gap_range=ARROW_ANCHOR_GAP):
    """Anchor on an object's own ink, with the arrow pointing straight into it.

    A sketch, a flowchart or a geometry figure has no bounding-box side to leave
    from, so the arrow attaches to its ink, perpendicular to the stroke there
    and pointing *into the object*. Perpendicular alone was not enough: taking
    whichever stroke was nearest the other end, with its normal flipped toward
    that end, could land on a stroke inside the object (a leaf, a chart's own
    arrow) or one running toward the arrow, so the arrow skimmed along the ink
    rather than pointing in.

    So each sampled point's normal is oriented outward, away from the object's
    centre, and the arrow travels against it. A point is only used if that
    outward normal faces the arrow's other end (ARROW_STROKE_MIN_FACING), so the
    arrow does not have to come round the object to point in, and if the outside
    is open there — none of the object's ink lies just beyond the stroke along
    the normal (ARROW_EXPOSED_REACH) — so it meets the object's outer edge. Of
    those, the one nearest the other end is used. The direction is then turned
    by a small random amount (ARROW_STROKE_NORMAL_JITTER), since nobody draws an
    arrow exactly square to a line, and the anchor stands a random
    ARROW_ANCHOR_GAP off the stroke.
    """
    samples = []
    for stroke in ink.all_strokes():
        n = len(stroke.x)
        step = max(1, n // 40)
        for i in range(0, n, step):
            j, k = min(i + 1, n - 1), max(i - 1, 0)
            tx, ty = stroke.x[j] - stroke.x[k], stroke.y[j] - stroke.y[k]
            length = math.hypot(tx, ty)
            if length > 1e-9:
                samples.append((stroke.x[i], stroke.y[i], -ty / length, tx / length))
    if not samples:
        return None, None

    x0, y0, x1, y1 = ink.bbox()
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    reach = max(x1 - x0, y1 - y0, 1e-6) * ARROW_EXPOSED_REACH
    ink_points = [(sx, sy) for sx, sy, _, _ in samples]

    candidates = []
    for px, py, nx, ny in samples:
        if nx * (px - cx) + ny * (py - cy) < 0:
            nx, ny = -nx, -ny               # outward, away from the object's centre
        dx, dy = towards[0] - px, towards[1] - py
        distance = math.hypot(dx, dy) or 1e-9
        if (nx * dx + ny * dy) / distance >= ARROW_STROKE_MIN_FACING:
            candidates.append((distance, px, py, nx, ny))
    candidates.sort()

    chosen = None
    for distance, px, py, nx, ny in candidates:
        # is more of the object just beyond the stroke along the normal?
        blocked = any(0.5 < (qx - px) * nx + (qy - py) * ny < reach
                      and abs((qx - px) * ny - (qy - py) * nx) < reach * 0.25
                      for qx, qy in ink_points)
        if not blocked:
            chosen = (px, py, nx, ny)
            break
    if chosen is None:
        # nowhere open faces the other end: the nearest point, pointing in from
        # the side the arrow comes from
        px, py, nx, ny = min(samples, key=lambda s: math.hypot(s[0] - towards[0],
                                                               s[1] - towards[1]))
        if nx * (towards[0] - px) + ny * (towards[1] - py) < 0:
            nx, ny = -nx, -ny
        chosen = (px, py, nx, ny)
    px, py, nx, ny = chosen
    normal = (nx, ny)

    # square to the stroke give or take a little, not exactly, and a small random
    # distance off it
    turn = max(-ARROW_STROKE_NORMAL_JITTER,
               min(ARROW_STROKE_NORMAL_JITTER, rng.gauss(0.0, ARROW_STROKE_NORMAL_JITTER / 2)))
    c, s = math.cos(turn), math.sin(turn)
    normal = (normal[0] * c - normal[1] * s, normal[0] * s + normal[1] * c)
    gap = rng.uniform(*gap_range)
    point = (px + normal[0] * gap, py + normal[1] * gap)
    return point, normal


def _anchor_for(node: Ink, towards, obstacles, rng: random.Random = random):
    """Pick the right anchoring strategy for whatever the arrow is attached to:
    the side of a text line's box, or the nearest stroke of a sketch, flowchart
    or geometry figure. A flowchart end stands further off (ARROW_FLOWCHART_GAP)."""
    if (node.meta or {}).get("flowchart"):
        return _stroke_anchor(node, towards, rng, ARROW_FLOWCHART_GAP)
    if node.class_name in ("drawing", "geometry"):
        return _stroke_anchor(node, towards, rng)
    return _box_anchor(node.bbox(), towards, obstacles, rng)


def _add_connector_arrows(blocks: List[Ink], page_w: float, page_h: float,
                           rng: random.Random = random, shape_bank=None) -> List[Ink]:
    """Link a few pairs of text lines with routed arrows, drawn in mined ink
    from `shape_bank` — so without a bank there are none."""
    wanted = rng.choices(list(ARROW_COUNT_WEIGHTS),
                         weights=list(ARROW_COUNT_WEIGHTS.values()))[0]
    if not wanted or shape_bank is None or not shape_bank.ok():
        return []
    # Loops are a habit of the writer, so they are decided once for the page:
    # usually none at all, and on the odd page most of its arrows have one.
    loop_prob = (bg.ARROW_LOOP_PROB_ON_PAGE if rng.random() < bg.ARROW_LOOP_PAGE_PROB
                 else 0.0)

    lines, obstacles = _text_lines_and_obstacles(blocks)
    if len(lines) < 2:
        return []

    scale = ARROW_DISTANCE_SCALE * math.hypot(page_w, page_h)
    pairs, weights = [], []
    for i, (block_a, a) in enumerate(lines):
        ax0, ay0, ax1, ay1 = a.bbox()
        for block_b, b in lines[i + 1:]:
            # Only ever link two different blocks. Two lines inside one
            # paragraph are the closest pairs on the page and so would dominate
            # the sampling, but they are also hemmed in by their own
            # neighbours — nearly every route between them gets rejected, and
            # an arrow from a paragraph to itself is not what this is for.
            if block_a == block_b:
                continue
            bx0, by0, bx1, by1 = b.bbox()
            distance = math.hypot((ax0 + ax1) / 2 - (bx0 + bx1) / 2,
                                   (ay0 + ay1) / 2 - (by0 + by1) / 2)
            if distance < ARROW_MIN_SPAN:
                continue
            pairs.append(((block_a, a), (block_b, b)))
            weights.append(math.exp(-distance / scale))
    if not pairs:
        return []

    arrows: List[Ink] = []
    drawn: List[np.ndarray] = []         # each arrow's ink as points (ARROW_MIN_SEPARATION)
    block_ends: Counter = Counter()      # arrows ending on each block (ARROW_MAX_ENDS_PER_BLOCK)
    used: set = set()
    for _ in range(wanted * ARROW_TRIES_PER_ARROW):     # extra tries; routes fail and arrows are dropped
        if len(arrows) >= wanted or not any(weights):
            break
        (source_block, source), (target_block, target) = rng.choices(
            pairs, weights=weights)[0]
        # Unordered, so a pair is linked at most once *in either direction*. An
        # ordered key let the same two lines get both an A->B and a B->A arrow;
        # since both are routed between the same two anchors they came out on top
        # of each other, reading as one confused double-headed scrawl.
        key = frozenset((id(source), id(target)))
        if key in used:
            continue
        used.add(key)
        if rng.random() < 0.5:
            source, target = target, source     # either direction
            source_block, target_block = target_block, source_block
        source_box, target_box = source.bbox(), target.bbox()
        # An arrow has to be able to leave the two things it joins, so their own
        # blocks' bounding boxes are not obstacles for it — but every other
        # block on the page is, whole.
        # Only the *whole-block* box is lifted for the two ends. Their
        # individual lines stay in, so an arrow leaving a paragraph still has to
        # get past that paragraph's other lines rather than straight through
        # them, and the two lines actually being joined are lifted by name.
        ends = (source_block, target_block)
        others = [box for box, owner, is_block in obstacles
                  if not (is_block and owner in ends)
                  and box != source_box and box != target_box]

        def centre(box):
            return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

        start, start_normal = _anchor_for(source, centre(target_box), others, rng)
        end, end_normal = _anchor_for(target, centre(source_box), others, rng)
        if start is None or end is None:
            continue        # every side of one of them was boxed in

        # Both normals point outward, away from the thing they are attached to.
        # The end stub therefore sits outside the target and the arrow drops
        # onto it square, rather than running along it.
        # The writing area, not the paper: every block is laid out inside the
        # margin, so an arrow that strays into it reads as a printing fault even
        # though it is technically still on the page.
        arrow = bg.make_connector_arrow(start, end, others, rng,
                                         start_normal=start_normal,
                                         end_normal=end_normal,
                                         shape_bank=shape_bank,
                                         loop_prob=loop_prob,
                                         bounds=(ly.MARGIN, ly.MARGIN,
                                                 page_w - ly.MARGIN,
                                                 page_h - ly.MARGIN))
        if arrow is None:
            continue
        points = _ink_points(arrow)
        if _near_any(points, drawn, ARROW_MIN_SEPARATION):
            continue        # too close to, or across, an arrow already drawn
        arrows.append(arrow)
        drawn.append(points)
        block_ends[source_block] += 1
        block_ends[target_block] += 1
        full = {b for b in (source_block, target_block) if block_ends[b] >= ARROW_MAX_ENDS_PER_BLOCK}
        if full:            # a block with its arrows is no longer offered as an end
            weights = [0.0 if (a[0] in full or b[0] in full) else w for (a, b), w in zip(pairs, weights)]
    return arrows


def _clamp_to_cell(block: Ink, cell: Tuple[float, float, float, float]) -> Ink:
    """Keep a block inside its cell, whatever the generator produced.

    The generators size their content to the cell, but a few things can still
    poke out — a descender on the last line, an expression taller than the row
    it was dropped into. Nudging the block back in (and, only if it is genuinely
    too big, scaling it) means nothing ever collides with a neighbouring cell or
    runs off the page, without every generator having to be perfect.
    """
    cx0, cy0, cx1, cy1 = cell
    bx0, by0, bx1, by1 = block.bbox()
    width, height = bx1 - bx0, by1 - by0
    if width <= 0 or height <= 0:
        return block

    scale = min(1.0, (cx1 - cx0) / width, (cy1 - cy0) / height)
    if scale < 1.0:
        block = transform_ink(block, scale, bx0 - bx0 * scale, by0 - by0 * scale)
        bx0, by0, bx1, by1 = block.bbox()

    dx = max(0.0, cx0 - bx0) - max(0.0, bx1 - cx1)
    dy = max(0.0, cy0 - by0) - max(0.0, by1 - cy1)
    if dx or dy:
        block = transform_ink(block, 1.0, dx, dy)
    return block


def generate_page(text_pool: List[Ink],
                   quickdraw_pool: List[Ink],
                   ohfd_pool: Optional[List[Ink]] = None,
                   math_pool: Optional[List[Ink]] = None,
                   shape_bank: Optional["gd.Bank"] = None,
                   glyph_pool: Optional[Dict[str, List[Ink]]] = None,
                   enclosure_prob: float = ENCLOSURE_PROB,
                   _attempt: int = 0) -> Dict:
    """
    text_pool:      line-level Ink from IAM-OnDB / DeepWriting loaders
    quickdraw_pool: sketch-level Ink from QuickDraw loader
    ohfd_pool:      diagram-level Ink from OHFD loader (optional; falls back
                     to procedural diagrams if not provided)
    math_pool:      expression-level Ink from MathWriting loader (optional;
                     fills whole maths sections and drops the odd expression
                     into a paragraph)
    glyph_pool:     character-level Ink keyed by character, from
                     loaders.load_emnist_pool (optional; it is what a geometry
                     figure names its corners and writes its lengths with, so
                     without it the figures come out unlabelled)
    shape_bank:     mined line/oval ink from geometry_diagrams.build_bank,
                     which geometry figures are drawn from (optional; defaults to
                     the cache at SHAPES_BANK_CACHE, and no cache means no
                     geometry rather than a multi-minute mine mid-run)

    Returns a dict with the rendered blocks, page size, JSON labels and
    per-level target strings ready for training.
    """
    # Prose comes from one hand (two at most), dealt without replacement so a
    # page doesn't show the same sentence twice.
    dealer = bg.LineDealer(_single_writer_pool(text_pool))

    cells = ly.build_grid(page_w=ly.PAGE_W, page_h=ly.PAGE_H)
    scale = ly.page_text_scale()
    min_h = ly.min_text_height(scale)

    if shape_bank is None:
        shape_bank = shape_bank_default()

    pools = {"ohfd": ohfd_pool, "quickdraw": quickdraw_pool, "math": math_pool,
             "shapes": shape_bank, "glyphs": glyph_pool}
    bank_ok = shape_bank is not None and shape_bank.ok()
    have = {
        "flowchart": True,          # falls back to a procedural diagram
        "geometry": bank_ok,
        "drawing": bool(quickdraw_pool),
        "table": bank_ok and bool(text_pool),   # its rules are drawn from the bank
        "textblock": bool(text_pool) or bool(math_pool),
    }

    usable_w = ly.PAGE_W - 2 * ly.MARGIN
    usable_h = ly.PAGE_H - 2 * ly.MARGIN
    kinds: List[str] = []
    for x0, y0, x1, y1 in cells:
        cell_have = dict(have)
        if y1 - y0 < min_h:
            cell_have["textblock"] = False      # not even one row fits
        kinds.append(_choose_kind((x1 - x0) / usable_w, (y1 - y0) / usable_h, cell_have))

    blocks: List[Ink] = []
    page_charts: set = set()        # the OHFD charts on this page: none is placed twice
    for cell_index, (cell, kind) in enumerate(zip(cells, kinds)):
        block = _fill_cell(kind, cell, scale, dealer, pools, page_charts)
        if block is None:
            continue
        # The order the cell was created in (build_grid's seed order), which is
        # what level-2 targets are ordered by (seg_prompts). It survives an
        # enclosure group wrapping the block, since it sits on the block itself.
        block.meta = dict(block.meta or {}, cell_index=cell_index)
        blocks.append(block)

    # Boxes and ovals round writing go on once every block is placed, so each can
    # be kept off all its neighbours rather than only the ones drawn before it.
    for index, block in enumerate(blocks):
        if block.class_name == "textblock":
            others = [other.bbox() for j, other in enumerate(blocks) if j != index]
            blocks[index] = bg.maybe_enclose(block, shape_bank, probability=enclosure_prob,
                                             keep_clear=others,
                                             bounds=(0.0, 0.0, ly.PAGE_W, ly.PAGE_H))

    if not blocks:      # guarantee a page is never empty
        fallback = bg.fill_text_cell(dealer, 40.0, 20.0)
        blocks.append(transform_ink(fallback, scale, ly.MARGIN, ly.MARGIN))

    # No two lines — textlines and formulas, in any pairing — may share more than
    # bg.LINE_MAX_OVERLAP of the smaller one's box. Every generator keeps to it; should a
    # page not, it is built again.
    if bg.line_overlaps(blocks):
        if _attempt + 1 >= LINE_OVERLAP_ATTEMPTS:
            raise RuntimeError(f"{LINE_OVERLAP_ATTEMPTS} pages in a row had lines overlapping by more "
                               f"than {bg.LINE_MAX_OVERLAP:.0%} of the smaller one")
        return generate_page(text_pool, quickdraw_pool, ohfd_pool, math_pool, shape_bank, glyph_pool,
                             enclosure_prob, _attempt=_attempt + 1)

    # Arrows go on last, once every block they might connect is in place.
    blocks.extend(_add_connector_arrows(blocks, ly.PAGE_W, ly.PAGE_H,
                                        shape_bank=shape_bank))

    # The page is built at the working size and handed on at OUTPUT_SIZE: the
    # ink, and everything measured from it — boxes, targets — is
    # scaled together, so the labels describe the 224 x 224 page, not the one it
    # was laid out on. transform_ink also carries each block's exact
    # construction (table grids, geometry) with it.
    shrink = ly.OUTPUT_SIZE / ly.PAGE_W
    placed = [transform_ink(block, shrink, 0.0, 0.0) for block in blocks]
    size = ly.OUTPUT_SIZE

    labels = extract_labels(placed)
    targets = {
        lvl: format_target_string(items, size, size)
        for lvl, items in labels.items()
    }

    return {
        "blocks": placed,
        "page_w": size,
        "page_h": size,
        "labels_json": export_json(placed, size, size),
        "targets_by_level": targets,
        # the same page before the shrink: a level-2 object is cut out of this
        # board, not the 224 page, before it is scaled up to its own 224 image
        "board_blocks": blocks,
        "board_size": ly.PAGE_W,
    }
