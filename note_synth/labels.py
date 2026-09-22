"""
labels.py — walk a placed (page-coordinate) Ink tree and extract hierarchical
detection labels at each of the three taxonomy levels, then format them either
as JSON (general purpose) or as the "x0 y0 x1 y1 class ..." style target string
used in the InkFM paper's segmentation task.

There are no relations between objects: a page is its per-level boxes, and a
`vertex` carries a few fields of its own beyond its box (see export_json).

NOTE on coordinate convention: the paper doesn't specify byte-for-byte whether
its "38 41 67 273 textline" example is (x0 y0 x1 y1) or (y0 x0 y1 x1) — PaliGemma's
native "detect" task historically uses <locXXXX> tokens in y0 x0 y1 x1 order on a
0-1023 grid. Verify against PaliGemma's documented detect format before training,
and adjust `COORD_ORDER` below accordingly — it's a one-line change.
"""

from __future__ import annotations
from typing import List, Tuple, Dict
import json

from core import Ink

# A bulleted list is a `textblock`: the same lines, with bullets that carry no
# class in front of them.
#
# A `flowchart` is boxed as a whole. Its shapes and arrows are level 0; its
# captions — in a shape, or the "yes" on an arrow — are its lines.
#
# Every line of writing is inside a textblock, a table, a geometry figure or a
# flowchart: a table cell's line, a label written on a figure (a corner's name,
# a length, an angle) and a flowchart's caption sit in the table, figure or
# chart directly. So line detection runs inside all four, and is not needed for
# a textblock that is a single line (is_single_line).
LEVEL2 = {"textblock", "geometry", "drawing", "table", "flowchart"}
# `formula` is a row of maths — a MathWriting expression — and sits beside
# `textline` rather than under it. The two are genuinely different content and
# the ink says which is which, but only if the generator commits to one: written
# out by hand, "x = i+1" is a formula and "if i is bigger" is a line of prose,
# and there is no reading of the strokes alone that separates them reliably.
# Every expression placed on a page is therefore labelled `formula`, including
# the rows of a maths section and the contents of a flowchart shape, so the
# class means the same thing everywhere it appears.
# `vertex` is a vertex of a geometry figure: a point where its lines end or meet,
# so its box has no area. Every one a figure has is emitted, named or not, and it
# carries what makes it a vertex rather than a dot — its angles, how they are
# marked, and which vertices it is joined to (geometry_diagrams.figure_vertices;
# the fields are listed at export_json).
# `arc` and `segment` are the rest of a geometry figure: every circle or arc
# of one it draws, as its start, middle and end going counterclockwise, and
# every straight line, as its two ends (geometry_diagrams.figure_parts). Marks — right-angle boxes, angle arcs, the
# ticks themselves — are neither. There are no ellipses.
# `enclosure` is every shape drawn round something — a flowchart's box, stadium,
# diamond, parallelogram or circle, and a box or oval drawn round a block, a line
# or some words, alike (a table's outer border is not labelled). Which shape it is is kept for
# rendering (meta["ohfd_type"], or the label: "rectangle", "oval"), not as a class.
# `line_mark` is a line drawn under or through writing — an underline (single or
# double) or a strikethrough — one class for both, with which it is kept as the
# label. Both are level 1, held by the object whose writing they mark (a
# textblock or a flowchart), and hold nothing; what they mark keeps its
# own place — a line is never split round the words it marks. Each records the
# text it marks (meta["attends_row"], or a flowchart shape's caption_key), which
# orders it among the level-1 targets.
LEVEL1 = {"textline", "formula", "arc", "segment", "vertex", "enclosure", "line_mark"}
# `unknown` is ink that is none of the other classes — a crossing out, an
# abandoned mark. Nothing currently GENERATES it: the procedural scribbles that
# used to fill it were invented ink rather than anything a source dataset
# carries, and teaching a class off those was worse than leaving it empty (see
# page_generator). The class is kept so real unnameable ink has a name waiting
# for it, but expect zero of these on a generated page today.
# `arrow` is the one object that stands on the page on its own: a flowchart's
# edge or a connector drawn between two blocks.
LEVEL0 = {"word", "arrow", "unknown"}

COORD_ORDER = "x0y0x1y1"  # or "y0x0y1x1" to match PaliGemma's native detect convention

# is_single_line: a block is one line when its stroke at this height percentile
# is at least this share of the block's height. Chosen on 150 pages (1,111
# single-line and 323 multi-line blocks) as the percentile catching the most
# single lines once the threshold sits 1.5x above the highest value any
# multi-line block reached (30.9% at the 89th percentile). On 150 unseen pages
# it made 0 false positives of 352 multi-line blocks and caught 77.9% of 1,054
# single lines. Thresholds with less headroom did not hold up: right at the
# worst multi-line value, and at 1.1x, both made false positives on the unseen
# pages. Averaging over an 8-percentile window instead was no better (74.0% at
# the same headroom).
SINGLE_LINE_PERCENTILE = 89
SINGLE_LINE_MIN_RATIO = 0.463


def is_single_line(strokes, percentile: float = SINGLE_LINE_PERCENTILE,
                   min_ratio: float = SINGLE_LINE_MIN_RATIO) -> bool:
    """Whether a block's ink is a single line of writing, so line detection can
    be skipped for it.

    The strokes are sorted by height and the one at `percentile` taken (nearest
    rank: index round(percentile/100 * (n - 1))); the block is one line when that
    stroke is at least `min_ratio` of the height of the box round all of them. A
    line's taller strokes span most of its height, while no stroke in a block of
    several lines comes close to the whole block's height.

    The rule is one-sided on purpose: calling a multi-line block a single line
    would skip its line detection, so the threshold sits above every multi-line
    block seen, with headroom, and some single lines are left for line detection
    to confirm. A median at 70% (the first rule tried) had no false positives
    either but caught only 15% of single lines; no per-stroke rule separates the
    two at all, since every single line has small strokes — dots, commas — that
    look exactly like a multi-line block's.

    Pass the block's own writing only. An enclosure, a line mark or an arrow drawn over a
    block changes what it means, not what it is, and its strokes can be as tall
    (or, for an underline, as wide) as the whole block, which would skew the
    percentile either way.
    """
    inked = [s for s in strokes if s.y]
    if not inked:
        return False
    top = min(min(s.y) for s in inked)
    bottom = max(max(s.y) for s in inked)
    if bottom - top <= 1e-9:
        return True
    heights = sorted(max(s.y) - min(s.y) for s in inked)
    at = heights[int(round(percentile / 100.0 * (len(heights) - 1)))]
    return at >= min_ratio * (bottom - top)


def _level_of(class_name: str) -> int:
    if class_name in LEVEL2:
        return 2
    if class_name in LEVEL1:
        return 1
    if class_name in LEVEL0:
        return 0
    return -1  # unlabeled structural node (e.g. plain textblock container before relabel)


def label_bbox(node: Ink) -> Tuple[float, float, float, float]:
    """The box a labelled object is given: its ink's own box.

    A geometry figure's box takes in the labels written on it — a corner's name,
    a length, an angle — since they are lines of the figure, as a table's lines
    are of the table.
    """
    return node.bbox()


def extract_labels(placed_blocks: List[Ink]) -> Dict[int, List[Tuple[Tuple[float, float, float, float], str]]]:
    """Returns {level: [(bbox, class_name), ...]} across the whole page."""
    out = {0: [], 1: [], 2: []}

    def walk(node: Ink):
        if node.class_name:
            lvl = _level_of(node.class_name)
            if lvl >= 0:
                out[lvl].append((label_bbox(node), node.class_name))
        for c in node.children:
            walk(c)

    for b in placed_blocks:
        walk(b)
    return out


def _quantize(box: Tuple[float, float, float, float], page_w: float, page_h: float,
              grid: int = 1000) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return (int(x0 / page_w * grid), int(y0 / page_h * grid),
            int(x1 / page_w * grid), int(y1 / page_h * grid))


def format_target_string(labels_at_level: List[Tuple[Tuple[float, float, float, float], str]],
                          page_w: float, page_h: float, grid: int = 1000) -> str:
    """
    Produces "x0 y0 x1 y1 class x0 y0 x1 y1 class ..." (or y0x0y1x1 order per
    COORD_ORDER), grouped by class as the paper recommends (Eq. 1: list all
    objects of one class together, in the same class order as the prompt).
    """
    by_class: Dict[str, List[Tuple[int, int, int, int]]] = {}
    for box, cls in labels_at_level:
        q = _quantize(box, page_w, page_h, grid)
        by_class.setdefault(cls, []).append(q)

    parts = []
    for cls, boxes in by_class.items():
        for (a, b, c, d) in boxes:
            if COORD_ORDER == "y0x0y1x1":
                parts.append(f"{b} {a} {d} {c} {cls}")
            else:
                parts.append(f"{a} {b} {c} {d} {cls}")
    return " ".join(parts)


VERTEX_FIELDS = ("id", "angles", "angle_mark", "angle_mark_count", "connected")


def export_json(placed_blocks: List[Ink], page_w: float, page_h: float) -> str:
    """Every labelled object on the page as {level, class, bbox}, level 0 first.

    A `vertex` adds what its box cannot say: `position` ([x, y], the point
    itself), `id` (unique on the page), `angles` (degrees, one per angle it makes
    inside the figure), `angle_mark` ("none", "right" or "arc"),
    `angle_mark_count` (how many arcs, or 1 for a right-angle box) and
    `connected` (the ids of the vertices a drawn line or arc joins it to).

    An `arc` adds `points`, [x1, y1, x2, y2, x3, y3]: its start, middle and
    end, counterclockwise. A `segment` adds `points`, [x1, y1, x2, y2], its two
    ends.
    """
    objects = []

    def walk(node: Ink):
        level = _level_of(node.class_name) if node.class_name else -1
        if level >= 0:
            box = label_bbox(node)
            entry = {"level": level, "class": node.class_name, "bbox": list(box)}
            if node.class_name == "vertex":
                entry["position"] = [box[0], box[1]]
                entry.update({key: (node.meta or {}).get(key) for key in VERTEX_FIELDS})
            elif node.class_name in ("arc", "segment"):
                entry["points"] = [v for s in node.strokes for v in (s.x[0], s.y[0])]
            objects.append(entry)
        for child in node.children:
            walk(child)

    for block in placed_blocks:
        walk(block)
    objects.sort(key=lambda entry: entry["level"])
    return json.dumps({"page_size": [page_w, page_h], "objects": objects}, indent=2)
