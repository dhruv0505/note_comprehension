"""
seg_prompts.py — the segmentation prompts asked of a page, the 224 x 224 image
each is asked of, and the target each should produce.

    python3 seg_prompts.py --check-order          # spot-check level-2 creation order on 25 pages

A page is asked two things as a whole, and one thing about each level-2 object
on it:

    full page     "Where are the level 2 objects? Detect multiple textblock, drawing, geometry, table, flowchart."
                  "Where are the arrows? Detect multiple arrow."
    textblock     "Where are the level 1 objects in textblock? Detect multiple textline, formula, enclosure, line mark."
    table         "Where are the level 1 objects in table? Detect multiple textline, formula."
    geometry      "Where are the level 1 objects in geometry? Detect multiple textline, arc, segment, vertex."
    flowchart     "Where are the level 1 objects in flowchart? Detect multiple textline, formula, enclosure."
    drawing       nothing — no level-1 prompt is defined for a drawing

Images. The page is rasterised on the 1000 x 1000 board it was laid out on
(render_board) and downsized to 224 x 224. A level-2 object's image is its box
cut out of that same board raster, scaled so its longer side is 224, and centred
on a white 224 x 224 square (crop_transform). Every sample keeps its transform,
image = board * scale + offset, so its coordinates can be taken back to the
board: a level-1 target is in its parent's crop, not in the page, and the two
are not comparable without it.

Targets. Each object is its coordinates then its class, all objects joined by
spaces. A box is its top-left then bottom-right corner, a vertex is its one
point, and an arrow is its start then its end — never reordered, since that is
its direction. An arc (a whole circle or part of one; there are no ellipses) is three
points, its start, middle and end, going counterclockwise as the image is seen:
"x1 y1 x2 y2 x3 y3 arc", a full circle starting and ending at its 0-degree
point with its 180-degree point between. A segment is its two ends, the upper
one first: "x1 y1 x2 y2 segment". Each coordinate is on a GRID-step scale over
the 224 image, x before y unless labels.COORD_ORDER says otherwise. Targets are grouped by class
in the order the prompt names them (PaliGemma's convention); within a class:

    level 2      the order the cell was created in (page_generator tags each
                 block with meta["cell_index"]), then the order in the block's
                 tree (LEVEL2_ORDER; "coords" sorts by exact (y0, x0) instead)
    level 1      creation order: a textblock's rows are laid top to bottom, a
                 table's cells row by row
    geometry     a figure's textlines, arcs, segments and vertices by exact
                 (y, x) of the top-left corner, the first point or the point,
                 since its labels are created going round the figure, not down it
    flowchart    a chart's textlines and formulas by exact (y, x) of the
                 top-left corner, since its captions come in the source chart's
                 order, not down it
    enclosure,   by the text each marks: in a textblock, the row it marks
    line mark    counted down the block (a box round the whole block first), then
                 left to right; in a flowchart, a shape by its caption's (y, x),
                 or its own if it has none
    arrow        exact (y, x) of its start

What counts as what:
  * A table holds its cells' textlines and formulas directly, with no
    textblock between; they are the table's level-1 objects.
  * A label written on a geometry figure is a textline held by the figure, inside
    its box, as a table holds its lines. So are its arcs, segments and
    vertices (geometry_diagrams.figure_parts, figure_vertices); the marks on it
    — right-angle boxes, angle arcs, ticks — are not objects. (An angle in radians is a formula, which
    the geometry prompt does not ask for; the shipped MathWriting excerpt has
    none.)
  * A flowchart's captions — in a shape or on an arrow — are textlines and
    formulas held by the chart, and its shapes are enclosures it holds.
  * Enclosures and line marks are level-1 objects of the textblock or flowchart
    whose writing they mark. A table's outer border is not labelled at all.
  * Arrows — a flowchart's edges and connectors between blocks — are the only
    objects asked for across the page besides level 2.
"""

from __future__ import annotations
import itertools
from typing import Dict, List, Tuple

import numpy as np

import layout as ly
from labels import COORD_ORDER, label_bbox
from make_pdf import _arrow_span, _graph_span

BOARD_SIZE = int(ly.PAGE_W)          # the board a page is laid out and rasterised on
IMAGE_SIZE = int(ly.OUTPUT_SIZE)     # every image a prompt is asked of


def configure(board: float = None, image: float = None) -> None:
    """Set the board a page is laid out on and the image every prompt is asked of.

    Sets layout's page and output size too, which page_generator reads each time
    it builds a page, so a page, its images and its targets all agree. Call it
    before generating anything.
    """
    global BOARD_SIZE, IMAGE_SIZE
    if board:
        ly.PAGE_W = ly.PAGE_H = float(board)
    if image:
        ly.OUTPUT_SIZE = float(image)
    BOARD_SIZE, IMAGE_SIZE = int(ly.PAGE_W), int(ly.OUTPUT_SIZE)
GRID = 1000                          # coordinate steps across an image (labels.format_target_string's)
INK_WIDTH_PX = 2.0                   # pen width on the board raster, in board pixels

LEVEL2_PROMPT = ("Where are the level 2 objects? "
                 "Detect multiple textblock, drawing, geometry, table, flowchart.")
ARROW_PROMPT = "Where are the arrows? Detect multiple arrow."
LEVEL1_PROMPTS = {
    "textblock": ("Where are the level 1 objects in textblock? "
                  "Detect multiple textline, formula, enclosure, line mark."),
    "table": ("Where are the level 1 objects in table? "
              "Detect multiple textline, formula."),
    "geometry": ("Where are the level 1 objects in geometry? "
                 "Detect multiple textline, arc, segment, vertex."),
    "flowchart": ("Where are the level 1 objects in flowchart? "
                  "Detect multiple textline, formula, enclosure."),
}

# Classes in the order their prompt names them, which is the order targets are
# grouped in.
LEVEL2_CLASSES = ("textblock", "drawing", "geometry", "table", "flowchart")
LEVEL1_CLASSES = {"textblock": ("textline", "formula", "enclosure", "line_mark"),
                  "table": ("textline", "formula"),
                  "geometry": ("textline", "arc", "segment", "vertex"),
                  "flowchart": ("textline", "formula", "enclosure")}
# A class is written in a target the way its prompt spells it.
CLASS_TOKEN = {"line_mark": "line mark"}

# "creation" (cell creation order, then tree order) or "coords" (exact y0, x0).
# Creation order, as checked on pages biased to 17+ cells, over every pair of
# level-2 objects whose y-ranges do not overlap:
#   * the grid alone (640,126 layouts, 33M pairs): 0 pairs where the later cell
#     sits wholly above the earlier. Seeds leave the heap in non-decreasing y, so
#     the seed/collision system cannot put a later cell above an earlier one.
#   * real level-2 boxes (300 pages, 31,030 pairs, flowchart a level-2 object):
#     5 such pairs (0.016%), on 5 pages (1.7%) — each a table, figure or drawing
#     fitted and centred low in its cell beside a later, top-anchored textblock.
LEVEL2_ORDER = "creation"

_NO_CELL = 10 ** 9      # a block no cell made (the empty-page fallback) goes last


# ----------------------------------------------------------------------------
# What is on the page
# ----------------------------------------------------------------------------

def _box_distance(points, box):
    """Each point's distance to `box` (0 inside it)."""
    x0, y0, x1, y1 = box
    dx = np.maximum(np.maximum(x0 - points[:, 0], 0), points[:, 0] - x1)
    dy = np.maximum(np.maximum(y0 - points[:, 1], 0), points[:, 1] - y1)
    return np.hypot(dx, dy)


def _snap_to_ink(node, start, end, boxes=None):
    """The arrow's own ink nearest where it was meant to start and end.

    A flowchart arrow's span comes from the graph (make_pdf._graph_span): the
    outlines of the two shapes it joins, facing each other. That settles which way
    it points, but the drawn arrow stops short of those points or runs past them,
    so the label is moved onto the ink. Snapping to the drawn point nearest those
    outline points put the end partway along the shaft whenever the arrow bent
    round a corner, since the point facing the other shape's centre is then beside
    the shaft rather than at the head. With the two shapes' boxes given, the end is
    the drawn point nearest the target's box and the start the one nearest the
    source's, the intended point breaking ties. Stroke ends alone are not enough:
    a head drawn in the same stroke as its shaft has its tip mid-stroke. A
    connector's span is already its shaft's first and last points, which snap to
    themselves."""
    ends = np.array([(x, y) for stroke in node.all_strokes() for x, y in zip(stroke.x, stroke.y)],
                    dtype=float)
    if len(ends) < 2:
        return start, end

    def cost(p, box):
        to_point = np.sqrt(((ends - np.asarray(p, dtype=float)) ** 2).sum(1))
        return to_point if box is None else _box_distance(ends, box) + 0.25 * to_point

    source_box, target_box = boxes or (None, None)
    cost_start, cost_end = cost(start, source_box), cost(end, target_box)
    j = int(np.argmin(cost_end))
    i = int(np.argmin(np.where(np.all(ends == ends[j], axis=1), np.inf, cost_start)))
    if not np.isfinite(cost_start[i]) or np.all(ends[i] == ends[j]):
        return start, end   # too small to tell apart on the ink: keep the intended ends
    return tuple(ends[i]), tuple(ends[j])


def _arrow(node, shapes) -> Dict:
    """An arrow as its start and end where its ink actually begins and ends, in
    the direction the graph or the drawing gives it (make_pdf._graph_span,
    _arrow_span, then _snap_to_ink)."""
    start = end = boxes = None
    if shapes:
        start, end = _graph_span(node, shapes)
        if start is not None:
            meta = node.meta or {}
            boxes = (shapes[meta["source_id"]].bbox(), shapes[meta["target_id"]].bbox())
    if start is None:
        start, end = _arrow_span(node)
    if start is not None and end is not None:
        start, end = _snap_to_ink(node, start, end, boxes)
        return {"class": "arrow",
                "board": [float(start[0]), float(start[1]), float(end[0]), float(end[1])]}
    # no way to tell which end is which: its box, flagged
    return {"class": "arrow", "board": [float(v) for v in node.bbox()],
            "no_direction": True}


def _collect(blocks) -> Tuple[List[Dict], List[Dict]]:
    """Every level-2 object and arrow on the page, unordered.

    A level-2 object carries its node (for its level-1 objects), its board box
    and `created`: its cell's creation index, then its place in the walk.
    """
    level2: List[Dict] = []
    arrows: List[Dict] = []
    walked = itertools.count()

    def walk(node, cell, shapes):
        meta = node.meta or {}
        cell = meta.get("cell_index", cell)
        if meta.get("flowchart"):
            shapes = {c.meta["id"]: c for c in node.children
                      if c.meta and c.meta.get("id") and c.class_name == "enclosure"}
        cls = node.class_name
        if cls in LEVEL2_CLASSES:
            level2.append({"class": cls, "node": node,
                           "board": [float(v) for v in label_bbox(node)],
                           "created": (_NO_CELL if cell is None else cell, next(walked))})
        elif cls == "arrow":
            arrows.append(_arrow(node, shapes))
        for child in node.children:
            walk(child, cell, shapes)

    for block in blocks:
        walk(block, None, None)
    return level2, arrows


def _level2_key(obj: Dict):
    if LEVEL2_ORDER == "coords":
        return (obj["board"][1], obj["board"][0])
    return obj["created"]


def _arrow_key(obj: Dict):
    return (obj["board"][1], obj["board"][0])       # an arrow's start


def _grouped(objects: List[Dict], classes, key) -> List[Dict]:
    return [obj for cls in classes
            for obj in sorted((o for o in objects if o["class"] == cls), key=key)]


def page_objects(blocks) -> Tuple[List[Dict], List[Dict]]:
    """The page's level-2 objects and arrows, each in target order."""
    level2, arrows = _collect(blocks)
    return (_grouped(level2, LEVEL2_CLASSES, _level2_key),
            _grouped(arrows, ("arrow",), _arrow_key))


def _mark_key(mark, parent, cls: str):
    """Where an enclosure or line mark goes among its class: by the text it marks.

    In a textblock, the row it marks counted down the block, then left to right
    (-1, a box round the whole block, comes first).
    In a flowchart, a shape goes where its caption does — the (y, x) of the
    caption's first line, the key its lines are sorted by — or by its own (y, x)
    when it has no caption.
    """
    meta = mark.meta or {}
    x0, y0, _, _ = label_bbox(mark)
    if cls == "flowchart":
        key = meta.get("caption_key")
        caption = [label_bbox(c) for c in parent.children
                   if key is not None and (c.meta or {}).get("caption_of") == key]
        if caption:
            return min((box[1], box[0]) for box in caption)
        return (y0, x0)
    return (meta.get("attends_row", -1), x0)


def level1_objects(node, cls: str) -> List[Dict]:
    """A level-2 object's level-1 objects in target order, in board coordinates.

    They are all its direct children. A textblock's rows and a table's cells go
    in the order they were created, and a geometry figure's labels, arcs,
    segments and vertices or a flowchart's captions by exact (y, x) — a line's
    top-left corner, an arc's or segment's first point, a vertex's point.
    Enclosures and line marks go by the text they mark (_mark_key).

    An arc's or segment's board coordinates are its points, flattened.
    """
    wanted = LEVEL1_CLASSES.get(cls, ())
    by_position = cls in ("geometry", "flowchart")
    objects = []
    for created, child in enumerate(node.children):
        name = child.class_name
        if name not in wanted:
            continue
        if name == "vertex":
            if not child.strokes:
                continue
            x, y = float(child.strokes[0].x[0]), float(child.strokes[0].y[0])
            objects.append({"class": name, "board": [x, y], "key": (y, x)})
            continue
        if name in ("arc", "segment"):
            # one single-point stroke per point (geometry_diagrams.Figure.to_ink)
            points = [float(v) for s in child.strokes for v in (s.x[0], s.y[0])]
            objects.append({"class": name, "board": points, "key": (points[1], points[0])})
            continue
        board = [float(v) for v in label_bbox(child)]
        if name in ("enclosure", "line_mark"):
            key = _mark_key(child, node, cls)
        else:
            key = (board[1], board[0]) if by_position else (created,)
        objects.append({"class": name, "board": board, "key": key})

    return [obj for name in wanted
            for obj in sorted((o for o in objects if o["class"] == name),
                              key=lambda o: o["key"])]

# ----------------------------------------------------------------------------
# Transforms and targets
# ----------------------------------------------------------------------------

def page_transform() -> Dict:
    """The whole board downsized to the image."""
    s = IMAGE_SIZE / BOARD_SIZE
    return {"board_box": [0.0, 0.0, float(BOARD_SIZE), float(BOARD_SIZE)],
            "size": [IMAGE_SIZE, IMAGE_SIZE], "pad": [0, 0],
            "scale": [s, s], "offset": [0.0, 0.0]}


def crop_transform(box) -> Dict:
    """A board box scaled so its longer side fills the image, centred.

    The scaled region is a whole number of pixels, so the two axes' scales can
    differ by a hair; both are kept. image = board * scale + offset.
    """
    x0, y0 = max(0.0, box[0]), max(0.0, box[1])
    w = max(min(float(BOARD_SIZE), box[2]) - x0, 1e-3)
    h = max(min(float(BOARD_SIZE), box[3]) - y0, 1e-3)
    fit = IMAGE_SIZE / max(w, h)
    out_w = int(min(IMAGE_SIZE, max(1, round(w * fit))))
    out_h = int(min(IMAGE_SIZE, max(1, round(h * fit))))
    pad_x, pad_y = (IMAGE_SIZE - out_w) // 2, (IMAGE_SIZE - out_h) // 2
    sx, sy = out_w / w, out_h / h
    return {"board_box": [x0, y0, x0 + w, y0 + h], "size": [out_w, out_h],
            "pad": [pad_x, pad_y], "scale": [sx, sy],
            "offset": [pad_x - x0 * sx, pad_y - y0 * sy]}


def to_image(transform: Dict, coords) -> List[float]:
    return [v * transform["scale"][i % 2] + transform["offset"][i % 2]
            for i, v in enumerate(coords)]


def _step(value: float) -> int:
    return min(GRID - 1, max(0, int(value / IMAGE_SIZE * GRID)))


def coord_tokens(image_coords) -> List[int]:
    out = []
    for x, y in zip(image_coords[0::2], image_coords[1::2]):
        out.extend((_step(y), _step(x)) if COORD_ORDER == "y0x0y1x1" else (_step(x), _step(y)))
    return out


def class_words(target: Dict) -> str:
    """What follows an object's coordinates: its class."""
    return CLASS_TOKEN.get(target["class"], target["class"])


def target_string(targets: List[Dict]) -> str:
    return " ".join(" ".join(str(v) for v in t["tokens"]) + " " + class_words(t)
                    for t in targets)


def prompt_entry(prompt: str, objects: List[Dict], transform: Dict) -> Dict:
    targets = []
    for obj in objects:
        image = to_image(transform, obj["board"])
        entry = {"class": obj["class"], "board": obj["board"], "image": image,
                 "tokens": coord_tokens(image)}
        if obj.get("no_direction"):
            entry["no_direction"] = True
        targets.append(entry)
    return {"prompt": prompt, "targets": targets, "target": target_string(targets)}


def build_samples(blocks) -> List[Dict]:
    """The page's samples: the full page first, then each level-2 object's crop
    in the order the level-2 target lists them.

    Each is {image: "page"|"object", transform, prompts: [{prompt, target,
    targets}]}; an object's sample adds `parent` (its index in the level-2
    target, class, board box). A drawing's has no prompts.
    """
    level2, arrows = page_objects(blocks)
    page_t = page_transform()
    samples = [{"image": "page", "transform": page_t,
                "prompts": [prompt_entry(LEVEL2_PROMPT, level2, page_t),
                            prompt_entry(ARROW_PROMPT, arrows, page_t)]}]
    for index, obj in enumerate(level2):
        transform = crop_transform(obj["board"])
        prompts = []
        if obj["class"] in LEVEL1_PROMPTS:
            prompts.append(prompt_entry(LEVEL1_PROMPTS[obj["class"]],
                                        level1_objects(obj["node"], obj["class"]),
                                        transform))
        samples.append({"image": "object",
                        "parent": {"index": index, "class": obj["class"],
                                   "board_bbox": obj["board"],
                                   # the source item it was built from ("fc:writer003_fc_005"), where it has one
                                   "item": (obj["node"].meta or {}).get("item")},
                        "transform": transform, "prompts": prompts})
    return samples


# ----------------------------------------------------------------------------
# Images
# ----------------------------------------------------------------------------

def render_board(blocks, width: float = INK_WIDTH_PX):
    """The page's ink, black on white, as a BOARD_SIZE-pixel square greyscale
    PIL image, one board unit to a pixel."""
    from PIL import Image
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure

    dpi = 100
    fig = Figure(figsize=(BOARD_SIZE / dpi, BOARD_SIZE / dpi), dpi=dpi, facecolor="white")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, BOARD_SIZE)
    ax.set_ylim(BOARD_SIZE, 0)
    ax.axis("off")
    # a vertex is a one-point stroke and draws nothing, as on every other page
    lines = [np.column_stack([s.x, s.y]) for block in blocks
             for s in block.all_strokes() if len(s.x) >= 2]
    ax.add_collection(LineCollection(lines, colors="black", linewidths=width * 72.0 / dpi,
                                     capstyle="round", joinstyle="round"))
    canvas.draw()
    return Image.fromarray(np.asarray(canvas.buffer_rgba())[..., 0].copy(), mode="L")


def sample_image(board, transform: Dict):
    """A sample's 224 x 224 image: its board box resampled, then padded."""
    from PIL import Image
    out_w, out_h = transform["size"]
    region = board.resize((out_w, out_h), resample=Image.LANCZOS,
                          box=tuple(transform["board_box"]))
    image = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 255)
    image.paste(region, tuple(transform["pad"]))
    return image


# ----------------------------------------------------------------------------
# Spot check: does level-2 creation order match where objects end up?
# ----------------------------------------------------------------------------

def order_report(pages_blocks) -> Dict:
    """How often cell creation order disagrees with final position, per class.

    `differs`: pages where sorting by exact (y0, x0) gives another order.
    `inverted`: pairs (of one class, on one page) the two orders disagree on.
    `above`: pairs where the later-created object lies wholly above the earlier
    one — the disagreement that actually reads wrong, as opposed to two objects
    side by side whose tops differ by a few units.
    `same_cell_*`: the part of `inverted` and `above` between two objects of one
    cell (a flowchart's captions, a figure's labels), which no cell ordering
    decides; the rest are between cells, which is the seed/collision question.
    """
    report = {cls: {"pages": 0, "differs": 0, "pairs": 0, "inverted": 0, "above": 0,
                    "same_cell_inverted": 0, "same_cell_above": 0}
              for cls in LEVEL2_CLASSES}
    for blocks in pages_blocks:
        level2, _ = _collect(blocks)
        for cls in LEVEL2_CLASSES:
            objects = sorted((o for o in level2 if o["class"] == cls),
                             key=lambda o: o["created"])
            if len(objects) < 2:
                continue
            row = report[cls]
            row["pages"] += 1
            by_coords = sorted(objects, key=lambda o: (o["board"][1], o["board"][0]))
            row["differs"] += any(a is not b for a, b in zip(objects, by_coords))
            for i, first in enumerate(objects):
                for later in objects[i + 1:]:
                    inverted = ((later["board"][1], later["board"][0])
                                < (first["board"][1], first["board"][0]))
                    above = later["board"][3] <= first["board"][1]
                    same_cell = later["created"][0] == first["created"][0]
                    row["pairs"] += 1
                    row["inverted"] += inverted
                    row["above"] += above
                    row["same_cell_inverted"] += inverted and same_cell
                    row["same_cell_above"] += above and same_cell
    return report


def main():
    import argparse
    import random

    ap = argparse.ArgumentParser()
    ap.add_argument("--check-order", action="store_true",
                    help="spot-check level-2 creation order against final positions")
    ap.add_argument("--pages", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not args.check_order:
        ap.error("nothing to do; pass --check-order (seg_pages.py renders the samples)")

    from make_pdf import build_pools
    from page_generator import generate_page

    random.seed(args.seed)
    pools = build_pools(with_shape_bank=True)
    pages = [generate_page(**pools)["board_blocks"] for _ in range(args.pages)]
    report = order_report(pages)
    print(f"level-2 creation order vs exact (y0, x0), {args.pages} pages "
          "(pages with 2+ of the class):")
    for cls, row in report.items():
        pairs = max(row["pairs"], 1)
        print(f"  {cls:10s} order differs on {row['differs']}/{row['pages']} pages; "
              f"{row['inverted']}/{row['pairs']} pairs inverted ({100 * row['inverted'] / pairs:.1f}%), "
              f"{row['above']} created later but wholly above ({100 * row['above'] / pairs:.1f}%); "
              f"within one cell: {row['same_cell_inverted']} inverted, {row['same_cell_above']} above")


if __name__ == "__main__":
    main()
