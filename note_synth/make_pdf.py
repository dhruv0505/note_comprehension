"""
make_pdf.py — generate N synthetic note pages and write them to one PDF for
eyeballing, alongside the label JSON for each page.

    python3 make_pdf.py                          # 25 pages -> pages.pdf
    python3 make_pdf.py --pages 20 --out try.pdf
    python3 make_pdf.py --no-label-pages         # ink pages only
    python3 make_pdf.py --color-by-class         # tint strokes by taxonomy class
    python3 make_pdf.py --boxes 1                # overlay level-1 label boxes

Each rendered page is followed by its **ground truth**: every labelled object
boxed by level, handwriting replaced with the text it actually says, maths with
its LaTeX, a drawing kept as ink under its category name, and each arrow reduced
to a dotted line from start to end. Read the pair together to check that what
the generator wrote and what it claims to have written agree.

The ink pages themselves are plain black, which is what a segmentation model
should be trained on. The two flags above exist only to inspect what the
generator did — they draw information that is supposed to come from the labels,
not the image.

Only a slice of each dataset is read (see POOL_LIMITS): a hundred pages resample
from the pools many times over, so there is no reason to parse all 12k IAM lines
or a multi-GB QuickDraw dump to make them. Datasets that aren't downloaded yet
fall back to the small stand-ins in data/ (see loaders.DATA_FALLBACKS).
"""

from __future__ import annotations
import argparse
import json
import math
import random
import re
import sys

import loaders
from page_generator import generate_page
from demo import CLASS_COLORS
from labels import label_bbox

# Enough variety that 100 pages rarely repeat a line, without a slow startup.
# FC ships 672 charts and they are small to parse, so load them all: at a
# 120 cap the generator was drawing 92 flowcharts from 35 distinct charts.
# EMNIST was 24 per character, and a geometry figure writing the same character
# twice (two "5"s, say) used the identical glyph for both 1 time in 10; at 200
# per character that went to 0 of 33 such pairs.
POOL_LIMITS = {"iam": 4000, "deepwriting": 4000, "quickdraw": 120,
               "math": 400, "fc": 672, "emnist": 200}


def _try(name, fn, *args, **kwargs):
    try:
        pool = fn(*args, **kwargs)
    except (FileNotFoundError, ImportError) as exc:
        print(f"  {name:12s} skipped ({exc})")
        return []
    print(f"  {name:12s} {len(pool)} items")
    return pool


def build_pools(with_shape_bank: bool = False) -> dict:
    """The dataset pools a page is filled from.

    `with_shape_bank` is off by default because geometry_diagrams.build_bank()
    mines these same pools, so asking for it from here while build_bank is the
    caller would recurse. Page generation asks for it separately.
    """
    print("Loading datasets from", loaders.DATA_DIR)
    text = (_try("iam-ondb", loaders.load_iam_ondb, max_items=POOL_LIMITS["iam"]) +
            _try("deepwriting", loaders.load_deepwriting, max_items=POOL_LIMITS["deepwriting"]))
    pools = {
        "text_pool": text,
        "quickdraw_pool": _try("quickdraw", loaders.load_quickdraw,
                               max_per_category=POOL_LIMITS["quickdraw"]),
        "flowchart_pool": _try("fc", loaders.load_fc, max_items=POOL_LIMITS["fc"]),
        "math_pool": _try("mathwriting", loaders.load_mathwriting,
                          max_items=POOL_LIMITS["math"]),
        # keyed by character, not a flat list: a geometry figure looks up the
        # ink for the letter it wants to write
        "glyph_pool": _try("emnist", loaders.load_emnist_pool,
                           max_per_class=POOL_LIMITS["emnist"]),
    }
    if with_shape_bank:
        from page_generator import SHAPES_BANK_CACHE, shape_bank_default
        bank = shape_bank_default()
        pools["shape_bank"] = bank
        if bank is None:
            print(f"  {'geometry':12s} skipped (no mined shape bank at "
                  f"{SHAPES_BANK_CACHE}; build one with "
                  f"'python geometry_diagrams.py --cache shapes_bank.json')")
        else:
            print(f"  {'geometry':12s} {len(bank.lines)} lines, "
                  f"{len(bank.ovals)} ovals")
    return pools


def draw_page(ax, page, color_by_class: bool, box_level):
    w, h = page["page_w"], page["page_h"]

    def walk(node):
        color = CLASS_COLORS.get(node.class_name, "#000") if color_by_class else "#000"
        for s in node.strokes:
            ax.plot(s.x, s.y, "-", lw=0.55, color=color, solid_capstyle="round")
        for c in node.children:
            walk(c)

    for block in page["blocks"]:
        walk(block)

    if box_level is not None:
        from matplotlib.patches import Rectangle
        for obj in json.loads(page["labels_json"])["objects"]:
            if obj["level"] != box_level:
                continue
            x0, y0, x1, y1 = obj["bbox"]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                    ec="#d62728", lw=0.4, alpha=0.7))

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)          # y increases downward, same as the page space
    ax.set_aspect("equal")
    ax.axis("off")


# Box colour per taxonomy level, for the label page only.
LEVEL_COLORS = {2: "#1f77b4", 1: "#2ca02c", 0: "#d62728"}
LEVEL_OF = {"textblock": 2, "geometry": 2, "drawing": 2, "table": 2, "flowchart": 2,
            "enclosure": 1, "line_mark": 1,
            "textline": 1, "formula": 1, "arc": 1, "segment": 1, "vertex": 1,
            "word": 0, "arrow": 0, "unknown": 0}

# Classes an arrow can be anchored to. Every flowchart shape is an `enclosure`;
# how its outline is approximated comes from what was drawn (see _node_shape).
NODE_CLASSES = ("enclosure",)
ROUND_OUTLINES = ("terminator", "connection", "oval", "circle")


MIN_LABEL_PT = 2.6      # below this the text is unreadable, so truncate instead
CHAR_EM = 0.55          # mean advance width of a proportional face, in em


def _fit_text(text: str, box_w: float, box_h: float, units_to_points: float):
    """A font size and a truncation that keep `text` inside its box.

    Sized against the box's width as well as its height. Height alone gave a
    font far too large for a long line, so almost everything was truncated to a
    few characters and a lone ellipsis — the transcription is the point of this
    page, so the size yields to it and only shrinks so far before cutting.
    """
    width_pt = max(box_w * units_to_points, 1.0)
    height_pt = max(box_h * units_to_points, 1.0)

    size = min(height_pt * 0.78, width_pt / (max(len(text), 1) * CHAR_EM))
    if size < MIN_LABEL_PT:
        size = MIN_LABEL_PT
        room = max(1, int(width_pt / (size * CHAR_EM)))
        if len(text) > room:
            text = text[:max(1, room - 1)] + "…"
    return text, size


# Matplotlib's mathtext is close to LaTeX but not identical. These are the gaps
# that actually show up in MathWriting's normalized labels: it wants the long
# spellings of the comparison operators, and it has no \underline at all. Left
# alone, `A\ge0.018` simply fails to parse and falls back to showing its source.
_MATHTEXT_FIXES = [
    (re.compile(r"\\ge(?![a-zA-Z])"), r"\\geq"),
    (re.compile(r"\\le(?![a-zA-Z])"), r"\\leq"),
    (re.compile(r"\\ne(?![a-zA-Z])"), r"\\neq"),
    (re.compile(r"\\iff(?![a-zA-Z])"), r"\\Leftrightarrow"),
    (re.compile(r"\\implies(?![a-zA-Z])"), r"\\Rightarrow"),
    # Braces, not a bare unwrap: `\in\underline{m}` unwrapped to `\inm`, which
    # runs the letter into the preceding command and fails to parse.
    (re.compile(r"\\underline\{([^{}]*)\}"), r"{\1}"),
]

# A two-row, one-column matrix wrapped in brackets is a binomial coefficient,
# which mathtext draws properly. Matched before the general case below.
# One cell of a matrix: no column separator, and no row separator either.
# Spelled as "any non-backslash, or a backslash not followed by another" so
# commands inside a cell (\lfloor, \sqrt) still match while a row break does
# not. Without the row-break exclusion this matched a three-row column vector
# as a binomial and produced a \binom with a stray row separator inside it.
_CELL = r"(?:[^&\\]|\\(?!\\))+?"
_BINOM_RE = re.compile(
    r"[\(\[]\s*\\begin\{[bp]?matrix\}\s*(" + _CELL + r")\s*\\\\\s*("
    + _CELL + r")\s*\\end\{[bp]?matrix\}\s*[\)\]]")

# Anything else with a matrix environment: mathtext has none, so the rows are
# written inline, cells separated by commas and rows by semicolons. That is a
# real convention for writing a matrix on one line, and it beats showing raw
# LaTeX source on a page whose whole job is to say what the ink means.
_MATRIX_RE = re.compile(r"\\begin\{[bp]?matrix\}(.*?)\\end\{[bp]?matrix\}", re.S)


def _inline_matrix(match: "re.Match") -> str:
    rows = [r.strip() for r in re.split(r"\\\\", match.group(1)) if r.strip()]
    return ";".join(",".join(c.strip() for c in row.split("&")) for row in rows)

_math_cache: dict = {}


def _math_metrics(expr: str):
    """(mathtext string, width, height) at 10pt, or None if it won't parse.

    Parsed rather than guessed because a rendered expression's width has no
    relation to its source length — `\\frac{1}{2}` is three characters wide and
    twelve long — so the character-count estimate used for prose would size
    maths wrongly in both directions.
    """
    if expr in _math_cache:
        return _math_cache[expr]

    from matplotlib import mathtext
    from matplotlib.font_manager import FontProperties

    fixed = _BINOM_RE.sub(
        lambda m: "\\binom{%s}{%s}" % (m.group(1), m.group(2)), expr)
    fixed = _MATRIX_RE.sub(_inline_matrix, fixed)
    for pattern, replacement in _MATHTEXT_FIXES:
        fixed = pattern.sub(replacement, fixed)

    result = None
    if "$" not in fixed:
        try:
            parser = mathtext.MathTextParser("path")
            width, height, _, _, _ = parser.parse(f"${fixed}$", dpi=72,
                                                   prop=FontProperties(size=10))
            result = (f"${fixed}$", width, height)
        except Exception:
            result = None       # unsupported construct; caller shows the source
    _math_cache[expr] = result
    return result


def _boundary_point(box, toward, shape="rect"):
    """Where the line from a shape's centre toward `toward` leaves the shape.

    `shape` follows the node's drawn outline, so an edge stops on the outline actually
    drawn: a diamond's corner is much closer in along the diagonals than its
    bounding box is, and anchoring a decision's arrows to the box left them
    floating in the white space outside the diamond.
    """
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    dx, dy = toward[0] - cx, toward[1] - cy
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return cx, cy
    rx, ry = max((x1 - x0) / 2.0, 1e-6), max((y1 - y0) / 2.0, 1e-6)
    if shape == "ellipse":
        t = 1.0 / max(((dx / rx) ** 2 + (dy / ry) ** 2) ** 0.5, 1e-9)
    elif shape == "diamond":
        t = 1.0 / max(abs(dx) / rx + abs(dy) / ry, 1e-9)
    else:
        t = min(rx / abs(dx) if dx else float("inf"),
                ry / abs(dy) if dy else float("inf"))
    return cx + dx * t, cy + dy * t


def _node_shape(node):
    """How an enclosure's outline is approximated, for edge anchoring and boxing.

    The class does not say — every shape is an `enclosure` — so this reads
    what was drawn: FC's own symbol type for a flowchart node, or the shape
    drawn round some writing, kept in its label.
    """
    drawn = (node.meta or {}).get("node_type") or node.label
    if drawn in ROUND_OUTLINES:
        return "ellipse"
    if drawn == "decision":
        return "diamond"
    return "rect"          # a box, and data's parallelogram is close enough


def _graph_span(node, shapes):
    """An arrow's endpoints taken from the flowchart graph, not from its ink.

    FC states which nodes each arrow joins, and that is what the arrow means.
    Deriving the endpoints from the ink instead put them wherever the pen
    happened to start and stop — short of the shapes, past them, or with the
    head on the wrong end when the writer drew the arrow backwards.
    """
    meta = node.meta or {}
    source = shapes.get(meta.get("source_id"))
    target = shapes.get(meta.get("target_id"))
    if source is None or target is None:
        return None, None
    source_box, target_box = source.bbox(), target.bbox()
    source_mid = ((source_box[0] + source_box[2]) / 2, (source_box[1] + source_box[3]) / 2)
    target_mid = ((target_box[0] + target_box[2]) / 2, (target_box[1] + target_box[3]) / 2)
    return (_boundary_point(source_box, target_mid, _node_shape(source)),
            _boundary_point(target_box, source_mid, _node_shape(target)))


# How far an arrow bows off the straight line joining its two nodes, as a
# fraction of the distance between them (matplotlib's arc3 `rad`).
#
# Straight is right when an arrow is the only one between its two shapes, and 28
# of OHFD's node pairs are joined in BOTH directions — a loop back up to the
# decision it came from. Drawn straight, those two land on exactly the same line
# between the same two outlines: what should read as "A goes to B, and B comes
# back to A" instead reads as one arrow with a head at each end, or as a single
# arrow whose direction is anyone's guess. Bowing them to opposite sides costs
# nothing where there is only one arrow and makes the pair unambiguous where
# there are two.
ARROW_BOW = 0.22


def _arrow_lanes(diagram):
    """arrow id -> how far that arrow bows, keeping parallel arrows apart.

    The bow is worked out in page space and then converted to the arrow's own
    frame, because matplotlib measures `rad` from the direction of travel: giving
    an A->B arrow and a B->A arrow the same rad would bow them to opposite sides,
    and giving two A->B arrows the same rad would lay them on top of each other.
    Grouping by the unordered pair and flipping the sign for arrows running
    against the group's first one handles both cases the same way.
    """
    groups = {}
    for child in diagram.children:
        if child.class_name != "arrow":
            continue
        meta = child.meta or {}
        source, target = meta.get("source_id"), meta.get("target_id")
        arrow_id = meta.get("id")
        if not (source and target and arrow_id):
            continue
        groups.setdefault(frozenset((source, target)), []).append(
            (arrow_id, (source, target)))

    lanes = {}
    for members in groups.values():
        if len(members) < 2:
            lanes[members[0][0]] = 0.0
            continue
        canonical = members[0][1]
        span = (len(members) - 1) / 2.0
        for i, (arrow_id, direction) in enumerate(members):
            side = (i - span) / span * ARROW_BOW          # -BOW .. +BOW
            lanes[arrow_id] = side if direction == canonical else -side
    return lanes


def _bowed_midpoint(start, end, rad: float):
    """Where a bowed arrow's own midpoint is, and the angle it runs at there.

    matplotlib's arc3 is a quadratic Bezier whose control point sits at the
    straight line's midpoint pushed `rad * length` sideways, so the curve passes
    through half that offset. The tangent there is parallel to the chord, which is
    why the angle does not depend on the bow.
    """
    mx, my = (start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0
    dx, dy = end[0] - start[0], end[1] - start[1]
    # matplotlib offsets the control point by rad * (dy, -dx)
    return (mx + 0.5 * rad * dy, my - 0.5 * rad * dx), (dx, dy)


def _arrow_span(node):
    """Where an arrow starts and where it points, as (start, end).

    A connector arrow is drawn shaft first, then its head (two barbs, one V, or
    an outline and its fill), so its last stroke ends somewhere on the head,
    not at the target. Taking the last stroke's last point left every
    ground-truth arrow stopping short and off to one side of the thing it was
    pointing at. The shaft alone carries the run — loop and all, it still starts
    on the anchor and ends on the tip.

    FC's own arrows are captured ink with no such convention, so there the two
    stroke endpoints furthest apart are used, with the head placed at whichever
    of them the last stroke finished nearest.
    """
    strokes = node.all_strokes()
    if not strokes:
        return None, None

    # connector arrows: "geometry" now their ink is mined, "procedural" before
    if node.source in ("geometry", "procedural"):
        shaft = strokes[0]
        return (shaft.x[0], shaft.y[0]), (shaft.x[-1], shaft.y[-1])

    ends = []
    for stroke in strokes:
        ends.append((stroke.x[0], stroke.y[0]))
        ends.append((stroke.x[-1], stroke.y[-1]))
    best = max(((a, b) for i, a in enumerate(ends) for b in ends[i + 1:]),
               key=lambda pair: (pair[0][0] - pair[1][0]) ** 2
                                + (pair[0][1] - pair[1][1]) ** 2,
               default=(None, None))
    if best[0] is None:
        return None, None

    tail = (strokes[-1].x[-1], strokes[-1].y[-1])
    a, b = best
    if ((a[0] - tail[0]) ** 2 + (a[1] - tail[1]) ** 2
            < (b[0] - tail[0]) ** 2 + (b[1] - tail[1]) ** 2):
        a, b = b, a         # head goes where the pen finished
    return a, b


def draw_label_page(ax, page, fig_width_in: float):
    """Render the page's ground truth rather than its ink.

    Every labelled object gets its box. Handwriting is replaced by the text it
    actually says, a drawing keeps its strokes and gains a caption, and a
    connector arrow collapses to a dotted line from its start to its end — so
    the page can be read as what the model is meant to predict, next to the
    image it has to predict it from.
    """
    from matplotlib.patches import Ellipse, Polygon, Rectangle

    w, h = page["page_w"], page["page_h"]
    units_to_points = fig_width_in * 72.0 / w

    def box(node, color, lw=0.6, alpha=0.85, shape="rect"):
        """Outline a node in its own shape, so a flowchart's shape vocabulary
        survives into the ground truth: an oval, a connector circle and a
        decision diamond are not rectangles, and drawing every symbol as a
        rectangle made the chart unreadable as a flowchart."""
        # the labelled box (labels.label_bbox)
        x0, y0, x1, y1 = label_bbox(node)
        if shape == "ellipse":
            ax.add_patch(Ellipse(((x0 + x1) / 2, (y0 + y1) / 2), x1 - x0, y1 - y0,
                                  fill=False, ec=color, lw=lw, alpha=alpha))
        elif shape == "diamond":
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            ax.add_patch(Polygon([(cx, y0), (x1, cy), (cx, y1), (x0, cy)],
                                  closed=True, fill=False, ec=color, lw=lw,
                                  alpha=alpha))
        else:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                    ec=color, lw=lw, alpha=alpha))
        return x0, y0, x1, y1

    def caption(text, x, y, size, color, va="bottom"):
        # '$' would switch matplotlib into mathtext and choke on LaTeX labels
        ax.text(x, y, str(text).replace("$", ""), fontsize=size, color=color,
                va=va, ha="left", clip_on=True)

    def draw_math(expr, x, y, box_w, box_h):
        """Render an expression as maths, falling back to its source if it won't."""
        metrics = _math_metrics(expr)
        if metrics is None:
            fitted, size = _fit_text(expr, box_w, box_h, units_to_points)
            caption(fitted, x, y, size, "#b07acc", va="center")   # dimmer: not rendered
            return
        text, width10, height10 = metrics
        # metrics are for 10pt, and scale linearly with the font size
        size = min(box_h * units_to_points * 0.95 / max(height10 / 10.0, 1e-6),
                   box_w * units_to_points / max(width10 / 10.0, 1e-6))
        ax.text(x, y, text, fontsize=max(1.6, size), color="#5b2d8e",
                va="center", ha="left", clip_on=True)

    def draw_geometry(node):
        """A figure drawn as the construction it was an attempt at.

        geometry_diagrams keeps the true segment or circle behind every stroke,
        so the ground truth here is not a tidied trace of the ink — it is the
        geometry the hand was aiming at, in which a tangent really is tangent and
        an inscribed circle really touches all three sides.
        """
        import geometry_diagrams as gd

        primitives = (node.meta or {}).get("exact") or []
        for prim in primitives:
            points = gd.exact_polyline(prim)
            # an annotation (a chord, an altitude, an angle mark) is drawn lightly
            # so the shape's own outline still reads as the outline
            light = prim.get("mark")
            ax.plot(points[:, 0], points[:, 1], "-",
                    lw=0.4 if light else 0.7,
                    color="#8a6bbf" if light else "#5b2d8e",
                    alpha=0.8 if light else 1.0, solid_capstyle="round")
        if not primitives:
            # a figure that lost its construction to an anisotropic fit; its ink
            # is the only thing left that says what it is
            for s in node.all_strokes():
                ax.plot(s.x, s.y, "-", lw=0.5, color="#5b2d8e",
                        solid_capstyle="round")

    def draw_table_grid(node, bounds):
        """Every row and column divider of a table, drawn from its structure.

        Not from its ink: `make_table` rules only some tables fully — a header
        rule only, just the rows, just the columns. The rows and columns are
        there regardless — they are what makes it a table — so the ground truth
        states the whole grid and lets the ink page be as sparsely ruled as a
        real hand-drawn table is.
        """
        meta = node.meta or {}
        segments = [p for p in (meta.get("exact") or [])
                    if p.get("kind") == "segment"]
        if not segments:
            # An anisotropic fit dropped the construction (core._scale_exact).
            # Where the dividers fall is kept as fractions of the table's size,
            # so the counts, the fractions and the box rebuild it.
            rows, cols = meta.get("n_rows"), meta.get("n_cols")
            if not (rows and cols):
                return
            from block_generators import table_grid_exact
            segments = table_grid_exact(*bounds, rows, cols,
                                        col_fracs=meta.get("col_fracs"),
                                        row_fracs=meta.get("row_fracs"))
        for prim in segments:
            (sx, sy), (ex, ey) = prim["a"], prim["b"]
            ax.plot([sx, ex], [sy, ey], "-", lw=0.5, color=LEVEL_COLORS[2],
                    alpha=0.75, solid_capstyle="round")

    def walk(node, inside_drawing=False, chart=None):
        cls = node.class_name
        level = LEVEL_OF.get(cls)

        if (node.meta or {}).get("flowchart"):
            # Index this chart by id so its arrows can be drawn between the nodes
            # the flowchart says they join, and its parallel arrows told apart.
            by_id = {c.meta["id"]: c for c in node.children
                     if c.meta and c.meta.get("id")}
            chart = {
                "shapes": {i: c for i, c in by_id.items()
                           if c.class_name in NODE_CLASSES},
                "arrows": {i: c for i, c in by_id.items()
                           if c.class_name == "arrow"},
                "lanes": _arrow_lanes(node),
            }

        if cls == "geometry":
            x0, y0, x1, y1 = box(node, LEVEL_COLORS[2], lw=0.8)
            draw_geometry(node)
            name = "geometry"
            caption(name, x0, y0 - 2.0, 4.2, LEVEL_COLORS[2])
            # The labels written on the figure and its vertices are ground truth
            # in their own right, so they are walked; the rest of the children
            # are the strokes, which draw_geometry has already drawn as the
            # construction behind them.
            for child in node.children:
                if child.class_name in ("textline", "formula", "vertex"):
                    walk(child, inside_drawing, chart)
            return

        if cls == "vertex":
            # a point: its box has no area, so it is marked rather than boxed,
            # with the angles it makes inside the figure written beside it
            px, py, _, _ = node.bbox()
            ax.plot([px], [py], "o", ms=1.5, color=LEVEL_COLORS[1], alpha=0.9,
                    zorder=4)
            angles = (node.meta or {}).get("angles") or []
            if angles:
                caption(" / ".join(f"{a:.0f}°" for a in angles), px + 0.8, py - 0.8,
                        MIN_LABEL_PT, LEVEL_COLORS[1])
            return

        if cls == "table":
            bounds = box(node, LEVEL_COLORS[2], lw=0.8)
            draw_table_grid(node, bounds)
            meta = node.meta or {}
            name = "table"
            if meta.get("n_rows") and meta.get("n_cols"):
                name = f"table ({meta['n_rows']}x{meta['n_cols']})"
            caption(name, bounds[0], bounds[1] - 2.0, 4.2, LEVEL_COLORS[2])
            # falls through to the children so each cell still gets its own
            # word box and transcription inside the grid
            for child in node.children:
                walk(child, inside_drawing, chart)
            return

        if cls == "drawing":
            x0, y0, x1, y1 = box(node, LEVEL_COLORS[2], lw=0.8)
            for s in node.all_strokes():      # a sketch is its own best label
                ax.plot(s.x, s.y, "-", lw=0.5, color="#444", solid_capstyle="round")
            label = node.label or "drawing"
            caption(label, x0, y0 - 2, min(7.0, max(3.0, (x1 - x0) * units_to_points * 0.12)),
                    LEVEL_COLORS[2])
            # A `drawing` is always a real QuickDraw sketch, so there is nothing
            # under it to box.
            return

        if cls == "unknown":
            # Meaningless ink: drawn, because the mark itself is the only thing
            # that says what it is. There is no transcription to substitute.
            x0, y0, x1, y1 = box(node, LEVEL_COLORS[0], lw=0.5)
            for s in node.all_strokes():
                ax.plot(s.x, s.y, "-", lw=0.45, color="#888", solid_capstyle="round")
            caption("unknown", x0, y0 - 1.5, 3.4, LEVEL_COLORS[0])
            return

        if cls == "arrow":
            start = end = None
            if chart:
                start, end = _graph_span(node, chart["shapes"])
            if start is None:
                start, end = _arrow_span(node)
            if start and end:
                # bowed away from any arrow sharing the same pair of nodes, so a
                # loop drawn in both directions reads as two arrows
                rad = (chart or {}).get("lanes", {}).get(
                    (node.meta or {}).get("id"), 0.0)
                ax.annotate("", xy=end, xytext=start,
                            arrowprops=dict(arrowstyle="-|>", color=LEVEL_COLORS[0],
                                            linewidth=0.7, linestyle=(0, (2.2, 1.8)),
                                            shrinkA=0, shrinkB=0, mutation_scale=7,
                                            joinstyle="miter",
                                            connectionstyle=f"arc3,rad={rad:.3f}"))
            return

        if cls == "formula":
            # its own ground truth is the maths it says, set as maths
            x0, y0, x1, y1 = box(node, LEVEL_COLORS[1])
            if node.label:
                draw_math(node.label, x0 + 1, (y0 + y1) / 2, x1 - x0, y1 - y0)
            else:
                caption("⟨formula⟩", x0 + 1, (y0 + y1) / 2,
                        max(MIN_LABEL_PT, min(4.0, (y1 - y0) * units_to_points * 0.5)),
                        "#999", va="center")
            return

        if cls == "textline":
            # Word boxes are deliberately not drawn here: a box per word on top
            # of the line's own box and its transcription made the line
            # unreadable. Level-0 words still exist in the labels themselves,
            # and a word outside a line (a table cell) is still boxed below.
            x0, y0, x1, y1 = box(node, LEVEL_COLORS[1])
            text = node.label or ""
            if not text:
                # A few IAM documents ship no transcription at all. Say so
                # rather than leaving bare boxes, which read as a rendering bug.
                caption("⟨no transcription⟩", x0 + 1, (y0 + y1) / 2,
                        max(MIN_LABEL_PT, min(4.0, (y1 - y0) * units_to_points * 0.5)),
                        "#999", va="center")
                return
            if node.source == "mathwriting":
                draw_math(text, x0 + 1, (y0 + y1) / 2, x1 - x0, y1 - y0)
            else:
                fitted, size = _fit_text(text, x1 - x0, y1 - y0, units_to_points)
                caption(fitted, x0 + 1, (y0 + y1) / 2, size, "#000", va="center")
            return

        if node.source == "mathwriting" and node.strokes:
            x0, y0, x1, y1 = box(node, LEVEL_COLORS[1])
            if node.label:
                draw_math(node.label, x0 + 1, (y0 + y1) / 2, x1 - x0, y1 - y0)
            return

        if cls == "word":
            # Only reached for a word that is NOT inside a text line — a table
            # cell holds its word directly. The textline branch above draws its
            # own words and returns, so those never arrive here.
            x0, y0, x1, y1 = box(node, LEVEL_COLORS[0], lw=0.4, alpha=0.7)
            if node.label:
                fitted, size = _fit_text(node.label, x1 - x0, y1 - y0, units_to_points)
                caption(fitted, x0 + 1, (y0 + y1) / 2, size, "#000", va="center")
            return

        if level is not None:
            x0, y0, x1, y1 = box(node, LEVEL_COLORS[level],
                                  lw=0.8 if level == 2 else 0.5,
                                  shape=_node_shape(node) if cls in NODE_CLASSES else "rect")
            # A line mark gets its box and no caption: the box is a stroke or two
            # high and sits on the words it marks, so a caption would land on them.
            if cls in NODE_CLASSES:
                # inside its own corner: placed above the box it sat on top of
                # whichever node happened to be there, which is what made a
                # flowchart's ground truth unreadable
                caption(cls, x0 + 1.5, y0 + 1.0, 3.4, LEVEL_COLORS[level], va="top")
            elif level == 2:
                caption(cls, x0, y0 - 2.0, 4.2, LEVEL_COLORS[level])

        for child in node.children:
            walk(child, inside_drawing, chart)

    for block in page["blocks"]:
        walk(block)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=25)
    ap.add_argument("--out", default="pages.pdf")
    ap.add_argument("--labels", default="pages_labels.json",
                    help="write every page's label JSON here ('' to skip)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--color-by-class", action="store_true",
                    help="debug only — tints strokes by class, leaking labels into the image")
    ap.add_argument("--boxes", type=int, default=None, choices=[0, 1, 2],
                    help="debug only — overlay ground-truth boxes for this level")
    ap.add_argument("--no-label-pages", action="store_true",
                    help="skip the ground-truth page that follows each rendered page")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    random.seed(args.seed)
    pools = build_pools(with_shape_bank=True)
    if not pools["text_pool"]:
        sys.exit("No text pool — IAM-OnDB and DeepWriting are both missing; see README.md")
    if not pools["quickdraw_pool"]:
        print("  (no quickdraw pool: pages will have no 'drawing' blocks)")

    all_labels = []
    with PdfPages(args.out) as pdf:
        for i in range(args.pages):
            page = generate_page(**pools)
            aspect = page["page_h"] / page["page_w"]

            fig, ax = plt.subplots(figsize=(8.5, 8.5 * aspect))
            draw_page(ax, page, args.color_by_class, args.boxes)
            fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
            pdf.savefig(fig)
            plt.close(fig)

            if not args.no_label_pages:
                fig, ax = plt.subplots(figsize=(8.5, 8.5 * aspect))
                draw_label_page(ax, page, 8.5)
                ax.set_title(f"page {i + 1} — ground truth", fontsize=7, color="#666")
                fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.02)
                pdf.savefig(fig)
                plt.close(fig)

            labels = json.loads(page["labels_json"])
            labels["page_index"] = i
            labels["targets_by_level"] = {str(k): v for k, v in page["targets_by_level"].items()}
            all_labels.append(labels)

            if (i + 1) % 10 == 0:
                print(f"  rendered {i + 1}/{args.pages} pages")

    print(f"Wrote {args.pages} pages -> {args.out}")
    if args.labels:
        with open(args.labels, "w") as f:
            json.dump(all_labels, f)
        counts = {}
        for page_labels in all_labels:
            for obj in page_labels["objects"]:
                counts[obj["class"]] = counts.get(obj["class"], 0) + 1
        print(f"Wrote labels for {len(all_labels)} pages -> {args.labels}")
        print("  objects by class:",
              ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
