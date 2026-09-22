"""
mark_crops.py — the enclosure / line-mark separation set.

For every enclosure and line mark inside a textblock, a crop of the page as
page_export.render_page draws it (1600 px), and the same crop without the mark:

    <name>_mNN_input.png   all the page's ink in the crop
    <name>_mNN_text.png    the crop without its marks: the writing underneath them

Greyscale PNGs, black ink on white, at the page drawing's resolution (no
resizing), so input is exactly text and marks drawn together. Marks whose own
boxes overlap — an underline under a circled word, two enclosures that cross —
are one sample, joined transitively: a crop holding only one of them would still
show part of the other, and the text image would then not be the writing alone.
One manifest row per sample:

    {"input": ..., "text": ...,
     "labels": [0, 1, 1, 0],                     one per MARK_TYPES: is that kind of mark in the crop
     "shapes": ["oval", "underline"],            the sample's marks, in page order
     "page": 123, "box": [x0, y0, x1, y1],       the crop, page pixels
     "mark_boxes": [[x0, y0, x1, y1], ...],      each mark's own box, page pixels, as `shapes`
     "text_label": "..."}                        what the marked rows (or block) say, where known

`labels` is the target for a per-type score with a sigmoid on each: several kinds
can be present at once, so it is multi-label, not one class out of four.

The crop (write_marks' `crop`):
  "context"  an enclosure's own box — it surrounds what it marks — or a line
             mark's box joined with the row it marks, since an underline alone is a
             sliver; a sample's crop is its marks' crops joined, padded by PAD board
             units
  "box"      the marks' own boxes joined and nothing else: a lone mark's box, or
             the joined box of marks that overlap. Padded only by half the pen, so
             the drawn ink is not cut off
Either is kept on the page. Flowchart shapes are enclosures of a flowchart rather
than marks on writing, and are not included.
"""

from __future__ import annotations
import math
import os

import numpy as np

PAD = 6.0       # board units of margin round a "context" crop
CROPS = ("context", "box")
# The order of a row's `labels`: line marks' kinds, then enclosures'.
MARK_TYPES = ("underline", "strikethrough", "oval", "rectangle")


def mark_targets(blocks):
    """(mark, its textblock's rows, the row it marks or None) for every enclosure and
    line mark that is a child of a textblock."""
    found = []

    def walk(node):
        if node.class_name == "textblock":
            rows = [c for c in node.children if c.class_name in ("textline", "formula")]
            for child in node.children:
                if child.class_name in ("enclosure", "line_mark"):
                    at = (child.meta or {}).get("attends_row", -1)
                    row = rows[at] if isinstance(at, int) and 0 <= at < len(rows) else None
                    found.append((child, rows, row))
        for child in node.children:
            walk(child)

    for block in blocks:
        walk(block)
    return found


def _render_region(strokes, box, scale, ink_width):
    """`strokes` (board units) drawn into the page-pixel `box`, as render_page draws the page."""
    from PIL import Image
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure

    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    dpi = 100
    # The canvas is figsize * dpi pixels rounded down, and 57 / 100 * 100 can land
    # a hair under 57 — so a little slack, and the result cut to the box exactly.
    fig = Figure(figsize=((width + 0.01) / dpi, (height + 0.01) / dpi), dpi=dpi, facecolor="white")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis("off")
    lines = [np.column_stack([np.asarray(s.x) * scale, np.asarray(s.y) * scale])
             for s in strokes if len(s.x) >= 2]
    if lines:
        ax.add_collection(LineCollection(lines, colors="black", linewidths=ink_width * scale * 72.0 / dpi,
                                         capstyle="round", joinstyle="round"))
    canvas.draw()
    drawn = np.asarray(canvas.buffer_rgba())[..., 0]
    exact = np.full((height, width), 255, dtype=np.uint8)
    h, w = min(height, drawn.shape[0]), min(width, drawn.shape[1])
    exact[:h, :w] = drawn[:h, :w]
    return Image.fromarray(exact, mode="L")


def _overlaps(a, b):
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def mark_groups(targets):
    """`targets` (as mark_targets gives them) split into lists of indices whose marks'
    own boxes overlap, joined transitively; groups in page order, and each in page order."""
    parent = list(range(len(targets)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    boxes = [mark.bbox() for mark, _, _ in targets]
    for i in range(len(targets)):
        for j in range(i + 1, len(targets)):
            if _overlaps(boxes[i], boxes[j]):
                parent[root(j)] = root(i)
    groups = {}
    for i in range(len(targets)):
        groups.setdefault(root(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def _text_label(mark, block_rows, row):
    if row is not None:
        return row.label
    if (mark.meta or {}).get("attends_row", -1) == -1:
        return "\n".join(r.label for r in block_rows if r.label) or None
    return None


def write_marks(blocks, scale, ink_width, page_px, folder, rel_dir, name, index, crop="context"):
    """Save every mark sample's two crops under `folder`/`rel_dir`; return manifest rows.
    `crop` is one of CROPS (see the module docstring)."""
    if crop not in CROPS:
        raise ValueError(f"crop {crop!r} is not one of {CROPS}")
    pad = PAD if crop == "context" else ink_width / 2.0
    targets = mark_targets(blocks)
    if not targets:
        return []
    os.makedirs(os.path.join(folder, rel_dir), exist_ok=True)
    everything = [s for block in blocks for s in block.all_strokes() if len(s.x) >= 2]
    rows = []
    for k, group in enumerate(mark_groups(targets)):
        members = [targets[i] for i in group]
        for mark, _, _ in members:
            if mark.label not in MARK_TYPES:
                raise ValueError(f"a {mark.class_name} labelled {mark.label!r} is not one of {MARK_TYPES}")
        box = None
        for mark, _, row in members:
            b = list(mark.bbox())
            if crop == "context" and mark.class_name == "line_mark" and row is not None:
                r = row.bbox()
                b = [min(b[0], r[0]), min(b[1], r[1]), max(b[2], r[2]), max(b[3], r[3])]
            box = b if box is None else [min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3])]
        px = (max(0, math.floor((box[0] - pad) * scale)), max(0, math.floor((box[1] - pad) * scale)),
              min(page_px, math.ceil((box[2] + pad) * scale)), min(page_px, math.ceil((box[3] + pad) * scale)))
        if px[2] - px[0] < 2 or px[3] - px[1] < 2:
            continue
        bx0, by0, bx1, by1 = (v / scale for v in px)
        near = [s for s in everything
                if max(s.x) >= bx0 and min(s.x) <= bx1 and max(s.y) >= by0 and min(s.y) <= by1]
        own = {id(s) for mark, _, _ in members for s in mark.all_strokes()}
        parts = {"input": near,
                 "text": [s for s in near if id(s) not in own]}
        stem = f"{rel_dir}/{name}_m{k:02d}"
        for part, strokes in parts.items():
            _render_region(strokes, px, scale, ink_width).save(os.path.join(folder, f"{stem}_{part}.png"))

        shapes = [mark.label for mark, _, _ in members]
        labels = []
        for mark, block_rows, row in members:
            label = _text_label(mark, block_rows, row)
            if label and label not in labels:
                labels.append(label)
        rows.append({"input": f"{stem}_input.png", "text": f"{stem}_text.png",
                     "labels": [int(t in shapes) for t in MARK_TYPES], "shapes": shapes, "page": index,
                     "box": list(px), "crop": crop,
                     "mark_boxes": [[round(float(v) * scale, 3) for v in mark.bbox()] for mark, _, _ in members],
                     "text_label": "\n".join(labels) or None})
    return rows
