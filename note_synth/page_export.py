"""
page_export.py — the files a generated page is written out as.

Shared by synthesize.py and by training/'s render_seg_data.py and
render_page_val.py, so a dataset run and those scripts write the same thing.

A page is laid out on the board (layout.PAGE_W, the 1000 units every generator
is tuned in) and drawn at PAGE_PX pixels with the pen scaled from the board's
(seg_prompts.INK_WIDTH_PX). Both views of the page come from that one drawing:

  samples   the segmentation samples (seg_prompts.build_samples), each
            seg_prompts.IMAGE_SIZE square: the full page downsized, and each
            level-2 object cut out of the PAGE_PX page, scaled so its longer
            side fills the sample and centred on white — the cut seg_eval.py
            makes at inference, from the same resolution. One manifest row per
            prompt: {"image", "prompt", "target"} (write_samples).
  page      the PAGE_PX image, beside a JSON of every labelled object on it in
            that image's pixels (page_objects, page_record). A textline or
            formula also carries its transcription (a formula: its LaTeX) as
            "text" and its strokes as "ink", for the recognition step that
            validation and testing run on detected lines (training/recog_eval.py).
            An enclosure or line mark carries its strokes as "ink" too (points
            without times), so recognition sees the marks that cross a line and
            the mark separation network can take them out.
"""

from __future__ import annotations
import os

import numpy as np

PAGE_PX = 1600
FORMAT = "note_synth_page/1"
TEXT_CLASSES = ("textline", "formula")
MARK_CLASSES = ("enclosure", "line_mark")


class NoDirection(SystemExit):
    """An arrow whose start and end could not be told apart. A SystemExit, so a
    script that does not catch it stops with this message."""


def render_page(blocks, size, board_size, ink_width):
    """seg_prompts.render_board at `size` pixels: the same strokes and pen, scaled."""
    from PIL import Image
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure

    dpi, scale = 100, size / board_size
    fig = Figure(figsize=(size / dpi, size / dpi), dpi=dpi, facecolor="white")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.axis("off")
    # a vertex is a one-point stroke and draws nothing, as in render_board
    lines = [np.column_stack([np.asarray(s.x) * scale, np.asarray(s.y) * scale])
             for block in blocks for s in block.all_strokes() if len(s.x) >= 2]
    ax.add_collection(LineCollection(lines, colors="black", linewidths=ink_width * scale * 72.0 / dpi,
                                     capstyle="round", joinstyle="round"))
    canvas.draw()
    return Image.fromarray(np.asarray(canvas.buffer_rgba())[..., 0].copy(), mode="L")


def page_objects(blocks, scale, arrows=True):
    """Every labelled object on the page, walked as seg_prompts._collect walks it.

    Coordinates are board units times `scale` — the page image's pixels. A
    level-1 object names the level-2 object it sits under as `parent`; words and
    unknown ink are left out. Raises NoDirection for an arrow with no start/end.
    With `arrows` False arrows are left out instead, and nothing is raised: for a
    page only its lines are needed from (recognition samples, recog_pages.py).
    """
    import labels
    import seg_prompts as sp

    objects = []
    px = lambda values: [round(float(v) * scale, 3) for v in values]

    def walk(node, parent, shapes):
        meta = node.meta or {}
        if meta.get("flowchart"):
            shapes = {c.meta["id"]: c for c in node.children
                      if c.meta and c.meta.get("id") and c.class_name == "enclosure"}
        cls, under = node.class_name, parent
        entry = {"id": len(objects), "class": cls}
        if cls in labels.LEVEL2:
            entry.update(level=2, bbox=px(labels.label_bbox(node)))
            under = entry["id"]
        elif cls in labels.LEVEL1:
            entry["level"] = 1
            if parent is not None:
                entry["parent"] = parent
            if cls == "vertex":
                point = (node.strokes[0].x[0], node.strokes[0].y[0]) if node.strokes else labels.label_bbox(node)[:2]
                entry["position"] = px(point)
            elif cls in ("arc", "segment"):
                # an arc's start, middle and end, a segment's two ends
                entry["points"] = px([v for s in node.strokes for v in (s.x[0], s.y[0])])
            else:
                entry["bbox"] = px(labels.label_bbox(node))
                if cls in TEXT_CLASSES:
                    entry.update(text=_text(node.label), ink=_ink(node, scale))
                elif cls in MARK_CLASSES:
                    entry["ink"] = _ink(node, scale)
        elif cls == "arrow" and arrows:
            arrow = sp._arrow(node, shapes)
            if arrow.get("no_direction"):
                raise NoDirection(f"an arrow at {px(labels.label_bbox(node))} has no start and end "
                                  "(seg_prompts._arrow found none)")
            entry.update(level=0, start=px(arrow["board"][:2]), end=px(arrow["board"][2:]),
                         bbox=px(labels.label_bbox(node)))
        else:
            entry = None
        if entry is not None:
            objects.append(entry)
        for child in node.children:
            walk(child, under, shapes)

    for block in blocks:
        walk(block, None, None)
    return objects


def _text(label):
    """A line's transcription with runs of whitespace made one space, or None."""
    return " ".join(str(label).split()) or None if label is not None else None


def _ink(node, scale):
    """A line's strokes in page pixels (to 0.1), each a list of [x, y, t] points with t the time since the
    line's earliest point, in the source's units (only time since the start colours recognition ink, and
    IAM-OnDB's own times are large); a stroke without times (ink built rather than loaded) has [x, y] points."""
    strokes = [s for s in node.all_strokes() if len(s.x)]
    timed = [s.t for s in strokes if s.t is not None and len(s.t) == len(s.x)]
    start = min(float(min(t)) for t in timed) if timed else 0.0
    out = []
    for s in strokes:
        points = [[round(float(x) * scale, 1), round(float(y) * scale, 1)] for x, y in zip(s.x, s.y)]
        if s.t is not None and len(s.t) == len(s.x):
            for point, t in zip(points, s.t):
                point.append(round(float(t) - start, 3))
        out.append(points)
    return out


def page_record(name, objects, size, board_size, seed, index):
    """The page JSON: `name`.png's objects, as render_page_val.py writes them."""
    return {"format": FORMAT, "image": name + ".png", "image_size": [size, size],
            "page_size": [size, size], "coord_order": "x0y0x1y1",
            "source": {"generator": "note_synth", "seed": seed, "page": index,
                       "board_size": board_size},
            "objects": objects}


def sample_image(page, transform, scale, image_size):
    """A sample's image cut out of the page drawing: its board box, in page pixels,
    resampled to the transform's size and pasted on white at its padding."""
    from PIL import Image
    out_w, out_h = transform["size"]
    box = tuple(float(v) * scale for v in transform["board_box"])
    region = page.resize((out_w, out_h), resample=Image.LANCZOS, box=box)
    image = Image.new("L", (image_size, image_size), 255)
    image.paste(region, tuple(transform["pad"]))
    return image


def write_samples(blocks, page, scale, folder, rel_dir, name):
    """Save every sample that has a prompt as `folder`/`rel_dir`/`name`_sNN.png.

    Returns (manifest rows with image paths relative to `folder`, images saved).
    A drawing's crop has no prompt and is not saved.
    """
    import seg_prompts as sp

    os.makedirs(os.path.join(folder, rel_dir), exist_ok=True)
    rows, saved = [], 0
    for j, sample in enumerate(sp.build_samples(blocks)):
        if not sample["prompts"]:
            continue
        rel = f"{rel_dir}/{name}_s{j:02d}.png"
        image = sample_image(page, sample["transform"], scale, sp.IMAGE_SIZE)
        assert image.size == (sp.IMAGE_SIZE, sp.IMAGE_SIZE) and image.mode == "L"
        image.save(os.path.join(folder, rel))
        saved += 1
        parent = sample.get("parent") or {}
        for prompt in sample["prompts"]:
            row = {"image": rel, "prompt": prompt["prompt"], "target": prompt["target"]}
            if parent.get("class") == "flowchart" and parent.get("item"):
                # which chart this crop is, for synthesize.join_manifests: one per chart in train
                row["level1_chart"] = parent["item"]
            rows.append(row)
    return rows, saved
