"""
recog_layout.py — how a recognition sample is drawn, and what it is asked.

    textline, formula   ROWS items stacked down one CANVAS x CANVAS image
    drawing             one QuickDraw sketch filling a CANVAS x CANVAS image

A text sample. Each item's ink is drawn so that it, pen included, is ROW_HEIGHT
pixels tall. An item too wide for the canvas at that height is shrunk as a
whole, ink and pen together, until it is CANVAS wide, and centred in its
ROW_HEIGHT slot. Every row is left-aligned, as writing is, and padded on the
right to CANVAS wide. Slots are stacked ROW_GAP apart; the gap under the last
row is not kept, and what remains of the canvas is split evenly above the first
row and below the last, the odd pixel going below (row_tops). With the
defaults: 10 x 55 + 9 x 20 = 730 pixels of rows, 19 above and 19 below.

A drawing sample. The sketch, pen included, is scaled so its longer side is
CANVAS and centred on white — the cut seg_prompts.crop_transform makes of a
level-2 drawing, so the classifier is shown what segmentation hands it.

Ink colour. Every item is drawn in RGB on white with time and distance in the
colour channels (Fadeeva et al., 2024, "Representing Online Handwriting for
Recognition in Large Vision-Language Models"). For an item I whose strokes are
s_i = [(x_i,0, y_i,0, t_i,0), ..., (x_i,n, y_i,n, t_i,n)]:

    R_i,j = (t_i,j - t_0,0) / max (t_m,n - t_0,0)
    G_i,j = |dx_i,j| / max |dx_m,n|
    B_i,j = |dy_i,j| / max |dy_m,n|

with dx_i,j = x_i,j+1 - x_i,j and dy likewise, the step to the stroke's next
point, as the paper defines it. The maxima are over the whole item; a channel
whose maximum is 0 is 0. A stroke's last point has no next point and takes its
previous point's step (a one-point stroke's is 0). t_0,0 is the item's first
point in writing order, taken as its earliest time: an IAM line's strokes are
listed by word, left to right, which is not always the order they were written.
The stretch from point j to j + 1 is drawn in point j's colour, round-capped,
and strokes are drawn in the order they were written, so a later stroke crosses
over an earlier one. Times are the source's own (IAM-OnDB's point times,
MathWriting's T channel, QuickDraw's raw times, all kept by note_synth's
loaders as Stroke.t), and an item without them is refused (polylines).

Ink is drawn SUPERSAMPLE times larger and box-filtered down.

Targets. A text sample's is each row's transcription, top to bottom, joined by
SEP, a newline: in Florence-2's byte-level BPE tokenizer that is one token of the
base vocabulary (Ċ, 50118), trained on from pretraining, where `<sep>` is one
processing_florence2.py adds for polygons. clean_label turns every run of
whitespace in a label into one space, so no transcription holds a newline, and
leaves out labels holding a string the tokenizer reads as a special token
(SPECIAL_TOKENS, <loc_N>), so every target is tokenized as the text it is
(training/check_recog_targets.py checks a manifest against the tokenizer). A
text sample holds up to ROWS rows, filling its layout's slots from the top. A
drawing's target is its QuickDraw category.

Nothing here imports note_synth: an item is anything with `all_strokes()`
(core.Ink) or a list of strokes, and a stroke anything with `.x`, `.y` and `.t`.
"""

from __future__ import annotations
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

CANVAS = 768
ROWS = 10
ROW_HEIGHT = 55
ROW_GAP = 20
SUPERSAMPLE = 4
BACKGROUND = (255, 255, 255)
# How ink is coloured, recorded in a run's config.json so that a run is never
# finished with images drawn another way (recog_synthesize._check_config).
INK = "rgb_time_dx_dy"

# Pen width in output pixels, drawn per row (text) or per image (drawing) from
# these ranges. Rough estimates, not measurements: a page is drawn with a 3.2 px
# pen at 1600 px (page_export), and a textline cut from it and brought to 55 px
# keeps roughly 2-4 px of that; a drawing's crop is scaled up far more, to 768 on
# its longer side, so its pen comes out several pixels wide. Tune with --pen.
TEXT_PEN = (2.0, 4.0)
DRAWING_PEN = (3.0, 9.0)

SEP = "\n"
# Strings Florence-2's tokenizer reads as one special token wherever they stand (BART's own and those
# processing_florence2.py adds): a label holding one would not be trained on as the text it is.
SPECIAL_TOKENS = ("<s>", "</s>", "<pad>", "<unk>", "<mask>", "<od>", "</od>", "<ocr>", "</ocr>",
                  "<cap>", "</cap>", "<ncap>", "</ncap>", "<dcap>", "</dcap>", "<grounding>", "</grounding>",
                  "<seg>", "</seg>", "<sep>", "<region_cap>", "</region_cap>", "<region_to_desciption>",
                  "</region_to_desciption>", "<proposal>", "</proposal>", "<poly>", "</poly>", "<and>")
LOC_TOKEN = re.compile(r"<loc_\d+>")
PROMPTS = {
    "textline": "What is written in this image in English?",
    "formula": "What is written in this image in LaTeX?",
    "drawing": "What is drawn in this image?",
}
TEXT_TASKS = ("textline", "formula")
TASKS = TEXT_TASKS + ("drawing",)

# Times are never made up. Recorded ink keeps its times and only ever moves as a whole section; a section without
# recorded times (a mark, which note_synth builds) takes a single time. Between two sections of writing — one line and
# the next below it, a line and a mark made on it — there is a pause of this share of the longer one's duration,
# drawn uniformly (section_pause). recog_synth's samples and training/recog_eval.py's cascade both time ink this way.
SECTION_PAUSE = (0.05, 0.3)


def section_pause(u: float, *durations: float) -> float:
    """The pause between two sections of writing, for a uniform draw `u` in [0, 1) and their durations."""
    lo, hi = SECTION_PAUSE
    return (lo + u * (hi - lo)) * max(max(durations, default=0.0), 1e-6)

_EPS = 1e-9


# ----------------------------------------------------------------------------
# Labels and targets
# ----------------------------------------------------------------------------

def clean_label(label) -> Optional[str]:
    """A label as it goes into a target — runs of whitespace, newlines included,
    made one space — or None when it cannot go in: missing, blank, or holding a
    string the tokenizer reads as a special token (SPECIAL_TOKENS, <loc_N>)."""
    if label is None:
        return None
    text = " ".join(str(label).split())
    if not text or SEP in text or any(token in text for token in SPECIAL_TOKENS) or LOC_TOKEN.search(text):
        return None
    return text


def text_target(labels: Sequence[str]) -> str:
    """The rows' transcriptions, top to bottom, joined by SEP."""
    cleaned = [clean_label(label) for label in labels]
    if any(text is None for text in cleaned):
        raise ValueError(f"a label cannot go into a target: {list(labels)!r}")
    return SEP.join(cleaned)


# ----------------------------------------------------------------------------
# Drawing ink
# ----------------------------------------------------------------------------

def polylines(item) -> List[np.ndarray]:
    """An item's strokes as (n, 3) float arrays of x, y, t, empty strokes dropped.
    Raises ValueError when a stroke does not carry one time per point."""
    strokes = item.all_strokes() if hasattr(item, "all_strokes") else item
    lines = []
    for s in strokes:
        if not len(s.x):
            continue
        t = getattr(s, "t", None)
        if t is None or len(t) != len(s.x):
            raise ValueError("an item's strokes carry no per-point times (Stroke.t), which the ink's "
                             "colour needs: its source was loaded or cached without them")
        lines.append(np.column_stack([np.asarray(s.x, dtype=np.float64), np.asarray(s.y, dtype=np.float64),
                                      np.asarray(t, dtype=np.float64)]))
    return lines


def ink_colours(lines) -> List[np.ndarray]:
    """Each point's (R, G, B), 0-255, one (n, 3) uint8 array per stroke of `lines`
    (polylines'): time since the item's first point and the absolute x and y step
    to the next point, each over its maximum in the item (module docstring)."""
    steps = []
    for line in lines:
        step = np.abs(np.diff(line[:, :2], axis=0))
        # the last point has no next one and keeps its previous point's step
        steps.append(np.vstack([step, step[-1:]]) if len(step) else np.zeros((1, 2)))
    t0 = min(float(line[:, 2].min()) for line in lines)
    t_span = max(float(line[:, 2].max()) for line in lines) - t0
    d_max = np.vstack(steps).max(axis=0)
    colours = []
    for line, step in zip(lines, steps):
        red = (line[:, 2] - t0) / t_span if t_span > 0 else np.zeros(len(line))
        green = step[:, 0] / d_max[0] if d_max[0] > 0 else np.zeros(len(line))
        blue = step[:, 1] / d_max[1] if d_max[1] > 0 else np.zeros(len(line))
        colours.append(np.rint(np.column_stack([red, green, blue]) * 255.0).clip(0, 255).astype(np.uint8))
    return colours


def _extent(lines) -> Tuple[np.ndarray, float, float]:
    points = np.concatenate(lines)[:, :2]
    origin = points.min(axis=0)
    w, h = points.max(axis=0) - origin
    return origin, float(w), float(h)


def _pixels(value: float) -> int:
    return max(1, int(math.ceil(value - 1e-6)))


def row_fit(lines, pen: float, height: int = ROW_HEIGHT, width: int = CANVAS) -> Dict:
    """How to draw `lines` as a row: {origin, scale, pen, size}, a point landing
    at (point - origin) * scale + pen / 2.

    Ink and pen together are `height` tall. Where that is wider than `width`,
    the whole row is shrunk by one factor until it is `width` wide — the pen with
    it, as resizing a drawn image would. A flat item (no height) is stretched to
    `width` instead, and a single point is just the pen's dot.
    """
    origin, w, h = _extent(lines)
    if h > _EPS:
        scale = (height - pen) / h
    elif w > _EPS:
        scale = (width - pen) / w
    else:
        scale = 1.0
    full = w * scale + pen
    if full > width:
        shrink = width / full
        scale, pen = scale * shrink, pen * shrink
    return {"origin": origin, "scale": scale, "pen": pen,
            "size": (min(width, _pixels(w * scale + pen)), min(height, _pixels(h * scale + pen)))}


def drawing_fit(lines, pen: float, canvas: int = CANVAS) -> Dict:
    """row_fit's placement for a drawing: ink and pen together `canvas` on the
    longer side."""
    origin, w, h = _extent(lines)
    longest = max(w, h)
    scale = (canvas - pen) / longest if longest > _EPS else 1.0
    return {"origin": origin, "scale": scale, "pen": pen,
            "size": (min(canvas, _pixels(w * scale + pen)), min(canvas, _pixels(h * scale + pen)))}


def draw(lines, placement: Dict) -> Image.Image:
    """`lines` drawn on white as `placement` says, at its size, in RGB: each point's
    stretch to the next in the point's ink_colours colour, round-capped, and the
    strokes in the order they were written."""
    w, h = placement["size"]
    ss = SUPERSAMPLE
    pen = placement["pen"] * ss
    radius = pen / 2.0
    width = max(1, int(round(pen)))
    big = Image.new("RGB", (w * ss, h * ss), BACKGROUND)
    canvas = ImageDraw.Draw(big)
    colours = ink_colours(lines)
    for k in sorted(range(len(lines)), key=lambda k: float(lines[k][0, 2])):
        points = ((lines[k][:, :2] - placement["origin"]) * placement["scale"] + placement["pen"] / 2.0) * ss
        coords = [tuple(p) for p in points.tolist()]
        fills = [tuple(c) for c in colours[k].tolist()]
        for j, ((x, y), fill) in enumerate(zip(coords, fills)):
            if j + 1 < len(coords):
                canvas.line([(x, y), coords[j + 1]], fill=fill, width=width)
            canvas.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)   # round caps and joins
    return big.resize((w, h), Image.BOX)


# ----------------------------------------------------------------------------
# Samples
# ----------------------------------------------------------------------------

def row_tops(rows: int = ROWS, row_height: int = ROW_HEIGHT, gap: int = ROW_GAP,
             canvas: int = CANVAS) -> List[int]:
    """The top pixel of each row's slot: rows and the gaps between them centred
    on the canvas, the odd pixel of the remainder below."""
    used = rows * row_height + (rows - 1) * gap
    if rows < 1 or used > canvas:
        raise ValueError(f"{rows} rows of {row_height} px, {gap} px apart, need {used} px; "
                         f"the canvas is {canvas}")
    top = (canvas - used) // 2
    return [top + i * (row_height + gap) for i in range(rows)]


def text_sample(items, pens: Sequence[float], row_height: int = ROW_HEIGHT,
                gap: int = ROW_GAP, canvas: int = CANVAS, slots: Optional[int] = None):
    """(image, rows): `items` stacked down a canvas-square RGB image, one row each,
    and each row's {box: [x0, y0, x1, y1] in the image, scale, pen, shrunk}.
    `pens` is each row's pen width at full row height. `slots` is how many rows
    the layout has (default: one per item): the items fill its first slots from
    the top and the rest stay white, as the cascade draws a group of fewer lines."""
    if len(pens) != len(items):
        raise ValueError(f"{len(items)} items but {len(pens)} pen widths")
    tops = row_tops(slots or len(items), row_height, gap, canvas)
    if len(items) > len(tops):
        raise ValueError(f"{len(items)} items but {len(tops)} slots")
    image = Image.new("RGB", (canvas, canvas), BACKGROUND)
    rows = []
    for item, pen, top in zip(items, pens, tops):
        lines = polylines(item)
        if not lines:
            raise ValueError("an item has no ink")
        placement = row_fit(lines, pen, row_height, canvas)
        w, h = placement["size"]
        y = top + (row_height - h) // 2
        image.paste(draw(lines, placement), (0, y))
        rows.append({"box": [0, y, w, y + h], "scale": placement["scale"],
                     "pen": placement["pen"], "shrunk": placement["pen"] < pen - 1e-9})
    return image, rows


def drawing_sample(item, pen: float, canvas: int = CANVAS):
    """(image, placed): `item` filling a canvas-square RGB image on its longer side,
    centred, and its {box, scale, pen}."""
    lines = polylines(item)
    if not lines:
        raise ValueError("an item has no ink")
    placement = drawing_fit(lines, pen, canvas)
    w, h = placement["size"]
    x, y = (canvas - w) // 2, (canvas - h) // 2
    image = Image.new("RGB", (canvas, canvas), BACKGROUND)
    image.paste(draw(lines, placement), (x, y))
    return image, {"box": [x, y, x + w, y + h], "scale": placement["scale"], "pen": placement["pen"]}
