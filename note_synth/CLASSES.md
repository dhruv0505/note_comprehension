# Label classes

Every class in the three-level taxonomy (`labels.py`), with an estimate of how
often each one appears on a generated page.

**Source of the estimate:** 100 pages generated with seed 0, the same generator
and seed `python make_pdf.py` uses (it matched that run's `pages_labels.json`
exactly). Counts are objects per page. "On pages" is the share of pages that have
at least one. If the generator's probabilities change, regenerate the pages and
recount.

**Pages** are square: laid out at 1000 × 1000 and scaled to **224 × 224**, labels
included. (The counts below predate the corner-grown layout — 9–19 cells a page —
and the size-weighted choice of what each cell holds; regenerate and recount.)

`word` (level 0) is left out on purpose.

## Level 2: blocks

| Class | Avg / page | Median | Min–Max | On pages | What it is |
|---|---:|---:|---:|---:|---|
| `textblock` | **8.71** | 9 | 0–28 | 90% | Any block of writing (see below) |
| `geometry`  | **0.52** | 1 | 0–1  | 52% | A textbook geometry figure built from mined ink, its labels as textlines inside its box |
| `drawing`   | **0.33** | 0 | 0–3  | 25% | A QuickDraw sketch |
| `table`     | **0.29** | 0 | 0–3  | 23% | A grid ruled in mined ink — every column divider, and either every row divider or just the header's — each cell a textline or formula, held by the table directly |
| `flowchart` | **0.76** train, **1.26** val | | | | An FC_1.0 flowchart boxed as a whole: its shapes `enclosure`, its arrows `arrow`, its captions (in a shape or on an arrow) textlines or formulas held by the chart directly |

**≈ 9.9 level-2 blocks per page**

**Every line of writing is inside a textblock, a table, a geometry figure or a
flowchart.** A table cell's line, a figure's label and a flowchart's caption (in
a shape or on an arrow) sit in the table, figure or chart directly. So a
textblock is a paragraph, a maths section or a bulleted list. List bullets are
unlabelled ink beside their rows, outside the rows' boxes and transcriptions.

**Single-line textblocks.** A textblock of one line doesn't need line detection
(captions, which were most of them, are now lines of their flowchart; the numbers
below were measured before that). `labels.is_single_line(strokes)` calls a block one line
when its stroke at the 89th height percentile is at least 46.3% of the block's
height. It's one-sided: on 150 unseen pages it called **no** multi-line block a
single line (0 of 352) and caught 77.9% of the 1,054 single-line blocks; the rest
still get line detection. Pass it the block's own strokes only, not an enclosure,
line mark or arrow drawn over it.

## Level 1: lines

| Class | Avg / page | Median | Min–Max | On pages | What it is |
|---|---:|---:|---:|---:|---|
| `formula`  | **14.07** | 13 | 0–51 | 90% | A row of maths (MathWriting), always inside a textblock, a table or a geometry figure |
| `textline` | **9.57**  | 9  | 0–26 | 89% | A line of handwritten prose, always inside a textblock, a table or a geometry figure |
| `arc`      | not yet counted | | | | Every circle or arc of one a geometry figure draws, as its start, middle and end, counterclockwise (below). No ellipses |
| `segment`  | not yet counted | | | | Every straight line of a geometry figure, as its two ends (below) |
| `vertex`   | not yet counted | | | | Every vertex of a geometry figure, named or not: a zero-area point box with fields of its own (below) |
| `enclosure` | **4.86** | 5 | 0–20 | 76% | Any shape drawn round something (see below), held by the flowchart or textblock it marks |
| `line_mark` | not yet counted | | | | An underline (single or double) or a strikethrough, one class for both, held by its textblock |

**≈ 23.6 level-1 lines per page** (before vertices were counted here)

A `vertex` object carries, besides its box: `position` `[x, y]`; `id`; `angles`
(degrees, one per angle it makes inside the figure — one at a polygon's corner,
more where more lines meet); `angle_mark` (`none`, `right` or `arc`) and
`angle_mark_count` (arcs, or 1 for a right-angle box); and `connected` (ids of
the vertices a drawn line or arc joins it to).

An `arc` target is `x1 y1 x2 y2 x3 y3 arc`: its start, middle and end,
going counterclockwise as the image is seen; a full circle starts and ends at its
0° point with its 180° point between. A `segment` target is `x1 y1 x2 y2
segment`: its upper end first.
In the page JSON both carry `points`. Right-angle boxes,
angle arcs and ticks are marks, not `arc` or `segment` objects.

Every textline carries its transcription, and no text is written twice on a page.

No two lines — textline against textline, textline against formula, or formula
against formula — overlap by more than **30% of the smaller one's box area**
(`block_generators.LINE_MAX_OVERLAP`). A paragraph row, prose or an inline
expression, that would reach further into the line above is pushed down (and rows
that no longer fit the cell are left out rather than squeezed); a flowchart caption
landing on another is taken off, an arrow's caption before a shape's; and a page
that still breaks the rule is generated again. Table cells, geometry labels and a
maths section's rows never overlap at all, which is stricter, and stay that way.

## Level 0: primitives (without `word`)

| Class | Avg / page | Median | Min–Max | On pages | What it is |
|---|---:|---:|---:|---:|---|
| `arrow`     | **5.27** | 5   | 0–22 | 84% | A flowchart edge (real FC_1.0 ink) or a connector arrow between blocks (~1.06 / page, mined ink) |
| `unknown`   | **0.00** | 0   | 0–0  | 0%  | Unnameable ink. The class exists but nothing generates it yet |

(Counts predate enclosures and line marks moving to level 1.) Arrows are the only
objects that stand on the page on their own.

**Enclosures and line marks** (level 1) hold nothing, and what they mark keeps its
own place. Each records the text it marks — the row, the whole block, or a
flowchart shape's caption — only to order it among its holder's level-1 targets.

| Enclosure drawn as… | Avg / page |
|---|---:|
| a flowchart's shapes | 4.57 |
| a box or oval round a textblock, a line, or words in a line | 0.07 |

An `enclosure` doesn't record its shape as a class: a flowchart's rectangle,
stadium, diamond, parallelogram or circle and a box or oval drawn in text are all
`enclosure`, with the shape kept in the label. A table's outer border is not
labelled. A
`line_mark` is an underline or a strikethrough on a text line (never on a
formula), with which it is kept in the label; a double underline is labelled
`underline` like a single one. Boxing, circling, underlining or striking through
a word never splits its line: the line stays whole.

A connector arrow's shaft is several real hand-drawn strokes placed end to end,
with a straight last stretch into its target. Its head is two barbs, one V stroke,
or a closed triangle of three strokes. On a text line it lands near the middle of
a side; on a drawing, flowchart or figure it points straight into the object.
Loops are uncommon and decided per page.

## Relations

There are none. A page is its per-level boxes; a caption on an arrow is a line of
its flowchart like any other caption, and a label on a geometry figure is a
textline of the figure.

## Hierarchy at a glance

This tree comes from checking which box contains which in the generated labels.

```
page (224 × 224)
├── textblock ─── textline / formula       (prose, maths, a list)
│            └── enclosure / line_mark     (a box or oval round text; an underline or strikethrough)
├── flowchart ─── textline / formula   (a caption, in a shape or on an arrow)
│            └── enclosure             (each shape)
├── table ─────── textline / formula   (one per cell, no textblock between)
├── geometry ──── textline / formula   (a label; inside the geometry's box)
│            └── arc / segment / vertex   (every circle or arc, straight line and vertex of the figure)
├── drawing       (no labelled children)
└── arrow         (flowchart edge, or connector between blocks)
```

A flowchart is boxed as a whole and holds its captions as lines and its shapes as
enclosures. Its arrows stand on the page on their own, as every arrow does.

Unlabelled ink: list bullets (beside their rows). Underlines, table rules, boxes
and ovals round writing, connector arrows (loops included) and geometry are all
drawn from mined hand-drawn lines and ovals.

## Removed classes

| Removed | Now |
|---|---|
| `diagram` | `flowchart` (level 2). Its shapes are `enclosure`, its arrows `arrow`, its captions its lines |
| `rectangle`, `oval`, `diamond`, `data`, `circle` | All `enclosure` |
| `augmentation` | Gone: split back into `enclosure` (shapes) and `line_mark` (underlines, strikethroughs) |
| `enclosure_around` | Gone. An enclosure is held by what it marks, and records the text it marks only for ordering |
| `textline_on_arrow`, `label_on_vertex`, `label_on_edge`, `label_on_angle` | Gone, with all relations. A caption is a line of its flowchart; a figure's label is a textline of the figure |
| `list` | `textblock` |
