# Synthetic note-page generator

Composes full-page handwritten "notes" for training a segmentation model, out of:
- **FC_1.0**     -> flowcharts, boxed as a level-2 `flowchart` (its shapes level-1 `enclosure`s and its captions lines of the chart, + arrows)
- **IAM-OnDB** -> textlines/words (feeds textblocks — bulleted lists included — and table cells, which hold textlines or formulas); the only prose source by default
- **DeepWriting** -> loaded but OFF for prose by default (see `PROSE_SOURCES`)
- **MathWriting** -> whole maths sections, plus the odd expression inline in prose
- **QuickDraw**   -> drawings
- **mined ink**   -> geometry figures: real lines/ovals found *inside* the pools above by `shape_mining.py`, re-warped into textbook constructions (every straight-sided shape, up to a decagon, is closed out of mined lines); also the box or oval drawn round a block, line or word, a table's rules, underlines, and the shaft and head of every connector arrow
- **procedural**  -> list bullets, and where connector arrows run (no source has these natively)

IAMonDo is intentionally NOT used here — hold it out entirely for evaluation.

## Files
- `core.py`             — Stroke/Ink data model, geometric transforms (fit-to-box). Strokes are x/y only — no timestamps
- `loaders.py`           — per-dataset parsers -> common Ink format (all five implemented; each reads straight out of the shipped archive, no unpacking)
- `iam_writers.py`       — recovers IAM-OnDB's writer IDs from DeepWriting (which carries IAM-OnDB's own ink, tagged by writer) -> `data/iamondb_writers.json`
- `block_generators.py`  — builds one labeled Ink tree per block (textblock/table/flowchart/drawing; a bulleted list is a textblock, its bullets unlabelled ink beside the rows, outside their boxes and transcriptions) + procedural primitives (rect/oval/arrow/bullet/grid strokes)
- `layout.py`            — the page's grid of cells, one page-wide text scale
- `labels.py`            — extracts hierarchical (level 0/1/2) labels, a vertex with its angles, marking and neighbours, exports JSON and the paper's "x0 y0 x1 y1 class ..." target-string format
- `shape_mining.py`      — finds real hand-drawn lines / ovals / rectangles inside the source pools
- `geometry_diagrams.py` — builds textbook geometry figures out of that mined ink, keeping the exact construction behind every stroke
- `page_generator.py`    — fills each grid cell with a flowchart / geometry figure / drawing / table / textblock, assembles a full page
- `demo.py`               — runnable end-to-end check against `data/`, falling back to mocks for anything missing, + SVG preview renderer
- `make_pdf.py`           — generate N pages into one PDF (each followed by its ground-truth page) plus their label JSON
- `seg_prompts.py`        — the segmentation prompts (level 2, arrows, level 1 per textblock/table/geometry/flowchart), their target ordering, the 224×224 page and per-object crop transforms, and the targets; `--check-order` spot-checks level-2 creation order
- `seg_pages.py`          — the same pages as segmentation samples: the full page and every level-2 object's crop, each as its 224×224 image, its ground truth and its prompts (3 PDF pages a sample), plus a label JSON of every sample

## Quickstart
```bash
python3 demo.py            # real datasets where present in data/, mocks for the rest
python3 demo.py --mock     # mocks only; needs no data files

python3 make_pdf.py                    # 25 pages -> pages.pdf + pages_labels.json
python3 make_pdf.py --pages 20 --boxes 1   # fewer pages, with level-1 boxes drawn
python3 make_pdf.py --no-label-pages       # ink pages only

python3 seg_pages.py                   # the same 25 pages -> seg_pages.pdf + seg_pages_labels.json
python3 seg_pages.py --only 9          # page 9 alone
```

`seg_pages.py` is the segmentation view, as prompts (`seg_prompts.py`). The full
page, rasterised on the 1000×1000 board and downsized to 224×224, is asked for its
level-2 objects (textblock, drawing, geometry, table, flowchart) and its arrows.
Each level-2 object is then cut out of the board, scaled and padded to its own
224×224 image, and asked for its level-1 objects: textline, formula, enclosure and
line mark in a textblock; textline and formula in a table; textline, formula and
enclosure in a flowchart;
textline, arc, segment and vertex in a geometry figure (a drawing has none).
Level-1 coordinates are in that crop, so every sample stores its transform. A box
is top-left then bottom-right, a vertex one point, an arrow start then end (found
as the ground-truth page finds them), an arc its start, middle and end going
counterclockwise (`x1 y1 x2 y2 x3 y3 arc`), and a segment its two ends
(`x1 y1 x2 y2 segment`). Targets are grouped by class in prompt order; level
2 in cell creation order (on real boxes over 300 pages of 17+ cells, 5 of 31,030
cross-row pairs came out of visual order); a textblock's or table's lines in
creation order; a geometry figure's and a flowchart's lines, arcs, segments and vertices, and
arrows, by exact (y, x); enclosures and line marks by the text they mark (the row
counted down the block, a box round the whole block first; a
flowchart shape by its caption). A table holds its cells' lines directly (its border
is not labelled), a geometry figure its labels, and a flowchart its captions and shapes.
Same seed and page count as `make_pdf.py`, so its page N is page N of `pages.pdf`.

Every rendered page is followed by a **ground-truth page**: each labelled object
boxed and coloured by level (blue level 2, green level 1, red level 0),
handwriting replaced by the text it actually says, maths by its LaTeX, a drawing
kept as ink under its category name, and every arrow reduced to a dotted line
from start to end. Reading the pair together is the quickest way to check that
what the generator drew and what it claims to have drawn agree.

Pages render as plain **black** ink, which is what the segmentation model should
see. `make_pdf.py --color-by-class` and `--boxes N` tint strokes / draw ground
truth for inspection only — never train on those, they leak the labels into the
image. Nothing in the `Ink` model carries color; it is purely a renderer option.

## Data

Loaders default to these paths and read **inside the archives** — nothing needs
unpacking. Every loader also accepts an explicit path or an extracted directory.

| File | Dataset | How to get it |
|---|---|---|
| `data/iamondb_linestrokes.tar.gz` | IAM-OnDB line strokes | `lineStrokes-all.tar.gz` from the [IAM-OnDB download page](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database) (registration required) |
| `data/iamondb_ascii.tar` | IAM-OnDB transcriptions | `ascii-all.tar.gz` from the same page |
| `data/iamondb_writers.json` | IAM-OnDB writer IDs | built here: `python iam_writers.py` (reads the two IAM-OnDB files and `data/deepwriting.zip`); optional — without it each form stands in for its writer |
| `data/deepwriting.zip` | DeepWriting | [DeepWriting dataset](https://ait.ethz.ch/deepwriting) (`deepwriting_training.npz` + `deepwriting_validation.npz`) |
| `data/mathwriting.tgz` | MathWriting 2024 | `curl -L -o data/mathwriting.tgz https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz` (~4GB) |
| `data/fc.zip` | FC_1.0 flowcharts (InkML) | the FC database of Bresler, Průša and Hlaváč (CMP, CTU Prague; IJDAR 2016, DOI 10.1007/s10032-016-0269-z), `FC_database_1.0.zip` as downloaded |
| `data/quickdraw.zip` | QuickDraw full raw | see below |
| `data/emnist.zip` | NIST SD19 (EMNIST's source) | the NIST Special Database 19 1st Edition (1995) release, zipped as downloaded |

### EMNIST / NIST SD19 is the odd one out

Every other source is **online** ink — a pen sampled as it moved. SD19 is
**offline**: scanned paper, 128×128 bilevel rasters, no pen path at all. So
`load_emnist` vectorizes it — thins each glyph to a one-pixel skeleton and
traces that into polylines — and what you get back is an honest *reconstruction*
of the pen's path, not a recording of one. Stroke order and direction are
inferred. Everything downstream sees the same
`Ink`/`Stroke` shape as the online sources.

No unpacking or format conversion is needed: the loader reads the `.mis` image
sets in place, out of the zip. Each is a NIST IHead header followed by one CCITT
Group 4 stream holding the whole class stacked into a single tall bitmap, and
because Group 4 decodes strictly top to bottom, asking for `max_per_class`
glyphs reads only those rather than all several thousand.

`load_emnist_pool` is the same thing keyed by character, which is what
`geometry_diagrams` writes with when it names a figure's corners.

DeepWriting's training and validation splits are concatenated: IAMonDo is what we
hold out for val/test, so DeepWriting's own split buys us nothing here. For the
same reason `load_mathwriting` reads only `train`/`synthetic`/`symbols` and skips
the release's `valid`/`test` directories.

**MathWriting must come from the official release, not Hugging Face.**
`andito/mathwriting-google` is rendered JPEGs plus LaTeX strings — it has no
stroke coordinates, so it cannot feed a pipeline that needs pen strokes.

QuickDraw (the raw dump is ~23GB; `load_quickdraw(max_per_category=...)` reads it
straight out of the zip):
```bash
mkdir -p data/quickdraw_raw && cd data/quickdraw_raw
gsutil -m cp 'gs://quickdraw_dataset/full/raw/*.ndjson' raw/
n=$(ls raw/*.ndjson | wc -l); echo "Downloaded $n files"
[ "$n" -ge 345 ] || { echo "expected 345, got $n — not deleting raw/"; exit 1; }
zip -r ../quickdraw.zip raw/ && zip -T ../quickdraw.zip
[ "$(unzip -l ../quickdraw.zip | grep -c '\.ndjson$')" -eq "$n" ] || { echo "count mismatch"; exit 1; }
cd .. && rm -rf quickdraw_raw
```
The loader reads the **raw** files' x/y and ignores their per-point times.

### Small stand-ins (already present)
`make_pdf.py` falls back to these when the full download is absent, so pages can
be generated before the big datasets land. Delete them once the real files are in
place; the loaders themselves always prefer the canonical name.

| File | Stands in for | What it is |
|---|---|---|
| `data/mathwriting_excerpt.tgz` | `data/mathwriting.tgz` | the official 500-ink excerpt build (1.6MB) |
| `data/quickdraw_sample.zip` | `data/quickdraw.zip` | ~400 sketches each for 16 categories, HTTP-range-fetched prefixes of the real raw `.ndjson` files (6.6MB) |

## Generating at scale
```python
from page_generator import generate_page
from demo import load_real_pools

pools = load_real_pools()                 # or call the loaders directly, without limits
pages = [generate_page(**pools) for _ in range(N)]
```
For actual training input, render `page["blocks"]` as a **plain image** (ink
   strokes on a plain background) — `demo.py`'s SVG renderer is already the right
   shape for this, it just needs nicer stroke rendering (proper width/antialiasing)
   instead of raw polylines. NOTE: the paper's Eq. 2 time/distance-in-color-channels
   rendering is specific to the *recognition* task (it helps disambiguate similar
   character shapes via stroke order/direction) — it is not used for segmentation,
   which trains on a plain rendering of the strokes. Don't apply Eq. 2 here.

## Building the dataset

Train, val and test are split **at the source**, before any page is made, and
then each split is generated in parallel from its own content only.

```bash
python iam_writers.py            # data/iamondb_writers.json — IAM-OnDB writer IDs, once
python dataset_splits.py         # splits.json — which content belongs to which split
python split_pools.py --cache cache --mathwriting /data/mathwriting-2024.tgz --quickdraw data/quickdraw.zip
python synthesize.py --cache cache --split train --pages 30000 --workers 32 --out dataset
python synthesize.py --cache cache --split val   --pages 2000  --workers 32 --out dataset
python synthesize.py --cache cache --split test  --pages 2000  --workers 32 --out dataset
```

**Splits** (`dataset_splits.py`), 80/10/10 on the unit that carries identity:
IAM-OnDB forms grouped so no writer and no text is in two splits (forms whose
writer is unknown stay in train, and forms sharing a passage stay together); FC by
layout AND writer: val and test each get 5 of the 28 layouts, chosen for room to
write in (`FC_EVAL_LAYOUTS`; layouts that are one chart drawn twice stay together),
and the release's own writer lists (000-009 / 010-016 / 017-023), so 180/35/35 of
its 672 charts are used; MathWriting
by the release's own train+synthetic / valid / test; EMNIST by SD19 writer set
(hsf_0-3,6 / hsf_4 / hsf_7); QuickDraw by a hash of each drawing's `key_id`.

**Caches** (`split_pools.py`): per split, that split's lines, charts, drawings and
glyphs pickled, its MathWriting as a memory-mapped store (`mathwriting_store.py`,
shared by every worker on a machine), and a shape bank mined from that split
alone — a bank records its split and refuses to load for another.

**Outputs** (`synthesize.py`), under `dataset/<split>/`. Every page is laid out
on the generator's 1000-unit board and drawn at 1600 × 1600 pixels
(`page_export.py`, which `training/render_seg_data.py` and
`render_page_val.py` write through too, so all three produce the same files):
- `samples/` (train, val): `manifest.jsonl`, one row per prompt
  (`{"image", "prompt", "target"}`), and the 768 × 768 sample images: the full
  page downsized, and each level-2 object cut out of the 1600 page and scaled to
  fit — the cut `seg_eval.py` makes at inference.
- `pages/` (val, test): `pageNNNNNNN.png`, the 1600 × 1600 page, beside
  `pageNNNNNNN.json` listing every labelled object on it in page pixels — boxes,
  vertex positions, arrow start and end, each object's level, and for a level-1
  object the `parent` level-2 object it sits in (read only by
  `render_page_val.py`'s checks; `seg_eval.py` ignores it and it is in no
  training target).
- `marks/` (every split): for each enclosure and line mark inside a textblock, a
  crop of the 1600 page (`_input.png`) and the same crop as the writing alone
  (`_text.png`) and the mark alone (`_mark.png`), with `manifest.jsonl` giving the
  crop box, the mark's class, shape and box, and the marked row's text
  (`mark_crops.py`). Flowchart shapes are not included.
  `--mark-crop box` crops each mark to its own bounding box instead (marks whose
  boxes overlap are one crop of their joined box), padded only by half the pen.
- Train, with `--geometry-per-page N`: N geometry figures a page drawn without a
  page (`synthesize.geometry_sample`), from the split's own shape bank, glyphs and
  MathWriting store. Each is fitted into a board cell of a page cell's size, cut out
  with its box grown a little on each side as a predicted box is, sometimes with a
  sketch beside it, and asked only the geometry level-1 prompt
  (`samples/images/<shard>/g0000123.png`, in the samples manifest). The report counts
  them as `geometry_samples`.
- `usage/` per shard and `report.json`: pages, samples, rows, mark crops, and for
  each source class how evenly its items were used, plus the run's `tuning` and
  `mark_crop`.

**Class tuning** (`class_tuning.py`): `--tuning tuning.json` overrides how often
each class appears — the cell weights (textblock, flowchart, geometry, table,
drawing), the prose/formula leans, enclosure and line-mark probabilities, and the
connector-arrow count weights — in every worker before its first page, e.g.
`{"flowchart_weight": 30, "underline_prob": 0.3}`. Without it a run is unchanged.
The synthesis tracker sets these, and the mark crop, for each run.

Manifests are written per shard under `shards/` as a run goes and joined into
`manifest.jsonl` when it ends. A val page's samples, page files and mark crops
are all one page.

Train and evaluate with them:
```bash
python train.py --manifest dataset/train/samples/manifest.jsonl \
    --val-manifest dataset/val/samples/manifest.jsonl --page-val dataset/val/pages
python seg_eval.py --checkpoint runs/seg/best --pages dataset/test/pages
```

**Reuse cap.** Content is reused across pages within a split, but no item may
take more than `--cap-share` (default 1%) of its class's uses — IAM-OnDB lines,
MathWriting expressions, FC charts, QuickDraw drawings; a pool too small for
that gets 2 / its size. FC charts are capped at 2% in train and 10% in val and test
(`CLASS_CAP_SHARE`, `EVAL_CLASS_CAP_SHARE`): only some of a split's charts fit a given
cell legibly, and a cap below one over that count left a split with almost no
flowcharts (0.03 a val page at the default). Capped items are held off the next page. Uses inside a
single-line textblock that `labels.is_single_line` fails to recognise are not
counted: those are the cases the classifier passes on in deployment, and they
stay at the rate the generator makes them. Counts are kept per worker, so with
few pages per worker the report's shares run above the cap (one use of an item
is always allowed); they settle as a run grows.

**Resuming.** Pages are written in shards; a shard whose usage file exists is
done and is skipped on a rerun, and each page's seed depends only on `--seed`,
the split and the page number.

## How a page is built

1. The page is **square**. It is laid out at 1000 x 1000 (`layout.PAGE_W`,
   `PAGE_H`), the size every generator's measurements are tuned in, and scaled
   to **224 x 224** (`layout.OUTPUT_SIZE`) once it is finished — ink, boxes,
   and targets together, so the labels describe the 224 page.
   `layout.build_grid` tiles the writing area with cells **grown from
   corners**: the first at the top-left, each with a width and height drawn
   from `CELL_SIZE` (15-39% of the writing area) and cut off at the page edge,
   and each seeding two more at its top-right and bottom-left. Seeds are filled
   top to bottom, left to right; one landing on a cell slides past it, and a
   cell stops short of a taller neighbour to its right. **No cell is under
   15%**: when the one at the end of a row or column would be cut below that,
   it is dropped and its space shared equally among the cells before it. A line
   across the page crosses **3.5 cells** on average (rows and columns alike),
   so a page carries 9-19 blocks, tiling the writing area instead of leaving
   most of it blank.
2. `layout.page_text_scale` picks **one handwriting size for the whole page**.
   Nothing is ever scaled to fit its cell — the content adapts instead: text
   takes as many lines as the cell's height allows and is cut at its right edge.
   Scaling to fit is what previously put 8px and 40px writing on one page.
3. `page_generator._choose_kind` decides what goes in each cell from its **size
   and shape**. Each kind's weight is a function of the cell's area (its share
   of the writing area) and aspect (width / height):

   | kind | weight | leans to |
   |---|---|---|
   | flowchart | `FLOWCHART_WEIGHT` × area; 0 below `MIN_FLOWCHART_AREA` (6%) | bigger cells, gently |
   | geometry | `GEOMETRY_WEIGHT` × area × fit(aspect, 1); 0 below `MIN_GEOMETRY_AREA` | big, squarish cells |
   | drawing | `DRAWING_WEIGHT` × (1 − area) × fit(aspect, 1) | small, squarish cells |
   | table | `TABLE_WEIGHT` × fit(area, 10%) × fit(aspect, 1.6) | mid-sized cells a little wider than tall |
   | textblock | `TEXTBLOCK_WEIGHT` | anywhere |

   `fit(x, best)` is 1 at `best` and falls away on both sides in log terms, so a
   table is rare in a thumbnail, in a cell spanning most of the page, and in a
   sliver. Figures, sketches and tables keep their own proportions when fitted
   into a cell, which is why they care about its shape. Over 3000 simulated
   pages (12.8 cells each):

   | cell's share of page | textblock | drawing | geometry | flowchart | table |
   |---|---|---|---|---|---|
   | 0-6% (61% of cells) | 84% | 10% | 5% | 0% | 2% |
   | 6-8% (22% of cells) | 67% | 8% | 13% | 8% | 3% |
   | 8-10% (11% of cells) | 62% | 8% | 17% | 9% | 4% |
   | 10%+ (6% of cells) | 57% | 8% | 20% | 11% | 4% |
   | all cells | 76% | 9% | 9% | 3.4% | 2.5% (0.32 a page) |

   A flowchart is refused below `MIN_FLOWCHART_AREA`, and its odds rise only
   gently above it, so flowcharts are spread over the mid-sized cells instead of
   piled into the few biggest (at a 9% floor and 4× the weight, the same 3% of
   blocks were 30-36% of cells over 10% and none under 8%). Below about 13% of
   the writing area no chart in the set meets `AREA_PER_SYMBOL`, so a smaller
   cell gets one of the roomiest few (`make_diagram`). That density is intended:
   a page stands in for a downsampled capture of a much larger screen, so small
   content is not meant to be legible.
4. A **textblock** is rows of prose and formulas with one of the two the
   majority (`bg.fill_textblock_cell`): mostly formulas with the odd line of
   prose (a worked problem), or mostly prose with the odd expression inline
   (notes). `MINORITY_ROW_PROB` (15%) of rows are the other kind. Paragraphs,
   bulleted lists (`BULLET_PROB`, 22%) and maths sections are all this one
   kind. A block leans to prose more the wider it is, since a prose line is cut
   at the cell's edge and a formula is scaled to fit: from 10% at 8 x-heights
   wide to 90% at 24 (`TEXT_HEAVY_WIDTH_XH`, `TEXT_HEAVY_PROB`). On this layout
   27% of textblocks lean to prose. A **table** is its own kind, not a
   textblock.
5. Prose lines are **chosen to fit the cell**, never re-wrapped and rarely cut.
   `LineDealer.take_best` auditions the next several undealt lines and keeps
   the one whose width lands in `LINE_FIT_TARGET` (80-100% of the cell). This
   both stops mid-sentence truncation and evens out the right-hand edge.

   A line that would reach the page **without a transcription** ranks behind
   every other candidate (`ROW_UNTRANSCRIBED_COST`): one with no label, or one
   that has to be cut and whose words carry no labels of their own, since a cut
   keeps the text only word by word. A table cell, narrow enough that nearly
   every line is cut, is written instead from a run of individually labelled
   words taken out of the writer's pool without dealing it (`_labelled_run`).
   Over 30 pages that took textlines with no transcription from 30 of 429 (table
   cells 28%, split enclosure pieces 5 of 6) to 0 of 373. A prose row in a
   textblock falls back the same way (`_pick_row`): once `ROW_LABEL_ATTEMPTS`
   lines have all come out cut with no word labels, it is written as a labelled
   run, and left out if even that fails. Settling for the first cut line instead
   left 134 of 905 prose rows (15%) untranscribed over 60 pages — in narrow
   cells, formula-heavy and prose-heavy blocks alike.

   **No text is written twice on a page.** The page's `LineDealer` keeps every
   transcription already written — by a prose row, a table cell or a flowchart
   caption — and each of those skips text that is already there (`used`,
   `is_used`, `mark`). It keeps the **source lines** too (`used_items`,
   `item_used`, `mark_item`, keyed by the loader's `meta["item"]`): a run of
   words a table cell took out of a line, or a row cut from it, has different
   text from the line itself, and without this one well-labelled line came out
   as row after row once a page had dealt through its writer's lines — "It
   would", "would be", "It would" … 13 times on one validation page. A row with
   no unused line left is now left out rather than repeated; over 10 validation
   pages that is 0 repeated source lines in 333 uses. Dealing prose without replacement was not enough on its
   own. A page's pool itself holds no repeated text (0 duplicate transcriptions
   across 40 pages' pools), but captions and table cells were picked from it
   without being dealt, so a line could be written as a caption and come round
   again as a prose row, and the deck reshuffles once a page has used it up. Captions were the worst: `shape_content` took the best-shaped
   of 14 picks from one writer's few dozen lines, and the best shape kept winning
   — 23% of prose captions repeated text already on their page, against 5% when
   the same 14 were picked at random, and loading every IAM line only moved it
   to 22.8%. Candidates are now drawn without replacement from text not yet on
   the page, and the best-shaped of those is kept. Formula rows still repeat
   about 6% within a page, which is the MathWriting excerpt's size (200 usable
   expressions against ~20 per page), not the sampling; the full dataset fixes it.
   Geometry labels read EMNIST at 200 glyphs per character (`POOL_LIMITS` in
   `make_pdf.py`); at 24, two labels sharing a character drew the identical glyph
   1 time in 10, and at 200 it was 0 of 33. Measured over 40 pages with both
   changes in place: 0 of 104 flowchart captions, 0 of 375 prose rows and 0 of 74
   table cells repeated text already on their page (before: 25%, 1.8%, 5.4%), 0
   of 25 same-character label pairs shared a glyph, and formula rows repeated
   3.8% (26 of 684) — the MathWriting excerpt's size. Captions on arrows are
   the chart's own ("yes", "no") and left as they are, so 1 of 33 repeated.

6. **Connector arrows** are drawn last, linking a text line in one block to one
   in another (`_add_connector_arrows`). Only the two endpoints are fixed: the
   route between them is *sampled*, not planned. A detour size is drawn from a
   distribution with most of its mass near zero and a long tail, the path is
   smoothed through 1-3 random waypoints, and any route crossing a text line,
   sketch or flowchart is thrown away and re-drawn — which is what makes arrows
   go around paragraphs instead of through them. Detours widen as attempts fail,
   so direct routes stay the likeliest outcome. Pairs are weighted by
   `exp(-distance / scale)`, so a link between neighbours is far more likely
   than one spanning the page.

   An arrow leaves and lands **square to the edge it is attached to** — vertical
   off a line's top or bottom, horizontal off its left or right — so it never
   sets off across the text it points at. A side with something parked against
   it is skipped, which is what stops an arrow diving into the next line of the
   same paragraph. On a text line the arrow lands within ±20% of the middle of
   that side, along it (`ARROW_ANCHOR_SPREAD`), and stands a random 3-9 units
   off it (`ARROW_ANCHOR_GAP`). A **drawing, flowchart or geometry figure** can
   be either end too, and the arrow **points straight into it**: it anchors on
   the object's own ink, perpendicular to the stroke there, travelling against
   the stroke's outward normal (away from the object's centre). Only points
   whose outward normal faces the arrow's other end (`ARROW_STROKE_MIN_FACING`)
   and where the object's outside is open — none of its ink just beyond the
   stroke (`ARROW_EXPOSED_REACH`) — are used, and of those the nearest to the
   other end. Taking the nearest stroke with its normal flipped toward the
   other end instead could land on ink inside the object, or on a stroke
   running toward the arrow, so it skimmed along rather than pointing in. The
   direction is perpendicular give or take up to 10°
   (`ARROW_STROKE_NORMAL_JITTER`), and the anchor stands a random 3-9 units off
   the ink.

   The **ink is mined**; only the route is procedural. The route is eased first
   (`_smooth_route`), which rounds the hook where the square lead-in met the
   first detour. The shaft is then **several mined lines end to end**
   (`_chain_ink`), 18-40 units each (`ARROW_PIECE_LENGTH`). A piece is a mined
   line cleaned for joining (`_piece_shape`): 8% trimmed off each end, where a
   pen hooks as it sets down and lifts; its waver re-measured from the trimmed
   ends, so it starts and finishes on its chord; and even steps along it, since
   a third of mined lines step backwards somewhere. Each piece starts where the
   last stopped, setting off along the direction that piece was travelling at
   its end, turned toward the point a piece-length ahead on the route by at
   most 6° (`ARROW_MAX_JOINT`); where the route turns harder, pieces get shorter
   rather than the line kinking. So the curvature is the joints between strokes
   a hand really drew, the route only steers, and the waver is a piece's own.
   The last 60+ units (`ARROW_FINAL_STRETCH`) are one mined line along a curve
   from the chain's heading into the arrival direction, so the head is met
   square instead of by a straight piece aimed at it from wherever the chain
   had got to. A loop is one **mined oval** drawn once round (`_loop_ink`, via
   `geometry_diagrams.arc`), with chains before and after, since a chain turning
   6° a join cannot follow a curl that tight; its ends are pinned onto the route
   by an offset blended along the stroke, which leaves the oval's own shape. A chain that draws more than `ARROW_CHAIN_SLACK` (1.8)
   times its route's length has lost the route and is circling — one drew 618
   units between anchors 117 apart — so that attempt is abandoned. The route
   itself varies little (`ARROW_DETOUR_SIGMA` 0.08, `ARROW_DETOUR_TAIL` 0.10,
   mostly one waypoint, `ARROW_WEAVE_PROB` 0.10). The head is mined lines
   too, drawn one of three ways (`ARROW_HEAD_STYLES`): two barbs, one V stroke,
   or a closed triangle of three separate mined lines (in along one side, out
   along the other, back across the base). There is no filled head — scribbled
   in, it never read as drawn. The V's corner is pinned to the tip. The last
   head-length of every shaft runs straight into the target, as its own mined
   line, and the head is aimed along it: a shaft that curved all the way to the
   tip did its tightest turning right where the head sits, and cut out through
   the side of a closed one. **Loops** are a
   writer's habit, and an uncommon one, so they are decided per page: about one
   page in twenty (`ARROW_LOOP_PAGE_PROB` 0.05) has its connectors drawn with
   loops, each getting one with `ARROW_LOOP_PROB_ON_PAGE` (0.6), and every other
   page has none. A
   loop sits partway along the shaft (`_add_loop`): it leaves the
   route tangent to it, goes once round while drifting forward, and rejoins it;
   it is added after the doubling-back tests (which a loop fails by definition)
   and before the obstacle tests. A mined line's offset is zero at both ends, so
   an arrow still starts and stops exactly on its anchors. No bank, no connector
   arrows.

   A route is rejected if it **doubles back** (`_turns_back`) or if either end
   runs against the arrow's own overall direction (`_reverses`) — that second
   test is what removes arrows that travelled one way and hooked round to arrive
   from the other side, so the head read as pointing backwards (`-----<-`). It
   has to be tested directly on the first and last legs: the reversal is spread
   smoothly over a long curve, so a corner-angle test does not see it.

   Measured with the chained shaft, over 40 pages (27 arrows, every fifth page
   forced loopy): departure deviates **3.7° median** from the intended normal
   over the first 12 units (p90 8.8°) and arrival 8.2° (p90 25.3°) — since
   brought to 1.7° (p90 3.8°, over 25 pages and 37 arrows) by the straight last
   head-length described above; **0
   crossing an obstacle**, every shaft exactly on its two anchors; detour ratio
   (path length / straight-line distance, loops left out) median 1.11, p90
   1.34, max 1.76; the sharpest turn over any 3 units of a shaft is 19° median
   and 30° at most. The arrival tail was the weakest part — the final curve bent
   from wherever the chain arrived into the target's direction, and on some
   arrows that bend was still inside the last 12 units — until the last
   head-length was made straight; with it, no closed head has shaft outside its
   triangle (the worst had 40% of its last head-length outside). Before pieces were
   trimmed and the final stretch curved, the same measure found 32° median turns
   (spikes at the joins) and arrival 24° median. Separately, with loops decided
   per page and the closed / scribbled heads, a 100-page run had 46 pages with
   connectors, 4 of them loopy (3 of their 4 arrows looped, and no arrow on any
   other page), and heads split 23 barbs, 28 V, 17 closed, 11 filled. The
   procedural shaft this whole scheme replaced measured 3.1° median departure,
   p90 7.3°, detour median 1.27, max 2.04.

7. **Scribbles are not generated**, and nothing on a page carries `unknown`.
   There used to be a family of procedural scribble generators — crossings-out,
   loops, spirals, wandering squiggles, hatching — filling a share of the drawing
   cells as level-0 `unknown`, on the argument that a real page has marks
   belonging to none of the taxonomy's classes and the model needs somewhere to
   put them.

   That argument does not survive looking at the output. Those marks were an
   invention, not something any source dataset carries: drawn by formula, sitting
   in a cell of their own with space around them rather than on top of the thing
   they were supposedly crossing out. They gave the page `unknown` boxes in a
   shape and a place no real page produces, and teaching a class off invented ink
   is worse than leaving the class empty. `drawing` is now always a real
   QuickDraw sketch.

   `unknown` stays in the taxonomy (`labels.LEVEL0`) so real unnameable ink has a
   name waiting for it — but the way to fill it is real marks, or scribbles drawn
   **on** real text, which is a different generator and a different label
   relationship.

8. **Geometry figures** — a triangle with an altitude, a circle with a chord,
   two shapes glued along a side — are built by `geometry_diagrams.py` out of
   *mined* ink: real pen strokes that `shape_mining.py` found in the pools,
   re-warped to the size the construction calls for. So a synthesised
   right-angle box carries the tremor of the hand that drew the line it came
   from. Nothing is a mined outline replayed whole: a square, a trapezium, a
   regular hexagon and a dented heptagon are all n mined lines laid corner to
   corner — joined the way a careful hand joins them: half close cleanly, the
   rest with a slight run-on past a corner now and then and hardly ever a gap (`FIGURE_JOIN`,
   `loose_sides`; only the ink moves, the construction keeps the true corners) —
   drawn from a distribution where sides get rarer as they multiply and
   the symmetric forms dominate within a given number of them (most convex, a
   minority dented). Equal sides are sometimes ticked (`mark_equal_sides`): one,
   two or three short strokes of mined line ink across the middle of each side,
   a count per group of equal lengths, read off the finished construction after
   any gluing so the marks say what is true of the figure as drawn. Equal
   angles are arced the same way (`mark_equal_angles`): one, two or three
   concentric arcs per group, also decided over the whole assembled figure, so
   an arc count is one size across every glued shape. Both compare the exact
   construction, to float noise only, never the ink: two marks alike are two
   sizes that measure the same. They are class `geometry` at level 2, boxed as a whole, as
   a `flowchart` is: a flowchart is a graph of captioned symbols joined by arrows
   — each shape a level-0 enclosure and each caption a line of the chart — while
   a figure is one construction of segments and circular arcs, each labelled
   as a part of it (see *Arcs and segments*). Every curve on a figure is a
   circle or an arc of one: there are no ellipses.

   A figure is chosen **per cell**, like every other kind (`_choose_kind`), and
   a page may carry several. Its weight grows with the cell's area and falls
   off as the cell gets further from square, since a figure keeps its own
   proportions when it is fitted in; below `MIN_GEOMETRY_AREA` it is refused.

   The ink is drawn from a **bank of mined primitives** cached in
   `shapes_bank.json`; build it with
   `python geometry_diagrams.py --cache shapes_bank.json`. Mining takes minutes,
   so if the cache is missing the generator simply leaves geometry off the page
   rather than mining mid-run. For a dataset split, mine the bank from **that
   split's pools alone** — `build_bank(pools=..., split="train", cache=...)` —
   since its ink draws every underline, box, table rule, arrow and figure side
   on the page. The cache then records the split and each primitive's source,
   and is refused for any other split. The shipped `shapes_bank.json` was mined
   from every dataset and records neither, so it is for the demo only. Each figure carries the exact construction it was
   drawn from in `meta["exact"]`, and **that** is what the ground-truth page
   renders — not a smoothed trace of the ink, but the geometry the hand was
   aiming at, in which an inscribed circle really is tangent to all three sides.

9. **Marks on writing** come in two level-1 classes. A box or oval round text is
   an `enclosure` — the same class as every flowchart shape — and an underline or strikethrough is a `line_mark`, one class for
   both; either way what was drawn is kept as the label. The marks that go on
   text blocks **never touch the block's rows**: the rows and their words stay
   exactly as they were, and the mark goes into the block beside them. Each mark
   records the row it marks (`meta["attends_row"]`, -1 for the whole block),
   which is only used to order it among the block's level-1 targets.

   **Underlines and strikethroughs** (`make_line_mark`, `_mark_rows`) go on once
   a block's rows are stacked. A text line is underlined (`UNDERLINE_PROB` 0.08)
   or, rarely, struck through (`STRIKETHROUGH_PROB` 0.01). A formula row is never
   marked — a line under maths is too easily read as a fraction bar — but still
   counts as a neighbour a text line's mark must stay out of. A mark covers a
   stretch of its row grown from a random word until it passes under (or
   through) at least `UNDERLINE_MIN_STROKES` (10) of the row's strokes and
   `UNDERLINE_MIN_WIDTH` (4) x-heights — IAM's word boundaries are geometric, so
   a "word" is often a couple of letters or a full stop, and a mark under one
   reads as a stray tick under a single character. It sits beside the rows in
   its textblock. A few underlines are drawn twice, one stroke just
   under the other (`UNDERLINE_DOUBLE_PROB` 0.08, `UNDERLINE_DOUBLE_GAP` 0.18
   x-heights), and are still labelled `underline`: a double underline is the
   same emphasis as a single one, and `meta["double"]` only records which was
   drawn.

   Each stroke is a nearly straight mined line (`UNDERLINE_WOBBLE` 0.2 of a
   mined line's waver) following the row's own drift, a slope fitted through
   every word — a level underline under a sloping line of handwriting cut up
   into its words at one end. That slope is capped at `UNDERLINE_MAX_SLOPE`
   (0.10). The cap was 0.25, and over a long line that ran an underline
   diagonally through the writing: over 60 pages 2 of 46 underlines, at slopes
   of 0.17 and 0.14, reached the middle of the next line. An underline sits
   `UNDERLINE_DROP` (0.40) x-heights below the lowest foot of the words it
   spans, so it cannot rise into any of them; a strikethrough runs through
   their middle. What is drawn is then checked (`_mark_fits`): an underline may
   not rise into its own writing, a strikethrough may not leave it, and neither
   may come within the middle half of the row above or below
   (`MARK_NEIGHBOUR_REACH` 0.25 — clipping the top of the next line is what a
   hand does now and then, reaching its middle strikes it through). A mark that
   fails is redrawn level, then single, then closer under its row, and dropped
   if none of those fit.

   **Boxes and ovals** go round part of a text block (`maybe_enclose`,
   `ENCLOSURE_PROB` 0.08 per text block): the whole block, one line of prose, a
   run of words (two or more, never the whole line), or a single word, each
   likelier than the one before (`ENCLOSURE_TARGET_WEIGHTS` 0.15 / 0.20 / 0.25 /
   0.40). Words are not split out of their line: a box round the middle of a line
   changes what those words mean, not which line they belong to. The shape is
   drawn the way a geometry figure is, from the mined bank: four mined lines for
   a rectangle — usually joined more loosely than a figure's sides, so its
   corners often overshoot, open up or miss, though 30% of boxes close cleanly
   (`ENCLOSURE_JOIN`, which a table's border uses too) — and a mined oval for an
   oval, so no boxes or ovals without the bank. An
   oval is sized to the smallest ellipse through the padded target box's
   corners, then tested against the target's own ink and redrawn larger if any
   of it falls outside — a mined oval dips inside its fitted ellipse, so the fit
   alone does not guarantee it. A rectangle's sides are held clear of the
   target's box the same way. The shape must also stay on the page and off every
   other block — an oval round a whole block reaches sqrt 2 past it, and half of
   them used to run into the block next door — so these go on in a pass after
   every block is placed; a shape that would collide is swapped for the other
   shape, and a target with no room for either is left alone. The same goes for
   a shape that would cut through the middle of a row of its own block that it
   does not mark (the test underlines get, `_mark_fits`): checked only against
   other blocks, 3 of 11 ovals and 2 of 25 boxes round text over 60 pages ran
   through the line above or below their target. A level a block
   cannot offer — words, in a maths section — is not swapped for a line either:
   doing so made lines the commonest box, the reverse of the weights. Only a run
   of words falls back, to a single word. The shape goes into its block, whose box
   grows to take it in.

`_clamp_to_cell` is the backstop: whatever a generator produces is kept inside
its cell, so nothing collides with a neighbour or runs off the page.

## `formula`: maths is its own level-1 class

A row of maths is labelled **`formula`**, beside `textline` rather than under
it. Written by hand, `x = i+1` is a formula and `check the next value` is prose,
and there is no reading of the strokes alone that separates them reliably — so
the generator commits to one and the label is known rather than guessed. Every
MathWriting expression it places carries this class, wherever it lands: the rows
of a maths section, an equation dropped mid-paragraph, the contents of a
flowchart shape, an angle written in radians. One class, one meaning everywhere.

### What goes inside a flowchart shape

The source chart's own node captions are **thrown away and rewritten** — text 70% of the
time, a formula 30%. Two reasons:

- **Labelling.** FC captions are often maths written in text (`$M=M+1$`,
  `x > y`), and nothing in the ink says which they are. Left alone, the same
  marks would be `textline` in one chart and `formula` in another.
- **Range.** Real flowcharts hold far more than FC's short phrases. The full
  text and maths pools cover ground the 672 charts do not.

Only shapes that *already had* a caption get one, so each chart keeps its own
balance of labelled and blank nodes rather than having captions invented for it.
A caption sitting on an **arrow** is left exactly as the source chart wrote it
(`Yes`, `Then`, `0`; the few written in LaTeX-in-dollars or with no transcription are left out).

Content is sized to the room inside the shape, and the room is measured against
the shape's **own outline** rather than worked out from its class
(`_interior_box`): a box of the shape's proportions is shrunk step by step until
no outline point falls inside it. That lands on about half-size for a diamond
and most of the box for a rectangle without being told either, and it follows
the slant of whichever parallelogram that chart happens to have drawn. The
candidate whose aspect ratio best matches the space is the one written in, since
`fit_ink_to_box` scales uniformly — a wide line dropped into a tall box is not
clipped, it is shrunk until it is unreadable. Containment is then by
construction, not by hope: uniform scale plus centring cannot leave the box.

## Geometry figures: labels and vertices

A page carries **no relations** between objects — only its per-level boxes. A
flowchart caption on an arrow is a line of its chart, like any other caption.

### Labels are lines of the figure

A textbook figure is half drawing and half writing: corners are named so the
prose can refer to them, a side carries its length, an angle carries its size.
`geometry_diagrams.annotate_figure` writes those on, one character at a time out
of the EMNIST pool, and each label is **a `textline` held directly by the
`geometry`** (its characters one `word`), the way a table holds its cells' lines
(`_label_line`). It counts toward the figure's box (`labels.label_bbox`), so a
label written just outside a corner stretches it. Nothing ties a label to the
corner, side or angle it sits beside.

An angle in **radians** is a formula, not a number: `\frac{\pi}{3}` is a
fraction with a Greek letter in it. Those are not assembled from character ink —
they are picked out of MathWriting by what their LaTeX says
(`radian_expressions`, matching a multiple of pi, a pi fraction, or pi over a
number), only ever one whose value is within half a degree of the angle, and
that label is a `formula` instead of a textline.
**This needs the full MathWriting download**: the shipped 200-expression excerpt
contains none, so with it every angle comes out in degrees.

Every number is **true to the construction**, not to the ink. An angle is its
size to the nearest degree. Lengths share one scale per figure
(`_length_scale`), so a side written 12 really is twice one written 6, and a
side is only given a number within 3% of its length on that scale. Two labels
that say the same thing are on two things of the same size: a number that would
repeat one already written on a different size is left off, and an unknown
letter is never used twice. An angle is not labelled where another line leaves
its corner inside it, since the number could not say which angle it measures.

Corners are named **all or none** — a triangle with one lettered corner and two
blank ones is not a figure anyone draws. Placement is collision-aware
(`_place_label`): each label is tried at increasing distances from its anchor
and dropped if none is clear of the figure's ink and of the labels already
written. Dropping is often the right answer — where two glued shapes bring their
corners together there is genuinely nowhere to write, and a number laid across
the lines is worse than no number.

### Vertices

**`vertex` is level 1**, and *every* vertex a figure has is emitted, named or
not (`figure_vertices`): each point where a drawn line or arc ends — on its own,
at another's end, or partway along another, like an altitude's foot or a chord's
end on its circle. A point where two lines only cross is not one, and neither
are right-angle boxes, angle arcs and the ticks across equal sides, which are
marks on the figure rather than lines of it. Its box is the point itself, and it has no ink of its own.
Vertices are found on the exact construction, not the ink, so two ends that are
one point in the construction are one vertex however the strokes wobble.

A vertex's object in the label JSON adds what its box cannot say:

| Field | What it is |
|---|---|
| `position` | the point, `[x, y]` |
| `id` | unique on the page, for `connected` |
| `angles` | degrees, one per gap between the lines meeting here that lies **inside** the figure: one at a polygon's corner (over 180 at a dent), more where more lines meet — an altitude's foot, an inscribed polygon's corner on its circle |
| `angle_mark` | `none`, `right` (the little square) or `arc` |
| `angle_mark_count` | how many concentric arcs (1–3 tell groups of equal angles apart), or 1 for a right-angle box |
| `connected` | ids of the vertices one drawn line or arc runs to from this one |

"Inside" is an even-odd test, a little way into each gap, against the figure's
outline: its sides, a sector's radii, a semicircle's diameter, a circle. A
construction drawn on the figure (a chord, an altitude, an inscribed polygon, an
incircle) joins vertices but is not outline, and neither is a side two glued
shapes share — `concatenate` flags it as it glues, since it is inside the joined
figure. On the ground-truth page each vertex is a dot with its angles beside it.

### Arcs and segments

**`arc` and `segment` are level 1** too, held by the figure beside its
vertices (`figure_parts`). Every circle or arc of one the figure draws — a
circle, a semicircle's or sector's arc, an incircle or circumcircle — is an
`arc`; every straight line — a side, a radius, a diameter, a chord, an
altitude and its extension, an inscribed polygon's side — is a `segment`. The
marks are neither: a right-angle box and an angle's arcs are what a vertex's
`angle_mark` says, and the ticks across equal sides are not labelled.

| Class | Target | Page JSON |
|---|---|---|
| `arc` | `x1 y1 x2 y2 x3 y3 arc`: start, middle, end, going **counterclockwise** as the image is seen (y down). A full circle starts and ends at its 0° point (the right-hand end of its horizontal diameter), its 180° point between | `points` |
| `segment` | `x1 y1 x2 y2 segment`: the upper end (then the left) first | `points` |

Like a vertex, each has no ink of its own: its points are
single-point strokes (so they move with the figure when the page places it),
taken from the exact construction.

### Arrows that share a pair of nodes
72 of FC's node pairs (28 in OHFD, the earlier source) are joined in **both directions** — a loop back up to the
decision it came from. Drawn straight, both land on exactly the same line between
the same two outlines, and what should read as "A goes to B, and B comes back to
A" instead reads as a single arrow with a head at each end. `_arrow_lanes` bows
each of them to a different side (`ARROW_BOW`, matplotlib's `arc3` rad). Because
`rad` is measured from the *direction of travel*, arrows are grouped by their
unordered pair and the sign is flipped for any running against the group's first
one — which handles both a back-and-forth pair and two arrows in the same
direction with the same rule. A pair with only one arrow stays straight.

The page's own **connector arrows** get the same guarantee more cheaply: pairs of
text lines are keyed by an unordered `frozenset`, so two lines are linked at most
once in either direction. An ordered key let the same two lines get both an A→B
and a B→A arrow, and since both were routed between the same two anchors they
came out on top of each other.

## Known simplifications / things to tune
- `unknown` is a level-0 class for ink that means nothing, and **nothing
  currently generates it** (see step 7). It is a declared class with no examples
  behind it, which is a real gap if you want the model to have anywhere to put
  stray marks — filling it needs real unnameable ink, not more procedural
  squiggles.
- A geometry label's *value* is arbitrary — the `12` against a side is not the
  side's actual length, and the `60` written in an angle is not its actual size.
  The figures are drawn freehand from mined ink, so no consistent set of numbers
  exists to write. A vertex's `angles`, by contrast, are measured off the
  construction and are true. Don't train a model to read lengths off the labels.
- `COORD_ORDER` in `labels.py` — verify against PaliGemma's actual detect-token
  coordinate order (native format is commonly y0 x0 y1 x1, not x0 y0 x1 y1) before training.
- **A maths section is labelled `textblock`, its expression rows `formula`.**
  Expressions get no word children, so a maths section contributes level-2 and
  level-1 boxes but nothing at level 0.
- Lines are chosen to fit and only cut as a last resort, never re-wrapped word
  by word. Re-wrapping was tried and is worse: IAM's word boundaries are
  geometric guesses that often split a word into two or three pieces (a "word"
  averages 1.9 x-heights), so laying them out on an even gap turned sentences
  into letter soup — "Whitewallfarceisitwhen". The writer's own spacing is the
  real thing, so the only edit made to it is where the line stops.
- Arrows link two *different* blocks, never two lines of one paragraph. Nearby
  pairs dominate the distance weighting, and two lines inside one paragraph are
  the nearest pairs there are — but they are also hemmed in by their own
  neighbours, so nearly every route between them was rejected. About 27% of
  attempted routes still fail and 16 pages in 61 have fewer than two
  text-bearing blocks to link, so roughly half of pages end up with no arrow.
- **One- and two-stroke items are dropped by the loaders** (`loaders.MIN_STROKES`,
  default 3), because a single-character "line" or "expression" placed on a page
  reads as a stray mark. MathWriting's `symbols` split was the main source (53%
  are one stroke, median 1) and is no longer in `load_mathwriting`'s default
  splits. The filter costs QuickDraw the most — ~32% of its sketches are two
  strokes or fewer and some are legitimate drawings — so pass a lower
  `min_strokes` to that loader if the sketch pool gets thin. Flowchart labels
  ("YES", "end") are exempt: they are children of a diagram, not standalone.
- **DeepWriting is off for prose by default.** Rendering its lines straight from
  the loader shows the problem is the data, not the pipeline: it is loose
  cursive, and its samples are fixed-length *chunks* that begin and end mid-word
  ("He smiled at his ow", "actually drippin", "k in"). It stays loaded, and
  clearing `PROSE_SOURCES` brings it back.

  Measured against IAM-OnDB, it is mostly IAM-OnDB: 35,279 of its 35,282 source
  texts are IAM-OnDB lines, and 186 of its 294 writers (23,907 samples) are
  IAM-OnDB's own ink, stroke for stroke; the other ~95 writers are new hands
  copying the same texts. 0.5% of samples are whole lines and 54% start or end
  mid-word (median 2 words, against IAM's 6), about 19% of decoded labels do not
  match the source text even ignoring case and punctuation, and 16% of samples
  are narrower per character than IAM's narrowest 1% (a squashed
  reconstruction). What it does carry that IAM-OnDB's parts here do not is a
  writer per sample — which is what `iam_writers.py` uses. **Never use both in
  one dataset split scheme without joining them by writer and text**: the same
  ink would otherwise sit on both sides of a split.
- A page uses up to `MAX_WRITERS_PER_PAGE` (6) hands. The cap was set when a
  "writer" was one IAM form, ~7 lines against a dense page's ~40, and a tighter
  cap just repeated sentences (measured then: 2 hands repeated 3.7% of lines, 4
  hands 0.8%). With `data/iamondb_writers.json` a writer is the real one — 200
  writers, a median of 55 lines — and only the 139 forms with no known writer
  still stand in for theirs.
- Every **table has all its column dividers** — the vertical rules are what make
  the columns, and without them the writing is spaced out in a block that could
  not be told from a textblock. Only two things are chosen: the horizontal
  dividers, every row (`full`, 79%) or only the header row's (`header_row`, 21%)
  (`TABLE_RULE_STYLES`), and, separately, whether there is an outer border
  (`TABLE_BORDER_PROB` 0.75).
- Table/list *layout* is 100% procedural since no source dataset has them (a
  table's rules are mined ink, but where they go is not) —
  the hand-drawn jitter in `block_generators.py` is a rough approximation; consider
  a small set of hand-collected real examples to fine-tune the "hand-drawn-ness" if
  synthetic tables/lists look too clean vs. real IAMonDo.
- Procedural **ovals are gone**: an ellipse drawn by formula read as an obvious
  artifact next to real ink, so procedural flowchart nodes are boxes. Every
  ellipse on a page is real hand-drawn ink: FC's terminators and connection
  circles, and the mined ovals drawn round words. A geometry figure draws only
  circles and arcs of them, out of mined oval ink.
- There are **no per-shape classes**. A flowchart's rectangle, stadium, diamond,
  parallelogram and circle, and a box or oval drawn round a block, line or words,
  are all `enclosure` (a table's outer border is not labelled) (`FC_CLASS_MAP` in `loaders.py`); where an enclosure sits says
  what kind it is. An underline (single or double) and a strikethrough are both
  `line_mark`. What was drawn survives as `meta["node_type"]` or the label
  (`rectangle`, `oval`, `underline`, `strikethrough`), for rendering only. A
  geometry figure's circles and straight lines are its `arc` and `segment`
  objects.
- **Enclosures and line marks are level 1**, held by the textblock or
  flowchart whose writing they mark, and hold nothing. What they mark keeps its
  own place — a line is never split round the words it marks. A flowchart's shapes
  and caption lines sit side by side in the chart (`_unnest_flowchart`), a shape
  sharing a key with its caption's lines; an underline, strikethrough or box round
  text sits beside the rows of its textblock. A table's border is not labelled. The holder's box takes in the mark's ink. Arrows are the only objects
  that stand on the page on their own.
- **Every line is inside a textblock, a table, a geometry figure or a
  flowchart.** A table cell's line, a label on a geometry figure and a flowchart
  caption (in a shape or on an arrow) sit in the table, figure or chart directly,
  with no textblock between. So line detection only ever runs inside those four,
  and not at all for a
  block that is a single line: **`labels.is_single_line(strokes)`** sorts the
  block's strokes by height and calls it one line when the stroke at the **89th
  percentile** is at least **46.3%** of the height of the box round them
  (`SINGLE_LINE_PERCENTILE`, `SINGLE_LINE_MIN_RATIO`). Pass it the block's own
  writing — not an enclosure, line mark or arrow drawn over it, whose strokes can be as tall as
  the whole block.

  The rule is one-sided: calling a multi-line block one line skips its line
  detection, so there must be no false positives, and a missed single line only
  costs a detection pass. The first rule tried — median at 70% — had none but
  caught 15% of single lines. No per-stroke rule separates the two at all: a
  hard-margin SVM on every stroke's (percentile, share of block height), 62,090
  strokes, has no solution, because every single line has small strokes (dots,
  commas) exactly like a multi-line block's — 60% of each class lies inside the
  other's convex hull. Per block, for every percentile the threshold was put at
  1.5x the highest value any multi-line block reached, and the percentile
  catching the most single lines kept: over 150 pages that is the 89th. On 150
  unseen pages it made **0 false positives** of 352 multi-line blocks and caught
  **77.9%** of 1,054 single lines. Less headroom did not survive unseen pages
  (at 1x, 5 false positives; at 1.1x, 2); 1.25x held (0, 86.3%) but with less
  margin for data not yet seen. Averaging each percentile over an 8-percentile
  window was no better (74.0% at 1.5x).
- `layout.py` scatters blocks at random non-overlapping positions rather than
  stacking them in rows, so content lands all over the page. Blocks that don't
  fit are dropped, never run off an edge. Placement is largest-first with
  rejection sampling (`PLACE_ATTEMPTS`, then `SHRINK_STEPS` shrinks) — good
  enough for these densities, but it is not a real packer, so very crowded pages
  will drop blocks rather than find a clever arrangement.
- **One handwriting size, and one hand, per page.** Three things combine here:
  `block_generators._core_band` sizes a line by its **x-height** (the arc-length
  -weighted middle of its ink), not its bbox — a bbox conflates x-height with
  ascender/descender reach, so "we ran on" rendered at triple the size of "of a
  proposed House of chiefs."; `layout._page_text_scale` then picks one scale for
  every text block, derived from the content so a typical block spans
  `TEXT_WIDTH_FRAC` of the page, and a text block is never rescaled to fit; and
  `page_generator._single_writer_pool` draws a page's prose from one writer,
  since equal x-heights still look inconsistent when a tight upright hand sits
  beside a loose sprawling one. It allows more hands (`MAX_WRITERS_PER_PAGE`)
  because IAM-OnDB's own parts name lines only by form — about 7 lines — and
  demanding more from one form either excludes IAM entirely (leaving pages of
  DeepWriting's one-word fragments) or makes a page repeat its own sentences.
  `data/iamondb_writers.json` now supplies the real writer for 1,591 of the
  1,730 forms, so a "hand" is normally a whole writer. Lines are dealt without replacement
  (`block_generators.LineDealer`) for the same reason. Flowchart captions are sized via `DIAGRAM_CAPTION_RATIO`; the residual
  spread in them is real variation in the FC ink, not a scaling artifact.
- Page composition deviates from the paper's prior on purpose, via
  `page_generator.COUNT_SCALE` (less prose), `block_generators.MATH_LINE_PROB`
  (35% of textblock lines are expressions) and `DIAGRAM_CROWDING` (when a
  flowchart is present it gets `layout.DIAGRAM_WIDTH_FRAC` of the page and
  everything else is thinned). `LEVEL2_STATS` still holds the paper's numbers
  unmodified.
- Every one of FC's 672 flowcharts has captions, but a shape whose rewritten
  caption would not stay legible at the chart's size is left blank, so some
  flowcharts show empty shapes. That is by design, not a parsing failure.
- **No timestamps.** A `Stroke` is x/y only. The per-point times IAM,
  MathWriting and QuickDraw record are dropped on load, and nothing synthesized
  is given any; the output describes ink as drawn, not the order or pace it was
  drawn in.
- IAM-OnDB word boundaries are **geometric guesses**
  (`naive_split_line_into_words`), because the `lineStrokes` distribution has no
  word segmentation — delayed strokes (i-dots, t-crossings) can land in the
  neighboring word. DeepWriting and FC symbol groups are real annotations.
  If level-0 word boxes turn out to be the weak point, the fix is to also
  download IAM-OnDB's `original` part and read its per-word time ranges.
