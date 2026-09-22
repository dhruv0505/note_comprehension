# Recognition samples

Recognition data for the three tasks, from the same sources and splits as note_synth's segmentation pages. Each
writes one prompt row per image:

| task | source | image | prompt | target |
|---|---|---|---|---|
| `textline` | the textlines of each synthesized train and val page | up to 10 lines stacked, as the cascade reads them | `What is written in this image in English?` | the transcriptions, top to bottom, one per line (joined by newlines) |
| `formula` | the formulas of each synthesized train and val page | up to 10 expressions stacked, as the cascade reads them | `What is written in this image in LaTeX?` | the LaTeX strings, one per line |
| `drawing` | QuickDraw sketches | one sketch | `What is drawn in this image?` | its category, e.g. `cat` |

Test has no recognition samples: test recognition is only ever scored through the cascade (`training/recog_eval.py`).

## Files
- `recog_layout.py` — prompts, the row separator (a newline), how an image is drawn and laid out, how ink sections are timed. Uses only numpy and PIL.
- `recog_pages.py` — textline and formula samples straight from a synthesized page, read the way the cascade reads one. `training/recog_eval.py` and note_synth's `synthesize.py` both use it. Uses only numpy, PIL and `recog_layout`.
- `recog_pools.py` — one split's items for a task, read from note_synth's `split_pools.py` cache or from the archives.
- `recog_marks.py` — for pool samples, now and then a mark on a textline row, as the cascade's mark separation leaves it.
- `recog_synthesize.py` — samples from the pools: the parallel, sharded, resumable run for `drawing` (and, by hand, pool-based textlines and formulas).
- `test_recog.py` — tests, on mock ink only: `python -m unittest test_recog`.

## Samples from the pages (textline, formula)

With `--recognition`, note_synth's `synthesize.py` makes these for every train and val page as it builds the page
(`recog_pages.write_samples`). A sample is what the cascade would read from the page with perfect boxes:

- **Grouping.** The page's textlines are taken ten at a time in the order the page lists them (a block's lines top
  to bottom), then its formulas the same way. The page's last group holds what is left, so a page with 23 textlines
  gives samples of 10, 10 and 3 rows. Its rows fill the top slots of the ten-row layout and the rest stay white.
- **Row ink.** A row is every stroke of the page with at least half its points inside the line's ground-truth box,
  taken whole: the line's own strokes, and any strokes of the lines above and below that reach far enough in. This
  is exactly how the cascade picks a detected line's ink. A line's box is its ink's own extent.
- **Marks never happened.** Underlines, strikethroughs and enclosures are left out altogether, using the ground
  truth's labels. The cascade removes them with the mark separation network; the samples don't need it.
- **Labels.** A line whose transcription can't go into a target (below) is not a row of its own, though its strokes
  are still read in the boxes of the lines beside it.
- **Pen.** The sources record x, y and time, not pen width, so every row is drawn with the same 2 px pen in the drawn
  row, whatever the line's size; only a row shrunk to fit the image's width gets a thinner one. (The page's own pen
  scaled with each line blew small lines up into unreadable blots.) The cascade draws rows the same way.

**Times are never made up.** Every line keeps its recorded times and moves only as a whole section. A page is written
top to bottom: within a block, each line starts a pause after the line above it ends (5–30% of the longer of the two
lines' durations, seeded by the page's ink), and blocks follow one another in page order. The page JSON times each line
from its own first point, so without this the strokes of the lines above and below would be timed over the row's own.
The cascade orders a page's times the same way (`recog_pages.write_order`). A textline or formula without recorded
times is refused.

Output, under `<out>/<split>/recognition/<task>/`:
- `images/<shard>/p0000123_NN.png` — the page's NNth sample of the task.
- `manifest.jsonl` — `{"image", "prompt", "target", "task", "page", "boxes"}`, `image` relative to this folder.
- `shards/` — each shard's rows, joined into `manifest.jsonl` when the split is done.
- `report.json` — samples, rows, samples with fewer than 10 rows (`short_samples`), and rows holding another line's strokes (`rows_with_other_lines`).

Each shard's counts are also in the split's `usage/shard_*.json` under `"recognition"`.

**Splits.** A sample only ever holds the ink of its own page, so no split's samples hold another split's ink.

## The image

Everything is 768 × 768 RGB, on white. The ink carries time and distance in its colour channels, as in
[Fadeeva et al., 2024](https://arxiv.org/abs/2402.15307). For an item whose strokes are sᵢ = [(xᵢ,₀, yᵢ,₀, tᵢ,₀), …]:

```
R(i,j) = (t(i,j) − t(0,0)) / max (t(m,n) − t(0,0))
G(i,j) = |dx(i,j)| / max |dx(m,n)|        dx(i,j) = x(i,j+1) − x(i,j)
B(i,j) = |dy(i,j)| / max |dy(m,n)|        dy(i,j) = y(i,j+1) − y(i,j)
```

The maxima are over the whole item (one row, or one drawing); a channel whose maximum is 0 is 0. A stroke's last point
has no next point and keeps its previous point's step. t(0,0) is the item's earliest point: an IAM line's strokes are
listed word by word, which is not always the order they were written. Each stretch between two points is drawn in the
first point's colour, and strokes are drawn in writing order, so a later stroke crosses over an earlier one.

**Times are the sources' own**: IAM-OnDB's point times, MathWriting's T channel and QuickDraw's raw times, which
note_synth's loaders keep as `Stroke.t`. A split cache built before the loaders kept times has none: rebuild it with
`split_pools.py` (the synthesis tracker does this itself, below).

**Text rows.** Each row's strokes are drawn so the ink, pen included, is **55 px** tall. The row is left-aligned and
padded on the right to **768 px** wide. Rows are stacked with **20 px** between them. The gap under the last row is
dropped, and what remains of the canvas is split evenly above and below:

```
10 × 55 + 9 × 20 = 730 px of rows      768 − 730 = 38  →  19 px above, 19 px below
row tops: 19, 94, 169, 244, 319, 394, 469, 544, 619, 694
```

A row that would be wider than 768 px at 55 px tall is shrunk as a whole, ink and pen together, until it is 768 px
wide, then centred in its 55 px slot. Nothing is cropped.

**Drawing.** The sketch, pen included, is scaled so its longer side is 768 px and centred. This is the same cut
`seg_prompts.crop_transform` makes of a level-2 drawing, so the classifier sees what segmentation hands it.

Strokes are drawn directly at the target size (4× supersampled, then box-filtered), not rendered large and resized.

## The target and its newlines

Rows are joined by a newline. In Florence-2's byte-level BPE tokenizer a newline is one token of the base vocabulary
(`Ċ`, id 50118), trained on from pretraining, where `<sep>` (51279) is a token `processing_florence2.py` adds for
polygon outputs. Labels have every run of whitespace, newlines included, collapsed to one space, so a row break is
always a separator. A label that is empty, or holds a string the tokenizer reads as a special token (`<s>`, `<pad>`,
`<sep>`, `<loc_12>`, … — `recog_layout.SPECIAL_TOKENS`), is left out: it would not be trained on as the text it is.

`training/check_recog_targets.py MANIFEST…` tokenizes targets with the real tokenizer. Each must decode back exactly,
hold no special or unknown token, and have one newline token per row break. Decode with
`clean_up_tokenization_spaces=False` and split on the newline.

**Do not feed these manifests through `training/seg_dataset.py`'s segmentation path.** Its `to_loc_target` treats every
whitespace-separated number in a target as a coordinate. Rows carry a `task` field, which `seg_dataset.py` uses to pass
a recognition target through unchanged.

## From the synthesis tracker

With *Recognition* switched on in the tracker's settings, a dataset run uploads this folder's scripts beside
note_synth and runs `synthesize.py --recognition` for each split: train and val get their samples as their pages are
built, test gets none. Before anything else it checks that the split caches keep stroke times (stage `cache_times`)
and, if they were built before they did, rebuilds them in place with `split_pools.py`, from the archives and
`splits.json` the first run built them with (`~/run/note_synth/data`). The Florence Tuning run trains on
`train/recognition/` together with the segmentation samples.

## Samples from the pools (`recog_synthesize.py`)

```bash
# from recog_synth/
python recog_synthesize.py --task drawing  --split train --quickdraw ../note_synth/data/quickdraw.zip --out dataset
python recog_synthesize.py --task textline --split train --cache ../note_synth/cache --out dataset --workers 32
# a dataset run's whole-pool textline passes (Recognition's "Whole-pool textline passes"), beside the page samples
python recog_synthesize.py --task textline --split train --cache ../note_synth/cache --out dataset --subdir recognition_pool --epochs 3
```

Page samples only read the IAM-OnDB lines a page happened to place (about 6,000 of train's 9,800, each ~74 times), so
a dataset run with recognition on also deals every train line through `--epochs` passes into
`train/recognition_pool/textline/`, which training reads beside `train/recognition/`. Val and test get no pool samples.

This deals a pool's items at random instead of reading pages. Output goes to `<out>/<split>/recognition/<task>/`:
`images/<shard>/<task>_NNNNNNN.png`, `manifest.jsonl` (`{"image", "prompt", "target", "task", "items"}`, `items` the
source ids top to bottom), `report.json`, and `config.json`, `shards/`, `done/` (a rerun skips finished shards; a
rerun with different settings is refused).

**How many samples.** By default, one pass over the pool: every item appears exactly once, so there are ⌈items / 10⌉
text samples, or one sample per drawing. `--epochs N` makes N passes, each through a fresh shuffle, and `--samples`
sets the count directly. One text sample in ten (`--short-prob`) holds 1 to 10 rows instead of 10. The pen width is
drawn at random for each row (text, 2–4 px) or each image (drawing, 3–9 px); set it with `--pen LO HI`.

**Marks (textline).** `--mark-prob` of textline rows (default 0.15) get an underline, strikethrough, box or oval
(`recog_marks.py`). Only the mark's points within one pen width of the writing are added, as mark separation would
leave them. A mark has no recorded times, so all its points take one time, a pause after the row's writing. Formulas
are never marked.

**Splits.** Pool items never cross splits, and they use the same assignment the pages do:

- **IAM-OnDB:** `splits.json`'s form groups, so no writer and no text is in two splits.
- **MathWriting:** the release's own directories (`train` and `synthetic`, `valid`, `test`).
- **QuickDraw:** `sha1(key_id) % 100`.

With `--cache`, the items are exactly what that split's pages are built from, and a `pools.pkl` from another split is
refused.

**QuickDraw source.** The page cache holds only 120 drawings per category for train (30 for val/test), and it drops
sketches under three strokes. With `--quickdraw`, the archive is read instead: 1000 per category for train and 100 for
val/test by default (`--qd-per-category`), keeping one- and two-stroke sketches, since a circle or a line is a real
drawing to a classifier.
