# Handwritten Page Understanding — Synthetic Ink Segmentation & Recognition
 
A from-scratch pipeline for detecting and (optionally) reading the structure of handwritten pages — text blocks, tables, geometry figures, drawings, flowcharts, and their sub-elements — trained entirely on procedurally synthesized data.
 
## Reproduce It Yourself
 
You can use the model that's already trained, or rebuild everything from the source datasets. Either way, the easiest way in is the **tracker**. It's a local web page that runs the dataset synthesis, the training runs, and the model explorer. It rents GPUs for them on Jarvis Labs.
 
### Setup
 
1. Install [PyTorch](https://pytorch.org/get-started/locally/) for your machine, then `pip install -r training/requirements.txt`.
2. Create a [Jarvis Labs](https://jarvislabs.ai) account, add credit, and set your API key as the `JL_API_KEY` environment variable. The tracker uses its bundled CLI (`tracker/jl/`) to create, resume, and pause VMs.
3. Start the tracker and open **http://127.0.0.1:8765**:
   ```bash
   python tracker/tracker_server.py
   ```
   This is the only process you need to start. It keeps itself and the job orchestrators running, and each button starts whatever it needs. To run it hidden at login on Windows, put `tracker/start_trackers.vbs` in your Startup folder. Change the folder path inside it to match your checkout first.
 
### Option A: use the trained model
 
Trained models are saved in [inference/](inference/), one folder each, sorted by kind. `inference/comprehension/<name>` holds a LoRA adapter for Florence-2-large, which segments and reads the page. `inference/mark_separation/<name>` holds the mark separation CNN (`model.pt`), which takes underlines and boxes out of each line before it's read. Each model's `run_info.json` and `stats.json` hold the training config and the validation scores it was saved with.
 
- **In the tracker:** open the **Explore** tab, choose a model in the Model menu and a mark separator in the Mark separator menu, and pick any validation page. The cascade runs on a 1 × H100 VM, which resumes automatically if it's paused. Every stage is drawn against the ground truth. The VM stays up until you press **Pause VM**, so pause it when you're done to stop billing. The **Models** tab compares the validation scores of every model saved in `inference/comprehension/`.
- **From the command line** (on a CUDA GPU):
  ```bash
  cd training
  python seg_eval.py --checkpoint ../inference/comprehension/<model> --pages data/val_pages --predictions preds.jsonl
  ```
  This runs the same cascade and writes per-class metrics, plus every page's raw outputs with `--predictions`. Both routes work on labeled pages: a page image beside a JSON of the same name, as `synthesize.py` and `render_page_val.py` write them, because the results are scored against ground truth.
 
### Option B: train it yourself
 
The tracker's tabs run in pipeline order:
 
1. **Dataset:** synthesizes the train/val/test splits on a dataset VM. **Sample run** makes a small batch to check first, and **Regenerate all** makes the full set. It needs the source datasets; download links are in [note_synth/README.md](note_synth/README.md#data).
2. **Mark separation:** trains the small CNN that removes underlines and boxes from detected lines. Florence Tuning needs a mark separator before it can start. You can train one here and use **Save** to copy it into `inference/mark_separation/<name>`, or pick one that's already saved there and skip this training.
3. **Florence Tuning:** the LoRA fine-tune. Set a budget, the LoRA rank, and optionally a model from `inference/comprehension/` to keep training. The mark separator chosen in step 2 goes to the GPU VM with the run. Watch loss, validation F1, recognition accuracy (correct characters over total characters in what the cascade's own detected lines are read as), and samples live, and use **Save** to copy the best or last checkpoint into `inference/comprehension/<name>`.
4. **Explore / Models:** inspect and compare what you trained, as in Option A. **Run on test** scores a model once on the 2,000 held-out test pages and shows only the final metrics.
 
To skip the tracker, [note_synth/README.md](note_synth/README.md#building-the-dataset) lists the same steps as plain commands (`dataset_splits.py` → `split_pools.py` → `synthesize.py` → `training/train.py` → `seg_eval.py`).
 
## Motivation
 
This project is inspired by Google DeepMind's **InkFM** paper, which unifies text recognition, math recognition, and page segmentation for *online* digital ink — multilingual, timestamped stroke sequences from a stylus or tablet.
 
This project deliberately scopes that down to a different problem: **English-only and no drawing classification with added complexity in geometry and flowchart comprehension**. No off-the-shelf dataset fits this scope, so building one was the first real problem.
 
## Pipeline Overview
 
```
Procedural layout (1600×1600 board)
        │
        ▼
Synthetic ink rendering (handwriting, formulas, geometry, arrows, flowcharts)
        │
        ▼
Hierarchical object labeling (level-2 / level-1 / arrows)
        │
        ▼
768×768 training images + structured text targets
        │
        ▼
LoRA fine-tuning (Florence-2)
        │
        ▼
Detection-style evaluation (IoU / endpoint matching, per-class F1)
```
 
## Object Taxonomy
 
**Level 2 (page-level objects):** `textblock`, `table`, `geometry`, `drawing`, `flowchart`
 
**Level 1 (nested inside a level-2 parent):**
- `textblock` → `textline`, `formula`, `enclosure`, `line mark`
- `table` → `textline`, `formula`
- `geometry` → `vertex`, `segment`, `arc`
- `flowchart` → `textline`, `enclosure`
**Connectors (independent, page-level, not tied to any parent):** `arrow`
 
`enclosure` and `line mark` nest directly under whichever object contains them rather than existing as independent page-level detections — an `enclosure` is either a box drawn around content inside a `textblock`, or a flowchart shape (itself rendered as an enclosure) inside `flowchart`; a `line mark` is a strikethrough or underline element inside a `textblock`. `arrow` is the only class that never has a natural parent to nest under, since it points *between* things rather than belonging to one of them, so it stays a page-level detection on its own.
 
Level-1 objects are labeled relative to a cropped, rescaled version of their parent's region on the board — not the full page — so fine sub-structure can be resolved at full resolution even when the parent object is small relative to the page.
 
## Key Architectural Decisions
 
**Procedural layout via seeded cell-carving.** Pages are laid out on a 1600×1600 working board, carved into cells by a corner-seeding algorithm (each placed cell spawns two new seeds, consumed top-to-bottom then left-to-right, with collision handling for overlapping seeds and undersized runs). This gives exact, noise-free ground-truth coordinates for every object — a property exploited heavily in the ordering and evaluation work below.
 
**Ink sampling.** Handwritten content (prose lines, formulas, table cells) is sampled from a writer-line pool, sized to fit its target row/cell, and — critically — the underlying content pool is split into disjoint train/val/test partitions *before* any page generation happens, so no handwriting sample ever appears across splits. Reuse within a split is allowed and capped as a percentage of a class's total appearances, since geometric labels don't depend on content identity and reuse acts as positional augmentation rather than duplication — except for genuinely rare edge cases (e.g. content a downstream classifier can't resolve into a single line), which are kept at their natural synthesis rate uncapped, since that classifier runs at inference time and those are the exact cases the model will really encounter.
 
**Arrows preserve direction — they are never canonicalized.** Every other box-shaped class is stored as top-left/bottom-right, but an arrow's two coordinate pairs are always start-then-end and are never reordered to a canonical corner form, since doing so would erase the one thing that makes it an arrow rather than a box.
 
**Geometry and flowchart synthesization.** Geometry figures are generated with their vertex labels nested *inside* the figure as level-1 objects (rather than as independent, ambiguous, often single-glyph level-2 textblocks), fixing a real degenerate case where blown-up single-letter labels were diluting the textblock training distribution. Flowchart shapes are sourced from an external offline handwritten flowchart dataset (OHFD) with their original, ambiguous shape content (which mixed text and math indistinguishably) replaced with definitively-labeled synthetic formula/textline substitutions, while keeping OHFD's real, exact bounding boxes.
 
**Inverted polarity for both tasks; grayscale-as-3-channel for segmentation, colored ink for recognition.** Images are rendered as dark ink on a white background, then inverted when loaded so that ink is encoded as 1 and background as 0 — inverted from the natural convention — specifically so that zero-value padding (introduced when cropping/rescaling level-1 objects) reads as background rather than as a solid stroke of ink. Segmentation images are natively single-channel, duplicated across all three channels (rather than zero-padding the unused channels) to stay in-distribution for a vision encoder pretrained on real RGB imagery. Recognition images are true RGB, following InkFM: each stroke is colored by time and pen movement. They go through the same per-channel inversion (each channel becomes 1 − value/255), so the model sees the complement of the rendered colors on a black background, identically in training and evaluation.
 
## Model & Training
 
- **Backbone:** Florence-2-large, fine-tuned via LoRA (frozen vision encoder, low-rank adapters on attention/projection layers).
- **Resolution:** 1600×1600 procedural board → 768×768 training images, matching Florence-2's native high-resolution tuning stage rather than the more common but resolution-limiting 224×224.
- **Coordinate quantization:** 0–999 bins, matching Florence-2's own native location-token scheme.
## Evaluation Methodology
 
Token-level cross-entropy is used for training, but never for model selection — it's the wrong metric for a structured detection task, since a single early token error desyncs everything after it without reflecting whether the actual detected objects are correct. Validation instead uses:
 
- **Per-class precision/recall/F1**, matched via IoU (box classes), pixel-distance threshold (vertex), and independent start/end distance matching with no direction-swapping allowed (arrow).
- **Parse validity rate**, tracked separately from detection accuracy, since a malformed generation and a well-formed-but-wrong one are different failure modes that need different fixes.
- **Cascaded (autoregressive) evaluation**, not just teacher-forced: level-1 crops are taken from the model's *own* predicted level-2 boxes, not ground truth, so the reported accuracy reflects true end-to-end error propagation rather than an artificially optimistic score.
This harness caught two real, non-obvious bugs during development: a class-imbalance problem where high-frequency classes (text, formulas) were dominating gradient signal at the expense of rare ones (tables, arrows, flowcharts) — fixed by rebalancing the synthetic generator's target object-density per page — and a decoding artifact (a duplicated leading special token) that was silently causing valid predictions to be scored as complete parse failures for several classes.
 
## Status
 
Segmentation and recognition are both implemented end-to-end, from synthetic data generation through fine-tuning and evaluation.

- **Segmentation:** all object classes (level-2, level-1, and arrows), scored with detection-style per-class F1 through the full cascade.
- **Recognition:** one Florence-2 model, trained on the recognition prompts next to the segmentation ones. It reads handwritten text lines as English, formulas as LaTeX, and drawings as their category. Its training images color the ink by time and pen movement, following InkFM ([recog_synth/](recog_synth/README.md)). In the cascade, the textlines and formulas the model detected are read ten at a time.
- **Mark separation:** a small CNN ([training/marks_cnn/](training/marks_cnn/)) removes underlines, strikethroughs, and boxes from each detected line before it's read, so the marks aren't read as text. It also says which of the four kinds of mark are in the crop.

## Note Comprehension Performance

These are the validation scores of the saved model, [inference/comprehension/comprehension-v3](inference/comprehension/comprehension-v3/). It's the best checkpoint of a 3-epoch LoRA fine-tune (rank 32, alpha 64) continued from `comprehension-v2`: step 15,000 of 15,564, on 4 × H100, global batch 128, lr 5e-5, over 664,048 training rows. It was scored on 200 held-out validation pages and, teacher-forced, on 300 manifest rows. The numbers are in its `stats.json`, computed by [seg_eval.py](training/seg_eval.py) and [recog_eval.py](training/recog_eval.py), and are what the tracker's Models tab shows.

| Score | comprehension-v3 |
| --- | --- |
| Pages mean F1 (cascade) | 0.977 |
| — Level 2 (blocks) | 0.983 |
| — Level 1 (inside blocks) | 0.975 |
| — Arrows | 0.953 |
| Pages parse-valid | 100.0% |
| Rows mean F1 (teacher-forced) | 0.966 |
| Rows parse-valid | 100.0% |
| Recognition accuracy (textlines) | 89.7% |
| Recognition accuracy (formulas) | 77.2% |
| Recognition CER (textlines) | 11.1% |
| Recognition CER (formulas) | 26.5% |

Per class, over all 200 pages through the cascade:

| Class | Group | Precision | Recall | F1 | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| textblock | level 2 | 0.997 | 0.998 | 0.997 | 1,836 | 6 | 4 |
| table | level 2 | 1.000 | 1.000 | 1.000 | 147 | 0 | 0 |
| flowchart | level 2 | 1.000 | 1.000 | 1.000 | 148 | 0 | 0 |
| geometry | level 2 | 0.986 | 0.986 | 0.986 | 216 | 3 | 3 |
| drawing | level 2 | 0.942 | 0.925 | 0.933 | 49 | 3 | 4 |
| textline | level 1 | 0.994 | 0.988 | 0.991 | 3,772 | 22 | 46 |
| formula | level 1 | 0.995 | 0.998 | 0.996 | 7,214 | 34 | 18 |
| enclosure | level 1 | 0.980 | 0.983 | 0.982 | 941 | 19 | 16 |
| line mark | level 1 | 0.993 | 0.983 | 0.988 | 291 | 2 | 5 |
| vertex | level 1 | 0.987 | 0.973 | 0.980 | 975 | 13 | 27 |
| segment | level 1 | 0.966 | 0.959 | 0.962 | 974 | 34 | 42 |
| arc | level 1 | 0.939 | 0.918 | 0.928 | 168 | 11 | 15 |
| arrow | page-level | 0.949 | 0.957 | 0.953 | 851 | 46 | 38 |

**How the numbers are computed** (in [seg_eval.py](training/seg_eval.py), with recognition in [recog_eval.py](training/recog_eval.py)):

- **The cascade.** A page is resized to 768 × 768 and asked for its level-2 objects, then for its arrows. Every *predicted* level-2 box — not the ground truth's — is cut out of the full-resolution page, scaled so its longer side is 768, and asked its level-1 prompt. Every prediction is mapped back to page pixels and scored against the whole page at once, so a level-2 box the model misses or misplaces also costs the level-1 objects inside it. This is why the page scores are the ones that matter: they carry the error propagation the model has at inference.
- **Matching**, per page and per class, greedy best-first and one-to-one: boxes at IoU ≥ 0.5 (a box thinner than 4 px, as an underline can be, is first widened to that about its centre, prediction and ground truth alike); a vertex within 20 page pixels; a segment when both ends are within that distance, either way round; an arc when its two ends match in order and its labelled middle lies within that distance of the true arc anywhere between them; an arrow when start-to-start *and* end-to-end are each within 30 page pixels, so an arrow predicted backwards does not match.
- **Precision / recall / F1.** TP, FP and FN are summed over all 200 pages before the ratios are taken, so the count column is what each class's F1 weighs.
- **Mean F1** is the *unweighted* mean of the per-class F1 over the 13 classes present in the ground truth, so rare classes (drawing, arc) count as much as formulas. The level 2 / level 1 / arrow rows are that same mean restricted to each group.
- **Parse-valid** is the share of generations that parse cleanly into whole objects: `</s><s>`, then each object exactly its class's number of `<loc_…>` tokens followed by a class its prompt asked for, then `</s>`. Stray text, a coordinate with no class, a box with x0 > x1, or running out of tokens all make an output invalid. It's tracked apart from F1 because a malformed generation and a well-formed wrong one are different failures. The well-formed objects of an invalid output still get matched.
- **Rows** is the teacher-forced check: each manifest row's prompt on its own image, a level-1 crop taken from its *ground-truth* parent box, scored in the row image's pixels (vertex 10 px, arrow 15 px). It isolates each prompt from cascade error, and it runs on a much smaller set (300 rows), so a handful of objects moves it — it's a training-time sanity check, not the headline number.
- **Recognition accuracy** is correct characters over total characters, for the textlines and then the formulas the cascade itself detected, read ten at a time from the page's own strokes with the mark separator applied first. A row's correct characters are the longest common subsequence of what was read and its target; its total is the longer of the two, so extra characters cost as much as missing ones. Every ground-truth line no detected box matched counts as read as nothing, all its characters wrong. **CER** is the edit distance of the read text against the target over the target's characters, on the same rows. Formulas are compared with whitespace removed, textlines with each run of whitespace made one space.

Token cross-entropy is not used for model selection here. `best/` is kept by a combined score, 0.6 × pages mean F1 + 0.4 × the mean of the two recognition accuracies — a box in the right place around text the model misreads is only half a result. For this checkpoint that score is 0.920.

## Mark Separation Performance

These are the validation scores of the saved mark separator, [inference/mark_separation/marks](inference/mark_separation/marks/). It's the last checkpoint of a 50-epoch run: 10,900 steps on 1 × H100, batch 256, over 55,742 training crops. It was scored on 4,625 validation crops. The numbers are in its `stats.json`, and are what the tracker's Mark separation tab shows for this run.

Each crop is one region of a page, scaled to fit 600 × 100 with its aspect ratio kept. The model gets the **input** (the writing with its marks drawn over it). It returns two things: the **writing** image with the marks removed, and four **labels** saying which kinds of mark are in the crop. One crop can hold several kinds.

| Output | Precision | Recall | F1 | IoU |
| --- | --- | --- | --- | --- |
| Writing | 0.954 | 0.991 | 0.972 | 0.946 |
| Labels (mean of the four) | – | – | 0.990 | – |

| Kind of mark | Crops with it | Label precision | Label recall | Label F1 | Writing F1 | Writing IoU | Mark ink kept as writing |
| --- | --- | --- | --- | --- | --- | --- | --- |
| underline | 2,562 | 0.996 | 0.995 | 0.995 | 0.972 | 0.945 | 0.3% |
| strikethrough | 487 | 0.966 | 0.979 | 0.972 | 0.975 | 0.951 | 4.0% |
| oval | 637 | 0.994 | 0.997 | 0.995 | 0.967 | 0.936 | 1.6% |
| rectangle | 1,146 | 0.999 | 0.996 | 0.997 | 0.973 | 0.947 | 0.8% |

All four labels right on 99.2% of crops · 0.7% of mark ink kept as writing · input copied as writing: F1 0.486.

**What the fields mean** (computed in [marks_eval.py](training/marks_cnn/marks_eval.py)):

- **Writing precision / recall / F1 / IoU:** pixel scores of the predicted writing against the true writing. A true pixel counts as ink when it's darker than mid-grey. A predicted pixel counts as ink when its probability is over 0.5. The pixel counts are summed over all crops before the ratios are taken, so larger crops weigh more. Precision is the share of predicted ink that is really writing, which is what drops when mark ink is left in. Recall is the share of the real writing that was kept, which is what drops when writing is erased with a mark. F1 is the Dice overlap of the two ink masks, and IoU is their intersection over union.
- **Labels (mean of the four):** the plain mean of the four per-kind label F1 scores.
- **Kind of mark rows:** each row covers only the crops that truly hold that kind of mark.
  - **Crops with it:** how many validation crops hold that kind. The column adds up to more than 4,625 because one crop can hold several kinds.
  - **Label precision / recall / F1:** a kind is predicted present when its probability is over 0.5. These are counted per crop over all 4,625 crops. For example, label precision for strikethrough is the share of crops called "strikethrough" that really have one.
  - **Writing F1 / IoU:** the writing scores above, pooled over only the crops that hold this kind.
  - **Mark ink kept as writing:** of the pixels that are ink in the input but not in the true writing (the mark's own ink), the share the model predicted as writing. Lower is better. This is the ink that would be read as text.
- **All four labels right:** the share of crops where every one of the four labels matches.
- **Input copied as writing:** the writing F1 of a model that changes nothing and returns the input as the writing. It's the baseline: the gap from 0.486 to 0.972 is what removing the marks gains.

During training, the checkpoint kept as `best/` is the one with the highest **mean F1**, the mean of the writing F1 and the label mean F1. That peaked at 0.983 at step 7,848. The saved model is the final checkpoint instead, which scores 0.981.
 
## Drawbacks
 
- **Geometry is limited to circles, circular arcs, and polygons.** Figures are assembled from straight segments and circular arcs, combined in arbitrary ways. Ellipses, free-form curves, function plots, and 3D solids aren't modeled. The model also has no notion of which figures make geometric sense.
- **Geometry labels and markings aren't tied to what they describe.** A corner letter, a length, or an angle value is detected as a `textline` or `formula` inside the figure. Angle marks (arcs, right-angle boxes) and equal-length tick marks on segments aren't output at all. None of these is linked to the vertex, angle, or segment it refers to.
- **Flowchart arrow labels aren't tied to their arrows.** A caption on an arrow, such as "yes" or "no", is detected as a line of the flowchart, but nothing records which arrow it labels.
- **Recognition relies on a separate mark-separation model.** Underlines, strikethroughs, and boxes are removed from each line by a second network before the line is read. Its mistakes carry straight into recognition: a mark it misses is read as part of the text, and writing it wrongly removes is lost. It's also one more model to train (before the segmentation run) and to run at inference.
 
## Acknowledgments
 
Built on [Florence-2](https://huggingface.co/microsoft/Florence-2-large) (Microsoft) and informed by the target-ordering conventions established in the InkFM (Google DeepMind) papers.
 
Every page is synthesized from the ink in these datasets. Thanks to their authors for making them available:
 
- **[IAM-OnDB](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database)**: handwritten text lines, the source of all prose, table cells, and captions. M. Liwicki and H. Bunke, "IAM-OnDB – an On-Line English Sentence Database Acquired from Handwritten Text on a Whiteboard," *ICDAR* 2005.
- **[MathWriting](https://github.com/google-research/google-research/tree/master/mathwriting)**: handwritten mathematical expressions, the source of every `formula`. P. Gervais, A. Fadeeva, and A. Maksai, "MathWriting: A Dataset for Handwritten Mathematical Expression Recognition," 2024.
- **OHFD (Online Handwritten Flowchart Dataset)**: flowchart layouts, shapes, arrows, and bounding boxes. A.-M. Awal, G. Feng, H. Mouchère, and C. Viard-Gaudin, "First Experiments on a New Online Handwritten Flowchart Database," *Document Recognition and Retrieval XVIII*, 2011 (IRCCyN / Université de Nantes).
- **[Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset)**: sketches for the `drawing` class. Google Creative Lab, CC BY 4.0.
- **NIST Special Database 19**: handprinted characters used to letter geometry figures. P. J. Grother, *NIST Special Database 19: Handprinted Forms and Characters Database*, NIST, 1995.
- **[DeepWriting](https://ait.ethz.ch/deepwriting)**: used to recover IAM-OnDB's writer IDs, so that no writer appears in more than one split. E. Aksan, F. Pece, and O. Hilliges, "DeepWriting: Making Digital Ink Editable via Deep Generative Modeling," *CHI* 2018.
 
The geometry figures, enclosures, underlines, table rules, and connector arrows reuse real pen strokes mined from the datasets above. Each dataset keeps its own license and terms of use (IAM-OnDB, for example, requires registration), so download them from their sources.