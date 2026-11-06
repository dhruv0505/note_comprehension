"""
synthesize.py — the dataset: pages generated in parallel for one split, from its
split_pools.py cache, written as training/'s render_seg_data.py and
render_page_val.py write theirs (page_export.py).

    python synthesize.py --cache cache --split train --pages 30000 --workers 32 --out dataset
    python synthesize.py --cache cache --split val   --pages 2000  --workers 32 --out dataset
    python synthesize.py --cache cache --split test  --pages 2000  --workers 32 --out dataset

A page is laid out on the generator's 1000-unit board and drawn at --size (1600)
pixels. What each split gets, under <out>/<split>/:

    samples/   train and val. manifest.jsonl, one row per prompt, {"image", "prompt",
               "target"}, and images/<shard>/p0000123_s00.png, each --image (768) square:
               the full page downsized, then each level-2 object cut out of the 1600 page
               and scaled to fit (render_seg_data.py's files). Train, with
               --geometry-per-page N: also N geometry figures a page drawn without a
               page (geometry_sample), each cut out as a cascade's geometry box is and
               asked only the geometry level-1 prompt, images/<shard>/g0000123.png.
    pages/     val and test. page0000123.png, the 1600 x 1600 page, beside
               page0000123.json listing every labelled object on it
               (render_page_val.py's format).
    marks/     every split. For each enclosure and line mark inside a textblock (marks
               that overlap taken together), a crop of the 1600 page and the same crop
               as the writing alone, with a 0/1 label per mark type in manifest.jsonl
               (mark_crops.py).
    recognition/<task>/   train and val, with --recognition. The page's textlines, then its
               formulas, ten to an image, each row read as the cascade reads a line with the
               ground truth's box, marks left out (recog_synth/recog_pages.py):
               images/<shard>/p0000123_NN.png, manifest.jsonl {"image", "prompt",
               "target", "task", "page", "boxes"} and report.json. Test gets none: its
               recognition is only scored through the cascade.
    usage/     per shard, how often each source item was used.
    report.json, once every shard is done: pages, samples, rows, marks, and per
               source class how evenly its items were used against the cap.

--tuning FILE overrides how often each class appears (class_tuning.py: cell
weights, formula and mark probabilities, connector arrow counts), set in every
worker before its first page. --mark-crop box crops each mark to its own box
(marks that overlap joined) instead of the default "context" crop. Both are
recorded in report.json; without them a run is as before.

Manifests are written per shard under <folder>/shards/ as shards finish and joined
into <folder>/manifest.jsonl at the end. A val page's samples, page files and mark
crops are all the same page. A run is resumable: a shard whose usage file exists
is done and is skipped, and every page's seed is a function of --seed, the split
and the page number, so a page comes out the same whichever worker makes it (bar
the usage cap, which is kept per worker).

Reuse cap. Within a split, content is reused freely across pages — positional
augmentation — but no single source item may take more than --cap-share of its
class's appearances (IAM-OnDB lines, MathWriting expressions, FC charts,
QuickDraw drawings). A class whose pool is too small for that share gets
CAP_MIN_MULTIPLE / pool size instead, the smallest cap a pool can meet, and the
report says so. Items at their cap are kept off the next page: filtered out of
the line, chart and drawing pools, and redirected elsewhere when a MathWriting
expression is drawn (which also keeps one expression from being drawn twice on a
page). Uses inside a textblock that is a single line the line-count classifier
(labels.is_single_line) cannot recognise as one are not counted against the cap:
those blocks reach the model as they would in deployment, and they are kept at
the rate the generator makes them.
"""

from __future__ import annotations
import argparse
import glob
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter

SHARD_PAGES = 250
DEFAULT_CAP_SHARE = 0.01
# A class's cap, where it must be looser than --cap-share. FC charts: a cell takes only a chart whose captions stay
# legible at its size (block_generators.make_diagram), and in a small cell that is a handful of the split's 180, which
# reached a 1% cap within a few hundred pages; every flowchart cell after that came out as a textblock, and a dataset
# run made 0.23 flowcharts a page against the 0.55 the OHFD one made. A 500-page simulation, flowcharts per page after
# page 200: 0.06 at the 1.1% cap, 0.66 at 2%, 1.02 at 3% (no cap reached).
CLASS_CAP_SHARE = {"fc": 0.02}
# Val and test hold 35 FC charts each (dataset_splits: held-out writers AND layouts), of which only some 10-15 fit a
# given flowchart cell legibly. A cap below one over that count can never be met by all of them at once, so every
# usable chart sits at its cap and flowcharts all but vanish: 0.05 a val page at 5.7% (2 / 35), 1.27 at 10%; test
# 0.70 and 1.31. 10% is the cap that keeps their flowcharts; train, with 180 charts, keeps 2% (0.55 a page).
EVAL_CLASS_CAP_SHARE = {"fc": 0.10}
CAP_MIN_MULTIPLE = 2.0
FLAT_RETRIES = 5
CLASSES = {"iam": "IAM-OnDB lines", "mw": "MathWriting expressions",
           "fc": "FC flowcharts", "qd": "QuickDraw drawings"}
MODES = {"train": ("samples", "marks"), "val": ("samples", "pages", "marks"), "test": ("pages", "marks")}
RECOGNITION_SPLITS = ("train", "val")      # --recognition's samples; test recognition is only scored through the cascade
RECOG_SYNTH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recog_synth")


def _recog_pages():
    """recog_synth's recog_pages, from the folder beside note_synth (the dataset VM's run folder holds both)."""
    if RECOG_SYNTH not in sys.path:
        sys.path.insert(0, RECOG_SYNTH)
    import recog_pages
    return recog_pages


def page_seed(seed: int, split: str, index: int, attempt: int = 0) -> int:
    digest = hashlib.sha1(f"{seed}:{split}:{index}:{attempt}".encode()).hexdigest()
    return int(digest[:12], 16)


def _item(ink):
    return (ink.meta or {}).get("item")


# ----------------------------------------------------------------------------
# Usage: counting, capping, and the exemption
# ----------------------------------------------------------------------------

class Usage:
    """One worker's count of how often each source item has been used."""

    def __init__(self, pools: dict, split: str, cap_share: float):
        math = pools.get("math_pool")
        self.split = split
        self.sizes = {"iam": len(pools.get("text_pool") or []),
                      "mw": len(math) if math is not None else 0,
                      "fc": len(pools.get("flowchart_pool") or []),
                      "qd": len(pools.get("quickdraw_pool") or [])}
        floors = {**CLASS_CAP_SHARE, **(EVAL_CLASS_CAP_SHARE if split in ("val", "test") else {})}
        self.share = {cls: (max(cap_share, floors.get(cls, 0.0), CAP_MIN_MULTIPLE / n) if n else 1.0)
                      for cls, n in self.sizes.items()}
        self.counts = {cls: Counter() for cls in CLASSES}
        self.totals = Counter()
        self.shard_uses = Counter()
        self.shard_exempt = Counter()
        self.rng = random.Random(0)
        self.page_served = set()        # MathWriting indices already handed out on this page

    def limit(self, cls: str) -> float:
        return max(1.0, self.share[cls] * self.totals[cls])

    def capped(self, item) -> bool:
        if not item:
            return False
        cls = item.split(":", 1)[0]
        return cls in self.counts and self.counts[cls][item] >= self.limit(cls)

    def _servable(self, index: int) -> bool:
        return index not in self.page_served and not self.capped(f"mw:{self.split}:{index}")

    def redirect_math(self, index: int) -> int:
        """The expression to serve for a pick of `index`: itself, unless it is at its
        cap or already on this page, in which case another that is neither."""
        if not self._servable(index):
            for _ in range(32):
                other = self.rng.randrange(self.sizes["mw"])
                if self._servable(other):
                    index = other
                    break
        self.page_served.add(index)
        return index

    def page_pools(self, pools: dict) -> dict:
        """The pools with every capped line, chart and drawing held back for a page.
        Called once at the start of each page, which also starts its MathWriting tally."""
        self.page_served = set()
        out = dict(pools)
        for key in ("text_pool", "flowchart_pool", "quickdraw_pool"):
            pool = pools.get(key)
            if pool:
                out[key] = [ink for ink in pool if not self.capped(_item(ink))] or pool
        return out

    def record(self, blocks) -> int:
        """Count a finished page's uses; returns the most uses of one item on it."""
        uses, exempt = page_uses(blocks)
        for item in uses:
            cls = item.split(":", 1)[0]
            if cls in self.counts:
                self.counts[cls][item] += 1
                self.totals[cls] += 1
            self.shard_uses[item] += 1
        for item in exempt:
            self.shard_exempt[item] += 1
        return max(Counter(uses).values(), default=0)


def page_uses(blocks):
    """(counted, exempt) source item ids, one entry per use on the page.

    A use is one labelled unit built from an item: a textline (its words carry
    the source line's id even when the row was rebuilt from them), a formula, a
    drawing, a flowchart. A flowchart's captions are uses of their own items.
    Everything inside a single-line textblock that is_single_line fails to call
    single-line is exempt from the cap.
    """
    import labels

    counted, exempt = [], []

    def ids(node):
        found = []
        own = _item(node)
        if own:
            found.append(own)
        if node.class_name == "textline":
            for child in node.children:
                child_item = _item(child)
                if child_item and child_item not in found:
                    found.append(child_item)
        return found

    def walk(node, exempt_block):
        cls = node.class_name
        if cls == "textblock":
            rows = [c for c in node.children if c.class_name in ("textline", "formula")]
            if len(rows) == 1 and not labels.is_single_line(rows[0].all_strokes()):
                exempt_block = True
        if cls in ("textline", "formula", "drawing") or (node.meta or {}).get("flowchart"):
            (exempt if exempt_block else counted).extend(ids(node))
            if cls in ("textline", "formula", "drawing"):
                return
        for child in node.children:
            walk(child, exempt_block)

    for block in blocks:
        walk(block, False)
    return counted, exempt


# ----------------------------------------------------------------------------
# Workers
# ----------------------------------------------------------------------------

_W: dict = {}


def _init(cache, split, out, seed, cap_share, size, image, tuning=None, mark_crop="context", recognition=False,
          only_recognition=False, geometry_per_page=0.0, only_geometry=False):
    _W["geometry_per_page"] = float(geometry_per_page or 0.0)
    _W["only_geometry"] = only_geometry
    sys.dont_write_bytecode = True
    _W["recognition"] = (recognition or only_recognition) and split in RECOGNITION_SPLITS
    _W["only_recognition"] = only_recognition
    if _W["recognition"]:
        _recog_pages()                  # fails here, before any page, when recog_synth isn't beside note_synth
    import seg_prompts as sp
    sp.configure(image=image)
    import class_tuning
    import split_pools

    class_tuning.apply(tuning)
    pools = split_pools.load_split(cache, split)
    usage = Usage(pools, split, cap_share)
    if pools.get("math_pool") is not None:
        pools["math_pool"].redirect = usage.redirect_math
    _W.update(pools=pools, usage=usage, split=split, out=out, seed=seed, size=size, sp=sp, mark_crop=mark_crop)


def _paths(out, split, shard=None):
    root = os.path.join(out, split)
    paths = {"root": root, "pages": os.path.join(root, "pages"), "samples": os.path.join(root, "samples"),
             "marks": os.path.join(root, "marks"), "recognition": os.path.join(root, "recognition"),
             "geometry": os.path.join(root, "geometry")}
    if shard is not None:
        paths["usage"] = os.path.join(root, "usage", f"shard_{shard:05d}.json")
    return paths


def _shard_manifest(folder, shard):
    return os.path.join(folder, "shards", f"manifest_{shard:05d}.jsonl")


def _write_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    os.replace(tmp, path)


def _run_shard(task):
    shard, first, count = task
    import mark_crops as mc
    import page_export as pe
    import page_generator as pg

    sp, usage, split, size = _W["sp"], _W["usage"], _W["split"], _W["size"]
    # --only-recognition and --only-geometry write nothing else
    modes = () if _W.get("only_recognition") or _W.get("only_geometry") else MODES[split]
    recognizing = _W.get("recognition", False)
    paths = _paths(_W["out"], split, shard)
    if "pages" in modes:
        os.makedirs(paths["pages"], exist_ok=True)
    usage.shard_uses.clear()
    usage.shard_exempt.clear()
    rows, mark_rows, stats = [], [], Counter()
    recog_rows, recog_stats = {}, {}
    started = time.time()
    scale = size / sp.BOARD_SIZE

    for index in (() if _W.get("only_geometry") else range(first, first + count)):     # --only-geometry: no page
        for attempt in range(FLAT_RETRIES):
            seed = page_seed(_W["seed"], split, index, attempt)
            random.seed(seed)
            usage.rng.seed(seed ^ 0x5EED)
            # ENCLOSURE_PROB passed, not left to the default argument, so a --tuning override reaches it
            blocks = pg.generate_page(**usage.page_pools(_W["pools"]),
                                      enclosure_prob=pg.ENCLOSURE_PROB)["board_blocks"]
            if "pages" not in modes:
                # no page files, so no page is turned away: recognition only needs its lines, arrows left out
                objects = pe.page_objects(blocks, scale, arrows=False) if recognizing else None
                break
            try:
                objects = pe.page_objects(blocks, scale)
                break
            except pe.NoDirection:
                stats["rejected"] += 1          # a page's ground truth must give every arrow a direction
        else:
            raise RuntimeError(f"{split} page {index}: {FLAT_RETRIES} attempts each had an arrow with no direction")

        stats["max_item_uses_on_a_page"] = max(stats["max_item_uses_on_a_page"], usage.record(blocks))
        page = pe.render_page(blocks, size, sp.BOARD_SIZE, sp.INK_WIDTH_PX) if modes else None   # recognition draws its own
        rel_dir = f"images/{shard:05d}"
        if "samples" in modes:
            page_rows, saved = pe.write_samples(blocks, page, scale, paths["samples"], rel_dir, f"p{index:07d}")
            rows.extend(page_rows)
            stats["samples"] += saved
        if "pages" in modes:
            name = f"page{index:07d}"
            page.save(os.path.join(paths["pages"], name + ".png"))
            with open(os.path.join(paths["pages"], name + ".json"), "w") as f:
                json.dump(pe.page_record(name, objects, size, sp.BOARD_SIZE, _W["seed"], index), f)
        if "marks" in modes:
            mark_rows.extend(mc.write_marks(blocks, scale, sp.INK_WIDTH_PX, size, paths["marks"],
                                            rel_dir, f"p{index:07d}", index, crop=_W["mark_crop"]))
        if recognizing:
            # the page's textlines and formulas, ten to an image, read as the cascade reads them with the truth's boxes
            page_rows, page_stats = _recog_pages().write_samples(objects, paths["recognition"], rel_dir, f"p{index:07d}")
            for kind, kind_rows in page_rows.items():
                recog_rows.setdefault(kind, []).extend(kind_rows)
                recog_stats.setdefault(kind, Counter()).update(page_stats[kind])
        stats["pages"] += 1

    if ("samples" in modes or _W.get("only_geometry")) and _W.get("geometry_per_page"):
        # figures without a page, numbered across the split so the shards add up to pages x geometry_per_page; with
        # --only-geometry into their own folder, geometry/, and nothing of the split's pages or samples is touched
        per, rel_dir = _W["geometry_per_page"], f"images/{shard:05d}"
        folder = paths["geometry"] if _W.get("only_geometry") else paths["samples"]
        os.makedirs(os.path.join(folder, rel_dir), exist_ok=True)
        for number in range(round(first * per), round((first + count) * per)):
            random.seed(page_seed(_W["seed"], f"{split}-geometry", number))
            made = geometry_sample(usage.page_pools(_W["pools"]), random)
            if made is None:
                stats["geometry_skipped"] += 1
                continue
            blocks, figure, crop = made
            page = pe.render_page(blocks, size, sp.BOARD_SIZE, sp.INK_WIDTH_PX)
            transform = sp.crop_transform(crop)
            entry = sp.prompt_entry(sp.LEVEL1_PROMPTS["geometry"], sp.level1_objects(figure, "geometry"), transform)
            rel = f"{rel_dir}/g{number:07d}.png"
            pe.sample_image(page, transform, scale, sp.IMAGE_SIZE).save(os.path.join(folder, rel))
            rows.append({"image": rel, "prompt": entry["prompt"], "target": entry["target"]})
            stats["samples"] += 1
            stats["geometry_samples"] += 1

    if _W.get("only_geometry"):
        _write_rows(_shard_manifest(paths["geometry"], shard), rows)
    if "samples" in modes:
        _write_rows(_shard_manifest(paths["samples"], shard), rows)
    if "marks" in modes:
        _write_rows(_shard_manifest(paths["marks"], shard), mark_rows)
    recognition = {}
    if recognizing:
        for kind in _recog_pages().TEXT_CLASSES:         # every task's shard manifest, empty or not, for the join
            _write_rows(_shard_manifest(os.path.join(paths["recognition"], kind), shard), recog_rows.get(kind, []))
            recognition[kind] = {k: int(v) for k, v in recog_stats.get(kind, Counter()).items()}
    stats["rows"] = len(rows)
    stats["marks"] = len(mark_rows)
    record = {"shard": shard, "first": first, "count": count, **stats,
              "seconds": round(time.time() - started, 2),
              "uses": dict(usage.shard_uses), "exempt": dict(usage.shard_exempt),
              "cap_share": usage.share, "pool_sizes": usage.sizes,
              **({"recognition": recognition} if recognizing else {})}
    os.makedirs(os.path.dirname(paths["usage"]), exist_ok=True)
    tmp = paths["usage"] + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f)
    os.replace(tmp, paths["usage"])            # written last: a shard with this file is done
    return shard, dict(stats), record["seconds"]


# ----------------------------------------------------------------------------
# Geometry figures without a page (train, --geometry-per-page)
# ----------------------------------------------------------------------------

# A figure's board cell: this share of the page across, a little off square. Val pages' figure boxes are 11-23% of
# the page (median 16%), drawn at page_generator.GEOMETRY_FILL of their cell.
GEOMETRY_CELL = (0.13, 0.30)
GEOMETRY_ASPECT = (0.8, 1.25)
# The crop is the figure's box grown on each side by up to this share of its longer side, as a predicted box is seldom
# tight; never shrunk, so every level-1 object stays inside the image.
GEOMETRY_CROP_GROW = 0.08
# How often a sketch stands beside the figure, close enough for some of its ink to fall inside the crop, as page
# neighbours do.
GEOMETRY_NEIGHBOUR_PROB = 0.35


def geometry_sample(pools, rng):
    """(board blocks, the figure, the crop's board box) for one geometry figure drawn on its own at a page cell's size,
    or None when the split's pools can't draw one. The figure is labelled as a page's is (seg_prompts.level1_objects)."""
    import block_generators as bg
    import page_generator as pg
    import seg_prompts as sp
    from core import fit_ink_to_box

    board = float(sp.BOARD_SIZE)
    figure = bg.make_geometry(pools.get("shape_bank"), glyphs=pools.get("glyph_pool"), math_pool=pools.get("math_pool"))
    if figure is None:
        return None
    side, aspect = rng.uniform(*GEOMETRY_CELL) * board, rng.uniform(*GEOMETRY_ASPECT)
    w, h = side * aspect ** 0.5, side / aspect ** 0.5
    x0, y0 = rng.uniform(0.0, board - w), rng.uniform(0.0, board - h)
    figure = fit_ink_to_box(figure, pg._inset((x0, y0, x0 + w, y0 + h), pg.GEOMETRY_FILL, rng), jitter=0.02)
    fx0, fy0, fx1, fy1 = figure.bbox()
    longer = max(fx1 - fx0, fy1 - fy0)
    grow = lambda: rng.uniform(0.0, GEOMETRY_CROP_GROW) * longer
    crop = [max(0.0, fx0 - grow()), max(0.0, fy0 - grow()), min(board, fx1 + grow()), min(board, fy1 + grow())]
    blocks = [figure]
    drawings = pools.get("quickdraw_pool")
    if drawings and rng.random() < GEOMETRY_NEIGHBOUR_PROB:
        size, gap = rng.uniform(0.5, 1.0) * longer, rng.uniform(0.0, GEOMETRY_CROP_GROW) * longer
        cx, cy = (fx0 + fx1) / 2.0, (fy0 + fy1) / 2.0
        box = {"left": (fx0 - gap - size, cy - size / 2, fx0 - gap, cy + size / 2),
               "right": (fx1 + gap, cy - size / 2, fx1 + gap + size, cy + size / 2),
               "above": (cx - size / 2, fy0 - gap - size, cx + size / 2, fy0 - gap),
               "below": (cx - size / 2, fy1 + gap, cx + size / 2, fy1 + gap + size)}[rng.choice(("left", "right", "above", "below"))]
        box = (max(0.0, box[0]), max(0.0, box[1]), min(board, box[2]), min(board, box[3]))
        if box[2] - box[0] > 0.02 * board and box[3] - box[1] > 0.02 * board:
            blocks.append(fit_ink_to_box(bg.make_drawing(drawings), box, jitter=0.02))
    return blocks, figure, crop


def load_json(path):
    with open(path) as f:
        return json.load(f)


def _W_prompt_geometry():
    import seg_prompts as sp
    return sp.LEVEL1_PROMPTS["geometry"]


def join_manifests(folder: str, shards, once_per_chart: bool = False):
    """Every shard's rows under `folder`/shards, in shard order, into `folder`/manifest.jsonl; (rows written,
    rows dropped).

    A flowchart's level-1 crop row names its chart (`level1_chart`, page_export.write_samples), which the joined
    manifest leaves out. With `once_per_chart` (train) only the first such row of each chart is kept: an FC chart
    comes back on many pages, and its level-1 sample is trained on once. The rest are dropped and their images
    deleted (a flowchart's crop has no other prompt). Decided here, after every shard, so no shard needs to know
    what the others placed, and a rerun drops the same rows."""
    target = os.path.join(folder, "manifest.jsonl")
    tmp = target + ".tmp"
    rows = dropped = 0
    charts = set()
    with open(tmp, "w", newline="\n") as joined:
        for shard, _, _ in shards:
            with open(_shard_manifest(folder, shard)) as part:
                for line in part:
                    if not line.strip():
                        continue
                    if '"level1_chart"' in line:
                        row = json.loads(line)
                        chart = row.pop("level1_chart")
                        if once_per_chart and chart in charts:
                            dropped += 1
                            try:
                                os.remove(os.path.join(folder, row["image"]))
                            except FileNotFoundError:       # a rerun: dropped last time
                                pass
                            continue
                        charts.add(chart)
                        line = json.dumps(row)
                    joined.write(line if line.endswith("\n") else line + "\n")
                    rows += 1
    os.replace(tmp, target)
    return rows, dropped


def write_recognition_report(out: str, split: str, task: str, size: int = 1600) -> dict:
    """<split>/recognition/<task>/report.json from every shard's usage record: what training/'s fetch checks
    (samples, target_separator), and how the samples came out. `size` is the page's drawing size, which sets its pen."""
    import seg_prompts as sp
    rp = _recog_pages()
    root = _paths(out, split)["root"]
    totals, pages = Counter(), 0
    for path in sorted(glob.glob(os.path.join(root, "usage", "shard_*.json"))):
        with open(path) as f:
            record = json.load(f)
        pages += record.get("pages", 0)
        totals.update((record.get("recognition") or {}).get(task) or {})
    report = {"task": task, "split": split, "source": "pages", "prompt": rp.rl.PROMPTS[task],
              "target_separator": rp.rl.SEP, "pages": pages, "samples": totals["samples"], "rows": totals["rows"],
              "short_samples": totals["short_samples"], "rows_with_other_lines": totals["rows_with_other_lines"],
              # lines left out for having no recorded times (geometry's letters, written from NIST character images)
              "lines_without_times": totals["lines_without_times"],
              "rows_per_sample": rp.rl.ROWS, "pen": rp.PEN}
    with open(os.path.join(root, "recognition", task, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    return {k: report[k] for k in ("samples", "rows", "short_samples", "rows_with_other_lines")}


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------

def write_report(out: str, split: str, settings: dict = None, dropped: int = 0) -> dict:
    """<split>/report.json from every shard's usage record. `dropped`: flowchart level-1 samples join_manifests took
    out as repeats of a chart's, each one image and one row, taken off the samples and rows the shards wrote."""
    root = _paths(out, split)["root"]
    totals, uses, exempt = Counter(), {cls: Counter() for cls in CLASSES}, Counter()
    shares, sizes = {}, {}
    for path in sorted(glob.glob(os.path.join(root, "usage", "shard_*.json"))):
        with open(path) as f:
            record = json.load(f)
        for key in ("pages", "samples", "rows", "marks", "rejected", "seconds", "geometry_samples", "geometry_skipped"):
            totals[key] += record.get(key, 0)
        totals["max_item_uses_on_a_page"] = max(totals["max_item_uses_on_a_page"],
                                                record.get("max_item_uses_on_a_page", 0))
        for item, n in record["uses"].items():
            cls = item.split(":", 1)[0]
            if cls in uses:
                uses[cls][item] += n
        for item, n in record["exempt"].items():
            exempt[item.split(":", 1)[0]] += n
        shares, sizes = record["cap_share"], record["pool_sizes"]

    classes = {}
    for cls, counter in uses.items():
        total = sum(counter.values())
        top_item, top = counter.most_common(1)[0] if counter else (None, 0)
        classes[cls] = {
            "name": CLASSES[cls], "pool_items": sizes.get(cls), "items_used": len(counter),
            "uses": total, "uses_per_item": round(total / len(counter), 2) if counter else 0,
            "most_used_item": top_item, "most_used_share": round(top / total, 5) if total else 0,
            "cap_share": shares.get(cls), "exempt_uses": exempt.get(cls, 0),
        }
    report = {"split": split, "pages": totals["pages"], "samples": totals["samples"] - dropped,
              "rows": totals["rows"] - dropped, "repeat_flowchart_level1_dropped": dropped, "marks": totals["marks"],
              "geometry_samples": totals["geometry_samples"], "geometry_skipped": totals["geometry_skipped"],
              "rejected_for_undirected_arrows": totals["rejected"],
              "max_item_uses_on_a_page": totals["max_item_uses_on_a_page"],
              "worker_seconds": round(totals["seconds"], 1),
              "seconds_per_page": round(totals["seconds"] / totals["pages"], 3) if totals["pages"] else None,
              "classes": classes, **(settings or {})}
    with open(os.path.join(root, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="split_pools.py's cache root")
    ap.add_argument("--split", required=True, choices=list(MODES))
    ap.add_argument("--pages", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap-share", type=float, default=DEFAULT_CAP_SHARE)
    ap.add_argument("--shard-pages", type=int, default=SHARD_PAGES)
    ap.add_argument("--size", type=int, default=1600, help="page drawing, pixels")
    ap.add_argument("--image", type=int, default=768, help="sample image width and height, pixels")
    ap.add_argument("--tuning", default=None, help="JSON of class_tuning.py overrides, e.g. {\"flowchart_weight\": 30}")
    ap.add_argument("--mark-crop", default="context", choices=("context", "box"),
                    help="context: a mark with its marked row and a margin; box: the mark's own box only")
    ap.add_argument("--recognition", action="store_true",
                    help="train and val: also each page's recognition samples, into <split>/recognition/ "
                         "(recog_synth/recog_pages.py)")
    ap.add_argument("--only-recognition", action="store_true",
                    help="train and val: only the recognition samples; the pages are built but never drawn or written")
    ap.add_argument("--geometry-per-page", type=float, default=0.0,
                    help="train: geometry figures without a page per page, each its own level-1 sample (geometry_sample)")
    ap.add_argument("--only-geometry", action="store_true",
                    help="train: only the geometry figures without a page, into <out>/train/geometry/ with its own "
                         "manifest and report; no page, sample, mark or recognition sample is made")
    args = ap.parse_args()
    if args.only_geometry and (args.split != "train" or not args.geometry_per_page or args.recognition
                               or args.only_recognition):
        ap.error("--only-geometry is for train with --geometry-per-page, and makes nothing else")
    if args.only_recognition and args.split not in RECOGNITION_SPLITS:
        ap.error(f"--only-recognition makes {' and '.join(RECOGNITION_SPLITS)} samples only")
    if args.geometry_per_page and (args.split != "train" or args.geometry_per_page < 0):
        ap.error("--geometry-per-page is for train only, and can't be negative")

    tuning = None
    if args.tuning:
        import class_tuning
        with open(args.tuning) as f:
            tuning = class_tuning.validate(json.load(f))       # refused here, before any worker starts

    shards = [(n, first, min(args.shard_pages, args.pages - first))
              for n, first in enumerate(range(0, args.pages, args.shard_pages))]
    todo = [s for s in shards if not os.path.exists(_paths(args.out, args.split, s[0])["usage"])]
    done_pages = sum(s[2] for s in shards) - sum(s[2] for s in todo)
    print(f"{args.split}: {args.pages} pages in {len(shards)} shards, {len(todo)} to do "
          f"({done_pages} pages already done), {args.workers} workers -> {os.path.join(args.out, args.split)}",
          flush=True)

    initargs = (args.cache, args.split, args.out, args.seed, args.cap_share, args.size, args.image,
                tuning, args.mark_crop, args.recognition, args.only_recognition, args.geometry_per_page,
                args.only_geometry)
    started, pages = time.time(), 0

    def progress(shard, stats, seconds):
        nonlocal pages
        pages += stats.get("pages", 0)
        elapsed = time.time() - started
        left = sum(s[2] for s in todo) - pages
        eta = elapsed / pages * left if pages else 0
        print(f"  shard {shard}: {stats.get('pages', 0)} pages in {seconds:.0f}s | "
              f"{pages + done_pages}/{args.pages} pages, {pages / elapsed:.2f} pages/s, "
              f"~{eta / 60:.0f} min left", flush=True)

    if args.workers <= 1:
        _init(*initargs)
        for task in todo:
            progress(*_run_shard(task))
    elif todo:
        import multiprocessing
        context = multiprocessing.get_context("spawn" if os.name == "nt" else "fork")
        with context.Pool(args.workers, initializer=_init, initargs=initargs) as pool:
            for result in pool.imap_unordered(_run_shard, todo):
                progress(*result)

    paths = _paths(args.out, args.split)
    dropped = 0
    for mode in ("samples", "marks"):
        if mode in MODES[args.split] and not (args.only_recognition or args.only_geometry):
            rows, gone = join_manifests(paths[mode], shards, once_per_chart=mode == "samples" and args.split == "train")
            dropped += gone
            print(f"joined {rows} rows -> {os.path.join(paths[mode], 'manifest.jsonl')}"
                  + (f" ({gone} repeat flowchart level-1 samples dropped)" if gone else ""), flush=True)
    if args.only_geometry:
        # geometry/ is all this run makes: its own report, the split's report.json is not written
        rows, _ = join_manifests(paths["geometry"], shards)
        records = [load_json(_paths(args.out, args.split, s[0])["usage"]) for s in shards]
        report = {"split": args.split, "pages": args.pages, "geometry_per_page": args.geometry_per_page,
                  "samples": rows, "skipped": sum(r.get("geometry_skipped", 0) for r in records),
                  "seed": args.seed, "size": args.size, "image": args.image,
                  "prompt": _W_prompt_geometry()}
        with open(os.path.join(paths["geometry"], "report.json"), "w") as f:
            json.dump(report, f, indent=1)
        print(f"joined {rows} geometry samples -> {os.path.join(paths['geometry'], 'manifest.jsonl')}")
        print(json.dumps(report, indent=1))
        return
    recognition = None
    if (args.recognition or args.only_recognition) and args.split in RECOGNITION_SPLITS:
        recognition = {}
        for kind in _recog_pages().TEXT_CLASSES:
            folder = os.path.join(paths["recognition"], kind)
            rows, _ = join_manifests(folder, shards)
            recognition[kind] = write_recognition_report(args.out, args.split, kind, args.size)
            print(f"joined {rows} {kind} recognition samples -> {os.path.join(folder, 'manifest.jsonl')}", flush=True)
    report = write_report(args.out, args.split, {"tuning": tuning or {}, "mark_crop": args.mark_crop,
                                                  "recognition": recognition, "only_recognition": args.only_recognition,
                                                  "geometry_per_page": args.geometry_per_page},
                          dropped=dropped)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
