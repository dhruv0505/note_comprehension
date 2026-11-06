"""
comprehension_server.py — the comprehension cascade, live, for the whiteboard (whiteboard_interface).

    python live/comprehension_server.py                                   # the newest model in inference/comprehension/
    python live/comprehension_server.py --model comprehension-v3          # a saved model by name, or an adapter folder
    python live/comprehension_server.py --model comprehension-v3 --seg-model seg-v2   # a second adapter for seg alone
    python live/comprehension_server.py --stub                            # no model: canned outputs, to test the wiring

The whiteboard posts its strokes (world units, each point's time in seconds when it has one) every few seconds and
gets back what the models found, in the same world units. Nothing here is a new way of reading a page: the page is
drawn and read by the training code itself, as seg_eval.py's cascade and recog_eval.py's recognition read a
validation page (README, "How the numbers are computed"):

  Page.     The strokes' bounding box is placed on a square board BOARD_SIZE units wide (note_synth's 1000-unit
            board), its top-left corner MARGIN in from the board's, at a scale that makes the board at least
            --min-page world units across, so handwriting of an ordinary size lands near the line heights pages are
            synthesized with (layout.LINE_PX_RANGE); more writing than fits makes the board bigger, never smaller.
            It is drawn by note_synth's page_export.render_page at PAGE_PX pixels with the training pen
            (seg_prompts.INK_WIDTH_PX), black on white: the pen the whiteboard drew with is not the model's.
  Cascade.  seg_eval.validate_pages on that page: 768 x 768 for the level-2 objects and the arrows, then every
            predicted textblock, table, geometry and flowchart box cut from the full page, its longer side scaled to
            768, and asked its level-1 prompt. Every prediction is mapped back to page pixels, then here to world units.
  Reading.  ("seg_recog" only) recog_eval.Recognizer: the detected textlines, then formulas, ten at a time, each box
            taking the strokes with at least half their points inside it, the mark separation network (--marks)
            taking underlines, strikethroughs and boxes out first, drawn by recog_pages.group_image (coloured by time
            and pen movement) and read with recog_layout.PROMPTS. Each level-2 drawing is read as its category too,
            drawn as recog_layout.drawing_sample draws one. Points without a time (strokes saved before the
            whiteboard kept times) are timed as recog_pages.row_lines times a mark's: never made up, in drawing order.

Every generation is greedy, so an unchanged image asked an unchanged prompt gives the same output: outputs are
cached by (adapter, prompt, image), and a stroke added to one block costs that block's level-1 prompt and the
two whole-page prompts, not the whole cascade again.

HTTP (one request at a time; the models are not shared between threads):
  GET  /health   {"ready", "model", "seg_model", "marks", "device", "dtype", "stub", "error"}
  POST /infer    {"mode": "seg" | "seg_recog", "strokes": [{"id", "points": [[x, y, t or null], ...]}]}
              -> {"objects": [{"class", "kind", "level", "coords": [...world...], "text"?}], "page": {...},
                  "timings": {...}, "mode"}
     kind: box (x0 y0 x1 y1), point (x y), segment (x1 y1 x2 y2), arc (start, middle, end), arrow (start, end).
"""

from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
import traceback
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # note_comprehension/
for folder in ("training", os.path.join("training", "marks_cnn"), "recog_synth", "note_synth"):
    sys.path.insert(0, os.path.join(ROOT, folder))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "training", "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np                                      # noqa: E402

import page_export                                      # noqa: E402  note_synth: how a page is drawn
import recog_layout as rl                               # noqa: E402  recog_synth: how a recognition sample is drawn
import seg_eval as E                                    # noqa: E402  training: the cascade
from recog_eval import MarkRemover, Recognizer, read_rows, to_ink as recog_ink   # noqa: E402
from recog_pages import PageInk, row_lines              # noqa: E402

INFERENCE = os.path.join(ROOT, "inference")
BOARD_SIZE = 1000.0                 # note_synth's board (layout.PAGE_W): every generator is tuned in these units
MARGIN = 40.0                       # layout.MARGIN: the board's blank border
PAGE_PX = page_export.PAGE_PX       # 1600: the page the cascade cuts its level-1 crops from
INK_WIDTH_PX = 2.0                  # seg_prompts.INK_WIDTH_PX: the pen, in board units
DRAWING_PEN = sum(rl.DRAWING_PEN) / 2.0     # a drawing's pen: the middle of the range its samples were drawn with
IMAGE_SIZE = 768                    # every image a prompt is asked of


# ----------------------------------------------------------------------------
# The page
# ----------------------------------------------------------------------------

class _Stroke:
    """A stroke as page_export.render_page and recog_layout.polylines read one: .x, .y, .t."""

    def __init__(self, x, y, t=None):
        self.x, self.y, self.t = x, y, t


class _Block:
    def __init__(self, strokes):
        self.strokes = strokes

    def all_strokes(self):
        return self.strokes


class Page:
    """The whiteboard's strokes placed on the board and drawn: world units <-> page pixels."""

    def __init__(self, strokes, min_page):
        # strokes: [(n, 3) arrays of world x, y and t (NaN: no time)]
        points = np.concatenate([s[:, :2] for s in strokes])
        lo, hi = points.min(axis=0), points.max(axis=0)
        span = float(max(hi[0] - lo[0], hi[1] - lo[1], 1e-6))
        usable = BOARD_SIZE - 2 * MARGIN
        side = max(min_page, span * BOARD_SIZE / usable)        # world units across the whole board
        self.board_per_world = BOARD_SIZE / side
        self.origin = lo - MARGIN / self.board_per_world        # world point at the board's top-left
        self.px_per_world = self.board_per_world * PAGE_PX / BOARD_SIZE
        self.world_side = side
        board = [self.to_board(s) for s in strokes]
        image = page_export.render_page([_Block([_Stroke(b[:, 0], b[:, 1]) for b in board])],
                                        PAGE_PX, BOARD_SIZE, INK_WIDTH_PX)
        self.image = image                                      # PIL "L", PAGE_PX square, black ink on white
        self.grey = np.asarray(image)
        # the strokes in page pixels, times kept, for recognition
        self.strokes = [np.column_stack([(s[:, :2] - self.origin) * self.px_per_world, s[:, 2]]) for s in strokes]

    def to_board(self, stroke):
        return (stroke[:, :2] - self.origin) * self.board_per_world

    def to_world(self, coords):
        """Page-pixel coordinates (x, y, x, y, ...) in world units."""
        return [float(v / self.px_per_world + self.origin[i % 2]) for i, v in enumerate(coords)]

    def describe(self):
        return {"origin": [float(v) for v in self.origin], "world_side": self.world_side,
                "px_per_world": self.px_per_world, "page_px": PAGE_PX}


# ----------------------------------------------------------------------------
# The models
# ----------------------------------------------------------------------------

def find_model(name):
    """An adapter folder: a path, or a name in inference/comprehension/; with no name, the newest one there."""
    if name and os.path.isdir(name):
        return os.path.abspath(name)
    folder = os.path.join(INFERENCE, "comprehension")
    if name:
        path = os.path.join(folder, name)
        if not os.path.isfile(os.path.join(path, "adapter_model.safetensors")):
            raise FileNotFoundError(f"no adapter at {path} (adapter_config.json and adapter_model.safetensors)")
        return path
    found = [os.path.join(folder, n) for n in (os.listdir(folder) if os.path.isdir(folder) else [])
             if os.path.isfile(os.path.join(folder, n, "adapter_model.safetensors"))]
    if not found:
        raise FileNotFoundError(f"no comprehension model in {folder}: save one there from the tracker, or give --model")
    return max(found, key=lambda p: os.path.getmtime(os.path.join(p, "adapter_model.safetensors")))


class Cache:
    """Generated outputs by (adapter, prompt, image bytes), least recently used dropped past `size`."""

    def __init__(self, size):
        self.size, self.items, self.hits, self.misses = size, OrderedDict(), 0, 0

    @staticmethod
    def key(adapter, request):
        image = request["image"]
        digest = hashlib.sha1(image.tobytes()).hexdigest()
        return adapter, request["prompt"], image.mode, image.size, digest

    def run(self, adapter, requests, generate):
        """generate(requests) -> (texts, truncated), called only for the requests not cached."""
        keys = [self.key(adapter, r) for r in requests]
        missing = [i for i, k in enumerate(keys) if k not in self.items]
        self.hits += len(requests) - len(missing)
        self.misses += len(missing)
        if missing:
            texts, cut = generate([requests[i] for i in missing])
            for i, text, c in zip(missing, texts, cut):
                self.items[keys[i]] = (text, c)
        out = []
        for k in keys:
            self.items.move_to_end(k)
            out.append(self.items[k])
        while len(self.items) > self.size:
            self.items.popitem(last=False)
        return [t for t, _ in out], [c for _, c in out]


class Models:
    def __init__(self, args):
        self.args, self.error, self.ready = args, None, False
        self.model = self.seg_model = self.marks = None
        self.device = self.dtype = None
        self.cache = Cache(args.cache)
        self.remove_marks = None

    def load(self):
        a = self.args
        try:
            if a.stub:
                self.model, self.device, self.dtype = "stub", "cpu", "none"
                self.generate = self.read = None
                self.ready = True
                return
            import torch
            from model import load_base, pick_dtype
            if a.threads:
                torch.set_num_threads(a.threads)
            main = find_model(a.model)
            seg = find_model(a.seg_model) if a.seg_model else None
            self.dtype = pick_dtype()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"loading Florence-2-large on {self.device} ({self.dtype}) ...", flush=True)
            base, processor = load_base(self.dtype)
            self.base = base.to(self.device, self.dtype).eval()
            from peft import PeftModel
            print(f"  adapter main: {main}", flush=True)
            self.peft = PeftModel.from_pretrained(self.base, main, adapter_name="main")
            if seg:
                print(f"  adapter seg:  {seg}", flush=True)
                self.peft.load_adapter(seg, adapter_name="seg")
            self.peft.to(self.device, self.dtype).eval()
            self.model, self.seg_model = main, seg
            # adapters are injected into the base in place: these run whichever one is set
            self.generate = E.model_generate_fn(self.base, processor, self.device, self.dtype, a.max_new_tokens,
                                                a.batch_size)
            self.processor = processor
            marks = a.marks or os.path.join(INFERENCE, "mark_separation", "marks", "model.pt")
            if marks.lower() != "none" and os.path.exists(marks):
                print(f"  mark separator: {marks}", flush=True)
                self.remove_marks = MarkRemover(marks, self.device)
                self.marks = marks
            self.ready = True
            print("ready", flush=True)
        except Exception as error:      # reported on /health; the server stays up to say why
            self.error = f"{type(error).__name__}: {error}"
            traceback.print_exc()

    def use(self, adapter):
        if self.model != "stub":
            self.peft.set_adapter(adapter)

    def generate_fn(self, adapter):
        def run(requests):
            if self.model == "stub":
                return self.cache.run(adapter, requests, stub_generate)
            return self.cache.run(adapter, requests, self.generate)
        return run

    def read_fn(self, adapter):
        """read_fn for recog_eval.Recognizer: the text only. The teacher-forced loss it can also give is of the
        ground truth's transcriptions, which a whiteboard has none of."""
        def run(requests):
            batch = [{"image": r["image"], "prompt": r["prompt"]} for r in requests]
            if self.model == "stub":
                texts, cut = self.cache.run(adapter, batch, stub_read)
            else:
                texts, cut = self.cache.run(adapter, batch, self._read)
            return texts, cut, [0.0] * len(texts), [0] * len(texts)
        return run

    def _read(self, requests):
        """Recognition images are RGB, coloured by time and pen movement: read as recog_eval.model_read_fn reads
        them (recog_eval.to_ink, each channel 1 - value / 255), generated greedily as everything else is."""
        from seg_dataset import Collator
        collator = Collator()
        collator.processor = self.processor
        texts, cut = [], []
        size = self.args.batch_size
        for i in range(0, len(requests), size):
            chunk = requests[i:i + size]
            batch = collator([(recog_ink(r["image"]), r["prompt"], "", "") for r in chunk])
            t, c = E.generate(self.base, batch, self.processor.tokenizer, self.device, self.dtype,
                              self.args.max_new_tokens)
            texts += t
            cut += c
        return texts, cut


def stub_generate(requests):
    """--stub: a textblock over the ink's middle, an arrow, a textline across each crop's middle."""
    out = []
    for r in requests:
        if r["stage"] == "level2":
            out.append("</s><s><loc_100><loc_100><loc_900><loc_900>textblock</s>")
        elif r["stage"] == "arrow":
            out.append("</s><s><loc_100><loc_950><loc_900><loc_950>arrow</s>")
        else:
            out.append("</s><s><loc_50><loc_400><loc_950><loc_600>textline</s>")
    return out, [False] * len(out)


def stub_read(requests):
    return ["</s><s>" + ("a cat" if "drawn" in r["prompt"] else "stub reading") + "</s>" for r in requests], \
        [False] * len(requests)


# ----------------------------------------------------------------------------
# One request
# ----------------------------------------------------------------------------

def parse_strokes(raw):
    strokes = []
    for stroke in raw:
        points = [(p[0], p[1], p[2] if len(p) > 2 and p[2] is not None else math.nan) for p in stroke["points"]]
        if points:
            strokes.append(np.asarray(points, dtype=np.float64))
    return strokes


def read_drawings(models, adapter, page, preds):
    """Each predicted level-2 drawing read as its category: the strokes with at least half their points inside its
    box (recog_pages.PageInk), timed as recog_pages.row_lines times a row, drawn by recog_layout.drawing_sample."""
    boxes = [coords for name, coords in preds if name == "drawing"]
    ink = PageInk([{"strokes": page.strokes}])
    requests, placed = [], []
    for k, box in enumerate(boxes):
        strokes = ink.inside(box)
        if not strokes:
            continue
        lines = row_lines(strokes)
        image, _ = rl.drawing_sample([_Stroke(s[:, 0], s[:, 1], s[:, 2]) for s in lines], DRAWING_PEN)
        requests.append({"image": image, "prompt": rl.PROMPTS["drawing"]})
        placed.append(k)
    labels = [None] * len(boxes)
    if requests:
        texts, _, _, _ = models.read_fn(adapter)(requests)
        for k, text in zip(placed, texts):
            labels[k] = read_rows(text)[0] or None
    return labels


def infer(models, body):
    mode = body.get("mode", "seg")
    if mode not in ("seg", "seg_recog"):
        raise ValueError(f"mode {mode!r}: seg or seg_recog")
    strokes = parse_strokes(body.get("strokes") or [])
    if not strokes:
        return {"mode": mode, "objects": [], "page": None, "timings": {}}
    a = models.args
    started = time.time()
    page = Page(strokes, a.min_page)
    timings = {"render": time.time() - started}

    # recognition reads with the main model; seg alone with --seg-model when there is one
    adapter = "seg" if mode == "seg" and models.seg_model else "main"
    models.use(adapter)
    recognize = None
    if mode == "seg_recog":
        recognize = Recognizer(models.read_fn(adapter), a.iou, a.page_min_box_px, remove_marks=models.remove_marks)
    # the whole page's ink as one object of no class: every box reads the strokes it covers, and there is no ground
    # truth to match them to. write_order moves a timed object's times as one section, which leaves them as written.
    text = [{"class": "ink", "bbox": (0.0, 0.0, float(PAGE_PX), float(PAGE_PX)), "text": None,
             "strokes": page.strokes, "parent": None}]
    metrics = E.SegMetrics(a.iou, a.page_vertex_px, a.page_arrow_px, a.page_min_box_px, unit="pages")
    hits = models.cache.hits
    t0 = time.time()
    _, samples = E.validate_pages([[{"path": "whiteboard", "image": page.grey, "truth": [], "text": text}]],
                                  models.generate_fn(adapter), metrics, IMAGE_SIZE, keep_samples=1,
                                  summarize=False, recognize=recognize)
    sample = samples[0]
    timings["cascade_and_reading"] = time.time() - t0
    preds = [(name, tuple(coords)) for name, coords in sample["predicted"]]

    # recognition's reads, back on their boxes: each task's boxes in the order they were predicted, ten a group
    texts = {}
    if mode == "seg_recog":
        for group_index, group in enumerate(sample.get("recognition") or []):
            order = [i for i, (name, _) in enumerate(preds) if name == group["task"]]
            first = sum(len(g["boxes"]) for g in (sample["recognition"][:group_index]) if g["task"] == group["task"])
            for k in range(len(group["boxes"])):
                said = group["predictions"][k] if k < len(group["predictions"]) else ""
                texts[order[first + k]] = said
        t0 = time.time()
        drawing_labels = read_drawings(models, adapter, page, preds)
        timings["drawings"] = time.time() - t0
        drawings = [i for i, (name, _) in enumerate(preds) if name == "drawing"]
        for i, label in zip(drawings, drawing_labels):
            if label:
                texts[i] = label

    objects = []
    for i, (name, coords) in enumerate(preds):
        kind = E.KIND[name]
        level = 2 if name in E.LEVEL2_CLASSES else 0 if name == "arrow" else 1
        obj = {"class": name, "kind": kind, "level": level, "coords": page.to_world(coords)}
        if i in texts:
            obj["text"] = texts[i]
        objects.append(obj)
    timings["total"] = time.time() - started
    stages = {o["stage"]: o["valid"] for o in sample["outputs"] if o["stage"] != "level1"}
    return {"mode": mode, "objects": objects, "page": page.describe(), "timings": timings,
            "valid": stages, "cache_hits": models.cache.hits - hits}


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------

def make_handler(models, lock):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, doc):
            data = json.dumps(doc).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") != "/health":
                return self._send(404, {"error": "not found"})
            self._send(200, {"ready": models.ready, "error": models.error, "model": models.model,
                             "seg_model": models.seg_model, "marks": models.marks, "device": str(models.device),
                             "dtype": str(models.dtype), "stub": models.args.stub})

        def do_POST(self):
            if self.path.rstrip("/") != "/infer":
                return self._send(404, {"error": "not found"})
            if not models.ready:
                return self._send(503, {"error": models.error or "the models are still loading"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                with lock:
                    out = infer(models, body)
                self._send(200, out)
            except Exception as error:
                traceback.print_exc()
                self._send(500, {"error": f"{type(error).__name__}: {error}"})

        def log_message(self, fmt, *args):
            if self.path.rstrip("/") != "/health":
                sys.stderr.write("%s %s\n" % (self.log_date_time_string(), fmt % args))

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--model", default=os.environ.get("COMPREHENSION_MODEL"),
                    help="segmentation + recognition adapter: a name in inference/comprehension/ or a folder "
                         "(default: the newest saved there)")
    ap.add_argument("--seg-model", default=os.environ.get("COMPREHENSION_SEG_MODEL"),
                    help="a second adapter used for seg mode alone (default: --model for both)")
    ap.add_argument("--marks", default=None, help="mark separation model.pt, or 'none' "
                                                  "(default: inference/mark_separation/marks/model.pt)")
    ap.add_argument("--min-page", type=float, default=2400.0,
                    help="world units across the smallest board the strokes are placed on: sets how big the "
                         "whiteboard's handwriting is to the model (default 2400: a 50-unit line is ~20 of 1000)")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=4, help="prompts generated at once")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (default: torch's own)")
    ap.add_argument("--cache", type=int, default=512, help="generated outputs kept")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--page-vertex-px", type=float, default=20.0)
    ap.add_argument("--page-arrow-px", type=float, default=30.0)
    ap.add_argument("--page-min-box-px", type=float, default=4.0)
    ap.add_argument("--stub", action="store_true", help="no model: canned outputs through the real cascade code")
    args = ap.parse_args()

    models = Models(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(models, threading.Lock()))
    print(f"comprehension server on http://{args.host}:{args.port} (loading the models)", flush=True)
    threading.Thread(target=models.load, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
