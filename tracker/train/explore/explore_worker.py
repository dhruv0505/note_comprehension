"""
explore_worker.py — the segmentation model on single validation pages, on demand, on the GPU VM.

    python3 explore_worker.py --poll [--submit '{"id": "p0000012-1", "page": 12}' ...]   # orchestrator.py, over jl exec
    ~/train/venv/bin/python explore_worker.py --serve                                   # the worker, started by --poll

--poll is quick and imports no torch: it queues the requests given, starts the
worker when its heartbeat is stale, and prints one JSON line: the worker's status,
the ids that have a result, whether the page index is written, and the home folder.

--serve loads the base model and the run's best/ (last/ if no validation chose one) once, writes
explore/index.json (every val page's object counts by class and its number of manifest rows), then takes
explore/requests/<id>.json {"id", "page", "model", "sig", "marks", "marks_sig"} one at a time. "model" names an
adapter that orchestrator.py uploaded from note_synth/inference/comprehension/ to explore/models/<model>/ (none: the
run's checkpoint). "marks" names a mark separator it uploaded from note_synth/inference/mark_separation/ to
explore/marks/<marks>/model.pt (none: the one the run was started with, marks/model.pt; NO_MARKS: none at all).
Each is loaded once and switched to per request, and a new signature (its files changed) loads it again:
  cascade   seg_eval.validate_pages on the full val page, exactly as validation runs it, recognition included
            (recog_eval.py), marks taken out of each line with the request's mark separator when there is one:
            each group of detected lines it read is saved as results/<id>_readNN.png
  rows      that page's val manifest rows, each prompt on its own ground-truth image
into explore/results/<id>.json. Only data/val is read; no test page is ever touched.
explore/status.json: state loading | ready | working | failed, with the request and step in hand.

    python explore_worker.py --serve --root DIR --code ../../training --checkpoint none --max-new-tokens 8   # local check
    python explore_worker.py --serve --root DIR --stub       # no model, every output empty: tests the plumbing only
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter

STALE_SECONDS = 30          # a heartbeat older than this means no worker is running
LAUNCH_GAP = 90             # seconds between two launches, so a slow start is not started twice
ROW_PAGE = re.compile(r"p(\d+)_s\d+")
REQUEST_ID = re.compile(r"[\w.~-]{1,128}")
MODEL_NAME = re.compile(r"[\w-][\w.-]{0,39}")      # as inference_models.NAME
NO_MARKS = "~none"                                 # as inference_models.NO_MARKS: no mark separator


def load(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, data):
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:         # Windows (local test): a reader has the file open
            time.sleep(0.05 * (attempt + 1))
    os.replace(tmp, path)


class Folders:
    def __init__(self, root):
        self.root = root
        self.pages = os.path.join(root, "data", "val", "pages")
        self.manifest = os.path.join(root, "data", "val", "samples", "manifest.jsonl")
        self.explore = os.path.join(root, "explore")
        self.requests = os.path.join(self.explore, "requests")
        self.results = os.path.join(self.explore, "results")
        self.status = os.path.join(self.explore, "status.json")
        self.index = os.path.join(self.explore, "index.json")
        self.launch = os.path.join(self.explore, "launch.json")
        self.log = os.path.join(self.explore, "worker.log")
        for folder in (self.requests, self.results):
            os.makedirs(folder, exist_ok=True)


# ----------------------------------------------------------------------------
# --poll
# ----------------------------------------------------------------------------

def poll(args, f):
    for text in args.submit:
        doc = json.loads(text)
        rid, model, marks = str(doc.get("id", "")), doc.get("model"), doc.get("marks")
        if (not REQUEST_ID.fullmatch(rid) or not isinstance(doc.get("page"), int)
                or not (model is None or MODEL_NAME.fullmatch(str(model)))
                or not (marks in (None, NO_MARKS) or MODEL_NAME.fullmatch(str(marks)))):
            continue
        if not os.path.exists(os.path.join(f.results, f"{rid}.json")):
            write_json(os.path.join(f.requests, f"{rid}.json"), {"id": rid, "page": doc["page"], "model": model,
                                                               "sig": doc.get("sig"), "marks": marks,
                                                               "marks_sig": doc.get("marks_sig"), "queued_at": time.time()})
    status = load(f.status) or {}
    alive = time.time() - status.get("heartbeat", 0) < STALE_SECONDS
    launched = False
    if not alive and time.time() - (load(f.launch) or {}).get("at", 0) > LAUNCH_GAP:
        write_json(f.launch, {"at": time.time()})
        command = [os.path.expanduser(args.python), os.path.abspath(__file__), "--serve", "--root", f.root, *args.worker_arg]
        options = {"stdin": subprocess.DEVNULL, "stdout": open(f.log, "a"), "stderr": subprocess.STDOUT, "close_fds": True}
        if os.name == "nt":
            options["creationflags"] = 0x00000008 | 0x00000200      # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        else:
            options["start_new_session"] = True                     # outlives the ssh session that started it
        subprocess.Popen(command, **options)
        launched = True
    finished = sorted(name[:-5] for name in os.listdir(f.results) if name.endswith(".json"))
    print(json.dumps({"now": time.time(), "home": os.path.expanduser("~"), "status": status, "alive": alive,
                      "launched": launched, "finished": finished, "index": os.path.exists(f.index)}))


# ----------------------------------------------------------------------------
# --serve
# ----------------------------------------------------------------------------

class Status:
    """status.json, rewritten on every change and every few seconds as a heartbeat."""

    def __init__(self, path):
        self.path, self.lock = path, threading.Lock()
        self.doc = {"state": "loading", "pid": os.getpid(), "started_at": time.time()}

    def set(self, **fields):
        with self.lock:
            self.doc.update(fields)
            write_json(self.path, {**self.doc, "heartbeat": time.time()})

    def beat(self):
        while True:
            time.sleep(5)
            self.set()


def build_index(f):
    """Writes index.json; returns each page's manifest row offsets."""
    offsets = {}
    with open(f.manifest, "rb") as handle:
        position = 0
        for line in handle:
            if line.strip():
                match = ROW_PAGE.search(json.loads(line)["image"])
                if match:
                    offsets.setdefault(int(match.group(1)), []).append(position)
            position += len(line)
    pages = []
    for name in sorted(os.listdir(f.pages)):
        match = re.fullmatch(r"page(\d+)\.json", name)
        if not match:
            continue
        objects = (load(os.path.join(f.pages, name)) or {}).get("objects") or []
        page = int(match.group(1))
        pages.append({"page": page, "objects": len(objects), "classes": dict(Counter(o["class"] for o in objects)),
                      "rows": len(offsets.get(page, []))})
    write_json(f.index, {"built_at": time.time(), "count": len(pages), "pages": pages})
    return offsets


class Runner:
    def __init__(self, args, f, status):
        self.args, self.f = args, f
        # the eval code orchestrator.py uploads beside this worker (explore/code/) comes before the run's own code/,
        # which is as old as the run and may predate recognition
        for position, folder in enumerate((args.code or os.path.join(f.explore, "code"), os.path.join(f.root, "code"))):
            sys.path.insert(position, os.path.abspath(folder))
        os.environ.setdefault("HF_HOME", os.path.join(f.root, "hf_cache"))
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import seg_eval
        from recog_eval import MarkRemover, Recognizer, model_read_fn
        from seg_dataset import to_loc_target
        self.E, self.to_loc_target, self.Recognizer, self.MarkRemover = seg_eval, to_loc_target, Recognizer, MarkRemover
        # the mark separator in use (remove_marks, and its label for results), the run's own, and saved ones loaded
        self.remove_marks = self.mark_model = self.run_marks = self.run_mark_model = None
        self.mark_removers = {}

        status.set(step="indexing the val pages")
        self.offsets = build_index(f)
        if args.stub:
            self.checkpoint, self.size = "stub", 768
            self.generate = lambda requests: (["</s><s></s>"] * len(requests), [False] * len(requests))
            self.read = lambda requests: (["</s><s></s>"] * len(requests), [False] * len(requests),
                                          [0.0] * len(requests), [0] * len(requests))
            return

        status.set(step="loading the model")
        import torch
        from model import load_base, pick_dtype
        run = os.path.join(f.root, "run")
        # the run's own checkpoint, for requests without a model ('none': the base model alone)
        self.run_checkpoint = args.checkpoint or next((os.path.join(run, name) for name in ("best", "last")
                                                       if os.path.exists(os.path.join(run, name, "adapter_model.safetensors"))), None)
        self.dtype = pick_dtype()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base, processor = load_base(self.dtype)
        # adapters are injected into this model in place, so one generate function serves every adapter
        self.model = base.to(self.device, self.dtype).eval()
        self.peft, self.adapters, self.active, self.checkpoint = None, {}, "unset", None
        self.size = processor.image_processor.size["height"]
        self.generate = seg_eval.model_generate_fn(self.model, processor, self.device, self.dtype, args.max_new_tokens, args.batch_size)
        self.read = model_read_fn(self.model, processor, self.device, self.dtype, args.max_new_tokens, args.batch_size)
        # the mark separation model the run was given (~/train/marks/model.pt), unless --mark-model says otherwise
        mark_model = args.mark_model or os.path.join(f.root, "marks", "model.pt")
        if mark_model.lower() != "none" and os.path.exists(mark_model):
            status.set(step="loading the mark separation model")
            self.run_marks, self.run_mark_model = MarkRemover(mark_model, self.device), mark_model
        self.use_marks(None, None)
        if self.run_checkpoint:
            self.use(None, None)

    def use_marks(self, marks, sig):
        """Makes a request's mark separator the one recognition takes marks out with: None is the run's own, NO_MARKS
        none, otherwise the one in explore/marks/<marks>/. Each (marks, sig) is loaded once."""
        if marks is None:
            self.remove_marks, self.mark_model = self.run_marks, self.run_mark_model
            return
        if marks == NO_MARKS:
            self.remove_marks = self.mark_model = None
            return
        label = f"inference/mark_separation/{marks}"
        if self.args.stub:
            self.remove_marks, self.mark_model = None, label
            return
        key = (marks, sig)
        if key not in self.mark_removers:
            path = os.path.join(self.f.explore, "marks", marks, "model.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"mark separator {marks} has not been uploaded to the VM")
            self.mark_removers[key] = self.MarkRemover(path, self.device)
        self.remove_marks, self.mark_model = self.mark_removers[key], label

    def use(self, model, sig):
        """Makes a request's model the one generating: None is the run's checkpoint, otherwise the adapter in
        explore/models/<model>/. Each (model, sig) is loaded once; switching between loaded ones is instant."""
        if self.args.stub:
            self.checkpoint = f"inference/comprehension/{model}" if model else "stub"
            return
        key = (model, sig)
        if key == self.active:
            return
        if model is None:
            path = self.run_checkpoint
            if path is None:
                raise FileNotFoundError(f"no best/ or last/ checkpoint in {os.path.join(self.f.root, 'run')}")
            label = "base model" if path.lower() == "none" else os.path.basename(path.rstrip("/\\")) + "/"
        else:
            path, label = os.path.join(self.f.explore, "models", model), f"inference/comprehension/{model}"
            if not os.path.exists(os.path.join(path, "adapter_model.safetensors")):
                raise FileNotFoundError(f"model {model} has not been uploaded to the VM")
        if path.lower() == "none":
            if self.peft is not None:
                self.peft.base_model.disable_adapter_layers()
        else:
            if key not in self.adapters:
                from peft import PeftModel
                name = f"adapter{len(self.adapters)}"
                if self.peft is None:
                    self.peft = PeftModel.from_pretrained(self.model, path, adapter_name=name)
                else:
                    self.peft.load_adapter(path, adapter_name=name)
                self.peft.to(self.device, self.dtype).eval()
                self.adapters[key] = name
            self.peft.base_model.enable_adapter_layers()
            self.peft.set_adapter(self.adapters[key])
        self.active, self.checkpoint = key, label

    def run(self, page, step, rid):
        import numpy as np
        from PIL import Image
        E, a = self.E, self.args
        name = f"page{page:07d}"
        image_path = os.path.join(self.f.pages, name + ".png")
        truth_path = os.path.join(self.f.pages, name + ".json")
        if not (os.path.exists(image_path) and os.path.exists(truth_path)):
            raise FileNotFoundError(f"there is no val page {page}")
        with Image.open(image_path) as image:
            grey = np.asarray(image.convert("L"))
        height, width = grey.shape
        truth, text = E.read_page(truth_path, width, height)

        calls = []

        def cascade_generate(requests):
            calls.append(len(requests))
            step("cascade: level 2 and arrows" if len(calls) == 1 else f"cascade: level 1 in {len(requests)} predicted boxes")
            return self.generate(requests)

        started = time.time()
        metrics = E.SegMetrics(a.iou, a.page_vertex_px, a.page_arrow_px, a.page_min_box_px, unit="pages")
        def read(requests):
            step(f"recognition: {len(requests)} groups of detected lines")
            return self.read(requests)

        recognize = self.Recognizer(read, a.iou, a.page_min_box_px, keep_images=True, remove_marks=self.remove_marks)
        summary, samples = E.validate_pages([[{"path": image_path, "image": grey, "truth": truth, "text": text}]],
                                            cascade_generate, metrics, self.size, keep_samples=1, recognize=recognize)
        for k, group in enumerate(samples[0].get("recognition") or []):
            image, group["image"] = group["image"], os.path.join(self.f.results, f"{rid}_read{k:02d}.png")
            image.save(group["image"])
        cascade = {**samples[0], "summary": summary, "seconds": round(time.time() - started, 1)}

        started = time.time()
        rows = []
        with open(self.f.manifest, "rb") as handle:
            for offset in self.offsets.get(page, []):
                handle.seek(offset)
                rows.append(json.loads(handle.readline()))
        folder = os.path.dirname(self.f.manifest)
        row_metrics = E.SegMetrics(a.iou, a.vertex_px, a.arrow_px)
        out = []
        for i in range(0, len(rows), a.batch_size):
            chunk = rows[i:i + a.batch_size]
            step(f"rows: {i} of {len(rows)} done")
            images = []
            for row in chunk:
                with Image.open(os.path.join(folder, row["image"])) as image:
                    images.append(image.convert("L"))
            texts, truncated = self.generate([{"image": im, "prompt": row["prompt"]} for im, row in zip(images, chunk)])
            for row, image, text, cut in zip(chunk, images, texts, truncated):
                target = self.to_loc_target(row["target"])
                parsed = row_metrics.add(row["prompt"], target, text, image.size, cut)
                out.append({"image": os.path.join(folder, row["image"]), "prompt": row["prompt"],
                            "target": E.from_loc_target(target), "prediction": E.from_loc_target(text),
                            "valid": parsed.valid, "errors": [reason for reason, _ in parsed.errors]})

        objects = (load(truth_path) or {}).get("objects") or []
        return {"name": name, "checkpoint": self.checkpoint, "mark_model": self.mark_model,
                "image": image_path, "image_size": [width, height],
                "truth": [{k: o[k] for k in ("class", "bbox", "position", "start", "end") if k in o} for o in objects],
                "cascade": cascade,
                "rows": {"samples": out, "summary": row_metrics.summary(), "seconds": round(time.time() - started, 1)}}


def serve(args, f):
    other = load(f.status) or {}
    if other.get("state") != "failed" and time.time() - other.get("heartbeat", 0) < STALE_SECONDS:
        print(f"a worker (pid {other.get('pid')}) is already running", flush=True)
        return
    status = Status(f.status)
    status.set(state="loading", step="starting")
    threading.Thread(target=status.beat, daemon=True).start()
    try:
        runner = Runner(args, f, status)
    except BaseException as error:
        status.set(state="failed", step=None, error=f"{type(error).__name__}: {error}\n{traceback.format_exc()[-2000:]}")
        raise
    status.set(state="ready", step=None, ready_at=time.time(), checkpoint=runner.checkpoint, error=None)
    print(f"ready with {runner.checkpoint}", flush=True)

    while True:
        pending = sorted(glob.glob(os.path.join(f.requests, "*.json")), key=os.path.getmtime)
        if not pending:
            time.sleep(1)
            continue
        path = pending[0]
        rid = os.path.basename(path)[:-5]
        request = load(path) or {}
        status.set(state="working", id=rid, page=request.get("page"), step="starting")
        started = time.time()
        try:
            if request.get("model"):
                status.set(step=f"loading model {request['model']}")
            runner.use(request.get("model"), request.get("sig"))
            if request.get("marks") not in (None, NO_MARKS):
                status.set(step=f"loading mark separator {request['marks']}")
            runner.use_marks(request.get("marks"), request.get("marks_sig"))
            status.set(checkpoint=runner.checkpoint)
            result = runner.run(int(request["page"]), lambda step: status.set(step=step), rid)
        except Exception as error:      # noqa: BLE001 - reported in the result, the worker carries on
            result = {"error": f"{type(error).__name__}: {error}\n{traceback.format_exc()[-2000:]}"}
            print(result["error"], flush=True)
        result.update(id=rid, page=request.get("page"), model=request.get("model"), marks=request.get("marks"),
                      seconds=round(time.time() - started, 1), finished_at=time.time())
        write_json(os.path.join(f.results, f"{rid}.json"), result)
        os.remove(path)
        status.set(state="ready", id=None, page=None, step=None, last_id=rid, last_seconds=result["seconds"])
        print(f"{rid}: page {request.get('page')} in {result['seconds']}s{' (error)' if 'error' in result else ''}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--poll", action="store_true")
    mode.add_argument("--serve", action="store_true")
    ap.add_argument("--root", default=os.path.join(os.path.expanduser("~"), "train"))
    ap.add_argument("--submit", action="append", default=[], help="poll: a request {\"id\", \"page\"} to queue")
    ap.add_argument("--python", default=sys.executable, help="poll: the python that runs --serve")
    ap.add_argument("--worker-arg", action="append", default=[], help="poll: an argument passed on to --serve")
    ap.add_argument("--code", default=None, help="the training code (default <root>/code)")
    ap.add_argument("--checkpoint", default=None, help="the checkpoint for requests without a model: an adapter folder, or 'none' for the base model (default run/best)")
    ap.add_argument("--stub", action="store_true", help="no model; every output empty (tests the plumbing)")
    ap.add_argument("--mark-model", default=None,
                    help="the mark separation model recognition takes marks out with (default <root>/marks/model.pt when "
                         "the run has one; 'none' reads lines with their marks)")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--vertex-px", type=float, default=10.0)
    ap.add_argument("--arrow-px", type=float, default=15.0)
    ap.add_argument("--page-vertex-px", type=float, default=20.0)
    ap.add_argument("--page-arrow-px", type=float, default=30.0)
    ap.add_argument("--page-min-box-px", type=float, default=4.0)
    args = ap.parse_args()
    folders = Folders(os.path.abspath(args.root))
    (poll if args.poll else serve)(args, folders)


if __name__ == "__main__":
    main()
