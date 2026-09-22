"""synth_api.py — the tracker's dataset (synthesis) tab; ../tracker_server.py serves it, for calls with ?job=synth.

Reads what synth_orchestrator.py writes (run/status.json, run/run_meta.json,
run/picks.jsonl and run/picks/, orchestrator.json) and turns the page's buttons into
request files for it:
  Sample run / Regenerate all   requests/START.json, created exclusively while no run is going,
                                with the run's settings (recognition, class tuning, mark crops)
                                checked by synth_orchestrator.run_options
  Stop                          run/KILL, and `jl pause` on the dataset VM straight away
/api/options gives the page what those settings can be: class_tuning's settings and
their defaults, read from the generator's source. Test-split content is never served:
only its counts.
"""
import json
import os
import subprocess
import threading
import time
import urllib.parse

import synth_orchestrator as so
import tracker_update

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "run")                 # configure() moves this, so.RUN and the rest together
PICKS = os.path.realpath(os.path.join(RUN, "picks"))
TRAIN_RUNS = os.path.join(os.path.dirname(HERE), "train", "runs")
JL_DIR = os.path.join(os.path.dirname(HERE), "jl")                # the Jarvis CLI, shared with train/
JL = os.path.join(JL_DIR, "bin", "jl.exe")
ENV = dict(os.environ, PYTHONPATH=JL_DIR, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
ALIVE_SECONDS = 45
kill_state = {"run_id": None, "requested_at": None, "result": None}


def load(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def orchestrator():
    doc = load(so.HEARTBEAT) or {}
    return {**doc, "alive": time.time() - doc.get("heartbeat_at", 0) < ALIVE_SECONDS}


def training_busy():
    """Why a run can't start now because a training job is using the dataset VM, or None."""
    for job in sorted(os.listdir(TRAIN_RUNS)) if os.path.isdir(TRAIN_RUNS) else []:
        state = load(os.path.join(TRAIN_RUNS, job, "state.json")) or {}
        if time.time() - state.get("heartbeat_at", 0) > ALIVE_SECONDS:
            continue
        stages = {s.get("name"): s.get("state") for s in state.get("stages") or []}
        if state.get("phase") == "provisioning" or (state.get("phase") == "training"
                                                    and stages.get("fetch_dataset") != "done"):
            return f"the {job} training run is copying the dataset from the dataset VM"
    return None


def idle_doc():
    return {"state": "idle", "pause_reason": None, "machine_id": None, "region": so.REGION, "vm": so.VM_DESC,
            "hourly": so.HOURLY, "budget": 0, "cost": 0, "started_at": None, "updated_at": time.time(), "poll": 0,
            "current_stage": None, "overall": {"pages_done": 0, "pages_target": 0, "pct": 0},
            "speed": {"wall_s_per_page": None, "worker_s_per_page": None, "pages_per_min": None, "eta_s": None},
            "stages": [], "splits": {}, "machine": {}}


def status_doc():
    meta = load(os.path.join(RUN, "run_meta.json")) or {}
    if os.path.exists(os.path.join(RUN, "status.json")) and meta:
        doc = tracker_update.build_doc(RUN)
        doc["mode"] = meta.get("mode", "full")
        doc["run_id"] = meta.get("run_id", "first-run")
    else:
        doc = idle_doc()
        doc["mode"], doc["run_id"] = None, None
    if kill_state["run_id"] != doc["run_id"]:
        kill_state.update(run_id=doc["run_id"], requested_at=None, result=None)
    doc.update(kill=kill_state, served_at=time.time(), orchestrator=orchestrator(), busy=training_busy(),
               resumable=resumable() is None,
               modes={k: {"pages": v["pages"], "budget": v["budget"], "vcpus": v["vm"]["vcpus"],
                          "hourly": round(so.vm_hourly(v["vm"]["vcpus"]), 4)}
                      for k, v in so.MODES.items()})
    return doc


def options_doc():
    return {"knobs": so.class_tuning.describe(), "classes": list(so.class_tuning.CLASS_ORDER),
            "max_weight": so.class_tuning.MAX_WEIGHT,
            "recognition": {"tasks": list(so.RECOG_TASKS)},
            "mark_crops": list(so.MARK_CROPS)}


def run_output():
    """The folder the current run writes, as synth_orchestrator.upload_code names it, spelled as it sits in a pick's
    remote paths ("/run/dataset_new/"); None before any run."""
    meta = load(os.path.join(RUN, "run_meta.json")) or {}
    mode, run_id = meta.get("mode"), meta.get("run_id")
    folder = {"full": "run/dataset_new", "recognition": "run/recognition_new", "additions": "run/additions_new"}.get(mode)
    if mode == "sample" and run_id:
        folder = f"regen/{run_id}/dataset"
    return f"/{folder}/" if folder else None


def picks_doc():
    out = []
    path = os.path.join(RUN, "picks.jsonl")
    if not os.path.exists(path):
        return out
    written = run_output()
    for line in open(path):
        if not line.strip():
            continue
        pick = json.loads(line)
        if pick["split"] == "test":         # test content is never shown
            continue
        remote = [p.replace("\\", "/") for p in (pick.get("remote") or {}).values()]
        if not written or not remote or not all(written in p for p in remote):
            continue                        # only what this run wrote: not the dataset it replaces or reads from

        url = lambda p: "/file?job=synth&p=" + urllib.parse.quote(p)
        doc = {"t": pick.get("pulled_at"), "split": pick["split"], "kind": pick["kind"], "page": pick["page"]}
        local = pick["local"]
        if pick["kind"] == "sample" or pick["kind"].startswith("recog_"):
            doc.update(image=url(local["image"]), prompt=pick["prompt"], target=pick["target"], task=pick.get("task"))
        elif pick["kind"] == "page":
            try:
                objects = json.load(open(local["json"]))["objects"]
            except (OSError, ValueError):
                objects = []
            doc.update(image=url(local["image"]), objects=[
                {k: o[k] for k in ("class", "bbox", "position", "points", "start", "end") if k in o} for o in objects])
        else:
            # a run from before marks were merged also has a "mark" image, a class and one shape
            doc.update({part: url(local[part]) for part in ("input", "text", "mark") if part in local},
                       shapes=pick.get("shapes") or [pick.get("shape") or pick.get("class")],
                       labels=pick.get("labels"), text_label=pick.get("text_label"), box=pick.get("box"))
        out.append(doc)
    return out


def start(body):
    mode = body.get("mode")
    if mode not in so.MODES:
        return 400, {"error": f"mode must be one of {', '.join(so.MODES)}"}
    if mode in ("full", "recognition", "additions") and body.get("confirm") is not True:
        return 400, {"error": "regenerating replaces files of the dataset and needs confirming"}
    if mode in ("recognition", "additions"):     # an additions run's pool passes are recognition samples
        body = {**body, "recognition": {**(body.get("recognition") or {}), "on": True}}   # nothing but recognition
    try:
        options = so.run_options(body)
        pages = so.run_pages(mode, body)
    except ValueError as error:
        return 400, {"error": f"settings: {error}"}
    # tracker_server.py has just started the orchestrator, or restarted it onto current code, if it needed to:
    # the request waits in requests/ until it looks
    orch = orchestrator()
    if orch.get("phase") in ("starting", "running"):
        return 409, {"error": f"a {orch.get('mode')} run is already going"}
    busy = training_busy()
    if busy:
        return 409, {"error": f"not now: {busy}"}
    os.makedirs(so.REQUESTS, exist_ok=True)
    try:
        fd = os.open(os.path.join(so.REQUESTS, "START.json"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return 409, {"error": "a run has already been requested"}
    with os.fdopen(fd, "w") as f:
        json.dump({"mode": mode, "pages": pages, "budget": so.run_budget(mode, pages), "requested_at": time.time(),
                   **options}, f)
    return 200, {"ok": True}


def resumable():
    """Why the last run can't be carried on (synth_orchestrator.resume_run), or None when it can: a full run that
    failed, with its VM known."""
    st = load(os.path.join(RUN, "orchestrator_state.json")) or {}
    if st.get("mode") != "full" or st.get("phase") != "failed":
        return "only a failed Regenerate all can be carried on"
    if not st.get("home"):
        return "the run failed before its VM was ready: start it again instead"
    return None


def resume():
    why = resumable()
    if why:
        return 409, {"error": why}
    orch = orchestrator()
    if orch.get("phase") in ("starting", "running"):
        return 409, {"error": f"a {orch.get('mode')} run is already going"}
    busy = training_busy()
    if busy:
        return 409, {"error": f"not now: {busy}"}
    os.makedirs(so.REQUESTS, exist_ok=True)
    try:
        fd = os.open(os.path.join(so.REQUESTS, "RESUME.json"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return 409, {"error": "carrying on has already been requested"}
    with os.fdopen(fd, "w") as f:
        json.dump({"requested_at": time.time()}, f)
    return 200, {"ok": True}


def pause_now(machine_id):
    result = subprocess.run([JL, "pause", str(machine_id), "--yes"], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", env=ENV, timeout=300)
    kill_state["result"] = {"ok": result.returncode == 0, "at": time.time(),
                            "output": (result.stdout + result.stderr)[-300:]}


def kill():
    meta = load(os.path.join(RUN, "run_meta.json")) or {}
    if not meta:
        return 409, {"error": "there is no run to stop"}
    if kill_state["requested_at"] is None:
        kill_state["requested_at"] = time.time()
        open(os.path.join(RUN, "KILL"), "w").write(str(time.time()))
        if meta.get("machine_id"):
            threading.Thread(target=pause_now, args=(meta["machine_id"],), daemon=True).start()
        else:
            kill_state["result"] = {"ok": True, "at": time.time(), "output": "no VM was running yet"}
    return 200, {"ok": True, "kill": kill_state}


def configure(root):
    """Keep every run file under `root` (default: this folder); a local test points it at a scratch folder."""
    global RUN, PICKS
    so.configure(root)
    RUN = so.RUN
    PICKS = os.path.realpath(os.path.join(RUN, "picks"))


def get(path, query):
    """(code, body) or (code, bytes, content type) for a GET; None for a path this tab doesn't serve."""
    if path == "/api/status":
        return 200, status_doc()
    if path == "/api/picks":
        return 200, picks_doc()
    if path == "/api/options":
        return 200, options_doc()
    if path == "/file":
        target = os.path.realpath(query.get("p", ""))
        if (not target.startswith(PICKS + os.sep) or not os.path.isfile(target)
                or target.startswith(os.path.join(PICKS, "test") + os.sep)):
            return 404, {"error": "not found"}
        return 200, open(target, "rb").read(), "image/png" if target.endswith(".png") else "application/json"
    return None


def post(path, body):
    if path == "/api/kill":
        return kill()
    if path == "/api/start":
        return start(body)
    if path == "/api/resume":
        return resume()
    return None
