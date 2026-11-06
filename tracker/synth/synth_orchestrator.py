"""
synth_orchestrator.py — the local side of a dataset regeneration on the dataset VM.

    python synth_orchestrator.py            # waits for "Sample run" or "Regenerate all" in the synthesis tracker

It needs nobody in the loop. tracker_server.py only writes requests/START.json
{"mode": "sample" | "full", "budget"}; this process then
  1. moves the previous run/ into runs_archive/ and starts a fresh run/
  2. resumes the dataset VM, found by name (Jarvis may give it a new id on resume)
  3. uploads the generator (note_synth/*.py) and remote/ as one small tarball to
     ~/regen/<run id>/ and starts regen_remote.sh there. It synthesizes from the split
     caches the first run built (~/run/cache), so nothing large is uploaded
  4. every --interval seconds runs status.py, copies new picks (up to 20 pages per
     split, never test) into run/picks/ as one tar, and writes run/status.json
  5. pauses the VM when the run finishes, a stage fails, the VM stops answering for
     5 polls, the cost reaches the budget, or run/KILL appears

Modes (MODES): sample = MODES' train + val pages into ~/regen/<run id>/dataset, with
~/run/dataset untouched; full = the train, val and test pages chosen in the tracker (default
30,000 + 2,000 + 2,000; run_pages) into ~/run/dataset_new, which replaces ~/run/dataset only
once every split is done. A full run's budget grows with its pages past the default (run_budget).

The dataset VM is resumed at the mode's size: a full run at 32 vCPUs / 128 GB, the
largest CPU VM Jarvis offers (synthesize.py runs a worker per vCPU, so about twice as
fast as 16), and a sample run back at 16 / 64. If the size is not to be had, the VM is
resumed at the size it last had, and the cost is counted at the size it actually got.

A request also carries the run's settings (run_options), chosen in the tracker:
  recognition   on or off: textline and formula samples from every train and val page's lines,
                read as the cascade reads a page (recog_synth/recog_pages.py), made by
                synthesize.py --recognition as it builds the pages, into <split>/recognition/<task>/.
                Test gets none: its recognition is only scored through the cascade
  tuning        class_tuning.py overrides: how often each class appears on a page
  mark_crop     "box" (each mark's own box, overlapping marks joined) or "context" (as before)
  recognition.pool_epochs
                with recognition on, passes over train's whole IAM-OnDB textline pool
                (recog_synthesize.py --subdir recognition_pool), lines no page placed included; 0 skips it
  geometry_per_page
                train geometry figures drawn without a page, per page, each its own level-1 sample
                (synthesize.py --geometry-per-page); val and test are never given any
"""
import argparse
import importlib.util
import io
import json
import os
import shlex
import shutil
import socket
import sys
import tarfile
import tempfile
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
TRACKER = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TRACKER, "train"))
from jlcmd import JL, JLError  # noqa: E402 - the Jarvis CLI wrapper the trainer uses

NOTE_SYNTH = os.path.join(os.path.dirname(TRACKER), "note_synth")
RECOG_SYNTH = os.path.join(os.path.dirname(TRACKER), "recog_synth")
RUN = os.path.join(HERE, "run")
ARCHIVE = os.path.join(HERE, "runs_archive")
REQUESTS = os.path.join(HERE, "requests")
REMOTE = os.path.join(HERE, "remote")
PACK = os.path.join(TRACKER, "train", "remote", "pack.py")
HEARTBEAT = os.path.join(HERE, "orchestrator.json")
LAST_FULL = os.path.join(HERE, "last_full_status.json")     # what the training tracker reads while a sample run is current
EVENTS = os.path.join(HERE, "orchestrator_events.log")
DATA_VM_NAME = "note-synth-dataset"
REGION = "IN2"
VM_DESC = "16 vCPU / 64 GB / 100 GB SSD"
HOURLY = 0.4105                 # the dataset VM with its 100 GB disk
STORAGE_HOURLY = 0.0137         # 100 GB
CPU_VM_HOURLY = {2: 0.0496, 4: 0.0992, 8: 0.1984, 16: 0.3968, 32: 0.7936}     # Jarvis CPU VMs (jl resources), by vCPUs
PICKS_PER_SPLIT = 20
RECOG_PICKS_PER_TASK = 6            # recognition samples per task per split copied here to look at (never test)
BUDGET_MARGIN = 0.10
MODES = {
    "sample": {"pages": {"train": 20, "val": 20}, "shard_pages": 5, "budget": 1.0, "vm": {"vcpus": 16, "ram": 64}},
    "full": {"pages": {"train": 30000, "val": 2000, "test": 2000}, "shard_pages": 100, "budget": 4.0,
             "vm": {"vcpus": 32, "ram": 128}},
    # only the train and val pages' recognition samples, the pages rebuilt in memory and never drawn
    # (synthesize.py --only-recognition), swapped into ~/run/dataset/<split>/recognition when both are done;
    # the dataset's pages, samples and mark crops are left as they are
    "recognition": {"pages": {"train": 30000, "val": 2000}, "shard_pages": 100, "budget": 4.0,
                    "vm": {"vcpus": 32, "ram": 128}},
    # train only, what the dataset lacks, added beside it: geometry figures without a page (the train pages' count x
    # geometry_per_page) and the textline pool passes. No page is drawn; val and test are never read or written
    "additions": {"pages": {"train": 30000}, "shard_pages": 100, "budget": 2.5, "vm": {"vcpus": 32, "ram": 128}},
}
RECOG_TASKS = ("textline", "formula")
MARK_CROPS = ("box", "context")
POOL_EPOCHS_MAX = 10
GEOMETRY_PER_PAGE_MAX = 4
# The paused dataset VM's disk lives on one Jarvis server, which can have no free cores for a while ("No contiguous
# block of 16 cores available on server N9001"). When the mode's size can't be had, these are tried after it, in
# turn (None: the size the VM last had), round after round for up to RESUME_WAIT seconds.
RESUME_SIZES = ({"vcpus": 16, "ram": 64}, {"vcpus": 8, "ram": 32}, None, {"vcpus": 4, "ram": 16}, {"vcpus": 2, "ram": 8})
# requests/UPSIZE.json: a running full run is paused to be carried on (RESUME.json) at a bigger size, when Jarvis has
# one free again; the stages done and the shards written are kept, so only the shards under way are made again
UPSIZE_REASON = "upsizing: paused to carry on with more vCPUs"
RESUME_WAIT = 45 * 60
RESUME_RETRY = 120


def vm_hourly(vcpus):
    return CPU_VM_HOURLY.get(vcpus, 0.0248 * vcpus) + STORAGE_HOURLY


def _load_class_tuning():
    """note_synth/class_tuning.py, loaded by path: it reads the generator's source, so nothing
    heavy is imported, and note_synth is neither put on the path nor given a __pycache__."""
    before, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec = importlib.util.spec_from_file_location("class_tuning", os.path.join(NOTE_SYNTH, "class_tuning.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.dont_write_bytecode = before


class_tuning = _load_class_tuning()


def run_options(request):
    """A run request's settings, checked: {recognition: {on, tasks, pool_epochs}, tuning, mark_crop, geometry_per_page}.
    Raises ValueError. A request without them gets recognition off, no tuning, box crops, no pool passes and no
    geometry figures without a page."""
    rec = request.get("recognition") or {}
    if not isinstance(rec, dict):
        raise ValueError("recognition must be {on}")
    on = rec.get("on") is True
    crop = request.get("mark_crop") or "box"
    if crop not in MARK_CROPS:
        raise ValueError(f"mark_crop must be one of {MARK_CROPS}")
    pool = rec.get("pool_epochs", 0)
    if isinstance(pool, bool) or not isinstance(pool, int) or not 0 <= pool <= POOL_EPOCHS_MAX:
        raise ValueError(f"recognition.pool_epochs must be a whole number from 0 to {POOL_EPOCHS_MAX}")
    geometry = request.get("geometry_per_page", 0)
    if isinstance(geometry, bool) or not isinstance(geometry, (int, float)) or not 0 <= geometry <= GEOMETRY_PER_PAGE_MAX:
        raise ValueError(f"geometry_per_page must be a number from 0 to {GEOMETRY_PER_PAGE_MAX}")
    return {"recognition": {"on": on, "tasks": list(RECOG_TASKS) if on else [], "pool_epochs": pool if on else 0},
            "tuning": class_tuning.validate(request.get("tuning")), "mark_crop": crop,
            "geometry_per_page": float(geometry)}


FULL_PAGE_LIMITS = {"train": 200_000, "val": 20_000, "test": 20_000}     # at least 1 page each: training reads all three


def run_pages(mode, request):
    """The pages per split a run makes: a full run's from the request ({"pages": {"train", "val", "test"}}, each
    split left out taking MODES' count), a sample run's always MODES'. Raises ValueError."""
    pages = dict(MODES[mode]["pages"])
    if mode != "full" or not request.get("pages"):
        return pages
    given = request["pages"]
    if not isinstance(given, dict) or set(given) - set(pages):
        raise ValueError(f"pages must be a count for each of {', '.join(pages)}")
    for split, value in given.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value) \
                or not 1 <= value <= FULL_PAGE_LIMITS[split]:
            raise ValueError(f"{split} pages must be a whole number from 1 to {FULL_PAGE_LIMITS[split]:,}")
        pages[split] = int(value)
    return pages


def run_budget(mode, pages):
    """MODES' budget, grown in proportion when a full run makes more pages than MODES' count."""
    default = MODES[mode]
    return round(default["budget"] * max(1.0, sum(pages.values()) / sum(default["pages"].values())), 2)
LOCAL_STAGES = ["resume_vm", "upload_code"]
GUARD_PORT = 8796


def configure(root):
    """Keep every run file under `root` (default: this folder); a local test points it at a scratch folder."""
    global RUN, ARCHIVE, REQUESTS, HEARTBEAT, LAST_FULL, EVENTS
    RUN, ARCHIVE, REQUESTS = (os.path.join(root, name) for name in ("run", "runs_archive", "requests"))
    HEARTBEAT = os.path.join(root, "orchestrator.json")
    LAST_FULL = os.path.join(root, "last_full_status.json")
    EVENTS = os.path.join(root, "orchestrator_events.log")


def load(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, data):
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:         # a reader has the file open
            time.sleep(0.1)
    os.replace(tmp, path)


class Stopped(Exception):
    pass


class Regen:
    def __init__(self, args):
        self.args, self.jl = args, JL()
        self.state = load(os.path.join(RUN, "orchestrator_state.json"))
        self.message, self.error, self.last_remote = "", None, None
        threading.Thread(target=self.beat, daemon=True).start()

    # --- bookkeeping ------------------------------------------------------------------
    def beat(self):
        while True:
            st = self.state or {}
            try:
                write_json(HEARTBEAT, {"heartbeat_at": time.time(), "pid": os.getpid(), "phase": st.get("phase", "idle"),
                                       "mode": st.get("mode"), "run_id": st.get("run_id"), "message": self.message,
                                       "error": self.error,
                                       # tells tracker_server a request's settings will be used, not dropped
                                       "accepts_settings": True})
            except OSError:
                pass
            time.sleep(5)

    def say(self, text):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {text}"
        print(line, flush=True)
        with open(EVENTS, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        self.message = text

    def save(self):
        write_json(os.path.join(RUN, "orchestrator_state.json"), self.state)

    def cost(self):
        st = self.state or {}
        if not st.get("billing_started"):
            return 0.0
        return max(0.0, (st.get("billing_ended") or time.time()) - st["billing_started"]) / 3600 * st.get("hourly", HOURLY)

    def publish(self):
        """run/status.json in the record format tracker_update.build_doc reads."""
        st, now = self.state, time.time()
        stages = [dict(s) for s in st["local_stages"]]
        if self.last_remote:
            status = {**self.last_remote, "stages": stages + self.last_remote["stages"]}
        else:
            status = {"now": now, "targets": st["pages"], "mode": st["mode"], "picks": [],
                      "stages": stages + [{"name": n, "state": "pending"} for n in st["remote_stages"]],
                      "splits": {s: {"target": n, "pages": 0, "samples": 0, "rows": 0, "marks": 0}
                                 for s, n in st["pages"].items()}}
        write_json(os.path.join(RUN, "status.json"), {
            "polled_at": now, "poll": st["poll"], "cost": round(self.cost(), 4), "budget": st["budget"],
            "hourly": st.get("hourly", HOURLY), "machine_id": st.get("machine_id"),
            "started": st.get("billing_started") or st["requested_at"], "status": status,
            "pause_reason": st.get("pause_reason"), "paused": bool(st.get("paused")), "mode": st["mode"]})

    def killed(self):
        return os.path.exists(os.path.join(RUN, "KILL"))

    def exec(self, command, timeout=300, check=True):
        code, out, err = self.jl.exec(self.state["machine_id"], command, timeout=timeout)
        if check and code != 0:
            raise RuntimeError(f"command failed ({code}): {command[:100]}: {(out + err).strip()[-600:]}")
        return code, out, err

    def settle(self, machine_id, timeout=900):
        end = time.time() + timeout
        while True:
            inst = self.jl.get(machine_id)
            status = inst.get("status") or ""
            if status == "Running" or not status.endswith("ing") or time.time() > end:
                return inst
            time.sleep(10)

    def stage(self, name, work):
        entry = next(s for s in self.state["local_stages"] if s["name"] == name)
        entry.clear()
        entry.update(name=name, state="running", start=time.time())
        self.save()
        self.publish()
        try:
            detail = work()
        except BaseException as error:
            entry.update(state="failed", end=time.time(), log=[f"{type(error).__name__}: {error}"[-600:]])
            self.save()
            self.publish()
            raise
        entry.update(state="done", end=time.time(), **({"log": [detail]} if detail else {}))
        self.save()
        self.publish()

    # --- a run ------------------------------------------------------------------------------
    def archive_run(self):
        if not os.path.isdir(RUN) or not os.listdir(RUN):
            return
        meta = load(os.path.join(RUN, "run_meta.json")) or {}
        watch = load(os.path.join(RUN, "status.json")) or {}
        mode = meta.get("mode", "full")         # the first run had no mode: it was a full one
        if mode == "full" and watch.get("pause_reason") == "finished" and not os.path.exists(LAST_FULL):
            shutil.copyfile(os.path.join(RUN, "status.json"), LAST_FULL)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(meta.get("started") or os.path.getmtime(RUN)))
        target = os.path.join(ARCHIVE, meta.get("run_id") or f"{stamp}-{mode}")
        os.makedirs(ARCHIVE, exist_ok=True)
        for attempt in range(10):
            try:
                os.rename(RUN, target)
                return
            except PermissionError:
                time.sleep(1)
        os.rename(RUN, target)

    def start_run(self, request):
        mode = request.get("mode")
        if mode not in MODES:
            self.say(f"ignoring a request for mode {mode!r}")
            return
        cfg = MODES[mode]
        try:
            options = run_options(request)
            asked = run_pages(mode, request)
        except ValueError as error:
            self.error = f"the run request's settings were refused: {error}"
            self.say(self.error)
            return
        self.archive_run()
        os.makedirs(RUN, exist_ok=True)
        run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{mode}"
        pages = self.args.pages or asked                   # --pages: a local test's small run
        splits = list(pages)
        budget = float(request.get("budget") or run_budget(mode, pages))
        recognition = options["recognition"]["on"]
        self.state = {"run_id": run_id, "mode": mode, "phase": "starting", "requested_at": time.time(),
                      "hourly": vm_hourly(cfg["vm"]["vcpus"]), "options": options,
                      "budget": budget, "pages": pages, "shard_pages": self.args.shard_pages or cfg["shard_pages"],
                      "remote_stages": (["check_cache", "cache_rebuild", "prepare_output", "synth_geometry",
                                         "recog_pool_train", "swap_additions", "all_done"] if mode == "additions" else
                                        ["check_cache", "cache_rebuild", "prepare_output", *[f"recog_{s}" for s in splits],
                                         "swap_recognition", "all_done"] if mode == "recognition" else
                                        ["check_cache", "cache_rebuild",
                                         "prepare_output", *[f"synth_{s}" for s in splits],
                                         *(["swap_dataset"] if mode == "full" else []), "count_classes", "all_done"]),
                      "local_stages": [{"name": n, "state": "pending"} for n in LOCAL_STAGES],
                      "poll": 0, "pulled": [], "machine_id": None, "billing_started": None}
        self.last_remote, self.error = None, None
        size = cfg["vm"]
        write_json(os.path.join(RUN, "run_meta.json"), {"mode": mode, "run_id": run_id, "region": REGION,
                                                        "vm": f"{size['vcpus']} vCPU / {size['ram']} GB / 100 GB SSD",
                                                        "machine_id": None, "hourly": self.state["hourly"],
                                                        "budget": budget, "started": time.time(), "options": options})
        self.save()
        self.publish()
        self.say(f"starting {run_id}: " + ", ".join(f"{n:,} {s}" for s, n in pages.items()) + " pages"
                 + f"; recognition {'on' if recognition else 'off'}"
                 + (f" with {options['recognition'].get('pool_epochs')} whole-pool textline passes"
                    if recognition and options["recognition"].get("pool_epochs") else "")
                 + f", mark crops {options['mark_crop']}, {options.get('geometry_per_page') or 0:g} geometry figures "
                 + f"without a page per train page, {len(options['tuning'])} tuned settings")
        try:
            self.stage("resume_vm", self.resume_vm)
            if self.killed():
                raise Stopped("stopped from the tracker")
            self.stage("upload_code", self.upload_code)
        except Stopped:
            return self.finish("kill requested")
        except Exception as error:      # noqa: BLE001 - reported on the stage and in the heartbeat
            failed = next((s["name"] for s in self.state["local_stages"] if s.get("state") == "failed"), "setup")
            self.error = f"{type(error).__name__}: {error}"[-600:]
            return self.finish(f"stage failed: {failed}")
        self.state["phase"] = "running"
        self.save()
        self.watch()

    def resume_vm(self):
        found = [i for i in self.jl.list() if i.get("name") == DATA_VM_NAME]
        if len(found) != 1:
            raise RuntimeError(f"{len(found)} instances are named {DATA_VM_NAME!r}")
        inst = self.settle(int(found[0]["machine_id"]))
        if inst.get("status") not in ("Paused", "Running"):
            raise RuntimeError(f"the dataset VM {inst['machine_id']} is {inst.get('status')}")
        st = self.state
        size = MODES[st["mode"]]["vm"]
        st.update(machine_id=int(inst["machine_id"]))
        self.save()
        if inst["status"] == "Paused":
            inst = self.resume_any_size(int(inst["machine_id"]), size)
            st["machine_id"] = int(inst["machine_id"])
        # billed from when the VM runs: waiting for its server to have free cores costs nothing
        st["billing_started"] = time.time()
        self.save()
        cores = int(inst.get("cores") or 0) or size["vcpus"]
        ram = int(inst.get("ram") or 0) or size["ram"]
        st["hourly"], st["cores"] = vm_hourly(cores), cores
        self.save()
        meta = load(os.path.join(RUN, "run_meta.json")) or {}
        write_json(os.path.join(RUN, "run_meta.json"), {**meta, "machine_id": st["machine_id"], "started": st["billing_started"],
                                                        "vm": f"{cores} vCPU / {ram} GB / 100 GB SSD", "hourly": st["hourly"]})
        end = time.time() + 900
        while True:
            if self.killed():
                raise Stopped("stopped from the tracker")
            try:
                if self.jl.exec(st["machine_id"], "true", timeout=90)[0] == 0:
                    break
            except JLError:
                pass
            if time.time() > end:
                raise RuntimeError("the dataset VM did not accept SSH within 15 minutes")
            self.message = "waiting for the dataset VM to accept SSH"
            self.publish()
            time.sleep(15)
        _, out, _ = self.exec('echo "HOME=$HOME"', timeout=120)
        st["home"] = out.strip().split("HOME=")[-1]
        self.save()
        return f"VM {st['machine_id']} running with {cores} vCPUs, ${st['hourly']:.2f}/h"

    def resume_run(self):
        """A full run that failed after its pages were made (at swap_dataset, say), carried on from where it stopped
        (requests/RESUME.json): the dataset VM resumed, this code's regen_remote.sh and status.py put in the run's
        folder, the failed stages' records cleared, and the script started again, which skips every stage already
        done. The run's cost so far is kept, against the same budget."""
        st = self.state
        if not st or st.get("phase") != "failed" or st.get("mode") != "full" or not st.get("home"):
            self.say("resume ignored: there is no failed full run to carry on")
            return
        spent = max(0.0, (st.get("billing_ended") or time.time()) - (st.get("billing_started") or time.time()))
        self.error = None
        st.update(phase="starting", pause_reason=None, paused=False, finished_at=None)
        self.say(f"carrying on {st['run_id']} from where it stopped")
        found = [i for i in self.jl.list() if i.get("name") == DATA_VM_NAME]
        if len(found) != 1:
            raise RuntimeError(f"{len(found)} instances are named {DATA_VM_NAME!r}")
        inst = self.settle(int(found[0]["machine_id"]))
        # billed from as far back as the run had already run, so cost() carries on from its total
        st.update(billing_started=time.time() - spent, billing_ended=None, machine_id=int(inst["machine_id"]))
        self.save()
        if inst.get("status") == "Paused":
            inst = self.resume_any_size(int(inst["machine_id"]), MODES["full"]["vm"])
        elif inst.get("status") != "Running":
            raise RuntimeError(f"the dataset VM {inst['machine_id']} is {inst.get('status')}")
        st["machine_id"] = int(inst["machine_id"])
        # the size it came back at sets the workers (launch) and the rate; time already run keeps its cost
        cores = int(inst.get("cores") or 0)
        if cores and cores != st.get("cores"):
            spent = self.cost()
            st["cores"], st["hourly"] = cores, vm_hourly(cores)
            st["billing_started"] = time.time() - spent / st["hourly"] * 3600
            self.say(f"carrying on with {cores} vCPUs, ${st['hourly']:.2f}/h")
        self.save()
        end = time.time() + 900
        while self.jl.exec(st["machine_id"], "true", timeout=90)[0] != 0:
            if time.time() > end:
                raise RuntimeError("the dataset VM did not accept SSH within 15 minutes")
            time.sleep(15)
        folder = f"{st['home']}/regen/{st['run_id']}"
        for name in ("regen_remote.sh", "status.py"):
            self.jl.upload(st["machine_id"], os.path.join(REMOTE, name), f"{folder}/{name}")
        # a failed stage's record would be read as the run failing again before the script reaches it
        self.exec(f"cd {folder}/stages && grep -l '\"state\": \"failed\"' *.json | xargs -r rm -f", check=False)
        self.launch()
        st["phase"] = "running"
        self.save()
        self.publish()
        self.watch()

    def resume_any_size(self, machine_id, size):
        """The paused dataset VM resumed at `size`, or failing that at each of RESUME_SIZES in turn, the round
        tried again every RESUME_RETRY seconds until RESUME_WAIT runs out, when the last error is raised."""
        end = time.time() + RESUME_WAIT
        while True:
            for choice in (size, *RESUME_SIZES):
                label = f"{choice['vcpus']} vCPUs / {choice['ram']} GB" if choice else "its last size"
                self.say(f"resuming the dataset VM {machine_id} at {label}")
                try:
                    return self.jl.resume(machine_id, **(choice or {}))
                except JLError as error:
                    last = error
                    if self.settle(machine_id).get("status") != "Paused":
                        raise
                    self.say(f"could not resume at {label}: {str(error)[-160:]}")
                if self.killed():
                    raise Stopped("stopped from the tracker")
            if time.time() + RESUME_RETRY > end:
                raise last
            wake = time.time() + RESUME_RETRY
            while time.time() < wake:
                if self.killed():
                    raise Stopped("stopped from the tracker")
                self.message = (f"no size of the dataset VM is free on its server right now; trying again at "
                                f"{time.strftime('%H:%M:%S', time.localtime(wake))}")
                self.publish()
                time.sleep(15)

    def upload_code(self):
        st = self.state
        run_id, home, full = st["run_id"], st["home"], st["mode"] == "full"
        options = st.get("options") or run_options({})
        rec = options["recognition"]
        targets = {"mode": st["mode"], "splits": list(st["pages"]), **st["pages"], "shard_pages": st["shard_pages"],
                   "stages": st["remote_stages"], "cache": "run/cache", "picks": PICKS_PER_SPLIT,
                   # a recognition run writes its splits' recognition/ beside the dataset, then swaps them in
                   "out": ("run/dataset_new" if full else "run/recognition_new" if st["mode"] == "recognition"
                           else "run/additions_new" if st["mode"] == "additions" else f"regen/{run_id}/dataset"),
                   "final": ("run/dataset" if full or st["mode"] in ("recognition", "additions")
                             else f"regen/{run_id}/dataset"),
                   "tuning": options["tuning"], "mark_crop": options["mark_crop"],
                   "geometry_per_page": options.get("geometry_per_page", 0.0),
                   "recognition": {"tasks": rec["tasks"], "picks": RECOG_PICKS_PER_TASK,
                                   "pool_epochs": rec.get("pool_epochs", 0)}}
        bundle = os.path.join(tempfile.gettempdir(), f"note_synth_regen_{run_id}.tgz")
        with tarfile.open(bundle, "w:gz") as tar:
            for name in sorted(os.listdir(NOTE_SYNTH)):
                if name.endswith(".py") or name == "splits.json":
                    tar.add(os.path.join(NOTE_SYNTH, name), arcname=f"{run_id}/note_synth/{name}")
            if rec["tasks"]:
                for name in sorted(os.listdir(RECOG_SYNTH)):
                    if name.endswith(".py") and not name.startswith("test_"):
                        tar.add(os.path.join(RECOG_SYNTH, name), arcname=f"{run_id}/recog_synth/{name}")
            tar.add(os.path.join(NOTE_SYNTH, "data", "iamondb_writers.json"),
                    arcname=f"{run_id}/note_synth/data/iamondb_writers.json")
            for name in ("regen_remote.sh", "status.py"):
                tar.add(os.path.join(REMOTE, name), arcname=f"{run_id}/{name}")
            tar.add(os.path.join(HERE, "class_counts.py"), arcname=f"{run_id}/class_counts.py")
            tar.add(PACK, arcname=f"{run_id}/pack.py")
            data = json.dumps(targets, indent=1).encode()
            info = tarfile.TarInfo(f"{run_id}/targets.json")
            info.size, info.mtime = len(data), time.time()
            tar.addfile(info, io.BytesIO(data))
        size = os.path.getsize(bundle)
        try:
            self.exec("mkdir -p ~/regen")
            self.jl.upload(st["machine_id"], bundle, f"{home}/regen/{run_id}.tgz")
        finally:
            os.remove(bundle)
        # earlier runs' folders go (a sample's dataset is small, and the full run's lives in ~/run)
        _, out, _ = self.exec(f"cd ~/regen && find . -mindepth 1 -maxdepth 1 ! -name {run_id}.tgz -exec rm -rf {{}} + ; "
                              f"tar xzf {run_id}.tgz && rm -f {run_id}.tgz && echo UNPACKED")
        if "UNPACKED" not in out:
            raise RuntimeError(f"the code did not unpack: {out[-300:]}")
        self.launch()
        return f"{size / 1e3:,.0f} KB"

    def launch(self):
        rid = self.state["run_id"]
        detach = "nohup" if self.jl.fake else "setsid nohup"        # Git Bash (local test) has no setsid
        # a worker per core starves sshd on a small VM and the status polls time out: one core is left free
        cores = int(self.state.get("cores") or 0)
        workers = f"env WORKERS={max(1, cores - 1)} " if cores else ""      # nohup needs env, not a bare assignment
        _, out, _ = self.exec(f"chmod +x ~/regen/{rid}/regen_remote.sh && mkdir -p ~/regen/{rid}/logs && "
                              f"({detach} {workers}bash ~/regen/{rid}/regen_remote.sh >> ~/regen/{rid}/logs/runner.log 2>&1 "
                              f"< /dev/null &) && sleep 3 && echo LAUNCHED")
        if "LAUNCHED" not in out:
            raise RuntimeError(f"regen_remote.sh did not start: {out[-300:]}")

    def remote_status(self):
        _, out, err = self.exec(f"python3 ~/regen/{self.state['run_id']}/status.py", timeout=180, check=False)
        line = next((l for l in reversed(out.splitlines()) if l.startswith("{") and '"stages"' in l), None)
        if line is None:
            raise RuntimeError((out + err).strip()[-300:] or "no output")
        return json.loads(line)

    def pull_picks(self, picks):
        """New picks' files in one tar, laid out as run/picks/<split>/<kind>/p<page>_<part>.<ext>."""
        st = self.state
        home, rid, pulled = st["home"], st["run_id"], set(st["pulled"])
        # picks name files as python on the VM spells its home (pack.py stores them relative to it)
        norm = lambda p: p.replace("\\", "/").rstrip("/")
        py_home = norm((self.last_remote or {}).get("home") or home)
        key = lambda p: f"{p['split']}/{p['kind']}/{p['page']}"
        new = [p for p in picks if p.get("split") != "test" and key(p) not in pulled]
        if not new:
            return
        paths = sorted({path for p in new for path in p["remote"].values()})
        remote_tar = f"{home}/regen/{rid}/picks.tar"
        self.exec(f"python3 ~/regen/{rid}/pack.py {shlex.quote(remote_tar)} " + " ".join(shlex.quote(p) for p in paths),
                  timeout=300)
        tmp = tempfile.mkdtemp(prefix="synth_picks_")
        try:
            local_tar = os.path.join(tmp, "picks.tar")
            self.jl.download(st["machine_id"], remote_tar, local_tar)
            with tarfile.open(local_tar) as tar:
                members = [m for m in tar.getmembers() if m.isfile() and ".." not in m.name.split("/")]
                tar.extractall(tmp, members=members, filter="data")
            for pick in new:
                folder = os.path.join(RUN, "picks", pick["split"], pick["kind"])
                os.makedirs(folder, exist_ok=True)
                local = {}
                for part, remote in pick["remote"].items():
                    path = norm(remote)
                    inside = path.lower().startswith(py_home.lower() + "/")
                    source = os.path.join(tmp, *path[len(py_home) + 1:].split("/")) if inside else None
                    if not source or not os.path.exists(source):
                        break
                    dest = os.path.join(folder, f"p{pick['page']:07d}_{part}{os.path.splitext(remote)[1]}")
                    shutil.copyfile(source, dest)
                    local[part] = dest
                else:
                    with open(os.path.join(RUN, "picks.jsonl"), "a") as f:
                        f.write(json.dumps({**pick, "local": local, "pulled_at": time.time()}) + "\n")
                    pulled.add(key(pick))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            self.exec(f"rm -f {shlex.quote(remote_tar)}", check=False)
        st["pulled"] = sorted(pulled)
        self.save()

    def pull_counts(self):
        st = self.state
        try:
            self.jl.download(st["machine_id"], f"{st['home']}/regen/{st['run_id']}/class_counts.json",
                             os.path.join(RUN, "class_counts.json"))
        except JLError as error:
            self.say(f"class counts were not copied: {str(error)[:200]}")

    def watch(self):
        st = self.state
        failures = relaunches = 0
        self.say(f"watching {st['run_id']}")
        while True:
            reason = None
            try:
                remote = self.remote_status()
                failures = 0
                self.last_remote = remote
            except Exception as error:      # noqa: BLE001 - any failure to reach the VM counts
                remote = None
                failures += 1
                self.say(f"status unavailable ({failures}/5): {str(error)[:200]}")
            if remote:
                try:
                    self.pull_picks(remote.get("picks") or [])
                except Exception as error:  # noqa: BLE001
                    self.say(f"picks were not copied this time: {str(error)[:200]}")
                states = {s["name"]: s.get("state") for s in remote["stages"]}
                failed = [n for n, s in states.items() if s == "failed"]
                if failed:
                    reason = f"stage failed: {', '.join(failed)}"
                elif states.get("all_done") == "done":
                    self.pull_counts()
                    reason = "finished"
                elif remote.get("runner_alive") is False:
                    if relaunches >= 3:
                        reason = "the run stopped on the VM and did not restart"
                    else:
                        relaunches += 1
                        self.say("the run is not going on the VM: starting it again (done stages are skipped)")
                        self.launch()
            if self.killed():
                reason = "kill requested"
            upsize = os.path.join(REQUESTS, "UPSIZE.json")
            if os.path.exists(upsize):
                os.remove(upsize)
                if st.get("mode") == "full" and not reason:
                    reason = UPSIZE_REASON
            if self.cost() >= st["budget"] - BUDGET_MARGIN:
                reason = f"budget reached (${self.cost():.2f} of ${st['budget']:.2f})"
            if failures >= 5:
                reason = "VM unreachable for 5 polls"
            st["poll"] += 1
            self.save()
            self.publish()
            if reason:
                return self.finish(reason)
            if remote:
                done = sum(s.get("pages", 0) for s in remote["splits"].values())
                target = sum(s.get("target") or 0 for s in remote["splits"].values())
                self.message = f"{done:,} of {target:,} pages"
            end = time.time() + self.args.interval
            while time.time() < end and not self.killed():
                time.sleep(2)

    def finish(self, reason):
        st = self.state
        paused = not st.get("billing_started") or bool(st.get("billing_ended"))
        if not paused and st.get("machine_id"):
            try:
                self.jl.pause(st["machine_id"])
                paused = True
            except JLError as error:
                try:
                    paused = self.settle(st["machine_id"]).get("status") == "Paused"
                except JLError:
                    paused = False
                if not paused:
                    self.error = f"PAUSE FAILED, check the Jarvis dashboard: {str(error)[-300:]}"
            if paused:
                st["billing_ended"] = time.time()
        phase = "finished" if reason == "finished" else "stopped" if reason == "kill requested" else "failed"
        st.update(phase=phase, pause_reason=reason, paused=paused, finished_at=time.time())
        self.save()
        self.publish()
        if st["mode"] == "full" and phase == "finished":
            shutil.copyfile(os.path.join(RUN, "status.json"), LAST_FULL)
        elif st["mode"] == "recognition" and phase == "finished" and os.path.exists(LAST_FULL):
            # the dataset's recognition/ folders were replaced: the last full run's status takes their new counts
            last, now = load(LAST_FULL) or {}, load(os.path.join(RUN, "status.json")) or {}
            for split, entry in ((now.get("status") or {}).get("splits") or {}).items():
                kept = ((last.get("status") or {}).get("splits") or {}).get(split)
                if kept is not None and entry.get("recognition"):
                    kept["recognition"] = entry["recognition"]
            write_json(LAST_FULL, last)
        elif st["mode"] == "additions" and phase == "finished" and os.path.exists(LAST_FULL):
            # train/geometry and train/recognition_pool were added to the dataset: the last full run's status counts them
            last, now = load(LAST_FULL) or {}, load(os.path.join(RUN, "status.json")) or {}
            entry = ((now.get("status") or {}).get("splits") or {}).get("train") or {}
            kept = ((last.get("status") or {}).get("splits") or {}).get("train")
            if kept is not None:
                for key in ("geometry_samples", "recognition_pool"):
                    if entry.get(key):
                        kept[key] = entry[key]
                write_json(LAST_FULL, last)
        self.say(f"{st['run_id']} {phase}: {reason}" + ("; the VM is paused" if paused else "; THE VM DID NOT PAUSE"))

    # --- main -----------------------------------------------------------------------------------
    def main(self):
        os.makedirs(REQUESTS, exist_ok=True)
        if self.state and self.state.get("phase") == "running":
            self.watch()
        elif self.state and self.state.get("phase") == "starting":
            self.error = "this process stopped while a run was being set up; start the run again"
            self.finish("orchestrator restarted during setup")
        self.say("waiting for a run request")
        last_pause_try = 0.0
        while True:
            st = self.state
            if st and st.get("billing_started") and not st.get("billing_ended") and time.time() - last_pause_try > 60:
                last_pause_try = time.time()
                self.finish(st.get("pause_reason") or "pausing")         # a pause that failed is tried again
            path = os.path.join(REQUESTS, "START.json")
            request = load(path)
            if request:
                os.remove(path)
                try:
                    self.start_run(request)
                except Exception as error:  # noqa: BLE001 - reported; any VM left running is paused
                    self.error = f"{type(error).__name__}: {error}"[-600:]
                    self.say(f"FAILED: {self.error}")
                    traceback.print_exc()
                    if self.state and self.state.get("billing_started") and not self.state.get("billing_ended"):
                        self.finish("the orchestrator failed")
                self.say("waiting for a run request")
            resume = os.path.join(REQUESTS, "RESUME.json")
            if os.path.exists(resume):
                os.remove(resume)
                try:
                    self.resume_run()
                except Exception as error:  # noqa: BLE001 - reported; the VM is paused again
                    self.error = f"{type(error).__name__}: {error}"[-600:]
                    self.say(f"RESUME FAILED: {self.error}")
                    traceback.print_exc()
                    if self.state and self.state.get("billing_started") and not self.state.get("billing_ended"):
                        self.finish("the orchestrator failed")
                self.say("waiting for a run request")
            time.sleep(2)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=60, help="seconds between status polls while a run is going")
    ap.add_argument("--root", default=HERE, help="where run/, requests/ and the heartbeat live (local test: a scratch folder)")
    ap.add_argument("--pages", default=None, help="local test: pages per split instead of the mode's, e.g. train=8,val=4")
    ap.add_argument("--shard-pages", type=int, default=None, help="local test: pages per shard instead of the mode's")
    ap.add_argument("--guard-port", type=int, default=GUARD_PORT)
    args = ap.parse_args()
    args.pages = {k: int(v) for k, v in (kv.split("=") for kv in args.pages.split(","))} if args.pages else None
    return args


if __name__ == "__main__":
    arguments = parse_args()
    configure(arguments.root)
    guard = socket.socket()
    try:
        guard.bind(("127.0.0.1", arguments.guard_port))     # one at a time: the login script may start a second
    except OSError:
        sys.exit("another synth_orchestrator.py is already running")
    Regen(arguments).main()
