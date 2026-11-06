"""orchestrator.py — the local side of the training run on Jarvis Labs.

    python orchestrator.py                         # run dir runs/train; waits for Start in the tracker
    python orchestrator.py --job marks             # the mark separation CNN: run dir runs/marks, one GPU
    JL_FAKE=DIR python orchestrator.py --run-dir ../run_fake --local-test ...   # everything on this laptop

Jobs (--job, JOBS below): seg is the Florence-2 segmentation run (training/, remote/);
marks is the mark separation CNN (training/marks_cnn/, remote_marks/). Each has its
own run dir, GPU VM and lock, and the dataset VM is shared between them. A seg run's Start
(train_api.py) needs a mark separator: the marks run's, whose best/ and last/ model.pt are
copied back when it ends (fetch_mark_model resumes a VM paused before the copy), or a saved one
from note_synth/inference/mark_separation/ picked in the tracker instead of training. Its copy
goes to the GPU VM in the bundle as ~/train/marks/model.pt, where validation, the test run and
the explorer take marks out of detected lines with it before recognition. The explorer can
take a page's marks out with another saved one instead, or with none.

It needs nobody in the loop. The tracker server (tracker_server.py, through train_api.py) only writes
request files into the run dir; this process acts on them and keeps state.json
current for the page.

Phases (state.json "phase"), each resumable from meta.json if this process restarts:
  waiting_for_start  nothing rented; START.json appears when Start is confirmed
  provisioning       dataset VM resumed, GPU VM created, SSH key from the GPU VM to the
                     dataset VM, code uploaded, train_remote.sh started on the GPU VM
  training           train_status.py polled every --interval s: log lines, validation
                     samples and their images, GPU stats. The dataset VM is paused as soon
                     as its data has been copied
  stopping           STOP sent (stop button or budget): the run saves last/ and exits,
                     then the GPU VM is paused (at most --stop-wait s)
  trained            GPU VM paused, nothing billed but disk; TEST.json starts the test
  testing            GPU VM resumed, test_remote.sh (once only), results pulled, VM paused
  done | stopped | failed
Once a run has ended (trained, done, stopped or failed; everything paused after a failure)
it stays as it is until RESET.json appears (Restart training in the tracker): everything is
paused, the run dir's files move to runs/archive/<run>-<time>/, and the phase goes back to
waiting_for_start. The VMs are kept; the next run's upload wipes the GPU VM's checkpoints,
dataset copy, test marker and stage records (not the venv), so it trains on the dataset as it
is then: from the pretrained weights, from a model in inference/comprehension/, or (seg, RESET.json
{"continue_from": "best" | "last"} and Start's continue_run) from the old run's checkpoint, set
aside before the wipe and given to train.py as --init-adapter with a fresh optimizer and schedule.

BUDGET.json (Change budget in the tracker) becomes the run's budget within a few seconds, whatever
phase it is in; a stop already sent for the old budget is not taken back.

Saving to note_synth/inference/comprehension/ (seg): when training ends, best/ and last/ are copied back into
files/train/run/ before the GPU VM is paused. A save (save/requests/<id>.json from the tracker, or the
save_to chosen at Start) copies one of them into inference/comprehension/<name> with run_info.json and stats.json, the
validation record logged at that checkpoint's step. If the copy isn't here (a run paused on its budget, or
one that ended before this existed), the VM is resumed through the explorer, the checkpoints are copied
back, and the VM is paused again unless pages were explored meanwhile.

Money: every create / resume opens a billing segment at that machine's hourly
rate, closed when its pause succeeds. At budget minus ~10 minutes of GPU time the
run is asked to stop and save; at the budget itself the GPU VM is paused outright.
"""
import argparse
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import sys
import tarfile
import threading
import time
import traceback

import inference_models
from jlcmd import JL, JLError

HERE = os.path.dirname(os.path.abspath(__file__))
# beside the checkout's tracker/, so renaming or moving the root does not need editing here
TRAINING = os.path.join(os.path.dirname(os.path.dirname(HERE)), "training")
STORAGE_HOURLY_PER_GB = 0.0137 / 100            # Jarvis storage, from the CPU VM's bill
REMOTE_STAGES = ["setup_env", "fetch_dataset", "fetch_model", "smoke", "train"]
LABELS = {"requested": "Start confirmed", "resume_data_vm": "Resume the dataset VM",
          "create_gpu_vm": "Rent the GPU VM", "link_vms": "Connect the GPU VM to the dataset VM",
          "upload_code": "Upload the training code", "launch": "Launch on the GPU VM",
          "setup_env": "Install PyTorch and libraries", "fetch_dataset": "Copy the dataset to the GPU VM",
          "fetch_model": "Download Florence-2-large", "smoke": "Smoke test on every GPU (6 steps)",
          "train": "Train"}
MARKS_REMOTE_STAGES = ["setup_env", "fetch_dataset", "prepare", "smoke", "train"]
MARKS_LABELS = {**LABELS, "setup_env": "Install PyTorch", "fetch_dataset": "Copy the mark crops to the GPU VM",
                "prepare": "Scale and pad the crops to 600 x 100", "smoke": "Smoke test (6 steps)"}
MARKS_CODE = os.path.join(TRAINING, "marks_cnn")
# a checkpoint's files, as copied back to files/train/run/<name>/ when training ends
JOBS = {
    "seg": {"run_dir": "train", "name": "seg-train", "storage": 200, "guard_port": 8798, "code": TRAINING,
            "remote": [("remote", None)], "labels": LABELS, "remote_stages": REMOTE_STAGES,
            "checkpoint_files": inference_models.FILES},
    "marks": {"run_dir": "marks", "name": "marks-cnn", "storage": 100, "guard_port": 8799,
              "code": MARKS_CODE, "remote": [("remote", ["pack.py"]), ("remote_marks", None)],
              "labels": MARKS_LABELS, "remote_stages": MARKS_REMOTE_STAGES,
              "checkpoint_files": inference_models.KINDS["mark_separation"]},
}
EXPLORE_WORKER = os.path.join(HERE, "explore", "explore_worker.py")
# what the explorer uploads from note_synth/inference/: the request field naming it, its folder in ~/train/explore/
EXPLORE_UPLOADS = {"comprehension": ("model", "models", "uploaded_models"),
                   "mark_separation": ("marks", "marks", "uploaded_marks")}
# recog_eval.py draws each detected line's ink with recog_synth's code (recog_pages.py, which reads recog_layout.py),
# the code the recognition samples are made with: sent beside it in code/
RECOG_CODE = [os.path.join(os.path.dirname(TRAINING), "recog_synth", name) for name in ("recog_layout.py", "recog_pages.py")]
# recog_eval.py takes marks out of detected lines with the mark separation network: its model and crop fitting
MARKS_EVAL_CODE = [os.path.join(MARKS_CODE, name) for name in ("marks_model.py", "marks_data.py")]
# remote/budget_guard.py on the GPU VM: stops training at the deadline the budget buys, then pauses the VM itself; and
# pauses a finished run's VM once polls stop. Jarvis's backends by region (jarvislabs/constants.py REGION_URLS).
GUARD_API = {"IN1": "https://backendc.jarvislabs.net/", "IN2": "https://backendn.jarvislabs.net/",
             "EU1": "https://backendeu.jarvislabs.net/"}
GUARD_STOP_WAIT = 360           # seconds the guard gives a stopped run to save before it pauses the VM anyway
GUARD_UNWATCHED = 1200          # seconds without a poll after training ends before the guard pauses the VM
GUARD_MARGIN_S = 120            # the guard's deadline: this much GPU time and GUARD_STOP_WAIT before the budget runs out
# stops the guard by the pid it wrote: `pkill -f budget_guard.py` also matches the shell sent to start it
STOP_GUARD = "kill $(cat ~/train/guard.pid 2>/dev/null) 2>/dev/null; "
# the eval code the explorer runs, uploaded beside its worker so a run's older code/ is not what reads pages
EXPLORE_CODE = [*(os.path.join(TRAINING, name) for name in ("seg_eval.py", "seg_dataset.py", "model.py", "recog_eval.py")),
                *RECOG_CODE, *MARKS_EVAL_CODE]
EXPLORE_ACTIVE = ("queued", "sent", "running")


def load(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, data):
    """`data` to `path` through a temporary file renamed over it. On Windows the rename is refused while a reader
    (the tracker server) has the file open: tried again with a growing wait, about 20 s in all, then the temporary
    file is removed and the error raised."""
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(min(0.1 * (attempt + 1), 1.5))
    try:
        os.replace(tmp, path)
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def rel_to_home(path, home):
    """A remote path relative to the remote home, '/'-separated, or None outside it."""
    norm = lambda p: p.replace("\\", "/").rstrip("/")
    p, h = norm(path), norm(home)
    if p.lower().startswith(h.lower() + "/"):
        return p[len(h) + 1:]
    return None


class Failure(RuntimeError):
    pass


class Orchestrator:
    def __init__(self, args):
        self.args = args
        self.dir = os.path.abspath(args.run_dir)
        self.files = os.path.join(self.dir, "files")
        os.makedirs(self.files, exist_ok=True)
        self.jl = JL()
        self.meta = load(self.path("meta.json")) or self.new_meta({}, args.data_vm)
        self.remote = self.meta.get("remote_snapshot") or {}
        self.message = ""
        self.job = args.job
        self.labels, self.remote_stages = JOBS[args.job]["labels"], JOBS[args.job]["remote_stages"]
        self.explore_thread, self.explore_outcome, self.explore_polled = None, None, 0.0
        ex = self.meta.get("explore")
        if ex and ex.get("state") == "resuming":        # this process stopped while a resume was under way
            ex["state"] = "starting" if self.running("gpu") else "paused"
        elif ex and ex.get("state") == "pausing":       # ... or a pause: ask for it again
            ex["state"] = "ready"
            os.makedirs(self.path("explore"), exist_ok=True)
            open(self.path("explore", "PAUSE"), "w").close()

    # --- files and state ------------------------------------------------------
    def path(self, *parts):
        return os.path.join(self.dir, *parts)

    def new_meta(self, machines, data_vm):
        return {"phase": "waiting_for_start", "stages": {}, "billing": [], "machines": machines, "log_offset": 0,
                "pulled": {}, "test": {"state": "not_run"}, "data_vm": data_vm, "region": self.args.region,
                "run_id": time.time()}

    def say(self, text):
        line = f"[{time.strftime('%H:%M:%S')}] {text}"
        print(line, flush=True)
        with open(self.path("orchestrator.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
        self.message = text

    def save(self):
        self.meta["remote_snapshot"] = self.remote
        write_json(self.path("meta.json"), self.meta)

    def cost(self, kind=None, since=None):
        now = time.time()
        total = 0.0
        for seg in self.meta["billing"]:
            if kind and seg["kind"] != kind:
                continue
            start = max(seg["start"], since or 0)
            end = seg["end"] or now
            if end > start:
                total += (end - start) / 3600 * seg["hourly"]
        return total

    def hourly_now(self):
        return sum(seg["hourly"] for seg in self.meta["billing"] if seg["end"] is None)

    def publish(self, **extra):
        """state.json for the tracker."""
        config = self.meta.get("config") or {}
        stages = []
        for name in ["requested", "resume_data_vm", "create_gpu_vm", "link_vms", "upload_code", "launch"]:
            stage = self.meta["stages"].get(name, {"state": "pending"})
            stages.append({"name": name, "label": self.labels[name], "where": "local", **stage})
        remote_stages = {s["name"]: s for s in (self.remote.get("stages") or [])}
        for name in self.remote_stages:
            stage = remote_stages.get(name, {"state": "pending"})
            stages.append({"name": name, "label": self.labels[name], "where": "gpu", **stage})
        test = dict(self.meta["test"])
        if test.get("started_at"):
            test["cost"] = round(self.cost("gpu", since=test["started_at"]), 4)
        doc = {
            "heartbeat_at": time.time(), "pid": os.getpid(), "run_id": self.meta.get("run_id"),
            "scrapped": self.meta.get("scrapped"), "phase": self.meta["phase"],
            "message": self.message, "error": self.meta.get("error"), "pause_reason": self.meta.get("pause_reason"),
            "config": config, "fake": bool(self.jl.fake),
            "cost": {"total": round(self.cost(), 4), "gpu": round(self.cost("gpu"), 4),
                     "data": round(self.cost("data"), 4), "hourly_now": round(self.hourly_now(), 4),
                     "budget": config.get("budget")},
            "machines": self.meta["machines"], "stages": stages, "remote": self.remote, "test": test,
            "training_started_at": self.meta.get("training_started_at"),
            "explore": self.explore_public() if self.job == "seg" else None,
            "saves": sorted((self.meta.get("saves") or {}).values(), key=lambda s: s.get("requested_at") or 0),
            "checkpoints_local": self.meta.get("checkpoints_local") or {},
            "mark_model_fetch": self.meta.get("mark_model_fetch"),
            "continue_from": self.meta.get("continue_from"),
            "log_offset": self.meta["log_offset"], **extra}
        try:
            write_json(self.path("state.json"), doc)
        except OSError as error:
            # only the page's view of the run, rewritten every few seconds: a write the file's readers kept out
            # is skipped rather than failing the run
            print(f"[{time.strftime('%H:%M:%S')}] state.json not written this time: {error}", flush=True)

    def stage(self, name, state, **fields):
        entry = self.meta["stages"].setdefault(name, {})
        if state == "running":
            entry.clear()
            entry["start"] = time.time()
        else:
            entry["end"] = time.time()
        entry.update(state=state, **fields)
        self.save()
        self.publish()

    def idle(self, seconds, watch_kill=True):
        """Sleeps in short steps, keeping the heartbeat fresh; True if KILL appeared."""
        end = time.time() + seconds
        while time.time() < end:
            if watch_kill and os.path.exists(self.path("KILL")):
                return True
            self.take_budget()
            self.publish()
            time.sleep(min(3.0, max(0.0, end - time.time())))
        return watch_kill and os.path.exists(self.path("KILL"))

    def take_budget(self):
        """BUDGET.json from the tracker: the run's budget from now on (the watch loop reads it every poll)."""
        path = self.path("BUDGET.json")
        if not os.path.exists(path):
            return
        request = load(path)
        try:
            os.remove(path)
        except OSError:
            return                                  # the tracker server is writing it; taken next time
        cfg = self.meta.get("config")
        try:
            budget = float((request or {})["budget"])
        except (KeyError, TypeError, ValueError):
            return
        if not cfg or not budget > 0:
            return
        old = cfg.get("budget")
        cfg["budget"] = budget
        self.meta.setdefault("budget_changes", []).append({"at": time.time(), "from": old, "to": budget})
        self.save()
        self.say(f"budget changed from ${old or 0:.2f} to ${budget:.2f}" +
                 (" (the run was already asked to stop; that stands)" if self.meta["phase"] == "stopping" else ""))
        if self.meta["phase"] == "training" and self.running("gpu") and self.meta.get("guard"):
            try:
                self.write_guard()                  # the GPU VM's guard stops at the new budget's deadline
            except (JLError, Failure) as error:
                self.say(f"the GPU VM's guard kept its old deadline: {str(error)[:200]}")

    # --- machines ----------------------------------------------------------------
    def open_segment(self, kind, machine_id, hourly):
        self.meta["billing"].append({"kind": kind, "machine_id": machine_id, "hourly": hourly,
                                     "start": time.time(), "end": None})
        self.save()

    def close_segments(self, kind):
        for seg in self.meta["billing"]:
            if seg["kind"] == kind and seg["end"] is None:
                seg["end"] = time.time()
        self.save()

    def remember(self, kind, inst, hourly=None):
        machine = self.meta["machines"].setdefault(kind, {})
        m = re.search(r"(\w+)@([\w.\-]+)", inst.get("ssh_command") or "")
        machine.update(id=int(inst["machine_id"]), status=inst.get("status"), private_ip=inst.get("private_ip"),
                       public_ip=inst.get("public_ip"), user=m.group(1) if m else "ubuntu",
                       gpu_type=inst.get("gpu_type"), num_gpus=inst.get("num_gpus"),
                       storage_gb=inst.get("storage_gb"), name=inst.get("name"))
        if hourly is not None:
            machine["hourly"] = hourly
        for seg in self.meta["billing"]:
            if seg["kind"] == kind and seg["end"] is None:
                seg["machine_id"] = machine["id"]
        self.save()
        return machine

    def running(self, kind):
        return any(seg["kind"] == kind and seg["end"] is None for seg in self.meta["billing"])

    def settle(self, machine_id, timeout=900):
        """The instance once it is no longer pausing, resuming or starting."""
        end = time.time() + timeout
        while True:
            inst = self.jl.get(machine_id)
            status = inst.get("status") or ""
            if status == "Running" or not status.endswith("ing") or time.time() > end:
                return inst
            time.sleep(10)

    def wait_for_ssh(self, kind, timeout=900):
        """A VM reported Running can take a minute or two before it accepts SSH."""
        machine_id = self.meta["machines"][kind]["id"]
        end = time.time() + timeout
        while True:
            try:
                code, _, err = self.jl.exec(machine_id, "true", timeout=90)
                if code == 0:
                    return
                detail = f"exit {code}: {err[-200:]}"
            except JLError as error:
                detail = str(error)[-200:]
            if time.time() > end:
                raise Failure(f"the {kind} VM {machine_id} did not accept SSH within {timeout // 60} minutes: {detail}")
            self.message = f"waiting for the {kind} VM to accept SSH"
            self.idle(15, watch_kill=False)

    def pause(self, kind, reason):
        machine = self.meta["machines"].get(kind)
        if not machine or not self.running(kind):
            return
        self.say(f"pausing the {kind} VM {machine['id']}: {reason}")
        if kind == "gpu" and self.meta.get("guard"):
            try:        # this pause is the orchestrator's own: the VM's budget guard has nothing left to do
                self.jl.exec(machine["id"], STOP_GUARD + "true", timeout=60)
            except JLError:
                pass
        try:
            self.jl.pause(machine["id"])
        except JLError:
            if self.settle(machine["id"]).get("status") != "Paused":      # already paused counts as paused
                raise
        machine["status"] = "Paused"
        self.close_segments(kind)
        if kind == "gpu" and (self.meta.get("explore") or {}).get("state", "paused") != "paused":
            self.meta["explore"]["state"] = "paused"
            self.explore_cancel(f"the VM was paused: {reason}")
            self.save()

    def pause_everything(self, reason):
        """Both VMs at once: a GPU VM's pause can take minutes and must not keep the other one billing."""
        def one(kind):
            try:
                self.pause(kind, reason)
            except JLError as error:
                self.meta["pause_error"] = f"{kind} VM did not pause: {error}"
                self.say(f"PAUSE FAILED for the {kind} VM — check the Jarvis dashboard: {error}")
        threads = [threading.Thread(target=one, args=(kind,)) for kind in ("gpu", "data")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.save()
        self.publish()

    # What train_remote.sh's fetch_dataset copies (its `parts` and the optional `extra` folders), less shards/
    COPIED = ("train/samples val/samples val/pages test/pages train/report.json val/report.json test/report.json "
              "train/recognition/textline train/recognition/formula train/recognition_pool/textline train/geometry")

    def copy_matches(self):
        """Whether the GPU VM holds every file of the dataset fetch_dataset copies, byte for byte in count and size,
        checked before the dataset VM is paused: the copy's own checks count manifest rows and page files but not every
        image. Counted with find on both VMs (~/run/dataset there, ~/train/data here), shards/ left out as the copy
        leaves them out. A mismatch keeps the dataset VM running and is reported; it is counted once per run."""
        if self.meta.get("copy_verified"):
            return True
        count = ("cd {root} && find " + self.COPIED + " -path '*/shards' -prune -o -type f -printf '%s\n' "
                 "2>/dev/null | awk '{{n++; s+=$1}} END {{print \"FILES\", n+0, s+0}}'")
        try:
            sides = {}
            for kind, root in (("data", "~/run/dataset"), ("gpu", "~/train/data")):
                _, out, _ = self.exec(kind, count.format(root=root), timeout=900, check=False)
                line = next((l for l in out.splitlines() if l.startswith("FILES")), "")
                sides[kind] = tuple(int(x) for x in line.split()[1:3]) if line else None
        except (JLError, Failure, ValueError) as error:
            self.say(f"could not compare the copy with the dataset yet: {str(error)[:200]}; the dataset VM stays up")
            return False
        if sides["data"] and sides["data"] == sides["gpu"] and sides["data"][0] > 0:
            self.meta["copy_verified"] = {"files": sides["data"][0], "bytes": sides["data"][1], "at": time.time()}
            self.save()
            self.say(f"copy checked: {sides['data'][0]:,} files, {sides['data'][1] / 1e9:.1f} GB on both VMs")
            return True
        if not self.meta.get("copy_mismatch"):
            self.meta["copy_mismatch"] = {"data": sides["data"], "gpu": sides["gpu"], "at": time.time()}
            self.save()
            self.say(f"COPY MISMATCH, the dataset VM is kept running: dataset VM {sides['data']}, GPU VM {sides['gpu']}"
                     " (files, bytes)")
        return False

    def exec(self, kind, command, timeout=300, check=True):
        code, out, err = self.jl.exec(self.meta["machines"][kind]["id"], command, timeout=timeout)
        if check and code != 0:
            raise Failure(f"command on the {kind} VM failed ({code}): {command[:120]}\n{(out + err)[-1500:]}")
        return code, out, err

    # --- phases -------------------------------------------------------------------
    def wait_for_start(self):
        self.say("waiting for Start in the tracker")
        while True:
            request = load(self.path("START.json"))
            if request:
                break
            if os.path.exists(self.path("KILL")):
                os.remove(self.path("KILL"))          # nothing is running to stop
            self.idle(3, watch_kill=False)
        self.meta["config"] = request
        self.meta["phase"] = "provisioning"
        detail = f"{request['num_gpus']} x {request['gpu_type']}, budget ${request['budget']:.2f}"
        if self.job == "seg":
            detail += f", LoRA rank {request.get('lora_rank', 16)}"
            if request.get("init_model"):
                detail += f", trained further from inference/comprehension/{request['init_model']['name']}"
            elif request.get("continue_from"):
                c = request["continue_from"]
                detail += f", continues the previous run's {c['checkpoint']}/ checkpoint" + (f" (step {c['step']})" if c.get("step") is not None else "")
            if request.get("save_to"):
                detail += f", saves {request['save_to']['checkpoint']}/ as inference/comprehension/{request['save_to']['name']} at the end"
            if request.get("mark_model"):
                m = request["mark_model"]
                detail += (f", takes marks out with inference/mark_separation/{m['name']}" if m.get("source") == "inference"
                           else f", takes marks out with mark separation {m['checkpoint']}/ (step {m.get('step')})")
        self.stage("requested", "done", detail=detail)

    def provision(self):
        cfg = self.meta["config"]
        if os.path.exists(self.path("KILL")):
            return self.stop_before_training()

        if self.meta["stages"].get("resume_data_vm", {}).get("state") != "done":
            self.stage("resume_data_vm", "running")
            # by name every time unless --data-vm pins it: Jarvis may give a VM a new id when it is resumed
            self.meta["data_vm"] = self.args.data_vm or self.find_data_vm()
            inst = self.settle(self.meta["data_vm"])
            if inst["status"] == "Paused":
                self.open_segment("data", inst["machine_id"], self.args.data_hourly)
                inst = self.jl.resume(inst["machine_id"])
            elif inst["status"] == "Running":
                self.open_segment("data", inst["machine_id"], self.args.data_hourly)
            else:
                raise Failure(f"dataset VM {inst['machine_id']} is {inst['status']}")
            machine = self.remember("data", inst, self.args.data_hourly)
            self.meta["data_vm"] = machine["id"]
            self.wait_for_ssh("data")
            self.stage("resume_data_vm", "done", detail=f"VM {machine['id']} running")
        if os.path.exists(self.path("KILL")):
            return self.stop_before_training()

        if self.meta["stages"].get("create_gpu_vm", {}).get("state") != "done":
            self.stage("create_gpu_vm", "running")
            offers = [g for g in self.jl.gpus() if g["gpu_type"] == cfg["gpu_type"] and g["region"] == self.args.region
                      and g.get("workload_type") == "vm"]
            if not offers:
                raise Failure(f"no {cfg['gpu_type']} VM offer in {self.args.region}")
            price = offers[0]["price_per_hour"]
            if max(g["num_free_devices"] for g in offers) < cfg["num_gpus"]:
                raise Failure(f"only {max(g['num_free_devices'] for g in offers)} {cfg['gpu_type']} free in "
                              f"{self.args.region}, {cfg['num_gpus']} needed; nothing was rented")
            hourly = price * cfg["num_gpus"] + self.args.storage * STORAGE_HOURLY_PER_GB
            existing = self.meta["machines"].get("gpu") or {}
            if existing.get("id"):
                # the VM as Jarvis lists it now, by name: its id may have changed since this run last saw it
                found = [i for i in self.jl.list() if i.get("name") == self.args.name]
                if len(found) > 1:
                    raise Failure(f"{len(found)} instances are named {self.args.name!r}; nothing was rented")
                existing = {"id": int(found[0]["machine_id"])} if found else {}
            if existing.get("id"):
                # a GPU VM rented by an earlier attempt: resume it (its disk may already hold work) with the GPU type
                # asked for at Start, which a resume otherwise leaves as it was (an L4 kept from an earlier run)
                inst = self.settle(existing["id"])
                if inst["status"] == "Running" and inst.get("gpu_type") and inst["gpu_type"] != cfg["gpu_type"]:
                    self.say(f"GPU VM {inst['machine_id']} is running with {inst['gpu_type']}: pausing it to resume with {cfg['gpu_type']}")
                    self.jl.pause(inst["machine_id"])
                    inst = self.settle(inst["machine_id"])
                if inst["status"] == "Paused":
                    self.open_segment("gpu", inst["machine_id"], hourly)
                    swap = cfg["gpu_type"] if inst.get("gpu_type") != cfg["gpu_type"] else None
                    inst = self.jl.resume(inst["machine_id"], num_gpus=cfg["num_gpus"], gpu=swap)
                elif inst["status"] == "Running":
                    self.open_segment("gpu", inst["machine_id"], hourly)
                else:
                    raise Failure(f"GPU VM {inst['machine_id']} is {inst['status']}")
                verb = "resumed"
            else:
                self.open_segment("gpu", None, hourly)
                try:
                    inst = self.jl.create_gpu_vm(cfg["gpu_type"], cfg["num_gpus"], self.args.storage,
                                                 self.args.name, self.args.region)
                except JLError as error:
                    found = [i for i in self.jl.list() if i.get("name") == self.args.name and i.get("status") == "Running"]
                    if not found:
                        self.close_segments("gpu")
                        raise Failure(f"could not create the GPU VM: {error}")
                    inst = found[0]
                verb = "rented"
            machine = self.remember("gpu", inst, hourly)
            actual = machine.get("gpu_type") or self.jl.get(machine["id"]).get("gpu_type")
            if actual != cfg["gpu_type"]:           # never train, or bill, as one GPU on another
                self.pause("gpu", f"it came up with {actual}, not the {cfg['gpu_type']} asked for")
                raise Failure(f"GPU VM {machine['id']} came up with {actual}, not the {cfg['gpu_type']} asked for; nothing was trained")
            self.wait_for_ssh("gpu")
            self.stage("create_gpu_vm", "done", detail=f"VM {machine['id']} {verb}: {cfg['num_gpus']} x {cfg['gpu_type']} "
                                                       f"at ${hourly:.2f}/h")
        if os.path.exists(self.path("KILL")):
            return self.stop_before_training()

        if self.meta["stages"].get("link_vms", {}).get("state") != "done":
            self.stage("link_vms", "running")
            data = self.meta["machines"]["data"]
            if self.jl.fake:
                host = "local"
            else:
                _, out, _ = self.exec("gpu", "mkdir -p ~/.ssh && (test -f ~/.ssh/datavm || ssh-keygen -q -t ed25519 "
                                             "-N '' -f ~/.ssh/datavm) && cat ~/.ssh/datavm.pub")
                key = next(line for line in out.splitlines() if line.startswith("ssh-"))
                self.exec("data", f"mkdir -p ~/.ssh && touch ~/.ssh/authorized_keys && "
                                  f"(grep -qF {shlex.quote(key)} ~/.ssh/authorized_keys || "
                                  f"echo {shlex.quote(key)} >> ~/.ssh/authorized_keys)")
                # tried again for --link-wait s: a VM just resumed onto other hardware (a GPU type swapped) can take a
                # while to bring its private network up. A host key that no longer matches the one this GPU VM
                # recorded (the VMs are inside one VPC) is forgotten and tried again rather than refused for good
                host, errors, tries, deadline = None, [], 0, time.time() + self.args.link_wait
                while host is None:
                    tries += 1
                    for candidate in (data["private_ip"], data["public_ip"]):
                        if not candidate:
                            continue
                        code, out, err = self.exec("gpu", f"ssh -i ~/.ssh/datavm -o StrictHostKeyChecking=accept-new "
                                                          f"-o ConnectTimeout=15 {data['user']}@{candidate} true", check=False)
                        if code == 0:
                            host = candidate
                            break
                        said = " ".join((out + err).split())[-300:] or f"exit {code}"
                        errors.append(f"{candidate}: {said}")
                        if "host key" in said.lower() or "identification has changed" in said.lower():
                            self.exec("gpu", f"ssh-keygen -R {shlex.quote(candidate)}", check=False)
                    if host is not None:
                        break
                    if os.path.exists(self.path("KILL")):
                        return self.stop_before_training()
                    if time.time() > deadline:
                        _, net, _ = self.exec("gpu", "ip -brief addr; ip route | head -5", check=False)
                        raise Failure(f"the GPU VM cannot reach the dataset VM over SSH (private or public IP) after {tries} "
                                      f"tries in {self.args.link_wait // 60} min:\n" + "\n".join(errors[-2:]) +
                                      f"\nthe GPU VM's network:\n{net.strip()[-600:]}")
                    self.message = f"the GPU VM can't reach the dataset VM yet (try {tries}): {errors[-1][:160]}"
                    self.stage("link_vms", "running", detail=self.message)
                    self.idle(20, watch_kill=False)
            self.meta["data_host"] = host
            self.stage("link_vms", "done", detail=f"dataset VM reachable at {host}")

        if self.meta["stages"].get("upload_code", {}).get("state") != "done":
            self.stage("upload_code", "running")
            bundle = self.path("trainer_bundle.tgz")
            self.build_bundle(bundle, cfg)
            self.refresh_home()
            keep = (cfg.get("continue_from") or {}).get("checkpoint") if self.job == "seg" else None
            if self.meta.get("wipe_remote"):
                # a restarted run on this VM: its checkpoints (train_remote.sh would resume from run/last),
                # dataset copy, test marker and stage records go; the venv and the model cache stay. A continued
                # run's checkpoint is moved aside to ~/train/continue first (again safe to repeat)
                aside = (f"if [ -f ~/train/run/{keep}/adapter_model.safetensors ]; then rm -rf ~/train/continue && "
                         f"mv ~/train/run/{keep} ~/train/continue; fi && test -f ~/train/continue/adapter_model.safetensors && "
                         if keep else "")
                code, out, err = self.exec("gpu", aside + "rm -rf ~/train/run ~/train/init ~/train/marks ~/train/data ~/train/prepared "
                                           "~/train/smoke ~/train/test ~/train/explore ~/train/outbox ~/train/logs "
                                           "~/train/guard.json ~/train/guard_paused.json ~/train/.last_poll ~/train/rescore && "
                                           "mkdir -p ~/train/stages && find ~/train/stages -type f ! -name setup_env.json -delete",
                                           timeout=600, check=False)
                if code != 0:
                    raise Failure((f"the previous run's {keep}/ checkpoint is not on the GPU VM to continue from"
                                   if keep else "could not clear the previous run from the GPU VM") + f":\n{(out + err)[-800:]}")
                self.meta["wipe_remote"] = False
                self.say("cleared the previous run from the GPU VM" + (f", keeping its {keep}/ checkpoint" if keep else ""))
            dest = f"{self.meta['gpu_home_shell']}/trainer_bundle.tgz"
            self.jl.upload(self.meta["machines"]["gpu"]["id"], bundle, dest)
            init = ("if [ -d ~/train/continue ]; then rm -rf ~/train/init && mv ~/train/continue ~/train/init; fi && "
                    "test -f ~/train/init/adapter_model.safetensors && " if keep else "rm -rf ~/train/init && ")
            # ~/train/marks goes too: the bundle holds the mark separation model this run was started with, if any
            self.exec("gpu", f"mkdir -p ~/train && rm -rf ~/train/code ~/train/marks && {init}tar xzf ~/trainer_bundle.tgz -C ~/train "
                             "&& mv ~/train/code/train_args.json ~/train/train_args.json && ls ~/train/code | head -30")
            self.stage("upload_code", "done", detail=f"{os.path.getsize(bundle) / 1e3:,.0f} KB")

        if self.meta["stages"].get("launch", {}).get("state") != "done":
            self.stage("launch", "running")
            env = {"DATA_HOST": self.meta["data_host"], "DATA_USER": self.meta["machines"]["data"]["user"],
                   "NPROC": str(cfg["num_gpus"]), **self.remote_env()}
            assign = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
            detach = "nohup" if self.jl.fake else "setsid nohup"
            self.write_guard()
            guard = ("" if self.jl.fake else
                     "(setsid nohup python3 ~/train/code/remote/budget_guard.py > ~/train/logs/guard.log 2>&1 < /dev/null &) && ")
            # The guard is stopped by its pid (budget_guard.py writes guard.pid): matching it by name would also match
            # this very shell, whose command line holds the guard's path, and killing that ends the call as a
            # connection error. train_remote.sh takes its own lock, so starting it twice is safe.
            self.exec("gpu", f"mkdir -p ~/train/logs && {STOP_GUARD}{guard}"
                             f"({detach} env {assign} bash ~/train/code/remote/train_remote.sh "
                             f"> ~/train/logs/runner.log 2>&1 < /dev/null &) && sleep 3 && echo LAUNCHED")
            self.meta["phase"] = "training"
            self.meta["training_started_at"] = time.time()
            self.stage("launch", "done")

    def refresh_home(self):
        """The GPU VM's home as its shell spells it (upload and download paths); read again after any resume."""
        _, out, _ = self.exec("gpu", 'echo "HOME=$HOME"')
        self.meta["gpu_home_shell"] = out.strip().split("HOME=")[-1]
        self.save()

    def remote_env(self):
        return dict(kv.split("=", 1) for kv in self.args.remote_env)

    def find_data_vm(self):
        """The dataset VM's id, looked up by name: Jarvis may give a VM a new id when it is resumed."""
        found = [i for i in self.jl.list() if i.get("name") == self.args.data_vm_name]
        if len(found) != 1:
            raise Failure(f"{len(found)} instances are named {self.args.data_vm_name!r}; pass --data-vm")
        return int(found[0]["machine_id"])

    def build_bundle(self, path, cfg):
        """The job's training code, its remote scripts and train_args.json, under code/."""
        job = JOBS[self.job]
        if self.job == "marks":
            args = ["--epochs", cfg["epochs"], "--batch-size", cfg["batch_size"], "--lr", cfg["lr"],
                    "--warmup-steps", cfg["warmup_steps"], "--val-every", cfg["val_every"],
                    "--save-every", cfg["save_every"], "--val-max", cfg["val_max"], "--samples", cfg["samples"],
                    "--log-every", cfg["log_every"], "--keep", 2]
        else:
            args = ["--epochs", cfg["epochs"], "--batch-size", cfg["batch_size"], "--grad-accum", cfg["grad_accum"],
                    "--lr", cfg["lr"], "--warmup-steps", cfg["warmup_steps"], "--val-every", cfg["val_every"],
                    "--save-every", cfg["save_every"], "--page-val-max", cfg["page_val_max"],
                    "--eval-batches", cfg["eval_batches"], "--samples", cfg["samples"], "--log-every", cfg["log_every"],
                    "--num-workers", cfg["num_workers"], "--keep", 2,
                    "--lora-rank", cfg.get("lora_rank", 16), "--lora-alpha", cfg.get("lora_alpha", 2 * cfg.get("lora_rank", 16))]
            if cfg.get("grad_checkpointing"):
                args.append("--grad-checkpointing")
            if cfg.get("vision_lora"):
                args.append("--vision-lora")
            if cfg.get("init_model") or cfg.get("continue_from"):
                args += ["--init-adapter", "~/train/init"]         # expanded by train.py; a continued run's is put there on the VM
        args += self.args.extra_train_args
        with tarfile.open(path, "w:gz") as tar:
            if self.job == "seg" and cfg.get("init_model"):
                # the model this run trains further from, as train_api.py copied it at Start; unpacked to ~/train/init
                for name in inference_models.FILES:
                    source = self.path("init_model", name)
                    if not os.path.isfile(source):
                        raise Failure(f"the copy of inference/comprehension/{cfg['init_model']['name']} made at Start is missing: {source}")
                    tar.add(source, arcname=f"init/{name}")
            for name in sorted(os.listdir(job["code"])):
                if name.endswith(".py") or name == "requirements.txt":
                    tar.add(os.path.join(job["code"], name), arcname=f"code/{name}")
            if self.job == "seg":
                for path in [*RECOG_CODE, *MARKS_EVAL_CODE]:
                    tar.add(path, arcname=f"code/{os.path.basename(path)}")
                if cfg.get("mark_model"):
                    # the mark separation model train_api.py copied at Start; unpacked to ~/train/marks, where
                    # train_remote.sh, test_remote.sh and the explorer look for it
                    source = self.path("mark_model", "model.pt")
                    if not os.path.isfile(source):
                        raise Failure(f"the copy of the mark separation model made at Start is missing: {source}")
                    tar.add(source, arcname="marks/model.pt")
            for folder, names in job["remote"]:
                for name in names or sorted(os.listdir(os.path.join(HERE, folder))):
                    if os.path.isfile(os.path.join(HERE, folder, name)):
                        tar.add(os.path.join(HERE, folder, name), arcname=f"code/remote/{name}")
            data = json.dumps([str(a) for a in args]).encode()
            info = tarfile.TarInfo("code/train_args.json")
            info.size, info.mtime = len(data), time.time()
            tar.addfile(info, io.BytesIO(data))

    # --- the GPU VM's budget guard (remote/budget_guard.py) -----------------------------------------
    def write_guard(self):
        """~/train/guard.json: the deadline the run's remaining budget buys at this VM's rate, less GUARD_STOP_WAIT and
        GUARD_MARGIN_S, and the VM's id and Jarvis backend for it to pause itself with. The API key (JL_API_KEY) goes
        to ~/train/.jl_key; without one the guard can still stop training but not pause the VM."""
        cfg, machine = self.meta["config"], self.meta["machines"]["gpu"]
        hourly = machine.get("hourly") or cfg["num_gpus"] * 2.69
        left = cfg["budget"] - self.cost() - 0.05
        doc = {"deadline": time.time() + max(0.0, left) / hourly * 3600 - GUARD_STOP_WAIT - GUARD_MARGIN_S,
               "machine_id": machine["id"], "api_base": GUARD_API.get(self.args.region, GUARD_API["IN2"]),
               "stop_wait": GUARD_STOP_WAIT, "unwatched": GUARD_UNWATCHED, "written_at": time.time()}
        self.exec("gpu", f"mkdir -p ~/train && printf %s {shlex.quote(json.dumps(doc))} > ~/train/guard.json.tmp "
                         "&& mv ~/train/guard.json.tmp ~/train/guard.json")
        key = os.environ.get("JL_API_KEY")
        if not key and os.name == "nt":
            try:        # set for the user after this process started, so not in its environment
                import winreg
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as reg:
                    key = winreg.QueryValueEx(reg, "JL_API_KEY")[0]
            except OSError:
                key = None
        if key:
            import tempfile
            handle, local = tempfile.mkstemp()
            try:
                with os.fdopen(handle, "w") as f:
                    f.write(key)
                self.jl.upload(machine["id"], local, f"{self.meta['gpu_home_shell']}/train/.jl_key")
            finally:
                os.remove(local)
            self.exec("gpu", "chmod 600 ~/train/.jl_key", check=False)
        elif not self.meta.get("guard"):
            self.say("JL_API_KEY is not set for the tracker: the GPU VM's guard can stop training at the budget, "
                     "but not pause the VM")
        self.meta["guard"] = doc
        self.save()

    def gpu_paused_itself(self):
        """Whether the GPU VM of a guarded training run is paused: a status poll failed because the guard paused it."""
        if not self.meta.get("guard"):
            return False
        try:
            found = [i for i in self.jl.list() if i.get("name") == self.args.name]
        except JLError:
            return False
        return len(found) == 1 and found[0].get("status") == "Paused"

    def recover_paused_run(self):
        """The GPU VM paused itself while this orchestrator was away (budget_guard.py). The run's billing ends when it
        paused (estimated now, set exactly from guard_paused.json below); the VM is resumed with one GPU for the rest
        of the log, the samples and best/ and last/, and paused again. The run ends trained when it has a checkpoint."""
        guard, progress = self.meta.get("guard") or {}, self.remote.get("progress") or {}
        polled = self.remote.get("polled_at") or time.time()
        ends = [time.time()]
        if guard.get("deadline") and guard["deadline"] < time.time():
            ends.append(guard["deadline"] + GUARD_STOP_WAIT)
        if progress.get("eta_s") is not None:
            ends.append(polled + progress["eta_s"] + GUARD_UNWATCHED)
        run_segments = [seg for seg in self.meta["billing"] if seg["kind"] == "gpu" and seg["end"] is None]
        for seg in run_segments:
            seg["end"] = max(seg["start"], min(ends))
            seg["note"] = "ended when the GPU VM paused itself (budget_guard.py)"
        self.meta["machines"]["gpu"]["status"] = "Paused"
        self.save()
        self.say("the GPU VM paused itself while this orchestrator was away: copying the rest of the run back")
        machine, cfg = self.meta["machines"]["gpu"], self.meta["config"]
        storage = self.args.storage * STORAGE_HOURLY_PER_GB
        found = [i for i in self.jl.list() if i.get("name") == self.args.name]
        inst = self.settle(int(found[0]["machine_id"]))
        self.open_segment("gpu", int(inst["machine_id"]), (machine["hourly"] - storage) / cfg["num_gpus"] + storage)
        self.meta["billing"][-1]["purpose"] = "recover"
        try:
            inst = self.jl.resume(inst["machine_id"], num_gpus=1)
            self.remember("gpu", inst)
            self.wait_for_ssh("gpu")
            self.refresh_home()
            _, out, _ = self.exec("gpu", STOP_GUARD + "cat ~/train/guard_paused.json", check=False)
            paused = next((json.loads(l) for l in out.splitlines() if l.startswith("{")), None)
            if paused:
                for seg in run_segments:
                    seg["end"] = max(seg["start"], paused["at"])
                self.say(f"the GPU VM had paused itself at {time.strftime('%H:%M:%S', time.localtime(paused['at']))}: "
                         f"{paused.get('reason')}")
            while True:
                st = self.status()
                st["gpus"] = []                     # the one-GPU recovery VM's stats are not the run's
                self.take_status(st)
                if st["log"]["next"] >= st["log"]["size"]:
                    break
            self.pull_samples(st.get("samples") or [])
            self.fetch_checkpoints()
        except (JLError, Failure) as error:
            self.say(f"could not copy the rest of the run back: {str(error)[:300]}")
        finally:
            self.pause("gpu", "the run's results copied back")
        progress = self.remote.get("progress") or {}
        finished = progress.get("phase") == "finished"
        self.meta["phase"] = "trained" if (finished or self.remote.get("checkpoints")) else "stopped"
        self.meta["pause_reason"] = ("training finished" if finished else
                                     "stopped by the GPU VM's budget guard while the tracker was away")
        self.save()
        self.publish()

    def stop_before_training(self):
        self.say("stop requested before training started")
        self.pause_everything("stopped from the tracker")
        self.meta["phase"] = "stopped"
        self.meta["pause_reason"] = "stopped from the tracker before training started"
        self.save()

    # --- watching ---------------------------------------------------------------------
    def status(self):
        py = self.args.remote_python
        _, out, _ = self.exec("gpu", f"{py} ~/train/code/remote/train_status.py {self.meta['log_offset']}",
                              timeout=180)
        line = next((l for l in reversed(out.splitlines()) if l.startswith("{") and '"stages"' in l), None)
        if line is None:
            raise Failure(f"train_status.py printed no status: {out[-500:]}")
        return json.loads(line)

    def take_status(self, st):
        """Log lines, GPU stats, the snapshot the page shows."""
        log = st.get("log") or {}
        if log.get("records"):
            with open(self.path("log.jsonl"), "a") as f:
                for record in log["records"]:
                    f.write(json.dumps(record) + "\n")
        if log.get("offset", 0) < self.meta["log_offset"]:       # the remote log started over
            self.say("remote log restarted")
        self.meta["log_offset"] = log.get("next", self.meta["log_offset"])
        if st.get("gpus"):
            with open(self.path("gpu.jsonl"), "a") as f:
                f.write(json.dumps({"t": st["now"], "gpus": st["gpus"]}) + "\n")
        self.meta["remote_home"] = st["home"]
        self.remote = {k: st.get(k) for k in ("now", "home", "stages", "progress", "config", "control", "control_ack",
                                              "stop_pending", "console", "checkpoints", "gpus", "processes",
                                              "disk_used_gb", "disk_total_gb", "test")}
        self.remote["polled_at"] = time.time()
        self.save()

    def pull(self, remote_paths):
        """Remote files (absolute paths under the remote home) into files/, one tar per call."""
        home = self.meta["remote_home"]
        wanted = [p for p in remote_paths if rel_to_home(p, home)]
        if not wanted:
            return 0
        name = hashlib.sha1("|".join(wanted).encode()).hexdigest()[:12]
        remote_tar = f"{self.meta.get('gpu_home_shell', '~')}/train/outbox/{name}.tar"
        py = self.args.remote_python
        quoted = " ".join(shlex.quote(p.replace("\\", "/")) for p in wanted)
        self.exec("gpu", f"mkdir -p ~/train/outbox && {py} ~/train/code/remote/pack.py ~/train/outbox/{name}.tar {quoted}",
                  timeout=300)
        local_tar = self.path(f"pull_{name}.tar")
        try:
            self.jl.download(self.meta["machines"]["gpu"]["id"], remote_tar, local_tar)
            with tarfile.open(local_tar) as tar:
                members = [m for m in tar.getmembers() if m.isfile() and ".." not in m.name.split("/")]
                tar.extractall(self.files, members=members, filter="data")
        finally:
            if os.path.exists(local_tar):
                os.remove(local_tar)
            self.exec("gpu", f"rm -f ~/train/outbox/{name}.tar", check=False)
        return len(wanted)

    def pull_samples(self, sample_files):
        """New or grown sample streams, then every image and page JSON they point at."""
        home = self.meta["remote_home"]
        pulled = self.meta["pulled"]
        changed = [s for s in sample_files if pulled.get(s["path"]) != s["size"]]
        if not changed:
            return
        self.pull([s["path"] for s in changed])
        needed = []
        for s in changed:
            pulled[s["path"]] = s["size"]
            local = os.path.join(self.files, rel_to_home(s["path"], home))
            try:
                rows = [json.loads(line) for line in open(local) if line.strip()]
            except (OSError, ValueError):
                continue
            for row in rows:
                image = row.get("image")
                if not image:
                    continue
                refs = [image]
                if "predicted" in row:                          # a page: its ground truth beside it
                    refs.append(os.path.splitext(image)[0] + ".json")
                for ref in refs:
                    if ref not in pulled:
                        needed.append(ref)
        for i in range(0, len(needed), 60):
            chunk = needed[i:i + 60]
            self.pull(chunk)
            for ref in chunk:
                pulled[ref] = True
        self.save()

    def watch_training(self):
        cfg = self.meta["config"]
        failures = 0
        stop_sent_at = self.meta.get("stop_sent_at")
        hourly = (self.meta["machines"].get("gpu") or {}).get("hourly") or cfg["num_gpus"] * 2.69
        reserve = hourly * 10 / 60 + 0.05
        while True:
            try:
                st = self.status()
                failures = 0
            except (JLError, Failure) as error:
                failures += 1
                self.say(f"status unavailable ({failures}/5): {str(error)[:200]}")
                if self.gpu_paused_itself():
                    return self.recover_paused_run()
                if failures >= 5:
                    raise Failure(f"GPU VM unreachable for 5 polls: {error}")
                self.idle(min(30, self.args.interval))
                continue
            self.take_status(st)
            try:
                self.pull_samples(st.get("samples") or [])
            except (JLError, Failure) as error:
                self.say(f"could not pull samples this time: {str(error)[:200]}")

            states = {s["name"]: s.get("state") for s in st["stages"]}
            progress = st.get("progress") or {}
            if states.get("fetch_dataset") == "done" and self.running("data") and self.copy_matches():
                self.pause("data", "dataset copied to the GPU VM")
            failed = [n for n, s in states.items() if s == "failed"]
            cost = self.cost()
            kill = os.path.exists(self.path("KILL"))

            if failed:
                stage = next(s for s in st["stages"] if s["name"] == failed[0])
                self.pause_everything(f"stage failed: {failed[0]}")
                raise Failure(f"stage {failed[0]} failed on the GPU VM:\n" + "\n".join(stage.get("log") or []))
            if states.get("train") == "done":
                # while the VM still runs: seg's for saving to inference/, marks' for the segmentation run to use
                self.fetch_checkpoints()
                self.pause_everything("training finished" if progress.get("phase") == "finished" else "training stopped")
                self.meta["phase"] = "trained"
                self.meta["pause_reason"] = ("training finished" if progress.get("phase") == "finished"
                                             else self.meta.get("stop_reason") or "training stopped")
                self.say(f"training ended at step {progress.get('step')}: VM paused, waiting for a test request")
                self.save()
                return
            if cost >= cfg["budget"]:
                self.pause_everything(f"budget reached (${cost:.2f})")
                self.meta["phase"] = "trained" if (self.remote.get("checkpoints") or []) else "stopped"
                self.meta["pause_reason"] = f"budget reached (${cost:.2f} of ${cfg['budget']:.2f}) before the run saved"
                self.save()
                return
            want_stop = kill or cost >= cfg["budget"] - reserve
            if want_stop and not stop_sent_at:
                self.meta["stop_reason"] = "stopped from the tracker" if kill else f"budget: ${cost:.2f} of ${cfg['budget']:.2f}"
                if states.get("train") in ("running",):
                    self.exec("gpu", "touch ~/train/run/STOP", check=False)
                    stop_sent_at = self.meta["stop_sent_at"] = time.time()
                    self.meta["phase"] = "stopping"
                    self.say(f"asked the run to save and stop ({self.meta['stop_reason']})")
                else:
                    self.pause_everything(self.meta["stop_reason"])
                    self.meta["phase"] = "stopped"
                    self.meta["pause_reason"] = self.meta["stop_reason"] + " (before training began)"
                    self.save()
                    return
                self.save()
            if stop_sent_at and time.time() - stop_sent_at > self.args.stop_wait:
                self.pause_everything("the run did not stop in time")
                self.meta["phase"] = "trained" if (self.remote.get("checkpoints") or []) else "stopped"
                self.meta["pause_reason"] = f"{self.meta['stop_reason']}; paused after {self.args.stop_wait}s without a clean stop"
                self.save()
                return

            train_phase = progress.get("phase")
            self.message = (f"{train_phase or 'starting'}: step {progress.get('step', 0)}/{progress.get('total_steps', '?')}"
                            if states.get("train") == "running" else
                            next((self.labels[n] for n in self.remote_stages if states.get(n) == "running"), "starting"))
            self.publish()
            interval = 15 if stop_sent_at else self.args.interval
            if self.idle(interval) and not stop_sent_at:
                continue

    def wait_for_test(self):
        """False if the run is scrapped instead."""
        while not load(self.path("TEST.json")):
            if self.reset_requested():
                return False
            if os.path.exists(self.path("KILL")):
                os.remove(self.path("KILL"))
            self.explore_tick()
            self.save_tick()
            self.idle(3, watch_kill=False)
        self.explore_pause_for_test()
        self.meta["phase"] = "testing"
        self.meta["test"] = {"state": "requested", "requested_at": time.time(), **load(self.path("TEST.json"))}
        self.save()

    def run_test(self):
        test = self.meta["test"]
        cfg = self.meta["config"]
        machine = self.meta["machines"]["gpu"]
        if not test.get("launched"):
            test["state"] = "resuming"
            self.say("resuming the GPU VM for the test run")
            self.publish()
            try:
                inst = self.jl.get(machine["id"])
                if inst["status"] == "Paused":
                    self.open_segment("gpu", inst["machine_id"], machine["hourly"])
                    test["started_at"] = time.time()
                    # all the run's GPUs: a resume without --num-gpus keeps the count last used, and the
                    # page explorer resumes this VM with one
                    inst = self.jl.resume(inst["machine_id"], num_gpus=cfg["num_gpus"])
                self.remember("gpu", inst)
                self.wait_for_ssh("gpu")
                self.refresh_home()
            except JLError as error:
                self.close_segments("gpu")
                os.replace(self.path("TEST.json"), self.path(f"TEST.failed.{int(time.time())}.json"))
                self.meta["phase"] = "trained"
                self.meta["test"] = {"state": "not_run", "last_error": f"could not resume the GPU VM: {error}"}
                self.say(f"test not started, the GPU VM did not resume: {error}")
                self.save()
                return False
            test.setdefault("started_at", time.time())
            env = " ".join(f"{k}={shlex.quote(v)}" for k, v in {"NPROC": str(cfg["num_gpus"]), **self.remote_env()}.items())
            detach = "nohup" if self.jl.fake else "setsid nohup"
            self.exec("gpu", f"({detach} env {env} bash ~/train/code/remote/test_remote.sh > ~/train/logs/test_runner.log "
                             f"2>&1 < /dev/null &) && sleep 3 && echo LAUNCHED")
            test.update(launched=True, state="running")
            self.save()

        cap = float(test.get("max_spend") or 4.0)
        failures = 0
        while True:
            try:
                st = self.status()
                failures = 0
            except (JLError, Failure) as error:
                failures += 1
                if failures >= 5:
                    self.pause_everything("unreachable during the test")
                    raise Failure(f"GPU VM unreachable during the test: {error}")
                self.idle(20, watch_kill=False)
                continue
            self.take_status(st)
            remote_test = st.get("test") or {}
            spent = self.cost("gpu", since=test["started_at"])
            if remote_test.get("done"):
                code = remote_test["done"]["exit_code"]
                home = st["home"]
                sep = "\\" if "\\" in home else "/"
                # aggregate metrics only: test pages and per-page predictions stay on the VM
                self.pull([f"{home}{sep}train{sep}test{sep}{n}" for n in ("test_metrics.json", "test.log")])
                self.pause_everything("test finished")
                if code == 3:
                    test.update(state="refused", message="the test set had already been evaluated on this VM")
                elif code == 0:
                    test.update(state="done", finished_at=time.time(), metrics=remote_test.get("metrics"))
                else:
                    test.update(state="failed", message="\n".join(remote_test.get("log") or [])[-1500:])
                self.meta["phase"] = "done"
                self.say(f"test {test['state']}")
                self.save()
                self.publish()
                return True
            if spent >= cap:
                self.pause_everything(f"test spend cap reached (${spent:.2f})")
                test.update(state="failed", message=f"stopped at the ${cap:.2f} test spend cap")
                self.meta["phase"] = "done"
                self.save()
                return True
            prog = remote_test.get("progress") or {}
            self.message = f"testing: {prog.get('done', 0)}/{prog.get('total', '?')} {prog.get('unit', 'pages')}"
            self.publish()
            self.idle(self.args.interval, watch_kill=False)

    # --- page explorer (seg) -----------------------------------------------------------------
    # Once training has ended the tracker can resume the GPU VM with one GPU and have the model
    # run on any val page (explore/explore_worker.py). The tracker server writes explore/RESUME,
    # explore/PAUSE and explore/requests/<id>.json into the run dir; this process acts on them
    # between its other work, bills the VM's time as segments of their own ("purpose": "explore")
    # and pulls each result and the images it shows into files/. It only ever pauses the VM
    # when asked to, when the test run needs it, or when something fails.
    def explore(self):
        ex = self.meta.setdefault("explore", {"state": "paused"})
        ex.setdefault("requests", {})
        return ex

    def explore_public(self):
        ex = self.meta.get("explore") or {"state": "paused", "requests": {}}
        spent = sum(((seg["end"] or time.time()) - seg["start"]) / 3600 * seg["hourly"]
                    for seg in self.meta["billing"] if seg.get("purpose") == "explore")
        return {**ex, "spent": round(spent, 4)}

    def explore_cancel(self, message):
        for request in (self.meta.get("explore") or {}).get("requests", {}).values():
            if request["state"] in EXPLORE_ACTIVE:
                request.update(state="cancelled", message=message, step=None)

    def explore_tick(self):
        """One look at the explorer's request files and VM. Never raises: the explorer must not fail the run."""
        if self.job != "seg" or not (self.meta.get("machines") or {}).get("gpu"):
            return
        try:
            self._explore_tick()
        except Exception as error:      # noqa: BLE001
            self.explore()["error"] = f"{type(error).__name__}: {error}"[-600:]
            self.message = f"explorer: {self.explore()['error'][:200]}"
            self.save()

    def _explore_tick(self):
        ex = self.explore()
        folder = self.path("explore")
        queue = os.path.join(folder, "requests")
        os.makedirs(queue, exist_ok=True)
        changed = False
        pause, resume = os.path.join(folder, "PAUSE"), os.path.join(folder, "RESUME")
        waiting = ex["state"] in ("paused", "pausing") and os.path.exists(resume) and not os.path.exists(pause)
        for name in sorted(os.listdir(queue)):
            if not name.endswith(".json") or waiting:       # asked on a paused VM: kept for the resume asked with it
                continue
            doc = load(os.path.join(queue, name))
            os.remove(os.path.join(queue, name))
            if doc and ex["state"] not in ("paused", "pausing"):
                ex["requests"][doc["id"]] = {"id": doc["id"], "page": int(doc["page"]), "model": doc.get("model"),
                                             "marks": doc.get("marks"), "state": "queued",
                                             "requested_at": doc.get("requested_at", time.time())}
                changed = True
        if len(ex["requests"]) > 300:                   # keep the newest; never drop one in progress
            old = sorted((r for r in ex["requests"].values() if r["state"] not in EXPLORE_ACTIVE),
                         key=lambda r: r.get("requested_at", 0))[:len(ex["requests"]) - 300]
            for request in old:
                del ex["requests"][request["id"]]

        if self.explore_thread is not None:
            if self.explore_thread.is_alive():
                if changed:
                    self.save()
                return
            self.explore_thread = None
            self.explore_finish(ex, self.explore_outcome)
            changed = True

        if os.path.exists(pause):
            os.remove(pause)
            if os.path.exists(resume):
                os.remove(resume)
            if ex["state"] != "paused":
                self.explore_start(ex, "pause")
        elif os.path.exists(resume) and ex["state"] != "pausing":      # one asked while pausing waits for the pause
            os.remove(resume)
            if ex["state"] == "paused":
                self.explore_start(ex, "resume")
        elif ex["state"] in ("starting", "ready"):
            busy = ex["state"] == "starting" or any(r["state"] in EXPLORE_ACTIVE for r in ex["requests"].values())
            if time.time() - self.explore_polled >= (4 if busy else 60):
                self.explore_poll(ex)
        if changed:
            self.save()

    def explore_start(self, ex, action):
        """Opens the billing segment and starts the resume or pause in a thread: it can take minutes,
        and this process's heartbeat must stay fresh. The thread only calls jl; meta is changed here."""
        machine, cfg = self.meta["machines"]["gpu"], self.meta["config"]
        if action == "resume":
            storage = self.args.storage * STORAGE_HOURLY_PER_GB
            hourly = (machine["hourly"] - storage) / cfg["num_gpus"] + storage
            self.open_segment("gpu", machine["id"], hourly)
            self.meta["billing"][-1]["purpose"] = "explore"
            ex.update(state="resuming", since=time.time(), error=None, hourly=round(hourly, 4))
            self.say(f"explorer: resuming GPU VM {machine['id']} with 1 GPU")
        else:
            ex.update(state="pausing", since=time.time())
            self.say(f"explorer: pausing GPU VM {machine['id']}")
        self.save()
        self.publish()
        outcome = self.explore_outcome = {"action": action, "machine_id": machine["id"]}

        def work():
            try:
                if action == "resume":
                    outcome["result"] = self.explore_resume_vm(outcome)
                else:
                    try:
                        self.jl.pause(outcome["machine_id"])
                    except JLError:
                        if self.settle(outcome["machine_id"]).get("status") != "Paused":
                            raise
                outcome["ok"] = True
            except Exception as error:      # noqa: BLE001 - reported by explore_finish
                outcome["error"] = f"{type(error).__name__}: {error}"[-600:]
                if action == "resume":      # never leave a VM running that nothing is tracking
                    try:
                        self.jl.pause(outcome["machine_id"])
                    except Exception:       # noqa: BLE001
                        pass

        self.explore_thread = threading.Thread(target=work, daemon=True)
        self.explore_thread.start()

    def explore_resume_vm(self, outcome):
        """(instance, home as the shell spells it); jl calls only."""
        inst = self.jl.get(outcome["machine_id"])
        if inst.get("status") == "Paused":
            inst = self.jl.resume(outcome["machine_id"], num_gpus=1)
        elif inst.get("status") != "Running":
            raise Failure(f"GPU VM {outcome['machine_id']} is {inst.get('status')}")
        outcome["machine_id"] = machine_id = int(inst["machine_id"])
        end = time.time() + 900
        while True:
            try:
                if self.jl.exec(machine_id, "true", timeout=90)[0] == 0:
                    break
            except JLError:
                pass
            if time.time() > end:
                raise Failure("the GPU VM did not accept SSH within 15 minutes")
            time.sleep(15)
        _, out, _ = self.jl.exec(machine_id, 'mkdir -p ~/train/explore/code && echo "HOME=$HOME"', timeout=120)
        home = out.strip().split("HOME=")[-1]
        self.jl.upload(machine_id, EXPLORE_WORKER, f"{home}/train/explore/explore_worker.py")
        for path in EXPLORE_CODE:
            self.jl.upload(machine_id, path, f"{home}/train/explore/code/{os.path.basename(path)}")
        return inst, home

    def explore_finish(self, ex, outcome):
        machine = self.meta["machines"]["gpu"]
        if outcome["action"] == "resume":
            if outcome.get("ok"):
                inst, home = outcome["result"]
                self.remember("gpu", inst)
                self.meta["gpu_home_shell"] = home
                ex.update(state="starting", since=time.time())
                self.explore_polled = 0.0
                self.say(f"explorer: GPU VM {machine['id']} is running; starting the model")
            else:
                machine["id"] = outcome["machine_id"]
                machine["status"] = "Paused"
                self.close_segments("gpu")
                ex.update(state="paused", since=time.time(), error=f"The VM did not resume: {outcome.get('error')}")
                self.explore_cancel("the VM did not resume")
                self.say(f"explorer: resume failed: {outcome.get('error')}")
        elif outcome.get("ok"):
            machine["status"] = "Paused"
            self.close_segments("gpu")
            ex.update(state="paused", since=time.time(), worker=None)
            self.explore_cancel("the VM was paused")
            self.say("explorer: GPU VM paused")
        else:
            ex.update(state="ready", error=f"The VM did not pause: {outcome.get('error')}. "
                                           f"It may still be billing; check the Jarvis dashboard or press Pause again.")
            self.say(f"explorer: PAUSE FAILED: {outcome.get('error')}")
        self.save()
        self.publish()

    def explore_poll(self, ex):
        """Hands queued requests to the VM, reads the worker's status and collects finished results."""
        self.explore_polled = time.time()
        submit = [r for r in ex["requests"].values() if r["state"] == "queued"]
        # a model or mark separator from note_synth/inference/ goes to the VM before the first request that needs it,
        # and again if it changed
        for kind, (field, _, key) in EXPLORE_UPLOADS.items():
            uploaded, what = ex.setdefault(key, {}), "model" if kind == "comprehension" else "mark separator"
            for name in sorted({r.get(field) for r in submit if r["state"] == "queued"} - {None, inference_models.NO_MARKS}):
                asking = [r for r in submit if r.get(field) == name and r["state"] == "queued"]
                model = inference_models.get(name, kind)
                if model is None:
                    for request in asking:
                        request.update(state="failed", message=f"there is no {what} {name} in note_synth/inference/{kind}/ any more", step=None)
                    continue
                if uploaded.get(name) != model["signature"]:
                    for request in asking:
                        request["step"] = f"uploading {what} {name} ({model['size_mb']} MB)"
                    self.say(f"explorer: uploading {what} {name} ({model['size_mb']} MB)")
                    self.publish()
                    self.explore_upload_model(model)
                    uploaded[name] = model["signature"]
                    for request in asking:
                        request["step"] = None
                    self.save()
        submit = [r for r in submit if r["state"] == "queued"]
        uploaded_models, uploaded_marks = ex["uploaded_models"], ex["uploaded_marks"]
        parts = [f"--python {shlex.quote(self.args.explore_python)}"]
        parts += [f"--worker-arg={shlex.quote(arg)}" for arg in self.args.explore_worker_args]
        parts += [f"--submit {shlex.quote(json.dumps({'id': r['id'], 'page': r['page'], 'model': r.get('model'), 'sig': uploaded_models.get(r.get('model')), 'marks': r.get('marks'), 'marks_sig': uploaded_marks.get(r.get('marks'))}))}"
                  for r in submit]
        code, out, err = self.exec("gpu", f"{self.args.remote_python} ~/train/explore/explore_worker.py --poll "
                                          f"{' '.join(parts)}", timeout=120, check=False)
        line = next((l for l in reversed(out.splitlines()) if l.startswith("{") and '"finished"' in l), None)
        if code != 0 or line is None:
            ex["failures"] = ex.get("failures", 0) + 1
            if ex["failures"] >= 3:
                ex["error"] = f"The explorer on the VM is not answering: {(out + err).strip()[-300:]}"
            self.save()
            return
        doc = json.loads(line)
        ex["failures"] = 0
        if (ex.get("error") or "").startswith("The explorer on the VM is not answering"):
            ex["error"] = None
        if doc.get("home"):
            self.meta["remote_home"] = self.remote["home"] = doc["home"]
        for request in submit:
            request.update(state="sent", sent_at=time.time())
        status = doc.get("status") or {}
        ex["worker"] = {k: status.get(k) for k in ("state", "id", "page", "step", "checkpoint", "error")}
        if status.get("state") == "failed" and status.get("error"):
            ex["error"] = f"The model did not load on the VM: {status['error'][:400]}"
        if ex["state"] == "starting" and doc.get("alive") and status.get("state") in ("ready", "working"):
            ex.update(state="ready", since=time.time(), error=None)
            self.say(f"explorer: model ready ({status.get('checkpoint')})")
        working = status.get("id") if status.get("state") == "working" else None
        for request in ex["requests"].values():
            if request["state"] in ("sent", "running") and request["id"] == working:
                request.update(state="running", step=status.get("step"))
        if doc.get("index") and not ex.get("index_pulled"):
            self.pull([f"{self.meta['remote_home']}/train/explore/index.json"])
            ex["index_pulled"] = True
        finished = set(doc.get("finished") or [])
        for request in [r for r in ex["requests"].values() if r["state"] in ("sent", "running") and r["id"] in finished]:
            self.publish()
            self.explore_collect(request)
        self.save()
        self.publish()

    def explore_upload_model(self, model):
        """Copies a model's weights to ~/train/explore/models/<name>/ (a mark separator's to ~/train/explore/marks/<name>/)
        on the GPU VM, swapped in whole so the worker never reads half a folder. Names are inference_models.NAME, safe in
        a shell command."""
        name, gpu, place = model["name"], self.meta["machines"]["gpu"]["id"], EXPLORE_UPLOADS[model["kind"]][1]
        folder = f"~/train/explore/{place}/{name}"
        self.exec("gpu", f"rm -rf {folder}.part && mkdir -p {folder}.part", timeout=120)
        for file in inference_models.KINDS[model["kind"]]:
            self.jl.upload(gpu, os.path.join(model["folder"], file),
                           f"{self.meta['gpu_home_shell']}/train/explore/{place}/{name}.part/{file}")
        self.exec("gpu", f"rm -rf {folder} && mv {folder}.part {folder}", timeout=120)

    def explore_collect(self, request):
        rel = f"train/explore/results/{request['id']}.json"
        self.pull([f"{self.meta['remote_home']}/{rel}"])
        doc = load(os.path.join(self.files, rel))
        if not doc:
            request.update(state="failed", message="the result could not be copied from the VM", step=None)
            return
        if doc.get("error"):
            request.update(state="failed", message=doc["error"][-1500:], step=None)
            return
        pulled = self.meta["pulled"]
        needed = [p for p in [doc["image"], *(row["image"] for row in doc["rows"]["samples"]),
                              *(group["image"] for group in doc["cascade"].get("recognition") or [])] if not pulled.get(p)]
        for i in range(0, len(needed), 60):
            self.pull(needed[i:i + 60])
            for path in needed[i:i + 60]:
                pulled[path] = True
        request.update(state="done", step=None, result=rel, finished_at=time.time(),
                       mean_f1=doc["cascade"]["summary"]["mean_f1"], rows_mean_f1=doc["rows"]["summary"]["mean_f1"])

    def explore_pause_for_test(self):
        """The test run resumes the VM with all its GPUs, so the explorer must have let it go."""
        if self.job != "seg":
            return
        if self.explore_thread is not None:
            self.explore_thread.join()
            self.explore_thread = None
            self.explore_finish(self.explore(), self.explore_outcome)
        if self.explore()["state"] != "paused" and self.running("gpu"):
            self.pause("gpu", "the test run needs the VM")
        self.explore()["state"] = "paused"
        self.save()

    def serve_explorer(self):
        self.say("serving the page explorer")
        while not self.reset_requested():
            self.explore_tick()
            self.save_tick()
            self.idle(3, watch_kill=False)

    # --- saving to note_synth/inference/comprehension/ (seg) ---------------------------------------
    SAVE_ACTIVE = ("queued", "waiting_for_vm", "copying")

    def local_checkpoint(self, name):
        """files/train/run/<name>/ if it holds that checkpoint as the GPU VM last listed it, else None."""
        entry = (self.meta.get("checkpoints_local") or {}).get(name)
        remote = next((c for c in self.remote.get("checkpoints") or [] if c.get("name") == name), None)
        folder = os.path.join(self.files, "train", "run", name)
        if (entry and remote and entry.get("mtime") == remote.get("mtime")
                and all(os.path.isfile(os.path.join(folder, f)) for f in JOBS[self.job]["checkpoint_files"])):
            return folder
        return None

    def fetch_checkpoints(self):
        """best/ and last/ from the running GPU VM into files/train/run/: seg's adapters, marks' model.pt. Never raises."""
        files = JOBS[self.job]["checkpoint_files"]
        progress = self.remote.get("progress") or {}
        steps = {"best": progress.get("best_step"), "last": progress.get("step")}
        wanted = [c for c in self.remote.get("checkpoints") or [] if c.get("name") in steps and not self.local_checkpoint(c["name"])]
        if not wanted:
            return
        home = self.meta["remote_home"]
        self.say(f"copying {' and '.join(c['name'] + '/' for c in wanted)} back from the GPU VM")
        self.publish()
        try:
            self.pull([f"{home}/train/run/{c['name']}/{f}" for c in wanted for f in files])
        except (JLError, Failure, OSError, tarfile.TarError) as error:
            self.say(f"could not copy the checkpoints back: {str(error)[:300]}")
            return
        local = self.meta.setdefault("checkpoints_local", {})
        for c in wanted:
            folder = os.path.join(self.files, "train", "run", c["name"])
            if all(os.path.isfile(os.path.join(folder, f)) for f in files):
                local[c["name"]] = {"mtime": c["mtime"], "step": steps[c["name"]], "at": time.time(),
                                    **({"mean_f1": progress.get("best_mean_f1")} if c["name"] == "best" else {})}
        self.save()

    def fetch_mark_model(self):
        """marks: once training has ended, best/ and last/ copied back for the segmentation run to use. A run paused
        before the copy (on its budget) has its GPU VM resumed for a few minutes to make it, once. Never raises."""
        if self.job != "marks" or self.local_checkpoint("best") or self.local_checkpoint("last"):
            return
        if not {"best", "last"} & {c.get("name") for c in self.remote.get("checkpoints") or []}:
            return
        if self.running("gpu"):
            return self.fetch_checkpoints()
        if self.meta.get("mark_model_fetch"):
            return                                  # tried already; reported in state.json
        machine, cfg = self.meta["machines"]["gpu"], self.meta.get("config") or {}
        self.meta["mark_model_fetch"] = {"at": time.time()}
        self.say("copying the trained model back for Florence Tuning: resuming the GPU VM for a few minutes")
        try:
            inst = self.settle(machine["id"])
            if inst.get("status") == "Paused":
                self.open_segment("gpu", inst["machine_id"], machine.get("hourly") or 0.0)
                inst = self.jl.resume(inst["machine_id"], num_gpus=cfg.get("num_gpus", 1))
            elif inst.get("status") == "Running":
                self.open_segment("gpu", inst["machine_id"], machine.get("hourly") or 0.0)
            else:
                raise Failure(f"GPU VM {inst['machine_id']} is {inst.get('status')}")
            self.remember("gpu", inst)
            self.wait_for_ssh("gpu")
            self.refresh_home()
            self.fetch_checkpoints()
        except (JLError, Failure) as error:
            self.say(f"could not copy the model back: {str(error)[:300]}")
        finally:
            self.meta["mark_model_fetch"]["done"] = bool(self.local_checkpoint("best") or self.local_checkpoint("last"))
            try:
                self.pause("gpu", "model copied back")
            except JLError as error:
                self.meta["pause_error"] = f"gpu VM did not pause: {error}"
                self.say(f"PAUSE FAILED for the gpu VM — check the Jarvis dashboard: {error}")
            self.save()
            self.publish()

    def save_tick(self):
        """One look at the saves asked for. Never raises: saving must not fail the run."""
        if self.job != "seg":
            return
        try:
            self._save_tick()
        except Exception as error:      # noqa: BLE001
            self.say(f"saving: {type(error).__name__}: {error}"[:600])
            for s in (self.meta.get("saves") or {}).values():
                if s["state"] in self.SAVE_ACTIVE:
                    s.update(state="failed", message=f"{type(error).__name__}: {error}"[-400:], finished_at=time.time())
            self.save()
            self.publish()

    def _save_tick(self):
        saves = self.meta.setdefault("saves", {})
        changed = False
        queue = self.path("save", "requests")
        for name in sorted(os.listdir(queue)) if os.path.isdir(queue) else []:
            if not name.endswith(".json"):
                continue
            doc = load(os.path.join(queue, name))
            os.remove(os.path.join(queue, name))
            if doc and doc.get("checkpoint") in ("best", "last") and doc.get("name"):
                saves[doc["id"]] = {"id": doc["id"], "checkpoint": doc["checkpoint"], "name": doc["name"], "origin": "tracker",
                                    "state": "queued", "requested_at": doc.get("requested_at") or time.time()}
                changed = True
        wanted = (self.meta.get("config") or {}).get("save_to")
        if wanted and not self.meta.get("save_to_queued"):         # chosen at Start, asked for once training has ended
            saves["start"] = {"id": "start", "checkpoint": wanted["checkpoint"], "name": wanted["name"], "origin": "start",
                              "state": "queued", "requested_at": time.time()}
            self.meta["save_to_queued"] = changed = True

        active = [s for s in saves.values() if s["state"] in self.SAVE_ACTIVE]
        for s in [s for s in active if s["checkpoint"] not in {c.get("name") for c in self.remote.get("checkpoints") or []}]:
            s.update(state="failed", message=f"this run has no {s['checkpoint']}/ checkpoint", finished_at=time.time())
            changed = True
        active = [s for s in active if s["state"] in self.SAVE_ACTIVE]
        missing = [s for s in active if not self.local_checkpoint(s["checkpoint"])]
        if missing:
            changed |= self.save_fetch(missing)
        for s in active:
            if s["state"] in self.SAVE_ACTIVE and self.local_checkpoint(s["checkpoint"]):
                self.save_one(s)
                changed = True
        changed |= self.save_release_vm()
        if changed:
            self.save()
            self.publish()

    def save_fetch(self, missing):
        """Gets the checkpoints the saves need back from the GPU VM, resuming it through the explorer if it is paused."""
        ex = self.explore()
        since = self.meta.get("save_resumed_vm")
        if self.running("gpu") and self.explore_thread is None and ex["state"] in ("starting", "ready"):
            for s in missing:
                s.update(state="copying", message="copying the checkpoint back from the GPU VM")
            self.fetch_checkpoints()
            for s in missing:
                if not self.local_checkpoint(s["checkpoint"]):
                    s.update(state="failed", message=f"{s['checkpoint']}/ could not be copied back from the GPU VM",
                             finished_at=time.time())
            return True
        if ex["state"] == "paused" and since and ex.get("error") and (ex.get("since") or 0) >= since:
            for s in missing:          # the resume asked for these saves failed
                s.update(state="failed", message=ex["error"][:400], finished_at=time.time())
            self.meta["save_resumed_vm"] = None
            return True
        if self.meta["phase"] == "trained" and os.path.exists(self.path("TEST.json")):
            return False                # the test run is about to take the VM; asked again once it has finished
        if ex["state"] == "paused" and not os.path.exists(self.path("explore", "RESUME")):
            os.makedirs(self.path("explore"), exist_ok=True)
            open(self.path("explore", "RESUME"), "w").close()
            self.meta["save_resumed_vm"] = time.time()
            self.say("saving: resuming the GPU VM with one GPU to copy the checkpoint back")
        for s in missing:
            s.update(state="waiting_for_vm", message="resuming the GPU VM with one GPU to copy the checkpoint back")
        return True

    def save_one(self, s):
        ck, local = s["checkpoint"], self.meta["checkpoints_local"][s["checkpoint"]]
        cfg, progress = self.meta.get("config") or {}, self.remote.get("progress") or {}
        machine = self.meta["machines"].get("gpu") or {}
        step = local.get("step")
        record = inference_models.val_record(self.path("log.jsonl"), step) if step is not None else None
        chosen = record and (record.get("pages/mean_f1") if record.get("pages/mean_f1") is not None else record.get("rows/mean_f1"))
        run_info = {
            "checkpoint": "best (val mean F1)" if ck == "best" else "last (end of training)", "step": step,
            "source": f"{self.meta.get('remote_home')}/train/run/{ck} (GPU VM {machine.get('id')})",
            "saved_at": time.strftime("%Y-%m-%d %H:%M"), "run_id": self.meta.get("run_id"),
            "base_model": "microsoft/Florence-2-large (LoRA adapter)",
            "lora_rank": cfg.get("lora_rank", 16), "lora_alpha": cfg.get("lora_alpha", 2 * cfg.get("lora_rank", 16)),
            "trained_from": (cfg.get("init_model") or {}).get("name") or (
                f"{(cfg['continue_from'].get('archive') or 'previous run')} {cfg['continue_from']['checkpoint']}/"
                if cfg.get("continue_from") else None),
            "val_mean_f1": chosen, "best_step": progress.get("best_step"), "total_steps": progress.get("total_steps"),
            "epochs": cfg.get("epochs"), "best_val_mean_f1": progress.get("best_mean_f1"),
            "train_config": {"gpus": f"{cfg.get('num_gpus')} x {cfg.get('gpu_type')}", "batch_size": cfg.get("batch_size"),
                             "grad_accum": cfg.get("grad_accum"), "global_batch": progress.get("global_batch"),
                             "lr": cfg.get("lr"), "warmup_steps": cfg.get("warmup_steps"), "train_rows": progress.get("train_rows")}}
        stats = {"checkpoint": ck, "step": step, "val_step": record["step"], "val_pages": progress.get("val_pages"),
                 "val_rows": progress.get("val_rows"), "run_id": self.meta.get("run_id"), "val": record} if record else None
        try:
            folder = inference_models.save(s["name"], self.local_checkpoint(ck), run_info, stats)
        except (FileExistsError, ValueError, OSError) as error:
            s.update(state="failed", message=str(error), finished_at=time.time())
            self.say(f"not saved as inference/comprehension/{s['name']}: {error}")
            return
        s.update(state="done", message=None, folder=folder, step=step, val_mean_f1=chosen, finished_at=time.time())
        self.say(f"saved {ck}/ (step {step}) as inference/comprehension/{s['name']}")

    def save_release_vm(self):
        """Pauses a VM resumed only to copy checkpoints back, once no save needs it; not if pages were explored meanwhile."""
        since = self.meta.get("save_resumed_vm")
        if not since or any(s["state"] in self.SAVE_ACTIVE for s in self.meta["saves"].values()):
            return False
        ex = self.explore()
        if ex["state"] in ("resuming", "pausing") or self.explore_thread is not None:
            return False
        if ex["state"] in ("starting", "ready") and not any((r.get("requested_at") or 0) >= since for r in ex["requests"].values()):
            open(self.path("explore", "PAUSE"), "w").close()
            self.say("saving: done; pausing the GPU VM again")
        self.meta["save_resumed_vm"] = None
        return True

    # --- scrapping a run -------------------------------------------------------------------
    def reset_requested(self):
        return os.path.exists(self.path("RESET.json"))

    def wait_for_reset(self):
        while not self.reset_requested():
            if os.path.exists(self.path("KILL")):
                os.remove(self.path("KILL"))          # nothing is running to stop
            self.idle(3, watch_kill=False)

    def reset(self):
        """RESET.json: everything paused, the run's files archived, a new run waiting for Start on the same VMs. With
        {"continue_from": "best" | "last"} (seg), the new run's Start offers to continue from that checkpoint."""
        request = load(self.path("RESET.json")) or {}
        keep = request.get("continue_from") if self.job == "seg" else None
        continued = None
        try:
            if keep:
                if keep not in [c.get("name") for c in (self.remote.get("checkpoints") or [])]:
                    raise Failure(f"there is no {keep}/ checkpoint on the GPU VM to continue from")
                prev, progress = self.meta.get("config") or {}, self.remote.get("progress") or {}
                rank = prev.get("lora_rank", 16)
                continued = {"checkpoint": keep, "step": progress.get("best_step" if keep == "best" else "step"),
                             "val_mean_f1": progress.get("best_mean_f1") if keep == "best" else None,
                             "lora_rank": rank, "lora_alpha": prev.get("lora_alpha") or 2 * rank,
                             "run_id": self.meta.get("run_id")}
            self.say("restarting training: pausing everything and archiving the run's files")
            if self.job == "seg":
                self.explore_pause_for_test()
            self.pause_everything("the run was scrapped")
            if any(seg["end"] is None for seg in self.meta["billing"]):
                raise Failure(f"a VM did not pause ({self.meta.get('pause_error')}); check the Jarvis dashboard")
            archive = os.path.join(os.path.dirname(self.dir), "archive",
                                   f"{os.path.basename(self.dir)}-{time.strftime('%Y%m%d-%H%M%S')}")
            os.makedirs(archive)
            for name in sorted(os.listdir(self.dir)):
                if os.path.splitext(name)[1] in (".out", ".err", ".log"):
                    continue                            # held open by the processes writing them
                source = self.path(name)
                if name == "state.json":                # the tracker keeps reading it: copied, then rewritten
                    shutil.copy2(source, archive)
                    continue
                for attempt in range(20):               # the tracker server may be reading a file just then
                    try:
                        shutil.move(source, os.path.join(archive, name))
                        break
                    except PermissionError:
                        if attempt == 19:
                            raise
                        time.sleep(0.5)
        except Exception as error:      # noqa: BLE001 - the run is left as it was, ready to be scrapped again
            self.save()
            if self.reset_requested():
                os.replace(self.path("RESET.json"), self.path(f"RESET.failed.{int(time.time())}.json"))
            self.say(f"NOT RESTARTED: {type(error).__name__}: {error}")
            self.publish()
            return
        machines = self.meta["machines"]
        self.meta = self.new_meta(machines, self.args.data_vm)     # looked up by name again at the next Start
        self.meta["wipe_remote"] = bool((machines.get("gpu") or {}).get("id"))
        self.meta["scrapped"] = {"at": time.time(), "archive": archive}
        if continued:
            self.meta["continue_from"] = {**continued, "archive": os.path.basename(archive)}
        self.remote = {}
        self.explore_thread, self.explore_outcome, self.explore_polled = None, None, 0.0
        os.makedirs(self.files, exist_ok=True)
        self.save()
        self.say(f"training restarted, the old run's files are in {archive}; the next Start " +
                 (f"can continue from its {keep}/ checkpoint" if continued else "trains from the pretrained weights or a saved model"))
        self.publish()

    # --- main ------------------------------------------------------------------------------
    def main(self):
        """A run, then the next one each time the tracker scraps it."""
        while True:
            self.run()
            self.wait_for_reset()
            self.reset()

    def run(self):
        try:
            if self.meta["phase"] == "waiting_for_start":
                self.wait_for_start()
            if self.meta["phase"] == "provisioning":
                self.provision()
            if self.meta["phase"] in ("training", "stopping"):
                self.watch_training()
            if self.meta["phase"] in ("trained", "done"):
                self.fetch_mark_model()         # marks: the segmentation run can't start without it
            while self.meta["phase"] == "trained":
                self.publish()
                if self.wait_for_test() is False:
                    break
                if self.run_test() is False:
                    continue
            if self.meta["phase"] == "testing":
                self.run_test()
            self.publish()
            self.say(f"finished in phase {self.meta['phase']}")
            if self.job == "seg" and self.meta["phase"] == "done":
                self.serve_explorer()           # the explorer stays available after the test
            return 0
        except Exception as error:      # noqa: BLE001 - everything is paused and reported
            self.meta["error"] = f"{type(error).__name__}: {error}\n{traceback.format_exc()[-3000:]}"
            for entry in self.meta["stages"].values():      # the page shows where it stopped, not a stage still running
                if entry.get("state") == "running":
                    entry.update(state="failed", end=time.time(), detail=str(error)[:600])
            self.say(f"FAILED: {error}")
            self.pause_everything("the orchestrator failed")
            self.meta["phase"] = "failed"
            self.save()
            self.publish()
            return 1


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", choices=sorted(JOBS), default="seg",
                    help="seg: the segmentation run; marks: the mark separation CNN")
    ap.add_argument("--run-dir", default=None, help="default runs/train (seg) or runs/marks (marks)")
    ap.add_argument("--data-vm", type=int, default=None,
                    help="the dataset VM (paused after synthesis); default: found by --data-vm-name when needed")
    ap.add_argument("--data-vm-name", default="note-synth-dataset")
    ap.add_argument("--data-hourly", type=float, default=0.4105)
    ap.add_argument("--region", default="IN2")
    ap.add_argument("--storage", type=int, default=None, help="GPU VM disk, GB (default 200 seg, 100 marks)")
    ap.add_argument("--name", default=None, help="GPU VM name (default seg-train, marks-cnn)")
    ap.add_argument("--interval", type=int, default=60, help="seconds between status polls while running")
    ap.add_argument("--stop-wait", type=int, default=600, help="seconds a stop may take before the VM is paused anyway")
    ap.add_argument("--link-wait", type=int, default=300,
                    help="seconds the GPU VM is given to reach the dataset VM over SSH before the run fails")
    ap.add_argument("--remote-python", default="python3", help="python on the VM for the status and pack scripts")
    ap.add_argument("--remote-env", nargs="*", default=[], help="KEY=VALUE for the remote scripts (local test)")
    ap.add_argument("--extra-train-args", default="", help="one string appended to train.py's arguments (local test)")
    ap.add_argument("--explore-python", default="~/train/venv/bin/python",
                    help="the python on the GPU VM that runs the page explorer's model")
    ap.add_argument("--explore-worker-args", default="",
                    help="one string of extra explore_worker.py --serve arguments (local test: --stub)")
    ap.add_argument("--guard-port", type=int, default=None,
                    help="port of the one-orchestrator-per-job lock (local test beside a running orchestrator)")
    args = ap.parse_args()
    args.extra_train_args = shlex.split(args.extra_train_args)
    args.explore_worker_args = shlex.split(args.explore_worker_args)
    job = JOBS[args.job]
    args.run_dir = args.run_dir or os.path.join(HERE, "runs", job["run_dir"])
    args.storage = args.storage or job["storage"]
    args.name = args.name or job["name"]
    return args


if __name__ == "__main__":
    import socket
    arguments = parse_args()
    guard = socket.socket()
    try:
        # one orchestrator per job at a time: the login script may start a second
        guard.bind(("127.0.0.1", arguments.guard_port or JOBS[arguments.job]["guard_port"]))
    except OSError:
        sys.exit(f"another orchestrator.py --job {arguments.job} is already running")
    sys.exit(Orchestrator(arguments).main())
