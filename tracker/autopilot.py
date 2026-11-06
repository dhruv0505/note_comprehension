"""
autopilot.py — walks the whole loop through the tracker, one step per call, so nobody has to press its buttons.

    python autopilot.py              # one pass: look at where things are, do the next thing if it is due, exit
    python autopilot.py --status     # print autopilot_status.json

Run every few minutes by Windows Task Scheduler (the "note-comprehension autopilot" task; see README). Each pass is
quick and safe to repeat: it reads the tracker's own state and only ever asks for what the tracker's buttons ask for,
through the same HTTP API (tracker_server.py, http://127.0.0.1:8765), plus synth/bootstrap_dataset_vm.py when there
is no dataset VM yet. A lock file keeps two passes from overlapping.

Steps, in order (autopilot_status.json "step"):
  ascii       IAM-OnDB's transcriptions (ascii-all.tar.gz, a login-only download) found in Downloads or the
              checkout's root, copied to note_synth/data/iamondb_ascii.tar
  splits      note_synth/iam_writers.py, then note_synth/dataset_splits.py: data/iamondb_writers.json and splits.json
  bootstrap   synth/bootstrap_dataset_vm.py in the background: the dataset VM made, archives fetched, caches built
  synth       Regenerate all (full) with the last full dataset's settings (SYNTH_REQUEST), waited for
  marks       the saved mark separator (MARK_SEPARATOR) picked for Florence Tuning
  train       Florence Tuning's Start with autopilot_config.json's "florence" settings, once the account can pay
              for its budget; waits here while that file has none
  training    waited for: the run saves its best checkpoint to inference/comprehension/<save_to> when it ends
  done        nothing more to do (the README is updated from the saved model's stats.json by hand)
A step that fails is retried on later passes up to MAX_ATTEMPTS times, then the autopilot stops with "error" set.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NOTE_SYNTH = os.path.join(ROOT, "note_synth")
DATA = os.path.join(NOTE_SYNTH, "data")
STATUS = os.path.join(HERE, "autopilot_status.json")
CONFIG = os.path.join(HERE, "autopilot_config.json")
LOG = os.path.join(HERE, "autopilot.log")
LOCK = os.path.join(HERE, "autopilot.lock")
BOOTSTRAP = os.path.join(HERE, "synth", "bootstrap_dataset_vm.py")
BOOTSTRAP_STATE = os.path.join(HERE, "synth", "bootstrap_state.json")
TRACKER = "http://127.0.0.1:8765"
STEPS = ["ascii", "splits", "bootstrap", "synth", "marks", "train", "training", "done"]
MAX_ATTEMPTS = 3
ASCII_NAMES = ("ascii-all.tar.gz", "ascii-all.tar", "ascii-all*.tar.gz", "ascii*.tar.gz", "iamondb_ascii*.tar*")
ASCII_PLACES = (os.path.join(os.path.expanduser("~"), "Downloads"), ROOT, DATA)
# the settings the last full dataset was built with: a full run with recognition, then the additions run's
# textline pool passes and geometry figures without a page (synth/runs_archive/20260915-*)
SYNTH_REQUEST = {"mode": "full", "confirm": True, "recognition": {"on": True, "pool_epochs": 10},
                 "mark_crop": "box", "geometry_per_page": 2.0, "tuning": {}}
MARK_SEPARATOR = "marks"
FLOWCHART_FLOOR = {"train": 0.4, "val": 0.8, "test": 0.8}     # flowcharts per page a finished dataset must reach
BUDGET_MARGIN = 1.05         # training budget kept this far above the projected cost of finishing
BUDGET_FINISH = 0.15         # hours of the billed rate allowed for saving and copying the checkpoints back
BUDGET_RESERVE = 1.0         # dollars of the balance never given to the training budget (paused VMs bill storage)
MAX_RESUMES = 6              # carry-ons of one failed dataset run, one a pass, before stopping for a person
UPSIZE_FROM = 16             # a dataset run on fewer vCPUs than this is moved to a bigger VM when one is free
UPSIZE_EVERY = 1500          # seconds between looks at what IN2 has free
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)       # no console window for the programs a pass runs


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def say(text):
    line = f"[{now()}] {text}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write(path, data):
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(path + ".tmp", path)


def api(method, path, job, body=None, timeout=120):
    url = f"{TRACKER}{path}?job={job}"
    data = json.dumps(body or {}).encode() if method == "POST" else None
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            return reply.status, json.loads(reply.read() or b"{}")
    except urllib.error.HTTPError as error:
        try:
            return error.code, json.loads(error.read() or b"{}")
        except ValueError:
            return error.code, {"error": str(error)}


def pid_alive(pid):
    if not pid:
        return False
    out = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}", "/NH"], capture_output=True, text=True,
                         creationflags=NO_WINDOW).stdout
    return str(int(pid)) in out


def flowchart_problem():
    """What is wrong with a finished dataset's flowcharts, or None: each split's flowcharts per page
    (synth/run/class_counts.json) against FLOWCHART_FLOOR. A split that came out with too few once took
    a whole val set's flowchart scores down to noise."""
    counts = load(os.path.join(HERE, "synth", "run", "class_counts.json")) or {}
    rates = {split: ((entry or {}).get("per_page") or {}).get("flowchart")
             for split, entry in (counts.get("splits") or {}).items()}
    if not rates:
        return "its class counts are missing"
    low = {s: r for s, r in rates.items() if r is None or r < FLOWCHART_FLOOR.get(s, 0)}
    return f"too few flowcharts per page: {low} (floors {FLOWCHART_FLOOR})" if low else None


def ensure_tracker():
    """tracker_server.py started, hidden, as start_trackers.vbs starts it, when it isn't answering (after a reboot,
    say): it keeps the orchestrators running itself. A second copy exits at once, so a false alarm costs nothing."""
    try:
        urllib.request.urlopen(f"{TRACKER}/api/status?job=synth", timeout=30).read()
        return
    except (OSError, urllib.error.URLError):
        pass
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW    # a hidden console its children share
    log = open(os.path.join(HERE, "tracker_server.log"), "a")
    subprocess.Popen([sys.executable.replace("pythonw.exe", "python.exe"), "-u", "tracker_server.py"], cwd=HERE,
                     stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags)
    say("the tracker was not answering: started tracker_server.py")
    time.sleep(20)


class Autopilot:
    def __init__(self):
        self.st = load(STATUS) or {"step": STEPS[0], "attempts": {}, "history": []}
        self.cfg = load(CONFIG) or {}

    def save(self, message=None):
        if message:
            self.st["message"] = message
        self.st["updated_at"] = now()
        write(STATUS, self.st)

    def advance(self, detail):
        step = self.st["step"]
        self.st["history"].append({"step": step, "done_at": now(), "detail": detail})
        self.st["step"] = STEPS[STEPS.index(step) + 1]
        self.st["error"] = None
        say(f"{step}: done ({detail}); next {self.st['step']}")
        self.save(f"{step} done: {detail}")

    def fail(self, detail):
        step = self.st["step"]
        n = self.st["attempts"].get(step, 0) + 1
        self.st["attempts"][step] = n
        self.st["error"] = f"{step} failed (attempt {n} of {MAX_ATTEMPTS}): {detail}"[-3000:]
        say(self.st["error"])
        self.save()

    # --- steps --------------------------------------------------------------------------------------
    def step_ascii(self):
        target = os.path.join(DATA, "iamondb_ascii.tar")
        if not os.path.exists(target):
            found = None
            for place in ASCII_PLACES:
                for pattern in ASCII_NAMES:
                    for path in glob.glob(os.path.join(place, pattern)):
                        if os.path.abspath(path) != os.path.abspath(target) and not path.endswith((".crdownload", ".part")):
                            found = found or path
            if not found:
                return self.save("waiting for IAM-OnDB's ascii-all.tar.gz in Downloads or the checkout folder")
            try:
                with tarfile.open(found, "r:*") as tar:
                    names = [m.name for m in tar.getmembers()[:2000] if m.name.endswith(".txt")]
            except (tarfile.TarError, OSError, EOFError) as error:
                return self.save(f"{found} is not a readable archive yet ({error}); waiting")
            if not names:
                return self.fail(f"{found} holds no .txt transcriptions")
            shutil.copyfile(found, target + ".tmp")
            os.replace(target + ".tmp", target)
            say(f"copied {found} to {target}")
        self.advance(f"{os.path.getsize(target) / 1e6:.1f} MB of transcriptions")

    def step_splits(self):
        before = load(os.path.join(NOTE_SYNTH, "splits.json")) or {}
        for script in ("iam_writers.py", "dataset_splits.py"):
            result = subprocess.run([sys.executable, "-W", "ignore", script], cwd=NOTE_SYNTH, capture_output=True, creationflags=NO_WINDOW,
                                    text=True, encoding="utf-8", errors="replace", timeout=3600)
            with open(os.path.join(HERE, f"autopilot_{script[:-3]}.log"), "w", encoding="utf-8") as f:
                f.write(result.stdout + result.stderr)
            if result.returncode != 0:
                return self.fail(f"{script} exited {result.returncode}: {(result.stdout + result.stderr)[-1500:]}")
        after = load(os.path.join(NOTE_SYNTH, "splits.json")) or {}
        old, new = before.get("iam_forms") or {}, after.get("iam_forms") or {}
        moved = sum(1 for form, split in new.items() if old.get(form) not in (None, split))
        self.st["iam_forms_moved"] = moved
        self.advance(f"summary {json.dumps(after.get('summary'))}; {moved} IAM forms changed split from the last splits.json")

    def step_bootstrap(self):
        boot = load(BOOTSTRAP_STATE) or {}
        if boot.get("phase") == "done":
            return self.advance(f"dataset VM {boot.get('machine_id')} set up for ${boot.get('cost', 0):.2f}")
        if boot.get("phase") == "running" and pid_alive(boot.get("pid")):
            return self.save(f"setting up the dataset VM (${boot.get('cost', 0):.2f} so far); see tracker/synth/bootstrap.log")
        if boot.get("phase") == "failed":
            if self.st.get("bootstrap_failure_seen") != boot.get("started_at"):
                self.st["bootstrap_failure_seen"] = boot.get("started_at")
                self.fail(boot.get("error") or "bootstrap failed")
                if self.st["attempts"].get("bootstrap", 0) >= MAX_ATTEMPTS:
                    return
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW    # a hidden console its children share
        with open(os.path.join(HERE, "synth", "bootstrap.out"), "a") as out:
            proc = subprocess.Popen([sys.executable, BOOTSTRAP, "--budget", str(self.cfg.get("bootstrap_budget", 3.0))],
                                    cwd=os.path.dirname(BOOTSTRAP), stdout=out, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, creationflags=flags)
        say(f"started the dataset VM bootstrap (pid {proc.pid})")
        self.save("setting up the dataset VM")

    def step_synth(self):
        code, status = api("GET", "/api/status", "synth")
        if code != 200:
            return self.save(f"the tracker is not answering ({code}); waiting")
        orch = status.get("orchestrator") or {}
        run_id = self.st.get("synth_run_id")
        mine = run_id and orch.get("run_id") == run_id
        phase = orch.get("phase")
        if mine and phase in ("starting", "running"):
            if phase == "running":
                self.maybe_upsize(status)
            return self.save(f"dataset run {run_id}: {orch.get('message') or phase}")
        if mine and phase in ("finished", "failed"):
            reason = status.get("pause_reason")
            if reason == "finished":
                problem = flowchart_problem()
                if problem:
                    # no training on a dataset whose flowcharts went missing: the step stays failed for a person
                    self.st["attempts"]["synth"] = MAX_ATTEMPTS
                    return self.fail(f"dataset run {run_id} finished but {problem}; not training on it")
                return self.advance(f"dataset run {run_id} finished, ${status.get('cost') or 0:.2f}")
            if reason and reason.startswith("upsizing"):
                code, reply = api("POST", "/api/resume", "synth")
                say(f"dataset run {run_id} paused to move to more vCPUs; carrying it on: {code} {reply}")
                return self.save("dataset run moving to a bigger VM")
            # A run that failed after its VM was ready is carried on, never started again: a new full run's first
            # stage deletes the pages the old one made. Jarvis itself can be down for an hour (2026-11-05: SSH and its
            # API both gone 14:00-15:15), so the carry-on is retried on later passes, up to MAX_RESUMES times.
            resumes = self.st.get("synth_resumes", 0)
            if status.get("resumable") and resumes < MAX_RESUMES:
                self.st["synth_resumes"] = resumes + 1
                code, reply = api("POST", "/api/resume", "synth")
                say(f"dataset run {run_id} ended ({reason}); carrying it on ({resumes + 1} of {MAX_RESUMES}): "
                    f"{code} {reply}")
                return self.save(f"dataset run ended ({reason}); asked to carry it on")
            if status.get("resumable"):
                # stop for a person rather than start a fresh run over the pages already made
                self.st["attempts"]["synth"] = MAX_ATTEMPTS
                return self.fail(f"dataset run {run_id} could not be carried on after {MAX_RESUMES} tries: {reason}; "
                                 f"{orch.get('error')}")
            self.st.pop("synth_run_id", None)
            self.st["synth_resumes"] = 0
            return self.fail(f"dataset run {run_id} ended: {reason}; {orch.get('error')}")
        if run_id and not mine and time.time() - self.st.get("synth_requested_at", 0) < 600:
            return self.save("waiting for the dataset run to start")
        if self.st["attempts"].get("synth", 0) >= MAX_ATTEMPTS:
            return
        code, reply = api("POST", "/api/start", "synth", SYNTH_REQUEST)
        if code != 200:
            return self.fail(f"Regenerate all was refused: {code} {reply}")
        self.st["synth_requested_at"] = time.time()
        for _ in range(60):             # the orchestrator takes the request within a few seconds
            time.sleep(5)
            orch = (api("GET", "/api/status", "synth")[1] or {}).get("orchestrator") or {}
            if orch.get("phase") in ("starting", "running") and orch.get("mode") == "full":
                self.st["synth_run_id"] = orch.get("run_id")
                break
        say(f"asked for Regenerate all: run {self.st.get('synth_run_id')}")
        self.save("dataset run requested")

    def maybe_upsize(self, status):
        """A full dataset run going on fewer than UPSIZE_FROM vCPUs (IN2 had nothing bigger when it started) is moved
        to the biggest CPU VM IN2 has free now, if that is at least twice its size and more than an hour of pages is
        left: the orchestrator pauses it (requests/UPSIZE.json) and step_synth carries it on, which resumes the VM at
        the biggest size free. Looked at once every UPSIZE_EVERY seconds."""
        if time.time() - self.st.get("upsize_checked_at", 0) < UPSIZE_EVERY:
            return
        self.st["upsize_checked_at"] = time.time()
        try:
            cores = int(str(status.get("vm") or "").split()[0])
        except (ValueError, IndexError):
            return
        eta = (status.get("speed") or {}).get("eta_s") or 0
        if cores >= UPSIZE_FROM or status.get("mode") != "full" or eta < 3600:
            return
        sys.path.insert(0, os.path.join(HERE, "train"))
        from jlcmd import JL, JLError
        try:
            offers = JL().json("cpus", timeout=120).get("combinations") or []
        except JLError as error:
            return say(f"could not read CPU VM sizes: {str(error)[-200:]}")
        free = [o["vcpus"] for o in offers if o.get("available") and (o.get("regions") or {}).get("IN2")
                and o["vcpus"] >= 2 * cores and o["vcpus"] <= 32]
        if not free:
            return
        with open(os.path.join(HERE, "synth", "requests", "UPSIZE.json"), "w") as f:
            json.dump({"requested_at": time.time(), "from": cores, "free": free}, f)
        say(f"IN2 has {max(free)} vCPU VMs free and the dataset run is on {cores} with {eta / 3600:.1f} h left: "
            "asked to move it to a bigger VM")

    def step_marks(self):
        code, reply = api("POST", "/api/mark_model", "marks", {"name": MARK_SEPARATOR})
        if code != 200:
            return self.fail(f"picking the mark separator {MARK_SEPARATOR}: {code} {reply}")
        self.advance(f"inference/mark_separation/{MARK_SEPARATOR}")

    def step_train(self):
        florence = self.cfg.get("florence")
        if not florence:
            return self.save("dataset ready; waiting for autopilot_config.json's florence settings (the budget)")
        code, account = api("GET", "/api/account", "seg")
        balance = (account or {}).get("balance")
        if florence.get("budget") == "balance":
            # whatever the account holds, less a little for the paused VMs' storage; raised through /api/budget if
            # the measured speed says the epochs need more and the account is topped up
            if balance is None:
                return self.save("could not read the Jarvis balance; waiting")
            florence = dict(florence, budget=round(balance - float(self.cfg.get("reserve", 1.5)), 2))
            floor = float(florence.get("min_balance") or 10)
            if balance < floor:
                return self.save(f"the Jarvis balance ${balance:.2f} is below the ${floor:.2f} this run waits for; "
                                 "waiting for a top-up")
        if balance is not None and balance < florence["budget"]:
            return self.save(f"the Jarvis balance ${balance:.2f} is short of the ${florence['budget']:.2f} budget; "
                             "waiting for a top-up")
        code, state = api("GET", "/api/status", "seg")
        phase = (state or {}).get("phase")
        if phase and phase != "waiting_for_start":
            self.st["train_run_id"] = (state or {}).get("run_id")
            return self.advance(f"training already {phase}")
        code, reply = api("POST", "/api/start", "seg", florence)
        if code != 200:
            return self.fail(f"Start was refused: {code} {reply}")
        time.sleep(10)
        self.st["train_run_id"] = (api("GET", "/api/status", "seg")[1] or {}).get("run_id")
        self.advance(f"Florence Tuning started: {json.dumps(florence)}")

    def step_training(self):
        code, state = api("GET", "/api/status", "seg")
        if code != 200:
            return self.save(f"the tracker is not answering ({code}); waiting")
        phase = state.get("phase")
        cost = (state.get("cost") or {}).get("total")
        progress = (state.get("remote") or {}).get("progress") or {}
        self.st["training"] = {"phase": phase, "cost": cost, "step": progress.get("step"),
                               "total_steps": progress.get("total_steps"), "step_seconds": progress.get("step_seconds"),
                               "eta_s": progress.get("eta_s"), "message": state.get("message"), "error": state.get("error")}
        # "stopped": the run reached its budget (the user's "spend down to the reserve"), which saves best/ all the same
        if phase in ("trained", "done") or (phase == "stopped" and "budget" in str(state.get("pause_reason") or "")):
            saves = state.get("saves") or []
            if saves and all(s.get("state") in ("done", "failed") for s in saves):
                return self.advance(f"training ended ({phase}), ${cost}; saves: "
                                    + ", ".join(f"{s.get('name')} {s.get('state')}" for s in saves))
            return self.save(f"training ended ({phase}); saving the checkpoint")
        if phase in ("stopped", "failed"):
            return self.fail(f"training {phase}: {state.get('error') or state.get('pause_reason')}")
        if phase == "training":
            self.keep_budget(state, progress)
        self.save(f"training: {phase}, step {progress.get('step')} of {progress.get('total_steps')}, ${cost}")

    def keep_budget(self, state, progress):
        """The run's budget kept at what finishing its epochs will cost, as far as the account allows.

        The projection is what has been spent, plus the run's own estimate of the time left (steps and the
        validations still to come, train.py's eta_s) at the rate being billed now, plus BUDGET_FINISH of that rate
        for saving and copying the checkpoints back, all times BUDGET_MARGIN. When that is over the budget the budget
        is raised (POST /api/budget, which also moves the GPU VM's own deadline), as far as the balance pays for:
        the tracker refuses a budget whose unspent part is more than the balance. What the balance cannot cover is
        kept in st["budget_short"] for the person to be told; once they top up, the next pass raises it the rest of
        the way, so a top-up is never left stopped by the old budget."""
        cost = state.get("cost") or {}
        spent, hourly = float(cost.get("total") or 0.0), float(cost.get("hourly_now") or 0.0)
        budget = float(cost.get("budget") or (state.get("config") or {}).get("budget") or 0.0)
        eta, steps = progress.get("eta_s"), progress.get("step") or 0
        if not (eta and hourly and budget) or steps < 50:         # the estimate settles after the first steps
            return
        need = round((spent + (eta / 3600.0 + BUDGET_FINISH) * hourly) * BUDGET_MARGIN, 2)
        self.st["budget_projection"] = {"need": need, "budget": budget, "spent": round(spent, 2), "hourly": hourly,
                                        "eta_h": round(eta / 3600.0, 2), "at": now()}
        if need <= budget:
            self.st["budget_short"] = 0.0
            return
        code, account = api("GET", "/api/account", "seg")
        balance = (account or {}).get("balance")
        if balance is None:
            return
        affordable = round(spent + float(balance) - float(self.cfg.get("reserve", BUDGET_RESERVE)), 2)
        target = min(need, affordable)
        self.st["budget_short"] = round(max(0.0, need - affordable), 2)
        if target > budget + 0.5:
            code, reply = api("POST", "/api/budget", "seg", {"budget": target})
            say(f"training needs about ${need:.2f} to finish (spent ${spent:.2f}, {eta / 3600:.2f} h left at "
                f"${hourly:.2f}/h); budget ${budget:.2f} -> ${target:.2f}: {code} {reply}")
        if self.st["budget_short"]:
            say(f"the balance ${balance:.2f} leaves training ${self.st['budget_short']:.2f} short of finishing")

    def run(self):
        step = self.st["step"]
        if step == "done":
            return self.save("everything is done")
        if self.st["attempts"].get(step, 0) >= MAX_ATTEMPTS:
            return self.save(f"stopped: {step} failed {MAX_ATTEMPTS} times; see autopilot.log")
        getattr(self, f"step_{step}")()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.status:
        print(json.dumps(load(STATUS), indent=1))
        return
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        if time.time() - os.path.getmtime(LOCK) < 3600:
            return
        os.remove(LOCK)             # left by a pass that died
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        ensure_tracker()
        Autopilot().run()
    except Exception as error:      # noqa: BLE001 - logged; the next pass tries again
        say(f"pass failed: {type(error).__name__}: {error}")
    finally:
        os.remove(LOCK)


if __name__ == "__main__":
    main()
