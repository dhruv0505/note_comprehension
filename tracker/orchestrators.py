"""orchestrators.py — keeps the three orchestrators running for tracker_server.py, so none is started by hand.

  synth   synth/synth_orchestrator.py          lock port 8796, heartbeat synth/orchestrator.json
  seg     train/orchestrator.py                lock port 8798, heartbeat train/runs/train/state.json
  marks   train/orchestrator.py --job marks    lock port 8799, heartbeat train/runs/marks/state.json

One whose heartbeat is stale and whose lock port is free is started, hidden and detached (it outlives the
tracker server), its output appended to its .out file. One that keeps exiting is started again after 10 s,
a minute, then every 5 minutes, and its last output is shown on the page. A held lock with a stale heartbeat
is an orchestrator inside a long call, and is left alone. When an orchestrator's code is newer than its
process it is restarted onto the new code, but only while it is idle: no run going, nothing billed, no
request waiting. prepare() does all this straight away when a button needs an orchestrator.
"""
import ctypes
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SYNTH, TRAIN = os.path.join(HERE, "synth"), os.path.join(HERE, "train")
CLASS_TUNING = os.path.join(os.path.dirname(HERE), "note_synth", "class_tuning.py")
ALIVE_SECONDS = 45              # as the tracker pages count it
BACKOFF = (10, 60, 300)         # seconds before starting one again after its 1st, 2nd, 3rd+ exit in a row
TICK = 5
SETTLE = 10                     # seconds an orchestrator's code must stay unchanged before it is restarted onto it


def _job(script, args, run_files, port, code):
    return {"command": [script, *args], "cwd": os.path.dirname(script), "port": port, "code": code, **run_files}


JOBS = {
    "synth": _job(os.path.join(SYNTH, "synth_orchestrator.py"), [],
                  {"out": os.path.join(SYNTH, "synth_orchestrator.out"), "heartbeat": os.path.join(SYNTH, "orchestrator.json")},
                  8796, [os.path.join(SYNTH, "synth_orchestrator.py"), os.path.join(TRAIN, "jlcmd.py"), CLASS_TUNING]),
    "seg": _job(os.path.join(TRAIN, "orchestrator.py"), [],
                {"out": os.path.join(TRAIN, "runs", "train", "orchestrator.out"),
                 "heartbeat": os.path.join(TRAIN, "runs", "train", "state.json")},
                8798, [os.path.join(TRAIN, "orchestrator.py"), os.path.join(TRAIN, "jlcmd.py"), os.path.join(TRAIN, "inference_models.py")]),
    "marks": _job(os.path.join(TRAIN, "orchestrator.py"), ["--job", "marks"],
                  {"out": os.path.join(TRAIN, "runs", "marks", "orchestrator.out"),
                   "heartbeat": os.path.join(TRAIN, "runs", "marks", "state.json")},
                  8799, [os.path.join(TRAIN, "orchestrator.py"), os.path.join(TRAIN, "jlcmd.py"), os.path.join(TRAIN, "inference_models.py")]),
}
TRAIN_BUSY = ("provisioning", "training", "stopping", "testing")
TRAIN_REQUESTS = ("RESET.json", "KILL", os.path.join("explore", "RESUME"), os.path.join("explore", "PAUSE"))


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def tail(path, lines=12):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 6000))
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])[-1500:]


def lock_held(port):
    probe = socket.socket()
    try:
        probe.bind(("127.0.0.1", port))
        return False
    except OSError:
        return True
    finally:
        probe.close()


_kernel32 = None


def process_started(pid):
    """When process `pid` started, in unix seconds; None if it isn't running (or not on Windows)."""
    global _kernel32
    if os.name != "nt" or not pid:
        return None
    if _kernel32 is None:
        _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        _kernel32.OpenProcess.restype = ctypes.c_void_p
        _kernel32.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_ulonglong)] * 4
        _kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = _kernel32.OpenProcess(0x1000, False, int(pid))        # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    try:
        times = [ctypes.c_ulonglong() for _ in range(4)]
        if not _kernel32.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
            return None
        return times[0].value / 1e7 - 11644473600                   # FILETIME: 100 ns steps since 1601
    finally:
        _kernel32.CloseHandle(handle)


def idle(name, beat):
    """True when restarting this orchestrator can't cut anything short."""
    if name == "synth":
        st = load(os.path.join(SYNTH, "run", "orchestrator_state.json")) or {}
        return (beat.get("phase") not in ("starting", "running")
                and not (st.get("billing_started") and not st.get("billing_ended"))
                and not os.path.exists(os.path.join(SYNTH, "requests", "START.json")))
    run, phase = os.path.dirname(JOBS[name]["heartbeat"]), beat.get("phase")
    queued = os.path.join(run, "explore", "requests")
    # START.json and TEST.json stay once used (the page reads them as "already asked"): waiting only in their phase
    waiting = TRAIN_REQUESTS + (("START.json",) if phase in (None, "waiting_for_start") else ()) \
        + (("TEST.json",) if phase == "trained" else ())
    return (phase not in TRAIN_BUSY and not (beat.get("cost") or {}).get("hourly_now")
            and ((beat.get("explore") or {}).get("state") or "paused") == "paused"
            and not any(os.path.exists(os.path.join(run, f)) for f in waiting)
            and not (os.path.isdir(queued) and os.listdir(queued)))


def code_time(name):
    return max([os.path.getmtime(path) for path in JOBS[name]["code"] if os.path.exists(path)], default=0.0)


def outdated(name, beat):
    """True when the running orchestrator started before its code last changed."""
    if name == "synth" and not beat.get("accepts_settings"):
        return True
    started = process_started(beat.get("pid"))
    return started is not None and started < code_time(name)


class Supervisor:
    def __init__(self):
        self.lock = threading.Lock()
        self.jobs = {name: {"proc": None, "launched_at": 0.0, "failures": 0, "retry_at": 0.0, "replaced_pid": None,
                            "restarted_for": None, "state": "starting", "note": None} for name in JOBS}

    def start(self):
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while True:
            for name in JOBS:
                try:
                    with self.lock:
                        self.tick(name)
                except Exception as error:      # noqa: BLE001 - kept going; shown on the page
                    self.jobs[name]["note"] = f"supervisor: {type(error).__name__}: {error}"
            time.sleep(TICK)

    def say(self, name, text):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {name} orchestrator: {text}", flush=True)

    def prepare(self, name):
        """A button needs this orchestrator: start it, or restart it onto new code, now rather than at the next tick."""
        with self.lock:
            self.tick(name, pressed=True)

    def status(self, name):
        s = self.jobs[name]
        doc = {"state": s["state"], "note": s["note"]}
        if s["state"] == "failing":
            doc.update(retry_in=max(0, round(s["retry_at"] - time.time())), output=tail(JOBS[name]["out"]))
        return doc

    def tick(self, name, pressed=False):
        job, s, now = JOBS[name], self.jobs[name], time.time()
        beat = load(job["heartbeat"]) or {}
        alive = now - beat.get("heartbeat_at", 0) < ALIVE_SECONDS and beat.get("pid") != s["replaced_pid"]
        proc = s["proc"]
        if proc is not None and proc.poll() is not None:
            s["proc"] = None
            ran = now - s["launched_at"]
            s["failures"] = s["failures"] + 1 if ran < 600 else 1
            s["retry_at"] = now + BACKOFF[min(s["failures"], len(BACKOFF)) - 1]
            s["note"] = f"exited with code {proc.returncode} after {ran:.0f} s"
            self.say(name, s["note"])
        if alive:
            if s["failures"] and now - s["launched_at"] > 120:
                s["failures"], s["note"] = 0, None
            # once per change of its code (a file dated in the future would otherwise restart it at every tick), and
            # only once the code has been still for SETTLE seconds, so an edit touching several files lands whole
            changed = code_time(name)
            if (outdated(name, beat) and s["restarted_for"] != changed and time.time() - changed > SETTLE
                    and idle(name, beat)):
                s["restarted_for"] = code_time(name)
                self.restart(name, beat)
            else:
                s["state"] = "running"
                if not s["failures"]:
                    s["note"] = None
            return
        if s["proc"] is not None or lock_held(job["port"]):
            s["state"] = "starting" if s["proc"] is not None and now - s["launched_at"] < 120 else "busy"
            return
        if s["failures"] and now < s["retry_at"] and not pressed:
            s["state"] = "failing"
            return
        self.launch(name)

    def restart(self, name, beat):
        s, pid = self.jobs[name], beat.get("pid")
        started = process_started(pid)
        if started is None or started > beat.get("heartbeat_at", 0):
            return                              # gone already, or the pid now belongs to another process
        s["state"], s["note"] = "restarting", "its code changed"
        self.say(name, f"restarting it (pid {pid}): its code changed")
        s["proc"], s["replaced_pid"] = None, pid        # stopped on purpose: not counted as an exit
        try:
            os.kill(int(pid), signal.SIGTERM)
        except OSError:
            pass
        for _ in range(40):
            if not lock_held(JOBS[name]["port"]):
                break
            time.sleep(0.25)
        else:
            return                              # still holding its lock: looked at again next tick
        self.launch(name)

    def launch(self, name):
        job, s = JOBS[name], self.jobs[name]
        os.makedirs(os.path.dirname(job["out"]), exist_ok=True)
        command = [sys.executable, "-u", *job["command"]]
        with open(job["out"], "a", encoding="utf-8") as out:
            out.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] started by tracker_server.py\n")
            out.flush()
            options = dict(cwd=job["cwd"], stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
            if os.name != "nt":
                s["proc"] = subprocess.Popen(command, start_new_session=True, **options)
            else:
                hidden = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
                try:        # CREATE_BREAKAWAY_FROM_JOB: not ended with the job the server may run in
                    s["proc"] = subprocess.Popen(command, creationflags=hidden | 0x01000000, **options)
                except OSError:
                    s["proc"] = subprocess.Popen(command, creationflags=hidden, **options)
        s["launched_at"], s["state"] = time.time(), "starting"
        self.say(name, f"started it (pid {s['proc'].pid})")
