"""budget_guard.py — keeps a training run on this GPU VM inside its budget when the tracker can't.

    python3 ~/train/code/remote/budget_guard.py        # started beside train_remote.sh by orchestrator.py

orchestrator.py writes ~/train/guard.json at launch and whenever the budget changes:
    {"deadline": epoch seconds, "machine_id": this VM's id, "api_base": Jarvis's backend for its region,
     "stop_wait": seconds, "unwatched": seconds}
and puts the Jarvis API key in ~/train/.jl_key. Every CHECK seconds this reads guard.json again and:

  deadline   once the deadline has passed while train.py runs, touches run/STOP, so the run saves last/ and exits;
             when it has, or stop_wait later if it hasn't, pauses this VM. Past the deadline with no train.py
             running, pauses it at once.
  unwatched  once training has ended (stages/train.json done or failed) and no status poll (train_status.py writes
             ~/train/.last_poll) has come for `unwatched` seconds, pauses this VM: a finished run is not left billing
             while the tracker is away.

Before pausing it writes ~/train/guard_paused.json {at, reason}, which the orchestrator reads when it next resumes the
VM, to end the run's billing there. The pause is Jarvis's own API call (jarvislabs Instances.pause:
POST <api_base>templates/vm/pause?machine_id=<id>, a Bearer token), made with the standard library only. The
orchestrator kills this process whenever it pauses the VM itself.
"""
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

ROOT = os.path.join(os.path.expanduser("~"), "train")
CHECK = 30


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + message, flush=True)


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def training_running():
    return subprocess.run(["pgrep", "-f", "train.py"], capture_output=True).returncode == 0


def last_poll(default):
    try:
        return os.path.getmtime(os.path.join(ROOT, ".last_poll"))
    except OSError:
        return default


def pause(cfg, reason):
    """True once Jarvis has accepted the pause."""
    with open(os.path.join(ROOT, "guard_paused.json"), "w") as f:
        json.dump({"at": time.time(), "reason": reason}, f)
    try:
        key = open(os.path.join(ROOT, ".jl_key")).read().strip()
    except OSError:
        log(f"would pause ({reason}) but there is no API key in ~/train/.jl_key")
        return False
    url = f"{cfg['api_base'].rstrip('/')}/templates/vm/pause?machine_id={int(cfg['machine_id'])}"
    request = urllib.request.Request(url, data=b"", method="POST", headers={"Authorization": f"Bearer {key}"})
    log(f"pausing this VM ({cfg['machine_id']}): {reason}")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            log(f"pause accepted: {response.status} {response.read(500).decode('utf-8', 'replace')}")
            return True
    except urllib.error.HTTPError as error:
        log(f"pause refused: {error.code} {error.read(500).decode('utf-8', 'replace')}")
    except (urllib.error.URLError, OSError) as error:
        log(f"pause failed: {error}")
    return False


def main():
    started, stop_sent = time.time(), None
    # orchestrator.py stops this by pid: matching the name would also match the shell that starts it, whose command
    # line holds this script's path
    try:
        with open(os.path.join(ROOT, "guard.pid"), "w") as f:
            f.write(str(os.getpid()))
    except OSError:
        pass
    log("guarding the run's budget")
    while True:
        cfg = load(os.path.join(ROOT, "guard.json")) or {}
        if cfg.get("stop"):
            log("told to stop")
            return
        now, running = time.time(), training_running()
        stage = (load(os.path.join(ROOT, "stages", "train.json")) or {}).get("state")
        deadline = cfg.get("deadline")
        reason = None
        if deadline and now >= deadline:
            if running and stop_sent is None:
                os.makedirs(os.path.join(ROOT, "run"), exist_ok=True)
                open(os.path.join(ROOT, "run", "STOP"), "w").close()
                stop_sent = now
                log("the budget's deadline has passed: asked the run to save and stop")
            elif not running:
                reason = "the budget's deadline passed" + (" and the run stopped" if stop_sent else "")
            elif now - stop_sent >= cfg.get("stop_wait", 360):
                reason = f"the budget's deadline passed and the run did not stop within {cfg.get('stop_wait', 360)}s"
        elif stage in ("done", "failed") and not running and now - last_poll(started) >= cfg.get("unwatched", 1200):
            reason = f"training {stage} and no status poll for {int(now - last_poll(started))}s"
        if reason and cfg.get("machine_id") and cfg.get("api_base") and pause(cfg, reason):
            return
        time.sleep(CHECK)


if __name__ == "__main__":
    main()
