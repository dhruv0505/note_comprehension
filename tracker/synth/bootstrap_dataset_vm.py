"""
bootstrap_dataset_vm.py — create the dataset VM from nothing, as the first ever run did by hand.

    python bootstrap_dataset_vm.py            # create (or resume) note-synth-dataset, set it up, pause it

synth_orchestrator.py only ever resumes a dataset VM that already has the source archives in ~/run/note_synth/data
and the split caches in ~/run/cache. When there is no such VM (the Jarvis account was emptied), this makes one:
  1. creates a CPU VM named DATA_VM_NAME in IN2 (16 vCPU / 64 GB / 100 GB), or resumes the one there is
  2. uploads the generator's code and splits.json to ~/run/note_synth/, and the archives that only this machine
     has (UPLOADS, from note_synth/data/) to ~/run/note_synth/data/
  3. starts remote/bootstrap_remote.sh there, which installs numpy and Pillow, fetches MathWriting, NIST SD19 and
     a QuickDraw subset onto the VM, and builds the split caches
  4. polls it until ~/bootstrap/ALL_DONE or ~/bootstrap/FAILED, then pauses the VM either way

Resumable: every step records itself in bootstrap_state.json, and the remote script skips the stages it finished.
Stops (pausing the VM) when its cost reaches --budget. Progress is in bootstrap.log.
"""
import argparse
import json
import os
import shlex
import sys
import tarfile
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "train"))
from jlcmd import JL, JLError  # noqa: E402

import synth_orchestrator as so  # noqa: E402 - the dataset VM's name, region and prices

NOTE_SYNTH = so.NOTE_SYNTH
DATA = os.path.join(NOTE_SYNTH, "data")
STATE = os.path.join(HERE, "bootstrap_state.json")
LOG = os.path.join(HERE, "bootstrap.log")
REMOTE_SCRIPT = os.path.join(HERE, "remote", "bootstrap_remote.sh")
UPLOADS = ("iamondb_linestrokes.tar.gz", "iamondb_ascii.tar", "iamondb_writers.json", "deepwriting.zip", "fc.zip")
SIZE = {"vcpus": 16, "ram": 64, "storage": 100}
CREATE_SIZES = ((16, 64), (8, 32), (4, 16))
POLL = 60


def say(text):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {text}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state():
    try:
        with open(STATE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(state):
    with open(STATE + ".tmp", "w") as f:
        json.dump(state, f, indent=1)
    os.replace(STATE + ".tmp", STATE)


class Bootstrap:
    def __init__(self, budget):
        self.jl = JL()
        self.budget = budget
        self.state = load_state()
        self.state.setdefault("phase", "starting")
        self.state.setdefault("billed_seconds", 0.0)

    def save(self):
        save_state(self.state)

    def cost(self):
        running = time.time() - self.state["running_since"] if self.state.get("running_since") else 0.0
        return (self.state["billed_seconds"] + running) / 3600 * so.vm_hourly(self.state.get("vcpus") or SIZE["vcpus"])

    def stop_billing(self):
        if self.state.get("running_since"):
            self.state["billed_seconds"] += time.time() - self.state["running_since"]
            self.state["running_since"] = None
            self.save()

    def exec(self, command, timeout=300, check=True):
        code, out, err = self.jl.exec(self.state["machine_id"], command, timeout=timeout)
        if check and code != 0:
            raise RuntimeError(f"command failed ({code}): {command[:120]}: {(out + err).strip()[-600:]}")
        return code, out, err

    def vm(self):
        """The dataset VM running, created if there is none."""
        found = [i for i in self.jl.list() if i.get("name") == so.DATA_VM_NAME]
        if len(found) > 1:
            raise RuntimeError(f"{len(found)} instances are named {so.DATA_VM_NAME!r}")
        if not found:
            inst = None
            for vcpus, ram in CREATE_SIZES:     # whichever IN2 has free: the bootstrap is mostly downloads
                say(f"creating the dataset VM {so.DATA_VM_NAME} ({vcpus} vCPU / {ram} GB / {SIZE['storage']} GB) "
                    f"in {so.REGION}")
                try:
                    inst = self.jl.json("create", "--vm", "--cpu", "--vcpus", vcpus, "--ram", ram,
                                        "--storage", SIZE["storage"], "--name", so.DATA_VM_NAME,
                                        "--region", so.REGION, "--yes", timeout=2400)
                    self.state["vcpus"] = vcpus
                    break
                except JLError as error:
                    say(f"not at that size: {str(error)[-160:]}")
            if inst is None:
                raise RuntimeError(f"no CPU VM size in {CREATE_SIZES} is free in {so.REGION}")
            self.state["created_at"] = time.time()
        else:
            inst = found[0]
            if inst.get("status") == "Paused":
                say(f"resuming the dataset VM {inst['machine_id']}")
                inst = self.jl.resume(int(inst["machine_id"]))
        self.state["machine_id"] = int(inst["machine_id"])
        if not self.state.get("running_since"):
            self.state["running_since"] = time.time()
        self.save()
        end = time.time() + 900
        while True:
            try:
                if self.jl.exec(self.state["machine_id"], "true", timeout=90)[0] == 0:
                    break
            except JLError:
                pass
            if time.time() > end:
                raise RuntimeError("the dataset VM did not accept SSH within 15 minutes")
            time.sleep(15)
        _, out, _ = self.exec('echo "HOME=$HOME"; nproc; df -h "$HOME" | tail -1', timeout=120)
        say(f"VM {self.state['machine_id']} up: {' | '.join(out.split())}")
        self.state["home"] = out.split("HOME=")[-1].split()[0]
        self.save()

    def upload(self):
        if self.state.get("uploaded"):
            return
        home = self.state["home"]
        bundle = os.path.join(tempfile.gettempdir(), "note_synth_bootstrap_code.tgz")
        with tarfile.open(bundle, "w:gz") as tar:
            for name in sorted(os.listdir(NOTE_SYNTH)):
                if name.endswith(".py") or name == "splits.json":
                    tar.add(os.path.join(NOTE_SYNTH, name), arcname=f"note_synth/{name}")
        self.exec(f"mkdir -p {home}/run/note_synth/data {home}/bootstrap")
        self.jl.upload(self.state["machine_id"], bundle, f"{home}/run/code.tgz")
        os.remove(bundle)
        self.exec(f"cd {home}/run && tar xzf code.tgz && rm -f code.tgz && ls note_synth | wc -l")
        for name in UPLOADS:
            local = os.path.join(DATA, name)
            remote = f"{home}/run/note_synth/data/{name}"
            _, out, _ = self.exec(f"stat -c %s {shlex.quote(remote)} 2>/dev/null || echo none", timeout=120)
            if out.strip().splitlines()[-1:] == [str(os.path.getsize(local))]:
                say(f"{name} is already on the VM")
                continue
            say(f"uploading {name} ({os.path.getsize(local) / 1e6:.0f} MB)")
            self.jl.upload(self.state["machine_id"], local, remote)
        self.jl.upload(self.state["machine_id"], REMOTE_SCRIPT, f"{home}/bootstrap/bootstrap_remote.sh")
        self.state["uploaded"] = True
        self.save()

    def launch(self):
        home = self.state["home"]
        _, out, _ = self.exec(f"cd {home}/bootstrap && chmod +x bootstrap_remote.sh && "
                              f"(setsid nohup bash bootstrap_remote.sh >> runner.log 2>&1 < /dev/null &) && "
                              f"sleep 3 && echo LAUNCHED")
        if "LAUNCHED" not in out:
            raise RuntimeError(f"bootstrap_remote.sh did not start: {out[-300:]}")
        say("bootstrap_remote.sh started")

    def watch(self):
        home, last = self.state["home"], None
        while True:
            if self.cost() >= self.budget:
                raise RuntimeError(f"the bootstrap reached its budget of ${self.budget:.2f}")
            try:
                _, out, _ = self.exec(f"cd {home}/bootstrap && tail -n 3 progress.log 2>/dev/null; "
                                      f"[ -f ALL_DONE ] && echo STATE=done; [ -f FAILED ] && echo STATE=failed:$(cat FAILED); "
                                      f"pgrep -f bootstrap_remote.sh >/dev/null && echo ALIVE", timeout=120, check=False)
            except JLError as error:
                say(f"poll failed: {str(error)[-200:]}")
                time.sleep(POLL)
                continue
            if out != last:
                say("remote: " + " | ".join(l for l in out.splitlines() if l.strip()))
                last = out
            if "STATE=done" in out:
                return
            if "STATE=failed" in out:
                stage = out.split("STATE=failed:")[-1].split()[0]
                _, log, _ = self.exec(f"tail -n 30 {home}/bootstrap/logs/{stage}.log", check=False)
                raise RuntimeError(f"stage {stage} failed:\n{log[-2500:]}")
            if "ALIVE" not in out:
                say("bootstrap_remote.sh is not running: starting it again (it skips what it finished)")
                self.launch()
            self.state["cost"] = round(self.cost(), 3)
            self.save()
            time.sleep(POLL)

    def pause(self):
        if not self.state.get("machine_id"):
            return
        try:
            self.jl.pause(self.state["machine_id"])
            say(f"paused the dataset VM {self.state['machine_id']}")
        finally:
            self.stop_billing()

    def run(self):
        missing = [n for n in UPLOADS if not os.path.exists(os.path.join(DATA, n))]
        if missing:
            raise SystemExit(f"missing from {DATA}: {', '.join(missing)}")
        if self.state.get("phase") == "done":
            say("the dataset VM was already set up")
            return 0
        self.state.update(phase="running", error=None, pid=os.getpid(), started_at=time.time())
        self.save()
        try:
            self.vm()
            self.upload()
            self.launch()
            self.watch()
            self.state["phase"] = "done"
            say(f"the dataset VM is set up; cost ${self.cost():.2f}")
        except BaseException as error:  # noqa: BLE001 - recorded, and the VM paused, whatever it was
            self.state.update(phase="failed", error=f"{type(error).__name__}: {error}"[-3000:])
            say(f"failed: {self.state['error']}")
        finally:
            self.pause()
            self.state["cost"] = round(self.cost(), 3)
            self.save()
        return 0 if self.state["phase"] == "done" else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=3.0, help="dollars; the VM is paused when its cost reaches this")
    args = ap.parse_args()
    sys.exit(Bootstrap(args.budget).run())


if __name__ == "__main__":
    main()
