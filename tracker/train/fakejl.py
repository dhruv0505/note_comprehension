"""fakejl.py — a local stand-in for the Jarvis Labs CLI, for testing the trainer flow.

    python fakejl.py --root DIR <jl command> ...

A VM is a folder DIR/home_<id> used as its home; its record is in DIR/machines.json.
exec runs the command in Git Bash with HOME (and USERPROFILE) pointed at that folder;
pause kills every process whose command line mentions the folder; resume gives
the VM a new id (as Jarvis may) by renaming its folder. Prices are Jarvis's.
Only the commands and flags jlcmd.py uses are understood.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time

GIT_BASH = r"C:\Program Files\Git\bin\bash.exe"


def load(root):
    try:
        return json.load(open(os.path.join(root, "machines.json")))
    except (OSError, ValueError):
        return {"next_id": 900001, "machines": {}}


def save(root, db):
    with open(os.path.join(root, "machines.json"), "w") as f:
        json.dump(db, f, indent=1)


def home(root, mid):
    return os.path.join(root, f"home_{mid}")


def windows_path(root, mid, path):
    """A remote path as the local file it stands for."""
    for prefix in ("/home/ubuntu/", "~/"):
        if path.startswith(prefix):
            return os.path.join(home(root, mid), path[len(prefix):])
    if path.startswith("/"):                        # a Git Bash path such as /tmp/... or /c/...
        return subprocess.run([GIT_BASH, "-c", f"cygpath -w '{path}'"], capture_output=True, text=True).stdout.strip()
    return path


def out(doc):
    print(json.dumps(doc))


def kill_processes(folder):
    needle = os.path.basename(folder)
    script = (f"Get-CimInstance Win32_Process | Where-Object {{ $_.CommandLine -like '*{needle}*' "
              f"-and $_.ProcessId -ne {os.getpid()} }} | ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force "
              f"-ErrorAction SilentlyContinue }}")
    subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, timeout=120)


def main():
    argv = sys.argv[1:]
    root = argv[argv.index("--root") + 1]
    del argv[argv.index("--root"):argv.index("--root") + 2]
    os.makedirs(root, exist_ok=True)
    db = load(root)
    cmd, rest = argv[0], argv[1:]
    flag = lambda name, default=None: rest[rest.index(name) + 1] if name in rest else default

    if cmd == "status":
        return out({"balance": {"balance": 17.5, "grants": 0.0}, "currency": "USD"})
    if cmd == "gpus":
        return out([{"gpu_type": "H100", "region": "IN2", "num_free_devices": 8, "price_per_hour": 2.69,
                     "workload_type": "vm"}])
    if cmd == "list":
        return out(list(db["machines"].values()))
    if cmd == "create":
        mid = db["next_id"]
        db["next_id"] += 1
        os.makedirs(home(root, mid), exist_ok=True)
        time.sleep(2)
        db["machines"][str(mid)] = {"machine_id": mid, "name": flag("--name"), "status": "Running",
                                    "gpu_type": flag("--gpu"), "num_gpus": int(flag("--num-gpus", 1)),
                                    "storage_gb": int(flag("--storage", 100)), "region": flag("--region"),
                                    "template": "vm", "ssh_command": "ssh ubuntu@127.0.0.1",
                                    "private_ip": "127.0.0.1", "public_ip": "127.0.0.1"}
        save(root, db)
        return out(db["machines"][str(mid)])

    mid = rest[0]
    machine = db["machines"].get(str(mid))
    if machine is None:
        print(f"Instance {mid} not found", file=sys.stderr)
        sys.exit(1)
    if cmd == "get":
        return out(machine)
    if cmd == "pause":
        kill_processes(home(root, mid))
        machine["status"] = "Paused"
        save(root, db)
        print(f"Instance {mid} paused.")
        return
    if cmd == "resume":
        if machine["status"] != "Paused":
            print(f"Can only resume a Paused instance (current status: {machine['status']})", file=sys.stderr)
            sys.exit(1)
        new = db["next_id"]
        db["next_id"] += 1
        os.replace(home(root, mid), home(root, new))
        del db["machines"][str(mid)]
        machine.update(machine_id=new, status="Running", gpu_type=flag("--gpu", machine.get("gpu_type")),
                       num_gpus=int(flag("--num-gpus", machine.get("num_gpus") or 1)))
        db["machines"][str(new)] = machine
        save(root, db)
        time.sleep(2)
        return out(machine)
    if machine["status"] != "Running":
        out({"machine_id": int(mid), "exit_code": 255, "stdout": "", "stderr": "instance is not running"})
        sys.exit(255)
    if cmd == "exec":
        command = rest[rest.index("--") + 3]
        folder = home(root, mid)
        env = dict(os.environ, HOME=folder, USERPROFILE=folder)
        env.pop("MSYS2_ARG_CONV_EXCL", None)
        env.pop("MSYS_NO_PATHCONV", None)           # paths handed to Windows programs must be converted
        done = subprocess.run([GIT_BASH, "-c", f"cd ~ && {command}"], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", env=env)
        out({"machine_id": int(mid), "exit_code": done.returncode, "stdout": done.stdout, "stderr": done.stderr})
        sys.exit(done.returncode)
    if cmd == "upload":
        local, remote = rest[1], windows_path(root, mid, rest[2])
        os.makedirs(os.path.dirname(remote), exist_ok=True)
        shutil.copyfile(local, remote)
        return
    if cmd == "download":
        remote, local = windows_path(root, mid, rest[1]), rest[2]
        shutil.copyfile(remote, local)
        return
    print(f"fakejl: {cmd} not supported", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
