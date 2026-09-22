"""train_status.py — one JSON line describing the training VM (train_remote.sh, test_remote.sh).

    python3 train_status.py [LOG_OFFSET] [MAX_LOG_BYTES]

LOG_OFFSET: bytes of run/log.jsonl already read; the whole lines after it come back
(at most MAX_LOG_BYTES of them) with the offset to ask for next time.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

HOME = os.path.expanduser("~")
ROOT = os.path.join(HOME, "train")
RUN = os.path.join(ROOT, "run")
TEST = os.path.join(ROOT, "test")
ORDER = ["setup_env", "fetch_dataset", "fetch_model", "smoke", "train"]


def load(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def tail(path, n=6):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 12000))
            text = f.read().decode("utf-8", "replace").replace("\r", "\n")
        return [line[-300:] for line in text.splitlines() if line.strip()][-n:]
    except OSError:
        return []


def new_log_lines(offset, limit):
    path = os.path.join(RUN, "log.jsonl")
    try:
        size = os.path.getsize(path)
    except OSError:
        return {"offset": 0, "next": 0, "size": 0, "records": []}
    if offset > size:                   # the log was replaced (a fresh run): start over
        offset = 0
    with open(path, "rb") as f:
        f.seek(offset)
        chunk = f.read(limit)
    end = chunk.rfind(b"\n") + 1
    records = []
    for line in chunk[:end].decode("utf-8", "replace").splitlines():
        try:
            records.append(json.loads(line))
        except ValueError:
            pass
    return {"offset": offset, "next": offset + end, "size": size, "records": records}


def gpus():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,"
                              "temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 7:
            continue
        num = lambda v: float(v) if re.fullmatch(r"[\d.]+", v) else None
        rows.append({"index": int(parts[0]), "name": parts[1], "util": num(parts[2]), "mem_used_mb": num(parts[3]),
                     "mem_total_mb": num(parts[4]), "temp_c": num(parts[5]), "power_w": num(parts[6])})
    return rows


def test_metrics_summary(path):
    doc = load(path)
    if not doc:
        return None
    keep = ("checkpoint", "units", "outputs", "parse_valid_rate", "parse_valid_by_stage", "mean_f1",
            "mean_f1_classes", "mean_f1_by_group", "per_class", "parse_errors", "recognition", "thresholds")
    return {k: doc[k] for k in keep if k in doc}


def main():
    # the orchestrator polls with this script: budget_guard.py pauses a finished run's VM once polls stop coming
    try:
        with open(os.path.join(ROOT, ".last_poll"), "w") as f:
            f.write(str(time.time()))
    except OSError:
        pass
    offset = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 2_000_000

    stages = []
    for name in ORDER:
        stage = load(os.path.join(ROOT, "stages", f"{name}.json"), {"state": "pending"})
        stage["name"] = name
        if stage.get("state") in ("running", "failed"):
            stage["log"] = tail(os.path.join(ROOT, "logs", f"{name}.log"))
        stages.append(stage)

    samples = [{"name": os.path.basename(p), "path": p, "size": os.path.getsize(p), "mtime": os.path.getmtime(p)}
               for p in sorted(glob.glob(os.path.join(RUN, "samples_*.jsonl")))]
    checkpoints = [{"name": os.path.basename(d), "mtime": os.path.getmtime(d)}
                   for d in glob.glob(os.path.join(RUN, "*"))
                   if re.fullmatch(r"step_\d+|last|best", os.path.basename(d))
                   and os.path.exists(os.path.join(d, "adapter_model.safetensors"))]

    test = None
    if os.path.isdir(TEST):
        test = {"started": os.path.exists(os.path.join(TEST, "STARTED")),
                "progress": load(os.path.join(TEST, "test_progress.json")),
                "done": load(os.path.join(TEST, "done.json")),
                "metrics": test_metrics_summary(os.path.join(TEST, "test_metrics.json")),
                "samples": {"path": os.path.join(TEST, "test_samples.jsonl"),
                            "size": os.path.getsize(os.path.join(TEST, "test_samples.jsonl"))}
                if os.path.exists(os.path.join(TEST, "test_samples.jsonl")) else None,
                "log": tail(os.path.join(TEST, "test.log"), 8)}
        try:
            test["checkpoint"] = open(os.path.join(TEST, "checkpoint")).read().strip()
        except OSError:
            pass

    try:
        running = subprocess.run(["pgrep", "-fc", "train.py|seg_eval.py"], capture_output=True, text=True,
                                 timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        running = None
    usage = shutil.disk_usage(ROOT if os.path.isdir(ROOT) else HOME)
    print(json.dumps({
        "now": time.time(), "home": HOME, "stages": stages,
        "progress": load(os.path.join(RUN, "progress.json")), "config": load(os.path.join(RUN, "config.json")),
        "control": load(os.path.join(RUN, "control.json")), "control_ack": load(os.path.join(RUN, "control_ack.json")),
        "stop_pending": os.path.exists(os.path.join(RUN, "STOP")),
        "log": new_log_lines(offset, limit), "console": tail(os.path.join(ROOT, "logs", "train.log"), 10),
        "samples": samples, "checkpoints": sorted(checkpoints, key=lambda c: c["mtime"]), "test": test,
        "gpus": gpus(), "processes": int(running) if running and running.isdigit() else None,
        "disk_used_gb": round((usage.total - usage.free) / 1e9, 1), "disk_total_gb": round(usage.total / 1e9, 1),
    }))


if __name__ == "__main__":
    main()
