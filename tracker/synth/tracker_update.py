"""tracker_update.py — the tracker's status document from the watchdog's status.json.

    python tracker_update.py RUN_DIR            # -> RUN_DIR/tracker_status.json, and prints new picks

build_doc(run_dir) returns the document; tracker_server.py serves it live.
"""
import json
import os
import sys

LABELS = {"setup_python": "Install Python packages", "fetch_mathwriting": "Download MathWriting (3 GB)",
          "fetch_quickdraw": "Fetch QuickDraw subset", "build_caches": "Build split caches and shape banks",
          "resume_vm": "Resume the dataset VM", "upload_code": "Upload the generator code",
          "check_cache": "Check the split caches and disk space",
          "cache_times": "Rebuild the split caches with stroke times (only if they lack them)",
          "cache_rebuild": "Rebuild the split caches if they are older than this code's splits or pools",
          "recog_pool_train": "Textline recognition samples from train's whole IAM-OnDB pool, in passes",
          "synth_geometry": "Geometry figures without a page, train only (no page drawn)",
          "swap_additions": "Add the new train data to the dataset (train/geometry, train/recognition_pool)",
          "prepare_output": "Clear the output folder",
          "synth_train": "Synthesize train", "synth_val": "Synthesize val", "synth_test": "Synthesize test",
          "recog_train": "Recognition samples, train", "recog_val": "Recognition samples, val",
          "recog_test": "Recognition samples, test",
          "swap_dataset": "Replace the dataset with the new one", "count_classes": "Count objects per page",
          "swap_recognition": "Replace the dataset's recognition samples with the new ones",
          "all_done": "Finished"}


def display_stages(stages, mode, options):
    """The stages as the tracker lists them, always the usual pipeline. A split's pages and its recognition samples are
    made together, so they are one stage. A recognition-only run's split stages (recog_<split>) are listed as that same
    Synthesize stage with only recognition written, and the stages it doesn't run as skipped."""
    recognition = bool(((options or {}).get("recognition") or {}).get("on"))
    out = []
    for s in stages:
        name = s["name"]
        entry = {"name": name, "label": LABELS.get(name, name), "state": s.get("state"),
                 "start": s.get("start"), "end": s.get("end"), "log": s.get("log", [])}
        if name == "recog_pool_train":          # its own stage in every mode, not a split's
            out.append(entry)
            continue
        if mode == "recognition":
            if name.startswith("recog_"):
                split = name[len("recog_"):]
                entry.update(name=f"synth_{split}",
                             label=f"Synthesize {split}: pages rebuilt in memory, only their recognition samples written")
            elif name == "swap_recognition":
                out.append({"name": "synth_test", "label": "Synthesize test (not part of a recognition run)",
                            "state": "skipped", "start": None, "end": None, "log": []})
            elif name == "all_done":
                out.append({"name": "count_classes", "label": "Count objects per page (not part of a recognition run)",
                            "state": "skipped", "start": None, "end": None, "log": []})
        elif name.startswith("recog_"):
            continue            # a full run's recognition samples are made with its pages: part of synth_<split>
        elif recognition and name in ("synth_train", "synth_val"):
            entry["label"] += ": pages, samples, mark crops and recognition samples"
        out.append(entry)
    return out


def build_doc(run_dir):
    watch = json.load(open(os.path.join(run_dir, "status.json")))
    status = watch.get("status") or {}
    meta = json.load(open(os.path.join(run_dir, "run_meta.json")))
    now = watch["polled_at"]
    splits = status.get("splits", {})
    done = sum(s.get("pages", 0) for s in splits.values())
    target = sum((s.get("target") or 0) for s in splits.values())
    stages = status.get("stages", [])
    remote_now = status.get("now", now)

    # a recognition run's recog_<split> stages build the pages too
    synth = [s for s in stages if s["name"].startswith(("synth_", "recog_")) and s["name"] != "recog_pool_train"
             and s.get("start")]
    synth_seconds = sum((s.get("end") or remote_now) - s["start"] for s in synth)
    worker_seconds = sum(s.get("worker_seconds", 0) for s in splits.values())
    wall_per_page = synth_seconds / done if done else None
    eta = wall_per_page * (target - done) if wall_per_page else None

    reason = watch.get("pause_reason")
    running = [s for s in stages if s.get("state") == "running"]
    if watch.get("paused"):
        state = {"finished": "done", "kill requested": "stopped"}.get(reason, "paused")
        if reason and reason.startswith("stage failed"):
            state = "failed"
    elif any(s.get("state") == "failed" for s in stages):
        state = "failed"
    elif not status:
        state = "unreachable"
    else:
        state = "running"

    shown = display_stages(stages, meta.get("mode"), meta.get("options"))
    doc = {
        "state": state, "pause_reason": reason,
        "machine_id": watch["machine_id"], "region": meta["region"], "vm": meta["vm"],
        "hourly": watch["hourly"], "budget": watch["budget"], "cost": watch["cost"],
        "started_at": watch["started"], "updated_at": now, "poll": watch["poll"],
        "current_stage": next((s["label"] for s in shown if s["state"] == "running"), None) if running else None,
        "overall": {"pages_done": done, "pages_target": target,
                    "pct": round(100 * done / target, 2) if target else 0},
        "speed": {"wall_s_per_page": round(wall_per_page, 3) if wall_per_page else None,
                  "worker_s_per_page": round(worker_seconds / done, 3) if done else None,
                  "pages_per_min": round(60 / wall_per_page, 1) if wall_per_page else None,
                  "eta_s": round(eta) if eta is not None else None},
        "stages": shown,
        "splits": {name: {k: s.get(k) for k in ("target", "pages", "samples", "rows", "marks",
                                                 "max_item_uses_on_a_page", "shards_done", "classes", "recognition",
                                                 "recognition_pool", "geometry_samples")}
                   for name, s in splits.items()},
        "options": meta.get("options"),     # the run's settings; absent for runs from before they existed
        "machine": {"load": status.get("load"), "cpus": status.get("cpus"),
                    "disk_used_gb": status.get("disk_used_gb"), "disk_total_gb": status.get("disk_total_gb")},
    }
    # class_counts.py's output, pulled from the dataset VM once a run is done
    counts_path = os.path.join(run_dir, "class_counts.json")
    if os.path.exists(counts_path):
        counts = json.load(open(counts_path))
        doc["class_order"] = counts.get("classes")
        for name, c in counts.get("splits", {}).items():
            if name in doc["splits"]:
                doc["splits"][name]["objects_per_page"] = c["per_page"]
                doc["splits"][name]["objects_source"] = c["source"]
    return doc


def main(run_dir):
    doc = build_doc(run_dir)
    with open(os.path.join(run_dir, "tracker_status.json"), "w") as f:
        json.dump(doc, f)
    print(json.dumps({"state": doc["state"], "pct": doc["overall"]["pct"], "cost": doc["cost"]}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
