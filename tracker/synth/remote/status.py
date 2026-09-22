"""status.py — one JSON line describing a dataset regeneration on this VM (regen_remote.sh).

    python3 ~/regen/<run id>/status.py

Stages, pages per split and output folders come from targets.json beside it. Picks:
up to targets["picks"] pages per split, spread evenly over the split and offered as
soon as their shard is done — each page's full-page sample, a mark crop from the same
shard, and for val the page drawing. With recognition on, each task's samples
(synthesize.py --recognition, train and val only) are counted from the shards' usage
records and one is picked from each of up to targets["recognition"]["picks"] shards spread
over the split. Test pages are counted, never picked.
"""
import glob
import json
import os
import shutil
import subprocess
import time

RUN = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~")


def load(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def tail(path, n=3):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 6000))
            lines = [l for l in f.read().decode("utf-8", "replace").replace("\r", "\n").splitlines() if l.strip()]
        return [l[-220:] for l in lines[-n:]]
    except OSError:
        return []


def rows(path):
    try:
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    except (OSError, ValueError):
        return []


# A sample pick is one prompt of its page. The picks go round these in turn, so the
# ones to look at cover every prompt rather than only each page's first (level 2).
# A page without the prompt due takes the next one it has.
PROMPT_KINDS = ("level 2 objects", "arrows", "in textblock", "in table", "in geometry", "in flowchart")


def split_picks(base, split, records, total, count):
    wanted = spread(total, count)
    picks = []
    for record in records:
        pages = [p for p in wanted if record["first"] <= p < record["first"] + record["count"]]
        if not pages:
            continue
        shard = record["shard"]
        samples = rows(f"{base}/samples/shards/manifest_{shard:05d}.jsonl")
        marks = rows(f"{base}/marks/shards/manifest_{shard:05d}.jsonl")
        marked = set()
        for page in pages:
            tag = f"p{page:07d}_s"
            page_rows = [r for r in samples if os.path.basename(r["image"]).startswith(tag)]
            turn = wanted.index(page)
            row = None
            for step in range(len(PROMPT_KINDS)):
                kind = PROMPT_KINDS[(turn + step) % len(PROMPT_KINDS)]
                row = next((r for r in page_rows if kind in r["prompt"]), None)
                if row:
                    break
            row = row or (page_rows[0] if page_rows else None)
            if row:
                picks.append({"split": split, "kind": "sample", "page": page,
                              "remote": {"image": f"{base}/samples/{row['image']}"},
                              "prompt": row["prompt"], "target": row["target"]})
            png = f"{base}/pages/page{page:07d}.png"
            if os.path.exists(png):
                picks.append({"split": split, "kind": "page", "page": page,
                              "remote": {"image": png, "json": png[:-4] + ".json"}})
            mark = (next((m for m in marks if m["page"] >= page and m["page"] not in marked), None)
                    or next((m for m in marks if m["page"] not in marked), None))
            if mark:
                marked.add(mark["page"])
                picks.append({"split": split, "kind": "mark", "page": mark["page"],
                              "remote": {part: f"{base}/marks/{mark[part]}" for part in ("input", "text")},
                              "shapes": mark["shapes"], "labels": mark["labels"], "text_label": mark["text_label"],
                              "box": mark["box"]})
    return picks


def spread(total, count):
    return sorted({min(total - 1, round(i * total / count)) for i in range(count)}) if total and count else []


def recognition(base, split, records, tasks, count, shards):
    """({task: counts}, picks) for the recognition samples synthesize.py --recognition writes under base/recognition/:
    counted from each done shard's usage record, and one sample picked from each of `count` shards spread over the
    split's `shards`, once that shard is done."""
    counts, picks = {}, []
    by_shard = {r["shard"]: r for r in records}
    for task in tasks:
        per_shard = [(r.get("recognition") or {}).get(task) or {} for r in records]
        counts[task] = {**{k: sum(s.get(k, 0) for s in per_shard)
                           for k in ("samples", "rows", "short_samples", "rows_with_other_lines", "lines_without_times")},
                        "shards_done": len(records)}
        for shard in spread(shards, count):
            if shard not in by_shard:
                continue
            shard_rows = rows(f"{base}/recognition/{task}/shards/manifest_{shard:05d}.jsonl")
            if shard_rows:
                row = shard_rows[len(shard_rows) // 2]
                picks.append({"split": split, "kind": f"recog_{task}", "page": int(row["page"].lstrip("p")),
                              "task": task, "remote": {"image": f"{base}/recognition/{task}/{row['image']}"},
                              "prompt": row["prompt"], "target": row["target"]})
    return counts, picks


def main():
    targets = load(os.path.join(RUN, "targets.json"), {})
    recog = targets.get("recognition") or {}
    stages = []
    for name in targets.get("stages", []):
        stage = load(os.path.join(RUN, "stages", f"{name}.json"), {"state": "pending"})
        stage["name"] = name
        if stage.get("state") in ("running", "failed"):
            stage["log"] = tail(os.path.join(RUN, "logs", f"{name}.log"))
        stages.append(stage)

    out = os.path.join(HOME, targets.get("out", ""))
    final = os.path.join(HOME, targets.get("final", ""))
    # A full run deletes the old dataset, writes the new one to `out` and swap_dataset moves it to `final`. Before the
    # swap nothing is counted or picked from the old one (before `out` exists), or its picks would stand for the new
    # dataset's pages, which the tracker keeps one pick of (synth_orchestrator.pull_picks keys them by split, kind and
    # page); after it the new dataset is counted where it now is
    mode = targets.get("mode")
    swapped = any(s["name"] in ("swap_dataset", "swap_additions") and s.get("state") == "done" for s in stages)
    if os.path.isdir(out):
        root = out
    elif mode in ("full", "additions"):         # an additions run counts only what it writes, never the dataset's own
        root = final if swapped and mode == "full" else out
    else:
        root = final
    splits, picks = {}, []
    for split in targets.get("splits", []):
        base = f"{root}/{split}"
        records = sorted((r for r in (load(p) for p in glob.glob(f"{base}/usage/shard_*.json")) if r),
                         key=lambda r: r["shard"])
        agg = {k: sum(r.get(k, 0) for r in records) for k in ("pages", "samples", "rows", "marks", "rejected")}
        agg["worker_seconds"] = round(sum(r.get("seconds", 0) for r in records), 1)
        agg["max_item_uses_on_a_page"] = max((r.get("max_item_uses_on_a_page", 0) for r in records), default=0)
        agg["shards_done"] = len(records)
        agg["target"] = targets.get(split)
        agg["geometry_samples"] = sum(r.get("geometry_samples", 0) for r in records)
        # an additions run's geometry/ and recognition_pool/ are in its own folder until swap_additions moves them
        extras = f"{final}/{split}" if mode == "additions" and swapped else base
        if mode == "additions":
            agg["pages"] = sum(r.get("count", 0) for r in records)     # the pages' worth of figures drawn
            geometry_report = load(f"{extras}/geometry/report.json")
            if geometry_report:
                agg["geometry_samples"] = geometry_report["samples"]
        report = load(f"{base}/report.json")
        if report:
            agg["classes"] = {c: {k: v[k] for k in ("items_used", "uses", "most_used_share", "cap_share")}
                              for c, v in report["classes"].items()}
            # the join drops repeat flowchart level-1 samples after the shards are done: the report counts after it
            agg["samples"], agg["rows"] = report.get("samples", agg["samples"]), report.get("rows", agg["rows"])
        if split == "train":
            # the textline pool pass (recog_synthesize.py --subdir recognition_pool): its report once joined, its
            # shards' records until then
            pool_dir = f"{extras}/recognition_pool/textline"
            pool_report = load(f"{pool_dir}/report.json")
            done = [load(p) or {} for p in glob.glob(f"{pool_dir}/done/shard_*.json")]
            if pool_report or done:
                agg["recognition_pool"] = {"textline": {
                    "samples": pool_report["samples"] if pool_report else sum(d.get("samples", 0) for d in done),
                    "rows": pool_report["rows"] if pool_report else sum(d.get("rows", 0) for d in done),
                    "pool_items": (pool_report or {}).get("pool_items"), "finished": bool(pool_report)}}
        # test has no recognition samples; an additions run makes none from pages
        if recog.get("tasks") and split != "test" and mode != "additions":
            shards = -(-(targets.get(split) or 0) // max(targets.get("shard_pages") or 1, 1))
            agg["recognition"], more = recognition(base, split, records, recog["tasks"], recog.get("picks", 6), shards)
            picks += more if root == out else []
        splits[split] = agg
        # test content is never offered, nor anything this run didn't write, nor an additions run's (no pages)
        if split != "test" and root == out and mode != "additions":
            picks += split_picks(base, split, records, targets.get(split) or 0, targets.get("picks", 20))

    try:
        alive = subprocess.run(["pgrep", "-f", "regen_remote.sh"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        alive = None
    usage = shutil.disk_usage(HOME)
    print(json.dumps({"now": time.time(), "home": HOME, "targets": {s: targets.get(s) for s in targets.get("splits", [])},
                      "mode": targets.get("mode"), "stages": stages, "splits": splits, "picks": picks,
                      "runner_alive": alive,
                      "load": [round(v, 1) for v in os.getloadavg()] if hasattr(os, "getloadavg") else None,
                      "cpus": os.cpu_count(),
                      "disk_used_gb": round((usage.total - usage.free) / 1e9, 1),
                      "disk_total_gb": round(usage.total / 1e9, 1)}))


if __name__ == "__main__":
    main()
