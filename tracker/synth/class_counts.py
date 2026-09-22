"""class_counts.py — average objects of each class per page, per split, from a finished dataset.

    python3 class_counts.py ~/run/dataset > class_counts.json      # on the dataset VM, stdlib only

A split with page files (val, test) is counted from each page JSON's objects. Train has
none, so it is counted from its sample manifest: every object is named once across a
page's prompts (level 2 and arrows in the full-page targets, level 1 in its parent's
crop), so the class words in a page's targets are its objects. Val has both, and its
manifest count is kept as `from_samples` to check the two agree. Test gives counts only.
"""
import glob
import json
import os
import re
import sys
import time
from collections import Counter

ORDER = ["textblock", "drawing", "geometry", "table", "flowchart",
         "textline", "formula", "enclosure", "line_mark", "arc", "segment", "vertex", "arrow"]
PAGE_OF = re.compile(r"p(\d+)_s\d+\.png$")


def from_pages(folder):
    totals, pages = Counter(), 0
    for path in glob.glob(os.path.join(folder, "page*.json")):
        with open(path) as f:
            totals.update(o["class"] for o in json.load(f)["objects"])
        pages += 1
    return totals, pages


def target_classes(target):
    tokens = target.split()
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if not t.isdigit():
            following = tokens[i + 1] if i + 1 < len(tokens) else None
            if (t, following) == ("line", "mark"):
                t = "line_mark"
                i += 1
            yield t
        i += 1


def from_samples(folder):
    joined = os.path.join(folder, "manifest.jsonl")
    paths = [joined] if os.path.exists(joined) else sorted(glob.glob(os.path.join(folder, "shards", "*.jsonl")))
    totals, pages = Counter(), set()
    for path in paths:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                page = PAGE_OF.search(row["image"])
                if page is None:
                    continue            # a geometry figure drawn without a page (g0000123.png): no page's objects
                pages.add(page.group(1))
                totals.update(target_classes(row["target"]))
    return totals, len(pages)


def summary(totals, pages, source):
    classes = ORDER + sorted(set(totals) - set(ORDER))
    return {"source": source, "pages": pages, "totals": {c: totals.get(c, 0) for c in classes},
            "per_page": {c: round(totals.get(c, 0) / pages, 3) if pages else 0 for c in classes}}


def main(root):
    out = {"generated_at": time.time(), "classes": ORDER, "splits": {}}
    for split in ("train", "val", "test"):
        pages_dir, samples_dir = os.path.join(root, split, "pages"), os.path.join(root, split, "samples")
        doc = None
        if os.path.isdir(pages_dir):
            doc = summary(*from_pages(pages_dir), "pages")
        if os.path.isdir(samples_dir):
            counted = summary(*from_samples(samples_dir), "samples")
            if doc is None:
                doc = counted
            elif split != "test":
                doc["from_samples"] = counted["per_page"]
        if doc:
            out["splits"][split] = doc
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main(os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/run/dataset"))
