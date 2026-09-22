"""
progress.py — the files a run leaves for whoever is watching it.

    ProgressFile     progress.json: what the run is doing now, rewritten every few seconds
    SampleStream     a JSON-lines file each kept validation sample is appended to as it is made

Only rank 0 writes; on the other ranks both are no-ops. Writes go to a
temporary file and are renamed into place, so a reader never sees half a file.
"""

from __future__ import annotations
import json
import os
import time

from distributed import is_main


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, data):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:         # Windows: a reader has the file open
            time.sleep(0.05 * (attempt + 1))
    os.replace(tmp, path)


class ProgressFile:
    """One JSON document of fields, written at most every `every` seconds unless forced."""

    def __init__(self, path, every=5.0, **fields):
        self.path, self.every, self.last = path, every, 0.0
        self.doc = {"started_at": time.time(), "pid": os.getpid(), **fields}

    def update(self, force=False, **fields):
        self.doc.update(fields)
        if not is_main() or self.path is None:
            return
        now = time.time()
        if force or now - self.last >= self.every:
            self.doc["updated_at"] = now
            write_json(self.path, self.doc)
            self.last = now


class SampleStream:
    """Rows appended to a JSON-lines file one at a time, up to `limit` (0 = no limit)."""

    def __init__(self, path, limit=0):
        self.path = path if is_main() else None
        self.limit, self.count = limit, 0
        if self.path:
            open(self.path, "w").close()

    def write(self, row):
        if self.path is None or (self.limit and self.count >= self.limit):
            return
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        self.count += 1
