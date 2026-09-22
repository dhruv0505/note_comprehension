"""
iam_writers.py — recover IAM-OnDB's writer IDs from DeepWriting.

    python iam_writers.py                 # -> data/iamondb_writers.json

IAM-OnDB's lineStrokes and ascii parts name each line by its form (a01-000u),
not by who wrote it, so both the page generator's "one hand per page" and a
writer-disjoint split had only the form to go on — about 7 lines a writer.

DeepWriting's release carries IAM-OnDB's own ink, re-cut into chunks of a few
words: 186 of its 294 subjects are IAM-OnDB writers, stroke for stroke, with
subject ids in IAM-OnDB's 10000 range (the other ~95 are new writers copying
IAM's texts). Each sample is tagged with its subject, so a form's writer is the
subject whose chunks are runs of that form's strokes, point for point.

A chunk matches an IAM line when its source text is that line's transcription
and its inner strokes — the first and last may be cut mid-stroke — have exactly
the point counts of a contiguous run of the line's strokes, at least
MIN_RUN strokes long. A form takes the subject with the most matching chunks,
provided no other subject has more than CONFLICT_SHARE of that. Forms with no
match, or a conflict, are left out of the file; loaders.load_iam_ondb keeps the
form as their writer, and a split should keep them in training only.
"""

from __future__ import annotations
import argparse
import collections
import io
import json
import os
import re
import tarfile
import xml.etree.ElementTree as ET

import numpy as np

import loaders

MIN_RUN = 3             # strokes a chunk's inner run must span to count as a match
CONFLICT_SHARE = 0.2    # a runner-up with more than this share of the winner's votes is a conflict


def _iam_stroke_counts(path: str):
    """Form -> [(line file, points per stroke)], in line order, read in one pass."""
    by_doc = collections.defaultdict(list)
    with tarfile.open(path, "r:*") as tar:
        for member in tar:
            if not (member.isfile() and member.name.endswith(".xml")):
                continue
            doc, index = loaders._iam_doc_and_line(member.name)
            root = ET.fromstring(tar.extractfile(member).read())
            counts = [sum(1 for _ in stroke.iter("Point")) for stroke in root.iter("Stroke")]
            by_doc[doc].append((index, member.name, counts))
    return {doc: [(name, counts) for _, name, counts in sorted(lines)]
            for doc, lines in by_doc.items()}


def _deepwriting_samples(path: str):
    """(subject, source text, points per stroke) for every DeepWriting sample."""
    archive = loaders._Archive(path)
    for member in archive.names(".npz"):
        data = np.load(io.BytesIO(archive.read(member)), allow_pickle=True)
        # NpzFile re-reads an array on every data[key], so take each once
        strokes, texts, subjects = data["strokes"], data["texts"], data["subject_labels"]
        for i in range(len(texts)):
            pen_up = strokes[i][:, 2]
            bounds = list(np.where(pen_up == 1)[0] + 1)
            lengths = [b - a for a, b in zip([0] + bounds, bounds + [len(pen_up)]) if b > a]
            yield int(np.asarray(subjects[i]).ravel()[0]), str(texts[i]), lengths


def _contains_run(inner, full) -> bool:
    n = len(inner)
    return any(full[k:k + n] == inner for k in range(len(full) - n + 1))


def build(out_path: str) -> dict:
    transcriptions = loaders._iam_parse_transcriptions(
        loaders._Archive(loaders._resolve(None, "iamondb_ascii")))
    strokes = _iam_stroke_counts(loaders._resolve(None, "iamondb_strokes"))

    # a line's transcription -> the stroke counts of every form line with that text,
    # only where a form's transcription lines up 1:1 with its stroke files (as the
    # loader requires)
    lines_by_text = collections.defaultdict(list)
    for doc, files in strokes.items():
        texts = transcriptions.get(doc)
        if texts is None or len(texts) != len(files):
            continue
        for text, (_, counts) in zip(texts, files):
            lines_by_text[text].append((doc, counts))

    votes = collections.defaultdict(collections.Counter)
    chunks = matched = 0
    for subject, text, lengths in _deepwriting_samples(loaders._resolve(None, "deepwriting")):
        candidates = lines_by_text.get(text)
        if not candidates:
            continue
        chunks += 1
        inner = lengths[1:-1]
        if len(inner) < MIN_RUN:
            continue
        for doc, counts in candidates:
            if _contains_run(inner, counts):
                votes[doc][subject] += 1
                matched += 1

    writers, conflicts = {}, {}
    for doc, tally in votes.items():
        (best, n), *rest = tally.most_common()
        if rest and rest[0][1] > CONFLICT_SHARE * n:
            conflicts[doc] = dict(tally)
        else:
            writers[doc] = best

    # what the map buys: lines per writer, and how a writer- and text-disjoint
    # split would have to group forms (a form's text can be copied by several
    # writers, and a writer copies several forms)
    line_count = {doc: len(files) for doc, files in strokes.items()}
    per_writer = collections.Counter()
    for doc, writer in writers.items():
        per_writer[writer] += line_count.get(doc, 0)

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for doc in strokes:
        text_key = ("text", re.sub(r"[a-z]$", "", doc))
        writer_key = ("writer", writers[doc]) if doc in writers else ("form", doc)
        parent[find(("doc", doc))] = find(text_key)
        parent[find(writer_key)] = find(text_key)
    groups = collections.Counter()
    for doc in strokes:
        groups[find(("doc", doc))] += line_count[doc]
    sizes = sorted(groups.values(), reverse=True)

    stats = {
        "forms_with_ink": len(strokes),
        "lines_with_ink": sum(line_count.values()),
        "deepwriting_chunks_on_iam_text": chunks,
        "chunk_line_matches": matched,
        "forms_with_writer": len(writers),
        "lines_with_writer": sum(line_count[d] for d in writers),
        "forms_conflicting": len(conflicts),
        "forms_unmatched": len(strokes) - len(writers) - len(conflicts),
        "writers": len(per_writer),
        "lines_per_writer_median": float(np.median(list(per_writer.values()))) if per_writer else 0,
        "split_groups": len(sizes),
        "split_group_lines_largest": sizes[:5],
        "split_group_lines_median": float(np.median(sizes)) if sizes else 0,
    }
    result = {"about": "IAM-OnDB form -> writer id, recovered from DeepWriting by iam_writers.py",
              "stats": stats, "writers": dict(sorted(writers.items())),
              "conflicts": conflicts}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=loaders.data_path("iamondb_writers"))
    args = ap.parse_args()
    stats = build(args.out)
    print(json.dumps(stats, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
