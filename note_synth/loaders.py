"""
loaders.py — convert each source dataset into the common `Ink` format from core.py.

Contract every loader must satisfy:
    loader(path_or_dir) -> List[Ink]

Where each returned Ink:
  - has .strokes populated with real (x, y, t) points
  - has .children populated with sub-Inks if the source gives you finer structure
    (e.g. a text-line Ink whose children are word Inks) — this is what lets
    hierarchy flow through to level-0 labels for free
  - has .label set to something human-readable (transcription, LaTeX, category)
  - has .source set to the dataset name, for balancing/debugging

Every loader reads straight out of the shipped archive — no unpacking required —
and defaults to the canonical location under `data/` (see DATA_DIR / DATA_FILES),
so in practice you just call `load_iam_ondb()` with no arguments.

    data/iamondb_linestrokes.tar.gz   IAM-OnDB per-line stroke XML
    data/iamondb_ascii.tar            IAM-OnDB transcriptions
    data/deepwriting.zip              DeepWriting .npz bundle
    data/mathwriting.tgz              MathWriting 2024 InkML release
    data/fc.zip                       FC_1.0 flowcharts (Bresler, Průša, Hlaváč; InkML)
    data/quickdraw.zip                QuickDraw "full raw" .ndjson files
    data/emnist.zip                   NIST SD19 (EMNIST's source) character rasters

See README.md for where each one comes from.

Coordinate convention: every loader emits y increasing DOWNWARD, matching
IAM-OnDB, FC and the layout engine's page space.

No timing is kept: a Stroke is x/y only, and the per-point times the online
sources carry are dropped on load.
"""

from __future__ import annotations
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import io
import json
import os
import re
import tarfile
import xml.etree.ElementTree as ET
import zipfile

from core import Stroke, Ink

# Where the datasets live, relative to this file, so loaders work regardless of cwd.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

DATA_FILES = {
    "iamondb_strokes": "iamondb_linestrokes.tar.gz",
    "iamondb_ascii": "iamondb_ascii.tar",
    # form -> writer id, built from DeepWriting by iam_writers.py (optional)
    "iamondb_writers": "iamondb_writers.json",
    "deepwriting": "deepwriting.zip",
    "mathwriting": "mathwriting.tgz",
    "fc": "fc.zip",
    "quickdraw": "quickdraw.zip",
    "emnist": "emnist.zip",
}

# Content with fewer strokes than this is dropped by the loaders. A one- or
# two-stroke "line" or "expression" is a single character, and dropped onto a
# page on its own it reads as a stray mark rather than as content — MathWriting's
# `symbols` split is the worst offender (53% are one stroke, median 1), which is
# also why it is no longer in load_mathwriting's default splits.
#
# Pass min_strokes=1 to switch the filter off. Note it costs QuickDraw more than
# the others: ~32% of sketches are two strokes or fewer, and some of those are
# perfectly good drawings, so lower it for that loader specifically if the sketch
# pool ends up too thin.
MIN_STROKES = 3

# DeepWriting ships normalized deltas, so a reconstructed line is only ~0.2 units
# tall where IAM's is ~600 and FC's ~2000. Nothing downstream should depend on
# absolute units — everything is refit into page boxes — but keeping the pools
# within an order of magnitude of each other makes mixed-pool output far easier
# to eyeball, and stops any absolute epsilon elsewhere from quietly biting.
DEEPWRITING_SCALE = 3000.0


# Smaller stand-ins used only when the full download isn't there yet, so pages
# can be generated before the multi-GB datasets land. The canonical file always
# wins when it exists.
DATA_FALLBACKS = {
    "mathwriting": "mathwriting_excerpt.tgz",
    "quickdraw": "quickdraw_sample.zip",
}


def data_path(key: str) -> str:
    """Canonical on-disk path for one of the DATA_FILES entries."""
    return os.path.join(DATA_DIR, DATA_FILES[key])


def _resolve(path: Optional[str], key: str) -> str:
    """Explicit path, else the canonical data/ file, else a registered stand-in."""
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{key} data not found at {path!r}")
        return path

    canonical = data_path(key)
    if os.path.exists(canonical):
        return canonical

    fallback = DATA_FALLBACKS.get(key)
    if fallback:
        fallback = os.path.join(DATA_DIR, fallback)
        if os.path.exists(fallback):
            return fallback

    raise FileNotFoundError(
        f"{key} data not found at {canonical!r}. "
        f"See README.md for how to obtain it, or pass an explicit path."
    )


# ----------------------------------------------------------------------------
# Archive access — every loader reads members in place, out of .zip or .tar[.gz]
# ----------------------------------------------------------------------------

class _Archive:
    """Uniform member access over a zip, a tar(.gz), or an already-extracted dir.

    The datasets are shipped as archives of many small files (12k XMLs for IAM,
    ~670 InkMLs for FC, 600k for MathWriting). Reading them in place avoids
    unpacking tens of gigabytes just to stream through them once.
    """

    def __init__(self, path: str):
        self._zip = self._tar = self._root = None
        if os.path.isdir(path):
            self._root = path
            self._names = [
                os.path.relpath(os.path.join(dirpath, fn), path).replace(os.sep, "/")
                for dirpath, _, filenames in os.walk(path) for fn in filenames
            ]
        elif zipfile.is_zipfile(path):
            self._zip = zipfile.ZipFile(path)
            self._names = [i.filename for i in self._zip.infolist() if not i.is_dir()]
        else:
            self._tar = tarfile.open(path, "r:*")  # transparent .tar / .tar.gz / .tgz
            self._names = [m.name for m in self._tar.getmembers() if m.isfile()]

    def names(self, suffix: str = "") -> List[str]:
        return sorted(n for n in self._names if n.endswith(suffix))

    def read(self, name: str) -> bytes:
        if self._zip is not None:
            return self._zip.read(name)
        if self._tar is not None:
            return self._tar.extractfile(name).read()
        with open(os.path.join(self._root, name.replace("/", os.sep)), "rb") as f:
            return f.read()

    def open_text(self, name: str, encoding: str = "utf-8"):
        """Line iterator for streaming big text members without buffering them."""
        if self._zip is not None:
            return io.TextIOWrapper(self._zip.open(name), encoding=encoding)
        if self._tar is not None:
            return io.TextIOWrapper(self._tar.extractfile(name), encoding=encoding)
        return open(os.path.join(self._root, name.replace("/", os.sep)),
                    "r", encoding=encoding)


# ----------------------------------------------------------------------------
# QuickDraw — ndjson "raw" format
# ----------------------------------------------------------------------------
# Each line of a QuickDraw raw ndjson file looks like:
# {"word": "cat", "countrycode": "US", "timestamp": "...", "recognized": true,
#  "key_id": "...", "drawing": [[[x0,x1,...],[y0,y1,...],[t0,t1,...]], ...]}
# x, y and the third array, each point's time in ms (Stroke.t), are read.

def load_quickdraw(path: str = None, max_items: int = None,
                    only_recognized: bool = True,
                    categories: Optional[Iterable[str]] = None,
                    max_per_category: int = None,
                    min_strokes: int = MIN_STROKES,
                    keep_key: Optional[Callable[[str], bool]] = None) -> List[Ink]:
    """
    Accepts a single .ndjson file, a directory of them, or the zip produced by
    the download script (`quickdraw.zip`, containing `raw/*.ndjson`).

    The full raw dump is ~23GB across 345 categories, so for a zip or directory
    you almost always want `max_per_category` (a few hundred each gives a
    well-balanced sketch pool) rather than `max_items`, which would otherwise
    exhaust itself on the alphabetically first categories.

    `categories` filters by file stem, e.g. ("cat", "house", "tree").
    `keep_key` filters drawings by their key_id before any limit is counted —
    how a dataset split takes only its own drawings
    (dataset_splits.quickdraw_split). Each drawing's meta carries its key_id and
    an `item` id.
    """
    path = _resolve(path, "quickdraw")

    if os.path.isfile(path) and path.endswith(".ndjson"):
        with open(path, "r") as f:
            return _read_quickdraw_lines(f, max_items, only_recognized, min_strokes, keep_key)

    archive = _Archive(path)
    members = archive.names(".ndjson")
    if categories is not None:
        wanted = set(categories)
        members = [m for m in members
                   if os.path.basename(m)[:-len(".ndjson")] in wanted]

    inks: List[Ink] = []
    for member in members:
        remaining = None if max_items is None else max_items - len(inks)
        if remaining is not None and remaining <= 0:
            break
        cap = min(x for x in (remaining, max_per_category) if x is not None) \
            if (remaining is not None or max_per_category is not None) else None
        with archive.open_text(member) as f:
            inks.extend(_read_quickdraw_lines(f, cap, only_recognized, min_strokes, keep_key))
    return inks


def _read_quickdraw_lines(lines: Iterable[str], max_items: Optional[int],
                           only_recognized: bool,
                           min_strokes: int = MIN_STROKES,
                           keep_key: Optional[Callable[[str], bool]] = None) -> List[Ink]:
    inks: List[Ink] = []
    for line in lines:
        if max_items is not None and len(inks) >= max_items:
            break
        rec = json.loads(line)
        if only_recognized and not rec.get("recognized", True):
            continue
        key = str(rec.get("key_id", ""))
        if keep_key is not None and not keep_key(key):
            continue
        strokes = []
        for stroke_arr in rec["drawing"]:
            xs, ys = stroke_arr[0], stroke_arr[1]
            ts = stroke_arr[2] if len(stroke_arr) > 2 and len(stroke_arr[2]) == len(xs) else None
            strokes.append(Stroke(x=[float(v) for v in xs],
                                   y=[float(v) for v in ys],
                                   t=[float(v) for v in ts] if ts is not None else None))
        if len(strokes) >= min_strokes:
            inks.append(Ink(strokes=strokes, label=rec.get("word"),
                             class_name="drawing", source="quickdraw",
                             meta={"key_id": key, "item": f"qd:{key}"}))
    return inks


# ----------------------------------------------------------------------------
# IAM-OnDB — the `lineStrokes` + `ascii` distribution
# ----------------------------------------------------------------------------
# Layout (extracted trees and the shipped archives are both accepted):
#
#   lineStrokes/<grp>/<doc>/<doc>-<NN>.xml   one file per text LINE, e.g. a01-000u-03.xml
#   ascii/<grp>/<doc>/<doc>.txt              one file per document, all its lines
#
# Each line XML is:
#   <WhiteboardCaptureSession>
#     <WhiteboardDescription>...</WhiteboardDescription>   <- line bbox, not ink
#     <StrokeSet>
#       <Stroke colour="black" start_time="769.05" end_time="769.64">
#         <Point x="1073" y="1058" time="769.05"/> ...
#
# x/y are whiteboard device units with y increasing DOWNWARD (verified: line 01 of
# a01-000u has mean y 1077, line 06 has 4657), so no flip is needed. Each point's
# `time` (seconds) is kept as Stroke.t.
#
# Each ascii file holds an `OCR:` block (the printed prompt) and a `CSR:` block
# (what the writer actually wrote). CSR is the correct transcription for the ink,
# and its non-blank lines correspond 1:1, in order, to that document's line XMLs
# (holds for 1728 of the 1730 documents that have both).
#
# IAM-OnDB's word-level segmentation lives in the `original` XML part, which this
# distribution does not include, so words are recovered geometrically via
# naive_split_line_into_words() and only labeled when the number of groups found
# happens to match the number of tokens in the transcription.

def _iam_doc_and_line(xml_name: str) -> Tuple[str, int]:
    """'lineStrokes/a01/a01-000/a01-000u-03.xml' -> ('a01-000u', 3)."""
    stem = os.path.basename(xml_name)[:-len(".xml")]
    doc, _, idx = stem.rpartition("-")
    try:
        return doc, int(idx)
    except ValueError:
        # no trailing line number — treat the whole stem as a single-line doc
        return stem, 0


def _iam_parse_line_strokes(data: bytes) -> List[Stroke]:
    """Parse one lineStrokes XML into Strokes, each point's time kept (Stroke.t)."""
    root = ET.fromstring(data)
    strokes: List[Stroke] = []
    for stroke_el in root.iter("Stroke"):
        xs, ys, ts = [], [], []
        for pt in stroke_el.iter("Point"):
            xs.append(float(pt.attrib["x"]))
            ys.append(float(pt.attrib["y"]))
            if "time" in pt.attrib:
                ts.append(float(pt.attrib["time"]))
        if xs:
            strokes.append(Stroke(x=xs, y=ys, t=ts if len(ts) == len(xs) else None))
    return strokes


def _iam_parse_transcriptions(archive: _Archive) -> Dict[str, List[str]]:
    """doc id -> transcribed lines, preferring the CSR block over OCR."""
    out: Dict[str, List[str]] = {}
    for name in archive.names(".txt"):
        # latin-1: the ascii files are ISO-8859-1, same as the stroke XML declares
        text = archive.read(name).decode("latin-1")
        match = re.search(r"^CSR:\s*$(.*)", text, re.M | re.S) or \
                re.search(r"^OCR:\s*$(.*)", text, re.M | re.S)
        if not match:
            continue
        lines = [ln.strip() for ln in match.group(1).splitlines() if ln.strip()]
        if lines:
            out[os.path.basename(name)[:-len(".txt")]] = lines
    return out


# How far one word's ink may be from the width its transcription predicts,
# as a ratio either way, before the whole line's labels are thrown out.
IAM_WORD_WIDTH_TOL = 2.6


def _words_align(words: List[Ink], tokens: Sequence[str]) -> bool:
    """Whether these ink groups plausibly *are* these tokens, in this order.

    Matching counts is not enough, and assuming it was is a real source of wrong
    labels. IAM's word boundaries here are recovered geometrically from the gaps
    between strokes, so a line where one word was split in two and two others
    were run together has exactly the right number of groups and the wrong label
    on every group between the split and the merge. It is invisible downstream —
    the box is right, and the text inside it belongs to the neighbour.

    Widths catch it. How much ink a word takes scales with how many characters
    were written, so one scale factor fitted across the whole line ought to
    predict every group in it; a line where some group is wildly off the
    prediction is one where the pairing has slipped. Judged per line rather than
    per word: the shift affects a run of words at once, and half-labelling a
    line would leave the wrong ones in.
    """
    widths = [w.bbox()[2] - w.bbox()[0] for w in words]
    lengths = [max(len(t), 1) for t in tokens]
    total_width, total_length = sum(widths), sum(lengths)
    if total_width <= 0 or total_length <= 0:
        return False

    per_character = total_width / total_length
    for width, length in zip(widths, lengths):
        expected = per_character * length
        if width <= 0 or expected <= 0:
            return False
        ratio = width / expected
        if ratio > IAM_WORD_WIDTH_TOL or ratio < 1.0 / IAM_WORD_WIDTH_TOL:
            return False
    return True


def _iam_writer_map(path: Optional[str] = None) -> Dict[str, int]:
    """IAM-OnDB form -> writer id, as iam_writers.py recovered it, or {} unbuilt."""
    path = path or data_path("iamondb_writers")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {form: int(writer) for form, writer in json.load(f).get("writers", {}).items()}


def load_iam_ondb(xml_dir: str = None, transcription_dir: str = None,
                   max_items: int = None,
                   min_strokes: int = MIN_STROKES,
                   writers_path: str = None) -> List[Ink]:
    """
    Load IAM-OnDB as one Ink per text LINE (class_name="textline"), each with
    .children = one Ink per WORD (class_name="word") and .label = the transcribed
    line text. Per the pipeline convention (see demo.py), a line that has word
    children keeps its own .strokes empty — all the ink lives in the children —
    so bboxes and stroke counts don't double-count.

    Both arguments accept an extracted tree or the shipped archive, and default
    to data/iamondb_linestrokes.tar.gz and data/iamondb_ascii.tar.

    `.writer` is the real writer ("iam:10042") where data/iamondb_writers.json
    (iam_writers.py, or `writers_path`) knows the form's writer, and the form
    ("iam:a01-000u") otherwise. `.meta` holds "form" and "writer_known", which a
    split needs: lines must be kept apart by writer and by text, and a form with
    no known writer can only go to training.

    Lines are returned in document order, then line order, so consecutive Inks are
    consecutive lines of the same page — handy for building multi-line textblocks.
    """
    strokes_archive = _Archive(_resolve(xml_dir, "iamondb_strokes"))
    transcriptions = _iam_parse_transcriptions(
        _Archive(_resolve(transcription_dir, "iamondb_ascii")))
    writers = _iam_writer_map(writers_path)

    # group the per-line XMLs by document, in line order
    by_doc: Dict[str, List[Tuple[int, str]]] = {}
    for name in strokes_archive.names(".xml"):
        doc, line_idx = _iam_doc_and_line(name)
        by_doc.setdefault(doc, []).append((line_idx, name))

    inks: List[Ink] = []
    for doc in sorted(by_doc):
        lines = sorted(by_doc[doc])
        texts = transcriptions.get(doc)
        # only trust the transcription when it lines up 1:1 with the stroke files
        if texts is not None and len(texts) != len(lines):
            texts = None

        for pos, (_, name) in enumerate(lines):
            if max_items is not None and len(inks) >= max_items:
                return inks
            strokes = _iam_parse_line_strokes(strokes_archive.read(name))
            if len(strokes) < min_strokes:
                continue        # a one- or two-stroke "line" is a stray mark

            text = texts[pos] if texts else None
            # `doc` (e.g. a01-000u) is the form: one copy of one text by one
            # writer. The writer themself comes from the map where it is known,
            # so a page's hand spans everything they wrote, not one form.
            writer_id = writers.get(doc)
            # `item` names this source line wherever its ink ends up — a row, a
            # table cell's run of words, a caption — so a synthesis run can count
            # how often each line is used. Words carry it too: a row rebuilt from
            # a line keeps the line's words, not the line itself.
            item = f"iam:{doc}:{pos}"
            line = Ink(strokes=strokes, label=text,
                       writer=f"iam:{writer_id}" if writer_id is not None else f"iam:{doc}",
                       class_name="textline", source="iam_ondb",
                       meta={"form": doc, "writer_known": writer_id is not None, "item": item})

            words = naive_split_line_into_words(line)
            if words:
                for word in words:
                    word.meta = {"item": item}
                tokens = text.split() if text else []
                if len(tokens) == len(words) and _words_align(words, tokens):
                    for word, token in zip(words, tokens):
                        word.label = token
                line.strokes = []       # all ink now lives in the word children
                line.children = words

            inks.append(line)

    return inks


# ----------------------------------------------------------------------------
# DeepWriting — npz bundle of normalized (dx, dy, pen_up) sequences
# ----------------------------------------------------------------------------
# `deepwriting.zip` holds deepwriting_training.npz (34,577 samples) and
# deepwriting_validation.npz (705). Both are loaded and concatenated by default:
# IAM-OnDo is what we hold out for val/test, so DeepWriting's own split is not
# needed here and the extra 705 samples are just more training content.
#
# Per-sample arrays (object arrays indexed by sample):
#   strokes      (T, 3) float32 — (dx, dy, pen_up), normalized by `mean`/`std`
#   texts        the full source line the writer was copying
#   char_labels  (T,) index into `alphabet`, per point
#   eoc_labels   (T,) 1 at the last point of each character
#   sow/eow_labels (T,) 1 at the first/last point of each word
#
# Reconstruction, verified by rendering: denormalize the first two channels
# (v*std + mean) and cumsum them. The result is already y-down like IAM and
# FC, so it needs NO flip. Worth stating explicitly because the mistake is
# invisible in a y-up plot (matplotlib's default), where the wrongly negated
# version is what looks upright — you only catch it once the ink lands on a
# y-down page next to IAM text and reads as a mirror image.
#
# `texts` cannot be used as the label: DeepWriting cuts lines into fixed-budget
# chunks, so a sample's ink is usually a *prefix or middle* of `texts` (only
# 65/705 validation samples contain as many words as `texts` claims). The
# per-point char/word annotations describe the ink that is actually present,
# so the real transcription is decoded from those instead.

def _deepwriting_label_words(char_labels, eoc, eow) -> List[List[int]]:
    """This sample's ink as a list of words, each a list of char_label values."""
    selected = eoc == 1
    words, current = [], []
    for label, end_of_word in zip(char_labels[selected], eow[selected]):
        current.append(int(label))
        if end_of_word:
            words.append(current)
            current = []
    if current:
        words.append(current)
    return words


def _deepwriting_char_table(data, limit: int = 8000) -> Dict[int, str]:
    """Recover the `char_labels` index -> character mapping for this file.

    The shipped `alphabet` array holds the right 70 characters but NOT in the
    order `char_labels` indexes them (that order turns out to be
    blank+punctuation, then digits, then uppercase, then lowercase). Decoding
    against `alphabet` directly, or against any single rotation of it, silently
    corrupts part of the range — uppercase in particular — which is exactly the
    kind of bug that survives a spot check.

    So derive it instead, with no bootstrap: `eoc`/`eow` give the exact character
    count of every word in the ink, and the ink is a contiguous run of words from
    `texts`, so matching those word lengths against `texts` usually pins the run
    to a single position. Every pinned sample then votes on what each label
    means, and the majority wins. On this release, ~38% of samples pin uniquely
    (13,076 of 34,577 in training), which resolves digits and both letter cases
    exactly; only rare punctuation stays noisy.
    """
    char_labels, eoc, eow, texts = (data["char_labels"], data["eoc_labels"],
                                    data["eow_labels"], data["texts"])
    votes: Dict[int, Dict[str, int]] = {}

    for i in range(min(limit, len(texts))):
        words = _deepwriting_label_words(char_labels[i], eoc[i], eow[i])
        target = str(texts[i]).split()
        if not words or len(words) > len(target):
            continue
        lengths = [len(w) for w in words]
        # the ink stops mid-line, so the final word may be a prefix of its target
        starts = [s for s in range(len(target) - len(words) + 1)
                  if all(len(target[s + k]) == lengths[k] for k in range(len(words) - 1))
                  and len(target[s + len(words) - 1]) >= lengths[-1]]
        if len(starts) != 1:
            continue    # ambiguous placement — it would poison the vote
        start = starts[0]
        for k, word in enumerate(words):
            for position, label in enumerate(word):
                counts = votes.setdefault(label, {})
                char = target[start + k][position]
                counts[char] = counts.get(char, 0) + 1

    return {label: max(counts.items(), key=lambda kv: kv[1])[0]
            for label, counts in votes.items()}


def _deepwriting_decode(words: List[List[int]], table: Dict[int, str]) -> List[str]:
    """Words actually present in this sample's ink, as text."""
    decoded = ["".join(table.get(label, "") for label in word) for word in words]
    return [w for w in decoded if w]


def load_deepwriting(path: str = None, max_items: int = None,
                      splits: Sequence[str] = ("training", "validation"),
                      scale: float = DEEPWRITING_SCALE,
                      min_strokes: int = MIN_STROKES) -> List[Ink]:
    """
    Load DeepWriting as one Ink per text LINE (class_name="textline") with
    .children = one Ink per WORD, matching the IAM-OnDB loader's shape.

    Word boundaries come from the dataset's own per-point `sow_labels`, which is
    a real annotation rather than the geometric guess used for IAM-OnDB — each
    stroke is assigned to whichever word owns most of its points, so strokes stay
    intact across cursive word joins.

    `splits` selects which .npz members to read; both are included by default.
    Requires numpy (only this loader does).
    """
    import numpy as np  # local: nothing else in the pipeline needs numpy

    archive = _Archive(_resolve(path, "deepwriting"))
    members = [n for n in archive.names(".npz")
               if any(s in os.path.basename(n) for s in splits)]
    if not members:
        raise FileNotFoundError(f"no .npz for splits {splits!r} in the DeepWriting archive")

    inks: List[Ink] = []
    for member in members:
        data = np.load(io.BytesIO(archive.read(member)), allow_pickle=True)
        strokes_all, mean, std = data["strokes"], data["mean"], data["std"]
        subjects = data["subject_labels"]
        char_labels, eoc, eow, sow = (data["char_labels"], data["eoc_labels"],
                                      data["eow_labels"], data["sow_labels"])
        table = _deepwriting_char_table(data)

        for i in range(len(strokes_all)):
            if max_items is not None and len(inks) >= max_items:
                return inks

            raw = strokes_all[i].astype(np.float64)
            deltas = raw[:, :2] * std[:2] + mean[:2]
            xy = np.cumsum(deltas, axis=0) * scale
            pen_up = raw[:, 2]

            # split into strokes at pen-up points, and tag each point's word index
            word_index = np.cumsum(sow[i]) - 1
            bounds = list(np.where(pen_up == 1)[0] + 1)
            starts = [0] + bounds
            ends = bounds + [len(xy)]
            spans = [(a, b) for a, b in zip(starts, ends) if b - a >= 2]
            if len(spans) < min_strokes:
                continue

            strokes, owners = [], []
            for a, b in spans:
                strokes.append(Stroke(x=[float(v) for v in xy[a:b, 0]],
                                       y=[float(v) for v in xy[a:b, 1]]))
                # a cursive stroke can straddle a word join — give it to the
                # word holding most of its points
                counts = np.bincount(np.maximum(word_index[a:b], 0))
                owners.append(int(counts.argmax()))

            texts = _deepwriting_decode(
                _deepwriting_label_words(char_labels[i], eoc[i], eow[i]), table)
            writer = f"dw:{int(subjects[i])}"

            # A very short word can end up with no stroke of its own (every
            # stroke it touches is dominated by a neighbour), so build the line
            # label from the words that actually carry ink rather than from the
            # full decode — otherwise the line claims text that isn't drawn.
            words: List[Ink] = []
            for w in sorted(set(owners)):
                grouped = [s for s, o in zip(strokes, owners) if o == w]
                if grouped:
                    words.append(Ink(strokes=grouped, class_name="word",
                                      label=texts[w] if w < len(texts) else None,
                                      writer=writer, source="deepwriting"))

            present = [w.label for w in words if w.label]
            inks.append(Ink(strokes=[], children=words, class_name="textline",
                             label=" ".join(present) if present else None,
                             writer=writer, source="deepwriting"))
    return inks


# ----------------------------------------------------------------------------
# MathWriting — InkML, same team/paper as InkFM
# ----------------------------------------------------------------------------
# Get the official release (the ink is not on the Hugging Face mirror — see the
# note in load_mathwriting):
#   curl -L -o data/mathwriting.tgz \
#     https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz
#
# The archive holds one InkML per expression under train/ valid/ test/ symbols/
# synthetic/, each:
#   <ink xmlns="http://www.w3.org/2003/InkML">
#     <annotation type="label">d\approx \sqrt{2 \cdot k\cdot R\cdot h}</annotation>
#     <annotation type="normalizedLabel">d\approx\sqrt{...}</annotation>
#     <traceFormat><channel name="X"/><channel name="Y"/><channel name="T" units="ms"/>
#     <trace id="0">1140.00 453.00 2600.0,1141.00 461.00 2730.0,...</trace>
#
# `normalizedLabel` is the ground truth per the dataset readme (it falls back to
# `label` for the symbols split, which has no normalized form). The T channel
# (ms) is kept as Stroke.t.

_INKML_NS = "{http://www.w3.org/2003/InkML}"


def _parse_inkml_traces(root: ET.Element) -> Tuple[List[Stroke], Dict[str, int]]:
    """Traces of one InkML file, plus a trace-id -> stroke-index map.

    X and Y, and a third channel where every point of a trace has one
    (MathWriting's T, in ms) as Stroke.t.
    """
    strokes: List[Stroke] = []
    index_of: Dict[str, int] = {}
    for trace in root.iter(_INKML_NS + "trace"):
        xs, ys, ts = [], [], []
        for point in (trace.text or "").split(","):
            parts = point.split()
            if len(parts) < 2:
                continue
            xs.append(float(parts[0]))
            ys.append(float(parts[1]))
            if len(parts) >= 3:
                ts.append(float(parts[2]))
        if not xs:
            continue
        index_of[trace.attrib.get("id", str(len(strokes)))] = len(strokes)
        strokes.append(Stroke(x=xs, y=ys, t=ts if len(ts) == len(xs) else None))
    return strokes, index_of


def _inkml_annotations(root: ET.Element) -> Dict[str, str]:
    return {a.attrib.get("type", ""): (a.text or "")
            for a in root.iter(_INKML_NS + "annotation")}


def load_mathwriting(path: str = None, max_items: int = None,
                      splits: Sequence[str] = ("train", "synthetic"),
                      min_strokes: int = MIN_STROKES) -> List[Ink]:
    """
    Load MathWriting as one flat Ink per expression, .label = normalized LaTeX.

    There is no meaningful word/line hierarchy to preserve (an expression is one
    unit) and .class_name is left None — math isn't a segmentation class in the
    target taxonomy, so these are used purely as visual filler inside textblocks
    and diagrams, for stroke-appearance diversity.

    `splits` selects which directories of the release to read. It defaults to the
    three training splits the dataset readme designates; `valid`/`test` are left
    out so held-out ink never leaks into generated training pages.

    NOTE on the Hugging Face mirror: `andito/mathwriting-google` is *rendered
    JPEGs* plus LaTeX strings (features: image, label, normalized_label, source,
    sample_id, ink_path) — it carries no stroke coordinates, so it cannot feed
    this pipeline, which needs (x, y, t). Use the official InkML release above.
    """
    archive = _Archive(_resolve(path, "mathwriting"))
    wanted = tuple(splits)

    inks: List[Ink] = []
    for name in archive.names(".inkml"):
        parts = name.split("/")
        if wanted and not any(p in wanted for p in parts[:-1]):
            continue
        if max_items is not None and len(inks) >= max_items:
            break

        root = ET.fromstring(archive.read(name))
        strokes, _ = _parse_inkml_traces(root)
        if len(strokes) < min_strokes:
            continue
        annotations = _inkml_annotations(root)
        inks.append(Ink(
            strokes=strokes,
            label=annotations.get("normalizedLabel") or annotations.get("label"),
            class_name=None,
            source="mathwriting",
        ))
    return inks


# ----------------------------------------------------------------------------
# FC — the Bresler / Průša / Hlaváč online flowchart database (FC_1.0, CMP, CTU Prague)
# ----------------------------------------------------------------------------
# Bresler M., Průša D., Hlaváč V.: On-line recognition of sketched arrow-connected diagrams. IJDAR,
# DOI 10.1007/s10032-016-0269-z. 672 InkML flowcharts: 28 layouts (001-021, some with a second drawing "b"),
# each drawn by all 24 writers (writer000-writer023), as FC_1.0/writer<WWW>_fc_<LLL>[b].inkml.
#
# Traces carry X Y T P; T is milliseconds from the trace's own timeStamp, so a point's time is the two added.
# y increases downward, as in IAM (the "Start" terminator of writer000_fc_001 sits at the top of the page).
# No InkML namespace.
#
#   <symbols>
#     <traceGroup id="0"><annotation type="truth">text</annotation>
#                        <annotation type="textMeaning">Read $N$</annotation><traceView traceDataRef="15"/>...
#     <traceGroup id="10"><annotation type="truth">arrow</annotation><head>traceViews</head><shaft>traceViews</shaft>
#     <traceGroup id="18"><annotation type="truth">data</annotation>...
#   <relations>
#     <symbolGroup><annotation type="truth">arrow_connection</annotation>   arrow, from-node, to-node
#     <symbolGroup><annotation type="truth">text_inside</annotation>        text, node
#     <symbolGroup><annotation type="truth">arrow_label</annotation>        text, arrow
#
# Symbol classes across the 672 files: text (5732), arrow (4569), process (1300), data (1221), terminator (908),
# decision (768), connection (382). Every text is either inside a node (4246) or an arrow's label (1486), and every
# arrow names the two nodes it joins.

# FC symbol class -> our taxonomy.
#
# Every node shape is one class, `enclosure`, the same class as a box or oval drawn round writing. Which shape it
# is — process rectangle, terminator stadium, data parallelogram, decision diamond, connection circle — is not a
# class: once the enclosure is found, where it sits and what it holds say what kind it is. The original annotation
# is kept per symbol in meta["node_type"], which is also what the label page reads to draw and anchor to its outline.
FC_CLASS_MAP = {
    "process": "enclosure",
    "terminator": "enclosure",
    "data": "enclosure",
    "decision": "enclosure",
    "connection": "enclosure",
    "arrow": "arrow",
    "text": "textline",
}

# The shapes an arrow may join, i.e. everything FC_CLASS_MAP treats as a node.
FC_NODE_CLASSES = ("enclosure",)


def _fc_arrow_label(text: Optional[str]) -> Optional[str]:
    """An arrow's caption as a textline's label, or None when it is not one to keep.

    A shape's caption is rewritten (block_generators.restock_diagram_text), but an arrow's is kept as written, so
    its transcription is the label. Nearly all are a word or two ("Yes", "Then", "0", "True"). Seven have no
    transcription, and three are written in the dataset's LaTeX-in-dollars ("Output $A$"), mixing text and maths a
    textline label cannot hold; their ink is left out rather than labelled with a guess.
    """
    if not text:
        return None
    text = text.strip()
    if not text or "$" in text or "\\" in text:
        return None
    return text


def _fc_traces(root: ET.Element) -> Tuple[List[Stroke], Dict[str, int]]:
    """Traces of one FC file and a trace-id -> stroke-index map; Stroke.t is timeStamp + T (ms)."""
    strokes: List[Stroke] = []
    index_of: Dict[str, int] = {}
    for trace in root.iter("trace"):
        start = float(trace.attrib.get("timeStamp") or 0.0)
        xs, ys, ts = [], [], []
        for point in (trace.text or "").split(","):
            parts = point.split()
            if len(parts) < 2:
                continue
            xs.append(float(parts[0]))
            ys.append(float(parts[1]))
            if len(parts) >= 3:
                ts.append(start + float(parts[2]))
        if not xs:
            continue
        index_of[trace.attrib.get("id", str(len(strokes)))] = len(strokes)
        strokes.append(Stroke(x=xs, y=ys, t=ts if len(ts) == len(xs) else None))
    return strokes, index_of


def _strip_namespaces(root: ET.Element) -> ET.Element:
    for element in root.iter():
        if isinstance(element.tag, str) and "}" in element.tag:
            element.tag = element.tag.split("}", 1)[1]
    return root


def load_fc(path: str = None, max_items: int = None) -> List[Ink]:
    """
    Load the FC database as one Ink per flowchart, whose children are the annotated symbols mapped onto the target
    taxonomy (enclosure / arrow / textline). The flowchart is marked meta["flowchart"], and is given its class,
    `flowchart`, when block_generators.make_diagram places it. A node's caption is nested inside that node here;
    make_diagram brings it out as a line of the chart. It drops straight into a flowchart slot without any
    procedural drawing.

    The flowchart's graph is attached to the diagram as meta["graph"], and each arrow carries
    meta["source_id"]/["target_id"], so an edge is drawn between the nodes the dataset says it joins rather than
    inferred from the ink.

    Captions belonging to an *arrow* rather than to a node are not nested (that would inflate the arrow's box over
    the text beside it); they stay at diagram level, and the arrow carries meta["caption_id"]. One the label of
    which cannot be kept (_fc_arrow_label) is left out, ink and all.

    Each chart's label is its file's stem ("writer000_fc_001b"), and its source "fc".
    """
    archive = _Archive(_resolve(path, "fc"))

    inks: List[Ink] = []
    for name in archive.names(".inkml"):
        if max_items is not None and len(inks) >= max_items:
            break
        root = _strip_namespaces(ET.fromstring(archive.read(name)))
        chart = os.path.basename(name)[:-len(".inkml")]

        strokes, index_of = _fc_traces(root)
        symbols_el, relations_el = root.find("symbols"), root.find("relations")
        if not strokes or symbols_el is None:
            continue

        groups: Dict[str, Tuple[str, Optional[str], List[Stroke]]] = {}     # symbol id -> (fc class, meaning, ink)
        for group in symbols_el.findall("traceGroup"):
            sid = group.attrib.get("id")
            kind = meaning = None
            for note in group.findall("annotation"):
                if note.attrib.get("type") == "truth":
                    kind = (note.text or "").strip()
                elif note.attrib.get("type") == "textMeaning":
                    meaning = note.text
            refs = [v.attrib.get("traceDataRef") for v in group.iter("traceView")]
            grouped = [strokes[index_of[r]] for r in refs if r in index_of]
            if sid is not None and kind and grouped:
                groups[sid] = (kind, meaning, grouped)

        edges: Dict[str, Tuple[str, str]] = {}          # arrow id -> (from node id, to node id)
        inside: Dict[str, str] = {}                     # text id -> node id
        arrow_caption: Dict[str, str] = {}              # arrow id -> text id
        labelled: set = set()                           # text ids that label an arrow
        for relation in (relations_el.findall("symbolGroup") if relations_el is not None else []):
            note = relation.find("annotation")
            kind = (note.text or "").strip() if note is not None else ""
            refs = [v.attrib.get("symbolDataRef") for v in relation.findall("symbolView")]
            if kind == "arrow_connection" and len(refs) == 3:
                edges[refs[0]] = (refs[1], refs[2])
            elif kind == "text_inside" and len(refs) == 2:
                inside[refs[0]] = refs[1]
            elif kind == "arrow_label" and len(refs) == 2:
                labelled.add(refs[0])
                arrow_caption.setdefault(refs[1], refs[0])

        symbols: Dict[str, Ink] = {}
        node_types: Dict[str, str] = {}
        for sid, (kind, meaning, grouped) in groups.items():
            class_name = FC_CLASS_MAP.get(kind)
            if class_name is None or class_name == "textline":
                continue
            meta = {"id": sid}
            if class_name == "enclosure":
                meta["node_type"] = node_types[sid] = kind
            symbols[sid] = Ink(strokes=grouped, class_name=class_name, source="fc", meta=meta)

        children: List[Ink] = []
        captions: Dict[str, str] = {}                   # text id -> its transcription, for the graph
        for sid, (kind, meaning, grouped) in groups.items():
            if FC_CLASS_MAP.get(kind) != "textline":
                continue
            owner = inside.get(sid)
            if owner in symbols and symbols[owner].class_name in FC_NODE_CLASSES:
                # a shape's caption: rewritten downstream, so its own transcription only rides along
                text = Ink(strokes=grouped, class_name="textline", source="fc",
                           label=(meaning or "").strip() or None, meta={"id": sid})
                symbols[owner].children.append(text)
                if text.label:
                    captions[sid] = text.label
                continue
            label = _fc_arrow_label(meaning) if sid in labelled else None
            if label is None:
                continue                    # see _fc_arrow_label; a text with no relation at all is not in the set
            captions[sid] = label
            children.append(Ink(strokes=grouped, class_name="textline", source="fc", label=label,
                                meta={"id": sid}))
        kept_text = {(c.meta or {}).get("id") for c in children}

        for sid, ink in symbols.items():
            if ink.class_name == "arrow":
                if sid in edges:
                    ink.meta["source_id"], ink.meta["target_id"] = edges[sid]
                if arrow_caption.get(sid) in kept_text:
                    ink.meta["caption_id"] = arrow_caption[sid]
        children = list(symbols.values()) + children
        if not children:
            continue

        # The flowchart's graph, carried on the diagram itself, restricted to nodes that made it into the tree, so
        # an edge present in graph["edges"] can always be drawn.
        present = {sid for sid, ink in symbols.items() if ink.class_name in FC_NODE_CLASSES}
        caption_of_node = {node: text for text, node in inside.items()}
        graph = {
            "nodes": {sid: {"class": symbols[sid].class_name, "type": node_types.get(sid),
                            "text": captions.get(caption_of_node.get(sid))}
                      for sid in present},
            "edges": [{"id": aid, "source": s, "target": t,
                       "caption_id": symbols[aid].meta.get("caption_id"),
                       "text": captions.get(symbols[aid].meta.get("caption_id"))}
                      for aid, (s, t) in edges.items()
                      if aid in symbols and s in present and t in present],
        }
        inks.append(Ink(strokes=[], children=children, label=chart, source="fc",
                        meta={"flowchart": True, "graph": graph}))
    return inks


# ----------------------------------------------------------------------------
# EMNIST / NIST Special Database 19 — isolated handwritten characters
# ----------------------------------------------------------------------------
# The odd one out: every other source here is online ink, captured pen-down as a
# sequence of points. SD19 is offline — scanned paper, 128x128 bilevel rasters,
# no pen and no clock. To reach the common format at all it has to be
# *vectorized*: thinned to a one-pixel skeleton, and the skeleton traced into
# polylines. What comes back is therefore an honest reconstruction of the pen's
# path rather than a recording of it. Stroke order and direction are inferred
# (top-left first, the order a right-hander tends to write in).
#
# Archive layout — data/emnist.zip is the NIST SD19 1st Edition (1995) release:
#     1stEdition1995/data/by_class/<hex>/hsf_<n>.mis     one file per writer set
#     1stEdition1995/data/by_class/<hex>/train_<hex>.mis
# <hex> is the character's ASCII code, so 41 is "A" and 6f is "o": 62 classes,
# the ten digits plus both cases.
#
# .mis ("multiple image set") is a 288-byte NUL-separated ASCII IHead header,
# then eight more bytes holding the entry height, then one CCITT Group 4 stream
# for the whole set — the glyphs stacked into a single tall bitmap, 128 wide by
# 128*count high. Group 4 is fax compression, so Pillow decodes it once it is
# wrapped in the TIFF container it expects; and because the stream is decoded
# strictly top to bottom, asking for fewer rows than the file holds reads only
# the glyphs wanted instead of all 6611 of them.

EMNIST_ENTRY = 128                  # every SD19 glyph is 128x128
EMNIST_HEADER = 288 + 8             # IHead, then the entry-height field
EMNIST_TARGET_H = 40.0              # unit height a vectorized glyph is scaled to

# The skeleton is traced at this height rather than at 128px. Thinning a stroke
# drawn ten pixels wide leaves the noise of its own edges behind as hairs and
# stubs; shrinking first turns that edge noise into grey, which the re-threshold
# drops. It also makes the trace about seven times faster.
EMNIST_WORK_H = 48

# A *dangling* branch shorter than this fraction of the glyph's height is a
# thinning artefact — the little spur thrown off at a corner or a stroke end —
# so it is dropped rather than drawn. Only dangling ones: a short run between
# two junctions is the waist of an "A" or the middle of an "8", and dropping
# those by length alone tore holes in the characters.
EMNIST_MIN_SPUR = 0.11

# Douglas-Peucker tolerance for the traced polyline, as a fraction of the glyph
# height. The skeleton is a staircase of single pixels; this is what turns it
# back into the handful of points a pen would have been sampled at.
EMNIST_SIMPLIFY = 0.02


def _mis_header(raw: bytes) -> Dict[str, str]:
    """The IHead fields of a .mis, in order, as strings.

    The header is a struct of fixed-width char arrays that NIST wrote as
    NUL-terminated strings, so the fields are recovered by splitting rather than
    by unpacking at fixed offsets — the padding between them is not uniform.
    """
    tokens = [t.decode("latin-1") for t in raw[:288].split(b"\x00") if t]
    names = ["hdr_size", "id", "created", "width", "height", "depth", "density",
             "compress", "complen", "align", "unitsize", "flags1", "whitepix",
             "flags2"]
    return dict(zip(names, tokens))


def _tiff_g4(data: bytes, width: int, height: int) -> bytes:
    """`data` (a raw CCITT Group 4 stream) wrapped in a minimal TIFF."""
    import struct
    tags = sorted([
        (256, 3, width),        # ImageWidth
        (257, 4, height),       # ImageLength
        (258, 3, 1),            # BitsPerSample
        (259, 3, 4),            # Compression: CCITT T.6
        (262, 3, 0),            # Photometric: WhiteIsZero, as MIS stores it
        (266, 3, 1),            # FillOrder: most significant bit first
        (273, 4, 8),            # StripOffsets — the data sits right after the header
        (277, 3, 1),            # SamplesPerPixel
        (278, 4, height),       # RowsPerStrip: the whole image is one strip
        (279, 4, len(data)),    # StripByteCounts
    ])
    out = io.BytesIO()
    out.write(b"II" + struct.pack("<HI", 42, 8 + len(data)))
    out.write(data)
    out.write(struct.pack("<H", len(tags)))
    for tag, typ, value in tags:
        out.write(struct.pack("<HHI", tag, typ, 1))
        # a SHORT is written into the low half of the four-byte value field
        out.write(struct.pack("<HH", value, 0) if typ == 3
                  else struct.pack("<I", value))
    out.write(struct.pack("<I", 0))     # no next IFD
    return out.getvalue()


def _mis_glyphs(raw: bytes, want: int):
    """The first `want` glyphs of a .mis, as boolean ink-is-True arrays."""
    import numpy as np
    from PIL import Image

    header = _mis_header(raw)
    total = int(header["height"]) // EMNIST_ENTRY
    count = min(want, total)
    rows = count * EMNIST_ENTRY
    sheet = Image.open(io.BytesIO(_tiff_g4(raw[EMNIST_HEADER:],
                                           int(header["width"]), rows)))
    # WhiteIsZero is declared above, so Pillow hands back True for paper; the
    # ink is what is left
    page = ~np.asarray(sheet, dtype=bool)
    return [page[i * EMNIST_ENTRY:(i + 1) * EMNIST_ENTRY] for i in range(count)]


def _shift(a, dr: int, dc: int):
    """`a` moved by (dr, dc), with zeros shifted in — no wraparound."""
    import numpy as np
    out = np.zeros_like(a)
    rows = slice(max(dr, 0), a.shape[0] + min(dr, 0))
    cols = slice(max(dc, 0), a.shape[1] + min(dc, 0))
    src_rows = slice(max(-dr, 0), a.shape[0] + min(-dr, 0))
    src_cols = slice(max(-dc, 0), a.shape[1] + min(-dc, 0))
    out[rows, cols] = a[src_rows, src_cols]
    return out


# The eight neighbours in the ring order Zhang-Suen counts them in: N, NE, E,
# SE, S, SW, W, NW. Rows increase downward, so north is one row *back*.
_RING = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def _thin(mask):
    """Zhang-Suen thinning: a blob of ink reduced to a one-pixel-wide skeleton.

    Both sub-iterations are applied to the whole image at once rather than pixel
    by pixel, which is what makes this fast enough in pure numpy to run over
    thousands of glyphs.
    """
    import numpy as np
    img = mask.astype(np.uint8)
    while True:
        removed = False
        for step in (0, 1):
            ring = [_shift(img, dr, dc) for dr, dc in _RING]
            p2, p3, p4, p5, p6, p7, p8, p9 = ring
            neighbours = sum(ring)
            # transitions from white to ink going once round the ring; exactly
            # one of them means the pixel is not a bridge between two branches
            loop = ring + [ring[0]]
            crossings = sum(((loop[i] == 0) & (loop[i + 1] == 1)).astype(np.uint8)
                            for i in range(8))
            if step == 0:
                free = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                free = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            doomed = ((img == 1) & (neighbours >= 2) & (neighbours <= 6)
                      & (crossings == 1) & free)
            if doomed.any():
                img[doomed] = 0
                removed = True
        if not removed:
            return img.astype(bool)


def _trace_skeleton(skeleton) -> List[Tuple[List[Tuple[float, float]], bool]]:
    """Break a skeleton into polylines, cut at its ends and its junctions.

    Each run of degree-two pixels between two such nodes becomes one polyline.
    A closed loop with no node on it at all — the bowl of an "o" — has no end to
    start from, so one is chosen arbitrarily and the loop is closed back onto it.

    Each polyline comes back paired with whether it dangles: whether either of
    its ends is a loose end rather than a junction. That is what tells a spur
    worth pruning from a short run that is holding two parts of the character
    together.
    """
    import numpy as np

    ink = {(int(r), int(c)) for r, c in zip(*np.nonzero(skeleton))}

    def neighbours(pixel):
        r, c = pixel
        return [(r + dr, c + dc) for dr, dc in _RING if (r + dr, c + dc) in ink]

    degree = {p: len(neighbours(p)) for p in ink}
    nodes = {p for p, d in degree.items() if d != 2}

    loose = {p for p, d in degree.items() if d == 1}
    paths: List[Tuple[List[Tuple[float, float]], bool]] = []
    walked = set()          # undirected pixel pairs already covered

    def walk(start, second):
        path = [start, second]
        walked.add(frozenset((start, second)))
        while path[-1] not in nodes:
            here = path[-1]
            ahead = [n for n in neighbours(here) if n != path[-2]]
            if not ahead:
                break
            step = ahead[0]
            if frozenset((here, step)) in walked:
                break
            walked.add(frozenset((here, step)))
            path.append(step)
        return path

    # sorted so the trace is deterministic and starts at the top-left, which is
    # roughly where a right-handed writer starts the character
    for node in sorted(nodes):
        for neighbour in neighbours(node):
            if frozenset((node, neighbour)) not in walked:
                path = walk(node, neighbour)
                paths.append((path, path[0] in loose or path[-1] in loose))

    # whatever is left is a loop of degree-two pixels with no node on it
    for pixel in sorted(ink):
        if degree[pixel] != 2:
            continue
        if any(frozenset((pixel, n)) in walked for n in neighbours(pixel)):
            continue
        loop = walk(pixel, neighbours(pixel)[0])
        if len(loop) > 2:
            loop.append(loop[0])
        paths.append((loop, False))       # a closed loop dangles nowhere

    return [([(float(c), float(r)) for r, c in path], dangling)
            for path, dangling in paths]


def _simplify(points: Sequence[Tuple[float, float]], tolerance: float):
    """Douglas-Peucker: the polyline through the same shape with fewer points."""
    if len(points) < 3:
        return list(points)
    (x0, y0), (x1, y1) = points[0], points[-1]
    dx, dy = x1 - x0, y1 - y0
    span = (dx * dx + dy * dy) ** 0.5
    worst, index = -1.0, 0
    for i in range(1, len(points) - 1):
        px, py = points[i]
        if span < 1e-9:
            offset = ((px - x0) ** 2 + (py - y0) ** 2) ** 0.5
        else:
            offset = abs(dy * (px - x0) - dx * (py - y0)) / span
        if offset > worst:
            worst, index = offset, i
    if worst <= tolerance:
        return [points[0], points[-1]]
    return (_simplify(points[:index + 1], tolerance)[:-1]
            + _simplify(points[index:], tolerance))


def _vectorize_glyph(bitmap, target_h: float = EMNIST_TARGET_H) -> List[Stroke]:
    """A scanned glyph turned into the strokes a pen would have drawn it with."""
    import numpy as np
    from PIL import Image

    rows, cols = np.nonzero(bitmap)
    if len(rows) == 0:
        return []
    # crop to the ink before scaling, so every glyph is worked on at the same
    # size whatever its share of the 128x128 cell
    patch = bitmap[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    height, width = patch.shape
    scale = EMNIST_WORK_H / max(height, 1)
    small = Image.fromarray(patch.astype(np.uint8) * 255).resize(
        (max(2, int(round(width * scale))), max(2, int(round(height * scale)))),
        Image.BILINEAR)
    # halfway: the greys the resize produced are the blurred edges of the stroke
    work = np.asarray(small) > 127
    if not work.any():
        return []

    skeleton = _thin(work)
    paths = _trace_skeleton(skeleton)
    if not paths:
        return []

    span = max(work.shape[0], 1)
    keep = [p for p, dangling in paths
            if len(p) > 1 and not (dangling
                                   and _path_length(p) < EMNIST_MIN_SPUR * span)]
    # a dot, a comma, the tittle of an "i": every path was a stub, and dropping
    # them all would leave nothing, so the longest one stands for the mark
    if not keep:
        keep = [max((p for p, _ in paths), key=_path_length)]

    unit = target_h / span                    # working pixels -> page units
    polylines = []
    for path in keep:
        simple = _simplify(path, EMNIST_SIMPLIFY * span)
        polylines.append([(x * unit, y * unit) for x, y in simple])

    # written in reading order: down the page, then across
    polylines.sort(key=lambda line: (round(line[0][1], 1), line[0][0]))
    return [Stroke(x=[p[0] for p in line], y=[p[1] for p in line])
            for line in polylines]


def _path_length(points: Sequence[Tuple[float, float]]) -> float:
    return sum(((points[i + 1][0] - points[i][0]) ** 2
                + (points[i + 1][1] - points[i][1]) ** 2) ** 0.5
               for i in range(len(points) - 1))


# The writer sets. hsf_4 is the one EMNIST and the NIST benchmarks hold out for
# testing, so it is left out of the default read rather than trained on.
EMNIST_TRAIN_SETS = ("hsf_0", "hsf_1", "hsf_2", "hsf_3", "hsf_6", "hsf_7")

EMNIST_DIGITS = "0123456789"
EMNIST_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
EMNIST_LOWER = "abcdefghijklmnopqrstuvwxyz"


def load_emnist(path: str = None, characters: Optional[Iterable[str]] = None,
                max_per_class: int = 40, sets: Sequence[str] = EMNIST_TRAIN_SETS,
                target_h: float = EMNIST_TARGET_H) -> List[Ink]:
    """Isolated handwritten characters, vectorized into strokes.

    Returns one Ink per glyph, `class_name="character"` and `.label` the single
    character it is — so a caller can pick the ink for a letter it wants to
    write. `characters` limits which classes are read at all, which is the
    difference between a second and a minute: the archive holds 62 of them.

    A glyph is scanned paper, so what comes back is a reconstruction of the pen
    path, not a recording of one — see the note above this function.
    """
    path = _resolve(path, "emnist")
    archive = _Archive(path)

    wanted = set(characters) if characters is not None else None
    members: Dict[str, List[str]] = {}
    for name in archive.names(".mis"):
        parts = name.split("/")
        if len(parts) < 2 or "by_class" not in parts:
            continue
        folder, stem = parts[-2], parts[-1][:-len(".mis")]
        try:
            char = chr(int(folder, 16))
        except ValueError:
            continue
        if wanted is not None and char not in wanted:
            continue
        if stem in sets:
            members.setdefault(char, []).append(name)

    inks: List[Ink] = []
    for char in sorted(members):
        taken = 0
        for name in sorted(members[char]):
            if taken >= max_per_class:
                break
            raw = archive.read(name)
            for bitmap in _mis_glyphs(raw, max_per_class - taken):
                strokes = _vectorize_glyph(bitmap, target_h)
                if not strokes:
                    continue
                taken += 1
                inks.append(Ink(strokes=strokes, label=char,
                                 class_name="character", source="emnist",
                                 writer=os.path.basename(name)[:-len(".mis")]))
    return inks


def load_emnist_pool(path: str = None, characters: Optional[Iterable[str]] = None,
                     max_per_class: int = 24) -> Dict[str, List[Ink]]:
    """load_emnist keyed by character — what a caller writing a word wants.

    Anything that writes text a character at a time (geometry_diagrams naming a
    corner "B", or writing "12" against a side) needs to look the ink up by the
    character it wants, and several samples of each so the same letter is not
    drawn identically twice on one page.
    """
    if characters is None:
        characters = EMNIST_DIGITS + EMNIST_UPPER + EMNIST_LOWER
    pool: Dict[str, List[Ink]] = {}
    for ink in load_emnist(path, characters=characters,
                            max_per_class=max_per_class):
        pool.setdefault(ink.label, []).append(ink)
    return pool


# ----------------------------------------------------------------------------
# Convenience: word-splitting fallback for sources that only give you full lines
# ----------------------------------------------------------------------------

def naive_split_line_into_words(line_ink: Ink, gap_threshold: float = None) -> List[Ink]:
    """
    Fallback word segmentation when a source only gives you line-level ink with
    no word boundaries. Groups strokes into words based on horizontal gaps
    between consecutive strokes' bounding boxes. Not as good as real annotations
    — prefer real word boundaries (IAM-OnDB's `original` part, DeepWriting's
    sow_labels) wherever available.
    """
    strokes = sorted(line_ink.strokes, key=lambda s: min(s.x))
    if not strokes:
        return []
    if gap_threshold is None:
        heights = [s.bbox()[3] - s.bbox()[1] for s in strokes]
        avg_h = sum(heights) / len(heights) if heights else 10.0
        gap_threshold = avg_h * 0.6  # heuristic: gap > ~60% of stroke height = word break

    words: List[List[Stroke]] = [[strokes[0]]]
    prev_right = strokes[0].bbox()[2]
    for s in strokes[1:]:
        left = s.bbox()[0]
        if left - prev_right > gap_threshold:
            words.append([])
        words[-1].append(s)
        prev_right = max(prev_right, s.bbox()[2])

    return [Ink(strokes=w, class_name="word", source=line_ink.source,
                 writer=line_ink.writer) for w in words]
