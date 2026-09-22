"""
recog_eval.py — recognition after the page cascade: the textlines and formulas it detected, read the way the
recognition data draws them, and the cross-entropy of their transcriptions.

    remove = MarkRemover("marks/model.pt", device)       # optional: the mark separation network
    recognize = Recognizer(model_read_fn(base, processor, device, dtype, max_new_tokens, batch_size), iou, min_box_px,
                           remove_marks=remove)
    seg_eval.validate_pages(..., recognize=recognize)       # train.py, seg_eval.py (the test run), explore_worker.py

Groups. On each page, the boxes the cascade predicted as textline, in the order it predicted them, are taken
ten at a time (recog_layout.ROWS; the last group may hold fewer), then the same for formula. A box's ink is
every stroke of the page's textlines, formulas, enclosures and line marks (the page JSON's "ink"; a line's
points carry their times, a mark's none) that has at least half its points inside the box, so a box that is
misplaced, too small or too large reads what it really covers, marks included.

Marks. Given a mark separation network (remove_marks: MarkRemover, training/marks_cnn's model.pt), each box's
crop of the page image, CROP_PAD pixels wider all round, is scaled into the network's 600 x 100 input the way
marks_data.py fits its crops, and a point of the box's ink is kept only where the network's writing channel is
over KEEP_PROB within a pixel of it; a stroke is cut where its points are dropped. Points off the crop are not
judged and stay. A mark's ink that crosses the writing survives this, without times of its own (see Image). The
recognition samples have no marks at all: they are made from the ground truth, which knows which strokes are marks.
Without the network every mark stroke in the box is read with the writing.

Image. A page's ink is timed, picked out of each box and drawn by recog_synth/recog_pages.py, the code the
recognition samples are made with (straight from the synthesized pages, with the ground truth's boxes and no
marks): each line's recorded times move as one section into the order the page is written (write_order), each row's
ink is drawn ROW_HEIGHT tall with the fixed pen recog_pages.PEN (2 px), shrunk as a whole to CANVAS wide when longer, left-aligned in its slot
of the ten-row layout from the top, coloured by time and pen movement; slots past the last row stay white. A mark's
points, which have no recorded times, are one section at a single time after the row's writing (row_lines). The
image is given to the model as the segmentation images are, background 0 and ink 1 (each channel 1 - value / 255),
asked recog_layout.PROMPTS[task] and generated greedily.

Target and loss. Each box takes the transcription (a formula: its LaTeX) of the ground-truth object of its
class it overlaps best at IoU >= iou, one to one, best first; a box matching none is an empty row. The rows
are joined by recog_layout.SEP (a newline, one pretrained token), as recognition targets are. recog_loss/<task> is the teacher-forced token cross-entropy
of those targets, the training loss, summed over the task's target tokens and divided by their count (with
the rows, matched rows and images behind it). Ground-truth lines no box matched are not read: the loss is
of what the cascade hands recognition.

Text. Each image is also generated greedily (not teacher-forced) and its output split into rows (read_rows).
recog_cer/<task> is the character error rate of that text: the edit distance of every row read against its
target, over the characters of the targets. A box matching no ground-truth line has an empty target, so all it
reads is error, as is a row read past the group's last box; and every ground-truth line no box matched counts as
read as nothing, all its characters wrong (recog_missed/<task> counts them), so a line the cascade misses costs
recognition too. A formula is compared with its whitespace taken out, a textline with each run of it made one
space. A ground-truth line whose transcription can't go into a target (clean_label gives None) isn't scored.
recog_acc/<task> is the share of characters read correctly, over the same rows and missed lines: a row's correct
characters are the longest common subsequence of what was read and its target (the target's characters read, in
order), its total the longer of the two, so characters read that aren't there cost as much as characters left out.
A box matching no line has no correct characters and a missed line none either, all its characters in the total.

A page JSON whose textlines and formulas carry no "ink" (written before page_export kept it) is skipped: its
page gets None and adds nothing to the counts. Enclosures and line marks without "ink" (pages written before
page_export kept theirs) add nothing to a box's ink.
"""

from __future__ import annotations
import math
import os
import sys

import numpy as np
from PIL import Image

try:
    import recog_layout as rl
except ImportError:         # training/ run in place: recog_synth/ beside it (the GPU VM has it in code/)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recog_synth"))
    import recog_layout as rl

from seg_dataset import IGNORE_INDEX, Collator
from seg_eval import EOS, PAD, START, box_iou, encoder_inputs, generate

# how a page's ink is timed, picked out of a box and drawn: recognition samples are made by the same code
from recog_pages import MIN_INSIDE, PageInk, group_image, row_lines, write_order  # noqa: E402,F401

TASKS = rl.TEXT_TASKS                       # textline, formula: the cascade's class names too
CROP_PAD = 10.0                            # page pixels round a box's crop for the mark separation network
KEEP_PROB = 0.5                             # the writing channel's probability a point needs to be kept


def to_ink(image):
    """An RGB PIL image as the model is given it: background 0, ink 1, each channel 1 - value / 255."""
    return 1.0 - np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def match_rows(boxes, truths, iou, min_box_px=0.0):
    """Each box's index in `truths` (dicts with a bbox) or None: one to one, best IoU first, at IoU >= iou."""
    pairs = sorted(((box_iou(box, truth["bbox"], min_box_px), i, j) for i, box in enumerate(boxes)
                    for j, truth in enumerate(truths)), reverse=True)
    found, used = [None] * len(boxes), set()
    for score, i, j in pairs:
        if score < iou - 1e-9:
            break
        if found[i] is None and j not in used:
            found[i] = j
            used.add(j)
    return found


def read_rows(text):
    """A generated output's rows: what lies between the decoder's start and end, split on recog_layout.SEP. Every
    "<s>" goes too: the model can repeat the start's, and no transcription holds one (recog_layout.clean_label)."""
    if text.startswith(START):
        text = text[len(START):]
    end = text.find(EOS)
    if end >= 0:
        text = text[:end]
    return [" ".join(row.replace(PAD, "").replace("<s>", "").split()) for row in text.split(rl.SEP)]


def text_key(task, text):
    """Text as its characters are compared: a formula's with its whitespace taken out, a textline's with each run
    of it made one space."""
    return "".join(text.split()) if task == "formula" else " ".join(text.split())


def edit_distance(a, b):
    """The Levenshtein distance of strings a and b: one per character inserted, deleted or substituted. Bit-parallel
    (Myers 1999, in Hyyrö's form for whole strings) over Python ints, so it takes one step per character of the
    shorter string whatever their lengths."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    peq = {}
    for i, ch in enumerate(a):
        peq[ch] = peq.get(ch, 0) | (1 << i)
    mask, last = (1 << len(a)) - 1, 1 << (len(a) - 1)
    pv, mv, score = mask, 0, len(a)
    for ch in b:
        eq = peq.get(ch, 0)
        xv = eq | mv
        xh = ((((eq & pv) + pv) & mask) ^ pv) | eq
        ph = (mv | ~(xh | pv)) & mask
        mh = pv & xh
        if ph & last:
            score += 1
        elif mh & last:
            score -= 1
        ph = ((ph << 1) | 1) & mask
        mh = (mh << 1) & mask
        pv = (mh | ~(xv | ph)) & mask
        mv = ph & xv
    return score


def common_length(a, b):
    """The length of the longest common subsequence of strings a and b. Bit-parallel (Allison and Dix 1986, in
    Hyyrö's form) over Python ints, one step per character of the shorter string."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return 0
    peq = {}
    for i, ch in enumerate(a):
        peq[ch] = peq.get(ch, 0) | (1 << i)
    mask = (1 << len(a)) - 1
    v = mask
    for ch in b:
        u = v & peq.get(ch, 0)
        v = ((v + u) | (v - u)) & mask
    return len(a) - bin(v).count("1")


def score_rows(task, read, targets, scored):
    """(edits, chars, correct, totals): per row, the edit distance of what was read against its target, the target's
    characters, the characters read correctly (their longest common subsequence) and the longer of the two lengths,
    all None for a row not `scored`. Rows run to the longer of `read` and `targets`: a row read past the last target
    is scored against an empty one, a target with nothing read for it against empty text."""
    edits, chars, correct, totals = [], [], [], []
    for k in range(max(len(read), len(targets))):
        if k < len(targets) and not scored[k]:
            for values in (edits, chars, correct, totals):
                values.append(None)
            continue
        target = text_key(task, targets[k]) if k < len(targets) else ""
        said = text_key(task, read[k]) if k < len(read) else ""
        edits.append(edit_distance(said, target))
        chars.append(len(target))
        correct.append(common_length(said, target))
        totals.append(max(len(said), len(target)))
    return edits, chars, correct, totals


def model_read_fn(base, processor, device, dtype, max_new_tokens, batch_size):
    """read_fn for Recognizer: requests {image, prompt, target} -> (texts, truncated, loss sums, target tokens),
    batch_size at a time. A loss sum is the request's token cross-entropy summed over its target tokens."""
    import torch
    import torch.nn.functional as F

    collator = Collator()
    collator.processor = processor

    @torch.no_grad()
    def run(requests):
        texts, truncated, sums, counts = [], [], [], []
        for i in range(0, len(requests), batch_size):
            chunk = requests[i:i + batch_size]
            batch = collator([(to_ink(r["image"]), r["prompt"], r["target"], "") for r in chunk])
            t, c = generate(base, batch, processor.tokenizer, device, dtype, max_new_tokens)
            texts += t
            truncated += c
            embeds, mask = encoder_inputs(base, batch, device, dtype)
            labels = batch["labels"].to(device)
            with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
                logits = base.language_model(inputs_embeds=embeds, attention_mask=mask, labels=labels,
                                             use_cache=False).logits
            for row in range(len(chunk)):
                keep = labels[row] != IGNORE_INDEX
                sums.append(F.cross_entropy(logits[row][keep].float(), labels[row][keep], reduction="sum").item())
                counts.append(int(keep.sum()))
        return texts, truncated, sums, counts
    return run


class MarkRemover:
    """remove_marks for Recognizer: the mark separation network (training/marks_cnn) over detected boxes.
    Called with requests [{image: the page, uint8 greyscale H x W; box: page pixels; strokes: (n, 3) arrays}],
    returns each request's strokes with the points the network finds no writing under taken out."""

    def __init__(self, path, device=None, batch_size=64):
        import torch
        try:
            import marks_model
            from marks_data import HEIGHT, WIDTH, fit
        except ImportError:     # training/ run in place: marks_cnn/ inside it (the GPU VM has both in code/)
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "marks_cnn"))
            import marks_model
            from marks_data import HEIGHT, WIDTH, fit
        self.torch, self.fit, self.size, self.batch_size = torch, fit, (HEIGHT, WIDTH), batch_size
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model = marks_model.from_checkpoint(checkpoint).to(self.device).eval()
        self.path, self.step = path, (checkpoint.get("state") or {}).get("step")

    def crop(self, image, box):
        """(the network's uint8 input, (x0, y0, scale, x offset, y offset, width, height)) for a box's padded crop
        of the page, fitted as marks_data.load_row fits a crop; None for a crop with no area."""
        height, width = image.shape
        x0, y0 = max(0, math.floor(box[0] - CROP_PAD)), max(0, math.floor(box[1] - CROP_PAD))
        x1, y1 = min(width, math.ceil(box[2] + CROP_PAD)), min(height, math.ceil(box[3] + CROP_PAD))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        scale, w, h, ox, oy = self.fit(x1 - x0, y1 - y0)
        out = np.full(self.size, 255, dtype=np.uint8)
        out[oy:oy + h, ox:ox + w] = np.asarray(Image.fromarray(np.ascontiguousarray(image[y0:y1, x0:x1])).resize((w, h), Image.BILINEAR))
        return out, (x0, y0, scale, ox, oy, w, h)

    def __call__(self, requests):
        torch = self.torch
        out = [r["strokes"] for r in requests]
        crops = [(i, self.crop(r["image"], r["box"])) for i, r in enumerate(requests) if r["strokes"]]
        crops = [(i, c) for i, c in crops if c is not None]
        for start in range(0, len(crops), self.batch_size):
            chunk = crops[start:start + self.batch_size]
            batch = torch.from_numpy(np.stack([c[0] for _, c in chunk])[:, None]).to(self.device)
            with torch.no_grad():
                writing = torch.sigmoid(self.model(1.0 - batch.float() / 255.0)[0].float())    # the labels aren't needed
                # the most the writing channel says within a pixel of each point, so a point on a stroke's edge counts
                writing = torch.nn.functional.max_pool2d(writing, 3, stride=1, padding=1)[:, 0].cpu().numpy()
            for (i, (_, place)), probs in zip(chunk, writing):
                out[i] = keep_writing(requests[i]["strokes"], probs, place)
        return out


def keep_writing(strokes, probs, place):
    """`strokes` cut down to their points over KEEP_PROB in `probs` (the writing channel, HEIGHT x WIDTH), placed as
    MarkRemover.crop placed them; points outside the fitted crop stay."""
    x0, y0, scale, ox, oy, w, h = place
    kept = []
    for s in strokes:
        cx = np.rint((s[:, 0] - x0) * scale + ox).astype(np.int64)
        cy = np.rint((s[:, 1] - y0) * scale + oy).astype(np.int64)
        judged = (cx >= ox) & (cx < ox + w) & (cy >= oy) & (cy < oy + h)
        keep = ~judged
        keep[judged] = probs[cy[judged], cx[judged]] > KEEP_PROB
        edges = np.flatnonzero(np.diff(np.concatenate([[0], keep.astype(np.int8), [0]])))
        kept += [s[a:b] for a, b in zip(edges[::2], edges[1::2])]
    return kept


class Recognizer:
    """recognize(pages, preds, metrics) for seg_eval.validate_pages. Reads every page's detected textlines and
    formulas, adds each image to metrics (SegMetrics.recognized, and its generated text's errors with
    SegMetrics.transcribed) and returns, per page, {"groups": [{task, boxes, targets, matched, predictions, edits,
    chars, correct, totals, loss, tokens, truncated}], "missed": [{task, bbox, text, chars}]} (a group with
    removed_points, the ink points remove_marks took out, when it is given, and the PIL "image" with keep_images;
    edits, chars, correct and totals as score_rows gives them; missed: the ground-truth lines no box matched), or None
    for a page whose JSON has no ink.
    remove_marks (MarkRemover) takes marks out of each box's ink before it is drawn."""

    def __init__(self, read_fn, iou=0.5, min_box_px=0.0, keep_images=False, remove_marks=None):
        self.read_fn, self.iou, self.min_box_px, self.keep_images = read_fn, iou, min_box_px, keep_images
        self.remove_marks = remove_marks

    def __call__(self, pages, preds, metrics):
        out, requests = [], []
        for p, page in enumerate(pages):
            text = page.get("text")
            out.append(None if text is None else {"groups": [], "missed": []})
            if text is None:
                continue
            ink = PageInk(write_order(text))
            for task in TASKS:
                boxes = [coords for name, coords in preds[p] if name == task]
                truths = [obj for obj in text if obj["class"] == task]
                labels = [rl.clean_label(obj["text"]) for obj in truths]
                found = match_rows(boxes, truths, self.iou, self.min_box_px)
                # a line no box matched is read as nothing: every character of it an error
                missed = [{"task": task, "bbox": [round(v, 1) for v in truths[j]["bbox"]], "text": labels[j],
                           "chars": len(text_key(task, labels[j]))}
                          for j in range(len(truths)) if j not in found and labels[j] is not None]
                if missed:
                    chars = sum(line["chars"] for line in missed)
                    metrics.transcribed(task, chars, chars, len(missed), total=chars)
                    out[p]["missed"] += missed
                for first in range(0, len(boxes), rl.ROWS):
                    group = range(first, min(first + rl.ROWS, len(boxes)))
                    targets = [(labels[found[k]] or "") if found[k] is not None else "" for k in group]
                    requests.append({"page": p, "task": task, "prompt": rl.PROMPTS[task],
                                     "target": rl.SEP.join(targets), "targets": targets,
                                     "rows": [ink.inside(boxes[k]) for k in group], "pixels": [boxes[k] for k in group],
                                     "boxes": [[round(v, 1) for v in boxes[k]] for k in group],
                                     "matched": [found[k] is not None for k in group],
                                     "scored": [found[k] is None or labels[found[k]] is not None for k in group]})
        if requests and self.remove_marks is not None:
            flat = [{"image": pages[r["page"]]["image"], "box": box, "strokes": strokes}
                    for r in requests for box, strokes in zip(r["pixels"], r["rows"])]
            cleaned = iter(self.remove_marks(flat))
            for request in requests:
                before = sum(len(s) for row in request["rows"] for s in row)
                request["rows"] = [next(cleaned) for _ in request["rows"]]
                request["removed_points"] = before - sum(len(s) for row in request["rows"] for s in row)
        for request in requests:
            request["image"] = group_image(request["rows"])
        if requests:
            for request, text, cut, loss, tokens in zip(requests, *self.read_fn(requests)):
                metrics.recognized(request["task"], loss, tokens, len(request["boxes"]), sum(request["matched"]))
                read = read_rows(text)
                edits, chars, correct, totals = score_rows(request["task"], read, request["targets"], request["scored"])
                known = lambda values: sum(v for v in values if v is not None)
                metrics.transcribed(request["task"], known(edits), known(chars), correct=known(correct),
                                    total=known(totals))
                group = {k: request[k] for k in ("task", "boxes", "targets", "matched", "removed_points") if k in request}
                group.update(predictions=read, edits=edits, chars=chars, correct=correct, totals=totals,
                             loss=loss / tokens if tokens else None, tokens=tokens, truncated=cut)
                if self.keep_images:
                    group["image"] = request["image"]
                out[request["page"]]["groups"].append(group)
        return out
