"""
marks_eval.py — scoring the mark separation network: validation while training, and the one test run.

    python marks_eval.py --checkpoint run/best --data prepared --split test --out test_metrics.json

Writing. A target pixel is ink when it is darker than mid-grey (ink > 0.5); a predicted pixel
is ink when its probability is over 0.5. Pooled over every crop: precision, recall, F1 (the
Dice of the ink masks) and IoU, and F1 / IoU again over the crops holding each kind of mark.
Also
  mark_in_text         of the marks' own ink (in the input, not in the writing), the share predicted as writing
  baseline_writing_f1  the writing's F1 if the input were copied out unchanged
Labels. A kind of mark is predicted present when its probability is over 0.5. Per kind,
precision, recall and F1 over every crop; labels_mean_f1 is their mean and labels_exact the
share of crops with all four right. mean_f1, the mean of the writing's F1 and labels_mean_f1,
is what chooses best/.

The test run writes aggregate numbers only: no crops, predictions or per-crop scores.
"""

from __future__ import annotations
import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from marks_data import MARK_TYPES
from marks_model import from_checkpoint, mark_loss, split_batch

# columns of crop_stats
TP, FP, FN, B_TP, B_FP, B_FN, MK_N, MK_D = range(8)


@torch.no_grad()
def crop_stats(writing_logits, ink_in, writing):
    """int64 (B, 8) pixel counts per crop, columns as named above."""
    pred, truth, base = writing_logits[:, 0] > 0, writing[:, 0] > 0.5, ink_in[:, 0] > 0.5
    mark_only = base & ~truth
    count = lambda mask: mask.sum((1, 2))
    return torch.stack([count(pred & truth), count(pred & ~truth), count(~pred & truth),
                        count(base & truth), count(base & ~truth), count(~base & truth),
                        count(pred & mark_only), count(mark_only)], 1).long()


def _ratio(a, b):
    return float(a) / float(b) if b else 0.0


def channel_scores(tp, fp, fn):
    return {"precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn), "f1": _ratio(2 * tp, 2 * tp + fp + fn),
            "iou": _ratio(tp, tp + fp + fn), "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def crop_scores(stats, label_prob, labels):
    """One crop's writing F1 and IoU, share of mark ink kept as writing, and its labels against the predicted ones."""
    s = [int(v) for v in stats]
    writing = channel_scores(s[TP], s[FP], s[FN])
    return {"f1": round(writing["f1"], 4), "iou": round(writing["iou"], 4), "mark_in_text": round(_ratio(s[MK_N], s[MK_D]), 4),
            "labels": [int(v) for v in labels], "label_probs": [round(float(p), 4) for p in label_prob]}


class Tally:
    """Counts summed over crops: the writing's over all of them and over the crops holding each kind of mark,
    and the labels' per kind."""

    def __init__(self):
        self.groups = {}
        self.labels = np.zeros((len(MARK_TYPES), 3), dtype=np.int64)       # tp, fp, fn per kind
        self.exact, self.loss_sum, self.label_loss_sum, self.units = 0, 0.0, 0.0, 0

    def add(self, stats, predicted, labels, loss_sum, label_loss_sum):
        """stats (B, 8), predicted and labels bool (B, kinds)."""
        self.loss_sum += loss_sum
        self.label_loss_sum += label_loss_sum
        self.units += len(stats)
        self.labels[:, 0] += (predicted & labels).sum(0)
        self.labels[:, 1] += (predicted & ~labels).sum(0)
        self.labels[:, 2] += (~predicted & labels).sum(0)
        self.exact += int((predicted == labels).all(1).sum())
        for row, present in zip(stats, labels):
            for key in ["all", *(MARK_TYPES[k] for k in np.flatnonzero(present))]:
                entry = self.groups.setdefault(key, {"crops": 0, "counts": np.zeros(8, dtype=np.int64)})
                entry["crops"] += 1
                entry["counts"] += row

    def summary(self):
        if not self.units:
            return {"units": 0, "loss": None, "label_loss": None, "mean_f1": None, "writing": {}, "labels": {},
                    "labels_mean_f1": None, "labels_exact": None, "by_type": {}}
        c = self.groups["all"]["counts"]
        writing = channel_scores(c[TP], c[FP], c[FN])
        labels = {}
        for k, kind in enumerate(MARK_TYPES):
            tp, fp, fn = self.labels[k]
            labels[kind] = {**{key: v for key, v in channel_scores(tp, fp, fn).items() if key != "iou"}, "support": int(tp + fn)}
        labels_mean_f1 = sum(v["f1"] for v in labels.values()) / len(labels)
        by_type = {}
        for kind in MARK_TYPES:
            if kind in self.groups:
                g = self.groups[kind]["counts"]
                scores = channel_scores(g[TP], g[FP], g[FN])
                by_type[kind] = {"crops": self.groups[kind]["crops"], "writing": {"f1": scores["f1"], "iou": scores["iou"]},
                                 "mark_in_text": _ratio(g[MK_N], g[MK_D])}
        return {"units": self.units, "loss": self.loss_sum / self.units, "label_loss": self.label_loss_sum / self.units,
                "mean_f1": (writing["f1"] + labels_mean_f1) / 2, "writing": writing,
                "mark_in_text": _ratio(c[MK_N], c[MK_D]), "baseline_writing_f1": channel_scores(c[B_TP], c[B_FP], c[B_FN])["f1"],
                "labels": labels, "labels_mean_f1": labels_mean_f1, "labels_exact": self.exact / self.units, "by_type": by_type}


def flat_record(summary, prefix):
    """The summary as flat log.jsonl keys."""
    rec = {f"{prefix}{k}": summary.get(k) for k in ("loss", "label_loss", "mean_f1", "mark_in_text", "baseline_writing_f1")}
    rec[f"{prefix}crops"] = summary.get("units")
    rec[f"{prefix}labels/mean_f1"] = summary.get("labels_mean_f1")
    rec[f"{prefix}labels/exact"] = summary.get("labels_exact")
    for k in ("precision", "recall", "f1", "iou"):
        if k in (summary.get("writing") or {}):
            rec[f"{prefix}{k}/writing"] = summary["writing"][k]
    for kind, v in (summary.get("labels") or {}).items():
        for k in ("precision", "recall", "f1", "support"):
            rec[f"{prefix}{k}/label/{kind}"] = v[k]
    for kind, g in (summary.get("by_type") or {}).items():
        rec[f"{prefix}crops/{kind}"] = g["crops"]
        rec[f"{prefix}f1/writing/{kind}"] = g["writing"]["f1"]
        rec[f"{prefix}iou/writing/{kind}"] = g["writing"]["iou"]
        rec[f"{prefix}mark_in_text/{kind}"] = g["mark_in_text"]
    return rec


def format_report(summary):
    if not summary.get("units"):
        return "  (no crops)"
    w = summary["writing"]
    lines = [f"  {'':14s} {'precision':>9s} {'recall':>7s} {'F1':>7s} {'IoU':>7s}",
             f"  {'writing':14s} {w['precision']:9.3f} {w['recall']:7.3f} {w['f1']:7.3f} {w['iou']:7.3f}",
             f"  {'label':14s} {'precision':>9s} {'recall':>7s} {'F1':>7s} {'crops':>7s}"]
    for kind, v in summary["labels"].items():
        lines.append(f"  {kind:14s} {v['precision']:9.3f} {v['recall']:7.3f} {v['f1']:7.3f} {v['support']:7d}")
    lines.append(f"  {'crops with':14s} {'crops':>9s} {'writing F1':>10s} {'mark ink kept':>13s}")
    for kind, g in summary["by_type"].items():
        lines.append(f"  {kind:14s} {g['crops']:9d} {g['writing']['f1']:10.3f} {g['mark_in_text']:13.3f}")
    lines.append(f"  loss {summary['loss']:.4f} (labels {summary['label_loss']:.4f})  mean F1 {summary['mean_f1']:.4f}  "
                 f"label F1 {summary['labels_mean_f1']:.3f}  all labels right {summary['labels_exact']:.3f}  "
                 f"mark ink kept as writing {summary['mark_in_text']:.3f}  "
                 f"(input copied as writing: F1 {summary['baseline_writing_f1']:.3f})")
    return "\n".join(lines)


def sample_image(crop, prob):
    """Three 600 x 100 strips stacked: input, writing, predicted writing."""
    crop = crop.cpu().numpy()
    pred = (255.0 * (1.0 - prob.float().clamp(0, 1))).round().to(torch.uint8).cpu().numpy()
    return Image.fromarray(np.ascontiguousarray(np.concatenate([crop[0], crop[1], pred[0]], 0)))


@torch.no_grad()
def evaluate(model, images, labels, indices, batch_size, device, dtype, progress=None, keep=(), on_sample=None):
    """The summary over images[indices] (uint8 (N, 2, H, W)) and labels[indices] (float (N, kinds)), on any device.
    For every index in `keep`, on_sample(index, crop, writing probabilities, label probabilities, crop_stats row) as
    it is scored."""
    was_training = model.training
    model.eval()
    tally, keep = Tally(), set(int(i) for i in keep)
    indices = [int(i) for i in indices]
    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]
        at = torch.as_tensor(idx, device=images.device)
        batch = images[at].to(device, non_blocking=True)
        truth = labels[at.to(labels.device)].to(device)
        x, target = split_batch(batch)
        if device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            logits, label_logits = model(x)
        logits, label_logits = logits.float(), label_logits.float()
        loss, parts = mark_loss(logits, target, label_logits, truth)
        stats = crop_stats(logits, x, target).cpu().numpy()
        label_prob = torch.sigmoid(label_logits)
        tally.add(stats, (label_prob > 0.5).cpu().numpy(), (truth > 0.5).cpu().numpy(),
                  float(loss) * len(idx), float(parts["label_bce"]) * len(idx))
        if on_sample is not None and keep:
            for j, i in enumerate(idx):
                if i in keep:
                    on_sample(i, batch[j], torch.sigmoid(logits[j]), label_prob[j], stats[j])
        if progress is not None:
            progress(min(len(indices), start + len(idx)))
    if was_training:
        model.train()
    return tally.summary()


def load_checkpoint_model(path, device):
    ckpt = torch.load(os.path.join(path, "model.pt"), map_location=device, weights_only=False)
    return from_checkpoint(ckpt).to(device), ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="a checkpoint folder holding model.pt")
    ap.add_argument("--data", required=True, help="marks_data.py's output")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True, help="metrics JSON")
    ap.add_argument("--progress", default=None, help="progress JSON, rewritten as crops are scored")
    ap.add_argument("--batch-size", type=int, default=64, help="crops per forward pass (23 MB each at dec1)")
    args = ap.parse_args()

    from marks_data import label_array, load_split

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model, ckpt = load_checkpoint_model(args.checkpoint, device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    array, metas = load_split(args.data, args.split, mmap=False)
    images, labels = torch.from_numpy(array), torch.from_numpy(label_array(metas))
    total = len(metas)
    last = {"t": 0.0}

    def progress(done):
        if args.progress and (done == total or time.time() - last["t"] > 2):
            last["t"] = time.time()
            tmp = args.progress + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"done": done, "total": total, "unit": "crops", "updated_at": time.time()}, f)
            os.replace(tmp, args.progress)

    progress(0)
    started = time.time()
    summary = evaluate(model, images, labels, range(total), args.batch_size, device, dtype, progress=progress)
    doc = {"checkpoint": os.path.abspath(args.checkpoint), "split": args.split,
           "step": (ckpt.get("state") or {}).get("step"), "seconds": round(time.time() - started, 1), **summary}
    with open(args.out + ".tmp", "w") as f:
        json.dump(doc, f, indent=1)
    os.replace(args.out + ".tmp", args.out)
    print(f"{args.split}: {total:,} crops, checkpoint {args.checkpoint} (step {doc['step']})")
    print(format_report(summary))


if __name__ == "__main__":
    main()
