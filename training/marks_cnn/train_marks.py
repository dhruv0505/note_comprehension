"""
train_marks.py — the mark separation network (marks_model.py) on marks_data.py's arrays, one GPU.

    python marks_data.py --root data --out prepared --splits train val test       # once
    python train_marks.py --data prepared --out runs/marks --epochs 40
    python train_marks.py --data prepared --out runs/marks --resume runs/marks/last
    python train_marks.py --data prepared --out smoke --max-steps 6 --val-max 64 --allow-cpu

The train and val arrays are loaded whole onto the GPU (120 KB of uint8 a crop: about
6.7 GB for 56k train crops and 0.6 GB for 4.6k val), so a batch is an index into them
and there is no data loader. Each step: the
crops as ink, the network's writing against the writing-alone crop with BCE + Dice and
its four labels against the crop's (underline, strikethrough, oval, rectangle) with BCE,
weighted by --label-weight (marks_model.mark_loss), AdamW, gradient clipping, and a
linear warmup into cosine decay.

Validation (marks_eval.py) runs every --val-every steps (0 = at the end of every
epoch) and after the last step, on the first --val-max crops of a fixed shuffle of
the val split (0 = all of them). best/ is the checkpoint with the highest mean F1:
the mean of the writing's F1 and the labels' mean F1. The test split is never read here.

Control while running, as train.py: control.json {"seq": 3, "lr": 5e-4} sets the
schedule's peak learning rate from the next step (control_ack.json and a log line say
so); STOP saves last/ and exits.

Output (--out), where the tracker reads train.py's:
  config.json, log.jsonl, progress.json
  step_000500/, last/, best/          model.pt: network, optimizer, schedule, position in the epoch
  samples_step_000500_marks.jsonl     the first --samples val crops of the shuffle: name, shapes,
                                      labels and predicted label probabilities, per-crop writing
                                      F1 / IoU, and samples/step_000500/<name>.png, three
                                      600 x 100 strips: input, writing, predicted writing
"""

from __future__ import annotations
import argparse
import json
import math
import os
import re
import shutil
import time
import traceback

import numpy as np
import torch

from marks_data import HEIGHT, MARK_TYPES, WIDTH, label_array, load_split
from marks_eval import crop_scores, evaluate, flat_record, format_report, sample_image
from marks_model import build_model, count_parameters, mark_loss, split_batch

CONTROL_EVERY = 5.0         # seconds between reads of the control files
MODEL_CONFIG = {"width": 32, "labels": len(MARK_TYPES)}


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


class Progress:
    """progress.json, written at most every `every` seconds unless forced."""

    def __init__(self, path, every=5.0, **fields):
        self.path, self.every, self.last = path, every, 0.0
        self.doc = {"started_at": time.time(), "pid": os.getpid(), **fields}

    def update(self, force=False, **fields):
        self.doc.update(fields)
        now = time.time()
        if force or now - self.last >= self.every:
            self.doc["updated_at"] = now
            write_json(self.path, self.doc)
            self.last = now


def log(args, record):
    with open(os.path.join(args.out, "log.jsonl"), "a") as f:
        f.write(json.dumps({"time": time.time(), **record}) + "\n")


def schedule(warmup, total):
    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))
    return factor


def set_peak_lr(optimizer, scheduler, peak):
    """The schedule's peak learning rate to `peak` from the next step; returns (old peak, rate now)."""
    old = scheduler.base_lrs[0]
    factor = peak / old
    scheduler.base_lrs = [lr * factor for lr in scheduler.base_lrs]
    for group in optimizer.param_groups:
        group["lr"] *= factor
        if "initial_lr" in group:
            group["initial_lr"] *= factor
    scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
    return old, optimizer.param_groups[0]["lr"]


def save_checkpoint(model, optimizer, scheduler, state, args, name):
    path = os.path.join(args.out, name)
    tmp = path + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    torch.save({"model": model.state_dict(), "model_config": MODEL_CONFIG, "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "state": dict(state), "input_size": [HEIGHT, WIDTH],
                "mark_types": list(MARK_TYPES)},
               os.path.join(tmp, "model.pt"))
    shutil.rmtree(path, ignore_errors=True)
    os.replace(tmp, path)
    return path


def prune_checkpoints(out, keep):
    steps = sorted(d for d in os.listdir(out) if re.fullmatch(r"step_\d+", d))
    for name in steps[:-keep] if keep > 0 else []:
        shutil.rmtree(os.path.join(out, name), ignore_errors=True)


class Control:
    def __init__(self, out):
        self.control, self.stop, self.last = os.path.join(out, "control.json"), os.path.join(out, "STOP"), 0.0

    def poll(self, applied_seq):
        """(stop requested, (seq, peak lr) of a new rate request or None)."""
        if time.time() - self.last < CONTROL_EVERY:
            return False, None
        self.last = time.time()
        request = None
        doc = read_json(self.control)
        try:
            seq, lr = int(doc["seq"]), float(doc["lr"])
            if seq > applied_seq and math.isfinite(lr) and lr > 0:
                request = (seq, lr)
        except (TypeError, KeyError, ValueError):
            pass
        return os.path.exists(self.stop), request


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="marks_data.py's output folder (train.npy, val.npy, ...)")
    ap.add_argument("--out", default=os.path.join("runs", "marks"))
    ap.add_argument("--resume", default=None, help="checkpoint folder, e.g. runs/marks/last")
    ap.add_argument("--epochs", type=float, default=40.0)
    ap.add_argument("--max-steps", type=int, default=None, help="optimizer steps; overrides --epochs")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--label-weight", type=float, default=1.0, help="weight of the labels' BCE against the writing's loss")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--val-every", type=int, default=500, help="validate every N steps (0 = at epoch ends)")
    ap.add_argument("--val-max", type=int, default=0, help="val crops per validation (0 = all)")
    # a 600 x 100 crop's widest activation (dec1's input, float32) is 23 MB: 256 at once ran a 24 GB L4 out of
    # memory beside the arrays held on the GPU, while 64 is what training already fits
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--samples", type=int, default=16, help="val crops whose outputs are saved each validation")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--keep", type=int, default=2, help="step_* checkpoints kept (last/ and best/ always are)")
    ap.add_argument("--allow-cpu", action="store_true", help="run without a GPU (a smoke test only)")
    return ap.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("train_marks.py needs a CUDA GPU; --allow-cpu runs a smoke test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    progress = Progress(os.path.join(args.out, "progress.json"), phase="loading", world=1)
    try:
        run(args, device, progress)
    except BaseException as error:
        progress.update(force=True, phase="stopped" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()[-4000:])
        raise


def run(args, device, progress):
    cuda = device.type == "cuda"
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({**vars(args), "world_size": 1, "input_size": [HEIGHT, WIDTH], "model": MODEL_CONFIG}, f, indent=2)
    progress.update(force=True)
    torch.manual_seed(args.seed)
    if cuda:
        torch.backends.cudnn.benchmark = True

    train_array, train_meta = load_split(args.data, "train", mmap=False)
    val_array, val_meta = load_split(args.data, "val", mmap=False)
    train_images, train_labels = torch.from_numpy(train_array).to(device), torch.from_numpy(label_array(train_meta)).to(device)
    val_images, val_labels = torch.from_numpy(val_array).to(device), torch.from_numpy(label_array(val_meta)).to(device)
    n_train = len(train_images)
    if not n_train:
        raise SystemExit("no training crops")
    val_order = np.random.default_rng(args.seed).permutation(len(val_meta))
    val_indices = (val_order[:args.val_max] if args.val_max else val_order).tolist()
    keep = val_indices[:args.samples]

    model = build_model(**MODEL_CONFIG).to(device)
    if cuda:
        model = model.to(memory_format=torch.channels_last)
    dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float32
    steps_per_epoch = math.ceil(n_train / args.batch_size)
    total_steps = args.max_steps or max(1, round(args.epochs * steps_per_epoch))
    print(f"train {n_train:,} crops ({steps_per_epoch:,} steps/epoch), val {len(val_indices):,} of {len(val_meta):,} "
          f"crops; {total_steps:,} steps at batch {args.batch_size}; {count_parameters(model):,} parameters; "
          f"{HEIGHT} x {WIDTH}; {dtype}; data on {device}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=cuda)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule(min(args.warmup_steps, total_steps // 10), total_steps))
    state = {"step": 0, "epoch": 0, "epoch_samples": 0, "best_mean_f1": -1.0, "best_step": None,
             "control_seq": 0, "last_val_step": None}
    if args.resume:
        ckpt = torch.load(os.path.join(args.resume, "model.pt"), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        state.update(ckpt["state"])
        print(f"resumed {args.resume}: step {state['step']}, epoch {state['epoch']} at crop {state['epoch_samples']:,}")
    model.train()

    val_seconds = {"last": None}

    def remaining_validations():
        left = total_steps - state["step"]
        count = left // args.val_every if args.val_every else math.ceil(left / steps_per_epoch)
        return max(1, count)

    def report(**fields):
        step_s = progress.doc.get("step_seconds")
        eta = None
        if step_s:
            eta = step_s * (total_steps - state["step"]) + (val_seconds["last"] or 0) * remaining_validations()
        progress.update(step=state["step"], total_steps=total_steps,
                        epoch=state["epoch"] + state["epoch_samples"] / n_train,
                        epochs=total_steps / steps_per_epoch if args.max_steps else args.epochs,
                        steps_per_epoch=steps_per_epoch, lr=optimizer.param_groups[0]["lr"],
                        peak_lr=scheduler.base_lrs[0], control_seq=state["control_seq"],
                        best_mean_f1=state["best_mean_f1"], best_step=state["best_step"],
                        last_val_step=state["last_val_step"], eta_s=eta, **fields)

    progress.update(force=True, world=1, batch_size=args.batch_size, grad_accum=1, global_batch=args.batch_size,
                    train_rows=n_train, val_rows=len(val_indices), val_every=args.val_every, save_every=args.save_every)
    log(args, {"event": "start", "step": state["step"], "total_steps": total_steps, "world": 1,
               "global_batch": args.batch_size, "steps_per_epoch": steps_per_epoch, "peak_lr": scheduler.base_lrs[0],
               "train_crops": n_train, "val_crops": len(val_indices), "parameters": count_parameters(model),
               "resumed_from": args.resume})

    def checkpoint(name):
        report(phase="checkpointing")
        progress.update(force=True)
        path = save_checkpoint(model, optimizer, scheduler, state, args, name)
        print(f"  saved {path}", flush=True)
        progress.update(last_checkpoint={"name": name, "step": state["step"], "at": time.time()})

    def run_validation():
        t = time.perf_counter()
        step, total = state["step"], len(val_indices)
        tag = f"step_{step:06d}"
        sample_dir = os.path.join(args.out, "samples", tag)
        os.makedirs(sample_dir, exist_ok=True)
        stream = os.path.join(args.out, f"samples_{tag}_marks.jsonl")
        open(stream, "w").close()

        def on_sample(i, crop, prob, label_prob, stats):
            meta = val_meta[i]
            path = os.path.join(sample_dir, f"{meta['name']}.png")
            sample_image(crop, prob).save(path)
            row = {"image": os.path.abspath(path), "name": meta["name"], "index": i, "shapes": meta.get("shapes"),
                   "text_label": meta.get("text_label"), "content": meta.get("content"),
                   **crop_scores(stats, label_prob, meta["labels"])}
            with open(stream, "a") as f:
                f.write(json.dumps(row) + "\n")

        report(phase="validating", val={"kind": "crops", "done": 0, "total": total, "step": step})
        progress.update(force=True)
        summary = evaluate(model, val_images, val_labels, val_indices, args.eval_batch_size, device, dtype,
                           keep=keep, on_sample=on_sample,
                           progress=lambda n: progress.update(val={"kind": "crops", "done": n, "total": total, "step": step}))
        record = {"step": step, "epoch": state["epoch"] + state["epoch_samples"] / n_train, **flat_record(summary, "val/")}
        record["val_seconds"] = val_seconds["last"] = time.perf_counter() - t
        state["last_val_step"] = step
        print(f"validation at step {step} on {total:,} val crops:\n{format_report(summary)}\n"
              f"  validation took {record['val_seconds']:.1f}s", flush=True)
        log(args, record)
        report(phase="training", val=None)
        if summary["mean_f1"] is not None and summary["mean_f1"] > state["best_mean_f1"]:
            state["best_mean_f1"], state["best_step"] = summary["mean_f1"], step
            checkpoint("best")
            log(args, {"event": "best", "step": step, "mean_f1": summary["mean_f1"]})

    control = Control(args.out)

    def fresh_window():
        return {"loss": torch.zeros((), device=device), "bce": torch.zeros((), device=device),
                "dice": torch.zeros((), device=device), "label_bce": torch.zeros((), device=device),
                "steps": 0, "samples": 0, "t": time.perf_counter()}

    window, stopped = fresh_window(), False
    optimizer.zero_grad(set_to_none=True)
    report(phase="training")
    progress.update(force=True)
    try:
        while state["step"] < total_steps and not stopped:
            generator = torch.Generator().manual_seed(args.seed * 100_003 + state["epoch"])
            order = torch.randperm(n_train, generator=generator)[state["epoch_samples"]:].to(device)
            for start in range(0, len(order), args.batch_size):
                idx = order[start:start + args.batch_size]
                x, target = split_batch(train_images[idx])
                if cuda:
                    x = x.contiguous(memory_format=torch.channels_last)
                with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                    logits, label_logits = model(x)
                loss, parts = mark_loss(logits, target, label_logits, train_labels[idx], args.label_weight)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                state["step"] += 1
                state["epoch_samples"] += len(idx)
                window["loss"] += loss.detach()
                window["bce"] += parts["bce"]
                window["dice"] += parts["dice"]
                window["label_bce"] += parts["label_bce"]
                window["steps"] += 1
                window["samples"] += len(idx)
                step = state["step"]

                stop, request = control.poll(state["control_seq"])
                if request is not None:
                    seq, peak = request
                    old_peak, lr_now = set_peak_lr(optimizer, scheduler, peak)
                    state["control_seq"] = seq
                    ack = {"seq": seq, "step": step, "old_peak_lr": old_peak, "peak_lr": peak, "lr": lr_now,
                           "at": time.time(), "status": "applied"}
                    write_json(os.path.join(args.out, "control_ack.json"), ack)
                    print(f"step {step}: peak learning rate {old_peak:.2e} -> {peak:.2e} (now {lr_now:.2e})", flush=True)
                    log(args, {"event": "lr_change", **ack})

                if step % args.log_every == 0 or step == total_steps:
                    steps = window["steps"]
                    loss_v, bce_v, dice_v, label_v = (window[k].item() / steps for k in ("loss", "bce", "dice", "label_bce"))
                    elapsed = time.perf_counter() - window["t"]
                    record = {"step": step, "epoch": state["epoch"] + state["epoch_samples"] / n_train, "loss": loss_v,
                              "bce": bce_v, "dice": dice_v, "label_bce": label_v, "lr": scheduler.get_last_lr()[0],
                              "peak_lr": scheduler.base_lrs[0], "grad_norm": float(grad_norm),
                              "samples_per_s": window["samples"] / elapsed,
                              "max_mem_gb": torch.cuda.max_memory_allocated(device) / 2 ** 30 if cuda else None}
                    progress.doc["step_seconds"] = elapsed / steps
                    mem = f"  mem {record['max_mem_gb']:.1f}G" if cuda else ""
                    print(f"step {step:>7}/{total_steps}  epoch {record['epoch']:.2f}  loss {loss_v:.4f} "
                          f"(bce {bce_v:.4f} dice {dice_v:.4f} labels {label_v:.4f})  lr {record['lr']:.2e}  gnorm {record['grad_norm']:.2f}  "
                          f"{record['samples_per_s']:.0f} crops/s{mem}", flush=True)
                    log(args, record)
                    progress.update(samples_per_s=record["samples_per_s"], loss=loss_v)
                    window = fresh_window()
                report(phase="training")

                if stop:
                    stopped = True
                    print(f"stop requested: saving at step {step}", flush=True)
                    log(args, {"event": "stop", "step": step})
                    checkpoint("last")
                    break
                validated = False
                if step == total_steps or (args.val_every and step % args.val_every == 0):
                    run_validation()
                    validated = True
                if step % args.save_every == 0 or step == total_steps:
                    checkpoint(f"step_{step:06d}")
                    checkpoint("last")
                    prune_checkpoints(args.out, args.keep)
                if validated:
                    window["t"] = time.perf_counter()       # validation time is not training throughput
                if step >= total_steps:
                    break
            else:
                state["epoch"] += 1
                state["epoch_samples"] = 0
                print(f"epoch {state['epoch']} finished at step {state['step']}", flush=True)
                log(args, {"event": "epoch", "step": state["step"], "epoch": state["epoch"]})
                if not args.val_every and state["last_val_step"] != state["step"]:
                    run_validation()
                    window["t"] = time.perf_counter()
                checkpoint("last")
    except KeyboardInterrupt:
        print("interrupted", flush=True)
        checkpoint("last")
        stopped = True
    report(phase="stopped" if stopped else "finished", val=None)
    progress.update(force=True, finished_at=time.time())
    log(args, {"event": "stopped" if stopped else "finished", "step": state["step"],
               "best_mean_f1": state["best_mean_f1"], "best_step": state["best_step"]})
    print(f"done: step {state['step']}, best val mean F1 {state['best_mean_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
