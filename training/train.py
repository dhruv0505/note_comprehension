"""
train.py — LoRA fine-tuning of Florence-2-large (model.py) on segmentation prompts (seg_dataset.py).

    python render_page_val.py --pages 200                   # once: 1600 x 1600 pages -> data/val_pages/
    python train.py                                         # data/manifest.jsonl (tail 1% held out) + data/val_pages/
    python train.py --manifest a.jsonl b.jsonl --val-manifest v.jsonl --page-val other_pages/ --epochs 3
    torchrun --nproc_per_node 2 train.py ...                # one process per GPU (distributed.py)
    python train.py --max-steps 20 --page-val-max 5         # quick smoke run
    python train.py --resume runs/seg/last                  # continue from a checkpoint
    python train.py --lora-rank 32                          # LoRA rank (alpha 2 x rank unless --lora-alpha)
    python train.py --init-adapter ../inference/comprehension/seg   # train further from a saved adapter, fresh optimizer

Each step: the frozen vision tower encodes the 768 x 768 image into 577 tokens,
the prompt tokens are appended, and the LoRA-wrapped encoder/decoder is trained
with cross-entropy on the loc-token target (padding = -100, ignored). That loss
is the only thing backpropagated.

The image + prompt are merged here rather than by Florence2ForConditionalGeneration
.forward, whose merge sets the prompt's attention mask to all ones, so padded
prompt tokens in a batch would be attended to. Its generate() drops the mask the
same way; validation calls language_model.generate with the real mask.

Several GPUs (torchrun): every process shuffles the rows alike and takes every
world-th of them, so --batch-size is per GPU and a step sees batch size x GPUs x
--grad-accum rows. Gradients are averaged across processes before each optimizer
step. Validation is split across the processes and its counts combined.

Validation runs every --val-every steps, at the end of every epoch and after the last step (seg_eval.py):
  rows    teacher-forced: held-out manifest rows, each prompt on its own image, so
          level-1 crops come from ground-truth level-2 boxes. Also the validation loss.
  pages   the cascade, with --page-val: whole pages (e.g. 1600 x 1600) resized to
          768 for the level-2 and arrow prompts, then the level-1 prompt on crops of
          the *predicted* level-2 boxes, everything scored against the page's ground
          truth in page pixels. The textlines and formulas the cascade found are then read
          ten at a time as the recognition data draws them (recog_eval.py), logging
          pages/recog_loss/textline and pages/recog_loss/formula: their token cross-entropy,
          pages/recog_cer/...: the character error rate of the generated text, and
          pages/recog_acc/...: its correct characters over total characters, lines the
          cascade missed counted as read as nothing.
Each logs parse validity and per-class precision / recall / F1 with their mean
(rows/... and pages/... in log.jsonl), so the two can be compared and a class
degrading is visible. best/ is the checkpoint with the highest score: with
--page-val, the pages' mean F1 (BEST_F1_WEIGHT of it) together with what the
cascade read, the textline and formula recognition accuracies sharing the rest
(best_score) — a box in the right place around misread text is half a result.
Without pages it is the rows' mean F1, and without recognition scores the mean
F1 alone.

Validation rows: --val-manifest if given, otherwise the last --val-fraction of rows.
Rows of one image are written next to each other, so a tail split keeps an
image's prompts on one side (bar the single boundary image). The first
--eval-batches x --eval-batch-size of them are used, the same rows every time.

Control while running, files in --out read every few seconds:
  control.json    {"seq": 3, "lr": 5e-5}: from the next step the schedule's peak learning
                  rate is 5e-5, warmup and cosine decay scaled to match. Applied once per
                  seq (a higher seq is a new request); control_ack.json and a log line say so.
  STOP            save last/ and exit cleanly, as if the run had ended there.

Output (--out, default runs/seg):
  config.json                        the arguments
  log.jsonl                          training lines, each validation, and events (start, lr_change, stop)
  progress.json                      what the run is doing now: phase, step, validation progress, speed, ETA
  step_000500/, last/, best/         adapter (adapter_config.json + adapter_model.safetensors,
                                     loadable with PeftModel.from_pretrained) and trainer_state.pt
  samples_step_000500_pages.jsonl    every prediction, output and recognition read for the first --samples pages,
                                     written as made (rows keep no samples)

Resume restores adapters, optimizer, scheduler (with any learning-rate change),
grad scaler, and the position in the epoch: each epoch's shuffle is seeded by
(seed, epoch), so the rest of the interrupted epoch is replayed in the same
order without re-reading its batches. Resume with the same number of GPUs.

--init-adapter only loads an adapter's weights (its rank must be --lora-rank):
the optimizer, schedule and step count start afresh. A --resume takes precedence,
so a run restarted from its own last/ does not go back to the starting adapter.
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
from torch.utils.data import DataLoader, Sampler, Subset

import distributed as D
from model import build_model, report_parameters
from progress import ProgressFile, SampleStream, read_json, write_json
from recog_eval import MarkRemover, Recognizer, model_read_fn
from recog_eval import rl as recog_layout
from seg_dataset import DEFAULT_MANIFEST, HERE, IGNORE_INDEX, Collator, SegDataset
from seg_eval import (ARROW_PROMPT, LEVEL1_PROMPTS, LEVEL2_PROMPT, EvalCollator, PageSet, SegMetrics,
                      add_metric_args, collate_pages, encoder_inputs, flat_record, format_report,
                      model_generate_fn, validate, validate_pages)


DEFAULT_PAGE_VAL = os.path.join(HERE, "data", "val_pages")     # render_page_val.py's output
BEST_F1_WEIGHT = 0.6            # best/: the rest is shared by the recognition accuracies (best_score)
CONTROL_EVERY = 5.0                                             # seconds between reads of the control files


class EpochSampler(Sampler):
    """A shuffle of `indices` seeded by (seed, epoch), started `start` samples in,
    then this rank's share of it. Every rank shuffles alike and takes every
    world-th of the rest, cut to a multiple of the world so all ranks run the same
    number of batches. `start` counts samples over all ranks."""

    def __init__(self, indices, seed, rank=0, world=1):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.seed, self.epoch, self.start = seed, 0, 0
        self.rank, self.world = rank, world

    def set_epoch(self, epoch, start=0):
        self.epoch, self.start = epoch, min(start, len(self.indices))

    def __iter__(self):
        order = np.random.default_rng((self.seed, self.epoch)).permutation(len(self.indices))
        rest = self.indices[order[self.start:]]
        rest = rest[:len(rest) - len(rest) % self.world]
        return iter(rest[self.rank::self.world].tolist())

    def __len__(self):
        n = len(self.indices) - self.start
        return (n - n % self.world) // self.world


def make_loader(dataset, sampler, collator, batch_size, workers, device):
    return DataLoader(dataset, sampler=sampler, collate_fn=collator, batch_size=batch_size,
                      num_workers=workers, persistent_workers=workers > 0,
                      pin_memory=device.type == "cuda")


def split(args):
    """(train dataset, train indices, val dataset, val indices)."""
    train = SegDataset(args.manifest)
    if args.val_manifest:
        val = SegDataset(args.val_manifest)
        return train, np.arange(len(train)), val, np.arange(len(val))
    n_val = min(len(train) - 1, math.ceil(len(train) * args.val_fraction)) if args.val_fraction > 0 else 0
    cut = len(train) - n_val
    return train, np.arange(cut), train, np.arange(cut, len(train))


def forward_loss(base, batch, device, dtype):
    embeds, mask = encoder_inputs(base, batch, device, dtype)
    labels = batch["labels"].to(device, non_blocking=True)
    with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
        out = base.language_model(inputs_embeds=embeds, attention_mask=mask, labels=labels, use_cache=False)
    return out.loss, int((labels != IGNORE_INDEX).sum())


def set_train(model):
    """Train mode for the adapters (LoRA dropout); the frozen vision tower stays
    in eval mode so its drop-path does not perturb the image features."""
    model.train()
    model.get_base_model().vision_tower.eval()


def save_checkpoint(model, optimizer, scheduler, scaler, state, args, name):
    path = os.path.join(args.out, name)
    tmp = path + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), **state}, os.path.join(tmp, "trainer_state.pt"))
    shutil.rmtree(path, ignore_errors=True)
    os.replace(tmp, path)
    return path


def prune_checkpoints(out, keep):
    steps = sorted(d for d in os.listdir(out) if re.fullmatch(r"step_\d+", d))
    for name in steps[:-keep] if keep > 0 else []:
        shutil.rmtree(os.path.join(out, name), ignore_errors=True)


def load_adapter_weights(model, path):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    adapters = load_file(os.path.join(path, "adapter_model.safetensors"))
    result = set_peft_model_state_dict(model, adapters)
    # a text-only checkpoint has no vision-tower adapters: with --vision-lora those start fresh rather than fail
    fresh = [k for k in result.missing_keys if "lora_" in k and "vision_tower" in k]
    mismatched = [k for k in result.unexpected_keys + result.missing_keys if "lora_" in k and k not in fresh]
    if mismatched:
        raise RuntimeError(f"{path}: adapter keys missing from the checkpoint or the model: {mismatched[:5]}")
    if fresh and D.is_main():
        print(f"{len(fresh)} vision-tower adapter tensors start fresh: {path} has none")


def load_checkpoint(model, optimizer, scheduler, scaler, path):
    load_adapter_weights(model, path)
    state = torch.load(os.path.join(path, "trainer_state.pt"), map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state.pop("optimizer"))
    scheduler.load_state_dict(state.pop("scheduler"))
    scaler.load_state_dict(state.pop("scaler"))
    return state


def log(args, record):
    if D.is_main():
        with open(os.path.join(args.out, "log.jsonl"), "a") as f:
            f.write(json.dumps({"time": time.time(), **record}) + "\n")


def warn_on_prompt_drift(dataset, indices, rows=2000):
    """The cascade asks seg_eval.py's copies of the generator's prompts; training
    prompts that are not among them mean the copies are stale."""
    known = {LEVEL2_PROMPT, ARROW_PROMPT, *LEVEL1_PROMPTS.values(), *recog_layout.PROMPTS.values()}   # recognition's too
    unknown = sorted({dataset.row(int(i))["prompt"] for i in indices[:rows]} - known)
    if unknown:
        print("WARNING: these training prompts are not ones the page cascade asks (seg_eval.py "
              "LEVEL2_PROMPT / ARROW_PROMPT / LEVEL1_PROMPTS):\n  " + "\n  ".join(unknown))


def set_peak_lr(optimizer, scheduler, peak):
    """The schedule's peak learning rate to `peak` from the next step: the base
    rates LambdaLR multiplies by warmup / cosine are scaled, and so is the rate
    in use now. Returns (old peak, rate now)."""
    old = scheduler.base_lrs[0]
    factor = peak / old
    scheduler.base_lrs = [lr * factor for lr in scheduler.base_lrs]
    for group in optimizer.param_groups:
        group["lr"] *= factor
        if "initial_lr" in group:
            group["initial_lr"] *= factor
    scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
    return old, optimizer.param_groups[0]["lr"]


class Control:
    """Requests from outside the run (control.json, STOP in --out). Rank 0 reads
    them every CONTROL_EVERY seconds; every rank receives the same decision at the
    same step, so the processes never disagree about a stop or a rate."""

    def __init__(self, out, device):
        self.control, self.stop = os.path.join(out, "control.json"), os.path.join(out, "STOP")
        self.device, self.last = device, 0.0

    def poll(self, applied_seq):
        """(stop requested, (seq, peak lr) of a new rate request or None)."""
        values = [0.0, -1.0, 0.0]
        if D.is_main() and time.time() - self.last >= CONTROL_EVERY:
            self.last = time.time()
            values[0] = float(os.path.exists(self.stop))
            doc = read_json(self.control)
            try:
                seq, lr = int(doc["seq"]), float(doc["lr"])
                if seq > applied_seq and math.isfinite(lr) and lr > 0:
                    values[1], values[2] = seq, lr
            except (TypeError, KeyError, ValueError):
                pass
        stop, seq, lr = D.broadcast_values(values, self.device)
        return bool(stop), ((int(seq), lr) if seq >= 0 else None)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", nargs="+", default=[DEFAULT_MANIFEST])
    ap.add_argument("--val-manifest", nargs="+", default=None)
    ap.add_argument("--val-fraction", type=float, default=0.01, help="tail rows held out when no --val-manifest")
    ap.add_argument("--page-val", nargs="+", default=[DEFAULT_PAGE_VAL],
                    help="page folder(s) for the cascade validation (default data/val_pages if it exists; 'none' skips)")
    ap.add_argument("--page-val-max", type=int, default=0, help="first N validation pages (0 = all)")
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "seg"))
    ap.add_argument("--resume", default=None, help="checkpoint folder, e.g. runs/seg/last")
    ap.add_argument("--init-adapter", default=None,
                    help="adapter folder whose weights training starts from (optimizer and schedule start afresh)")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=None, help="default 2 x --lora-rank")

    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=None, help="optimizer steps; overrides --epochs")
    ap.add_argument("--batch-size", type=int, default=6, help="rows per GPU")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--grad-checkpointing", action="store_true", help="less memory, ~30%% slower")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--val-every", type=int, default=0, help="also validate every N optimizer steps (0 = epoch ends only)")
    ap.add_argument("--eval-batches", type=int, default=25, help="validation row batches per pass (0 = all)")
    ap.add_argument("--samples", type=int, default=4, help="validation pages whose outputs are saved")
    add_metric_args(ap)
    ap.add_argument("--vision-lora", action="store_true",
                    help="adapt the DaViT vision tower too (its own weights stay frozen): what the decoder reads "
                         "is otherwise pretrained on photos and documents, never on pen strokes")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--keep", type=int, default=3, help="step_* checkpoints kept (last/ and best/ always are)")
    ap.add_argument("--allow-cpu", action="store_true", help="run without a GPU, in float32 (a smoke test only)")
    args = ap.parse_args()
    if args.page_val == ["none"] or (args.page_val == [DEFAULT_PAGE_VAL] and not os.path.isdir(DEFAULT_PAGE_VAL)):
        args.page_val = None
    if args.init_adapter:
        args.init_adapter = os.path.expanduser(args.init_adapter)      # given unexpanded by train_remote.sh
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("train.py needs a CUDA GPU (the base model is loaded in half precision); "
                         "--allow-cpu runs a float32 smoke test")
    rank, world, device = D.setup()
    os.makedirs(args.out, exist_ok=True)
    progress = ProgressFile(os.path.join(args.out, "progress.json"), phase="loading", world=world)
    try:
        run(args, rank, world, device, progress)
    except BaseException as error:
        progress.update(force=True, phase="stopped" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()[-4000:])
        raise
    finally:
        D.cleanup()


def run(args, rank, world, device, progress):
    if D.is_main():
        with open(os.path.join(args.out, "config.json"), "w") as f:
            json.dump({**vars(args), "world_size": world}, f, indent=2)
    progress.update(force=True)
    torch.manual_seed(args.seed)

    model, processor, tokenizer = build_model(args.lora_rank, args.lora_alpha, vision_lora=args.vision_lora)
    model = model.to(device)
    if D.is_main():
        report_parameters(model)
    base = model.get_base_model()
    dtype = base.language_model.lm_head.weight.dtype
    image_size = processor.image_processor.size["height"]
    if args.grad_checkpointing:
        base.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    set_train(model)

    train, train_indices, val, val_indices = split(args)
    if args.eval_batches > 0:
        val_indices = val_indices[:args.eval_batches * args.eval_batch_size]
    sampler = EpochSampler(train_indices, args.seed, rank, world)
    train_loader = make_loader(train, sampler, Collator(), args.batch_size, args.num_workers, device)
    val_loader = (make_loader(val, D.shard(val_indices).tolist(), EvalCollator(), args.eval_batch_size,
                              args.num_workers, device) if len(val_indices) else None)
    page_loader, n_pages = None, 0
    remove_marks = MarkRemover(args.mark_model, device) if args.mark_model and args.page_val else None
    if remove_marks is not None and D.is_main():
        print(f"recognition takes marks out with {args.mark_model} (step {remove_marks.step})")
    if args.page_val:
        pages = PageSet(args.page_val, args.page_val_max)
        n_pages = len(pages)
        page_loader = DataLoader(Subset(pages, D.shard(range(n_pages))), batch_size=args.page_batch,
                                 collate_fn=collate_pages, num_workers=args.num_workers)
        if D.is_main():
            warn_on_prompt_drift(train, train_indices)

    batches_per_epoch = len(sampler) // args.batch_size + bool(len(sampler) % args.batch_size)
    steps_per_epoch = max(1, batches_per_epoch // args.grad_accum)
    total_steps = args.max_steps or max(1, int(args.epochs * batches_per_epoch) // args.grad_accum)
    global_batch = args.batch_size * world * args.grad_accum
    if D.is_main():
        print(f"train {len(train_indices):,} rows ({batches_per_epoch:,} batches/epoch per GPU x {world} GPU), "
              f"val {len(val_indices):,} rows, {n_pages:,} pages; "
              f"{total_steps:,} optimizer steps at {args.batch_size} x {world} x {args.grad_accum} per step; {dtype}")
        if val_loader is None and page_loader is None:
            print("no validation rows or pages: validation and best/ are skipped")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, fused=device.type == "cuda")
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, min(args.warmup_steps, total_steps // 10), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)

    # best_mean_f1 is the score best/ was chosen by (best_score: the mean F1 with what the cascade read);
    # best_pages_mean_f1 is that same checkpoint's plain mean F1, so both can be shown
    state = {"step": 0, "epoch": 0, "epoch_samples": 0, "best_mean_f1": -1.0, "best_step": None,
             "best_pages_mean_f1": None, "control_seq": 0, "last_val_step": None}
    if args.resume:
        state.update(load_checkpoint(model, optimizer, scheduler, scaler, args.resume))
        if D.is_main():
            print(f"resumed {args.resume}: step {state['step']}, epoch {state['epoch']} "
                  f"at sample {state['epoch_samples']:,}")
    elif args.init_adapter:
        rank = (read_json(os.path.join(args.init_adapter, "adapter_config.json")) or {}).get("r")
        if rank is not None and rank != args.lora_rank:
            raise SystemExit(f"{args.init_adapter} has LoRA rank {rank}; this run's --lora-rank is {args.lora_rank}")
        load_adapter_weights(model, args.init_adapter)
        if D.is_main():
            print(f"starting from the adapter in {args.init_adapter} (rank {args.lora_rank})")

    has_validation = val_loader is not None or page_loader is not None
    val_seconds = {"last": None}

    def remaining_validations():
        if not has_validation:
            return 0
        left = total_steps - state["step"]
        count = left // args.val_every if args.val_every else 0
        count += max(0, total_steps // steps_per_epoch - state["step"] // steps_per_epoch)
        return max(1, count)

    def report(**fields):
        step_s = progress.doc.get("step_seconds")
        eta = None
        if step_s:
            eta = step_s * (total_steps - state["step"]) + (val_seconds["last"] or 0) * remaining_validations()
        progress.update(step=state["step"], total_steps=total_steps, epoch=state["epoch"] + state["epoch_samples"] /
                        max(1, len(train_indices)), epochs=args.max_steps and total_steps / steps_per_epoch or args.epochs,
                        steps_per_epoch=steps_per_epoch, lr=optimizer.param_groups[0]["lr"],
                        peak_lr=scheduler.base_lrs[0], control_seq=state["control_seq"],
                        best_mean_f1=state["best_mean_f1"], best_step=state["best_step"],
                        best_pages_mean_f1=state["best_pages_mean_f1"],
                        last_val_step=state["last_val_step"], eta_s=eta, **fields)

    progress.update(force=True, world=world, batch_size=args.batch_size, grad_accum=args.grad_accum,
                    global_batch=global_batch, train_rows=len(train_indices), val_rows=len(val_indices),
                    val_pages=n_pages, val_every=args.val_every, save_every=args.save_every)
    log(args, {"event": "start", "step": state["step"], "total_steps": total_steps, "world": world,
               "global_batch": global_batch, "steps_per_epoch": steps_per_epoch, "peak_lr": scheduler.base_lrs[0],
               "resumed_from": args.resume, "init_adapter": None if args.resume else args.init_adapter,
               "lora_rank": args.lora_rank})

    def checkpoint(name):
        report(phase="checkpointing")
        progress.update(force=True)
        if D.is_main():
            path = save_checkpoint(model, optimizer, scheduler, scaler, state, args, name)
            print(f"  saved {path}")
            progress.update(last_checkpoint={"name": name, "step": state["step"], "at": time.time()})
        D.barrier()

    def best_score(summary):
        """What best/ is judged by when pages are validated: the cascade's mean F1 together with what it read. A box
        in the right place around text the model misreads is half a result, so the recognition accuracies count too,
        with detection keeping most of the weight (BEST_F1_WEIGHT). Without recognition scores it is the mean F1."""
        read = [task["accuracy"] for task in (summary.get("recognition") or {}).values()
                if task.get("accuracy") is not None]
        if not read:
            return summary["mean_f1"]
        return BEST_F1_WEIGHT * summary["mean_f1"] + (1 - BEST_F1_WEIGHT) * sum(read) / len(read)

    def run_validation():
        t = time.perf_counter()
        model.eval()
        record, chosen = {"step": state["step"], "epoch": state["epoch"]}, None
        tag = f"step_{state['step']:06d}"
        if val_loader is not None:
            total = len(val_indices)
            report(phase="validating", val={"kind": "rows", "done": 0, "total": total, "step": state["step"]})
            progress.update(force=True)
            metrics, (loss_sum, tokens), _ = validate(
                base, val_loader, tokenizer, device, dtype, args.max_new_tokens,
                SegMetrics(args.iou, args.vertex_px, args.arrow_px),
                loss_fn=lambda batch: forward_loss(base, batch, device, dtype), summarize=False,
                progress=lambda n: progress.update(val={"kind": "rows", "done": min(total, n * world),
                                                        "total": total, "step": state["step"]}))
            rows = D.gather_metrics(metrics).summary()
            loss_sum, tokens = D.sum_values([loss_sum, tokens], device)
            rows["val_loss"] = loss_sum / max(tokens, 1)
            if D.is_main():
                print(f"validation rows at step {state['step']}, teacher-forced (ground-truth crops):\n"
                      f"{format_report(rows)}")
            record.update(flat_record(rows, "rows/"))
            chosen = rows["mean_f1"]
        if page_loader is not None:
            report(phase="validating", val={"kind": "pages", "done": 0, "total": n_pages, "step": state["step"]})
            progress.update(force=True)
            stream = SampleStream(os.path.join(args.out, f"samples_{tag}_pages.jsonl"))
            generate_fn = model_generate_fn(base, processor, device, dtype, args.max_new_tokens,
                                            args.eval_batch_size)
            metrics = SegMetrics(args.iou, args.page_vertex_px, args.page_arrow_px, args.page_min_box_px,
                                 unit="pages")
            recognize = Recognizer(model_read_fn(base, processor, device, dtype, args.max_new_tokens,
                                                 args.eval_batch_size), args.iou, args.page_min_box_px,
                                   remove_marks=remove_marks)
            metrics, _ = validate_pages(
                page_loader, generate_fn, metrics, image_size, keep_samples=args.samples if D.is_main() else 0,
                summarize=False, on_sample=stream.write, recognize=recognize,
                progress=lambda n: progress.update(val={"kind": "pages", "done": min(n_pages, n * world),
                                                        "total": n_pages, "step": state["step"]}))
            pages = D.gather_metrics(metrics).summary()
            if D.is_main():
                print(f"validation pages at step {state['step']}, cascade (predicted level-2 boxes -> level-1 crops):\n"
                      f"{format_report(pages)}")
            record.update(flat_record(pages, "pages/"))
            chosen = best_score(pages)
            record["pages/best_score"] = chosen
        set_train(model)
        record["val_seconds"] = val_seconds["last"] = time.perf_counter() - t
        state["last_val_step"] = state["step"]
        if D.is_main():
            print(f"  validation took {record['val_seconds']:.0f}s")
        log(args, record)
        report(phase="training", val=None)
        if chosen is not None and chosen > state["best_mean_f1"]:
            state["best_mean_f1"], state["best_step"] = chosen, state["step"]
            state["best_pages_mean_f1"] = record.get("pages/mean_f1")
            checkpoint("best")
            # not "pages/mean_f1": that key is what marks a validation record, and this is an event
            log(args, {"event": "best", "step": state["step"], "mean_f1": chosen,
                       "best/pages_mean_f1": record.get("pages/mean_f1")})

    control = Control(args.out, device)
    window = {"loss": 0.0, "batches": 0, "samples": 0, "tokens": 0, "t": time.perf_counter()}
    micro, stopped, step_clock = 0, False, time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    report(phase="training")
    progress.update(force=True)
    try:
        while state["step"] < total_steps and not stopped:
            sampler.set_epoch(state["epoch"], start=state["epoch_samples"])
            for batch in train_loader:
                loss, tokens = forward_loss(base, batch, device, dtype)
                scaler.scale(loss / args.grad_accum).backward()
                micro += 1
                state["epoch_samples"] += batch["labels"].shape[0] * world
                window["loss"] += loss.item()
                window["batches"] += 1
                window["samples"] += batch["labels"].shape[0]
                window["tokens"] += tokens
                if micro % args.grad_accum:
                    continue

                D.average_gradients(params)
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm).item()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                state["step"] += 1
                step = state["step"]
                now = time.perf_counter()
                previous = progress.doc.get("step_seconds")
                progress.doc["step_seconds"] = (now - step_clock) if previous is None else 0.9 * previous + 0.1 * (now - step_clock)
                step_clock = now

                stop, request = control.poll(state["control_seq"])
                if request is not None:
                    seq, peak = request
                    old_peak, lr_now = set_peak_lr(optimizer, scheduler, peak)
                    state["control_seq"] = seq
                    ack = {"seq": seq, "step": step, "old_peak_lr": old_peak, "peak_lr": peak, "lr": lr_now,
                           "at": time.time(), "status": "applied"}
                    if D.is_main():
                        write_json(os.path.join(args.out, "control_ack.json"), ack)
                        print(f"step {step}: peak learning rate {old_peak:.2e} -> {peak:.2e} (now {lr_now:.2e})")
                    log(args, {"event": "lr_change", **ack})

                if step % args.log_every == 0 or step == total_steps:
                    elapsed = time.perf_counter() - window["t"]
                    loss_sum, batches, samples, tokens_sum = D.sum_values(
                        [window["loss"], window["batches"], window["samples"], window["tokens"]], device)
                    record = {"step": step, "epoch": state["epoch"] + state["epoch_samples"] / len(train_indices),
                              "loss": loss_sum / max(batches, 1), "lr": scheduler.get_last_lr()[0],
                              "peak_lr": scheduler.base_lrs[0], "grad_norm": grad_norm,
                              "samples_per_s": samples / elapsed, "target_tokens_per_s": tokens_sum / elapsed,
                              "max_mem_gb": (torch.cuda.max_memory_allocated(device) / 2 ** 30
                                             if device.type == "cuda" else None)}
                    if D.is_main():
                        mem = f"  mem {record['max_mem_gb']:.1f}G" if record["max_mem_gb"] is not None else ""
                        print(f"step {step:>7}/{total_steps}  epoch {record['epoch']:.3f}  loss {record['loss']:.4f}  "
                              f"lr {record['lr']:.2e}  gnorm {grad_norm:.2f}  {record['samples_per_s']:.1f} img/s{mem}")
                    log(args, record)
                    progress.update(samples_per_s=record["samples_per_s"], loss=record["loss"])
                    window.update(loss=0.0, batches=0, samples=0, tokens=0, t=time.perf_counter())
                report(phase="training")

                if stop:
                    stopped = True
                    if D.is_main():
                        print(f"stop requested: saving at step {step}")
                    log(args, {"event": "stop", "step": step})
                    checkpoint("last")
                    break
                validated = False
                if has_validation and (step == total_steps or (args.val_every and step % args.val_every == 0)):
                    run_validation()
                    validated = True
                if step % args.save_every == 0 or step == total_steps:
                    checkpoint(f"step_{step:06d}")
                    checkpoint("last")
                    if D.is_main():
                        prune_checkpoints(args.out, args.keep)
                if validated:
                    window["t"] = step_clock = time.perf_counter()     # validation time is not training throughput
                if step >= total_steps:
                    break
            else:
                state["epoch"] += 1
                state["epoch_samples"] = 0
                if D.is_main():
                    print(f"epoch {state['epoch']} finished at step {state['step']}")
                log(args, {"event": "epoch", "step": state["step"], "epoch": state["epoch"]})
                if has_validation and state["last_val_step"] != state["step"]:
                    run_validation()
                    window["t"] = step_clock = time.perf_counter()
                checkpoint("last")
    except KeyboardInterrupt:
        # the unfinished accumulation window is dropped; its samples are replayed on resume
        if micro % args.grad_accum:
            state["epoch_samples"] -= (micro % args.grad_accum) * args.batch_size * world
            state["epoch_samples"] = max(state["epoch_samples"], 0)
        if D.is_main():
            print("interrupted")
        checkpoint("last")
        stopped = True
    report(phase="stopped" if stopped else "finished", val=None)
    progress.update(force=True, finished_at=time.time())
    log(args, {"event": "stopped" if stopped else "finished", "step": state["step"],
               "best_mean_f1": state["best_mean_f1"], "best_step": state["best_step"]})
    if D.is_main():
        print(f"done: step {state['step']}, best val mean F1 {state['best_mean_f1']:.4f}")


if __name__ == "__main__":
    main()
