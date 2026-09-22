"""
distributed.py — one process per GPU under torchrun, or a single process without it.

    torchrun --nproc_per_node 2 train.py ...        # WORLD_SIZE / RANK / LOCAL_RANK set by torchrun
    python train.py ...                             # world of one: every helper here is a no-op

Gradients are averaged by hand (average_gradients) rather than through
DistributedDataParallel: train.py calls the language model and the vision tower
directly instead of the peft model's forward, which DDP's reducer needs to see.
Only the LoRA matrices train, so the flattened gradient is a few million numbers.

Validation is split by rank (shard) and its counts brought back together
(gather_metrics), so every rank reports the same numbers and makes the same
best-checkpoint decision.
"""

from __future__ import annotations
import os
from datetime import timedelta

import torch
import torch.distributed as dist


def setup():
    """(rank, world size, device). Initialises the process group under torchrun."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    if world > 1 and not dist.is_initialized():
        # validation and checkpoint saving run long between collectives
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(hours=3))
    return rank, world, device


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def is_main():
    return rank() == 0


def barrier():
    if dist.is_initialized():
        dist.barrier()


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def shard(items, world=None, rank_=None):
    """This rank's share of a sequence: every world-th item from its rank."""
    world = world_size() if world is None else world
    rank_ = rank() if rank_ is None else rank_
    return items[rank_::world]


def average_gradients(params):
    """Every rank's gradients replaced by their mean over the ranks."""
    if not dist.is_initialized():
        return
    grads = [p.grad for p in params if p.grad is not None]
    flat = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat)
    flat /= dist.get_world_size()
    for grad, synced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
        grad.copy_(synced)


def sum_values(values, device):
    """Numbers summed over the ranks (a list of floats in, the sums out)."""
    if not dist.is_initialized():
        return list(values)
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor)
    return tensor.tolist()


def broadcast_values(values, device):
    """Rank 0's numbers on every rank."""
    if not dist.is_initialized():
        return list(values)
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.broadcast(tensor, src=0)
    return tensor.tolist()


def gather_objects(obj):
    """Every rank's picklable object, in rank order."""
    if not dist.is_initialized():
        return [obj]
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out


def gather_metrics(metrics):
    """A SegMetrics holding every rank's counts (the rank's own when not distributed)."""
    if not dist.is_initialized():
        return metrics
    merged = metrics.empty_like()
    for state in gather_objects(metrics.state()):
        merged.merge(state)
    return merged
