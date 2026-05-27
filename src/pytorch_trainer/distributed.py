"""Distributed training utilities for multi-GPU data parallelism.

Thin wrapper around PyTorch DDP. When RANK env var is absent (no torchrun),
all functions degrade to single-GPU no-ops for backward compatibility.
"""

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def init_distributed() -> tuple[int, int, torch.device]:
    """Initialize distributed training if launched via torchrun.

    Returns:
        (rank, world_size, device)
    """
    if "RANK" not in os.environ:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return 0, 1, device

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    return rank, world_size, device


def cleanup_distributed():
    """Destroy process group if initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int = 0) -> bool:
    """Check if this is the main process (rank 0)."""
    return rank == 0


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying model, unwrapping DDP if necessary."""
    if isinstance(model, DDP):
        return model.module
    return model


def wrap_model_ddp(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Wrap model in DDP if distributed is initialized, otherwise no-op."""
    if dist.is_initialized():
        return DDP(model, device_ids=[device.index])
    return model


def reduce_metrics(metrics: dict, world_size: int) -> dict:
    """All-reduce (average) numeric metric values across ranks.

    Non-numeric values are passed through unchanged.
    """
    if world_size <= 1 or not dist.is_initialized():
        return metrics

    reduced = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            tensor = torch.tensor(value, dtype=torch.float64, device="cuda")
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            reduced[key] = tensor.item()
        else:
            reduced[key] = value
    return reduced
