"""Distributed training utilities."""

import os
from typing import Optional
import torch
import torch.distributed as dist


def setup_distributed(
    backend: str = "nccl",
    init_method: str = "env://",
) -> None:
    """
    Initialize distributed training.

    Args:
        backend: Distributed backend ("nccl" for GPU, "gloo" for CPU)
        init_method: Initialization method
    """
    if dist.is_initialized():
        return

    # Check if running in distributed mode
    if "RANK" not in os.environ:
        return

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # Initialize process group
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )

    print(f"Distributed training initialized: rank {rank}/{world_size}")


def cleanup_distributed() -> None:
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main process."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """Get current process rank."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get total number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_local_rank() -> int:
    """Get local rank (within node)."""
    return int(os.environ.get("LOCAL_RANK", 0))


def synchronize() -> None:
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def reduce_tensor(
    tensor: torch.Tensor,
    average: bool = True,
) -> torch.Tensor:
    """
    Reduce tensor across all processes.

    Args:
        tensor: Tensor to reduce
        average: Whether to average (vs sum)

    Returns:
        Reduced tensor
    """
    if not dist.is_initialized():
        return tensor

    world_size = get_world_size()

    # Clone to avoid modifying original
    reduced = tensor.clone()

    # All-reduce
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)

    if average:
        reduced /= world_size

    return reduced
