"""Model checkpointing utilities."""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    epoch: int = 0,
    global_step: int = 0,
    config: Optional[Dict[str, Any]] = None,
    path: Union[str, Path] = "./checkpoints/checkpoint",
    save_optimizer: bool = True,
) -> str:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        global_step: Current global step
        config: Training configuration
        path: Save path (directory or file)
        save_optimizer: Whether to save optimizer state

    Returns:
        Path to saved checkpoint
    """
    path = Path(path)

    # Create directory if needed
    if not path.suffix:
        path.mkdir(parents=True, exist_ok=True)
        checkpoint_path = path / "pytorch_model.bin"
        config_path = path / "config.pt"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path = path
        config_path = path.parent / "config.pt"

    # Build checkpoint dict
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }

    if save_optimizer and optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    # Save model checkpoint
    torch.save(checkpoint, checkpoint_path)

    # Save config separately
    if config is not None:
        torch.save(config, config_path)

    return str(checkpoint_path)


def load_checkpoint(
    path: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        device: Device to load to
        strict: Whether to strictly enforce state dict matching

    Returns:
        Checkpoint dictionary with metadata
    """
    path = Path(path)

    # Find checkpoint file
    if path.is_dir():
        checkpoint_path = path / "pytorch_model.bin"
    else:
        checkpoint_path = path

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    map_location = device if device else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # Load model weights
    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def get_latest_checkpoint(
    checkpoint_dir: Union[str, Path],
    prefix: str = "checkpoint-",
) -> Optional[Path]:
    """
    Find the latest checkpoint in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints
        prefix: Checkpoint name prefix

    Returns:
        Path to latest checkpoint or None
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return None

    checkpoints = []
    for item in checkpoint_dir.iterdir():
        if item.is_dir() and item.name.startswith(prefix):
            try:
                step = int(item.name.replace(prefix, ""))
                checkpoints.append((step, item))
            except ValueError:
                continue

    if not checkpoints:
        return None

    # Sort by step and return latest
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


def cleanup_checkpoints(
    checkpoint_dir: Union[str, Path],
    keep_last: int = 3,
    prefix: str = "checkpoint-",
) -> None:
    """
    Clean up old checkpoints, keeping only the most recent.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last: Number of checkpoints to keep
        prefix: Checkpoint name prefix
    """
    import shutil

    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return

    checkpoints = []
    for item in checkpoint_dir.iterdir():
        if item.is_dir() and item.name.startswith(prefix):
            try:
                step = int(item.name.replace(prefix, ""))
                checkpoints.append((step, item))
            except ValueError:
                continue

    # Sort by step (newest first)
    checkpoints.sort(key=lambda x: x[0], reverse=True)

    # Remove old checkpoints
    for step, path in checkpoints[keep_last:]:
        print(f"Removing old checkpoint: {path}")
        shutil.rmtree(path)
