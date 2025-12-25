"""Logging utilities with W&B integration."""

import os
from typing import Dict, Any, Optional
import wandb


_wandb_initialized = False


def setup_wandb(
    project: str = "modality-adaptive-attention",
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    resume: bool = False,
    **kwargs,
) -> None:
    """
    Initialize Weights & Biases logging.

    Args:
        project: W&B project name
        name: Run name
        config: Configuration to log
        resume: Whether to resume a previous run
        **kwargs: Additional wandb.init arguments
    """
    global _wandb_initialized

    if _wandb_initialized:
        return

    # Check if W&B is disabled
    if os.environ.get("WANDB_DISABLED", "false").lower() == "true":
        wandb.init(mode="disabled")
        _wandb_initialized = True
        return

    try:
        wandb.init(
            project=project,
            name=name,
            config=config,
            resume=resume,
            **kwargs,
        )
        _wandb_initialized = True
        print(f"W&B initialized: {wandb.run.name}")
    except Exception as e:
        print(f"W&B initialization failed: {e}")
        wandb.init(mode="disabled")
        _wandb_initialized = True


def log_metrics(
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    commit: bool = True,
) -> None:
    """
    Log metrics to W&B.

    Args:
        metrics: Dictionary of metric name -> value
        step: Global step (optional)
        commit: Whether to commit the log
    """
    if not _wandb_initialized:
        return

    try:
        if step is not None:
            wandb.log(metrics, step=step, commit=commit)
        else:
            wandb.log(metrics, commit=commit)
    except Exception:
        pass


def log_summary(
    summary: Dict[str, Any],
) -> None:
    """
    Log summary metrics to W&B.

    Args:
        summary: Dictionary of summary metric name -> value
    """
    if not _wandb_initialized:
        return

    try:
        for key, value in summary.items():
            wandb.run.summary[key] = value
    except Exception:
        pass


def log_artifact(
    name: str,
    artifact_type: str,
    path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log an artifact to W&B.

    Args:
        name: Artifact name
        artifact_type: Type of artifact (e.g., "model", "dataset")
        path: Path to artifact files
        metadata: Optional metadata
    """
    if not _wandb_initialized:
        return

    try:
        artifact = wandb.Artifact(name, type=artifact_type, metadata=metadata)
        artifact.add_dir(path)
        wandb.log_artifact(artifact)
    except Exception as e:
        print(f"Failed to log artifact: {e}")


def finish() -> None:
    """Finish W&B run."""
    global _wandb_initialized

    if _wandb_initialized:
        try:
            wandb.finish()
        except Exception:
            pass
        _wandb_initialized = False
