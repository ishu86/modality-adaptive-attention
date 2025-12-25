#!/usr/bin/env python3
"""Main training script for Modality-Adaptive Attention models."""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from omegaconf import OmegaConf

from src.data import ConceptualCaptionsDataset, VisionTextCollator
from src.models import VisionTextModel
from src.models.vision_text_model import create_model
from src.training import Trainer
from src.utils.logging import setup_wandb, finish
from src.utils.checkpointing import get_latest_checkpoint


def load_config(config_path: str) -> dict:
    """Load and merge configuration files."""
    config_path = Path(config_path)

    # Load base config
    base_config = OmegaConf.load(config_path.parent / "base.yaml")

    # Load experiment config
    exp_config = OmegaConf.load(config_path)

    # Handle defaults/inheritance
    if "defaults" in exp_config:
        # Already loaded base, just merge
        del exp_config["defaults"]

    # Merge configs
    config = OmegaConf.merge(base_config, exp_config)

    return OmegaConf.to_container(config, resolve=True)


def create_dataloaders(config: dict):
    """Create training and validation dataloaders."""
    data_config = config.get("data", {})
    training_config = config.get("training", {})

    # Training dataset
    train_dataset = ConceptualCaptionsDataset(
        subset_size=data_config.get("train_subset_size", 100000),
        cache_dir=data_config.get("cache_dir", "./cache"),
        use_cached_features=True,
    )

    # Collator
    collator = VisionTextCollator(use_cached_features=True)

    # Training dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training_config.get("batch_size", 8),
        shuffle=True,
        num_workers=data_config.get("num_workers", 4),
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )

    # Validation dataloader (optional)
    val_loader = None
    # Can add COCO validation here

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train Vision-Text Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume from latest checkpoint",
    )
    parser.add_argument(
        "--wandb-disabled",
        action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    args = parser.parse_args()

    # Set wandb mode
    if args.wandb_disabled:
        os.environ["WANDB_DISABLED"] = "true"

    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    experiment_name = config.get("experiment_name", "experiment")
    print(f"Experiment: {experiment_name}")

    # Setup W&B
    logging_config = config.get("logging", {})
    setup_wandb(
        project=logging_config.get("wandb_project", "modality-adaptive-attention"),
        name=experiment_name,
        config=config,
    )

    # Override checkpoint directory to be experiment-specific
    config["checkpoint"]["save_dir"] = f"./checkpoints/{experiment_name}"

    # Create model
    print("Creating model...")
    model = create_model(config)
    print(f"Trainable parameters: {model.get_num_params():,}")

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"Training samples: {len(train_loader.dataset)}")

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=config,
        device=device,
    )

    # Resume from checkpoint
    resume_from = args.resume
    if args.resume_latest:
        checkpoint_dir = config.get("checkpoint", {}).get("save_dir", "./checkpoints")
        resume_from = get_latest_checkpoint(checkpoint_dir)
        if resume_from:
            print(f"Resuming from latest checkpoint: {resume_from}")

    # Train
    print("\nStarting training...")
    try:
        metrics = trainer.train(resume_from=resume_from)
        print("\nTraining completed!")
        print(f"Final loss: {metrics['train_loss']:.4f}")
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        trainer.save("interrupted")
    finally:
        finish()


if __name__ == "__main__":
    main()
