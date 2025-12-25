#!/usr/bin/env python3
"""Training script for tuned modality-adaptive attention models."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
import wandb
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from src.models.vision_text_model import create_model
from src.data import ConceptualCaptionsDataset, VisionTextCollator
from src.training import Trainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def log_attention_stats(model, step: int):
    """Log attention module statistics (for V2 models)."""
    stats = {}
    for name, module in model.named_modules():
        if hasattr(module, 'get_attention_stats'):
            layer_stats = module.get_attention_stats()
            for key, value in layer_stats.items():
                stats[f"{name}/{key}"] = value

    if stats and wandb.run is not None:
        wandb.log(stats, step=step)


def main():
    parser = argparse.ArgumentParser(description="Train tuned modality-adaptive model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tuned_v2.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name (overrides config)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override experiment name if provided
    if args.experiment_name:
        config["logging"]["wandb_run_name"] = args.experiment_name
        config["checkpoint"]["save_dir"] = f"./checkpoints/{args.experiment_name}"

    experiment_name = config["logging"].get("wandb_run_name", "tuned_experiment")
    print(f"Running experiment: {experiment_name}")
    print(f"Attention type: {config['model']['attention_type']}")
    print(f"Attention config: {config.get('attention', {})}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Enable TF32 for A100
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Create model
    print("\nCreating model...")
    model = create_model(config)
    model = model.to(device)

    num_params = model.get_num_params(trainable_only=True)
    print(f"Trainable parameters: {num_params:,}")

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("\nLoading dataset...")
    train_dataset = ConceptualCaptionsDataset(
        cache_dir=config["data"]["cache_dir"],
        max_text_length=config["data"]["max_caption_length"],
        use_cached_features=config["data"]["use_cached_features"],
    )

    # Create data loader
    collator = VisionTextCollator(
        use_cached_features=config["data"]["use_cached_features"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )

    # Convert config to trainer format
    trainer_config = {
        "model": config["model"],
        "training": {
            "learning_rate": config["training"]["learning_rate"],
            "weight_decay": config["training"]["weight_decay"],
            "warmup_steps": config["training"]["warmup_steps"],
            "total_steps": config["training"]["max_steps"],
            "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
            "max_grad_norm": config["training"]["max_grad_norm"],
            "mixed_precision": config["training"].get("fp16", True),
        },
        "checkpoint": {
            "save_every": config["checkpoint"]["save_steps"],
            "save_dir": config["checkpoint"]["save_dir"],
        },
        "logging": {
            "log_every": config["logging"]["log_steps"],
        },
        "evaluation": {
            "eval_every": config["checkpoint"]["eval_steps"],
        },
    }

    # Initialize wandb
    wandb.init(
        project=config["logging"]["wandb_project"],
        name=experiment_name,
        config=config,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        config=trainer_config,
        device=device,
    )

    # Run training
    print("\nStarting training...")
    trainer.train(resume_from=args.resume)

    print(f"\nTraining complete!")
    wandb.finish()


if __name__ == "__main__":
    main()
