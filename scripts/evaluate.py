#!/usr/bin/env python3
"""Evaluation script for trained models."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from omegaconf import OmegaConf
from transformers import GPT2Tokenizer

from src.data import COCOCaptionsDataset, VisionTextCollator
from src.models.vision_text_model import create_model
from src.evaluation import Evaluator, compute_caption_metrics
from src.utils.checkpointing import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Evaluate Vision-Text Model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (uses checkpoint config if not provided)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=30,
        help="Maximum caption length",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    if args.config:
        config = OmegaConf.load(args.config)
        config = OmegaConf.to_container(config, resolve=True)
    else:
        config_path = checkpoint_path / "config.pt" if checkpoint_path.is_dir() else checkpoint_path.parent / "config.pt"
        if config_path.exists():
            config = torch.load(config_path)
        else:
            # Use default config
            config = {
                "model": {
                    "hidden_dim": 768,
                    "num_heads": 12,
                    "num_layers": 6,
                    "attention_type": "softmax",
                }
            }

    # Create model
    print("Loading model...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = create_model(config)

    # Load checkpoint
    load_checkpoint(checkpoint_path, model=model, device=device)
    model.eval()

    print(f"Model loaded. Parameters: {model.get_num_params():,}")

    # Create evaluation dataloader
    print("Loading COCO validation set...")
    from src.data.transforms import get_image_transform
    image_transform = get_image_transform(is_train=False)

    dataset = COCOCaptionsDataset(
        cache_dir=config.get("data", {}).get("cache_dir", "./cache"),
        use_cached_features=True,
        image_transform=image_transform,
    )

    # Check if dataset is using cached features or images
    use_cached = dataset.use_cached_features
    collator = VisionTextCollator(use_cached_features=use_cached, is_eval=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collator,
    )

    print(f"Evaluating on {len(dataset)} images...")

    # Create evaluator
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    evaluator = Evaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=args.max_length,
    )

    # Run evaluation
    results = evaluator.evaluate(dataloader, compute_metrics=True)

    # Save results
    results_file = output_dir / f"{checkpoint_path.stem}_results.json"

    # Prepare serializable results
    save_results = {
        "checkpoint": str(checkpoint_path),
        "metrics": results.get("metrics", {}),
        "perplexity": results.get("perplexity"),
        "num_samples": len(dataset),
    }

    with open(results_file, "w") as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Perplexity: {results.get('perplexity', 'N/A'):.2f}")
    if results.get("metrics"):
        for name, value in results["metrics"].items():
            print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
