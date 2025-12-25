#!/usr/bin/env python3
"""Analyze and visualize evaluation results."""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer

from src.models.vision_text_model import create_model
from src.utils.checkpointing import load_checkpoint
from src.data import COCOCaptionsDataset, VisionTextCollator


def load_results(results_dir: str) -> dict:
    """Load all results from the results directory."""
    results_dir = Path(results_dir)
    results = {}

    for subdir in results_dir.iterdir():
        if subdir.is_dir():
            for json_file in subdir.glob("*.json"):
                with open(json_file) as f:
                    data = json.load(f)
                    results[subdir.name] = data

    return results


def plot_metrics_comparison(results: dict, output_path: str):
    """Create bar chart comparing metrics across models."""
    models = list(results.keys())
    metrics = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L", "CIDEr"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Caption metrics
    ax1 = axes[0]
    x = np.arange(len(metrics))
    width = 0.25

    colors = {'baseline': '#2ecc71', 'proposed': '#3498db', 'ablation': '#e74c3c'}

    for i, model in enumerate(models):
        if model in results:
            values = [results[model]["metrics"].get(m, 0) for m in metrics]
            # Cap very small values for visualization
            values = [max(v, 0) for v in values]
            offset = (i - len(models)/2 + 0.5) * width
            bars = ax1.bar(x + offset, values, width, label=model.replace('_', ' ').title(),
                          color=colors.get(model, '#95a5a6'))

    ax1.set_xlabel('Metric', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Caption Quality Metrics', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45, ha='right')
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Perplexity comparison
    ax2 = axes[1]
    perplexities = []
    model_names = []
    bar_colors = []

    for model in models:
        if model in results:
            ppl = results[model].get("perplexity", 0)
            perplexities.append(ppl)
            model_names.append(model.replace('_', ' ').title())
            bar_colors.append(colors.get(model, '#95a5a6'))

    bars = ax2.bar(model_names, perplexities, color=bar_colors)
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Perplexity', fontsize=12)
    ax2.set_title('Model Perplexity (Lower is Better*)', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, perplexities):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved metrics comparison to: {output_path}")
    plt.close()


def plot_bleu_breakdown(results: dict, output_path: str):
    """Create detailed BLEU score breakdown."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = ['baseline', 'proposed', 'ablation']
    bleu_metrics = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]

    colors = {'baseline': '#2ecc71', 'proposed': '#3498db', 'ablation': '#e74c3c'}
    markers = {'baseline': 'o', 'proposed': 's', 'ablation': '^'}

    for model in models:
        if model in results:
            values = [results[model]["metrics"].get(m, 0) for m in bleu_metrics]
            ax.plot(bleu_metrics, values, marker=markers[model],
                   label=model.replace('_', ' ').title(),
                   color=colors[model], linewidth=2, markersize=8)

    ax.set_xlabel('BLEU N-gram', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('BLEU Score Breakdown by N-gram', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved BLEU breakdown to: {output_path}")
    plt.close()


def investigate_model_outputs(checkpoint_path: str, model_name: str, num_samples: int = 10):
    """Investigate what a model is actually generating."""
    print(f"\n{'='*60}")
    print(f"Investigating: {model_name}")
    print(f"{'='*60}")

    checkpoint_path = Path(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    config_path = checkpoint_path / "config.pt"
    if config_path.exists():
        config = torch.load(config_path)
    else:
        config = {
            "model": {
                "hidden_dim": 768,
                "num_heads": 12,
                "num_layers": 6,
                "attention_type": "softmax",
            }
        }

    # Create and load model
    model = create_model(config)
    load_checkpoint(checkpoint_path, model=model, device=device)
    model = model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Load a few samples
    dataset = COCOCaptionsDataset(
        cache_dir="./cache",
        use_cached_features=True,
    )

    collator = VisionTextCollator(use_cached_features=True, is_eval=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
    )

    print(f"\nSample generations:")
    print("-" * 60)

    generated_captions = []

    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break
        if batch is None:
            continue

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        vision_features = batch.get("vision_features")

        with torch.no_grad():
            generated_ids = model.generate(
                vision_features=vision_features,
                max_length=30,
                do_sample=False,
            )

        caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        generated_captions.append(caption)

        ref_captions = batch.get("reference_captions", [[]])[0]
        ref = ref_captions[0] if ref_captions else "N/A"

        print(f"\nSample {i+1}:")
        print(f"  Reference: {ref[:80]}...")
        print(f"  Generated: {caption}")

    # Analyze token distribution
    print(f"\n{'='*60}")
    print("Token Analysis:")
    print("-" * 60)

    all_tokens = []
    for cap in generated_captions:
        tokens = tokenizer.encode(cap)
        all_tokens.extend(tokens)

    if all_tokens:
        token_counts = Counter(all_tokens)
        print(f"Total tokens generated: {len(all_tokens)}")
        print(f"Unique tokens: {len(token_counts)}")
        print(f"\nMost common tokens:")
        for token_id, count in token_counts.most_common(10):
            token_str = tokenizer.decode([token_id])
            print(f"  '{token_str}' (id={token_id}): {count} times")

    # Check if model is just repeating same tokens
    unique_captions = set(generated_captions)
    print(f"\nUnique captions: {len(unique_captions)} out of {len(generated_captions)}")

    if len(unique_captions) == 1:
        print("WARNING: Model generating identical captions for all inputs!")

    return generated_captions


def create_summary_table(results: dict, output_path: str):
    """Create a summary table as an image."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    # Prepare data
    columns = ['Model', 'BLEU-1', 'BLEU-4', 'ROUGE-L', 'CIDEr', 'Perplexity', 'Status']

    rows = []
    for model in ['baseline', 'proposed', 'ablation']:
        if model in results:
            r = results[model]
            m = r.get('metrics', {})
            ppl = r.get('perplexity', 0)

            # Determine status
            if m.get('CIDEr', 0) < 0.001:
                status = 'COLLAPSED'
            elif model == 'baseline':
                status = 'BEST'
            else:
                status = 'OK'

            rows.append([
                model.replace('_', ' ').title(),
                f"{m.get('BLEU-1', 0):.3f}",
                f"{m.get('BLEU-4', 0):.4f}",
                f"{m.get('ROUGE-L', 0):.3f}",
                f"{m.get('CIDEr', 0):.4f}",
                f"{ppl:.1f}",
                status
            ])

    # Create table
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.18, 0.12, 0.12, 0.12, 0.12, 0.14, 0.12]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for i, col in enumerate(columns):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Color status column
    for i, row in enumerate(rows, 1):
        if row[-1] == 'COLLAPSED':
            table[(i, 6)].set_facecolor('#e74c3c')
            table[(i, 6)].set_text_props(color='white')
        elif row[-1] == 'BEST':
            table[(i, 6)].set_facecolor('#2ecc71')
            table[(i, 6)].set_text_props(color='white')

    ax.set_title('Modality-Adaptive Attention: Evaluation Results Summary',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved summary table to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation results")
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--output-dir", type=str, default="./visualizations")
    parser.add_argument("--investigate", action="store_true",
                       help="Investigate model outputs")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Specific checkpoint to investigate")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    print("Loading results...")
    results = load_results(args.results_dir)

    if not results:
        print("No results found!")
        return

    print(f"Found results for: {list(results.keys())}")

    # Create visualizations
    print("\nCreating visualizations...")
    plot_metrics_comparison(results, str(output_dir / "metrics_comparison.png"))
    plot_bleu_breakdown(results, str(output_dir / "bleu_breakdown.png"))
    create_summary_table(results, str(output_dir / "summary_table.png"))

    # Investigate specific model if requested
    if args.investigate:
        checkpoints = {
            'baseline': './checkpoints/baseline_uniform_softmax/checkpoint-28000',
            'ablation': './checkpoints/ablation_uniform_linear/checkpoint-28000',
            'proposed': './checkpoints/checkpoint-28000',
        }

        if args.checkpoint:
            investigate_model_outputs(args.checkpoint, "specified model")
        else:
            # Investigate ablation (collapsed) model
            print("\n" + "="*60)
            print("INVESTIGATING COLLAPSED ABLATION MODEL")
            print("="*60)
            investigate_model_outputs(
                checkpoints['ablation'],
                'Ablation (Uniform Linear)',
                num_samples=5
            )

    print(f"\nAll visualizations saved to: {output_dir}/")


if __name__ == "__main__":
    main()
