#!/usr/bin/env python3
"""
Compare attention patterns across different model variants.

This script loads the baseline (softmax), proposed (modality-adaptive),
and ablation (linear) models, runs inference on the same image, and
generates comparative visualizations of their attention patterns.

Usage:
    python scripts/compare_attention_variants.py --image path/to/image.jpg
    python scripts/compare_attention_variants.py  # Uses default sample image
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from transformers import GPT2Tokenizer

from src.models.vision_text_model import create_model
from src.utils.checkpointing import load_checkpoint
from src.data.transforms import get_image_transform
from src.data.collator import create_causal_mask


# Default checkpoint paths
CHECKPOINTS = {
    "baseline": "checkpoints/baseline_uniform_softmax/checkpoint-28000",
    "proposed": "checkpoints/proposed_modality_adaptive/checkpoint-28000",
    "ablation": "checkpoints/ablation_uniform_linear/checkpoint-28000",
}

# Display names for plots
DISPLAY_NAMES = {
    "baseline": "Baseline (Softmax)",
    "proposed": "Proposed (Modality-Adaptive)",
    "ablation": "Ablation (Linear)",
}


def load_model_variant(name: str, checkpoint_path: str, device: torch.device):
    """Load a model variant from checkpoint."""
    checkpoint_path = Path(checkpoint_path)

    # Load config
    config_path = checkpoint_path / "config.pt"
    if config_path.exists():
        config = torch.load(config_path, map_location="cpu")
    else:
        # Fallback config based on variant
        attention_types = {
            "baseline": "softmax",
            "proposed": "modality_adaptive",
            "ablation": "linear",
        }
        config = {
            "model": {
                "hidden_dim": 768,
                "num_heads": 12,
                "num_layers": 6,
                "attention_type": attention_types.get(name, "softmax"),
            }
        }

    model = create_model(config)
    load_checkpoint(checkpoint_path, model=model, device=device)
    model.eval()
    model.to(device)

    return model, config


def compute_attention_weights_manually(
    model,
    hidden_states: torch.Tensor,
    layer_idx: int,
    num_vision_tokens: int,
    device: torch.device,
):
    """
    Manually compute attention weights for visualization.
    Works for all attention types by directly computing Q @ K.T.
    """
    import math

    # Get the attention layer
    layer = model.transformer.layers[layer_idx]
    attn = layer.attention

    batch_size, seq_len, hidden_dim = hidden_states.shape
    num_heads = attn.num_heads
    head_dim = attn.head_dim
    scale = math.sqrt(head_dim)

    # Project to Q, K
    query = attn.q_proj(hidden_states)
    key = attn.k_proj(hidden_states)

    # Reshape for multi-head
    query = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    # Compute raw attention scores
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) / scale

    # Apply causal mask for text tokens
    text_len = seq_len - num_vision_tokens
    if text_len > 0:
        # Create causal mask: text can attend to vision + previous text
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        # Text-to-text region: causal
        text_to_text_mask = torch.tril(torch.ones(text_len, text_len, dtype=torch.bool, device=device))
        mask[num_vision_tokens:, num_vision_tokens:] = text_to_text_mask

        # Apply mask
        attn_scores = attn_scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    # Softmax to get attention weights (for visualization purposes)
    attn_weights = torch.softmax(attn_scores, dim=-1)

    return attn_weights


def get_attention_weights(
    model,
    image_tensor: torch.Tensor,
    tokenizer,
    device: torch.device,
    layer_idx: int = 5,
    max_tokens: int = 10,
):
    """
    Generate caption and extract attention weights.
    Computes attention weights manually for all model types.

    Returns:
        attention_weights: (num_heads, seq_len, seq_len)
        generated_text: str
        tokens: list of token strings
    """
    with torch.no_grad():
        # Get vision features
        vision_features = model.vision_encoder(image_tensor)
        vision_embeds = model.vision_projection(vision_features)

        # Initialize with BOS
        generated_ids = torch.tensor([[tokenizer.eos_token_id]], device=device)

        last_hidden_states = None

        for _ in range(max_tokens):
            text_embeds = model.text_embeddings(generated_ids)
            hidden_states = torch.cat([vision_embeds, text_embeds], dim=1)
            seq_len = hidden_states.size(1)

            # Add embeddings
            position_ids = torch.arange(seq_len, device=device)
            hidden_states = hidden_states + model.position_embeddings(position_ids)

            modality_mask = torch.zeros(1, seq_len, dtype=torch.long, device=device)
            modality_mask[:, model.num_vision_tokens:] = 1
            hidden_states = hidden_states + model.modality_embeddings(modality_mask)

            # Store for attention computation
            last_hidden_states = hidden_states.clone()

            # Get attention
            causal_mask = create_causal_mask(seq_len, model.num_vision_tokens, device)

            hidden_states, _, _ = model.transformer(
                hidden_states,
                modality_mask=modality_mask,
                causal_mask=causal_mask,
                num_vision_tokens=model.num_vision_tokens,
                output_attentions=False,  # Don't need returned attention
            )

            # Get next token
            logits = model.lm_head(hidden_states[:, -1:, :])
            next_token = logits.argmax(dim=-1)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode caption
    caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    tokens = tokenizer.tokenize(caption)

    # Compute attention weights manually (works for all model types)
    attn_weights = None
    if last_hidden_states is not None:
        attn_weights = compute_attention_weights_manually(
            model, last_hidden_states, layer_idx,
            model.num_vision_tokens, device
        )

    return attn_weights, caption, tokens


def visualize_attention_comparison(
    attentions: dict,
    captions: dict,
    num_vision_tokens: int = 197,
    head_idx: int = 0,
    save_path: str = None,
):
    """
    Create a 3x3 comparison grid of attention patterns.

    Rows: Baseline, Proposed, Ablation
    Cols: Vision-to-Vision, Text-to-Vision, Text-to-Text
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    model_order = ["baseline", "proposed", "ablation"]

    for row, name in enumerate(model_order):
        attn = attentions.get(name)
        caption = captions.get(name, "")

        if attn is None:
            for col in range(3):
                axes[row, col].text(0.5, 0.5, "No attention data",
                                   ha='center', va='center', fontsize=12)
                axes[row, col].set_title(f"{DISPLAY_NAMES[name]}")
            continue

        # Handle batch dimension
        if attn.dim() == 4:
            attn = attn[0]

        attn = attn[head_idx].detach().cpu().numpy()
        seq_len = attn.shape[0]
        text_len = seq_len - num_vision_tokens

        # Vision-to-Vision
        v2v = attn[:num_vision_tokens, :num_vision_tokens]
        im = axes[row, 0].imshow(v2v, cmap='viridis', aspect='auto')
        if row == 0:
            axes[row, 0].set_title("Vision → Vision", fontsize=12, fontweight='bold')
        axes[row, 0].set_ylabel(f"{DISPLAY_NAMES[name]}", fontsize=10, fontweight='bold')
        plt.colorbar(im, ax=axes[row, 0], fraction=0.046, pad=0.04)

        # Text-to-Vision (cross-modal)
        if text_len > 0:
            t2v = attn[num_vision_tokens:, :num_vision_tokens]
            im = axes[row, 1].imshow(t2v, cmap='viridis', aspect='auto')
        else:
            axes[row, 1].text(0.5, 0.5, "No text tokens", ha='center', va='center')
        if row == 0:
            axes[row, 1].set_title("Text → Vision", fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.04)

        # Text-to-Text (causal)
        if text_len > 0:
            t2t = attn[num_vision_tokens:, num_vision_tokens:]
            # Apply causal mask for visualization
            mask = np.triu(np.ones_like(t2t), k=1)
            t2t_masked = np.ma.masked_array(t2t, mask=mask)
            im = axes[row, 2].imshow(t2t_masked, cmap='viridis', aspect='auto')
        else:
            axes[row, 2].text(0.5, 0.5, "No text tokens", ha='center', va='center')
        if row == 0:
            axes[row, 2].set_title("Text → Text (Causal)", fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)

    plt.suptitle("Attention Pattern Comparison Across Model Variants\n" +
                 f"(Head {head_idx}, Layer 5)", fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


def visualize_attention_statistics(
    attentions: dict,
    num_vision_tokens: int = 197,
    save_path: str = None,
):
    """
    Visualize attention statistics (entropy, sparsity) across variants.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    stats = {}
    model_order = ["baseline", "proposed", "ablation"]

    for name in model_order:
        attn = attentions.get(name)
        if attn is None:
            continue

        if attn.dim() == 4:
            attn = attn[0]

        attn = attn.detach().cpu().numpy()

        # Compute statistics across all heads
        v2v = attn[:, :num_vision_tokens, :num_vision_tokens]
        t2v = attn[:, num_vision_tokens:, :num_vision_tokens]
        t2t = attn[:, num_vision_tokens:, num_vision_tokens:]

        # Entropy (measure of attention distribution)
        def entropy(p):
            p = np.clip(p, 1e-10, 1)
            return -np.sum(p * np.log(p), axis=-1)

        stats[name] = {
            'v2v_entropy': entropy(v2v).mean(),
            't2v_entropy': entropy(t2v).mean() if t2v.size > 0 else 0,
            't2t_entropy': entropy(t2t).mean() if t2t.size > 0 else 0,
            'v2v_max': v2v.max(axis=-1).mean(),
            't2v_max': t2v.max(axis=-1).mean() if t2v.size > 0 else 0,
            't2t_max': t2t.max(axis=-1).mean() if t2t.size > 0 else 0,
        }

    # Plot entropy comparison
    x = np.arange(3)
    width = 0.25

    colors = {'baseline': '#2ecc71', 'proposed': '#3498db', 'ablation': '#e74c3c'}

    for i, name in enumerate(model_order):
        if name not in stats:
            continue
        s = stats[name]
        axes[0, 0].bar(x[0] + i*width, s['v2v_entropy'], width, label=DISPLAY_NAMES[name], color=colors[name])
        axes[0, 0].bar(x[1] + i*width, s['t2v_entropy'], width, color=colors[name])
        axes[0, 0].bar(x[2] + i*width, s['t2t_entropy'], width, color=colors[name])

    axes[0, 0].set_xticks(x + width)
    axes[0, 0].set_xticklabels(['V→V', 'T→V', 'T→T'])
    axes[0, 0].set_ylabel('Entropy')
    axes[0, 0].set_title('Attention Entropy by Region\n(Higher = More Distributed)')
    axes[0, 0].legend()

    # Plot max attention
    for i, name in enumerate(model_order):
        if name not in stats:
            continue
        s = stats[name]
        axes[0, 1].bar(x[0] + i*width, s['v2v_max'], width, label=DISPLAY_NAMES[name], color=colors[name])
        axes[0, 1].bar(x[1] + i*width, s['t2v_max'], width, color=colors[name])
        axes[0, 1].bar(x[2] + i*width, s['t2t_max'], width, color=colors[name])

    axes[0, 1].set_xticks(x + width)
    axes[0, 1].set_xticklabels(['V→V', 'T→V', 'T→T'])
    axes[0, 1].set_ylabel('Max Attention Weight')
    axes[0, 1].set_title('Peak Attention by Region\n(Higher = More Focused)')
    axes[0, 1].legend()

    # Attention distribution histogram for V2V
    for name in model_order:
        attn = attentions.get(name)
        if attn is None:
            continue
        if attn.dim() == 4:
            attn = attn[0]
        v2v = attn[:, :num_vision_tokens, :num_vision_tokens].detach().cpu().numpy().flatten()
        axes[1, 0].hist(v2v, bins=50, alpha=0.5, label=DISPLAY_NAMES[name], color=colors[name])

    axes[1, 0].set_xlabel('Attention Weight')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Vision-to-Vision Attention Distribution')
    axes[1, 0].legend()
    axes[1, 0].set_yscale('log')

    # Head-wise entropy for baseline vs proposed
    head_entropies = {}
    for name in ['baseline', 'proposed']:
        attn = attentions.get(name)
        if attn is None:
            continue
        if attn.dim() == 4:
            attn = attn[0]

        def entropy(p):
            p = np.clip(p, 1e-10, 1)
            return -np.sum(p * np.log(p), axis=-1)

        # Per-head entropy for V2V
        v2v = attn[:, :num_vision_tokens, :num_vision_tokens].detach().cpu().numpy()
        head_entropies[name] = [entropy(v2v[h]).mean() for h in range(v2v.shape[0])]

    if 'baseline' in head_entropies and 'proposed' in head_entropies:
        heads = np.arange(len(head_entropies['baseline']))
        axes[1, 1].plot(heads, head_entropies['baseline'], 'o-', label='Baseline', color=colors['baseline'])
        axes[1, 1].plot(heads, head_entropies['proposed'], 's-', label='Proposed', color=colors['proposed'])
        axes[1, 1].set_xlabel('Head Index')
        axes[1, 1].set_ylabel('Entropy')
        axes[1, 1].set_title('Per-Head V2V Attention Entropy')
        axes[1, 1].legend()

    plt.suptitle('Attention Statistics Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


def visualize_cross_modal_attention(
    attentions: dict,
    image: np.ndarray,
    num_vision_tokens: int = 197,
    text_token_idx: int = 0,
    save_path: str = None,
):
    """
    Visualize how each model variant attends to image regions for a specific text token.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    model_order = ["baseline", "proposed", "ablation"]
    grid_size = int(np.sqrt(num_vision_tokens - 1))  # -1 for CLS token

    for idx, name in enumerate(model_order):
        attn = attentions.get(name)
        if attn is None:
            axes[idx + 1].text(0.5, 0.5, "No attention data", ha='center', va='center')
            axes[idx + 1].set_title(DISPLAY_NAMES[name])
            continue

        if attn.dim() == 4:
            attn = attn[0]

        # Average across heads
        attn = attn.mean(0).detach().cpu().numpy()

        # Get attention from text token to vision patches (excluding CLS)
        text_to_vision = attn[num_vision_tokens + text_token_idx, 1:num_vision_tokens]

        if len(text_to_vision) == grid_size * grid_size:
            attn_map = text_to_vision.reshape(grid_size, grid_size)
        else:
            # Fallback
            attn_map = text_to_vision[:grid_size*grid_size].reshape(grid_size, grid_size)

        # Resize to image size
        from PIL import Image as PILImage
        attn_pil = PILImage.fromarray(attn_map.astype(np.float32))
        attn_resized = np.array(attn_pil.resize((image.shape[1], image.shape[0]), PILImage.BILINEAR))

        # Normalize
        attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)

        # Overlay on image
        axes[idx + 1].imshow(image)
        axes[idx + 1].imshow(attn_resized, cmap='jet', alpha=0.5)
        axes[idx + 1].set_title(f"{DISPLAY_NAMES[name]}", fontsize=11, fontweight='bold')
        axes[idx + 1].axis('off')

    plt.suptitle(f"Cross-Modal Attention: Where Text Token {text_token_idx} Looks in the Image",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Compare Attention Patterns Across Model Variants")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image (uses sample image if not specified)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./visualizations",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=5,
        help="Layer index to visualize (0-indexed)",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=0,
        help="Head index to visualize",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load image
    if args.image:
        image_path = args.image
    else:
        # Use matplotlib sample image
        import matplotlib
        sample_data_dir = Path(matplotlib.__file__).parent / "mpl-data" / "sample_data"
        image_path = sample_data_dir / "grace_hopper.jpg"
        print(f"Using sample image: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image) / 255.0

    transform = get_image_transform(is_train=False)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Load models and get attention
    attentions = {}
    captions = {}

    for name, checkpoint_path in CHECKPOINTS.items():
        print(f"\nLoading {DISPLAY_NAMES[name]}...")
        try:
            model, config = load_model_variant(name, checkpoint_path, device)
            attn, caption, tokens = get_attention_weights(
                model, image_tensor, tokenizer, device, layer_idx=args.layer
            )
            attentions[name] = attn
            captions[name] = caption
            print(f"  Caption: {caption}")
        except Exception as e:
            print(f"  Error loading {name}: {e}")
            attentions[name] = None
            captions[name] = ""

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Attention pattern comparison grid
    visualize_attention_comparison(
        attentions, captions,
        num_vision_tokens=197,
        head_idx=args.head,
        save_path=output_dir / "attention_comparison.png",
    )

    # 2. Attention statistics
    visualize_attention_statistics(
        attentions,
        num_vision_tokens=197,
        save_path=output_dir / "attention_statistics.png",
    )

    # 3. Cross-modal attention overlay
    visualize_cross_modal_attention(
        attentions, image_np,
        num_vision_tokens=197,
        text_token_idx=0,
        save_path=output_dir / "cross_modal_attention.png",
    )

    # Save captions
    with open(output_dir / "generated_captions.txt", "w") as f:
        for name, caption in captions.items():
            f.write(f"{DISPLAY_NAMES[name]}:\n{caption}\n\n")

    print(f"\nVisualization complete! Results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - attention_comparison.png: 3x3 grid comparing attention patterns")
    print("  - attention_statistics.png: Entropy and distribution analysis")
    print("  - cross_modal_attention.png: Attention overlay on image")
    print("  - generated_captions.txt: Captions from each model")


if __name__ == "__main__":
    main()
