"""Attention visualization utilities."""

from typing import Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_attention(
    attention_weights: torch.Tensor,
    image: Optional[np.ndarray] = None,
    tokens: Optional[List[str]] = None,
    layer_idx: int = 0,
    head_idx: int = 0,
    num_vision_tokens: int = 197,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Visualize attention patterns.

    Args:
        attention_weights: Attention weights (num_heads, seq_len, seq_len) or
                          (batch, num_heads, seq_len, seq_len)
        image: Original image for overlaying attention (H, W, 3)
        tokens: Token strings for labeling
        layer_idx: Layer index (for title)
        head_idx: Head index to visualize
        num_vision_tokens: Number of vision tokens
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    # Handle batch dimension
    if attention_weights.dim() == 4:
        attention_weights = attention_weights[0]  # Take first sample

    # Get specific head
    attn = attention_weights[head_idx].cpu().numpy()
    seq_len = attn.shape[0]
    text_len = seq_len - num_vision_tokens

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Vision-to-Vision attention (top-left)
    ax = axes[0, 0]
    v2v = attn[:num_vision_tokens, :num_vision_tokens]
    sns.heatmap(v2v, ax=ax, cmap="viridis", cbar=True)
    ax.set_title(f"Vision → Vision (Layer {layer_idx}, Head {head_idx})")
    ax.set_xlabel("Key (Vision)")
    ax.set_ylabel("Query (Vision)")

    # 2. Text-to-Vision attention (top-right)
    ax = axes[0, 1]
    t2v = attn[num_vision_tokens:, :num_vision_tokens]
    sns.heatmap(t2v, ax=ax, cmap="viridis", cbar=True)
    ax.set_title("Text → Vision (Cross-Modal)")
    ax.set_xlabel("Key (Vision)")
    ax.set_ylabel("Query (Text)")
    if tokens:
        ax.set_yticklabels(tokens[-text_len:], rotation=0, fontsize=8)

    # 3. Text-to-Text attention (bottom-left)
    ax = axes[1, 0]
    t2t = attn[num_vision_tokens:, num_vision_tokens:]
    sns.heatmap(t2t, ax=ax, cmap="viridis", cbar=True, mask=np.triu(np.ones_like(t2t), k=1))
    ax.set_title("Text → Text (Causal)")
    ax.set_xlabel("Key (Text)")
    ax.set_ylabel("Query (Text)")
    if tokens:
        ax.set_xticklabels(tokens[-text_len:], rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(tokens[-text_len:], rotation=0, fontsize=8)

    # 4. Full attention matrix (bottom-right)
    ax = axes[1, 1]
    sns.heatmap(attn, ax=ax, cmap="viridis", cbar=True)
    ax.set_title("Full Attention Matrix")
    ax.axhline(y=num_vision_tokens, color="red", linestyle="--", linewidth=1)
    ax.axvline(x=num_vision_tokens, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Key")
    ax.set_ylabel("Query")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_attention_on_image(
    attention_weights: torch.Tensor,
    image: np.ndarray,
    text_token_idx: int,
    num_vision_tokens: int = 196,  # Excluding CLS
    patch_size: int = 14,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Overlay attention weights on image.

    Shows which image patches a specific text token attends to.

    Args:
        attention_weights: Attention weights (num_heads, seq_len, seq_len)
        image: Original image (H, W, 3), values in [0, 1]
        text_token_idx: Index of text token to visualize
        num_vision_tokens: Number of vision tokens (excluding CLS)
        patch_size: Image patch size (14 for ViT-B/16)
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    # Average across heads
    if attention_weights.dim() == 3:
        attn = attention_weights.mean(0).cpu().numpy()
    else:
        attn = attention_weights.cpu().numpy()

    # Get attention from text token to vision tokens (excluding CLS)
    # Assuming vision tokens are [CLS, patch_0, patch_1, ...]
    text_to_vision = attn[num_vision_tokens + 1 + text_token_idx, 1:num_vision_tokens + 1]

    # Reshape to 2D grid
    grid_size = int(np.sqrt(num_vision_tokens))
    attn_map = text_to_vision.reshape(grid_size, grid_size)

    # Resize attention map to image size
    from PIL import Image as PILImage
    attn_map_pil = PILImage.fromarray(attn_map)
    attn_map_resized = np.array(
        attn_map_pil.resize((image.shape[1], image.shape[0]), PILImage.BILINEAR)
    )

    # Normalize
    attn_map_resized = (attn_map_resized - attn_map_resized.min()) / (
        attn_map_resized.max() - attn_map_resized.min() + 1e-8
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Attention map
    axes[1].imshow(attn_map_resized, cmap="hot")
    axes[1].set_title(f"Attention (Token {text_token_idx})")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(attn_map_resized, cmap="jet", alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def compare_attention_patterns(
    attn_softmax: torch.Tensor,
    attn_linear: torch.Tensor,
    attn_adaptive: torch.Tensor,
    num_vision_tokens: int = 197,
    head_idx: int = 0,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10),
) -> plt.Figure:
    """
    Compare attention patterns across model variants.

    Args:
        attn_softmax: Attention from baseline model
        attn_linear: Attention from ablation model
        attn_adaptive: Attention from proposed model
        num_vision_tokens: Number of vision tokens
        head_idx: Head index to visualize
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(3, 3, figsize=figsize)

    models = [
        ("Baseline (Softmax)", attn_softmax),
        ("Proposed (Adaptive)", attn_adaptive),
        ("Ablation (Linear)", attn_linear),
    ]

    for row, (name, attn) in enumerate(models):
        if attn.dim() == 4:
            attn = attn[0]
        attn = attn[head_idx].cpu().numpy()

        # Vision-to-Vision
        v2v = attn[:num_vision_tokens, :num_vision_tokens]
        sns.heatmap(v2v, ax=axes[row, 0], cmap="viridis", cbar=True)
        axes[row, 0].set_title(f"{name}\nVision → Vision")

        # Text-to-Vision
        t2v = attn[num_vision_tokens:, :num_vision_tokens]
        sns.heatmap(t2v, ax=axes[row, 1], cmap="viridis", cbar=True)
        axes[row, 1].set_title("Text → Vision")

        # Text-to-Text
        t2t = attn[num_vision_tokens:, num_vision_tokens:]
        mask = np.triu(np.ones_like(t2t), k=1)
        sns.heatmap(t2t, ax=axes[row, 2], cmap="viridis", cbar=True, mask=mask)
        axes[row, 2].set_title("Text → Text")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
