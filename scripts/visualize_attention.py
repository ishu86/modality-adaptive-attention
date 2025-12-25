#!/usr/bin/env python3
"""Visualize attention patterns from trained models."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
from transformers import GPT2Tokenizer

from src.models.vision_text_model import create_model
from src.utils.checkpointing import load_checkpoint
from src.evaluation.visualization import (
    visualize_attention,
    visualize_attention_on_image,
    compare_attention_patterns,
)
from src.data.transforms import get_image_transform, denormalize_image


def load_model(checkpoint_path: str, device: str):
    """Load model from checkpoint."""
    checkpoint_path = Path(checkpoint_path)

    # Load config
    config_path = checkpoint_path / "config.pt" if checkpoint_path.is_dir() else checkpoint_path.parent / "config.pt"
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

    model = create_model(config)
    load_checkpoint(checkpoint_path, model=model, device=device)
    model.eval()

    return model, config


def main():
    parser = argparse.ArgumentParser(description="Visualize Attention Patterns")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./attention_viz",
        help="Output directory",
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

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    model = model.to(device)

    # Load and preprocess image
    print(f"Processing image: {args.image}")
    image = Image.open(args.image).convert("RGB")
    transform = get_image_transform(is_train=False)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Tokenizer for caption
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Generate caption with attention
    print("Generating caption...")
    with torch.no_grad():
        # Get vision features
        vision_features = model.vision_encoder(image_tensor)
        vision_embeds = model.vision_projection(vision_features)

        # Initialize with BOS
        generated_ids = torch.tensor([[tokenizer.eos_token_id]], device=device)

        all_attentions = []

        for _ in range(30):
            text_embeds = model.text_embeddings(generated_ids)
            hidden_states = torch.cat([vision_embeds, text_embeds], dim=1)
            seq_len = hidden_states.size(1)

            # Add embeddings
            position_ids = torch.arange(seq_len, device=device)
            hidden_states = hidden_states + model.position_embeddings(position_ids)

            modality_mask = torch.zeros(1, seq_len, dtype=torch.long, device=device)
            modality_mask[:, model.num_vision_tokens:] = 1
            hidden_states = hidden_states + model.modality_embeddings(modality_mask)

            # Get attention
            from src.data.collator import create_causal_mask
            causal_mask = create_causal_mask(seq_len, model.num_vision_tokens, device)

            hidden_states, _, attentions = model.transformer(
                hidden_states,
                modality_mask=modality_mask,
                causal_mask=causal_mask,
                num_vision_tokens=model.num_vision_tokens,
                output_attentions=True,
            )

            if attentions and attentions[args.layer] is not None:
                all_attentions.append(attentions[args.layer])

            # Get next token
            logits = model.lm_head(hidden_states[:, -1:, :])
            next_token = logits.argmax(dim=-1)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode caption
    caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Generated caption: {caption}")

    # Get tokens for visualization
    tokens = tokenizer.tokenize(caption)

    # Visualize attention from last generation step
    if all_attentions:
        last_attn = all_attentions[-1]

        # Create attention visualization
        print("Creating attention visualization...")
        fig = visualize_attention(
            last_attn,
            tokens=tokens,
            layer_idx=args.layer,
            head_idx=args.head,
            num_vision_tokens=model.num_vision_tokens,
        )
        fig.savefig(output_dir / "attention_patterns.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {output_dir / 'attention_patterns.png'}")

        # Visualize attention on image for each text token
        image_np = np.array(image) / 255.0

        for i, token in enumerate(tokens[:5]):  # First 5 tokens
            fig = visualize_attention_on_image(
                last_attn,
                image_np,
                text_token_idx=i,
                num_vision_tokens=model.num_vision_tokens - 1,  # Exclude CLS
            )
            fig.savefig(
                output_dir / f"attention_token_{i}_{token.strip()}.png",
                dpi=150,
                bbox_inches="tight",
            )

        print(f"Saved attention overlays for first 5 tokens")

    # Save caption
    with open(output_dir / "caption.txt", "w") as f:
        f.write(caption)

    print(f"\nVisualization complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
