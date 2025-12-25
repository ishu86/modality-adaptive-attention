"""Full Vision-Text Model for Image Captioning."""

from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_encoder import VisionEncoder, VisionProjection
from .transformer import TransformerEncoder
from ..data.collator import create_causal_mask


class VisionTextModel(nn.Module):
    """
    Vision-Text Transformer for Image Captioning.

    Architecture:
    1. Vision Encoder (Frozen CLIP ViT-B/16)
    2. Vision Projection (learnable)
    3. Text Embeddings (learnable)
    4. Positional Embeddings (learnable)
    5. Transformer Encoder (configurable attention)
    6. Language Modeling Head (weight-tied with text embeddings)

    The model is trained with next-token prediction on captions.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 6,
        ffn_dim: int = 3072,
        vocab_size: int = 50257,  # GPT2 tokenizer
        max_seq_length: int = 273,  # 197 vision + 76 text
        dropout: float = 0.1,
        attention_type: str = "softmax",
        vision_model_name: str = "openai/clip-vit-base-patch16",
        freeze_vision: bool = True,
        attention_kwargs: dict = None,
    ):
        """
        Initialize Vision-Text model.

        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ffn_dim: FFN intermediate dimension
            vocab_size: Vocabulary size
            max_seq_length: Maximum sequence length (vision + text)
            dropout: Dropout rate
            attention_type: "softmax", "linear", "modality_adaptive",
                          "modality_adaptive_v2", or "progressive"
            vision_model_name: CLIP model name
            freeze_vision: Whether to freeze vision encoder
            attention_kwargs: Additional kwargs for attention modules (V2 tuning)
        """
        super().__init__()
        self.attention_kwargs = attention_kwargs or {}

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.attention_type = attention_type

        # Vision encoder (frozen CLIP)
        self.vision_encoder = VisionEncoder(
            model_name=vision_model_name,
            freeze=freeze_vision,
        )
        self.num_vision_tokens = self.vision_encoder.num_tokens  # 197

        # Vision projection
        self.vision_projection = VisionProjection(
            vision_dim=self.vision_encoder.hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Text embeddings
        self.text_embeddings = nn.Embedding(vocab_size, hidden_dim)

        # Positional embeddings
        self.position_embeddings = nn.Embedding(max_seq_length, hidden_dim)

        # Modality embeddings (to distinguish vision from text)
        self.modality_embeddings = nn.Embedding(2, hidden_dim)  # 0: vision, 1: text

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Transformer
        self.transformer = TransformerEncoder(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            attention_type=attention_type,
            attention_kwargs=self.attention_kwargs,
        )

        # Language modeling head (weight-tied with text embeddings)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.text_embeddings.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize learnable weights."""
        # Text embeddings
        nn.init.normal_(self.text_embeddings.weight, std=0.02)

        # Positional embeddings
        nn.init.normal_(self.position_embeddings.weight, std=0.02)

        # Modality embeddings
        nn.init.normal_(self.modality_embeddings.weight, std=0.02)

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Get number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        vision_features: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        modality_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Text token IDs (batch, text_len)
            vision_features: Pre-computed CLIP features (batch, num_vision_tokens, hidden_dim)
            images: Raw images (batch, 3, 224, 224) - used if vision_features is None
            attention_mask: Full attention mask (batch, seq_len)
            modality_mask: 0 for vision, 1 for text (batch, seq_len)
            labels: Target token IDs for LM loss (batch, text_len)
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states

        Returns:
            Dictionary with:
                - loss: Language modeling loss (if labels provided)
                - logits: Predicted logits (batch, text_len, vocab_size)
                - hidden_states: All hidden states (if requested)
                - attentions: All attention weights (if requested)
        """
        batch_size = input_ids.size(0)
        text_len = input_ids.size(1)

        # Get vision features
        if vision_features is None:
            if images is None:
                raise ValueError("Either vision_features or images must be provided")
            vision_features = self.vision_encoder(images)

        # Project vision features
        vision_embeds = self.vision_projection(vision_features)

        # Get text embeddings
        text_embeds = self.text_embeddings(input_ids)

        # Concatenate vision and text embeddings
        # Shape: (batch, num_vision_tokens + text_len, hidden_dim)
        hidden_states = torch.cat([vision_embeds, text_embeds], dim=1)
        seq_len = hidden_states.size(1)

        # Add positional embeddings
        position_ids = torch.arange(seq_len, device=hidden_states.device)
        position_embeds = self.position_embeddings(position_ids)
        hidden_states = hidden_states + position_embeds

        # Add modality embeddings
        if modality_mask is None:
            modality_mask = torch.zeros(batch_size, seq_len, dtype=torch.long, device=hidden_states.device)
            modality_mask[:, self.num_vision_tokens:] = 1

        modality_embeds = self.modality_embeddings(modality_mask)
        hidden_states = hidden_states + modality_embeds

        # Dropout
        hidden_states = self.dropout(hidden_states)

        # Create causal mask
        causal_mask = create_causal_mask(
            seq_len=seq_len,
            num_vision_tokens=self.num_vision_tokens,
            device=hidden_states.device,
        )

        # Transform
        hidden_states, all_hidden_states, all_attentions = self.transformer(
            hidden_states,
            attention_mask=attention_mask,
            modality_mask=modality_mask,
            causal_mask=causal_mask,
            num_vision_tokens=self.num_vision_tokens,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # Get text portion for LM head
        text_hidden_states = hidden_states[:, self.num_vision_tokens:, :]

        # Language modeling head
        logits = self.lm_head(text_hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Flatten
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
        }

    @torch.no_grad()
    def generate(
        self,
        vision_features: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        max_length: int = 30,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = False,
        eos_token_id: int = 50256,
        pad_token_id: int = 50256,
    ) -> torch.Tensor:
        """
        Generate captions for images.

        Args:
            vision_features: Pre-computed CLIP features
            images: Raw images
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample or use greedy decoding
            eos_token_id: End-of-sequence token ID
            pad_token_id: Padding token ID

        Returns:
            generated_ids: Generated token IDs (batch, max_length)
        """
        # Get vision features
        if vision_features is None:
            if images is None:
                raise ValueError("Either vision_features or images must be provided")
            vision_features = self.vision_encoder(images)

        batch_size = vision_features.size(0)
        device = vision_features.device

        # Project vision features
        vision_embeds = self.vision_projection(vision_features)

        # Initialize with BOS token (using EOS as BOS for GPT2)
        generated_ids = torch.full(
            (batch_size, 1),
            eos_token_id,
            dtype=torch.long,
            device=device,
        )

        # Track which sequences are done
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length - 1):
            # Get text embeddings for generated tokens
            text_embeds = self.text_embeddings(generated_ids)

            # Concatenate vision and text
            hidden_states = torch.cat([vision_embeds, text_embeds], dim=1)
            seq_len = hidden_states.size(1)

            # Add positional embeddings
            position_ids = torch.arange(seq_len, device=device)
            position_embeds = self.position_embeddings(position_ids)
            hidden_states = hidden_states + position_embeds

            # Add modality embeddings
            modality_mask = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
            modality_mask[:, self.num_vision_tokens:] = 1
            modality_embeds = self.modality_embeddings(modality_mask)
            hidden_states = hidden_states + modality_embeds

            # Create causal mask
            causal_mask = create_causal_mask(
                seq_len=seq_len,
                num_vision_tokens=self.num_vision_tokens,
                device=device,
            )

            # Transform
            hidden_states, _, _ = self.transformer(
                hidden_states,
                modality_mask=modality_mask,
                causal_mask=causal_mask,
                num_vision_tokens=self.num_vision_tokens,
            )

            # Get logits for last token
            last_hidden = hidden_states[:, -1, :]
            logits = self.lm_head(last_hidden)

            # Apply temperature
            logits = logits / temperature

            if do_sample:
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')

                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = logits.argmax(dim=-1, keepdim=True)

            # Update generated ids
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Check for EOS
            unfinished = unfinished & (next_token.squeeze(-1) != eos_token_id)

            if not unfinished.any():
                break

        return generated_ids


def create_model(config: Dict[str, Any]) -> VisionTextModel:
    """
    Create model from config dictionary.

    Args:
        config: Model configuration

    Returns:
        VisionTextModel instance
    """
    model_config = config.get("model", {})
    vision_config = config.get("vision", {})
    attention_config = config.get("attention", {})

    return VisionTextModel(
        hidden_dim=model_config.get("hidden_dim", 768),
        num_heads=model_config.get("num_heads", 12),
        num_layers=model_config.get("num_layers", 6),
        ffn_dim=model_config.get("ffn_dim", 3072),
        vocab_size=model_config.get("vocab_size", 50257),
        max_seq_length=model_config.get("max_seq_length", 273),
        dropout=model_config.get("dropout", 0.1),
        attention_type=model_config.get("attention_type", "softmax"),
        vision_model_name=vision_config.get("model_name", "openai/clip-vit-base-patch16"),
        freeze_vision=vision_config.get("freeze", True),
        attention_kwargs=attention_config,
    )
