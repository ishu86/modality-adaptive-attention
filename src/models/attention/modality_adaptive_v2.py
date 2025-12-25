"""Modality-Adaptive Attention V2: Improved version with tuning mechanisms."""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear_attention import elu_feature_map


class ModalityAdaptiveAttentionV2(nn.Module):
    """
    Improved Modality-Adaptive Attention with tuning mechanisms.

    Key improvements over V1:
    1. Temperature scaling for linear attention (learnable sharpness)
    2. Gated mixing between linear and softmax (adaptive blending)
    3. Layer-aware behavior (more softmax in early layers)
    4. Entropy regularization support

    Design philosophy:
    - Vision-to-Vision: Primarily linear (efficiency) with softmax residual (sharpness)
    - Text-to-Vision: Full softmax (grounding quality)
    - Text-to-Text: Full softmax (generation quality)
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        bias: bool = True,
        eps: float = 1e-6,
        # V2 tuning parameters
        linear_temperature: float = 1.0,
        use_gated_mixing: bool = True,
        initial_softmax_ratio: float = 0.3,  # Start with 30% softmax for vision
        layer_idx: int = 0,
        num_layers: int = 6,
    ):
        """
        Initialize improved modality-adaptive attention.

        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            bias: Whether to use bias
            eps: Small constant for numerical stability
            linear_temperature: Temperature for linear attention scaling
            use_gated_mixing: Whether to use gated mixing of linear+softmax
            initial_softmax_ratio: Initial ratio of softmax attention for vision
            layer_idx: Current layer index (for layer-wise scheduling)
            num_layers: Total number of layers
        """
        super().__init__()

        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.eps = eps
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        self.use_gated_mixing = use_gated_mixing

        # Shared Q, K, V projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # V2: Learnable temperature for linear attention
        self.linear_temperature = nn.Parameter(torch.tensor(linear_temperature))

        # V2: Gated mixing - learnable gate to blend linear and softmax for vision
        if use_gated_mixing:
            # Layer-wise schedule: early layers use more softmax
            # softmax_ratio decreases as we go deeper (more linear in later layers)
            layer_ratio = layer_idx / max(1, num_layers - 1)
            init_gate = initial_softmax_ratio * (1 - 0.5 * layer_ratio)

            self.vision_gate = nn.Parameter(torch.tensor(init_gate))
            # Per-head gates for more flexibility
            self.head_gates = nn.Parameter(torch.full((num_heads,), init_gate))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        modality_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        num_vision_tokens: int = 197,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with improved modality-adaptive attention.
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Split into vision and text portions
        vision_query = query[:, :, :num_vision_tokens, :]
        vision_key = key[:, :, :num_vision_tokens, :]
        vision_value = value[:, :, :num_vision_tokens, :]

        text_query = query[:, :, num_vision_tokens:, :]
        text_key = key[:, :, num_vision_tokens:, :]
        text_value = value[:, :, num_vision_tokens:, :]

        # ============================================
        # 1. Vision-to-Vision: Gated Linear + Softmax
        # ============================================
        vision_output = self._vision_attention(
            vision_query, vision_key, vision_value
        )

        # ============================================
        # 2. Text Attention (to both vision and text)
        # ============================================
        text_output = self._text_attention(
            text_query,
            key,  # Full key (vision + text)
            value,  # Full value (vision + text)
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            num_vision_tokens=num_vision_tokens,
        )

        # ============================================
        # 3. Combine outputs
        # ============================================
        attn_output = torch.cat([vision_output, text_output], dim=2)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )

        # Output projection
        output = self.out_proj(attn_output)
        output = self.proj_dropout(output)

        return output, None

    def _vision_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gated attention for vision tokens: blend of linear and softmax.

        Linear attention is efficient but over-smooths.
        Softmax attention is sharp but O(n^2).
        Gating allows the model to learn the optimal blend.
        """
        if self.use_gated_mixing:
            # Compute both attention types
            linear_out = self._linear_attention(query, key, value)
            softmax_out = self._softmax_attention_vision(query, key, value)

            # Gated mixing with per-head gates
            # gate: 0 = all linear, 1 = all softmax
            gate = torch.sigmoid(self.head_gates).view(1, -1, 1, 1)

            return gate * softmax_out + (1 - gate) * linear_out
        else:
            return self._linear_attention(query, key, value)

    def _linear_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Linear attention with temperature scaling.
        """
        # Apply ELU+1 feature map
        query_feat = elu_feature_map(query)
        key_feat = elu_feature_map(key)

        # Temperature scaling (learnable sharpness)
        # Higher temperature = sharper attention
        temp = torch.clamp(self.linear_temperature, min=0.1, max=10.0)
        query_feat = query_feat * temp

        # K'^T @ V: (batch, num_heads, head_dim, head_dim)
        kv = torch.einsum('bhnd,bhnv->bhdv', key_feat, value)

        # Q' @ (K'^T @ V): (batch, num_heads, vision_len, head_dim)
        numerator = torch.einsum('bhnd,bhdv->bhnv', query_feat, kv)

        # Normalization
        key_sum = key_feat.sum(dim=2)  # (batch, num_heads, head_dim)
        denominator = torch.einsum('bhnd,bhd->bhn', query_feat, key_sum)
        denominator = denominator.unsqueeze(-1) + self.eps

        return numerator / denominator

    def _softmax_attention_vision(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Standard softmax attention for vision tokens.
        """
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        return torch.matmul(attn_weights, value)

    def _text_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        causal_mask: Optional[torch.Tensor],
        num_vision_tokens: int,
    ) -> torch.Tensor:
        """
        Softmax attention for text tokens (unchanged from V1).
        Text needs sharp attention for good generation.
        """
        batch_size = query.size(0)
        text_len = query.size(2)
        seq_len = key.size(2)

        # Compute attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        # Create attention mask for text queries
        text_attn_mask = torch.ones(text_len, seq_len, dtype=torch.bool, device=query.device)

        # Text can attend to all vision tokens
        text_attn_mask[:, :num_vision_tokens] = True

        # Text-to-text: causal (lower triangular)
        text_to_text_mask = torch.tril(
            torch.ones(text_len, text_len, dtype=torch.bool, device=query.device)
        )
        text_attn_mask[:, num_vision_tokens:] = text_to_text_mask

        # Apply to attention scores
        text_attn_mask = text_attn_mask.unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(~text_attn_mask, float('-inf'))

        # Apply padding mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        return torch.matmul(attn_weights, value)

    def get_attention_stats(self) -> dict:
        """Get current attention configuration for logging."""
        stats = {
            "linear_temperature": self.linear_temperature.item(),
            "layer_idx": self.layer_idx,
        }
        if self.use_gated_mixing:
            stats["vision_gate"] = torch.sigmoid(self.vision_gate).item()
            stats["head_gates_mean"] = torch.sigmoid(self.head_gates).mean().item()
            stats["head_gates_std"] = torch.sigmoid(self.head_gates).std().item()
        return stats


class ProgressiveModalityAdaptive(nn.Module):
    """
    Progressive modality-adaptive attention.

    Early layers: Pure softmax (learn good representations)
    Middle layers: Mixed (transition)
    Later layers: Modality-adaptive (efficient processing)

    This addresses the issue where linear attention in early layers
    prevents learning of good visual representations.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        layer_idx: int = 0,
        num_layers: int = 6,
        softmax_layers: int = 2,  # First N layers use pure softmax
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.use_softmax_only = layer_idx < softmax_layers

        if self.use_softmax_only:
            # Pure softmax for early layers
            from .softmax_attention import SoftmaxAttention
            self.attention = SoftmaxAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
        else:
            # Modality-adaptive for later layers
            self.attention = ModalityAdaptiveAttentionV2(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                layer_idx=layer_idx - softmax_layers,
                num_layers=num_layers - softmax_layers,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        modality_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        num_vision_tokens: int = 197,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if self.use_softmax_only:
            return self.attention(
                hidden_states,
                attention_mask=attention_mask,
                causal_mask=causal_mask,
                output_attentions=output_attentions,
            )
        else:
            return self.attention(
                hidden_states,
                attention_mask=attention_mask,
                modality_mask=modality_mask,
                causal_mask=causal_mask,
                num_vision_tokens=num_vision_tokens,
                output_attentions=output_attentions,
            )
