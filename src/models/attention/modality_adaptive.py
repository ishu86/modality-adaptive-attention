"""Modality-Adaptive Attention: Different attention for vision vs text tokens."""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .softmax_attention import SoftmaxAttention
from .linear_attention import LinearAttention, elu_feature_map


class ModalityAdaptiveAttention(nn.Module):
    """
    Modality-Adaptive Attention module.

    Routes tokens to different attention mechanisms based on modality:
    - Vision tokens (modality_mask == 0): Linear attention (O(n) complexity)
    - Text tokens (modality_mask == 1): Softmax attention (O(n^2) complexity)

    For cross-modal attention (text attending to vision), uses softmax
    to preserve grounding quality.

    This approach aims to:
    1. Speed up vision-heavy computations with linear attention
    2. Maintain text quality with full softmax attention
    3. Preserve vision-text grounding with softmax cross-attention
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        bias: bool = True,
        eps: float = 1e-6,
    ):
        """
        Initialize modality-adaptive attention.

        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            bias: Whether to use bias
            eps: Small constant for numerical stability
        """
        super().__init__()

        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.eps = eps

        # Shared Q, K, V projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

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
        Forward pass with modality-adaptive attention.

        The computation is split into regions:
        1. Vision-to-Vision: Linear attention
        2. Text-to-Vision: Softmax attention (cross-modal grounding)
        3. Text-to-Text: Softmax attention with causal mask

        Args:
            hidden_states: Input tensor (batch, seq_len, hidden_dim)
            attention_mask: Padding mask (batch, seq_len)
            modality_mask: 0 for vision, 1 for text (batch, seq_len)
            causal_mask: Causal mask for text tokens (seq_len, seq_len)
            num_vision_tokens: Number of vision tokens (for splitting)
            output_attentions: Whether to return attention weights

        Returns:
            output: Attention output (batch, seq_len, hidden_dim)
            attention_weights: Combined attention weights if output_attentions=True
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

        text_len = seq_len - num_vision_tokens

        # ============================================
        # 1. Vision-to-Vision: Linear Attention
        # ============================================
        vision_output = self._linear_attention(
            vision_query, vision_key, vision_value
        )

        # ============================================
        # 2. Text Attention (to both vision and text)
        # ============================================
        # Text uses softmax for both self-attention and cross-attention
        # to preserve grounding quality

        # Full attention for text: attends to all tokens
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

    def _linear_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute linear attention for vision tokens.

        Args:
            query, key, value: (batch, num_heads, vision_len, head_dim)

        Returns:
            output: (batch, num_heads, vision_len, head_dim)
        """
        # Apply ELU+1 feature map
        query = elu_feature_map(query)
        key = elu_feature_map(key)

        # K'^T @ V: (batch, num_heads, head_dim, head_dim)
        kv = torch.einsum('bhnd,bhnv->bhdv', key, value)

        # Q' @ (K'^T @ V): (batch, num_heads, vision_len, head_dim)
        numerator = torch.einsum('bhnd,bhdv->bhnv', query, kv)

        # Normalization
        key_sum = key.sum(dim=2)  # (batch, num_heads, head_dim)
        denominator = torch.einsum('bhnd,bhd->bhn', query, key_sum)
        denominator = denominator.unsqueeze(-1) + self.eps

        return numerator / denominator

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
        Compute softmax attention for text tokens.

        Text can attend to:
        - All vision tokens (bidirectional)
        - Previous text tokens only (causal)

        Args:
            query: (batch, num_heads, text_len, head_dim)
            key, value: (batch, num_heads, seq_len, head_dim)
            attention_mask: (batch, seq_len)
            causal_mask: (seq_len, seq_len) or None
            num_vision_tokens: Number of vision tokens

        Returns:
            output: (batch, num_heads, text_len, head_dim)
        """
        batch_size = query.size(0)
        text_len = query.size(2)
        seq_len = key.size(2)

        # Compute attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        # Create attention mask for text queries
        # Shape: (text_len, seq_len)
        text_attn_mask = torch.ones(text_len, seq_len, dtype=torch.bool, device=query.device)

        # Text can attend to all vision tokens
        text_attn_mask[:, :num_vision_tokens] = True

        # Text-to-text: causal (lower triangular)
        text_to_text_mask = torch.tril(
            torch.ones(text_len, text_len, dtype=torch.bool, device=query.device)
        )
        text_attn_mask[:, num_vision_tokens:] = text_to_text_mask

        # Apply to attention scores
        text_attn_mask = text_attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, text_len, seq_len)
        attn_scores = attn_scores.masked_fill(~text_attn_mask, float('-inf'))

        # Apply padding mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply to values
        output = torch.matmul(attn_weights, value)

        return output


def create_attention_module(
    attention_type: str,
    hidden_dim: int = 768,
    num_heads: int = 12,
    dropout: float = 0.1,
    layer_idx: int = 0,
    num_layers: int = 6,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create attention modules.

    Args:
        attention_type: "softmax", "linear", "modality_adaptive",
                       "modality_adaptive_v2", or "progressive"
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        layer_idx: Current layer index (for layer-wise scheduling)
        num_layers: Total number of layers
        **kwargs: Additional arguments

    Returns:
        Attention module
    """
    if attention_type == "softmax":
        return SoftmaxAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
    elif attention_type == "linear":
        return LinearAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
    elif attention_type == "modality_adaptive":
        return ModalityAdaptiveAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
    elif attention_type == "modality_adaptive_v2":
        from .modality_adaptive_v2 import ModalityAdaptiveAttentionV2
        return ModalityAdaptiveAttentionV2(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            layer_idx=layer_idx,
            num_layers=num_layers,
            use_gated_mixing=kwargs.get("use_gated_mixing", True),
            initial_softmax_ratio=kwargs.get("initial_softmax_ratio", 0.3),
            linear_temperature=kwargs.get("linear_temperature", 1.0),
        )
    elif attention_type == "progressive":
        from .modality_adaptive_v2 import ProgressiveModalityAdaptive
        return ProgressiveModalityAdaptive(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            layer_idx=layer_idx,
            num_layers=num_layers,
            softmax_layers=kwargs.get("softmax_layers", 2),
        )
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")
