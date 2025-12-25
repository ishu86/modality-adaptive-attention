"""Linear Attention with ELU+1 feature map."""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """
    ELU+1 feature map for linear attention.

    φ(x) = elu(x) + 1

    This ensures non-negative values for valid attention computation.

    Args:
        x: Input tensor

    Returns:
        Transformed tensor with non-negative values
    """
    return F.elu(x) + 1


class LinearAttention(nn.Module):
    """
    Linear Attention using ELU+1 feature map.

    Instead of computing attention as:
        Attention = softmax(Q @ K^T / sqrt(d)) @ V

    We compute:
        Q' = φ(Q), K' = φ(K)
        Attention = Q' @ (K'^T @ V) / (Q' @ sum(K'^T))

    This reduces complexity from O(n^2) to O(n) for sequence length n.

    Reference:
        - Transformers are RNNs (Katharopoulos et al., 2020)
        - Performers (Choromanski et al., 2020)
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
        Initialize linear attention.

        Args:
            hidden_dim: Hidden dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout rate
            bias: Whether to use bias in linear projections
            eps: Small constant for numerical stability
        """
        super().__init__()

        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.eps = eps

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        # Dropout
        self.proj_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for linear attention.

        Args:
            hidden_states: Input tensor (batch, seq_len, hidden_dim)
            attention_mask: Padding mask (batch, seq_len) - 1 for valid, 0 for padding
            causal_mask: Not used for linear attention (see CausalLinearAttention)
            output_attentions: Whether to return attention weights (expensive for linear attention)

        Returns:
            output: Attention output (batch, seq_len, hidden_dim)
            attention_weights: None (computing full attention matrix defeats the purpose)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        # (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply feature map
        query = elu_feature_map(query)
        key = elu_feature_map(key)

        # Apply attention mask by zeroing out padded positions in key
        if attention_mask is not None:
            # attention_mask: (batch, seq_len), 1 = valid, 0 = padding
            mask = attention_mask.unsqueeze(1).unsqueeze(-1)  # (batch, 1, seq_len, 1)
            key = key * mask
            value = value * mask

        # Compute linear attention
        # Standard form: Q' @ K'^T @ V (but this is O(n^2))
        # Linear form: Q' @ (K'^T @ V)  (this is O(n))

        # K'^T @ V: (batch, num_heads, head_dim, head_dim)
        kv = torch.einsum('bhnd,bhnv->bhdv', key, value)

        # Q' @ (K'^T @ V): (batch, num_heads, seq_len, head_dim)
        numerator = torch.einsum('bhnd,bhdv->bhnv', query, kv)

        # Normalization: Q' @ sum(K'^T, dim=seq)
        # K' sum: (batch, num_heads, head_dim)
        key_sum = key.sum(dim=2)

        # Q' @ key_sum: (batch, num_heads, seq_len)
        denominator = torch.einsum('bhnd,bhd->bhn', query, key_sum)

        # Add eps and expand for division
        denominator = denominator.unsqueeze(-1) + self.eps

        # Final attention output
        attn_output = numerator / denominator

        # Reshape back
        # (batch, seq_len, hidden_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        # Output projection
        output = self.out_proj(attn_output)
        output = self.proj_dropout(output)

        # For linear attention, returning full attention weights is expensive and defeats the purpose
        return output, None


class CausalLinearAttention(nn.Module):
    """
    Causal Linear Attention for autoregressive generation.

    Uses a cumulative sum formulation to maintain O(n) complexity
    while respecting causal constraints.

    For each position i:
        S_i = sum(K'_j @ V_j for j <= i)
        Z_i = sum(K'_j for j <= i)
        output_i = Q'_i @ S_i / (Q'_i @ Z_i)
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        bias: bool = True,
        eps: float = 1e-6,
    ):
        """Initialize causal linear attention."""
        super().__init__()

        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.eps = eps

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        self.proj_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,  # Ignored, always causal
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for causal linear attention.

        Uses cumulative sum for O(n) causal attention.
        """
        batch_size, seq_len, _ = hidden_states.shape

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply feature map
        query = elu_feature_map(query)
        key = elu_feature_map(key)

        # Apply attention mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(-1)
            key = key * mask
            value = value * mask

        # Compute K' outer V: (batch, num_heads, seq_len, head_dim, head_dim)
        kv = torch.einsum('bhnd,bhnv->bhndv', key, value)

        # Cumulative sum for causal: S_i = sum(K'_j @ V_j for j <= i)
        kv_cumsum = torch.cumsum(kv, dim=2)

        # Cumulative sum of keys: Z_i = sum(K'_j for j <= i)
        key_cumsum = torch.cumsum(key, dim=2)

        # Q' @ S_i: (batch, num_heads, seq_len, head_dim)
        numerator = torch.einsum('bhnd,bhndv->bhnv', query, kv_cumsum)

        # Q' @ Z_i: (batch, num_heads, seq_len)
        denominator = torch.einsum('bhnd,bhnd->bhn', query, key_cumsum)
        denominator = denominator.unsqueeze(-1) + self.eps

        attn_output = numerator / denominator

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(attn_output)
        output = self.proj_dropout(output)

        return output, None
