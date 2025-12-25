"""Standard Softmax Multi-Head Attention."""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxAttention(nn.Module):
    """
    Standard scaled dot-product multi-head attention with softmax.

    This is the baseline attention mechanism used in transformers.
    Complexity: O(n^2) where n is sequence length.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """
        Initialize softmax attention.

        Args:
            hidden_dim: Hidden dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout rate
            bias: Whether to use bias in linear projections
        """
        super().__init__()

        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Q, K, V projections
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
        causal_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for softmax attention.

        Args:
            hidden_states: Input tensor (batch, seq_len, hidden_dim)
            attention_mask: Padding mask (batch, seq_len) - 1 for valid, 0 for padding
            causal_mask: Causal mask (seq_len, seq_len) - True for valid, False for masked
            output_attentions: Whether to return attention weights

        Returns:
            output: Attention output (batch, seq_len, hidden_dim)
            attention_weights: Attention weights if output_attentions=True
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

        # Compute attention scores
        # (batch, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        # Apply causal mask (for autoregressive generation)
        if causal_mask is not None:
            # causal_mask: (seq_len, seq_len), True = can attend
            # Convert to attention mask format: 0 for valid, -inf for masked
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
            attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))

        # Apply padding mask
        if attention_mask is not None:
            # attention_mask: (batch, seq_len), 1 = valid, 0 = padding
            # Expand for broadcasting: (batch, 1, 1, seq_len)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        # (batch, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, value)

        # Reshape back
        # (batch, seq_len, hidden_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        # Output projection
        output = self.out_proj(attn_output)
        output = self.proj_dropout(output)

        if output_attentions:
            return output, attn_weights
        return output, None


class CrossAttention(nn.Module):
    """
    Cross-attention module for attending to a different sequence.

    Used when text tokens need to attend to vision tokens.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """
        Initialize cross-attention.

        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            bias: Whether to use bias
        """
        super().__init__()

        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_states: torch.Tensor,
        key_value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for cross-attention.

        Args:
            query_states: Query tensor (batch, query_len, hidden_dim)
            key_value_states: Key/Value tensor (batch, kv_len, hidden_dim)
            attention_mask: Mask for key_value_states (batch, kv_len)
            output_attentions: Whether to return attention weights

        Returns:
            output: Attention output (batch, query_len, hidden_dim)
            attention_weights: Weights if output_attentions=True
        """
        batch_size, query_len, _ = query_states.shape
        kv_len = key_value_states.size(1)

        query = self.q_proj(query_states)
        key = self.k_proj(key_value_states)
        value = self.v_proj(key_value_states)

        query = query.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, query_len, self.hidden_dim)

        output = self.out_proj(attn_output)
        output = self.proj_dropout(output)

        if output_attentions:
            return output, attn_weights
        return output, None
