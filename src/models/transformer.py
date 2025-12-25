"""Transformer Block with configurable attention."""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from .attention import SoftmaxAttention, LinearAttention, ModalityAdaptiveAttention
from .attention.modality_adaptive import create_attention_module


class FeedForward(nn.Module):
    """
    Feed-forward network (FFN) with GELU activation.

    FFN(x) = Linear(GELU(Linear(x)))
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
    ):
        """
        Initialize feed-forward network.

        Args:
            hidden_dim: Input/output dimension
            ffn_dim: Intermediate dimension (usually 4x hidden_dim)
            dropout: Dropout rate
        """
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer Block with configurable attention.

    Architecture:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
        attention_type: str = "softmax",
        layer_idx: int = 0,
        num_layers: int = 6,
        attention_kwargs: dict = None,
    ):
        """
        Initialize transformer block.

        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            ffn_dim: FFN intermediate dimension
            dropout: Dropout rate
            attention_type: "softmax", "linear", "modality_adaptive",
                          "modality_adaptive_v2", or "progressive"
            layer_idx: Current layer index (for layer-wise attention)
            num_layers: Total number of layers
            attention_kwargs: Additional kwargs for attention module
        """
        super().__init__()

        self.attention_type = attention_type

        # Layer norms (pre-norm)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Attention
        self.attention = create_attention_module(
            attention_type=attention_type,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            layer_idx=layer_idx,
            num_layers=num_layers,
            **(attention_kwargs or {}),
        )

        # Feed-forward
        self.ffn = FeedForward(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
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
        """
        Forward pass.

        Args:
            hidden_states: Input tensor (batch, seq_len, hidden_dim)
            attention_mask: Padding mask (batch, seq_len)
            modality_mask: 0 for vision, 1 for text (batch, seq_len)
            causal_mask: Causal mask (seq_len, seq_len)
            num_vision_tokens: Number of vision tokens
            output_attentions: Whether to return attention weights

        Returns:
            hidden_states: Output tensor (batch, seq_len, hidden_dim)
            attention_weights: Attention weights if output_attentions=True
        """
        # Pre-norm attention
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)

        if self.attention_type in ("modality_adaptive", "modality_adaptive_v2", "progressive"):
            hidden_states, attn_weights = self.attention(
                hidden_states,
                attention_mask=attention_mask,
                modality_mask=modality_mask,
                causal_mask=causal_mask,
                num_vision_tokens=num_vision_tokens,
                output_attentions=output_attentions,
            )
        else:
            hidden_states, attn_weights = self.attention(
                hidden_states,
                attention_mask=attention_mask,
                causal_mask=causal_mask,
                output_attentions=output_attentions,
            )

        hidden_states = residual + hidden_states

        # Pre-norm FFN
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, attn_weights


class TransformerEncoder(nn.Module):
    """
    Stack of Transformer blocks.
    """

    def __init__(
        self,
        num_layers: int = 6,
        hidden_dim: int = 768,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
        attention_type: str = "softmax",
        attention_kwargs: dict = None,
    ):
        """
        Initialize transformer encoder.

        Args:
            num_layers: Number of transformer blocks
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            ffn_dim: FFN intermediate dimension
            dropout: Dropout rate
            attention_type: Attention mechanism type
            attention_kwargs: Additional kwargs for attention modules
        """
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                attention_type=attention_type,
                layer_idx=i,
                num_layers=num_layers,
                attention_kwargs=attention_kwargs,
            )
            for i in range(num_layers)
        ])

        self.final_ln = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        modality_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        num_vision_tokens: int = 197,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple], Optional[Tuple]]:
        """
        Forward pass through all layers.

        Args:
            hidden_states: Input tensor (batch, seq_len, hidden_dim)
            attention_mask: Padding mask
            modality_mask: Modality indicators
            causal_mask: Causal mask
            num_vision_tokens: Number of vision tokens
            output_attentions: Whether to return all attention weights
            output_hidden_states: Whether to return all hidden states

        Returns:
            hidden_states: Final output
            all_hidden_states: Tuple of hidden states from each layer (if requested)
            all_attentions: Tuple of attention weights from each layer (if requested)
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states, attn_weights = layer(
                hidden_states,
                attention_mask=attention_mask,
                modality_mask=modality_mask,
                causal_mask=causal_mask,
                num_vision_tokens=num_vision_tokens,
                output_attentions=output_attentions,
            )

            if output_attentions and attn_weights is not None:
                all_attentions = all_attentions + (attn_weights,)

        hidden_states = self.final_ln(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states, all_hidden_states, all_attentions
