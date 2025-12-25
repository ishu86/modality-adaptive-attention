"""Loss functions for training."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CaptioningLoss(nn.Module):
    """
    Cross-entropy loss for image captioning.

    Supports:
    - Label smoothing
    - Ignoring padding tokens
    - Optional focal loss weighting
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        """
        Initialize captioning loss.

        Args:
            vocab_size: Vocabulary size
            ignore_index: Index to ignore in loss computation
            label_smoothing: Label smoothing factor (0 = no smoothing)
            reduction: Reduction method ("mean", "sum", "none")
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss.

        Args:
            logits: Predicted logits (batch, seq_len, vocab_size)
            labels: Target labels (batch, seq_len)

        Returns:
            Loss value
        """
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Flatten
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)

        # Compute cross-entropy with label smoothing
        loss = F.cross_entropy(
            shift_logits,
            shift_labels,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
        )

        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        ignore_index: int = -100,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        reduction: str = "mean",
    ):
        """
        Initialize focal loss.

        Args:
            vocab_size: Vocabulary size
            ignore_index: Index to ignore
            gamma: Focusing parameter
            alpha: Weighting factor
            reduction: Reduction method
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss."""
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Flatten
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)

        # Create mask for valid tokens
        mask = shift_labels != self.ignore_index

        # Get valid logits and labels
        valid_logits = shift_logits[mask]
        valid_labels = shift_labels[mask]

        if valid_logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Compute probabilities
        probs = F.softmax(valid_logits, dim=-1)

        # Get probability of true class
        p_t = probs.gather(1, valid_labels.unsqueeze(1)).squeeze(1)

        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Compute cross-entropy
        ce_loss = F.cross_entropy(
            valid_logits,
            valid_labels,
            reduction="none",
        )

        # Apply focal weight
        loss = focal_weight * ce_loss

        # Apply alpha if provided
        if self.alpha is not None:
            loss = self.alpha * loss

        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def compute_perplexity(loss: torch.Tensor) -> torch.Tensor:
    """
    Compute perplexity from cross-entropy loss.

    PPL = exp(loss)

    Args:
        loss: Cross-entropy loss value

    Returns:
        Perplexity value
    """
    return torch.exp(loss)
