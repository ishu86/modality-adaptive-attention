"""Caption evaluation metrics using pycocoevalcap."""

from typing import Dict, List, Any
import torch


def compute_caption_metrics(
    predictions: Dict[int, List[str]],
    references: Dict[int, List[str]],
) -> Dict[str, float]:
    """
    Compute image captioning metrics.

    Uses pycocoevalcap for:
    - BLEU (1-4)
    - METEOR
    - ROUGE-L
    - CIDEr
    - SPICE (if available)

    Args:
        predictions: Dict mapping image_id -> list of predicted captions
        references: Dict mapping image_id -> list of reference captions

    Returns:
        Dictionary of metric name -> score
    """
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider
    except ImportError:
        print("pycocoevalcap not installed. Install with: pip install pycocoevalcap")
        return {}

    # Format for pycocoevalcap: {id: [caption1, caption2, ...]}
    # Predictions should have single caption per image
    gts = {}
    res = {}

    for img_id in predictions:
        if img_id in references:
            gts[img_id] = references[img_id]
            res[img_id] = predictions[img_id]

    if len(gts) == 0:
        return {}

    # Initialize scorers
    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Rouge(), "ROUGE-L"),
        (Cider(), "CIDEr"),
    ]

    # METEOR requires Java - add only if available
    try:
        scorers.insert(1, (Meteor(), "METEOR"))
    except Exception as e:
        print(f"METEOR unavailable (requires Java): {e}")

    # Compute scores
    metrics = {}
    for scorer, method in scorers:
        try:
            score, _ = scorer.compute_score(gts, res)
            if isinstance(method, list):
                for m, s in zip(method, score):
                    metrics[m] = s
            else:
                metrics[method] = score
        except Exception as e:
            print(f"Error computing {method}: {e}")

    return metrics


def compute_perplexity_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """
    Compute perplexity from model logits.

    Args:
        logits: Model output logits (batch, seq_len, vocab_size)
        labels: Target labels (batch, seq_len)
        ignore_index: Index to ignore

    Returns:
        Perplexity value
    """
    import torch.nn.functional as F

    # Flatten
    shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
    shift_labels = labels[:, 1:].contiguous().view(-1)

    # Compute cross-entropy
    loss = F.cross_entropy(
        shift_logits,
        shift_labels,
        ignore_index=ignore_index,
        reduction="mean",
    )

    return torch.exp(loss).item()


class MetricsTracker:
    """
    Track and aggregate metrics over time.
    """

    def __init__(self):
        self.metrics = {}
        self.counts = {}

    def update(self, metrics: Dict[str, float], count: int = 1):
        """Update metrics with new values."""
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = 0.0
                self.counts[name] = 0
            self.metrics[name] += value * count
            self.counts[name] += count

    def compute(self) -> Dict[str, float]:
        """Compute average metrics."""
        return {
            name: self.metrics[name] / max(1, self.counts[name])
            for name in self.metrics
        }

    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}
