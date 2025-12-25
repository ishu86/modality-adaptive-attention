"""Evaluation pipeline for image captioning."""

from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer

from .metrics import compute_caption_metrics, MetricsTracker


class Evaluator:
    """
    Evaluator for image captioning models.

    Handles:
    - Caption generation with beam search
    - Metric computation (BLEU, CIDEr, etc.)
    - Perplexity evaluation
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[GPT2Tokenizer] = None,
        device: Optional[torch.device] = None,
        max_length: int = 30,
        beam_size: int = 5,
    ):
        """
        Initialize evaluator.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for decoding
            device: Device to use
            max_length: Maximum caption length
            beam_size: Beam search size
        """
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.tokenizer = tokenizer or GPT2Tokenizer.from_pretrained("gpt2")
        self.max_length = max_length
        self.beam_size = beam_size

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        compute_metrics: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run evaluation on a dataset.

        Args:
            dataloader: Evaluation data loader
            compute_metrics: Whether to compute caption metrics
            verbose: Whether to print progress

        Returns:
            Dictionary with:
                - metrics: Caption metrics (if compute_metrics=True)
                - predictions: Generated captions
                - perplexity: Average perplexity
        """
        self.model.eval()

        all_predictions = {}
        all_references = {}
        total_loss = 0.0
        num_batches = 0

        iterator = tqdm(dataloader, desc="Evaluating") if verbose else dataloader

        for batch in iterator:
            if batch is None:
                continue

            # Move to device
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Get vision features
            vision_features = batch.get("vision_features")
            images = batch.get("images")

            # Generate captions
            generated_ids = self.model.generate(
                vision_features=vision_features,
                images=images,
                max_length=self.max_length,
                do_sample=False,  # Greedy for evaluation
            )

            # Decode predictions
            for i, (gen_ids, img_id) in enumerate(
                zip(generated_ids, batch.get("image_ids", range(len(generated_ids))))
            ):
                # Decode, removing special tokens
                caption = self.tokenizer.decode(
                    gen_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                all_predictions[img_id] = [caption.strip()]

            # Store references
            if "reference_captions" in batch:
                for img_id, refs in zip(batch["image_ids"], batch["reference_captions"]):
                    all_references[img_id] = refs

            # Compute loss for perplexity
            if "labels" in batch:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    vision_features=vision_features,
                    images=images,
                    attention_mask=batch.get("full_attention_mask"),
                    modality_mask=batch.get("modality_mask"),
                    labels=batch["labels"],
                )
                if outputs["loss"] is not None:
                    total_loss += outputs["loss"].item()
                    num_batches += 1

        # Compute metrics
        results = {
            "predictions": all_predictions,
            "perplexity": torch.exp(
                torch.tensor(total_loss / max(1, num_batches))
            ).item() if num_batches > 0 else None,
        }

        if compute_metrics and all_references:
            metrics = compute_caption_metrics(all_predictions, all_references)
            results["metrics"] = metrics

            if verbose:
                print("\nEvaluation Results:")
                print(f"  Perplexity: {results['perplexity']:.2f}")
                for name, value in metrics.items():
                    print(f"  {name}: {value:.4f}")

        return results

    @torch.no_grad()
    def generate_caption(
        self,
        vision_features: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate a caption for a single image.

        Args:
            vision_features: Pre-computed CLIP features (1, num_tokens, hidden_dim)
            image: Raw image tensor (1, 3, 224, 224)
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling

        Returns:
            Generated caption string
        """
        self.model.eval()

        # Move to device
        if vision_features is not None:
            vision_features = vision_features.to(self.device)
        if image is not None:
            image = image.to(self.device)

        # Generate
        generated_ids = self.model.generate(
            vision_features=vision_features,
            images=image,
            max_length=self.max_length,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # Decode
        caption = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return caption.strip()
