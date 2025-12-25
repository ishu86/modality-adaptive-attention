"""Frozen CLIP Vision Encoder wrapper."""

from typing import Optional
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPProcessor


class VisionEncoder(nn.Module):
    """
    Wrapper around CLIP ViT-B/16 for extracting vision features.

    Outputs patch embeddings (before final projection) that can be used
    as vision tokens in the multimodal transformer.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        freeze: bool = True,
    ):
        """
        Initialize CLIP vision encoder.

        Args:
            model_name: HuggingFace model name
            freeze: Whether to freeze all parameters
        """
        super().__init__()

        self.model_name = model_name
        self.clip_vision = CLIPVisionModel.from_pretrained(model_name)

        # Get hidden dimension from config
        self.hidden_dim = self.clip_vision.config.hidden_size  # 768 for ViT-B/16
        self.num_patches = (
            self.clip_vision.config.image_size // self.clip_vision.config.patch_size
        ) ** 2  # 196 for 224/16

        if freeze:
            self._freeze_parameters()

    def _freeze_parameters(self):
        """Freeze all parameters in the vision encoder."""
        for param in self.clip_vision.parameters():
            param.requires_grad = False

    @property
    def num_tokens(self) -> int:
        """Total number of output tokens (patches + CLS)."""
        return self.num_patches + 1  # 197 for ViT-B/16

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_hidden_states: bool = False,
    ) -> torch.Tensor:
        """
        Extract vision features from images.

        Args:
            pixel_values: Image tensor (batch, 3, 224, 224)
            return_hidden_states: Whether to return all hidden states

        Returns:
            Vision features: (batch, num_tokens, hidden_dim)
            Shape: (batch, 197, 768) for ViT-B/16
        """
        outputs = self.clip_vision(
            pixel_values=pixel_values,
            output_hidden_states=return_hidden_states,
        )

        # Use last_hidden_state (before final projection)
        # Shape: (batch, 197, 768)
        vision_features = outputs.last_hidden_state

        return vision_features

    def get_processor(self) -> CLIPProcessor:
        """Get the CLIP processor for image preprocessing."""
        return CLIPProcessor.from_pretrained(self.model_name)


class VisionProjection(nn.Module):
    """
    Project vision features to match text embedding dimension.

    This is used when vision and text have different hidden dimensions,
    or to add learnable parameters for vision feature adaptation.
    """

    def __init__(
        self,
        vision_dim: int = 768,
        hidden_dim: int = 768,
        dropout: float = 0.1,
    ):
        """
        Initialize vision projection.

        Args:
            vision_dim: Input dimension from vision encoder
            hidden_dim: Output dimension for transformer
            dropout: Dropout rate
        """
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Project vision features.

        Args:
            vision_features: (batch, num_tokens, vision_dim)

        Returns:
            Projected features: (batch, num_tokens, hidden_dim)
        """
        return self.projection(vision_features)


def create_vision_encoder(
    model_name: str = "openai/clip-vit-base-patch16",
    freeze: bool = True,
    project: bool = True,
    hidden_dim: int = 768,
    dropout: float = 0.1,
) -> nn.Module:
    """
    Create vision encoder with optional projection layer.

    Args:
        model_name: CLIP model name
        freeze: Whether to freeze CLIP parameters
        project: Whether to add projection layer
        hidden_dim: Target hidden dimension
        dropout: Dropout rate

    Returns:
        Vision encoder module
    """
    encoder = VisionEncoder(model_name=model_name, freeze=freeze)

    if project:
        projection = VisionProjection(
            vision_dim=encoder.hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        return nn.Sequential(encoder, projection)

    return encoder
