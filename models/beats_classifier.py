"""BEATs encoder + linear classification head for bioacoustic species classification.

Wraps the pretrained BEATs model with a task-specific linear head.
Supports encoder freezing for warmup epochs and extraction of encoder-only
state dicts for task vector computation.

Usage:
    model = BEATsClassifier("checkpoints/BEATs_iter3+.pt", num_classes=336)
    logits = model(waveform)  # waveform: (batch, 80000) at 16kHz
    encoder_weights = model.get_encoder_state_dict()
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Add BEATs source to path
_BEATS_DIR = str(Path(__file__).resolve().parent.parent / "external" / "unilm" / "beats")
if _BEATS_DIR not in sys.path:
    sys.path.insert(0, _BEATS_DIR)

from BEATs import BEATs, BEATsConfig  # type: ignore


class BEATsClassifier(nn.Module):
    """BEATs encoder with a linear classification head.

    Architecture:
        Input waveform (batch, samples) at 16kHz
        → BEATs encoder → (batch, time_steps, embed_dim)
        → Mean pooling over time → (batch, embed_dim)
        → Linear head → (batch, num_classes)

    For task arithmetic, only encoder weights are used to compute task vectors.
    The classification head is group-specific and discarded during merging.

    Args:
        checkpoint_path: Path to pretrained BEATs checkpoint (.pt file).
        num_classes: Number of species classes for the linear head.
        freeze_epochs: Number of initial epochs to freeze encoder weights.
            During frozen epochs, only the classification head is trained.
    """

    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int,
        freeze_epochs: int = 2,
    ) -> None:
        super().__init__()

        # Load pretrained BEATs
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        cfg = BEATsConfig(ckpt["cfg"])
        self.encoder = BEATs(cfg)
        self.encoder.load_state_dict(ckpt["model"])

        # Remove the pretrained AudioSet predictor head.
        # With predictor=None, extract_features() returns raw encoder output
        # (batch, time_steps, 768) instead of AudioSet logits (batch, 527).
        self.encoder.predictor = None

        self.embed_dim: int = cfg.encoder_embed_dim  # 768 for base model
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        self.freeze_epochs = freeze_epochs
        self._frozen = False

        # Initialize classifier with small weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters (only head trains)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self._frozen = True

    def unfreeze_encoder(self) -> None:
        """Unfreeze all encoder parameters (full fine-tuning)."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self._frozen = False

    @property
    def is_frozen(self) -> bool:
        """Whether the encoder is currently frozen."""
        return self._frozen

    def forward(
        self,
        audio: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass: raw waveform → class logits.

        Args:
            audio: Raw waveform tensor, shape (batch, samples) at 16kHz.
            padding_mask: Optional boolean mask, shape (batch, samples).
                True indicates padded positions. If None, assumes no padding.

        Returns:
            Logits tensor, shape (batch, num_classes).
        """
        if padding_mask is None:
            padding_mask = torch.zeros(
                audio.shape, dtype=torch.bool, device=audio.device
            )

        # BEATs extract_features returns (features, output_padding_mask)
        features, _ = self.encoder.extract_features(
            audio, padding_mask=padding_mask
        )
        # features: (batch, time_steps, embed_dim)

        # Mean-pool over time dimension
        pooled = features.mean(dim=1)  # (batch, embed_dim)

        logits = self.classifier(pooled)
        return logits

    def get_encoder_state_dict(self) -> dict[str, torch.Tensor]:
        """Return a copy of encoder-only weights (for task vector computation).

        These are the weights used in: τ = θ_finetuned - θ_pretrained
        The classification head is excluded since it's group-specific.
        """
        return {k: v.clone() for k, v in self.encoder.state_dict().items()}

    def load_encoder_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        strict: bool = True,
    ) -> None:
        """Load encoder weights (e.g., from a merged model).

        Args:
            state_dict: Encoder-only state dict.
            strict: Whether to require exact key matching.
        """
        self.encoder.load_state_dict(state_dict, strict=strict)

    def get_num_params(self, trainable_only: bool = False) -> int:
        """Count model parameters.

        Args:
            trainable_only: If True, count only parameters with requires_grad.

        Returns:
            Total parameter count.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())