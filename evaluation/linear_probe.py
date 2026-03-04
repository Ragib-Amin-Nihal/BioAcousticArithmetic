"""Linear probe evaluation for merged encoders.

Evaluates merged encoders by training a lightweight linear head on frozen
encoder representations. This is the required evaluation protocol for
task arithmetic — it isolates encoder quality from classification head artifacts.

Protocol:
    1. Load merged encoder into BEATsClassifier
    2. Freeze encoder (no gradients)
    3. Train a fresh linear head (768 → num_classes)
    4. Evaluate on held-out test set
    5. Report metrics with bootstrap 95% CIs

Usage:
    evaluator = LinearProbeEvaluator(base_checkpoint_path, config)
    results = evaluator.evaluate(merged_encoder, train_loader, test_loader, num_classes)
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root for imports
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.beats_classifier import BEATsClassifier

logger = logging.getLogger(__name__)


@dataclass
class ProbeConfig:
    """Configuration for linear probe training."""

    learning_rate: float = 1e-3
    epochs: int = 10
    batch_size: int = 64
    weight_decay: float = 0.0
    # Bootstrap CI
    bootstrap_n_resamples: int = 1000
    bootstrap_confidence: float = 0.95


@dataclass
class ProbeResult:
    """Result of a linear probe evaluation."""

    accuracy: float
    loss: float
    top5_accuracy: float
    macro_f1: float
    accuracy_ci_low: float = 0.0
    accuracy_ci_high: float = 0.0
    per_class_accuracy: dict[int, float] = field(default_factory=dict)
    train_epochs: int = 0
    train_time_s: float = 0.0


class LinearProbeEvaluator:
    """Evaluates merged encoders via linear probing.

    The encoder is frozen. Only a fresh linear head is trained.
    This isolates the quality of the merged encoder representations.

    Args:
        base_checkpoint_path: Path to BEATs pretrained checkpoint (for model init).
        config: Probe training configuration.
        device: Torch device.
    """

    def __init__(
        self,
        base_checkpoint_path: str,
        config: Optional[ProbeConfig] = None,
        device: str = "cuda",
    ) -> None:
        self.base_checkpoint_path = base_checkpoint_path
        self.config = config or ProbeConfig()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def _build_model(
        self,
        encoder_state_dict: dict[str, torch.Tensor],
        num_classes: int,
    ) -> BEATsClassifier:
        """Build model with merged encoder and fresh classification head.

        Args:
            encoder_state_dict: Merged encoder weights.
            num_classes: Number of output classes.

        Returns:
            BEATsClassifier with frozen encoder and trainable head.
        """
        model = BEATsClassifier(
            self.base_checkpoint_path, num_classes, freeze_epochs=999
        )
        model.load_encoder_state_dict(encoder_state_dict, strict=False)
        model.freeze_encoder()

        # Reinitialize the classification head
        nn.init.xavier_uniform_(model.classifier.weight)
        nn.init.zeros_(model.classifier.bias)

        return model.to(self.device)

    def _extract_features(
        self,
        model: BEATsClassifier,
        loader: DataLoader,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract frozen encoder features for all samples.

        Pre-computing features avoids redundant forward passes through the
        encoder during linear head training. This is a major speedup.

        Args:
            model: BEATsClassifier with frozen encoder.
            loader: DataLoader yielding (audio, labels).

        Returns:
            (features, labels) tensors on CPU.
        """
        all_features: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        model.eval()
        n_batches = len(loader)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for batch_idx, (audio, labels) in enumerate(loader):
                audio = audio.to(self.device)
                padding_mask = torch.zeros(
                    audio.shape, dtype=torch.bool, device=self.device
                )
                features, _ = model.encoder.extract_features(
                    audio, padding_mask=padding_mask
                )
                # Mean pool over time
                pooled = features.mean(dim=1)  # (batch, 768)
                all_features.append(pooled.cpu().float())
                all_labels.append(labels)

                if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == n_batches:
                    n_done = sum(f.shape[0] for f in all_features)
                    logger.info(
                        "    batch %d/%d (%d samples extracted)",
                        batch_idx + 1, n_batches, n_done,
                    )

        return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)

    def _train_head(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int,
    ) -> nn.Linear:
        """Train a linear head on pre-extracted features.

        Args:
            features: (N, 768) feature tensor.
            labels: (N,) label tensor.
            num_classes: Number of classes.

        Returns:
            Trained linear head.
        """
        head = nn.Linear(features.shape[1], num_classes).to(self.device)
        nn.init.xavier_uniform_(head.weight)
        nn.init.zeros_(head.bias)

        optimizer = torch.optim.Adam(
            head.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        dataset = torch.utils.data.TensorDataset(features, labels)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True,
        )

        head.train()
        for epoch in range(self.config.epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            for feat_batch, label_batch in loader:
                feat_batch = feat_batch.to(self.device)
                label_batch = label_batch.to(self.device)

                logits = head(feat_batch)
                loss = criterion(logits, label_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                correct += (logits.argmax(1) == label_batch).sum().item()
                total += label_batch.size(0)

            avg_loss = total_loss / len(loader)
            acc = correct / max(total, 1)
            logger.info(
                "    probe epoch %d/%d: loss=%.4f, acc=%.4f",
                epoch + 1, self.config.epochs, avg_loss, acc,
            )

        return head

    @torch.no_grad()
    def _evaluate_head(
        self,
        head: nn.Linear,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run inference with trained head, return predictions and labels.

        Args:
            head: Trained linear head.
            features: (N, 768) feature tensor.
            labels: (N,) label tensor.

        Returns:
            (predictions, labels) as numpy arrays.
        """
        head.eval()
        dataset = torch.utils.data.TensorDataset(features, labels)
        loader = DataLoader(dataset, batch_size=self.config.batch_size * 4)

        all_preds: list[np.ndarray] = []
        all_probs: list[np.ndarray] = []
        all_labels_list: list[np.ndarray] = []

        for feat_batch, label_batch in loader:
            feat_batch = feat_batch.to(self.device)
            logits = head(feat_batch)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels_list.append(label_batch.numpy())

        predictions = np.concatenate(all_preds)
        labels_np = np.concatenate(all_labels_list)
        return predictions, labels_np

    def _compute_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        num_classes: int,
    ) -> ProbeResult:
        """Compute accuracy, macro-F1, and bootstrap CIs.

        Args:
            predictions: (N,) predicted class indices.
            labels: (N,) true class indices.
            num_classes: Number of classes.

        Returns:
            ProbeResult with all metrics.
        """
        correct = predictions == labels
        accuracy = correct.mean()

        # Top-5 not applicable without probabilities in this path;
        # set to 0 (computed separately when logits are available)
        top5_accuracy = 0.0

        # Per-class accuracy
        per_class_acc: dict[int, float] = {}
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() > 0:
                per_class_acc[c] = correct[mask].mean()

        # Macro F1
        macro_f1 = self._macro_f1(predictions, labels, num_classes)

        # Bootstrap CI for accuracy
        ci_low, ci_high = self._bootstrap_ci(correct)

        return ProbeResult(
            accuracy=float(accuracy),
            loss=0.0,  # Populated by caller if needed
            top5_accuracy=top5_accuracy,
            macro_f1=macro_f1,
            accuracy_ci_low=ci_low,
            accuracy_ci_high=ci_high,
            per_class_accuracy=per_class_acc,
        )

    def _macro_f1(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        num_classes: int,
    ) -> float:
        """Compute macro-averaged F1 score.

        Args:
            predictions: (N,) predicted class indices.
            labels: (N,) true class indices.
            num_classes: Number of classes.

        Returns:
            Macro F1 score.
        """
        f1_scores: list[float] = []
        for c in range(num_classes):
            tp = ((predictions == c) & (labels == c)).sum()
            fp = ((predictions == c) & (labels != c)).sum()
            fn = ((predictions != c) & (labels == c)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            if precision + recall > 0:
                f1_scores.append(2 * precision * recall / (precision + recall))
            else:
                f1_scores.append(0.0)

        return float(np.mean(f1_scores)) if f1_scores else 0.0

    def _bootstrap_ci(
        self,
        correct: np.ndarray,
    ) -> tuple[float, float]:
        """Compute bootstrap confidence interval for accuracy.

        Args:
            correct: Boolean array of per-sample correctness.

        Returns:
            (ci_low, ci_high) tuple.
        """
        rng = np.random.default_rng(42)
        n = len(correct)
        boot_means = np.array([
            rng.choice(correct, size=n, replace=True).mean()
            for _ in range(self.config.bootstrap_n_resamples)
        ])

        alpha = 1.0 - self.config.bootstrap_confidence
        ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
        ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
        return ci_low, ci_high

    def evaluate(
        self,
        encoder_state_dict: dict[str, torch.Tensor],
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_classes: int,
    ) -> ProbeResult:
        """Full linear probe evaluation pipeline.

        1. Load encoder, freeze it
        2. Extract features from train and test sets
        3. Train linear head on train features
        4. Evaluate on test features
        5. Compute metrics with bootstrap CIs

        Args:
            encoder_state_dict: Merged encoder weights.
            train_loader: DataLoader for probe training (audio, labels).
            test_loader: DataLoader for evaluation (audio, labels).
            num_classes: Number of species classes.

        Returns:
            ProbeResult with accuracy, F1, CIs.
        """
        start_time = time.time()

        # Build model with merged encoder
        model = self._build_model(encoder_state_dict, num_classes)

        # Extract features (much faster than running encoder every epoch)
        logger.info("  Extracting train features...")
        train_features, train_labels = self._extract_features(model, train_loader)
        logger.info("  Extracting test features...")
        test_features, test_labels = self._extract_features(model, test_loader)

        logger.info(
            "  Features: train=%s, test=%s",
            train_features.shape, test_features.shape,
        )

        # Train linear head
        logger.info("  Training linear probe (%d epochs)...", self.config.epochs)
        head = self._train_head(train_features, train_labels, num_classes)

        # Evaluate
        predictions, labels = self._evaluate_head(head, test_features, test_labels)
        result = self._compute_metrics(predictions, labels, num_classes)
        result.train_epochs = self.config.epochs
        result.train_time_s = time.time() - start_time

        logger.info(
            "  Probe result: acc=%.4f [%.4f, %.4f], F1=%.4f (%.1fs)",
            result.accuracy, result.accuracy_ci_low, result.accuracy_ci_high,
            result.macro_f1, result.train_time_s,
        )

        # Cleanup
        del model, head, train_features, test_features
        torch.cuda.empty_cache()

        return result

    def evaluate_per_group(
        self,
        encoder_state_dict: dict[str, torch.Tensor],
        group_loaders: dict[str, tuple[DataLoader, DataLoader, int]],
    ) -> dict[str, ProbeResult]:
        """Evaluate a merged encoder on multiple groups independently.

        For each group, trains a separate linear probe and evaluates.
        This measures whether merging degraded per-group performance.

        Args:
            encoder_state_dict: Merged encoder weights.
            group_loaders: Dict mapping group_name → (train_loader, test_loader, num_classes).

        Returns:
            Dict mapping group_name → ProbeResult.
        """
        results: dict[str, ProbeResult] = {}
        for group_name, (train_loader, test_loader, n_classes) in group_loaders.items():
            logger.info("Evaluating group: %s (%d classes)", group_name, n_classes)
            results[group_name] = self.evaluate(
                encoder_state_dict, train_loader, test_loader, n_classes,
            )
        return results