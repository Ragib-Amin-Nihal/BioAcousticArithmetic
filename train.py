#!/usr/bin/env python3
"""Fine-tune BEATs encoder on a species-group dataset.

CRITICAL CONSTRAINT: All fine-tuning runs MUST use identical hyperparameters.
Hyperparameters are read from the YAML config and logged to W&B for verification.
Only the data subset (--group) changes between runs.

Usage:
    python train.py --group G1_passerines --config configs/base.yaml
    python train.py --group ALL_birds --config configs/base.yaml
    python train.py --group G4_marine_mammals --config configs/base.yaml --gpu 1

All checkpoints include:
    - model_state_dict (full model: encoder + head)
    - encoder_state_dict (encoder only, for task vectors)
    - optimizer_state_dict
    - config, epoch, metrics, git hash, group name
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import yaml
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML config and resolve paths."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg["paths"]["project_root"]).resolve()
    for k, v in cfg["paths"].items():
        if k != "project_root" and isinstance(v, str):
            cfg["paths"][k] = str(root / v)
    return cfg


def get_git_hash() -> str:
    """Get current git commit hash for reproducibility."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AudioDataset(Dataset):
    """Load preprocessed 5-second audio clips from species-group JSON manifests.

    Each sample is a dict with 'path', 'label', 'species' keys.
    Audio is loaded as raw waveform at 16kHz mono.

    Args:
        manifest_path: Path to the species-group JSON file.
        split: One of 'train', 'val', 'test'.
        target_sr: Target sample rate (should match preprocessing).
        clip_duration_s: Expected clip duration in seconds.
    """

    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        target_sr: int = 16000,
        clip_duration_s: float = 5.0,
    ) -> None:
        with open(manifest_path) as f:
            data = json.load(f)

        self.samples: list[dict[str, Any]] = data[split]
        self.metadata: dict[str, Any] = data["metadata"]
        self.target_sr = target_sr
        self.target_length = int(target_sr * clip_duration_s)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        try:
            waveform, sr = torchaudio.load(sample["path"])
        except Exception as e:
            # Return silence + label on load failure (logged but doesn't crash)
            log.debug(f"Failed to load {sample['path']}: {e}")
            waveform = torch.zeros(1, self.target_length)
            sr = self.target_sr

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed (shouldn't be, clips are preprocessed)
        if sr != self.target_sr:
            waveform = torchaudio.transforms.Resample(sr, self.target_sr)(waveform)

        # Pad or trim to exact target length
        if waveform.shape[1] < self.target_length:
            pad = torch.zeros(1, self.target_length - waveform.shape[1])
            waveform = torch.cat([waveform, pad], dim=1)
        else:
            waveform = waveform[:, :self.target_length]

        return waveform.squeeze(0), sample["label"]


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

class SpecAugment(nn.Module):
    """SpecAugment-style augmentation applied to raw waveforms.

    Since BEATs computes spectrograms internally, we apply time masking
    in the waveform domain (zero out random time segments). Frequency
    masking is not possible on raw waveforms, but the time masking
    provides regularization.

    For proper SpecAugment on mel spectrograms, this would need to
    hook into BEATs' internal spectrogram computation, which we avoid
    to keep the encoder as a black box.

    Args:
        time_mask_param: Maximum width of time mask in samples.
        n_time_masks: Number of time masks to apply.
        sample_rate: Audio sample rate.
    """

    def __init__(
        self,
        time_mask_param: int = 50,
        n_time_masks: int = 2,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()
        # Convert ms to samples: 50ms * 16 = 800 samples per mask
        self.time_mask_samples = int(time_mask_param * sample_rate / 1000)
        self.n_time_masks = n_time_masks

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply time masking to waveform.

        Args:
            waveform: (batch, samples) or (samples,)

        Returns:
            Masked waveform of same shape.
        """
        if not self.training:
            return waveform

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        batch_size, length = waveform.shape
        masked = waveform.clone()

        for _ in range(self.n_time_masks):
            mask_len = torch.randint(0, self.time_mask_samples, (batch_size,))
            mask_start = torch.randint(0, max(1, length - self.time_mask_samples), (batch_size,))

            for b in range(batch_size):
                start = mask_start[b].item()
                end = min(start + mask_len[b].item(), length)
                masked[b, start:end] = 0.0

        if squeeze:
            masked = masked.squeeze(0)

        return masked


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply mixup augmentation.

    Args:
        x: Input batch (batch, samples).
        y: Labels (batch,) as class indices.
        alpha: Beta distribution parameter. Higher = more mixing.

    Returns:
        mixed_x: Mixed input.
        y_a: Original labels.
        y_b: Shuffled labels.
        lam: Mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute loss for mixup samples.

    Args:
        criterion: Loss function (e.g., CrossEntropyLoss).
        logits: Model predictions.
        y_a: Original labels.
        y_b: Shuffled labels.
        lam: Mixing coefficient.

    Returns:
        Weighted loss combining both label sets.
    """
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    augment: Optional[nn.Module],
    mixup_alpha: float,
    grad_clip: float,
    device: torch.device,
    precision_dtype: torch.dtype,
) -> dict[str, float]:
    """Train for one epoch.

    Args:
        model: BEATsClassifier model.
        loader: Training DataLoader.
        optimizer: Optimizer.
        scheduler: LR scheduler (stepped per batch).
        criterion: Loss function.
        scaler: GradScaler for mixed precision.
        augment: Optional augmentation module (SpecAugment).
        mixup_alpha: Mixup alpha parameter. 0 disables mixup.
        grad_clip: Maximum gradient norm.
        device: CUDA device.
        precision_dtype: torch.bfloat16 or torch.float16.

    Returns:
        Dict with 'loss', 'accuracy', 'lr' metrics.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (audio, labels) in enumerate(loader):
        audio = audio.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Augmentation
        if augment is not None:
            audio = augment(audio)

        # Mixup
        use_mixup = mixup_alpha > 0 and np.random.random() < 0.5
        if use_mixup:
            audio, labels_a, labels_b, lam = mixup_data(audio, labels, mixup_alpha)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(dtype=precision_dtype):
            logits = model(audio)

            if use_mixup:
                loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            else:
                loss = criterion(logits, labels)

        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * audio.size(0)
        if not use_mixup:
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        else:
            # For mixup, approximate accuracy using dominant label
            preds = logits.argmax(dim=1)
            correct += (lam * (preds == labels_a).float() +
                        (1 - lam) * (preds == labels_b).float()).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    current_lr = scheduler.get_last_lr()[0]

    return {"loss": avg_loss, "accuracy": accuracy, "lr": current_lr}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    precision_dtype: torch.dtype,
) -> dict[str, float]:
    """Evaluate on validation set.

    Args:
        model: BEATsClassifier model.
        loader: Validation DataLoader.
        criterion: Loss function.
        device: CUDA device.
        precision_dtype: Precision for autocast.

    Returns:
        Dict with 'loss', 'accuracy', 'top5_accuracy' metrics.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0

    for audio, labels in loader:
        audio = audio.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(dtype=precision_dtype):
            logits = model(audio)
            loss = criterion(logits, labels)

        total_loss += loss.item() * audio.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()

        # Top-5 accuracy
        if logits.size(1) >= 5:
            _, top5_preds = logits.topk(5, dim=1)
            correct_top5 += (top5_preds == labels.unsqueeze(1)).any(dim=1).sum().item()
        else:
            correct_top5 += correct

        total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    top5_accuracy = correct_top5 / max(total, 1)

    return {"loss": avg_loss, "accuracy": accuracy, "top5_accuracy": top5_accuracy}


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
    group_name: str,
    output_path: str,
) -> None:
    """Save checkpoint with full reproducibility metadata.

    Saves both the full model state dict and the encoder-only state dict
    to support both resuming training and computing task vectors.

    Args:
        model: BEATsClassifier model.
        optimizer: Current optimizer state.
        scheduler: Current scheduler state.
        epoch: Current epoch number.
        metrics: Training/validation metrics for this epoch.
        config: Full training config dict.
        group_name: Species group identifier.
        output_path: Path to save the checkpoint.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "encoder_state_dict": model.get_encoder_state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
        "config": config,
        "group_name": group_name,
        "git_hash": get_git_hash(),
        "timestamp": datetime.now().isoformat(),
        "num_classes": model.classifier.out_features,
        "embed_dim": model.embed_dim,
    }

    torch.save(checkpoint, output_path)
    log.info(f"  Checkpoint saved: {output_path}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    group_name: str,
    config_path: str,
    gpu: int = 0,
    resume: Optional[str] = None,
    no_wandb: bool = False,
) -> None:
    """Run full fine-tuning pipeline for one species group.

    Args:
        group_name: Name of the species group (e.g., 'G1_passerines').
        config_path: Path to base YAML config.
        gpu: GPU device index.
        resume: Optional path to checkpoint to resume from.
        no_wandb: If True, skip W&B logging.
    """
    cfg = load_config(config_path)
    train_cfg = cfg["training"]
    aug_cfg = cfg["augmentation"]
    audio_cfg = cfg["audio"]

    set_seed(train_cfg["seed"])

    # Resolve paths
    root = Path(cfg["paths"]["project_root"]).resolve()
    manifest_path = str(root / "data" / "species_groups" / f"{group_name}.json")
    checkpoint_path = str(root / cfg["paths"]["beats_base_checkpoint"])
    output_dir = root / "results" / "finetuned" / group_name

    if not Path(manifest_path).exists():
        log.error(f"Manifest not found: {manifest_path}")
        sys.exit(1)
    if not Path(checkpoint_path).exists():
        log.error(f"BEATs checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Load manifest metadata
    with open(manifest_path) as f:
        manifest = json.load(f)
    metadata = manifest["metadata"]
    num_classes = metadata["n_classes"]

    log.info(f"Group: {group_name}")
    log.info(f"Classes: {num_classes}")
    log.info(f"Train: {len(manifest['train'])}, Val: {len(manifest['val'])}, "
             f"Test: {len(manifest['test'])}")
    log.info(f"Output: {output_dir}")

    # Device
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Precision
    precision_dtype = torch.bfloat16 if train_cfg["precision"] == "bf16" else torch.float16

    # W&B init
    run_name = f"beats-{group_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not no_wandb:
        try:
            import wandb
            wandb.init(
                project="bioacoustic-task-arithmetic",
                name=run_name,
                config={
                    "group": group_name,
                    "num_classes": num_classes,
                    "train_samples": len(manifest["train"]),
                    "val_samples": len(manifest["val"]),
                    **train_cfg,
                    **aug_cfg,
                    "audio": audio_cfg,
                    "git_hash": get_git_hash(),
                },
                tags=["finetune", group_name],
            )
        except Exception as e:
            log.warning(f"W&B init failed: {e}. Continuing without logging.")
            no_wandb = True

    # Model
    model = BEATsClassifier(checkpoint_path, num_classes, train_cfg["freeze_epochs"])
    model = model.to(device)
    log.info(f"Model params: {model.get_num_params():,} total, "
             f"encoder embed_dim={model.embed_dim}")

    # Freeze encoder for warmup
    model.freeze_encoder()
    log.info(f"Encoder frozen for first {train_cfg['freeze_epochs']} epochs "
             f"({model.get_num_params(trainable_only=True):,} trainable)")

    # Datasets
    train_ds = AudioDataset(
        manifest_path, split="train",
        target_sr=audio_cfg["sample_rate"],
        clip_duration_s=audio_cfg["clip_duration_s"],
    )
    val_ds = AudioDataset(
        manifest_path, split="val",
        target_sr=audio_cfg["sample_rate"],
        clip_duration_s=audio_cfg["clip_duration_s"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
        drop_last=True,
        persistent_workers=True if train_cfg["num_workers"] > 0 else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
        persistent_workers=True if train_cfg["num_workers"] > 0 else False,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    # Scheduler: OneCycleLR (cosine with linear warmup)
    total_steps = len(train_loader) * train_cfg["epochs"]
    warmup_frac = train_cfg["warmup_steps"] / max(total_steps, 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_cfg["learning_rate"],
        total_steps=total_steps,
        pct_start=min(warmup_frac, 0.3),  # cap warmup fraction
        anneal_strategy="cos",
    )

    # Loss
    criterion = nn.CrossEntropyLoss(
        label_smoothing=train_cfg["label_smoothing"],
    )

    # Grad scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()

    # Augmentation
    augment = SpecAugment(
        time_mask_param=aug_cfg["spec_augment"]["time_mask_param"],
        n_time_masks=aug_cfg["spec_augment"]["n_time_masks"],
        sample_rate=audio_cfg["sample_rate"],
    ).to(device)

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["metrics"].get("best_val_loss", float("inf"))
        log.info(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # Save config hash for verification that all runs use identical hyperparameters
    config_str = json.dumps(train_cfg, sort_keys=True) + json.dumps(aug_cfg, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    log.info(f"Config hash: {config_hash} (must match across all groups)")

    # Training loop
    log.info(f"\nStarting training: {train_cfg['epochs']} epochs, "
             f"{len(train_loader)} batches/epoch")
    log.info("=" * 60)

    for epoch in range(start_epoch, train_cfg["epochs"]):
        epoch_start = time.time()

        # Unfreeze encoder after freeze_epochs
        if epoch == train_cfg["freeze_epochs"] and model.is_frozen:
            model.unfreeze_encoder()
            # Re-create optimizer to include encoder params with fresh state
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=train_cfg["learning_rate"],
                weight_decay=train_cfg["weight_decay"],
            )
            remaining_steps = len(train_loader) * (train_cfg["epochs"] - epoch)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=train_cfg["learning_rate"],
                total_steps=remaining_steps,
                pct_start=min(0.1, 100 / max(remaining_steps, 1)),
                anneal_strategy="cos",
            )
            scaler = torch.cuda.amp.GradScaler()
            log.info(f"  Epoch {epoch}: encoder unfrozen "
                     f"({model.get_num_params(trainable_only=True):,} trainable)")

        # Train
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            scaler=scaler,
            augment=augment,
            mixup_alpha=aug_cfg["mixup_alpha"],
            grad_clip=train_cfg["grad_clip"],
            device=device,
            precision_dtype=precision_dtype,
        )

        # Validate
        val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            precision_dtype=precision_dtype,
        )

        epoch_time = time.time() - epoch_start
        frozen_str = " [FROZEN]" if model.is_frozen else ""

        log.info(
            f"  Epoch {epoch:3d}/{train_cfg['epochs']}{frozen_str} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_top5={val_metrics['top5_accuracy']:.4f} | "
            f"lr={train_metrics['lr']:.2e} | {epoch_time:.0f}s"
        )

        # W&B logging
        if not no_wandb:
            try:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "train/accuracy": train_metrics["accuracy"],
                    "train/lr": train_metrics["lr"],
                    "val/loss": val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                    "val/top5_accuracy": val_metrics["top5_accuracy"],
                    "encoder_frozen": model.is_frozen,
                    "epoch_time_s": epoch_time,
                })
            except Exception:
                pass

        # Save best model
        all_metrics = {
            "train": train_metrics,
            "val": val_metrics,
            "best_val_loss": min(best_val_loss, val_metrics["loss"]),
            "config_hash": config_hash,
        }

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=all_metrics,
                config={
                    "training": train_cfg,
                    "augmentation": aug_cfg,
                    "audio": audio_cfg,
                },
                group_name=group_name,
                output_path=str(output_dir / "best_model.pt"),
            )
        else:
            patience_counter += 1
            if patience_counter >= train_cfg["patience"]:
                log.info(f"  Early stopping at epoch {epoch} "
                         f"(patience={train_cfg['patience']})")
                break

        # Save periodic checkpoints
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=all_metrics,
                config={
                    "training": train_cfg,
                    "augmentation": aug_cfg,
                    "audio": audio_cfg,
                },
                group_name=group_name,
                output_path=str(output_dir / f"checkpoint_epoch{epoch:03d}.pt"),
            )

    # Save final model
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        metrics=all_metrics,
        config={
            "training": train_cfg,
            "augmentation": aug_cfg,
            "audio": audio_cfg,
        },
        group_name=group_name,
        output_path=str(output_dir / "final_model.pt"),
    )

    log.info("=" * 60)
    log.info(f"Training complete for {group_name}")
    log.info(f"Best val loss: {best_val_loss:.4f}")
    log.info(f"Checkpoints saved to: {output_dir}")

    if not no_wandb:
        try:
            wandb.finish()
        except Exception:
            pass


# Need to import BEATsClassifier — do it here to keep argparse fast
# when just checking --help
BEATsClassifier = None  # type: ignore


def _lazy_import() -> None:
    global BEATsClassifier
    if BEATsClassifier is None:
        from models.beats_classifier import BEATsClassifier as _cls
        BEATsClassifier = _cls


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune BEATs on a species-group dataset.",
    )
    parser.add_argument(
        "--group", type=str, required=True,
        help="Species group name (e.g., G1_passerines, ALL_birds)",
    )
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml",
        help="Path to base config YAML",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device index",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable W&B logging",
    )
    args = parser.parse_args()

    _lazy_import()
    train(
        group_name=args.group,
        config_path=args.config,
        gpu=args.gpu,
        resume=args.resume,
        no_wandb=args.no_wandb,
    )


if __name__ == "__main__":
    main()