#!/usr/bin/env python3
"""Supplementary Experiment: Continual Learning Comparison.

Tests whether task arithmetic avoids catastrophic forgetting when adding a new
species group to an existing multi-group system.

Scenario: an existing system covers G1-G4 (birds + marine mammals).
G5 (amphibians) arrives as new data. Three strategies to incorporate it:

    Method A  (oracle):      Retrain on all G1-G5 data from scratch     = beats-ALL
    Method B  (fine-tune):   Take beats-G1234, fine-tune encoder on G5  (forgetting risk)
    Method C  (merge):       θ_base + ½(τ_G1234 + τ_G5)               (no forgetting by construction)
    Method C' (merge-indiv): θ_base + (1/5)Σ τ_Gi for i=1..5          (existing Exp2 merge)

All methods evaluated identically via linear probing on:
    - Per-group test sets (G1, G2, G3, G4, G5 individually)
    - All-group unified test set (661 classes)

The key measurement is catastrophic forgetting: how much does G1-G4 performance
drop when we add G5 via fine-tuning vs. merging?

Prerequisites:
    - G1-G5 individual fine-tuned models (Phase 1)
    - beats-ALL joint model (Phase 1)
    - G1234_joint manifest + model (created/trained by this script)
    - Task vectors for all groups

Usage:
    # Phase 0: Create G1234_joint manifest
    python analysis/continual_learning.py --config configs/base.yaml --phase 0

    # Phase 1: Train G1234_joint model (will call train.py)
    python analysis/continual_learning.py --config configs/base.yaml --phase 1

    # Phase 2: Fine-tune G1234_joint on G5 data
    python analysis/continual_learning.py --config configs/base.yaml --phase 2 --device cuda

    # Phase 3: Evaluate all methods
    python analysis/continual_learning.py --config configs/base.yaml --phase 3 --device cuda

    # Run all phases sequentially
    python analysis/continual_learning.py --config configs/base.yaml --phase 0 1 2 3 --device cuda
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Groups in the "existing system" vs the "new arrival"
EXISTING_GROUPS = ["G1_passerines", "G2_nonpasserine_birds",
                   "G3_raptors_waterbirds", "G4_marine_mammals"]
NEW_GROUP = "G5_amphibians"
ALL_GROUPS = EXISTING_GROUPS + [NEW_GROUP]
JOINT_GROUP_NAME = "G1234_joint"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class MethodResult:
    """Per-method evaluation result."""
    method: str
    description: str
    per_group_accuracy: dict[str, float] = field(default_factory=dict)
    per_group_f1: dict[str, float] = field(default_factory=dict)
    all_group_accuracy: float = 0.0
    all_group_f1: float = 0.0
    old_group_mean_accuracy: float = 0.0  # Mean over G1-G4
    new_group_accuracy: float = 0.0  # G5 only
    forgetting: float = 0.0  # Drop in old-group accuracy vs baseline


@dataclass
class ContinualLearningResults:
    """Full experiment results."""
    g1234_baseline: dict[str, float] = field(default_factory=dict)  # Per-group acc of G1234 model
    methods: list[MethodResult] = field(default_factory=list)
    finetune_per_epoch: list[dict[str, Any]] = field(default_factory=list)  # Epoch-by-epoch tracking
    total_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Phase 0: Create G1234_joint manifest
# ---------------------------------------------------------------------------

def create_g1234_manifest(config_path: str) -> Path:
    """Create a joint manifest for G1-G4 by merging individual group manifests.

    The merged manifest has a unified label space: G1 labels [0, n1),
    G2 labels [n1, n1+n2), etc. This is the same approach as ALL_birds
    but excludes G5.

    Args:
        config_path: Path to base.yaml.

    Returns:
        Path to the created manifest.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["paths"]["project_root"]).resolve()
    species_groups_dir = root / cfg["paths"]["species_groups"]
    output_path = species_groups_dir / f"{JOINT_GROUP_NAME}.json"

    if output_path.exists():
        logger.info("G1234_joint manifest already exists: %s", output_path)
        return output_path

    all_train: list[dict[str, Any]] = []
    all_val: list[dict[str, Any]] = []
    all_test: list[dict[str, Any]] = []
    all_species: list[str] = []
    species2label: dict[str, int] = {}
    label_offset = 0

    for group in EXISTING_GROUPS:
        manifest_path = species_groups_dir / f"{group}.json"
        if not manifest_path.exists():
            logger.error("Group manifest not found: %s", manifest_path)
            sys.exit(1)

        with open(manifest_path) as f:
            manifest = json.load(f)
        meta = manifest["metadata"]
        n_classes = meta["n_classes"]

        logger.info(
            "  %s: %d classes, %d train, %d val, %d test",
            group, n_classes,
            len(manifest["train"]), len(manifest["val"]), len(manifest["test"]),
        )

        # Remap labels to unified space
        for split_name, target_list in [
            ("train", all_train), ("val", all_val), ("test", all_test),
        ]:
            for entry in manifest[split_name]:
                target_list.append({
                    "path": entry["path"],
                    "species": entry["species"],
                    "label": entry["label"] + label_offset,
                    "source_group": group,
                })

        # Track species
        sp_list = meta.get("species_list", sorted(meta["species2label"].keys()))
        for sp in sp_list:
            global_label = meta["species2label"][sp] + label_offset
            species2label[sp] = global_label
            all_species.append(sp)

        label_offset += n_classes

    total_classes = label_offset

    manifest = {
        "metadata": {
            "group": JOINT_GROUP_NAME,
            "description": "Joint G1-G4 (without G5 amphibians) for continual learning baseline",
            "n_classes": total_classes,
            "n_species": len(all_species),
            "species_list": all_species,
            "species2label": species2label,
            "label2species": {v: k for k, v in species2label.items()},
            "n_train": len(all_train),
            "n_val": len(all_val),
            "n_test": len(all_test),
            "source_groups": EXISTING_GROUPS,
        },
        "train": all_train,
        "val": all_val,
        "test": all_test,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "Created %s: %d classes, %d train, %d val, %d test",
        output_path, total_classes, len(all_train), len(all_val), len(all_test),
    )
    return output_path


# ---------------------------------------------------------------------------
# Phase 1: Train G1234_joint model
# ---------------------------------------------------------------------------

def train_g1234_joint(config_path: str, gpu: int = 0) -> Path:
    """Train the G1234_joint model by calling train.py.

    Args:
        config_path: Path to base.yaml.
        gpu: GPU device index.

    Returns:
        Path to the trained checkpoint.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg["paths"]["project_root"]).resolve()
    ckpt_dir = root / cfg["paths"]["finetuned"] / JOINT_GROUP_NAME
    ckpt_path = ckpt_dir / "best_model.pt"

    if ckpt_path.exists():
        logger.info("G1234_joint checkpoint already exists: %s", ckpt_path)
        return ckpt_path

    logger.info("Training G1234_joint model via train.py...")
    cmd = [
        sys.executable, str(root / "train.py"),
        "--group", JOINT_GROUP_NAME,
        "--config", config_path,
        "--gpu", str(gpu),
    ]
    logger.info("  Command: %s", " ".join(cmd))

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("Training failed with return code %d", result.returncode)
        sys.exit(1)

    if not ckpt_path.exists():
        logger.error("Expected checkpoint not found after training: %s", ckpt_path)
        sys.exit(1)

    logger.info("G1234_joint training complete: %s", ckpt_path)
    return ckpt_path


# ---------------------------------------------------------------------------
# Phase 2: Fine-tune G1234_joint on G5 data
# ---------------------------------------------------------------------------

def finetune_on_new_group(
    config_path: str,
    device: str = "cuda",
    epochs: int = 20,
    learning_rates: Optional[list[float]] = None,
    output_dir: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Fine-tune the G1234_joint model's encoder on G5 data.

    This simulates the continual learning approach: take existing model,
    train on new data. We track per-epoch performance on BOTH old and new
    groups to measure forgetting dynamics.

    We test multiple learning rates because the optimal LR for continual
    learning may differ from the original LR. Lower LR = less forgetting
    but slower learning.

    Args:
        config_path: Path to base.yaml.
        device: Torch device.
        epochs: Number of fine-tuning epochs.
        learning_rates: LR values to test. Default: [1e-5, 5e-6, 1e-6].
        output_dir: Where to save fine-tuned checkpoints.

    Returns:
        List of per-LR results with epoch-by-epoch metrics.
    """
    if learning_rates is None:
        learning_rates = [1e-5, 5e-6, 1e-6]

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["paths"]["project_root"]).resolve()
    base_path = str(root / cfg["paths"]["beats_base_checkpoint"])
    g1234_ckpt_path = root / cfg["paths"]["finetuned"] / JOINT_GROUP_NAME / "best_model.pt"
    species_groups_dir = str(root / cfg["paths"]["species_groups"])
    processed_dir = cfg["paths"]["data_processed"]

    if output_dir is None:
        output_dir = str(root / "results" / "continual_learning")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load G1234_joint checkpoint
    logger.info("Loading G1234_joint checkpoint...")
    g1234_ckpt = torch.load(str(g1234_ckpt_path), map_location="cpu", weights_only=False)
    g1234_encoder_sd = g1234_ckpt["encoder_state_dict"]

    # Load G5 manifest for training
    g5_manifest_path = Path(species_groups_dir) / f"{NEW_GROUP}.json"
    with open(g5_manifest_path) as f:
        g5_manifest = json.load(f)
    g5_n_classes = g5_manifest["metadata"]["n_classes"]

    # Build G5 data loaders for fine-tuning
    from data.dataset import AudioDataset
    from torch.utils.data import DataLoader

    def _make_loader(split: str, shuffle: bool) -> DataLoader:
        entries = g5_manifest[split]
        files = [(e["path"], e["label"]) for e in entries]
        return DataLoader(
            AudioDataset(files),
            batch_size=cfg["training"]["batch_size"],
            shuffle=shuffle,
            num_workers=cfg["training"]["num_workers"],
            pin_memory=True,
        )

    g5_train_loader = _make_loader("train", shuffle=True)
    g5_val_loader = _make_loader("val", shuffle=False)

    # Build evaluation loaders for all groups
    from data.dataset import build_group_loaders, build_all_group_loader

    logger.info("Building evaluation loaders for all groups...")
    group_loaders = build_group_loaders(
        ALL_GROUPS, species_groups_dir, processed_dir,
        batch_size=cfg["evaluation"]["linear_probe"]["batch_size"],
    )

    from evaluation.linear_probe import LinearProbeEvaluator, ProbeConfig
    probe_cfg = ProbeConfig(
        learning_rate=cfg["evaluation"]["linear_probe"]["learning_rate"],
        epochs=cfg["evaluation"]["linear_probe"]["epochs"],
        batch_size=cfg["evaluation"]["linear_probe"]["batch_size"],
        bootstrap_n_resamples=100,  # Reduced for speed during tracking
        bootstrap_confidence=0.95,
    )
    evaluator = LinearProbeEvaluator(base_path, probe_cfg, device)

    # Lazy import
    from models.beats_classifier import BEATsClassifier

    all_lr_results: list[dict[str, Any]] = []

    for lr in learning_rates:
        logger.info("=" * 60)
        logger.info("Fine-tuning G1234 on G5 with LR=%.1e", lr)

        # Build model from G1234_joint encoder + fresh G5 head
        model = BEATsClassifier(base_path, g5_n_classes, freeze_epochs=0)
        model.load_encoder_state_dict(g1234_encoder_sd)
        model = model.to(device)
        model.unfreeze_encoder()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=cfg["training"]["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr,
            total_steps=epochs * len(g5_train_loader),
        )
        criterion = nn.CrossEntropyLoss(
            label_smoothing=cfg["training"]["label_smoothing"],
        )
        scaler = torch.amp.GradScaler() if device != "cpu" else None
        precision_dtype = torch.bfloat16

        lr_result = {
            "learning_rate": lr,
            "epochs": [],
        }

        for epoch in range(epochs):
            # --- Train one epoch ---
            model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_idx, (audio, labels) in enumerate(g5_train_loader):
                audio, labels = audio.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.amp.autocast(device_type=device.split(":")[0], dtype=precision_dtype):
                    logits = model(audio)
                    loss = criterion(logits, labels)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
                    optimizer.step()

                scheduler.step()

                epoch_loss += loss.item() * audio.size(0)
                preds = logits.argmax(dim=-1)
                epoch_correct += (preds == labels).sum().item()
                epoch_total += audio.size(0)

            train_acc = epoch_correct / max(epoch_total, 1)
            train_loss = epoch_loss / max(epoch_total, 1)

            # --- Evaluate via linear probes on all groups ---
            encoder_sd = model.get_encoder_state_dict()

            epoch_metrics: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc_g5_head": train_acc,
                "per_group": {},
            }

            old_accs = []
            for group_name in ALL_GROUPS:
                if group_name not in group_loaders:
                    continue
                train_ld, test_ld, n_cls = group_loaders[group_name]
                probe_result = evaluator.evaluate(encoder_sd, train_ld, test_ld, n_cls)
                epoch_metrics["per_group"][group_name] = probe_result.accuracy
                if group_name in EXISTING_GROUPS:
                    old_accs.append(probe_result.accuracy)

            epoch_metrics["old_group_mean"] = float(np.mean(old_accs)) if old_accs else 0.0
            epoch_metrics["new_group"] = epoch_metrics["per_group"].get(NEW_GROUP, 0.0)

            lr_result["epochs"].append(epoch_metrics)

            logger.info(
                "  Epoch %2d/%d: loss=%.4f, G5_head_acc=%.4f, "
                "old_mean=%.4f, G5_probe=%.4f",
                epoch + 1, epochs, train_loss, train_acc,
                epoch_metrics["old_group_mean"], epoch_metrics["new_group"],
            )

            # Early stop if old groups completely collapse
            if epoch >= 3 and epoch_metrics["old_group_mean"] < 0.20:
                logger.warning("  Old group accuracy collapsed. Stopping early.")
                break

        # Save the best encoder (best G5 accuracy without catastrophic forgetting)
        # Heuristic: best epoch = max(0.5 * old_mean + 0.5 * new_acc)
        best_epoch_idx = max(
            range(len(lr_result["epochs"])),
            key=lambda i: (
                0.5 * lr_result["epochs"][i]["old_group_mean"]
                + 0.5 * lr_result["epochs"][i]["new_group"]
            ),
        )
        lr_result["best_epoch"] = best_epoch_idx
        lr_result["best_old_mean"] = lr_result["epochs"][best_epoch_idx]["old_group_mean"]
        lr_result["best_new_acc"] = lr_result["epochs"][best_epoch_idx]["new_group"]

        logger.info(
            "  Best epoch: %d (old_mean=%.4f, new=%.4f)",
            best_epoch_idx + 1,
            lr_result["best_old_mean"],
            lr_result["best_new_acc"],
        )

        all_lr_results.append(lr_result)

        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    # Save fine-tuning results
    ft_results_path = out_path / "finetune_results.json"
    with open(ft_results_path, "w") as f:
        json.dump(all_lr_results, f, indent=2)
    logger.info("Fine-tuning results saved to %s", ft_results_path)

    return all_lr_results


# ---------------------------------------------------------------------------
# Phase 3: Evaluate all methods
# ---------------------------------------------------------------------------

def evaluate_all_methods(
    config_path: str,
    device: str = "cuda",
    output_dir: Optional[str] = None,
) -> ContinualLearningResults:
    """Evaluate all continual learning strategies.

    Methods compared:
        baseline: G1234_joint model (before adding G5) — reference for forgetting
        method_a: beats-ALL (retrain on G1-G5 from scratch) — oracle
        method_b: Best fine-tuned G1234+G5 model (from Phase 2)
        method_c: θ_base + τ_G1234 + τ_G5 (task arithmetic with joint TV)
        method_c_indiv: θ_base + Σ τ_Gi for i=1..5 (existing Exp2 merge)

    Args:
        config_path: Path to base.yaml.
        device: Torch device.
        output_dir: Where to save results.

    Returns:
        ContinualLearningResults.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["paths"]["project_root"]).resolve()
    base_path = str(root / cfg["paths"]["beats_base_checkpoint"])
    finetuned_dir = root / cfg["paths"]["finetuned"]
    tv_dir = root / cfg["paths"]["task_vectors"]
    species_groups_dir = str(root / cfg["paths"]["species_groups"])
    processed_dir = cfg["paths"]["data_processed"]

    if output_dir is None:
        output_dir = str(root / "results" / "continual_learning")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = ContinualLearningResults()
    t0 = time.time()

    # Setup evaluation
    from data.dataset import build_group_loaders, build_all_group_loader
    from evaluation.linear_probe import LinearProbeEvaluator, ProbeConfig
    from merging.task_vectors import apply_task_vector, compose_task_vectors

    probe_cfg = ProbeConfig(
        learning_rate=cfg["evaluation"]["linear_probe"]["learning_rate"],
        epochs=cfg["evaluation"]["linear_probe"]["epochs"],
        batch_size=cfg["evaluation"]["linear_probe"]["batch_size"],
        bootstrap_n_resamples=cfg["evaluation"]["bootstrap_n_resamples"],
        bootstrap_confidence=cfg["evaluation"]["bootstrap_confidence"],
    )
    evaluator = LinearProbeEvaluator(base_path, probe_cfg, device)

    # Build evaluation loaders
    logger.info("Building evaluation loaders...")
    group_loaders = build_group_loaders(
        ALL_GROUPS, species_groups_dir, processed_dir,
        batch_size=cfg["evaluation"]["linear_probe"]["batch_size"],
    )
    all_train_ld, all_test_ld, total_classes = build_all_group_loader(
        ALL_GROUPS, species_groups_dir, processed_dir,
        batch_size=cfg["evaluation"]["linear_probe"]["batch_size"],
    )

    def _evaluate_encoder(
        encoder_sd: dict[str, torch.Tensor],
        method_name: str,
        description: str,
    ) -> MethodResult:
        """Evaluate one encoder on all groups."""
        mr = MethodResult(method=method_name, description=description)

        # Per-group evaluation
        old_accs = []
        for group in ALL_GROUPS:
            if group not in group_loaders:
                continue
            train_ld, test_ld, n_cls = group_loaders[group]
            probe_result = evaluator.evaluate(encoder_sd, train_ld, test_ld, n_cls)
            mr.per_group_accuracy[group] = probe_result.accuracy
            mr.per_group_f1[group] = probe_result.macro_f1
            if group in EXISTING_GROUPS:
                old_accs.append(probe_result.accuracy)
            if group == NEW_GROUP:
                mr.new_group_accuracy = probe_result.accuracy

        mr.old_group_mean_accuracy = float(np.mean(old_accs)) if old_accs else 0.0

        # All-group unified evaluation
        all_result = evaluator.evaluate(encoder_sd, all_train_ld, all_test_ld, total_classes)
        mr.all_group_accuracy = all_result.accuracy
        mr.all_group_f1 = all_result.macro_f1

        logger.info(
            "  %s: old_mean=%.4f, new=%.4f, all=%.4f",
            method_name, mr.old_group_mean_accuracy,
            mr.new_group_accuracy, mr.all_group_accuracy,
        )
        return mr

    # -------------------------------------------------------------------
    # Baseline: G1234_joint model (before G5)
    # -------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Evaluating baseline: G1234_joint model")

    g1234_ckpt_path = finetuned_dir / JOINT_GROUP_NAME / "best_model.pt"
    if g1234_ckpt_path.exists():
        ckpt = torch.load(str(g1234_ckpt_path), map_location="cpu", weights_only=False)
        g1234_encoder = ckpt["encoder_state_dict"]
        baseline = _evaluate_encoder(g1234_encoder, "baseline_g1234", "G1234 model (before G5)")
        results.g1234_baseline = baseline.per_group_accuracy.copy()
        results.methods.append(baseline)
        baseline_old_mean = baseline.old_group_mean_accuracy
    else:
        logger.error("G1234_joint checkpoint not found. Run phase 1 first.")
        baseline_old_mean = 0.0

    # -------------------------------------------------------------------
    # Method A: Retrain all (beats-ALL)
    # -------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Method A: Retrain all (beats-ALL)")

    all_ckpt_path = finetuned_dir / "ALL_birds" / "best_model.pt"
    if all_ckpt_path.exists():
        ckpt = torch.load(str(all_ckpt_path), map_location="cpu", weights_only=False)
        all_encoder = ckpt["encoder_state_dict"]
        method_a = _evaluate_encoder(all_encoder, "retrain_all", "Retrain from scratch on G1-G5")
        method_a.forgetting = baseline_old_mean - method_a.old_group_mean_accuracy
        results.methods.append(method_a)
        del all_encoder
    else:
        logger.warning("beats-ALL checkpoint not found. Skipping Method A.")

    # -------------------------------------------------------------------
    # Method B: Best fine-tuned model (from Phase 2)
    # -------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Method B: Fine-tune G1234 on G5")

    ft_results_path = out_path / "finetune_results.json"
    if ft_results_path.exists():
        with open(ft_results_path) as f:
            ft_results = json.load(f)

        # Find best LR/epoch combination
        best_score = -1.0
        best_lr_result = None
        for lr_result in ft_results:
            for ep in lr_result["epochs"]:
                score = 0.5 * ep["old_group_mean"] + 0.5 * ep["new_group"]
                if score > best_score:
                    best_score = score
                    best_lr_result = lr_result
                    best_epoch_metrics = ep

        if best_lr_result is not None:
            # Re-run fine-tuning to get the encoder at the best epoch
            # (or reconstruct from saved per-epoch data)
            # For efficiency, use the per-epoch metrics already collected
            method_b = MethodResult(
                method="finetune_joint",
                description=(
                    f"Fine-tune G1234 on G5 "
                    f"(LR={best_lr_result['learning_rate']:.0e}, "
                    f"epoch={best_lr_result['best_epoch']+1})"
                ),
                per_group_accuracy=best_epoch_metrics["per_group"],
                old_group_mean_accuracy=best_epoch_metrics["old_group_mean"],
                new_group_accuracy=best_epoch_metrics["new_group"],
            )
            method_b.forgetting = baseline_old_mean - method_b.old_group_mean_accuracy
            results.methods.append(method_b)
            results.finetune_per_epoch = ft_results

            logger.info(
                "  Best fine-tune: LR=%.0e, epoch=%d, old_mean=%.4f, new=%.4f, "
                "forgetting=%.4f",
                best_lr_result["learning_rate"],
                best_lr_result["best_epoch"] + 1,
                method_b.old_group_mean_accuracy,
                method_b.new_group_accuracy,
                method_b.forgetting,
            )
    else:
        logger.warning("Fine-tuning results not found. Run phase 2 first.")

    # -------------------------------------------------------------------
    # Method C: Task arithmetic (θ_base + τ_G1234 + τ_G5)
    # -------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Method C: Task arithmetic (τ_G1234 + τ_G5)")

    tv_g1234_path = tv_dir / f"tau_{JOINT_GROUP_NAME}.pt"
    tv_g5_path = tv_dir / f"tau_{NEW_GROUP}.pt"

    if not tv_g1234_path.exists():
        # Compute from checkpoint
        logger.info("  Computing τ_G1234 from checkpoint...")
        from merging.task_vectors import compute_task_vector
        compute_task_vector(str(g1234_ckpt_path), base_path, str(tv_g1234_path))

    if tv_g1234_path.exists() and tv_g5_path.exists():
        tv_g1234 = torch.load(str(tv_g1234_path), map_location="cpu")
        tv_g5 = torch.load(str(tv_g5_path), map_location="cpu")

        # Merge: θ_base + (1/2)(τ_G1234 + τ_G5) — simple average of 2 TVs
        merged_tv: dict[str, torch.Tensor] = {}
        scale = 0.5  # 1/N where N=2 task vectors
        for key in tv_g1234:
            merged_tv[key] = scale * tv_g1234[key].float() + scale * tv_g5.get(key, torch.zeros_like(tv_g1234[key])).float()

        from merging.task_vectors import apply_merged_tv_to_base
        merged_encoder = apply_merged_tv_to_base(base_path, merged_tv)

        method_c = _evaluate_encoder(
            merged_encoder,
            "merge_joint_tv",
            "Simple avg: θ_base + ½(τ_G1234 + τ_G5)",
        )
        method_c.forgetting = baseline_old_mean - method_c.old_group_mean_accuracy
        results.methods.append(method_c)
        del merged_encoder, merged_tv
    else:
        logger.warning("Task vectors not found for G1234 or G5.")

    # -------------------------------------------------------------------
    # Method C': Individual task vector merge (existing Exp2 approach)
    # -------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Method C': Individual TV merge (Σ τ_Gi)")

    individual_tvs: dict[str, dict[str, torch.Tensor]] = {}
    all_found = True
    for group in ALL_GROUPS:
        tv_path = tv_dir / f"tau_{group}.pt"
        if tv_path.exists():
            individual_tvs[group] = torch.load(str(tv_path), map_location="cpu")
        else:
            logger.warning("  Task vector not found: %s", tv_path)
            all_found = False

    if all_found and len(individual_tvs) == len(ALL_GROUPS):
        # Simple average: θ_base + (1/N) Σ τ_Gi — matches Experiment 2 / bootstrap
        n_tvs = len(ALL_GROUPS)
        scale = 1.0 / n_tvs
        merged_tv = {}
        for key in individual_tvs[ALL_GROUPS[0]]:
            merged_tv[key] = scale * sum(
                individual_tvs[g][key].float()
                for g in ALL_GROUPS
                if key in individual_tvs[g]
            )

        from merging.task_vectors import apply_merged_tv_to_base
        merged_encoder = apply_merged_tv_to_base(base_path, merged_tv)

        method_c_indiv = _evaluate_encoder(
            merged_encoder,
            "merge_individual_tvs",
            "Simple avg: θ_base + (1/5) Σ τ_Gi (i=1..5)",
        )
        method_c_indiv.forgetting = baseline_old_mean - method_c_indiv.old_group_mean_accuracy
        results.methods.append(method_c_indiv)
        del merged_encoder
    else:
        logger.warning("Not all individual task vectors found.")

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    results.total_time_s = time.time() - t0

    results_dict = {
        "experiment": "continual_learning",
        "existing_groups": EXISTING_GROUPS,
        "new_group": NEW_GROUP,
        "total_time_s": results.total_time_s,
        "g1234_baseline": results.g1234_baseline,
        "methods": [asdict(m) for m in results.methods],
        "finetune_per_epoch": results.finetune_per_epoch,
    }

    out_file = out_path / "continual_learning_results.json"
    with open(out_file, "w") as f:
        json.dump(results_dict, f, indent=2)
    logger.info("Results saved to %s", out_file)

    # Summary
    logger.info("=" * 60)
    logger.info("CONTINUAL LEARNING SUMMARY")
    logger.info("-" * 60)
    logger.info("%-25s  old_mean  new_acc  forgetting  all_acc", "Method")
    for m in results.methods:
        logger.info(
            "%-25s  %.4f   %.4f    %+.4f     %.4f",
            m.method[:25],
            m.old_group_mean_accuracy,
            m.new_group_accuracy,
            m.forgetting,
            m.all_group_accuracy,
        )
    logger.info("=" * 60)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continual Learning Comparison: adding G5 to G1-G4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phases:
  0 = Create G1234_joint manifest
  1 = Train G1234_joint model (calls train.py, ~4-5 GPU-hours)
  2 = Fine-tune G1234 on G5 data (tests multiple LRs, ~3-4 GPU-hours)
  3 = Evaluate all methods via linear probing (~1-2 GPU-hours)

Examples:
  # Full pipeline
  python analysis/continual_learning.py --config configs/base.yaml --phase 0 1 2 3 --device cuda

  # Just create manifest + train (no GPU for manifest, GPU for training)
  python analysis/continual_learning.py --config configs/base.yaml --phase 0 1

  # Just evaluate (after training is done)
  python analysis/continual_learning.py --config configs/base.yaml --phase 3 --device cuda
        """,
    )
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument(
        "--phase", nargs="+", type=int, required=True,
        help="Phase(s) to run: 0=manifest, 1=train, 2=finetune, 3=evaluate",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--finetune-epochs", type=int, default=20,
        help="Epochs for fine-tuning in Phase 2.",
    )
    parser.add_argument(
        "--finetune-lrs", nargs="*", type=float, default=None,
        help="Learning rates to test in Phase 2. Default: 1e-5 5e-6 1e-6",
    )
    args = parser.parse_args()

    phases = set(args.phase)

    if 0 in phases:
        logger.info("=" * 60)
        logger.info("PHASE 0: Create G1234_joint manifest")
        create_g1234_manifest(args.config)

    if 1 in phases:
        logger.info("=" * 60)
        logger.info("PHASE 1: Train G1234_joint model")
        train_g1234_joint(args.config, gpu=args.gpu)

    if 2 in phases:
        logger.info("=" * 60)
        logger.info("PHASE 2: Fine-tune G1234 on G5")
        finetune_on_new_group(
            args.config,
            device=args.device,
            epochs=args.finetune_epochs,
            learning_rates=args.finetune_lrs,
            output_dir=args.output,
        )

    if 3 in phases:
        logger.info("=" * 60)
        logger.info("PHASE 3: Evaluate all methods")
        evaluate_all_methods(args.config, device=args.device, output_dir=args.output)


if __name__ == "__main__":
    main()