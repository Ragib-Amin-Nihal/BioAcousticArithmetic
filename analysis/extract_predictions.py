#!/usr/bin/env python3
"""Extract and cache full prediction arrays for paper figures.

bootstrap_gap.py only saves per-sample correctness (boolean arrays).
The confusion matrix (Fig C) and calibration diagram (Fig E) need
raw predictions, labels, and probability vectors.

This script extracts them once and caches to disk. Subsequent runs
of paper_figures.py can load from cache without GPU.

Outputs per encoder (joint + merged):
    {name}_predictions.npy   — (N,) int64 predicted class indices
    {name}_labels.npy         — (N,) int64 true class indices
    {name}_probabilities.npy  — (N, C) float32 softmax probabilities

Usage:
    python analysis/extract_predictions.py --config configs/base.yaml --device cuda
    python analysis/extract_predictions.py --config configs/base.yaml --device cuda \
        --output results/gap_analysis/predictions/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ALL_GROUPS = [
    "G1_passerines", "G2_nonpasserine_birds", "G3_raptors_waterbirds",
    "G4_marine_mammals", "G5_amphibians",
]


def extract_full_predictions(
    encoder_sd: dict[str, torch.Tensor],
    base_path: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features, train probe, return predictions + labels + probs.

    Args:
        encoder_sd: Encoder state dict.
        base_path: BEATs base checkpoint path.
        train_loader: Training DataLoader (for probe training).
        test_loader: Test DataLoader (for evaluation).
        num_classes: Number of output classes.
        device: Torch device string.

    Returns:
        (predictions, labels, probabilities) — shapes (N,), (N,), (N, C).
    """
    from evaluation.linear_probe import LinearProbeEvaluator, ProbeConfig

    probe_cfg = ProbeConfig()
    evaluator = LinearProbeEvaluator(base_path, probe_cfg, device)

    # Build model with given encoder
    model = evaluator._build_model(encoder_sd, num_classes)

    # Extract features
    logger.info("    Extracting train features...")
    train_feats, train_labs = evaluator._extract_features(model, train_loader)
    logger.info("    Extracting test features...")
    test_feats, test_labs = evaluator._extract_features(model, test_loader)

    # Train probe head
    logger.info("    Training linear probe...")
    head = evaluator._train_head(train_feats, train_labs, num_classes)

    # Run inference — capture probabilities too
    head.eval()
    dataset = TensorDataset(test_feats, test_labs)
    loader = DataLoader(dataset, batch_size=probe_cfg.batch_size * 4)

    all_preds: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for feat_batch, label_batch in loader:
            feat_batch = feat_batch.to(device)
            logits = head(feat_batch)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(label_batch.numpy())

    predictions = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    probabilities = np.concatenate(all_probs)

    del model, head, train_feats, test_feats
    torch.cuda.empty_cache()

    return predictions, labels, probabilities


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract full predictions for paper figures",
    )
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["paths"]["project_root"]).resolve()
    base_path = str(root / cfg["paths"]["beats_base_checkpoint"])
    finetuned_dir = root / cfg["paths"]["finetuned"]
    tv_dir = root / cfg["paths"]["task_vectors"]
    species_groups_dir = str(root / cfg["paths"]["species_groups"])
    processed_dir = cfg["paths"]["data_processed"]

    out_dir = Path(args.output) if args.output else (
        root / cfg["paths"]["results"] / "gap_analysis" / "predictions"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    from data.dataset import build_all_group_loader

    logger.info("Building unified data loaders...")
    all_train_ld, all_test_ld, total_classes = build_all_group_loader(
        ALL_GROUPS, species_groups_dir, processed_dir,
        batch_size=cfg["evaluation"]["linear_probe"]["batch_size"],
    )

    # -----------------------------------------------------------------------
    # Joint encoder (beats-ALL)
    # -----------------------------------------------------------------------
    joint_preds_path = out_dir / "joint_all_predictions.npy"
    joint_labels_path = out_dir / "joint_all_labels.npy"
    joint_probs_path = out_dir / "joint_all_probabilities.npy"

    if joint_preds_path.exists() and joint_labels_path.exists() and joint_probs_path.exists():
        logger.info("Joint predictions already cached. Skipping.")
    else:
        logger.info("Extracting JOINT encoder predictions...")
        all_ckpt_path = finetuned_dir / "ALL_birds" / "best_model.pt"
        if not all_ckpt_path.exists():
            logger.error("beats-ALL checkpoint not found: %s", all_ckpt_path)
            sys.exit(1)

        ckpt = torch.load(str(all_ckpt_path), map_location="cpu", weights_only=False)
        joint_sd = ckpt["encoder_state_dict"]

        preds, labels, probs = extract_full_predictions(
            joint_sd, base_path, all_train_ld, all_test_ld, total_classes, args.device,
        )
        np.save(str(joint_preds_path), preds)
        np.save(str(joint_labels_path), labels)
        np.save(str(joint_probs_path), probs)
        logger.info("  Saved: %d predictions, %d classes", len(preds), probs.shape[1])
        del joint_sd, preds, labels, probs
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Merged encoder (simple average of task vectors, λ=1.0)
    # -----------------------------------------------------------------------
    merged_preds_path = out_dir / "merged_all_predictions.npy"
    merged_labels_path = out_dir / "merged_all_labels.npy"
    merged_probs_path = out_dir / "merged_all_probabilities.npy"

    if merged_preds_path.exists() and merged_labels_path.exists() and merged_probs_path.exists():
        logger.info("Merged predictions already cached. Skipping.")
    else:
        logger.info("Extracting MERGED encoder predictions...")

        # Build merged encoder: θ_base + (1/N) Σ τ_Gi  (simple average)
        # Must match bootstrap_gap.py which uses (1/n) scaling.
        base_ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
        base_sd = base_ckpt["model"]

        # Count available task vectors first for correct scaling
        available_tvs = [
            g for g in ALL_GROUPS
            if (tv_dir / f"tau_{g}.pt").exists()
        ]
        scale = 1.0 / max(len(available_tvs), 1)

        merged_sd = {k: v.clone().float() for k, v in base_sd.items()}
        for group in available_tvs:
            tv = torch.load(str(tv_dir / f"tau_{group}.pt"), map_location="cpu")
            for key in tv:
                if key in merged_sd:
                    merged_sd[key] += scale * tv[key].float()
            del tv

        logger.info("  Merged %d group task vectors (scale=1/%d=%.4f)", len(available_tvs), len(available_tvs), scale)

        preds, labels, probs = extract_full_predictions(
            merged_sd, base_path, all_train_ld, all_test_ld, total_classes, args.device,
        )
        np.save(str(merged_preds_path), preds)
        np.save(str(merged_labels_path), labels)
        np.save(str(merged_probs_path), probs)
        logger.info("  Saved: %d predictions, %d classes", len(preds), probs.shape[1])

    logger.info("=" * 60)
    logger.info("Prediction extraction complete. Output: %s", out_dir)
    logger.info("Files:")
    for p in sorted(out_dir.glob("*.npy")):
        arr = np.load(str(p))
        logger.info("  %-40s shape=%s dtype=%s", p.name, arr.shape, arr.dtype)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()