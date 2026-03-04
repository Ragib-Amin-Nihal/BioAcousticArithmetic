#!/usr/bin/env python3
"""Bootstrap confidence intervals on the composition gap.

The composition gap = acc(joint_training) - acc(merged_model).
A point estimate of 9.3% is meaningless without a CI on the *gap itself*.

This script:
    1. Loads test predictions from joint and merged encoders
    2. Computes the gap with bootstrap CIs by resampling PAIRED predictions
    3. Reports whether the 95% CI excludes zero (statistically significant gap)
    4. Separates per-group (independent probes) vs all-group (unified probe)
       to diagnose whether the gap is encoder-level or probe-level

If the per-group gap CI includes zero but the all-group gap CI excludes zero,
the encoder representations are fine — the unified linear head is the bottleneck.

Usage:
    # From saved predictions (fastest, no GPU)
    python analysis/bootstrap_gap.py \
        --predictions-dir results/composition/predictions/ \
        --output results/gap_analysis/

    # Re-extract predictions from encoders (needs GPU)
    python analysis/bootstrap_gap.py \
        --config configs/base.yaml \
        --output results/gap_analysis/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class GapCI:
    """Bootstrap CI for one accuracy gap measurement."""

    name: str
    acc_a: float  # Accuracy of model A (e.g., joint)
    acc_b: float  # Accuracy of model B (e.g., merged)
    gap: float  # acc_a - acc_b
    ci_low: float  # 2.5th percentile of gap
    ci_high: float  # 97.5th percentile of gap
    significant: bool  # CI excludes zero
    n_samples: int
    n_bootstrap: int


@dataclass
class GapAnalysis:
    """Full gap analysis across evaluation modes."""

    all_group_gap: Optional[GapCI] = None
    per_group_gaps: dict[str, GapCI] = field(default_factory=dict)
    mean_per_group_gap: Optional[GapCI] = None


# ---------------------------------------------------------------------------
# Core bootstrap
# ---------------------------------------------------------------------------

def bootstrap_paired_gap(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
    name: str = "",
) -> GapCI:
    """Compute bootstrap CI on the accuracy gap between two models.

    Uses PAIRED resampling: the same bootstrap indices are used for both
    models, preserving per-sample correlation. This is critical because
    joint and merged models often agree on easy samples and disagree on
    hard ones — ignoring this correlation inflates the CI.

    Args:
        correct_a: (N,) boolean array — model A correct per sample.
        correct_b: (N,) boolean array — model B correct per sample.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (default 0.95).
        seed: Random seed.
        name: Label for this comparison.

    Returns:
        GapCI with gap estimate and confidence interval.
    """
    assert len(correct_a) == len(correct_b), \
        f"Mismatched lengths: {len(correct_a)} vs {len(correct_b)}"

    n = len(correct_a)
    acc_a = correct_a.mean()
    acc_b = correct_b.mean()
    gap = float(acc_a - acc_b)

    rng = np.random.default_rng(seed)
    boot_gaps = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_gaps[i] = correct_a[idx].mean() - correct_b[idx].mean()

    alpha = 1.0 - confidence
    ci_low = float(np.percentile(boot_gaps, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_gaps, 100 * (1 - alpha / 2)))
    significant = (ci_low > 0) or (ci_high < 0)  # CI excludes zero

    return GapCI(
        name=name,
        acc_a=float(acc_a),
        acc_b=float(acc_b),
        gap=gap,
        ci_low=ci_low,
        ci_high=ci_high,
        significant=significant,
        n_samples=n,
        n_bootstrap=n_bootstrap,
    )


# ---------------------------------------------------------------------------
# Prediction extraction
# ---------------------------------------------------------------------------

def get_predictions(
    encoder_state_dict: dict[str, torch.Tensor],
    base_checkpoint_path: str,
    loader: DataLoader,
    num_classes: int,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract test predictions from a frozen encoder + trained linear probe.

    Runs the full linear probe pipeline (feature extraction → head training →
    inference) and returns per-sample correctness.

    Args:
        encoder_state_dict: Encoder weights.
        base_checkpoint_path: BEATs base checkpoint path.
        loader: Test DataLoader (train loader needed for probe — pass tuple).
        num_classes: Number of classes.
        device: Torch device.

    Returns:
        (predictions, labels) as numpy arrays.
    """
    from evaluation.linear_probe import LinearProbeEvaluator, ProbeConfig

    evaluator = LinearProbeEvaluator(base_checkpoint_path, ProbeConfig(), device)
    model = evaluator._build_model(encoder_state_dict, num_classes)

    # Need both train and test features
    # Caller should pass a tuple of (train_loader, test_loader)
    raise NotImplementedError(
        "Use run_gap_analysis() which handles train/test loader pairs properly"
    )


def _extract_and_probe(
    encoder_sd: dict[str, torch.Tensor],
    base_path: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    device: str,
) -> np.ndarray:
    """Extract features, train probe, return per-sample correctness array.

    Args:
        encoder_sd: Encoder state dict.
        base_path: BEATs base checkpoint.
        train_loader: Training data for probe.
        test_loader: Test data for evaluation.
        num_classes: Number of output classes.
        device: Torch device.

    Returns:
        Boolean array of shape (N_test,) indicating correctness per sample.
    """
    from evaluation.linear_probe import LinearProbeEvaluator, ProbeConfig

    evaluator = LinearProbeEvaluator(base_path, ProbeConfig(), device)
    model = evaluator._build_model(encoder_sd, num_classes)

    train_feats, train_labs = evaluator._extract_features(model, train_loader)
    test_feats, test_labs = evaluator._extract_features(model, test_loader)

    head = evaluator._train_head(train_feats, train_labs, num_classes)
    preds, labels = evaluator._evaluate_head(head, test_feats, test_labs)

    correct = (preds == labels).astype(np.bool_)

    del model, head, train_feats, test_feats
    torch.cuda.empty_cache()

    return correct


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def run_gap_analysis(
    config_path: str,
    output_dir: str,
    device: str = "cuda",
    n_bootstrap: int = 10000,
    predictions_dir: Optional[str] = None,
) -> GapAnalysis:
    """Run full bootstrap gap analysis: joint vs. best merged encoder.

    Evaluates both models on identical test sets and computes paired
    bootstrap CIs on the accuracy gap.

    Args:
        config_path: Path to base.yaml.
        output_dir: Output directory for results.
        device: Torch device.
        n_bootstrap: Bootstrap resamples.
        predictions_dir: If set, load/save predictions here to avoid re-extraction.

    Returns:
        GapAnalysis with all-group and per-group gap CIs.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["paths"]["project_root"]).resolve()
    finetuned_dir = root / cfg["paths"]["finetuned"]
    tv_dir = root / cfg["paths"]["task_vectors"]
    species_groups_dir = root / cfg["paths"]["species_groups"]
    base_path = str(root / cfg["paths"]["beats_base_checkpoint"])

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    pred_dir = Path(predictions_dir) if predictions_dir else out_path / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    groups = [
        "G1_passerines", "G2_nonpasserine_birds", "G3_raptors_waterbirds",
        "G4_marine_mammals", "G5_amphibians",
    ]

    # -----------------------------------------------------------------------
    # Load encoders
    # -----------------------------------------------------------------------
    logger.info("Loading encoders...")

    # Joint baseline
    all_ckpt = torch.load(
        str(finetuned_dir / "ALL_birds" / "best_model.pt"),
        map_location="cpu", weights_only=False,
    )
    joint_sd = all_ckpt["encoder_state_dict"]

    # Best merged: simple average (θ_base + (1/n) Σ τ_i)
    base_ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
    base_sd = base_ckpt["model"]
    task_vectors = {}
    for g in groups:
        tv_path = tv_dir / f"tau_{g}.pt"
        if tv_path.exists():
            task_vectors[g] = torch.load(str(tv_path), map_location="cpu")
    assert len(task_vectors) == len(groups), \
        f"Missing task vectors: have {list(task_vectors.keys())}"

    n = len(groups)
    merged_sd: dict[str, torch.Tensor] = {}
    for key in base_sd:
        merged_sd[key] = base_sd[key].float().clone()
        for g in groups:
            if key in task_vectors[g]:
                merged_sd[key] += (1.0 / n) * task_vectors[g][key].float()

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    from data.dataset import build_group_loaders, build_all_group_loader

    group_loaders = build_group_loaders(
        groups=groups,
        species_groups_dir=str(species_groups_dir),
        processed_dir=cfg["paths"]["data_processed"],
        batch_size=cfg["evaluation"]["linear_probe"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
    )
    all_train_ld, all_test_ld, total_classes = build_all_group_loader(
        groups=groups,
        species_groups_dir=str(species_groups_dir),
        processed_dir=cfg["paths"]["data_processed"],
        batch_size=cfg["evaluation"]["linear_probe"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
    )

    analysis = GapAnalysis()

    # -----------------------------------------------------------------------
    # All-group gap (unified label space, single probe)
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("ALL-GROUP GAP ANALYSIS (unified probe)")

    joint_all_path = pred_dir / "joint_all_correct.npy"
    merged_all_path = pred_dir / "merged_all_correct.npy"

    if joint_all_path.exists() and merged_all_path.exists():
        logger.info("  Loading cached predictions...")
        joint_correct = np.load(str(joint_all_path))
        merged_correct = np.load(str(merged_all_path))
    else:
        logger.info("  Extracting joint baseline predictions...")
        joint_correct = _extract_and_probe(
            joint_sd, base_path, all_train_ld, all_test_ld, total_classes, device,
        )
        np.save(str(joint_all_path), joint_correct)

        logger.info("  Extracting merged model predictions...")
        merged_correct = _extract_and_probe(
            merged_sd, base_path, all_train_ld, all_test_ld, total_classes, device,
        )
        np.save(str(merged_all_path), merged_correct)

    analysis.all_group_gap = bootstrap_paired_gap(
        joint_correct, merged_correct,
        n_bootstrap=n_bootstrap,
        name="all_group",
    )

    g = analysis.all_group_gap
    logger.info(
        "  Joint: %.4f  Merged: %.4f  Gap: %.4f [%.4f, %.4f] %s",
        g.acc_a, g.acc_b, g.gap, g.ci_low, g.ci_high,
        "*** SIGNIFICANT ***" if g.significant else "(not significant)",
    )

    # -----------------------------------------------------------------------
    # Per-group gaps (independent probes per group)
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PER-GROUP GAP ANALYSIS (independent probes)")

    per_group_correct_joint: dict[str, np.ndarray] = {}
    per_group_correct_merged: dict[str, np.ndarray] = {}

    for group, (train_ld, test_ld, n_cls) in group_loaders.items():
        logger.info("  Group: %s (%d classes)", group, n_cls)

        j_path = pred_dir / f"joint_{group}_correct.npy"
        m_path = pred_dir / f"merged_{group}_correct.npy"

        if j_path.exists() and m_path.exists():
            j_correct = np.load(str(j_path))
            m_correct = np.load(str(m_path))
        else:
            logger.info("    Extracting joint predictions...")
            j_correct = _extract_and_probe(
                joint_sd, base_path, train_ld, test_ld, n_cls, device,
            )
            np.save(str(j_path), j_correct)

            logger.info("    Extracting merged predictions...")
            m_correct = _extract_and_probe(
                merged_sd, base_path, train_ld, test_ld, n_cls, device,
            )
            np.save(str(m_path), m_correct)

        per_group_correct_joint[group] = j_correct
        per_group_correct_merged[group] = m_correct

        gap_ci = bootstrap_paired_gap(
            j_correct, m_correct,
            n_bootstrap=n_bootstrap,
            name=group,
        )
        analysis.per_group_gaps[group] = gap_ci
        logger.info(
            "    Joint: %.4f  Merged: %.4f  Gap: %.4f [%.4f, %.4f] %s",
            gap_ci.acc_a, gap_ci.acc_b, gap_ci.gap,
            gap_ci.ci_low, gap_ci.ci_high,
            "*** SIG ***" if gap_ci.significant else "",
        )

    # Mean per-group gap (concatenate all per-group correct arrays)
    all_j = np.concatenate(list(per_group_correct_joint.values()))
    all_m = np.concatenate(list(per_group_correct_merged.values()))
    analysis.mean_per_group_gap = bootstrap_paired_gap(
        all_j, all_m, n_bootstrap=n_bootstrap, name="mean_per_group",
    )

    # -----------------------------------------------------------------------
    # Diagnostic summary
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("COMPOSITION GAP DIAGNOSTIC")
    logger.info("-" * 60)

    ag = analysis.all_group_gap
    mg = analysis.mean_per_group_gap
    logger.info("  All-group gap (unified probe):    %.4f [%.4f, %.4f]  sig=%s",
                ag.gap, ag.ci_low, ag.ci_high, ag.significant)
    logger.info("  Mean per-group gap (indep probes): %.4f [%.4f, %.4f]  sig=%s",
                mg.gap, mg.ci_low, mg.ci_high, mg.significant)

    if ag.significant and not mg.significant:
        logger.info("  → DIAGNOSIS: Gap is PROBE-LEVEL, not encoder-level.")
        logger.info("    The merged encoder's representations are comparable to joint")
        logger.info("    training, but the unified linear head struggles with the")
        logger.info("    combined label space.")
    elif ag.significant and mg.significant:
        logger.info("  → DIAGNOSIS: Gap is ENCODER-LEVEL.")
        logger.info("    Merging degrades representations even for per-group evaluation.")
        if ag.gap > mg.gap * 1.5:
            logger.info("    However, the all-group gap is %.1f× the per-group gap,",
                        ag.gap / max(mg.gap, 1e-6))
            logger.info("    suggesting a probe-level component as well.")
    elif not ag.significant:
        logger.info("  → DIAGNOSIS: NO significant gap. Merging matches joint training.")
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    results = {
        "all_group_gap": asdict(analysis.all_group_gap) if analysis.all_group_gap else None,
        "per_group_gaps": {k: asdict(v) for k, v in analysis.per_group_gaps.items()},
        "mean_per_group_gap": asdict(analysis.mean_per_group_gap) if analysis.mean_per_group_gap else None,
        "n_bootstrap": n_bootstrap,
    }
    out_file = out_path / "gap_analysis.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_file)

    return analysis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap CI on the composition gap (joint vs. merged)",
    )
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--output", type=str, default="results/gap_analysis/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument(
        "--predictions-dir", type=str, default=None,
        help="Directory with cached per-sample predictions (.npy). "
             "If provided, skips feature extraction (no GPU needed).",
    )
    args = parser.parse_args()

    run_gap_analysis(
        config_path=args.config,
        output_dir=args.output,
        device=args.device,
        n_bootstrap=args.n_bootstrap,
        predictions_dir=args.predictions_dir,
    )


if __name__ == "__main__":
    main()