#!/usr/bin/env python3
"""Experiment 4: Focal→Soundscape Domain Negation (RQ4) — with resume support.

Tests whether subtracting a "focal-domain" task vector removes recording-quality
bias and improves soundscape performance. This is the most practically impactful
experiment: focal→soundscape domain shift degrades every deployed bioacoustic model.

Protocol:
    1. Load τ_focal (focal-domain task vector)
    2. Load encoders for soundscape and mixed models
    3. Apply negation at varying strengths:
         θ_negated = θ_soundscape − β · τ_focal    (β ∈ {0, 0.1, ..., 1.0, 1.5, 2.0})
         θ_negated_mixed = θ_mixed − β · τ_focal
    4. Evaluate on both focal test set and soundscape test set at each β
    5. CRITICAL: Random-vector control with matched per-layer L2 norms
    6. Plot Pareto frontier: focal accuracy vs soundscape accuracy across β
    7. Layer-wise analysis: where is the focal bias concentrated?

The random-vector control is MANDATORY (key constraint). If a random vector
of the same magnitude also improves soundscape performance, the improvement
is not specific to focal-domain removal — it's just a regularization effect.

Resume support:
    Results are saved incrementally after every completed sweep. Use --resume
    to reload partial results and skip already-completed trials. This allows
    the experiment to survive 24-hour server time limits.

Usage:
    # Full run
    python evaluation/domain_negation.py \
        --config configs/base.yaml \
        --output results/negation/ \
        --device cuda

    # Resume after interruption
    python evaluation/domain_negation.py \
        --config configs/base.yaml \
        --output results/negation/ \
        --device cuda --resume

    # Run only specific phases
    python evaluation/domain_negation.py \
        --config configs/base.yaml \
        --output results/negation/ \
        --device cuda --resume --phases 4

    # Quick test with fewer β values
    python evaluation/domain_negation.py \
        --config configs/base.yaml \
        --output results/negation/ \
        --beta-grid 0.0 0.5 1.0

    # Start random controls from seed index 2 (0-indexed) to resume mid-phase-4
    python evaluation/domain_negation.py \
        --config configs/base.yaml \
        --output results/negation/ \
        --device cuda --resume --phases 4 --random-seed-start 2
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.dataset import build_group_loaders
from evaluation.linear_probe import LinearProbeEvaluator, ProbeConfig, ProbeResult
from merging.task_vectors import (
    apply_task_vector,
    compute_task_vector,
    negate_task_vector,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class NegationTrialResult:
    """Result of one negation trial (one β value, one vector type)."""

    vector_type: str  # "focal", "random_control", or specific method
    source_model: str  # "soundscape" or "mixed"
    beta: float
    focal_accuracy: float  # Performance on focal test set
    soundscape_accuracy: float  # Performance on soundscape test set
    focal_f1: float = 0.0
    soundscape_f1: float = 0.0
    eval_time_s: float = 0.0


@dataclass
class LayerAnalysis:
    """Per-layer magnitude analysis of the focal task vector."""

    layer_name: str
    l2_norm: float
    l1_norm: float
    fraction_of_total: float  # This layer's L2 as fraction of total L2
    mean_abs: float
    max_abs: float


@dataclass
class NegationExperimentResult:
    """Full Experiment 4 results."""

    trials: list[NegationTrialResult] = field(default_factory=list)
    focal_baseline_accuracy: Optional[float] = None
    soundscape_baseline_accuracy: Optional[float] = None
    mixed_baseline_accuracy: Optional[float] = None
    # Additional baseline cross-eval
    focal_on_soundscape_accuracy: Optional[float] = None
    soundscape_on_focal_accuracy: Optional[float] = None
    mixed_on_focal_accuracy: Optional[float] = None
    mixed_on_soundscape_accuracy: Optional[float] = None
    layer_analysis: list[LayerAnalysis] = field(default_factory=list)
    n_random_seeds: int = 0


# ---------------------------------------------------------------------------
# Checkpoint I/O (incremental saving and resume)
# ---------------------------------------------------------------------------

CHECKPOINT_FILENAME = "negation_checkpoint.json"


def _trial_key(t: NegationTrialResult) -> tuple[str, str, float]:
    """Unique key for deduplicating trials."""
    return (t.vector_type, t.source_model, round(t.beta, 4))


def _trial_key_from_dict(d: dict) -> tuple[str, str, float]:
    """Unique key from a trial dict (as stored in JSON)."""
    return (d["vector_type"], d["source_model"], round(d["beta"], 4))


def save_checkpoint(
    experiment: NegationExperimentResult,
    output_path: Path,
    beta_grid: list[float],
    n_random_seeds: int,
    completed_phases: set[int],
    elapsed_s: float,
) -> None:
    """Save incremental checkpoint to disk.

    Args:
        experiment: Current experiment state.
        output_path: Output directory.
        beta_grid: Full beta grid for reference.
        n_random_seeds: Total random seeds planned.
        completed_phases: Set of fully completed phase numbers.
        elapsed_s: Total elapsed time so far.
    """
    ckpt = {
        "experiment": "domain_negation",
        "beta_grid": beta_grid,
        "n_random_seeds": n_random_seeds,
        "completed_phases": sorted(completed_phases),
        "total_trials": len(experiment.trials),
        "elapsed_s": elapsed_s,
        "baselines": {
            "focal_model_accuracy": experiment.focal_baseline_accuracy,
            "soundscape_model_accuracy": experiment.soundscape_baseline_accuracy,
            "mixed_model_accuracy": experiment.mixed_baseline_accuracy,
            "focal_on_soundscape_accuracy": experiment.focal_on_soundscape_accuracy,
            "soundscape_on_focal_accuracy": experiment.soundscape_on_focal_accuracy,
            "mixed_on_focal_accuracy": experiment.mixed_on_focal_accuracy,
            "mixed_on_soundscape_accuracy": experiment.mixed_on_soundscape_accuracy,
        },
        "layer_analysis": [asdict(la) for la in experiment.layer_analysis[:20]],
        "trials": [asdict(t) for t in experiment.trials],
    }

    ckpt_path = output_path / CHECKPOINT_FILENAME
    # Write to temp file first, then rename (atomic on same filesystem)
    tmp_path = ckpt_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(ckpt, f, indent=2)
    tmp_path.rename(ckpt_path)

    logger.info(
        "Checkpoint saved: %d trials, phases %s, %.1f min elapsed → %s",
        len(experiment.trials),
        sorted(completed_phases),
        elapsed_s / 60,
        ckpt_path,
    )


def load_checkpoint(
    output_path: Path,
) -> Optional[dict[str, Any]]:
    """Load checkpoint if it exists.

    Args:
        output_path: Output directory.

    Returns:
        Checkpoint dict or None.
    """
    ckpt_path = output_path / CHECKPOINT_FILENAME
    if not ckpt_path.exists():
        return None

    with open(ckpt_path) as f:
        ckpt = json.load(f)

    logger.info(
        "Loaded checkpoint: %d trials, phases %s, %.1f min elapsed",
        ckpt.get("total_trials", 0),
        ckpt.get("completed_phases", []),
        ckpt.get("elapsed_s", 0) / 60,
    )
    return ckpt


def restore_experiment_from_checkpoint(
    ckpt: dict[str, Any],
) -> tuple[NegationExperimentResult, set[int], set[tuple[str, str, float]]]:
    """Restore experiment state from checkpoint.

    Args:
        ckpt: Loaded checkpoint dict.

    Returns:
        Tuple of (experiment, completed_phases, completed_trial_keys).
    """
    experiment = NegationExperimentResult(
        n_random_seeds=ckpt.get("n_random_seeds", 0),
    )

    # Restore baselines
    baselines = ckpt.get("baselines", {})
    experiment.focal_baseline_accuracy = baselines.get("focal_model_accuracy")
    experiment.soundscape_baseline_accuracy = baselines.get("soundscape_model_accuracy")
    experiment.mixed_baseline_accuracy = baselines.get("mixed_model_accuracy")
    experiment.focal_on_soundscape_accuracy = baselines.get("focal_on_soundscape_accuracy")
    experiment.soundscape_on_focal_accuracy = baselines.get("soundscape_on_focal_accuracy")
    experiment.mixed_on_focal_accuracy = baselines.get("mixed_on_focal_accuracy")
    experiment.mixed_on_soundscape_accuracy = baselines.get("mixed_on_soundscape_accuracy")

    # Restore layer analysis
    experiment.layer_analysis = [
        LayerAnalysis(**la) for la in ckpt.get("layer_analysis", [])
    ]

    # Restore trials
    completed_keys: set[tuple[str, str, float]] = set()
    for td in ckpt.get("trials", []):
        trial = NegationTrialResult(**td)
        experiment.trials.append(trial)
        completed_keys.add(_trial_key(trial))

    completed_phases = set(ckpt.get("completed_phases", []))

    logger.info(
        "Restored: %d trials, %d unique keys, phases %s",
        len(experiment.trials),
        len(completed_keys),
        sorted(completed_phases),
    )

    return experiment, completed_phases, completed_keys


# ---------------------------------------------------------------------------
# Random vector control
# ---------------------------------------------------------------------------

def generate_matched_random_vector(
    tau: dict[str, torch.Tensor],
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Generate a random task vector with per-layer L2 norms matching τ.

    This is the CRITICAL control for Experiment 4. Each layer's random vector
    has the same L2 norm as the corresponding layer in τ_focal, but with
    random direction. If negating this random vector also improves soundscape
    performance, the effect is not direction-specific.

    Implementation: sample from N(0,1) per layer, then rescale to match L2 norm.

    Args:
        tau: Reference task vector whose per-layer norms to match.
        seed: Random seed for reproducibility.

    Returns:
        Random task vector with matched per-layer L2 norms.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    random_tv: dict[str, torch.Tensor] = {}

    for key, tensor in tau.items():
        # Generate random values from N(0, 1)
        random_values = torch.randn(tensor.shape, generator=rng, dtype=torch.float32)

        # Compute target L2 norm (from the real task vector)
        target_l2 = tensor.float().norm(p=2).item()

        # Compute current L2 norm of random vector
        current_l2 = random_values.norm(p=2).item()

        # Rescale to match
        if current_l2 > 0:
            random_values = random_values * (target_l2 / current_l2)

        random_tv[key] = random_values

    # Verify norm matching
    total_target_l2 = sum(t.float().norm(p=2).item() ** 2 for t in tau.values()) ** 0.5
    total_random_l2 = sum(t.norm(p=2).item() ** 2 for t in random_tv.values()) ** 0.5
    logger.info(
        "Random vector control: target total L2=%.4f, random total L2=%.4f (ratio=%.6f)",
        total_target_l2, total_random_l2,
        total_random_l2 / max(total_target_l2, 1e-10),
    )

    return random_tv


# ---------------------------------------------------------------------------
# Layer-wise analysis
# ---------------------------------------------------------------------------

def analyze_focal_layers(
    tau_focal: dict[str, torch.Tensor],
) -> list[LayerAnalysis]:
    """Analyze which layers have the largest focal-domain bias.

    If focal-domain encoding is concentrated in early layers (spectral processing),
    negation should primarily affect low-level features. If it's in later layers,
    the effect is more about decision boundaries.

    Args:
        tau_focal: Focal task vector.

    Returns:
        List of LayerAnalysis sorted by L2 norm (descending).
    """
    # Compute total L2
    total_l2_sq = sum(t.float().norm(p=2).item() ** 2 for t in tau_focal.values())
    total_l2 = total_l2_sq ** 0.5

    analyses: list[LayerAnalysis] = []
    for name, tensor in sorted(tau_focal.items()):
        t = tensor.float()
        l2 = t.norm(p=2).item()
        l1 = t.norm(p=1).item()
        analyses.append(LayerAnalysis(
            layer_name=name,
            l2_norm=l2,
            l1_norm=l1,
            fraction_of_total=l2 / max(total_l2, 1e-10),
            mean_abs=t.abs().mean().item(),
            max_abs=t.abs().max().item(),
        ))

    # Sort by L2 norm descending
    analyses.sort(key=lambda a: a.l2_norm, reverse=True)

    # Log top layers
    logger.info("Top 10 layers by L2 norm in τ_focal:")
    for a in analyses[:10]:
        logger.info(
            "  %s: L2=%.4f (%.1f%% of total), mean_abs=%.6f",
            a.layer_name, a.l2_norm, a.fraction_of_total * 100, a.mean_abs,
        )

    # Categorize by layer type
    layer_groups: dict[str, float] = {}
    for a in analyses:
        # Heuristic categorization based on BEATs layer naming
        if "embed" in a.layer_name.lower() or "patch" in a.layer_name.lower():
            group = "embedding"
        elif any(k in a.layer_name.lower() for k in ["q_proj", "k_proj", "v_proj", "out_proj"]):
            group = "attention"
        elif any(k in a.layer_name.lower() for k in ["fc1", "fc2", "ffn"]):
            group = "ffn"
        elif "norm" in a.layer_name.lower():
            group = "layernorm"
        else:
            group = "other"

        layer_groups[group] = layer_groups.get(group, 0) + a.l2_norm ** 2

    logger.info("Focal bias by layer type (L2 norm):")
    for group, l2_sq in sorted(layer_groups.items(), key=lambda x: -x[1]):
        logger.info("  %s: L2=%.4f", group, l2_sq ** 0.5)

    return analyses


# ---------------------------------------------------------------------------
# Negation sweep (single source model × single vector)
# ---------------------------------------------------------------------------

def run_negation_sweep(
    base_path: str,
    source_encoder: dict[str, torch.Tensor],
    source_name: str,
    tau_to_negate: dict[str, torch.Tensor],
    vector_type: str,
    beta_grid: list[float],
    evaluator: LinearProbeEvaluator,
    focal_loaders: tuple,
    soundscape_loaders: tuple,
    completed_keys: set[tuple[str, str, float]] | None = None,
) -> list[NegationTrialResult]:
    """Sweep β for negation: θ_result = θ_source − β · τ.

    Skips trials whose (vector_type, source_name, beta) key is already in
    completed_keys (for resume support).

    Args:
        base_path: BEATs base checkpoint path.
        source_encoder: Encoder state dict to negate from.
        source_name: "soundscape" or "mixed".
        tau_to_negate: Task vector to subtract (focal or random control).
        vector_type: "focal" or "random_control_seed{N}".
        beta_grid: Negation strengths to try.
        evaluator: LinearProbeEvaluator.
        focal_loaders: (train_loader, test_loader, n_classes) for focal evaluation.
        soundscape_loaders: (train_loader, test_loader, n_classes) for soundscape.
        completed_keys: Set of already-completed trial keys to skip.

    Returns:
        List of NegationTrialResult for each *newly computed* β.
    """
    if completed_keys is None:
        completed_keys = set()

    focal_train, focal_test, focal_n_cls = focal_loaders
    ss_train, ss_test, ss_n_cls = soundscape_loaders

    results: list[NegationTrialResult] = []
    skipped = 0

    for beta in beta_grid:
        key = (vector_type, source_name, round(beta, 4))
        if key in completed_keys:
            skipped += 1
            continue

        logger.info(
            "  Negation: source=%s, vector=%s, β=%.2f",
            source_name, vector_type, beta,
        )
        t0 = time.time()

        # Apply negation
        negated = negate_task_vector(source_encoder, tau_to_negate, beta)

        # Evaluate on focal test set
        logger.info("    Evaluating on focal test set (%d classes)...", focal_n_cls)
        focal_result = evaluator.evaluate(negated, focal_train, focal_test, focal_n_cls)

        # Evaluate on soundscape test set
        logger.info("    Evaluating on soundscape test set (%d classes)...", ss_n_cls)
        ss_result = evaluator.evaluate(negated, ss_train, ss_test, ss_n_cls)

        eval_time = time.time() - t0

        trial = NegationTrialResult(
            vector_type=vector_type,
            source_model=source_name,
            beta=beta,
            focal_accuracy=focal_result.accuracy,
            soundscape_accuracy=ss_result.accuracy,
            focal_f1=focal_result.macro_f1,
            soundscape_f1=ss_result.macro_f1,
            eval_time_s=eval_time,
        )
        results.append(trial)

        logger.info(
            "    [%s β=%.2f] focal_acc=%.4f, soundscape_acc=%.4f (%.1f min)",
            vector_type, beta, focal_result.accuracy, ss_result.accuracy,
            eval_time / 60,
        )

        del negated
        torch.cuda.empty_cache()

    if skipped > 0:
        logger.info(
            "  Skipped %d/%d betas (already completed)",
            skipped, len(beta_grid),
        )

    return results


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_negation_experiment(
    config_path: str,
    output_dir: str,
    device: str = "cuda",
    beta_grid: Optional[list[float]] = None,
    n_random_seeds: int = 3,
    resume: bool = False,
    phases: Optional[list[int]] = None,
    random_seed_start: int = 0,
) -> NegationExperimentResult:
    """Run the full Experiment 4 pipeline with resume support.

    Phases:
        1 = Baseline evaluations (focal/soundscape/mixed models on both test sets)
        2 = Layer-wise analysis of τ_focal
        3 = Focal negation sweep (real τ_focal)
        4 = Random-vector control sweeps

    Args:
        config_path: Path to base.yaml.
        output_dir: Directory to save results.
        device: Torch device.
        beta_grid: Negation strengths. Default: fine grid + extended range.
        n_random_seeds: Number of random vector controls to average over.
        resume: If True, load checkpoint and skip completed work.
        phases: List of phase numbers to run. Default: all [1,2,3,4].
        random_seed_start: Start random controls from this seed index (0-indexed).
            Useful for resuming mid-phase-4 without --resume (e.g., if you know
            seeds 0 and 1 finished but 2 didn't).

    Returns:
        NegationExperimentResult.
    """
    if beta_grid is None:
        beta_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]

    if phases is None:
        phases = [1, 2, 3, 4]
    phases_set = set(phases)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["paths"]["project_root"]).resolve()
    base_path = str(root / cfg["paths"]["beats_base_checkpoint"])
    tv_dir = root / cfg["paths"]["task_vectors"]
    finetuned_dir = root / cfg["paths"]["finetuned"]
    species_groups_dir = str(root / cfg["paths"]["species_groups"])
    processed_dir = str(root / cfg["paths"]["data_processed"])
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Resume from checkpoint if requested
    # -----------------------------------------------------------------------
    completed_phases: set[int] = set()
    completed_keys: set[tuple[str, str, float]] = set()

    if resume:
        ckpt = load_checkpoint(output_path)
        if ckpt is not None:
            experiment, completed_phases, completed_keys = (
                restore_experiment_from_checkpoint(ckpt)
            )
            experiment.n_random_seeds = n_random_seeds
            prior_elapsed = ckpt.get("elapsed_s", 0.0)
            logger.info(
                "Resuming: %d completed phases, %d completed trials, "
                "%.1f min prior elapsed",
                len(completed_phases), len(completed_keys), prior_elapsed / 60,
            )
        else:
            logger.info("No checkpoint found — starting fresh.")
            experiment = NegationExperimentResult(n_random_seeds=n_random_seeds)
            prior_elapsed = 0.0
    else:
        experiment = NegationExperimentResult(n_random_seeds=n_random_seeds)
        prior_elapsed = 0.0

    experiment_start = time.time()

    # -----------------------------------------------------------------------
    # Load or compute task vectors
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("EXPERIMENT 4: Focal→Soundscape Domain Negation")
    logger.info("  Phases to run: %s", sorted(phases_set))
    logger.info("  Beta grid: %s", beta_grid)
    logger.info("  Random seeds: %d (starting from index %d)", n_random_seeds, random_seed_start)
    if resume:
        logger.info("  Resume mode: ON (skipping %d completed trials)", len(completed_keys))

    # Load focal task vector
    focal_tv_path = tv_dir / "tau_focal_domain.pt"
    if focal_tv_path.exists():
        tau_focal = torch.load(str(focal_tv_path), map_location="cpu")
        logger.info("Loaded τ_focal: %d keys", len(tau_focal))
    else:
        # Try to compute from checkpoint
        focal_ckpt = finetuned_dir / "focal_domain" / "best_model.pt"
        if not focal_ckpt.exists():
            logger.error(
                "Neither task vector (%s) nor checkpoint (%s) found for focal domain. "
                "Run fine-tuning first: python train.py --group focal_domain",
                focal_tv_path, focal_ckpt,
            )
            sys.exit(1)
        logger.info("Computing τ_focal from checkpoint...")
        tau_focal = compute_task_vector(
            str(focal_ckpt), base_path, str(focal_tv_path),
        )

    # Load soundscape and mixed encoders
    domain_encoders: dict[str, dict[str, torch.Tensor]] = {}
    for domain in ["soundscape_domain", "mixed_domain"]:
        # First try task vector
        tv_path = tv_dir / f"tau_{domain}.pt"
        if tv_path.exists():
            tv = torch.load(str(tv_path), map_location="cpu")
            encoder = apply_task_vector(base_path, tv, scaling=1.0)
            domain_encoders[domain] = encoder
            logger.info("Loaded %s encoder via task vector", domain)
        else:
            # Try checkpoint
            ckpt_path = finetuned_dir / domain / "best_model.pt"
            if ckpt_path.exists():
                ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
                domain_encoders[domain] = ckpt["encoder_state_dict"]
                logger.info("Loaded %s encoder from checkpoint", domain)
            else:
                logger.warning(
                    "No encoder found for %s (tried %s and %s)",
                    domain, tv_path, ckpt_path,
                )

    if not domain_encoders:
        logger.error("Need at least one domain encoder (soundscape or mixed)")
        sys.exit(1)

    # Also get the focal encoder for baseline evaluation
    focal_ckpt_path = finetuned_dir / "focal_domain" / "best_model.pt"
    focal_encoder: Optional[dict[str, torch.Tensor]] = None
    if focal_ckpt_path.exists():
        ckpt = torch.load(str(focal_ckpt_path), map_location="cpu", weights_only=False)
        focal_encoder = ckpt["encoder_state_dict"]
    elif (tv_dir / "tau_focal_domain.pt").exists():
        focal_encoder = apply_task_vector(base_path, tau_focal, scaling=1.0)

    # -----------------------------------------------------------------------
    # Build data loaders for focal and soundscape evaluation
    # -----------------------------------------------------------------------
    logger.info("Building evaluation loaders...")

    focal_loaders_dict = build_group_loaders(
        ["focal_domain"], species_groups_dir, processed_dir,
    )
    ss_loaders_dict = build_group_loaders(
        ["soundscape_domain"], species_groups_dir, processed_dir,
    )

    if "focal_domain" not in focal_loaders_dict:
        logger.error("Focal domain data loaders not built. Check manifest.")
        sys.exit(1)
    if "soundscape_domain" not in ss_loaders_dict:
        logger.error("Soundscape domain data loaders not built. Check manifest.")
        sys.exit(1)

    focal_loaders = focal_loaders_dict["focal_domain"]  # (train, test, n_classes)
    ss_loaders = ss_loaders_dict["soundscape_domain"]

    # -----------------------------------------------------------------------
    # Setup evaluator
    # -----------------------------------------------------------------------
    probe_cfg = ProbeConfig(
        learning_rate=cfg["evaluation"]["linear_probe"]["learning_rate"],
        epochs=cfg["evaluation"]["linear_probe"]["epochs"],
        batch_size=cfg["evaluation"]["linear_probe"]["batch_size"],
        bootstrap_n_resamples=cfg["evaluation"]["bootstrap_n_resamples"],
        bootstrap_confidence=cfg["evaluation"]["bootstrap_confidence"],
    )
    evaluator = LinearProbeEvaluator(base_path, probe_cfg, device)

    # Helper to checkpoint after each sweep
    def _save_incremental() -> None:
        elapsed = prior_elapsed + (time.time() - experiment_start)
        save_checkpoint(
            experiment, output_path, beta_grid,
            n_random_seeds, completed_phases, elapsed,
        )

    # -----------------------------------------------------------------------
    # Phase 1: Baseline evaluation
    # -----------------------------------------------------------------------
    if 1 in phases_set and 1 not in completed_phases:
        logger.info("=" * 60)
        logger.info("Phase 1: Baseline evaluations")

        # Focal model on both sets
        if focal_encoder is not None:
            logger.info("Evaluating focal model...")
            focal_on_focal = evaluator.evaluate(
                focal_encoder, focal_loaders[0], focal_loaders[1], focal_loaders[2],
            )
            focal_on_ss = evaluator.evaluate(
                focal_encoder, ss_loaders[0], ss_loaders[1], ss_loaders[2],
            )
            experiment.focal_baseline_accuracy = focal_on_focal.accuracy
            experiment.focal_on_soundscape_accuracy = focal_on_ss.accuracy
            logger.info(
                "  Focal model: focal_acc=%.4f, soundscape_acc=%.4f",
                focal_on_focal.accuracy, focal_on_ss.accuracy,
            )
            del focal_encoder
            torch.cuda.empty_cache()

        # Soundscape model on both sets
        if "soundscape_domain" in domain_encoders:
            logger.info("Evaluating soundscape model...")
            ss_enc = domain_encoders["soundscape_domain"]
            ss_on_focal = evaluator.evaluate(
                ss_enc, focal_loaders[0], focal_loaders[1], focal_loaders[2],
            )
            ss_on_ss = evaluator.evaluate(
                ss_enc, ss_loaders[0], ss_loaders[1], ss_loaders[2],
            )
            experiment.soundscape_baseline_accuracy = ss_on_ss.accuracy
            experiment.soundscape_on_focal_accuracy = ss_on_focal.accuracy
            logger.info(
                "  Soundscape model: focal_acc=%.4f, soundscape_acc=%.4f",
                ss_on_focal.accuracy, ss_on_ss.accuracy,
            )

        # Mixed model on both sets
        if "mixed_domain" in domain_encoders:
            logger.info("Evaluating mixed model...")
            mixed_enc = domain_encoders["mixed_domain"]
            mixed_on_focal = evaluator.evaluate(
                mixed_enc, focal_loaders[0], focal_loaders[1], focal_loaders[2],
            )
            mixed_on_ss = evaluator.evaluate(
                mixed_enc, ss_loaders[0], ss_loaders[1], ss_loaders[2],
            )
            experiment.mixed_baseline_accuracy = mixed_on_ss.accuracy
            experiment.mixed_on_focal_accuracy = mixed_on_focal.accuracy
            experiment.mixed_on_soundscape_accuracy = mixed_on_ss.accuracy
            logger.info(
                "  Mixed model: focal_acc=%.4f, soundscape_acc=%.4f",
                mixed_on_focal.accuracy, mixed_on_ss.accuracy,
            )

        completed_phases.add(1)
        _save_incremental()
    elif 1 in completed_phases:
        logger.info("Phase 1: SKIPPED (already completed)")

    # -----------------------------------------------------------------------
    # Phase 2: Layer-wise analysis of τ_focal
    # -----------------------------------------------------------------------
    if 2 in phases_set and 2 not in completed_phases:
        logger.info("=" * 60)
        logger.info("Phase 2: Layer-wise analysis of τ_focal")
        experiment.layer_analysis = analyze_focal_layers(tau_focal)
        completed_phases.add(2)
        _save_incremental()
    elif 2 in completed_phases:
        logger.info("Phase 2: SKIPPED (already completed)")

    # -----------------------------------------------------------------------
    # Phase 3: Focal negation sweep
    # -----------------------------------------------------------------------
    if 3 in phases_set:
        logger.info("=" * 60)
        logger.info("Phase 3: Focal negation sweep (β ∈ %s)", beta_grid)

        for domain_name, encoder in domain_encoders.items():
            short_name = domain_name.replace("_domain", "")
            logger.info("Source model: %s", short_name)

            trials = run_negation_sweep(
                base_path=base_path,
                source_encoder=encoder,
                source_name=short_name,
                tau_to_negate=tau_focal,
                vector_type="focal",
                beta_grid=beta_grid,
                evaluator=evaluator,
                focal_loaders=focal_loaders,
                soundscape_loaders=ss_loaders,
                completed_keys=completed_keys,
            )

            # Merge new trials
            for t in trials:
                experiment.trials.append(t)
                completed_keys.add(_trial_key(t))

            # Save after each source model sweep
            _save_incremental()
            logger.info(
                "  Checkpoint saved after focal sweep for %s (%d total trials)",
                short_name, len(experiment.trials),
            )

        if 3 not in completed_phases:
            completed_phases.add(3)
            _save_incremental()
    elif 3 in completed_phases:
        logger.info("Phase 3: SKIPPED (already completed)")

    # -----------------------------------------------------------------------
    # Phase 4: Random-vector control (MANDATORY)
    # -----------------------------------------------------------------------
    if 4 in phases_set:
        logger.info("=" * 60)
        logger.info(
            "Phase 4: Random-vector control (%d seeds, starting at index %d)",
            n_random_seeds, random_seed_start,
        )

        for seed_idx in range(random_seed_start, n_random_seeds):
            seed = 42 + seed_idx * 1000
            logger.info(
                "  Random control seed %d/%d (seed=%d)",
                seed_idx + 1, n_random_seeds, seed,
            )

            tau_random = generate_matched_random_vector(tau_focal, seed=seed)

            # Verify L2 norm matching per-layer
            mismatches = 0
            for key in tau_focal:
                real_l2 = tau_focal[key].float().norm(p=2).item()
                rand_l2 = tau_random[key].norm(p=2).item()
                ratio = rand_l2 / max(real_l2, 1e-10)
                if abs(ratio - 1.0) > 0.01:
                    mismatches += 1
            if mismatches > 0:
                logger.warning(
                    "  L2 norm mismatches: %d/%d layers", mismatches, len(tau_focal),
                )

            for domain_name, encoder in domain_encoders.items():
                short_name = domain_name.replace("_domain", "")

                trials = run_negation_sweep(
                    base_path=base_path,
                    source_encoder=encoder,
                    source_name=short_name,
                    tau_to_negate=tau_random,
                    vector_type=f"random_control_seed{seed}",
                    beta_grid=beta_grid,
                    evaluator=evaluator,
                    focal_loaders=focal_loaders,
                    soundscape_loaders=ss_loaders,
                    completed_keys=completed_keys,
                )

                # Merge new trials
                for t in trials:
                    experiment.trials.append(t)
                    completed_keys.add(_trial_key(t))

                # Save after each (seed, source_model) combination
                _save_incremental()
                logger.info(
                    "  Checkpoint saved after random seed=%d, source=%s "
                    "(%d total trials)",
                    seed, short_name, len(experiment.trials),
                )

            del tau_random
            torch.cuda.empty_cache()

        if 4 not in completed_phases:
            completed_phases.add(4)
            _save_incremental()
    elif 4 in completed_phases:
        logger.info("Phase 4: SKIPPED (already completed)")

    # -----------------------------------------------------------------------
    # Save final results (same format as original, for downstream compatibility)
    # -----------------------------------------------------------------------
    total_time = prior_elapsed + (time.time() - experiment_start)

    results_dict = {
        "experiment": "domain_negation",
        "beta_grid": beta_grid,
        "n_random_seeds": n_random_seeds,
        "total_trials": len(experiment.trials),
        "total_time_s": total_time,
        "baselines": {
            "focal_model_accuracy": experiment.focal_baseline_accuracy,
            "soundscape_model_accuracy": experiment.soundscape_baseline_accuracy,
            "mixed_model_accuracy": experiment.mixed_baseline_accuracy,
            "focal_on_soundscape_accuracy": experiment.focal_on_soundscape_accuracy,
            "soundscape_on_focal_accuracy": experiment.soundscape_on_focal_accuracy,
            "mixed_on_focal_accuracy": experiment.mixed_on_focal_accuracy,
            "mixed_on_soundscape_accuracy": experiment.mixed_on_soundscape_accuracy,
        },
        "layer_analysis": [asdict(la) for la in experiment.layer_analysis[:20]],
        "trials": [asdict(t) for t in experiment.trials],
    }

    # Final results file (always overwritten)
    out_file = output_path / "negation_results.json"
    with open(out_file, "w") as f:
        json.dump(results_dict, f, indent=2)
    logger.info("Final results saved to: %s", out_file)

    # -----------------------------------------------------------------------
    # Summary & Pareto analysis
    # -----------------------------------------------------------------------
    _log_summary(experiment, total_time)

    return experiment


def _log_summary(experiment: NegationExperimentResult, total_time: float) -> None:
    """Log experiment summary and Pareto analysis."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 4 SUMMARY")
    logger.info("-" * 60)

    for source in ["soundscape", "mixed"]:
        focal_trials = [
            t for t in experiment.trials
            if t.source_model == source and t.vector_type == "focal"
        ]
        if not focal_trials:
            continue

        logger.info("Source model: %s", source)
        logger.info("  β     focal_acc   ss_acc   Δss")

        baseline_ss = (
            focal_trials[0].soundscape_accuracy
            if focal_trials[0].beta == 0.0
            else None
        )
        best_ss_trial = max(focal_trials, key=lambda t: t.soundscape_accuracy)

        for t in sorted(focal_trials, key=lambda t: t.beta):
            delta_ss = (
                (t.soundscape_accuracy - baseline_ss)
                if baseline_ss is not None
                else 0.0
            )
            marker = " ← BEST" if t is best_ss_trial else ""
            logger.info(
                "  %.2f   %.4f    %.4f    %+.4f%s",
                t.beta, t.focal_accuracy, t.soundscape_accuracy, delta_ss, marker,
            )

        # Compare to random control average at the best β
        best_beta = best_ss_trial.beta
        random_trials_at_best = [
            t for t in experiment.trials
            if t.source_model == source
            and t.vector_type.startswith("random_control")
            and abs(t.beta - best_beta) < 0.01
        ]
        if random_trials_at_best:
            avg_random_ss = np.mean(
                [t.soundscape_accuracy for t in random_trials_at_best],
            )
            logger.info(
                "  At β=%.2f: focal negation ss_acc=%.4f, "
                "random control ss_acc=%.4f (avg of %d seeds)",
                best_beta,
                best_ss_trial.soundscape_accuracy,
                avg_random_ss,
                len(random_trials_at_best),
            )
            if best_ss_trial.soundscape_accuracy > avg_random_ss + 0.01:
                logger.info(
                    "  → FOCAL NEGATION IS DIRECTION-SPECIFIC "
                    "(not just regularization)",
                )
            elif best_ss_trial.soundscape_accuracy < avg_random_ss - 0.01:
                logger.info(
                    "  → Random control BETTER than focal negation (unexpected)",
                )
            else:
                logger.info(
                    "  → No significant difference — may be regularization effect",
                )

    logger.info("Total time: %.1f min", total_time / 60)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 4: Focal→Soundscape Domain Negation (with resume)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run from scratch
  python evaluation/domain_negation.py --config configs/base.yaml --output results/negation/ --device cuda

  # Resume after server timeout
  python evaluation/domain_negation.py --config configs/base.yaml --output results/negation/ --device cuda --resume

  # Run only random controls (phase 4), resuming completed trials
  python evaluation/domain_negation.py --config configs/base.yaml --output results/negation/ --device cuda --resume --phases 4

  # Run only random seed index 2 (third seed)
  python evaluation/domain_negation.py --config configs/base.yaml --output results/negation/ --device cuda --resume --phases 4 --random-seed-start 2

  # Quick mode: reduced grid
  python evaluation/domain_negation.py --config configs/base.yaml --output results/negation/ --beta-grid 0.0 0.5 1.0 --n-random-seeds 1
        """,
    )
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--output", type=str, default="results/negation/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--beta-grid", nargs="*", type=float, default=None,
        help="Custom β values. Default: 0.0 through 2.0 in 13 steps.",
    )
    parser.add_argument(
        "--n-random-seeds", type=int, default=3,
        help="Number of random vector control seeds.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint (skip completed trials).",
    )
    parser.add_argument(
        "--phases", nargs="*", type=int, default=None,
        help=(
            "Run only these phases. "
            "1=baselines, 2=layer analysis, 3=focal sweep, 4=random controls. "
            "Default: all."
        ),
    )
    parser.add_argument(
        "--random-seed-start", type=int, default=0,
        help="Start random controls from this seed index (0-indexed). Default: 0.",
    )
    args = parser.parse_args()

    run_negation_experiment(
        config_path=args.config,
        output_dir=args.output,
        device=args.device,
        beta_grid=args.beta_grid,
        n_random_seeds=args.n_random_seeds,
        resume=args.resume,
        phases=args.phases,
        random_seed_start=args.random_seed_start,
    )


if __name__ == "__main__":
    main()