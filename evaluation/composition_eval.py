"""Experiment 2: Species-Group Task Vector Composition.

Tests whether adding species-group task vectors produces multi-taxa classifiers.
Sweeps over merging methods and hyperparameters, evaluating each merged encoder
via linear probing (frozen encoder + fresh trained linear head).

Merging methods evaluated:
    1. Simple addition with uniform λ sweep
    2. TIES-Merging with trim fraction sweep
    3. DARE + averaging with drop rate sweep
    4. DARE + TIES combination
    5. DELLA + TIES (magnitude-proportional dropout)
    6. Simple average (model soups baseline)

For each merged encoder, evaluates:
    - Per-group accuracy (did merging degrade individual group performance?)
    - All-group accuracy (multi-taxa classification with unified label space)
    - Gap to joint baseline (beats-ALL, when available)

Usage:
    python evaluation/composition_eval.py \\
        --config configs/base.yaml \\
        --output results/composition/ \\
        --groups G1_passerines G2_nonpasserine_birds G3_raptors_waterbirds \\
                 G4_marine_mammals G5_amphibians
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from evaluation.linear_probe import LinearProbeEvaluator, ProbeConfig, ProbeResult
from merging.task_vectors import (
    apply_merged_tv_to_base,
    apply_task_vector,
    compose_task_vectors,
)
from merging.ties_dare import (
    dare_average,
    dare_ties_merge,
    della_ties_merge,
    simple_average,
    ties_merge,
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
class CompositionTrialResult:
    """Result of one composition trial (one method + one hyperparameter setting)."""

    method: str
    hyperparams: dict
    per_group_results: dict[str, dict]  # group_name → {accuracy, f1, ...}
    all_group_accuracy: float = 0.0
    all_group_macro_f1: float = 0.0
    gap_to_joint: Optional[float] = None
    merge_time_s: float = 0.0
    eval_time_s: float = 0.0


@dataclass
class CompositionExperimentResult:
    """Full Experiment 2 results."""

    trials: list[CompositionTrialResult] = field(default_factory=list)
    joint_baseline_accuracy: Optional[float] = None
    base_accuracy: Optional[float] = None  # beats-base (no fine-tuning)
    groups_used: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Composition methods
# ---------------------------------------------------------------------------

def _run_simple_addition(
    base_path: str,
    task_vectors: list[dict[str, torch.Tensor]],
    lambda_grid: list[float],
    evaluator: LinearProbeEvaluator,
    group_loaders: dict[str, tuple],
    all_group_loader: Optional[tuple],
    joint_baseline_acc: Optional[float],
) -> list[CompositionTrialResult]:
    """Sweep uniform scaling λ for simple task vector addition.

    θ_merged = θ_base + (λ/n) · Σ τ_i

    Args:
        base_path: Path to BEATs base checkpoint.
        task_vectors: List of per-group task vectors.
        lambda_grid: Scaling factors to try.
        evaluator: LinearProbeEvaluator instance.
        group_loaders: Dict of group_name → (train_loader, test_loader, n_classes).
        all_group_loader: Optional (train_loader, test_loader, n_classes) for all groups.
        joint_baseline_acc: Joint baseline accuracy if available.

    Returns:
        List of CompositionTrialResult for each λ.
    """
    n = len(task_vectors)
    results: list[CompositionTrialResult] = []

    for lam in lambda_grid:
        logger.info("Simple addition: λ=%.2f", lam)
        t0 = time.time()

        weights = [lam / n] * n
        merged = compose_task_vectors(base_path, task_vectors, weights)
        merge_time = time.time() - t0

        trial = _evaluate_merged_encoder(
            merged_encoder=merged,
            method="task_arithmetic",
            hyperparams={"lambda": lam},
            evaluator=evaluator,
            group_loaders=group_loaders,
            all_group_loader=all_group_loader,
            joint_baseline_acc=joint_baseline_acc,
            merge_time=merge_time,
        )
        results.append(trial)

    return results


def _run_ties(
    base_path: str,
    task_vectors: list[dict[str, torch.Tensor]],
    trim_fractions: list[float],
    evaluator: LinearProbeEvaluator,
    group_loaders: dict[str, tuple],
    all_group_loader: Optional[tuple],
    joint_baseline_acc: Optional[float],
) -> list[CompositionTrialResult]:
    """Sweep TIES-Merging trim fractions.

    Args:
        base_path: Path to BEATs base checkpoint.
        task_vectors: List of per-group task vectors.
        trim_fractions: Trim fraction values to try.
        evaluator: LinearProbeEvaluator instance.
        group_loaders: Dict of group_name → (train_loader, test_loader, n_classes).
        all_group_loader: Optional all-group loader.
        joint_baseline_acc: Joint baseline accuracy if available.

    Returns:
        List of CompositionTrialResult for each trim fraction.
    """
    results: list[CompositionTrialResult] = []

    for k in trim_fractions:
        logger.info("TIES: trim_fraction=%.2f", k)
        t0 = time.time()

        merged_tv = ties_merge(task_vectors, trim_fraction=k)
        merged = apply_merged_tv_to_base(base_path, merged_tv)
        merge_time = time.time() - t0

        trial = _evaluate_merged_encoder(
            merged_encoder=merged,
            method="ties",
            hyperparams={"trim_fraction": k},
            evaluator=evaluator,
            group_loaders=group_loaders,
            all_group_loader=all_group_loader,
            joint_baseline_acc=joint_baseline_acc,
            merge_time=merge_time,
        )
        results.append(trial)

    return results


def _run_dare_average(
    base_path: str,
    task_vectors: list[dict[str, torch.Tensor]],
    drop_rates: list[float],
    evaluator: LinearProbeEvaluator,
    group_loaders: dict[str, tuple],
    all_group_loader: Optional[tuple],
    joint_baseline_acc: Optional[float],
    seed: int = 42,
) -> list[CompositionTrialResult]:
    """Sweep DARE drop rates with simple averaging.

    Args:
        base_path: Path to BEATs base checkpoint.
        task_vectors: List of per-group task vectors.
        drop_rates: Drop rate values to try.
        evaluator: LinearProbeEvaluator instance.
        group_loaders: Dict of group_name → (train_loader, test_loader, n_classes).
        all_group_loader: Optional all-group loader.
        joint_baseline_acc: Joint baseline accuracy if available.
        seed: Random seed for DARE.

    Returns:
        List of CompositionTrialResult for each drop rate.
    """
    results: list[CompositionTrialResult] = []

    for p in drop_rates:
        logger.info("DARE+average: drop_rate=%.2f", p)
        t0 = time.time()

        merged_tv = dare_average(task_vectors, drop_rate=p, seed=seed)
        merged = apply_merged_tv_to_base(base_path, merged_tv)
        merge_time = time.time() - t0

        trial = _evaluate_merged_encoder(
            merged_encoder=merged,
            method="dare_average",
            hyperparams={"drop_rate": p},
            evaluator=evaluator,
            group_loaders=group_loaders,
            all_group_loader=all_group_loader,
            joint_baseline_acc=joint_baseline_acc,
            merge_time=merge_time,
        )
        results.append(trial)

    return results


def _run_dare_ties(
    base_path: str,
    task_vectors: list[dict[str, torch.Tensor]],
    drop_rates: list[float],
    trim_fractions: list[float],
    evaluator: LinearProbeEvaluator,
    group_loaders: dict[str, tuple],
    all_group_loader: Optional[tuple],
    joint_baseline_acc: Optional[float],
    seed: int = 42,
) -> list[CompositionTrialResult]:
    """Sweep DARE+TIES combinations.

    Args:
        base_path: Path to BEATs base checkpoint.
        task_vectors: List of per-group task vectors.
        drop_rates: DARE drop rates.
        trim_fractions: TIES trim fractions.
        evaluator: LinearProbeEvaluator instance.
        group_loaders: Dict of group_name → (train_loader, test_loader, n_classes).
        all_group_loader: Optional all-group loader.
        joint_baseline_acc: Joint baseline accuracy if available.
        seed: Random seed.

    Returns:
        List of CompositionTrialResult for each (drop_rate, trim_fraction) pair.
    """
    results: list[CompositionTrialResult] = []

    for p in drop_rates:
        for k in trim_fractions:
            logger.info("DARE+TIES: drop_rate=%.2f, trim=%.2f", p, k)
            t0 = time.time()

            merged_tv = dare_ties_merge(
                task_vectors, drop_rate=p, trim_fraction=k, seed=seed
            )
            merged = apply_merged_tv_to_base(base_path, merged_tv)
            merge_time = time.time() - t0

            trial = _evaluate_merged_encoder(
                merged_encoder=merged,
                method="dare_ties",
                hyperparams={"drop_rate": p, "trim_fraction": k},
                evaluator=evaluator,
                group_loaders=group_loaders,
                all_group_loader=all_group_loader,
                joint_baseline_acc=joint_baseline_acc,
                merge_time=merge_time,
            )
            results.append(trial)

    return results


def _run_della_ties(
    base_path: str,
    task_vectors: list[dict[str, torch.Tensor]],
    drop_rates: list[float],
    trim_fractions: list[float],
    evaluator: LinearProbeEvaluator,
    group_loaders: dict[str, tuple],
    all_group_loader: Optional[tuple],
    joint_baseline_acc: Optional[float],
    seed: int = 42,
) -> list[CompositionTrialResult]:
    """Sweep DELLA+TIES combinations.

    Args:
        base_path: Path to BEATs base checkpoint.
        task_vectors: List of per-group task vectors.
        drop_rates: DELLA target drop rates.
        trim_fractions: TIES trim fractions.
        evaluator: LinearProbeEvaluator instance.
        group_loaders: Dict of group_name → (train_loader, test_loader, n_classes).
        all_group_loader: Optional all-group loader.
        joint_baseline_acc: Joint baseline accuracy if available.
        seed: Random seed.

    Returns:
        List of CompositionTrialResult.
    """
    results: list[CompositionTrialResult] = []

    for p in drop_rates:
        for k in trim_fractions:
            logger.info("DELLA+TIES: drop_rate=%.2f, trim=%.2f", p, k)
            t0 = time.time()

            merged_tv = della_ties_merge(
                task_vectors, target_drop_rate=p, trim_fraction=k, seed=seed
            )
            merged = apply_merged_tv_to_base(base_path, merged_tv)
            merge_time = time.time() - t0

            trial = _evaluate_merged_encoder(
                merged_encoder=merged,
                method="della_ties",
                hyperparams={"target_drop_rate": p, "trim_fraction": k},
                evaluator=evaluator,
                group_loaders=group_loaders,
                all_group_loader=all_group_loader,
                joint_baseline_acc=joint_baseline_acc,
                merge_time=merge_time,
            )
            results.append(trial)

    return results


def _run_simple_avg(
    base_path: str,
    task_vectors: list[dict[str, torch.Tensor]],
    evaluator: LinearProbeEvaluator,
    group_loaders: dict[str, tuple],
    all_group_loader: Optional[tuple],
    joint_baseline_acc: Optional[float],
) -> list[CompositionTrialResult]:
    """Simple average baseline (model soups).

    θ_merged = θ_base + (1/n) · Σ τ_i

    Args:
        base_path: Path to BEATs base checkpoint.
        task_vectors: List of per-group task vectors.
        evaluator: LinearProbeEvaluator instance.
        group_loaders: Dict of group_name → (train_loader, test_loader, n_classes).
        all_group_loader: Optional all-group loader.
        joint_baseline_acc: Joint baseline accuracy if available.

    Returns:
        Single-element list with the trial result.
    """
    logger.info("Simple average (model soups)")
    t0 = time.time()

    merged_tv = simple_average(task_vectors)
    merged = apply_merged_tv_to_base(base_path, merged_tv)
    merge_time = time.time() - t0

    trial = _evaluate_merged_encoder(
        merged_encoder=merged,
        method="simple_average",
        hyperparams={},
        evaluator=evaluator,
        group_loaders=group_loaders,
        all_group_loader=all_group_loader,
        joint_baseline_acc=joint_baseline_acc,
        merge_time=merge_time,
    )
    return [trial]


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def _evaluate_merged_encoder(
    merged_encoder: dict[str, torch.Tensor],
    method: str,
    hyperparams: dict,
    evaluator: LinearProbeEvaluator,
    group_loaders: dict[str, tuple],
    all_group_loader: Optional[tuple],
    joint_baseline_acc: Optional[float],
    merge_time: float,
) -> CompositionTrialResult:
    """Evaluate a merged encoder with linear probing on all groups.

    Args:
        merged_encoder: Merged encoder state dict.
        method: Merging method name.
        hyperparams: Method-specific hyperparameters.
        evaluator: LinearProbeEvaluator instance.
        group_loaders: Dict of group_name → (train_loader, test_loader, n_classes).
        all_group_loader: Optional (train_loader, test_loader, n_classes) for all groups.
        joint_baseline_acc: Joint baseline accuracy if available.
        merge_time: Time taken for the merge operation.

    Returns:
        CompositionTrialResult with all metrics.
    """
    t0 = time.time()

    # Per-group evaluation
    per_group_results: dict[str, dict] = {}
    for group_name, (train_loader, test_loader, n_classes) in group_loaders.items():
        logger.info("  Evaluating %s (%d classes)...", group_name, n_classes)
        result = evaluator.evaluate(
            merged_encoder, train_loader, test_loader, n_classes,
        )
        per_group_results[group_name] = {
            "accuracy": result.accuracy,
            "accuracy_ci_low": result.accuracy_ci_low,
            "accuracy_ci_high": result.accuracy_ci_high,
            "macro_f1": result.macro_f1,
        }

    # All-group evaluation (unified label space)
    all_acc = 0.0
    all_f1 = 0.0
    if all_group_loader is not None:
        train_all, test_all, n_all = all_group_loader
        logger.info("  Evaluating ALL groups (%d classes)...", n_all)
        all_result = evaluator.evaluate(
            merged_encoder, train_all, test_all, n_all,
        )
        all_acc = all_result.accuracy
        all_f1 = all_result.macro_f1

    eval_time = time.time() - t0

    # Gap to joint baseline
    gap = None
    if joint_baseline_acc is not None and all_acc > 0:
        gap = joint_baseline_acc - all_acc

    trial = CompositionTrialResult(
        method=method,
        hyperparams=hyperparams,
        per_group_results=per_group_results,
        all_group_accuracy=all_acc,
        all_group_macro_f1=all_f1,
        gap_to_joint=gap,
        merge_time_s=merge_time,
        eval_time_s=eval_time,
    )

    # Log summary
    mean_group_acc = np.mean(
        [r["accuracy"] for r in per_group_results.values()]
    ) if per_group_results else 0.0

    logger.info(
        "  [%s %s] mean_group_acc=%.4f, all_acc=%.4f, gap=%s, merge=%.1fs, eval=%.1fs",
        method, hyperparams,
        mean_group_acc, all_acc,
        f"{gap:.4f}" if gap is not None else "n/a",
        merge_time, eval_time,
    )

    return trial


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_composition_experiment(
    config_path: str,
    groups: list[str],
    group_loaders: dict[str, tuple],
    all_group_loader: Optional[tuple] = None,
    joint_baseline_acc: Optional[float] = None,
    output_dir: Optional[str] = None,
    device: str = "cuda",
) -> CompositionExperimentResult:
    """Run the full species-group composition experiment.

    Loads task vectors, sweeps over all merging methods and hyperparameters,
    evaluates each via linear probing.

    Args:
        config_path: Path to base.yaml config.
        groups: List of species group names.
        group_loaders: Dict mapping group_name → (train_loader, test_loader, n_classes).
        all_group_loader: Optional loader for all-group evaluation.
        joint_baseline_acc: Accuracy of beats-ALL if available.
        output_dir: Directory to save results.
        device: Torch device.

    Returns:
        CompositionExperimentResult with all trials.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    base_path = cfg["paths"]["beats_base_checkpoint"]
    tv_dir = Path(cfg["paths"]["task_vectors"])

    # Load task vectors
    task_vectors: list[dict[str, torch.Tensor]] = []
    loaded_groups: list[str] = []
    for group in groups:
        tv_path = tv_dir / f"tau_{group}.pt"
        if not tv_path.exists():
            logger.warning("Task vector not found: %s", tv_path)
            continue
        tv = torch.load(str(tv_path), map_location="cpu")
        task_vectors.append(tv)
        loaded_groups.append(group)
        logger.info("Loaded τ_%s: %d keys", group, len(tv))

    if len(task_vectors) < 2:
        logger.error("Need at least 2 task vectors, got %d", len(task_vectors))
        sys.exit(1)

    # Filter group_loaders to match loaded groups
    available_loaders = {
        g: group_loaders[g] for g in loaded_groups if g in group_loaders
    }

    # Set up evaluator
    probe_cfg = ProbeConfig(
        learning_rate=cfg["evaluation"]["linear_probe"]["learning_rate"],
        epochs=cfg["evaluation"]["linear_probe"]["epochs"],
        batch_size=cfg["evaluation"]["linear_probe"]["batch_size"],
        bootstrap_n_resamples=cfg["evaluation"]["bootstrap_n_resamples"],
        bootstrap_confidence=cfg["evaluation"]["bootstrap_confidence"],
    )
    evaluator = LinearProbeEvaluator(base_path, probe_cfg, device)

    experiment = CompositionExperimentResult(
        joint_baseline_accuracy=joint_baseline_acc,
        groups_used=loaded_groups,
    )

    # Parse method configs
    merging_cfg = cfg["merging"]["methods"]
    method_map = {m["name"]: m for m in merging_cfg}

    logger.info("=" * 60)
    logger.info("EXPERIMENT 2: Species-Group Composition")
    logger.info("Groups: %s", loaded_groups)
    logger.info("Methods: %s", list(method_map.keys()))
    logger.info("=" * 60)

    # Count total trials for progress tracking
    total_trials = 0
    if "simple_average" in method_map:
        total_trials += 1
    if "task_arithmetic" in method_map:
        total_trials += len(method_map["task_arithmetic"].get("lambda_grid", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
    if "ties" in method_map:
        total_trials += len(method_map["ties"].get("trim_fractions", [0.1, 0.2, 0.5, 0.8]))
    if "dare_average" in method_map:
        total_trials += len(method_map["dare_average"].get("drop_rates", [0.5, 0.7, 0.9, 0.95, 0.99]))
    if "dare_ties" in method_map:
        dr = method_map["dare_ties"].get("drop_rates", [0.7, 0.9])
        tf = method_map["dare_ties"].get("trim_fractions", [0.2])
        total_trials += len(dr) * len(tf)
    if "della" in method_map:
        total_trials += 2 * 1  # 2 drop rates × 1 trim fraction
    logger.info("Total trials to run: %d", total_trials)

    trial_counter = [0]  # mutable for closure access
    experiment_start = time.time()

    def _log_trial_progress() -> None:
        trial_counter[0] += 1
        elapsed = time.time() - experiment_start
        avg_per_trial = elapsed / trial_counter[0]
        remaining = avg_per_trial * (total_trials - trial_counter[0])
        logger.info(
            ">>> Trial %d/%d complete (elapsed: %.0fm, est. remaining: %.0fm)",
            trial_counter[0], total_trials, elapsed / 60, remaining / 60,
        )

    # --- Run each method ---

    # 1. Simple average (model soups baseline)
    if "simple_average" in method_map:
        trials = _run_simple_avg(
            base_path, task_vectors, evaluator,
            available_loaders, all_group_loader, joint_baseline_acc,
        )
        experiment.trials.extend(trials)
        _log_trial_progress()

    # 2. Task arithmetic (uniform λ sweep)
    if "task_arithmetic" in method_map:
        lam_grid = method_map["task_arithmetic"].get(
            "lambda_grid", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        trials = _run_simple_addition(
            base_path, task_vectors, lam_grid, evaluator,
            available_loaders, all_group_loader, joint_baseline_acc,
        )
        experiment.trials.extend(trials)
        for _ in trials:
            _log_trial_progress()

    # 3. TIES-Merging
    if "ties" in method_map:
        trim_fracs = method_map["ties"].get("trim_fractions", [0.1, 0.2, 0.5, 0.8])
        trials = _run_ties(
            base_path, task_vectors, trim_fracs, evaluator,
            available_loaders, all_group_loader, joint_baseline_acc,
        )
        experiment.trials.extend(trials)
        for _ in trials:
            _log_trial_progress()

    # 4. DARE + average
    if "dare_average" in method_map:
        drop_rates = method_map["dare_average"].get(
            "drop_rates", [0.5, 0.7, 0.9, 0.95, 0.99]
        )
        trials = _run_dare_average(
            base_path, task_vectors, drop_rates, evaluator,
            available_loaders, all_group_loader, joint_baseline_acc,
        )
        experiment.trials.extend(trials)
        for _ in trials:
            _log_trial_progress()

    # 5. DARE + TIES
    if "dare_ties" in method_map:
        drop_rates = method_map["dare_ties"].get("drop_rates", [0.7, 0.9])
        trim_fracs = method_map["dare_ties"].get("trim_fractions", [0.2])
        trials = _run_dare_ties(
            base_path, task_vectors, drop_rates, trim_fracs, evaluator,
            available_loaders, all_group_loader, joint_baseline_acc,
        )
        experiment.trials.extend(trials)
        for _ in trials:
            _log_trial_progress()

    # 6. DELLA + TIES
    if "della" in method_map:
        # DELLA uses same grid as DARE+TIES
        drop_rates = [0.7, 0.9]
        trim_fracs = [0.2]
        trials = _run_della_ties(
            base_path, task_vectors, drop_rates, trim_fracs, evaluator,
            available_loaders, all_group_loader, joint_baseline_acc,
        )
        experiment.trials.extend(trials)
        for _ in trials:
            _log_trial_progress()

    # Save results
    if output_dir:
        _save_results(experiment, output_dir)

    # Print summary table
    _print_summary(experiment)

    return experiment


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _save_results(
    experiment: CompositionExperimentResult,
    output_dir: str,
) -> None:
    """Save experiment results to JSON."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    serializable = {
        "groups_used": experiment.groups_used,
        "joint_baseline_accuracy": experiment.joint_baseline_accuracy,
        "base_accuracy": experiment.base_accuracy,
        "trials": [
            {
                "method": t.method,
                "hyperparams": t.hyperparams,
                "per_group_results": t.per_group_results,
                "all_group_accuracy": t.all_group_accuracy,
                "all_group_macro_f1": t.all_group_macro_f1,
                "gap_to_joint": t.gap_to_joint,
                "merge_time_s": t.merge_time_s,
                "eval_time_s": t.eval_time_s,
            }
            for t in experiment.trials
        ],
    }

    results_path = out_path / "composition_results.json"
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)

    logger.info("Results saved to %s", results_path)


def _print_summary(experiment: CompositionExperimentResult) -> None:
    """Print a compact summary table of all trials."""
    logger.info("\n" + "=" * 80)
    logger.info("COMPOSITION EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    header = (
        f"{'Method':<20s} {'Params':<30s} {'MeanGrpAcc':>12s} "
        f"{'AllAcc':>10s} {'GapToJoint':>12s}"
    )
    logger.info(header)
    logger.info("-" * len(header))

    # Sort by all-group accuracy descending
    sorted_trials = sorted(
        experiment.trials,
        key=lambda t: t.all_group_accuracy,
        reverse=True,
    )

    for t in sorted_trials:
        mean_acc = np.mean(
            [r["accuracy"] for r in t.per_group_results.values()]
        ) if t.per_group_results else 0.0

        params_str = ", ".join(f"{k}={v}" for k, v in t.hyperparams.items())
        if not params_str:
            params_str = "—"

        gap_str = f"{t.gap_to_joint:.4f}" if t.gap_to_joint is not None else "n/a"

        logger.info(
            f"{t.method:<20s} {params_str:<30s} {mean_acc:>12.4f} "
            f"{t.all_group_accuracy:>10.4f} {gap_str:>12s}"
        )

    logger.info("=" * 80)

    if experiment.joint_baseline_accuracy:
        logger.info(
            "Joint baseline (beats-ALL) accuracy: %.4f",
            experiment.joint_baseline_accuracy,
        )

    # Best result
    if sorted_trials:
        best = sorted_trials[0]
        logger.info(
            "Best: %s (%s) → all_acc=%.4f",
            best.method, best.hyperparams, best.all_group_accuracy,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Experiment 2: Species-Group Composition"
    )
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--output", type=str, default="results/composition")
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--joint-baseline-checkpoint", type=str, default=None,
        help="Path to beats-ALL checkpoint for computing gap-to-joint",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    groups = args.groups or [
        "G1_passerines", "G2_nonpasserine_birds", "G3_raptors_waterbirds",
        "G4_marine_mammals", "G5_amphibians",
    ]

    # -----------------------------------------------------------------------
    # DATA LOADING HOOK
    # -----------------------------------------------------------------------
    # Replace with your AudioDataset construction.
    #
    # Expected interfaces:
    #   group_loaders[group_name] → (train_DataLoader, test_DataLoader, n_classes)
    #   all_group_loader → (train_DataLoader, test_DataLoader, total_n_classes)
    #     where the all_group loader maps each sample to a global label index
    #     spanning the union of all group species.
    # -----------------------------------------------------------------------

    try:
        from data.dataset import build_group_loaders, build_all_group_loader  # type: ignore
        group_loaders = build_group_loaders(
            groups=groups,
            species_groups_dir=cfg["paths"]["species_groups"],
            processed_dir=cfg["paths"]["data_processed"],
            batch_size=cfg["evaluation"]["linear_probe"]["batch_size"],
            num_workers=cfg["training"]["num_workers"],
        )
        all_group_loader = build_all_group_loader(
            groups=groups,
            species_groups_dir=cfg["paths"]["species_groups"],
            processed_dir=cfg["paths"]["data_processed"],
            batch_size=cfg["evaluation"]["linear_probe"]["batch_size"],
            num_workers=cfg["training"]["num_workers"],
        )
    except ImportError:
        logger.error(
            "Cannot import data loading functions. "
            "Implement data.dataset.build_group_loaders and "
            "data.dataset.build_all_group_loader."
        )
        sys.exit(1)

    # Joint baseline accuracy (if checkpoint exists)
    joint_acc = None
    if args.joint_baseline_checkpoint:
        jb_path = Path(args.joint_baseline_checkpoint)
        if jb_path.exists():
            logger.info("Computing joint baseline accuracy...")
            jb_evaluator = LinearProbeEvaluator(
                cfg["paths"]["beats_base_checkpoint"],
                ProbeConfig(
                    learning_rate=cfg["evaluation"]["linear_probe"]["learning_rate"],
                    epochs=cfg["evaluation"]["linear_probe"]["epochs"],
                    batch_size=cfg["evaluation"]["linear_probe"]["batch_size"],
                ),
                args.device,
            )
            jb_ckpt = torch.load(str(jb_path), map_location="cpu", weights_only=False)
            jb_encoder = jb_ckpt["encoder_state_dict"]
            train_all, test_all, n_all = all_group_loader
            jb_result = jb_evaluator.evaluate(jb_encoder, train_all, test_all, n_all)
            joint_acc = jb_result.accuracy
            logger.info("Joint baseline accuracy: %.4f", joint_acc)
        else:
            logger.warning("Joint baseline checkpoint not found: %s", jb_path)

    result = run_composition_experiment(
        config_path=args.config,
        groups=groups,
        group_loaders=group_loaders,
        all_group_loader=all_group_loader,
        joint_baseline_acc=joint_acc,
        output_dir=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()