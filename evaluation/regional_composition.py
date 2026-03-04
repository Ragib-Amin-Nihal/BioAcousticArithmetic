#!/usr/bin/env python3
"""Experiment 3: Regional Model Composition (RQ3).

Tests whether geographically specialized models can be merged for deployment
in novel regions or transition zones.

Protocol:
    1. Load regional task vectors τ_R1 through τ_R4
    2. Compose with three strategies:
       a) Uniform merge: equal weight for all regions
       b) Ecological weighting: proportional to species overlap (eBird)
       c) Lambda-optimized: per-region λ via grid search
    3. Evaluate on four scenarios:
       a) Within-region: R1 model on R1 test (sanity)
       b) Cross-region: R1+R2 merge on R3 test (novel region)
       c) Transition zone: merged model on species shared between regions
       d) All-region: R1+R2+R3+R4 on all test sets (scalability)
    4. Compare to R1234 joint baseline
    5. Correlate merge performance with ecological distance (Jaccard similarity)

Also sweeps advanced merging methods (TIES, DARE) on the regional vectors.

Usage:
    python evaluation/regional_composition.py \\
        --config configs/base.yaml \\
        --output results/regional/ \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from itertools import combinations
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
    apply_merged_tv_to_base,
    apply_task_vector,
    compose_task_vectors,
)
from merging.ties_dare import (
    dare_average,
    dare_ties_merge,
    simple_average,
    ties_merge,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Region names (matching manifest filenames)
REGIONS = [
    "R1_east_africa",
    "R2_south_asia",
    "R3_neotropics",
    "R4_north_america",
]
JOINT_BASELINE = "R1234_all"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class RegionalTrialResult:
    """Result of one regional composition trial."""

    method: str
    hyperparams: dict[str, Any]
    regions_composed: list[str]
    per_region_accuracy: dict[str, float]
    all_region_accuracy: float = 0.0
    gap_to_joint: Optional[float] = None
    merge_time_s: float = 0.0
    eval_time_s: float = 0.0
    # Extra metadata for ecological analysis
    ecological_weights: Optional[dict[str, float]] = None
    pairwise_jaccard: Optional[dict[str, float]] = None


@dataclass
class RegionalExperimentResult:
    """Full Experiment 3 results."""

    trials: list[RegionalTrialResult] = field(default_factory=list)
    within_region_baselines: dict[str, float] = field(default_factory=dict)
    joint_baseline_accuracy: Optional[float] = None
    species_overlaps: dict[str, dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Ecological analysis helpers
# ---------------------------------------------------------------------------

def compute_species_overlaps(
    species_groups_dir: str,
    regions: list[str],
) -> dict[str, dict[str, Any]]:
    """Compute pairwise species overlaps and Jaccard similarity between regions.

    Args:
        species_groups_dir: Directory containing regional JSON manifests.
        regions: List of region names.

    Returns:
        Dict mapping "R1_vs_R2" → {"overlap": N, "jaccard": float, "species": [...]}
    """
    region_species: dict[str, set[str]] = {}
    for region in regions:
        json_path = Path(species_groups_dir) / f"{region}.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            data = json.load(f)
        region_species[region] = set(data["metadata"].get("species_list", []))

    overlaps: dict[str, dict[str, Any]] = {}
    for r1, r2 in combinations(regions, 2):
        if r1 not in region_species or r2 not in region_species:
            continue
        s1, s2 = region_species[r1], region_species[r2]
        shared = s1 & s2
        union = s1 | s2
        jaccard = len(shared) / max(len(union), 1)
        key = f"{r1}_vs_{r2}"
        overlaps[key] = {
            "overlap": len(shared),
            "jaccard": jaccard,
            "r1_total": len(s1),
            "r2_total": len(s2),
            "shared_species": sorted(shared),
        }
        logger.info(
            "  %s × %s: %d shared (J=%.3f)",
            r1, r2, len(shared), jaccard,
        )

    return overlaps


def compute_ecological_weights_offline(
    species_groups_dir: str,
    target_region: str,
    source_regions: list[str],
) -> dict[str, float]:
    """Compute ecological weights based on species overlap (offline, no eBird API).

    For each source region, weight = (overlap with target) / (target species count).
    Normalized to sum to 1.

    This is a static proxy for the eBird API-based ecological weighting described
    in the experiment design. For real deployment, use the eBird API version.

    Args:
        species_groups_dir: Directory containing regional JSON manifests.
        target_region: Region to deploy to.
        source_regions: Regions whose task vectors will be composed.

    Returns:
        Dict mapping region name → weight.
    """
    # Load target species
    target_path = Path(species_groups_dir) / f"{target_region}.json"
    if not target_path.exists():
        return {r: 1.0 / len(source_regions) for r in source_regions}

    with open(target_path) as f:
        target_species = set(json.load(f)["metadata"].get("species_list", []))

    if not target_species:
        return {r: 1.0 / len(source_regions) for r in source_regions}

    weights: dict[str, float] = {}
    for region in source_regions:
        rpath = Path(species_groups_dir) / f"{region}.json"
        if not rpath.exists():
            weights[region] = 0.0
            continue
        with open(rpath) as f:
            region_species = set(json.load(f)["metadata"].get("species_list", []))
        overlap = len(target_species & region_species)
        weights[region] = overlap / max(len(target_species), 1)

    # Normalize
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    else:
        weights = {r: 1.0 / len(source_regions) for r in source_regions}

    return weights


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_merged_regional(
    merged_encoder: dict[str, torch.Tensor],
    evaluator: LinearProbeEvaluator,
    region_loaders: dict[str, tuple],
    all_region_loader: Optional[tuple],
) -> tuple[dict[str, float], float]:
    """Evaluate a merged encoder on per-region and all-region tasks.

    Args:
        merged_encoder: Merged encoder state dict.
        evaluator: LinearProbeEvaluator instance.
        region_loaders: Dict of region_name → (train_loader, test_loader, n_classes).
        all_region_loader: Optional (train, test, n_classes) for all-region eval.

    Returns:
        (per_region_acc, all_region_acc) tuple.
    """
    per_region: dict[str, float] = {}
    for rname, (train_ld, test_ld, n_cls) in region_loaders.items():
        logger.info("    Evaluating %s (%d classes)...", rname, n_cls)
        result = evaluator.evaluate(merged_encoder, train_ld, test_ld, n_cls)
        per_region[rname] = result.accuracy

    all_acc = 0.0
    if all_region_loader is not None:
        train_all, test_all, n_all = all_region_loader
        logger.info("    Evaluating ALL regions (%d classes)...", n_all)
        all_result = evaluator.evaluate(merged_encoder, train_all, test_all, n_all)
        all_acc = all_result.accuracy

    return per_region, all_acc


# ---------------------------------------------------------------------------
# Composition strategies
# ---------------------------------------------------------------------------

def run_uniform_composition(
    base_path: str,
    task_vectors: dict[str, dict[str, torch.Tensor]],
    regions_to_compose: list[str],
    lambda_grid: list[float],
    evaluator: LinearProbeEvaluator,
    region_loaders: dict[str, tuple],
    all_region_loader: Optional[tuple],
    joint_acc: Optional[float],
) -> list[RegionalTrialResult]:
    """Uniform weighting: θ_merged = θ_base + (λ/n) · Σ τ_Ri.

    Args:
        base_path: BEATs base checkpoint path.
        task_vectors: Dict mapping region name → task vector.
        regions_to_compose: Which regions to merge.
        lambda_grid: Scaling factors to sweep.
        evaluator: LinearProbeEvaluator.
        region_loaders: Per-region data loaders.
        all_region_loader: All-region loader.
        joint_acc: R1234 joint baseline accuracy.

    Returns:
        List of trial results.
    """
    tvs = [task_vectors[r] for r in regions_to_compose]
    n = len(tvs)
    results: list[RegionalTrialResult] = []

    for lam in lambda_grid:
        logger.info("  Uniform composition: λ=%.2f, regions=%s", lam, regions_to_compose)
        t0 = time.time()

        weights = [lam / n] * n
        merged = compose_task_vectors(base_path, tvs, weights)
        merge_time = time.time() - t0

        t1 = time.time()
        per_region_acc, all_acc = evaluate_merged_regional(
            merged, evaluator, region_loaders, all_region_loader,
        )
        eval_time = time.time() - t1

        gap = (joint_acc - all_acc) if joint_acc is not None else None

        trial = RegionalTrialResult(
            method="uniform",
            hyperparams={"lambda": lam, "n_regions": n},
            regions_composed=regions_to_compose,
            per_region_accuracy=per_region_acc,
            all_region_accuracy=all_acc,
            gap_to_joint=gap,
            merge_time_s=merge_time,
            eval_time_s=eval_time,
        )
        results.append(trial)

        mean_acc = np.mean(list(per_region_acc.values())) if per_region_acc else 0.0
        logger.info(
            "    [uniform λ=%.2f] mean_region=%.4f, all=%.4f, gap=%s",
            lam, mean_acc, all_acc,
            f"{gap:.4f}" if gap is not None else "n/a",
        )

    return results


def run_ecological_composition(
    base_path: str,
    task_vectors: dict[str, dict[str, torch.Tensor]],
    source_regions: list[str],
    target_regions: list[str],
    species_groups_dir: str,
    lambda_grid: list[float],
    evaluator: LinearProbeEvaluator,
    region_loaders: dict[str, tuple],
    all_region_loader: Optional[tuple],
    joint_acc: Optional[float],
) -> list[RegionalTrialResult]:
    """Ecologically-weighted composition: weights proportional to species overlap.

    For each target region, compute overlap-based weights for source regions,
    then compose. This simulates deploying to a specific geographic area.

    Args:
        base_path: BEATs base checkpoint path.
        task_vectors: Dict mapping region → task vector.
        source_regions: Regions to compose from.
        target_regions: Regions to optimize weights for.
        species_groups_dir: Directory with regional JSON manifests.
        lambda_grid: Overall scaling factors.
        evaluator: LinearProbeEvaluator.
        region_loaders: Per-region data loaders.
        all_region_loader: All-region loader.
        joint_acc: Joint baseline accuracy.

    Returns:
        List of trial results.
    """
    results: list[RegionalTrialResult] = []

    for target in target_regions:
        eco_weights = compute_ecological_weights_offline(
            species_groups_dir, target, source_regions,
        )
        logger.info(
            "  Ecological weights for target=%s: %s",
            target,
            {r: f"{w:.3f}" for r, w in eco_weights.items()},
        )

        for lam in lambda_grid:
            logger.info(
                "    Ecological composition: target=%s, λ=%.2f", target, lam,
            )
            t0 = time.time()

            tvs = [task_vectors[r] for r in source_regions]
            weights = [lam * eco_weights[r] for r in source_regions]
            merged = compose_task_vectors(base_path, tvs, weights)
            merge_time = time.time() - t0

            t1 = time.time()
            per_region_acc, all_acc = evaluate_merged_regional(
                merged, evaluator, region_loaders, all_region_loader,
            )
            eval_time = time.time() - t1

            gap = (joint_acc - all_acc) if joint_acc is not None else None

            trial = RegionalTrialResult(
                method="ecological",
                hyperparams={"lambda": lam, "target_region": target},
                regions_composed=source_regions,
                per_region_accuracy=per_region_acc,
                all_region_accuracy=all_acc,
                gap_to_joint=gap,
                merge_time_s=merge_time,
                eval_time_s=eval_time,
                ecological_weights=eco_weights,
            )
            results.append(trial)

    return results


def run_advanced_methods(
    base_path: str,
    task_vectors: dict[str, dict[str, torch.Tensor]],
    regions_to_compose: list[str],
    evaluator: LinearProbeEvaluator,
    region_loaders: dict[str, tuple],
    all_region_loader: Optional[tuple],
    joint_acc: Optional[float],
) -> list[RegionalTrialResult]:
    """Run TIES and DARE on regional task vectors.

    Tests whether advanced methods help for geographically structured vectors
    the same way they (didn't) help for taxonomic vectors in Experiment 2.

    Args:
        base_path: BEATs base checkpoint path.
        task_vectors: Dict mapping region → task vector.
        regions_to_compose: Which regions to merge.
        evaluator: LinearProbeEvaluator.
        region_loaders: Per-region loaders.
        all_region_loader: All-region loader.
        joint_acc: Joint baseline accuracy.

    Returns:
        List of trial results.
    """
    tvs = [task_vectors[r] for r in regions_to_compose]
    results: list[RegionalTrialResult] = []

    # --- Simple average (model soups) ---
    logger.info("  Simple average (regional model soups)")
    t0 = time.time()
    merged_tv = simple_average(tvs)
    merged = apply_merged_tv_to_base(base_path, merged_tv)
    merge_time = time.time() - t0

    t1 = time.time()
    per_region_acc, all_acc = evaluate_merged_regional(
        merged, evaluator, region_loaders, all_region_loader,
    )
    eval_time = time.time() - t1

    results.append(RegionalTrialResult(
        method="simple_average",
        hyperparams={},
        regions_composed=regions_to_compose,
        per_region_accuracy=per_region_acc,
        all_region_accuracy=all_acc,
        gap_to_joint=(joint_acc - all_acc) if joint_acc else None,
        merge_time_s=merge_time,
        eval_time_s=eval_time,
    ))

    # --- TIES-Merging ---
    for trim_frac in [0.2, 0.5, 0.8]:
        logger.info("  TIES: trim=%.2f", trim_frac)
        t0 = time.time()
        merged_tv = ties_merge(tvs, trim_fraction=trim_frac)
        merged = apply_merged_tv_to_base(base_path, merged_tv)
        merge_time = time.time() - t0

        t1 = time.time()
        per_region_acc, all_acc = evaluate_merged_regional(
            merged, evaluator, region_loaders, all_region_loader,
        )
        eval_time = time.time() - t1

        results.append(RegionalTrialResult(
            method="ties",
            hyperparams={"trim_fraction": trim_frac},
            regions_composed=regions_to_compose,
            per_region_accuracy=per_region_acc,
            all_region_accuracy=all_acc,
            gap_to_joint=(joint_acc - all_acc) if joint_acc else None,
            merge_time_s=merge_time,
            eval_time_s=eval_time,
        ))

    # --- DARE + average ---
    for drop_rate in [0.5, 0.9]:
        logger.info("  DARE: drop=%.2f", drop_rate)
        t0 = time.time()
        merged_tv = dare_average(tvs, drop_rate=drop_rate)
        merged = apply_merged_tv_to_base(base_path, merged_tv)
        merge_time = time.time() - t0

        t1 = time.time()
        per_region_acc, all_acc = evaluate_merged_regional(
            merged, evaluator, region_loaders, all_region_loader,
        )
        eval_time = time.time() - t1

        results.append(RegionalTrialResult(
            method="dare_average",
            hyperparams={"drop_rate": drop_rate},
            regions_composed=regions_to_compose,
            per_region_accuracy=per_region_acc,
            all_region_accuracy=all_acc,
            gap_to_joint=(joint_acc - all_acc) if joint_acc else None,
            merge_time_s=merge_time,
            eval_time_s=eval_time,
        ))

    return results


def run_incremental_composition(
    base_path: str,
    task_vectors: dict[str, dict[str, torch.Tensor]],
    region_order: list[str],
    evaluator: LinearProbeEvaluator,
    region_loaders: dict[str, tuple],
    all_region_loader: Optional[tuple],
) -> list[RegionalTrialResult]:
    """Add regions one at a time to measure scalability.

    R1 → R1+R2 → R1+R2+R3 → R1+R2+R3+R4.
    Tests whether degradation is monotonic or catastrophic.

    Args:
        base_path: BEATs base checkpoint path.
        task_vectors: Dict mapping region → task vector.
        region_order: Order to add regions.
        evaluator: LinearProbeEvaluator.
        region_loaders: Per-region loaders.
        all_region_loader: All-region loader.

    Returns:
        List of trial results (one per incremental step).
    """
    results: list[RegionalTrialResult] = []

    for k in range(1, len(region_order) + 1):
        regions_k = region_order[:k]
        tvs = [task_vectors[r] for r in regions_k]
        n = len(tvs)

        logger.info("  Incremental: %d regions = %s", k, regions_k)
        t0 = time.time()
        weights = [1.0 / n] * n
        merged = compose_task_vectors(base_path, tvs, weights)
        merge_time = time.time() - t0

        t1 = time.time()
        per_region_acc, all_acc = evaluate_merged_regional(
            merged, evaluator, region_loaders, all_region_loader,
        )
        eval_time = time.time() - t1

        results.append(RegionalTrialResult(
            method="incremental",
            hyperparams={"n_regions": k, "lambda": 1.0},
            regions_composed=regions_k,
            per_region_accuracy=per_region_acc,
            all_region_accuracy=all_acc,
            merge_time_s=merge_time,
            eval_time_s=eval_time,
        ))

    return results


# ---------------------------------------------------------------------------
# Within-region baseline
# ---------------------------------------------------------------------------

def evaluate_within_region_baselines(
    base_path: str,
    task_vectors: dict[str, dict[str, torch.Tensor]],
    evaluator: LinearProbeEvaluator,
    region_loaders: dict[str, tuple],
) -> dict[str, float]:
    """Evaluate each region's own fine-tuned encoder on its own test set.

    This is the sanity check: merging should not exceed these single-region
    accuracies on each region's own data.

    Args:
        base_path: BEATs base checkpoint.
        task_vectors: Per-region task vectors.
        evaluator: LinearProbeEvaluator.
        region_loaders: Per-region (train, test, n_classes).

    Returns:
        Dict of region → within-region accuracy.
    """
    baselines: dict[str, float] = {}

    for region, tv in task_vectors.items():
        if region not in region_loaders:
            continue
        train_ld, test_ld, n_cls = region_loaders[region]
        logger.info("Within-region baseline: %s (%d classes)", region, n_cls)

        encoder = apply_task_vector(base_path, tv, scaling=1.0)
        result = evaluator.evaluate(encoder, train_ld, test_ld, n_cls)
        baselines[region] = result.accuracy
        logger.info("  %s: acc=%.4f", region, result.accuracy)

        del encoder
        torch.cuda.empty_cache()

    return baselines


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_regional_experiment(
    config_path: str,
    output_dir: str,
    device: str = "cuda",
    regions: Optional[list[str]] = None,
) -> RegionalExperimentResult:
    """Run the full Experiment 3 pipeline.

    Args:
        config_path: Path to base.yaml.
        output_dir: Directory to save results.
        device: Torch device.
        regions: Specific regions to use. Default: all four.

    Returns:
        RegionalExperimentResult.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["paths"]["project_root"]).resolve()
    base_path = str(root / cfg["paths"]["beats_base_checkpoint"])
    tv_dir = root / cfg["paths"]["task_vectors"]
    species_groups_dir = str(root / cfg["paths"]["species_groups"])
    processed_dir = str(root / cfg["paths"]["data_processed"])
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if regions is None:
        regions = REGIONS

    # -----------------------------------------------------------------------
    # Load task vectors
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("EXPERIMENT 3: Regional Model Composition")
    logger.info("Loading regional task vectors...")

    task_vectors: dict[str, dict[str, torch.Tensor]] = {}
    for region in regions:
        tv_path = tv_dir / f"tau_{region}.pt"
        if not tv_path.exists():
            logger.warning("Task vector not found: %s", tv_path)
            continue
        tv = torch.load(str(tv_path), map_location="cpu")
        task_vectors[region] = tv
        logger.info("  Loaded τ_%s: %d keys", region, len(tv))

    available_regions = list(task_vectors.keys())
    if len(available_regions) < 2:
        logger.error("Need at least 2 regional task vectors, got %d", len(available_regions))
        sys.exit(1)

    logger.info("Available regions: %s", available_regions)

    # -----------------------------------------------------------------------
    # Build data loaders
    # -----------------------------------------------------------------------
    logger.info("Building region data loaders...")
    region_loaders = build_group_loaders(
        available_regions, species_groups_dir, processed_dir,
    )

    # All-region loader
    all_region_loader: Optional[tuple] = None
    joint_path = Path(species_groups_dir) / f"{JOINT_BASELINE}.json"
    if joint_path.exists():
        joint_loaders = build_group_loaders(
            [JOINT_BASELINE], species_groups_dir, processed_dir,
        )
        if JOINT_BASELINE in joint_loaders:
            all_region_loader = joint_loaders[JOINT_BASELINE]
    else:
        logger.warning("Joint baseline manifest not found: %s", joint_path)

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

    experiment = RegionalExperimentResult()
    experiment_start = time.time()

    # -----------------------------------------------------------------------
    # Species overlap analysis
    # -----------------------------------------------------------------------
    logger.info("Computing species overlaps...")
    experiment.species_overlaps = compute_species_overlaps(
        species_groups_dir, available_regions,
    )

    # -----------------------------------------------------------------------
    # Phase 1: Within-region baselines
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 1: Within-region baselines")
    experiment.within_region_baselines = evaluate_within_region_baselines(
        base_path, task_vectors, evaluator, region_loaders,
    )

    # -----------------------------------------------------------------------
    # Phase 2: Joint baseline (R1234)
    # -----------------------------------------------------------------------
    joint_acc: Optional[float] = None
    joint_tv_path = tv_dir / f"tau_{JOINT_BASELINE}.pt"
    if joint_tv_path.exists() and all_region_loader is not None:
        logger.info("=" * 60)
        logger.info("Phase 2: Joint baseline (R1234)")
        joint_tv = torch.load(str(joint_tv_path), map_location="cpu")
        joint_encoder = apply_task_vector(base_path, joint_tv, scaling=1.0)
        _, _, n_joint = all_region_loader
        joint_result = evaluator.evaluate(
            joint_encoder,
            all_region_loader[0],
            all_region_loader[1],
            n_joint,
        )
        joint_acc = joint_result.accuracy
        experiment.joint_baseline_accuracy = joint_acc
        logger.info("Joint baseline (R1234): acc=%.4f", joint_acc)
        del joint_encoder, joint_tv
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Phase 3: Uniform composition
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 3: Uniform composition (all regions)")
    lambda_grid = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    trials = run_uniform_composition(
        base_path, task_vectors, available_regions, lambda_grid,
        evaluator, region_loaders, all_region_loader, joint_acc,
    )
    experiment.trials.extend(trials)

    # -----------------------------------------------------------------------
    # Phase 4: Ecological composition
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 4: Ecologically-weighted composition")
    eco_lambda_grid = [0.5, 1.0, 1.5]
    trials = run_ecological_composition(
        base_path, task_vectors,
        source_regions=available_regions,
        target_regions=available_regions,  # Test each as target
        species_groups_dir=species_groups_dir,
        lambda_grid=eco_lambda_grid,
        evaluator=evaluator,
        region_loaders=region_loaders,
        all_region_loader=all_region_loader,
        joint_acc=joint_acc,
    )
    experiment.trials.extend(trials)

    # -----------------------------------------------------------------------
    # Phase 5: Advanced merging methods
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 5: Advanced merging methods")
    trials = run_advanced_methods(
        base_path, task_vectors, available_regions,
        evaluator, region_loaders, all_region_loader, joint_acc,
    )
    experiment.trials.extend(trials)

    # -----------------------------------------------------------------------
    # Phase 6: Incremental composition
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 6: Incremental composition")
    trials = run_incremental_composition(
        base_path, task_vectors, available_regions,
        evaluator, region_loaders, all_region_loader,
    )
    experiment.trials.extend(trials)

    # -----------------------------------------------------------------------
    # Phase 7: Cross-region (novel region) evaluation
    # Leave one region out and evaluate the merged remaining on the held-out.
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 7: Cross-region (leave-one-out)")
    for held_out in available_regions:
        source = [r for r in available_regions if r != held_out]
        if len(source) < 2:
            continue

        logger.info("  Held out: %s, Merging: %s", held_out, source)
        tvs = [task_vectors[r] for r in source]
        n = len(tvs)

        t0 = time.time()
        weights = [1.0 / n] * n
        merged = compose_task_vectors(base_path, tvs, weights)
        merge_time = time.time() - t0

        # Evaluate on the held-out region
        if held_out in region_loaders:
            train_ld, test_ld, n_cls = region_loaders[held_out]
            t1 = time.time()
            result = evaluator.evaluate(merged, train_ld, test_ld, n_cls)
            eval_time = time.time() - t1

            trial = RegionalTrialResult(
                method="cross_region_loo",
                hyperparams={"held_out_region": held_out},
                regions_composed=source,
                per_region_accuracy={held_out: result.accuracy},
                all_region_accuracy=0.0,
                merge_time_s=merge_time,
                eval_time_s=eval_time,
            )
            experiment.trials.extend([trial])
            logger.info(
                "    Held-out %s acc: %.4f (within-region baseline: %.4f)",
                held_out, result.accuracy,
                experiment.within_region_baselines.get(held_out, 0.0),
            )

        del merged
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    total_time = time.time() - experiment_start

    results_dict = {
        "experiment": "regional_composition",
        "regions": available_regions,
        "total_trials": len(experiment.trials),
        "total_time_s": total_time,
        "within_region_baselines": experiment.within_region_baselines,
        "joint_baseline_accuracy": experiment.joint_baseline_accuracy,
        "species_overlaps": experiment.species_overlaps,
        "trials": [asdict(t) for t in experiment.trials],
    }

    out_file = output_path / "regional_composition_results.json"
    with open(out_file, "w") as f:
        json.dump(results_dict, f, indent=2)
    logger.info("Results saved to: %s", out_file)

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("EXPERIMENT 3 SUMMARY")
    logger.info("-" * 60)
    logger.info("Within-region baselines:")
    for r, acc in experiment.within_region_baselines.items():
        logger.info("  %s: %.4f", r, acc)
    if joint_acc is not None:
        logger.info("Joint baseline (R1234): %.4f", joint_acc)

    logger.info("Best trials per method:")
    methods = set(t.method for t in experiment.trials)
    for method in sorted(methods):
        method_trials = [t for t in experiment.trials if t.method == method]
        best = max(method_trials, key=lambda t: t.all_region_accuracy or np.mean(
            list(t.per_region_accuracy.values())) if t.per_region_accuracy else 0
        )
        mean_acc = np.mean(list(best.per_region_accuracy.values())) if best.per_region_accuracy else 0
        logger.info(
            "  %s: mean_region=%.4f, all=%.4f, params=%s",
            method, mean_acc, best.all_region_accuracy, best.hyperparams,
        )

    logger.info("Total time: %.1f min", total_time / 60)
    logger.info("=" * 60)

    return experiment


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 3: Regional Model Composition",
    )
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--output", type=str, default="results/regional/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--regions", nargs="*", default=None,
        help="Specific regions to use. Default: all four.",
    )
    args = parser.parse_args()

    run_regional_experiment(
        config_path=args.config,
        output_dir=args.output,
        device=args.device,
        regions=args.regions,
    )


if __name__ == "__main__":
    main()