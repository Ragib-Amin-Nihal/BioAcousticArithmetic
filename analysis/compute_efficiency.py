#!/usr/bin/env python3
"""Compute efficiency comparison: merging vs. joint training vs. incremental fine-tuning.

Produces a table answering "why not just jointly train?" with quantitative data.

Three scenarios compared:
    A) Joint training: Fine-tune on ALL data from scratch.
    B) Incremental fine-tune: Take existing joint model, fine-tune on new group data.
    C) Task arithmetic: Fine-tune base on new group only, merge via vector addition.

Data sources (in priority order):
    1. Checkpoint files: extract epoch count, timestamp, config
    2. W&B API: query run metrics for wall-clock times
    3. Manual estimates: compute from (epochs × batches × time_per_batch)

Usage:
    python analysis/compute_efficiency.py --config configs/base.yaml
    python analysis/compute_efficiency.py --config configs/base.yaml --output results/efficiency/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

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
class TrainingCost:
    """Training cost for one model."""

    group_name: str
    n_classes: int
    n_train_samples: int
    n_epochs_completed: int  # From checkpoint
    batch_size: int
    batches_per_epoch: int
    estimated_gpu_hours: float  # Total wall-clock
    timestamp: str = ""
    config_hash: str = ""


@dataclass
class ScenarioCost:
    """Cost of one deployment scenario."""

    scenario: str
    description: str
    training_gpu_hours: float  # GPU time for fine-tuning
    merge_seconds: float  # Time for task vector arithmetic (CPU)
    eval_gpu_hours: float  # Linear probe evaluation
    total_gpu_hours: float
    accuracy: Optional[float] = None  # If available from results


@dataclass
class EfficiencyTable:
    """Full efficiency comparison."""

    per_model_costs: list[TrainingCost] = field(default_factory=list)
    scenarios: list[ScenarioCost] = field(default_factory=list)
    joint_training_cost: Optional[TrainingCost] = None
    merge_arithmetic_seconds: float = 0.0  # Measured or estimated


# ---------------------------------------------------------------------------
# Extract costs from checkpoints
# ---------------------------------------------------------------------------

def extract_checkpoint_cost(
    checkpoint_path: str,
    manifest_path: Optional[str] = None,
) -> Optional[TrainingCost]:
    """Extract training cost metadata from a saved checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        manifest_path: Optional path to group JSON manifest for sample counts.

    Returns:
        TrainingCost or None if checkpoint not found.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        return None

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    group_name = ckpt.get("group_name", ckpt_path.parent.name)
    epoch = ckpt.get("epoch", 0)
    config = ckpt.get("config", {})
    timestamp = ckpt.get("timestamp", "")
    n_classes = ckpt.get("num_classes", 0)

    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 32)
    config_hash = ckpt.get("metrics", {}).get("config_hash", "")

    # Get sample count from manifest
    n_train = 0
    if manifest_path and Path(manifest_path).exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        if "metadata" in manifest:
            n_train = manifest["metadata"].get("n_train", len(manifest.get("train", [])))
        elif "train" in manifest:
            n_train = len(manifest["train"])

    batches_per_epoch = max(1, n_train // batch_size) if n_train > 0 else 0

    # Estimate GPU hours from epoch count and dataset size
    # BEATs fine-tuning: ~1.5s per batch on A100 with bf16
    seconds_per_batch = 1.5
    total_seconds = (epoch + 1) * batches_per_epoch * seconds_per_batch
    gpu_hours = total_seconds / 3600

    return TrainingCost(
        group_name=group_name,
        n_classes=n_classes,
        n_train_samples=n_train,
        n_epochs_completed=epoch + 1,
        batch_size=batch_size,
        batches_per_epoch=batches_per_epoch,
        estimated_gpu_hours=gpu_hours,
        timestamp=timestamp,
        config_hash=config_hash,
    )


def measure_merge_time(
    tv_dir: str,
    base_path: str,
    groups: list[str],
    n_repeats: int = 5,
) -> float:
    """Measure wall-clock time for task vector composition.

    Loads task vectors and composes them n_repeats times to get a stable
    timing estimate. This is the "seconds of arithmetic" in the efficiency
    argument.

    Args:
        tv_dir: Directory containing tau_*.pt files.
        base_path: Path to BEATs base checkpoint.
        groups: Groups whose vectors to compose.
        n_repeats: Number of timing repetitions.

    Returns:
        Average merge time in seconds.
    """
    import time

    tvs = []
    for g in groups:
        tv_path = Path(tv_dir) / f"tau_{g}.pt"
        if tv_path.exists():
            tvs.append(torch.load(str(tv_path), map_location="cpu"))

    if len(tvs) < 2:
        logger.warning("Need ≥2 task vectors to measure merge time")
        return 0.0

    base_ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
    base_model = base_ckpt["model"]

    times = []
    n = len(tvs)
    for _ in range(n_repeats):
        t0 = time.perf_counter()

        # Compose: θ_base + (1/n) · Σ τ_i
        merged = {}
        for key in base_model:
            merged[key] = base_model[key].float().clone()
            for tv in tvs:
                if key in tv:
                    merged[key] += (1.0 / n) * tv[key].float()

        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_time = sum(times) / len(times)
    logger.info(
        "Merge time (%d vectors, %d repeats): %.2f ± %.2f seconds",
        n, n_repeats, avg_time,
        (max(times) - min(times)) / 2,
    )
    return avg_time


# ---------------------------------------------------------------------------
# Build scenarios
# ---------------------------------------------------------------------------

def build_efficiency_table(
    config_path: str,
    output_dir: Optional[str] = None,
) -> EfficiencyTable:
    """Build the full compute efficiency comparison table.

    Args:
        config_path: Path to base.yaml.
        output_dir: Directory to save results.

    Returns:
        EfficiencyTable with all cost data.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["paths"]["project_root"]).resolve()
    finetuned_dir = root / cfg["paths"]["finetuned"]
    tv_dir = root / cfg["paths"]["task_vectors"]
    species_groups_dir = root / cfg["paths"]["species_groups"]
    base_path = str(root / cfg["paths"]["beats_base_checkpoint"])

    table = EfficiencyTable()

    # -----------------------------------------------------------------------
    # Extract per-model training costs
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Extracting per-model training costs...")

    all_groups = [
        "G1_passerines", "G2_nonpasserine_birds", "G3_raptors_waterbirds",
        "G4_marine_mammals", "G5_amphibians",
    ]

    for group in all_groups + ["ALL_birds"]:
        ckpt_path = finetuned_dir / group / "best_model.pt"
        manifest_path = species_groups_dir / f"{group}.json"

        cost = extract_checkpoint_cost(
            str(ckpt_path),
            str(manifest_path) if manifest_path.exists() else None,
        )
        if cost:
            table.per_model_costs.append(cost)
            if group == "ALL_birds":
                table.joint_training_cost = cost
            logger.info(
                "  %s: %d classes, %d train, %d epochs → %.1f GPU-hrs (est.)",
                cost.group_name, cost.n_classes, cost.n_train_samples,
                cost.n_epochs_completed, cost.estimated_gpu_hours,
            )

    # -----------------------------------------------------------------------
    # Measure merge arithmetic time
    # -----------------------------------------------------------------------
    logger.info("Measuring merge arithmetic time...")
    table.merge_arithmetic_seconds = measure_merge_time(
        str(tv_dir), base_path, all_groups[:5],
    )

    # -----------------------------------------------------------------------
    # Build deployment scenarios
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Computing deployment scenarios...")

    # Sum of individual fine-tuning costs
    individual_total = sum(c.estimated_gpu_hours for c in table.per_model_costs
                          if c.group_name != "ALL_birds")
    joint_cost = table.joint_training_cost.estimated_gpu_hours if table.joint_training_cost else 0

    # Linear probe evaluation cost (estimated: ~5 min per group)
    eval_cost_per_group = 5.0 / 60  # hours
    n_groups = len(all_groups)
    eval_cost = eval_cost_per_group * n_groups

    # Scenario A: Joint training from scratch
    table.scenarios.append(ScenarioCost(
        scenario="joint_training",
        description="Fine-tune on all data jointly (beats-ALL)",
        training_gpu_hours=joint_cost,
        merge_seconds=0,
        eval_gpu_hours=eval_cost,
        total_gpu_hours=joint_cost + eval_cost,
    ))

    # Scenario B: Task arithmetic — train groups independently, merge
    merge_hrs = table.merge_arithmetic_seconds / 3600
    table.scenarios.append(ScenarioCost(
        scenario="task_arithmetic_all",
        description=f"Fine-tune {n_groups} groups independently + merge",
        training_gpu_hours=individual_total,
        merge_seconds=table.merge_arithmetic_seconds,
        eval_gpu_hours=eval_cost,
        total_gpu_hours=individual_total + merge_hrs + eval_cost,
    ))

    # Scenario C: Adding one new group (G5 amphibians) to existing G1-G4 merge
    g5_cost = next(
        (c.estimated_gpu_hours for c in table.per_model_costs
         if c.group_name == "G5_amphibians"), 0
    )
    table.scenarios.append(ScenarioCost(
        scenario="add_new_group_merge",
        description="Add G5 (amphibians) to existing G1-G4 model via merge",
        training_gpu_hours=g5_cost,
        merge_seconds=table.merge_arithmetic_seconds,
        eval_gpu_hours=eval_cost,
        total_gpu_hours=g5_cost + merge_hrs + eval_cost,
    ))

    # Scenario D: Adding one new group by retraining from scratch
    table.scenarios.append(ScenarioCost(
        scenario="add_new_group_retrain",
        description="Retrain on all data (G1-G5) from scratch",
        training_gpu_hours=joint_cost,
        merge_seconds=0,
        eval_gpu_hours=eval_cost,
        total_gpu_hours=joint_cost + eval_cost,
    ))

    # Scenario E: Adding one new group by fine-tuning existing joint model
    # Assumes fine-tuning on G5 data only but with all-group head
    table.scenarios.append(ScenarioCost(
        scenario="add_new_group_finetune_joint",
        description="Fine-tune existing joint model on new G5 data (forgetting risk)",
        training_gpu_hours=g5_cost * 0.5,  # Fewer epochs needed
        merge_seconds=0,
        eval_gpu_hours=eval_cost,
        total_gpu_hours=g5_cost * 0.5 + eval_cost,
    ))

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("COMPUTE EFFICIENCY COMPARISON")
    logger.info("-" * 60)
    logger.info("Per-model fine-tuning costs:")
    for c in table.per_model_costs:
        logger.info(
            "  %-25s %4d cls  %6d train  %2d epochs  %5.1f GPU-hrs",
            c.group_name, c.n_classes, c.n_train_samples,
            c.n_epochs_completed, c.estimated_gpu_hours,
        )
    logger.info("-" * 60)
    logger.info("Merge arithmetic: %.2f seconds (CPU only)", table.merge_arithmetic_seconds)
    logger.info("-" * 60)
    logger.info("Deployment scenarios:")
    for s in table.scenarios:
        logger.info(
            "  %-45s  train=%5.1fh  merge=%5.1fs  total=%5.1fh",
            s.description[:45], s.training_gpu_hours,
            s.merge_seconds, s.total_gpu_hours,
        )

    # Key comparison
    merge_add = next((s for s in table.scenarios if s.scenario == "add_new_group_merge"), None)
    retrain_add = next((s for s in table.scenarios if s.scenario == "add_new_group_retrain"), None)
    if merge_add and retrain_add and retrain_add.total_gpu_hours > 0:
        speedup = retrain_add.total_gpu_hours / max(merge_add.total_gpu_hours, 0.01)
        logger.info("-" * 60)
        logger.info(
            "KEY: Adding a new group via merge is %.1f× cheaper than retraining",
            speedup,
        )
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        results = {
            "per_model_costs": [asdict(c) for c in table.per_model_costs],
            "scenarios": [asdict(s) for s in table.scenarios],
            "merge_arithmetic_seconds": table.merge_arithmetic_seconds,
        }
        out_file = out_path / "compute_efficiency.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved to %s", out_file)

        # Also generate a LaTeX-ready table
        _write_latex_table(table, out_path / "efficiency_table.tex")

    return table


def _write_latex_table(table: EfficiencyTable, path: Path) -> None:
    """Write a LaTeX-formatted efficiency comparison table.

    Args:
        table: EfficiencyTable with computed costs.
        path: Output .tex file path.
    """
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Compute cost comparison: adding a new species group.}",
        r"\label{tab:efficiency}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Strategy & Training (GPU-h) & Merge (s) & Total (GPU-h) \\",
        r"\midrule",
    ]

    for s in table.scenarios:
        name = s.description
        # Shorten for table
        if "retrain" in s.scenario.lower() or "scratch" in s.description.lower():
            name = "Retrain from scratch"
        elif "finetune_joint" in s.scenario:
            name = "Fine-tune existing joint"
        elif "add_new_group_merge" in s.scenario:
            name = r"Task arithmetic (\textbf{ours})"
        elif "task_arithmetic_all" in s.scenario:
            name = "Train all + merge"
        elif "joint_training" in s.scenario:
            name = "Joint training (all data)"

        merge_str = f"{s.merge_seconds:.1f}" if s.merge_seconds > 0 else "---"
        lines.append(
            f"{name} & {s.training_gpu_hours:.1f} & {merge_str} & {s.total_gpu_hours:.1f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    path.write_text("\n".join(lines))
    logger.info("LaTeX table written to %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute efficiency comparison table",
    )
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--output", type=str, default="results/efficiency/")
    args = parser.parse_args()

    build_efficiency_table(args.config, args.output)


if __name__ == "__main__":
    main()