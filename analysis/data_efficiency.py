#!/usr/bin/env python3
"""Supplementary Experiment: Data Efficiency of Task Arithmetic.

Tests how the amount of training data for a small group affects merged model
performance. This is the most novel contribution — no task arithmetic paper
has measured this in any domain.

Scenario: G4 (marine mammals, 21 classes, ~1400 training samples) is the
smallest and most data-limited group. Fine-tune G4 on {25%, 50%, 75%, 100%}
of its training data, merge each variant with the full G1-G3+G5 vectors,
and measure merged accuracy across all groups.

This answers the practical question: "How much data do I need for a new
species group before task arithmetic produces useful results?"

Prerequisites:
    - G1, G2, G3, G5 fine-tuned models and task vectors (Phase 1)
    - G4 species group manifest
    - beats-ALL joint model (for joint training comparison)

Usage:
    # Phase 0: Create sub-sampled G4 manifests
    python analysis/data_efficiency.py --config configs/base.yaml --phase 0

    # Phase 1: Train G4 at each data fraction (4 runs, ~30 min each)
    python analysis/data_efficiency.py --config configs/base.yaml --phase 1

    # Phase 2: Compute task vectors + merge + evaluate
    python analysis/data_efficiency.py --config configs/base.yaml --phase 2 --device cuda

    # All phases
    python analysis/data_efficiency.py --config configs/base.yaml --phase 0 1 2 --device cuda

    # Custom fractions
    python analysis/data_efficiency.py --config configs/base.yaml --phase 0 1 2 \
        --fractions 0.1 0.25 0.5 0.75 1.0
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import subprocess
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

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# The group whose data we vary
TARGET_GROUP = "G4_marine_mammals"

# The other groups (always use full data)
OTHER_GROUPS = [
    "G1_passerines", "G2_nonpasserine_birds",
    "G3_raptors_waterbirds", "G5_amphibians",
]
ALL_GROUPS = OTHER_GROUPS + [TARGET_GROUP]

DEFAULT_FRACTIONS = [0.25, 0.50, 0.75, 1.0]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class FractionResult:
    """Result for one data fraction."""
    fraction: float
    n_train_samples: int
    n_species_retained: int  # Some species may drop below minimum at low fractions
    # G4 standalone accuracy (fine-tuned on this fraction)
    g4_standalone_accuracy: float = 0.0
    # Merged model metrics
    g4_merged_accuracy: float = 0.0
    all_group_accuracy: float = 0.0
    per_group_accuracy: dict[str, float] = field(default_factory=dict)
    # Task vector properties
    tv_l2_norm: float = 0.0
    tv_cosine_with_full: float = 0.0  # Cosine similarity with 100% task vector


@dataclass
class DataEfficiencyResults:
    """Full experiment results."""
    target_group: str = TARGET_GROUP
    fractions: list[FractionResult] = field(default_factory=list)
    joint_baseline_accuracy: float = 0.0  # beats-ALL all-group acc
    full_merge_accuracy: float = 0.0  # Merge with 100% G4
    total_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Phase 0: Create sub-sampled manifests
# ---------------------------------------------------------------------------

def create_subsampled_manifests(
    config_path: str,
    fractions: list[float],
    seed: int = 42,
) -> list[Path]:
    """Create sub-sampled versions of the target group manifest.

    For each fraction, stratified sub-sampling is performed: each species
    retains the same fraction of its training clips. Species that drop below
    the minimum sample threshold after sub-sampling are removed entirely.

    Validation and test sets are kept IDENTICAL across fractions for fair
    comparison.

    Args:
        config_path: Path to base.yaml.
        fractions: Data fractions to create (e.g., [0.25, 0.5, 0.75, 1.0]).
        seed: Random seed for sub-sampling.

    Returns:
        List of paths to created manifests.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["paths"]["project_root"]).resolve()
    species_groups_dir = root / cfg["paths"]["species_groups"]

    # Load the full G4 manifest
    full_manifest_path = species_groups_dir / f"{TARGET_GROUP}.json"
    if not full_manifest_path.exists():
        logger.error("Target group manifest not found: %s", full_manifest_path)
        sys.exit(1)

    with open(full_manifest_path) as f:
        full_manifest = json.load(f)

    full_train = full_manifest["train"]
    val_data = full_manifest["val"]  # Kept identical
    test_data = full_manifest["test"]  # Kept identical
    metadata = full_manifest["metadata"]

    logger.info(
        "Full %s manifest: %d classes, %d train, %d val, %d test",
        TARGET_GROUP, metadata["n_classes"],
        len(full_train), len(val_data), len(test_data),
    )

    # Group training samples by species for stratified sampling
    species_train: dict[str, list[dict]] = {}
    for entry in full_train:
        sp = entry["species"]
        if sp not in species_train:
            species_train[sp] = []
        species_train[sp].append(entry)

    rng = random.Random(seed)
    created_paths: list[Path] = []

    min_samples = cfg["splits"].get("min_samples_per_species", 10)
    # For sub-sampled manifests, we relax the minimum to avoid dropping
    # too many species at low fractions. Use 3 as absolute floor.
    min_samples_sub = max(3, min_samples // 3)

    for frac in fractions:
        group_name = f"{TARGET_GROUP}_frac{int(frac * 100):03d}"
        out_path = species_groups_dir / f"{group_name}.json"

        if frac >= 1.0:
            # 100% = just copy original manifest with new name
            sub_manifest = copy.deepcopy(full_manifest)
            sub_manifest["metadata"]["group"] = group_name
            sub_manifest["metadata"]["data_fraction"] = 1.0
            with open(out_path, "w") as f:
                json.dump(sub_manifest, f, indent=2)
            created_paths.append(out_path)
            logger.info(
                "  frac=%.0f%%: %d species, %d train (full copy)",
                frac * 100, metadata["n_classes"], len(full_train),
            )
            continue

        # Stratified sub-sampling
        sub_train: list[dict] = []
        retained_species: list[str] = []
        species2label: dict[str, int] = {}
        label_counter = 0

        for sp in sorted(species_train.keys()):
            samples = species_train[sp].copy()
            rng.shuffle(samples)
            n_keep = max(1, int(len(samples) * frac))
            kept = samples[:n_keep]

            if len(kept) < min_samples_sub:
                continue  # Drop this species

            # Remap label
            new_label = label_counter
            for entry in kept:
                sub_train.append({
                    "path": entry["path"],
                    "species": sp,
                    "label": new_label,
                })
            species2label[sp] = new_label
            retained_species.append(sp)
            label_counter += 1

        # Remap val and test to only retained species
        sub_val = []
        sub_test = []
        for entry in val_data:
            if entry["species"] in species2label:
                sub_val.append({
                    "path": entry["path"],
                    "species": entry["species"],
                    "label": species2label[entry["species"]],
                })
        for entry in test_data:
            if entry["species"] in species2label:
                sub_test.append({
                    "path": entry["path"],
                    "species": entry["species"],
                    "label": species2label[entry["species"]],
                })

        rng.shuffle(sub_train)

        sub_manifest = {
            "metadata": {
                "group": group_name,
                "description": (
                    f"{TARGET_GROUP} sub-sampled to {frac*100:.0f}% "
                    f"of training data for data efficiency experiment"
                ),
                "n_classes": len(retained_species),
                "n_species": len(retained_species),
                "species_list": retained_species,
                "species2label": species2label,
                "label2species": {v: k for k, v in species2label.items()},
                "n_train": len(sub_train),
                "n_val": len(sub_val),
                "n_test": len(sub_test),
                "data_fraction": frac,
                "source_group": TARGET_GROUP,
            },
            "train": sub_train,
            "val": sub_val,
            "test": sub_test,
        }

        with open(out_path, "w") as f:
            json.dump(sub_manifest, f, indent=2)
        created_paths.append(out_path)

        logger.info(
            "  frac=%.0f%%: %d/%d species, %d train, %d val, %d test",
            frac * 100,
            len(retained_species), len(species_train),
            len(sub_train), len(sub_val), len(sub_test),
        )

    return created_paths


# ---------------------------------------------------------------------------
# Phase 1: Train G4 at each fraction
# ---------------------------------------------------------------------------

def train_subsampled_models(
    config_path: str,
    fractions: list[float],
    gpu: int = 0,
) -> list[Path]:
    """Train the target group model at each data fraction.

    Calls train.py for each sub-sampled manifest. Skips fractions where
    a checkpoint already exists.

    Args:
        config_path: Path to base.yaml.
        fractions: Data fractions to train.
        gpu: GPU device index.

    Returns:
        List of checkpoint paths.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg["paths"]["project_root"]).resolve()
    finetuned_dir = root / cfg["paths"]["finetuned"]

    checkpoint_paths: list[Path] = []

    for frac in fractions:
        group_name = f"{TARGET_GROUP}_frac{int(frac * 100):03d}"
        ckpt_dir = finetuned_dir / group_name
        ckpt_path = ckpt_dir / "best_model.pt"

        if frac >= 1.0:
            # For 100%, use the existing full G4 checkpoint
            full_ckpt = finetuned_dir / TARGET_GROUP / "best_model.pt"
            if full_ckpt.exists():
                logger.info("  frac=100%%: Using existing %s checkpoint", TARGET_GROUP)
                checkpoint_paths.append(full_ckpt)
                continue
            else:
                logger.info("  frac=100%%: Full G4 checkpoint not found, training...")

        if ckpt_path.exists():
            logger.info("  frac=%.0f%%: Checkpoint exists, skipping", frac * 100)
            checkpoint_paths.append(ckpt_path)
            continue

        logger.info("  Training %s...", group_name)
        cmd = [
            sys.executable, str(root / "train.py"),
            "--group", group_name,
            "--config", config_path,
            "--gpu", str(gpu),
        ]
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            logger.error("Training failed for %s (return code %d)", group_name, result.returncode)
            continue

        if ckpt_path.exists():
            checkpoint_paths.append(ckpt_path)
        else:
            logger.error("Expected checkpoint not found: %s", ckpt_path)

    return checkpoint_paths


# ---------------------------------------------------------------------------
# Phase 2: Compute task vectors, merge, and evaluate
# ---------------------------------------------------------------------------

def evaluate_data_efficiency(
    config_path: str,
    fractions: list[float],
    device: str = "cuda",
    output_dir: Optional[str] = None,
) -> DataEfficiencyResults:
    """Evaluate merged model quality across data fractions.

    For each fraction:
    1. Compute task vector from the sub-sampled G4 checkpoint
    2. Merge with full G1-G3+G5 task vectors (task arithmetic, λ=1.0)
    3. Evaluate merged encoder via linear probing on all groups
    4. Compare cosine similarity of sub-sampled TV with full TV

    Args:
        config_path: Path to base.yaml.
        fractions: Data fractions to evaluate.
        device: Torch device.
        output_dir: Where to save results.

    Returns:
        DataEfficiencyResults.
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
        output_dir = str(root / "results" / "data_efficiency")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = DataEfficiencyResults()
    t0 = time.time()

    from data.dataset import build_group_loaders, build_all_group_loader
    from evaluation.linear_probe import LinearProbeEvaluator, ProbeConfig
    from merging.task_vectors import compute_task_vector, apply_merged_tv_to_base

    probe_cfg = ProbeConfig(
        learning_rate=cfg["evaluation"]["linear_probe"]["learning_rate"],
        epochs=cfg["evaluation"]["linear_probe"]["epochs"],
        batch_size=cfg["evaluation"]["linear_probe"]["batch_size"],
        bootstrap_n_resamples=cfg["evaluation"]["bootstrap_n_resamples"],
        bootstrap_confidence=cfg["evaluation"]["bootstrap_confidence"],
    )
    evaluator = LinearProbeEvaluator(base_path, probe_cfg, device)

    # Build evaluation loaders for all groups
    logger.info("Building evaluation loaders...")
    group_loaders = build_group_loaders(
        ALL_GROUPS, species_groups_dir, processed_dir,
        batch_size=cfg["evaluation"]["linear_probe"]["batch_size"],
    )
    all_train_ld, all_test_ld, total_classes = build_all_group_loader(
        ALL_GROUPS, species_groups_dir, processed_dir,
        batch_size=cfg["evaluation"]["linear_probe"]["batch_size"],
    )

    # Load task vectors for other groups (always full)
    other_tvs: dict[str, dict[str, torch.Tensor]] = {}
    for group in OTHER_GROUPS:
        tv_path = tv_dir / f"tau_{group}.pt"
        if tv_path.exists():
            other_tvs[group] = torch.load(str(tv_path), map_location="cpu")
        else:
            logger.error("Task vector not found for %s: %s", group, tv_path)
            sys.exit(1)

    # Load full G4 task vector for cosine comparison
    full_tv_path = tv_dir / f"tau_{TARGET_GROUP}.pt"
    full_tv: Optional[dict[str, torch.Tensor]] = None
    if full_tv_path.exists():
        full_tv = torch.load(str(full_tv_path), map_location="cpu")

    # -------------------------------------------------------------------
    # Evaluate each fraction
    # -------------------------------------------------------------------
    for frac in fractions:
        logger.info("=" * 60)
        logger.info("Evaluating data fraction: %.0f%%", frac * 100)

        frac_result = FractionResult(fraction=frac, n_train_samples=0, n_species_retained=0)

        # Find checkpoint
        group_name = f"{TARGET_GROUP}_frac{int(frac * 100):03d}"
        if frac >= 1.0:
            ckpt_path = finetuned_dir / TARGET_GROUP / "best_model.pt"
        else:
            ckpt_path = finetuned_dir / group_name / "best_model.pt"

        if not ckpt_path.exists():
            logger.warning("  Checkpoint not found: %s. Skipping.", ckpt_path)
            continue

        # Load manifest for metadata
        if frac >= 1.0:
            manifest_path = Path(species_groups_dir) / f"{TARGET_GROUP}.json"
        else:
            manifest_path = Path(species_groups_dir) / f"{group_name}.json"

        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest_data = json.load(f)
            meta = manifest_data["metadata"]
            frac_result.n_train_samples = meta.get(
                "n_train", len(manifest_data.get("train", [])),
            )
            frac_result.n_species_retained = meta["n_classes"]

        # Compute task vector
        tv_save_path = out_path / f"tau_{group_name}.pt"
        if tv_save_path.exists():
            sub_tv = torch.load(str(tv_save_path), map_location="cpu")
        elif frac >= 1.0 and (tv_dir / f"tau_{TARGET_GROUP}.pt").exists():
            # 100% — reuse existing full task vector
            logger.info("  Reusing existing full task vector for %s", TARGET_GROUP)
            sub_tv = torch.load(str(tv_dir / f"tau_{TARGET_GROUP}.pt"), map_location="cpu")
        else:
            logger.info("  Computing task vector for %s...", group_name)
            compute_task_vector(str(ckpt_path), base_path, str(tv_save_path))
            sub_tv = torch.load(str(tv_save_path), map_location="cpu")

        # Task vector properties
        frac_result.tv_l2_norm = sum(
            t.float().norm(p=2).item() ** 2 for t in sub_tv.values()
        ) ** 0.5

        if full_tv is not None:
            # Cosine similarity with full TV
            dot = sum(
                (sub_tv[k].float() * full_tv[k].float()).sum().item()
                for k in sub_tv if k in full_tv
            )
            norm_sub = frac_result.tv_l2_norm
            norm_full = sum(
                t.float().norm(p=2).item() ** 2 for t in full_tv.values()
            ) ** 0.5
            frac_result.tv_cosine_with_full = dot / max(norm_sub * norm_full, 1e-10)

        logger.info(
            "  TV L2=%.4f, cosine_with_full=%.4f",
            frac_result.tv_l2_norm, frac_result.tv_cosine_with_full,
        )

        # --- G4 standalone accuracy ---
        if TARGET_GROUP in group_loaders:
            train_ld, test_ld, n_cls = group_loaders[TARGET_GROUP]
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            standalone_result = evaluator.evaluate(
                ckpt["encoder_state_dict"], train_ld, test_ld, n_cls,
            )
            frac_result.g4_standalone_accuracy = standalone_result.accuracy
            logger.info("  G4 standalone: %.4f", standalone_result.accuracy)

        # --- Merge with other groups (simple average: θ_base + (1/N) Σ τ_i) ---
        # This matches bootstrap_gap.py and Experiment 2's best method.
        # N = total number of task vectors being merged (target + others).
        n_vectors = 1 + len(OTHER_GROUPS)  # sub_tv + other groups
        scale = 1.0 / n_vectors

        # Collect ALL unique keys across all task vectors
        all_keys: set[str] = set(sub_tv.keys())
        for group in OTHER_GROUPS:
            all_keys.update(other_tvs[group].keys())

        merged_tv: dict[str, torch.Tensor] = {}
        for key in sorted(all_keys):
            val = torch.zeros_like(
                sub_tv.get(key, next(
                    tv[key] for tv in other_tvs.values() if key in tv
                )),
            ).float()
            if key in sub_tv:
                val = val + scale * sub_tv[key].float()
            for group in OTHER_GROUPS:
                if key in other_tvs[group]:
                    val = val + scale * other_tvs[group][key].float()
            merged_tv[key] = val

        merged_encoder = apply_merged_tv_to_base(base_path, merged_tv)

        # Per-group evaluation
        for group in ALL_GROUPS:
            if group not in group_loaders:
                continue
            train_ld, test_ld, n_cls = group_loaders[group]
            probe_result = evaluator.evaluate(merged_encoder, train_ld, test_ld, n_cls)
            frac_result.per_group_accuracy[group] = probe_result.accuracy
            if group == TARGET_GROUP:
                frac_result.g4_merged_accuracy = probe_result.accuracy

        # All-group evaluation
        all_result = evaluator.evaluate(merged_encoder, all_train_ld, all_test_ld, total_classes)
        frac_result.all_group_accuracy = all_result.accuracy

        results.fractions.append(frac_result)

        logger.info(
            "  frac=%.0f%%: G4_standalone=%.4f, G4_merged=%.4f, all=%.4f",
            frac * 100,
            frac_result.g4_standalone_accuracy,
            frac_result.g4_merged_accuracy,
            frac_result.all_group_accuracy,
        )

        del merged_encoder, merged_tv
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------
    # Reference baselines
    # -------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Reference baselines:")

    # beats-ALL joint baseline
    all_ckpt = finetuned_dir / "ALL_birds" / "best_model.pt"
    if all_ckpt.exists():
        ckpt = torch.load(str(all_ckpt), map_location="cpu", weights_only=False)
        all_result = evaluator.evaluate(
            ckpt["encoder_state_dict"], all_train_ld, all_test_ld, total_classes,
        )
        results.joint_baseline_accuracy = all_result.accuracy
        logger.info("  Joint baseline (beats-ALL): %.4f", all_result.accuracy)

    # Full merge (100% of all groups)
    if len(results.fractions) > 0:
        full_frac = [fr for fr in results.fractions if fr.fraction >= 1.0]
        if full_frac:
            results.full_merge_accuracy = full_frac[0].all_group_accuracy

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    results.total_time_s = time.time() - t0

    results_dict = {
        "experiment": "data_efficiency",
        "target_group": TARGET_GROUP,
        "other_groups": OTHER_GROUPS,
        "joint_baseline_accuracy": results.joint_baseline_accuracy,
        "full_merge_accuracy": results.full_merge_accuracy,
        "total_time_s": results.total_time_s,
        "fractions": [asdict(fr) for fr in results.fractions],
    }

    out_file = out_path / "data_efficiency_results.json"
    with open(out_file, "w") as f:
        json.dump(results_dict, f, indent=2)
    logger.info("Results saved to %s", out_file)

    # Summary
    logger.info("=" * 60)
    logger.info("DATA EFFICIENCY SUMMARY")
    logger.info("-" * 60)
    logger.info(
        "%-8s  %-7s  %-7s  %-10s  %-10s  %-10s  %-8s",
        "Frac", "N_samp", "N_spp", "G4_alone", "G4_merged", "All_acc", "cos_full",
    )
    for fr in results.fractions:
        logger.info(
            "%-8.0f%%  %-7d  %-7d  %-10.4f  %-10.4f  %-10.4f  %-8.4f",
            fr.fraction * 100,
            fr.n_train_samples,
            fr.n_species_retained,
            fr.g4_standalone_accuracy,
            fr.g4_merged_accuracy,
            fr.all_group_accuracy,
            fr.tv_cosine_with_full,
        )
    logger.info("-" * 60)
    logger.info("Joint baseline: %.4f", results.joint_baseline_accuracy)
    logger.info("Full merge:     %.4f", results.full_merge_accuracy)
    logger.info("=" * 60)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Data Efficiency: how much data does task arithmetic need?",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phases:
  0 = Create sub-sampled G4 manifests
  1 = Train G4 at each data fraction (calls train.py)
  2 = Compute TVs, merge, evaluate all fractions

Examples:
  # Full pipeline
  python analysis/data_efficiency.py --config configs/base.yaml --phase 0 1 2 --device cuda

  # Just manifests
  python analysis/data_efficiency.py --config configs/base.yaml --phase 0

  # Custom fractions
  python analysis/data_efficiency.py --config configs/base.yaml --phase 0 1 2 \
      --fractions 0.1 0.25 0.5 0.75 1.0

  # Just evaluate (after training done)
  python analysis/data_efficiency.py --config configs/base.yaml --phase 2 --device cuda
        """,
    )
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument(
        "--phase", nargs="+", type=int, required=True,
        help="Phase(s) to run: 0=manifests, 1=train, 2=evaluate",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--fractions", nargs="*", type=float, default=None,
        help="Data fractions to test. Default: 0.25 0.50 0.75 1.0",
    )
    args = parser.parse_args()

    fractions = args.fractions or DEFAULT_FRACTIONS
    phases = set(args.phase)

    if 0 in phases:
        logger.info("=" * 60)
        logger.info("PHASE 0: Create sub-sampled manifests")
        create_subsampled_manifests(args.config, fractions)

    if 1 in phases:
        logger.info("=" * 60)
        logger.info("PHASE 1: Train sub-sampled models")
        train_subsampled_models(args.config, fractions, gpu=args.gpu)

    if 2 in phases:
        logger.info("=" * 60)
        logger.info("PHASE 2: Evaluate data efficiency")
        evaluate_data_efficiency(
            args.config, fractions,
            device=args.device, output_dir=args.output,
        )


if __name__ == "__main__":
    main()