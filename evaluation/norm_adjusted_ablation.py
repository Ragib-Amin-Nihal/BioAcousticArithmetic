"""Norm-adjusted weighting ablation for task arithmetic composition.

Standard task arithmetic uses uniform weights: merged = base + λ * mean(τ_i).
Norm-adjusted weighting scales each task vector by 1/||τ_i||₂ before averaging,
so all groups contribute equally regardless of training set size.

This addresses the observed magnitude imbalance:
  G1: L2=13.58, G2: L2=9.51, G3: L2=6.86, G5: L2=4.79, G4: L2=1.45

Usage:
    python evaluation/norm_adjusted_ablation.py \
        --config configs/base.yaml \
        --output results/composition/norm_adjusted/ \
        --device cuda \
        --joint-baseline-checkpoint results/finetuned/ALL_birds/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def compute_l2_norm(task_vector: dict[str, torch.Tensor]) -> float:
    """Compute the global L2 norm of a task vector."""
    total = sum(v.float().pow(2).sum().item() for v in task_vector.values())
    return total ** 0.5


def merge_norm_adjusted(
    base_state: dict[str, torch.Tensor],
    task_vectors: dict[str, dict[str, torch.Tensor]],
    lam: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Merge task vectors with inverse-L2-norm weighting.

    Each τ_i is scaled by w_i = (1/||τ_i||₂) / Σ_j(1/||τ_j||₂)
    so that all groups contribute equal "angular" signal regardless of magnitude.

    Args:
        base_state: Base encoder state dict.
        task_vectors: Dict mapping group_name → task vector state dict.
        lam: Global scaling factor applied after weighted average.

    Returns:
        Merged encoder state dict.
    """
    # Compute inverse-norm weights
    norms = {name: compute_l2_norm(tv) for name, tv in task_vectors.items()}
    inv_norms = {name: 1.0 / n for name, n in norms.items()}
    total_inv = sum(inv_norms.values())
    weights = {name: inv_n / total_inv for name, inv_n in inv_norms.items()}

    logger.info("Norm-adjusted weights:")
    for name in sorted(weights.keys()):
        logger.info(
            "  %s: L2=%.2f, weight=%.4f (uniform would be %.4f)",
            name, norms[name], weights[name], 1.0 / len(task_vectors),
        )

    # Weighted sum of task vectors
    keys = list(next(iter(task_vectors.values())).keys())
    merged = {}
    for key in keys:
        combined = torch.zeros_like(base_state[key], dtype=torch.float32)
        for name, tv in task_vectors.items():
            combined += weights[name] * tv[key].float()
        merged[key] = base_state[key].float() + lam * combined

    return merged


def merge_uniform(
    base_state: dict[str, torch.Tensor],
    task_vectors: dict[str, dict[str, torch.Tensor]],
    lam: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Standard uniform-weight task arithmetic (for comparison)."""
    n = len(task_vectors)
    keys = list(next(iter(task_vectors.values())).keys())
    merged = {}
    for key in keys:
        combined = torch.zeros_like(base_state[key], dtype=torch.float32)
        for tv in task_vectors.values():
            combined += tv[key].float() / n
        merged[key] = base_state[key].float() + lam * combined
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Norm-adjusted weighting ablation"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/composition/norm_adjusted")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--joint-baseline-checkpoint", type=str, default=None,
        help="Path to ALL_birds checkpoint for gap-to-joint computation",
    )
    parser.add_argument(
        "--lambda-grid", type=float, nargs="+",
        default=[0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        help="Lambda values to sweep",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Imports that depend on project code ---
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.dataset import build_group_loaders, build_all_group_loader
    from evaluation.linear_probe import LinearProbeEvaluator, ProbeConfig

    # --- Load task vectors ---
    groups = cfg.get("experiment", {}).get("groups", [
        "G1_passerines", "G2_nonpasserine_birds", "G3_raptors_waterbirds",
        "G4_marine_mammals", "G5_amphibians",
    ])
    tv_dir = Path(cfg["paths"]["task_vectors"])

    task_vectors: dict[str, dict[str, torch.Tensor]] = {}
    for g in groups:
        tv_path = tv_dir / f"tau_{g}.pt"
        task_vectors[g] = torch.load(tv_path, map_location="cpu")
        logger.info("Loaded τ_%s: %d keys", g, len(task_vectors[g]))

    # --- Load base encoder ---
    base_ckpt = torch.load(
        cfg["paths"]["beats_base_checkpoint"], map_location="cpu"
    )
    base_model_state = base_ckpt["model"]

    # Filter to encoder keys only (matching task vector keys)
    reference_keys = set(next(iter(task_vectors.values())).keys())
    base_state = {k: v for k, v in base_model_state.items() if k in reference_keys}
    logger.info("Base encoder: %d keys", len(base_state))

    # --- Build data loaders ---
    species_groups_dir = cfg["paths"]["species_groups"]
    processed_dir = cfg["paths"]["data_processed"]
    batch_size = cfg.get("evaluation", {}).get("batch_size", 64)

    group_loaders = build_group_loaders(
        groups=groups,
        species_groups_dir=species_groups_dir,
        processed_dir=processed_dir,
        batch_size=batch_size,
        num_workers=4,
    )
    for name, (train_dl, test_dl, n_cls) in group_loaders.items():
        logger.info(
            "Built loaders for %s: %d train, %d test, %d classes",
            name, len(train_dl.dataset), len(test_dl.dataset), n_cls,
        )

    all_group_loader = build_all_group_loader(
        groups=groups,
        species_groups_dir=species_groups_dir,
        processed_dir=processed_dir,
        batch_size=batch_size,
        num_workers=4,
    )
    all_train_dl, all_test_dl, total_classes = all_group_loader
    logger.info(
        "All-group loader: %d train, %d test, %d classes",
        len(all_train_dl.dataset), len(all_test_dl.dataset), total_classes,
    )

    # --- Joint baseline ---
    joint_baseline_acc: Optional[float] = None
    if args.joint_baseline_checkpoint:
        logger.info("Computing joint baseline...")
        evaluator_tmp = LinearProbeEvaluator(
            base_checkpoint_path=cfg["paths"]["beats_base_checkpoint"],
            config=ProbeConfig(),
            device=args.device,
        )
        jb_encoder = torch.load(
            args.joint_baseline_checkpoint, map_location="cpu"
        )["encoder_state_dict"]
        jb_result = evaluator_tmp.evaluate(
            encoder_state_dict=jb_encoder,
            train_loader=all_train_dl,
            test_loader=all_test_dl,
            num_classes=total_classes,
        )
        joint_baseline_acc = jb_result.accuracy
        logger.info("Joint baseline accuracy: %.4f", joint_baseline_acc)

    # --- Run ablation ---
    evaluator = LinearProbeEvaluator(
        base_checkpoint_path=cfg["paths"]["beats_base_checkpoint"],
        config=ProbeConfig(),
        device=args.device,
    )

    results: list[dict] = []
    total_trials = len(args.lambda_grid) * 2  # norm-adjusted + uniform per lambda
    trial_idx = 0
    start_time = time.time()

    for lam in args.lambda_grid:
        for method_name, merge_fn in [
            ("norm_adjusted", merge_norm_adjusted),
            ("uniform", merge_uniform),
        ]:
            trial_idx += 1
            logger.info(
                "=== Trial %d/%d: %s, λ=%.2f ===",
                trial_idx, total_trials, method_name, lam,
            )

            merged_state = merge_fn(base_state, task_vectors, lam=lam)

            # Per-group evaluation
            per_group: dict[str, dict] = {}
            for g_name, (train_dl, test_dl, n_cls) in group_loaders.items():
                res = evaluator.evaluate(
                    encoder_state_dict=merged_state,
                    train_loader=train_dl,
                    test_loader=test_dl,
                    num_classes=n_cls,
                )
                per_group[g_name] = {
                    "accuracy": res.accuracy,
                    "accuracy_ci_low": res.accuracy_ci_low,
                    "accuracy_ci_high": res.accuracy_ci_high,
                    "macro_f1": res.macro_f1,
                }
                logger.info("  %s: acc=%.4f", g_name, res.accuracy)

            # All-group evaluation
            all_res = evaluator.evaluate(
                encoder_state_dict=merged_state,
                train_loader=all_train_dl,
                test_loader=all_test_dl,
                num_classes=total_classes,
            )

            mean_grp_acc = sum(
                r["accuracy"] for r in per_group.values()
            ) / len(per_group)

            gap = (joint_baseline_acc - all_res.accuracy) if joint_baseline_acc else None

            trial_result = {
                "method": method_name,
                "lambda": lam,
                "per_group_results": per_group,
                "all_group_accuracy": all_res.accuracy,
                "mean_group_accuracy": mean_grp_acc,
                "gap_to_joint": gap,
            }
            results.append(trial_result)

            elapsed = time.time() - start_time
            remaining = elapsed / trial_idx * (total_trials - trial_idx)
            logger.info(
                "  AllAcc=%.4f, MeanGrpAcc=%.4f, GapToJoint=%s "
                "(elapsed: %.0fm, est. remaining: %.0fm)",
                all_res.accuracy, mean_grp_acc,
                f"{gap:.4f}" if gap else "n/a",
                elapsed / 60, remaining / 60,
            )

    # --- Save results ---
    with open(output_dir / "norm_adjusted_results.json", "w") as f:
        json.dump({"trials": results, "joint_baseline_acc": joint_baseline_acc}, f, indent=2)

    # --- Print summary ---
    logger.info("")
    logger.info("=" * 100)
    logger.info("NORM-ADJUSTED ABLATION SUMMARY")
    logger.info("=" * 100)
    logger.info(
        "%-15s %8s %12s %12s %12s",
        "Method", "Lambda", "MeanGrpAcc", "AllAcc", "GapToJoint",
    )
    logger.info("-" * 65)

    for r in sorted(results, key=lambda x: -x["all_group_accuracy"]):
        gap_str = f"{r['gap_to_joint']:.4f}" if r["gap_to_joint"] is not None else "n/a"
        logger.info(
            "%-15s %8.2f %12.4f %12.4f %12s",
            r["method"], r["lambda"], r["mean_group_accuracy"],
            r["all_group_accuracy"], gap_str,
        )

    # --- Print per-group comparison at best lambda ---
    logger.info("")
    logger.info("PER-GROUP COMPARISON (best AllAcc per method):")

    for method in ["norm_adjusted", "uniform"]:
        method_trials = [r for r in results if r["method"] == method]
        best = max(method_trials, key=lambda x: x["all_group_accuracy"])
        logger.info(
            "\n  %s (λ=%.2f, AllAcc=%.4f):", method, best["lambda"],
            best["all_group_accuracy"],
        )
        for g in sorted(best["per_group_results"].keys()):
            acc = best["per_group_results"][g]["accuracy"]
            logger.info("    %s: %.4f", g, acc)

    logger.info("")
    logger.info("Results saved to %s", output_dir / "norm_adjusted_results.json")


if __name__ == "__main__":
    main()