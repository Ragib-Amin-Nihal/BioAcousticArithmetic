"""Recompute LMC barriers with the correct metric.

The original barrier formula (max_loss - avg_endpoints) is invalid when
endpoints have very different losses — which always happens for cross-task
evaluation where α=0 means "wrong encoder + this task's head."

Correct metric: does any INTERMEDIATE α have loss exceeding BOTH endpoints?
    barrier = max(0, max_intermediate_loss - max(endpoint_losses))

If the curve is monotonic (which it should be for same-basin models),
barrier = 0 regardless of absolute loss values.

Additionally computes:
    - near_end_degradation: fractional loss increase at α=0.9 vs α=1.0
      (how much does a 10% contribution from the other model hurt?)
    - monotonic: whether the curve is strictly monotonic (no bumps)

Usage:
    python scripts/recompute_lmc_barriers.py --input results/lmc --output results/lmc
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def compute_correct_barrier(points: list[dict]) -> dict:
    """Compute the correct barrier metric for an interpolation curve.

    Args:
        points: List of {"alpha": float, "loss": float, "accuracy": float}.

    Returns:
        Dict with corrected metrics.
    """
    losses = [p["loss"] for p in points]
    accs = [p["accuracy"] for p in points]
    alphas = [p["alpha"] for p in points]

    endpoint_max_loss = max(losses[0], losses[-1])
    endpoint_min_loss = min(losses[0], losses[-1])
    intermediate_losses = losses[1:-1]  # exclude endpoints

    # True barrier: does any intermediate point exceed BOTH endpoints?
    if intermediate_losses:
        max_intermediate = max(intermediate_losses)
        true_barrier = max(0.0, max_intermediate - endpoint_max_loss)
    else:
        true_barrier = 0.0

    # Monotonicity check
    # For the "own task" direction (α=1 is best), loss should decrease as α→1
    # For the "other task" direction (α=0 is best), loss should increase as α→1
    increasing = all(losses[i] <= losses[i + 1] + 1e-6 for i in range(len(losses) - 1))
    decreasing = all(losses[i] >= losses[i + 1] - 1e-6 for i in range(len(losses) - 1))
    is_monotonic = increasing or decreasing

    # Near-endpoint degradation: how much does α=0.9 differ from α=1.0?
    # (measures sensitivity to small contribution from the other model)
    # Find the "good" endpoint (lower loss)
    if losses[-1] < losses[0]:
        # α=1.0 is the good endpoint, check α=0.9
        good_loss = losses[-1]
        near_loss = next((p["loss"] for p in points if abs(p["alpha"] - 0.9) < 0.01), None)
        good_acc = accs[-1]
        near_acc = next((p["accuracy"] for p in points if abs(p["alpha"] - 0.9) < 0.01), None)
    else:
        # α=0.0 is the good endpoint, check α=0.1
        good_loss = losses[0]
        near_loss = next((p["loss"] for p in points if abs(p["alpha"] - 0.1) < 0.01), None)
        good_acc = accs[0]
        near_acc = next((p["accuracy"] for p in points if abs(p["alpha"] - 0.1) < 0.01), None)

    if near_loss is not None and good_loss > 0:
        loss_degradation_frac = (near_loss - good_loss) / good_loss
    else:
        loss_degradation_frac = None

    if near_acc is not None and good_acc > 0:
        acc_degradation_abs = good_acc - near_acc
    else:
        acc_degradation_abs = None

    return {
        "true_barrier": true_barrier,
        "is_monotonic": is_monotonic,
        "loss_degradation_at_10pct": loss_degradation_frac,
        "acc_drop_at_10pct": acc_degradation_abs,
        "endpoint_loss_good": good_loss,
        "endpoint_loss_bad": endpoint_max_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute LMC barriers")
    parser.add_argument("--input", type=str, default="results/lmc")
    parser.add_argument("--output", type=str, default="results/lmc")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # Load combined results
    results_path = input_dir / "lmc_results.json"
    if not results_path.exists():
        logger.error("No lmc_results.json found. Run aggregate_lmc.py first.")
        return

    with open(results_path) as f:
        data = json.load(f)

    all_groups: set[str] = set()
    corrected_pairs: list[dict] = []

    for pair in data["pairs"]:
        metrics = compute_correct_barrier(pair["interpolation_points"])
        corrected = {**pair, **metrics}
        corrected_pairs.append(corrected)
        all_groups.add(pair["model_a"])
        all_groups.add(pair["model_b"])

    # Build corrected barrier matrix
    groups = sorted(all_groups)
    barrier_matrix: dict[str, dict[str, float]] = {g: {} for g in groups}

    for ga in groups:
        for gb in groups:
            if ga == gb:
                continue
            pair_barriers = [
                p["true_barrier"] for p in corrected_pairs
                if {p["model_a"], p["model_b"]} == {ga, gb}
            ]
            if pair_barriers:
                barrier_matrix[ga][gb] = max(pair_barriers)

    # Save corrected results
    corrected_data = {
        "pairs": corrected_pairs,
        "barrier_matrix": barrier_matrix,
    }
    with open(output_dir / "lmc_results_corrected.json", "w") as f:
        json.dump(corrected_data, f, indent=2)

    # Print corrected barrier matrix
    logger.info("")
    logger.info("CORRECTED Barrier Height Matrix (true barrier = max(0, max_intermediate - max_endpoint))")
    logger.info("")
    header = f"{'':>25s}" + "".join(f"{g:>15s}" for g in groups)
    logger.info(header)
    logger.info("-" * len(header))
    for g_row in groups:
        row = f"{g_row:>25s}"
        for g_col in groups:
            if g_row == g_col:
                row += f"{'—':>15s}"
            elif g_col in barrier_matrix.get(g_row, {}):
                val = barrier_matrix[g_row][g_col]
                row += f"{val:>15.4f}"
            else:
                row += f"{'n/a':>15s}"
        logger.info(row)

    # Print per-pair detail table
    logger.info("")
    logger.info("Per-direction detail:")
    logger.info(
        f"{'Pair':<50s} {'EvalOn':<25s} {'Barrier':>8s} {'Mono':>6s} "
        f"{'LossDeg%':>9s} {'AccDrop':>8s}"
    )
    logger.info("-" * 110)

    for p in sorted(corrected_pairs, key=lambda x: (x["model_a"], x["model_b"], x["eval_group"])):
        pair_str = f"{p['model_a']} ↔ {p['model_b']}"
        mono = "✓" if p["is_monotonic"] else "✗"
        loss_deg = f"{p['loss_degradation_at_10pct']*100:.1f}%" if p["loss_degradation_at_10pct"] is not None else "n/a"
        acc_drop = f"{p['acc_drop_at_10pct']:.4f}" if p["acc_drop_at_10pct"] is not None else "n/a"
        barrier_flag = "" if p["true_barrier"] < 0.01 else " ⚠️"

        logger.info(
            f"{pair_str:<50s} {p['eval_group']:<25s} {p['true_barrier']:>8.4f}{barrier_flag:2s} {mono:>6s} "
            f"{loss_deg:>9s} {acc_drop:>8s}"
        )

    # Summary
    all_barriers = [p["true_barrier"] for p in corrected_pairs]
    all_monotonic = all(p["is_monotonic"] for p in corrected_pairs)
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("  Max true barrier across all pairs: %.4f", max(all_barriers))
    logger.info("  All curves monotonic: %s", "YES ✓" if all_monotonic else "NO ✗")
    if all_monotonic and max(all_barriers) < 0.01:
        logger.info("  CONCLUSION: All models share a loss basin. Merging is safe.")
    elif max(all_barriers) < 0.1:
        logger.info("  CONCLUSION: Minor barriers detected but within safe range.")
    else:
        logger.info("  CONCLUSION: Significant barriers detected. Check non-monotonic pairs.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()