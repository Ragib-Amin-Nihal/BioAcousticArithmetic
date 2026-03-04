"""Aggregate per-pair LMC results into a single results file + barrier matrix.

Run after all parallel LMC jobs finish.

Usage:
    python scripts/aggregate_lmc.py --input results/lmc/ --output results/lmc/
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate LMC pair results")
    parser.add_argument("--input", type=str, default="results/lmc")
    parser.add_argument("--output", type=str, default="results/lmc")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all per-pair result files
    pair_files = sorted(input_dir.glob("lmc_*_vs_*.json"))
    if not pair_files:
        logger.error("No per-pair files found in %s", input_dir)
        return

    logger.info("Found %d pair result files", len(pair_files))

    # Merge all pairs
    all_pairs: list[dict] = []
    all_groups: set[str] = set()

    for pf in pair_files:
        with open(pf) as f:
            data = json.load(f)
        for pair_result in data["pairs"]:
            all_pairs.append(pair_result)
            all_groups.add(pair_result["model_a"])
            all_groups.add(pair_result["model_b"])
        logger.info("  Loaded %s: %d pair-direction results", pf.name, len(data["pairs"]))

    # Build barrier matrix (max barrier across both eval directions per pair)
    groups = sorted(all_groups)
    barrier_matrix: dict[str, dict[str, float]] = {g: {} for g in groups}

    for ga in groups:
        for gb in groups:
            if ga == gb:
                continue
            # Find all results for this unordered pair
            pair_barriers = [
                p["barrier_height"] for p in all_pairs
                if {p["model_a"], p["model_b"]} == {ga, gb}
            ]
            if pair_barriers:
                barrier_matrix[ga][gb] = max(pair_barriers)

    # Save combined results
    combined = {
        "pairs": all_pairs,
        "barrier_matrix": barrier_matrix,
    }
    with open(output_dir / "lmc_results.json", "w") as f:
        json.dump(combined, f, indent=2)

    # Save barrier matrix table
    lines = ["Barrier Height Matrix (max across both eval directions)\n"]
    header = f"{'':>25s}" + "".join(f"{g:>15s}" for g in groups)
    lines.append(header)
    lines.append("-" * len(header))

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
        lines.append(row)

    table_text = "\n".join(lines) + "\n"
    with open(output_dir / "barrier_matrix.txt", "w") as f:
        f.write(table_text)

    # Print summary
    logger.info("\n%s", table_text)

    # Flag high barriers
    for ga in groups:
        for gb in groups:
            if ga >= gb:
                continue
            b = barrier_matrix.get(ga, {}).get(gb)
            if b is not None and b > 0.1:
                logger.warning("HIGH BARRIER: %s ↔ %s = %.4f", ga, gb, b)

    logger.info("Combined results: %s/lmc_results.json", output_dir)
    logger.info("Barrier matrix: %s/barrier_matrix.txt", output_dir)


if __name__ == "__main__":
    main()