"""Sparsity and structure analysis for bioacoustic task vectors.

Characterizes the geometry of fine-tuning deltas in weight space:
- Magnitude distributions (L1, L2 norms, near-zero fractions)
- Pairwise cosine similarity and sign agreement
- Per-layer statistics

This analysis is independent of the merging experiments and contributes
to understanding how bioacoustic fine-tuning differs from NLP fine-tuning
(where deltas are typically within ±0.002).

Usage:
    python analysis/sparsity.py --config configs/base.yaml --output results/analysis/
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def analyze_task_vector(
    tau: dict[str, torch.Tensor],
    name: str = "",
) -> dict:
    """Characterize the geometry of a single bioacoustic task vector.

    Computes global and per-layer statistics: norms, sparsity,
    magnitude distribution.

    Args:
        tau: Task vector state dict.
        name: Identifier for logging.

    Returns:
        Dict with global and per-layer analysis.
    """
    all_params = torch.cat([v.flatten().float() for v in tau.values()])
    magnitudes = all_params.abs()

    analysis = {
        "name": name,
        "n_params": int(len(all_params)),
        "n_keys": len(tau),
        # Global norms
        "l1_norm": float(magnitudes.sum().item()),
        "l2_norm": float(all_params.norm(2).item()),
        "linf_norm": float(magnitudes.max().item()),
        # Magnitude statistics
        "mean_magnitude": float(magnitudes.mean().item()),
        "median_magnitude": float(magnitudes.median().item()),
        "std_magnitude": float(magnitudes.std().item()),
        "max_magnitude": float(magnitudes.max().item()),
        # Sparsity at various thresholds
        "frac_near_zero_1e-6": float((magnitudes < 1e-6).float().mean().item()),
        "frac_near_zero_1e-5": float((magnitudes < 1e-5).float().mean().item()),
        "frac_near_zero_1e-4": float((magnitudes < 1e-4).float().mean().item()),
        "frac_near_zero_1e-3": float((magnitudes < 1e-3).float().mean().item()),
        "frac_near_zero_1e-2": float((magnitudes < 1e-2).float().mean().item()),
        # Sign distribution
        "frac_positive": float((all_params > 0).float().mean().item()),
        "frac_negative": float((all_params < 0).float().mean().item()),
        "frac_zero": float((all_params == 0).float().mean().item()),
    }

    # Per-layer analysis
    layer_stats: dict[str, dict] = {}
    for key, tensor in tau.items():
        flat = tensor.flatten().float()
        layer_stats[key] = {
            "n_params": int(flat.numel()),
            "l2_norm": float(flat.norm(2).item()),
            "mean_magnitude": float(flat.abs().mean().item()),
            "max_magnitude": float(flat.abs().max().item()),
            "sparsity_1e-4": float((flat.abs() < 1e-4).float().mean().item()),
            "sparsity_1e-3": float((flat.abs() < 1e-3).float().mean().item()),
        }
    analysis["per_layer"] = layer_stats

    return analysis


def pairwise_analysis(
    tau_a: dict[str, torch.Tensor],
    tau_b: dict[str, torch.Tensor],
    name_a: str = "A",
    name_b: str = "B",
) -> dict:
    """Compare two task vectors: cosine similarity, sign agreement.

    Args:
        tau_a: First task vector.
        tau_b: Second task vector.
        name_a: Name of first vector.
        name_b: Name of second vector.

    Returns:
        Dict with pairwise metrics.
    """
    shared_keys = sorted(set(tau_a.keys()) & set(tau_b.keys()))
    flat_a = torch.cat([tau_a[k].flatten().float() for k in shared_keys])
    flat_b = torch.cat([tau_b[k].flatten().float() for k in shared_keys])

    # Global cosine similarity
    cosine_sim = torch.nn.functional.cosine_similarity(
        flat_a.unsqueeze(0), flat_b.unsqueeze(0)
    ).item()

    # Sign agreement (fraction of params with same sign)
    sign_a = torch.sign(flat_a)
    sign_b = torch.sign(flat_b)
    sign_agreement = (sign_a == sign_b).float().mean().item()

    # Magnitude correlation (Pearson)
    mag_a = flat_a.abs()
    mag_b = flat_b.abs()
    mag_corr = float(torch.corrcoef(torch.stack([mag_a, mag_b]))[0, 1].item())

    # Per-layer cosine similarity
    per_layer_cosine: dict[str, float] = {}
    for key in shared_keys:
        a = tau_a[key].flatten().float()
        b = tau_b[key].flatten().float()
        if a.norm() > 0 and b.norm() > 0:
            cos = torch.nn.functional.cosine_similarity(
                a.unsqueeze(0), b.unsqueeze(0)
            ).item()
        else:
            cos = 0.0
        per_layer_cosine[key] = cos

    return {
        "pair": f"{name_a}_vs_{name_b}",
        "cosine_similarity": float(cosine_sim),
        "sign_agreement": float(sign_agreement),
        "magnitude_correlation": float(mag_corr),
        "n_shared_keys": len(shared_keys),
        "n_shared_params": int(flat_a.numel()),
        "per_layer_cosine": per_layer_cosine,
    }


def compute_all_pairwise(
    task_vectors: dict[str, dict[str, torch.Tensor]],
) -> list[dict]:
    """Compute pairwise analysis for all pairs of task vectors.

    Args:
        task_vectors: Dict mapping name → task vector.

    Returns:
        List of pairwise analysis dicts.
    """
    names = sorted(task_vectors.keys())
    results = []
    for name_a, name_b in itertools.combinations(names, 2):
        result = pairwise_analysis(
            task_vectors[name_a], task_vectors[name_b],
            name_a, name_b,
        )
        results.append(result)
        logger.info(
            "  %s ↔ %s: cosine=%.4f, sign_agree=%.4f",
            name_a, name_b, result["cosine_similarity"], result["sign_agreement"],
        )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Sparsity & structure analysis")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--output", type=str, default="results/analysis")
    parser.add_argument("--groups", nargs="*", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tv_dir = Path(cfg["paths"]["task_vectors"])
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = args.groups or [
        "G1_passerines", "G2_nonpasserine_birds", "G3_raptors_waterbirds",
        "G4_marine_mammals", "G5_amphibians",
    ]

    # Load task vectors
    task_vectors: dict[str, dict[str, torch.Tensor]] = {}
    for group in groups:
        tv_path = tv_dir / f"tau_{group}.pt"
        if not tv_path.exists():
            logger.warning("Not found: %s", tv_path)
            continue
        task_vectors[group] = torch.load(str(tv_path), map_location="cpu")
        logger.info("Loaded τ_%s", group)

    # Per-vector analysis
    logger.info("=" * 60)
    logger.info("PER-VECTOR ANALYSIS")
    logger.info("=" * 60)

    individual_analyses = {}
    for name, tau in task_vectors.items():
        analysis = analyze_task_vector(tau, name)
        individual_analyses[name] = analysis
        logger.info(
            "  %s: %d params, L2=%.2f, mean_mag=%.2e, "
            "sparsity@1e-3=%.1f%%, sparsity@1e-4=%.1f%%",
            name, analysis["n_params"], analysis["l2_norm"],
            analysis["mean_magnitude"],
            analysis["frac_near_zero_1e-3"] * 100,
            analysis["frac_near_zero_1e-4"] * 100,
        )

    # Pairwise analysis
    logger.info("=" * 60)
    logger.info("PAIRWISE ANALYSIS")
    logger.info("=" * 60)

    pairwise = compute_all_pairwise(task_vectors)

    # Save results (strip per-layer data for the summary, keep in detail file)
    summary = {
        "individual": {
            name: {k: v for k, v in a.items() if k != "per_layer"}
            for name, a in individual_analyses.items()
        },
        "pairwise": [
            {k: v for k, v in p.items() if k != "per_layer_cosine"}
            for p in pairwise
        ],
    }

    with open(out_dir / "sparsity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save full detail (including per-layer)
    detail = {
        "individual": individual_analyses,
        "pairwise": pairwise,
    }
    with open(out_dir / "sparsity_detail.json", "w") as f:
        json.dump(detail, f, indent=2, default=str)

    logger.info("Results saved to %s", out_dir)

    # Print cosine similarity matrix
    names = sorted(task_vectors.keys())
    logger.info("\nCosine Similarity Matrix:")
    header = f"{'':>25s}" + "".join(f"{n:>15s}" for n in names)
    logger.info(header)
    cos_lookup = {(p["pair"].split("_vs_")[0], p["pair"].split("_vs_")[1]): p["cosine_similarity"] for p in pairwise}
    for na in names:
        row = f"{na:>25s}"
        for nb in names:
            if na == nb:
                row += f"{'1.0000':>15s}"
            elif (na, nb) in cos_lookup:
                row += f"{cos_lookup[(na, nb)]:>15.4f}"
            elif (nb, na) in cos_lookup:
                row += f"{cos_lookup[(nb, na)]:>15.4f}"
            else:
                row += f"{'n/a':>15s}"
        logger.info(row)


if __name__ == "__main__":
    main()