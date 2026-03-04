"""Task vector computation, composition, and sanity checking.

Implements the core operations for bioacoustic task arithmetic:
  τ = θ_finetuned_encoder − θ_base_encoder  (encoder weights only)

All operations work on state_dict-level dictionaries of tensors.
DARE rescaling is computed in FP32 to avoid BF16 numerical instability.

Usage:
    # Compute
    python merging/task_vectors.py --config configs/base.yaml

    # Sanity check
    python merging/task_vectors.py --config configs/base.yaml --sanity-check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
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


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def compute_task_vector(
    finetuned_path: str,
    base_path: str,
    save_path: Optional[str] = None,
) -> dict[str, torch.Tensor]:
    """Compute τ = θ_finetuned_encoder − θ_base_encoder.

    Extracts encoder-only weights from each checkpoint and subtracts.
    Only keys present in BOTH state dicts with matching shapes are included.
    The base BEATs checkpoint stores weights under ckpt["model"], while
    fine-tuned checkpoints store encoder weights under ckpt["encoder_state_dict"].

    Args:
        finetuned_path: Path to fine-tuned checkpoint (.pt).
        base_path: Path to pretrained BEATs checkpoint (.pt).
        save_path: If provided, save the task vector to this path.

    Returns:
        Task vector as a state dict of deltas.
    """
    ft_ckpt = torch.load(finetuned_path, map_location="cpu", weights_only=False)
    base_ckpt = torch.load(base_path, map_location="cpu", weights_only=False)

    ft_encoder = ft_ckpt["encoder_state_dict"]
    base_model = base_ckpt["model"]

    # Iterate over fine-tuned keys (subset of base keys, since predictor=None)
    tau: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    for key in ft_encoder:
        if key not in base_model:
            skipped.append(key)
            continue
        if ft_encoder[key].shape != base_model[key].shape:
            skipped.append(key)
            logger.warning(
                "Shape mismatch for %s: ft=%s, base=%s",
                key, ft_encoder[key].shape, base_model[key].shape,
            )
            continue
        tau[key] = ft_encoder[key].float() - base_model[key].float()

    if skipped:
        logger.warning("Skipped %d keys not in base or shape mismatch: %s", len(skipped), skipped)

    logger.info(
        "Task vector: %d keys, %.1f MB",
        len(tau),
        sum(v.numel() * v.element_size() for v in tau.values()) / 1e6,
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(tau, save_path)
        logger.info("Saved to %s", save_path)

    return tau


def apply_task_vector(
    base_path: str,
    tau: dict[str, torch.Tensor],
    scaling: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Apply a single task vector: θ_merged = θ_base + scaling * τ.

    Args:
        base_path: Path to pretrained BEATs checkpoint.
        tau: Task vector state dict.
        scaling: Scaling factor (λ).

    Returns:
        Merged encoder state dict.
    """
    base_ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
    base_model = base_ckpt["model"]

    merged: dict[str, torch.Tensor] = {}
    for key in base_model:
        if key in tau:
            merged[key] = base_model[key].float() + scaling * tau[key].float()
        else:
            merged[key] = base_model[key].clone()

    return merged


def compose_task_vectors(
    base_path: str,
    task_vectors: list[dict[str, torch.Tensor]],
    weights: list[float],
) -> dict[str, torch.Tensor]:
    """Compose multiple task vectors via weighted addition.

    θ_merged = θ_base + Σ_i (w_i · τ_i)

    Args:
        base_path: Path to pretrained BEATs checkpoint.
        task_vectors: List of task vector state dicts.
        weights: Per-vector scaling weights.

    Returns:
        Merged encoder state dict.
    """
    assert len(task_vectors) == len(weights), (
        f"Got {len(task_vectors)} vectors but {len(weights)} weights"
    )

    base_ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
    base_model = base_ckpt["model"]

    merged: dict[str, torch.Tensor] = {}
    for key in base_model:
        merged[key] = base_model[key].float().clone()
        for tv, w in zip(task_vectors, weights):
            if key in tv:
                merged[key] = merged[key] + w * tv[key].float()

    return merged


def negate_task_vector(
    model_encoder: dict[str, torch.Tensor],
    tau: dict[str, torch.Tensor],
    beta: float,
) -> dict[str, torch.Tensor]:
    """Apply domain negation: θ_result = θ_model − β · τ.

    Args:
        model_encoder: Encoder state dict to negate from.
        tau: Task vector to negate.
        beta: Negation strength.

    Returns:
        Negated encoder state dict.
    """
    result: dict[str, torch.Tensor] = {}
    for key in model_encoder:
        result[key] = model_encoder[key].float().clone()
        if key in tau:
            result[key] = result[key] - beta * tau[key].float()
    return result


def apply_merged_tv_to_base(
    base_path: str,
    merged_tv: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Apply an already-merged task vector (e.g., from TIES) to the base model.

    θ_merged = θ_base + merged_tv

    Args:
        base_path: Path to pretrained BEATs checkpoint.
        merged_tv: Pre-merged task vector (output of TIES/DARE).

    Returns:
        Merged encoder state dict.
    """
    base_ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
    base_model = base_ckpt["model"]

    merged: dict[str, torch.Tensor] = {}
    for key in base_model:
        if key in merged_tv:
            merged[key] = base_model[key].float() + merged_tv[key].float()
        else:
            merged[key] = base_model[key].clone()

    return merged


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def sanity_check_reconstruction(
    finetuned_path: str,
    base_path: str,
    tau: dict[str, torch.Tensor],
    atol: float = 1e-5,
) -> bool:
    """Verify that base + 1.0 * τ exactly reconstructs the fine-tuned encoder.

    This is the fundamental correctness test: if this fails, something is
    wrong with the checkpoint format or extraction logic.

    Args:
        finetuned_path: Path to fine-tuned checkpoint.
        base_path: Path to pretrained BEATs checkpoint.
        tau: Task vector to verify.
        atol: Absolute tolerance for floating-point comparison.

    Returns:
        True if reconstruction is exact (within tolerance).
    """
    reconstructed = apply_task_vector(base_path, tau, scaling=1.0)

    ft_ckpt = torch.load(finetuned_path, map_location="cpu", weights_only=False)
    ft_encoder = ft_ckpt["encoder_state_dict"]

    max_diff = 0.0
    failed_keys: list[str] = []

    for key in tau:
        if key not in reconstructed or key not in ft_encoder:
            failed_keys.append(key)
            continue
        diff = (reconstructed[key].float() - ft_encoder[key].float()).abs().max().item()
        max_diff = max(max_diff, diff)
        if diff > atol:
            failed_keys.append(key)
            logger.error("Key %s: max diff = %.2e (exceeds atol=%.2e)", key, diff, atol)

    if failed_keys:
        logger.error(
            "SANITY CHECK FAILED: %d/%d keys exceed tolerance. Max diff: %.2e",
            len(failed_keys), len(tau), max_diff,
        )
        return False

    logger.info(
        "SANITY CHECK PASSED: %d keys, max diff = %.2e (atol=%.2e)",
        len(tau), max_diff, atol,
    )
    return True


# ---------------------------------------------------------------------------
# Key consistency check
# ---------------------------------------------------------------------------

def verify_key_consistency(
    task_vectors: dict[str, dict[str, torch.Tensor]],
) -> bool:
    """Verify all task vectors share exactly the same set of keys.

    This is required for valid task arithmetic — if vectors have different
    keys, the merging operations are not well-defined.

    Args:
        task_vectors: Dict mapping group name → task vector state dict.

    Returns:
        True if all key sets are identical.
    """
    names = list(task_vectors.keys())
    reference_keys = set(task_vectors[names[0]].keys())

    all_match = True
    for name in names[1:]:
        keys = set(task_vectors[name].keys())
        if keys != reference_keys:
            only_in_ref = reference_keys - keys
            only_in_curr = keys - reference_keys
            logger.error(
                "Key mismatch: %s vs %s. Only in %s: %s. Only in %s: %s",
                names[0], name,
                names[0], only_in_ref,
                name, only_in_curr,
            )
            all_match = False

    if all_match:
        logger.info("All %d task vectors share %d keys", len(names), len(reference_keys))
    return all_match


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute and verify task vectors")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument(
        "--sanity-check", action="store_true",
        help="Run reconstruction sanity check after computing",
    )
    parser.add_argument(
        "--groups", nargs="*", default=None,
        help="Specific groups to process (default: all available)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    base_path = cfg["paths"]["beats_base_checkpoint"]
    finetuned_dir = Path(cfg["paths"]["finetuned"])
    tv_dir = Path(cfg["paths"]["task_vectors"])
    tv_dir.mkdir(parents=True, exist_ok=True)

    # Discover available fine-tuned models
    all_groups = [
        "G1_passerines", "G2_nonpasserine_birds", "G3_raptors_waterbirds",
        "G4_marine_mammals", "G5_amphibians",
    ]
    if args.groups:
        all_groups = args.groups

    computed: dict[str, dict[str, torch.Tensor]] = {}
    all_passed = True

    for group in all_groups:
        ft_path = finetuned_dir / group / "best_model.pt"
        if not ft_path.exists():
            logger.warning("Checkpoint not found: %s — skipping", ft_path)
            continue

        save_path = tv_dir / f"tau_{group}.pt"
        logger.info("Computing task vector for %s", group)

        tau = compute_task_vector(
            finetuned_path=str(ft_path),
            base_path=base_path,
            save_path=str(save_path),
        )
        computed[group] = tau

        if args.sanity_check:
            passed = sanity_check_reconstruction(
                finetuned_path=str(ft_path),
                base_path=base_path,
                tau=tau,
            )
            if not passed:
                all_passed = False

    # Verify key consistency across all computed vectors
    if len(computed) > 1:
        verify_key_consistency(computed)

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY: Computed %d/%d task vectors", len(computed), len(all_groups))
    if args.sanity_check:
        status = "ALL PASSED" if all_passed else "SOME FAILED"
        logger.info("Sanity check: %s", status)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()