"""Advanced merging methods for bioacoustic task vectors.

Implements TIES-Merging, DARE, DELLA, and their combinations.
All operations are performed in FP32 to avoid numerical instability
from DARE's 1/(1-p) rescaling at high drop rates.

Methods:
    - TIES-Merging (Yadav et al., NeurIPS 2023): Trim, Elect sign, Merge
    - DARE (Yu et al., ICML 2024): Drop And REscale
    - DELLA (2024): Magnitude-proportional dropout (replaces DARE's uniform)
    - DARE+TIES: DARE preprocessing followed by TIES merging

References:
    Yadav et al. "Resolving Interference When Merging Models" NeurIPS 2023
    Yu et al. "Language Models are Super Mario" ICML 2024
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def ties_merge(
    task_vectors: list[dict[str, torch.Tensor]],
    trim_fraction: float = 0.8,
    weights: Optional[list[float]] = None,
) -> dict[str, torch.Tensor]:
    """TIES-Merging: Trim, Elect sign, Merge.

    For each parameter:
    1. Trim: Zero out the bottom `trim_fraction` by magnitude in each vector.
    2. Elect sign: Majority vote on parameter sign across vectors.
    3. Merge: Weighted average of values that agree with the elected sign.

    Args:
        task_vectors: List of task vector state dicts.
        trim_fraction: Fraction of smallest-magnitude params to zero (0.0–1.0).
            Higher = more aggressive trimming. Paper default: 0.8.
        weights: Per-vector weights. Default: uniform 1/n.

    Returns:
        Merged task vector (to be added to base model).
    """
    n = len(task_vectors)
    if weights is None:
        weights = [1.0 / n] * n

    assert len(weights) == n
    keys = list(task_vectors[0].keys())
    merged: dict[str, torch.Tensor] = {}

    for key in keys:
        tensors = [tv[key].float() for tv in task_vectors]

        # Step 1: Trim — zero out bottom trim_fraction by magnitude
        trimmed: list[torch.Tensor] = []
        for t in tensors:
            if trim_fraction > 0.0 and t.numel() > 1:
                threshold = torch.quantile(t.abs().flatten(), trim_fraction)
                mask = t.abs() >= threshold
                trimmed.append(t * mask)
            else:
                trimmed.append(t.clone())

        # Step 2: Elect sign — weighted majority vote
        # Sum the signs weighted by magnitude (magnitude-weighted vote)
        sign_votes = torch.zeros_like(trimmed[0])
        for t, w in zip(trimmed, weights):
            sign_votes = sign_votes + w * torch.sign(t)
        elected_sign = torch.sign(sign_votes)
        # Tie-break: use first vector's sign where vote is exactly zero
        tie_mask = elected_sign == 0
        if tie_mask.any():
            elected_sign[tie_mask] = torch.sign(trimmed[0])[tie_mask]

        # Step 3: Merge — average only values agreeing with elected sign
        result = torch.zeros_like(trimmed[0])
        count = torch.zeros_like(trimmed[0])
        for t, w in zip(trimmed, weights):
            agree = torch.sign(t) == elected_sign
            result = result + w * t * agree
            count = count + agree.float()

        # Disjoint merge: divide by count of agreeing vectors
        count = count.clamp(min=1)
        merged[key] = result / count

    return merged


def dare_sparsify(
    task_vector: dict[str, torch.Tensor],
    drop_rate: float = 0.9,
    rescale: bool = True,
    seed: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """DARE: Drop And REscale.

    Randomly drops `drop_rate` fraction of parameters and rescales
    survivors by 1/(1-p) to preserve expected magnitude.

    IMPORTANT: Computed in FP32. At p=0.9, rescaling = 10×;
    at p=0.95, rescaling = 20×. BF16 would overflow/lose precision.

    Args:
        task_vector: Task vector state dict.
        drop_rate: Fraction of parameters to drop (0.0–1.0).
        rescale: Whether to apply 1/(1-p) rescaling. Default True.
        seed: Optional RNG seed for reproducibility.

    Returns:
        Sparsified task vector.
    """
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None

    sparsified: dict[str, torch.Tensor] = {}
    for key, tensor in task_vector.items():
        t = tensor.float()
        if generator is not None:
            mask = torch.bernoulli(
                torch.full_like(t, 1.0 - drop_rate), generator=generator
            )
        else:
            mask = torch.bernoulli(torch.full_like(t, 1.0 - drop_rate))

        if rescale and drop_rate < 1.0:
            sparsified[key] = t * mask / (1.0 - drop_rate)
        else:
            sparsified[key] = t * mask

    return sparsified


def della_sparsify(
    task_vector: dict[str, torch.Tensor],
    target_drop_rate: float = 0.9,
    epsilon: float = 1e-6,
    seed: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """DELLA: Magnitude-proportional dropout.

    Unlike DARE's uniform dropout, DELLA drops parameters with probability
    inversely proportional to their magnitude: small deltas are more likely
    to be dropped. This preserves the most important weight changes.

    Drop probability for parameter i:
        p_i = (1 - |δ_i| / max(|δ|)) * target_drop_rate

    Rescaling is applied per-parameter: survivor i is multiplied by 1/(1-p_i).

    Args:
        task_vector: Task vector state dict.
        target_drop_rate: Average fraction of parameters to drop.
        epsilon: Small constant to avoid division by zero.
        seed: Optional RNG seed for reproducibility.

    Returns:
        Sparsified task vector with magnitude-proportional dropout.
    """
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None

    sparsified: dict[str, torch.Tensor] = {}
    for key, tensor in task_vector.items():
        t = tensor.float()
        abs_t = t.abs()
        max_abs = abs_t.max().clamp(min=epsilon)

        # Drop probability: higher for smaller magnitudes
        # Normalized magnitude in [0, 1], then invert
        normalized_mag = abs_t / max_abs  # 0 = smallest, 1 = largest
        drop_prob = (1.0 - normalized_mag) * target_drop_rate
        drop_prob = drop_prob.clamp(0.0, 0.999)  # avoid exactly 1.0

        # Sample mask: keep_prob = 1 - drop_prob
        keep_prob = 1.0 - drop_prob
        if generator is not None:
            mask = torch.bernoulli(keep_prob, generator=generator)
        else:
            mask = torch.bernoulli(keep_prob)

        # Per-parameter rescaling
        rescale_factor = 1.0 / keep_prob.clamp(min=epsilon)
        sparsified[key] = t * mask * rescale_factor

    return sparsified


def dare_ties_merge(
    task_vectors: list[dict[str, torch.Tensor]],
    drop_rate: float = 0.9,
    trim_fraction: float = 0.2,
    weights: Optional[list[float]] = None,
    seed: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """DARE preprocessing followed by TIES merging.

    First sparsifies each task vector with DARE, then applies TIES
    to resolve sign conflicts among surviving parameters.

    Args:
        task_vectors: List of task vector state dicts.
        drop_rate: DARE drop rate.
        trim_fraction: TIES trim fraction (applied after DARE).
        weights: Per-vector weights for TIES.
        seed: Base seed; each vector gets seed+i for reproducibility.

    Returns:
        Merged task vector.
    """
    sparsified = []
    for i, tv in enumerate(task_vectors):
        s = seed + i if seed is not None else None
        sparsified.append(dare_sparsify(tv, drop_rate=drop_rate, seed=s))

    return ties_merge(sparsified, trim_fraction=trim_fraction, weights=weights)


def della_ties_merge(
    task_vectors: list[dict[str, torch.Tensor]],
    target_drop_rate: float = 0.9,
    trim_fraction: float = 0.2,
    weights: Optional[list[float]] = None,
    seed: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """DELLA preprocessing followed by TIES merging.

    Args:
        task_vectors: List of task vector state dicts.
        target_drop_rate: DELLA target drop rate.
        trim_fraction: TIES trim fraction.
        weights: Per-vector weights for TIES.
        seed: Base seed.

    Returns:
        Merged task vector.
    """
    sparsified = []
    for i, tv in enumerate(task_vectors):
        s = seed + i if seed is not None else None
        sparsified.append(della_sparsify(tv, target_drop_rate=target_drop_rate, seed=s))

    return ties_merge(sparsified, trim_fraction=trim_fraction, weights=weights)


def simple_average(
    task_vectors: list[dict[str, torch.Tensor]],
    weights: Optional[list[float]] = None,
) -> dict[str, torch.Tensor]:
    """Simple weighted average of task vectors (model soups baseline).

    Args:
        task_vectors: List of task vector state dicts.
        weights: Per-vector weights. Default: uniform 1/n.

    Returns:
        Averaged task vector.
    """
    n = len(task_vectors)
    if weights is None:
        weights = [1.0 / n] * n

    keys = list(task_vectors[0].keys())
    averaged: dict[str, torch.Tensor] = {}

    for key in keys:
        averaged[key] = sum(
            w * tv[key].float() for w, tv in zip(weights, task_vectors)
        )

    return averaged


def dare_average(
    task_vectors: list[dict[str, torch.Tensor]],
    drop_rate: float = 0.9,
    weights: Optional[list[float]] = None,
    seed: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """DARE preprocessing followed by simple averaging.

    Args:
        task_vectors: List of task vector state dicts.
        drop_rate: DARE drop rate.
        weights: Per-vector weights.
        seed: Base seed.

    Returns:
        Merged task vector.
    """
    sparsified = []
    for i, tv in enumerate(task_vectors):
        s = seed + i if seed is not None else None
        sparsified.append(dare_sparsify(tv, drop_rate=drop_rate, seed=s))

    return simple_average(sparsified, weights=weights)