"""Experiment 1: Linear Mode Connectivity (LMC) Verification.

Tests whether BEATs models fine-tuned on different species groups remain in
the same loss basin. For each pair of fine-tuned encoders (θ_A, θ_B),
evaluates performance along the linear interpolation path:

    θ(α) = α · θ_A + (1 − α) · θ_B,  α ∈ {0.0, 0.1, ..., 1.0}

If LMC holds, the loss along the path never exceeds max(L(θ_A), L(θ_B))
by more than a small margin. A convex loss barrier indicates models are in
different basins and merging will fail.

For pairs with non-overlapping label spaces (e.g., G1 passerines ↔ G4 marine
mammals), we evaluate each interpolated encoder on EACH endpoint group's test
set using that group's trained classification head. This gives two loss curves
per pair — one per task.

Usage:
    python evaluation/lmc_analysis.py \\
        --config configs/base.yaml \\
        --output results/lmc/ \\
        --groups G1_passerines G2_nonpasserine_birds G3_raptors_waterbirds \\
                 G4_marine_mammals G5_amphibians
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import yaml

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.beats_classifier import BEATsClassifier

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
class InterpolationPoint:
    """Result at a single interpolation point."""
    alpha: float
    loss: float
    accuracy: float


@dataclass
class LMCPairResult:
    """LMC result for one pair of models, evaluated on one task."""
    model_a: str
    model_b: str
    eval_group: str
    interpolation_points: list[InterpolationPoint]
    barrier_height: float
    accuracy_at_midpoint: float
    endpoint_avg_loss: float


@dataclass
class LMCExperimentResult:
    """Full LMC experiment results across all pairs."""
    pairs: list[LMCPairResult] = field(default_factory=list)
    barrier_matrix: dict[str, dict[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core LMC evaluation
# ---------------------------------------------------------------------------

def interpolate_encoders(
    encoder_a: dict[str, torch.Tensor],
    encoder_b: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Linearly interpolate between two encoder state dicts.

    θ(α) = α · θ_A + (1 − α) · θ_B

    Args:
        encoder_a: First encoder state dict.
        encoder_b: Second encoder state dict.
        alpha: Interpolation coefficient. α=1 → model_a, α=0 → model_b.

    Returns:
        Interpolated encoder state dict.
    """
    assert set(encoder_a.keys()) == set(encoder_b.keys()), (
        "Encoder key sets do not match"
    )
    return {
        key: alpha * encoder_a[key].float() + (1 - alpha) * encoder_b[key].float()
        for key in encoder_a
    }


@torch.no_grad()
def evaluate_encoder_with_head(
    encoder_state_dict: dict[str, torch.Tensor],
    head_weight: torch.Tensor,
    head_bias: torch.Tensor,
    base_checkpoint_path: str,
    eval_loader: DataLoader,
    num_classes: int,
    device: str = "cuda",
) -> tuple[float, float]:
    """Evaluate an encoder using a pre-trained classification head.

    Loads the encoder weights into a BEATsClassifier, attaches the given
    head weights, and evaluates on the provided data.

    Args:
        encoder_state_dict: Encoder weights to evaluate.
        head_weight: Classification head weight matrix (num_classes, 768).
        head_bias: Classification head bias vector (num_classes,).
        base_checkpoint_path: Path to BEATs base checkpoint (for model init).
        eval_loader: DataLoader yielding (audio, labels).
        num_classes: Number of classes.
        device: Torch device.

    Returns:
        (loss, accuracy) tuple.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    model = BEATsClassifier(base_checkpoint_path, num_classes)
    model.load_encoder_state_dict(encoder_state_dict, strict=False)
    model.classifier.weight.data = head_weight.clone()
    model.classifier.bias.data = head_bias.clone()
    model = model.to(dev).eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for audio, labels in eval_loader:
            audio, labels = audio.to(dev), labels.to(dev)
            logits = model(audio)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

    loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)

    del model
    torch.cuda.empty_cache()

    return loss, accuracy


def run_lmc_pair(
    encoder_a: dict[str, torch.Tensor],
    encoder_b: dict[str, torch.Tensor],
    head_weight: torch.Tensor,
    head_bias: torch.Tensor,
    base_checkpoint_path: str,
    eval_loader: DataLoader,
    num_classes: int,
    name_a: str,
    name_b: str,
    eval_group: str,
    n_alphas: int = 11,
    device: str = "cuda",
) -> LMCPairResult:
    """Run LMC analysis for one pair of models on one evaluation task.

    Evaluates performance at n_alphas points along the linear interpolation
    path. Computes the loss barrier height: max loss along path minus the
    average of endpoint losses.

    Args:
        encoder_a: First fine-tuned encoder.
        encoder_b: Second fine-tuned encoder.
        head_weight: Classification head weights for eval_group.
        head_bias: Classification head bias for eval_group.
        base_checkpoint_path: Path to BEATs base checkpoint.
        eval_loader: DataLoader for the evaluation group.
        num_classes: Number of classes in eval_group.
        name_a: Name of model A (for logging).
        name_b: Name of model B (for logging).
        eval_group: Name of evaluation group (for logging).
        n_alphas: Number of interpolation points.
        device: Torch device.

    Returns:
        LMCPairResult with all interpolation points and barrier height.
    """
    alphas = np.linspace(0.0, 1.0, n_alphas)
    points: list[InterpolationPoint] = []

    logger.info(
        "LMC: %s ↔ %s (eval on %s, %d points)",
        name_a, name_b, eval_group, n_alphas,
    )

    for alpha in alphas:
        interp = interpolate_encoders(encoder_a, encoder_b, alpha)
        loss, acc = evaluate_encoder_with_head(
            interp, head_weight, head_bias,
            base_checkpoint_path, eval_loader, num_classes, device,
        )
        points.append(InterpolationPoint(alpha=float(alpha), loss=loss, accuracy=acc))
        logger.info(
            "  α=%.2f: loss=%.4f, acc=%.4f", alpha, loss, acc,
        )

    # Barrier height
    endpoint_avg_loss = (points[0].loss + points[-1].loss) / 2.0
    max_loss = max(p.loss for p in points)
    barrier = max_loss - endpoint_avg_loss

    # Midpoint accuracy (α=0.5)
    midpoint_idx = n_alphas // 2
    midpoint_acc = points[midpoint_idx].accuracy

    result = LMCPairResult(
        model_a=name_a,
        model_b=name_b,
        eval_group=eval_group,
        interpolation_points=points,
        barrier_height=barrier,
        accuracy_at_midpoint=midpoint_acc,
        endpoint_avg_loss=endpoint_avg_loss,
    )

    logger.info(
        "  Barrier height: %.4f | Midpoint acc: %.4f", barrier, midpoint_acc,
    )

    return result


def extract_head_from_checkpoint(
    checkpoint_path: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Extract the trained classification head from a fine-tuned checkpoint.

    Args:
        checkpoint_path: Path to fine-tuned checkpoint (.pt).

    Returns:
        (head_weight, head_bias, num_classes) tuple.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]

    # The classifier head keys in BEATsClassifier
    head_weight = state["classifier.weight"]
    head_bias = state["classifier.bias"]
    num_classes = head_weight.shape[0]

    return head_weight, head_bias, num_classes


# ---------------------------------------------------------------------------
# Full experiment runner
# ---------------------------------------------------------------------------

def run_lmc_experiment(
    base_checkpoint_path: str,
    finetuned_dir: str,
    group_names: list[str],
    group_eval_loaders: dict[str, DataLoader],
    n_alphas: int = 11,
    output_dir: Optional[str] = None,
    device: str = "cuda",
    only_pair: Optional[tuple[str, str]] = None,
) -> LMCExperimentResult:
    """Run full LMC experiment across all pairs of fine-tuned models.

    For each pair (A, B), evaluates on BOTH A's and B's test sets.
    This handles the key issue of non-overlapping label spaces across groups.

    Args:
        base_checkpoint_path: Path to BEATs base checkpoint.
        finetuned_dir: Directory containing fine-tuned checkpoints.
        group_names: List of group names to include.
        group_eval_loaders: Dict mapping group_name → DataLoader for evaluation.
        n_alphas: Number of interpolation points.
        output_dir: If provided, save results here.
        device: Torch device.
        only_pair: If set, run only this (group_a, group_b) pair.

    Returns:
        LMCExperimentResult with all pair results and barrier matrix.
    """
    finetuned_path = Path(finetuned_dir)
    experiment = LMCExperimentResult()

    # Load all encoders and heads
    encoders: dict[str, dict[str, torch.Tensor]] = {}
    heads: dict[str, tuple[torch.Tensor, torch.Tensor, int]] = {}

    for group in group_names:
        ckpt_path = finetuned_path / group / "best_model.pt"
        if not ckpt_path.exists():
            logger.warning("Checkpoint not found: %s", ckpt_path)
            continue

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        encoders[group] = ckpt["encoder_state_dict"]
        heads[group] = extract_head_from_checkpoint(str(ckpt_path))
        logger.info("Loaded %s: %d encoder keys, %d classes",
                     group, len(encoders[group]), heads[group][2])

    available = sorted(encoders.keys())
    logger.info("Running LMC on %d groups: %s", len(available), available)

    # Initialize barrier matrix
    barrier_matrix: dict[str, dict[str, float]] = {
        g: {} for g in available
    }

    # Run all pairwise comparisons (or just one if only_pair is set)
    if only_pair:
        pair_list = [only_pair]
    else:
        pair_list = list(itertools.combinations(available, 2))

    for name_a, name_b in pair_list:
        encoder_a = encoders[name_a]
        encoder_b = encoders[name_b]

        # Evaluate on BOTH groups' test sets
        for eval_group in [name_a, name_b]:
            if eval_group not in group_eval_loaders:
                logger.warning("No eval loader for %s, skipping", eval_group)
                continue

            head_w, head_b, n_classes = heads[eval_group]
            eval_loader = group_eval_loaders[eval_group]

            result = run_lmc_pair(
                encoder_a=encoder_a,
                encoder_b=encoder_b,
                head_weight=head_w,
                head_bias=head_b,
                base_checkpoint_path=base_checkpoint_path,
                eval_loader=eval_loader,
                num_classes=n_classes,
                name_a=name_a,
                name_b=name_b,
                eval_group=eval_group,
                n_alphas=n_alphas,
                device=device,
            )
            experiment.pairs.append(result)

        # Store max barrier across both eval directions
        pair_barriers = [
            r.barrier_height for r in experiment.pairs
            if (r.model_a == name_a and r.model_b == name_b)
        ]
        max_barrier = max(pair_barriers) if pair_barriers else 0.0
        barrier_matrix[name_a][name_b] = max_barrier
        barrier_matrix[name_b][name_a] = max_barrier

    experiment.barrier_matrix = barrier_matrix

    # Save results
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Save full results
        serializable = {
            "pairs": [
                {
                    "model_a": r.model_a,
                    "model_b": r.model_b,
                    "eval_group": r.eval_group,
                    "barrier_height": r.barrier_height,
                    "accuracy_at_midpoint": r.accuracy_at_midpoint,
                    "endpoint_avg_loss": r.endpoint_avg_loss,
                    "interpolation_points": [
                        {"alpha": p.alpha, "loss": p.loss, "accuracy": p.accuracy}
                        for p in r.interpolation_points
                    ],
                }
                for r in experiment.pairs
            ],
            "barrier_matrix": barrier_matrix,
        }

        # Use pair-specific filename when running a single pair
        if only_pair:
            fname = f"lmc_{only_pair[0]}_vs_{only_pair[1]}.json"
        else:
            fname = "lmc_results.json"

        with open(out_path / fname, "w") as f:
            json.dump(serializable, f, indent=2)

        if not only_pair:
            # Save barrier matrix as a compact table (only for full run)
            _save_barrier_table(barrier_matrix, out_path / "barrier_matrix.txt")

        logger.info("Results saved to %s/%s", out_path, fname)

    return experiment


def _save_barrier_table(
    barrier_matrix: dict[str, dict[str, float]],
    path: Path,
) -> None:
    """Save barrier matrix as a human-readable table."""
    groups = sorted(barrier_matrix.keys())
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

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for LMC experiment."""
    parser = argparse.ArgumentParser(
        description="Experiment 1: Linear Mode Connectivity"
    )
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--output", type=str, default="results/lmc")
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument(
        "--pair", nargs=2, default=None, metavar=("GROUP_A", "GROUP_B"),
        help="Run a single pair only (for parallel execution across machines)",
    )
    parser.add_argument("--n-alphas", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    groups = args.groups or [
        "G1_passerines", "G2_nonpasserine_birds", "G3_raptors_waterbirds",
        "G4_marine_mammals", "G5_amphibians",
    ]
    n_alphas = args.n_alphas or cfg["evaluation"]["lmc_alphas"]

    logger.info(
        "LMC experiment: %d groups, %d interpolation points",
        len(groups), n_alphas,
    )

    # If --pair is specified, only run that one pair
    single_pair = None
    if args.pair:
        single_pair = tuple(args.pair)
        # Still need both groups in the groups list for loading
        for g in single_pair:
            if g not in groups:
                groups.append(g)
        logger.info("Single-pair mode: %s ↔ %s", single_pair[0], single_pair[1])

    # -----------------------------------------------------------------------
    # DATA LOADING HOOK
    # -----------------------------------------------------------------------
    # This section must be connected to your actual AudioDataset and data
    # loading pipeline. Replace the placeholder with your dataset construction.
    #
    # Expected interface:
    #   group_eval_loaders[group_name] -> DataLoader yielding (audio, labels)
    #   where audio is (batch, 80000) float32 at 16kHz
    #   and labels is (batch,) int64 class indices
    # -----------------------------------------------------------------------

    try:
        from data.dataset import build_eval_loaders  # type: ignore
        group_eval_loaders = build_eval_loaders(
            groups=groups,
            species_groups_dir=cfg["paths"]["species_groups"],
            processed_dir=cfg["paths"]["data_processed"],
            batch_size=cfg["evaluation"]["linear_probe"]["batch_size"],
            num_workers=cfg["training"]["num_workers"],
        )
    except ImportError:
        logger.error(
            "Cannot import data.dataset.build_eval_loaders. "
            "You need to implement this function that returns:\n"
            "  dict[str, DataLoader] mapping group_name → eval DataLoader\n"
            "See data/dataset.py template for the expected interface."
        )
        sys.exit(1)

    result = run_lmc_experiment(
        base_checkpoint_path=cfg["paths"]["beats_base_checkpoint"],
        finetuned_dir=cfg["paths"]["finetuned"],
        group_names=groups,
        group_eval_loaders=group_eval_loaders,
        n_alphas=n_alphas,
        output_dir=args.output,
        device=args.device,
        only_pair=single_pair,
    )

    # Summary
    logger.info("=" * 60)
    logger.info("LMC SUMMARY")
    logger.info("=" * 60)
    for pair_key, barrier in sorted(
        (
            (f"{r.model_a} ↔ {r.model_b} (eval={r.eval_group})", r.barrier_height)
            for r in result.pairs
        ),
        key=lambda x: -x[1],
    ):
        flag = " ⚠️  HIGH" if barrier > 0.1 else ""
        logger.info("  %s: barrier=%.4f%s", pair_key, barrier, flag)


if __name__ == "__main__":
    main()