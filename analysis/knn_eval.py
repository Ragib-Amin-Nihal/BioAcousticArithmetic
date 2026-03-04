#!/usr/bin/env python3
"""KNN evaluation of merged encoders — alternative to linear probing.

Complements the linear probe evaluation to diagnose whether the 9.3%
composition gap is a property of the encoder or the evaluation protocol.

If KNN accuracy on merged encoders is closer to joint training than
linear probe accuracy, the gap is partly a probe-training artifact
(the unified linear head struggles with the combined label space).
If the gap persists under KNN, the encoder itself is the bottleneck.

KNN evaluation:
    1. Extract features with frozen encoder (same as linear_probe.py)
    2. L2-normalize features
    3. Classify test samples by majority vote among K nearest training neighbors
    No learned parameters — pure geometric evaluation of feature space quality.

Usage:
    # Evaluate a single merged encoder
    python analysis/knn_eval.py --config configs/base.yaml \
        --encoder results/finetuned/ALL_birds/best_model.pt \
        --output results/knn/

    # Compare linear probe vs KNN across composition methods
    python analysis/knn_eval.py --config configs/base.yaml \
        --compare-composition results/composition/composition_results.json \
        --output results/knn/

    # Just run KNN on pre-extracted features (no GPU needed)
    python analysis/knn_eval.py --features-dir results/features/ \
        --output results/knn/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

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
class KNNResult:
    """Result of a KNN evaluation."""

    accuracy: float
    macro_f1: float
    k: int
    accuracy_ci_low: float = 0.0
    accuracy_ci_high: float = 0.0
    per_class_accuracy: dict[int, float] = field(default_factory=dict)
    eval_time_s: float = 0.0
    n_train: int = 0
    n_test: int = 0


@dataclass
class ProtocolComparisonResult:
    """Side-by-side comparison of linear probe vs KNN."""

    encoder_name: str
    num_classes: int
    linear_probe_accuracy: float
    linear_probe_ci: tuple[float, float]
    knn_accuracy: float
    knn_ci: tuple[float, float]
    gap_linear: float  # gap to joint under linear probe
    gap_knn: float  # gap to joint under KNN
    k: int = 10


# ---------------------------------------------------------------------------
# Core KNN evaluation
# ---------------------------------------------------------------------------

def knn_classify(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    k: int = 10,
    normalize: bool = True,
) -> np.ndarray:
    """Classify test samples via K-nearest-neighbor majority vote.

    Uses brute-force L2 distance. For bioacoustic feature sizes
    (N~50k, D=768), this is fast enough without approximate NN.

    Args:
        train_features: (N_train, D) feature matrix.
        train_labels: (N_train,) integer labels.
        test_features: (N_test, D) feature matrix.
        k: Number of neighbors.
        normalize: L2-normalize features before distance computation.

    Returns:
        (N_test,) predicted labels.
    """
    if normalize:
        train_norms = np.linalg.norm(train_features, axis=1, keepdims=True)
        train_norms = np.maximum(train_norms, 1e-8)
        train_features = train_features / train_norms

        test_norms = np.linalg.norm(test_features, axis=1, keepdims=True)
        test_norms = np.maximum(test_norms, 1e-8)
        test_features = test_features / test_norms

    # Compute pairwise distances in chunks to manage memory
    # For N_test=40k, N_train=140k, D=768: chunked to avoid 40k×140k float32 matrix
    chunk_size = 512
    n_test = test_features.shape[0]
    predictions = np.empty(n_test, dtype=np.int64)

    for start in range(0, n_test, chunk_size):
        end = min(start + chunk_size, n_test)
        chunk = test_features[start:end]  # (chunk, D)

        # Cosine similarity via dot product (features are normalized)
        if normalize:
            sims = chunk @ train_features.T  # (chunk, N_train)
            # Top-k by similarity (highest = nearest)
            topk_idx = np.argpartition(-sims, k, axis=1)[:, :k]
        else:
            # L2 distance
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
            chunk_sq = np.sum(chunk ** 2, axis=1, keepdims=True)
            train_sq = np.sum(train_features ** 2, axis=1, keepdims=True).T
            dists = chunk_sq + train_sq - 2 * (chunk @ train_features.T)
            topk_idx = np.argpartition(dists, k, axis=1)[:, :k]

        # Majority vote
        for i in range(end - start):
            neighbor_labels = train_labels[topk_idx[i]]
            counts = Counter(neighbor_labels.tolist())
            predictions[start + i] = counts.most_common(1)[0][0]

    return predictions


def evaluate_knn(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    num_classes: int,
    k: int = 10,
    normalize: bool = True,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> KNNResult:
    """Full KNN evaluation with metrics and bootstrap CIs.

    Args:
        train_features: (N_train, D) training features.
        train_labels: (N_train,) training labels.
        test_features: (N_test, D) test features.
        test_labels: (N_test,) test labels.
        num_classes: Number of classes.
        k: Number of neighbors.
        normalize: L2-normalize before distance computation.
        n_bootstrap: Bootstrap resamples for CI.
        seed: Random seed.

    Returns:
        KNNResult with accuracy, F1, CIs.
    """
    t0 = time.time()

    predictions = knn_classify(
        train_features, train_labels, test_features, k=k, normalize=normalize,
    )

    correct = (predictions == test_labels)
    accuracy = correct.mean()

    # Per-class accuracy
    per_class: dict[int, float] = {}
    for c in range(num_classes):
        mask = test_labels == c
        if mask.sum() > 0:
            per_class[c] = float(correct[mask].mean())

    # Macro F1
    f1_scores = []
    for c in range(num_classes):
        tp = ((predictions == c) & (test_labels == c)).sum()
        fp = ((predictions == c) & (test_labels != c)).sum()
        fn = ((predictions != c) & (test_labels == c)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

    # Bootstrap CI
    rng = np.random.default_rng(seed)
    boot_accs = np.array([
        rng.choice(correct, size=len(correct), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    ci_low = float(np.percentile(boot_accs, 2.5))
    ci_high = float(np.percentile(boot_accs, 97.5))

    elapsed = time.time() - t0

    return KNNResult(
        accuracy=float(accuracy),
        macro_f1=macro_f1,
        k=k,
        accuracy_ci_low=ci_low,
        accuracy_ci_high=ci_high,
        per_class_accuracy=per_class,
        eval_time_s=elapsed,
        n_train=len(train_labels),
        n_test=len(test_labels),
    )


# ---------------------------------------------------------------------------
# Feature extraction (reuses linear_probe infrastructure)
# ---------------------------------------------------------------------------

def extract_features_for_encoder(
    encoder_state_dict: dict[str, torch.Tensor],
    base_checkpoint_path: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract frozen encoder features for KNN evaluation.

    Reuses the same feature extraction as linear_probe.py to ensure
    a fair comparison between evaluation protocols.

    Args:
        encoder_state_dict: Encoder weights (merged or fine-tuned).
        base_checkpoint_path: BEATs pretrained checkpoint for model init.
        train_loader: Training data loader.
        test_loader: Test data loader.
        num_classes: Number of classes (for model init only).
        device: Torch device.

    Returns:
        (train_features, train_labels, test_features, test_labels) as numpy arrays.
    """
    from evaluation.linear_probe import LinearProbeEvaluator, ProbeConfig

    evaluator = LinearProbeEvaluator(base_checkpoint_path, ProbeConfig(), device)
    model = evaluator._build_model(encoder_state_dict, num_classes)

    logger.info("  Extracting train features...")
    train_feats, train_labs = evaluator._extract_features(model, train_loader)
    logger.info("  Extracting test features...")
    test_feats, test_labs = evaluator._extract_features(model, test_loader)

    del model
    torch.cuda.empty_cache()

    return (
        train_feats.numpy(),
        train_labs.numpy(),
        test_feats.numpy(),
        test_labs.numpy(),
    )


def save_features(
    features: np.ndarray,
    labels: np.ndarray,
    path: str,
) -> None:
    """Save extracted features to disk for reuse.

    Args:
        features: (N, D) feature matrix.
        labels: (N,) label array.
        path: Output .npz file path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=features, labels=labels)
    logger.info("  Saved features: %s (%s)", path, features.shape)


def load_features(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load previously extracted features.

    Args:
        path: Path to .npz file.

    Returns:
        (features, labels) numpy arrays.
    """
    data = np.load(path)
    return data["features"], data["labels"]


# ---------------------------------------------------------------------------
# K sweep
# ---------------------------------------------------------------------------

def sweep_k(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    num_classes: int,
    k_values: list[int] = [1, 3, 5, 10, 20, 50],
) -> list[KNNResult]:
    """Evaluate KNN across multiple K values.

    Args:
        train_features: Training features.
        train_labels: Training labels.
        test_features: Test features.
        test_labels: Test labels.
        num_classes: Number of classes.
        k_values: K values to evaluate.

    Returns:
        List of KNNResult, one per K.
    """
    results = []
    for k in k_values:
        logger.info("  K=%d ...", k)
        result = evaluate_knn(
            train_features, train_labels,
            test_features, test_labels,
            num_classes, k=k,
        )
        results.append(result)
        logger.info("    acc=%.4f [%.4f, %.4f]", result.accuracy, result.accuracy_ci_low, result.accuracy_ci_high)
    return results


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def run_knn_evaluation(
    config_path: str,
    output_dir: str,
    device: str = "cuda",
    k_values: list[int] = [1, 5, 10, 20],
    save_feats: bool = True,
) -> dict[str, Any]:
    """Run KNN evaluation on all composition methods and compare with linear probe.

    Evaluates the same encoders used in Experiment 2 (composition_eval.py)
    under KNN instead of linear probing, producing a side-by-side comparison.

    Requires: composition_results.json from a prior composition_eval.py run.

    Args:
        config_path: Path to base.yaml.
        output_dir: Directory for output JSON and cached features.
        device: Torch device for feature extraction.
        k_values: K values to sweep.
        save_feats: Cache extracted features to disk.

    Returns:
        Results dict with per-encoder KNN metrics.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["paths"]["project_root"]).resolve()
    finetuned_dir = root / cfg["paths"]["finetuned"]
    tv_dir = root / cfg["paths"]["task_vectors"]
    species_groups_dir = root / cfg["paths"]["species_groups"]
    base_path = str(root / cfg["paths"]["beats_base_checkpoint"])
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    feat_dir = out_path / "features"
    feat_dir.mkdir(exist_ok=True)

    groups = [
        "G1_passerines", "G2_nonpasserine_birds", "G3_raptors_waterbirds",
        "G4_marine_mammals", "G5_amphibians",
    ]

    # Load data
    from data.dataset import build_group_loaders, build_all_group_loader
    group_loaders = build_group_loaders(
        groups=groups,
        species_groups_dir=str(species_groups_dir),
        processed_dir=cfg["paths"]["data_processed"],
        batch_size=cfg["evaluation"]["linear_probe"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
    )
    all_train_loader, all_test_loader, total_classes = build_all_group_loader(
        groups=groups,
        species_groups_dir=str(species_groups_dir),
        processed_dir=cfg["paths"]["data_processed"],
        batch_size=cfg["evaluation"]["linear_probe"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
    )

    results: dict[str, Any] = {
        "k_values": k_values,
        "encoders": {},
    }

    # -----------------------------------------------------------------------
    # Key encoders to evaluate
    # -----------------------------------------------------------------------
    encoders_to_eval: dict[str, dict[str, torch.Tensor]] = {}

    # 1. Joint baseline (ALL_birds)
    all_ckpt_path = finetuned_dir / "ALL_birds" / "best_model.pt"
    if all_ckpt_path.exists():
        ckpt = torch.load(str(all_ckpt_path), map_location="cpu", weights_only=False)
        encoders_to_eval["joint_baseline"] = ckpt["encoder_state_dict"]
        logger.info("Loaded joint baseline encoder")

    # 2. Base pretrained (no fine-tuning)
    base_ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
    encoders_to_eval["pretrained_base"] = base_ckpt["model"]

    # 3. Best composition methods from Experiment 2
    # Simple average
    task_vectors = {}
    for g in groups:
        tv_path = tv_dir / f"tau_{g}.pt"
        if tv_path.exists():
            task_vectors[g] = torch.load(str(tv_path), map_location="cpu")

    if len(task_vectors) == len(groups):
        base_sd = base_ckpt["model"]

        # Simple average: θ_base + (1/n) Σ τ_i
        merged_avg = {}
        n = len(groups)
        for key in base_sd:
            merged_avg[key] = base_sd[key].float().clone()
            for g in groups:
                if key in task_vectors[g]:
                    merged_avg[key] += (1.0 / n) * task_vectors[g][key].float()
        encoders_to_eval["simple_average"] = merged_avg

        # Task arithmetic λ=1.0: θ_base + (1/n) Σ τ_i (same as above)
        # Task arithmetic λ=0.5
        merged_05 = {}
        for key in base_sd:
            merged_05[key] = base_sd[key].float().clone()
            for g in groups:
                if key in task_vectors[g]:
                    merged_05[key] += (0.5 / n) * task_vectors[g][key].float()
        encoders_to_eval["task_arith_lambda_0.5"] = merged_05

    # 4. Per-group fine-tuned encoders (individual models)
    for g in groups:
        ckpt_path = finetuned_dir / g / "best_model.pt"
        if ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            encoders_to_eval[f"individual_{g}"] = ckpt["encoder_state_dict"]

    # -----------------------------------------------------------------------
    # Evaluate each encoder
    # -----------------------------------------------------------------------
    for enc_name, encoder_sd in encoders_to_eval.items():
        logger.info("=" * 60)
        logger.info("Evaluating encoder: %s", enc_name)

        # Check for cached features
        feat_path = feat_dir / f"{enc_name}_all.npz"
        if feat_path.exists():
            logger.info("  Loading cached features from %s", feat_path)
            all_train_feats, all_train_labs = load_features(
                str(feat_dir / f"{enc_name}_all_train.npz")
            )
            all_test_feats, all_test_labs = load_features(
                str(feat_dir / f"{enc_name}_all_test.npz")
            )
        else:
            # Extract features (needs GPU)
            all_train_feats, all_train_labs, all_test_feats, all_test_labs = \
                extract_features_for_encoder(
                    encoder_sd, base_path,
                    all_train_loader, all_test_loader,
                    total_classes, device=device,
                )
            if save_feats:
                save_features(all_train_feats, all_train_labs,
                              str(feat_dir / f"{enc_name}_all_train.npz"))
                save_features(all_test_feats, all_test_labs,
                              str(feat_dir / f"{enc_name}_all_test.npz"))
                # Touch the marker file
                np.savez(str(feat_path), dummy=np.array([0]))

        # KNN sweep on all-group evaluation
        logger.info("  All-group KNN evaluation (%d classes):", total_classes)
        knn_results = sweep_k(
            all_train_feats, all_train_labs,
            all_test_feats, all_test_labs,
            total_classes, k_values=k_values,
        )

        results["encoders"][enc_name] = {
            "all_group_knn": [asdict(r) for r in knn_results],
            "n_train": len(all_train_labs),
            "n_test": len(all_test_labs),
            "n_classes": int(total_classes),
        }

        # Also evaluate per-group if this is a composed encoder
        if enc_name in ("simple_average", "task_arith_lambda_0.5", "joint_baseline"):
            per_group_knn: dict[str, list[dict]] = {}
            for g, (train_ld, test_ld, n_cls) in group_loaders.items():
                gfeat_train_path = feat_dir / f"{enc_name}_{g}_train.npz"
                gfeat_test_path = feat_dir / f"{enc_name}_{g}_test.npz"

                if gfeat_train_path.exists() and gfeat_test_path.exists():
                    g_train_f, g_train_l = load_features(str(gfeat_train_path))
                    g_test_f, g_test_l = load_features(str(gfeat_test_path))
                else:
                    g_train_f, g_train_l, g_test_f, g_test_l = \
                        extract_features_for_encoder(
                            encoder_sd, base_path,
                            train_ld, test_ld, n_cls, device=device,
                        )
                    if save_feats:
                        save_features(g_train_f, g_train_l, str(gfeat_train_path))
                        save_features(g_test_f, g_test_l, str(gfeat_test_path))

                best_k = k_values[len(k_values) // 2]  # Use middle K for per-group
                knn_r = evaluate_knn(
                    g_train_f, g_train_l, g_test_f, g_test_l,
                    n_cls, k=best_k,
                )
                per_group_knn[g] = asdict(knn_r)
                logger.info("    %s (k=%d): acc=%.4f", g, best_k, knn_r.accuracy)

            results["encoders"][enc_name]["per_group_knn"] = per_group_knn

    # -----------------------------------------------------------------------
    # Summary comparison
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("KNN EVALUATION SUMMARY")
    logger.info("-" * 60)

    # Find best K per encoder
    for enc_name, enc_data in results["encoders"].items():
        knn_list = enc_data["all_group_knn"]
        best = max(knn_list, key=lambda r: r["accuracy"])
        logger.info(
            "  %-30s best_k=%2d  acc=%.4f [%.4f, %.4f]",
            enc_name, best["k"], best["accuracy"],
            best["accuracy_ci_low"], best["accuracy_ci_high"],
        )

    # Gap comparison if joint baseline exists
    joint_data = results["encoders"].get("joint_baseline")
    avg_data = results["encoders"].get("simple_average")
    if joint_data and avg_data:
        joint_best = max(joint_data["all_group_knn"], key=lambda r: r["accuracy"])
        avg_best = max(avg_data["all_group_knn"], key=lambda r: r["accuracy"])
        knn_gap = joint_best["accuracy"] - avg_best["accuracy"]
        logger.info("-" * 60)
        logger.info(
            "KNN gap (joint - merged): %.4f  (compare with linear probe gap)",
            knn_gap,
        )
        results["knn_composition_gap"] = knn_gap

    logger.info("=" * 60)

    # Save
    out_file = out_path / "knn_evaluation.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_file)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="KNN evaluation of merged encoders",
    )
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--output", type=str, default="results/knn/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--k-values", type=int, nargs="*", default=[1, 5, 10, 20],
        help="K values to sweep",
    )
    parser.add_argument(
        "--no-save-features", action="store_true",
        help="Don't cache extracted features to disk",
    )
    args = parser.parse_args()

    run_knn_evaluation(
        config_path=args.config,
        output_dir=args.output,
        device=args.device,
        k_values=args.k_values,
        save_feats=not args.no_save_features,
    )


if __name__ == "__main__":
    main()