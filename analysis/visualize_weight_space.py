"""Weight-space visualizations: model PCA and per-layer task vector analysis.

Produces:
  - figures/weight_pca.pdf — All models projected into 2D PCA
  - figures/layer_heatmap.pdf — Per-layer L2 norms + pairwise cosine similarity
  - figures/magnitude_flow.pdf — Parameter type breakdown per group

Loads task vectors and checkpoints; can run on CPU (no forward passes needed).

Usage:
    python analysis/visualize_weight_space.py \
        --config configs/base.yaml \
        --output figures/
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import re
import sys
from pathlib import Path

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# Publication settings
RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
plt.rcParams.update(RCPARAMS)

GROUP_COLORS = {
    "G1_passerines": "#0072B2",
    "G2_nonpasserine_birds": "#E69F00",
    "G3_raptors_waterbirds": "#009E73",
    "G4_marine_mammals": "#CC79A7",
    "G5_amphibians": "#56B4E9",
    "ALL_birds": "#D55E00",
    "base": "#000000",
    "merged": "#F0E442",
}
GROUP_SHORT = {
    "G1_passerines": "G1 Pass.",
    "G2_nonpasserine_birds": "G2 Non-pass.",
    "G3_raptors_waterbirds": "G3 Rapt/Water",
    "G4_marine_mammals": "G4 Marine",
    "G5_amphibians": "G5 Amphib.",
    "ALL_birds": "Joint",
    "base": "Base",
    "merged": "Merged",
}


# ---------------------------------------------------------------------------
# Flatten state dicts for PCA
# ---------------------------------------------------------------------------

def flatten_state_dict(sd: dict[str, torch.Tensor]) -> np.ndarray:
    """Flatten a state dict into a single 1D vector for PCA.

    Args:
        sd: State dict of tensors.

    Returns:
        1D numpy array of all concatenated parameters.
    """
    parts = [sd[k].flatten().float().numpy() for k in sorted(sd.keys())]
    return np.concatenate(parts)


def flatten_encoder_from_checkpoint(
    ckpt_path: str,
    key: str = "encoder_state_dict",
) -> np.ndarray:
    """Load checkpoint and flatten encoder weights.

    Args:
        ckpt_path: Path to checkpoint .pt file.
        key: Key in checkpoint dict for encoder weights.

    Returns:
        1D numpy array.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt[key]
    return flatten_state_dict(sd)


# ---------------------------------------------------------------------------
# Weight-space PCA
# ---------------------------------------------------------------------------

def plot_weight_pca(
    model_vectors: dict[str, np.ndarray],
    output_path: str,
) -> None:
    """Plot PCA projection of all models into 2D weight space.

    Args:
        model_vectors: Maps model_name → flattened weight vector.
        output_path: Path to save PDF.
    """
    from sklearn.decomposition import PCA

    names = list(model_vectors.keys())
    X = np.stack([model_vectors[n] for n in names])

    # Center on base model
    if "base" in model_vectors:
        center = model_vectors["base"]
    else:
        center = X.mean(axis=0)

    X_centered = X - center

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_centered)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    # Plot each model
    for i, name in enumerate(names):
        color = GROUP_COLORS.get(name, "#666666")
        short = GROUP_SHORT.get(name, name)
        marker = "o"
        size = 60
        zorder = 3

        if name == "base":
            marker = "*"
            size = 150
            zorder = 5
        elif name == "merged":
            marker = "^"
            size = 100
            zorder = 5
        elif name == "ALL_birds":
            marker = "D"
            size = 80
            zorder = 4

        ax.scatter(
            coords[i, 0], coords[i, 1],
            c=color, marker=marker, s=size,
            edgecolors="black", linewidths=0.5,
            label=short, zorder=zorder,
        )
        # Annotate
        ax.annotate(
            short, (coords[i, 0], coords[i, 1]),
            textcoords="offset points", xytext=(8, 6),
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="gray", alpha=0.7),
        )

    # Draw lines from base to each fine-tuned model (task vectors)
    base_idx = names.index("base") if "base" in names else None
    if base_idx is not None:
        for i, name in enumerate(names):
            if name in ("base", "merged"):
                continue
            ax.plot(
                [coords[base_idx, 0], coords[i, 0]],
                [coords[base_idx, 1], coords[i, 1]],
                color=GROUP_COLORS.get(name, "#999999"),
                linewidth=0.6, alpha=0.4, linestyle="--",
            )

    var_explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% variance)")
    ax.set_title("Weight-Space PCA: All Models in One Basin", fontweight="bold")

    ax.legend(
        loc="best", frameon=True, framealpha=0.9,
        edgecolor="gray", fancybox=True,
    )

    ax.axhline(0, color="gray", linewidth=0.3, alpha=0.3)
    ax.axvline(0, color="gray", linewidth=0.3, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved weight PCA to %s", output_path)


# ---------------------------------------------------------------------------
# Per-layer task vector heatmap
# ---------------------------------------------------------------------------

def classify_layer(key: str) -> tuple[int, str]:
    """Classify a BEATs state dict key into (layer_idx, param_type).

    BEATs keys follow patterns like:
        encoder.layers.{N}.self_attn.{k_proj,v_proj,q_proj,out_proj}.{weight,bias}
        encoder.layers.{N}.fc1.weight
        encoder.layers.{N}.fc2.weight
        encoder.layers.{N}.self_attn_layer_norm.{weight,bias}
        encoder.layers.{N}.final_layer_norm.{weight,bias}
        layer_norm.{weight,bias}
        post_extract_proj.{weight,bias}
        patch_embedding.proj.{weight,bias}

    Args:
        key: State dict key.

    Returns:
        (layer_index, param_category). layer_index=-1 for non-layer params.
    """
    # Transformer layers
    layer_match = re.match(r"encoder\.layers\.(\d+)\.", key)
    if layer_match:
        layer_idx = int(layer_match.group(1))
        if "self_attn" in key and "layer_norm" not in key:
            return layer_idx, "attention"
        elif "fc1" in key or "fc2" in key:
            return layer_idx, "ffn"
        elif "layer_norm" in key or "final_layer_norm" in key:
            return layer_idx, "layernorm"
        else:
            return layer_idx, "other"

    # Non-layer params
    if "patch_embedding" in key or "post_extract_proj" in key:
        return -1, "embedding"
    if "layer_norm" in key:
        return -1, "layernorm"
    return -1, "other"


def compute_per_layer_stats(
    task_vectors: dict[str, dict[str, torch.Tensor]],
) -> tuple[dict, dict, int]:
    """Compute per-layer L2 norms and pairwise cosine similarity.

    Args:
        task_vectors: Maps group_name → task vector state dict.

    Returns:
        (norm_dict, cosine_dict, n_layers):
            norm_dict: {group: {layer_idx: l2_norm}}
            cosine_dict: {(group_a, group_b): {layer_idx: cosine_sim}}
            n_layers: Number of transformer layers detected.
    """
    groups = sorted(task_vectors.keys())

    # Discover layers
    all_layer_indices: set[int] = set()
    for tau in task_vectors.values():
        for key in tau:
            layer_idx, _ = classify_layer(key)
            all_layer_indices.add(layer_idx)

    layer_indices = sorted(all_layer_indices)
    n_layers = max(i for i in layer_indices if i >= 0) + 1

    # Per-group per-layer L2 norm
    norm_dict: dict[str, dict[int, float]] = {}
    layer_vectors: dict[str, dict[int, torch.Tensor]] = {}

    for gname, tau in task_vectors.items():
        norm_dict[gname] = {}
        layer_vectors[gname] = {}

        for layer_idx in layer_indices:
            # Collect all params in this layer
            parts = []
            for key, val in tau.items():
                l_idx, _ = classify_layer(key)
                if l_idx == layer_idx:
                    parts.append(val.flatten().float())

            if parts:
                layer_vec = torch.cat(parts)
                norm_dict[gname][layer_idx] = layer_vec.norm(2).item()
                layer_vectors[gname][layer_idx] = layer_vec
            else:
                norm_dict[gname][layer_idx] = 0.0

    # Pairwise cosine per layer
    cosine_dict: dict[tuple[str, str], dict[int, float]] = {}
    for ga, gb in itertools.combinations(groups, 2):
        cosine_dict[(ga, gb)] = {}
        for layer_idx in layer_indices:
            va = layer_vectors.get(ga, {}).get(layer_idx)
            vb = layer_vectors.get(gb, {}).get(layer_idx)
            if va is not None and vb is not None and va.numel() > 0:
                cos = torch.nn.functional.cosine_similarity(
                    va.unsqueeze(0), vb.unsqueeze(0)
                ).item()
                cosine_dict[(ga, gb)][layer_idx] = cos
            else:
                cosine_dict[(ga, gb)][layer_idx] = 0.0

    return norm_dict, cosine_dict, n_layers


def compute_param_type_breakdown(
    task_vectors: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, float]]:
    """Compute L2 norm breakdown by parameter type for each group.

    Args:
        task_vectors: Maps group_name → task vector state dict.

    Returns:
        {group_name: {param_type: l2_norm}}
    """
    breakdown: dict[str, dict[str, float]] = {}

    for gname, tau in task_vectors.items():
        type_parts: dict[str, list[torch.Tensor]] = {}
        for key, val in tau.items():
            _, ptype = classify_layer(key)
            if ptype not in type_parts:
                type_parts[ptype] = []
            type_parts[ptype].append(val.flatten().float())

        breakdown[gname] = {}
        for ptype, parts in type_parts.items():
            vec = torch.cat(parts)
            breakdown[gname][ptype] = vec.norm(2).item()

    return breakdown


def plot_layer_heatmap(
    norm_dict: dict[str, dict[int, float]],
    cosine_dict: dict[tuple[str, str], dict[int, float]],
    n_layers: int,
    output_path: str,
) -> None:
    """Plot dual heatmap: L2 norms (top) and mean pairwise cosine (bottom).

    Args:
        norm_dict: {group: {layer_idx: l2_norm}}.
        cosine_dict: {(group_a, group_b): {layer_idx: cosine_sim}}.
        n_layers: Number of transformer layers.
        output_path: Path to save PDF.
    """
    groups = sorted(norm_dict.keys())
    # Only use actual transformer layers (≥0), plus -1 for embedding
    layer_indices = [-1] + list(range(n_layers))
    layer_labels = ["Embed"] + [str(i) for i in range(n_layers)]

    # Build norm matrix (groups × layers)
    norm_matrix = np.zeros((len(groups), len(layer_indices)))
    for i, g in enumerate(groups):
        for j, l_idx in enumerate(layer_indices):
            norm_matrix[i, j] = norm_dict[g].get(l_idx, 0.0)

    # Build mean cosine matrix (layers only — average over all pairs)
    mean_cosine = np.zeros(len(layer_indices))
    for j, l_idx in enumerate(layer_indices):
        vals = [cosine_dict[pair].get(l_idx, 0.0) for pair in cosine_dict]
        mean_cosine[j] = np.mean(vals) if vals else 0.0

    fig = plt.figure(figsize=(8, 5.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1.2], hspace=0.35)

    # Top: L2 norm heatmap
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(norm_matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    cbar1 = fig.colorbar(im1, ax=ax1, label="L2 Norm", pad=0.02, shrink=0.9)
    cbar1.ax.tick_params(labelsize=7)

    ax1.set_yticks(range(len(groups)))
    ax1.set_yticklabels([GROUP_SHORT.get(g, g) for g in groups])
    ax1.set_xticks(range(len(layer_labels)))
    ax1.set_xticklabels(layer_labels)
    ax1.set_xlabel("Layer")
    ax1.set_title("Per-Layer Task Vector L2 Norm", fontweight="bold")

    # Annotate cells with values
    for i in range(norm_matrix.shape[0]):
        for j in range(norm_matrix.shape[1]):
            val = norm_matrix[i, j]
            if val > 0.01:
                color = "white" if val > norm_matrix.max() * 0.6 else "black"
                ax1.text(j, i, f"{val:.2f}", ha="center", va="center",
                         fontsize=6, color=color)

    # Bottom: mean pairwise cosine per layer
    ax2 = fig.add_subplot(gs[1])
    bars = ax2.bar(range(len(layer_labels)), mean_cosine, color="#4C72B0", edgecolor="black", linewidth=0.3)
    ax2.set_xticks(range(len(layer_labels)))
    ax2.set_xticklabels(layer_labels)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Mean Cosine\nSimilarity")
    ax2.set_title("Mean Pairwise Cosine Similarity per Layer", fontweight="bold", fontsize=9)
    ax2.axhline(0, color="gray", linewidth=0.3)

    # Annotate bar values
    for bar, val in zip(bars, mean_cosine):
        if abs(val) > 0.005:
            ax2.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=6,
            )

    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved layer heatmap to %s", output_path)


def plot_magnitude_flow(
    breakdown: dict[str, dict[str, float]],
    output_path: str,
) -> None:
    """Stacked bar chart of task vector magnitude by parameter type.

    Args:
        breakdown: {group_name: {param_type: l2_norm}}.
        output_path: Path to save PDF.
    """
    groups = sorted(breakdown.keys())
    all_types = sorted(set(t for d in breakdown.values() for t in d))

    fig, ax = plt.subplots(figsize=(6, 3.5))

    x = np.arange(len(groups))
    width = 0.65
    bottom = np.zeros(len(groups))

    type_colors = {
        "attention": "#0072B2",
        "ffn": "#E69F00",
        "layernorm": "#009E73",
        "embedding": "#CC79A7",
        "other": "#999999",
    }

    for ptype in all_types:
        vals = np.array([breakdown[g].get(ptype, 0.0) for g in groups])
        color = type_colors.get(ptype, "#666666")
        ax.bar(x, vals, width, bottom=bottom, label=ptype.capitalize(), color=color, edgecolor="black", linewidth=0.3)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_SHORT.get(g, g) for g in groups], rotation=15, ha="right")
    ax.set_ylabel("L2 Norm")
    ax.set_title("Task Vector Magnitude by Parameter Type", fontweight="bold")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="gray", fontsize=7)

    plt.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved magnitude flow to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Weight-space visualizations: PCA + layer heatmap"
    )
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--output", type=str, default="figures/")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_path = cfg["paths"]["beats_base_checkpoint"]
    tv_dir = Path(cfg["paths"]["task_vectors"])
    finetuned_dir = Path(cfg["paths"]["finetuned"])

    groups = [
        "G1_passerines", "G2_nonpasserine_birds", "G3_raptors_waterbirds",
        "G4_marine_mammals", "G5_amphibians",
    ]

    # ------------------------------------------------------------------
    # Load all task vectors
    # ------------------------------------------------------------------
    logger.info("Loading task vectors...")
    task_vectors: dict[str, dict[str, torch.Tensor]] = {}
    for g in groups:
        tv_path = tv_dir / f"tau_{g}.pt"
        if tv_path.exists():
            task_vectors[g] = torch.load(tv_path, map_location="cpu")
            logger.info("  %s: %d keys", g, len(task_vectors[g]))
        else:
            logger.warning("  Missing: %s", tv_path)

    # ------------------------------------------------------------------
    # Weight-space PCA
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("WEIGHT-SPACE PCA")
    logger.info("=" * 60)

    # Flatten base encoder
    base_ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
    base_model = base_ckpt["model"]

    # Only use keys present in task vectors (encoder keys)
    reference_keys = sorted(list(task_vectors.values())[0].keys()) if task_vectors else []

    def flatten_encoder_keys(sd: dict[str, torch.Tensor]) -> np.ndarray:
        parts = []
        for k in reference_keys:
            if k in sd:
                parts.append(sd[k].flatten().float().numpy())
        return np.concatenate(parts) if parts else np.array([])

    model_vectors: dict[str, np.ndarray] = {}
    model_vectors["base"] = flatten_encoder_keys(base_model)

    # Fine-tuned models
    for g in groups:
        ft_path = finetuned_dir / g / "best_model.pt"
        if ft_path.exists():
            ft_ckpt = torch.load(ft_path, map_location="cpu", weights_only=False)
            model_vectors[g] = flatten_encoder_keys(ft_ckpt["encoder_state_dict"])

    # ALL_birds
    all_birds_path = finetuned_dir / "ALL_birds" / "best_model.pt"
    if all_birds_path.exists():
        ab_ckpt = torch.load(all_birds_path, map_location="cpu", weights_only=False)
        model_vectors["ALL_birds"] = flatten_encoder_keys(ab_ckpt["encoder_state_dict"])

    # Merged model (task arithmetic λ=1.0)
    merged_sd: dict[str, torch.Tensor] = {}
    for k in reference_keys:
        if k in base_model:
            merged_sd[k] = base_model[k].float().clone()
            for tv in task_vectors.values():
                if k in tv:
                    merged_sd[k] = merged_sd[k] + tv[k].float()
    model_vectors["merged"] = flatten_encoder_keys(merged_sd)

    if len(model_vectors) >= 3:
        plot_weight_pca(model_vectors, str(output_dir / "weight_pca.pdf"))
    else:
        logger.warning("Not enough models for PCA (need ≥3, got %d)", len(model_vectors))

    del model_vectors, base_model, merged_sd
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ------------------------------------------------------------------
    # Per-layer heatmap
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PER-LAYER TASK VECTOR HEATMAP")
    logger.info("=" * 60)

    norm_dict, cosine_dict, n_layers = compute_per_layer_stats(task_vectors)
    logger.info("Detected %d transformer layers", n_layers)

    plot_layer_heatmap(
        norm_dict, cosine_dict, n_layers,
        str(output_dir / "layer_heatmap.pdf"),
    )

    # ------------------------------------------------------------------
    # Magnitude flow (parameter type breakdown)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("MAGNITUDE FLOW BY PARAMETER TYPE")
    logger.info("=" * 60)

    breakdown = compute_param_type_breakdown(task_vectors)
    for g, types in breakdown.items():
        logger.info("  %s: %s", g, {t: f"{v:.2f}" for t, v in sorted(types.items())})

    plot_magnitude_flow(breakdown, str(output_dir / "magnitude_flow.pdf"))

    logger.info("Weight-space visualizations complete.")


if __name__ == "__main__":
    main()