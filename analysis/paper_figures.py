#!/usr/bin/env python3
"""Paper figures: new analyses and visualizations not covered by existing scripts.

Existing scripts handle:
  - visualize_results.py:       LMC curves, composition bars, cosine matrix,
                                sparsity CDF, taxonomy-vs-cosine scatter
  - visualize_weight_space.py:  Weight PCA, per-layer heatmap, magnitude flow
  - visualize_gpu.py:           UMAP triptych, G4 zoom, 2D loss landscape

This script adds:
  Fig 4a: Spectral distribution distance vs task vector cosine scatter
  Fig 4b: Asymmetric gap bar chart (per-group gap with bootstrap CIs)
  Fig 5:  Domain negation Pareto curves (real vs random control)
  Fig C:  Group-level confusion matrix (5x5) after merging
  Fig D:  Regional task vector cosine similarity matrix
  Fig E:  Calibration reliability diagrams (joint vs merged)
  Fig G:  Data efficiency curve (if results exist)
  Fig H:  Forgetting dynamics curve (if results exist)
  Fig I:  Per-layer pairwise cosine heatmap (full pair x layer grid)
  Supp:   Per-group gap vs training set size scatter

All figures use shared figure_style module for consistent sizing.
No titles on any figure --- paper \\caption{} handles that.

Usage:
    # All figures that can be produced from existing results
    python analysis/paper_figures.py --config configs/base.yaml --output figures/

    # Only specific figures
    python analysis/paper_figures.py --config configs/base.yaml --output figures/ \\
        --only spectral_vs_cosine asymmetric_gap confusion regional_cosine

    # With GPU (needed for calibration analysis)
    python analysis/paper_figures.py --config configs/base.yaml --output figures/ --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Style --- shared module preferred, inline fallback
# ---------------------------------------------------------------------------

try:
    from figure_style import (
        apply_style, GROUP_COLORS, GROUP_SHORT, TAXA_BIRD, TAXA_OTHER,
        REGIONAL_COLORS, REGIONAL_SHORT,
        COLUMN_WIDTH, FULL_WIDTH,
    )
    apply_style()
except ImportError:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    GROUP_COLORS = {
        "G1_passerines": "#0072B2",
        "G2_nonpasserine_birds": "#E69F00",
        "G3_raptors_waterbirds": "#009E73",
        "G4_marine_mammals": "#CC79A7",
        "G5_amphibians": "#56B4E9",
    }
    GROUP_SHORT = {
        "G1_passerines": "G1 Pass.",
        "G2_nonpasserine_birds": "G2 Non-pass.",
        "G3_raptors_waterbirds": "G3 Rapt.",
        "G4_marine_mammals": "G4 Marine",
        "G5_amphibians": "G5 Amphib.",
    }
    TAXA_BIRD = {"G1_passerines", "G2_nonpasserine_birds", "G3_raptors_waterbirds"}
    TAXA_OTHER = {"G4_marine_mammals", "G5_amphibians"}
    REGIONAL_COLORS = {}
    REGIONAL_SHORT = {}
    COLUMN_WIDTH = 3.35
    FULL_WIDTH = 7.0

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

try:
    import scipy.stats
except ImportError:
    scipy = None  # type: ignore[assignment]

ALL_GROUPS = [
    "G1_passerines", "G2_nonpasserine_birds", "G3_raptors_waterbirds",
    "G4_marine_mammals", "G5_amphibians",
]

# Approximate training set sizes (fallback if manifests unavailable)
_FALLBACK_TRAIN_SIZES = {
    "G1_passerines": 80761,
    "G2_nonpasserine_birds": 37574,
    "G3_raptors_waterbirds": 20709,
    "G4_marine_mammals": 1402,
    "G5_amphibians": 11898,
}


def load_train_sizes(species_groups_dir: str) -> dict[str, int]:
    """Load training set sizes from group manifests, falling back to estimates.

    Args:
        species_groups_dir: Path to directory containing group JSON files.

    Returns:
        Dict mapping group name to training sample count.
    """
    sizes = dict(_FALLBACK_TRAIN_SIZES)
    for group in ALL_GROUPS:
        manifest_path = Path(species_groups_dir) / f"{group}.json"
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    meta = json.load(f)["metadata"]
                sizes[group] = meta.get("n_train", sizes.get(group, 0))
            except (KeyError, json.JSONDecodeError):
                pass
    return sizes


# ============================================================================
# Fig 4a --- Spectral Distribution Distance vs. Task Vector Cosine Similarity
# ============================================================================

def plot_spectral_vs_cosine(
    pairs: list[dict],
    output_path: str | Path,
    distance_metric: str = "jsd",
) -> None:
    """Scatter plot: spectral distribution distance vs task vector cosine.

    Each point is one pairwise group comparison (n=10).
    Color distinguishes pair types; triangle marks hydrophone pairs.

    Args:
        pairs: List of pair dicts from spectral_distance_results.json["pairs"].
        output_path: Output PDF path.
        distance_metric: x-axis metric key ("jsd", "l2", "l2_norm").
    """
    COLORS = {
        "bird_bird":   "#0072B2",
        "bird_other":  "#E69F00",
        "other_other": "#CC79A7",
    }
    MARKERS = {"hydrophone": "^", "standard": "o"}

    metric_labels = {
        "jsd":     "Jensen-Shannon Divergence (spectral distribution)",
        "l2":      "L2 Distance (mean log-mel energy)",
        "l2_norm": "L2 Distance (mean-centered, shape only)",
    }
    x_label = metric_labels.get(distance_metric, distance_metric)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 1.5, 3.5))

    xs, ys = [], []
    for p in pairs:
        x_val = p[f"distance_{distance_metric}"]
        y_val = p["cosine_similarity"]
        xs.append(x_val)
        ys.append(y_val)

        color = COLORS[p["pair_type"]]
        marker = MARKERS["hydrophone"] if p["has_hydrophone"] else MARKERS["standard"]

        ax.scatter(
            x_val, y_val,
            c=color, marker=marker,
            s=90, edgecolors="black", linewidths=0.5, zorder=3,
        )
        ax.annotate(
            p["short_label"],
            (x_val, y_val),
            fontsize=9,
            xytext=(5, 4),
            textcoords="offset points",
        )

    # OLS regression line
    xs_arr, ys_arr = np.array(xs), np.array(ys)
    if len(xs_arr) > 2:
        m, b = np.polyfit(xs_arr, ys_arr, 1)
        x_line = np.linspace(xs_arr.min(), xs_arr.max(), 100)
        ax.plot(x_line, m * x_line + b, color="gray", linewidth=1.2,
                linestyle="--", alpha=0.7, zorder=2)

    # Spearman annotation
    # if scipy is not None:
    #     rho, p_val = scipy.stats.spearmanr(xs_arr, ys_arr)
    #     ax.text(
    #         0.04, 0.96,
    #         f"Spearman $\\rho$ = {rho:.3f}\n($p$ = {p_val:.4f})",
    #         transform=ax.transAxes, fontsize=10,
    #         verticalalignment="top",
    #         bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
    #                   edgecolor="gray", linewidth=0.5),
    #     )

    # Legend
    legend_elements = [
        Patch(facecolor=COLORS["bird_bird"], edgecolor="black",
              linewidth=0.5, label="Bird-Bird"),
        Patch(facecolor=COLORS["bird_other"], edgecolor="black",
              linewidth=0.5, label="Bird-Other"),
        Patch(facecolor=COLORS["other_other"], edgecolor="black",
              linewidth=0.5, label="Other-Other"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=8, linewidth=0,
               label="Hydrophone recording (G4)"),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=8.5,
              framealpha=0.9, edgecolor="gray")

    ax.set_xlabel(x_label)
    ax.set_ylabel("Task Vector Cosine Similarity")
    ax.grid(True, linewidth=0.3, alpha=0.4)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), format="pdf")
    plt.close(fig)
    logger.info("Saved spectral vs cosine to %s", output_path)


# ============================================================================
# Fig 4b --- Asymmetric Composition Gap
# ============================================================================

def plot_asymmetric_gap(
    gap_data: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Per-group composition gap bar chart with bootstrap CIs.

    Positive gap = joint training better. Negative = merging better.
    The visual highlight: bird groups have positive gaps while marine
    mammals and amphibians have negative gaps.

    Args:
        gap_data: Loaded from gap_analysis.json.
        output_path: Output PDF path.
    """
    per_group = gap_data.get("per_group", gap_data.get("per_group_gaps", {}))
    if not per_group:
        logger.warning("No per-group gap data found")
        return

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 1.5, 3.5))

    group_order = [g for g in ALL_GROUPS if g in per_group]

    gaps: list[float] = []
    ci_low: list[float] = []
    ci_high: list[float] = []
    colors: list[str] = []

    for g in group_order:
        gdata = per_group[g]
        gap = gdata.get("gap_pp", gdata.get("gap", 0))
        if isinstance(gap, float) and abs(gap) < 1:
            gap *= 100  # convert fraction to pp
        gaps.append(gap)

        ci = gdata.get("ci_95", gdata.get("bootstrap_ci",
                 [gdata.get("ci_low", gap), gdata.get("ci_high", gap)]))
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            lo, hi = ci[0], ci[1]
            if abs(lo) < 1:
                lo *= 100
                hi *= 100
            ci_low.append(gap - lo)
            ci_high.append(hi - gap)
        else:
            ci_low.append(0)
            ci_high.append(0)

        colors.append(GROUP_COLORS.get(g, "#999999"))

    x = np.arange(len(group_order))
    bars = ax.bar(x, gaps, color=colors, edgecolor="black", linewidth=0.5,
                  width=0.65, zorder=3)

    # Error bars
    if any(c > 0 for c in ci_high):
        ax.errorbar(x, gaps, yerr=[ci_low, ci_high],
                    fmt="none", ecolor="black", elinewidth=0.8, capsize=4, zorder=4)

    # Annotations on each bar
    for i, (bar, gap_val) in enumerate(zip(bars, gaps)):
        sign = "+" if gap_val > 0 else ""
        y_pos = gap_val + (ci_high[i] if gap_val > 0 else -ci_low[i]) + 0.3
        ax.text(
            i, y_pos, f"{sign}{gap_val:.1f}pp",
            ha="center", va="bottom" if gap_val > 0 else "top",
            fontsize=9, fontweight="bold",
        )

    # Zero line + context labels
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    y_lim = ax.get_ylim()
    ax.text(
        len(group_order) - 0.5, max(y_lim) * 0.85,
        "Joint training better",
        fontsize=8.5, fontstyle="italic", ha="right", color="#666666",
    )
    ax.text(
        len(group_order) - 0.5, min(y_lim) * 0.85,
        "Merging better",
        fontsize=8.5, fontstyle="italic", ha="right", color="#666666",
    )

    # Overall mean gap annotation
    overall = gap_data.get("overall", gap_data.get("all_group",
                  gap_data.get("all_group_gap", {})))
    mean_gap = overall.get("gap_pp", overall.get("gap", None))
    if mean_gap is not None:
        if isinstance(mean_gap, float) and abs(mean_gap) < 1:
            mean_gap *= 100
        ax.axhline(mean_gap, color="#D55E00", linewidth=1.2, linestyle="--",
                    zorder=2, label=f"All-group mean: {mean_gap:.1f}pp")
        ax.legend(fontsize=9, loc="best")

    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_SHORT.get(g, g) for g in group_order])
    ax.set_ylabel("Accuracy Gap (Joint - Merged, pp)")

    plt.tight_layout()
    fig.savefig(str(output_path), format="pdf")
    plt.close(fig)
    logger.info("Saved asymmetric gap to %s", output_path)


# ============================================================================
# Fig 5 --- Domain Negation Curves
# ============================================================================

def plot_domain_negation(
    negation_data: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Domain negation: accuracy vs subtraction strength beta.

    Two panels: (left) mixed-domain source, (right) soundscape source.
    Solid = focal negation, dashed = random-vector control.

    Args:
        negation_data: Loaded from negation_results.json.
        output_path: Output PDF path.
    """
    # Handle both sweep-based and trial-based data formats
    sweeps = negation_data.get("sweeps", negation_data.get("results", []))

    # Trial-based format (from existing code) -> reshape into sweeps
    if not sweeps and "trials" in negation_data:
        trials = negation_data["trials"]
        sweep_map: dict[tuple[str, str], dict] = {}
        for t in trials:
            key = (t.get("source_model", "unknown"), t.get("vector_type", "focal"))
            if key not in sweep_map:
                sweep_map[key] = {
                    "source_model": key[0],
                    "vector_type": key[1],
                    "betas": [],
                    "focal_accuracies": [],
                    "soundscape_accuracies": [],
                }
            sweep_map[key]["betas"].append(t["beta"])
            sweep_map[key]["focal_accuracies"].append(t.get("focal_accuracy", 0))
            sweep_map[key]["soundscape_accuracies"].append(t.get("soundscape_accuracy", 0))
        sweeps = list(sweep_map.values())

    if not sweeps:
        logger.warning("No sweep data in negation results")
        return

    # Group sweeps by source model
    source_groups: dict[str, list[dict]] = {}
    for sweep in sweeps:
        source = sweep.get("source_model", sweep.get("source", "unknown"))
        source_groups.setdefault(source, []).append(sweep)

    n_sources = len(source_groups)
    if n_sources == 0:
        logger.warning("No source models found in negation data")
        return

    fig, axes = plt.subplots(1, n_sources, figsize=(FULL_WIDTH, 3.5),
                              squeeze=False, sharey=True)

    source_labels = {
        "mixed": "Source: Mixed Model",
        "mixed_domain": "Source: Mixed Model",
        "soundscape": "Source: Soundscape Model",
        "soundscape_only": "Source: Soundscape Model",
    }

    color_focal = "#0072B2"
    color_soundscape = "#E69F00"

    for s_idx, (source_name, source_sweeps) in enumerate(sorted(source_groups.items())):
        ax = axes[0][s_idx]

        for sweep in source_sweeps:
            betas = sweep.get("betas", sweep.get("beta_values", []))
            vector_type = sweep.get("vector_type", sweep.get("negation_type", "focal"))
            is_control = "random" in vector_type.lower() or "control" in vector_type.lower()

            focal_accs = sweep.get("focal_accuracies", sweep.get("focal_acc", []))
            sound_accs = sweep.get("soundscape_accuracies", sweep.get("soundscape_acc", []))

            linestyle = "--" if is_control else "-"
            alpha = 0.7 if is_control else 1.0
            label_suffix = " (random ctrl)" if is_control else " (focal neg.)"
            marker = "s" if is_control else "o"

            if focal_accs:
                focal_pct = [a * 100 if a <= 1.0 else a for a in focal_accs]
                ax.plot(betas, focal_pct, linestyle=linestyle, color=color_focal,
                        linewidth=2.5, alpha=alpha, marker=marker, markersize=5,
                        label=f"Focal acc{label_suffix}")

            if sound_accs:
                sound_pct = [a * 100 if a <= 1.0 else a for a in sound_accs]
                ax.plot(betas, sound_pct, linestyle=linestyle, color=color_soundscape,
                        linewidth=2.5, alpha=alpha, marker=marker, markersize=5,
                        label=f"Soundscape acc{label_suffix}")

        # Source label as in-plot text
        label_text = source_labels.get(source_name, f"Source: {source_name}")
        ax.text(
            0.04, 0.66, label_text,
            transform=ax.transAxes, fontsize=9, fontweight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", linewidth=0.5, alpha=0.9),
        )

        ax.set_xlabel("Negation Strength ($\\beta$)")
        if s_idx == 0:
            ax.set_ylabel("Accuracy (%)")
        ax.legend(fontsize=7.5, loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(str(output_path), format="pdf")
    plt.close(fig)
    logger.info("Saved domain negation curves to %s", output_path)


# ============================================================================
# Fig C --- Group-Level Confusion Matrix
# ============================================================================

def plot_group_confusion(
    predictions_dir: str,
    species_groups_dir: str,
    output_path: str | Path,
) -> None:
    """5x5 confusion matrix showing where merged-model errors go.

    Uses cached predictions from bootstrap_gap.py.  For each test sample,
    determines which group the predicted label belongs to (based on label
    offset ranges in the unified label space).

    Args:
        predictions_dir: Dir containing *_predictions.npy and *_labels.npy.
        species_groups_dir: Dir with group manifests (for label offsets).
        output_path: Path to save PDF.
    """
    groups = ALL_GROUPS
    group_ranges: dict[str, tuple[int, int]] = {}
    offset = 0
    for group in groups:
        manifest_path = Path(species_groups_dir) / f"{group}.json"
        if not manifest_path.exists():
            logger.warning("Manifest not found: %s", manifest_path)
            continue
        with open(manifest_path) as f:
            n_cls = json.load(f)["metadata"]["n_classes"]
        group_ranges[group] = (offset, offset + n_cls)
        offset += n_cls

    def _label_to_group(label: int) -> str:
        for g, (lo, hi) in group_ranges.items():
            if lo <= label < hi:
                return g
        return "unknown"

    pred_dir = Path(predictions_dir)
    merged_preds_path = pred_dir / "merged_all_predictions.npy"
    merged_labels_path = pred_dir / "merged_all_labels.npy"

    if not (merged_preds_path.exists() and merged_labels_path.exists()):
        logger.warning(
            "Raw prediction arrays not found in %s. "
            "Group confusion requires predictions, not just correctness arrays. "
            "Skipping. To generate: re-run bootstrap_gap.py with --save-predictions.",
            pred_dir,
        )
        # Placeholder figure
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 2, COLUMN_WIDTH + 1.5))
        ax.text(
            0.5, 0.5,
            "Group confusion requires raw predictions.\n"
            "Re-run bootstrap_gap.py with --save-predictions",
            ha="center", va="center", fontsize=10,
            transform=ax.transAxes,
        )
        fig.savefig(str(output_path), format="pdf")
        plt.close(fig)
        return

    merged_preds = np.load(str(merged_preds_path))
    merged_labels = np.load(str(merged_labels_path))

    n_groups = len(groups)
    confusion = np.zeros((n_groups, n_groups), dtype=int)

    for true_label, pred_label in zip(merged_labels, merged_preds):
        true_group = _label_to_group(int(true_label))
        pred_group = _label_to_group(int(pred_label))
        if true_group == "unknown" or pred_group == "unknown":
            continue
        i = groups.index(true_group)
        j = groups.index(pred_group)
        confusion[i, j] += 1

    # Normalize by row (true group)
    row_sums = confusion.sum(axis=1, keepdims=True)
    confusion_pct = confusion / np.maximum(row_sums, 1) * 100

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 2, COLUMN_WIDTH + 1.5))
    im = ax.imshow(confusion_pct, cmap="Blues", vmin=0, vmax=100, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, label="% of True Group Predictions", shrink=0.85)
    cbar.ax.tick_params(labelsize=9)

    short_names = [GROUP_SHORT.get(g, g) for g in groups]
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(short_names, rotation=30, ha="right")
    ax.set_yticks(range(n_groups))
    ax.set_yticklabels(short_names)
    ax.set_xlabel("Predicted Group")
    ax.set_ylabel("True Group")

    for i in range(n_groups):
        for j in range(n_groups):
            val = confusion_pct[i, j]
            color = "white" if val > 50 else "black"
            fontweight = "bold" if i == j else "normal"
            ax.text(
                j, i, f"{val:.1f}%",
                ha="center", va="center", fontsize=9,
                color=color, fontweight=fontweight,
            )

    plt.tight_layout()
    fig.savefig(str(output_path), format="pdf")
    plt.close(fig)
    logger.info("Saved group confusion to %s", output_path)


# ============================================================================
# Fig D --- Regional Task Vector Cosine Similarity
# ============================================================================

def plot_regional_cosine(
    regional_data: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Heatmap of pairwise cosine similarity between regional task vectors.

    Two panels: cosine similarity (left) and sign agreement (right).

    Args:
        regional_data: From regional_composition_results.json.
        output_path: Output PDF path.
    """
    pairwise = regional_data.get("pairwise", regional_data.get("task_vector_geometry", []))

    # Alternate: compute from task vectors directly
    if not pairwise and "tv_dir" in regional_data:
        _plot_regional_cosine_from_tvs(regional_data["tv_dir"], output_path)
        return

    if not pairwise:
        logger.warning("No regional pairwise data found. Skipping.")
        return

    all_regions: set[str] = set()
    for entry in pairwise:
        pair = entry.get("pair", "")
        if "_vs_" in pair:
            a, b = pair.split("_vs_", 1)
            all_regions.add(a)
            all_regions.add(b)

    regions = sorted(all_regions)
    n = len(regions)
    cos_mat = np.zeros((n, n))
    sign_mat = np.zeros((n, n))
    np.fill_diagonal(cos_mat, 1.0)
    np.fill_diagonal(sign_mat, 1.0)

    for entry in pairwise:
        pair = entry.get("pair", "")
        if "_vs_" not in pair:
            continue
        a, b = pair.split("_vs_", 1)
        if a in regions and b in regions:
            i, j = regions.index(a), regions.index(b)
            cos = entry.get("cosine_similarity", 0.0)
            sign = entry.get("sign_agreement", 0.5)
            cos_mat[i, j] = cos_mat[j, i] = cos
            sign_mat[i, j] = sign_mat[j, i] = sign

    short_labels = [REGIONAL_SHORT.get(r, r.replace("_", " ")) for r in regions]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.5))

    im1 = ax1.imshow(cos_mat, cmap="RdBu_r", vmin=0.0, vmax=0.2, interpolation="nearest")
    cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label("Cosine Similarity", fontsize=10)
    cbar1.ax.tick_params(labelsize=9)
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(short_labels, rotation=45, ha="right")
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(short_labels)
    for i in range(n):
        for j in range(n):
            if i != j:
                ax1.text(j, i, f"{cos_mat[i, j]:.3f}",
                         ha="center", va="center", fontsize=10, fontweight="bold")

    im2 = ax2.imshow(sign_mat, cmap="YlOrRd", vmin=0.49, vmax=0.56, interpolation="nearest")
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label("Sign Agreement", fontsize=10)
    cbar2.ax.tick_params(labelsize=9)
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(short_labels, rotation=45, ha="right")
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(short_labels)
    ax2.text(0.5, 1.02, "chance = 0.50", transform=ax2.transAxes,
             fontsize=9, fontstyle="italic", ha="center", va="bottom")
    for i in range(n):
        for j in range(n):
            if i != j:
                ax2.text(j, i, f"{sign_mat[i, j]:.3f}",
                         ha="center", va="center", fontsize=10)

    plt.tight_layout()
    fig.savefig(str(output_path), format="pdf")
    plt.close(fig)
    logger.info("Saved regional cosine matrix to %s", output_path)


def _plot_regional_cosine_from_tvs(
    tv_dir: str,
    output_path: str | Path,
) -> None:
    """Fallback: compute regional cosine directly from saved task vectors."""
    import torch

    regions = [
        ("R1_east_africa", "R1 E.Africa"),
        ("R2_south_asia", "R2 S.Asia"),
        ("R3_neotropics", "R3 Neotrop."),
        ("R4_north_america", "R4 N.Amer."),
    ]

    tv_path = Path(tv_dir)
    tvs: dict[str, dict[str, torch.Tensor]] = {}
    for region_id, _ in regions:
        p = tv_path / f"tau_{region_id}.pt"
        if p.exists():
            tvs[region_id] = torch.load(str(p), map_location="cpu")

    found_regions = [(rid, rname) for rid, rname in regions if rid in tvs]
    if len(found_regions) < 2:
        logger.warning("Need at least 2 regional TVs. Found: %d. Skipping.", len(found_regions))
        return

    n = len(found_regions)
    cosine_matrix = np.zeros((n, n))

    for i, (rid_a, _) in enumerate(found_regions):
        for j, (rid_b, _) in enumerate(found_regions):
            if i == j:
                cosine_matrix[i, j] = 1.0
                continue
            if j < i:
                cosine_matrix[i, j] = cosine_matrix[j, i]
                continue
            shared_keys = sorted(set(tvs[rid_a].keys()) & set(tvs[rid_b].keys()))
            flat_a = torch.cat([tvs[rid_a][k].flatten().float() for k in shared_keys])
            flat_b = torch.cat([tvs[rid_b][k].flatten().float() for k in shared_keys])
            cos = torch.nn.functional.cosine_similarity(
                flat_a.unsqueeze(0), flat_b.unsqueeze(0),
            ).item()
            cosine_matrix[i, j] = cos
            cosine_matrix[j, i] = cos

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 1.0, COLUMN_WIDTH + 0.5))
    im = ax.imshow(cosine_matrix, cmap="RdBu_r", vmin=-0.1, vmax=0.3, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, label="Cosine Similarity", shrink=0.85)
    cbar.ax.tick_params(labelsize=9)

    labels = [rname for _, rname in found_regions]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)

    for i in range(n):
        for j in range(n):
            val = cosine_matrix[i, j]
            color = "white" if abs(val) > 0.15 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, color=color,
                    fontweight="bold" if i == j else "normal")

    ax.text(0.5, -0.18, "Species-group pairwise cosines: 0.01-0.09",
            transform=ax.transAxes, ha="center", fontsize=8, style="italic", color="gray")

    plt.tight_layout()
    fig.savefig(str(output_path), format="pdf")
    plt.close(fig)
    logger.info("Saved regional cosine matrix (from TVs) to %s", output_path)


# ============================================================================
# Fig E --- Calibration Reliability Diagrams
# ============================================================================

def plot_calibration(
    predictions_dir: str,
    output_path: str | Path,
    n_bins: int = 10,
) -> None:
    """Reliability diagrams for joint vs merged model predictions.

    Args:
        predictions_dir: Dir with *_probabilities.npy files.
        output_path: Path to save PDF.
        n_bins: Number of calibration bins.
    """
    pred_dir = Path(predictions_dir)
    joint_probs_path = pred_dir / "joint_all_probabilities.npy"
    merged_probs_path = pred_dir / "merged_all_probabilities.npy"
    labels_path = pred_dir / "joint_all_labels.npy"

    if not (joint_probs_path.exists() and merged_probs_path.exists() and labels_path.exists()):
        logger.warning(
            "Calibration requires probability arrays (*_probabilities.npy). "
            "Re-run bootstrap_gap.py with --save-probabilities. Skipping.",
        )
        return

    joint_probs = np.load(str(joint_probs_path))
    merged_probs = np.load(str(merged_probs_path))
    labels = np.load(str(labels_path))

    def _compute_calibration(
        probs: np.ndarray, labels_arr: np.ndarray, n_b: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Compute calibration curve -> (bin_centers, bin_accs, bin_counts, ECE)."""
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        correct = (predictions == labels_arr).astype(float)

        bin_edges = np.linspace(0, 1, n_b + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_accs = np.zeros(n_b)
        bin_confs = np.zeros(n_b)
        bin_counts = np.zeros(n_b)

        for i in range(n_b):
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_accs[i] = correct[mask].mean()
                bin_confs[i] = confidences[mask].mean()
                bin_counts[i] = mask.sum()

        ece = np.sum(
            bin_counts / max(len(labels_arr), 1) * np.abs(bin_accs - bin_confs)
        )
        return bin_centers, bin_accs, bin_counts, ece

    j_centers, j_accs, _, j_ece = _compute_calibration(joint_probs, labels, n_bins)
    m_centers, m_accs, _, m_ece = _compute_calibration(merged_probs, labels, n_bins)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.5))

    for ax, centers, accs, ece, color, label in [
        (ax1, j_centers, j_accs, j_ece, "#0072B2", "Joint"),
        (ax2, m_centers, m_accs, m_ece, "#E69F00", "Merged"),
    ]:
        ax.bar(centers, accs, width=0.08, alpha=0.6, color=color,
               edgecolor="black", linewidth=0.3)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.5, label="Perfect calibration")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        ax.text(
            0.05, 0.92, f"{label} (ECE = {ece:.3f})",
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", linewidth=0.5),
        )

    plt.tight_layout()
    fig.savefig(str(output_path), format="pdf")
    plt.close(fig)
    logger.info("Saved calibration diagram to %s", output_path)


# ============================================================================
# Supp --- Gap vs Training Set Size
# ============================================================================

def plot_gap_vs_size(
    gap_data: dict[str, Any],
    output_path: str | Path,
    train_sizes: Optional[dict[str, int]] = None,
) -> None:
    """Scatter: training set size (log x) vs composition gap (y).

    Shows that larger groups lose more from merging, smaller groups gain.

    Args:
        gap_data: Loaded from gap_analysis.json.
        output_path: Path to save PDF.
        train_sizes: Dict mapping group name to sample count.
    """
    if train_sizes is None:
        train_sizes = _FALLBACK_TRAIN_SIZES

    per_group = gap_data.get("per_group", gap_data.get("per_group_gaps", {}))
    if not per_group:
        return

    groups = [g for g in ALL_GROUPS if g in per_group and g in train_sizes]
    sizes = [train_sizes[g] for g in groups]

    gaps_pp: list[float] = []
    ci_low_pp: list[float] = []
    ci_high_pp: list[float] = []

    for g in groups:
        gdata = per_group[g]
        gap = gdata.get("gap_pp", gdata.get("gap", 0))
        if isinstance(gap, float) and abs(gap) < 1:
            gap *= 100
        gaps_pp.append(gap)

        ci = gdata.get("ci_95", gdata.get("bootstrap_ci",
                 [gdata.get("ci_low", gap / 100), gdata.get("ci_high", gap / 100)]))
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            lo, hi = ci[0], ci[1]
            if abs(lo) < 1:
                lo *= 100
                hi *= 100
            ci_low_pp.append(gap - lo)
            ci_high_pp.append(hi - gap)
        else:
            ci_low_pp.append(0)
            ci_high_pp.append(0)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 1.0, 3.2))

    for i, g in enumerate(groups):
        color = GROUP_COLORS.get(g, "#999999")
        ax.errorbar(
            sizes[i], gaps_pp[i],
            yerr=[[ci_low_pp[i]], [ci_high_pp[i]]],
            fmt="o", markersize=8, color=color,
            markeredgecolor="black", markeredgewidth=0.5,
            capsize=4, elinewidth=0.8,
        )
        ax.annotate(
            GROUP_SHORT.get(g, g),
            xy=(sizes[i], gaps_pp[i]),
            xytext=(sizes[i] * 1.15, gaps_pp[i] + 0.5),
            fontsize=8.5, color=color, fontweight="bold",
        )

    ax.axhline(0, color="black", linewidth=0.6, linestyle="-")
    ax.set_xscale("log")
    ax.set_xlabel("Training Set Size (clips)")
    ax.set_ylabel("Composition Gap (pp)")

    # Trend line
    log_sizes = np.log10(sizes)
    coeffs = np.polyfit(log_sizes, gaps_pp, 1)
    trend_x = np.logspace(np.log10(min(sizes) * 0.7), np.log10(max(sizes) * 1.3), 50)
    trend_y = coeffs[0] * np.log10(trend_x) + coeffs[1]
    ax.plot(trend_x, trend_y, "--", color="gray", linewidth=0.8, alpha=0.6)

    r_sq = np.corrcoef(log_sizes, gaps_pp)[0, 1] ** 2
    ax.text(0.05, 0.05, f"$R^2 = {r_sq:.2f}$ (log-linear)",
            transform=ax.transAxes, fontsize=8, color="gray")

    # Shade positive/negative regions
    ax.axhspan(0, ax.get_ylim()[1], alpha=0.04, color="#D55E00", zorder=0)
    ax.axhspan(ax.get_ylim()[0], 0, alpha=0.04, color="#009E73", zorder=0)
    ax.text(0.95, 0.95, "Joint better", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="#D55E00", alpha=0.7)
    ax.text(0.95, 0.05, "Merging better", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="#009E73", alpha=0.7)

    ax.grid(True, linewidth=0.3, alpha=0.4)
    plt.tight_layout()
    fig.savefig(str(output_path), format="pdf")
    plt.close(fig)
    logger.info("Saved gap vs size to %s", output_path)


# ============================================================================
# Fig G --- Data Efficiency Curve
# ============================================================================

def plot_data_efficiency(
    results_path: str,
    output_path: str | Path,
) -> None:
    """Plot merged accuracy vs data fraction for the target group.

    Args:
        results_path: Path to data_efficiency_results.json.
        output_path: Path to save PDF.
    """
    with open(results_path) as f:
        results = json.load(f)

    fracs = results.get("fractions", [])
    if not fracs:
        logger.warning("No data efficiency results. Skipping.")
        return

    x = [fr["fraction"] * 100 for fr in fracs]
    g4_standalone = [fr["g4_standalone_accuracy"] * 100 for fr in fracs]
    g4_merged = [fr["g4_merged_accuracy"] * 100 for fr in fracs]
    all_acc = [fr["all_group_accuracy"] * 100 for fr in fracs]
    cosines = [fr["tv_cosine_with_full"] for fr in fracs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH + 1, 3.5))

    ax1.plot(x, g4_standalone, "o-", color="#CC79A7", label="G4 standalone", markersize=6)
    ax1.plot(x, g4_merged, "s-", color="#009E73", label="G4 after merging", markersize=6)
    ax1.plot(x, all_acc, "^-", color="#0072B2", label="All-group (merged)", markersize=6)

    if results.get("joint_baseline_accuracy"):
        jb = results["joint_baseline_accuracy"] * 100
        ax1.axhline(jb, color="gray", linewidth=0.8, linestyle="--")
        ax1.annotate("Joint baseline", xy=(x[-1], jb + 0.5),
                     fontsize=8, color="gray", ha="right")

    ax1.set_xlabel("G4 Training Data (%)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)

    ax2.plot(x, cosines, "D-", color="#D55E00", markersize=6)
    ax2.set_xlabel("G4 Training Data (%)")
    ax2.set_ylabel("Cosine Similarity with Full TV")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(str(output_path), format="pdf")
    plt.close(fig)
    logger.info("Saved data efficiency curve to %s", output_path)


# ============================================================================
# Fig H --- Forgetting Dynamics (Continual Learning)
# ============================================================================

def plot_forgetting_dynamics(
    results_path: str,
    output_path: str | Path,
) -> None:
    """Plot per-group accuracy across fine-tuning epochs.

    Shows how G1-G4 accuracy degrades as the encoder adapts to G5,
    with task arithmetic as a horizontal reference line.

    Args:
        results_path: Path to continual_learning_results.json.
        output_path: Path to save PDF.
    """
    with open(results_path) as f:
        results = json.load(f)

    ft_results = results.get("finetune_per_epoch", [])
    if not ft_results:
        logger.warning("No fine-tuning epoch data. Skipping forgetting dynamics.")
        return

    lr_result = max(ft_results, key=lambda r: len(r.get("epochs", [])))
    epochs_data = lr_result["epochs"]
    lr_val = lr_result["learning_rate"]

    if not epochs_data:
        return

    epochs = [e["epoch"] + 1 for e in epochs_data]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 2.5, 3.8))

    for group in ALL_GROUPS:
        if group not in epochs_data[0].get("per_group", {}):
            continue
        accs = [e["per_group"][group] * 100 for e in epochs_data]
        color = GROUP_COLORS.get(group, "#999999")
        label = GROUP_SHORT.get(group, group)
        linestyle = "--" if group in TAXA_OTHER else "-"
        ax.plot(epochs, accs, linestyle, color=color, label=label, linewidth=1.5)

    if "old_group_mean" in epochs_data[0]:
        old_means = [e["old_group_mean"] * 100 for e in epochs_data]
        ax.plot(epochs, old_means, "k-", linewidth=2, label="G1-G4 mean", alpha=0.6)

    # Task arithmetic reference line
    merge_method = None
    for m in results.get("methods", []):
        if m["method"] in ("merge_joint_tv", "merge_individual_tvs"):
            merge_method = m
            break

    if merge_method:
        old_merge = merge_method["old_group_mean_accuracy"] * 100
        ax.axhline(old_merge, color="green", linewidth=1.5, linestyle=":",
                    label=f"Task arith. (old mean: {old_merge:.1f}%)")

    ax.set_xlabel("Fine-tuning Epoch")
    ax.set_ylabel("Accuracy (%)")

    ax.text(
        0.98, 0.02, f"LR = {lr_val:.0e}",
        transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="gray", linewidth=0.5),
    )

    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(str(output_path), format="pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved forgetting dynamics to %s", output_path)


# ============================================================================
# Fig I --- Per-Layer Pairwise Cosine Heatmap
# ============================================================================

def plot_perlayer_pairwise_cosine(
    sparsity_detail_path: str,
    output_path: str | Path,
) -> None:
    """Full heatmap: rows = group pairs, columns = transformer layers.

    Shows where in the network groups agree vs diverge.

    Args:
        sparsity_detail_path: Path to sparsity_detail.json.
        output_path: Path to save PDF.
    """
    with open(sparsity_detail_path) as f:
        detail = json.load(f)

    pairwise = detail.get("pairwise", [])
    if not pairwise:
        logger.warning("No pairwise data in sparsity_detail.json. Skipping.")
        return

    sg_pairs = [
        p for p in pairwise
        if any(g in p["pair"] for g in ["G1_", "G2_", "G3_", "G4_", "G5_"])
    ]
    if not sg_pairs:
        sg_pairs = pairwise

    def _layer_index(key: str) -> int:
        m = re.search(r"layers\.(\d+)\.", key)
        if m:
            return int(m.group(1))
        if "embed" in key.lower() or "patch" in key.lower():
            return -1
        return -2

    all_layer_keys: set[int] = set()
    for p in sg_pairs:
        per_layer = p.get("per_layer_cosine", {})
        for key in per_layer:
            all_layer_keys.add(_layer_index(key))

    layer_indices = sorted([li for li in all_layer_keys if li >= -1])
    layer_labels = ["Emb" if li == -1 else str(li) for li in layer_indices]

    pair_short: list[str] = []
    for p in sg_pairs:
        pn = p["pair"].replace("_vs_", " vs ")
        short = pn
        for full, s in GROUP_SHORT.items():
            short = short.replace(full, s.split()[0])
        pair_short.append(short)

    matrix = np.zeros((len(sg_pairs), len(layer_indices)))
    for i, p in enumerate(sg_pairs):
        per_layer = p.get("per_layer_cosine", {})
        layer_vals: dict[int, list[float]] = {li: [] for li in layer_indices}
        for key, cos_val in per_layer.items():
            l_idx = _layer_index(key)
            if l_idx in layer_vals:
                layer_vals[l_idx].append(cos_val)

        for j, l_idx in enumerate(layer_indices):
            vals = layer_vals[l_idx]
            matrix[i, j] = np.mean(vals) if vals else 0.0

    fig_height = max(3, 0.4 * len(sg_pairs) + 1.5)
    fig, ax = plt.subplots(figsize=(FULL_WIDTH + 1, fig_height))

    im = ax.imshow(
        matrix, aspect="auto", cmap="RdBu_r",
        vmin=-0.15, vmax=0.15, interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, label="Cosine Similarity", shrink=0.85)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xticks(range(len(layer_labels)))
    ax.set_xticklabels(layer_labels, fontsize=9)
    ax.set_yticks(range(len(pair_short)))
    ax.set_yticklabels(pair_short, fontsize=9)
    ax.set_xlabel("Transformer Layer")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if abs(val) > 0.03:
                color = "white" if abs(val) > 0.08 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    plt.tight_layout()
    fig.savefig(str(output_path), format="pdf")
    plt.close(fig)
    logger.info("Saved per-layer pairwise cosine to %s", output_path)


# ============================================================================
# Main --- config-driven orchestrator
# ============================================================================

FIGURE_REGISTRY = {
    "spectral_vs_cosine": "Fig 4a: Spectral distance vs task vector cosine scatter",
    "asymmetric_gap":     "Fig 4b: Asymmetric gap bar chart with bootstrap CIs",
    "domain_negation":    "Fig 5:  Domain negation Pareto curves",
    "confusion":          "Fig C:  Group-level confusion matrix (5x5)",
    "regional_cosine":    "Fig D:  Regional task vector cosine similarity",
    "calibration":        "Fig E:  Calibration reliability diagrams",
    "gap_vs_size":        "Supp:   Per-group gap vs training set size scatter",
    "data_efficiency":    "Fig G:  Data efficiency curve",
    "forgetting":         "Fig H:  Forgetting dynamics curve",
    "perlayer_cosine":    "Fig I:  Per-layer pairwise cosine heatmap",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper figures from experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {k:22s} {v}" for k, v in FIGURE_REGISTRY.items()
        ),
    )
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--output", type=str, default="figures/")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--only", nargs="*", type=str, default=None,
        help="Only generate these figures (by key). Default: all available.",
    )
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["paths"]["project_root"]).resolve()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_dir = root / cfg["paths"]["results"]
    tv_dir = str(root / cfg["paths"]["task_vectors"])
    species_groups_dir = str(root / cfg["paths"]["species_groups"])

    figures_to_make = set(args.only) if args.only else set(FIGURE_REGISTRY.keys())
    made = 0

    logger.info("=" * 60)
    logger.info("PAPER FIGURES GENERATOR")
    logger.info("Figures to make: %s", sorted(figures_to_make))
    logger.info("Output: %s", out_dir)
    logger.info("=" * 60)

    # --- Fig 4a: Spectral vs Cosine ---
    if "spectral_vs_cosine" in figures_to_make:
        for candidate in [
            results_dir / "spectral_distance" / "spectral_distance_results.json",
            results_dir / "spectral" / "spectral_distance_results.json",
            results_dir / "analysis" / "spectral_distance_results.json",
        ]:
            if candidate.exists():
                with open(candidate) as f:
                    data = json.load(f)
                plot_spectral_vs_cosine(
                    data["pairs"],
                    str(out_dir / "spectral_vs_cosine.pdf"),
                )
                made += 1
                break
        else:
            logger.warning("Spectral distance results not found.")

    # --- Fig 4b: Asymmetric gap ---
    if "asymmetric_gap" in figures_to_make:
        gap_path = results_dir / "gap_analysis" / "gap_analysis.json"
        if gap_path.exists():
            with open(gap_path) as f:
                gap_data = json.load(f)
            plot_asymmetric_gap(gap_data, str(out_dir / "asymmetric_gap.pdf"))
            made += 1
        else:
            logger.warning("gap_analysis.json not found. Run bootstrap_gap.py first.")

    # --- Fig 5: Domain negation ---
    if "domain_negation" in figures_to_make:
        for candidate in [
            results_dir / "negation" / "negation_results.json",
            results_dir / "domain_negation" / "domain_negation_results.json",
            results_dir / "negation" / "negation_checkpoint.json",
        ]:
            if candidate.exists():
                with open(candidate) as f:
                    data = json.load(f)
                plot_domain_negation(data, str(out_dir / "domain_negation_curves.pdf"))
                made += 1
                break
        else:
            logger.warning("Domain negation results not found.")

    # --- Fig C: Group confusion ---
    if "confusion" in figures_to_make:
        pred_dir = str(results_dir / "gap_analysis" / "predictions")
        plot_group_confusion(pred_dir, species_groups_dir,
                             str(out_dir / "group_confusion.pdf"))
        made += 1

    # --- Fig D: Regional cosine ---
    if "regional_cosine" in figures_to_make:
        for candidate in [
            results_dir / "regional" / "regional_composition_results.json",
            results_dir / "regional" / "regional_sparsity.json",
        ]:
            if candidate.exists():
                with open(candidate) as f:
                    data = json.load(f)
                plot_regional_cosine(data, str(out_dir / "regional_cosine_matrix.pdf"))
                made += 1
                break
        else:
            # Fallback: compute from task vectors directly
            plot_regional_cosine(
                {"tv_dir": tv_dir},
                str(out_dir / "regional_cosine_matrix.pdf"),
            )
            made += 1

    # --- Fig E: Calibration ---
    if "calibration" in figures_to_make:
        pred_dir = str(results_dir / "gap_analysis" / "predictions")
        plot_calibration(pred_dir, str(out_dir / "calibration.pdf"))
        made += 1

    # --- Supp: Gap vs size ---
    if "gap_vs_size" in figures_to_make:
        gap_path = results_dir / "gap_analysis" / "gap_analysis.json"
        if gap_path.exists():
            with open(gap_path) as f:
                gap_data = json.load(f)
            train_sizes = load_train_sizes(species_groups_dir)
            plot_gap_vs_size(gap_data, str(out_dir / "gap_vs_training_size.pdf"), train_sizes)
            made += 1
        else:
            logger.warning("gap_analysis.json not found. Run bootstrap_gap.py first.")

    # --- Fig G: Data efficiency ---
    if "data_efficiency" in figures_to_make:
        de_path = results_dir / "data_efficiency" / "data_efficiency_results.json"
        if de_path.exists():
            plot_data_efficiency(str(de_path), str(out_dir / "data_efficiency.pdf"))
            made += 1
        else:
            logger.warning("data_efficiency_results.json not found.")

    # --- Fig H: Forgetting dynamics ---
    if "forgetting" in figures_to_make:
        cl_path = results_dir / "continual_learning" / "continual_learning_results.json"
        if cl_path.exists():
            plot_forgetting_dynamics(str(cl_path), str(out_dir / "forgetting_dynamics.pdf"))
            made += 1
        else:
            logger.warning("continual_learning_results.json not found.")

    # --- Fig I: Per-layer pairwise cosine ---
    if "perlayer_cosine" in figures_to_make:
        sp_path = results_dir / "analysis" / "sparsity_detail.json"
        if sp_path.exists():
            plot_perlayer_pairwise_cosine(str(sp_path),
                                          str(out_dir / "perlayer_pairwise_cosine.pdf"))
            made += 1
        else:
            logger.warning("sparsity_detail.json not found.")

    logger.info("=" * 60)
    logger.info("Generated %d/%d figures in %s", made, len(figures_to_make), out_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()