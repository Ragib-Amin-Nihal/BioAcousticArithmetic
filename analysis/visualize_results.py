"""Static results visualizations from JSON experiment outputs.

Produces (no GPU needed — reads pre-computed results):
  - figures/lmc_curves.pdf          — Small-multiples grid of LMC interpolation curves
  - figures/composition_bars.pdf    — Grouped bar chart: methods ranked by AllAcc
  - figures/cosine_matrix.pdf       — Pairwise cosine similarity + sign agreement heatmap
  - figures/norm_adjusted_comparison.pdf — Norm-adjusted vs uniform per-group bars
  - figures/sparsity_profile.pdf    — Sparsity bar chart + magnitude summary
  - figures/taxonomy_vs_cosine.pdf  — Taxonomy category vs cosine scatter

Changes from previous version:
  - All font sizes increased (minimum ~8 pt at print size)
  - All suptitle() calls removed (captions in paper)
  - LMC figure: uses set_title() for pair identity (outside axes area),
    rendered at 10" width with 13pt fonts so LaTeX 0.7x scaling yields
    ~9pt at print.  Other figures use FULL_WIDTH directly.
  - Annotation sizes increased from 6-7pt to 8-9pt
  - Marker sizes increased
  - Colorbar label sizes increased

Usage:
    python analysis/visualize_results.py \\
        --results-dir results/ \\
        --output figures/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --- Style ---
# Import shared style if available, otherwise define inline
try:
    from figure_style import (
        apply_style, GROUP_COLORS, GROUP_SHORT, TAXA_BIRD, TAXA_OTHER,
        COLUMN_WIDTH, FULL_WIDTH, method_color,
    )
    apply_style()
except ImportError:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RCPARAMS = {
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
    }
    plt.rcParams.update(RCPARAMS)

    GROUP_COLORS = {
        "G1_passerines": "#0072B2",
        "G2_nonpasserine_birds": "#E69F00",
        "G3_raptors_waterbirds": "#009E73",
        "G4_marine_mammals": "#CC79A7",
        "G5_amphibians": "#56B4E9",
    }
    GROUP_SHORT = {
        "G1_passerines": "G1",
        "G2_nonpasserine_birds": "G2",
        "G3_raptors_waterbirds": "G3",
        "G4_marine_mammals": "G4",
        "G5_amphibians": "G5",
    }
    TAXA_BIRD = {"G1_passerines", "G2_nonpasserine_birds", "G3_raptors_waterbirds"}
    TAXA_OTHER = {"G4_marine_mammals", "G5_amphibians"}
    COLUMN_WIDTH = 3.35
    FULL_WIDTH = 7.0

    def method_color(name: str) -> str:
        n = name.lower()
        if "dare" in n: return "#E69F00"
        if "task_arithmetic" in n: return "#0072B2"
        if "ties" in n: return "#CC79A7"
        if "simple_average" in n: return "#009E73"
        if "della" in n: return "#56B4E9"
        return "#999999"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# LMC Interpolation Curves  (Paper Figure 2)
# ---------------------------------------------------------------------------

def plot_lmc_curves(
    lmc_data: dict,
    output_path: str,
) -> None:
    """Plot small-multiples grid of LMC interpolation curves.

    Shows loss and accuracy vs alpha for each pair.  Each subplot has two
    y-axes: loss (left, red) and accuracy (right, blue).

    Subplot titles identify pairs; no suptitle (paper caption provides
    context).

    Layout strategy:
      The figure is rendered wider than the final print column
      (10 in vs 7 in) so that each subplot has enough room for dual
      y-axes.  LaTeX scales by ~0.7x, so all font sizes here are
      set ~1.4x larger than the desired print size:
          13 pt title  -> 9.1 pt print
          12 pt label  -> 8.4 pt print
          11 pt ticks  -> 7.7 pt print
          10 pt annot. -> 7.0 pt print

    Args:
        lmc_data: Loaded from lmc_results_corrected.json.
        output_path: Path to save PDF.
    """
    pairs_raw = lmc_data["pairs"]

    # Group by (model_a, model_b) pair — each pair has 2 eval directions
    pair_dict: dict[tuple[str, str], list[dict]] = {}
    for p in pairs_raw:
        key = (p["model_a"], p["model_b"])
        if key not in pair_dict:
            pair_dict[key] = []
        pair_dict[key].append(p)

    unique_pairs = sorted(pair_dict.keys())
    n_pairs = len(unique_pairs)

    # Layout: 5 cols x 2 rows (or fewer if < 10 pairs)
    n_cols = min(5, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols

    # Wider-than-print figure: 10" gives 2" per subplot.
    # LaTeX \includegraphics[width=\textwidth] scales to 7" -> factor 0.7.
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(10, 2.3 * n_rows),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.55, wspace=0.55)

    color_loss = "#D62728"
    color_acc = "#1F77B4"

    for idx, pair_key in enumerate(unique_pairs):
        row, col = divmod(idx, n_cols)
        ax_loss = axes[row][col]
        ax_acc = ax_loss.twinx()

        directions = pair_dict[pair_key]
        ga, gb = pair_key
        short_a = GROUP_SHORT.get(ga, ga)
        short_b = GROUP_SHORT.get(gb, gb)

        for d_idx, direction in enumerate(directions):
            points = direction["interpolation_points"]
            if isinstance(points[0], dict):
                alphas = [p["alpha"] for p in points]
                losses = [p["loss"] for p in points]
                accs = [p["accuracy"] for p in points]
            else:
                alphas = [p[0] for p in points]
                losses = [p[1] for p in points]
                accs = [p[2] for p in points]

            linestyle = "-" if d_idx == 0 else "--"

            ax_loss.plot(
                alphas, losses,
                color=color_loss, linestyle=linestyle, linewidth=1.5,
                alpha=0.85,
            )
            ax_acc.plot(
                alphas, [a * 100 for a in accs],
                color=color_acc, linestyle=linestyle, linewidth=1.5,
                alpha=0.85,
            )

        # Pair title above the subplot (doesn't eat plot area)
        ax_loss.set_title(
            f"{short_a} ↔ {short_b}",
            fontsize=13, fontweight="bold", pad=4,
        )

        ax_loss.set_xlabel("α", fontsize=12)
        ax_loss.tick_params(axis="y", labelcolor=color_loss, labelsize=11)
        ax_loss.tick_params(axis="x", labelsize=11)
        ax_acc.tick_params(axis="y", labelcolor=color_acc, labelsize=11)

        # Shared y-axis labels: only edge columns
        if col == 0:
            ax_loss.set_ylabel("Loss", color=color_loss, fontsize=12)
        else:
            ax_loss.set_ylabel("")
        if col == n_cols - 1 or idx == n_pairs - 1:
            ax_acc.set_ylabel("Acc (%)", color=color_acc, fontsize=12)
        else:
            ax_acc.set_ylabel("")

        # Barrier annotation (bottom-right, compact)
        barrier = direction.get("true_barrier", 0.0)
        mono = direction.get("is_monotonic", True)
        status = "✓ mono" if mono else f"barrier={barrier:.3f}"
        ax_loss.text(
            0.97, 0.05, status,
            transform=ax_loss.transAxes, fontsize=10,
            ha="right", va="bottom",
            color="#2ca02c" if mono else "#d62728",
            fontstyle="italic",
        )

    # Hide unused subplots
    for idx in range(n_pairs, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    # Shared legend at bottom
    legend_elements = [
        Line2D([0], [0], color=color_loss, linewidth=1.8, label="Loss"),
        Line2D([0], [0], color=color_acc, linewidth=1.8, label="Accuracy"),
        Line2D([0], [0], color="gray", linewidth=1.5, linestyle="-", label="Eval on A"),
        Line2D([0], [0], color="gray", linewidth=1.5, linestyle="--", label="Eval on B"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center", ncol=4,
        bbox_to_anchor=(0.5, -0.01),
        frameon=False, fontsize=11,
    )

    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved LMC curves to %s", output_path)


# ---------------------------------------------------------------------------
# Composition Results Bar Chart
# ---------------------------------------------------------------------------

def plot_composition_bars(
    comp_data: dict,
    output_path: str,
    joint_baseline_acc: float = 68.48,
) -> None:
    """Grouped bar chart of composition methods ranked by AllAcc.

    Args:
        comp_data: Loaded from composition_results.json.
        output_path: Path to save PDF.
        joint_baseline_acc: Joint training AllAcc for reference line.
    """
    trials = comp_data.get("trials", comp_data.get("results", []))
    if not trials:
        logger.warning("No trials found in composition results")
        return

    # Sort by AllAcc descending
    trials_sorted = sorted(
        trials,
        key=lambda t: t.get("all_acc", t.get("unified_accuracy", 0)),
        reverse=True,
    )
    trials_sorted = trials_sorted[:12]

    fig, (ax_main, ax_group) = plt.subplots(
        2, 1, figsize=(FULL_WIDTH, 7),
        gridspec_kw={"height_ratios": [1, 1.5]},
    )

    # --- Top panel: AllAcc bar chart ---
    method_labels = []
    all_accs = []

    for t in trials_sorted:
        method = t.get("method", t.get("trial_name", "unknown"))
        params = t.get("params", {})
        if isinstance(params, dict):
            param_str = ", ".join(f"{k}={v}" for k, v in params.items() if k != "method")
            label = f"{method}\n({param_str})" if param_str else method
        else:
            label = method
        method_labels.append(label)
        all_accs.append(t.get("all_acc", t.get("unified_accuracy", 0)) * 100)

    x = np.arange(len(method_labels))
    bar_width = 0.65
    bar_colors = [method_color(t.get("method", "")) for t in trials_sorted]

    bars = ax_main.barh(x, all_accs, bar_width, color=bar_colors,
                        edgecolor="black", linewidth=0.4)

    # CIs
    ci_lows, ci_highs = [], []
    for t in trials_sorted:
        ci = t.get("all_acc_ci", t.get("unified_ci", [0, 0]))
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            ci_lows.append(ci[0] * 100)
            ci_highs.append(ci[1] * 100)
        else:
            ci_lows.append(0)
            ci_highs.append(0)

    if any(ci > 0 for ci in ci_highs):
        xerr_low = [max(0, a - cl) for a, cl in zip(all_accs, ci_lows)]
        xerr_high = [max(0, ch - a) for a, ch in zip(all_accs, ci_highs)]
        ax_main.errorbar(
            all_accs, x, xerr=[xerr_low, xerr_high],
            fmt="none", ecolor="black", elinewidth=0.7, capsize=3,
        )

    # Joint baseline
    ax_main.axvline(
        joint_baseline_acc, color="#D55E00", linewidth=1.5, linestyle="--",
        label=f"Joint baseline ({joint_baseline_acc:.1f}%)",
    )

    # Value annotations
    for i, (bar, val) in enumerate(zip(bars, all_accs)):
        ax_main.text(
            val + 0.3, i, f"{val:.1f}%",
            va="center", ha="left", fontsize=9, fontweight="bold",
        )

    ax_main.set_yticks(x)
    ax_main.set_yticklabels(method_labels, fontsize=9)
    ax_main.set_xlabel("All-Group Accuracy (%)")
    ax_main.legend(loc="lower right", fontsize=9)
    ax_main.invert_yaxis()

    # --- Bottom panel: Per-group breakdown for top methods ---
    top_n = min(6, len(trials_sorted))
    top_trials = trials_sorted[:top_n]
    group_names = sorted(GROUP_COLORS.keys())
    n_groups = len(group_names)
    group_bar_width = 0.8 / top_n
    x_groups = np.arange(n_groups)

    for i, trial in enumerate(top_trials):
        method = trial.get("method", trial.get("trial_name", ""))
        per_group = trial.get("per_group", trial.get("per_group_accuracy", {}))
        vals = []
        for gname in group_names:
            acc = per_group.get(gname, per_group.get(GROUP_SHORT.get(gname, ""), 0))
            if isinstance(acc, dict):
                acc = acc.get("accuracy", acc.get("acc", 0))
            vals.append(acc * 100 if acc <= 1.0 else acc)
        offset = (i - top_n / 2 + 0.5) * group_bar_width
        ax_group.bar(
            x_groups + offset, vals, group_bar_width,
            label=method, edgecolor="black", linewidth=0.3, alpha=0.85,
        )

    ax_group.set_xticks(x_groups)
    ax_group.set_xticklabels([GROUP_SHORT[g] for g in group_names], fontsize=10)
    ax_group.set_ylabel("Accuracy (%)")
    ax_group.legend(fontsize=8, ncol=2, loc="lower right", frameon=True)

    plt.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved composition bars to %s", output_path)


# ---------------------------------------------------------------------------
# Cosine Similarity Matrix  (Paper Figure 3)
# ---------------------------------------------------------------------------

def plot_cosine_matrix(
    sparsity_data: dict,
    output_path: str,
) -> None:
    """Plot pairwise cosine similarity + sign agreement heatmaps.

    Args:
        sparsity_data: Loaded from sparsity_summary.json.
        output_path: Path to save PDF.
    """
    pairwise = sparsity_data.get("pairwise", sparsity_data.get("pairwise_analysis", []))
    if not pairwise:
        logger.warning("No pairwise data in sparsity summary")
        return

    # Discover groups
    all_groups: set[str] = set()
    for entry in pairwise:
        pair = entry.get("pair", "")
        if "_vs_" in pair:
            a, b = pair.split("_vs_", 1)
            all_groups.add(a)
            all_groups.add(b)

    groups = sorted(all_groups)
    n = len(groups)

    # Build matrices
    cosine_mat = np.zeros((n, n))
    sign_mat = np.zeros((n, n))
    np.fill_diagonal(cosine_mat, 1.0)
    np.fill_diagonal(sign_mat, 1.0)

    for entry in pairwise:
        pair = entry.get("pair", "")
        if "_vs_" not in pair:
            continue
        a, b = pair.split("_vs_", 1)
        if a in groups and b in groups:
            i, j = groups.index(a), groups.index(b)
            cos = entry.get("cosine_similarity", 0.0)
            sign = entry.get("sign_agreement", 0.5)
            cosine_mat[i, j] = cos
            cosine_mat[j, i] = cos
            sign_mat[i, j] = sign
            sign_mat[j, i] = sign

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.8))

    short_labels = [GROUP_SHORT.get(g, g) for g in groups]

    # --- Cosine similarity heatmap ---
    im1 = ax1.imshow(cosine_mat, cmap="RdBu_r", vmin=-0.1, vmax=0.2,
                      interpolation="nearest")
    cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label("Cosine Similarity", fontsize=10)
    cbar1.ax.tick_params(labelsize=9)

    ax1.set_xticks(range(n))
    ax1.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=10)
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(short_labels, fontsize=10)

    # Cell annotations — large enough to read
    for i in range(n):
        for j in range(n):
            if i != j:
                ax1.text(
                    j, i, f"{cosine_mat[i, j]:.3f}",
                    ha="center", va="center", fontsize=9,
                    fontweight="bold" if cosine_mat[i, j] > 0.05 else "normal",
                )

    # Bird taxonomy boundary box
    bird_indices = [groups.index(g) for g in groups if g in TAXA_BIRD]
    if bird_indices:
        min_b, max_b = min(bird_indices), max(bird_indices)
        rect = plt.Rectangle(
            (min_b - 0.5, min_b - 0.5),
            max_b - min_b + 1, max_b - min_b + 1,
            fill=False, edgecolor="green", linewidth=2, linestyle="--",
        )
        ax1.add_patch(rect)
        ax1.text(
            min_b - 0.3, min_b - 0.7, "Birds",
            fontsize=9, color="green", fontweight="bold",
        )

    # --- Sign agreement heatmap ---
    im2 = ax2.imshow(sign_mat, cmap="YlOrRd", vmin=0.49, vmax=0.55,
                      interpolation="nearest")
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label("Sign Agreement", fontsize=10)
    cbar2.ax.tick_params(labelsize=9)

    ax2.set_xticks(range(n))
    ax2.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=10)
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(short_labels, fontsize=10)

    # Subtitle as axis text (since we don't use set_title)
    ax2.text(
        0.5, 1.02, "chance = 0.50",
        transform=ax2.transAxes, fontsize=9, fontstyle="italic",
        ha="center", va="bottom",
    )

    for i in range(n):
        for j in range(n):
            if i != j:
                ax2.text(
                    j, i, f"{sign_mat[i, j]:.3f}",
                    ha="center", va="center", fontsize=9,
                )

    plt.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved cosine matrix to %s", output_path)


# ---------------------------------------------------------------------------
# Norm-Adjusted vs Uniform Comparison
# ---------------------------------------------------------------------------

def plot_norm_adjusted_comparison(
    norm_data: dict,
    output_path: str,
) -> None:
    """Side-by-side per-group accuracy comparison: uniform vs norm-adjusted.

    Args:
        norm_data: Loaded from norm_adjusted_results.json.
        output_path: Path to save PDF.
    """
    trials = norm_data.get("trials", norm_data.get("results", []))
    if not trials:
        logger.warning("No trials in norm-adjusted results")
        return

    uniform_trials = [t for t in trials if t.get("weighting", "") == "uniform"]
    normed_trials = [t for t in trials if t.get("weighting", "") == "norm_adjusted"]

    def best_trial(trial_list: list[dict]) -> Optional[dict]:
        if not trial_list:
            return None
        return max(
            trial_list,
            key=lambda t: t.get("all_acc", t.get("unified_accuracy", 0)),
        )

    best_uniform = best_trial(uniform_trials)
    best_normed = best_trial(normed_trials)

    if best_uniform is None or best_normed is None:
        logger.warning("Cannot find uniform/norm_adjusted split — attempting lambda-based grouping")
        _plot_lambda_sweep(trials, output_path)
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.8))

    group_names = sorted(GROUP_COLORS.keys())
    x = np.arange(len(group_names))
    width = 0.35

    uniform_accs = _extract_per_group(best_uniform, group_names)
    normed_accs = _extract_per_group(best_normed, group_names)

    ax1.bar(x - width / 2, uniform_accs, width,
            label=f"Uniform (λ={best_uniform.get('lambda', '?')})",
            color="#0072B2", edgecolor="black", linewidth=0.4)
    ax1.bar(x + width / 2, normed_accs, width,
            label=f"Norm-adj (λ={best_normed.get('lambda', '?')})",
            color="#E69F00", edgecolor="black", linewidth=0.4)

    ax1.set_xticks(x)
    ax1.set_xticklabels([GROUP_SHORT[g] for g in group_names], rotation=15, ha="right")
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend(fontsize=9)

    # Difference annotations
    for i, (u, n_val) in enumerate(zip(uniform_accs, normed_accs)):
        diff = n_val - u
        color = "green" if diff > 0 else "red"
        ax1.text(
            i, max(u, n_val) + 0.5, f"{diff:+.1f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color=color,
        )

    # --- Lambda sweep ---
    lambdas_uniform = [(t.get("lambda", 0), t.get("all_acc", 0) * 100) for t in uniform_trials]
    lambdas_normed = [(t.get("lambda", 0), t.get("all_acc", 0) * 100) for t in normed_trials]
    lambdas_uniform.sort()
    lambdas_normed.sort()

    if lambdas_uniform:
        ax2.plot([l for l, _ in lambdas_uniform], [a for _, a in lambdas_uniform],
                 "o-", color="#0072B2", markersize=6, label="Uniform")
    if lambdas_normed:
        ax2.plot([l for l, _ in lambdas_normed], [a for _, a in lambdas_normed],
                 "s-", color="#E69F00", markersize=6, label="Norm-adjusted")

    ax2.set_xlabel("λ (scaling factor)")
    ax2.set_ylabel("All-Group Accuracy (%)")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved norm-adjusted comparison to %s", output_path)


def _extract_per_group(trial: dict, group_names: list[str]) -> list[float]:
    """Extract per-group accuracies from a trial dict."""
    per_group = trial.get("per_group", trial.get("per_group_accuracy", {}))
    result = []
    for g in group_names:
        acc = per_group.get(g, per_group.get(GROUP_SHORT.get(g, ""), 0))
        if isinstance(acc, dict):
            acc = acc.get("accuracy", acc.get("acc", 0))
        result.append(acc * 100 if acc <= 1.0 else acc)
    return result


def _plot_lambda_sweep(trials: list[dict], output_path: str) -> None:
    """Fallback: plot all trials as lambda vs accuracy."""
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 3.0))

    lambdas = [t.get("lambda", t.get("scaling", 0)) for t in trials]
    all_accs = [t.get("all_acc", t.get("unified_accuracy", 0)) * 100 for t in trials]
    mean_grp = [t.get("mean_group_acc", 0) * 100 for t in trials]

    ax.plot(lambdas, all_accs, "o-", label="All-group Acc", markersize=6)
    if any(m > 0 for m in mean_grp):
        ax.plot(lambdas, mean_grp, "s--", label="Mean per-group Acc", markersize=6)

    ax.set_xlabel("λ (scaling)")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved lambda sweep fallback to %s", output_path)


# ---------------------------------------------------------------------------
# Sparsity Profile
# ---------------------------------------------------------------------------

def plot_sparsity_cdf(
    sparsity_data: dict,
    output_path: str,
) -> None:
    """Sparsity bar chart and task vector magnitude summary.

    Args:
        sparsity_data: Loaded from sparsity_summary.json.
        output_path: Path to save PDF.
    """
    per_group = sparsity_data.get("per_group", sparsity_data.get("groups", {}))
    if not per_group:
        logger.warning("No per-group data in sparsity summary")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.5))

    groups = sorted(per_group.keys())
    thresholds = ["pct_below_1e3", "pct_below_1e4"]
    threshold_labels = ["% < 1e-3", "% < 1e-4"]
    colors_thresh = ["#4C72B0", "#DD8452"]

    x = np.arange(len(groups))
    width = 0.35

    for t_idx, (thresh_key, thresh_label) in enumerate(zip(thresholds, threshold_labels)):
        vals = []
        for g in groups:
            gdata = per_group[g]
            v = gdata.get(thresh_key, gdata.get(
                thresh_key.replace("pct_", "fraction_").replace("_1e3", "_below_0.001"),
                0
            ))
            if isinstance(v, (int, float)) and v <= 1.0:
                v *= 100
            vals.append(v)

        offset = (t_idx - 0.5) * width
        ax1.bar(x + offset, vals, width, label=thresh_label,
                color=colors_thresh[t_idx], edgecolor="black", linewidth=0.4)

    ax1.set_xticks(x)
    ax1.set_xticklabels([GROUP_SHORT.get(g, g) for g in groups], rotation=15, ha="right")
    ax1.set_ylabel("% of Parameters")
    ax1.legend(fontsize=9)

    # --- Right: L2 norm + mean magnitude ---
    norms = [per_group[g].get("l2_norm", per_group[g].get("global_l2_norm", 0)) for g in groups]
    mean_mags = [per_group[g].get("mean_magnitude", per_group[g].get("mean_abs_magnitude", 0))
                 for g in groups]

    ax2_twin = ax2.twinx()
    ax2.bar(x - 0.2, norms, 0.35, label="L2 Norm", color="#0072B2",
            edgecolor="black", linewidth=0.4)
    ax2_twin.bar(x + 0.2, [m * 1000 for m in mean_mags], 0.35,
                 label="Mean |Δ| (×10³)", color="#E69F00",
                 edgecolor="black", linewidth=0.4)

    ax2.set_xticks(x)
    ax2.set_xticklabels([GROUP_SHORT.get(g, g) for g in groups], rotation=15, ha="right")
    ax2.set_ylabel("L2 Norm", color="#0072B2")
    ax2_twin.set_ylabel("Mean |Δ| (×10³)", color="#E69F00")

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

    plt.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved sparsity profile to %s", output_path)


# ---------------------------------------------------------------------------
# Taxonomy vs Cosine Scatter
# ---------------------------------------------------------------------------

def plot_taxonomy_vs_cosine(
    sparsity_data: dict,
    output_path: str,
) -> None:
    """Scatter plot: taxonomic category vs cosine similarity.

    Args:
        sparsity_data: Loaded from sparsity_summary.json.
        output_path: Path to save PDF.
    """
    pairwise = sparsity_data.get("pairwise", sparsity_data.get("pairwise_analysis", []))
    if not pairwise:
        return

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 1.5, 3.5))

    categories = {"Bird–Bird": [], "Bird–Other": [], "Other–Other": []}

    for entry in pairwise:
        pair = entry.get("pair", "")
        if "_vs_" not in pair:
            continue
        a, b = pair.split("_vs_", 1)
        cos = entry.get("cosine_similarity", 0)

        a_bird = a in TAXA_BIRD
        b_bird = b in TAXA_BIRD

        if a_bird and b_bird:
            categories["Bird–Bird"].append(cos)
        elif a_bird or b_bird:
            categories["Bird–Other"].append(cos)
        else:
            categories["Other–Other"].append(cos)

    colors = {"Bird–Bird": "#0072B2", "Bird–Other": "#E69F00", "Other–Other": "#CC79A7"}

    for cat, vals in categories.items():
        if not vals:
            continue
        x_base = list(categories.keys()).index(cat)
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter(
            np.full(len(vals), x_base) + jitter, vals,
            c=colors[cat], s=60, edgecolors="black", linewidths=0.4,
            label=f"{cat} (n={len(vals)}, μ={np.mean(vals):.4f})",
            zorder=3,
        )
        ax.hlines(
            np.mean(vals), x_base - 0.3, x_base + 0.3,
            color=colors[cat], linewidth=2.5, linestyle="-", zorder=4,
        )

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(list(categories.keys()))
    ax.set_ylabel("Cosine Similarity")
    ax.legend(fontsize=8, loc="upper right")
    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")

    plt.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved taxonomy vs cosine to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Static results visualizations from JSON outputs"
    )
    parser.add_argument("--results-dir", type=str, default="results/")
    parser.add_argument("--output", type=str, default="figures/")
    parser.add_argument(
        "--joint-baseline-acc", type=float, default=68.48,
        help="Joint baseline AllAcc for reference line",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # LMC Curves
    lmc_path = results_dir / "lmc" / "lmc_results_corrected.json"
    if lmc_path.exists():
        logger.info("Generating LMC curves...")
        with open(lmc_path) as f:
            lmc_data = json.load(f)
        plot_lmc_curves(lmc_data, str(output_dir / "lmc_curves.pdf"))
    else:
        logger.warning("LMC results not found: %s", lmc_path)

    # Composition Results
    comp_path = results_dir / "composition" / "composition_results.json"
    if comp_path.exists():
        logger.info("Generating composition bars...")
        with open(comp_path) as f:
            comp_data = json.load(f)
        plot_composition_bars(
            comp_data, str(output_dir / "composition_bars.pdf"),
            joint_baseline_acc=args.joint_baseline_acc,
        )
    else:
        logger.warning("Composition results not found: %s", comp_path)

    # Sparsity Analysis
    sparsity_path = results_dir / "analysis" / "sparsity_summary.json"
    if sparsity_path.exists():
        logger.info("Generating cosine matrix + sparsity figures...")
        with open(sparsity_path) as f:
            sparsity_data = json.load(f)
        plot_cosine_matrix(sparsity_data, str(output_dir / "cosine_matrix.pdf"))
        plot_sparsity_cdf(sparsity_data, str(output_dir / "sparsity_profile.pdf"))
        plot_taxonomy_vs_cosine(sparsity_data, str(output_dir / "taxonomy_vs_cosine.pdf"))
    else:
        logger.warning("Sparsity results not found: %s", sparsity_path)

    # Norm-Adjusted Comparison
    norm_path = results_dir / "composition" / "norm_adjusted" / "norm_adjusted_results.json"
    if norm_path.exists():
        logger.info("Generating norm-adjusted comparison...")
        with open(norm_path) as f:
            norm_data = json.load(f)
        plot_norm_adjusted_comparison(norm_data, str(output_dir / "norm_adjusted_comparison.pdf"))
    else:
        logger.warning("Norm-adjusted results not found: %s", norm_path)

    logger.info("All static visualizations complete.")


if __name__ == "__main__":
    main()