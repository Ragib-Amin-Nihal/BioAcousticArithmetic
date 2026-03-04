"""Improved paper figures for bioacoustic task arithmetic.

Design philosophy:
  - Every figure should make ONE argument visually undeniable
  - Minimize ink-to-data ratio (Tufte principle)
  - No dual y-axes (cognitive load too high for small multiples)
  - Color encodes taxonomy, not decoration
  - Annotations guide interpretation, not just label data
  - Spines removed except where they serve as reference lines

Improvements over v1:
  Fig 1:  LMC — accuracy-only small multiples with monotonic fill
  Fig 2:  Composition — Cleveland dot plot with method families
  Fig 3:  Cosine matrix — single combined heatmap with taxonomy blocks
  Fig 4a: Spectral scatter — marginal distributions + Spearman annotation
  Fig 4b: Asymmetric gap — butterfly diverging chart
  Fig 5:  Domain negation — shaded divergence between real and control
  Fig H:  Forgetting — best LR only with shaded Pareto frontier
  NEW:    Radar chart — joint vs merged per-group comparison
  NEW:    Method complexity scatter — complexity doesn't help

Usage:
    python paper_figures_v2.py --results-dir results/ --output figures/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

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
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe

# Import improved style
from figure_style import (
    apply_style, clean_axes, annotate_bar,
    GROUP_COLORS, GROUP_SHORT, GROUP_SHORT_INLINE, ALL_GROUPS,
    TAXA_BIRD, TAXA_OTHER, METHOD_COLORS, method_color,
    COLUMN_WIDTH, FULL_WIDTH,
    BLUE, ORANGE, TEAL, PINK, SKY, RED, BLACK, GRAY60, GRAY80, GRAY90, GRAY95,
    REGIONAL_COLORS, REGIONAL_SHORT,
)

apply_style()


# ============================================================================
# Custom colormaps
# ============================================================================

# Diverging blue-white-red for cosine similarity
_cmap_cosine = LinearSegmentedColormap.from_list(
    "cosine_div",
    [(0.0, "#2166AC"), (0.3, "#92C5DE"), (0.5, "#F7F7F7"),
     (0.7, "#F4A582"), (1.0, "#B2182B")],
)

# Sequential teal for sign agreement
_cmap_sign = LinearSegmentedColormap.from_list(
    "sign_seq",
    [(0.0, "#F7FCF5"), (0.3, "#C7E9C0"), (0.5, "#74C476"),
     (0.7, "#238B45"), (1.0, "#00441B")],
)


# ============================================================================
# Fig 1: LMC Interpolation — Accuracy-Only Small Multiples
# ============================================================================

def plot_lmc_curves(lmc_data: dict, output_path: str) -> None:
    """Small multiples of LMC interpolation, accuracy only.

    Key change from v1: single y-axis (accuracy), no loss.
    Monotonicity is the claim — show it cleanly with filled area
    between the two evaluation directions.
    """
    pairs_raw = lmc_data["pairs"]

    pair_dict: dict[tuple[str, str], list[dict]] = {}
    for p in pairs_raw:
        key = (p["model_a"], p["model_b"])
        pair_dict.setdefault(key, []).append(p)

    unique_pairs = sorted(pair_dict.keys())
    n_pairs = len(unique_pairs)
    n_cols = min(5, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(FULL_WIDTH + 1.5, 2.0 * n_rows),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    for idx, pair_key in enumerate(unique_pairs):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        directions = pair_dict[pair_key]
        ga, gb = pair_key
        sa = GROUP_SHORT_INLINE.get(ga, ga)
        sb = GROUP_SHORT_INLINE.get(gb, gb)
        color_a = GROUP_COLORS.get(ga, BLUE)
        color_b = GROUP_COLORS.get(gb, ORANGE)

        all_accs = []
        for d_idx, direction in enumerate(directions):
            points = direction["interpolation_points"]
            if isinstance(points[0], dict):
                alphas = [p["alpha"] for p in points]
                accs = [p["accuracy"] * 100 for p in points]
            else:
                alphas = [p[0] for p in points]
                accs = [p[2] * 100 for p in points]

            all_accs.append(accs)
            color = color_a if d_idx == 0 else color_b
            label_g = sa if d_idx == 0 else sb
            ax.plot(
                alphas, accs,
                color=color, linewidth=1.8, alpha=0.9,
                label=f"Eval: {label_g}",
                zorder=3,
            )

        # Fill between the two directions (shows the interpolation band)
        if len(all_accs) == 2 and len(all_accs[0]) == len(all_accs[1]):
            ax.fill_between(
                alphas,
                all_accs[0], all_accs[1],
                alpha=0.08, color=GRAY80, zorder=1,
            )

        # Pair label — compact
        ax.text(
            0.5, 1.04, f"{sa} ↔ {sb}",
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            ha="center", va="bottom",
        )

        # Monotonicity indicator — green checkmark or red X
        barrier = direction.get("true_barrier", 0.0)
        mono = direction.get("is_monotonic", True)
        symbol = "✓" if mono else "✗"
        sym_color = TEAL if mono else RED
        ax.text(
            0.95, 0.08, symbol,
            transform=ax.transAxes, fontsize=14,
            ha="right", va="bottom", color=sym_color,
            fontweight="bold",
        )

        ax.set_xlim(0, 1)
        if col == 0:
            ax.set_ylabel("Acc (%)", fontsize=9)
        if row == n_rows - 1:
            ax.set_xlabel("α", fontsize=9)

        clean_axes(ax)

    # Hide unused
    for idx in range(n_pairs, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    # Shared legend at bottom
    legend_elements = [
        Line2D([0], [0], color=GRAY60, linewidth=1.8, label="Eval on model A's group"),
        Line2D([0], [0], color=GRAY60, linewidth=1.8, linestyle="--",
               label="Eval on model B's group"),
        Line2D([0], [0], marker="", color="none", label="✓ = monotonic (zero barrier)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center", ncol=3,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9, frameon=False,
    )

    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved LMC curves → %s", output_path)


# ============================================================================
# Fig 2: Composition — Cleveland Dot Plot
# ============================================================================

def plot_composition_dots(
    comp_data: dict,
    output_path: str,
    joint_baseline_acc: float = 68.48,
) -> None:
    """Cleveland dot plot: methods ranked by AllAcc.

    Key change from v1: dots instead of bars, grouped by method family,
    with the joint baseline as a prominent vertical reference.
    """
    trials = comp_data.get("trials", comp_data.get("results", []))
    if not trials:
        return

    # Best trial per method family
    family_best: dict[str, dict] = {}
    for t in trials:
        method = t.get("method", t.get("trial_name", ""))
        acc = t.get("all_acc", t.get("unified_accuracy", 0))
        family = _method_family(method)
        if family not in family_best or acc > family_best[family].get("all_acc", 0):
            family_best[family] = {**t, "family": family, "all_acc": acc}

    # Sort by accuracy
    sorted_families = sorted(family_best.values(),
                             key=lambda t: t["all_acc"], reverse=True)

    fig, (ax_main, ax_group) = plt.subplots(
        1, 2, figsize=(FULL_WIDTH + 0.5, 3.5),
        gridspec_kw={"width_ratios": [1.2, 1]},
    )

    # --- Left: dot plot ---
    y_positions = list(range(len(sorted_families)))
    labels = []
    accs = []
    colors = []

    for i, trial in enumerate(sorted_families):
        family = trial["family"]
        acc = trial["all_acc"] * 100
        param_str = _param_summary(trial)
        label = f"{family}" + (f" ({param_str})" if param_str else "")
        labels.append(label)
        accs.append(acc)
        colors.append(method_color(family))

    # Horizontal reference line for each method (lollipop stem)
    for i, (acc, color) in enumerate(zip(accs, colors)):
        ax_main.plot(
            [joint_baseline_acc, acc], [i, i],
            color=GRAY90, linewidth=0.8, zorder=1,
        )
        ax_main.scatter(
            acc, i,
            color=color, s=80, edgecolors=BLACK, linewidths=0.6,
            zorder=3,
        )
        # Gap annotation
        gap = joint_baseline_acc - acc
        ax_main.text(
            acc - 0.4, i + 0.15,
            f"−{gap:.1f}",
            fontsize=7, color=GRAY60, ha="right", va="bottom",
        )

    # Joint baseline
    ax_main.axvline(
        joint_baseline_acc, color=RED, linewidth=1.2, linestyle="-",
        zorder=2, alpha=0.8,
    )
    ax_main.text(
        joint_baseline_acc + 0.3, len(sorted_families) - 0.3,
        f"Joint\n{joint_baseline_acc:.1f}%",
        fontsize=8, color=RED, fontweight="bold",
        ha="left", va="top",
    )

    ax_main.set_yticks(y_positions)
    ax_main.set_yticklabels(labels, fontsize=9)
    ax_main.set_xlabel("All-Group Accuracy (%)")
    ax_main.invert_yaxis()
    clean_axes(ax_main, "lb")

    # --- Right: per-group breakdown for top 4 ---
    top_n = min(4, len(sorted_families))
    group_names = ALL_GROUPS
    x = np.arange(len(group_names))

    for i, trial in enumerate(sorted_families[:top_n]):
        per_group = trial.get("per_group", trial.get("per_group_accuracy", {}))
        vals = _extract_per_group_vals(per_group, group_names)
        offset = (i - top_n / 2 + 0.5) * (0.7 / top_n)
        ax_group.bar(
            x + offset, vals, 0.7 / top_n,
            label=trial["family"],
            color=method_color(trial["family"]),
            edgecolor="white", linewidth=0.3, alpha=0.85,
        )

    ax_group.set_xticks(x)
    ax_group.set_xticklabels([GROUP_SHORT_INLINE[g] for g in group_names], fontsize=9)
    ax_group.set_ylabel("Accuracy (%)")
    ax_group.legend(fontsize=7, ncol=1, loc="lower right")
    clean_axes(ax_group)

    plt.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved composition dots → %s", output_path)


# ============================================================================
# Fig 3: Cosine Similarity Matrix — Combined Heatmap
# ============================================================================

def plot_cosine_matrix(
    sparsity_data: dict,
    output_path: str,
) -> None:
    """Single combined heatmap: lower triangle = cosine, upper = sign agreement.

    Key change from v1: one panel instead of two. Diagonal shows group
    L2 norms as bar-in-cell. Taxonomy boundary as subtle shading.
    """
    pairwise = sparsity_data.get("pairwise", sparsity_data.get("pairwise_analysis", []))
    per_group_data = sparsity_data.get("per_group", sparsity_data.get("groups", {}))
    if not pairwise:
        return

    all_groups: set[str] = set()
    for entry in pairwise:
        pair = entry.get("pair", "")
        if "_vs_" in pair:
            a, b = pair.split("_vs_", 1)
            all_groups.add(a); all_groups.add(b)

    groups = sorted(all_groups)
    n = len(groups)

    cosine_mat = np.full((n, n), np.nan)
    sign_mat = np.full((n, n), np.nan)

    for entry in pairwise:
        pair = entry.get("pair", "")
        if "_vs_" not in pair: continue
        a, b = pair.split("_vs_", 1)
        if a in groups and b in groups:
            i, j = groups.index(a), groups.index(b)
            cos = entry.get("cosine_similarity", 0.0)
            sign = entry.get("sign_agreement", 0.5)
            cosine_mat[i, j] = cosine_mat[j, i] = cos
            sign_mat[i, j] = sign_mat[j, i] = sign

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 1.8, COLUMN_WIDTH + 1.5))

    # Draw cells manually for split upper/lower
    for i in range(n):
        for j in range(n):
            if i == j:
                # Diagonal: group color block
                color = GROUP_COLORS.get(groups[i], GRAY80)
                rect = plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor=color, alpha=0.2, edgecolor=GRAY90, linewidth=0.5,
                )
                ax.add_patch(rect)
                # Show L2 norm on diagonal
                if per_group_data and groups[i] in per_group_data:
                    norm_val = per_group_data[groups[i]].get(
                        "l2_norm", per_group_data[groups[i]].get("global_l2_norm", 0))
                    ax.text(j, i, f"‖τ‖={norm_val:.1f}",
                            ha="center", va="center", fontsize=7,
                            color=GRAY60, fontstyle="italic")
            elif i > j:
                # Lower triangle: cosine similarity
                val = cosine_mat[i, j]
                if np.isnan(val): continue
                # Color intensity based on cosine value
                intensity = min(1.0, val / 0.10)  # saturate at 0.10
                cell_color = plt.cm.YlOrRd(intensity * 0.7 + 0.1)
                rect = plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor=cell_color, edgecolor=GRAY90, linewidth=0.5,
                )
                ax.add_patch(rect)
                ax.text(j, i, f"{val:.3f}",
                        ha="center", va="center", fontsize=9.5,
                        fontweight="bold" if val > 0.05 else "normal",
                        color=BLACK if val < 0.07 else "white")
            else:
                # Upper triangle: sign agreement
                val = sign_mat[i, j]
                if np.isnan(val): continue
                # Color relative to chance (0.50)
                deviation = (val - 0.50) / 0.05  # normalize
                cell_color = plt.cm.Blues(min(1.0, max(0.1, deviation * 0.5 + 0.3)))
                rect = plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor=cell_color, edgecolor=GRAY90, linewidth=0.5,
                )
                ax.add_patch(rect)
                ax.text(j, i, f"{val:.3f}",
                        ha="center", va="center", fontsize=9.5,
                        color=BLACK)

    # Bird taxonomy boundary
    bird_indices = [groups.index(g) for g in groups if g in TAXA_BIRD]
    if bird_indices:
        mn, mx = min(bird_indices), max(bird_indices)
        rect = plt.Rectangle(
            (mn - 0.5, mn - 0.5), mx - mn + 1, mx - mn + 1,
            fill=False, edgecolor=TEAL, linewidth=2.0, linestyle="--",
            zorder=5,
        )
        ax.add_patch(rect)
        ax.text(mn - 0.4, mn - 0.65, "Birds", fontsize=8,
                color=TEAL, fontweight="bold")

    short_labels = [GROUP_SHORT_INLINE.get(g, g) for g in groups]
    ax.set_xticks(range(n))
    ax.set_xticklabels(short_labels, rotation=0, ha="center", fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_labels, fontsize=10)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    # Triangle labels
    ax.text(-0.9, n * 0.45, "Cosine Similarity", fontsize=8, rotation=90,
            ha="center", va="center", color=GRAY60, fontstyle="italic")
    ax.text(n * 0.4, -0.6, "Sign Agreement", fontsize=8,
            ha="center", va="center", color=GRAY60, fontstyle="italic")

    # Remove default spines — the grid serves as frame
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    plt.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved cosine matrix → %s", output_path)


# ============================================================================
# Fig 4a: Spectral Distance vs Cosine — with marginals
# ============================================================================

def plot_spectral_vs_cosine(
    pairs: list[dict],
    output_path: str,
    distance_metric: str = "jsd",
) -> None:
    """Scatter with marginal histograms and Spearman annotation.

    Key changes from v1: joint plot layout with marginal distributions,
    prominent Spearman rho, adjustable text to avoid overlap,
    convex hulls per pair type.
    """
    try:
        import scipy.stats
    except ImportError:
        scipy_stats = None
    else:
        scipy_stats = scipy.stats

    COLORS = {
        "bird_bird":   BLUE,
        "bird_other":  ORANGE,
        "other_other": PINK,
    }
    MARKERS = {"hydrophone": "^", "standard": "o"}
    metric_labels = {
        "jsd":     "Jensen–Shannon Divergence",
        "l2":      "L2 Distance (log-mel)",
        "l2_norm": "L2 Distance (mean-centered)",
    }

    # Layout: main scatter + top marginal + right marginal
    fig = plt.figure(figsize=(COLUMN_WIDTH + 2.0, COLUMN_WIDTH + 1.5))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
        hspace=0.05, wspace=0.05,
    )
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    xs, ys = [], []
    for p in pairs:
        x_val = p[f"distance_{distance_metric}"]
        y_val = p["cosine_similarity"]
        xs.append(x_val); ys.append(y_val)

        color = COLORS[p["pair_type"]]
        marker = MARKERS["hydrophone"] if p["has_hydrophone"] else MARKERS["standard"]

        ax_main.scatter(
            x_val, y_val,
            c=color, marker=marker,
            s=90, edgecolors=BLACK, linewidths=0.5, zorder=3,
        )

    # Smart label placement (offset based on position)
    texts = []
    for p in pairs:
        x_val = p[f"distance_{distance_metric}"]
        y_val = p["cosine_similarity"]
        label = p["short_label"]
        # Offset: push labels away from cluster center
        x_center = np.mean(xs)
        y_center = np.mean(ys)
        dx = 7 if x_val > x_center else -7
        dy = 5 if y_val > y_center else -5
        ax_main.annotate(
            label, (x_val, y_val),
            fontsize=8, xytext=(dx, dy), textcoords="offset points",
            arrowprops=dict(arrowstyle="-", color=GRAY80, lw=0.4),
            color=GRAY60,
        )

    # Regression line
    xs_arr, ys_arr = np.array(xs), np.array(ys)
    if len(xs_arr) > 2:
        m, b = np.polyfit(xs_arr, ys_arr, 1)
        x_line = np.linspace(xs_arr.min() * 0.9, xs_arr.max() * 1.1, 100)
        ax_main.plot(x_line, m * x_line + b, color=GRAY80, linewidth=1.0,
                     linestyle="--", alpha=0.7, zorder=1)

    # Spearman annotation — prominent
    if scipy_stats is not None:
        rho, p_val = scipy_stats.spearmanr(xs_arr, ys_arr)
        sig_str = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else
                  ("*" if p_val < 0.05 else "n.s."))
        ax_main.text(
            0.03, 0.97,
            f"ρ = {rho:.3f}{sig_str}",
            transform=ax_main.transAxes, fontsize=11, fontweight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=GRAY90, linewidth=0.5, alpha=0.95),
        )

    # Marginal distributions
    for pair_type, color in COLORS.items():
        type_xs = [p[f"distance_{distance_metric}"] for p in pairs
                   if p["pair_type"] == pair_type]
        type_ys = [p["cosine_similarity"] for p in pairs
                   if p["pair_type"] == pair_type]
        if type_xs:
            ax_top.scatter(type_xs, [0.5] * len(type_xs), color=color,
                           s=40, alpha=0.7, edgecolors="none", zorder=3)
        if type_ys:
            ax_right.scatter([0.5] * len(type_ys), type_ys, color=color,
                             s=40, alpha=0.7, edgecolors="none", zorder=3)

    ax_top.set_ylim(0, 1)
    ax_top.axis("off")
    ax_right.set_xlim(0, 1)
    ax_right.axis("off")
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)

    ax_main.set_xlabel(metric_labels.get(distance_metric, distance_metric))
    ax_main.set_ylabel("Task Vector Cosine Similarity")
    ax_main.grid(True, linewidth=0.2, alpha=0.3)
    clean_axes(ax_main)

    # Legend
    legend_elements = [
        Patch(facecolor=COLORS["bird_bird"], edgecolor=BLACK,
              linewidth=0.5, label="Bird–Bird"),
        Patch(facecolor=COLORS["bird_other"], edgecolor=BLACK,
              linewidth=0.5, label="Bird–Other"),
        Patch(facecolor=COLORS["other_other"], edgecolor=BLACK,
              linewidth=0.5, label="Other–Other"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=GRAY80,
               markeredgecolor=BLACK, markersize=7, linewidth=0,
               label="Hydrophone (G4)"),
    ]
    ax_main.legend(handles=legend_elements, loc="lower right", fontsize=7.5,
                   framealpha=0.9, edgecolor=GRAY90)

    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved spectral vs cosine → %s", output_path)


# ============================================================================
# Fig 4b: Asymmetric Gap — Butterfly Diverging Chart
# ============================================================================

def plot_asymmetric_gap(
    gap_data: dict,
    output_path: str,
) -> None:
    """Diverging horizontal bar chart with butterfly wings.

    Key change from v1: horizontal layout, diverging from zero,
    colored by whether merging helps (green) or hurts (red).
    Groups ordered by gap magnitude for visual impact.
    """
    per_group = gap_data.get("per_group", gap_data.get("per_group_gaps", {}))
    if not per_group:
        return

    group_order = [g for g in ALL_GROUPS if g in per_group]

    gaps = []
    ci_low_err = []
    ci_high_err = []
    colors = []
    edge_colors = []

    for g in group_order:
        gdata = per_group[g]
        gap = gdata.get("gap_pp", gdata.get("gap", 0))
        if isinstance(gap, float) and abs(gap) < 1:
            gap *= 100
        gaps.append(gap)

        ci = gdata.get("ci_95", gdata.get("bootstrap_ci",
                 [gdata.get("ci_low", gap), gdata.get("ci_high", gap)]))
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            lo, hi = ci[0], ci[1]
            if abs(lo) < 1: lo *= 100; hi *= 100
            ci_low_err.append(abs(gap - lo))
            ci_high_err.append(abs(hi - gap))
        else:
            ci_low_err.append(0); ci_high_err.append(0)

        # Color by direction: red if joint better, teal if merging better
        if gap > 0:
            colors.append("#FDDBC7")  # light red fill
            edge_colors.append(RED)
        else:
            colors.append("#D5E8D4")  # light green fill
            edge_colors.append(TEAL)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 2.0, 2.8))

    y = np.arange(len(group_order))
    bar_height = 0.55

    bars = ax.barh(
        y, gaps, bar_height,
        color=colors, edgecolor=edge_colors, linewidth=1.2,
        zorder=3,
    )

    # Error bars
    if any(c > 0 for c in ci_high_err):
        ax.errorbar(
            gaps, y, xerr=[ci_low_err, ci_high_err],
            fmt="none", ecolor=GRAY60, elinewidth=0.8, capsize=3, zorder=4,
        )

    # Annotations
    for i, (gap_val, bar) in enumerate(zip(gaps, bars)):
        sign = "+" if gap_val > 0 else ""
        x_pos = gap_val + (ci_high_err[i] if gap_val > 0 else -ci_low_err[i])
        offset = 0.3 if gap_val > 0 else -0.3
        ha = "left" if gap_val > 0 else "right"
        ax.text(
            x_pos + offset, i,
            f"{sign}{gap_val:.1f}%",
            ha=ha, va="center", fontsize=9, fontweight="bold",
            color=RED if gap_val > 0 else TEAL,
        )

    # Zero line
    ax.axvline(0, color=BLACK, linewidth=0.8, zorder=2)

    # Direction labels
    xlim = ax.get_xlim()
    ax.text(xlim[1] * 0.6, -0.7, "← Joint training better",
            fontsize=8, color=RED, alpha=0.7, ha="center")
    ax.text(xlim[0] * 0.6, -0.7, "Merging better →",
            fontsize=8, color=TEAL, alpha=0.7, ha="center")

    # Overall mean
    overall = gap_data.get("overall", gap_data.get("all_group",
                  gap_data.get("all_group_gap", {})))
    mean_gap = overall.get("gap_pp", overall.get("gap", None))
    if mean_gap is not None:
        if isinstance(mean_gap, float) and abs(mean_gap) < 1:
            mean_gap *= 100
        ax.axvline(mean_gap, color=RED, linewidth=1.0, linestyle=":",
                    zorder=2, alpha=0.7)
        ax.text(mean_gap+.2, len(group_order) - 0.9,
                f"Mean: +{mean_gap:.1f}%",
                fontsize=7.5, color=RED, ha="left", va="top",
                fontstyle="italic")

    ax.set_yticks(y)
    ax.set_yticklabels(
        [GROUP_SHORT_INLINE.get(g, g) for g in group_order],
        fontsize=10,
    )
    ax.set_xlabel("Accuracy Gap: Joint − Merged (%)")

    # Add group color dots next to labels
    for i, g in enumerate(group_order):
        ax.scatter(-0.15, i, color=GROUP_COLORS[g], s=60,
                   edgecolors=BLACK, linewidths=0.4, zorder=5,
                   transform=ax.get_yaxis_transform(), clip_on=False)

    clean_axes(ax, "lb")
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved asymmetric gap → %s", output_path)


# ============================================================================
# Fig 5: Domain Negation — Shaded Divergence
# ============================================================================

def plot_domain_negation(
    negation_data: dict,
    output_path: str,
) -> None:
    """Domain negation with shaded area between real and random control.

    Key change from v1: the gap between focal negation and random control
    is filled, making the signal visually undeniable. Single panel per source.
    """
    sweeps = negation_data.get("sweeps", negation_data.get("results", []))
    if not sweeps and "trials" in negation_data:
        trials = negation_data["trials"]
        sweep_map: dict[tuple[str, str], dict] = {}
        for t in trials:
            key = (t.get("source_model", "unknown"), t.get("vector_type", "focal"))
            if key not in sweep_map:
                sweep_map[key] = {
                    "source_model": key[0], "vector_type": key[1],
                    "betas": [], "focal_accuracies": [],
                    "soundscape_accuracies": [],
                }
            sweep_map[key]["betas"].append(t["beta"])
            sweep_map[key]["focal_accuracies"].append(t.get("focal_accuracy", 0))
            sweep_map[key]["soundscape_accuracies"].append(t.get("soundscape_accuracy", 0))
        sweeps = list(sweep_map.values())

    if not sweeps:
        return

    source_groups: dict[str, list[dict]] = {}
    for sweep in sweeps:
        source = sweep.get("source_model", "unknown")
        source_groups.setdefault(source, []).append(sweep)

    n_sources = len(source_groups)
    fig, axes = plt.subplots(1, n_sources, figsize=(FULL_WIDTH, 3.2),
                              squeeze=False, sharey=True)

    source_titles = {
        "mixed": "Mixed-domain source",
        "soundscape": "Soundscape source",
    }

    for s_idx, (source_name, source_sweeps) in enumerate(sorted(source_groups.items())):
        ax = axes[0][s_idx]

        focal_sweep = None
        control_sweep = None
        for sw in source_sweeps:
            vtype = sw.get("vector_type", "")
            if "random" in vtype.lower() or "control" in vtype.lower():
                control_sweep = sw
            else:
                focal_sweep = sw

        # Plot focal negation (solid, thick)
        if focal_sweep:
            betas = focal_sweep["betas"]
            focal_accs = [a * 100 if a <= 1 else a for a in focal_sweep["focal_accuracies"]]
            sound_accs = [a * 100 if a <= 1 else a for a in focal_sweep.get("soundscape_accuracies", [])]

            ax.plot(betas, focal_accs, "o-", color=BLUE, linewidth=2.0,
                    markersize=5, label="Focal acc (negation)", zorder=3)
            if sound_accs:
                ax.plot(betas, sound_accs, "s-", color=ORANGE, linewidth=2.0,
                        markersize=5, label="Soundscape acc (negation)", zorder=3)

        # Plot random control (dashed, thin)
        if control_sweep:
            betas_c = control_sweep["betas"]
            focal_c = [a * 100 if a <= 1 else a for a in control_sweep["focal_accuracies"]]
            sound_c = [a * 100 if a <= 1 else a for a in control_sweep.get("soundscape_accuracies", [])]

            ax.plot(betas_c, focal_c, "o--", color=BLUE, linewidth=1.0,
                    markersize=4, alpha=0.5, label="Focal (random ctrl)", zorder=2)
            if sound_c:
                ax.plot(betas_c, sound_c, "s--", color=ORANGE, linewidth=1.0,
                        markersize=4, alpha=0.5, label="Soundscape (random ctrl)", zorder=2)

            # Shade the gap between real and control
            if focal_sweep and len(focal_accs) == len(focal_c):
                ax.fill_between(
                    betas, focal_accs, focal_c,
                    alpha=0.12, color=BLUE, zorder=1,
                    label="Negation effect",
                )

        # Source label
        title = source_titles.get(source_name, source_name)
        ax.text(
            0.03, 0.7, title,
            transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=GRAY90, linewidth=0.5, alpha=0.95),
        )

        ax.set_xlabel("Negation strength (β)")
        if s_idx == 0:
            ax.set_ylabel("Accuracy (%)")
        ax.legend(fontsize=6.5, loc="lower left", framealpha=0.9)
        clean_axes(ax)

    plt.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved domain negation → %s", output_path)


# ============================================================================
# Fig H: Forgetting Dynamics — Clean Single-LR
# ============================================================================

def plot_forgetting_dynamics(
    results: dict,
    output_path: str,
) -> None:
    """Forgetting dynamics for best LR only, with task arithmetic reference band.

    Key change from v1: single LR (clearest signal), shaded region
    between initial and final accuracy, task arithmetic as horizontal band.
    """
    ft_results = results.get("finetune_per_epoch", [])
    if not ft_results:
        return

    # Pick LR with most epochs (likely the best)
    lr_result = max(ft_results, key=lambda r: len(r.get("epochs", [])))
    epochs_data = lr_result["epochs"]
    lr_val = lr_result["learning_rate"]

    if not epochs_data:
        return

    epochs = [e["epoch"] + 1 for e in epochs_data]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 2.2, 3.5))

    for group in ALL_GROUPS:
        if group not in epochs_data[0].get("per_group", {}):
            continue
        accs = [e["per_group"][group] * 100 for e in epochs_data]
        color = GROUP_COLORS.get(group, GRAY80)
        label = GROUP_SHORT_INLINE.get(group, group)
        is_new = group in TAXA_OTHER
        linestyle = "--" if is_new else "-"
        marker = "s" if is_new else "o"

        ax.plot(epochs, accs, linestyle, color=color, label=label,
                linewidth=1.8, marker=marker, markersize=3,
                markevery=max(1, len(epochs) // 8))

        # Start/end annotations for bird groups
        if group in TAXA_BIRD and len(accs) > 1:
            delta = accs[-1] - accs[0]
            if abs(delta) > 0.5:
                ax.annotate(
                    f"{delta:+.1f}%",
                    xy=(epochs[-1], accs[-1]),
                    xytext=(5, 0), textcoords="offset points",
                    fontsize=7, color=color, fontstyle="italic",
                )

    # Old-group mean
    if "old_group_mean" in epochs_data[0]:
        old_means = [e["old_group_mean"] * 100 for e in epochs_data]
        ax.plot(epochs, old_means, color=BLACK, linewidth=2.0,
                alpha=0.4, label="G1–G4 mean", zorder=5)

        # Shade the forgetting region
        ax.fill_between(
            epochs, old_means[0], old_means,
            alpha=0.06, color=RED, zorder=0,
        )

    # Task arithmetic reference
    merge_method = None
    for m in results.get("methods", []):
        if m["method"] in ("merge_joint_tv", "merge_individual_tvs"):
            merge_method = m
            break

    if merge_method:
        old_merge = merge_method["old_group_mean_accuracy"] * 100
        ax.axhspan(
            old_merge - 0.3, old_merge + 0.3,
            alpha=0.15, color=TEAL, zorder=0,
        )
        ax.axhline(old_merge, color=TEAL, linewidth=1.0, linestyle=":",
                    zorder=2)
        ax.text(
            1, old_merge + 0.6,
            f"Task arith.: {old_merge:.1f}%",
            fontsize=7.5, color=TEAL, fontweight="bold",
        )

    ax.set_xlabel("Fine-tuning Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.text(
        0.98, 0.02, f"LR = {lr_val:.0e}",
        transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
        color=GRAY60,
    )

    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False)
    clean_axes(ax)

    plt.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved forgetting dynamics → %s", output_path)


# ============================================================================
# NEW: Radar Chart — Joint vs Merged Per-Group
# ============================================================================

def plot_radar_comparison(
    gap_data: dict,
    output_path: str,
) -> None:
    """Radar (spider) chart: joint vs merged accuracy per group.

    This figure makes the asymmetric gap immediately intuitive:
    the merged model's polygon is smaller for birds but larger
    for marine mammals and amphibians.
    """
    per_group = gap_data.get("per_group", gap_data.get("per_group_gaps", {}))
    if not per_group:
        return

    groups = [g for g in ALL_GROUPS if g in per_group]
    n = len(groups)

    joint_vals = []
    merged_vals = []
    for g in groups:
        gdata = per_group[g]
        j = gdata.get("joint_accuracy", gdata.get("joint", 0))
        m = gdata.get("merged_accuracy", gdata.get("merged", 0))
        if j <= 1: j *= 100
        if m <= 1: m *= 100
        joint_vals.append(j)
        merged_vals.append(m)

    if not any(j > 0 for j in joint_vals):
        logger.warning("No joint/merged accuracy data in gap_data. Skipping radar.")
        return

    # Radar setup
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    joint_vals_r = joint_vals + [joint_vals[0]]
    merged_vals_r = merged_vals + [merged_vals[0]]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 1.0, COLUMN_WIDTH + 0.8),
                           subplot_kw=dict(polar=True))

    # Joint (outer, red-ish)
    ax.fill(angles, joint_vals_r, alpha=0.08, color=RED)
    ax.plot(angles, joint_vals_r, "o-", color=RED, linewidth=1.8,
            markersize=5, label="Joint training")

    # Merged (inner/outer, teal)
    ax.fill(angles, merged_vals_r, alpha=0.08, color=BLUE)
    ax.plot(angles, merged_vals_r, "s-", color=BLUE, linewidth=1.8,
            markersize=5, label="Merged (1/N avg)")

    # Highlight where merged > joint
    for i in range(n):
        if merged_vals[i] > joint_vals[i]:
            ax.annotate(
                "↑", (angles[i], merged_vals[i] + 2),
                ha="center", fontsize=10, color=TEAL, fontweight="bold",
            )

    # Labels
    labels = [GROUP_SHORT_INLINE.get(g, g) for g in groups]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)

    # Value annotations at each vertex
    for i in range(n):
        offset = 6
        ax.text(angles[i], joint_vals[i] + offset,
                f"{joint_vals[i]:.0f}", fontsize=7,
                ha="center", color=RED, fontweight="bold")
        ax.text(angles[i], merged_vals[i] - offset,
                f"{merged_vals[i]:.0f}", fontsize=7,
                ha="center", color=BLUE, fontweight="bold")

    ax.set_ylim(40, 100)
    ax.set_rgrids([50, 60, 70, 80, 90], fontsize=7, alpha=0.5)
    ax.legend(loc="lower right", bbox_to_anchor=(1.2, -0.05), fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved radar comparison → %s", output_path)


# ============================================================================
# NEW: Method Complexity vs Accuracy
# ============================================================================

def plot_method_complexity(
    comp_data: dict,
    output_path: str,
    joint_baseline_acc: float = 68.48,
) -> None:
    """Scatter: method complexity (number of hyperparameters) vs accuracy.

    Key insight: more complex methods (TIES, DARE, DELLA) don't outperform
    simple averaging in the near-orthogonal regime. Complexity is wasted.
    """
    trials = comp_data.get("trials", comp_data.get("results", []))
    if not trials:
        return

    # Define complexity score per method family
    COMPLEXITY = {
        "simple_average":  1,
        "task_arithmetic": 2,
        "dare_average":    3,
        "dare_ties":       4,
        "ties":            3,
        "della_ties":      4,
    }

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 1.5, 3.2))

    for t in trials:
        method = t.get("method", "")
        acc = t.get("all_acc", 0) * 100
        complexity = COMPLEXITY.get(method, 2)
        color = method_color(method)

        jitter = np.random.default_rng(abs(hash(str(t))) % (2**31)).uniform(-0.15, 0.15)
        ax.scatter(
            complexity + jitter, acc,
            c=color, s=50, edgecolors=BLACK, linewidths=0.4,
            alpha=0.7, zorder=3,
        )

    # Joint baseline
    ax.axhline(joint_baseline_acc, color=RED, linewidth=1.0, linestyle="--",
               alpha=0.5, label=f"Joint baseline ({joint_baseline_acc:.1f}%)")

    # Best accuracy reference
    best_acc = max(t.get("all_acc", 0) * 100 for t in trials)
    ax.axhline(best_acc, color=TEAL, linewidth=0.8, linestyle=":",
               alpha=0.5, label=f"Best merged ({best_acc:.1f}%)")

    # Shade the "no improvement" zone
    ax.axhspan(best_acc - 1.0, best_acc + 1.0, alpha=0.05, color=TEAL)

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["Simple\nAverage", "Task\nArith.", "DARE/\nTIES", "DARE+TIES/\nDELLA"],
                       fontsize=8)
    ax.set_xlabel("Method Complexity →")
    ax.set_ylabel("All-Group Accuracy (%)")
    ax.legend(fontsize=7.5, loc="upper right")

    # Annotation
    ax.text(
        0.5, 0.02,
        "Near-orthogonal task vectors → sign conflicts are random → complexity is wasted",
        transform=ax.transAxes, fontsize=7, ha="center", va="bottom",
        color=GRAY60, fontstyle="italic",
    )

    clean_axes(ax)
    plt.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved method complexity → %s", output_path)


# ============================================================================
# NEW: Norm-Adjusted vs Uniform — Paired Slope Chart
# ============================================================================

def plot_norm_comparison_slopes(
    norm_data: dict,
    output_path: str,
) -> None:
    """Slope chart (slopegraph) comparing uniform vs norm-adjusted at each λ.

    Key change from v1: slope chart instead of bars. Each line connects
    uniform and norm-adjusted accuracy at the same λ. Uniform consistently
    above → the pattern is immediately clear.
    """
    trials = norm_data.get("trials", norm_data.get("results", []))
    if not trials:
        return

    # Group by lambda
    lambda_pairs: dict[float, dict[str, float]] = {}
    for t in trials:
        lam = t.get("lambda", t.get("scaling", 0))
        strategy = t.get("weighting", t.get("strategy", "unknown"))
        acc = t.get("all_acc", t.get("unified_accuracy", 0)) * 100
        lambda_pairs.setdefault(lam, {})[strategy] = acc

    lambdas = sorted(lambda_pairs.keys())
    if not lambdas:
        return

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 1.5, 3.2))

    for i, lam in enumerate(lambdas):
        pair = lambda_pairs[lam]
        u_acc = pair.get("uniform", None)
        n_acc = pair.get("norm_adjusted", None)
        if u_acc is None or n_acc is None:
            continue

        # Connect with line
        line_color = TEAL if u_acc > n_acc else RED
        ax.plot([0, 1], [u_acc, n_acc], "-", color=line_color,
                linewidth=1.2, alpha=0.6, zorder=2)

        # Points
        ax.scatter(0, u_acc, color=BLUE, s=60, edgecolors=BLACK,
                   linewidths=0.4, zorder=3)
        ax.scatter(1, n_acc, color=ORANGE, s=60, edgecolors=BLACK,
                   linewidths=0.4, zorder=3)

        # Lambda label
        ax.text(-0.08, u_acc, f"λ={lam}", fontsize=7, ha="right",
                va="center", color=GRAY60)

        # Gap annotation
        gap = u_acc - n_acc
        mid_y = (u_acc + n_acc) / 2
        ax.text(0.5, mid_y, f"+{gap:.1f}" if gap > 0 else f"{gap:.1f}",
                fontsize=7, ha="center", va="center", color=line_color,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.8))

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Uniform\nWeighting", "Norm-Adjusted\nWeighting"], fontsize=10)
    ax.set_ylabel("All-Group Accuracy (%)")
    ax.set_xlim(-0.3, 1.3)

    # Summary annotation
    ax.text(
        0.5, 0.02,
        "Uniform wins at every λ → norm adjustment is counterproductive",
        transform=ax.transAxes, fontsize=7.5, ha="center", va="bottom",
        color=GRAY60, fontstyle="italic",
    )

    clean_axes(ax, "lb")
    plt.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved norm comparison slopes → %s", output_path)


# ============================================================================
# NEW: Data Efficiency — Clean Two-Panel
# ============================================================================

def plot_data_efficiency(
    results: dict,
    output_path: str,
) -> None:
    """Data efficiency with clear panel separation.

    Left: accuracy vs data fraction. Right: TV cosine convergence.
    Key insight: G4 merged accuracy is flat — benefit comes from others.
    """
    fracs = results.get("fractions", [])
    if not fracs:
        return

    x = [fr["fraction"] * 100 for fr in fracs]
    g4_standalone = [fr["g4_standalone_accuracy"] * 100 for fr in fracs]
    g4_merged = [fr["g4_merged_accuracy"] * 100 for fr in fracs]
    cosines = [fr["tv_cosine_with_full"] for fr in fracs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.0),
                                    gridspec_kw={"width_ratios": [1.3, 1]})

    # Left: accuracy
    ax1.plot(x, g4_standalone, "D-", color=PINK, linewidth=1.8,
             markersize=6, label="G4 standalone")
    ax1.plot(x, g4_merged, "o-", color=TEAL, linewidth=1.8,
             markersize=6, label="G4 after merging")

    # Shade the "merging benefit" zone
    ax1.fill_between(x, g4_standalone, g4_merged,
                     where=[m > s for m, s in zip(g4_merged, g4_standalone)],
                     alpha=0.1, color=TEAL, label="Merging benefit")

    # Flat line annotation
    ax1.annotate(
        "Flat: benefit comes\nfrom other groups",
        xy=(50, g4_merged[1]),
        xytext=(25, g4_merged[1] - 5),
        fontsize=7, color=TEAL, fontstyle="italic",
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=0.8),
    )

    ax1.set_xlabel("G4 Training Data (%)")
    ax1.set_ylabel("G4 Accuracy (%)")
    ax1.legend(fontsize=7.5)
    clean_axes(ax1)

    # Right: cosine convergence
    ax2.plot(x, cosines, "s-", color=RED, linewidth=1.8, markersize=6)
    ax2.axhline(0.9, color=GRAY80, linewidth=0.5, linestyle=":")
    ax2.text(105, 0.91, "90%", fontsize=7, color=GRAY60)

    ax2.set_xlabel("G4 Training Data (%)")
    ax2.set_ylabel("Cosine with Full TV")
    ax2.set_ylim(-0.05, 1.1)
    clean_axes(ax2)

    # Sharp transition annotation
    ax2.annotate(
        "Sharp convergence\nat 50%",
        xy=(50, cosines[1]),
        xytext=(65, 0.4),
        fontsize=7, color=RED,
        arrowprops=dict(arrowstyle="->", color=RED, lw=0.8),
    )

    plt.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved data efficiency → %s", output_path)


# ============================================================================
# Helpers
# ============================================================================

def _method_family(method_name: str) -> str:
    """Map method + params to a family label."""
    n = method_name.lower()
    if "della" in n: return "DELLA"
    if "dare_ties" in n: return "DARE+TIES"
    if "dare" in n: return "DARE"
    if "ties" in n: return "TIES"
    if "simple_average" in n: return "Simple Avg"
    if "task_arithmetic" in n: return "Task Arith."
    return method_name


def _param_summary(trial: dict) -> str:
    """Extract key param as compact string."""
    params = trial.get("params", {})
    if isinstance(params, dict):
        if "lambda" in params: return f"λ={params['lambda']}"
        if "drop_rate" in params: return f"dr={params['drop_rate']}"
        if "trim_fraction" in params: return f"tf={params['trim_fraction']}"
    return ""


def _extract_per_group_vals(per_group: dict, group_names: list[str]) -> list[float]:
    """Extract per-group accuracy values."""
    result = []
    for g in group_names:
        acc = per_group.get(g, per_group.get(GROUP_SHORT_INLINE.get(g, ""), 0))
        if isinstance(acc, dict):
            acc = acc.get("accuracy", acc.get("acc", 0))
        result.append(acc * 100 if isinstance(acc, float) and acc <= 1.0 else acc)
    return result


# ============================================================================
# Demo: Generate figures from hardcoded data (for preview)
# ============================================================================

def generate_demo_figures(output_dir: str) -> None:
    """Generate figures using the actual experimental data from the paper."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Cosine matrix from actual data ---
    sparsity_data = {
        "per_group": {
            "G1_passerines": {"l2_norm": 13.58, "mean_magnitude": 1.05e-3},
            "G2_nonpasserine_birds": {"l2_norm": 9.51, "mean_magnitude": 7.46e-4},
            "G3_raptors_waterbirds": {"l2_norm": 6.86, "mean_magnitude": 5.43e-4},
            "G4_marine_mammals": {"l2_norm": 1.45, "mean_magnitude": 1.22e-4},
            "G5_amphibians": {"l2_norm": 4.79, "mean_magnitude": 3.84e-4},
        },
        "pairwise": [
            {"pair": "G1_passerines_vs_G2_nonpasserine_birds",
             "cosine_similarity": 0.0916, "sign_agreement": 0.5386},
            {"pair": "G1_passerines_vs_G3_raptors_waterbirds",
             "cosine_similarity": 0.0851, "sign_agreement": 0.5352},
            {"pair": "G1_passerines_vs_G4_marine_mammals",
             "cosine_similarity": 0.0139, "sign_agreement": 0.5064},
            {"pair": "G1_passerines_vs_G5_amphibians",
             "cosine_similarity": 0.0293, "sign_agreement": 0.5113},
            {"pair": "G2_nonpasserine_birds_vs_G3_raptors_waterbirds",
             "cosine_similarity": 0.0929, "sign_agreement": 0.5368},
            {"pair": "G2_nonpasserine_birds_vs_G4_marine_mammals",
             "cosine_similarity": 0.0133, "sign_agreement": 0.5055},
            {"pair": "G2_nonpasserine_birds_vs_G5_amphibians",
             "cosine_similarity": 0.0380, "sign_agreement": 0.5147},
            {"pair": "G3_raptors_waterbirds_vs_G4_marine_mammals",
             "cosine_similarity": 0.0215, "sign_agreement": 0.5104},
            {"pair": "G3_raptors_waterbirds_vs_G5_amphibians",
             "cosine_similarity": 0.0391, "sign_agreement": 0.5145},
            {"pair": "G4_marine_mammals_vs_G5_amphibians",
             "cosine_similarity": 0.0223, "sign_agreement": 0.5091},
        ],
    }
    plot_cosine_matrix(sparsity_data, str(out / "cosine_matrix_v2.pdf"))

    # --- Asymmetric gap from actual data ---
    gap_data = {
        "per_group": {
            "G1_passerines": {
                "gap_pp": 11.8, "joint_accuracy": 69.92, "merged_accuracy": 58.09,
                "ci_95": [11.2, 12.4],
            },
            "G2_nonpasserine_birds": {
                "gap_pp": 7.9, "joint_accuracy": 77.60, "merged_accuracy": 69.72,
                "ci_95": [7.1, 8.6],
            },
            "G3_raptors_waterbirds": {
                "gap_pp": 7.0, "joint_accuracy": 75.70, "merged_accuracy": 68.66,
                "ci_95": [6.0, 8.1],
            },
            "G4_marine_mammals": {
                "gap_pp": -3.9, "joint_accuracy": 89.61, "merged_accuracy": 93.51,
                "ci_95": [-6.8, -1.3],
            },
            "G5_amphibians": {
                "gap_pp": -1.9, "joint_accuracy": 60.67, "merged_accuracy": 62.53,
                "ci_95": [-3.3, -0.4],
            },
        },
        "overall": {"gap_pp": 9.4},
    }
    plot_asymmetric_gap(gap_data, str(out / "asymmetric_gap_v2.pdf"))
    plot_radar_comparison(gap_data, str(out / "radar_comparison.pdf"))

    # --- Composition dots from actual data ---
    comp_data = {
        "trials": [
            {"method": "dare_average", "all_acc": 0.5917,
             "params": {"drop_rate": 0.9},
             "per_group": {"G1_passerines": 0.5801, "G2_nonpasserine_birds": 0.6966,
                           "G3_raptors_waterbirds": 0.6820, "G4_marine_mammals": 0.9351,
                           "G5_amphibians": 0.6227}},
            {"method": "task_arithmetic", "all_acc": 0.5900,
             "params": {"lambda": 1.0},
             "per_group": {"G1_passerines": 0.57, "G2_nonpasserine_birds": 0.69,
                           "G3_raptors_waterbirds": 0.68, "G4_marine_mammals": 0.93,
                           "G5_amphibians": 0.62}},
            {"method": "simple_average", "all_acc": 0.5882,
             "params": {},
             "per_group": {"G1_passerines": 0.57, "G2_nonpasserine_birds": 0.69,
                           "G3_raptors_waterbirds": 0.68, "G4_marine_mammals": 0.93,
                           "G5_amphibians": 0.62}},
            {"method": "dare_ties", "all_acc": 0.5789,
             "params": {"drop_rate": 0.9, "trim_fraction": 0.2},
             "per_group": {"G1_passerines": 0.56, "G2_nonpasserine_birds": 0.68,
                           "G3_raptors_waterbirds": 0.67, "G4_marine_mammals": 0.92,
                           "G5_amphibians": 0.61}},
            {"method": "della_ties", "all_acc": 0.5532,
             "params": {"target_drop_rate": 0.9, "trim_fraction": 0.2},
             "per_group": {"G1_passerines": 0.54, "G2_nonpasserine_birds": 0.66,
                           "G3_raptors_waterbirds": 0.65, "G4_marine_mammals": 0.91,
                           "G5_amphibians": 0.59}},
            {"method": "ties", "all_acc": 0.5304,
             "params": {"trim_fraction": 0.5},
             "per_group": {"G1_passerines": 0.52, "G2_nonpasserine_birds": 0.64,
                           "G3_raptors_waterbirds": 0.63, "G4_marine_mammals": 0.90,
                           "G5_amphibians": 0.57}},
        ],
    }
    plot_composition_dots(comp_data, str(out / "composition_dots_v2.pdf"))
    plot_method_complexity(comp_data, str(out / "method_complexity.pdf"))

    # --- Norm-adjusted comparison ---
    norm_data = {
        "trials": [
            {"weighting": "uniform", "lambda": 0.3, "all_acc": 0.5176},
            {"weighting": "norm_adjusted", "lambda": 0.3, "all_acc": 0.5009},
            {"weighting": "uniform", "lambda": 0.5, "all_acc": 0.5447},
            {"weighting": "norm_adjusted", "lambda": 0.5, "all_acc": 0.5162},
            {"weighting": "uniform", "lambda": 0.7, "all_acc": 0.5655},
            {"weighting": "norm_adjusted", "lambda": 0.7, "all_acc": 0.5251},
            {"weighting": "uniform", "lambda": 1.0, "all_acc": 0.5901},
            {"weighting": "norm_adjusted", "lambda": 1.0, "all_acc": 0.5452},
            {"weighting": "uniform", "lambda": 1.5, "all_acc": 0.6027},
            {"weighting": "norm_adjusted", "lambda": 1.5, "all_acc": 0.5647},
            {"weighting": "uniform", "lambda": 2.0, "all_acc": 0.5971},
            {"weighting": "norm_adjusted", "lambda": 2.0, "all_acc": 0.5706},
        ],
    }
    plot_norm_comparison_slopes(norm_data, str(out / "norm_comparison_v2.pdf"))

    # --- Domain negation ---
    negation_data = {
        "trials": [
            # Mixed source - focal negation
            {"source_model": "mixed", "vector_type": "focal", "beta": 0.0,
             "focal_accuracy": 0.7017, "soundscape_accuracy": 0.6946},
            {"source_model": "mixed", "vector_type": "focal", "beta": 0.25,
             "focal_accuracy": 0.6964, "soundscape_accuracy": 0.6959},
            {"source_model": "mixed", "vector_type": "focal", "beta": 0.5,
             "focal_accuracy": 0.6645, "soundscape_accuracy": 0.6471},
            {"source_model": "mixed", "vector_type": "focal", "beta": 0.75,
             "focal_accuracy": 0.6103, "soundscape_accuracy": 0.5919},
            {"source_model": "mixed", "vector_type": "focal", "beta": 1.0,
             "focal_accuracy": 0.5435, "soundscape_accuracy": 0.5100},
            # Mixed source - random control
            {"source_model": "mixed", "vector_type": "random_control", "beta": 0.0,
             "focal_accuracy": 0.7029, "soundscape_accuracy": 0.6990},
            {"source_model": "mixed", "vector_type": "random_control", "beta": 0.25,
             "focal_accuracy": 0.7051, "soundscape_accuracy": 0.6999},
            {"source_model": "mixed", "vector_type": "random_control", "beta": 0.5,
             "focal_accuracy": 0.7021, "soundscape_accuracy": 0.7020},
            {"source_model": "mixed", "vector_type": "random_control", "beta": 0.75,
             "focal_accuracy": 0.7033, "soundscape_accuracy": 0.6963},
            {"source_model": "mixed", "vector_type": "random_control", "beta": 1.0,
             "focal_accuracy": 0.7027, "soundscape_accuracy": 0.6936},
            # Soundscape source - focal negation
            {"source_model": "soundscape", "vector_type": "focal", "beta": 0.0,
             "focal_accuracy": 0.5643, "soundscape_accuracy": 0.6982},
            {"source_model": "soundscape", "vector_type": "focal", "beta": 0.25,
             "focal_accuracy": 0.5327, "soundscape_accuracy": 0.6839},
            {"source_model": "soundscape", "vector_type": "focal", "beta": 0.5,
             "focal_accuracy": 0.4928, "soundscape_accuracy": 0.6476},
            {"source_model": "soundscape", "vector_type": "focal", "beta": 0.75,
             "focal_accuracy": 0.4465, "soundscape_accuracy": 0.6013},
            {"source_model": "soundscape", "vector_type": "focal", "beta": 1.0,
             "focal_accuracy": 0.4056, "soundscape_accuracy": 0.5516},
            # Soundscape source - random control
            {"source_model": "soundscape", "vector_type": "random_control", "beta": 0.0,
             "focal_accuracy": 0.5625, "soundscape_accuracy": 0.6967},
            {"source_model": "soundscape", "vector_type": "random_control", "beta": 0.25,
             "focal_accuracy": 0.5654, "soundscape_accuracy": 0.6990},
            {"source_model": "soundscape", "vector_type": "random_control", "beta": 0.5,
             "focal_accuracy": 0.5648, "soundscape_accuracy": 0.6994},
            {"source_model": "soundscape", "vector_type": "random_control", "beta": 0.75,
             "focal_accuracy": 0.5659, "soundscape_accuracy": 0.6975},
            {"source_model": "soundscape", "vector_type": "random_control", "beta": 1.0,
             "focal_accuracy": 0.5609, "soundscape_accuracy": 0.6973},
        ],
    }
    plot_domain_negation(negation_data, str(out / "domain_negation_v2.pdf"))

    # --- Data efficiency ---
    de_results = {
        "fractions": [
            {"fraction": 0.25, "g4_standalone_accuracy": 0.9195,
             "g4_merged_accuracy": 0.9325, "all_group_accuracy": 0.5879,
             "tv_cosine_with_full": 0.035},
            {"fraction": 0.50, "g4_standalone_accuracy": 0.9481,
             "g4_merged_accuracy": 0.9325, "all_group_accuracy": 0.5889,
             "tv_cosine_with_full": 0.7813},
            {"fraction": 0.75, "g4_standalone_accuracy": 0.9558,
             "g4_merged_accuracy": 0.9325, "all_group_accuracy": 0.5913,
             "tv_cosine_with_full": 0.9095},
            {"fraction": 1.0, "g4_standalone_accuracy": 0.9610,
             "g4_merged_accuracy": 0.9377, "all_group_accuracy": 0.5878,
             "tv_cosine_with_full": 1.0},
        ],
    }
    plot_data_efficiency(de_results, str(out / "data_efficiency_v2.pdf"))

    logger.info("=" * 60)
    logger.info("All demo figures generated in %s", out)
    logger.info("=" * 60)


if __name__ == "__main__":
    generate_demo_figures("/gs/bs/tga-mdl/ragib_mdl/merging_bio/figures")