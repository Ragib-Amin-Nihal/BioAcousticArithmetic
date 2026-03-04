#!/usr/bin/env python3
"""V2 figure redesign — fixes all issues from V1 prototype review.

Design system:
  - Palette: desaturated, cohesive, CB-safe
  - Spines: top/right removed everywhere
  - Grid: y-only, whisper-light
  - Annotations: earn their space or get cut
  - Labels: collision-checked per figure
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from scipy import stats
from scipy.stats import gaussian_kde

# ============================================================================
# DESIGN SYSTEM
# ============================================================================

PAL = {
    "G1": "#3A6EA5",  "G2": "#D4882C",  "G3": "#3D8B5F",
    "G4": "#B5527A",  "G5": "#6B8EC4",
    "bb": "#3A6EA5",  "bo": "#D4882C",  "oo": "#B5527A",
    "accent": "#D64933",
    "gray": "#9CA3AF", "gray2": "#6B7280", "gray3": "#374151",
    "bg": "#F8F9FA", "grid": "#E5E7EB",
    "pos": "#B5527A", "neg": "#3A6EA5",  # diverging: merging better / joint better
}

# GN = {  # Group names (short)
#     "G1": "Passerines", "G2": "Non-pass. birds", "G3": "Raptors/Waterb.",
#     "G4": "Marine mammals", "G5": "Amphibians",
# }
GN = {  # Group names (short)
    "G1": "G1", "G2": "G2", "G3": "G3",
    "G4": "G4", "G5": "G5",
}
GK = ["G1", "G2", "G3", "G4", "G5"]

PAIRS = [
    ("G1","G2"), ("G1","G3"), ("G1","G4"), ("G1","G5"),
    ("G2","G3"), ("G2","G4"), ("G2","G5"),
    ("G3","G4"), ("G3","G5"),
    ("G4","G5"),
]

def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 10, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "legend.fontsize": 8.5,
        "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
        "axes.linewidth": 0.5, "axes.edgecolor": "#CBD5E1",
        "axes.facecolor": "white", "figure.facecolor": "white",
        "xtick.major.width": 0.4, "ytick.major.width": 0.4,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.direction": "out", "ytick.direction": "out",
        "lines.linewidth": 1.4,
        "grid.linewidth": 0.3, "grid.color": PAL["grid"], "grid.alpha": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "legend.frameon": True, "legend.fancybox": True,
        "legend.framealpha": 0.95, "legend.edgecolor": "#E5E7EB",
    })

apply_style()
OUT = Path("/gs/bs/tga-mdl/ragib_mdl/merging_bio/figures/figure_prototypes_v2")
OUT.mkdir(parents=True, exist_ok=True)

def clean_ax(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, linewidth=0.3, alpha=0.5)


# ============================================================================
# SYNTHETIC DATA
# ============================================================================

def make_lmc():
    np.random.seed(42)
    base_loss = {"G1": 4.8, "G2": 3.9, "G3": 3.3, "G4": 2.1, "G5": 3.0}
    base_acc  = {"G1": 0.52, "G2": 0.58, "G3": 0.65, "G4": 0.84, "G5": 0.62}
    alpha = np.linspace(0, 1, 11)
    out = {}
    for a, b in PAIRS:
        la, lb = base_loss[a], base_loss[b]
        aa, ab = base_acc[a], base_acc[b]
        t = alpha
        # Smooth monotonic with cubic easing
        ease = 3*t**2 - 2*t**3
        losses = la + (lb - la) * ease + np.random.randn(11)*0.06
        accs = (aa + (ab - aa) * ease + np.random.randn(11)*0.003) * 100
        out[(a,b)] = {"alpha": alpha, "loss": losses, "acc": accs}
    return out

def make_cos():
    return {
        ("G1","G2"): 0.089, ("G1","G3"): 0.072, ("G1","G4"): 0.013,
        ("G1","G5"): 0.031, ("G2","G3"): 0.078, ("G2","G4"): 0.016,
        ("G2","G5"): 0.034, ("G3","G4"): 0.018, ("G3","G5"): 0.038,
        ("G4","G5"): 0.015,
    }

def make_gap():
    return {
        "G1": (-11.8, -13.2, -10.4), "G2": (-6.3, -8.1, -4.5),
        "G3": (-3.7, -5.8, -1.6),    "G4": (9.0, 5.2, 12.8),
        "G5": (2.1, -0.5, 4.7),
    }

def make_spectral():
    np.random.seed(123)
    jsds    = np.array([0.12, 0.15, 0.35, 0.22, 0.18, 0.38, 0.28, 0.40, 0.32, 0.45])
    cosines = np.array([0.089, 0.072, 0.013, 0.031, 0.078, 0.016, 0.034, 0.018, 0.038, 0.015])
    ptypes = ["bb","bb","bo","bo","bb","bo","bo","bo","bo","oo"]
    hydro  = [False]*5 + [True, False, True, False, True]
    return jsds, cosines, ptypes, hydro

PAIR_SHORT = {
    ("G1","G2"): "1–2", ("G1","G3"): "1–3", ("G1","G4"): "1–4",
    ("G1","G5"): "1–5", ("G2","G3"): "2–3", ("G2","G4"): "2–4",
    ("G2","G5"): "2–5", ("G3","G4"): "3–4", ("G3","G5"): "3–5",
    ("G4","G5"): "4–5",
}


# ============================================================================
# FIG 2: LMC OVERLAY — two-panel, loss + accuracy
# ============================================================================

def fig_lmc_overlay():
    """All 10 LMC curves overlaid: loss (left) and accuracy (right).

    Message: every pair interpolates monotonically.
    Highlighted representatives show bird-bird / bird-other / other-other.
    """
    data = make_lmc()

    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(7.0, 2.8),
                                      gridspec_kw={"wspace": 0.32})

    def pair_color(a, b):
        if {a, b} <= {"G1","G2","G3"}: return PAL["bb"]
        if {a, b} <= {"G4","G5"}:      return PAL["oo"]
        return PAL["bo"]

    # Background curves (all 10, thin + translucent)
    for pair in PAIRS:
        d = data[pair]
        c = pair_color(*pair)
        ax_l.plot(d["alpha"], d["loss"], color=c, lw=1.0, alpha=0.30)
        ax_a.plot(d["alpha"], d["acc"],  color=c, lw=1.0, alpha=0.30)

    # Highlight one representative per type
    reps = [("G1","G2", PAL["bb"]),
            ("G1","G4", PAL["bo"]),
            ("G4","G5", PAL["oo"])]
    for a, b, c in reps:
        d = data[(a,b)]
        ax_l.plot(d["alpha"], d["loss"], color=c, lw=2.4, alpha=0.95, zorder=5)
        ax_a.plot(d["alpha"], d["acc"],  color=c, lw=2.4, alpha=0.95, zorder=5)

    for ax in (ax_l, ax_a):
        ax.set_xlabel("Interpolation coefficient $\\alpha$")
        ax.set_xlim(-0.02, 1.02)
        clean_ax(ax)

    ax_l.set_ylabel("Cross-entropy loss")
    ax_a.set_ylabel("Top-1 accuracy (%)")

    # "Zero barrier" callout — placed where there's space
    ax_l.text(0.50, 0.97, "All 10 pairs: zero LMC barrier",
              transform=ax_l.transAxes, fontsize=9, ha="center", va="top",
              fontweight="bold", color=PAL["gray3"],
              bbox=dict(boxstyle="round,pad=0.35", fc="white",
                        ec=PAL["grid"], lw=0.5))

    # Compact legend
    legs = [
        Line2D([0],[0], color=PAL["bb"], lw=2.2, label="Bird–Bird (3 pairs)"),
        Line2D([0],[0], color=PAL["bo"], lw=2.2, label="Bird–Other (6 pairs)"),
        Line2D([0],[0], color=PAL["oo"], lw=2.2, label="Other–Other (1 pair)"),
    ]
    ax_a.legend(handles=legs, loc="lower right", fontsize=7.5, borderpad=0.4)

    fig.savefig(OUT / "fig2_lmc_overlay.pdf")
    plt.close(fig)
    print("  fig2_lmc_overlay")


# ============================================================================
# FIG 2-ALT: LMC SMALL MULTIPLES — clean dual-axis version
# ============================================================================

def fig_lmc_multiples():
    """10-panel grid, each with loss (dark line) + accuracy (colored fill).

    Fixes from V1:
      - Fill anchored to y_min (not 0) so it reads as "area under curve"
      - No checkmark glyph (plain text "mono" instead)
      - Legend placed outside axes, no overlap with alpha labels
      - Title color encodes pair type
    """
    data = make_lmc()

    def pair_color(a, b):
        if {a, b} <= {"G1","G2","G3"}: return PAL["bb"]
        if {a, b} <= {"G4","G5"}:      return PAL["oo"]
        return PAL["bo"]

    fig, axes = plt.subplots(2, 5, figsize=(10.5, 4.4), squeeze=False)
    fig.subplots_adjust(hspace=0.50, wspace=0.40)

    for idx, pair in enumerate(PAIRS):
        row, col = divmod(idx, 5)
        ax = axes[row][col]
        d = data[pair]
        a, b = pair
        pc = pair_color(a, b)

        # Loss: bold dark line on left axis
        ax.plot(d["alpha"], d["loss"], color=PAL["gray3"], lw=2.0, zorder=3)
        ax.set_xlim(-0.02, 1.02)
        clean_ax(ax, grid_axis="y")

        # Accuracy: filled area on twin axis, anchored to bottom
        ax2 = ax.twinx()
        ymin_acc = np.floor(d["acc"].min() / 5) * 5  # round down
        ax2.fill_between(d["alpha"], ymin_acc, d["acc"],
                         color=pc, alpha=0.15, zorder=1)
        ax2.plot(d["alpha"], d["acc"], color=pc, lw=1.2, alpha=0.6, zorder=2)
        ax2.set_ylim(ymin_acc, d["acc"].max() + 5)
        ax2.spines["top"].set_visible(False)

        if col == 4:
            ax2.set_ylabel("Acc (%)", fontsize=9, color=pc)
            ax2.tick_params(labelsize=8, colors=pc)
        else:
            ax2.set_yticks([])

        # Title in pair color
        ax.set_title(f"{a} \u2194 {b}", fontsize=11, fontweight="bold",
                     color=pc, pad=5)

        if col == 0:
            ax.set_ylabel("Loss", fontsize=9)
        ax.set_xlabel("$\\alpha$", fontsize=9)
        ax.tick_params(labelsize=8)

        # Monotonic indicator — bottom-right, plain text
        ax.text(0.94, 0.06, "mono", transform=ax.transAxes,
                fontsize=7.5, ha="right", va="bottom",
                color="#16a34a", fontstyle="italic", fontweight="bold")

    # Shared legend below — spaced well below x-labels
    legs = [
        Line2D([0],[0], color=PAL["gray3"], lw=2, label="Loss (left axis)"),
        Line2D([0],[0], color=PAL["bo"], lw=1.5, alpha=0.6, label="Accuracy (right axis, filled)"),
    ]
    fig.legend(handles=legs, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=9)

    fig.savefig(OUT / "fig2_lmc_multiples.pdf")
    plt.close(fig)
    print("  fig2_lmc_multiples")


# ============================================================================
# FIG 3: COSINE — LOWER-TRIANGLE HEATMAP
# ============================================================================

def fig_cosine():
    """Lower-triangle heatmap.  Custom warm colormap.  Taxonomy boundary.

    Fixes from V1: bottom annotation repositioned below colorbar, not
    overlapping x-labels.
    """
    cos_data = make_cos()
    n = 5

    mat = np.full((n, n), np.nan)
    for i in range(n):
        mat[i, i] = np.nan  # diagonal blank
        for j in range(i):
            key = (GK[j], GK[i]) if (GK[j], GK[i]) in cos_data else (GK[i], GK[j])
            mat[i, j] = cos_data.get(key, 0)

    cmap = LinearSegmentedColormap.from_list(
        "warm", ["#FFFBEB", "#FDE68A", "#F59E0B", "#DC2626"], N=256)

    fig, ax = plt.subplots(figsize=(4.0, 3.8))

    # Mask upper triangle + diagonal
    mask = np.triu(np.ones((n,n), dtype=bool), k=0)
    masked = np.ma.array(mat, mask=mask)

    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=0.10,
                   interpolation="nearest", aspect="equal")

    # Gray diagonal blocks
    for i in range(n):
        ax.add_patch(Rectangle((i-0.5, i-0.5), 1, 1, fc=PAL["bg"],
                                ec="white", lw=2, zorder=2))
        # Group color dot on diagonal
        ax.scatter(i, i, s=60, color=PAL[GK[i]], edgecolors="white",
                  linewidths=1.2, zorder=3)

    # Value annotations (lower triangle only)
    for i in range(n):
        for j in range(i):
            val = mat[i, j]
            fw = "bold" if val > 0.06 else "normal"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9.5, fontweight=fw, color=PAL["gray3"])

    # Taxonomy boundary
    for coord in [2.5]:
        ax.axvline(coord, color=PAL["accent"], lw=1.2, ls="--", alpha=0.4, zorder=3)
        ax.axhline(coord, color=PAL["accent"], lw=1.2, ls="--", alpha=0.4, zorder=3)

    # Bracket labels
    ax.text(1.0, -0.85, "Birds", ha="center", fontsize=8,
            color=PAL["bb"], fontweight="bold")
    ax.text(3.5, -0.85, "Non-bird", ha="center", fontsize=8,
            color=PAL["oo"], fontweight="bold")

    labels = [f"{k} {GN[k]}" for k in GK]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=8, rotation=40, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.03, aspect=20)
    cbar.set_label("Cosine similarity", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_linewidth(0.3)

    # Context: placed as figure text well below the plot
    fig.text(0.52, -0.02,
             "Vision task vectors: 0.1–0.5   |   Bioacoustic: 0.01–0.09",
             ha="center", fontsize=7.5, color=PAL["gray2"], fontstyle="italic")

    fig.savefig(OUT / "fig3_cosine.pdf")
    plt.close(fig)
    print("  fig3_cosine")


# ============================================================================
# FIG 4a: SPECTRAL vs COSINE — with KDE marginals + pair labels
# ============================================================================

def fig_spectral():
    """JSD vs cosine scatter.  KDE marginals.  Pair labels.  CI band.

    Fixes from V1: KDE instead of histogram for marginals, pair labels
    on each point, cleaner legend placement.
    """
    jsds, cosines, ptypes, hydro = make_spectral()

    fig = plt.figure(figsize=(4.8, 4.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5],
                          hspace=0.04, wspace=0.04)
    ax = fig.add_subplot(gs[1, 0])
    ax_t = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_r = fig.add_subplot(gs[1, 1], sharey=ax)

    type_colors = {"bb": PAL["bb"], "bo": PAL["bo"], "oo": PAL["oo"]}

    # Scatter
    for i, (j, c, pt, h) in enumerate(zip(jsds, cosines, ptypes, hydro)):
        mk = "^" if h else "o"
        ax.scatter(j, c, c=type_colors[pt], marker=mk, s=80,
                  edgecolors="white", linewidths=0.8, zorder=4)

    # Pair labels
    for i, pair in enumerate(PAIRS):
        label = PAIR_SHORT[pair]
        ax.annotate(label, (jsds[i], cosines[i]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=6.5, color=PAL["gray2"])

    # Regression + CI band
    m, b = np.polyfit(jsds, cosines, 1)
    xf = np.linspace(jsds.min()*0.85, jsds.max()*1.1, 100)
    ax.plot(xf, m*xf + b, color=PAL["gray"], lw=1.2, ls="--", alpha=0.7, zorder=2)
    resid_std = np.std(cosines - (m*jsds + b))
    ax.fill_between(xf, m*xf+b - 1.96*resid_std, m*xf+b + 1.96*resid_std,
                    color=PAL["gray"], alpha=0.06, zorder=1)

    rho, pv = stats.spearmanr(jsds, cosines)
    ax.text(0.96, 0.96, f"$\\rho$ = {rho:.3f}",
            transform=ax.transAxes, fontsize=11, ha="right", va="top",
            fontweight="bold", color=PAL["gray3"],
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=PAL["grid"], lw=0.5))

    ax.set_xlabel("Jensen–Shannon divergence (spectral)")
    ax.set_ylabel("Task vector cosine similarity")
    clean_ax(ax)

    # KDE marginals
    kde_x = gaussian_kde(jsds)
    xm = np.linspace(jsds.min()*0.8, jsds.max()*1.15, 200)
    ax_t.fill_between(xm, kde_x(xm), color=PAL["gray"], alpha=0.15)
    ax_t.plot(xm, kde_x(xm), color=PAL["gray"], lw=0.8)
    ax_t.set_yticks([]); ax_t.tick_params(labelbottom=False)
    for s in ["top","right","left"]: ax_t.spines[s].set_visible(False)

    kde_y = gaussian_kde(cosines)
    ym = np.linspace(cosines.min()*0.5, cosines.max()*1.2, 200)
    ax_r.fill_betweenx(ym, kde_y(ym), color=PAL["gray"], alpha=0.15)
    ax_r.plot(kde_y(ym), ym, color=PAL["gray"], lw=0.8)
    ax_r.set_xticks([]); ax_r.tick_params(labelleft=False)
    for s in ["top","right","bottom"]: ax_r.spines[s].set_visible(False)

    # Legend — bottom of main plot
    legs = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor=PAL["bb"],
               markersize=7, markeredgecolor="white", lw=0, label="Bird–Bird"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=PAL["bo"],
               markersize=7, markeredgecolor="white", lw=0, label="Bird–Other"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=PAL["oo"],
               markersize=7, markeredgecolor="white", lw=0, label="Other–Other"),
        Line2D([0],[0], marker="^", color="w", markerfacecolor=PAL["gray"],
               markersize=7, markeredgecolor="white", lw=0, label="Hydrophone"),
    ]
    ax.legend(handles=legs, loc="lower left", fontsize=7, ncol=2,
             borderpad=0.5, columnspacing=0.8)

    fig.savefig(OUT / "fig4a_spectral.pdf")
    plt.close(fig)
    print("  fig4a_spectral")


# ============================================================================
# FIG 4b: ASYMMETRIC GAP — horizontal lollipop
# ============================================================================

def fig_gap_lollipop():
    """Horizontal diverging lollipop.

    Fixes from V1: label placement collision with y-axis resolved by
    putting values OUTSIDE the dot (not between dot and axis label).
    CI shown as thicker translucent stem.
    """
    gd = make_gap()

    fig, ax = plt.subplots(figsize=(4.5, 2.6))

    groups = list(reversed(GK))
    y = np.arange(len(groups))

    for i, g in enumerate(groups):
        gap, ci_lo, ci_hi = gd[g]
        color = PAL["pos"] if gap > 0 else PAL["neg"]

        # CI: wide translucent bar
        ax.barh(i, ci_hi - ci_lo, left=ci_lo, height=0.25,
                color=color, alpha=0.12, edgecolor="none", zorder=1)

        # Stem
        ax.plot([0, gap], [i, i], color=color, lw=2.0,
                solid_capstyle="round", zorder=2)

        # Dot
        ax.scatter(gap, i, s=110, color=color, edgecolors="white",
                  linewidths=1.5, zorder=4)

        # Value — outside the dot; for extreme values, go above stem
        if gap > 0:
            ax.text(gap + 1.0, i, f"+{gap:.1f}", ha="left", va="center",
                    fontsize=8.5, fontweight="bold", color=color)
        elif gap < -10:
            # Extreme negative: label above stem to avoid y-tick collision
            ax.text(gap / 2, i + 0.30, f"{gap:.1f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color=color)
        else:
            ax.text(gap - 1.0, i, f"{gap:.1f}", ha="right", va="center",
                    fontsize=8.5, fontweight="bold", color=color)

    # Zero line
    ax.axvline(0, color=PAL["gray3"], lw=0.7, zorder=3)

    # Zone labels — at the very top
    ax.text(-7.5, len(groups) + 0.15, "Joint training better",
            ha="center", fontsize=7.5, fontstyle="italic", color=PAL["neg"], alpha=0.5)
    ax.text(7.5, len(groups) + 0.15, "Merging better",
            ha="center", fontsize=7.5, fontstyle="italic", color=PAL["pos"], alpha=0.5)

    # Mean gap
    mean_gap = np.mean([gd[g][0] for g in GK])
    ax.axvline(mean_gap, color=PAL["accent"], lw=0.9, ls=":", zorder=2, alpha=0.6)
    ax.text(mean_gap, -0.65, f"mean {mean_gap:+.1f}pp",
            ha="center", fontsize=7, color=PAL["accent"])

    ax.set_yticks(y)
    ax.set_yticklabels([f"{g}  {GN[g]}" for g in groups], fontsize=9)
    ax.set_xlabel("Accuracy gap (pp)")
    ax.set_ylim(-0.8, len(groups) + 0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(True, axis="x", lw=0.3, alpha=0.4)

    fig.savefig(OUT / "fig4b_gap.pdf")
    plt.close(fig)
    print("  fig4b_gap")


# ============================================================================
# FIG 5: DOMAIN NEGATION — two-panel Pareto
# ============================================================================

def fig_negation():
    """Two-panel negation curves.

    Fixes from V1: legend INSIDE panels (upper-right), not below where
    it overlaps x-labels.  Fill between focal and random is made
    more subtle and only on the soundscape line.
    """
    np.random.seed(77)
    betas = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0),
                                     gridspec_kw={"wspace": 0.28})

    for title, ax in [("Mixed-domain source", ax1), ("Soundscape source", ax2)]:
        # Focal negation curves
        f_sound = 45 + 18*(1 - np.exp(-betas*1.2)) + np.random.randn(len(betas))*0.5
        f_focal = 72 - 25*(1 - np.exp(-betas*0.8)) + np.random.randn(len(betas))*0.5
        # Random control
        r_sound = 45 + 2*betas + np.random.randn(len(betas))*1.5
        r_focal = 72 - 3*betas + np.random.randn(len(betas))*1.5

        # Focal (solid, bold)
        ax.plot(betas, f_sound, "o-", color=PAL["bo"], lw=2.0, ms=5, zorder=4,
                label="Soundscape (focal neg.)")
        ax.plot(betas, f_focal, "s-", color=PAL["bb"], lw=2.0, ms=5, zorder=4,
                label="Focal (focal neg.)")
        # Random (dashed, muted)
        ax.plot(betas, r_sound, "o--", color=PAL["bo"], lw=1.0, ms=3,
                alpha=0.4, zorder=3)
        ax.plot(betas, r_focal, "s--", color=PAL["bb"], lw=1.0, ms=3,
                alpha=0.4, zorder=3)

        # Fill: gap between focal and random on soundscape line
        ax.fill_between(betas, r_sound, f_sound, color=PAL["bo"], alpha=0.06, zorder=1)

        # Random control label (once, in-plot)
        ax.text(betas[-1]+0.05, r_focal[-1], "ctrl", fontsize=6.5,
                color=PAL["gray"], va="center", fontstyle="italic")
        ax.text(betas[-1]+0.05, r_sound[-1], "ctrl", fontsize=6.5,
                color=PAL["gray"], va="center", fontstyle="italic")

        ax.text(0.04, 0.97, title, transform=ax.transAxes,
                fontsize=9.5, fontweight="bold", va="top", color=PAL["gray3"],
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=PAL["grid"], lw=0.5))

        ax.set_xlabel("Negation strength $\\beta$")
        clean_ax(ax)

    ax1.set_ylabel("Accuracy (%)")

    # Legend inside first panel, top-right — clear of curves
    ax1.legend(fontsize=7, loc="center right", borderpad=0.5,
              handlelength=1.5)

    # Dashed line legend annotation
    ax2.plot([], [], "--", color=PAL["gray"], lw=1, alpha=0.5, label="Random control")
    ax2.legend(fontsize=7, loc="center right", borderpad=0.5)

    fig.savefig(OUT / "fig5_negation.pdf")
    plt.close(fig)
    print("  fig5_negation")


# ============================================================================
# SUPP: GAP vs SIZE
# ============================================================================

def fig_gap_size():
    """Log(size) vs gap — clean labeled scatter.

    Fixes from V1: G5 label repositioned to avoid zero-line collision.
    """
    gd = make_gap()
    sizes = {"G1": 80761, "G2": 37574, "G3": 20709, "G4": 1402, "G5": 11898}

    fig, ax = plt.subplots(figsize=(4.2, 3.2))

    xs_log, ys = [], []
    for g in GK:
        x, y = sizes[g], gd[g][0]
        xs_log.append(np.log10(x))
        ys.append(y)
        ax.scatter(x, y, s=100, color=PAL[g], edgecolors="white",
                  linewidths=1.5, zorder=4)

    # Annotations with smart offsets
    offsets = {
        "G1": (1.12, -1.8), "G2": (1.12, -1.5), "G3": (1.12, 1.5),
        "G4": (1.15, 1.5), "G5": (0.7, 2.5),
    }
    for g in GK:
        x, y = sizes[g], gd[g][0]
        dx, dy = offsets[g]
        ax.annotate(f"{g} {GN[g]}", xy=(x, y), xytext=(x*dx, y+dy),
                    fontsize=7.5, color=PAL[g], fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=PAL[g],
                                    lw=0.4, alpha=0.4) if g == "G5" else None)

    # Trend
    c = np.polyfit(xs_log, ys, 1)
    tx = np.logspace(np.log10(800), np.log10(120000), 100)
    ax.plot(tx, c[0]*np.log10(tx)+c[1], "--", color=PAL["gray"], lw=0.9, alpha=0.5)

    r2 = np.corrcoef(xs_log, ys)[0,1]**2
    ax.text(0.05, 0.05, f"$R^2$ = {r2:.2f}", transform=ax.transAxes,
            fontsize=8, color=PAL["gray2"])

    ax.axhline(0, color=PAL["gray3"], lw=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Training set size (clips)")
    ax.set_ylabel("Composition gap (pp)")
    clean_ax(ax)

    # Zone shading
    yl = ax.get_ylim()
    ax.axhspan(0, yl[1], alpha=0.025, color=PAL["pos"], zorder=0)
    ax.axhspan(yl[0], 0, alpha=0.025, color=PAL["neg"], zorder=0)

    fig.savefig(OUT / "supp_gap_size.pdf")
    plt.close(fig)
    print("  supp_gap_size")


# ============================================================================
# COMBINED COMPOSITION FIGURE — Gap + Spectral in one 2-panel
# ============================================================================

def fig_composition_combined():
    """Two-panel figure combining the gap lollipop (left) with spectral
    scatter (right) into one figure.  This could replace Fig 4a+4b as a
    single Figure 4, saving a figure slot.
    """
    gd = make_gap()
    jsds, cosines, ptypes, hydro = make_spectral()

    fig, (ax_gap, ax_sp) = plt.subplots(1, 2, figsize=(7.0, 3.0),
                                         gridspec_kw={"wspace": 0.38,
                                                      "width_ratios": [1, 1.2]})

    # --- Left: lollipop ---
    groups = list(reversed(GK))
    for i, g in enumerate(groups):
        gap, ci_lo, ci_hi = gd[g]
        color = PAL["pos"] if gap > 0 else PAL["neg"]
        ax_gap.barh(i, ci_hi - ci_lo, left=ci_lo, height=0.25,
                    color=color, alpha=0.12, edgecolor="none", zorder=1)
        ax_gap.plot([0, gap], [i, i], color=color, lw=2.0,
                    solid_capstyle="round", zorder=2)
        ax_gap.scatter(gap, i, s=90, color=color, edgecolors="white",
                      linewidths=1.3, zorder=4)
        if gap > 0:
            ax_gap.text(gap+0.8, i, f"+{gap:.1f}", ha="left", va="center",
                        fontsize=7.5, fontweight="bold", color=color)
        elif gap < -10:
            ax_gap.text(gap/2, i+0.30, f"{gap:.1f}", ha="center", va="bottom",
                        fontsize=7.5, fontweight="bold", color=color)
        else:
            ax_gap.text(gap-0.8, i, f"{gap:.1f}", ha="right", va="center",
                        fontsize=7.5, fontweight="bold", color=color)

    ax_gap.axvline(0, color=PAL["gray3"], lw=0.7, zorder=3)
    ax_gap.set_yticks(range(len(groups)))
    ax_gap.set_yticklabels([f"{g} {GN[g]}" for g in groups], fontsize=8)
    ax_gap.set_xlabel("Gap (pp)", fontsize=9)
    ax_gap.spines["top"].set_visible(False)
    ax_gap.spines["right"].set_visible(False)
    ax_gap.spines["left"].set_visible(False)
    ax_gap.tick_params(axis="y", length=0)
    ax_gap.grid(True, axis="x", lw=0.3, alpha=0.4)
    ax_gap.text(0.02, 0.98, "(a)", transform=ax_gap.transAxes,
                fontsize=10, fontweight="bold", va="top")

    # --- Right: spectral scatter ---
    type_colors = {"bb": PAL["bb"], "bo": PAL["bo"], "oo": PAL["oo"]}
    for i, (j, c, pt, h) in enumerate(zip(jsds, cosines, ptypes, hydro)):
        mk = "^" if h else "o"
        ax_sp.scatter(j, c, c=type_colors[pt], marker=mk, s=70,
                     edgecolors="white", linewidths=0.7, zorder=4)
        ax_sp.annotate(PAIR_SHORT[PAIRS[i]], (j, c), xytext=(3, 3),
                       textcoords="offset points", fontsize=5.5, color=PAL["gray2"])

    m, b = np.polyfit(jsds, cosines, 1)
    xf = np.linspace(jsds.min()*0.85, jsds.max()*1.1, 100)
    ax_sp.plot(xf, m*xf+b, "--", color=PAL["gray"], lw=1, alpha=0.6, zorder=2)
    rs = np.std(cosines - (m*jsds+b))
    ax_sp.fill_between(xf, m*xf+b-1.96*rs, m*xf+b+1.96*rs,
                       color=PAL["gray"], alpha=0.06, zorder=1)

    rho, _ = stats.spearmanr(jsds, cosines)
    ax_sp.text(0.96, 0.96, f"$\\rho$ = {rho:.3f}",
               transform=ax_sp.transAxes, fontsize=10, ha="right", va="top",
               fontweight="bold", color=PAL["gray3"],
               bbox=dict(boxstyle="round,pad=0.25", fc="white",
                         ec=PAL["grid"], lw=0.4))

    ax_sp.set_xlabel("Jensen–Shannon divergence", fontsize=9)
    ax_sp.set_ylabel("Task vector cosine sim.", fontsize=9)
    clean_ax(ax_sp)
    ax_sp.text(0.02, 0.98, "(b)", transform=ax_sp.transAxes,
               fontsize=10, fontweight="bold", va="top")

    legs = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor=PAL["bb"],
               ms=6, markeredgecolor="white", lw=0, label="Bird–Bird"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=PAL["bo"],
               ms=6, markeredgecolor="white", lw=0, label="Bird–Other"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=PAL["oo"],
               ms=6, markeredgecolor="white", lw=0, label="Other–Other"),
    ]
    ax_sp.legend(handles=legs, loc="upper left", fontsize=6.5, borderpad=0.4)

    fig.savefig(OUT / "fig4_combined.pdf")
    plt.close(fig)
    print("  fig4_combined")


# ============================================================================

if __name__ == "__main__":
    print("V2 figure prototypes:\n")
    fig_lmc_overlay()
    fig_lmc_multiples()
    fig_cosine()
    fig_spectral()
    fig_gap_lollipop()
    fig_negation()
    fig_gap_size()
    fig_composition_combined()

    # Convert to PNG for inspection
    import subprocess
    for pdf in OUT.glob("*.pdf"):
        subprocess.run(["pdftoppm", "-png", "-r", "250", str(pdf),
                        str(pdf.with_suffix(""))], check=True)

    print(f"\nAll saved to {OUT}/")