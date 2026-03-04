# """Shared figure style for Interspeech 2026 publication figures.

# Interspeech uses a two-column format:
#   - Column width:  ~3.3 in (84 mm)
#   - Full width:    ~7.0 in (178 mm)
#   - Text body:     9–10 pt

# Font-size targets (minimum legible after PDF compilation):
#   - Axis labels, annotations: ≥ 8 pt at final print size
#   - Tick labels:              ≥ 7 pt
#   - Legend:                   ≥ 7.5 pt

# These RCPARAMS assume figures are authored at their final size
# (figsize in inches matches the intended column/full width), so
# font sizes here correspond directly to what appears in the PDF.

# Usage:
#     from figure_style import apply_style, GROUP_COLORS, GROUP_SHORT, COLUMN_WIDTH, FULL_WIDTH
#     apply_style()
# """

# from __future__ import annotations

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# # ---------------------------------------------------------------------------
# # Layout constants
# # ---------------------------------------------------------------------------
# COLUMN_WIDTH = 3.35   # inches — single-column Interspeech figure
# FULL_WIDTH = 7.0      # inches — full-width Interspeech figure

# # ---------------------------------------------------------------------------
# # rcParams — applies globally when apply_style() is called
# # ---------------------------------------------------------------------------
# RCPARAMS: dict[str, object] = {
#     # Fonts
#     "font.family": "serif",
#     "font.serif": ["Times New Roman", "DejaVu Serif"],
#     "font.size": 11,            # base (was 9)
#     "axes.titlesize": 11,       # we disable titles, but keep parity
#     "axes.labelsize": 11,       # axis labels (was 9)
#     "xtick.labelsize": 10,      # tick labels (was 8)
#     "ytick.labelsize": 10,      # tick labels (was 8)
#     "legend.fontsize": 9,       # legend entries (was 8)
#     "legend.title_fontsize": 10,
#     # Lines and edges
#     "axes.linewidth": 0.8,      # (was 0.6)
#     "xtick.major.width": 0.6,   # (was 0.5)
#     "ytick.major.width": 0.6,
#     "xtick.minor.width": 0.4,
#     "ytick.minor.width": 0.4,
#     "lines.linewidth": 1.5,     # (was 1.2)
#     "lines.markersize": 6,
#     # Export
#     "figure.dpi": 300,
#     "savefig.dpi": 300,
#     "savefig.bbox": "tight",
#     "savefig.pad_inches": 0.05,
#     "pdf.fonttype": 42,         # TrueType — required for camera-ready
#     "ps.fonttype": 42,
#     # Grid
#     "axes.grid": False,
#     "grid.linewidth": 0.4,
#     "grid.alpha": 0.4,
# }


# def apply_style() -> None:
#     """Apply publication rcParams globally."""
#     plt.rcParams.update(RCPARAMS)


# # ---------------------------------------------------------------------------
# # Color palette (colorblind-safe — Wong 2011)
# # ---------------------------------------------------------------------------
# GROUP_COLORS: dict[str, str] = {
#     "G1_passerines":          "#0072B2",  # blue
#     "G2_nonpasserine_birds":  "#E69F00",  # orange
#     "G3_raptors_waterbirds":  "#009E73",  # teal
#     "G4_marine_mammals":      "#CC79A7",  # pink
#     "G5_amphibians":          "#56B4E9",  # sky blue
# }

# GROUP_SHORT: dict[str, str] = {
#     "G1_passerines":          "G1",
#     "G2_nonpasserine_birds":  "G2",
#     "G3_raptors_waterbirds":  "G3",
#     "G4_marine_mammals":      "G4",
#     "G5_amphibians":          "G5",
# }

# REGIONAL_COLORS: dict[str, str] = {
#     "R1_east_africa":    "#0072B2",
#     "R2_south_asia":     "#E69F00",
#     "R3_neotropics":     "#009E73",
#     "R4_north_america":  "#CC79A7",
# }

# REGIONAL_SHORT: dict[str, str] = {
#     "R1_east_africa":    "R1 E. Africa",
#     "R2_south_asia":     "R2 S. Asia",
#     "R3_neotropics":     "R3 Neotropics",
#     "R4_north_america":  "R4 N. America",
# }

# # Taxonomy supersets (for cosine matrix annotation)
# TAXA_BIRD = {"G1_passerines", "G2_nonpasserine_birds", "G3_raptors_waterbirds"}
# TAXA_OTHER = {"G4_marine_mammals", "G5_amphibians"}

# # Method family → color (for composition bar charts)
# METHOD_COLORS: dict[str, str] = {
#     "dare":            "#E69F00",
#     "task_arithmetic": "#0072B2",
#     "ties":            "#CC79A7",
#     "simple_average":  "#009E73",
#     "della":           "#56B4E9",
#     "default":         "#999999",
# }


# def method_color(method_name: str) -> str:
#     """Return color for a merging method based on its family."""
#     name_lower = method_name.lower()
#     for key, color in METHOD_COLORS.items():
#         if key in name_lower:
#             return color
#     return METHOD_COLORS["default"]


"""Improved shared figure style for Interspeech publication figures.

Key changes from v1:
  - Spine-free aesthetic (left + bottom only) for cleaner look
  - Tufte-inspired minimal ink: no unnecessary gridlines, borders, or fills
  - Consistent colorblind-safe palette (Wong 2011) with semantic mapping
  - Predefined figure size presets for common layouts
  - Helper functions for common annotation patterns
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
COLUMN_WIDTH = 3.35
FULL_WIDTH = 7.0
MARGIN = 0.12  # figure margin fraction

# ---------------------------------------------------------------------------
# Color palette (Wong 2011 — colorblind-safe)
# ---------------------------------------------------------------------------
BLUE     = "#0072B2"
ORANGE   = "#E69F00"
TEAL     = "#009E73"
PINK     = "#CC79A7"
SKY      = "#56B4E9"
RED      = "#D55E00"
YELLOW   = "#F0E442"
BLACK    = "#000000"
GRAY60   = "#666666"
GRAY80   = "#999999"
GRAY90   = "#CCCCCC"
GRAY95   = "#E5E5E5"

# GROUP_COLORS = {
#     "G1_passerines":          BLUE,
#     "G2_nonpasserine_birds":  ORANGE,
#     "G3_raptors_waterbirds":  TEAL,
#     "G4_marine_mammals":      PINK,
#     "G5_amphibians":          SKY,
# }

GROUP_COLORS = {
    "G1_passerines":          "#8da0cb",  # soft blue
    "G2_nonpasserine_birds":  "#fc8d62",  # soft orange
    "G3_raptors_waterbirds":  "#66c2a5",  # soft teal
    "G4_marine_mammals":      "#e78ac3",  # soft pink
    "G5_amphibians":          "#a6d854",  # soft green
}

GROUP_SHORT = {
    "G1_passerines":          "G1\nPasserines",
    "G2_nonpasserine_birds":  "G2\nNon-pass.",
    "G3_raptors_waterbirds":  "G3\nRaptors",
    "G4_marine_mammals":      "G4\nMarine",
    "G5_amphibians":          "G5\nAmphibians",
}

GROUP_SHORT_INLINE = {
    "G1_passerines":          "G1",
    "G2_nonpasserine_birds":  "G2",
    "G3_raptors_waterbirds":  "G3",
    "G4_marine_mammals":      "G4",
    "G5_amphibians":          "G5",
}

REGIONAL_COLORS = {
    "R1_east_africa":    BLUE,
    "R2_south_asia":     ORANGE,
    "R3_neotropics":     TEAL,
    "R4_north_america":  PINK,
}

REGIONAL_SHORT = {
    "R1_east_africa":    "R1 E. Africa",
    "R2_south_asia":     "R2 S. Asia",
    "R3_neotropics":     "R3 Neotropics",
    "R4_north_america":  "R4 N. America",
}

TAXA_BIRD = {"G1_passerines", "G2_nonpasserine_birds", "G3_raptors_waterbirds"}
TAXA_OTHER = {"G4_marine_mammals", "G5_amphibians"}

ALL_GROUPS = [
    "G1_passerines", "G2_nonpasserine_birds", "G3_raptors_waterbirds",
    "G4_marine_mammals", "G5_amphibians",
]

METHOD_COLORS = {
    "simple_average":  TEAL,
    "task_arithmetic": BLUE,
    "dare":            ORANGE,
    "ties":            PINK,
    "della":           SKY,
    "norm_adjusted":   RED,
    "joint":           BLACK,
}


def method_color(name: str) -> str:
    n = name.lower()
    for key, color in METHOD_COLORS.items():
        if key in n:
            return color
    return GRAY80


# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------
RCPARAMS: dict[str, object] = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "legend.title_fontsize": 9,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.3,
    "ytick.minor.width": 0.3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.pad": 3,
    "ytick.major.pad": 3,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.grid": False,
    "legend.frameon": False,
    "legend.borderpad": 0.3,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.4,
    "legend.columnspacing": 1.0,
}


def apply_style() -> None:
    plt.rcParams.update(RCPARAMS)


# ---------------------------------------------------------------------------
# Helper: remove spines, clean up axes
# ---------------------------------------------------------------------------

def clean_axes(ax: plt.Axes, spines: str = "lb") -> None:
    """Remove unnecessary spines. 'l'=left, 'b'=bottom, 'r'=right, 't'=top."""
    for spine, key in [("top", "t"), ("right", "r"), ("bottom", "b"), ("left", "l")]:
        ax.spines[spine].set_visible(key in spines)


def add_significance_bracket(
    ax: plt.Axes,
    x1: float, x2: float, y: float,
    text: str = "*",
    lw: float = 0.8,
    color: str = BLACK,
) -> None:
    """Draw a significance bracket between two x positions."""
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.015
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=lw, color=color)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom",
            fontsize=8, color=color)


def annotate_bar(
    ax: plt.Axes,
    x: float, y: float, text: str,
    color: str = GRAY60,
    fontsize: float = 8,
    offset: tuple[float, float] = (0, 3),
    ha: str = "center",
    va: str = "bottom",
) -> None:
    """Place a text annotation above/below a bar."""
    ax.annotate(
        text, (x, y),
        xytext=offset, textcoords="offset points",
        ha=ha, va=va, fontsize=fontsize, color=color,
        fontweight="bold",
    )