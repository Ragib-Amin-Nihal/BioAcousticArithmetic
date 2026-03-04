#!/usr/bin/env python3
"""Apply figure style fixes to visualize_weight_space.py and spectral_distance.py.

These changes ensure all figures are readable at Interspeech column width:
1. RCPARAMS font sizes bumped up
2. All suptitle() calls removed  
3. All set_title() calls removed
4. Inline fontsize literals increased
5. Marker and annotation sizes increased

Usage:
    python scripts/patch_weight_and_spectral.py [--dry-run]
"""

import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Exact string replacements (safer than regex for complex expressions)
# ---------------------------------------------------------------------------

WEIGHT_SPACE_REPLACEMENTS = [
    # RCPARAMS block
    (
        '''RCPARAMS = {
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
}''',
        '''RCPARAMS = {
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
}''',
    ),

    # PCA title
    (
        '    ax.set_title("Weight-Space PCA: All Models in One Basin", fontweight="bold")',
        '    # Title removed — paper caption handles this',
    ),

    # PCA annotations
    (
        '            fontsize=7,',
        '            fontsize=9,',
    ),

    # Layer heatmap title
    (
        '    ax1.set_title("Per-Layer Task Vector L2 Norm", fontweight="bold")',
        '    # Title removed — paper caption handles this',
    ),

    # Layer heatmap cell annotations
    (
        '                ax1.text(j, i, f"{val:.2f}", ha="center", va="center",\n                         fontsize=6, color=color)',
        '                ax1.text(j, i, f"{val:.2f}", ha="center", va="center",\n                         fontsize=8, color=color)',
    ),

    # Cosine bar title
    (
        '    ax2.set_title("Mean Pairwise Cosine Similarity per Layer", fontweight="bold", fontsize=9)',
        '    # Title removed — paper caption handles this',
    ),

    # Cosine bar annotations
    (
        '                f"{val:.3f}", ha="center", va="bottom", fontsize=6,',
        '                f"{val:.3f}", ha="center", va="bottom", fontsize=8,',
    ),

    # Colorbar tick labelsize
    (
        '    cbar1.ax.tick_params(labelsize=7)',
        '    cbar1.ax.tick_params(labelsize=9)',
    ),

    # Magnitude flow title
    (
        '    ax.set_title("Task Vector Magnitude by Parameter Type", fontweight="bold")',
        '    # Title removed — paper caption handles this',
    ),

    # Magnitude flow legend
    (
        '    ax.legend(frameon=True, framealpha=0.9, edgecolor="gray", fontsize=7)',
        '    ax.legend(frameon=True, framealpha=0.9, edgecolor="gray", fontsize=9)',
    ),
]


SPECTRAL_DISTANCE_REPLACEMENTS = [
    # RCPARAMS block (inside plot_spectral_vs_cosine)
    (
        '''    RCPARAMS = {
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
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }''',
        '''    RCPARAMS = {
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
        "lines.linewidth": 1.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }''',
    ),

    # Scatter marker size
    (
        '            s=55, edgecolors="black", linewidths=0.5, zorder=3,',
        '            s=70, edgecolors="black", linewidths=0.5, zorder=3,',
    ),

    # Annotation fontsize
    (
        '            fontsize=6.5,',
        '            fontsize=8.5,',
    ),

    # Spearman rho annotation fontsize
    (
        '        fontsize=8.5,',
        '        fontsize=10,',
    ),

    # Legend fontsize
    (
        '    ax.legend(handles=legend_elements, loc="upper left", fontsize=7,',
        '    ax.legend(handles=legend_elements, loc="upper left", fontsize=8.5,',
    ),

    # Remove title
    (
        '''    ax.set_title("Spectral Distribution Distance Predicts\\nTask Vector Geometry",
                 fontweight="bold")''',
        '    # Title removed — paper caption handles this',
    ),

    # Hydrophone marker markersize
    (
        '               markeredgecolor="black", markersize=7, linewidth=0,',
        '               markeredgecolor="black", markersize=8, linewidth=0,',
    ),
]


def apply_replacements(content: str, replacements: list[tuple[str, str]], filepath: str) -> str:
    """Apply exact string replacements, reporting each match."""
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new, 1)
            # Show first 60 chars of the replacement
            old_preview = old.strip().splitlines()[0][:60]
            print(f"  ✓ {old_preview}...")
        else:
            old_preview = old.strip().splitlines()[0][:60]
            print(f"  ✗ NOT FOUND: {old_preview}...")
    return content


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    targets = {
        "analysis/visualize_weight_space.py": WEIGHT_SPACE_REPLACEMENTS,
        "analysis/spectral_distance.py": SPECTRAL_DISTANCE_REPLACEMENTS,
    }

    # Also try non-analysis/ paths (scripts may be at project root)
    alt_targets = {}
    for path, reps in targets.items():
        alt = path.replace("analysis/", "")
        alt_targets[alt] = reps
    targets.update(alt_targets)

    for filepath, replacements in targets.items():
        path = Path(filepath)
        if not path.exists():
            continue

        print(f"\n{'[DRY RUN] ' if dry_run else ''}Patching {filepath}:")
        content = path.read_text()
        patched = apply_replacements(content, replacements, filepath)

        if content != patched:
            if not dry_run:
                path.write_text(patched)
                print(f"  → Written")
            else:
                print(f"  → Would write")
        else:
            print(f"  → No changes needed")


if __name__ == "__main__":
    main()