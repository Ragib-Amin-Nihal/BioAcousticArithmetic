#!/usr/bin/env bash
# scripts/build_page_assets.sh
# Rasterize the curated set of paper figures (figures/*.pdf) into PNGs for the
# project page (docs/static/images/). Uses ghostscript (gs); no Python deps.
#
# Usage:
#   bash scripts/build_page_assets.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

SRC="figures"
OUT="docs/static/images"
mkdir -p "$OUT"

# Curated figures: PDF basename (without extension) -> output PNG name.
# Prefer the *_v2 variants (final paper versions) where they exist.
declare -A FIGS=(
  [cosine_matrix_v2]=cosine_matrix
  [composition_dots_v2]=composition
  [asymmetric_gap_v2]=asymmetric_gap
  [radar_comparison]=radar_comparison
  [weight_pca]=weight_pca
  [spectral_vs_cosine]=spectral_vs_cosine
  [spectral_profiles_by_group]=spectral_profiles
  [lmc_curves]=lmc_curves
  [regional_cosine_matrix]=regional_cosine
  [domain_negation_v2]=domain_negation
  [method_complexity]=method_complexity
  [norm_comparison_v2]=norm_comparison
  [data_efficiency_v2]=data_efficiency
  [layer_heatmap]=layer_heatmap
  [magnitude_flow]=magnitude_flow
  [vision_cosine_distribution]=vision_cosine
  [umap_triptych]=umap_triptych
)

# Default render DPI; umap_triptych is huge so render it lower.
render() {
  local stem="$1" outname="$2" dpi="${3:-300}"
  local pdf="$SRC/$stem.pdf"
  if [[ ! -f "$pdf" ]]; then
    echo "  SKIP (missing): $pdf"
    return
  fi
  gs -sDEVICE=png16m -r"$dpi" -dNOPAUSE -dBATCH -dQUIET \
     -dFirstPage=1 -dLastPage=1 \
     -dTextAlphaBits=4 -dGraphicsAlphaBits=4 \
     -sOutputFile="$OUT/$outname.png" "$pdf"
  echo "  $pdf -> $OUT/$outname.png ($dpi dpi)"
}

echo "==> Rasterizing figures to $OUT"
for stem in "${!FIGS[@]}"; do
  if [[ "$stem" == "umap_triptych" ]]; then
    render "$stem" "${FIGS[$stem]}" 150
  else
    render "$stem" "${FIGS[$stem]}" 300
  fi
done

echo "==> Done. Generated:"
ls -1 "$OUT"
