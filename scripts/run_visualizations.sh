#!/bin/bash
# scripts/run_visualizations.sh
# ============================================================================
# Generate all paper figures.
#
# Order:
#   1. Static plots from JSON (fast, no GPU) — ~30 seconds
#   2. Weight-space PCA + layer heatmap (CPU, loads task vectors) — ~2 minutes
#   3. GPU plots: UMAP + loss landscape — ~30-60 minutes
#
# Usage:
#   bash scripts/run_visualizations.sh                # all figures
#   bash scripts/run_visualizations.sh --static-only  # JSON plots only
#   bash scripts/run_visualizations.sh --skip-landscape  # skip slow landscape
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

OUTPUT_DIR="figures"
mkdir -p "$OUTPUT_DIR"

# Parse args
STATIC_ONLY=false
SKIP_LANDSCAPE=false
SKIP_UMAP=false

for arg in "$@"; do
    case $arg in
        --static-only) STATIC_ONLY=true ;;
        --skip-landscape) SKIP_LANDSCAPE=true ;;
        --skip-umap) SKIP_UMAP=true ;;
    esac
done

echo "============================================================"
echo "Phase 1/3: Static Results Plots (from JSON)"
echo "============================================================"
python analysis/visualize_results.py \
    --results-dir results/ \
    --output "$OUTPUT_DIR/"

if [ "$STATIC_ONLY" = true ]; then
    echo "Static-only mode. Done."
    exit 0
fi

echo ""
echo "============================================================"
echo "Phase 2/3: Weight-Space Visualizations (CPU)"
echo "============================================================"
python analysis/visualize_weight_space.py \
    --config configs/base.yaml \
    --output "$OUTPUT_DIR/"

echo ""
echo "============================================================"
echo "Phase 3/3: GPU Visualizations (UMAP + Loss Landscape)"
echo "============================================================"

GPU_ARGS=""
if [ "$SKIP_LANDSCAPE" = true ]; then
    GPU_ARGS="$GPU_ARGS --skip-landscape"
fi
if [ "$SKIP_UMAP" = true ]; then
    GPU_ARGS="$GPU_ARGS --skip-umap"
fi

python analysis/visualize_gpu.py \
    --config configs/base.yaml \
    --output "$OUTPUT_DIR/" \
    --device cuda \
    --umap-samples 3000 \
    --landscape-steps 21 \
    $GPU_ARGS

echo ""
echo "============================================================"
echo "ALL FIGURES GENERATED"
echo "============================================================"
echo "Output directory: $OUTPUT_DIR/"
ls -la "$OUTPUT_DIR/"*.pdf 2>/dev/null || echo "(no PDFs yet)"