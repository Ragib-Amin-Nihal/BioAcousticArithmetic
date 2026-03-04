#!/bin/bash
# scripts/run_supplementary.sh
#
# Runs the Tier 2 supplementary experiments:
#   - Continual Learning: adding G5 to existing G1-G4 system
#   - Data Efficiency: how much G4 data does task arithmetic need?
#
# Prerequisites:
#   - Phase 1 fine-tuning complete (G1-G5 individual + ALL_birds)
#   - Task vectors computed for G1-G5
#   - Bootstrap CI run (not strictly needed but helpful context)
#
# Estimated total GPU time: ~10-12 hours
#   - G1234_joint training: ~4-5h
#   - G1234 fine-tuning on G5 (3 LRs × 20 epochs): ~3-4h
#   - G4 sub-sampled training (3 fractions × ~30 min): ~1.5h
#   - Evaluation (both experiments): ~2h
#
# Usage:
#   bash scripts/run_supplementary.sh                          # Full pipeline
#   bash scripts/run_supplementary.sh --continual-only         # Just continual learning
#   bash scripts/run_supplementary.sh --efficiency-only        # Just data efficiency
#   bash scripts/run_supplementary.sh --eval-only              # Just evaluation (skip training)
#   bash scripts/run_supplementary.sh --gpu 1                  # Specific GPU

set -euo pipefail

CONFIG="configs/base.yaml"
GPU="${GPU:-0}"
CONTINUAL_ONLY=false
EFFICIENCY_ONLY=false
EVAL_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --continual-only)  CONTINUAL_ONLY=true; shift ;;
        --efficiency-only) EFFICIENCY_ONLY=true; shift ;;
        --eval-only)       EVAL_ONLY=true; shift ;;
        --gpu)             GPU="$2"; shift 2 ;;
        --config)          CONFIG="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES="${GPU}"

echo "=================================================================="
echo " Supplementary Experiments (Tier 2)"
echo " GPU: ${GPU}"
echo " Config: ${CONFIG}"
echo "=================================================================="

# ---------------------------------------------------------------------------
# Continual Learning Experiment
# ---------------------------------------------------------------------------
if [ "$EFFICIENCY_ONLY" = false ]; then
    echo ""
    echo "=== CONTINUAL LEARNING EXPERIMENT ==="

    if [ "$EVAL_ONLY" = false ]; then
        echo ""
        echo "--- Phase 0: Create G1234_joint manifest ---"
        python analysis/continual_learning.py --config "${CONFIG}" --phase 0

        echo ""
        echo "--- Phase 1: Train G1234_joint model (~4-5 GPU-hours) ---"
        python analysis/continual_learning.py --config "${CONFIG}" --phase 1 --gpu "${GPU}"

        echo ""
        echo "--- Phase 2: Fine-tune G1234 on G5 (~3-4 GPU-hours) ---"
        python analysis/continual_learning.py --config "${CONFIG}" --phase 2 \
            --device cuda --gpu "${GPU}"
    fi

    echo ""
    echo "--- Phase 3: Evaluate all methods ---"
    python analysis/continual_learning.py --config "${CONFIG}" --phase 3 --device cuda

    echo ""
    echo "   Results: results/continual_learning/continual_learning_results.json"
fi

# ---------------------------------------------------------------------------
# Data Efficiency Experiment
# ---------------------------------------------------------------------------
if [ "$CONTINUAL_ONLY" = false ]; then
    echo ""
    echo "=== DATA EFFICIENCY EXPERIMENT ==="

    if [ "$EVAL_ONLY" = false ]; then
        echo ""
        echo "--- Phase 0: Create sub-sampled G4 manifests ---"
        python analysis/data_efficiency.py --config "${CONFIG}" --phase 0

        echo ""
        echo "--- Phase 1: Train G4 at each fraction (~1.5 GPU-hours) ---"
        python analysis/data_efficiency.py --config "${CONFIG}" --phase 1 --gpu "${GPU}"
    fi

    echo ""
    echo "--- Phase 2: Evaluate all fractions ---"
    python analysis/data_efficiency.py --config "${CONFIG}" --phase 2 --device cuda

    echo ""
    echo "   Results: results/data_efficiency/data_efficiency_results.json"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo " Supplementary experiments complete!"
echo "=================================================================="