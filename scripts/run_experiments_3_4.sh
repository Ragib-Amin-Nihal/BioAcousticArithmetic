#!/bin/bash
# scripts/run_experiments_3_4.sh
#
# End-to-end pipeline for Experiments 3 (Regional Composition) and 4 (Domain Negation).
#
# Prerequisites:
#   - Phase 0 data download + preprocessing complete (BirdCLEF, BirdSet POW)
#   - Phase 1 fine-tuning complete for G1-G5, ALL_birds
#   - BEATs base checkpoint at checkpoints/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
#
# This script:
#   1. Creates regional manifests (R1-R4 + R1234 joint)
#   2. Creates domain manifests (focal/soundscape/mixed)
#   3. Fine-tunes per-region models (beats-R1 through R4 + R1234)
#   4. Fine-tunes per-domain models (beats-focal, beats-soundscape, beats-mixed)
#   5. Computes task vectors for all new models
#   6. Runs Experiment 3 (regional composition evaluation)
#   7. Runs Experiment 4 (domain negation evaluation)
#
# Usage:
#   bash scripts/run_experiments_3_4.sh                    # Full pipeline
#   bash scripts/run_experiments_3_4.sh --skip-finetune    # Skip to evaluation
#   bash scripts/run_experiments_3_4.sh --exp3-only        # Just Experiment 3
#   bash scripts/run_experiments_3_4.sh --exp4-only        # Just Experiment 4
#   bash scripts/run_experiments_3_4.sh --gpu 1            # Use specific GPU
#   bash scripts/run_experiments_3_4.sh --quick            # Reduced beta grid + fewer trials
#
# Resume support for Experiment 4 (survives 24h server limits):
#   bash scripts/run_experiments_3_4.sh --exp4-only --resume
#   bash scripts/run_experiments_3_4.sh --exp4-only --resume --exp4-phases 4
#   bash scripts/run_experiments_3_4.sh --exp4-only --resume --exp4-phases 4 --exp4-seed-start 2

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG="configs/base.yaml"
GPU="${GPU:-0}"
SKIP_FINETUNE=false
EXP3_ONLY=false
EXP4_ONLY=false
QUICK=false
RESUME=false
EXP4_PHASES=""
EXP4_SEED_START=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-finetune)    SKIP_FINETUNE=true; shift ;;
        --exp3-only)        EXP3_ONLY=true; shift ;;
        --exp4-only)        EXP4_ONLY=true; shift ;;
        --gpu)              GPU="$2"; shift 2 ;;
        --config)           CONFIG="$2"; shift 2 ;;
        --quick)            QUICK=true; shift ;;
        --resume)           RESUME=true; shift ;;
        --exp4-phases)      EXP4_PHASES="$2"; shift 2 ;;
        --exp4-seed-start)  EXP4_SEED_START="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES="${GPU}"

echo "=================================================================="
echo " Experiments 3 & 4 Pipeline"
echo " GPU: ${GPU}"
echo " Config: ${CONFIG}"
echo " Skip fine-tuning: ${SKIP_FINETUNE}"
echo " Quick mode: ${QUICK}"
echo " Resume mode: ${RESUME}"
if [ -n "${EXP4_PHASES}" ]; then
    echo " Exp4 phases: ${EXP4_PHASES}"
fi
if [ -n "${EXP4_SEED_START}" ]; then
    echo " Exp4 seed start: ${EXP4_SEED_START}"
fi
echo "=================================================================="

# ---------------------------------------------------------------------------
# Step 1: Create regional manifests (R1-R4 + R1234)
# ---------------------------------------------------------------------------
if [ "$EXP4_ONLY" = false ]; then
    echo ""
    echo "=== Step 1: Creating regional manifests ==="
    python data/create_regional_manifests.py --config "${CONFIG}"
fi

# ---------------------------------------------------------------------------
# Step 2: Create domain manifests (focal/soundscape/mixed)
# ---------------------------------------------------------------------------
if [ "$EXP3_ONLY" = false ]; then
    echo ""
    echo "=== Step 2: Creating domain manifests ==="
    python data/create_domain_manifests.py --config "${CONFIG}"
fi

# ---------------------------------------------------------------------------
# Step 3: Fine-tune regional models
# ---------------------------------------------------------------------------
if [ "$SKIP_FINETUNE" = false ] && [ "$EXP4_ONLY" = false ]; then
    echo ""
    echo "=== Step 3: Fine-tuning regional models ==="
    echo "    CRITICAL: Using identical config hash as Phase 1 models"

    for region in R1_east_africa R2_south_asia R3_neotropics R4_north_america; do
        echo ""
        echo "--- Fine-tuning beats-${region} ---"
        python train.py \
            --group "${region}" \
            --config "${CONFIG}" \
            --gpu 0
    done

    echo ""
    echo "--- Fine-tuning beats-R1234 (joint baseline) ---"
    python train.py \
        --group "R1234_all" \
        --config "${CONFIG}" \
        --gpu 0
fi

# ---------------------------------------------------------------------------
# Step 4: Fine-tune domain models
# ---------------------------------------------------------------------------
if [ "$SKIP_FINETUNE" = false ] && [ "$EXP3_ONLY" = false ]; then
    echo ""
    echo "=== Step 4: Fine-tuning domain models ==="

    for domain in focal_domain soundscape_domain mixed_domain; do
        echo ""
        echo "--- Fine-tuning beats-${domain} ---"
        python train.py \
            --group "${domain}" \
            --config "${CONFIG}" \
            --gpu 0
    done
fi

# ---------------------------------------------------------------------------
# Step 5: Compute task vectors
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 5: Computing task vectors ==="

if [ "$EXP4_ONLY" = false ]; then
    echo "--- Regional task vectors ---"
    python merging/task_vectors.py \
        --config "${CONFIG}" \
        --groups R1_east_africa R2_south_asia R3_neotropics R4_north_america R1234_all \
        --sanity-check
fi

if [ "$EXP3_ONLY" = false ]; then
    echo "--- Domain task vectors ---"
    python merging/task_vectors.py \
        --config "${CONFIG}" \
        --groups focal_domain soundscape_domain mixed_domain \
        --sanity-check
fi

# ---------------------------------------------------------------------------
# Step 6: Run Experiment 3 (Regional Composition)
# ---------------------------------------------------------------------------
if [ "$EXP4_ONLY" = false ]; then
    echo ""
    echo "=== Step 6: Experiment 3 — Regional Composition ==="
    python evaluation/regional_composition.py \
        --config "${CONFIG}" \
        --output results/regional/ \
        --device cuda
fi

# ---------------------------------------------------------------------------
# Step 7: Run Experiment 4 (Domain Negation)
# ---------------------------------------------------------------------------
if [ "$EXP3_ONLY" = false ]; then
    echo ""
    echo "=== Step 7: Experiment 4 — Domain Negation ==="

    # Build argument list
    EXP4_ARGS=""
    N_SEEDS=3

    if [ "$QUICK" = true ]; then
        EXP4_ARGS="${EXP4_ARGS} --beta-grid 0.0 0.25 0.5 0.75 1.0"
        N_SEEDS=1
    fi

    if [ "$RESUME" = true ]; then
        EXP4_ARGS="${EXP4_ARGS} --resume"
    fi

    if [ -n "${EXP4_PHASES}" ]; then
        EXP4_ARGS="${EXP4_ARGS} --phases ${EXP4_PHASES}"
    fi

    if [ -n "${EXP4_SEED_START}" ]; then
        EXP4_ARGS="${EXP4_ARGS} --random-seed-start ${EXP4_SEED_START}"
    fi

    python evaluation/domain_negation.py \
        --config "${CONFIG}" \
        --output results/negation/ \
        --device cuda \
        --n-random-seeds ${N_SEEDS} \
        ${EXP4_ARGS}
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo " Experiments 3 & 4 complete!"
echo " Results:"
if [ "$EXP4_ONLY" = false ]; then
    echo "   Experiment 3: results/regional/regional_composition_results.json"
fi
if [ "$EXP3_ONLY" = false ]; then
    echo "   Experiment 4: results/negation/negation_results.json"
    echo "   Checkpoint:   results/negation/negation_checkpoint.json"
fi
echo "=================================================================="