#!/usr/bin/env bash
# setup.sh — Environment setup for bioacoustic task arithmetic experiments.
#
# Creates a Python venv, installs all dependencies, builds project directory tree,
# clones external model repos, and downloads the BEATs base checkpoint.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh [--cpu]        # --cpu skips CUDA PyTorch install
#   source .venv/bin/activate  # activate after setup
#
# Prerequisites:
#   - Python 3.9+ available as `python3`
#   - git
#   - Kaggle API credentials (~/.kaggle/kaggle.json)
#   - eBird API key (set EBIRD_API_KEY env var or add to .env)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
USE_CPU=false
for arg in "$@"; do
    case "$arg" in
        --cpu) USE_CPU=true ;;
        --help|-h)
            echo "Usage: ./setup.sh [--cpu]"
            echo "  --cpu    Install CPU-only PyTorch (no CUDA)"
            exit 0
            ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Check Python version
# ---------------------------------------------------------------------------
PYTHON="python3"
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+."
    exit 1
fi

PY_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$($PYTHON -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$($PYTHON -c 'import sys; print(sys.version_info.minor)')

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]; }; then
    echo "ERROR: Python 3.9+ required, found $PY_VERSION"
    exit 1
fi
echo "==> Python $PY_VERSION detected"

# ---------------------------------------------------------------------------
# Create virtual environment
# ---------------------------------------------------------------------------
VENV_DIR=".venv"
if [ -d "$VENV_DIR" ]; then
    echo "==> Virtual environment already exists at $VENV_DIR"
else
    echo "==> Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
fi

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# Install PyTorch
# ---------------------------------------------------------------------------
echo "==> Installing PyTorch..."
if [ "$USE_CPU" = true ]; then
    pip install torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu
else
    # CUDA 12.1 — change index URL if you need a different CUDA version
    pip install torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
fi

# Verify torch import
python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# ---------------------------------------------------------------------------
# Install Python dependencies
# ---------------------------------------------------------------------------
echo "==> Installing Python dependencies..."

# Pin numpy<2 — torch 2.2.0 was compiled against NumPy 1.x ABI
pip install "numpy<2"

# Audio processing
pip install librosa soundfile audioread resampy

# Data, experiment tracking, ML utilities
# Pin datasets<3.0 — BirdSet uses a custom loading script which datasets 3.x dropped
pip install \
    "datasets>=2.18,<3.0" \
    transformers \
    wandb \
    scipy \
    scikit-learn \
    pandas \
    matplotlib \
    seaborn \
    torchmetrics \
    pyyaml \
    tqdm \
    requests \
    python-dotenv

# Kaggle CLI (for BirdCLEF downloads)
pip install kaggle

# eBird API client
pip install ebird-api

# NOTE: We do NOT install xenopy/xeno-canto — BirdCLEF data comes via Kaggle.
# NOTE: We do NOT install mergekit here — it's LLM-focused and we implement
#       TIES/DARE directly on raw state dicts. Install later if needed:
#       pip install mergekit

echo "==> All Python packages installed"

# ---------------------------------------------------------------------------
# Create project directory tree
# ---------------------------------------------------------------------------
echo "==> Creating project directory tree..."

DIRS=(
    # Config
    "configs"
    # Data: raw downloads, processed clips, manifests
    "data/raw/birdclef2023"
    "data/raw/birdclef2024"
    "data/raw/birdclef2025"
    "data/raw/birdset"
    "data/raw/watkins"
    "data/raw/anuraset"
    "data/raw/ebird_taxonomy"
    "data/processed/birdclef2023"
    "data/processed/birdclef2024"
    "data/processed/birdclef2025"
    "data/processed/birdset_pow"
    "data/processed/watkins"
    "data/processed/anuraset"
    "data/manifests"
    "data/species_groups"
    # Source code
    "models"
    "merging"
    "evaluation"
    "analysis"
    "scripts"
    # Outputs
    "checkpoints"
    "results/finetuned"
    "results/task_vectors"
    "results/lmc"
    "results/composition"
    "results/regional"
    "results/negation"
    "results/methods"
    "results/analysis"
    "figures"
    # External repos
    "external"
)

for d in "${DIRS[@]}"; do
    mkdir -p "$d"
done

echo "==> Directory tree created"

# ---------------------------------------------------------------------------
# Clone external model repositories
# ---------------------------------------------------------------------------
echo "==> Cloning external repositories..."

# BEATs (Microsoft unilm — large repo, sparse checkout just beats/)
if [ -d "external/unilm" ]; then
    echo "  unilm already cloned"
else
    echo "  Cloning microsoft/unilm (sparse: beats/ only)..."
    git clone --filter=blob:none --sparse https://github.com/microsoft/unilm.git external/unilm
    cd external/unilm
    git sparse-checkout set beats
    cd "$SCRIPT_DIR"
fi

# AVES (Earth Species Project)
if [ -d "external/aves" ]; then
    echo "  aves already cloned"
else
    echo "  Cloning earthspecies/aves..."
    git clone https://github.com/earthspecies/aves.git external/aves
fi

# Watkins downloader
if [ -d "external/getWHOIdata" ]; then
    echo "  getWHOIdata already cloned"
else
    echo "  Cloning mopg/getWHOIdata..."
    git clone https://github.com/mopg/getWHOIdata.git external/getWHOIdata || \
        echo "  WARNING: getWHOIdata clone failed. Watkins download will require manual steps."
fi

echo "==> External repos ready"

# ---------------------------------------------------------------------------
# Download BEATs base checkpoint
# ---------------------------------------------------------------------------
echo "==> Downloading BEATs checkpoint..."

BEATS_CKPT="checkpoints/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
if [ -f "$BEATS_CKPT" ]; then
    echo "  BEATs checkpoint already exists"
else
    # The download URL is from the BEATs README table.
    # This URL may change — if it fails, download manually from:
    # https://github.com/microsoft/unilm/tree/master/beats
    BEATS_URL="https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
    echo "  Downloading from: $BEATS_URL"
    if curl -fSL -o "$BEATS_CKPT" "$BEATS_URL" 2>/dev/null || \
       wget -q -O "$BEATS_CKPT" "$BEATS_URL" 2>/dev/null; then
        echo "  BEATs checkpoint downloaded ($(du -h "$BEATS_CKPT" | cut -f1))"
    else
        echo "  WARNING: BEATs download failed."
        echo "  Download manually from: https://github.com/microsoft/unilm/tree/master/beats"
        echo "  Save as: $BEATS_CKPT"
        rm -f "$BEATS_CKPT"
    fi
fi

# ---------------------------------------------------------------------------
# Create .env template
# ---------------------------------------------------------------------------
if [ ! -f ".env" ]; then
    cat > .env <<'EOF'
# API Keys — fill these in before running download/grouping scripts
EBIRD_API_KEY=
KAGGLE_USERNAME=
KAGGLE_KEY=
WANDB_API_KEY=
EOF
    echo "==> Created .env template — fill in your API keys"
else
    echo "==> .env already exists"
fi

# ---------------------------------------------------------------------------
# Verify BEATs checkpoint loads
# ---------------------------------------------------------------------------
echo "==> Verifying BEATs checkpoint..."
if [ -f "$BEATS_CKPT" ]; then
    python -c "
import torch, sys
ckpt = torch.load('$BEATS_CKPT', map_location='cpu', weights_only=False)
keys = list(ckpt.keys())
n_params = sum(p.numel() for p in ckpt.get('model', {}).values())
print(f'  Checkpoint keys: {keys}')
print(f'  Encoder params: {n_params:,}')
if n_params < 1_000_000:
    print('  WARNING: Unexpectedly few parameters. Verify checkpoint.')
    sys.exit(1)
print('  BEATs checkpoint OK')
" || echo "  WARNING: BEATs verification failed. Check the checkpoint file."
else
    echo "  SKIPPED: BEATs checkpoint not found"
fi

# ---------------------------------------------------------------------------
# Record setup metadata
# ---------------------------------------------------------------------------
python -c "
import json, datetime, subprocess, sys
meta = {
    'setup_date': datetime.datetime.now().isoformat(),
    'python_version': sys.version,
    'git_hash': subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                               capture_output=True, text=True, cwd='$SCRIPT_DIR').stdout.strip() or 'not a git repo',
}
try:
    import torch
    meta['torch_version'] = torch.__version__
    meta['cuda_available'] = torch.cuda.is_available()
    if torch.cuda.is_available():
        meta['cuda_version'] = torch.version.cuda
        meta['gpu_name'] = torch.cuda.get_device_name(0)
except:
    pass
with open('configs/setup_metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)
print(f'  Setup metadata saved to configs/setup_metadata.json')
"

# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. source .venv/bin/activate"
echo "  2. Fill in .env with your API keys"
echo "  3. python scripts/download_datasets.py"
echo "  4. python scripts/preprocess_audio.py"
echo "  5. python scripts/create_species_groups.py"
echo ""
echo "If BEATs checkpoint download failed, get it from:"
echo "  https://github.com/microsoft/unilm/tree/master/beats"
echo ""