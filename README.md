# Ecologically-Constrained Task Arithmetic for Bioacoustic Species Classification

Code for the paper:  
**"Ecologically-Constrained Task Arithmetic for Bioacoustic Species Classification"**  
*Submitted to Interspeech 2025*

---

## Overview

This repository implements weight-space model merging (task vector arithmetic) applied to BEATs audio encoders independently fine-tuned across taxonomic species groups. The central finding is that bioacoustic task vectors are near-orthogonal (cosine similarity 0.01–0.09), a structural property arising from spectral partitioning across ecological niches. This makes simple averaging optimal over sophisticated conflict-resolution methods like TIES and DARE in this domain.

**Taxonomic groups used in experiments:**

| Group | Description | Classes | Train clips | Val clips | Test clips |
|-------|-------------|---------|-------------|-----------|------------|
| G1 | Passerines | 336 | 80,761 | 11,443 | 22,966 |
| G2 | Non-passerine birds | 157 | 37,574 | 5,312 | 10,674 |
| G3 | Raptors / waterbirds | 84 | 20,709 | 2,935 | 5,890 |
| G4 | Marine mammals | 21 | 1,402 | 188 | 385 |
| G5 | Amphibians | 63 | 11,898 | 1,682 | 3,379 |
| ALL_birds | Joint baseline | 597 | 143,446 | 20,312 | 40,778 |

Note: G6 (insects) was dropped — only 3 classes with 111 training samples, insufficient for fine-tuning. 9 terrestrial mammal species from BirdCLEF 2025 (raccoons, jaguars, sloths, etc.) were also dropped as they fit no taxonomic group.

**Regional subsets used in Experiment 3:**

| Region | Source | Geographic scope | Species |
|--------|--------|-----------------|---------|
| R1 | BirdCLEF 2023 | East Africa (Kenya) | 264 |
| R2 | BirdCLEF 2024 | South Asia (India) | 182 |
| R3 | BirdCLEF 2025 | Neotropics (Colombia) | 206 |
| R4 | BirdSet POW | North America (Pennsylvania) | 48 |

---

## Repository Structure

```
.
├── setup.sh                      # Environment setup (venv, deps, BEATs checkpoint)
├── base.yaml                     # Master config — all hyperparameters live here
│
├── # ── Data ──────────────────────────────────────────────────────────
├── download_datasets.py          # BirdCLEF 2023/24/25, BirdSet POW, Watkins, AnuraSet
├── preprocess_audio.py           # Resample → 16kHz mono, 5s clips, write manifests
├── dataset.py                    # PyTorch Dataset for manifest-based loading
├── create_domain_manifests.py    # Focal / soundscape / mixed domain splits
├── create_regional_manifests.py  # R1–R4 regional splits
│
├── # ── Model ─────────────────────────────────────────────────────────
├── beats_classifier.py           # BEATsClassifier: encoder + mean pool + linear head
│
├── # ── Training & task vectors ────────────────────────────────────────
├── train.py                      # Fine-tune BEATs on one species group
├── task_vectors.py               # τ = θ_finetuned_encoder − θ_pretrained_encoder; merge
├── ties_dare.py                  # TIES, DARE, DELLA merging implementations
│
├── # ── Experiments ────────────────────────────────────────────────────
├── lmc_analysis.py               # Exp 1: Linear Mode Connectivity (10 specialist pairs)
├── recompute_lmc_barriers.py     # Corrected LMC barrier metric (avoids cross-task artifacts)
├── aggregate_lmc.py              # Aggregate LMC results across pairs
├── composition_eval.py           # Exp 2: Species-group composition, method comparison
├── regional_composition.py       # Exp 3: Regional composition (R1–R4)
├── domain_negation.py            # Exp 4: Focal→soundscape domain negation + random controls
├── norm_adjusted_ablation.py     # Ablation: norm-adjusted vs uniform coefficient weighting
├── linear_probe.py               # Frozen encoder + trained linear head evaluation
├── knn_eval.py                   # k-NN evaluation on merged encoder feature space
├── bootstrap_gap.py              # Bootstrap 95% CI on composition gap
├── compute_efficiency.py         # Compute cost: merge vs retrain vs fine-tune
├── spectral_distance.py          # Jensen-Shannon spectral distance correlation analysis
│
├── # ── Supplementary experiments ──────────────────────────────────────
├── continual_learning.py         # Sequential fine-tuning forgetting dynamics
├── data_efficiency.py            # Merge quality vs G4 data fraction
├── extract_predictions.py        # Raw predictions for confusion / calibration figures
│
├── # ── Visualisation ──────────────────────────────────────────────────
├── visualize_results.py          # Static figures from JSON results (CPU)
├── visualize_weight_space.py     # PCA, per-layer heatmap, magnitude flow (CPU)
├── visualize_gpu.py              # UMAP triptych, loss landscape (GPU, ~45 min)
│
└── # ── Pipeline scripts ────────────────────────────────────────────────
    ├── run_experiments_3_4.sh    # End-to-end Experiments 3 & 4
    └── run_visualizations.sh     # Generate all figures from completed results
```

**Expected directory layout after setup and data download:**
```
checkpoints/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
external/unilm/beats/            # BEATs source (sparse clone)
external/aves/                   # AVES source
data/raw/                        # Downloaded dataset archives
data/processed/                  # 16kHz mono 5s clips (.wav)
data/manifests/                  # Per-dataset JSON clip manifests
data/species_groups/             # Train/val/test/zeroshot splits per group
results/finetuned/{GROUP}/       # best_model.pt per group
results/task_vectors/            # tau_{GROUP}.pt
results/lmc/
results/composition/
results/regional/
results/negation/
results/analysis/
figures/
```

---

## Datasets

All datasets are downloaded automatically by `download_datasets.py`. Manual prerequisites are noted below.

### BirdCLEF 2023 / 2024 / 2025 (Kaggle)

Focal bird recordings used to construct species groups G1–G3 and regional subsets R1–R3.

- BirdCLEF 2023 (East Africa): https://www.kaggle.com/competitions/birdclef-2023 — 142,923 clips, 264 species after preprocessing
- BirdCLEF 2024 (South Asia): https://www.kaggle.com/competitions/birdclef-2024 — 209,236 clips, 182 species
- BirdCLEF 2025 (Neotropics): https://www.kaggle.com/competitions/birdclef-2025 — 210,855 clips, 206 species (multi-taxa; uses `train.csv` + `taxonomy.csv`, not `train_metadata.csv`)

**Requires:** Kaggle API credentials at `~/.kaggle/kaggle.json`. Accept each competition's rules on Kaggle before downloading.

### BirdSet POW (HuggingFace)

North America soundscape recordings. Used as regional subset R4.

- Dataset: https://huggingface.co/datasets/DBD-research-group/BirdSet (POW subset)
- Paper: https://arxiv.org/abs/2403.10380
- 166,536 clips, 48 species after preprocessing

**Note:** BirdSet uses HuggingFace `datasets` with `decode=False` audio and integer `ClassLabel` indices for `ebird_code`. The download script handles this transparently.

### Watkins Marine Mammal Sound Database

Used to construct species group G4.

- Website: https://cis.whoi.edu/science/B/whalesounds/index.cfm
- ~2,000 recordings, 32 high-quality species ("Best of" subset used)
- 3,881 clips after preprocessing
- Download helper: https://github.com/mopg/getWHOIdata (cloned automatically by `setup.sh`)
- **License:** Free for academic use; no commercial use; no reposting

### AnuraSet

Neotropical frog recordings. Used to construct species group G5.

- Zenodo: https://zenodo.org/records/8056090
- Paper: https://www.nature.com/articles/s41597-023-02666-2
- 141,039 clips, 42 species after preprocessing
- **License:** CC0
- **Note:** Multi-label CSV format; audio at `audio/{site}/{fname}_{min_t}_{max_t}.wav`. Label extraction is handled by the preprocessing script.

### Audio preprocessing

All audio is standardized to: **16kHz, mono, 5-second non-overlapping clips, −60 dB energy filter** (clips below −60 dB RMS are discarded, not zero-padded). Files shorter than 0.5 s are dropped. No zero-padding is applied to short source files.

---

## Model

**BEATs iter3+ AS2M** — ViT-based audio encoder, ~90M parameters, LayerNorm throughout (no BatchNorm — no recalibration needed after merging).

- Repository: https://github.com/microsoft/unilm/tree/master/beats
- Paper: https://arxiv.org/abs/2212.09058
- Checkpoint: `BEATs_iter3+_AS2M_finetuned_on_AS2M_cpt2.pt` (~360 MB) — download link in the repository README table
- **License:** MIT

**Important:** The base checkpoint is already fine-tuned on AudioSet-2M (not purely self-supervised). Task vectors therefore encode the delta from AudioSet-supervised representations.

**Architecture note:** Raw BEATs checkpoints store weights under `ckpt["model"]` and config under `ckpt["cfg"]`. `BEATsClassifier` removes the predictor head (`self.encoder.predictor = None`) so that `extract_features()` returns `(batch, 248, 768)` encoder output rather than AudioSet logits. The classifier head is `Linear(768, num_classes)` with Xavier uniform initialization, applied after mean pooling over the 248 time frames.

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (≥24 GB VRAM recommended for training; 16 GB sufficient for evaluation-only)
- [Kaggle API credentials](https://www.kaggle.com/docs/api) at `~/.kaggle/kaggle.json` with competition rules accepted
- W&B account (free tier sufficient)

### Install

```bash
git clone <this-repo>
cd <repo-root>
chmod +x setup.sh
./setup.sh          # CUDA machine
./setup.sh --cpu    # CPU-only (evaluation and figures only, no training)
source .venv/bin/activate
```

`setup.sh` creates a Python venv at `.venv/`, installs all dependencies, sparse-clones `microsoft/unilm` (BEATs source only) and `earthspecies/aves`, attempts to download the BEATs base checkpoint, and creates a `.env` template.

**BEATs checkpoint fallback:** The download URL in `setup.sh` (`valle.blob.core.windows.net`) may become unavailable. If the automatic download fails, retrieve the checkpoint manually from the [BEATs README](https://github.com/microsoft/unilm/tree/master/beats) and save it as `checkpoints/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt`.

### Fill in API keys

```bash
# .env (created by setup.sh — fill in before running download or training)
EBIRD_API_KEY=       # https://ebird.org/api/keygen
KAGGLE_USERNAME=
KAGGLE_KEY=
WANDB_API_KEY=
```

---

## Reproducing the Paper

### Phase 0 — Data Download and Preprocessing

**Time:** ~2–4 hours depending on bandwidth

```bash
# Download all datasets
python download_datasets.py --config base.yaml

# Skip or target individual datasets:
python download_datasets.py --config base.yaml --only birdclef2023
python download_datasets.py --config base.yaml --skip watkins

# Preprocess: resample, clip, energy filter, write manifests
python preprocess_audio.py --config base.yaml

# Build domain and regional manifests
python create_domain_manifests.py --config base.yaml
python create_regional_manifests.py --config base.yaml
```

**Verify before proceeding:**
- Each species group has ≥15 species with ≥100 clips
- Manifests exist at `data/manifests/{G1..G5,R1..R4,focal,soundscape}.json`
- Train/val/test splits are stratified and non-overlapping
- Zero-shot held-out species exist for each group

---

### Phase 1 — Fine-tuning

**Time:** ~6–8 GPU-hours per group (~50 GPU-hours total for all 7 groups)

```bash
for GROUP in G1_passerines G2_nonpasserine_birds G3_raptors_waterbirds \
             G4_marine_mammals G5_amphibians ALL_birds; do
    python train.py --config base.yaml --group $GROUP --device cuda
done
```

**Critical constraint: all runs must use identical hyperparameters.** The config hash `c4c3cf3b` in `base.yaml` identifies the canonical training configuration. Changing any value in the `training:` block between runs invalidates task arithmetic. Verify all runs share the same config hash in W&B before proceeding to Phase 2.

Training configuration (reproduced here for visibility — `base.yaml` is the authoritative source):

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-5 |
| Weight decay | 0.01 |
| LR schedule | OneCycleLR (cosine + linear warmup, 500 steps) |
| Batch size | 32 |
| Max epochs | 20 |
| Early stopping patience | 5 |
| Encoder freeze | First 2 epochs, then unfreeze all 90M params |
| Mixed precision | bf16 (torch.cuda.amp) |
| Label smoothing | 0.1 |
| Gradient clipping | 1.0 |
| Augmentation | Time-domain waveform masking + Mixup (α=0.3, 50% of batches) |

**Expected fine-tuning results (reference, for sanity-checking your runs):**

| Group | Classes | Best epoch | Val acc |
|-------|---------|-----------|---------|
| G1 Passerines | 336 | 19 | 61.18% |
| G2 Non-passerine birds | 157 | 18 | 72.08% |
| G3 Raptors/waterbirds | 84 | 19 | 72.91% |
| G4 Marine mammals | 21 | 19 | 84.04% |
| G5 Amphibians | 63 | 19 | 66.11% |

**Checkpoint format.** Each `best_model.pt` contains:
- `encoder_state_dict` — encoder only (used for task vector computation)
- `model_state_dict` — full model (encoder + classifier head)
- `metrics` — train/val loss, accuracy, top-5 accuracy, config hash
- `config` — training/augmentation/audio config snapshot
- `group_name`, `num_classes`, `embed_dim`, `epoch`, `git_hash`, `timestamp`

---

### Phase 2 — Task Vectors and Core Experiments

**Time:** ~3–4 GPU-hours

```bash
# Compute task vectors: τ = encoder_state_dict(finetuned) − encoder_state_dict(pretrained)
python task_vectors.py --config base.yaml
# Sanity check runs automatically: base + 1.0 * tau_G1 must reconstruct G1 exactly
# Each tau_{GROUP}.pt should be ~360 MB (same size as encoder)
```

**Experiment 1 — Linear Mode Connectivity:**
```bash
python lmc_analysis.py --config base.yaml --output results/lmc/
python recompute_lmc_barriers.py --config base.yaml   # use corrected barrier metric
python aggregate_lmc.py --config base.yaml
```

**Experiment 2 — Species-Group Composition (method comparison):**
```bash
python composition_eval.py --config base.yaml --output results/composition/
python norm_adjusted_ablation.py --config base.yaml --output results/analysis/
```

**Supporting analyses:**
```bash
python sparsity.py --config base.yaml --output results/analysis/
python knn_eval.py --config base.yaml --output results/knn/
python bootstrap_gap.py --config base.yaml --output results/gap_analysis/
python compute_efficiency.py --config base.yaml --output results/efficiency/
python spectral_distance.py --config base.yaml --output results/analysis/
```

---

### Phase 3 — Regional and Domain Experiments

**Time:** ~12 GPU-hours (includes additional fine-tuning for R1–R4)

```bash
bash run_experiments_3_4.sh                  # Full pipeline
bash run_experiments_3_4.sh --skip-finetune  # Skip if R1–R4 models already exist
bash run_experiments_3_4.sh --exp3-only      # Experiment 3 (regional) only
bash run_experiments_3_4.sh --exp4-only      # Experiment 4 (domain negation) only
bash run_experiments_3_4.sh --gpu 1          # Specify GPU index
bash run_experiments_3_4.sh --quick          # Reduced beta grid for testing
```

**Experiment 4 note:** Domain negation includes a random-vector control with matched per-layer L2 norms. Do not skip `--phases 4` when running Experiment 4 in isolation — the random control is required for a valid causal interpretation.

---

### Phase 4 — Figures

**Time:** ~2 GPU-hours for full figure set

```bash
# Static figures from JSON results (~2 min, CPU only)
python visualize_results.py --config base.yaml --output figures/
python visualize_weight_space.py --config base.yaml --output figures/

# GPU-required figures: UMAP triptych, loss landscape (~45 min)
python visualize_gpu.py --config base.yaml --output figures/

# Or run everything:
bash run_visualizations.sh
```

**Supplementary experiment figures** (optional, ~14 additional GPU-hours):
```bash
python continual_learning.py --config base.yaml --output results/continual/
python data_efficiency.py --config base.yaml --output results/data_efficiency/
python extract_predictions.py --config base.yaml --output results/predictions/
```

---

## Key Implementation Notes

**Why simple averaging outperforms TIES/DARE here.** Bioacoustic task vectors have pairwise cosine similarities of 0.01–0.09, compared to 0.1–0.5 in vision task arithmetic. In this near-orthogonal regime, TIES sign-conflict resolution operates on near-random disagreements, making it no better than simple averaging. DARE's 10× parameter amplification at drop rate 0.9 adds noise without resolving genuine conflicts. This is verified empirically in `composition_eval.py` and explained mechanistically via `spectral_distance.py` (Spearman ρ = −0.915 between pairwise Jensen-Shannon spectral divergence and task vector cosine similarity).

**LMC barrier correction.** The original barrier metric produced spurious high values due to cross-task evaluation artifacts. `recompute_lmc_barriers.py` applies the corrected metric; all 10 specialist pairs show zero barriers.

**DARE numerical stability.** At drop rate ≥ 0.9, the rescaling factor `1/(1−p)` amplifies surviving parameters by ≥10×. `ties_dare.py` forces FP32 for DARE computations regardless of training precision.

**No BatchNorm recalibration needed.** BEATs uses LayerNorm throughout — there are no `running_mean`/`running_var` buffers in the encoder. `setup.sh` verifies this at install time.

**Linear probing protocol.** All merged encoder evaluations use a frozen encoder with a freshly trained linear head (`linear_probe.py`). This isolates merge quality from head compatibility issues across groups.

**eBird API rate limits.** `spectral_distance.py` and ecological weighting scripts cache eBird API responses locally. For large-scale experiments, consider downloading the full eBird Basic Dataset (EBD) instead of using the live API.

---

## Merging Methods Compared

| Method | Implementation |
|--------|---------------|
| Simple average (uniform) | `task_vectors.py` |
| Task arithmetic (coefficient sweep) | `task_vectors.py` |
| Norm-adjusted weighting | `norm_adjusted_ablation.py` |
| TIES-merging | `ties_dare.py` |
| DARE | `ties_dare.py` |
| DELLA (MagPrune) | `ties_dare.py` |

---

## Reference Papers

| Paper | Link |
|-------|------|
| Ilharco et al. — Task Arithmetic | https://arxiv.org/abs/2212.04089 |
| Yadav et al. — TIES-Merging | https://arxiv.org/abs/2306.01708 |
| Yu et al. — DARE | https://arxiv.org/abs/2311.03099 |
| Bhardwaj et al. — DELLA | https://arxiv.org/abs/2406.11617 |
| Yang et al. — AdaMerging | https://arxiv.org/abs/2310.02575 |
| Wortsman et al. — Model Soups | https://arxiv.org/abs/2203.05482 |
| Frankle et al. — Linear Mode Connectivity | https://arxiv.org/abs/1912.05671 |
| Chen et al. — BEATs | https://arxiv.org/abs/2212.09058 |
| Ghani et al. — BirdSet | https://arxiv.org/abs/2403.10380 |

---


## License

Code: MIT  
BEATs model weights: MIT (Microsoft)  
BirdCLEF competition data: Competition rules apply (research use permitted)  
BirdSet/Xeno-canto recordings: CC BY-NC-SA 4.0  
Watkins Marine Mammal data: Academic use only — see [WHOI terms](https://cis.whoi.edu/science/B/whalesounds/about.cfm)  
AnuraSet: CC0
