#!/usr/bin/env python3
"""Spectral distribution distance vs. task-vector cosine similarity correlation.

Tests the mechanism behind near-orthogonal task vectors: if spectral partitioning
drives weight-space orthogonality, pairwise spectral distribution distances should
correlate positively with pairwise task vector cosine similarities.

This analysis directly addresses the confound raised in reviewer feedback:
recording-device and environment differences co-vary with taxonomy. We
cannot fully disentangle these — and we say so — but showing that spectral
distribution distance (a signal-level property capturing ALL of these factors)
predicts task vector geometry moves the claim from "ecological niche causes
orthogonality" to the stronger, more defensible:

    "Spectral distribution distance predicts task vector geometry —
     regardless of whether that distance stems from taxonomy, recording
     environment, or vocal mechanism."

Algorithm:
    1. For each species group, subsample up to N training clips.
    2. Compute log-mel spectrograms (128 bins, 25ms/10ms, 16kHz).
    3. Average over time and clips → 128-d mean log-power distribution.
    4. Compute all pairwise distances (Jensen-Shannon, L2-on-log-mel, L2-on-normalized).
    5. Load pairwise cosine similarities from existing sparsity_summary.json.
    6. Compute Spearman ρ with permutation-based p-value (N=10 pairs, small n).
    7. Produce publication-ready scatter plot with pair annotations.

Outputs:
    results/spectral_distance/spectral_profiles.json   -- per-group mean log-mel
    results/spectral_distance/spectral_distance_results.json  -- distances + correlations
    figures/spectral_vs_cosine.pdf                     -- main figure

Usage:
    python analysis/spectral_distance.py --config configs/base.yaml
    python analysis/spectral_distance.py --config configs/base.yaml \\
        --max-clips 200 --output results/spectral_distance/ \\
        --sparsity-json results/analysis/sparsity_summary.json

    # Re-use cached profiles (fast, no audio I/O):
    python analysis/spectral_distance.py --config configs/base.yaml \\
        --cached-profiles results/spectral_distance/spectral_profiles.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.stats
import torch
import torchaudio
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Groups that share broad avian acoustic properties (for confound annotation)
AVIAN_GROUPS = {"G1_passerines", "G2_nonpasserine_birds", "G3_raptors_waterbirds"}
# Groups with substantially different recording environments
HYDROPHONE_GROUPS = {"G4_marine_mammals"}  # Watkins = hydrophone recordings


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load YAML config and resolve paths relative to project root."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg["paths"]["project_root"]).resolve()
    for k, v in cfg["paths"].items():
        if k != "project_root" and isinstance(v, str):
            cfg["paths"][k] = str(root / v)
    return cfg


# ---------------------------------------------------------------------------
# Audio → spectral profile
# ---------------------------------------------------------------------------

def build_mel_transform(
    sample_rate: int = 16000,
    n_mels: int = 128,
    n_fft: int = 400,        # 25ms at 16kHz
    hop_length: int = 160,   # 10ms at 16kHz
    f_min: float = 0.0,
    f_max: Optional[float] = None,
) -> torchaudio.transforms.MelSpectrogram:
    """Build a mel spectrogram transform matching BEATs internal settings.

    Parameters match base.yaml: 128 mel bins, 25ms window, 10ms hop.

    Args:
        sample_rate: Audio sample rate in Hz.
        n_mels: Number of mel filterbank bins.
        n_fft: FFT window size in samples.
        hop_length: Hop size in samples.
        f_min: Minimum frequency for mel filterbank.
        f_max: Maximum frequency (None = Nyquist).

    Returns:
        Configured MelSpectrogram transform.
    """
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max if f_max is not None else float(sample_rate // 2),
        power=2.0,
        norm="slaney",
        mel_scale="slaney",
    )


def load_clip(
    path: str,
    target_sr: int = 16000,
    target_samples: int = 80000,
) -> Optional[torch.Tensor]:
    """Load a single audio clip, returning a mono (samples,) tensor.

    Returns None on load failure so callers can skip gracefully.

    Args:
        path: Path to audio file.
        target_sr: Expected sample rate (clips should be pre-processed to 16kHz).
        target_samples: Expected clip length in samples (5s = 80,000).

    Returns:
        Mono waveform tensor or None on failure.
    """
    try:
        waveform, sr = torchaudio.load(path)
    except Exception as exc:
        logger.debug("Failed to load %s: %s", path, exc)
        return None

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    if waveform.shape[0] < target_samples:
        pad = target_samples - waveform.shape[0]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    else:
        waveform = waveform[:target_samples]

    return waveform


def compute_group_spectral_profile(
    clips: list[dict],
    mel_transform: torchaudio.transforms.MelSpectrogram,
    max_clips: int = 500,
    seed: int = 42,
    log_offset: float = 1e-6,
) -> dict[str, list[float]]:
    """Compute the mean log-mel energy profile for one species group.

    Randomly subsamples up to `max_clips` from the training set, computes
    log-mel spectrograms, averages over time within each clip, then averages
    over all clips. The result is a 128-d vector representing the group's
    mean spectral energy distribution.

    The log transformation brings the dynamic range to a linear scale suitable
    for L2 comparison. The mean-over-clips is taken in log space (i.e. geometric
    mean of per-clip energy profiles), which is appropriate because perceptual
    loudness and acoustic power vary multiplicatively.

    Args:
        clips: List of clip dicts with at least a "path" key.
        mel_transform: Configured MelSpectrogram transform.
        max_clips: Maximum number of clips to process.
        seed: Random seed for subsampling.
        log_offset: Small constant added before log to avoid -inf.

    Returns:
        Dict with keys "mean_log_mel" (128-d profile, list[float]),
        "std_log_mel" (128-d std), "n_clips_processed" (int).
    """
    rng = random.Random(seed)
    if len(clips) > max_clips:
        clips = rng.sample(clips, max_clips)

    mel_transform.eval()
    all_profiles: list[torch.Tensor] = []

    for clip in clips:
        waveform = load_clip(clip["path"])
        if waveform is None:
            continue

        with torch.no_grad():
            mel = mel_transform(waveform.unsqueeze(0))  # (1, n_mels, T)
            mel = mel.squeeze(0)                         # (n_mels, T)
            log_mel = torch.log(mel + log_offset)        # (n_mels, T)
            # Mean over time → (n_mels,)
            profile = log_mel.mean(dim=1)

        all_profiles.append(profile)

    if not all_profiles:
        raise RuntimeError("No clips could be loaded for profile computation.")

    stacked = torch.stack(all_profiles, dim=0)  # (N, n_mels)
    mean_profile = stacked.mean(dim=0)          # (n_mels,)
    std_profile = stacked.std(dim=0)            # (n_mels,)

    return {
        "mean_log_mel": mean_profile.tolist(),
        "std_log_mel": std_profile.tolist(),
        "n_clips_processed": len(all_profiles),
        "n_clips_requested": len(clips),
    }


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------

def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence between two distributions.

    Converts log-mel energy profiles to probability distributions via
    softmax normalization before computing JSD. This is symmetric,
    bounded in [0, 1] (using log base 2), and well-defined even when
    one distribution has near-zero mass at some bins.

    Args:
        p: First distribution (n_mels,). Values are mean log-mel energies.
        q: Second distribution (n_mels,). Same scale as p.

    Returns:
        JSD in [0, 1].
    """
    # Softmax to get proper probability distributions (positive, sum to 1)
    def softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    p_prob = softmax(p)
    q_prob = softmax(q)

    m = 0.5 * (p_prob + q_prob)

    # KL(P || M) + KL(Q || M), using scipy for numerical stability
    kl_pm = scipy.stats.entropy(p_prob, m)
    kl_qm = scipy.stats.entropy(q_prob, m)
    jsd = 0.5 * (kl_pm + kl_qm)

    # Clip to [0, 1] to handle floating point rounding
    return float(np.clip(jsd, 0.0, 1.0))


def l2_distance_log_mel(p: np.ndarray, q: np.ndarray) -> float:
    """L2 distance between raw mean log-mel energy profiles.

    Both vectors are in log(power) units. L2 distance here is
    equivalent to comparing the log-spectral envelopes directly,
    capturing differences in both overall level and spectral shape.

    Args:
        p: Mean log-mel profile (n_mels,).
        q: Mean log-mel profile (n_mels,).

    Returns:
        L2 (Euclidean) distance.
    """
    return float(np.linalg.norm(p - q))


def l2_distance_normalized(p: np.ndarray, q: np.ndarray) -> float:
    """L2 distance after mean-centering (removes overall level differences).

    This isolates *spectral shape* differences from level differences.
    If two groups differ only in recording gain but have the same spectral
    distribution, this distance is 0.

    Args:
        p: Mean log-mel profile (n_mels,).
        q: Mean log-mel profile (n_mels,).

    Returns:
        L2 distance between mean-centered profiles.
    """
    p_centered = p - p.mean()
    q_centered = q - q.mean()
    return float(np.linalg.norm(p_centered - q_centered))


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def spearman_with_permutation_pvalue(
    x: list[float],
    y: list[float],
    n_permutations: int = 100_000,
    seed: int = 42,
) -> dict[str, float]:
    """Compute Spearman ρ with permutation-based p-value.

    With only 10 data points (all pairwise combinations of 5 groups),
    parametric p-values from the t-approximation are unreliable. A
    permutation test gives the exact null distribution for n=10.

    Args:
        x: First variable (e.g., spectral distances).
        y: Second variable (e.g., cosine similarities).
        n_permutations: Number of random permutations for null distribution.
        seed: Random seed for reproducibility.

    Returns:
        Dict with rho, p_value (permutation), p_value_parametric, n.
    """
    rng = np.random.default_rng(seed)
    x_arr = np.array(x)
    y_arr = np.array(y)

    observed_rho, parametric_p = scipy.stats.spearmanr(x_arr, y_arr)

    # Permutation null: shuffle x, recompute rho
    null_rhos = np.empty(n_permutations)
    for i in range(n_permutations):
        x_perm = rng.permutation(x_arr)
        null_rhos[i], _ = scipy.stats.spearmanr(x_perm, y_arr)

    # Two-tailed p-value
    p_perm = float(np.mean(np.abs(null_rhos) >= np.abs(observed_rho)))

    return {
        "rho": float(observed_rho),
        "p_value_permutation": p_perm,
        "p_value_parametric": float(parametric_p),
        "n_pairs": len(x),
        "n_permutations": n_permutations,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_spectral_vs_cosine(
    pairs: list[dict],
    output_path: Path,
    distance_metric: str = "jsd",
) -> None:
    """Scatter plot of spectral distribution distance vs. task vector cosine similarity.

    Each point is one pairwise group comparison (n=10).
    Annotations distinguish pair types: bird-bird, bird-other, other-other.
    G4 (marine mammal / hydrophone) pairs are marked with a different symbol
    to visually flag the recording-environment confound.

    Args:
        pairs: List of pair dicts with distance metrics, cosine similarity, pair name.
        output_path: Output PDF path.
        distance_metric: Which distance metric to use as x-axis ("jsd", "l2", "l2_norm").
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RCPARAMS = {
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
    }
    plt.rcParams.update(RCPARAMS)

    COLORS = {
        "bird_bird": "#0072B2",   # Blue
        "bird_other": "#E69F00",  # Orange
        "other_other": "#CC79A7", # Purple
    }
    MARKERS = {
        "hydrophone": "^",  # Triangle for G4-involved pairs (confound flag)
        "standard": "o",
    }

    metric_labels = {
        "jsd": "Jensen–Shannon Divergence (spectral distribution)",
        "l2": "L2 Distance (mean log-mel energy)",
        "l2_norm": "L2 Distance (mean-centered log-mel, shape only)",
    }
    x_label = metric_labels.get(distance_metric, distance_metric)

    fig, ax = plt.subplots(figsize=(5, 3.8))

    xs, ys = [], []
    for p in pairs:
        x_val = p[f"distance_{distance_metric}"]
        y_val = p["cosine_similarity"]
        xs.append(x_val)
        ys.append(y_val)

        pair_type = p["pair_type"]
        has_hydrophone = p["has_hydrophone"]

        color = COLORS[pair_type]
        marker = MARKERS["hydrophone"] if has_hydrophone else MARKERS["standard"]

        ax.scatter(
            x_val, y_val,
            c=color, marker=marker,
            s=55, edgecolors="black", linewidths=0.5, zorder=3,
        )
        # Offset label to avoid overlap (manual nudge per pair)
        label = p["short_label"]
        ax.annotate(
            label,
            (x_val, y_val),
            fontsize=6.5,
            xytext=(4, 3),
            textcoords="offset points",
        )

    # Regression line (on rank-transformed data for Spearman visualization)
    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    if len(xs_arr) > 2:
        m, b = np.polyfit(xs_arr, ys_arr, 1)
        x_line = np.linspace(xs_arr.min(), xs_arr.max(), 100)
        ax.plot(x_line, m * x_line + b, color="gray", linewidth=1.0,
                linestyle="--", alpha=0.7, label="OLS fit", zorder=2)

    # Compute Spearman for annotation on plot
    rho, _ = scipy.stats.spearmanr(xs_arr, ys_arr)
    ax.text(
        0.04, 0.96,
        f"Spearman ρ = {rho:.3f}",
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="gray", linewidth=0.5),
    )

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=COLORS["bird_bird"], edgecolor="black",
              linewidth=0.5, label="Bird–Bird"),
        Patch(facecolor=COLORS["bird_other"], edgecolor="black",
              linewidth=0.5, label="Bird–Other"),
        Patch(facecolor=COLORS["other_other"], edgecolor="black",
              linewidth=0.5, label="Other–Other"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=7, linewidth=0,
               label="Hydrophone recording (G4)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=7,
              framealpha=0.9, edgecolor="gray")

    ax.set_xlabel(x_label)
    ax.set_ylabel("Task Vector Cosine Similarity")
    ax.set_title("Spectral Distribution Distance Predicts\nTask Vector Geometry",
                 fontweight="bold")
    ax.grid(True, linewidth=0.3, alpha=0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path))
    plt.close(fig)
    logger.info("Figure saved: %s", output_path)


def plot_mean_profiles(
    profiles: dict[str, dict],
    output_path: Path,
    n_mels: int = 128,
    sample_rate: int = 16000,
    f_max: float = 8000.0,
) -> None:
    """Plot mean log-mel energy profiles for all groups on one axes.

    This provides visual evidence of spectral partitioning and is useful
    as a supplementary figure or sanity check. G4 (marine mammals) should
    show a distinctly different profile due to both biology and hydrophone
    recording chain.

    Args:
        profiles: Dict mapping group_name → profile dict (from compute_group_spectral_profile).
        output_path: Output PDF path.
        n_mels: Number of mel bins.
        sample_rate: Audio sample rate.
        f_max: Maximum frequency for x-axis labeling.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    GROUP_COLORS = {
        "G1_passerines": "#0072B2",
        "G2_nonpasserine_birds": "#56B4E9",
        "G3_raptors_waterbirds": "#009E73",
        "G4_marine_mammals": "#D55E00",
        "G5_amphibians": "#CC79A7",
    }
    GROUP_LABELS = {
        "G1_passerines": "G1 Passerines",
        "G2_nonpasserine_birds": "G2 Non-passerine birds",
        "G3_raptors_waterbirds": "G3 Raptors/waterbirds",
        "G4_marine_mammals": "G4 Marine mammals†",
        "G5_amphibians": "G5 Amphibians",
    }

    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.labelsize": 9, "legend.fontsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.dpi": 300, "savefig.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(6, 3.5))

    mel_bins = np.arange(n_mels)
    freq_ticks_hz = [500, 1000, 2000, 4000, 8000]
    # Approximate mel bin index for each frequency (linear approximation)
    freq_tick_bins = [
        int(n_mels * (np.log(f / 700 + 1) / np.log(f_max / 700 + 1)))
        for f in freq_ticks_hz
    ]

    for group_name, profile_data in profiles.items():
        mean_profile = np.array(profile_data["mean_log_mel"])
        std_profile = np.array(profile_data["std_log_mel"])
        color = GROUP_COLORS.get(group_name, "gray")
        label = GROUP_LABELS.get(group_name, group_name)

        ax.plot(mel_bins, mean_profile, color=color, linewidth=1.5, label=label)
        ax.fill_between(
            mel_bins,
            mean_profile - 0.5 * std_profile,
            mean_profile + 0.5 * std_profile,
            color=color, alpha=0.15,
        )

    ax.set_xticks(freq_tick_bins)
    ax.set_xticklabels([f"{f}Hz" for f in freq_ticks_hz], fontsize=7)
    ax.set_xlabel("Approximate Frequency (mel scale)")
    ax.set_ylabel("Mean Log-Mel Energy (log W)")
    ax.set_title("Mean Spectral Energy Profiles by Taxonomic Group",
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.annotate(
        "† Marine mammals: hydrophone recording chain",
        xy=(0.01, 0.02), xycoords="axes fraction",
        fontsize=6.5, color="gray",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path))
    plt.close(fig)
    logger.info("Figure saved: %s", output_path)


# ---------------------------------------------------------------------------
# Pair classification helpers
# ---------------------------------------------------------------------------

def classify_pair(group_a: str, group_b: str) -> tuple[str, bool]:
    """Classify a pair as bird-bird, bird-other, or other-other.

    Also flags whether either group uses hydrophone recordings (G4).

    Args:
        group_a: First group name.
        group_b: Second group name.

    Returns:
        (pair_type, has_hydrophone) where pair_type is one of
        "bird_bird", "bird_other", "other_other".
    """
    a_bird = group_a in AVIAN_GROUPS
    b_bird = group_b in AVIAN_GROUPS
    has_hydrophone = (group_a in HYDROPHONE_GROUPS) or (group_b in HYDROPHONE_GROUPS)

    if a_bird and b_bird:
        pair_type = "bird_bird"
    elif a_bird or b_bird:
        pair_type = "bird_other"
    else:
        pair_type = "other_other"

    return pair_type, has_hydrophone


SHORT_NAMES = {
    "G1_passerines": "G1",
    "G2_nonpasserine_birds": "G2",
    "G3_raptors_waterbirds": "G3",
    "G4_marine_mammals": "G4",
    "G5_amphibians": "G5",
}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_analysis(
    config_path: str,
    output_dir: str,
    sparsity_json: Optional[str],
    max_clips: int,
    figures_dir: str,
    cached_profiles: Optional[str],
    groups: Optional[list[str]],
    distance_metrics: list[str],
    n_mels: int = 128,
    seed: int = 42,
) -> None:
    """Full spectral distance vs. cosine similarity correlation analysis.

    Loads training clips, builds spectral profiles (or reuses cache),
    computes pairwise distances, correlates against task-vector cosine
    similarities, and writes results + figures.

    Args:
        config_path: Path to base.yaml.
        output_dir: Directory for JSON results.
        sparsity_json: Path to sparsity_summary.json (from sparsity.py).
                       If None, searches default location.
        max_clips: Max training clips per group (subsampled if needed).
        figures_dir: Directory for output figures.
        cached_profiles: If set, skip audio processing and load profiles
                         from this JSON file.
        groups: List of group names to process (None = all 5 standard groups).
        distance_metrics: Metrics to compute ("jsd", "l2", "l2_norm").
        n_mels: Number of mel bins (must match BEATs config).
        seed: Random seed for subsampling.
    """
    cfg = load_config(config_path)
    root = Path(cfg["paths"]["project_root"]).resolve()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    target_groups = groups or [
        "G1_passerines", "G2_nonpasserine_birds", "G3_raptors_waterbirds",
        "G4_marine_mammals", "G5_amphibians",
    ]

    # ------------------------------------------------------------------
    # Step 1: Build or load spectral profiles
    # ------------------------------------------------------------------
    if cached_profiles:
        logger.info("Loading cached spectral profiles from %s", cached_profiles)
        with open(cached_profiles) as f:
            profiles = json.load(f)
        logger.info("Loaded %d profiles", len(profiles))
    else:
        logger.info("Building spectral profiles (max %d clips/group)", max_clips)
        mel_transform = build_mel_transform(
            sample_rate=cfg["audio"]["sample_rate"],
            n_mels=n_mels,
            n_fft=int(cfg["spectrogram"]["win_length_ms"] * cfg["audio"]["sample_rate"] / 1000),
            hop_length=int(cfg["spectrogram"]["hop_length_ms"] * cfg["audio"]["sample_rate"] / 1000),
        )

        species_groups_dir = Path(cfg["paths"]["species_groups"])
        profiles: dict[str, dict] = {}

        for group in target_groups:
            manifest_path = species_groups_dir / f"{group}.json"
            if not manifest_path.exists():
                logger.warning("Manifest not found, skipping: %s", manifest_path)
                continue

            with open(manifest_path) as f:
                manifest = json.load(f)
            train_clips = manifest.get("train", [])

            if not train_clips:
                logger.warning("No training clips for %s", group)
                continue

            logger.info(
                "  %s: %d training clips → sampling %d",
                group, len(train_clips), min(len(train_clips), max_clips),
            )

            try:
                profile = compute_group_spectral_profile(
                    clips=train_clips,
                    mel_transform=mel_transform,
                    max_clips=max_clips,
                    seed=seed,
                )
            except RuntimeError as e:
                logger.error("Failed to build profile for %s: %s", group, e)
                continue

            profiles[group] = profile
            logger.info(
                "    → %d clips processed, profile shape: %d bins",
                profile["n_clips_processed"], len(profile["mean_log_mel"]),
            )

        # Cache profiles for fast re-runs
        profiles_cache_path = out_dir / "spectral_profiles.json"
        with open(profiles_cache_path, "w") as f:
            json.dump(profiles, f, indent=2)
        logger.info("Profiles cached: %s", profiles_cache_path)

    if len(profiles) < 2:
        logger.error("Need ≥2 groups with valid profiles to compute pairwise distances.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Load pairwise cosine similarities
    # ------------------------------------------------------------------
    if sparsity_json:
        sparsity_path = Path(sparsity_json)
    else:
        sparsity_path = root / "results" / "analysis" / "sparsity_summary.json"
        if not sparsity_path.exists():
            # Also check the old location from Project_Status_Phase2.md
            sparsity_path = root / "results" / "sparsity" / "sparsity_summary.json"

    if not sparsity_path.exists():
        logger.error(
            "sparsity_summary.json not found at %s. "
            "Run analysis/sparsity.py first, or pass --sparsity-json.",
            sparsity_path,
        )
        sys.exit(1)

    with open(sparsity_path) as f:
        sparsity_data = json.load(f)

    # Build lookup: (group_a, group_b) → cosine_similarity
    cosine_lookup: dict[tuple[str, str], float] = {}
    for entry in sparsity_data.get("pairwise", []):
        pair_str = entry.get("pair", "")
        if "_vs_" not in pair_str:
            continue
        a, b = pair_str.split("_vs_", 1)
        cos = float(entry["cosine_similarity"])
        cosine_lookup[(a, b)] = cos
        cosine_lookup[(b, a)] = cos  # symmetric

    logger.info("Loaded %d cosine similarity entries", len(cosine_lookup) // 2)

    # ------------------------------------------------------------------
    # Step 3: Compute pairwise spectral distances
    # ------------------------------------------------------------------
    group_names = sorted(profiles.keys())
    pair_results: list[dict] = []

    import itertools
    for group_a, group_b in itertools.combinations(group_names, 2):
        pa = np.array(profiles[group_a]["mean_log_mel"])
        pb = np.array(profiles[group_b]["mean_log_mel"])

        if len(pa) != len(pb):
            logger.warning("Profile length mismatch: %s vs %s", group_a, group_b)
            continue

        pair_type, has_hydrophone = classify_pair(group_a, group_b)
        sa = SHORT_NAMES.get(group_a, group_a)
        sb = SHORT_NAMES.get(group_b, group_b)

        cosine = cosine_lookup.get((group_a, group_b))
        if cosine is None:
            logger.warning("No cosine similarity found for pair: %s vs %s", group_a, group_b)
            continue

        entry: dict = {
            "pair": f"{group_a}_vs_{group_b}",
            "short_label": f"{sa}–{sb}",
            "group_a": group_a,
            "group_b": group_b,
            "pair_type": pair_type,
            "has_hydrophone": has_hydrophone,
            "cosine_similarity": cosine,
        }

        if "jsd" in distance_metrics:
            entry["distance_jsd"] = jensen_shannon_divergence(pa, pb)
        if "l2" in distance_metrics:
            entry["distance_l2"] = l2_distance_log_mel(pa, pb)
        if "l2_norm" in distance_metrics:
            entry["distance_l2_norm"] = l2_distance_normalized(pa, pb)

        pair_results.append(entry)

    if not pair_results:
        logger.error("No valid pairs found.")
        sys.exit(1)

    logger.info("Computed distances for %d pairs", len(pair_results))

    # ------------------------------------------------------------------
    # Step 4: Spearman correlation for each distance metric
    # ------------------------------------------------------------------
    correlation_results: dict[str, dict] = {}
    cosines = [p["cosine_similarity"] for p in pair_results]

    for metric in distance_metrics:
        key = f"distance_{metric}"
        if key not in pair_results[0]:
            continue
        distances = [p[key] for p in pair_results]
        stats = spearman_with_permutation_pvalue(distances, cosines, seed=seed)
        correlation_results[metric] = stats

        sig = "✓ significant" if stats["p_value_permutation"] < 0.05 else "✗ not significant"
        logger.info(
            "Spearman ρ [%s]: %.4f  (p_perm=%.4f, p_param=%.4f)  %s",
            metric, stats["rho"], stats["p_value_permutation"],
            stats["p_value_parametric"], sig,
        )

    # Also compute correlation excluding G4 (hydrophone) pairs
    for metric in distance_metrics:
        key = f"distance_{metric}"
        if key not in pair_results[0]:
            continue
        non_hydro_pairs = [p for p in pair_results if not p["has_hydrophone"]]
        if len(non_hydro_pairs) < 4:
            continue
        distances_nh = [p[key] for p in non_hydro_pairs]
        cosines_nh = [p["cosine_similarity"] for p in non_hydro_pairs]
        stats_nh = spearman_with_permutation_pvalue(distances_nh, cosines_nh, seed=seed)
        correlation_results[f"{metric}_excl_hydrophone"] = {
            **stats_nh,
            "note": "Excludes G4 (marine mammals / hydrophone) pairs",
            "n_pairs_excluded": len(pair_results) - len(non_hydro_pairs),
        }
        logger.info(
            "Spearman ρ [%s, excl. G4]: %.4f  (p_perm=%.4f, n=%d pairs)",
            metric, stats_nh["rho"], stats_nh["p_value_permutation"],
            len(non_hydro_pairs),
        )

    # ------------------------------------------------------------------
    # Step 5: Save results
    # ------------------------------------------------------------------
    results = {
        "config": {
            "max_clips_per_group": max_clips,
            "n_mels": n_mels,
            "distance_metrics": distance_metrics,
            "seed": seed,
            "sparsity_source": str(sparsity_path),
        },
        "groups_processed": group_names,
        "n_pairs": len(pair_results),
        "pairs": pair_results,
        "correlations": correlation_results,
        "interpretation": {
            "primary_metric": "jsd",
            "note": (
                "Jensen-Shannon divergence is the primary metric: symmetric, bounded [0,1], "
                "and robust to zero bins. L2 metrics confirm robustness. "
                "G4 (marine mammals) pairs are flagged as 'has_hydrophone=True' because "
                "Watkins recordings use hydrophone transducers, creating a recording-chain "
                "confound that cannot be separated from taxonomic/biological differences "
                "in this dataset. Correlation excluding G4 pairs tests whether the "
                "relationship holds within the avian + amphibian subspace."
            ),
        },
    }

    results_path = out_dir / "spectral_distance_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved: %s", results_path)

    # ------------------------------------------------------------------
    # Step 6: Figures
    # ------------------------------------------------------------------
    primary_metric = "jsd" if "jsd" in distance_metrics else distance_metrics[0]
    plot_spectral_vs_cosine(
        pairs=pair_results,
        output_path=fig_dir / "spectral_vs_cosine.pdf",
        distance_metric=primary_metric,
    )

    # Secondary figure: L2 on normalized profiles (shape-only)
    if "l2_norm" in distance_metrics:
        plot_spectral_vs_cosine(
            pairs=pair_results,
            output_path=fig_dir / "spectral_vs_cosine_shape.pdf",
            distance_metric="l2_norm",
        )

    # Supplementary: mean profiles per group
    if len(profiles) >= 2:
        plot_mean_profiles(
            profiles=profiles,
            output_path=fig_dir / "spectral_profiles_by_group.pdf",
            n_mels=n_mels,
        )

    # ------------------------------------------------------------------
    # Step 7: Console summary for the paper
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY FOR PAPER")
    logger.info("=" * 60)
    logger.info("Pairwise spectral distances and cosine similarities:")
    logger.info(
        "  %-30s  %-7s  %-7s  %-6s  %s",
        "Pair", "JSD", "Cosine", "Type", "Hydro?"
    )
    for p in sorted(pair_results, key=lambda x: x["distance_jsd"] if "distance_jsd" in x else 0):
        jsd_str = f"{p.get('distance_jsd', float('nan')):.4f}"
        hydro_flag = "⚠" if p["has_hydrophone"] else ""
        logger.info(
            "  %-30s  %-7s  %-7.4f  %-6s  %s",
            p["short_label"], jsd_str, p["cosine_similarity"],
            p["pair_type"], hydro_flag,
        )
    for metric, stats in correlation_results.items():
        sig = "p<0.05 ✓" if stats["p_value_permutation"] < 0.05 else "p≥0.05 ✗"
        logger.info(
            "Spearman ρ [%s]: %.3f  (%s, perm p=%.4f)",
            metric, stats["rho"], sig, stats["p_value_permutation"],
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spectral distribution distance vs. task-vector cosine similarity.",
    )
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml",
        help="Path to base.yaml",
    )
    parser.add_argument(
        "--output", type=str, default="results/spectral_distance/",
        help="Output directory for JSON results",
    )
    parser.add_argument(
        "--figures", type=str, default="figures/",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--sparsity-json", type=str, default=None,
        help="Path to sparsity_summary.json. Default: auto-detect.",
    )
    parser.add_argument(
        "--max-clips", type=int, default=500,
        help="Max training clips to use per group (subsampled uniformly). "
             "500 takes ~3-5 min on CPU. Use 100 for a quick sanity check.",
    )
    parser.add_argument(
        "--cached-profiles", type=str, default=None,
        help="Path to a previously saved spectral_profiles.json. "
             "Skips audio I/O entirely — useful for re-running correlation/plotting.",
    )
    parser.add_argument(
        "--groups", nargs="+", default=None,
        help="Groups to analyze. Default: all 5 standard groups.",
    )
    parser.add_argument(
        "--distance-metrics", nargs="+",
        default=["jsd", "l2", "l2_norm"],
        choices=["jsd", "l2", "l2_norm"],
        help="Distance metrics to compute.",
    )
    parser.add_argument(
        "--n-mels", type=int, default=128,
        help="Number of mel bins (must match BEATs config).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for subsampling.",
    )

    args = parser.parse_args()

    run_analysis(
        config_path=args.config,
        output_dir=args.output,
        sparsity_json=args.sparsity_json,
        max_clips=args.max_clips,
        figures_dir=args.figures,
        cached_profiles=args.cached_profiles,
        groups=args.groups,
        distance_metrics=args.distance_metrics,
        n_mels=args.n_mels,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()