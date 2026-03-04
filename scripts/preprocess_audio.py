#!/usr/bin/env python3
"""Preprocess all downloaded datasets into standardized 16kHz mono 5-second clips.

For each dataset:
  1. Discovers audio files in the raw download directory
  2. Resamples to 16kHz mono
  3. Segments into non-overlapping 5-second clips (zero-pads final clip if needed)
  4. Saves clips as WAV files organized by species
  5. Builds a JSON manifest with per-clip metadata
  6. Reports per-species sample counts and flags species with <10 clips

Handles dataset-specific formats:
  - BirdCLEF 2023/2024: train_audio/<species_code>/*.ogg + train_metadata.csv
  - BirdCLEF 2025: train_audio/<species_or_taxon_id>/*.ogg + train.csv + taxonomy.csv
  - BirdSet POW: HuggingFace dataset with audio arrays
  - Watkins: audio/<Species_Name>/*.wav
  - AnuraSet: multi-label CSV + flat WAV directory (species are column headers)

Usage:
    python scripts/preprocess_audio.py --config configs/base.yaml
    python scripts/preprocess_audio.py --config configs/base.yaml --only birdclef2023
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torchaudio
import yaml
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML config and resolve paths relative to project root."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg["paths"]["project_root"]).resolve()
    for k, v in cfg["paths"].items():
        if k != "project_root" and isinstance(v, str):
            cfg["paths"][k] = str(root / v)
    for ds_cfg in cfg["datasets"].values():
        if isinstance(ds_cfg, dict):
            for k in ["raw_dir", "processed_dir", "manifest"]:
                if k in ds_cfg:
                    ds_cfg[k] = str(root / ds_cfg[k])
    return cfg


# ---------------------------------------------------------------------------
# Core audio processing
# ---------------------------------------------------------------------------

def process_audio_file(
    input_path: Path,
    output_dir: Path,
    species: str,
    target_sr: int = 16000,
    clip_duration_s: float = 5.0,
    max_clips: int = 0,
    min_duration_s: float = 0.5,
    energy_threshold_db: float = -60.0,
) -> list[dict[str, Any]]:
    """Load an audio file, resample to mono, segment into fixed-length clips.

    Clips whose RMS energy falls below energy_threshold_db are discarded
    (silence-only or near-silence clips that would inject label noise).

    Args:
        input_path: Path to source audio file.
        output_dir: Base output directory (clips saved to output_dir/species/).
        species: Species label for this file.
        target_sr: Target sample rate in Hz.
        clip_duration_s: Duration of each clip in seconds.
        max_clips: Maximum clips to extract per file (0 = all possible).
        min_duration_s: Minimum file duration to process; shorter files skipped.
        energy_threshold_db: Minimum RMS energy in dB (relative to full scale).
            Clips below this are discarded. Default -60 dB removes only
            near-digital-silence. Set to None to disable filtering.

    Returns:
        List of clip metadata dicts.
    """
    clip_samples = int(clip_duration_s * target_sr)
    clips_meta: list[dict[str, Any]] = []

    try:
        info = torchaudio.info(str(input_path))
        file_duration = info.num_frames / info.sample_rate
        if file_duration < min_duration_s:
            return []
        waveform, sr = torchaudio.load(str(input_path))
    except Exception as e:
        log.debug(f"Failed to load {input_path}: {e}")
        return []

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    total_samples = waveform.shape[1]
    n_full_clips = total_samples // clip_samples
    has_remainder = (total_samples % clip_samples) > 0

    # Total clips: full clips + 1 zero-padded remainder if any audio left
    n_clips = n_full_clips + (1 if has_remainder and total_samples > 0 else 0)

    if max_clips > 0:
        n_clips = min(n_clips, max_clips)

    # If file is shorter than one clip, produce one zero-padded clip
    if n_clips == 0 and total_samples > 0:
        n_clips = 1

    species_dir = output_dir / species
    species_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    for i in range(n_clips):
        start = i * clip_samples
        end = start + clip_samples

        if end <= total_samples:
            clip = waveform[:, start:end]
            padded = False
        else:
            clip = torch.zeros(1, clip_samples)
            remaining = total_samples - start
            if remaining > 0:
                clip[:, :remaining] = waveform[:, start:]
            padded = True

        # Energy filter: discard clips that are near-silence
        if energy_threshold_db is not None:
            rms = clip.float().pow(2).mean().sqrt()
            if rms > 0:
                rms_db = 20 * torch.log10(rms).item()
            else:
                rms_db = -100.0
            if rms_db < energy_threshold_db:
                continue

        clip_filename = f"{stem}_clip{i:03d}.wav"
        clip_path = species_dir / clip_filename
        torchaudio.save(str(clip_path), clip, target_sr)

        clips_meta.append({
            "path": str(clip_path),
            "species": species,
            "duration": clip_duration_s,
            "original_file": input_path.name,
            "clip_index": i,
            "zero_padded": padded,
        })

    return clips_meta


# ---------------------------------------------------------------------------
# BirdCLEF 2023 / 2024
# ---------------------------------------------------------------------------

def preprocess_birdclef_2023_2024(
    raw_dir: str,
    processed_dir: str,
    dataset_name: str,
    audio_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Preprocess BirdCLEF 2023 or 2024.

    Structure: train_audio/<species_code>/*.ogg, train_metadata.csv
    """
    raw_path = Path(raw_dir)
    out_path = Path(processed_dir)
    ext_set = {e.lower() for e in audio_cfg["extensions"]}

    train_audio = raw_path / "train_audio"
    if not train_audio.exists():
        log.error(f"  {dataset_name}: train_audio not found at {train_audio}")
        return []

    # Load metadata CSV for scientific/common names
    metadata_map: dict[str, dict[str, str]] = {}
    meta_csv = raw_path / "train_metadata.csv"
    if meta_csv.exists():
        with open(meta_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row.get("primary_label", "")
                if label and label not in metadata_map:
                    metadata_map[label] = {
                        "scientific_name": row.get("scientific_name", ""),
                        "common_name": row.get("common_name", ""),
                    }
        log.info(f"  {dataset_name}: metadata for {len(metadata_map)} species")

    species_dirs = sorted([d for d in train_audio.iterdir() if d.is_dir()])
    log.info(f"  {dataset_name}: {len(species_dirs)} species directories")

    all_clips: list[dict[str, Any]] = []

    for sp_dir in tqdm(species_dirs, desc=f"  {dataset_name}", unit="species"):
        species = sp_dir.name
        audio_files = [f for f in sp_dir.iterdir()
                       if f.is_file() and f.suffix.lower() in ext_set]

        for audio_file in audio_files:
            clips = process_audio_file(
                input_path=audio_file,
                output_dir=out_path,
                species=species,
                target_sr=audio_cfg["sample_rate"],
                clip_duration_s=audio_cfg["clip_duration_s"],
                max_clips=audio_cfg["max_clips_per_file"],
                min_duration_s=audio_cfg["min_duration_s"],
                energy_threshold_db=audio_cfg.get("energy_threshold_db", -60.0),
            )
            for clip in clips:
                clip["dataset_source"] = dataset_name
                if species in metadata_map:
                    clip["scientific_name"] = metadata_map[species]["scientific_name"]
                    clip["common_name"] = metadata_map[species]["common_name"]
            all_clips.extend(clips)

    return all_clips


# ---------------------------------------------------------------------------
# BirdCLEF 2025 (multi-taxa)
# ---------------------------------------------------------------------------

def preprocess_birdclef_2025(
    raw_dir: str,
    processed_dir: str,
    audio_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Preprocess BirdCLEF 2025 (Colombia, multi-taxa).

    Structure:
        train_audio/<primary_label>/*.ogg
        train.csv: primary_label, filename, scientific_name, common_name, ...
        taxonomy.csv: primary_label, inat_taxon_id, scientific_name, common_name, class_name
    Where primary_label is an eBird code (birds) or iNat taxon ID (non-birds).
    """
    raw_path = Path(raw_dir)
    out_path = Path(processed_dir)
    ext_set = {e.lower() for e in audio_cfg["extensions"]}
    dataset_name = "birdclef2025"

    train_audio = raw_path / "train_audio"
    if not train_audio.exists():
        log.error(f"  {dataset_name}: train_audio not found at {train_audio}")
        return []

    # Load taxonomy.csv — maps primary_label to class_name (Aves/Insecta/Amphibia)
    taxonomy: dict[str, dict[str, str]] = {}
    tax_csv = raw_path / "taxonomy.csv"
    if tax_csv.exists():
        with open(tax_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row.get("primary_label", "")
                if label:
                    taxonomy[label] = {
                        "scientific_name": row.get("scientific_name", ""),
                        "common_name": row.get("common_name", ""),
                        "class_name": row.get("class_name", ""),
                        "inat_taxon_id": row.get("inat_taxon_id", ""),
                    }
        log.info(f"  {dataset_name}: taxonomy for {len(taxonomy)} species")

        # Report class breakdown
        class_counts = Counter(t["class_name"] for t in taxonomy.values())
        for cls, count in sorted(class_counts.items()):
            log.info(f"    {cls}: {count} species")
    else:
        log.warning(f"  {dataset_name}: taxonomy.csv not found")

    # Load train.csv for per-file metadata
    file_metadata: dict[str, dict[str, str]] = {}
    train_csv = raw_path / "train.csv"
    if train_csv.exists():
        with open(train_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row.get("filename", "")
                label = row.get("primary_label", "")
                if fname and label:
                    file_metadata[fname] = {
                        "primary_label": label,
                        "scientific_name": row.get("scientific_name", ""),
                        "common_name": row.get("common_name", ""),
                    }
        log.info(f"  {dataset_name}: train.csv has {len(file_metadata)} entries")

    species_dirs = sorted([d for d in train_audio.iterdir() if d.is_dir()])
    log.info(f"  {dataset_name}: {len(species_dirs)} species directories")

    all_clips: list[dict[str, Any]] = []

    for sp_dir in tqdm(species_dirs, desc=f"  {dataset_name}", unit="species"):
        species = sp_dir.name
        audio_files = [f for f in sp_dir.iterdir()
                       if f.is_file() and f.suffix.lower() in ext_set]

        for audio_file in audio_files:
            clips = process_audio_file(
                input_path=audio_file,
                output_dir=out_path,
                species=species,
                target_sr=audio_cfg["sample_rate"],
                clip_duration_s=audio_cfg["clip_duration_s"],
                max_clips=audio_cfg["max_clips_per_file"],
                min_duration_s=audio_cfg["min_duration_s"],
                energy_threshold_db=audio_cfg.get("energy_threshold_db", -60.0),
            )
            for clip in clips:
                clip["dataset_source"] = dataset_name
                # Add taxonomy info
                if species in taxonomy:
                    clip["scientific_name"] = taxonomy[species]["scientific_name"]
                    clip["common_name"] = taxonomy[species]["common_name"]
                    clip["class_name"] = taxonomy[species]["class_name"]
                # Per-file metadata from train.csv
                rel_path = f"{species}/{audio_file.name}"
                if rel_path in file_metadata:
                    fm = file_metadata[rel_path]
                    clip.setdefault("scientific_name", fm.get("scientific_name", ""))
                    clip.setdefault("common_name", fm.get("common_name", ""))

            all_clips.extend(clips)

    return all_clips


# ---------------------------------------------------------------------------
# BirdSet POW (HuggingFace)
# ---------------------------------------------------------------------------

def preprocess_birdset_pow(
    raw_dir: str,
    processed_dir: str,
    audio_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Preprocess BirdSet POW from HuggingFace dataset.

    BirdSet stores audio as numpy arrays in the HF dataset. We extract them
    to WAV files organized by species code.
    """
    raw_path = Path(raw_dir)
    out_path = Path(processed_dir)
    dataset_name = "birdset_pow"

    meta_file = raw_path / "birdset_pow_meta.json"
    if not meta_file.exists():
        log.error(f"  {dataset_name}: metadata not found. Run download_datasets.py first.")
        return []

    with open(meta_file) as f:
        meta = json.load(f)

    log.info(f"  {dataset_name}: loading HuggingFace dataset...")
    try:
        from datasets import load_dataset
        ds = load_dataset(
            meta["hf_dataset"],
            meta["hf_subset"],
            trust_remote_code=True,
            cache_dir=meta["cache_dir"],
        )
    except Exception as e:
        log.error(f"  {dataset_name}: failed to load HF dataset: {e}")
        return []

    target_sr = audio_cfg["sample_rate"]
    clip_samples = int(audio_cfg["clip_duration_s"] * target_sr)
    all_clips: list[dict[str, Any]] = []

    for split_name in ["train"]:
        if split_name not in ds:
            continue

        split = ds[split_name]
        columns = split.column_names
        log.info(f"  {dataset_name}/{split_name}: {len(split)} samples, columns: {columns}")

        # ebird_code is a ClassLabel — get the names list to map int→string
        ebird_feature = split.features.get("ebird_code")
        if ebird_feature is not None and hasattr(ebird_feature, "names"):
            label_names = ebird_feature.names
            log.info(f"  {dataset_name}: {len(label_names)} species: {label_names[:5]}...")
        else:
            log.error(f"  {dataset_name}: ebird_code feature has no names mapping")
            return []

        for idx in tqdm(range(len(split)), desc=f"  {dataset_name}", unit="sample"):
            try:
                sample = split[idx]

                # Get species name from ClassLabel index
                label_idx = sample.get("ebird_code")
                if label_idx is None or not isinstance(label_idx, int):
                    continue
                species = label_names[label_idx]

                # Load audio from filepath (audio field has decode=False)
                filepath = sample.get("filepath", "")
                if not filepath or not Path(filepath).exists():
                    # Try the audio dict's path field
                    audio_data = sample.get("audio", {})
                    if isinstance(audio_data, dict):
                        filepath = audio_data.get("path", "")
                    if not filepath or not Path(filepath).exists():
                        continue

                waveform, sr = torchaudio.load(filepath)

                # Mono
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                # Resample
                if sr != target_sr:
                    resampler = torchaudio.transforms.Resample(sr, target_sr)
                    waveform = resampler(waveform)

                total_samples = waveform.shape[1]
                n_clips = max(1, total_samples // clip_samples)

                species_dir = out_path / species
                species_dir.mkdir(parents=True, exist_ok=True)

                original_name = Path(filepath).stem

                for i in range(n_clips):
                    start = i * clip_samples
                    end = start + clip_samples
                    if end <= total_samples:
                        clip = waveform[:, start:end]
                        padded = False
                    else:
                        clip = torch.zeros(1, clip_samples)
                        remaining = total_samples - start
                        if remaining > 0:
                            clip[:, :remaining] = waveform[:, start:]
                        padded = True

                    # Energy filter
                    energy_threshold_db = audio_cfg.get("energy_threshold_db", -60.0)
                    if energy_threshold_db is not None:
                        rms = clip.float().pow(2).mean().sqrt()
                        rms_db = 20 * torch.log10(rms).item() if rms > 0 else -100.0
                        if rms_db < energy_threshold_db:
                            continue

                    clip_path = species_dir / f"pow_{original_name}_clip{i:03d}.wav"
                    torchaudio.save(str(clip_path), clip, target_sr)

                    all_clips.append({
                        "path": str(clip_path),
                        "species": species,
                        "duration": audio_cfg["clip_duration_s"],
                        "dataset_source": dataset_name,
                        "original_file": Path(filepath).name,
                        "clip_index": i,
                        "zero_padded": padded,
                    })

            except Exception as e:
                log.debug(f"  {dataset_name}: error on sample {idx}: {e}")
                continue

    return all_clips


# ---------------------------------------------------------------------------
# Watkins Marine Mammal Sound Database
# ---------------------------------------------------------------------------

def preprocess_watkins(
    raw_dir: str,
    processed_dir: str,
    audio_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Preprocess Watkins Marine Mammal Sound Database.

    Structure: audio/<Species_Name>/*.wav
    Species names are directory names like "Humpback_Whale", "Bottlenose_Dolphin".
    """
    raw_path = Path(raw_dir)
    out_path = Path(processed_dir)
    dataset_name = "watkins"
    ext_set = {e.lower() for e in audio_cfg["extensions"]} | {".aif", ".aiff"}

    # Watkins files are inside audio/ subdirectory, organized by species
    audio_root = raw_path / "audio"
    if not audio_root.exists():
        audio_root = raw_path
        log.warning(f"  {dataset_name}: no audio/ subdirectory, using {raw_path}")

    species_dirs = sorted([d for d in audio_root.iterdir() if d.is_dir()])
    if not species_dirs:
        log.error(f"  {dataset_name}: no species directories found in {audio_root}")
        return []

    log.info(f"  {dataset_name}: {len(species_dirs)} species directories in {audio_root}")

    all_clips: list[dict[str, Any]] = []

    for sp_dir in tqdm(species_dirs, desc=f"  {dataset_name}", unit="species"):
        # Species name from directory name, normalized
        species = sp_dir.name.lower().replace(" ", "_")

        audio_files = [f for f in sp_dir.rglob("*")
                       if f.is_file() and f.suffix.lower() in ext_set]

        for audio_file in audio_files:
            clips = process_audio_file(
                input_path=audio_file,
                output_dir=out_path,
                species=species,
                target_sr=audio_cfg["sample_rate"],
                clip_duration_s=audio_cfg["clip_duration_s"],
                max_clips=audio_cfg["max_clips_per_file"],
                min_duration_s=audio_cfg["min_duration_s"],
                energy_threshold_db=audio_cfg.get("energy_threshold_db", -60.0),
            )
            for clip in clips:
                clip["dataset_source"] = dataset_name
            all_clips.extend(clips)

    return all_clips


# ---------------------------------------------------------------------------
# AnuraSet (multi-label anuran vocalizations)
# ---------------------------------------------------------------------------

def preprocess_anuraset(
    raw_dir: str,
    processed_dir: str,
    audio_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Preprocess AnuraSet dataset.

    Structure:
        anuraset/metadata.csv — multi-label: rows are samples, species are columns
            Columns: sample_name, fname, min_t, max_t, site, date, species_number,
                     subset, SPHSUR, BOABIS, SCIPER, ... (0/1 per species)
        anuraset/audio/SAMPLE_00000.wav, SAMPLE_00001.wav, ...

    Each sample is a 3-second clip. Species columns use 6-letter codes
    (e.g., SPHSUR = Sphaenorhynchus surdus).
    Multiple species can be active per sample (multi-label).

    Strategy for single-label task vectors:
    - Samples with exactly 1 species → assigned to that species
    - Samples with multiple species → assigned to each active species
      (clip stored once per species, flagged as multi_label)
    - Samples with 0 species → skipped
    """
    raw_path = Path(raw_dir)
    out_path = Path(processed_dir)
    dataset_name = "anuraset"

    # Find metadata.csv
    meta_csv = None
    for candidate in [
        raw_path / "anuraset" / "metadata.csv",
        raw_path / "metadata.csv",
    ]:
        if candidate.exists():
            meta_csv = candidate
            break

    if meta_csv is None:
        log.error(f"  {dataset_name}: metadata.csv not found")
        return []

    # Find audio directory
    audio_dir = None
    for candidate in [
        meta_csv.parent / "audio",
        raw_path / "audio",
        raw_path / "anuraset" / "audio",
    ]:
        if candidate.exists() and candidate.is_dir():
            audio_dir = candidate
            break

    if audio_dir is None:
        log.error(f"  {dataset_name}: audio directory not found")
        return []

    log.info(f"  {dataset_name}: metadata at {meta_csv}")
    log.info(f"  {dataset_name}: audio at {audio_dir}")

    # Parse metadata CSV — identify species columns
    with open(meta_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    # Non-species metadata columns
    meta_columns = {
        "sample_name", "fname", "min_t", "max_t", "site", "date",
        "species_number", "subset",
    }
    species_columns = [c for c in fieldnames if c not in meta_columns]
    log.info(
        f"  {dataset_name}: {len(rows)} samples, {len(species_columns)} species: "
        f"{species_columns}"
    )

    target_sr = audio_cfg["sample_rate"]
    clip_samples = int(audio_cfg["clip_duration_s"] * target_sr)
    all_clips: list[dict[str, Any]] = []
    skipped_no_species = 0
    multi_label_count = 0

    for row in tqdm(rows, desc=f"  {dataset_name}", unit="sample"):
        sample_name = row.get("sample_name", "")
        if not sample_name:
            continue

        # Construct actual file path from metadata columns:
        #   audio/{site}/{fname}_{min_t}_{max_t}.wav
        fname = row.get("fname", "")
        min_t = row.get("min_t", "")
        max_t = row.get("max_t", "")
        site = row.get("site", "")

        audio_path = None
        if fname and min_t and max_t and site:
            audio_path = audio_dir / site / f"{fname}_{min_t}_{max_t}.wav"

        # Fallback: try sample_name directly
        if audio_path is None or not audio_path.exists():
            audio_path = audio_dir / sample_name
        if not audio_path.exists() and not sample_name.endswith(".wav"):
            audio_path = audio_dir / f"{sample_name}.wav"
        if not audio_path.exists():
            continue

        # Determine active species
        active_species = []
        for sp_col in species_columns:
            try:
                if int(row.get(sp_col, 0)) == 1:
                    active_species.append(sp_col.lower())
            except (ValueError, TypeError):
                continue

        if not active_species:
            skipped_no_species += 1
            continue

        if len(active_species) > 1:
            multi_label_count += 1

        # Load audio
        try:
            waveform, sr = torchaudio.load(str(audio_path))
        except Exception as e:
            log.debug(f"  {dataset_name}: failed to load {audio_path}: {e}")
            continue

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)

        total_samples = waveform.shape[1]

        # AnuraSet samples are 3s. Zero-pad to 5s clip duration.
        if total_samples < clip_samples:
            padded_waveform = torch.zeros(1, clip_samples)
            padded_waveform[:, :total_samples] = waveform
            waveform = padded_waveform
            padded = True
        else:
            waveform = waveform[:, :clip_samples]
            padded = False

        # Energy filter: skip near-silent samples
        energy_threshold_db = audio_cfg.get("energy_threshold_db", -60.0)
        if energy_threshold_db is not None:
            rms = waveform.float().pow(2).mean().sqrt()
            rms_db = 20 * torch.log10(rms).item() if rms > 0 else -100.0
            if rms_db < energy_threshold_db:
                continue

        # Save one copy per active species
        for species in active_species:
            species_dir = out_path / species
            species_dir.mkdir(parents=True, exist_ok=True)

            stem = audio_path.stem
            clip_path = species_dir / f"{stem}.wav"
            torchaudio.save(str(clip_path), waveform, target_sr)

            all_clips.append({
                "path": str(clip_path),
                "species": species,
                "duration": audio_cfg["clip_duration_s"],
                "dataset_source": dataset_name,
                "original_file": audio_path.name,
                "clip_index": 0,
                "zero_padded": padded,
                "multi_label": len(active_species) > 1,
                "all_species": active_species,
            })

    log.info(
        f"  {dataset_name}: {skipped_no_species} samples skipped (no active species), "
        f"{multi_label_count} multi-label samples"
    )

    return all_clips


# ---------------------------------------------------------------------------
# Manifest building
# ---------------------------------------------------------------------------

def build_manifest(
    clips: list[dict[str, Any]],
    dataset_name: str,
    manifest_path: str,
    min_samples_threshold: int = 10,
) -> dict[str, Any]:
    """Build and save a JSON manifest for a preprocessed dataset.

    Args:
        clips: List of clip metadata dicts from preprocessing.
        dataset_name: Name of the dataset.
        manifest_path: Output path for the JSON manifest.
        min_samples_threshold: Warn about species below this count.

    Returns:
        Summary statistics dict.
    """
    species_counts = Counter(c["species"] for c in clips)
    n_species = len(species_counts)

    low_sample_species = {
        sp: count for sp, count in species_counts.items()
        if count < min_samples_threshold
    }

    manifest = {
        "dataset": dataset_name,
        "total_clips": len(clips),
        "n_species": n_species,
        "species_counts": dict(sorted(species_counts.items())),
        "low_sample_species": dict(sorted(low_sample_species.items())),
        "clips": clips,
    }

    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    summary = {
        "dataset": dataset_name,
        "total_clips": len(clips),
        "n_species": n_species,
        "n_low_sample_species": len(low_sample_species),
    }

    log.info(f"  {dataset_name}: {len(clips)} clips, {n_species} species")
    if low_sample_species:
        log.warning(
            f"  {dataset_name}: {len(low_sample_species)} species with "
            f"<{min_samples_threshold} clips: "
            f"{list(low_sample_species.keys())[:10]}"
            f"{'...' if len(low_sample_species) > 10 else ''}"
        )
    log.info(f"  {dataset_name}: manifest → {manifest_path}")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess audio datasets into standardized 16kHz mono 5s clips.",
    )
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml",
        help="Path to base config YAML",
    )
    parser.add_argument(
        "--only", type=str, nargs="+", default=None,
        help="Process only these datasets",
    )
    parser.add_argument(
        "--skip", type=str, nargs="+", default=None,
        help="Skip these datasets",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    audio_cfg = cfg["audio"]
    datasets_cfg = cfg["datasets"]
    min_samples = cfg["splits"]["min_samples_per_species"]

    all_datasets = ["birdclef2023", "birdclef2024", "birdclef2025",
                    "birdset_pow", "watkins", "anuraset"]

    targets = all_datasets
    if args.only:
        targets = [d for d in args.only if d in all_datasets]
    if args.skip:
        targets = [d for d in targets if d not in args.skip]

    log.info(f"Preprocessing: {targets}")
    log.info(f"Audio: {audio_cfg['sample_rate']}Hz, {audio_cfg['clip_duration_s']}s clips")
    log.info("")

    summaries: list[dict[str, Any]] = []

    # --- BirdCLEF 2023 ---
    if "birdclef2023" in targets:
        ds_cfg = datasets_cfg["birdclef2023"]
        log.info(f"[birdclef2023] {ds_cfg['description']}")
        clips = preprocess_birdclef_2023_2024(
            raw_dir=ds_cfg["raw_dir"],
            processed_dir=ds_cfg["processed_dir"],
            dataset_name="birdclef2023",
            audio_cfg=audio_cfg,
        )
        if clips:
            summaries.append(build_manifest(
                clips, "birdclef2023", ds_cfg["manifest"], min_samples,
            ))

    # --- BirdCLEF 2024 ---
    if "birdclef2024" in targets:
        ds_cfg = datasets_cfg["birdclef2024"]
        log.info(f"[birdclef2024] {ds_cfg['description']}")
        clips = preprocess_birdclef_2023_2024(
            raw_dir=ds_cfg["raw_dir"],
            processed_dir=ds_cfg["processed_dir"],
            dataset_name="birdclef2024",
            audio_cfg=audio_cfg,
        )
        if clips:
            summaries.append(build_manifest(
                clips, "birdclef2024", ds_cfg["manifest"], min_samples,
            ))

    # --- BirdCLEF 2025 ---
    if "birdclef2025" in targets:
        ds_cfg = datasets_cfg["birdclef2025"]
        log.info(f"[birdclef2025] {ds_cfg['description']}")
        clips = preprocess_birdclef_2025(
            raw_dir=ds_cfg["raw_dir"],
            processed_dir=ds_cfg["processed_dir"],
            audio_cfg=audio_cfg,
        )
        if clips:
            summaries.append(build_manifest(
                clips, "birdclef2025", ds_cfg["manifest"], min_samples,
            ))

    # --- BirdSet POW ---
    if "birdset_pow" in targets:
        ds_cfg = datasets_cfg["birdset_pow"]
        log.info(f"[birdset_pow] {ds_cfg['description']}")
        clips = preprocess_birdset_pow(
            raw_dir=ds_cfg["raw_dir"],
            processed_dir=ds_cfg["processed_dir"],
            audio_cfg=audio_cfg,
        )
        if clips:
            summaries.append(build_manifest(
                clips, "birdset_pow", ds_cfg["manifest"], min_samples,
            ))

    # --- Watkins ---
    if "watkins" in targets:
        ds_cfg = datasets_cfg["watkins"]
        log.info(f"[watkins] {ds_cfg['description']}")
        clips = preprocess_watkins(
            raw_dir=ds_cfg["raw_dir"],
            processed_dir=ds_cfg["processed_dir"],
            audio_cfg=audio_cfg,
        )
        if clips:
            summaries.append(build_manifest(
                clips, "watkins", ds_cfg["manifest"], min_samples,
            ))

    # --- AnuraSet ---
    if "anuraset" in targets:
        ds_cfg = datasets_cfg["anuraset"]
        log.info(f"[anuraset] {ds_cfg['description']}")
        clips = preprocess_anuraset(
            raw_dir=ds_cfg["raw_dir"],
            processed_dir=ds_cfg["processed_dir"],
            audio_cfg=audio_cfg,
        )
        if clips:
            summaries.append(build_manifest(
                clips, "anuraset", ds_cfg["manifest"], min_samples,
            ))

    # --- Summary ---
    log.info("")
    log.info("=" * 60)
    log.info("PREPROCESSING SUMMARY")
    log.info("=" * 60)
    total_clips = 0
    total_species = 0
    for s in summaries:
        low = f"  ⚠ {s['n_low_sample_species']} low-sample" if s["n_low_sample_species"] > 0 else ""
        log.info(
            f"  {s['dataset']:20s}: {s['total_clips']:>8,} clips, "
            f"{s['n_species']:>5} species{low}"
        )
        total_clips += s["total_clips"]
        total_species += s["n_species"]

    log.info(f"  {'TOTAL':20s}: {total_clips:>8,} clips, {total_species:>5} species (with overlap)")

    summary_path = Path(cfg["paths"]["data_manifests"]) / "preprocess_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    log.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()