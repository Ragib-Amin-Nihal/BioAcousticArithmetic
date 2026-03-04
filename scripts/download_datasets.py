#!/usr/bin/env python3
"""Download all datasets for bioacoustic task arithmetic experiments.

Downloads:
  - BirdCLEF 2023/2024/2025 via Kaggle API
  - BirdSet POW subset via HuggingFace datasets
  - Watkins Marine Mammal Sound Database via WHOI scraper
  - AnuraSet from Zenodo

Usage:
    python scripts/download_datasets.py --config configs/base.yaml
    python scripts/download_datasets.py --config configs/base.yaml --only birdclef2023
    python scripts/download_datasets.py --config configs/base.yaml --skip watkins

Prerequisites:
    - Kaggle credentials in ~/.kaggle/kaggle.json
    - Internet access
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import requests
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML config and resolve paths relative to project root."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg["paths"]["project_root"]).resolve()
    # Resolve all path values
    for k, v in cfg["paths"].items():
        if k != "project_root" and isinstance(v, str):
            cfg["paths"][k] = str(root / v)
    for ds_cfg in cfg["datasets"].values():
        if isinstance(ds_cfg, dict):
            for k in ["raw_dir", "processed_dir", "manifest"]:
                if k in ds_cfg:
                    ds_cfg[k] = str(root / ds_cfg[k])
    return cfg


def count_files(directory: str | Path, extensions: list[str] | None = None) -> int:
    """Count files in directory tree, optionally filtered by extension."""
    directory = Path(directory)
    if not directory.exists():
        return 0
    if extensions:
        ext_set = {e.lower() for e in extensions}
        return sum(1 for f in directory.rglob("*") if f.suffix.lower() in ext_set)
    return sum(1 for f in directory.rglob("*") if f.is_file())


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress reporting. Returns True on success."""
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        dest.parent.mkdir(parents=True, exist_ok=True)
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)",
                          end="", flush=True)
        print()
        return True
    except Exception as e:
        log.error(f"Download failed: {e}")
        return False


# ---------------------------------------------------------------------------
# BirdCLEF downloads (Kaggle)
# ---------------------------------------------------------------------------

def download_birdclef(
    competition: str,
    raw_dir: str,
    description: str,
) -> dict[str, Any]:
    """Download a BirdCLEF competition dataset via Kaggle API.

    Args:
        competition: Kaggle competition slug (e.g. 'birdclef-2023')
        raw_dir: Directory to download and extract into
        description: Human-readable dataset description

    Returns:
        Status dict with file counts and any errors.
    """
    raw_path = Path(raw_dir)
    status: dict[str, Any] = {
        "dataset": competition,
        "description": description,
        "raw_dir": raw_dir,
        "success": False,
        "error": None,
    }

    # Check if already downloaded — look for train_audio or train_metadata.csv
    audio_extensions = [".ogg", ".wav", ".mp3", ".flac"]
    existing = count_files(raw_path, audio_extensions)
    if existing > 100:
        log.info(f"  {competition}: already downloaded ({existing} audio files)")
        status["success"] = True
        status["audio_files"] = existing
        return status

    # Check kaggle credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        env_user = os.environ.get("KAGGLE_USERNAME")
        env_key = os.environ.get("KAGGLE_KEY")
        if not (env_user and env_key):
            status["error"] = (
                "Kaggle credentials not found. Either place kaggle.json in "
                "~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY env vars."
            )
            log.error(f"  {competition}: {status['error']}")
            return status

    raw_path.mkdir(parents=True, exist_ok=True)

    log.info(f"  {competition}: downloading via Kaggle API...")
    try:
        result = subprocess.run(
            [
                "kaggle", "competitions", "download",
                "-c", competition,
                "-p", str(raw_path),
            ],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout for large datasets
        )
        if result.returncode != 0:
            # Check if it's an acceptance error
            if "403" in result.stderr or "accept" in result.stderr.lower():
                status["error"] = (
                    f"Kaggle returned 403. You must accept competition rules at "
                    f"https://www.kaggle.com/competitions/{competition}/rules "
                    f"before downloading."
                )
            else:
                status["error"] = f"kaggle CLI failed: {result.stderr[:500]}"
            log.error(f"  {competition}: {status['error']}")
            return status

    except FileNotFoundError:
        status["error"] = "kaggle CLI not found. Install with: pip install kaggle"
        log.error(f"  {competition}: {status['error']}")
        return status
    except subprocess.TimeoutExpired:
        status["error"] = "Download timed out after 1 hour"
        log.error(f"  {competition}: {status['error']}")
        return status

    # Extract zip files
    for zf in raw_path.glob("*.zip"):
        log.info(f"  {competition}: extracting {zf.name}...")
        try:
            with zipfile.ZipFile(zf, "r") as z:
                z.extractall(raw_path)
            zf.unlink()  # Remove zip after extraction
        except zipfile.BadZipFile:
            log.warning(f"  {competition}: corrupt zip {zf.name}, skipping")

    audio_count = count_files(raw_path, audio_extensions)
    status["success"] = audio_count > 0
    status["audio_files"] = audio_count
    log.info(f"  {competition}: {audio_count} audio files downloaded")

    # Count species from directory structure
    train_audio = raw_path / "train_audio"
    if train_audio.exists():
        species_dirs = [d for d in train_audio.iterdir() if d.is_dir()]
        status["n_species_dirs"] = len(species_dirs)
        log.info(f"  {competition}: {len(species_dirs)} species directories")

    return status


# ---------------------------------------------------------------------------
# BirdSet POW (HuggingFace)
# ---------------------------------------------------------------------------

def download_birdset_pow(
    raw_dir: str,
    description: str,
) -> dict[str, Any]:
    """Download BirdSet POW subset via HuggingFace datasets library.

    The HF dataset stores audio as arrays. We save the raw dataset object
    to disk here; audio extraction to WAV files happens in preprocess_audio.py.

    Args:
        raw_dir: Directory for HuggingFace cache
        description: Human-readable description

    Returns:
        Status dict.
    """
    status: dict[str, Any] = {
        "dataset": "birdset_pow",
        "description": description,
        "raw_dir": raw_dir,
        "success": False,
        "error": None,
    }

    raw_path = Path(raw_dir)
    marker = raw_path / ".birdset_pow_downloaded"

    if marker.exists():
        log.info("  birdset_pow: already downloaded")
        status["success"] = True
        return status

    raw_path.mkdir(parents=True, exist_ok=True)

    log.info("  birdset_pow: downloading via HuggingFace datasets...")
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "DBD-research-group/BirdSet",
            "POW",
            trust_remote_code=True,
            cache_dir=str(raw_path / "hf_cache"),
        )

        # Save split sizes for verification
        split_info = {}
        for split_name in ds:
            split_info[split_name] = len(ds[split_name])
            log.info(f"  birdset_pow: {split_name} = {len(ds[split_name])} samples")

        # Save metadata
        meta = {
            "splits": split_info,
            "cache_dir": str(raw_path / "hf_cache"),
            "hf_dataset": "DBD-research-group/BirdSet",
            "hf_subset": "POW",
        }
        with open(raw_path / "birdset_pow_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        marker.touch()
        status["success"] = True
        status["splits"] = split_info

    except Exception as e:
        status["error"] = f"HuggingFace download failed: {e}"
        log.error(f"  birdset_pow: {status['error']}")

    return status


# ---------------------------------------------------------------------------
# Watkins Marine Mammal Sound Database
# ---------------------------------------------------------------------------

def download_watkins(
    raw_dir: str,
    external_dir: str,
    description: str,
) -> dict[str, Any]:
    """Download Watkins Marine Mammal Sound Database.

    Attempts to use the getWHOIdata scraper if available. Falls back to
    providing manual download instructions, since the WHOI website structure
    is fragile and the scraper may not work.

    Args:
        raw_dir: Target directory for downloaded audio
        external_dir: Path to external repos (for getWHOIdata)
        description: Human-readable description

    Returns:
        Status dict.
    """
    status: dict[str, Any] = {
        "dataset": "watkins",
        "description": description,
        "raw_dir": raw_dir,
        "success": False,
        "error": None,
    }

    raw_path = Path(raw_dir)
    audio_extensions = [".wav", ".aif", ".aiff", ".mp3"]
    existing = count_files(raw_path, audio_extensions)
    if existing > 50:
        log.info(f"  watkins: already downloaded ({existing} audio files)")
        status["success"] = True
        status["audio_files"] = existing
        return status

    raw_path.mkdir(parents=True, exist_ok=True)

    # Try the getWHOIdata scraper
    scraper_dir = Path(external_dir) / "getWHOIdata"
    scraper_script = scraper_dir / "getdata.py"

    if scraper_script.exists():
        log.info("  watkins: attempting download via getWHOIdata scraper...")
        log.info("  watkins: NOTE — this scraper depends on WHOI website structure")
        log.info("  watkins: and may fail if the site has changed.")
        try:
            # The getWHOIdata script downloads to the current directory
            result = subprocess.run(
                [sys.executable, str(scraper_script), "--bestof"],
                cwd=str(raw_path),
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min timeout
            )
            if result.returncode != 0:
                log.warning(f"  watkins: scraper returned non-zero: {result.stderr[:300]}")
            else:
                audio_count = count_files(raw_path, audio_extensions)
                if audio_count > 0:
                    log.info(f"  watkins: {audio_count} files downloaded via scraper")
                    status["success"] = True
                    status["audio_files"] = audio_count
                    return status

        except subprocess.TimeoutExpired:
            log.warning("  watkins: scraper timed out")
        except Exception as e:
            log.warning(f"  watkins: scraper failed: {e}")

    # Fallback: try direct download of a known subset
    # The "Best of Cuts" WAV files are sometimes directly accessible
    log.warning("  watkins: automated download unavailable or failed.")
    log.warning("  watkins: MANUAL DOWNLOAD REQUIRED:")
    log.warning("  watkins:   1. Visit https://cis.whoi.edu/science/B/whalesounds/index.cfm")
    log.warning("  watkins:   2. Navigate to 'Best of Cuts' and download WAV files")
    log.warning(f"  watkins:   3. Save to: {raw_dir}")
    log.warning("  watkins:   4. Re-run this script to verify")

    status["error"] = "Automated download failed. Manual download required (see log)."
    # Mark as partial success if any files exist
    if existing > 0:
        status["success"] = True
        status["audio_files"] = existing

    return status


# ---------------------------------------------------------------------------
# AnuraSet (Zenodo)
# ---------------------------------------------------------------------------

def download_anuraset(
    raw_dir: str,
    zenodo_url: str,
    description: str,
) -> dict[str, Any]:
    """Download AnuraSet processed dataset from Zenodo.

    Downloads the v2 processed dataset (zenodo 8342596) which contains
    pre-segmented 3-second WAV clips with species labels.

    Args:
        raw_dir: Target directory
        zenodo_url: Base Zenodo URL for the record files
        description: Human-readable description

    Returns:
        Status dict.
    """
    status: dict[str, Any] = {
        "dataset": "anuraset",
        "description": description,
        "raw_dir": raw_dir,
        "success": False,
        "error": None,
    }

    raw_path = Path(raw_dir)
    audio_extensions = [".wav", ".flac"]
    existing = count_files(raw_path, audio_extensions)
    if existing > 100:
        log.info(f"  anuraset: already downloaded ({existing} audio files)")
        status["success"] = True
        status["audio_files"] = existing
        return status

    raw_path.mkdir(parents=True, exist_ok=True)

    # AnuraSet v2 processed dataset — contains labeled 3s segments
    # Zenodo record 8342596 has the processed version
    # The raw data is at 8056090
    files_to_download = {
        "anuraset_processed.zip": f"{zenodo_url}/anuraset.zip",
    }

    # Also try the raw data record as fallback
    zenodo_raw_url = "https://zenodo.org/records/8056090/files"

    for filename, url in files_to_download.items():
        dest = raw_path / filename
        if dest.exists():
            log.info(f"  anuraset: {filename} already downloaded")
            continue

        log.info(f"  anuraset: downloading {filename}...")
        success = download_file(url, dest)

        if not success:
            # Try alternate URL structure
            alt_url = f"{zenodo_raw_url}/anuraset.zip"
            log.info(f"  anuraset: trying alternate URL...")
            success = download_file(alt_url, dest)

        if not success:
            # Try with Zenodo API
            for record_id in ["8342596", "8056090"]:
                api_url = f"https://zenodo.org/api/records/{record_id}"
                log.info(f"  anuraset: querying Zenodo API for record {record_id}...")
                try:
                    resp = requests.get(api_url, timeout=30)
                    if resp.ok:
                        record = resp.json()
                        for file_entry in record.get("files", []):
                            if file_entry["key"].endswith(".zip"):
                                dl_link = file_entry["links"]["self"]
                                log.info(f"  anuraset: found {file_entry['key']} → {dl_link}")
                                success = download_file(dl_link, dest)
                                if success:
                                    break
                except Exception as e:
                    log.warning(f"  anuraset: Zenodo API query failed: {e}")
                if success:
                    break

        if not success:
            status["error"] = (
                f"Failed to download {filename}. "
                f"Manual download: visit https://zenodo.org/records/8342596 "
                f"or https://zenodo.org/records/8056090 and save to {raw_dir}"
            )
            log.error(f"  anuraset: {status['error']}")
            return status

    # Extract zips
    for zf in raw_path.glob("*.zip"):
        log.info(f"  anuraset: extracting {zf.name}...")
        try:
            with zipfile.ZipFile(zf, "r") as z:
                z.extractall(raw_path)
        except zipfile.BadZipFile:
            log.warning(f"  anuraset: corrupt zip {zf.name}")
            status["error"] = f"Corrupt zip file: {zf.name}"
            return status

    audio_count = count_files(raw_path, audio_extensions)
    status["success"] = audio_count > 0
    status["audio_files"] = audio_count
    log.info(f"  anuraset: {audio_count} audio files available")

    return status


# ---------------------------------------------------------------------------
# eBird Taxonomy (utility — needed by create_species_groups.py)
# ---------------------------------------------------------------------------

def download_ebird_taxonomy(data_raw_dir: str) -> dict[str, Any]:
    """Download eBird taxonomy CSV for species-to-order mapping.

    Requires EBIRD_API_KEY env var. The taxonomy is a ~6MB CSV with columns:
    speciesCode, comName, sciName, familyComName, familySciName, order, etc.

    Args:
        data_raw_dir: Base raw data directory

    Returns:
        Status dict with path to taxonomy CSV.
    """
    status: dict[str, Any] = {
        "dataset": "ebird_taxonomy",
        "success": False,
        "error": None,
    }

    out_dir = Path(data_raw_dir) / "ebird_taxonomy"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "ebird_taxonomy.csv"

    if out_file.exists() and out_file.stat().st_size > 100_000:
        log.info(f"  ebird_taxonomy: already downloaded ({out_file.stat().st_size / 1e6:.1f} MB)")
        status["success"] = True
        status["path"] = str(out_file)
        return status

    api_key = os.environ.get("EBIRD_API_KEY", "")
    if not api_key:
        # Also check .env file
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("EBIRD_API_KEY=") and len(line.split("=", 1)[1].strip()) > 0:
                    api_key = line.split("=", 1)[1].strip()
                    break

    if not api_key:
        status["error"] = (
            "EBIRD_API_KEY not set. Request a key at https://ebird.org/api/keygen "
            "and set it in .env or as an environment variable. "
            "Species grouping will fall back to heuristic taxonomy mapping."
        )
        log.warning(f"  ebird_taxonomy: {status['error']}")
        return status

    log.info("  ebird_taxonomy: downloading from eBird API...")
    try:
        resp = requests.get(
            "https://api.ebird.org/v2/ref/taxonomy/ebird",
            headers={"x-ebirdapitoken": api_key},
            params={"fmt": "csv"},
            timeout=60,
        )
        resp.raise_for_status()
        out_file.write_text(resp.text, encoding="utf-8")
        n_lines = resp.text.count("\n")
        log.info(f"  ebird_taxonomy: saved ({n_lines} entries, {out_file.stat().st_size / 1e6:.1f} MB)")
        status["success"] = True
        status["path"] = str(out_file)
        status["n_entries"] = n_lines

    except Exception as e:
        status["error"] = f"eBird API request failed: {e}"
        log.error(f"  ebird_taxonomy: {status['error']}")

    return status


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download datasets for bioacoustic task arithmetic experiments.",
    )
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml",
        help="Path to base config YAML",
    )
    parser.add_argument(
        "--only", type=str, nargs="+", default=None,
        help="Download only these datasets (e.g. --only birdclef2023 anuraset)",
    )
    parser.add_argument(
        "--skip", type=str, nargs="+", default=None,
        help="Skip these datasets (e.g. --skip watkins)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    datasets_cfg = cfg["datasets"]

    # all_datasets = ["birdclef2023", "birdclef2024", "birdclef2025",
    #                 "birdset_pow", "watkins", "anuraset", "ebird_taxonomy"]
    all_datasets = ["watkins"]

    if args.only:
        targets = [d for d in args.only if d in all_datasets]
        if not targets:
            log.error(f"No valid datasets in --only. Options: {all_datasets}")
            sys.exit(1)
    else:
        targets = all_datasets

    if args.skip:
        targets = [d for d in targets if d not in args.skip]

    log.info(f"Datasets to download: {targets}")
    log.info(f"Config: {args.config}")
    log.info("")

    results: dict[str, dict[str, Any]] = {}

    # --- BirdCLEF competitions ---
    for ds_name in ["birdclef2023", "birdclef2024", "birdclef2025"]:
        if ds_name not in targets:
            continue
        ds_cfg = datasets_cfg[ds_name]
        log.info(f"[{ds_name}] {ds_cfg['description']}")
        results[ds_name] = download_birdclef(
            competition=ds_cfg["kaggle_competition"],
            raw_dir=ds_cfg["raw_dir"],
            description=ds_cfg["description"],
        )

    # --- BirdSet POW ---
    if "birdset_pow" in targets:
        ds_cfg = datasets_cfg["birdset_pow"]
        log.info(f"[birdset_pow] {ds_cfg['description']}")
        results["birdset_pow"] = download_birdset_pow(
            raw_dir=ds_cfg["raw_dir"],
            description=ds_cfg["description"],
        )

    # --- Watkins ---
    if "watkins" in targets:
        ds_cfg = datasets_cfg["watkins"]
        log.info(f"[watkins] {ds_cfg['description']}")
        results["watkins"] = download_watkins(
            raw_dir=ds_cfg["raw_dir"],
            external_dir=cfg["paths"]["external"],
            description=ds_cfg["description"],
        )

    # --- AnuraSet ---
    if "anuraset" in targets:
        ds_cfg = datasets_cfg["anuraset"]
        log.info(f"[anuraset] {ds_cfg['description']}")
        results["anuraset"] = download_anuraset(
            raw_dir=ds_cfg["raw_dir"],
            zenodo_url=ds_cfg["zenodo_url"],
            description=ds_cfg["description"],
        )

    # --- eBird taxonomy ---
    if "ebird_taxonomy" in targets:
        log.info("[ebird_taxonomy] eBird species taxonomy for grouping")
        results["ebird_taxonomy"] = download_ebird_taxonomy(cfg["paths"]["data_raw"])

    # --- Summary ---
    log.info("")
    log.info("=" * 60)
    log.info("DOWNLOAD SUMMARY")
    log.info("=" * 60)

    success_count = 0
    fail_count = 0
    for ds_name, status in results.items():
        icon = "✓" if status.get("success") else "✗"
        msg = f"  {icon} {ds_name}"
        if "audio_files" in status:
            msg += f" — {status['audio_files']} audio files"
        if "splits" in status:
            msg += f" — splits: {status['splits']}"
        if "n_entries" in status:
            msg += f" — {status['n_entries']} taxonomy entries"
        if status.get("error"):
            msg += f"\n    ERROR: {status['error']}"
        log.info(msg)
        if status.get("success"):
            success_count += 1
        else:
            fail_count += 1

    log.info("")
    log.info(f"Succeeded: {success_count}/{len(results)}, Failed: {fail_count}/{len(results)}")

    # Save download report
    report_path = Path(cfg["paths"]["data_manifests"]) / "download_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Report saved to {report_path}")

    if fail_count > 0:
        log.warning("Some downloads failed. Check errors above and re-run with --only <dataset>.")
        sys.exit(1)


if __name__ == "__main__":
    main()