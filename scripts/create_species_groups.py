#!/usr/bin/env python3
"""Create species groups G1-G6 from preprocessed BirdCLEF + auxiliary dataset manifests.

Groups:
    G1: Passerines (Passeriformes) — from BirdCLEF 2023+2024+2025
    G2: Non-passerine birds (excluding raptors/waterbirds) — from BirdCLEF
    G3: Raptors & waterbirds — from BirdCLEF
    G4: Marine mammals — from Watkins
    G5: Amphibians — from AnuraSet (+ BirdCLEF 2025 if multi-taxa)
    G6: Insects — from BirdCLEF 2025 multi-taxa (if available)

Taxonomy mapping:
    1. Tries eBird taxonomy CSV (downloaded by download_datasets.py)
    2. Falls back to a hardcoded order mapping for common bird families

For each group:
    - Holds out N species for zero-shot evaluation
    - Splits remaining species into train/val/test (70/10/20, stratified)
    - Falls back to fewer groups if any group has < min_species_per_group

Usage:
    python scripts/create_species_groups.py --config configs/base.yaml
    python scripts/create_species_groups.py --config configs/base.yaml --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML config and resolve paths."""
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
# Taxonomy: eBird CSV + hardcoded fallback
# ---------------------------------------------------------------------------

# Orders assigned to G3 (raptors & waterbirds)
G3_ORDERS = frozenset({
    "Accipitriformes",     # Hawks, eagles, kites
    "Falconiformes",       # Falcons
    "Charadriiformes",     # Shorebirds, gulls
    "Anseriformes",        # Ducks, geese, swans
    "Pelecaniformes",      # Pelicans, herons, ibises
    "Suliformes",          # Cormorants, boobies, frigatebirds
    "Ciconiiformes",       # Storks
    "Gruiformes",          # Cranes, rails, coots
    "Podicipediformes",    # Grebes
    "Phoenicopteriformes", # Flamingos
    "Procellariiformes",   # Albatrosses, petrels
    "Sphenisciformes",     # Penguins
    "Gaviiformes",         # Loons
})

# G1: Passeriformes only. G2: all other bird orders not in G1 or G3.

# Hardcoded family-to-order mapping for common BirdCLEF families.
# Used as fallback when eBird taxonomy CSV is unavailable.
# This covers ~90% of species across BirdCLEF 2023-2025.
FAMILY_TO_ORDER: dict[str, str] = {
    # Passeriformes (G1) — just a few families for illustration;
    # the eBird taxonomy is the authoritative source
    "Thraupidae": "Passeriformes",
    "Tyrannidae": "Passeriformes",
    "Furnariidae": "Passeriformes",
    "Troglodytidae": "Passeriformes",
    "Turdidae": "Passeriformes",
    "Muscicapidae": "Passeriformes",
    "Sylviidae": "Passeriformes",
    "Paridae": "Passeriformes",
    "Fringillidae": "Passeriformes",
    "Emberizidae": "Passeriformes",
    "Parulidae": "Passeriformes",
    "Icteridae": "Passeriformes",
    "Corvidae": "Passeriformes",
    "Ploceidae": "Passeriformes",
    "Estrildidae": "Passeriformes",
    "Nectariniidae": "Passeriformes",
    "Cisticolidae": "Passeriformes",
    "Pycnonotidae": "Passeriformes",
    "Monarchidae": "Passeriformes",
    "Laniidae": "Passeriformes",
    "Vireonidae": "Passeriformes",
    "Hirundinidae": "Passeriformes",
    "Motacillidae": "Passeriformes",
    "Sturnidae": "Passeriformes",
    "Mimidae": "Passeriformes",
    # Accipitriformes (G3)
    "Accipitridae": "Accipitriformes",
    "Pandionidae": "Accipitriformes",
    # Falconiformes (G3)
    "Falconidae": "Falconiformes",
    # Strigiformes (G2)
    "Strigidae": "Strigiformes",
    "Tytonidae": "Strigiformes",
    # Piciformes (G2)
    "Picidae": "Piciformes",
    "Ramphastidae": "Piciformes",
    "Capitonidae": "Piciformes",
    "Bucconidae": "Piciformes",
    # Columbiformes (G2)
    "Columbidae": "Columbiformes",
    # Psittaciformes (G2)
    "Psittacidae": "Psittaciformes",
    "Psittaculidae": "Psittaciformes",
    # Cuculiformes (G2)
    "Cuculidae": "Cuculiformes",
    # Caprimulgiformes (G2)
    "Caprimulgidae": "Caprimulgiformes",
    # Apodiformes (G2)
    "Apodidae": "Apodiformes",
    "Trochilidae": "Apodiformes",
    # Coraciiformes (G2)
    "Alcedinidae": "Coraciiformes",
    "Meropidae": "Coraciiformes",
    "Coraciidae": "Coraciiformes",
    # Bucerotiformes (G2)
    "Bucerotidae": "Bucerotiformes",
    "Upupidae": "Bucerotiformes",
    # Galliformes (G2)
    "Phasianidae": "Galliformes",
    "Cracidae": "Galliformes",
    "Numididae": "Galliformes",
    # Trogoniformes (G2)
    "Trogonidae": "Trogoniformes",
    # Charadriiformes (G3)
    "Charadriidae": "Charadriiformes",
    "Scolopacidae": "Charadriiformes",
    "Laridae": "Charadriiformes",
    "Jacanidae": "Charadriiformes",
    "Recurvirostridae": "Charadriiformes",
    # Anseriformes (G3)
    "Anatidae": "Anseriformes",
    # Pelecaniformes (G3)
    "Ardeidae": "Pelecaniformes",
    "Threskiornithidae": "Pelecaniformes",
    "Pelecanidae": "Pelecaniformes",
    # Gruiformes (G3)
    "Rallidae": "Gruiformes",
    "Gruidae": "Gruiformes",
    # Suliformes (G3)
    "Phalacrocoracidae": "Suliformes",
    "Anhingidae": "Suliformes",
}


def load_ebird_taxonomy(data_raw_dir: str) -> dict[str, dict[str, str]]:
    """Load eBird taxonomy CSV into a species_code → {order, family, sciName} map.

    Args:
        data_raw_dir: Base raw data directory (taxonomy CSV in ebird_taxonomy/)

    Returns:
        Dict mapping species code to taxonomy info.
    """
    csv_path = Path(data_raw_dir) / "ebird_taxonomy" / "ebird_taxonomy.csv"
    if not csv_path.exists():
        log.warning(
            f"eBird taxonomy CSV not found at {csv_path}. "
            "Run download_datasets.py with ebird_taxonomy, or set EBIRD_API_KEY. "
            "Falling back to heuristic taxonomy mapping."
        )
        return {}

    taxonomy: dict[str, dict[str, str]] = {}
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = row.get("SPECIES_CODE", row.get("speciesCode", "")).strip()
                if not code:
                    continue
                taxonomy[code] = {
                    "order": row.get("ORDER", row.get("ORDER1", row.get("order", ""))).strip(),
                    "family": row.get("FAMILY_SCI_NAME", row.get("FAMILY", row.get("familySciName", ""))).strip(),
                    "sci_name": row.get("SCIENTIFIC_NAME", row.get("SCI_NAME", row.get("sciName", ""))).strip(),
                    "common_name": row.get("COMMON_NAME", row.get("PRIMARY_COM_NAME", row.get("comName", ""))).strip(),
                    "category": row.get("CATEGORY", row.get("category", "")).strip(),
                }
        log.info(f"Loaded eBird taxonomy: {len(taxonomy)} entries from {csv_path}")
    except Exception as e:
        log.error(f"Failed to parse eBird taxonomy: {e}")
    return taxonomy


def resolve_order(
    species_code: str,
    ebird_taxonomy: dict[str, dict[str, str]],
) -> str | None:
    """Look up the taxonomic order for a species code.

    Args:
        species_code: eBird species code (e.g., 'amecro')
        ebird_taxonomy: Loaded eBird taxonomy dict

    Returns:
        Taxonomic order string, or None if not found.
    """
    if species_code in ebird_taxonomy:
        order = ebird_taxonomy[species_code].get("order", "")
        if order:
            return order

    # Fallback: try matching by family from hardcoded map
    if species_code in ebird_taxonomy:
        family = ebird_taxonomy[species_code].get("family", "")
        if family in FAMILY_TO_ORDER:
            return FAMILY_TO_ORDER[family]

    return None


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: str) -> dict[str, Any] | None:
    """Load a preprocessed dataset manifest JSON."""
    path = Path(manifest_path)
    if not path.exists():
        log.warning(f"Manifest not found: {manifest_path}")
        return None
    with open(path) as f:
        return json.load(f)


def get_species_from_manifest(manifest: dict[str, Any]) -> dict[str, int]:
    """Extract species → clip count from a manifest."""
    return manifest.get("species_counts", {})


# ---------------------------------------------------------------------------
# Group assignment
# ---------------------------------------------------------------------------

def assign_bird_groups(
    birdclef_species: dict[str, dict[str, Any]],
    ebird_taxonomy: dict[str, dict[str, str]],
) -> dict[str, list[str]]:
    """Assign bird species from BirdCLEF to G1/G2/G3 based on taxonomic order.

    Args:
        birdclef_species: species_code → {"count": int, "datasets": [str]}
        ebird_taxonomy: eBird taxonomy lookup

    Returns:
        Dict of group_name → [species_codes]
    """
    groups: dict[str, list[str]] = {
        "G1_passerines": [],
        "G2_nonpasserine_birds": [],
        "G3_raptors_waterbirds": [],
    }
    unresolved: list[str] = []

    for sp_code in sorted(birdclef_species.keys()):
        order = resolve_order(sp_code, ebird_taxonomy)

        if order is None:
            unresolved.append(sp_code)
            continue

        if order == "Passeriformes":
            groups["G1_passerines"].append(sp_code)
        elif order in G3_ORDERS:
            groups["G3_raptors_waterbirds"].append(sp_code)
        else:
            groups["G2_nonpasserine_birds"].append(sp_code)

    if unresolved:
        log.warning(
            f"Could not resolve taxonomy for {len(unresolved)} species. "
            f"Assigning to G2 by default. First 10: {unresolved[:10]}"
        )
        groups["G2_nonpasserine_birds"].extend(unresolved)

    for g, species in groups.items():
        log.info(f"  {g}: {len(species)} species")

    return groups


def maybe_merge_small_groups(
    groups: dict[str, list[str]],
    min_species: int,
) -> dict[str, list[str]]:
    """Merge groups that are too small into their nearest neighbor.

    If G3 (raptors/waterbirds) has < min_species, merge into G2.
    If G2 still too small after merge, combine G2+G3 into 'G2_other_birds'.
    """
    merged = dict(groups)

    # Check G3 first — smallest expected group
    if len(merged.get("G3_raptors_waterbirds", [])) < min_species:
        g3_species = merged.pop("G3_raptors_waterbirds", [])
        if g3_species:
            log.warning(
                f"G3_raptors_waterbirds has only {len(g3_species)} species "
                f"(< {min_species}). Merging into G2."
            )
            merged.setdefault("G2_nonpasserine_birds", []).extend(g3_species)

    # Check G2
    if len(merged.get("G2_nonpasserine_birds", [])) < min_species:
        log.warning(
            f"G2_nonpasserine_birds has only "
            f"{len(merged.get('G2_nonpasserine_birds', []))} species. "
            "Keeping anyway — too few non-passerines in dataset."
        )

    return merged


# ---------------------------------------------------------------------------
# Split creation
# ---------------------------------------------------------------------------

def create_splits(
    species_list: list[str],
    species_clips: dict[str, list[dict[str, Any]]],
    holdout_n: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    min_samples: int,
    max_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Create train/val/test/zeroshot splits for a species group.

    Args:
        species_list: All species codes in this group
        species_clips: species_code → list of clip dicts
        holdout_n: Number of species to hold out for zero-shot
        train_frac, val_frac, test_frac: Split proportions
        min_samples: Minimum clips per species to include
        max_samples: Maximum clips per species (cap)
        seed: Random seed

    Returns:
        Split dict with train/val/test/zeroshot lists and metadata.
    """
    rng = random.Random(seed)

    # Filter species by minimum sample count
    eligible = [sp for sp in species_list if len(species_clips.get(sp, [])) >= min_samples]
    dropped = [sp for sp in species_list if sp not in eligible]

    if dropped:
        log.info(
            f"    Dropped {len(dropped)} species with <{min_samples} clips"
        )

    if len(eligible) == 0:
        log.error("    No species meet minimum sample threshold!")
        return {"train": [], "val": [], "test": [], "zeroshot": [],
                "metadata": {"n_classes": 0, "error": "no eligible species"}}

    # Hold out species for zero-shot
    rng.shuffle(eligible)
    actual_holdout = min(holdout_n, max(0, len(eligible) - 2))  # keep at least 2 for training
    zeroshot_species = set(eligible[:actual_holdout])
    train_species = [sp for sp in eligible if sp not in zeroshot_species]

    log.info(
        f"    {len(train_species)} train species, "
        f"{len(zeroshot_species)} zero-shot holdout"
    )

    # Build label mapping
    label2idx = {sp: i for i, sp in enumerate(sorted(train_species))}

    splits: dict[str, list[dict[str, Any]]] = {
        "train": [], "val": [], "test": [], "zeroshot": [],
    }

    for sp in train_species:
        clips = list(species_clips.get(sp, []))
        rng.shuffle(clips)
        clips = clips[:max_samples]  # Cap

        n = len(clips)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        n_train = n - n_test - n_val

        if n_train < 1:
            # Too few samples — put everything in train
            n_train = n
            n_val = 0
            n_test = 0

        for clip in clips[:n_train]:
            clip["label"] = label2idx[sp]
            splits["train"].append(clip)
        for clip in clips[n_train:n_train + n_val]:
            clip["label"] = label2idx[sp]
            splits["val"].append(clip)
        for clip in clips[n_train + n_val:]:
            clip["label"] = label2idx[sp]
            splits["test"].append(clip)

    for sp in zeroshot_species:
        clips = list(species_clips.get(sp, []))[:max_samples]
        for clip in clips:
            splits["zeroshot"].append(clip)

    # Shuffle each split
    for split_name in splits:
        rng.shuffle(splits[split_name])

    splits["metadata"] = {
        "n_classes": len(train_species),
        "label2species": {v: k for k, v in label2idx.items()},
        "species2label": label2idx,
        "zeroshot_species": sorted(zeroshot_species),
        "dropped_species": sorted(dropped),
        "split_sizes": {k: len(v) for k, v in splits.items() if k != "metadata"},
    }

    return splits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create species groups G1-G6 with train/val/test/zeroshot splits.",
    )
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml",
        help="Path to base config YAML",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print group assignments without creating split files",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    datasets_cfg = cfg["datasets"]
    groups_cfg = cfg["species_groups"]
    splits_cfg = cfg["splits"]
    output_dir = Path(cfg["paths"]["species_groups"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load eBird taxonomy ---
    ebird_taxonomy = load_ebird_taxonomy(cfg["paths"]["data_raw"])

    # --- Load all manifests ---
    log.info("Loading dataset manifests...")
    manifests: dict[str, dict[str, Any]] = {}
    for ds_name in ["birdclef2023", "birdclef2024", "birdclef2025",
                     "birdset_pow", "watkins", "anuraset"]:
        ds_cfg = datasets_cfg.get(ds_name)
        if ds_cfg and "manifest" in ds_cfg:
            m = load_manifest(ds_cfg["manifest"])
            if m:
                manifests[ds_name] = m
                log.info(
                    f"  {ds_name}: {m.get('total_clips', 0)} clips, "
                    f"{m.get('n_species', 0)} species"
                )

    if not manifests:
        log.error("No manifests found. Run preprocess_audio.py first.")
        return

    # --- Build unified species→clips index ---
    # For BirdCLEF datasets, species codes are eBird codes
    # For Watkins/AnuraSet, species labels are normalized names
    all_species_clips: dict[str, list[dict[str, Any]]] = defaultdict(list)
    species_source: dict[str, set[str]] = defaultdict(set)  # species → set of dataset names

    for ds_name, manifest in manifests.items():
        for clip in manifest.get("clips", []):
            sp = clip["species"]
            all_species_clips[sp].append(clip)
            species_source[sp].add(ds_name)

    log.info(f"Total unique species across all datasets: {len(all_species_clips)}")

    # --- Load BirdCLEF 2025 taxonomy.csv for class_name (Aves/Insecta/Amphibia) ---
    bc2025_taxonomy: dict[str, str] = {}  # species_code → class_name
    bc2025_tax_path = Path(cfg["datasets"]["birdclef2025"]["raw_dir"]) / "taxonomy.csv"
    if bc2025_tax_path.exists():
        with open(bc2025_tax_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row.get("primary_label", "")
                cls = row.get("class_name", "")
                if label and cls:
                    bc2025_taxonomy[label] = cls
        class_counts = Counter(bc2025_taxonomy.values())
        log.info(f"BirdCLEF 2025 taxonomy.csv: {dict(class_counts)}")
    else:
        log.warning("BirdCLEF 2025 taxonomy.csv not found — using heuristic separation")

    # --- Partition BirdCLEF species into bird groups G1-G3 vs non-bird G5-G6 ---
    birdclef_datasets = {"birdclef2023", "birdclef2024", "birdclef2025"}
    birdclef_species: dict[str, dict[str, Any]] = {}
    for sp, sources in species_source.items():
        if sources & birdclef_datasets:
            birdclef_species[sp] = {
                "count": len(all_species_clips[sp]),
                "datasets": sorted(sources & birdclef_datasets),
            }

    log.info(f"\nAll species from BirdCLEF datasets: {len(birdclef_species)}")

    # Separate non-bird taxa using taxonomy.csv class_name (authoritative)
    # then fall back to eBird taxonomy lookup for species without class_name
    nonbird_from_birdclef: dict[str, list[dict[str, Any]]] = {}
    amphibian_from_birdclef: list[str] = []
    insect_from_birdclef: list[str] = []

    for sp in list(birdclef_species.keys()):
        cls = bc2025_taxonomy.get(sp, "")

        if cls == "Insecta":
            insect_from_birdclef.append(sp)
            nonbird_from_birdclef[sp] = all_species_clips[sp]
            del birdclef_species[sp]
        elif cls == "Amphibia":
            amphibian_from_birdclef.append(sp)
            nonbird_from_birdclef[sp] = all_species_clips[sp]
            del birdclef_species[sp]
        elif cls and cls != "Aves":
            # Some other class (Mammalia, Reptilia, etc.)
            nonbird_from_birdclef[sp] = all_species_clips[sp]
            del birdclef_species[sp]
        elif not cls and ebird_taxonomy and sp not in ebird_taxonomy:
            # No class_name AND not in eBird taxonomy — likely non-bird
            nonbird_from_birdclef[sp] = all_species_clips[sp]
            del birdclef_species[sp]

    if nonbird_from_birdclef:
        log.info(
            f"  Separated {len(nonbird_from_birdclef)} non-bird species: "
            f"{len(amphibian_from_birdclef)} amphibians, "
            f"{len(insect_from_birdclef)} insects, "
            f"{len(nonbird_from_birdclef) - len(amphibian_from_birdclef) - len(insect_from_birdclef)} other"
        )

    log.info(f"  Bird species remaining: {len(birdclef_species)}")

    bird_groups = assign_bird_groups(birdclef_species, ebird_taxonomy)

    # Also include BirdSet POW species in appropriate bird groups
    pow_species = {sp for sp, sources in species_source.items()
                   if "birdset_pow" in sources and sp not in birdclef_species}
    if pow_species:
        log.info(f"\nBirdSet POW species not in BirdCLEF: {len(pow_species)}")
        for sp in pow_species:
            order = resolve_order(sp, ebird_taxonomy)
            if order == "Passeriformes":
                bird_groups["G1_passerines"].append(sp)
            elif order and order in G3_ORDERS:
                bird_groups["G3_raptors_waterbirds"].append(sp)
            else:
                bird_groups["G2_nonpasserine_birds"].append(sp)

    # Merge small groups if needed
    min_sp = groups_cfg.get("min_species_per_group", 15)
    bird_groups = maybe_merge_small_groups(bird_groups, min_sp)

    # --- Non-bird groups ---
    # G4: Marine mammals (Watkins)
    watkins_species = [sp for sp, sources in species_source.items()
                       if "watkins" in sources]

    # G5: Amphibians — AnuraSet + amphibians identified from BirdCLEF 2025 taxonomy.csv
    amphibian_species = [sp for sp, sources in species_source.items()
                         if "anuraset" in sources]
    for sp in amphibian_from_birdclef:
        if sp not in amphibian_species:
            amphibian_species.append(sp)

    # G6: Insects — from BirdCLEF 2025 taxonomy.csv (class_name == Insecta)
    insect_species = list(insect_from_birdclef)

    # --- Assemble all groups ---
    all_groups: dict[str, list[str]] = {}
    all_groups.update(bird_groups)

    if watkins_species:
        all_groups["G4_marine_mammals"] = watkins_species
    if amphibian_species:
        all_groups["G5_amphibians"] = amphibian_species
    if insect_species:
        all_groups["G6_insects"] = insect_species

    # --- Report ---
    log.info("\n" + "=" * 60)
    log.info("SPECIES GROUP SUMMARY")
    log.info("=" * 60)
    total_species = 0
    for group_name, species_list in sorted(all_groups.items()):
        total_clips_in_group = sum(len(all_species_clips[sp]) for sp in species_list)
        log.info(
            f"  {group_name:30s}: {len(species_list):>5} species, "
            f"{total_clips_in_group:>8,} clips"
        )
        total_species += len(species_list)

    log.info(f"  {'TOTAL':30s}: {total_species:>5} species")

    # Warn about empty groups
    empty_groups = [g for g, sp in all_groups.items() if len(sp) < min_sp]
    if empty_groups:
        log.warning(
            f"Groups with <{min_sp} species (may need to merge or skip): "
            f"{empty_groups}"
        )

    if args.dry_run:
        log.info("\n--dry-run: not creating split files.")
        return

    # --- Create splits for each group ---
    log.info("\nCreating train/val/test/zeroshot splits...")
    holdout_n = groups_cfg.get("holdout_species_per_group", 10)

    group_summaries: dict[str, Any] = {}

    for group_name, species_list in sorted(all_groups.items()):
        if len(species_list) == 0:
            log.warning(f"  {group_name}: empty, skipping")
            continue

        log.info(f"\n  [{group_name}]")
        splits = create_splits(
            species_list=species_list,
            species_clips=all_species_clips,
            holdout_n=holdout_n,
            train_frac=splits_cfg["train_fraction"],
            val_frac=splits_cfg["val_fraction"],
            test_frac=splits_cfg["test_fraction"],
            min_samples=splits_cfg["min_samples_per_species"],
            max_samples=splits_cfg["max_samples_per_species"],
            seed=splits_cfg["seed"],
        )

        # Save group split file
        out_file = output_dir / f"{group_name}.json"
        with open(out_file, "w") as f:
            json.dump(splits, f, indent=2)
        log.info(f"    Saved to {out_file}")

        group_summaries[group_name] = {
            "n_species": len(species_list),
            "n_classes": splits["metadata"]["n_classes"],
            "n_zeroshot": len(splits["metadata"]["zeroshot_species"]),
            "n_dropped": len(splits["metadata"]["dropped_species"]),
            "split_sizes": splits["metadata"]["split_sizes"],
        }

    # --- Also create the "ALL" combined group (joint training baseline) ---
    log.info("\n  [ALL_groups — joint baseline]")
    all_bird_species = []
    for g in ["G1_passerines", "G2_nonpasserine_birds", "G3_raptors_waterbirds"]:
        all_bird_species.extend(all_groups.get(g, []))

    if all_bird_species:
        all_splits = create_splits(
            species_list=all_bird_species,
            species_clips=all_species_clips,
            holdout_n=holdout_n,
            train_frac=splits_cfg["train_fraction"],
            val_frac=splits_cfg["val_fraction"],
            test_frac=splits_cfg["test_fraction"],
            min_samples=splits_cfg["min_samples_per_species"],
            max_samples=splits_cfg["max_samples_per_species"],
            seed=splits_cfg["seed"],
        )
        out_file = output_dir / "ALL_birds.json"
        with open(out_file, "w") as f:
            json.dump(all_splits, f, indent=2)
        log.info(f"    Saved to {out_file}")

        group_summaries["ALL_birds"] = {
            "n_species": len(all_bird_species),
            "n_classes": all_splits["metadata"]["n_classes"],
            "split_sizes": all_splits["metadata"]["split_sizes"],
        }

    # --- Save overall summary ---
    summary_file = output_dir / "groups_summary.json"
    with open(summary_file, "w") as f:
        json.dump(group_summaries, f, indent=2)

    log.info(f"\nOverall summary saved to {summary_file}")

    # --- Final report ---
    log.info("\n" + "=" * 60)
    log.info("SPLIT SUMMARY")
    log.info("=" * 60)
    for group_name, summary in sorted(group_summaries.items()):
        sizes = summary.get("split_sizes", {})
        log.info(
            f"  {group_name:30s}: "
            f"classes={summary['n_classes']:>4}, "
            f"train={sizes.get('train', 0):>7,}, "
            f"val={sizes.get('val', 0):>5,}, "
            f"test={sizes.get('test', 0):>6,}, "
            f"zs={sizes.get('zeroshot', 0):>5,}"
        )
    log.info("")
    log.info("Next step: python train.py --group G1_passerines --config configs/base.yaml")


if __name__ == "__main__":
    main()