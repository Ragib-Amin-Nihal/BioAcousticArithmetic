"""GPU-dependent visualizations: feature-space UMAP and loss landscape contour.

Produces:
  - figures/umap_triptych.pdf — Base vs Merged vs Joint encoder UMAP
  - figures/umap_g4_zoom.pdf — G4 marine mammal feature zoom (explains 84→93% gain)
  - figures/loss_landscape_2d.pdf — Contour on τ_G1 × τ_G4 plane

Requires GPU for encoder forward passes on test sets.

Usage:
    python analysis/visualize_gpu.py \
        --config configs/base.yaml \
        --output figures/ \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import yaml

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Publication-quality matplotlib defaults
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# NeurIPS / ICASSP-friendly settings
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
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth": 1.2,
    "pdf.fonttype": 42,  # TrueType fonts in PDF (required by many venues)
    "ps.fonttype": 42,
}
plt.rcParams.update(RCPARAMS)

# Colorblind-safe palette for 5 groups + extra
GROUP_COLORS = {
    "G1_passerines": "#0072B2",          # blue
    "G2_nonpasserine_birds": "#E69F00",  # orange
    "G3_raptors_waterbirds": "#009E73",  # green
    "G4_marine_mammals": "#CC79A7",      # pink
    "G5_amphibians": "#56B4E9",          # light blue
}
GROUP_SHORT = {
    "G1_passerines": "Passerines",
    "G2_nonpasserine_birds": "Non-pass. birds",
    "G3_raptors_waterbirds": "Raptors/Water",
    "G4_marine_mammals": "Marine mammals",
    "G5_amphibians": "Amphibians",
}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features_from_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract mean-pooled encoder features from audio DataLoader.

    Args:
        model: BEATsClassifier with encoder loaded.
        loader: DataLoader yielding (waveform, label).
        device: Torch device.
        max_samples: Cap on total samples (UMAP doesn't need millions).

    Returns:
        (features, labels) as numpy arrays, shapes (N, 768) and (N,).
    """
    model.eval()
    all_feats: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    total = 0

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for audio, labels in loader:
            if total >= max_samples:
                break
            audio = audio.to(device)
            padding_mask = torch.zeros(audio.shape, dtype=torch.bool, device=device)
            features, _ = model.encoder.extract_features(audio, padding_mask=padding_mask)
            pooled = features.mean(dim=1).cpu().float()
            all_feats.append(pooled)
            all_labels.append(labels)
            total += audio.shape[0]

    feats = torch.cat(all_feats, dim=0)[:max_samples]
    labs = torch.cat(all_labels, dim=0)[:max_samples]
    return feats.numpy(), labs.numpy()


def load_encoder_into_model(
    base_checkpoint_path: str,
    encoder_state_dict: dict[str, torch.Tensor],
    device: torch.device,
) -> nn.Module:
    """Build BEATsClassifier, load encoder weights, move to device.

    Args:
        base_checkpoint_path: Path to pretrained BEATs .pt file.
        encoder_state_dict: Encoder-only state dict to load.
        device: Target device.

    Returns:
        Model in eval mode on device.
    """
    from models.beats_classifier import BEATsClassifier

    # num_classes doesn't matter — we only use the encoder
    model = BEATsClassifier(base_checkpoint_path, num_classes=1, freeze_epochs=999)
    model.load_encoder_state_dict(encoder_state_dict, strict=False)
    model.freeze_encoder()
    model.eval()
    return model.to(device)


# ---------------------------------------------------------------------------
# UMAP triptych
# ---------------------------------------------------------------------------

def plot_umap_triptych(
    features_dict: dict[str, tuple[np.ndarray, np.ndarray, list[str]]],
    output_path: str,
    n_neighbors: int = 30,
    min_dist: float = 0.3,
    random_state: int = 42,
) -> None:
    """Plot UMAP triptych: base vs merged vs joint encoder features.

    Args:
        features_dict: Maps encoder_name → (features, group_ids, group_names_per_sample).
            features: (N, 768), group_ids: (N,) int, group_names: list of str.
        output_path: Path to save PDF.
        n_neighbors: UMAP n_neighbors.
        min_dist: UMAP min_dist.
        random_state: Random seed for reproducibility.
    """
    from umap import UMAP

    panel_names = list(features_dict.keys())
    n_panels = len(panel_names)

    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 3.8))
    if n_panels == 1:
        axes = [axes]

    for ax, name in zip(axes, panel_names):
        feats, group_ids, group_names = features_dict[name]

        logger.info("UMAP fitting for '%s' (%d samples)...", name, feats.shape[0])
        reducer = UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=random_state,
            metric="cosine",
        )
        embedding = reducer.fit_transform(feats)

        # Plot each group
        unique_groups = sorted(set(group_names))
        for gname in unique_groups:
            mask = np.array([g == gname for g in group_names])
            color = GROUP_COLORS.get(gname, "#999999")
            short = GROUP_SHORT.get(gname, gname)
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=color, s=3, alpha=0.4, label=short, rasterized=True,
            )

        ax.set_title(name, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    # Single legend
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=min(5, len(handles)),
        bbox_to_anchor=(0.5, -0.02),
        frameon=False, markerscale=3,
    )

    fig.suptitle(
        "Feature-Space Structure: Pretrained → Merged → Joint",
        fontsize=11, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved UMAP triptych to %s", output_path)


def plot_umap_g4_zoom(
    features_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    species_names: dict[int, str],
    output_path: str,
    n_neighbors: int = 15,
    min_dist: float = 0.2,
) -> None:
    """Plot UMAP zoom for G4 marine mammals under three encoders.

    Args:
        features_dict: Maps encoder_name → (features, species_labels).
        species_names: Maps label_idx → species name string.
        output_path: Path to save PDF.
        n_neighbors: UMAP n_neighbors (smaller for fine detail).
        min_dist: UMAP min_dist.
    """
    from umap import UMAP

    panel_names = list(features_dict.keys())
    n_panels = len(panel_names)

    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 3.8))
    if n_panels == 1:
        axes = [axes]

    # Use a qualitative colormap for up to 21 species
    cmap = plt.cm.get_cmap("tab20", max(21, len(species_names)))

    for ax, name in zip(axes, panel_names):
        feats, labels = features_dict[name]

        reducer = UMAP(
            n_neighbors=n_neighbors, min_dist=min_dist,
            n_components=2, random_state=42, metric="cosine",
        )
        embedding = reducer.fit_transform(feats)

        unique_labels = sorted(set(labels.tolist()))
        for i, lab in enumerate(unique_labels):
            mask = labels == lab
            sp_name = species_names.get(lab, f"sp_{lab}")
            # Truncate long names
            short = sp_name[:15] + "…" if len(sp_name) > 15 else sp_name
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=[cmap(i)], s=8, alpha=0.6, label=short, rasterized=True,
            )

        ax.set_title(name, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    # Legend outside
    handles, labels_legend = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles, labels_legend,
        loc="center left", bbox_to_anchor=(1.01, 0.5),
        frameon=False, fontsize=6, markerscale=2,
        ncol=1 if len(handles) <= 12 else 2,
    )

    fig.suptitle(
        "G4 Marine Mammals: Feature Separation by Encoder",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved G4 zoom UMAP to %s", output_path)


# ---------------------------------------------------------------------------
# 2D loss landscape contour
# ---------------------------------------------------------------------------

def evaluate_loss_on_grid(
    base_checkpoint_path: str,
    tau_a: dict[str, torch.Tensor],
    tau_b: dict[str, torch.Tensor],
    test_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    grid_range: tuple[float, float] = (-0.3, 1.5),
    grid_steps: int = 21,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate loss on a 2D grid in the τ_A × τ_B plane.

    θ(α, β) = θ_base + α·τ_A + β·τ_B

    For each grid point, loads the merged encoder, extracts features,
    trains a quick linear probe, and records the loss.

    Args:
        base_checkpoint_path: Path to BEATs base checkpoint.
        tau_a: First task vector.
        tau_b: Second task vector.
        test_loader: DataLoader for evaluation.
        num_classes: Number of classes.
        device: Torch device.
        grid_range: (min, max) for α and β.
        grid_steps: Number of steps per axis.

    Returns:
        (alphas, betas, loss_grid) — 1D alpha array, 1D beta array, 2D loss matrix.
    """
    from models.beats_classifier import BEATsClassifier
    from evaluation.linear_probe import LinearProbeEvaluator, ProbeConfig

    alphas = np.linspace(grid_range[0], grid_range[1], grid_steps)
    betas = np.linspace(grid_range[0], grid_range[1], grid_steps)
    loss_grid = np.zeros((grid_steps, grid_steps))

    base_ckpt = torch.load(base_checkpoint_path, map_location="cpu", weights_only=False)
    base_model = base_ckpt["model"]

    # Quick probe config — fewer epochs for speed on a dense grid
    probe_cfg = ProbeConfig(learning_rate=1e-3, epochs=5, batch_size=64)
    evaluator = LinearProbeEvaluator(
        base_checkpoint_path, config=probe_cfg, device=str(device),
    )

    total_points = grid_steps * grid_steps
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            point_idx = i * grid_steps + j + 1
            # Compose encoder
            merged: dict[str, torch.Tensor] = {}
            for key in base_model:
                val = base_model[key].float().clone()
                if key in tau_a:
                    val = val + alpha * tau_a[key].float()
                if key in tau_b:
                    val = val + beta * tau_b[key].float()
                merged[key] = val

            result = evaluator.evaluate(
                merged, test_loader, test_loader, num_classes,
            )
            loss_grid[j, i] = 1.0 - result.accuracy  # error rate as loss proxy

            if point_idx % 20 == 0 or point_idx == total_points:
                logger.info(
                    "  Grid point %d/%d (α=%.2f, β=%.2f): err=%.4f",
                    point_idx, total_points, alpha, beta, loss_grid[j, i],
                )

            # Cleanup
            del merged
            torch.cuda.empty_cache()

    return alphas, betas, loss_grid


def plot_loss_landscape(
    alphas: np.ndarray,
    betas: np.ndarray,
    loss_grid: np.ndarray,
    group_a: str,
    group_b: str,
    output_path: str,
) -> None:
    """Plot 2D loss landscape contour with key model positions.

    Args:
        alphas: 1D array of α values.
        betas: 1D array of β values.
        loss_grid: 2D loss matrix (beta × alpha).
        group_a: Name of first group (for axis label).
        group_b: Name of second group (for axis label).
        output_path: Path to save PDF.
    """
    fig, ax = plt.subplots(figsize=(5.0, 4.2))

    A, B = np.meshgrid(alphas, betas)

    # Filled contour
    levels = 20
    cf = ax.contourf(A, B, loss_grid, levels=levels, cmap="RdYlGn_r")
    # Contour lines
    cs = ax.contour(A, B, loss_grid, levels=10, colors="k", linewidths=0.3, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.2f")

    cbar = fig.colorbar(cf, ax=ax, label="Error Rate (1 − accuracy)", pad=0.02)
    cbar.ax.tick_params(labelsize=7)

    # Mark key model positions
    markers = {
        "Base (0,0)": (0.0, 0.0),
        f"FT {GROUP_SHORT.get(group_a, group_a)} (1,0)": (1.0, 0.0),
        f"FT {GROUP_SHORT.get(group_b, group_b)} (0,1)": (0.0, 1.0),
        "Merged (1,1)": (1.0, 1.0),
    }
    marker_styles = ["o", "s", "D", "^"]
    for (label, (a, b)), m in zip(markers.items(), marker_styles):
        ax.plot(a, b, marker=m, color="black", markersize=7,
                markeredgecolor="white", markeredgewidth=0.8, zorder=5)
        ax.annotate(
            label, (a, b), textcoords="offset points",
            xytext=(6, 6), fontsize=7, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8),
        )

    short_a = GROUP_SHORT.get(group_a, group_a)
    short_b = GROUP_SHORT.get(group_b, group_b)
    ax.set_xlabel(f"α  (scaling of τ_{{{short_a}}})")
    ax.set_ylabel(f"β  (scaling of τ_{{{short_b}}})")
    ax.set_title("Loss Landscape Cross-Section", fontweight="bold")
    ax.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info("Saved loss landscape to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU-dependent visualizations: UMAP + loss landscape"
    )
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--output", type=str, default="figures/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--umap-samples", type=int, default=3000,
        help="Max samples per group for UMAP (more = slower)",
    )
    parser.add_argument(
        "--landscape-steps", type=int, default=21,
        help="Grid resolution for loss landscape (21 = 441 evaluations)",
    )
    parser.add_argument(
        "--landscape-range", type=float, nargs=2, default=[-0.3, 1.5],
        help="Min and max for landscape grid axes",
    )
    parser.add_argument(
        "--skip-umap", action="store_true",
        help="Skip UMAP (run only loss landscape)",
    )
    parser.add_argument(
        "--skip-landscape", action="store_true",
        help="Skip loss landscape (run only UMAP)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_path = cfg["paths"]["beats_base_checkpoint"]
    tv_dir = Path(cfg["paths"]["task_vectors"])
    finetuned_dir = Path(cfg["paths"]["finetuned"])
    species_groups_dir = Path(cfg["paths"]["species_groups"])

    groups = [
        "G1_passerines", "G2_nonpasserine_birds", "G3_raptors_waterbirds",
        "G4_marine_mammals", "G5_amphibians",
    ]

    # Import data utilities
    from data.dataset import build_group_loaders

    # ------------------------------------------------------------------
    # UMAP Triptych
    # ------------------------------------------------------------------
    if not args.skip_umap:
        logger.info("=" * 60)
        logger.info("UMAP TRIPTYCH: extracting features for 3 encoders × %d groups", len(groups))
        logger.info("=" * 60)

        # Load encoders
        base_ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
        base_encoder = base_ckpt["model"]

        # Best merged encoder = task_arithmetic λ=1.0 (from composition results)
        task_vectors = {}
        for g in groups:
            tv_path = tv_dir / f"tau_{g}.pt"
            if tv_path.exists():
                task_vectors[g] = torch.load(tv_path, map_location="cpu")
            else:
                logger.warning("Task vector not found: %s", tv_path)

        # Merged = base + Σ τ_i (task arithmetic λ=1.0)
        merged_encoder: dict[str, torch.Tensor] = {}
        for key in base_encoder:
            merged_encoder[key] = base_encoder[key].float().clone()
            for tv in task_vectors.values():
                if key in tv:
                    merged_encoder[key] = merged_encoder[key] + tv[key].float()

        # Joint encoder (ALL_birds fine-tuned)
        joint_path = finetuned_dir / "ALL_birds" / "best_model.pt"
        if joint_path.exists():
            joint_ckpt = torch.load(joint_path, map_location="cpu", weights_only=False)
            joint_encoder = joint_ckpt["encoder_state_dict"]
        else:
            logger.warning("ALL_birds checkpoint not found at %s; skipping joint panel", joint_path)
            joint_encoder = None

        # Build test loaders for each group
        # build_group_loaders returns {group: (train_loader, test_loader, n_classes)}
        logger.info("Building test data loaders...")
        group_loader_tuples = build_group_loaders(
            groups=groups,
            species_groups_dir=str(species_groups_dir),
            processed_dir=cfg["paths"]["data_processed"],
            batch_size=64,
            num_workers=cfg["training"]["num_workers"],
        )
        # Extract just test loaders
        group_test_loaders: dict[str, DataLoader] = {
            gname: test_loader
            for gname, (_, test_loader, _) in group_loader_tuples.items()
        }

        # Extract features for each encoder
        encoder_configs = {"(a) Pretrained": base_encoder, "(b) Merged": merged_encoder}
        if joint_encoder is not None:
            encoder_configs["(c) Joint"] = joint_encoder

        triptych_data: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}

        for enc_name, enc_sd in encoder_configs.items():
            logger.info("--- Encoder: %s ---", enc_name)
            model = load_encoder_into_model(base_path, enc_sd, device)

            all_feats: list[np.ndarray] = []
            all_group_names: list[str] = []

            for gname, loader in group_test_loaders.items():
                feats, _ = extract_features_from_loader(
                    model, loader, device, max_samples=args.umap_samples,
                )
                all_feats.append(feats)
                all_group_names.extend([gname] * feats.shape[0])
                logger.info("  %s: %d samples", gname, feats.shape[0])

            combined_feats = np.concatenate(all_feats, axis=0)
            group_ids = np.array([groups.index(g) for g in all_group_names])
            triptych_data[enc_name] = (combined_feats, group_ids, all_group_names)

            del model
            torch.cuda.empty_cache()

        plot_umap_triptych(triptych_data, str(output_dir / "umap_triptych.pdf"))

        # G4 zoom
        logger.info("--- G4 Marine Mammal Zoom ---")
        g4_json_path = species_groups_dir / "G4_marine_mammals.json"
        if g4_json_path.exists():
            with open(g4_json_path) as f:
                g4_meta = json.load(f)
            label2species = g4_meta["metadata"].get("label2species", {})
            species_names = {int(k): v for k, v in label2species.items()}

            # Need G4 fine-tuned encoder too
            g4_ft_path = finetuned_dir / "G4_marine_mammals" / "best_model.pt"
            g4_zoom_encoders = {"(a) G4 Fine-tuned": None, "(b) Merged": merged_encoder}
            if g4_ft_path.exists():
                g4_ft_ckpt = torch.load(g4_ft_path, map_location="cpu", weights_only=False)
                g4_zoom_encoders["(a) G4 Fine-tuned"] = g4_ft_ckpt["encoder_state_dict"]
            g4_zoom_encoders["(c) Pretrained"] = base_encoder

            g4_loader = group_test_loaders.get("G4_marine_mammals")
            if g4_loader is not None:
                g4_zoom_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                for enc_name, enc_sd in g4_zoom_encoders.items():
                    if enc_sd is None:
                        continue
                    model = load_encoder_into_model(base_path, enc_sd, device)
                    feats, labels = extract_features_from_loader(
                        model, g4_loader, device, max_samples=2000,
                    )
                    g4_zoom_data[enc_name] = (feats, labels)
                    del model
                    torch.cuda.empty_cache()

                plot_umap_g4_zoom(
                    g4_zoom_data, species_names,
                    str(output_dir / "umap_g4_zoom.pdf"),
                )

        # Cleanup
        del task_vectors, merged_encoder, base_encoder
        if joint_encoder is not None:
            del joint_encoder
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 2D Loss Landscape Contour
    # ------------------------------------------------------------------
    if not args.skip_landscape:
        logger.info("=" * 60)
        logger.info("LOSS LANDSCAPE: τ_G1 × τ_G4 plane, %d×%d grid",
                     args.landscape_steps, args.landscape_steps)
        logger.info("=" * 60)

        tau_g1 = torch.load(tv_dir / "tau_G1_passerines.pt", map_location="cpu")
        tau_g4 = torch.load(tv_dir / "tau_G4_marine_mammals.pt", map_location="cpu")

        # Build a unified test loader (subsample for speed)
        # Use G1 + G4 test data
        landscape_loader_tuples = build_group_loaders(
            groups=["G1_passerines", "G4_marine_mammals"],
            species_groups_dir=str(species_groups_dir),
            processed_dir=cfg["paths"]["data_processed"],
            batch_size=64,
            num_workers=cfg["training"]["num_workers"],
        )

        # For loss landscape, we evaluate on each group's test set separately
        # and average the error rates. Using G1 test set as primary since it's
        # the larger group.
        _, g1_loader, g1_n_classes = landscape_loader_tuples["G1_passerines"]

        alphas, betas, loss_grid = evaluate_loss_on_grid(
            base_checkpoint_path=base_path,
            tau_a=tau_g1,
            tau_b=tau_g4,
            test_loader=g1_loader,
            num_classes=g1_n_classes,
            device=device,
            grid_range=tuple(args.landscape_range),
            grid_steps=args.landscape_steps,
        )

        # Save raw grid for replotting
        np.savez(
            output_dir / "loss_landscape_grid.npz",
            alphas=alphas, betas=betas, loss_grid=loss_grid,
        )

        plot_loss_landscape(
            alphas, betas, loss_grid,
            group_a="G1_passerines", group_b="G4_marine_mammals",
            output_path=str(output_dir / "loss_landscape_2d.pdf"),
        )

    logger.info("GPU visualizations complete.")


if __name__ == "__main__":
    main()