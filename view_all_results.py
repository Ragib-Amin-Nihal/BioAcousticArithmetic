#!/usr/bin/env python3
"""Dump every experiment result JSON into a single readable text file.

Reads the standard results/ directory layout and outputs a markdown-formatted
summary suitable for upload to project knowledge.

Usage:
    python view_all_results.py --results-dir results/
    python view_all_results.py --results-dir results/ --output results_summary.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def load(path: Path) -> Optional[dict]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def pct(v: float, d: int = 2) -> str:
    return f"{v * 100:.{d}f}%" if isinstance(v, (int, float)) else str(v)


def pp(v: float, d: int = 1) -> str:
    return f"{v * 100:+.{d}f}pp" if isinstance(v, (int, float)) else str(v)


W = 80  # column width for dividers


# ──────────────────────────────────────────────────────────────────────
# 1. LMC
# ──────────────────────────────────────────────────────────────────────

def dump_lmc(R: Path) -> str:
    o = f"\n{'=' * W}\n1. EXPERIMENT 1: LINEAR MODE CONNECTIVITY\n{'=' * W}\n\n"

    # Try aggregated then direct
    data =  load(R / "lmc" / "lmc_results_corrected.json")
    if not data:
        return o + "  FILE NOT FOUND\n"

    pairs = data.get("pairs", [])
    o += f"Pairs evaluated: {len(pairs)}\n\n"
    o += f"{'Model A':<28s} {'Model B':<28s} {'Eval Group':<20s} {'Barrier':>8s} {'Acc@0.5':>8s}\n"
    o += "-" * 94 + "\n"

    for p in pairs:
        o += (f"{p['model_a']:<28s} {p['model_b']:<28s} "
              f"{p.get('eval_group', '—'):<20s} "
              f"{p['barrier_height']:>8.4f} "
              f"{pct(p.get('accuracy_at_midpoint', 0)):>8s}\n")

    bm = data.get("barrier_matrix", {})
    if bm:
        groups = sorted(bm.keys())
        o += "\nBarrier matrix:\n"
        o += f"{'':15s}" + "".join(f"{g[-12:]:>14s}" for g in groups) + "\n"
        for ga in groups:
            row = f"{ga[-14:]:<15s}"
            for gb in groups:
                if ga == gb:
                    row += f"{'—':>14s}"
                else:
                    row += f"{bm.get(ga, {}).get(gb, 0):>14.4f}"
            o += row + "\n"

    return o


# ──────────────────────────────────────────────────────────────────────
# 2. Composition
# ──────────────────────────────────────────────────────────────────────

def dump_composition(R: Path) -> str:
    o = f"\n{'=' * W}\n2. EXPERIMENT 2: SPECIES-GROUP COMPOSITION\n{'=' * W}\n\n"

    data = load(R / "composition" / "composition_results.json")
    if not data:
        return o + "  FILE NOT FOUND\n"

    joint = data.get("joint_baseline_accuracy")
    if joint is not None:
        o += f"Joint baseline (ALL_birds): {pct(joint)}\n"
    o += f"Groups used: {data.get('groups_used', '?')}\n\n"

    trials = sorted(data.get("trials", []),
                    key=lambda t: t.get("all_group_accuracy", 0), reverse=True)

    o += f"{'Method':<22s} {'Params':<32s} {'AllAcc':>8s} {'Gap':>8s}\n"
    o += "-" * 72 + "\n"

    for t in trials:
        method = t["method"]
        hp = t.get("hyperparams", {})
        hp_str = ", ".join(f"{k}={v}" for k, v in hp.items()) or "—"
        if len(hp_str) > 30:
            hp_str = hp_str[:30] + "…"
        all_acc = t.get("all_group_accuracy", 0)
        gap = t.get("gap_to_joint")
        gap_str = pp(gap) if gap is not None else "—"
        o += f"{method:<22s} {hp_str:<32s} {pct(all_acc):>8s} {gap_str:>8s}\n"

    # Per-group breakdown for best trial
    if trials:
        best = trials[0]
        pg = best.get("per_group_results", {})
        if pg:
            o += f"\nBest method per-group ({best['method']}):\n"
            o += f"  {'Group':<35s} {'Accuracy':>10s}\n"
            o += "  " + "-" * 47 + "\n"
            for gn in sorted(pg):
                gd = pg[gn]
                acc = gd["accuracy"] if isinstance(gd, dict) else gd
                o += f"  {gn:<35s} {pct(acc):>10s}\n"

    return o


# ──────────────────────────────────────────────────────────────────────
# 3. Regional
# ──────────────────────────────────────────────────────────────────────

def dump_regional(R: Path) -> str:
    o = f"\n{'=' * W}\n3. EXPERIMENT 3: REGIONAL COMPOSITION\n{'=' * W}\n\n"

    data = load(R / "regional" / "regional_composition_results.json")
    if not data:
        return o + "  FILE NOT FOUND\n"

    o += f"Joint baseline: {data.get('joint_baseline_accuracy', 'N/A')}\n\n"

    trials = sorted(data.get("trials", []),
                    key=lambda t: t.get("all_region_accuracy",
                                        t.get("all_group_accuracy", 0)), reverse=True)
    if trials:
        o += f"{'Strategy':<35s} {'AllRegAcc':>10s} {'Gap':>10s}\n"
        o += "-" * 57 + "\n"
        for t in trials:
            method = t.get("method", t.get("strategy", "?"))
            params = t.get("hyperparams", {})
            param_str = " ".join(f"{k}={v}" for k, v in params.items())
            label = f"{method} {param_str}".strip()
            acc = t.get("all_region_accuracy", t.get("all_group_accuracy", 0))
            gap = t.get("gap_to_joint")
            gap_str = pp(gap) if gap is not None else "—"
            o += f"  {label:<33s} {pct(acc):>10s} {gap_str:>10s}\n"

            # Per-region breakdown
            pg = t.get("per_region_accuracy", t.get("per_group_results", {}))
            if pg:
                for rn in sorted(pg):
                    rd = pg[rn]
                    racc = rd["accuracy"] if isinstance(rd, dict) else rd
                    o += f"    {rn:<31s} {pct(racc):>10s}\n"

    # Within-region baselines
    baselines = data.get("within_region_baselines", {})
    if baselines:
        o += "\nWithin-region baselines (dedicated model):\n"
        for rn in sorted(baselines):
            o += f"  {rn:<35s} {pct(baselines[rn])}\n"

    # Species overlaps
    overlaps = data.get("species_overlaps", {})
    if overlaps:
        o += "\nSpecies overlaps between regions:\n"
        for pair, info in sorted(overlaps.items()):
            if isinstance(info, dict):
                jaccard = info.get("jaccard", info.get("overlap", "?"))
                n_shared = info.get("n_shared", info.get("shared", "?"))
                o += f"  {pair:<35s} jaccard={jaccard}, shared={n_shared}\n"
            else:
                o += f"  {pair:<35s} {info}\n"

    # Regional sparsity (separate file)
    reg_sparsity = load(R / "regional" / "regional_sparsity.json" / "sparsity_summary.json")
    if reg_sparsity:
        o += "\nRegional task vector geometry:\n"
        pw = reg_sparsity.get("pairwise", [])
        for p in pw:
            pair = p.get("pair", f"{p.get('name_a', '?')}_vs_{p.get('name_b', '?')}")
            cos = p.get("cosine_similarity", 0)
            sa = p.get("sign_agreement", 0)
            o += f"  {pair:<45s} cos={cos:.4f}  sign={sa:.4f}\n"

    return o


# ──────────────────────────────────────────────────────────────────────
# 4. Domain negation
# ──────────────────────────────────────────────────────────────────────

def dump_negation(R: Path) -> str:
    o = f"\n{'=' * W}\n4. EXPERIMENT 4: DOMAIN NEGATION\n{'=' * W}\n\n"

    data = None
    for d in ["negation", "domain_negation"]:
        for f in ["negation_results.json", "domain_negation_results.json",
                  "negation_checkpoint.json"]:
            data = load(R / d / f)
            if data:
                break
        if data:
            break
    if not data:
        return o + "  FILE NOT FOUND\n"

    baselines = data.get("baselines", {})
    if baselines:
        o += "Baselines:\n"
        for k, v in sorted(baselines.items()):
            if v is not None:
                o += f"  {k:<45s} {pct(v)}\n"
        o += "\n"

    trials = data.get("trials", [])
    # Group by source_model
    sources = sorted(set(t.get("source_model", "?") for t in trials))
    for src in sources:
        o += f"--- Source: {src} ---\n"
        src_trials = sorted(
            [t for t in trials if t.get("source_model") == src],
            key=lambda t: (t.get("vector_type", ""), t.get("beta", 0)),
        )
        o += f"  {'Type':<18s} {'Beta':>6s} {'Focal Acc':>10s} {'Soundsc Acc':>12s}\n"
        o += "  " + "-" * 48 + "\n"
        for t in src_trials:
            o += (f"  {t.get('vector_type', '?'):<18s} "
                  f"{t.get('beta', 0):>6.2f} "
                  f"{pct(t.get('focal_accuracy', 0)):>10s} "
                  f"{pct(t.get('soundscape_accuracy', 0)):>12s}\n")
        o += "\n"

    return o


# ──────────────────────────────────────────────────────────────────────
# 5. Sparsity & geometry
# ──────────────────────────────────────────────────────────────────────

def dump_sparsity(R: Path) -> str:
    o = f"\n{'=' * W}\n5. TASK VECTOR GEOMETRY & SPARSITY\n{'=' * W}\n\n"

    summary = load(R / "analysis" / "sparsity_summary.json")
    if not summary:
        return o + "  FILE NOT FOUND\n"

    indiv = summary.get("individual", {})
    if indiv:
        o += "Per-group properties:\n"
        o += f"  {'Group':<30s} {'N params':>10s} {'L2 norm':>10s} {'Mean mag':>10s} {'Sparse@1e-3':>12s}\n"
        o += "  " + "-" * 74 + "\n"
        for gn in sorted(indiv):
            g = indiv[gn]
            o += (f"  {gn:<30s} "
                  f"{g.get('n_params', 0):>10d} "
                  f"{g.get('l2_norm', 0):>10.4f} "
                  f"{g.get('mean_magnitude', 0):>10.2e} "
                  f"{g.get('frac_near_zero_1e-3', 0) * 100:>11.1f}%\n")

    pw = summary.get("pairwise", [])
    if pw:
        o += "\nPairwise cosine similarity & sign agreement:\n"
        o += f"  {'Pair':<55s} {'Cosine':>8s} {'Sign Agr':>9s}\n"
        o += "  " + "-" * 74 + "\n"
        for p in pw:
            pair = p.get("pair", f"{p.get('name_a', '?')}_vs_{p.get('name_b', '?')}")
            cos = p.get("cosine_similarity", 0)
            sa = p.get("sign_agreement", 0)
            o += f"  {pair:<55s} {cos:>8.4f} {sa:>8.4f}\n"

    return o


# ──────────────────────────────────────────────────────────────────────
# 6. Norm-adjusted ablation
# ──────────────────────────────────────────────────────────────────────

def dump_norm_adjusted(R: Path) -> str:
    o = f"\n{'=' * W}\n6. NORM-ADJUSTED WEIGHTING ABLATION\n{'=' * W}\n\n"

    data = load(R / "composition" / "norm_adjusted" / "norm_adjusted_results.json")
    if not data:
        return o + "  FILE NOT FOUND\n"

    joint = data.get("joint_baseline_acc")
    if joint:
        o += f"Joint baseline: {pct(joint)}\n\n"

    trials = data.get("trials", [])
    o += f"{'Strategy':<20s} {'Lambda':>8s} {'AllAcc':>10s} {'MeanGrp':>10s} {'Gap':>8s}\n"
    o += "-" * 58 + "\n"
    for t in trials:
        strategy = t.get("strategy", t.get("method", "?"))
        lam = t.get("lambda", t.get("scaling", "?"))
        all_acc = t.get("all_group_accuracy", 0)
        mean_grp = t.get("mean_group_accuracy", 0)
        gap = t.get("gap_to_joint")
        gap_str = pp(gap) if gap is not None else "—"
        o += f"{strategy:<20s} {str(lam):>8s} {pct(all_acc):>10s} {pct(mean_grp):>10s} {gap_str:>8s}\n"

    return o


# ──────────────────────────────────────────────────────────────────────
# 7. Bootstrap CI
# ──────────────────────────────────────────────────────────────────────

def dump_bootstrap(R: Path) -> str:
    o = f"\n{'=' * W}\n7. BOOTSTRAP CONFIDENCE INTERVALS\n{'=' * W}\n\n"

    data = load(R / "gap_analysis" / "gap_analysis.json")
    if not data:
        return o + "  FILE NOT FOUND\n"

    o += f"Bootstrap resamples: {data.get('n_bootstrap', '?')}\n\n"

    ag = data.get("all_group_gap", {})
    if ag:
        o += "All-group gap (unified 661-class probe):\n"
        o += f"  Joint accuracy:  {pct(ag['acc_a'])}\n"
        o += f"  Merged accuracy: {pct(ag['acc_b'])}\n"
        o += f"  Gap:             {pp(ag['gap'])}\n"
        o += f"  95% CI:          [{pp(ag['ci_low'])}, {pp(ag['ci_high'])}]\n"
        o += f"  Significant:     {ag['significant']}\n"
        o += f"  N samples:       {ag.get('n_samples', '?')}\n\n"

    mpg = data.get("mean_per_group_gap", {})
    if mpg:
        o += "Mean per-group gap (independent probes):\n"
        o += f"  Gap:             {pp(mpg['gap'])}\n"
        o += f"  95% CI:          [{pp(mpg['ci_low'])}, {pp(mpg['ci_high'])}]\n"
        o += f"  Significant:     {mpg['significant']}\n\n"

    pg = data.get("per_group_gaps", {})
    if pg:
        o += "Per-group gaps:\n"
        o += f"  {'Group':<35s} {'Joint':>8s} {'Merged':>8s} {'Gap':>8s} {'CI Low':>8s} {'CI Hi':>8s} {'Sig':>5s}\n"
        o += "  " + "-" * 82 + "\n"
        for gn in sorted(pg):
            g = pg[gn]
            o += (f"  {gn:<35s} "
                  f"{pct(g['acc_a']):>8s} {pct(g['acc_b']):>8s} "
                  f"{pp(g['gap']):>8s} {pp(g['ci_low']):>8s} {pp(g['ci_high']):>8s} "
                  f"{'Yes' if g['significant'] else 'No':>5s}\n")

    return o


# ──────────────────────────────────────────────────────────────────────
# 8. KNN evaluation
# ──────────────────────────────────────────────────────────────────────

def dump_knn(R: Path) -> str:
    o = f"\n{'=' * W}\n8. KNN EVALUATION\n{'=' * W}\n\n"

    data = load(R / "knn" / "knn_evaluation.json")
    if not data:
        return o + "  FILE NOT FOUND\n"

    encoders = data.get("encoders", {})
    for enc_name in sorted(encoders):
        ed = encoders[enc_name]
        o += f"Encoder: {enc_name}\n"
        knn_list = ed.get("all_group_knn", [])
        if knn_list:
            o += f"  {'K':>4s} {'Accuracy':>10s}\n"
            o += "  " + "-" * 16 + "\n"
            for kr in sorted(knn_list, key=lambda x: x.get("k", 0)):
                o += f"  {kr.get('k', '?'):>4} {pct(kr.get('accuracy', 0)):>10s}\n"

        # Per-group if available
        per_group = ed.get("per_group_knn", {})
        if per_group:
            o += f"\n  Per-group (best K):\n"
            for gn in sorted(per_group):
                glist = per_group[gn]
                if isinstance(glist, list) and glist:
                    best = max(glist, key=lambda x: x.get("accuracy", 0))
                    o += f"    {gn:<35s} {pct(best['accuracy']):>10s} (k={best.get('k', '?')})\n"
        o += "\n"

    gap = data.get("knn_composition_gap")
    if gap is not None:
        o += f"KNN composition gap: {pp(gap)}\n"

    return o


# ──────────────────────────────────────────────────────────────────────
# 9. Data efficiency
# ──────────────────────────────────────────────────────────────────────

def dump_data_efficiency(R: Path) -> str:
    o = f"\n{'=' * W}\n9. DATA EFFICIENCY\n{'=' * W}\n\n"

    data = load(R / "data_efficiency" / "data_efficiency_results.json")
    if not data:
        return o + "  FILE NOT FOUND\n"

    o += f"Target group: {data.get('target_group', '?')}\n"
    o += f"Joint baseline: {pct(data.get('joint_baseline_accuracy', 0))}\n"
    o += f"Full merge: {pct(data.get('full_merge_accuracy', 0))}\n\n"

    fracs = data.get("fractions", [])
    o += f"{'Frac':>6s} {'N_samp':>8s} {'N_spp':>7s} {'G4 Alone':>10s} {'G4 Merged':>11s} {'All Acc':>10s} {'Cos Full':>10s}\n"
    o += "-" * 64 + "\n"
    for fr in fracs:
        o += (f"{fr['fraction'] * 100:>5.0f}% "
              f"{fr.get('n_train_samples', 0):>8d} "
              f"{fr.get('n_species_retained', 0):>7d} "
              f"{pct(fr.get('g4_standalone_accuracy', 0)):>10s} "
              f"{pct(fr.get('g4_merged_accuracy', 0)):>11s} "
              f"{pct(fr.get('all_group_accuracy', 0)):>10s} "
              f"{fr.get('tv_cosine_with_full', 0):>10.4f}\n")

    # Per-group at each fraction
    for fr in fracs:
        pg = fr.get("per_group_accuracy", {})
        if pg:
            o += f"\n  Fraction {fr['fraction'] * 100:.0f}% per-group:\n"
            for gn in sorted(pg):
                o += f"    {gn:<35s} {pct(pg[gn])}\n"

    return o


# ──────────────────────────────────────────────────────────────────────
# 10. Continual learning
# ──────────────────────────────────────────────────────────────────────

def dump_continual(R: Path) -> str:
    o = f"\n{'=' * W}\n10. CONTINUAL LEARNING\n{'=' * W}\n\n"

    data = load(R / "continual_learning" / "continual_learning_results.json")
    if not data:
        return o + "  FILE NOT FOUND\n"

    o += f"Existing groups: {data.get('existing_groups', '?')}\n"
    o += f"New group: {data.get('new_group', '?')}\n\n"

    # G1234 baseline
    g1234 = data.get("g1234_baseline", {})
    if g1234:
        o += "G1234 baseline per-group accuracy:\n"
        for gn in sorted(g1234):
            o += f"  {gn:<35s} {pct(g1234[gn])}\n"
        o += "\n"

    methods = data.get("methods", [])
    o += f"{'Method':<28s} {'Old Mean':>10s} {'G5 Acc':>10s} {'Forget':>10s} {'All Acc':>10s}\n"
    o += "-" * 70 + "\n"
    for m in methods:
        o += (f"{m['method']:<28s} "
              f"{pct(m.get('old_group_mean_accuracy', 0)):>10s} "
              f"{pct(m.get('new_group_accuracy', 0)):>10s} "
              f"{pp(m.get('forgetting', 0)):>10s} "
              f"{pct(m.get('all_group_accuracy', 0)):>10s}\n")

    # Per-group per method
    for m in methods:
        pg = m.get("per_group_accuracy", {})
        if pg:
            o += f"\n  {m['method']} per-group:\n"
            for gn in sorted(pg):
                o += f"    {gn:<35s} {pct(pg[gn])}\n"

    # Fine-tuning dynamics
    ft = data.get("finetune_per_epoch", [])
    if ft:
        o += "\nFine-tuning dynamics (per LR, per epoch):\n"
        for lr_run in ft:
            lr = lr_run.get("learning_rate", "?")
            epochs = lr_run.get("epochs", lr_run.get("per_epoch", []))
            o += f"\n  LR = {lr}\n"
            if isinstance(epochs, list):
                o += f"  {'Epoch':>6s} {'Old Mean':>10s} {'G5 Acc':>10s}\n"
                o += "  " + "-" * 28 + "\n"
                for e in epochs:
                    epoch = e.get("epoch", "?")
                    pg_e = e.get("per_group", {})
                    old_groups = [v for k, v in pg_e.items() if "G5" not in k]
                    old_mean = sum(old_groups) / len(old_groups) if old_groups else 0
                    g5 = pg_e.get("G5_amphibians", 0)
                    o += f"  {str(epoch):>6s} {pct(old_mean):>10s} {pct(g5):>10s}\n"

    return o


# ──────────────────────────────────────────────────────────────────────
# 11. Compute efficiency
# ──────────────────────────────────────────────────────────────────────

def dump_efficiency(R: Path) -> str:
    o = f"\n{'=' * W}\n11. COMPUTE EFFICIENCY\n{'=' * W}\n\n"

    data = load(R / "efficiency" / "compute_efficiency.json")
    if not data:
        return o + "  FILE NOT FOUND\n"

    o += f"Merge arithmetic time: {data.get('merge_arithmetic_seconds', '?')}s\n\n"

    costs = data.get("per_model_costs", [])
    if costs:
        o += "Per-model training costs:\n"
        o += f"  {'Model':<30s} {'GPU Hours':>10s} {'Epochs':>8s} {'Samples':>10s}\n"
        o += "  " + "-" * 60 + "\n"
        for c in costs:
            name = c.get("group_name", c.get("model_name", c.get("group", "?")))
            gpu_h = c.get("estimated_gpu_hours", c.get("gpu_hours", 0))
            epochs = c.get("n_epochs_completed", c.get("epochs_completed", "?"))
            n_train = c.get("n_train_samples", c.get("n_samples", "?"))
            o += f"  {name:<30s} {gpu_h:>10.2f} {str(epochs):>8s} {str(n_train):>10s}\n"

    scenarios = data.get("scenarios", [])
    if scenarios:
        o += "\nDeployment scenarios:\n"
        o += f"  {'Scenario':<35s} {'Train GPU-h':>12s} {'Merge (s)':>10s} {'Total GPU-h':>12s} {'Acc':>8s}\n"
        o += "  " + "-" * 79 + "\n"
        for s in scenarios:
            acc = s.get("accuracy")
            acc_str = pct(acc) if acc is not None else "—"
            o += (f"  {s.get('scenario', '?'):<35s} "
                  f"{s.get('training_gpu_hours', 0):>12.2f} "
                  f"{s.get('merge_seconds', 0):>10.2f} "
                  f"{s.get('total_gpu_hours', 0):>12.2f} "
                  f"{acc_str:>8s}\n")
        o += "\n"
        for s in scenarios:
            desc = s.get("description", "")
            if desc:
                o += f"  {s['scenario']}: {desc}\n"

    return o


# ──────────────────────────────────────────────────────────────────────
# 12. Fine-tuned model checkpoints
# ──────────────────────────────────────────────────────────────────────

def dump_checkpoints(R: Path) -> str:
    o = f"\n{'=' * W}\n12. FINE-TUNED MODEL CHECKPOINTS\n{'=' * W}\n\n"

    ft_dir = R / "finetuned"
    if not ft_dir.exists():
        return o + f"  NOT FOUND: {ft_dir}\n"

    o += f"{'Group':<30s} {'Checkpoint?':>12s} {'Size (MB)':>10s}\n"
    o += "-" * 54 + "\n"
    for d in sorted(ft_dir.iterdir()):
        if not d.is_dir():
            continue
        ckpt = d / "best_model.pt"
        exists = ckpt.exists()
        size = f"{ckpt.stat().st_size / 1e6:.0f}" if exists else "—"
        o += f"{d.name:<30s} {'Yes' if exists else 'NO':>12s} {size:>10s}\n"

    return o


# ──────────────────────────────────────────────────────────────────────
# 13. Task vector files
# ──────────────────────────────────────────────────────────────────────

def dump_task_vectors(R: Path) -> str:
    o = f"\n{'=' * W}\n13. TASK VECTOR FILES\n{'=' * W}\n\n"

    tv_dir = R / "task_vectors"
    if not tv_dir.exists():
        return o + f"  NOT FOUND: {tv_dir}\n"

    for f in sorted(tv_dir.glob("tau_*.pt")):
        o += f"  {f.name:<45s} {f.stat().st_size / 1e6:>8.1f} MB\n"

    return o


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="View all experiment results")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Path to results/ directory")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save to file (default: print to stdout)")
    args = parser.parse_args()

    R = Path(args.results_dir).resolve()
    if not R.exists():
        print(f"ERROR: results directory not found: {R}", file=sys.stderr)
        sys.exit(1)

    parts = []
    parts.append(f"# Complete Experiment Results\n# Source: {R}\n")
    parts.append(dump_checkpoints(R))
    parts.append(dump_task_vectors(R))
    parts.append(dump_lmc(R))
    parts.append(dump_composition(R))
    parts.append(dump_sparsity(R))
    parts.append(dump_norm_adjusted(R))
    parts.append(dump_regional(R))
    parts.append(dump_negation(R))
    parts.append(dump_bootstrap(R))
    parts.append(dump_knn(R))
    parts.append(dump_data_efficiency(R))
    parts.append(dump_continual(R))
    parts.append(dump_efficiency(R))

    text = "\n".join(parts)

    if args.output:
        Path(args.output).write_text(text)
        print(f"Saved to {args.output}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()