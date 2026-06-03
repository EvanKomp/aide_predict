#!/usr/bin/env python
"""
new_05c_bnn2_sensitivity.py — Training-Substrate Sensitivity Sweep
====================================================================

Orchestrates many substrate-split CV runs to characterise how each matched
(BNN reference mode, null) pair scales with the **number of training
substrates** available ("k"). For each held-out *active* target substrate S
we iterate over subsets of size k = 2 .. N-1 drawn from the remaining
substrates — actives and inactives are pooled and sampled uniformly. When
C(N-1, k) ≤ --exhaustive-threshold we enumerate all subsets; otherwise we
sample R = --max-samples-per-k random subsets (seeded).

Held-out targets remain *active-only* (the user-facing axis of "which
substrate are we predicting?"). The training pool, however, is the union of
actives and inactives minus the held-out target — every substrate is an
ordinary candidate, with no always-on injection. This is the change from
earlier sweeps, which forced inactives into every fold; their presence was
masking subset-level variability of the nearest-neighbor and reference-pool
nulls.

Each subset triggers:
    new_05_bnn2_train.py  --target-substrate S --subsample-train-substrates …
    new_05b_bnn2_score.py --run-dir …

After all runs finish, we read every ``metrics.json`` back in and produce a
summary table + a set of sensitivity plots (BNN vs matched null, by mode,
faceted by target substrate; and a pooled overlay).

Sensitivity outputs land under
``results/new_05_bnn2/sensitivity/<timestamp>/`` (or ``--output-dir``), which
is **separate** from a single comprehensive run produced by
``new_05_bnn2_train.py + new_05b_bnn2_score.py`` (which writes to
``results/new_05_bnn2/{split}/{run_id}/``). To get a single non-sensitivity
result with full training data, just invoke train+score directly with no
``--target-substrate`` / ``--subsample-train-substrates``.

Metrics tracked (per (mode, BNN/null, target_substrate, k, subset)):
  - Spearman ρ
  - MAE
  - NDCG (per-position mean)
  - Per-position top-1 recovery
  - Top-3 / top-5 global recovery

Subset variability (recommended ``--max-samples-per-k`` ≥ 10) gives mean ± std
across random subsets at every k.

Usage:
  python new_05c_bnn2_sensitivity.py --device cuda:0
  python new_05c_bnn2_sensitivity.py --held-out-substrates '["Formaldehyde"]' \\
                                     --max-samples-per-k 3
  python new_05c_bnn2_sensitivity.py --dry-run   # just print what would run
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from importlib.util import spec_from_file_location, module_from_spec

logger = logging.getLogger(__name__)

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _plot_style import apply_talk_style as _apply_talk_style
    _apply_talk_style()
except Exception as _e:
    logger.debug("apply_talk_style not applied: %s", _e)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

_common_spec = spec_from_file_location(
    "bnn2_common", SCRIPT_DIR / "05_bnn2_common.py")
_common = module_from_spec(_common_spec)
_common_spec.loader.exec_module(_common)

setup_logging = _common.setup_logging
load_config = _common.load_config
load_multi_substrate_data = _common.load_multi_substrate_data
load_substrate_metadata = _common.load_substrate_metadata

TRAIN_SCRIPT = SCRIPT_DIR / "new_05_bnn2_train.py"
SCORE_SCRIPT = SCRIPT_DIR / "new_05b_bnn2_score.py"

ALL_MODES = ("formaldehyde", "nearest", "avg_all", "distance_weighted")


def _subset_hash(subset: Tuple[str, ...]) -> str:
    return hashlib.md5(",".join(sorted(subset)).encode()).hexdigest()[:6]


def enumerate_subsets(
    available: List[str],
    max_samples_per_k: int,
    exhaustive_threshold: int,
    rng: random.Random,
    min_k: int = 2,
) -> List[Tuple[int, Tuple[str, ...]]]:
    """For each k in min_k..len(available), return a list of (k, subset) pairs.

    `available` is the candidate training pool for one held-out target — the
    union of actives and inactives, minus the held-out target. Subsets are
    drawn uniformly without distinguishing actives vs inactives.

    Exhaustive when C(N, k) ≤ exhaustive_threshold; otherwise sample
    `max_samples_per_k` unique subsets uniformly at random.

    `min_k` defaults to 2 because pairwise expansion requires ref ≠ target —
    a single training substrate produces zero training triplets and the fold
    is silently skipped, so k=1 runs contribute nothing to the summary.
    """
    available = sorted(available)
    n = len(available)
    pairs: List[Tuple[int, Tuple[str, ...]]] = []
    for k in range(min_k, n + 1):
        n_total = math.comb(n, k)
        if n_total <= exhaustive_threshold:
            for subset in itertools.combinations(available, k):
                pairs.append((k, subset))
        else:
            seen = set()
            while len(seen) < min(max_samples_per_k, n_total):
                subset = tuple(sorted(rng.sample(available, k)))
                seen.add(subset)
            for subset in sorted(seen):
                pairs.append((k, subset))
    return pairs


def _run_id_for(target: str, k: int, subset: Tuple[str, ...]) -> str:
    return f"S={target}_k={k}_h={_subset_hash(subset)}"


def plan_sweep(
    held_out: List[str],
    candidate_pool: List[str],
    max_samples_per_k: int,
    exhaustive_threshold: int,
    seed: int,
) -> List[Dict[str, object]]:
    """Build the full list of planned runs.

    `held_out` is the list of (active) substrates each iteration holds out.
    `candidate_pool` is every substrate eligible for training — both actives
    and inactives. For each target we drop only the target itself from the
    pool, then sample subsets of size k.
    """
    rng = random.Random(seed)
    plan: List[Dict[str, object]] = []
    for target in sorted(held_out):
        remaining = [s for s in candidate_pool if s != target]
        subsets = enumerate_subsets(
            remaining, max_samples_per_k, exhaustive_threshold, rng)
        for k, subset in subsets:
            plan.append({
                "target_substrate": target,
                "k": k,
                "subset": list(subset),
                "run_id": _run_id_for(target, k, subset),
            })
    return plan


def run_one(
    item: Dict[str, object],
    sweep_root: Path,
    device: Optional[str],
    extra_train_args: List[str],
    dry_run: bool,
) -> Tuple[bool, Optional[Path]]:
    """Execute one (train → score) pair. Returns (success, run_dir)."""
    target = str(item["target_substrate"])
    run_id = str(item["run_id"])
    subset = list(item["subset"])  # type: ignore[arg-type]
    run_dir = sweep_root / "substrate" / run_id
    train_cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--split", "substrate",
        "--target-substrate", target,
        "--subsample-train-substrates", json.dumps(subset),
        "--run-id", run_id,
        "--output-root", str(sweep_root),
        "--skip-final-model",
    ]
    if device:
        train_cmd.extend(["--device", device])
    train_cmd.extend(extra_train_args)

    score_cmd = [
        sys.executable, str(SCORE_SCRIPT),
        "--run-dir", str(run_dir),
    ]

    logger.info("▶ %s  k=%d  subs=%s", target, item["k"], subset)
    if dry_run:
        logger.info("  train: %s", " ".join(train_cmd))
        logger.info("  score: %s", " ".join(score_cmd))
        return True, run_dir

    metrics_path = run_dir / "scoring" / "metrics.json"
    if metrics_path.exists():
        logger.info("  ↻ resume: metrics.json present — skipping")
        return True, run_dir

    try:
        subprocess.run(train_cmd, check=True)
        subprocess.run(score_cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error("  run failed: %s", e)
        return False, run_dir
    return True, run_dir


# ═══════════════════════════════════════════════════════════════════════════
# Summarisation + plots
# ═══════════════════════════════════════════════════════════════════════════

def _safe_get(d: dict, *path, default=float("nan")):
    """Walk nested dicts; return default if any key is missing or value is None."""
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur or cur[p] is None:
            return default
        cur = cur[p]
    return cur


def collect_metrics(sweep_root: Path, plan: List[Dict[str, object]]) -> pd.DataFrame:
    """Walk every planned run and pull headline metrics into long form.

    Pulls Spearman, MAE, NDCG (per-position mean and "overall" per-substrate
    mean), per-position top-1 recovery, and overall top-3 / top-5 recovery
    (per-substrate mean) for both BNN and matched null. All "overall" metrics
    are computed within each held-out substrate then equal-weight averaged —
    they're no longer pooled across substrates. Missing fields (older runs)
    fall back to NaN.
    """
    rows = []
    for item in plan:
        run_dir = sweep_root / "substrate" / str(item["run_id"])
        metrics_path = run_dir / "scoring" / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        for mode, mr in metrics.get("modes", {}).items():
            m = mr["metrics"]
            bnn = m.get("bnn", {})
            null = m.get("null", {})
            rows.append({
                "target_substrate": item["target_substrate"],
                "k": int(item["k"]),
                "subset_hash": _subset_hash(tuple(item["subset"])),  # type: ignore[arg-type]
                "subset": ",".join(sorted(item["subset"])),          # type: ignore[arg-type]
                "mode": mode,
                # Overall regression-style metrics (per-substrate mean)
                "bnn_spearman": _safe_get(bnn, "spearman_rho"),
                "null_spearman": _safe_get(null, "spearman_rho"),
                "bnn_mae": _safe_get(bnn, "mae"),
                "null_mae": _safe_get(null, "mae"),
                # NDCG (per-position mean — within-(substrate, position) two-step)
                "bnn_ndcg": _safe_get(bnn, "per_position_ndcg_mean", "mean"),
                "null_ndcg": _safe_get(null, "per_position_ndcg_mean", "mean"),
                # NDCG (overall — within-substrate then averaged across substrates)
                "bnn_ndcg_overall": _safe_get(bnn, "ndcg"),
                "null_ndcg_overall": _safe_get(null, "ndcg"),
                # Per-position top-1 recovery
                "bnn_per_pos_top1": _safe_get(bnn, "per_position_top1_recovery", "recovery"),
                "null_per_pos_top1": _safe_get(null, "per_position_top1_recovery", "recovery"),
                # Overall top-k recovery (per-substrate mean)
                "bnn_top3": _safe_get(bnn, "top3_recovery", "recovery"),
                "null_top3": _safe_get(null, "top3_recovery", "recovery"),
                "bnn_top5": _safe_get(bnn, "top5_recovery", "recovery"),
                "null_top5": _safe_get(null, "top5_recovery", "recovery"),
                # Deltas
                "delta_spearman": m.get("delta_spearman", float("nan")),
                "delta_mae": m.get("delta_mae", float("nan")),
                "n_rows": m.get("n_rows", 0),
            })
    return pd.DataFrame(rows)


def plot_sensitivity_grid(
    df: pd.DataFrame,
    out_path: Path,
    metric_col: str = "bnn_spearman",
    null_col: str = "null_spearman",
    ylabel: str = "Spearman ρ",
):
    """Grid of (mode rows × target columns) showing scatter + mean BNN line
    and matched-null line per facet. One reference mode per row makes mode-vs-
    mode comparisons within a target easy to read; per-target columns let you
    spot held-out substrates that behave anomalously (e.g. Glycoaldehyde /
    Pyruvate where the nearest-null collapses to a constant fallback)."""
    if df.empty:
        logger.warning("No metrics to plot — skipping %s", out_path)
        return

    targets = sorted(df["target_substrate"].unique())
    modes = [m for m in ALL_MODES if m in df["mode"].unique()]
    if not targets or not modes:
        return
    nrows = len(modes)
    ncols = len(targets)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(3.6 * ncols, 3.0 * nrows),
                              squeeze=False, sharex=True, sharey=True)

    bnn_color = "#4477aa"
    null_color = "#ee6677"

    for ri, mode in enumerate(modes):
        for ci, target in enumerate(targets):
            ax = axes[ri][ci]
            mdf = df[(df["target_substrate"] == target) & (df["mode"] == mode)]
            if mdf.empty:
                ax.set_facecolor("#f7f7f7")
                ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                        transform=ax.transAxes, color="#888", fontsize=10)
            else:
                ax.scatter(mdf["k"], mdf[metric_col], color=bnn_color, alpha=0.7,
                           s=42, edgecolors="black", linewidths=1.0,
                           label="BNN" if (ri == 0 and ci == 0) else None)
                bnn_mean = mdf.groupby("k")[metric_col].mean().sort_index()
                ax.plot(bnn_mean.index, bnn_mean.values, color=bnn_color, lw=2.5)

                null_mean = mdf.groupby("k")[null_col].mean().sort_index()
                ax.plot(null_mean.index, null_mean.values, color=null_color,
                        lw=2.0, ls="--",
                        label="Null" if (ri == 0 and ci == 0) else None)
            if ri == 0:
                ax.set_title(target, fontsize=11)
            if ci == 0:
                ax.set_ylabel(f"{mode}\n{ylabel}", fontsize=10)
            if ri == nrows - 1:
                ax.set_xlabel("k")
            ax.axhline(0, color="#444", lw=1.0)
            ax.grid(ls="--", lw=0.6, alpha=0.4)

    handles = [
        plt.Line2D([0], [0], color=bnn_color, lw=2.5, marker="o",
                   markeredgecolor="black", linestyle="-", label="BNN (mean)"),
        plt.Line2D([0], [0], color=null_color, lw=2.0, ls="--", label="Null (mean)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.01))
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _null_subset_variance_meaningful(
    mdf: pd.DataFrame,
    null_col: str,
) -> bool:
    """True iff the null actually varies across subsets *within* some (target, k).

    Several null modes are deterministic given the target:
      - ``formaldehyde``: predicts from a fixed always-on reference substrate.
      - ``nearest``: collapses to a single training substrate (and that nearest
        is often an always-on inactive, making it bit-identical across subsets).
    For these, a "std across subsets" band is misleading — it'd reflect
    target-to-target heterogeneity, not subset variability. We only render
    the null band/error if at least one (target, k) cell has >1 distinct null.
    """
    g = mdf.groupby(["target_substrate", "k"])[null_col].nunique(dropna=True)
    return bool((g > 1).any())


def _two_step_pool(mdf_: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per-target mean & subset-std, then average over targets at each k.

    Returns a DataFrame indexed by k with columns ``mean`` (mean of per-target
    means) and ``std`` (mean of per-target subset stds). Keeps the band
    interpretable as 'typical subset spread' rather than a mix of target-to-
    target heterogeneity + subset noise.
    """
    per_tgt = (mdf_.groupby(["target_substrate", "k"])[col]
                   .agg(["mean", "std", "count"]).reset_index())
    per_tgt["std"] = per_tgt["std"].fillna(0.0)
    agg = per_tgt.groupby("k").agg(mean=("mean", "mean"),
                                    std=("std", "mean")).sort_index()
    return agg


def _apply_transform_and_clip(
    by_k: pd.DataFrame,
    transform: Optional[str],
    yscale: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply optional `transform` to means, then return (mean, low, high) arrays
    suitable for plotting under the given `yscale`. With `yscale="log"` we clip
    non-positive values to a small epsilon so log can render them."""
    mean = by_k["mean"].astype(float).to_numpy().copy()
    std = by_k["std"].fillna(0.0).astype(float).to_numpy()
    if transform == "regret":
        mean = 1.0 - mean
    low = mean - std
    high = mean + std
    if yscale == "log":
        eps = 1e-4
        mean = np.clip(mean, eps, None)
        low = np.clip(low, eps, None)
        high = np.clip(high, eps, None)
    return mean, low, high


def plot_sensitivity_pooled(
    df: pd.DataFrame,
    out_path: Path,
    metric_col: str = "bnn_spearman",
    null_col: str = "null_spearman",
    ylabel: str = "Spearman ρ",
    yscale: str = "linear",
    transform: Optional[str] = None,
):
    """Pooled across targets: mean ± std of metric vs k, one panel per mode.

    BNN gets a mean line + ±1 std band across (target, subset) rows. Null gets
    a mean line; a std band is drawn ONLY when the null is actually subset-
    sensitive within some (target, k) cell — otherwise the "std" would be
    target-to-target heterogeneity dressed up as subset variability, which is
    misleading for deterministic null modes (formaldehyde, often nearest).

    `yscale="log"` and `transform="regret"` (plot 1 − metric) are useful for
    [0,1]-bounded metrics: log raw highlights small absolute values (top-k
    recoveries), regret-log highlights tiny gaps near 1 (e.g. NDCG).
    """
    if df.empty:
        return
    modes = [m for m in ALL_MODES if m in df["mode"].unique()]
    if not modes:
        return

    n = len(modes)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(6.5 * ncols, 4.0 * nrows),
                              squeeze=False, sharex=True, sharey=True)
    bnn_color = "#4477aa"
    null_color = "#ee6677"

    eff_ylabel = f"1 − {ylabel}" if transform == "regret" else ylabel

    for mi, mode in enumerate(modes):
        ax = axes[mi // ncols][mi % ncols]
        mdf = df[df["mode"] == mode]
        bnn_by_k = _two_step_pool(mdf, metric_col)
        null_by_k = _two_step_pool(mdf, null_col)

        bnn_mean, bnn_low, bnn_high = _apply_transform_and_clip(bnn_by_k, transform, yscale)
        null_mean, null_low, null_high = _apply_transform_and_clip(null_by_k, transform, yscale)

        # BNN: line + band
        ax.plot(bnn_by_k.index, bnn_mean, color=bnn_color, lw=3.0,
                marker="o", label="BNN (mean ± std across subsets)")
        ax.fill_between(bnn_by_k.index, bnn_low, bnn_high,
                         color=bnn_color, alpha=0.20)

        # Null: line; band only if subset variability exists
        null_varies = _null_subset_variance_meaningful(mdf, null_col)
        null_label = ("Null (mean ± std)" if null_varies
                      else "Null (deterministic per target — no std)")
        ax.plot(null_by_k.index, null_mean, color=null_color, lw=2.4,
                ls="--", marker="x", label=null_label)
        if null_varies:
            ax.fill_between(null_by_k.index, null_low, null_high,
                             color=null_color, alpha=0.18)

        if yscale == "linear":
            ax.axhline(0, color="#444", lw=1.0)
        ax.set_yscale(yscale)
        ax.set_title(f"reference mode = {mode}")
        ax.set_xlabel("k (training substrates)")
        ax.set_ylabel(eff_ylabel)
        ax.grid(ls="--", lw=0.8, alpha=0.4)
        ax.legend(loc="best", fontsize=10)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    suffix = ""
    if yscale == "log":
        suffix += "  [log y]"
    if transform == "regret":
        suffix += "  [regret = 1 − metric]"
    fig.suptitle(f"Sensitivity to number of training substrates (pooled){suffix}", y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_best_pooled(
    df: pd.DataFrame,
    out_path: Path,
    metric_col: str,
    null_col: str,
    ylabel: str,
    higher_is_better: bool = True,
    yscale: str = "linear",
    transform: Optional[str] = None,
):
    """Single-axis pooled plot: best BNN reference mode vs best null reference
    mode for the given metric. 'Best' picks the mode whose mean across all
    rows of `df` maximises (or minimises, if `higher_is_better=False`) the
    raw metric — selected BEFORE any transform. The chosen mode names appear
    in the legend so it's obvious which reference mode is being compared.
    """
    if df.empty:
        return
    modes = [m for m in ALL_MODES if m in df["mode"].unique()]
    if not modes:
        return

    by_mode_bnn = df.groupby("mode")[metric_col].mean().reindex(modes).dropna()
    by_mode_null = df.groupby("mode")[null_col].mean().reindex(modes).dropna()
    if by_mode_bnn.empty or by_mode_null.empty:
        return
    if higher_is_better:
        best_bnn_mode = str(by_mode_bnn.idxmax())
        best_null_mode = str(by_mode_null.idxmax())
    else:
        best_bnn_mode = str(by_mode_bnn.idxmin())
        best_null_mode = str(by_mode_null.idxmin())

    bnn_color = "#4477aa"
    null_color = "#ee6677"

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    eff_ylabel = f"1 − {ylabel}" if transform == "regret" else ylabel

    bnn_mdf = df[df["mode"] == best_bnn_mode]
    null_mdf = df[df["mode"] == best_null_mode]
    bnn_by_k = _two_step_pool(bnn_mdf, metric_col)
    null_by_k = _two_step_pool(null_mdf, null_col)

    bnn_mean, bnn_low, bnn_high = _apply_transform_and_clip(bnn_by_k, transform, yscale)
    null_mean, null_low, null_high = _apply_transform_and_clip(null_by_k, transform, yscale)

    ax.plot(bnn_by_k.index, bnn_mean, color=bnn_color, lw=3.0,
            marker="o", label=f"BNN — best mode: {best_bnn_mode}")
    ax.fill_between(bnn_by_k.index, bnn_low, bnn_high,
                     color=bnn_color, alpha=0.20)

    null_varies = _null_subset_variance_meaningful(null_mdf, null_col)
    null_label = (f"Null — best mode: {best_null_mode}"
                  if null_varies
                  else f"Null — best mode: {best_null_mode} (deterministic)")
    ax.plot(null_by_k.index, null_mean, color=null_color, lw=2.4,
            ls="--", marker="x", label=null_label)
    if null_varies:
        ax.fill_between(null_by_k.index, null_low, null_high,
                         color=null_color, alpha=0.18)

    if yscale == "linear":
        ax.axhline(0, color="#444", lw=1.0)
    ax.set_yscale(yscale)
    direction = "higher = better" if higher_is_better else "lower = better"
    suffix = ""
    if yscale == "log":
        suffix += "  [log y]"
    if transform == "regret":
        suffix += "  [regret = 1 − metric]"
    ax.set_title(f"Best BNN mode vs best null mode  ({direction}){suffix}")
    ax.set_xlabel("k (training substrates)")
    ax.set_ylabel(eff_ylabel)
    ax.grid(ls="--", lw=0.8, alpha=0.4)
    ax.legend(loc="best", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# Stable color per reference mode for any plot that mixes modes on one axis.
_MODE_COLORS: Dict[str, str] = {
    "formaldehyde":      "#4477aa",
    "nearest":           "#ee6677",
    "avg_all":           "#228833",
    "distance_weighted": "#ccbb44",
}


def plot_sensitivity_best_per_k(
    df: pd.DataFrame,
    out_path: Path,
    metric_col: str,
    null_col: str,
    ylabel: str,
    higher_is_better: bool = True,
    yscale: str = "linear",
    transform: Optional[str] = None,
):
    """Same idea as `plot_sensitivity_best_pooled`, but the winning reference
    mode is picked **independently at each k** for both BNN and null. Marker
    color = chosen mode at that k; BNN vs null is distinguished by marker
    shape (filled circle vs ×) and the gray connector linestyle (solid vs
    dashed). Error bars use the same two-step subset-std pooling.

    Use this when you suspect a mode-crossover with k — e.g. a mode that
    dominates at large k but loses to nearest-neighbor at k=2.
    """
    if df.empty:
        return
    modes = [m for m in ALL_MODES if m in df["mode"].unique()]
    if not modes:
        return

    bnn_pools = {m: _two_step_pool(df[df["mode"] == m], metric_col) for m in modes}
    null_pools = {m: _two_step_pool(df[df["mode"] == m], null_col) for m in modes}
    null_varies_by_mode = {
        m: _null_subset_variance_meaningful(df[df["mode"] == m], null_col)
        for m in modes
    }

    all_ks = sorted({k for p in bnn_pools.values() for k in p.index}
                    | {k for p in null_pools.values() for k in p.index})

    def _winner(pools, k):
        cands = [(m, float(p.loc[k, "mean"]), float(p.loc[k, "std"]))
                 for m, p in pools.items()
                 if k in p.index and np.isfinite(p.loc[k, "mean"])]
        if not cands:
            return None
        return (max if higher_is_better else min)(cands, key=lambda x: x[1])

    bnn_w = [(k, _winner(bnn_pools, k)) for k in all_ks]
    null_w = [(k, _winner(null_pools, k)) for k in all_ks]

    eps = 1e-4

    def _xy(winners, is_null):
        xs, ys, errs, colors = [], [], [], []
        for k, w in winners:
            if w is None:
                continue
            mode, mean, std = w
            y = (1.0 - mean) if transform == "regret" else mean
            std_use = std if (not is_null) or null_varies_by_mode.get(mode, False) else 0.0
            lo = y - std_use
            hi = y + std_use
            if yscale == "log":
                yp = max(y, eps)
                lop = max(lo, eps)
                hip = max(hi, eps)
                xs.append(k); ys.append(yp)
                errs.append([yp - lop, hip - yp])
            else:
                xs.append(k); ys.append(y)
                errs.append([std_use, std_use])
            colors.append(_MODE_COLORS.get(mode, "#888"))
        return xs, ys, errs, colors

    bnn_xs, bnn_ys, bnn_errs, bnn_colors = _xy(bnn_w, is_null=False)
    null_xs, null_ys, null_errs, null_colors = _xy(null_w, is_null=True)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    eff_ylabel = f"1 − {ylabel}" if transform == "regret" else ylabel

    # Gray connector lines (BNN solid, null dashed) so the eye tracks across k
    ax.plot(bnn_xs, bnn_ys, color="#888", lw=1.6, ls="-",  alpha=0.55, zorder=1)
    ax.plot(null_xs, null_ys, color="#888", lw=1.6, ls="--", alpha=0.55, zorder=1)

    # Markers + per-point error bars colored by chosen mode
    for x, y, e, c in zip(bnn_xs, bnn_ys, bnn_errs, bnn_colors):
        yerr = np.array([[e[0]], [e[1]]])
        ax.errorbar(x, y, yerr=yerr, fmt="o", markersize=11, color=c,
                    markeredgecolor="black", markeredgewidth=1.2,
                    elinewidth=1.5, capsize=4, zorder=3)
    for x, y, e, c in zip(null_xs, null_ys, null_errs, null_colors):
        yerr = np.array([[e[0]], [e[1]]])
        ax.errorbar(x, y, yerr=yerr, fmt="X", markersize=12, color=c,
                    markeredgecolor="black", markeredgewidth=1.2,
                    elinewidth=1.5, capsize=4, zorder=3)

    if yscale == "linear":
        ax.axhline(0, color="#444", lw=1.0)
    ax.set_yscale(yscale)
    direction = "higher = better" if higher_is_better else "lower = better"
    suffix = ""
    if yscale == "log":
        suffix += "  [log y]"
    if transform == "regret":
        suffix += "  [regret = 1 − metric]"
    ax.set_title(f"Best reference mode at each k  ({direction}){suffix}")
    ax.set_xlabel("k (training substrates)")
    ax.set_ylabel(eff_ylabel)
    ax.grid(ls="--", lw=0.8, alpha=0.4)

    mode_handles = [
        plt.Line2D([0], [0], marker="s", linestyle="none",
                   markerfacecolor=_MODE_COLORS.get(m, "#888"),
                   markeredgecolor="black", markersize=10, label=m)
        for m in modes
    ]
    shape_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="-", color="#888",
                   markerfacecolor="white", markeredgecolor="black",
                   markersize=10, label="BNN"),
        plt.Line2D([0], [0], marker="X", linestyle="--", color="#888",
                   markerfacecolor="white", markeredgecolor="black",
                   markersize=11, label="Null"),
    ]
    leg1 = ax.legend(handles=mode_handles, title="reference mode",
                     loc="upper left", fontsize=9, title_fontsize=10)
    ax.add_artist(leg1)
    ax.legend(handles=shape_handles, loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _draw_mode_bars(
    ax,
    mode_df: pd.DataFrame,
    metric_col: str,
    null_col: str,
    x_center: float,
    bar_width: float,
    cmap,
    show_legend_handles: dict,
    ks_global,
):
    """Render one mode-group on `ax` at center x_center.

    `show_legend_handles` is a dict that the caller seeds and reads back
    after one mode draws — used to dedupe k-color legend entries.
    """
    if mode_df.empty:
        return
    grp_bnn = mode_df.groupby("k")[metric_col].agg(["mean", "std", "count"]).sort_index()
    grp_null = mode_df.groupby("k")[null_col].agg(["mean"]).sort_index()
    ks = sorted(grp_bnn.index.tolist())
    n_k = len(ks_global)
    # Symmetric offsets across the full k range so x positions align between modes
    k_to_offset = {k: (ks_global.index(k) - (n_k - 1) / 2) * bar_width for k in ks_global}

    for k in ks:
        x = x_center + k_to_offset[k]
        mean_b = grp_bnn.loc[k, "mean"]
        std_b = grp_bnn.loc[k, "std"]
        if not np.isfinite(mean_b):
            continue
        color = cmap((ks_global.index(k)) / max(n_k - 1, 1))
        bar = ax.bar(x, mean_b, bar_width * 0.9,
                     yerr=(std_b if np.isfinite(std_b) else 0.0),
                     color=color, edgecolor="black", linewidth=1.5,
                     error_kw={"elinewidth": 1.5, "capsize": 4, "alpha": 0.9})
        if k not in show_legend_handles:
            show_legend_handles[k] = bar[0]

        if k in grp_null.index:
            mean_n = grp_null.loc[k, "mean"]
            if np.isfinite(mean_n):
                ax.hlines(mean_n,
                          x - bar_width * 0.45, x + bar_width * 0.45,
                          colors="black", linestyles="-", linewidth=2.8, zorder=5)


def plot_grouped_bars_by_mode_and_k(
    df: pd.DataFrame,
    out_path: Path,
    metric_col: str,
    null_col: str,
    ylabel: str,
    title: str,
):
    """Pooled across held-out targets.

    Outer x = reference mode; inner bars = one per k (BNN mean across subsets,
    yerr = std). A short black horizontal line atop each bar marks the matched
    null mean at the same (mode, k).
    """
    if df.empty:
        logger.warning("Empty df — skipping %s", out_path)
        return
    modes = [m for m in ALL_MODES if m in df["mode"].unique()]
    if not modes:
        return
    ks_global = sorted(df["k"].unique().tolist())
    n_k = len(ks_global)

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(modes) * n_k * 0.45 + 4), 5.0))
    cmap = plt.cm.viridis
    bar_width = 0.8 / max(n_k, 1)
    x_centers = np.arange(len(modes))
    legend_handles: dict = {}

    for mi, mode in enumerate(modes):
        mode_df = df[df["mode"] == mode]
        _draw_mode_bars(
            ax, mode_df, metric_col, null_col,
            x_center=float(x_centers[mi]),
            bar_width=bar_width, cmap=cmap,
            show_legend_handles=legend_handles, ks_global=ks_global,
        )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(modes)
    ax.axhline(0, color="#444", lw=1.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title + "  (cap line = matched null)")
    ax.grid(axis="y", ls="--", lw=0.8, alpha=0.4)

    # Legend: k entries + a separate handle for the null cap line
    handles = [legend_handles[k] for k in sorted(legend_handles.keys())]
    labels = [f"k={k}" for k in sorted(legend_handles.keys())]
    null_handle = plt.Line2D([0], [0], color="black", lw=2.8)
    handles.append(null_handle)
    labels.append("null (mode-matched)")
    ax.legend(handles, labels, ncol=min(len(handles), 6),
              loc="upper center", bbox_to_anchor=(0.5, -0.10))

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_bars_by_mode_and_k_per_target(
    df: pd.DataFrame,
    out_path: Path,
    metric_col: str,
    null_col: str,
    ylabel: str,
    title: str,
):
    """Same as plot_grouped_bars_by_mode_and_k but faceted by held-out target."""
    if df.empty:
        return
    targets = sorted(df["target_substrate"].unique())
    n = len(targets)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    modes = [m for m in ALL_MODES if m in df["mode"].unique()]
    if not modes:
        return
    ks_global = sorted(df["k"].unique().tolist())
    n_k = len(ks_global)

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(max(6, 1.4 * len(modes) * n_k * 0.4 + 3) * ncols,
                                       4.5 * nrows),
                              squeeze=False, sharey=True)
    cmap = plt.cm.viridis
    bar_width = 0.8 / max(n_k, 1)
    x_centers = np.arange(len(modes))
    legend_handles: dict = {}

    for ti, target in enumerate(targets):
        ax = axes[ti // ncols][ti % ncols]
        sub = df[df["target_substrate"] == target]
        for mi, mode in enumerate(modes):
            mode_df = sub[sub["mode"] == mode]
            _draw_mode_bars(
                ax, mode_df, metric_col, null_col,
                x_center=float(x_centers[mi]),
                bar_width=bar_width, cmap=cmap,
                show_legend_handles=legend_handles, ks_global=ks_global,
            )
        ax.set_xticks(x_centers)
        ax.set_xticklabels(modes, rotation=15, ha="right")
        ax.axhline(0, color="#444", lw=1.5)
        ax.set_title(f"held out = {target}")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", ls="--", lw=0.8, alpha=0.4)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    handles = [legend_handles[k] for k in sorted(legend_handles.keys())]
    labels = [f"k={k}" for k in sorted(legend_handles.keys())]
    null_handle = plt.Line2D([0], [0], color="black", lw=2.8)
    handles.append(null_handle)
    labels.append("null (mode-matched)")
    fig.legend(handles, labels, ncol=min(len(handles), 8),
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title + "  (cap line = matched null)", y=1.0)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# CLI + main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training-substrate sensitivity sweep over BNN2 substrate-split CV.",
    )
    parser.add_argument("--held-out-substrates", type=str, default=None,
                        help="JSON list of substrates to hold out (default: all active)")
    parser.add_argument("--max-samples-per-k", type=int, default=10)
    parser.add_argument("--exhaustive-threshold", type=int, default=30,
                        help="Enumerate all subsets when C(N-1,k) ≤ this value")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Root dir for sweep outputs (default: "
                             "results/new_05_bnn2/sensitivity/<timestamp>)")
    parser.add_argument("--device", type=str, default=None,
                        help="Passed through to new_05_bnn2_train.py")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seeds the RNG used to sample subsets")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned runs, don't execute")
    parser.add_argument("--summary-only", action="store_true",
                        help="Skip training; re-summarise an existing sweep directory "
                             "(requires --output-dir pointing at a prior sweep)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("train_args", nargs=argparse.REMAINDER,
                        help="Extra args forwarded to new_05_bnn2_train.py (prefix with --).")
    return parser.parse_args()


def main():
    args = parse_args()
    t_start = time.time()

    held_out_cli = json.loads(args.held_out_substrates) if args.held_out_substrates else None

    # Output directory
    if args.output_dir:
        sweep_root = Path(args.output_dir).resolve()
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        sweep_root = PROJECT_ROOT / "results" / "new_05_bnn2" / "sensitivity" / stamp
    sweep_root.mkdir(parents=True, exist_ok=True)
    summary_dir = sweep_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(summary_dir / "sweep.log")
    logger.info("Sweep root: %s", sweep_root)

    # Refuse to resume into a pre-redesign sweep dir. Earlier runs forced
    # inactive substrates into every fold; their cached metrics.json files
    # would be silently re-used because _subset_hash is name-only. Honour
    # --summary-only on existing redesigned dirs (the marker will be present)
    # but block both run and summary on stale dirs.
    plan_path = sweep_root / "sweep_plan.json"
    if plan_path.exists():
        try:
            with open(plan_path) as f:
                prior = json.load(f)
        except Exception:
            prior = {}
        if not prior.get("sweep_axis_includes_inactives", False):
            raise SystemExit(
                f"Refusing to write into {sweep_root}: existing sweep_plan.json "
                "was produced by the pre-redesign sweep (inactives forced into "
                "every fold). Pass a fresh --output-dir or delete the stale "
                "sweep_plan.json to proceed."
            )

    # Resolve active substrates from project data
    config = load_config(args.config)
    processed_dir = PROJECT_ROOT / config["data"]["output_dir"]
    df = load_multi_substrate_data(processed_dir)
    substrate_meta = load_substrate_metadata(processed_dir)
    data_substrates = set(df["substrate"].unique())
    active_substrates = sorted(
        s for s, m in substrate_meta.items()
        if m.get("is_active", True) and s in data_substrates
    )
    inactive_substrates = sorted(
        s for s, m in substrate_meta.items()
        if not m.get("is_active", True) and s in data_substrates
    )
    candidate_pool = sorted(set(active_substrates) | set(inactive_substrates))
    logger.info("Held-out targets (sweep loop, actives only): %s", active_substrates)
    logger.info("Training-pool candidates (any active or inactive, minus held-out): %s",
                candidate_pool)

    held_out = sorted(held_out_cli) if held_out_cli else active_substrates

    plan = plan_sweep(
        held_out=held_out,
        candidate_pool=candidate_pool,
        max_samples_per_k=args.max_samples_per_k,
        exhaustive_threshold=args.exhaustive_threshold,
        seed=args.seed,
    )
    logger.info("Plan: %d runs across %d held-out substrates", len(plan), len(held_out))
    with open(sweep_root / "sweep_plan.json", "w") as f:
        json.dump({
            "held_out": held_out,
            "active_substrates": active_substrates,
            "inactive_substrates": inactive_substrates,
            "candidate_pool": candidate_pool,
            "sweep_axis_includes_inactives": True,
            "max_samples_per_k": args.max_samples_per_k,
            "exhaustive_threshold": args.exhaustive_threshold,
            "seed": args.seed,
            "plan": plan,
        }, f, indent=2)

    # Execute
    if not args.summary_only:
        failures = 0
        for i, item in enumerate(plan):
            logger.info("── [%d/%d] ──", i + 1, len(plan))
            ok, _ = run_one(
                item=item,
                sweep_root=sweep_root,
                device=args.device,
                extra_train_args=[a for a in (args.train_args or []) if a != "--"],
                dry_run=args.dry_run,
            )
            if not ok:
                failures += 1
        logger.info("Sweep finished: %d/%d succeeded (%d failures)",
                    len(plan) - failures, len(plan), failures)

    if args.dry_run:
        logger.info("Dry run — skipping summary.")
        return

    # Summarise
    df_metrics = collect_metrics(sweep_root, plan)
    if df_metrics.empty:
        logger.warning("No metrics collected — did any run succeed?")
        return
    summary_dir.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(summary_dir / "sensitivity_summary.csv", index=False)
    logger.info("Wrote sensitivity_summary.csv (%d rows)", len(df_metrics))

    # Per-target grid + pooled (line + ±1 std band) for every headline metric.
    # `extra_views` lists the additional yscale/transform variants for each
    # metric. For metrics bounded in [0,1]:
    #   - "log_raw"     — log y on the metric directly; expands small values
    #                     (right call for top-k recoveries that hover near 0)
    #   - "log_regret"  — plot 1 − metric on a log axis; expands tiny gaps
    #                     near 1 (right call for NDCG which often peaks high)
    # Spearman/Δ-Spearman/MAE skip log because their range isn't [0,1].
    pooled_specs = [
        # (metric_col, null_col, ylabel, stem, higher_is_better, extra_views)
        ("bnn_spearman",     "null_spearman",     "Spearman ρ (per-substrate mean)",        "spearman",       True,  []),
        ("bnn_mae",          "null_mae",          "MAE (log_fc, per-substrate mean)",       "mae",            False, []),
        ("delta_spearman",   "delta_spearman",    "Δ Spearman (BNN − null)",                "delta_spearman", True,  []),
        ("bnn_ndcg",         "null_ndcg",         "NDCG (per-position mean)",               "ndcg",           True,  ["log_raw", "log_regret"]),
        ("bnn_ndcg_overall", "null_ndcg_overall", "NDCG (overall, per-substrate mean)",     "ndcg_overall",   True,  ["log_raw", "log_regret"]),
        ("bnn_per_pos_top1", "null_per_pos_top1", "Per-position top-1 recovery",            "per_pos_top1",   True,  ["log_raw"]),
        ("bnn_top3",         "null_top3",         "Top-3 recovery (per-substrate mean)",    "top3",           True,  ["log_raw"]),
        ("bnn_top5",         "null_top5",         "Top-5 recovery (per-substrate mean)",    "top5",           True,  ["log_raw"]),
    ]

    def _view_kwargs(view: str) -> Dict[str, object]:
        if view == "linear":
            return {"yscale": "linear", "transform": None}
        if view == "log_raw":
            return {"yscale": "log", "transform": None}
        if view == "log_regret":
            return {"yscale": "log", "transform": "regret"}
        raise ValueError(view)

    for metric_col, null_col, ylabel, stem, higher_is_better, extra_views in pooled_specs:
        all_views = ["linear"] + list(extra_views)
        for view in all_views:
            view_suffix = "" if view == "linear" else f"_{view}"
            view_kw = _view_kwargs(view)
            try:
                plot_sensitivity_pooled(
                    df_metrics,
                    summary_dir / f"sensitivity_{stem}_pooled{view_suffix}.png",
                    metric_col=metric_col, null_col=null_col, ylabel=ylabel,
                    **view_kw)
                plot_sensitivity_best_pooled(
                    df_metrics,
                    summary_dir / f"sensitivity_{stem}_best{view_suffix}.png",
                    metric_col=metric_col, null_col=null_col, ylabel=ylabel,
                    higher_is_better=higher_is_better,
                    **view_kw)
                plot_sensitivity_best_per_k(
                    df_metrics,
                    summary_dir / f"sensitivity_{stem}_best_per_k{view_suffix}.png",
                    metric_col=metric_col, null_col=null_col, ylabel=ylabel,
                    higher_is_better=higher_is_better,
                    **view_kw)
            except Exception as e:
                logger.warning("Failed to render sensitivity_%s_*%s: %s",
                               stem, view_suffix, e)
        # Per-target grid is linear-only — log scaling 24+ small panels rarely
        # helps and the comparison the grid is meant to support (held-out
        # target × reference mode) is read off the shape, not the absolute y.
        try:
            plot_sensitivity_grid(
                df_metrics, summary_dir / f"sensitivity_{stem}_by_target.png",
                metric_col=metric_col, null_col=null_col, ylabel=ylabel)
        except Exception as e:
            logger.warning("Failed to render sensitivity_%s_by_target: %s", stem, e)

    # Grouped-bar plots: outer = reference mode, inner = bars by k,
    # cap line on each bar = matched null at that (mode, k).
    bar_plot_specs = [
        ("bnn_ndcg",         "null_ndcg",         "NDCG (per-position mean)",                 "bars_ndcg_by_mode_k"),
        ("bnn_ndcg_overall", "null_ndcg_overall", "NDCG (overall, per-substrate mean)",       "bars_ndcg_overall_by_mode_k"),
        ("bnn_spearman",     "null_spearman",     "Spearman ρ (per-substrate mean)",          "bars_spearman_by_mode_k"),
        ("bnn_per_pos_top1", "null_per_pos_top1", "Per-position top-1 recovery",              "bars_per_pos_top1_by_mode_k"),
        ("bnn_top5",         "null_top5",         "Top-5 recovery (per-substrate mean)",      "bars_top5_by_mode_k"),
    ]
    for metric_col, null_col, ylabel, fname_stem in bar_plot_specs:
        try:
            plot_grouped_bars_by_mode_and_k(
                df_metrics, summary_dir / f"{fname_stem}.png",
                metric_col=metric_col, null_col=null_col,
                ylabel=ylabel,
                title=f"{ylabel} by reference mode and k (across held-out targets)",
            )
            plot_grouped_bars_by_mode_and_k_per_target(
                df_metrics, summary_dir / f"{fname_stem}_by_target.png",
                metric_col=metric_col, null_col=null_col,
                ylabel=ylabel,
                title=f"{ylabel} by reference mode and k",
            )
        except Exception as e:
            logger.warning("Failed to render %s: %s", fname_stem, e)

    elapsed = time.time() - t_start
    logger.info("Sweep + summary done (%.1fs)  —  %s", elapsed, sweep_root)


if __name__ == "__main__":
    main()
