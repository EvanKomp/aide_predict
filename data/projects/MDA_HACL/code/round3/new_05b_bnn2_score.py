#!/usr/bin/env python
"""
new_05b_bnn2_score.py — Aggregation, Matched-Pair Scoring, and Plotting
=========================================================================

Loads the raw pairwise predictions produced by ``new_05_bnn2_train.py`` and
scores them under four **matched** (BNN reference mode, null model) pairs:

    formaldehyde        ↔ null_formaldehyde
    nearest             ↔ null_nearest
    avg_all             ↔ null_avg_all
    distance_weighted   ↔ null_distance_weighted

For every mode the script:

  - aggregates pairwise predictions back to one prediction per
    (test mutation, target substrate),
  - computes a matching null prediction from the fold's training data,
  - saves `predictions_by_mode/{mode}.csv`,
  - records headline metrics (MAE, Spearman, NLPD, CRPS, calibration) +
    per-substrate / per-position breakdowns for both the BNN and its null.

A single combined plot (`per_substrate_spearman_all_modes.png`) compares
BNN-vs-null across modes per held-out substrate.

Usage:
  python new_05b_bnn2_score.py --run-dir results/new_05_bnn2/substrate/run_XYZ
  python new_05b_bnn2_score.py --run-dir … --modes formaldehyde nearest
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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
from scipy import stats

logger = logging.getLogger(__name__)

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _plot_style import apply_talk_style as _apply_talk_style
    _apply_talk_style()
except Exception as _e:
    logger.debug("apply_talk_style not applied: %s", _e)

# ---------------------------------------------------------------------------
# Paths + common module import
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

_common_spec = spec_from_file_location(
    "bnn2_common", SCRIPT_DIR / "05_bnn2_common.py")
_common = module_from_spec(_common_spec)
_common_spec.loader.exec_module(_common)

setup_logging = _common.setup_logging
load_config = _common.load_config
load_all_embeddings = _common.load_all_embeddings
load_substrate_metadata = _common.load_substrate_metadata
aggregate_pairwise_predictions = _common.aggregate_pairwise_predictions
compute_null_for_mode = _common.compute_null_for_mode
compute_nlpd = _common.compute_nlpd
compute_crps_gaussian = _common.compute_crps_gaussian
compute_calibration = _common.compute_calibration
compute_per_group_metrics = _common.compute_per_group_metrics
plot_parity = _common.plot_parity
plot_residuals = _common.plot_residuals
plot_calibration = _common.plot_calibration
plot_uncertainty_vs_error = _common.plot_uncertainty_vs_error
plot_uncertainty_decomposition = _common.plot_uncertainty_decomposition
plot_training_curves = _common.plot_training_curves
plot_loss_decomposition = _common.plot_loss_decomposition
compute_ndcg = _common.compute_ndcg
_break_ties = _common._break_ties

FORMALDEHYDE_SUBSTRATE = _common.FORMALDEHYDE_SUBSTRATE
ALL_MODES = ("formaldehyde", "nearest", "avg_all", "distance_weighted")

# UCB β sweep. Only RANKING metrics depend on the score ordering, so only these
# are recomputed across β (= y_pred + β·tot_std). Regression/uncertainty metrics
# (mae, rmse, nlpd, crps, sharpness, calibration) are β-meaningless and stay at
# β=0. The null baseline has no uncertainty, so it is β-invariant.
RANKING_METRIC_KEYS = ("spearman_rho", "pearson_r", "ndcg",
                       "top1_recovery", "top3_recovery", "top5_recovery")
DEFAULT_BETA_GRID = (0.0, 0.5, 1.0, 1.5, 2.0)   # mirrors new_opt_05.py --ucb-betas


class _HistoryNS:
    """Lightweight stand-in for TrainingHistory — exposes the attributes the
    existing plot_training_curves / plot_loss_decomposition functions read."""
    def __init__(self, d: dict):
        self.train_loss = d.get("train_loss") or []
        self.val_loss = d.get("val_loss") or []
        self.train_nll = d.get("train_nll") or []
        self.val_nll = d.get("val_nll") or []
        self.train_kl = d.get("train_kl") or []
        self.kl_weight_schedule = d.get("kl_weight_schedule") or []
        self.best_epoch = d.get("best_epoch")


def load_fold_histories(run_dir: Path) -> list:
    path = run_dir / "training_histories.json"
    if not path.exists():
        return []
    with open(path) as f:
        payload = json.load(f)
    return [_HistoryNS(h) for h in payload.get("histories", [])]


# ═══════════════════════════════════════════════════════════════════════════
# Per-mode aggregation + scoring
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_and_null_per_fold(
    mode: str,
    pairwise_df: pd.DataFrame,
    train_lookup_df: pd.DataFrame,
    embeddings: dict,
    substrate_embedding_type: str,
    distance_metric: str,
    distance_weight_temperature: float,
    formaldehyde_substrate: str = FORMALDEHYDE_SUBSTRATE,
    acq_sigma: str = "within_epi_ale",
) -> pd.DataFrame:
    """Run aggregation + null computation for one reference mode, per fold,
    then concatenate the results.

    ``acq_sigma`` selects the σ used by the UCB β-ranking (NOT the reported
    uncertainty). It is built from the within-reference variance components so β
    means the same thing for single- and multi-reference modes (see
    BETA_ACQUISITION_VARIANCE.md):
      - "within_epi" (default):     epi_within                — epistemic only.
      - "within_epi_ale":           sqrt(epi_within² + ale²)  — adds aleatoric.
      - "total":                    tot_std                   — OLD behavior.
    For single-reference modes var_means≡0, so "within_epi_ale" == "total";
    "within_epi" additionally drops aleatoric, so it changes single-ref modes too.

    Returns a DataFrame with columns:
        mutation_string, substrate, position, wt_aa, mut_aa, fold,
        log_fc, fold_change, is_active_substrate, ref_type,
        y_pred, epi_std, ale_std, tot_std, epi_within_std, acq_std, n_refs,
        null_pred, (cls_prob if hurdle)
    """
    # pairwise_df's "y_pred" is absolute log_fc already (fc_ref added in trainer).
    # The aggregator expects columns _y_pred / _epi_std / _ale_std / _tot_std, so
    # we rename transparently before the call.
    agg_chunks = []
    for fold_i, fold_pw in pairwise_df.groupby("fold", sort=False):
        fold_pw = fold_pw.copy()
        rename = {"y_pred": "_y_pred", "epi_std": "_epi_std",
                  "ale_std": "_ale_std", "tot_std": "_tot_std"}
        if "cls_prob" in fold_pw.columns:
            rename["cls_prob"] = "_cls_prob"
        fold_pw_agg_input = fold_pw.rename(columns=rename)

        cls_prob_arr = (fold_pw_agg_input["_cls_prob"].values
                        if "_cls_prob" in fold_pw_agg_input.columns else None)

        # For "formaldehyde" mode we rely on the aggregator's filtering pass
        # (it drops self-reference targets and rows whose ref isn't formaldehyde).
        _, _, _, _, agg_df = aggregate_pairwise_predictions(
            fold_pw_agg_input["_y_pred"].values,
            fold_pw_agg_input["_epi_std"].values,
            fold_pw_agg_input["_ale_std"].values,
            fold_pw_agg_input["_tot_std"].values,
            fold_pw_agg_input,
            cls_prob_expanded=cls_prob_arr,
            aggregation_mode=mode,
            distance_weight_temperature=distance_weight_temperature,
            formaldehyde_substrate=formaldehyde_substrate,
        )

        if len(agg_df) == 0:
            logger.warning("  Fold %s / mode %s: aggregated set is empty "
                           "(all rows filtered out)", fold_i, mode)
            continue

        # Null for this fold, using only this fold's training lookup
        fold_train = train_lookup_df[train_lookup_df["fold"] == fold_i]
        null_pred = compute_null_for_mode(
            mode=mode,
            agg_df=agg_df,
            df_train=fold_train,
            embeddings=embeddings,
            substrate_embedding_type=substrate_embedding_type,
            distance_metric=distance_metric,
            distance_weight_temperature=distance_weight_temperature,
            formaldehyde_substrate=formaldehyde_substrate,
        )
        agg_df = agg_df.copy()
        agg_df["null_pred"] = null_pred
        agg_df["fold"] = fold_i
        agg_chunks.append(agg_df)

    if not agg_chunks:
        return pd.DataFrame()
    out = pd.concat(agg_chunks, ignore_index=True)
    rename_back = {"_y_pred": "y_pred", "_epi_std": "epi_std",
                   "_ale_std": "ale_std", "_tot_std": "tot_std",
                   "_epi_within_std": "epi_within_std",
                   "_cls_prob": "cls_prob"}
    out.rename(columns=rename_back, inplace=True)

    # Build the UCB acquisition σ used for β-ranking (tot_std stays as-is for all
    # uncertainty diagnostics). within_* variants exclude the between-reference
    # disagreement, so β is mode-consistent.
    if acq_sigma == "total":
        out["acq_std"] = out["tot_std"]
    elif acq_sigma == "within_epi":
        out["acq_std"] = out["epi_within_std"]
    elif acq_sigma == "within_epi_ale":
        out["acq_std"] = np.sqrt(out["epi_within_std"] ** 2 + out["ale_std"] ** 2)
    else:
        raise ValueError(f"Unknown acq_sigma: {acq_sigma!r}")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Metrics computation per mode
# ═══════════════════════════════════════════════════════════════════════════

def _safe_spearman(y_true, y_pred):
    if len(y_true) < 2 or np.std(y_true) < 1e-10 or np.std(y_pred) < 1e-10:
        return float("nan")
    mask = ~(np.isnan(y_pred) | np.isnan(y_true))
    if mask.sum() < 2:
        return float("nan")
    rho, _ = stats.spearmanr(y_true[mask], y_pred[mask])
    return float(rho)


def _safe_mae(y_true, y_pred):
    mask = ~(np.isnan(y_pred) | np.isnan(y_true))
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def _safe_pearson(y_true, y_pred):
    mask = ~(np.isnan(y_pred) | np.isnan(y_true))
    if mask.sum() < 2 or np.std(y_true[mask]) < 1e-10 or np.std(y_pred[mask]) < 1e-10:
        return float("nan")
    r, _ = stats.pearsonr(y_true[mask], y_pred[mask])
    return float(r)


def _topk_recovery(y_true: np.ndarray, y_score: np.ndarray, k: int) -> dict:
    """Fraction of the true top-k captured by the top-k predicted scores.

    Ties are broken by ``y_score`` order, so randomly-tied nulls won't game
    the metric when k << n. Returns NaN if fewer than k valid rows.
    """
    mask = ~(np.isnan(y_score) | np.isnan(y_true))
    yt = y_true[mask]
    ys = y_score[mask]
    n = len(yt)
    if n == 0 or k <= 0:
        return {"k": int(k), "n": int(n), "recovered": 0, "recovery": float("nan")}
    k_eff = min(k, n)
    true_top = set(np.argsort(-yt, kind="stable")[:k_eff].tolist())
    pred_top = set(np.argsort(-ys, kind="stable")[:k_eff].tolist())
    recovered = len(true_top & pred_top)
    return {
        "k": int(k_eff),
        "n": int(n),
        "recovered": int(recovered),
        "recovery": float(recovered / k_eff),
    }


def _per_position_aggregate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    positions: np.ndarray,
    fn,
    min_rows: int = 2,
) -> dict:
    """Apply `fn(yt, ys) -> float` per position and return mean + per-position dict.

    Positions with fewer than ``min_rows`` valid rows (post-NaN mask) are skipped.
    ``fn`` must return NaN when it can't score a position (e.g. zero variance);
    NaNs are dropped from the mean. Returns:
      {"n_positions": <int>, "mean": <float>, "per_position": {pos: value}}
    """
    per_pos: dict = {}
    vals = []
    for p in np.unique(positions):
        mask = positions == p
        yt = y_true[mask]; ys = y_score[mask]
        valid = ~(np.isnan(yt) | np.isnan(ys))
        if valid.sum() < min_rows:
            continue
        val = fn(yt[valid], ys[valid])
        per_pos[str(int(p))] = float(val) if (val is not None and not math.isnan(val)) else float("nan")
        if val is not None and not math.isnan(val):
            vals.append(float(val))
    mean_val = float(np.mean(vals)) if vals else float("nan")
    return {
        "n_positions": int(len(vals)),
        "mean": mean_val,
        "per_position": per_pos,
    }


def _per_substrate_aggregate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    substrates: np.ndarray,
    fn,
    min_rows: int = 2,
) -> dict:
    """Apply `fn(yt, ys) -> float` per substrate, mean across substrates.

    Each held-out substrate is weighted equally. Substrates with fewer than
    ``min_rows`` valid rows (post-NaN mask) are skipped. Substrates where
    ``fn`` returns NaN (e.g. zero-variance) are recorded in the per-substrate
    breakdown but dropped from the cross-substrate mean.

    Returns:
      {"n_substrates": int, "mean": float, "per_substrate": {sub: float}}
    """
    per_sub: dict = {}
    vals = []
    for s in np.unique(substrates):
        smask = substrates == s
        yt = y_true[smask]; ys = y_score[smask]
        valid = ~(np.isnan(yt) | np.isnan(ys))
        if valid.sum() < min_rows:
            continue
        val = fn(yt[valid], ys[valid])
        per_sub[str(s)] = float(val) if (val is not None and not math.isnan(val)) else float("nan")
        if val is not None and not math.isnan(val):
            vals.append(float(val))
    mean_val = float(np.mean(vals)) if vals else float("nan")
    return {
        "n_substrates": int(len(vals)),
        "mean": mean_val,
        "per_substrate": per_sub,
    }


def _per_substrate_position_aggregate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    substrates: np.ndarray,
    positions: np.ndarray,
    fn,
    min_rows: int = 2,
) -> dict:
    """Two-step pool: per (substrate, position) `fn`, mean within each
    substrate's positions, then mean across substrates.

    Each held-out substrate is weighted equally, regardless of how many of
    its positions had a valid `fn` value. Substrates whose positions all
    NaN-out (e.g. constant log_fc → zero variance) are dropped from the
    overall mean and surfaced via `n_substrates`.

    Returns:
      {"n_substrates": int,
       "n_position_groups_total": int,
       "mean": float,
       "per_substrate": {sub: {"mean": float, "per_position": {pos: float}}}}
    """
    per_sub: dict = {}
    sub_means = []
    total_groups = 0
    for s in np.unique(substrates):
        smask = substrates == s
        sub_result = _per_position_aggregate(
            y_true[smask], y_score[smask], positions[smask], fn, min_rows=min_rows
        )
        per_sub[str(s)] = {
            "mean": sub_result["mean"],
            "per_position": sub_result["per_position"],
        }
        total_groups += int(sub_result["n_positions"])
        if not math.isnan(sub_result["mean"]):
            sub_means.append(sub_result["mean"])
    overall = float(np.mean(sub_means)) if sub_means else float("nan")
    return {
        "n_substrates": int(len(sub_means)),
        "n_position_groups_total": int(total_groups),
        "mean": overall,
        "per_substrate": per_sub,
    }


def _per_position_spearman_mean(y_true, y_score, substrates, positions) -> dict:
    def _fn(yt, ys):
        if np.std(yt) < 1e-10 or np.std(ys) < 1e-10:
            return float("nan")
        rho, _ = stats.spearmanr(yt, ys)
        return float(rho)
    return _per_substrate_position_aggregate(
        y_true, y_score, substrates, positions, _fn, min_rows=3)


def _per_position_mae_mean(y_true, y_score, substrates, positions) -> dict:
    def _fn(yt, ys):
        return float(np.mean(np.abs(yt - ys)))
    return _per_substrate_position_aggregate(
        y_true, y_score, substrates, positions, _fn, min_rows=1)


def _per_position_ndcg_mean(y_true, y_score, substrates, positions,
                             k: Optional[int] = None) -> dict:
    """Within-(substrate,position) NDCG with continuous relevance, averaged
    over each substrate's positions, then over substrates.

    NDCG semantics:
      - y_score is used ONLY to produce the predicted ordering.
      - y_true is the relevance; we shift per (substrate, position) so
        rel = yt - min(yt) ≥ 0. Constant-log_fc held-out substrates (e.g.
        inactives) yield idcg=0 → NaN at every position → substrate dropped
        from the cross-substrate mean.

    Uses linear gain (not exponential ``2^rel - 1``) to preserve the
    continuous scale of log_fc differences. k=None = full ranking.
    """
    def _fn(yt, ys):
        n = len(yt)
        if n < 2:
            return float("nan")
        k_eff = n if k is None else min(k, n)
        rel = yt - yt.min()
        discounts = 1.0 / np.log2(np.arange(2, k_eff + 2))
        pred_order = np.argsort(-ys, kind="stable")[:k_eff]
        ideal_order = np.argsort(-rel, kind="stable")[:k_eff]
        dcg = float(np.sum(rel[pred_order] * discounts))
        idcg = float(np.sum(rel[ideal_order] * discounts))
        if idcg < 1e-12:
            return float("nan")
        return dcg / idcg
    return _per_substrate_position_aggregate(
        y_true, y_score, substrates, positions, _fn, min_rows=2)


def _per_position_topk_recovery_mean(
    y_true, y_score, substrates, positions, k: int,
) -> dict:
    """Top-k recovery within each (substrate, position), averaged within
    substrate then across substrates."""
    def _fn(yt, ys):
        n = len(yt)
        if n == 0:
            return float("nan")
        k_eff = min(k, n)
        true_top = set(np.argsort(-yt, kind="stable")[:k_eff].tolist())
        pred_top = set(np.argsort(-ys, kind="stable")[:k_eff].tolist())
        return float(len(true_top & pred_top) / k_eff)
    return _per_substrate_position_aggregate(
        y_true, y_score, substrates, positions, _fn, min_rows=1)


def _per_position_top1_recovery(
    y_true: np.ndarray,
    y_score: np.ndarray,
    substrates: np.ndarray,
    positions: np.ndarray,
) -> dict:
    """For each (substrate, position), does predicted top-1 match true top-1?
    Average within substrate (fraction of that substrate's positions where
    the predicted argmax matches the true argmax), then mean across
    substrates. Each substrate weighted equally.
    """
    def _fn(yt, ys):
        if len(yt) == 0:
            return float("nan")
        return float(int(np.argmax(yt)) == int(np.argmax(ys)))
    agg = _per_substrate_position_aggregate(
        y_true, y_score, substrates, positions, _fn, min_rows=1)
    # Same shape as the other wrappers, but the headline key is "recovery"
    # for backwards compatibility with callers that read .recovery.
    return {
        "n_substrates": agg["n_substrates"],
        "n_position_groups_total": agg["n_position_groups_total"],
        "recovery": agg["mean"],
        "per_substrate": agg["per_substrate"],
    }


# ───────────────────────────────────────────────────────────────────────────
# Per-substrate wrappers (the "overall" metrics — compute within each held-out
# substrate, then equal-weight average across substrates). Exposed at module
# level so new_opt_05.py can import them by name.
# ───────────────────────────────────────────────────────────────────────────

def _per_substrate_spearman_mean(y_true, y_score, substrates) -> dict:
    def _fn(yt, ys):
        if np.std(yt) < 1e-10 or np.std(ys) < 1e-10:
            return float("nan")
        rho, _ = stats.spearmanr(yt, ys)
        return float(rho)
    return _per_substrate_aggregate(y_true, y_score, substrates, _fn, min_rows=3)


def _per_substrate_pearson_mean(y_true, y_score, substrates) -> dict:
    def _fn(yt, ys):
        if np.std(yt) < 1e-10 or np.std(ys) < 1e-10:
            return float("nan")
        r, _ = stats.pearsonr(yt, ys)
        return float(r)
    return _per_substrate_aggregate(y_true, y_score, substrates, _fn, min_rows=3)


def _per_substrate_mae_mean(y_true, y_score, substrates) -> dict:
    def _fn(yt, ys):
        return float(np.mean(np.abs(yt - ys)))
    return _per_substrate_aggregate(y_true, y_score, substrates, _fn, min_rows=1)


def _per_substrate_rmse_mean(y_true, y_score, substrates) -> dict:
    def _fn(yt, ys):
        return float(np.sqrt(np.mean((yt - ys) ** 2)))
    return _per_substrate_aggregate(y_true, y_score, substrates, _fn, min_rows=1)


def _per_substrate_ndcg_mean(y_true, y_score, substrates,
                             k: Optional[int] = None) -> dict:
    def _fn(yt, ys):
        if len(yt) < 2:
            return float("nan")
        return float(compute_ndcg(yt, ys, k=k))
    return _per_substrate_aggregate(y_true, y_score, substrates, _fn, min_rows=2)


def _per_substrate_topk_recovery_mean(y_true, y_score, substrates, k: int) -> dict:
    """Per-substrate top-k recovery, equal-weight averaged.

    Returns the same dotted-path-friendly shape as the old global
    ``_topk_recovery``: a top-level ``recovery`` scalar (the cross-substrate
    mean), with a ``per_substrate`` breakdown carrying full {k, n, recovered,
    recovery} dicts per substrate so the bar plots and JSON exports keep
    working unchanged.
    """
    per_sub: dict = {}
    recoveries = []
    for s in np.unique(substrates):
        smask = substrates == s
        result = _topk_recovery(y_true[smask], y_score[smask], k)
        per_sub[str(s)] = result
        if not math.isnan(result["recovery"]):
            recoveries.append(result["recovery"])
    mean_rec = float(np.mean(recoveries)) if recoveries else float("nan")
    return {
        "k": int(k),
        "recovery": mean_rec,
        "n_substrates": int(len(recoveries)),
        "per_substrate": per_sub,
    }


def _ranking_metrics_at_score(y_true, y_score, substrates) -> dict:
    """The six ranking metrics for one score vector, per-substrate-averaged.

    Reuses the existing ``_per_substrate_*`` wrappers verbatim (no metric math
    reimplemented). β is applied by the caller (``y_score = y_pred + β·tot_std``);
    this function is β-agnostic. Returns both the cross-substrate ``mean`` scalars
    and the ``per_substrate`` breakdowns so plots can draw faint per-substrate
    lines plus a bold mean line.
    """
    spear = _per_substrate_spearman_mean(y_true, y_score, substrates)
    pear = _per_substrate_pearson_mean(y_true, y_score, substrates)
    ndcg = _per_substrate_ndcg_mean(y_true, y_score, substrates)
    top1 = _per_substrate_topk_recovery_mean(y_true, y_score, substrates, k=1)
    top3 = _per_substrate_topk_recovery_mean(y_true, y_score, substrates, k=3)
    top5 = _per_substrate_topk_recovery_mean(y_true, y_score, substrates, k=5)

    def _m(d):
        v = d.get("mean", float("nan"))
        return float(v) if (v is not None and not math.isnan(v)) else float("nan")

    def _topk_per_sub(topk):
        # per_substrate values are full {k,n,recovered,recovery} dicts → keep the
        # scalar recovery for plotting.
        return {str(s): (d["recovery"] if isinstance(d, dict) else float(d))
                for s, d in topk["per_substrate"].items()}

    return {
        "mean": {
            "spearman_rho": _m(spear),
            "pearson_r": _m(pear),
            "ndcg": _m(ndcg),
            "top1_recovery": float(top1["recovery"]),
            "top3_recovery": float(top3["recovery"]),
            "top5_recovery": float(top5["recovery"]),
        },
        "per_substrate": {
            "spearman_rho": {str(k): v for k, v in spear["per_substrate"].items()},
            "pearson_r": {str(k): v for k, v in pear["per_substrate"].items()},
            "ndcg": {str(k): v for k, v in ndcg["per_substrate"].items()},
            "top1_recovery": _topk_per_sub(top1),
            "top3_recovery": _topk_per_sub(top3),
            "top5_recovery": _topk_per_sub(top5),
        },
    }


def compute_beta_sweep_for_mode(mode_df: pd.DataFrame, beta_grid: List[float]) -> dict:
    """Ranking metrics vs UCB β for one mode.

    For each β in ``beta_grid`` the rows are scored ``y_pred + β·acq_std`` (the
    mode-consistent acquisition σ) and the six per-substrate-averaged ranking
    metrics are computed (with per-substrate breakdowns). The null prediction is
    uncertainty-free and therefore β-invariant: computed exactly once.

    Returns::

        {"betas": [...],
         "bnn":  {metric: {"mean": [per-β], "per_substrate": {sub: [per-β]}}},
         "null": {metric: {"mean": float,   "per_substrate": {sub: float}}}}
    """
    y_true = mode_df["log_fc"].values.astype(np.float64)
    y_pred = mode_df["y_pred"].values.astype(np.float64)
    acq_std = mode_df["acq_std"].values.astype(np.float64)
    null_pred = mode_df["null_pred"].values.astype(np.float64)
    subs = mode_df["substrate"].values

    betas = sorted({float(b) for b in beta_grid})
    all_subs = sorted({str(s) for s in np.unique(subs)})

    # Null once (β-invariant).
    null_rm = _ranking_metrics_at_score(y_true, null_pred, subs)

    bnn_means = {mk: [] for mk in RANKING_METRIC_KEYS}
    bnn_persub = {mk: {s: [] for s in all_subs} for mk in RANKING_METRIC_KEYS}
    for beta in betas:
        rm = _ranking_metrics_at_score(y_true, y_pred + beta * acq_std, subs)
        for mk in RANKING_METRIC_KEYS:
            bnn_means[mk].append(rm["mean"][mk])
            for s in all_subs:
                bnn_persub[mk][s].append(rm["per_substrate"][mk].get(s, float("nan")))

    return {
        "betas": betas,
        "bnn": {mk: {"mean": bnn_means[mk], "per_substrate": bnn_persub[mk]}
                for mk in RANKING_METRIC_KEYS},
        "null": {mk: {"mean": null_rm["mean"][mk],
                      "per_substrate": null_rm["per_substrate"][mk]}
                 for mk in RANKING_METRIC_KEYS},
    }


def _per_substrate_nlpd_mean(y_true, y_score, tot_std, substrates) -> dict:
    """Per-substrate row-mean NLPD, equal-weight averaged across substrates."""
    per_sub: dict = {}
    vals = []
    for s in np.unique(substrates):
        smask = substrates == s
        yt = y_true[smask]; yp = y_score[smask]; ts = tot_std[smask]
        if len(yt) < 1:
            continue
        v = float(compute_nlpd(yt, yp, np.clip(ts, 1e-6, None)))
        per_sub[str(s)] = v
        if not math.isnan(v):
            vals.append(v)
    return {
        "n_substrates": int(len(vals)),
        "mean": float(np.mean(vals)) if vals else float("nan"),
        "per_substrate": per_sub,
    }


def _per_substrate_crps_mean(y_true, y_score, tot_std, substrates) -> dict:
    """Per-substrate row-mean CRPS, equal-weight averaged across substrates."""
    per_sub: dict = {}
    vals = []
    for s in np.unique(substrates):
        smask = substrates == s
        yt = y_true[smask]; yp = y_score[smask]; ts = tot_std[smask]
        if len(yt) < 1:
            continue
        v = float(compute_crps_gaussian(yt, yp, np.clip(ts, 1e-6, None)))
        per_sub[str(s)] = v
        if not math.isnan(v):
            vals.append(v)
    return {
        "n_substrates": int(len(vals)),
        "mean": float(np.mean(vals)) if vals else float("nan"),
        "per_substrate": per_sub,
    }


def _per_substrate_sharpness_mean(tot_std, substrates) -> dict:
    """Per-substrate mean of total std, equal-weight averaged across substrates."""
    per_sub: dict = {}
    vals = []
    for s in np.unique(substrates):
        smask = substrates == s
        ts = tot_std[smask]
        if len(ts) < 1:
            continue
        v = float(np.mean(ts))
        per_sub[str(s)] = v
        vals.append(v)
    return {
        "n_substrates": int(len(vals)),
        "mean": float(np.mean(vals)) if vals else float("nan"),
        "per_substrate": per_sub,
    }


def _per_substrate_calibration(y_true, y_score, tot_std, substrates) -> dict:
    """Per-substrate calibration curves, element-wise averaged.

    Returns the same {levels, expected_coverage, observed_coverage} shape as
    ``compute_calibration``, with observed_coverage being the equal-weight
    mean of per-substrate observed_coverage arrays at the shared CI levels.
    """
    per_sub: dict = {}
    covs = []
    levels = None
    for s in np.unique(substrates):
        smask = substrates == s
        yt = y_true[smask]; yp = y_score[smask]; ts = tot_std[smask]
        if len(yt) < 1:
            continue
        cal = compute_calibration(yt, yp, np.clip(ts, 1e-6, None))
        per_sub[str(s)] = cal
        if levels is None:
            levels = cal["levels"]
        covs.append(np.array(cal["observed_coverage"], dtype=np.float64))
    if not covs:
        return {"levels": [], "expected_coverage": [],
                "observed_coverage": [], "per_substrate": {}}
    mean_cov = np.mean(np.stack(covs, axis=0), axis=0)
    return {
        "levels": list(levels),
        "expected_coverage": list(levels),
        "observed_coverage": [float(x) for x in mean_cov],
        "per_substrate": per_sub,
    }


def compute_metrics_for_mode(
    mode: str,
    mode_df: pd.DataFrame,
    best_beta: float = 0.0,
    beta_grid: Optional[List[float]] = None,
) -> dict:
    """Compute BNN + null headline and per-group metrics for one mode.

    All "overall" / top-level scalar metrics (spearman_rho, mae, rmse,
    pearson_r, ndcg, top{1,3,5}_recovery, nlpd, crps, sharpness, calibration)
    use per-substrate-averaged semantics: compute within each held-out
    substrate, equal-weight mean across substrates. Per-position metrics
    (per_position_*_mean) use the substrate-first two-step pool (within
    (substrate, position) → mean over substrate's positions → mean over
    substrates) — unchanged.

    The existing ``bnn``/``null``/``delta_*`` keys are all computed at β=0 (the
    bare posterior mean) and are left exactly as before. On top of that, this
    function adds a UCB β layer (``best_beta`` defaults to 0.0 → no-op):

      - ``best_beta``           : the β used for the headline below.
      - ``bnn_at_best_beta``    : RANKING metrics (overall + per-position) for the
                                  BNN scored at ``y_pred + best_beta·tot_std``.
      - ``delta_at_best_beta``  : ``bnn_at_best_beta`` minus the (β-invariant) null,
                                  positive = BNN wins.
      - ``beta_sweep``          : full per-β arrays for the 6 overall ranking
                                  metrics (BNN curve + flat null), for plotting.

    Regression/uncertainty metrics are β-meaningless and are NOT recomputed.
    """
    y_true = mode_df["log_fc"].values.astype(np.float64)
    y_pred = mode_df["y_pred"].values.astype(np.float64)
    null_pred = mode_df["null_pred"].values.astype(np.float64)
    tot_std = mode_df["tot_std"].values.astype(np.float64)   # diagnostics only
    acq_std = mode_df["acq_std"].values.astype(np.float64)   # β-ranking only
    subs = mode_df["substrate"].values
    pos = mode_df["position"].values

    # Overall (per-substrate-averaged) — ranking & regression metrics
    bnn_spearman = _per_substrate_spearman_mean(y_true, y_pred, subs)
    null_spearman = _per_substrate_spearman_mean(y_true, null_pred, subs)
    bnn_pearson = _per_substrate_pearson_mean(y_true, y_pred, subs)
    null_pearson = _per_substrate_pearson_mean(y_true, null_pred, subs)
    bnn_mae = _per_substrate_mae_mean(y_true, y_pred, subs)
    null_mae = _per_substrate_mae_mean(y_true, null_pred, subs)
    bnn_rmse = _per_substrate_rmse_mean(y_true, y_pred, subs)
    null_rmse = _per_substrate_rmse_mean(y_true, null_pred, subs)
    bnn_ndcg = _per_substrate_ndcg_mean(y_true, y_pred, subs)
    null_ndcg = _per_substrate_ndcg_mean(y_true, null_pred, subs)

    # Overall (per-substrate-averaged) — uncertainty diagnostics (BNN only)
    nlpd = _per_substrate_nlpd_mean(y_true, y_pred, tot_std, subs)
    crps = _per_substrate_crps_mean(y_true, y_pred, tot_std, subs)
    sharpness = _per_substrate_sharpness_mean(tot_std, subs)
    calibration = _per_substrate_calibration(y_true, y_pred, tot_std, subs)

    # Overall (per-substrate-averaged) — top-k recovery
    bnn_top1 = _per_substrate_topk_recovery_mean(y_true, y_pred, subs, k=1)
    null_top1 = _per_substrate_topk_recovery_mean(y_true, null_pred, subs, k=1)
    bnn_top3 = _per_substrate_topk_recovery_mean(y_true, y_pred, subs, k=3)
    null_top3 = _per_substrate_topk_recovery_mean(y_true, null_pred, subs, k=3)
    bnn_top5 = _per_substrate_topk_recovery_mean(y_true, y_pred, subs, k=5)
    null_top5 = _per_substrate_topk_recovery_mean(y_true, null_pred, subs, k=5)

    # Per-position (unchanged): per-(substrate, position) two-step pool
    bnn_per_pos_top1 = _per_position_top1_recovery(y_true, y_pred, subs, pos)
    null_per_pos_top1 = _per_position_top1_recovery(y_true, null_pred, subs, pos)
    bnn_per_pos_spearman_mean = _per_position_spearman_mean(y_true, y_pred, subs, pos)
    null_per_pos_spearman_mean = _per_position_spearman_mean(y_true, null_pred, subs, pos)
    bnn_per_pos_mae_mean = _per_position_mae_mean(y_true, y_pred, subs, pos)
    null_per_pos_mae_mean = _per_position_mae_mean(y_true, null_pred, subs, pos)
    bnn_per_pos_ndcg_mean = _per_position_ndcg_mean(y_true, y_pred, subs, pos, k=None)
    null_per_pos_ndcg_mean = _per_position_ndcg_mean(y_true, null_pred, subs, pos, k=None)
    bnn_per_pos_top3_rec = _per_position_topk_recovery_mean(y_true, y_pred, subs, pos, k=3)
    null_per_pos_top3_rec = _per_position_topk_recovery_mean(y_true, null_pred, subs, pos, k=3)
    bnn_per_pos_top5_rec = _per_position_topk_recovery_mean(y_true, y_pred, subs, pos, k=5)
    null_per_pos_top5_rec = _per_position_topk_recovery_mean(y_true, null_pred, subs, pos, k=5)

    # Per-substrate dict breakdowns (kept for JSON consumers + plots).
    # These are sources the new scalar metrics average over; we expose the
    # full breakdowns so consumers can drill in.
    bnn_per_sub_spearman = bnn_spearman["per_substrate"]
    null_per_sub_spearman = null_spearman["per_substrate"]
    bnn_per_sub_mae = bnn_mae["per_substrate"]
    null_per_sub_mae = null_mae["per_substrate"]
    # Per-substrate activity range (used by some downstream tooling)
    _, _, per_sub_range = compute_per_group_metrics(y_true, y_pred, subs)
    # Per-position (within-position pooled across substrates) — kept for
    # the existing bar plots that read these dicts.
    per_pos_bnn_rho, per_pos_bnn_mae, _ = compute_per_group_metrics(y_true, y_pred, pos)
    per_pos_null_rho, per_pos_null_mae, _ = compute_per_group_metrics(y_true, null_pred, pos)

    def _scalar(d):
        """Extract the mean scalar from a per-substrate aggregate dict."""
        return float(d["mean"]) if not math.isnan(d["mean"]) else float("nan")

    def _key_scalar(d, key):
        """Extract `key` from an aggregate dict, NaN-safe."""
        v = d.get(key, float("nan"))
        return float(v) if (v is not None and not math.isnan(v)) else float("nan")

    # ── UCB β layer: headline @ best β (ranking only) + full β sweep ──────────
    grid = list(beta_grid) if beta_grid is not None else list(DEFAULT_BETA_GRID)
    if float(best_beta) not in grid:
        grid = sorted(set(grid) | {float(best_beta)})
    beta_sweep = compute_beta_sweep_for_mode(mode_df, grid)

    bb = float(best_beta)
    score_bb = y_pred + bb * acq_std
    overall_bb = _ranking_metrics_at_score(y_true, score_bb, subs)["mean"]
    # per-position ranking metrics at best β (same wrappers used at β=0)
    bnn_at_best_beta = dict(overall_bb)
    bnn_at_best_beta.update({
        "per_position_spearman_mean":
            _key_scalar(_per_position_spearman_mean(y_true, score_bb, subs, pos), "mean"),
        "per_position_ndcg_mean":
            _key_scalar(_per_position_ndcg_mean(y_true, score_bb, subs, pos, k=None), "mean"),
        "per_position_top1_recovery":
            _key_scalar(_per_position_top1_recovery(y_true, score_bb, subs, pos), "recovery"),
        "per_position_top3_recovery_mean":
            _key_scalar(_per_position_topk_recovery_mean(y_true, score_bb, subs, pos, k=3), "mean"),
        "per_position_top5_recovery_mean":
            _key_scalar(_per_position_topk_recovery_mean(y_true, score_bb, subs, pos, k=5), "mean"),
    })
    # Matched null values (β-invariant — reuse the already-computed β=0 results).
    null_ranking = {
        "spearman_rho": _scalar(null_spearman),
        "pearson_r": _scalar(null_pearson),
        "ndcg": _scalar(null_ndcg),
        "top1_recovery": null_top1["recovery"],
        "top3_recovery": null_top3["recovery"],
        "top5_recovery": null_top5["recovery"],
        "per_position_spearman_mean": _key_scalar(null_per_pos_spearman_mean, "mean"),
        "per_position_ndcg_mean": _key_scalar(null_per_pos_ndcg_mean, "mean"),
        "per_position_top1_recovery": _key_scalar(null_per_pos_top1, "recovery"),
        "per_position_top3_recovery_mean": _key_scalar(null_per_pos_top3_rec, "mean"),
        "per_position_top5_recovery_mean": _key_scalar(null_per_pos_top5_rec, "mean"),
    }

    def _delta(b, n):
        return (b - n) if not (math.isnan(b) or math.isnan(n)) else float("nan")
    delta_at_best_beta = {
        k: _delta(bnn_at_best_beta[k], null_ranking[k]) for k in bnn_at_best_beta
    }

    return {
        "n_rows": int(len(mode_df)),
        "best_beta": bb,
        "bnn_at_best_beta": bnn_at_best_beta,
        "delta_at_best_beta": delta_at_best_beta,
        "beta_sweep": beta_sweep,
        "bnn": {
            "mae": _scalar(bnn_mae),
            "rmse": _scalar(bnn_rmse),
            "spearman_rho": _scalar(bnn_spearman),
            "pearson_r": _scalar(bnn_pearson),
            "nlpd": _scalar(nlpd),
            "crps": _scalar(crps),
            "sharpness": _scalar(sharpness),
            "calibration": calibration,
            "per_substrate_spearman": {str(k): v for k, v in bnn_per_sub_spearman.items()},
            "per_substrate_mae": {str(k): v for k, v in bnn_per_sub_mae.items()},
            "per_substrate_ndcg": bnn_ndcg["per_substrate"],
            "per_position_spearman": {str(k): v for k, v in per_pos_bnn_rho.items()},
            "per_position_mae": {str(k): v for k, v in per_pos_bnn_mae.items()},
            "top1_recovery": bnn_top1,
            "top3_recovery": bnn_top3,
            "top5_recovery": bnn_top5,
            "ndcg": _scalar(bnn_ndcg),
            "per_position_top1_recovery": bnn_per_pos_top1,
            "per_position_top3_recovery_mean": bnn_per_pos_top3_rec,
            "per_position_top5_recovery_mean": bnn_per_pos_top5_rec,
            "per_position_spearman_mean": bnn_per_pos_spearman_mean,
            "per_position_mae_mean": bnn_per_pos_mae_mean,
            "per_position_ndcg_mean": bnn_per_pos_ndcg_mean,
        },
        "null": {
            "mae": _scalar(null_mae),
            "rmse": _scalar(null_rmse),
            "spearman_rho": _scalar(null_spearman),
            "pearson_r": _scalar(null_pearson),
            "per_substrate_spearman": {str(k): v for k, v in null_per_sub_spearman.items()},
            "per_substrate_mae": {str(k): v for k, v in null_per_sub_mae.items()},
            "per_substrate_ndcg": null_ndcg["per_substrate"],
            "per_position_spearman": {str(k): v for k, v in per_pos_null_rho.items()},
            "per_position_mae": {str(k): v for k, v in per_pos_null_mae.items()},
            "top1_recovery": null_top1,
            "top3_recovery": null_top3,
            "top5_recovery": null_top5,
            "ndcg": _scalar(null_ndcg),
            "per_position_top1_recovery": null_per_pos_top1,
            "per_position_top3_recovery_mean": null_per_pos_top3_rec,
            "per_position_top5_recovery_mean": null_per_pos_top5_rec,
            "per_position_spearman_mean": null_per_pos_spearman_mean,
            "per_position_mae_mean": null_per_pos_mae_mean,
            "per_position_ndcg_mean": null_per_pos_ndcg_mean,
        },
        "delta_spearman": _scalar(bnn_spearman) - _scalar(null_spearman)
            if not (math.isnan(_scalar(bnn_spearman)) or math.isnan(_scalar(null_spearman))) else float("nan"),
        "delta_mae": _scalar(null_mae) - _scalar(bnn_mae)
            if not (math.isnan(_scalar(bnn_mae)) or math.isnan(_scalar(null_mae))) else float("nan"),
        "delta_ndcg": _scalar(bnn_ndcg) - _scalar(null_ndcg)
            if not (math.isnan(_scalar(bnn_ndcg)) or math.isnan(_scalar(null_ndcg))) else float("nan"),
        "delta_top1_recovery": (bnn_top1["recovery"] - null_top1["recovery"])
            if not (math.isnan(bnn_top1["recovery"]) or math.isnan(null_top1["recovery"])) else float("nan"),
        "delta_top3_recovery": (bnn_top3["recovery"] - null_top3["recovery"])
            if not (math.isnan(bnn_top3["recovery"]) or math.isnan(null_top3["recovery"])) else float("nan"),
        "delta_top5_recovery": (bnn_top5["recovery"] - null_top5["recovery"])
            if not (math.isnan(bnn_top5["recovery"]) or math.isnan(null_top5["recovery"])) else float("nan"),
        "delta_per_position_top1_recovery": (bnn_per_pos_top1["recovery"] - null_per_pos_top1["recovery"])
            if not (math.isnan(bnn_per_pos_top1["recovery"]) or math.isnan(null_per_pos_top1["recovery"])) else float("nan"),
        "per_substrate_range": {str(k): v for k, v in per_sub_range.items()},
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_per_mode_diagnostics(
    mode: str,
    mode_df: pd.DataFrame,
    mode_metrics: dict,
    out_dir: Path,
):
    """Parity, residuals, calibration, uncertainty-vs-error per mode."""
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = mode_df["log_fc"].values.astype(np.float64)
    y_pred = mode_df["y_pred"].values.astype(np.float64)
    tot_std = np.clip(mode_df["tot_std"].values.astype(np.float64), 1e-6, None)
    epi_std = mode_df["epi_std"].values.astype(np.float64)
    ale_std = mode_df["ale_std"].values.astype(np.float64)
    subs = mode_df["substrate"].values

    plot_parity(y_true, y_pred, tot_std, subs, out_dir / f"parity_{mode}.png")
    plot_residuals(y_true, y_pred, subs, out_dir / f"residuals_{mode}.png")
    plot_calibration(mode_metrics["bnn"]["calibration"],
                     out_dir / f"calibration_{mode}.png")
    plot_uncertainty_vs_error(y_true, y_pred, tot_std, subs,
                              out_dir / f"uncertainty_vs_error_{mode}.png")
    plot_uncertainty_decomposition(epi_std, ale_std, subs,
                                   out_dir / f"uncertainty_decomposition_{mode}.png")


def plot_per_substrate_spearman_all_modes(
    mode_results: Dict[str, dict],
    out_path: Path,
):
    """Grouped-bar plot: per-substrate Spearman for BNN (@ best β) vs null across modes."""
    def _bnn_persub_best_beta(metrics: dict) -> dict:
        """Per-substrate BNN spearman evaluated at this mode's best β (falls back
        to the β=0 dict if no sweep is present)."""
        bs = metrics.get("beta_sweep")
        if not bs:
            return metrics["bnn"]["per_substrate_spearman"]
        betas = bs["betas"]
        bb = float(metrics.get("best_beta", 0.0))
        idx = betas.index(bb) if bb in betas else 0
        return {s: arr[idx] for s, arr in bs["bnn"]["spearman_rho"]["per_substrate"].items()}

    substrates = sorted({
        s for m in mode_results.values()
        for s in m["metrics"]["bnn"]["per_substrate_spearman"].keys()
    })
    modes = [m for m in ALL_MODES if m in mode_results]
    if not substrates or not modes:
        logger.warning("Not enough data for per-substrate comparison plot — skipping")
        return

    n_modes = len(modes)
    n_subs = len(substrates)
    fig, ax = plt.subplots(figsize=(max(10, 1.2 * n_subs * n_modes), 5.5))

    bar_width = 0.8 / (2 * n_modes)
    x = np.arange(n_subs)

    colors_bnn = plt.cm.tab10(np.linspace(0, 1, n_modes))
    for i, mode in enumerate(modes):
        metrics = mode_results[mode]["metrics"]
        bnn_persub = _bnn_persub_best_beta(metrics)
        bb = float(metrics.get("best_beta", 0.0))
        bnn_rhos = [bnn_persub.get(s, np.nan) for s in substrates]
        null_rhos = [metrics["null"]["per_substrate_spearman"].get(s, np.nan)
                     for s in substrates]
        offset = (i - n_modes / 2 + 0.5) * (2 * bar_width)
        ax.bar(x + offset - bar_width / 2, bnn_rhos, bar_width,
                label=f"BNN ({mode}, β={bb:g})", color=colors_bnn[i],
                edgecolor="black", linewidth=1.5)
        ax.bar(x + offset + bar_width / 2, null_rhos, bar_width,
                label=f"Null ({mode})", color=colors_bnn[i], alpha=0.4,
                edgecolor="black", linewidth=1.5, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(substrates, rotation=30, ha="right")
    ax.axhline(0, color="#444", lw=1.5)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Per-substrate Spearman: BNN (@ best β) vs matched null, all reference modes")
    ax.legend(ncol=2, loc="lower right")
    ax.grid(axis="y", ls="--", lw=0.8, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _extract(mode_dict: dict, key: str):
    """Fetch a possibly-nested metric from a mode_dict (supports ``top3_recovery.recovery``)."""
    cur = mode_dict
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return float("nan")
    return cur


def plot_mode_summary_bars(
    mode_results: Dict[str, dict],
    out_path: Path,
    metric: str = "spearman_rho",
    title: Optional[str] = None,
    bnn_source: str = "bnn",
    bnn_metric: Optional[str] = None,
):
    """Overall BNN vs null across modes, one pair of bars per mode.

    `metric` (the null-side key) may be a top-level key (e.g. "spearman_rho")
    or a dotted path into nested dicts (e.g. "top3_recovery.recovery").

    For ranking metrics the BNN bar should reflect the per-mode best β: pass
    ``bnn_source="bnn_at_best_beta"`` and ``bnn_metric=<flat key>`` (the best-β
    block stores flat scalars, e.g. "top3_recovery"). Each mode's β is then
    printed under its x-tick. Regression bars keep the default β=0 source.
    """
    modes = [m for m in ALL_MODES if m in mode_results]
    if not modes:
        return
    at_best_beta = (bnn_source != "bnn")
    bnn_key = bnn_metric or metric
    bnn_vals = [_extract(mode_results[m]["metrics"].get(bnn_source, {}), bnn_key)
                for m in modes]
    null_vals = [_extract(mode_results[m]["metrics"]["null"], metric) for m in modes]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(modes))
    w = 0.4
    bnn_label = "BNN @ best β" if at_best_beta else "BNN"
    ax.bar(x - w / 2, bnn_vals, w, label=bnn_label, color="#4477aa",
           edgecolor="black", linewidth=1.5)
    ax.bar(x + w / 2, null_vals, w, label="Null", color="#ee6677",
            edgecolor="black", linewidth=1.5, hatch="//")
    ax.set_xticks(x)
    if at_best_beta:
        xlabels = [f"{m}\nβ={float(mode_results[m]['metrics'].get('best_beta', 0.0)):g}"
                   for m in modes]
    else:
        xlabels = list(modes)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel(metric)
    _suffix = "  (BNN @ best β, null β-invariant)" if at_best_beta else ""
    ax.set_title(title or
                 f"Overall {metric}: BNN vs matched null, per reference mode{_suffix}")
    for i, (b, n) in enumerate(zip(bnn_vals, null_vals)):
        if not np.isnan(b):
            ax.text(i - w / 2, b, f"{b:.3f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")
        if not np.isnan(n):
            ax.text(i + w / 2, n, f"{n:.3f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")
    ax.legend()
    ax.axhline(0, color="#444", lw=1.5)
    ax.grid(axis="y", ls="--", lw=0.8, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# UCB β sweep (BNN-vs-null ranking metrics as a function of β)
# ═══════════════════════════════════════════════════════════════════════════

def plot_beta_sweep_mode(
    mode: str,
    beta_sweep: dict,
    best_beta: float,
    out_path: Path,
):
    """One subplot per ranking metric: BNN metric vs UCB β, against the null.

    Per subplot: faint per-substrate BNN lines, a bold BNN mean curve, a flat
    red-dashed null mean line (β-invariant) with a shaded per-substrate null
    band, and a green dotted vertical at the hyperopt ``best_beta``. The null
    has no uncertainty, so it cannot move with β — any place the BNN curve rises
    above the null line is where the BNN's uncertainty buys real ranking signal.
    """
    betas = np.asarray(beta_sweep["betas"], dtype=float)
    metrics = list(RANKING_METRIC_KEYS)
    ncols = 3
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    for idx, mk in enumerate(metrics):
        ax = axes[idx // ncols][idx % ncols]
        bnn = beta_sweep["bnn"][mk]
        nul = beta_sweep["null"][mk]

        # faint per-substrate BNN lines
        for arr in bnn["per_substrate"].values():
            ax.plot(betas, arr, "-", alpha=0.3, lw=1)
        # bold BNN mean
        ax.plot(betas, bnn["mean"], "s-", color="black", lw=2.5, ms=7,
                label="BNN mean", zorder=10)
        # null mean (flat) + per-substrate spread band
        null_mean = nul["mean"]
        if null_mean is not None and not math.isnan(null_mean):
            ax.axhline(null_mean, color="#ee6677", ls="--", lw=2, label="Null mean")
        sub_vals = [v for v in nul["per_substrate"].values()
                    if v is not None and not math.isnan(v)]
        if sub_vals:
            ax.fill_between(betas, min(sub_vals), max(sub_vals),
                            color="#ee6677", alpha=0.12)
        # best β marker
        ax.axvline(float(best_beta), color="#228833", ls=":", lw=1.8,
                   label=f"best β={best_beta:g}")

        ax.set_xlabel("UCB β")
        ax.set_ylabel(mk)
        ax.set_title(mk)
        ax.grid(ls="--", lw=0.6, alpha=0.4)
        if idx == 0:
            ax.legend(fontsize=8, loc="best")

    # blank any unused axes
    for j in range(len(metrics), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"UCB β sweep — mode: {mode}  (null is β-invariant)", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_beta_sweep_all_modes(
    mode_results: Dict[str, dict],
    metric: str,
    out_path: Path,
):
    """For one ranking metric: BNN mean curve per mode (vs β) + matched nulls."""
    modes = [m for m in ALL_MODES if m in mode_results
             and "beta_sweep" in mode_results[m]["metrics"]]
    if not modes:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(modes)))
    for c, m in zip(colors, modes):
        bs = mode_results[m]["metrics"]["beta_sweep"]
        betas = bs["betas"]
        ax.plot(betas, bs["bnn"][metric]["mean"], "o-", color=c, lw=2,
                label=f"BNN {m}")
        null_mean = bs["null"][metric]["mean"]
        if null_mean is not None and not math.isnan(null_mean):
            ax.axhline(null_mean, color=c, ls="--", lw=1.2, alpha=0.7)
    ax.set_xlabel("UCB β")
    ax.set_ylabel(metric)
    ax.set_title(f"UCB β sweep across modes: {metric}\n(dashed = matched null, β-invariant)")
    ax.legend(fontsize=8)
    ax.grid(ls="--", lw=0.6, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Recovery curves (global, per-mode, per-position)
# ═══════════════════════════════════════════════════════════════════════════

# The recovery plots rank by a SINGLE BNN acquisition series at the per-mode
# best β (UCB = y_pred + β·tot_std). β=0 collapses to the bare posterior mean.
# Fixed key "BNN" so cross-mode indexing is stable even when β differs by mode;
# the β value is surfaced in the plot labels, not the key.
def _acq_scores(y_pred: np.ndarray, tot_std: np.ndarray,
                best_beta: float) -> Dict[str, np.ndarray]:
    return {"BNN": y_pred + float(best_beta) * tot_std}


def _recovery_curve(
    scores: np.ndarray,
    true_top_idx: set,
    k: int,
    checkpoints: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Recall of true top-k for each checkpoint cutoff m.

    Returns (m_values, recall_values).
    """
    order = np.argsort(-scores, kind="stable")
    recalls = np.empty(len(checkpoints), dtype=np.float64)
    for i, m in enumerate(checkpoints):
        m = int(m)
        if m <= 0:
            recalls[i] = 0.0
        else:
            recalls[i] = len(set(order[:m].tolist()) & true_top_idx) / k
    return checkpoints.astype(np.float64), recalls


def _delta_auc(fracs: np.ndarray, recalls: np.ndarray) -> float:
    """AUC of recall(frac) minus the random-diagonal AUC of 0.5."""
    return float(np.trapz(recalls, fracs)) - 0.5


def _hit_rate(scores: np.ndarray, y_true: np.ndarray, wt_activity: float,
              budget_frac: float) -> float:
    n = len(y_true)
    m = max(1, int(round(budget_frac * n)))
    top_idx = np.argsort(-scores, kind="stable")[:m]
    return float(np.mean(y_true[top_idx] > wt_activity))


def _per_substrate_recall_runs(
    mode_df: pd.DataFrame,
    k: int,
    best_beta: float,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], List[str], np.ndarray]:
    """For each (substrate, acquisition score) pair, compute recall@m for
    every cutoff m = 1..n_s, where the "true top-k" is defined within that
    substrate's mutations. The BNN is ranked by its best-β UCB score.

    Returns:
      recalls: {series_name -> {substrate -> recall_array of len n_s}}
      substrate_names: list of substrate names (in iteration order)
      substrate_sizes: ndarray of n_s per substrate (same order)
    """
    y_true = mode_df["log_fc"].values.astype(np.float64)
    y_pred = mode_df["y_pred"].values.astype(np.float64)
    acq_std = np.clip(mode_df["acq_std"].values.astype(np.float64), 1e-6, None)
    null_pred = mode_df["null_pred"].values.astype(np.float64)
    subs = mode_df["substrate"].values

    acq = _acq_scores(y_pred, acq_std, best_beta)
    null_tb = _break_ties(null_pred, seed=len(null_pred))
    series_names = list(acq.keys()) + ["Null"]
    series_arrays = {**acq, "Null": null_tb}

    recalls: Dict[str, Dict[str, np.ndarray]] = {n: {} for n in series_names}
    substrate_names: List[str] = []
    substrate_sizes: List[int] = []

    for s in np.unique(subs):
        smask = subs == s
        yt = y_true[smask]
        valid = ~np.isnan(yt)
        n_s = int(valid.sum())
        if n_s < max(2, k):
            continue
        yt_v = yt[valid]
        k_eff = min(k, n_s)
        true_top = set(np.argsort(-yt_v, kind="stable")[:k_eff].tolist())
        substrate_names.append(str(s))
        substrate_sizes.append(n_s)
        for name in series_names:
            scores = series_arrays[name][smask][valid]
            order = np.argsort(-scores, kind="stable")
            running = np.empty(n_s, dtype=np.float64)
            count = 0
            for i in range(n_s):
                if int(order[i]) in true_top:
                    count += 1
                running[i] = count / k_eff
            recalls[name][str(s)] = running
    return recalls, substrate_names, np.asarray(substrate_sizes, dtype=int)


def _avg_recall_at_fracs(
    series_recalls: Dict[str, np.ndarray],
    substrate_sizes: np.ndarray,
    frac_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample each substrate's recall curve at the shared fractional grid and
    average across substrates. Returns (mean, ±1 SE) arrays."""
    sub_names = list(series_recalls.keys())
    n_subs = len(sub_names)
    grid = np.full((n_subs, len(frac_grid)), np.nan)
    for i, sn in enumerate(sub_names):
        n_s = substrate_sizes[i]
        recall = series_recalls[sn]
        for j, f in enumerate(frac_grid):
            m = int(np.clip(np.ceil(f * n_s), 0, n_s))
            grid[i, j] = 0.0 if m == 0 else recall[m - 1]
    mean = np.nanmean(grid, axis=0)
    n_valid = np.sum(~np.isnan(grid), axis=0)
    se = np.nanstd(grid, axis=0, ddof=1) / np.sqrt(np.maximum(n_valid, 1))
    return mean, se


def _avg_recall_at_counts(
    series_recalls: Dict[str, np.ndarray],
    substrate_sizes: np.ndarray,
    count_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample each substrate's recall curve at the shared absolute-count grid
    (NaN where m > that substrate's size) and average across substrates."""
    sub_names = list(series_recalls.keys())
    n_subs = len(sub_names)
    grid = np.full((n_subs, len(count_grid)), np.nan)
    for i, sn in enumerate(sub_names):
        n_s = substrate_sizes[i]
        recall = series_recalls[sn]
        for j, m in enumerate(count_grid):
            m = int(m)
            if m <= 0:
                grid[i, j] = 0.0
            elif m <= n_s:
                grid[i, j] = recall[m - 1]
    mean = np.nanmean(grid, axis=0)
    n_valid = np.sum(~np.isnan(grid), axis=0)
    se = np.nanstd(grid, axis=0, ddof=1) / np.sqrt(np.maximum(n_valid, 1))
    return mean, se


def _mean_delta_auc(series_recalls: Dict[str, np.ndarray]) -> float:
    """Mean of per-substrate ΔAUC = AUC(recall vs frac) - 0.5."""
    daucs = []
    for sn, recall in series_recalls.items():
        n_s = len(recall)
        if n_s < 2:
            continue
        # Recall at m=0 is 0; build curve including m=0
        fracs = np.arange(0, n_s + 1) / n_s
        rec_curve = np.concatenate(([0.0], recall))
        daucs.append(float(np.trapz(rec_curve, fracs)) - 0.5)
    return float(np.mean(daucs)) if daucs else float("nan")


def _per_substrate_hit_rates(
    mode_df: pd.DataFrame,
    wt_activity: float,
    budget_fracs: Tuple[float, ...],
    best_beta: float,
) -> Dict[str, np.ndarray]:
    """For the best-β BNN acquisition score and the matched null, compute hit
    rate per substrate at each budget fraction, then equal-weight average across
    substrates. Returns {series_name -> ndarray of length len(budget_fracs)}.
    """
    y_true = mode_df["log_fc"].values.astype(np.float64)
    y_pred = mode_df["y_pred"].values.astype(np.float64)
    acq_std = np.clip(mode_df["acq_std"].values.astype(np.float64), 1e-6, None)
    null_pred = mode_df["null_pred"].values.astype(np.float64)
    subs = mode_df["substrate"].values

    acq = _acq_scores(y_pred, acq_std, best_beta)
    null_tb = _break_ties(null_pred, seed=len(null_pred))
    series_arrays = {**acq, "Null": null_tb}

    out: Dict[str, np.ndarray] = {n: np.zeros(len(budget_fracs)) for n in series_arrays}
    counts = np.zeros(len(budget_fracs), dtype=int)
    for s in np.unique(subs):
        smask = subs == s
        yt = y_true[smask]
        if len(yt) < 1:
            continue
        for j, bf in enumerate(budget_fracs):
            counts[j] += 1
        for name, sa in series_arrays.items():
            sc = sa[smask]
            for j, bf in enumerate(budget_fracs):
                out[name][j] += _hit_rate(sc, yt, wt_activity, bf)
    for name in out:
        out[name] = out[name] / np.maximum(counts, 1)
    return out


def _draw_substrate_avg_recovery(
    ax,
    *,
    series_names: List[str],
    series_colors: Dict[str, str],
    means: Dict[str, np.ndarray],
    ses: Dict[str, np.ndarray],
    daucs: Dict[str, float],
    x: np.ndarray,
    x_unit: str,
    k: int,
    x_max: Optional[float] = None,
    label_map: Optional[Dict[str, str]] = None,
):
    """Render per-substrate-averaged recovery curves with ±1 SE shading."""
    label_map = label_map or {}
    if x_unit == "frac":
        ax.plot([0, 1], [0, 1], "k--", alpha=0.35, lw=1.2, label="Random")
        ax.set_xlabel("Fraction of substrate library screened")
        ax.set_xlim(0, x_max if x_max is not None else 1)
    else:
        n_ref = float(x_max) if x_max else float(x.max() if len(x) else 1.0)
        # Random baseline at the typical substrate size
        diag_x = np.array([0, x_max if x_max is not None else x.max()], dtype=float)
        diag_y = diag_x / max(n_ref, 1)
        ax.plot(diag_x, diag_y, "k--", alpha=0.35, lw=1.2, label="Random")
        ax.set_xlabel("Number of variants screened per substrate")
        ax.set_xlim(0, x_max if x_max is not None else x.max())

    for name in series_names:
        mean = means[name]
        se = ses[name]
        c = series_colors[name]
        lw = 2.5 if name == "Null" else 2
        ls = "--" if name == "Null" else "-"
        alpha = 0.85 if name == "Null" else 1.0
        dauc = daucs.get(name, float("nan"))
        disp = label_map.get(name, name)
        ax.plot(x, mean, color=c, lw=lw, linestyle=ls, alpha=alpha,
                label=f"{disp} (ΔAUC={dauc:+.3f})")
        ax.fill_between(x, np.clip(mean - se, 0, 1), np.clip(mean + se, 0, 1),
                        color=c, alpha=0.15)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel(f"Mean recall of true top-{k} within substrate")


def plot_mode_recovery_panels(
    mode: str,
    mode_df: pd.DataFrame,
    out_path: Path,
    best_beta: float,
    wt_activity: float = 0.0,
    top_k_count: int = 5,
    budget_fracs: Tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50),
    zoom_budget_frac: float = 0.05,
):
    """4-panel per-substrate-averaged recovery figure for one reference mode.

    Panels: (1) recovery of each substrate's true top-``top_k_count`` mutations
    on its own library, averaged across substrates with ±1 SE shading;
    (2) early-budget zoom of the same; (3) per-substrate hit rate vs screening
    budget, averaged across substrates; (4) KDE of activity for the
    top-5%-selected variants vs all (pooled, diagnostic only — distributions
    differ by substrate so averaging KDEs isn't meaningful). The BNN is ranked
    by a single UCB score at this mode's best β; the matched null (β-invariant)
    and a random baseline are overlaid on every panel.
    """
    from scipy.stats import gaussian_kde

    bnn_label = f"BNN (β={best_beta:g})"
    label_map = {"BNN": bnn_label, "Null": "Null"}

    y_true = mode_df["log_fc"].values.astype(np.float64)
    y_pred = mode_df["y_pred"].values.astype(np.float64)
    acq_std = np.clip(mode_df["acq_std"].values.astype(np.float64), 1e-6, None)
    null_pred = mode_df["null_pred"].values.astype(np.float64)
    n = len(y_true)
    if n < max(5, top_k_count):
        logger.warning("plot_mode_recovery_panels(%s): only %d rows — skipping",
                       mode, n)
        return

    k = top_k_count
    recalls, sub_names, sub_sizes = _per_substrate_recall_runs(mode_df, k, best_beta)
    if not sub_names:
        logger.warning("plot_mode_recovery_panels(%s): no substrates with >= k=%d "
                       "rows — skipping", mode, k)
        return
    n_substrates = len(sub_names)
    median_n_s = int(np.median(sub_sizes))

    series_colors = {"BNN": "tab:blue", "Null": "tab:purple"}
    series_names = list(recalls.keys())

    # Shared fractional grid (200 points) and absolute zoom grid (up to
    # zoom_budget_frac of the median substrate library)
    frac_grid = np.linspace(0, 1, 200)
    zoom_n = max(1, int(round(zoom_budget_frac * median_n_s)))
    count_grid = np.arange(0, zoom_n + 1)

    means_f: Dict[str, np.ndarray] = {}
    ses_f: Dict[str, np.ndarray] = {}
    means_c: Dict[str, np.ndarray] = {}
    ses_c: Dict[str, np.ndarray] = {}
    daucs: Dict[str, float] = {}
    for name in series_names:
        m_f, s_f = _avg_recall_at_fracs(recalls[name], sub_sizes, frac_grid)
        m_c, s_c = _avg_recall_at_counts(recalls[name], sub_sizes, count_grid)
        means_f[name] = m_f; ses_f[name] = s_f
        means_c[name] = m_c; ses_c[name] = s_c
        daucs[name] = _mean_delta_auc(recalls[name])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Panel 1: per-substrate-averaged recovery on the substrate library
    _draw_substrate_avg_recovery(
        axes[0, 0],
        series_names=series_names, series_colors=series_colors,
        means=means_f, ses=ses_f, daucs=daucs,
        x=frac_grid, x_unit="frac", k=k, label_map=label_map)
    axes[0, 0].set_title(
        f"Recovery of true top-{k} per substrate (avg over {n_substrates} substrates)")
    axes[0, 0].legend(fontsize=8, loc="lower right")

    # Panel 2: zoom (absolute count per substrate)
    _draw_substrate_avg_recovery(
        axes[0, 1],
        series_names=series_names, series_colors=series_colors,
        means=means_c, ses=ses_c, daucs=daucs,
        x=count_grid.astype(float), x_unit="count", k=k, x_max=float(zoom_n),
        label_map=label_map)
    axes[0, 1].set_title(
        f"Recovery zoom (top-{k}): first {zoom_n} variants per substrate "
        f"(~{zoom_budget_frac*100:.0f}% of substrate library)")
    axes[0, 1].legend(fontsize=8, loc="lower right")

    # Panel 3: per-substrate-averaged hit rate vs screening budget
    ax = axes[1, 0]
    bg_hit_rate = float(np.mean(y_true > wt_activity))
    n_beneficial = int(np.sum(y_true > wt_activity))
    budget_pcts = np.array([b * 100 for b in budget_fracs])
    hit_rates = _per_substrate_hit_rates(mode_df, wt_activity, budget_fracs, best_beta)
    ax.axhline(bg_hit_rate * 100, color="gray", linestyle="--", lw=1.2, alpha=0.7,
               label=f"Background ({bg_hit_rate*100:.1f}%, n={n_beneficial})")
    for name in series_names:
        rates = hit_rates[name]
        c = series_colors[name]
        lw = 2.5 if name == "Null" else 2
        ls = "--" if name == "Null" else "-"
        marker = "s" if name == "Null" else "o"
        ax.plot(budget_pcts, rates * 100, color=c, lw=lw, linestyle=ls,
                marker=marker, ms=5, label=label_map.get(name, name))
    ax.set_xscale("log")
    ax.set_xticks(budget_pcts)
    ax.set_xticklabels([f"{b:g}%" for b in budget_pcts], fontsize=8)
    ax.set_xlabel("Screening budget per substrate (top N% of substrate library)")
    ax.set_ylabel("Hit rate (% selected with log_fc > WT, avg over substrates)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_title(f"Hit rate vs budget (avg over {n_substrates} substrates)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", ls="--", lw=0.8, alpha=0.4)

    # Panel 4: KDE of activity of top-5%-selected vs all (POOLED, diagnostic)
    ax = axes[1, 1]
    dist_budget = 0.05
    k_dist = max(5, int(round(dist_budget * n)))
    null_tb = _break_ties(null_pred, seed=n)
    acq = _acq_scores(y_pred, acq_std, best_beta)
    x_range = np.linspace(y_true.min() - 0.3, y_true.max() + 0.3, 300)
    try:
        kde_all = gaussian_kde(y_true)
        ax.fill_between(x_range, kde_all(x_range), alpha=0.2, color="gray",
                        label="All data")
        ax.plot(x_range, kde_all(x_range), color="gray", lw=1.5, alpha=0.5)
    except Exception:
        pass
    null_top = np.argsort(-null_tb, kind="stable")[:k_dist]
    y_null_sel = y_true[null_top]
    if len(y_null_sel) >= 3:
        try:
            kde_null = gaussian_kde(y_null_sel)
            ax.plot(x_range, kde_null(x_range), color=series_colors["Null"], lw=2.5,
                    linestyle="--", alpha=0.85,
                    label=f"Null top {dist_budget*100:.0f}%")
        except Exception:
            pass
    for name, s in acq.items():
        top = np.argsort(-s, kind="stable")[:k_dist]
        y_sel = y_true[top]
        if len(y_sel) >= 3:
            try:
                kde_sel = gaussian_kde(y_sel)
                ax.plot(x_range, kde_sel(x_range), color=series_colors[name],
                        lw=2, label=f"{label_map.get(name, name)} top {dist_budget*100:.0f}%")
            except Exception:
                pass
    ax.axvline(wt_activity, color="black", linestyle="--", lw=1.2, alpha=0.7,
               label=f"WT ({wt_activity:.3f})")
    ax.set_xlabel("log_fc")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Activity dist of top {dist_budget*100:.0f}%-selected (pooled, diagnostic)")
    ax.legend(fontsize=8)

    fig.suptitle(
        f"Acquisition Recovery — {mode}  |  per-substrate avg over "
        f"{n_substrates} substrates, median n_s≈{median_n_s} rows",
        fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cross_mode_recovery_summary(
    mode_results: Dict[str, dict],
    out_path: Path,
    top_k_count: int = 5,
    zoom_budget_frac: float = 0.05,
):
    """One figure: BNN @ best β vs matched null across all reference modes.

    Two panels: full library curve and early-budget zoom. One color per mode
    (BNN solid, matched null dashed). Each mode's BNN curve is ranked by that
    mode's own best-β UCB score (shown in the legend). Target = each substrate's
    true top-k mutations; curves are per-substrate-averaged within each mode
    (matching the per-mode panels), so cross-mode comparison is apples-to-apples.
    """
    modes = [m for m in ALL_MODES if m in mode_results and "df" in mode_results[m]]
    if not modes:
        logger.warning("plot_cross_mode_recovery_summary: no mode dataframes — skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    palette = plt.cm.tab10(np.linspace(0, 1, max(3, len(modes))))
    mode_colors = {m: palette[i] for i, m in enumerate(modes)}

    # Diagonals
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.35, lw=1.2, label="Random")
    axes[0].set_xlabel("Fraction of substrate library screened")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title(
        f"Recovery of true top-{top_k_count} per substrate "
        "— BNN @ best β vs null, per mode")

    axes[1].set_xlabel("Number of variants screened per substrate")
    axes[1].set_ylim(0, 1.02)

    frac_grid = np.linspace(0, 1, 200)
    # Pick a single zoom bound from the median substrate size across modes
    all_median_n_s = []

    for mode in modes:
        mdf = mode_results[mode]["df"]
        bb = float(mode_results[mode]["metrics"]["best_beta"])
        recalls, sub_names, sub_sizes = _per_substrate_recall_runs(
            mdf, k=top_k_count, best_beta=bb)
        if not sub_names:
            continue
        all_median_n_s.append(int(np.median(sub_sizes)))

        # BNN best-β curve (frac axis)
        bnn_m_f, _ = _avg_recall_at_fracs(
            recalls["BNN"], sub_sizes, frac_grid)
        null_m_f, _ = _avg_recall_at_fracs(
            recalls["Null"], sub_sizes, frac_grid)
        bnn_dauc = _mean_delta_auc(recalls["BNN"])
        null_dauc = _mean_delta_auc(recalls["Null"])
        c = mode_colors[mode]
        axes[0].plot(frac_grid, bnn_m_f, color=c, lw=2,
                     label=f"{mode} BNN β={bb:g} (ΔAUC={bnn_dauc:+.3f})")
        axes[0].plot(frac_grid, null_m_f, color=c, lw=2, linestyle="--",
                     alpha=0.7, label=f"{mode} null (ΔAUC={null_dauc:+.3f})")

    if all_median_n_s:
        zoom_n_max = max(1, int(round(zoom_budget_frac * max(all_median_n_s))))
        count_grid = np.arange(0, zoom_n_max + 1)
        for mode in modes:
            mdf = mode_results[mode]["df"]
            bb = float(mode_results[mode]["metrics"]["best_beta"])
            recalls, sub_names, sub_sizes = _per_substrate_recall_runs(
                mdf, k=top_k_count, best_beta=bb)
            if not sub_names:
                continue
            bnn_m_c, _ = _avg_recall_at_counts(
                recalls["BNN"], sub_sizes, count_grid)
            null_m_c, _ = _avg_recall_at_counts(
                recalls["Null"], sub_sizes, count_grid)
            c = mode_colors[mode]
            axes[1].plot(count_grid, bnn_m_c, color=c, lw=2,
                         label=f"{mode} BNN β={bb:g}")
            axes[1].plot(count_grid, null_m_c, color=c, lw=2,
                         linestyle="--", alpha=0.7, label=f"{mode} null")
        ref_n = max(all_median_n_s)
        axes[1].plot([0, zoom_n_max], [0, zoom_n_max / ref_n], "k--",
                     alpha=0.35, lw=1.2, label="Random (median n_s)")
        axes[1].set_xlim(0, zoom_n_max)
    axes[1].set_title(
        f"Recovery zoom (top-{top_k_count}, first ~{zoom_budget_frac*100:.0f}% per substrate)")

    axes[0].set_ylabel(f"Mean recall of true top-{top_k_count} within substrate")
    axes[1].set_ylabel(f"Mean recall of true top-{top_k_count} within substrate")
    axes[0].legend(fontsize=8, loc="lower right", ncol=2)
    axes[1].legend(fontsize=8, loc="lower right", ncol=2)

    fig.suptitle("BNN vs matched null across reference modes "
                 "(per-substrate-averaged)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_position_recovery(
    mode: str,
    mode_df: pd.DataFrame,
    out_path: Path,
    k: int,
    best_beta: float,
    x_max: int = 20,
):
    """Within-(substrate,position) recovery, averaged across groups.

    For each (substrate, position) group of ≤19 mutations, rank under the BNN's
    best-β UCB score and compute recall of true top-k within the group at every
    cutoff m. Average recall across groups at each m. Curves naturally taper as
    m grows past smaller groups.

    Two series: BNN @ best β, matched null. ±1 SE shading.
    """
    y_true = mode_df["log_fc"].values.astype(np.float64)
    y_pred = mode_df["y_pred"].values.astype(np.float64)
    acq_std = np.clip(mode_df["acq_std"].values.astype(np.float64), 1e-6, None)
    null_pred = mode_df["null_pred"].values.astype(np.float64)
    subs = mode_df["substrate"].values
    pos = mode_df["position"].values

    acq_scores = _acq_scores(y_pred, acq_std, best_beta)
    bnn_label = f"BNN (β={best_beta:g})"
    label_map = {"BNN": bnn_label, "Null": "Null"}

    series_names = list(acq_scores.keys()) + ["Null"]
    series_scores = {**acq_scores, "Null": _break_ties(null_pred, seed=len(null_pred))}
    series_colors = {"BNN": "tab:blue", "Null": "tab:purple"}

    # Collect per-group per-m recall: dict[name] -> list of (m, recall_at_m) arrays
    per_group_recalls: Dict[str, List[np.ndarray]] = {n: [] for n in series_names}
    per_group_sizes: List[int] = []

    for s in np.unique(subs):
        smask = subs == s
        for p in np.unique(pos[smask]):
            gmask = smask & (pos == p)
            yt = y_true[gmask]
            valid = ~np.isnan(yt)
            if valid.sum() < 2:
                continue
            yt_g = yt[valid]
            n_g = len(yt_g)
            k_eff = min(k, n_g)
            if k_eff < 1:
                continue
            true_top = set(np.argsort(-yt_g, kind="stable")[:k_eff].tolist())
            per_group_sizes.append(n_g)
            for name, all_scores in series_scores.items():
                ys = all_scores[gmask][valid]
                if np.any(np.isnan(ys)):
                    nan_mask = ~np.isnan(ys)
                    ys = ys[nan_mask]
                    yt_use = yt_g[nan_mask]
                    if len(ys) < 2:
                        per_group_recalls[name].append(np.full(x_max, np.nan))
                        continue
                    true_top_use = set(
                        np.argsort(-yt_use, kind="stable")[:min(k, len(ys))].tolist())
                else:
                    true_top_use = true_top
                order = np.argsort(-ys, kind="stable")
                recall_at_m = np.full(x_max, np.nan)
                running = 0
                for m in range(1, min(len(order), x_max) + 1):
                    running = len(set(order[:m].tolist()) & true_top_use)
                    recall_at_m[m - 1] = running / k_eff
                per_group_recalls[name].append(recall_at_m)

    if not per_group_sizes:
        logger.warning("plot_per_position_recovery(%s, k=%d): no valid groups",
                       mode, k)
        return

    # Stack into (n_groups, x_max) arrays with NaN for cutoffs past group size
    stacked: Dict[str, np.ndarray] = {
        n: np.stack(per_group_recalls[n], axis=0) for n in series_names
    }

    ms = np.arange(1, x_max + 1)
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Random baseline: average of k/n_g across groups, scaled by m
    # E[recall@m] under random selection = (m / n_g) for m ≤ n_g
    sizes = np.array(per_group_sizes, dtype=float)
    rand = np.array([
        float(np.nanmean(np.where(sizes >= m, m / sizes, np.nan)))
        for m in ms
    ])
    ax.plot(ms, np.clip(rand, 0, 1), "k--", alpha=0.4, lw=1.2,
            label="Random (avg over groups)")

    for name in series_names:
        arr = stacked[name]
        mean = np.nanmean(arr, axis=0)
        n_valid = np.sum(~np.isnan(arr), axis=0)
        se = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(n_valid, 1))
        c = series_colors[name]
        lw = 2.5 if name == "Null" else 2
        ls = "--" if name == "Null" else "-"
        alpha = 0.85 if name == "Null" else 1.0
        ax.plot(ms, mean, color=c, lw=lw, linestyle=ls, alpha=alpha,
                label=label_map.get(name, name))
        ax.fill_between(ms, mean - se, mean + se, color=c, alpha=0.15)

    ax.set_xlim(1, x_max)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(ms[::max(1, x_max // 10)])
    ax.set_xlabel("Number of mutations selected per (substrate, position)")
    ax.set_ylabel(f"Mean recall of true top-{k} within group")
    ax.set_title(
        f"Per-position recovery — {mode}, top-{k}  "
        f"(avg over {len(per_group_sizes)} groups)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(ls="--", lw=0.8, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# CLI + main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate, score, and plot BNN2 CV pairwise predictions.",
    )
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Directory produced by new_05_bnn2_train.py")
    parser.add_argument("--modes", type=str, nargs="*", default=list(ALL_MODES),
                        choices=list(ALL_MODES),
                        help="Reference modes to evaluate (default: all four)")
    parser.add_argument("--output-subdir", type=str, default="scoring",
                        help="Subdirectory under run-dir for outputs (default: scoring)")
    parser.add_argument("--distance-weighted-temperature", type=float, default=None,
                        help="Override softmax temperature for both BNN and null "
                             "distance-weighted modes; defaults to train_metadata value")
    parser.add_argument("--config", type=str, default=None,
                        help="Config path (for processed_dir lookup); defaults to "
                             "the repo's round3/config.yaml")
    parser.add_argument("--beta", type=float, default=None,
                        help="Override the headline UCB β for ALL modes. Default: "
                             "read each mode's best_beta from hyperparams.json "
                             "(per_mode_results), falling back to the global "
                             "best_beta, then 0.0.")
    parser.add_argument("--beta-grid", type=str, default=None,
                        help="JSON list of βs for the sweep plots/arrays, e.g. "
                             f"'[0.0, 0.5, 1.0]'. Default: {list(DEFAULT_BETA_GRID)} "
                             "(mirrors the hyperopt --ucb-betas grid).")
    parser.add_argument("--acq-sigma", type=str, default="within_epi_ale",
                        choices=["within_epi_ale", "within_epi", "total"],
                        help="σ used by the UCB β-ranking (NOT the reported "
                             "uncertainty). 'within_epi' (default) = epistemic-only, "
                             "mode-consistent, explores reducible uncertainty. "
                             "'within_epi_ale' adds aleatoric; 'total' reproduces "
                             "the old (pre-fix) behavior.")
    return parser.parse_args()


def main():
    args = parse_args()
    t_start = time.time()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"--run-dir does not exist: {run_dir}")
    out_dir = run_dir / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir / "score.log")

    logger.info("=" * 60)
    logger.info("new_05b_bnn2_score.py — scoring %s", run_dir)
    logger.info("=" * 60)

    # Load artifacts
    pairwise_df = pd.read_csv(run_dir / "pairwise_predictions.csv")
    train_lookup_df = pd.read_csv(run_dir / "train_lookup.csv")
    with open(run_dir / "train_metadata.json") as f:
        train_metadata = json.load(f)
    with open(run_dir / "hyperparams.json") as f:
        hyperparams = json.load(f)

    logger.info("Loaded: %d pairwise rows, %d train-lookup rows, split=%s",
                len(pairwise_df), len(train_lookup_df),
                train_metadata.get("split_type"))

    # UCB β config. best_beta comes from the hyperparams.json the trainer wrote
    # (it preserved the hyperopt fields); --beta overrides it for all modes.
    if args.beta_grid is not None:
        beta_grid = [float(b) for b in json.loads(args.beta_grid)]
    else:
        beta_grid = list(DEFAULT_BETA_GRID)
    global_best_beta = float(hyperparams.get("best_beta", 0.0) or 0.0)
    per_mode_best = hyperparams.get("per_mode_results", {}) or {}
    logger.info("UCB β: grid=%s  global best_beta=%g  (override=%s)",
                beta_grid, global_best_beta,
                args.beta if args.beta is not None else "none")

    config = load_config(args.config)
    processed_dir = PROJECT_ROOT / config["data"]["output_dir"]
    embeddings = load_all_embeddings(processed_dir)
    substrate_meta = load_substrate_metadata(processed_dir)

    null_emb = train_metadata["null_model_embedding"]
    null_metric = train_metadata["null_model_distance_metric"]
    dw_temp = (args.distance_weighted_temperature
               if args.distance_weighted_temperature is not None
               else float(train_metadata.get("distance_weight_temperature", 1.0)))

    logger.info("Null/distance config: embedding=%s  metric=%s  τ=%.3f",
                null_emb, null_metric, dw_temp)
    logger.info("Modes to evaluate: %s", args.modes)

    # Per-mode aggregation + scoring
    pred_by_mode_dir = out_dir / "predictions_by_mode"
    pred_by_mode_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    mode_results: Dict[str, dict] = {}
    for mode in args.modes:
        logger.info("── Mode: %s ──", mode)
        mode_df = aggregate_and_null_per_fold(
            mode=mode,
            pairwise_df=pairwise_df,
            train_lookup_df=train_lookup_df,
            embeddings=embeddings,
            substrate_embedding_type=null_emb,
            distance_metric=null_metric,
            distance_weight_temperature=dw_temp,
            acq_sigma=args.acq_sigma,
        )
        if len(mode_df) == 0:
            logger.warning("  Mode %s: no rows after aggregation — skipping", mode)
            continue

        if args.beta is not None:
            mode_best_beta = float(args.beta)
        else:
            mode_best_beta = float(
                (per_mode_best.get(mode, {}) or {}).get("best_beta", global_best_beta))
        mode_metrics = compute_metrics_for_mode(
            mode, mode_df, best_beta=mode_best_beta, beta_grid=beta_grid)
        logger.info("  %-18s BNN: MAE=%.4f ρ=%.4f | Null: MAE=%.4f ρ=%.4f | Δρ=%.4f",
                    mode,
                    mode_metrics["bnn"]["mae"], mode_metrics["bnn"]["spearman_rho"],
                    mode_metrics["null"]["mae"], mode_metrics["null"]["spearman_rho"],
                    mode_metrics["delta_spearman"])
        logger.info("  %-18s @best β=%g: ρ=%.4f (Δ=%+.4f) | ndcg=%.4f (Δ=%+.4f) | "
                    "top3=%.3f (Δ=%+.3f) vs null",
                    mode, mode_best_beta,
                    mode_metrics["bnn_at_best_beta"]["spearman_rho"],
                    mode_metrics["delta_at_best_beta"]["spearman_rho"],
                    mode_metrics["bnn_at_best_beta"]["ndcg"],
                    mode_metrics["delta_at_best_beta"]["ndcg"],
                    mode_metrics["bnn_at_best_beta"]["top3_recovery"],
                    mode_metrics["delta_at_best_beta"]["top3_recovery"])
        logger.info("  %-18s Top-3 recovery (per-substrate mean): BNN=%.3f | Null=%.3f | Δ=%.3f  (n_substrates=%d)",
                    mode,
                    mode_metrics["bnn"]["top3_recovery"]["recovery"],
                    mode_metrics["null"]["top3_recovery"]["recovery"],
                    mode_metrics["delta_top3_recovery"],
                    mode_metrics["bnn"]["top3_recovery"]["n_substrates"])
        logger.info("  %-18s Top-5 recovery (per-substrate mean): BNN=%.3f | Null=%.3f | Δ=%.3f  (n_substrates=%d)",
                    mode,
                    mode_metrics["bnn"]["top5_recovery"]["recovery"],
                    mode_metrics["null"]["top5_recovery"]["recovery"],
                    mode_metrics["delta_top5_recovery"],
                    mode_metrics["bnn"]["top5_recovery"]["n_substrates"])
        logger.info("  %-18s Per-position top-1:     BNN=%.3f | Null=%.3f | Δ=%.3f  (n_substrates=%d)",
                    mode,
                    mode_metrics["bnn"]["per_position_top1_recovery"]["recovery"],
                    mode_metrics["null"]["per_position_top1_recovery"]["recovery"],
                    mode_metrics["delta_per_position_top1_recovery"],
                    mode_metrics["bnn"]["per_position_top1_recovery"]["n_substrates"])

        mode_df.to_csv(pred_by_mode_dir / f"{mode}.csv", index=False)
        plot_per_mode_diagnostics(mode, mode_df, mode_metrics, figures_dir)
        plot_mode_recovery_panels(
            mode, mode_df,
            figures_dir / f"acquisition_recovery_{mode}.png",
            best_beta=mode_metrics["best_beta"])
        for k_pp in (1, 3):
            plot_per_position_recovery(
                mode, mode_df,
                figures_dir / f"per_position_recovery_top{k_pp}_{mode}.png",
                k=k_pp, best_beta=mode_metrics["best_beta"])
        plot_beta_sweep_mode(
            mode, mode_metrics["beta_sweep"], mode_metrics["best_beta"],
            figures_dir / f"beta_sweep_{mode}.png")

        mode_results[mode] = {
            "metrics": mode_metrics,
            "n_rows": int(len(mode_df)),
            "df": mode_df,
        }

    # Cross-mode comparison plots
    if mode_results:
        plot_per_substrate_spearman_all_modes(
            mode_results, figures_dir / "per_substrate_spearman_all_modes.png")
        # Ranking-metric bars: BNN @ best β (per mode) vs β-invariant null.
        plot_mode_summary_bars(
            mode_results, figures_dir / "overall_spearman_by_mode.png",
            metric="spearman_rho",
            bnn_source="bnn_at_best_beta", bnn_metric="spearman_rho")
        # Regression metric: β-meaningless, stays at β=0.
        plot_mode_summary_bars(
            mode_results, figures_dir / "overall_mae_by_mode.png",
            metric="mae")
        plot_mode_summary_bars(
            mode_results, figures_dir / "overall_ndcg_by_mode.png",
            metric="ndcg", title="Overall NDCG: BNN vs matched null",
            bnn_source="bnn_at_best_beta", bnn_metric="ndcg")
        plot_mode_summary_bars(
            mode_results, figures_dir / "overall_top1_recovery_by_mode.png",
            metric="top1_recovery.recovery",
            title="Top-1 recovery (overall): BNN vs matched null",
            bnn_source="bnn_at_best_beta", bnn_metric="top1_recovery")
        plot_mode_summary_bars(
            mode_results, figures_dir / "overall_top3_recovery_by_mode.png",
            metric="top3_recovery.recovery",
            title="Top-3 recovery (overall): BNN vs matched null",
            bnn_source="bnn_at_best_beta", bnn_metric="top3_recovery")
        plot_mode_summary_bars(
            mode_results, figures_dir / "overall_top5_recovery_by_mode.png",
            metric="top5_recovery.recovery",
            title="Top-5 recovery (overall): BNN vs matched null",
            bnn_source="bnn_at_best_beta", bnn_metric="top5_recovery")
        plot_mode_summary_bars(
            mode_results, figures_dir / "per_position_top1_recovery_by_mode.png",
            metric="per_position_top1_recovery.recovery",
            title="Per-position top-1 recovery: BNN vs matched null",
            bnn_source="bnn_at_best_beta", bnn_metric="per_position_top1_recovery")
        plot_mode_summary_bars(
            mode_results, figures_dir / "per_position_top3_recovery_mean_by_mode.png",
            metric="per_position_top3_recovery_mean.mean",
            title="Per-position top-3 recovery (mean): BNN vs matched null",
            bnn_source="bnn_at_best_beta", bnn_metric="per_position_top3_recovery_mean")
        plot_mode_summary_bars(
            mode_results, figures_dir / "per_position_top5_recovery_mean_by_mode.png",
            metric="per_position_top5_recovery_mean.mean",
            title="Per-position top-5 recovery (mean): BNN vs matched null",
            bnn_source="bnn_at_best_beta", bnn_metric="per_position_top5_recovery_mean")
        plot_mode_summary_bars(
            mode_results, figures_dir / "per_position_spearman_mean_by_mode.png",
            metric="per_position_spearman_mean.mean",
            title="Per-position Spearman ρ (mean): BNN vs matched null",
            bnn_source="bnn_at_best_beta", bnn_metric="per_position_spearman_mean")
        # Regression metric: β-meaningless, stays at β=0.
        plot_mode_summary_bars(
            mode_results, figures_dir / "per_position_mae_mean_by_mode.png",
            metric="per_position_mae_mean.mean",
            title="Per-position MAE (mean): BNN vs matched null")
        plot_mode_summary_bars(
            mode_results, figures_dir / "per_position_ndcg_mean_by_mode.png",
            metric="per_position_ndcg_mean.mean",
            title="Per-position NDCG (mean): BNN vs matched null",
            bnn_source="bnn_at_best_beta", bnn_metric="per_position_ndcg_mean")
        plot_cross_mode_recovery_summary(
            mode_results, figures_dir / "acquisition_recovery_all_modes.png")
        # Cross-mode β sweep: one figure per ranking metric.
        for _mk in RANKING_METRIC_KEYS:
            plot_beta_sweep_all_modes(
                mode_results, _mk,
                figures_dir / f"beta_sweep_all_modes_{_mk}.png")

    # Training curves (one file covers all folds — reuses existing plot funcs)
    fold_histories = load_fold_histories(run_dir)
    if fold_histories:
        held_out = train_metadata.get("held_out_substrates") or None
        plot_training_curves(fold_histories,
                             figures_dir / "training_curves.png",
                             fold_labels=held_out)
        plot_loss_decomposition(fold_histories,
                                figures_dir / "loss_decomposition.png")
    else:
        logger.info("No training_histories.json — skipping training-curve plots")

    # Assemble unified metrics.json (drop the cached DataFrames — not JSON-able)
    modes_serializable = {
        m: {k: v for k, v in r.items() if k != "df"}
        for m, r in mode_results.items()
    }
    overall = {
        "run_dir": str(run_dir),
        "acq_sigma": args.acq_sigma,
        "split_type": train_metadata.get("split_type"),
        "held_out_substrates": train_metadata.get("held_out_substrates"),
        "training_substrates_kept": train_metadata.get("training_substrates_kept"),
        "training_substrates_considered": train_metadata.get("training_substrates_considered"),
        "null_model_embedding": null_emb,
        "null_model_distance_metric": null_metric,
        "distance_weighted_temperature": dw_temp,
        "modes": modes_serializable,
        "hyperparams": hyperparams,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(overall, f, indent=2, default=str)

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("new_05b_bnn2_score.py done (%.1fs)  —  %s", elapsed, out_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
