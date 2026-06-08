#!/usr/bin/env python
"""
new_opt_05.py — Optuna hyperopt for BNN2, targeting per-position top-1 recovery
===============================================================================

Wraps the new pipeline:
  - `new_05_bnn2_train.run_cv_and_collect_predictions` for CV training,
  - `new_05b_bnn2_score.aggregate_and_null_per_fold` + recovery helpers for
    scoring.

Key differences from the legacy ``opt_05_bnn2.py``:

  1. Objective defaults to **per-position top-1 recovery** — the metric that
     matches the downstream use case of recommending one mutation per position.
  2. Each trial scores predictions under **every reference mode × β-grid**
     combination (``μ + β·σ`` for the four matched-pair modes
     formaldehyde / nearest / avg_all / distance_weighted) and returns the
     ``max`` over the full grid. The winning (reference_mode, β) pair is
     recorded per trial, so the optimiser targets the real objective:
     "best top-1-per-position regardless of how you score".
  3. ``--scope formaldehyde`` restricts CV to the single fold holding out
     Formaldehyde (via ``new_05``'s ``--target-substrate``), ~6× faster than
     full substrate-split CV.
  4. After the study finishes we export a **working range** summarising which
     hyperparameter values cluster in the top ``--top-frac`` trials:
     ``working_range.json`` (machine-readable) and
     ``narrowed_config_suggestion.yaml`` (human-friendly). A follow-up run can
     use ``--narrow-from <working_range.json>`` to intersect the search space
     with the prior good values — the "wide → narrow" two-stage workflow.

Aggregation semantics (post-2026-05 fix):
  - All "overall" metric names (``spearman_rho``, ``mae``, ``ndcg``,
    ``top3_recovery``) compute the metric within each held-out substrate and
    equal-weight average across substrates. Previously these names meant
    pooled-across-substrates rows, which inflated scores via substrate-level
    offsets. **If you ran prior hyperopt studies with one of those metrics,
    those studies optimised against the wrong target and should be re-run.**
    The default ``per_position_top1_recovery`` was always per-substrate-correct
    and is unaffected.

Usage:
    python new_opt_05.py --scope formaldehyde --n-trials 100
    python new_opt_05.py --scope all-substrates --narrow-from prior/working_range.json --n-trials 40
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from importlib.util import spec_from_file_location, module_from_spec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution + import the other scripts (they start with digits)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR.parent))  # for `from bnns import ...`

_common_spec = spec_from_file_location(
    "bnn2_common", SCRIPT_DIR / "05_bnn2_common.py")
_common = module_from_spec(_common_spec)
_common_spec.loader.exec_module(_common)

_train_spec = spec_from_file_location(
    "new_05_bnn2_train", SCRIPT_DIR / "new_05_bnn2_train.py")
_train = module_from_spec(_train_spec)
_train_spec.loader.exec_module(_train)

_score_spec = spec_from_file_location(
    "new_05b_bnn2_score", SCRIPT_DIR / "new_05b_bnn2_score.py")
_score = module_from_spec(_score_spec)
_score_spec.loader.exec_module(_score)

# From common
load_config = _common.load_config
resolve_param = _common.resolve_param
resolve_config_block = _common.resolve_config_block
get_device = _common.get_device
parse_pca_value = _common.parse_pca_value
setup_logging = _common.setup_logging
load_multi_substrate_data = _common.load_multi_substrate_data
load_all_embeddings = _common.load_all_embeddings
load_substrate_metadata = _common.load_substrate_metadata
load_bnn1_backbone = _common.load_bnn1_backbone
load_bnn1_preprocessing = _common.load_bnn1_preprocessing
select_best_substrate_metric = _common.select_best_substrate_metric
FORMALDEHYDE_SUBSTRATE = _common.FORMALDEHYDE_SUBSTRATE
compute_ndcg = _common.compute_ndcg

# From the new train script
run_cv_and_collect_predictions = _train.run_cv_and_collect_predictions

# From the scorer
aggregate_and_null_per_fold = _score.aggregate_and_null_per_fold
_topk_recovery = _score._topk_recovery
_per_position_top1_recovery = _score._per_position_top1_recovery
_per_position_spearman_mean = _score._per_position_spearman_mean
_per_position_mae_mean = _score._per_position_mae_mean
_per_position_ndcg_mean = _score._per_position_ndcg_mean
_per_position_topk_recovery_mean = _score._per_position_topk_recovery_mean
_per_substrate_spearman_mean = _score._per_substrate_spearman_mean
_per_substrate_mae_mean = _score._per_substrate_mae_mean
_per_substrate_ndcg_mean = _score._per_substrate_ndcg_mean
_per_substrate_topk_recovery_mean = _score._per_substrate_topk_recovery_mean
_safe_spearman = _score._safe_spearman
_safe_mae = _score._safe_mae


ALL_REFERENCE_MODES = ("formaldehyde", "nearest", "avg_all", "distance_weighted")
ALL_METRICS = (
    # per-position (within-(substrate, position), averaged over positions then substrates)
    "per_position_top1_recovery",
    "per_position_top3_recovery_mean",
    "per_position_spearman_mean",
    "per_position_mae_mean",
    "per_position_ndcg_mean",
    # overall (per-substrate, averaged over substrates) — historically named
    # "global" but no longer pooled across substrates
    "top3_recovery",
    "spearman_rho",
    "mae",
    "ndcg",
)

_COLLAPSE_PENALTY = 0.2   # subtracted from maximise-direction objective
_COLLAPSE_THRESHOLD = 0.9


# ═══════════════════════════════════════════════════════════════════════════
# Search-space helpers (with optional "narrow-from" filter)
# ═══════════════════════════════════════════════════════════════════════════

def _value_to_key(v) -> str:
    """Canonical hashable key for dict-based frequency counting."""
    if isinstance(v, list):
        return json.dumps(v)
    if v is None:
        return "null"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _narrow_entry_allows(entry: dict, value) -> bool:
    """Return True if a candidate value is within the narrow-from entry."""
    kind = entry.get("kind", "categorical")
    if kind == "numeric":
        try:
            v = float(value)
        except (TypeError, ValueError):
            return value in entry.get("good_values", []) or _value_to_key(value) in {
                _value_to_key(x) for x in entry.get("good_values", [])}
        lo = entry.get("min_good"); hi = entry.get("max_good")
        if lo is None or hi is None:
            return True
        return lo <= v <= hi
    good = entry.get("good_values", [])
    good_keys = {_value_to_key(g) for g in good}
    return _value_to_key(value) in good_keys


def _apply_narrow_list(
    name: str,
    search_space: list,
    narrow_map: Optional[Dict[str, dict]],
) -> list:
    """Intersect a config search list with the working range (if supplied)."""
    if not narrow_map or name not in narrow_map:
        return search_space
    entry = narrow_map[name]
    # numeric_distribution entries are handled via _narrow_distribution on the
    # dict side; on the list side we let every value pass through.
    if entry.get("kind") == "numeric_distribution":
        return search_space
    kept = [v for v in search_space if _narrow_entry_allows(entry, v)]
    if not kept:
        logger.warning("narrow-from: '%s' intersection with working range is empty — "
                       "falling back to original search space %s", name, search_space)
        return search_space
    if len(kept) != len(search_space):
        logger.info("narrow-from: '%s' → %s (was %s)", name, kept, search_space)
    return kept


def _narrow_distribution(
    name: str,
    dist: dict,
    narrow_map: Optional[Dict[str, dict]],
) -> dict:
    """If `narrow_map[name]` is a numeric_distribution entry, clamp the bounds."""
    if not narrow_map or name not in narrow_map:
        return dist
    entry = narrow_map[name]
    if entry.get("kind") != "numeric_distribution":
        return dist
    t = dist.get("type")
    if t not in ("uniform", "loguniform", "int"):
        return dist
    try:
        orig_low = float(dist["low"])
        orig_high = float(dist["high"])
    except (KeyError, TypeError, ValueError):
        return dist
    min_good = entry.get("min_good")
    max_good = entry.get("max_good")
    if min_good is None or max_good is None:
        return dist
    new_low = max(orig_low, float(min_good))
    new_high = min(orig_high, float(max_good))
    if not (new_low < new_high):
        logger.warning("narrow-from: '%s' narrowed bounds collapsed "
                       "(new_low=%g ≥ new_high=%g) — keeping original %g..%g",
                       name, new_low, new_high, orig_low, orig_high)
        return dist
    if (new_low, new_high) != (orig_low, orig_high):
        logger.info("narrow-from: '%s' clamped low=%g high=%g (was %g..%g)",
                    name, new_low, new_high, orig_low, orig_high)
    out = dict(dist)
    if t == "int":
        out["low"] = int(round(new_low))
        out["high"] = int(round(new_high))
    else:
        out["low"] = new_low
        out["high"] = new_high
    return out


def _suggest_from_distribution(trial, name: str, dist: dict, is_pca: bool = False):
    """Dispatch an Optuna suggestion from a dict-form distribution entry."""
    if is_pca:
        raise ValueError(
            f"Parameter '{name}' is declared with is_pca=True but search is a "
            f"distribution dict. PCA params (None / int / float) must use list form.")
    t = dist.get("type")
    if t is None:
        raise ValueError(f"Parameter '{name}': distribution dict missing 'type' key: {dist!r}")
    if t == "uniform":
        low = float(dist["low"]); high = float(dist["high"])
        return trial.suggest_float(name, low, high)
    if t == "loguniform":
        low = float(dist["low"]); high = float(dist["high"])
        if low <= 0:
            raise ValueError(f"Parameter '{name}': loguniform requires low > 0 (got {low})")
        return trial.suggest_float(name, low, high, log=True)
    if t == "int":
        low = int(dist["low"]); high = int(dist["high"])
        step = int(dist.get("step", 1))
        log = bool(dist.get("log", False))
        if log and step != 1:
            step = 1  # Optuna disallows step!=1 with log
        return trial.suggest_int(name, low, high, step=step, log=log)
    if t == "categorical":
        values = list(dist["values"])
        return trial.suggest_categorical(name, values)
    raise ValueError(f"Parameter '{name}': unknown distribution type '{t}'")


def search_or_fixed(
    trial,
    name: str,
    config_entry,
    is_pca: bool = False,
    narrow_map: Optional[Dict[str, dict]] = None,
):
    """Dispatch an Optuna suggestion.

    Supports two ``search:`` forms:
      - list  (e.g. ``search: [0.1, 0.3, 0.5]``) → suggest_categorical
      - dict  (e.g. ``search: {type: loguniform, low: 1e-5, high: 1e-2}``)
                                              → suggest_float / suggest_int

    Both respect `narrow_map` (from a prior ``working_range.json``):
      - list form is intersected with the good values,
      - dict form has its bounds clamped to the good range.
    """
    if not isinstance(config_entry, dict) or "search" not in config_entry:
        return resolve_param(config_entry)
    search_space = config_entry["search"]
    if search_space is None:
        return resolve_param(config_entry)

    # ── Dict (distribution) form ──
    if isinstance(search_space, dict) and "type" in search_space:
        dist = _narrow_distribution(name, search_space, narrow_map)
        return _suggest_from_distribution(trial, name, dist, is_pca=is_pca)

    # ── List (categorical) form ──
    search_space = _apply_narrow_list(name, search_space, narrow_map)

    if is_pca:
        options = ["none" if v is None else str(v) for v in search_space]
        result_str = trial.suggest_categorical(name, options)
        return parse_pca_value(result_str)

    if search_space and isinstance(search_space[0], list):
        result_str = trial.suggest_categorical(
            name, [json.dumps(d) for d in search_space])
        return json.loads(result_str)

    return trial.suggest_categorical(name, search_space)


# ═══════════════════════════════════════════════════════════════════════════
# Metric dispatch + β-grid evaluation
# ═══════════════════════════════════════════════════════════════════════════

def _eval_metric(
    metric: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    substrates: np.ndarray,
    positions: np.ndarray,
) -> float:
    """Return a "higher is better" scalar for each supported metric.

    Per-position metrics use a substrate-first two-step pool: per
    (substrate, position) → mean across that substrate's positions →
    mean across substrates. Overall metrics use a one-step per-substrate
    pool: within each substrate → mean across substrates. Both treat each
    held-out substrate equally; neither pools rows across substrates.
    """
    # ── per-(substrate,position) → substrate-first two-step mean ──
    if metric == "per_position_top1_recovery":
        return _per_position_top1_recovery(y_true, y_score, substrates, positions)["recovery"]
    if metric == "per_position_top3_recovery_mean":
        return _per_position_topk_recovery_mean(y_true, y_score, substrates, positions, k=3)["mean"]
    if metric == "per_position_spearman_mean":
        return _per_position_spearman_mean(y_true, y_score, substrates, positions)["mean"]
    if metric == "per_position_mae_mean":
        v = _per_position_mae_mean(y_true, y_score, substrates, positions)["mean"]
        return -v if not math.isnan(v) else float("nan")
    if metric == "per_position_ndcg_mean":
        return _per_position_ndcg_mean(y_true, y_score, substrates, positions, k=None)["mean"]
    # ── overall (per-substrate, averaged over substrates) ──
    if metric == "top3_recovery":
        return _per_substrate_topk_recovery_mean(y_true, y_score, substrates, k=3)["recovery"]
    if metric == "spearman_rho":
        return _per_substrate_spearman_mean(y_true, y_score, substrates)["mean"]
    if metric == "mae":
        v = _per_substrate_mae_mean(y_true, y_score, substrates)["mean"]
        return -v if not math.isnan(v) else float("nan")
    if metric == "ndcg":
        return _per_substrate_ndcg_mean(y_true, y_score, substrates)["mean"]
    raise ValueError(f"Unknown metric: {metric}")


def _evaluate_beta_grid_for_mode(
    metric: str,
    mode_df,
    betas: List[float],
) -> Tuple[float, float, Dict[float, float], float]:
    """Score one mode_df under each β; return (best_bnn, best_beta, per_beta, null_val)."""
    y_true = mode_df["log_fc"].values.astype(np.float64)
    y_pred = mode_df["y_pred"].values.astype(np.float64)
    # UCB acquisition σ (mode-consistent: excludes between-reference variance);
    # built by aggregate_and_null_per_fold per its acq_sigma. Was tot_std.
    acq_std = mode_df["acq_std"].values.astype(np.float64)
    subs = mode_df["substrate"].values
    pos = mode_df["position"].values
    null_pred = mode_df["null_pred"].values.astype(np.float64)

    per_beta: Dict[float, float] = {}
    best = -math.inf
    best_beta = float(betas[0])
    for beta in betas:
        score = y_pred + float(beta) * acq_std
        val = _eval_metric(metric, y_true, score, subs, pos)
        per_beta[float(beta)] = float(val) if not math.isnan(val) else float("nan")
        if not math.isnan(val) and val > best:
            best = float(val)
            best_beta = float(beta)
    null_val = _eval_metric(metric, y_true, null_pred, subs, pos)
    return best, best_beta, per_beta, float(null_val) if not math.isnan(null_val) else float("nan")


# ═══════════════════════════════════════════════════════════════════════════
# Full-param builder (mirrors new_05_bnn2_train.resolve_all_params, driven
# by Optuna trial suggestions rather than argparse)
# ═══════════════════════════════════════════════════════════════════════════

def build_trial_params(
    trial,
    config: dict,
    narrow_map: Optional[Dict[str, dict]],
) -> Tuple[dict, dict]:
    """Return (params, trial_config).

    params matches what new_05_bnn2_train.resolve_all_params produces, except
    the trial's categorical choices override the config defaults.
    trial_config is the config dict with the pairwise block patched to
    reflect the trial's `exclude_self_ref` choice (read by expand_to_pairwise).
    """
    bnn2 = config["bnn2"]
    train_cfg = bnn2["training"]
    preproc = config["preprocessing"]
    pairwise_cfg_raw = bnn2.get("pairwise", {})
    features_cfg_raw = bnn2.get("features", {})
    lds_cfg_raw = bnn2.get("lds", {})

    # Model architecture
    hidden_dims = search_or_fixed(trial, "hidden_dims", bnn2["hidden_dims"], narrow_map=narrow_map)
    prior_std = search_or_fixed(trial, "prior_std", bnn2["prior_std"], narrow_map=narrow_map)
    dropout_rate = search_or_fixed(trial, "dropout_rate", bnn2["dropout_rate"], narrow_map=narrow_map)
    activation = search_or_fixed(trial, "activation", bnn2["activation"], narrow_map=narrow_map)
    x_aa_freeze = search_or_fixed(trial, "x_aa_freeze", bnn2["x_aa_freeze"], narrow_map=narrow_map)
    substrate_embedding_type = search_or_fixed(
        trial, "substrate_embedding_type", bnn2["substrate_embedding_type"], narrow_map=narrow_map)

    # Training
    learning_rate = search_or_fixed(trial, "learning_rate", train_cfg["learning_rate"], narrow_map=narrow_map)
    kl_weight = search_or_fixed(trial, "kl_weight", train_cfg.get("kl_weight", {"value": 1.0}), narrow_map=narrow_map)
    batch_size = search_or_fixed(trial, "batch_size", train_cfg.get("batch_size", {"value": 32}), narrow_map=narrow_map)
    kl_anneal_epochs = search_or_fixed(trial, "kl_anneal_epochs", train_cfg.get("kl_anneal_epochs", {"value": 30}), narrow_map=narrow_map)
    clip_grad_norm = search_or_fixed(trial, "clip_grad_norm", train_cfg.get("clip_grad_norm", {"value": 5.0}), narrow_map=narrow_map)

    # Loss & regularization
    loss_type = search_or_fixed(trial, "loss_type",
                                bnn2.get("loss_type", {"value": "gaussian_nll"}),
                                narrow_map=narrow_map)
    null_reg_weight = search_or_fixed(trial, "null_reg_weight",
                                       train_cfg.get("null_reg_weight", {"value": 0.0}),
                                       narrow_map=narrow_map)
    log_var_floor = search_or_fixed(trial, "log_var_floor",
                                     train_cfg.get("log_var_floor", {"value": None}),
                                     narrow_map=narrow_map)

    # LDS
    use_lds = search_or_fixed(trial, "use_lds",
                               lds_cfg_raw.get("use_lds", {"value": False}),
                               narrow_map=narrow_map)
    lds_cfg = {
        "use_lds": use_lds,
        "n_bins": lds_cfg_raw.get("n_bins", 50),
        "kernel_size": lds_cfg_raw.get("kernel_size", 5),
        "sigma": lds_cfg_raw.get("sigma", 2.0),
    }

    # Features
    features = {}
    for feat_name in ["fc_ref", "ref_distance", "x_target_substrate",
                       "x_ref_substrate", "x_aa", "esm_wt", "esm_mut", "saprot_zs"]:
        default_cfg = {"value": True} if feat_name != "x_aa" else {"value": False}
        feat_cfg = features_cfg_raw.get(feat_name, default_cfg)
        features[feat_name] = search_or_fixed(
            trial, f"feat_{feat_name}", feat_cfg, narrow_map=narrow_map)

    # Pairwise (opt side — the aggregation mode is fixed at the wrapper level,
    # but exclude_self_ref affects expand_to_pairwise)
    exclude_self_ref = search_or_fixed(
        trial, "exclude_self_ref",
        pairwise_cfg_raw.get("exclude_self_ref", {"value": True}),
        narrow_map=narrow_map)
    distance_weight_temperature = resolve_param(
        pairwise_cfg_raw.get("distance_weight_temperature", {"value": 1.0}))

    # Preprocessing
    x_sub_scaler = search_or_fixed(
        trial, "x_substrate_scaler",
        preproc.get("x_substrate", {}).get("scaler", "none"),
        narrow_map=narrow_map)
    x_sub_pca = search_or_fixed(
        trial, "x_substrate_pca",
        preproc.get("x_substrate", {}).get("pca", None),
        is_pca=True, narrow_map=narrow_map)
    saprot_scaler = search_or_fixed(
        trial, "saprot_zs_scaler",
        preproc.get("saprot_zs", {}).get("scaler", "none"),
        narrow_map=narrow_map)
    esm_wt_scaler = search_or_fixed(
        trial, "esm_wt_scaler",
        preproc.get("esm_wt", {}).get("scaler", "standard"),
        narrow_map=narrow_map)
    esm_wt_pca = search_or_fixed(
        trial, "esm_wt_pca",
        preproc.get("esm_wt", {}).get("pca", {"value": 0.99}),
        is_pca=True, narrow_map=narrow_map)
    esm_mut_scaler = search_or_fixed(
        trial, "esm_mut_scaler",
        preproc.get("esm_mut", {}).get("scaler", "standard"),
        narrow_map=narrow_map)
    esm_mut_pca = search_or_fixed(
        trial, "esm_mut_pca",
        preproc.get("esm_mut", {}).get("pca", {"value": 0.99}),
        is_pca=True, narrow_map=narrow_map)

    params = {
        # Model
        "hidden_dims": hidden_dims,
        "prior_std": prior_std,
        "dropout_rate": dropout_rate,
        "activation": activation,
        "x_aa_freeze": x_aa_freeze,
        "substrate_embedding_type": substrate_embedding_type,
        # Training
        "learning_rate": learning_rate,
        "kl_weight": kl_weight,
        "batch_size": batch_size,
        "kl_anneal_epochs": kl_anneal_epochs,
        "n_epochs": resolve_param(train_cfg["n_epochs"]),
        "early_stopping_patience": resolve_param(train_cfg["early_stopping_patience"]),
        "n_inference_samples": resolve_param(train_cfg["n_inference_samples"]),
        "clip_grad_norm": clip_grad_norm,
        # Loss & regularization
        "loss_type": loss_type,
        "null_reg_weight": null_reg_weight,
        "log_var_floor": log_var_floor,
        "prediction_floor": resolve_param(train_cfg.get("prediction_floor", {"value": None})),
        # LDS
        "lds": lds_cfg,
        # Features
        "features": features,
        # Pairwise
        "distance_weight_temperature": distance_weight_temperature,
        # Preprocessing
        "x_substrate_scaler": x_sub_scaler,
        "x_substrate_pca": x_sub_pca,
        "saprot_zs_scaler": saprot_scaler,
        "esm_wt_scaler": esm_wt_scaler,
        "esm_wt_pca": esm_wt_pca,
        "esm_mut_scaler": esm_mut_scaler,
        "esm_mut_pca": esm_mut_pca,
        # Hurdle (fixed; only consulted when loss_type == "hurdle")
        "hurdle": resolve_config_block(bnn2.get("hurdle", {})),
    }

    # Config with the trial's exclude_self_ref patched in so expand_to_pairwise sees it
    trial_config = copy.deepcopy(config)
    trial_config.setdefault("bnn2", {}).setdefault("pairwise", {})
    trial_config["bnn2"]["pairwise"]["exclude_self_ref"] = exclude_self_ref
    # Keep the remaining pairwise entries intact
    for k, v in pairwise_cfg_raw.items():
        if k != "exclude_self_ref":
            trial_config["bnn2"]["pairwise"].setdefault(k, v)

    return params, trial_config


# ═══════════════════════════════════════════════════════════════════════════
# Objective
# ═══════════════════════════════════════════════════════════════════════════

def _log_trial_panel(
    trial_number: int,
    objective_metric: str,
    objective_value: float,
    raw_objective: float,
    collapse_penalty: float,
    mean_collapse: float,
    best_mode: str,
    best_beta: float,
    best_null: float,
    delta: float,
    median_best_epoch: int,
    per_fold_best_epoch: list,
    effective_modes: list,
    ucb_betas: list,
    per_mode_results: Dict[str, dict],
):
    """Emit a multi-line box panel summarising a completed trial.

    Layout:
      - header: trial number + overall objective
      - winner: mode / β / Δ vs null / n_rows
      - per-mode table: BNN score at each β (winning β marked ✓),
                         null score, winner mark
      - training meta: median best epoch, per-fold epochs, collapse
    """
    # Column widths: mode name, each β in the grid, null, n_rows
    beta_cols = [f"β={b:g}" for b in ucb_betas]
    header_cols = ["mode"] + beta_cols + ["null", "n"]
    widths = [max(10, max(len(m) for m in effective_modes) + 1)]
    widths += [max(7, len(c)) for c in beta_cols]
    widths += [7, 5]

    def _fmt_row(cells):
        return "│   " + " ".join(f"{c:>{w}}" for c, w in zip(cells, widths))

    def _fmt_val(v, width):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "—"
        if isinstance(v, int):
            return f"{v}"
        return f"{v:.4f}"

    logger.info("┌─── TRIAL %d POST-PROCESSING ────────────────────────────┐", trial_number)
    logger.info("│ Objective (%s):  %.4f  (raw=%.4f − collapse penalty=%.2f)",
                objective_metric, objective_value, raw_objective, collapse_penalty)
    logger.info("│ Winner:  mode=%s  β=%.2f  null=%s  Δ=%s",
                best_mode, best_beta,
                f"{best_null:.4f}" if not math.isnan(best_null) else "—",
                f"{delta:+.4f}" if not math.isnan(delta) else "—")
    logger.info("│ Epochs:  median_best=%d  per_fold=%s",
                median_best_epoch,
                ", ".join(str(e) for e in per_fold_best_epoch) or "—")
    logger.info("│ Collapse score: %.3f  (penalty triggers at > %.2f)",
                mean_collapse, _COLLAPSE_THRESHOLD)
    logger.info("│")
    logger.info("│ Per-mode breakdown (BNN at each β, then matched null):")
    # Column header row
    header = _fmt_row(header_cols)
    logger.info(header)
    logger.info("│   " + "─" * (len(header) - 4))
    # One row per mode
    for mode_name in effective_modes:
        row = per_mode_results.get(mode_name, {})
        status = row.get("status", "missing")
        if status != "ok":
            cells = [mode_name] + ["—"] * len(beta_cols) + ["—", str(row.get("n_rows", 0))]
            logger.info(_fmt_row([f"{c:<{widths[0]}}" if i == 0 else c
                                    for i, c in enumerate(cells)]))
            continue
        per_beta = row.get("per_beta", {}) or {}
        row_cells = [f"{mode_name:<{widths[0]}}"]
        mode_best_beta = row.get("best_beta")
        for b, bcol in zip(ucb_betas, beta_cols):
            val = per_beta.get(str(float(b)))
            if val is None:
                val = per_beta.get(f"{float(b)}")
            val_str = _fmt_val(val, widths[len(row_cells)])
            # Mark the winning β within this mode
            if mode_best_beta is not None and abs(float(b) - float(mode_best_beta)) < 1e-9:
                val_str = "*" + val_str
            row_cells.append(val_str)
        row_cells.append(_fmt_val(row.get("null"), widths[-2]))
        row_cells.append(str(row.get("n_rows", 0)))
        line = _fmt_row(row_cells)
        # Append a trailing marker on the globally-winning row
        if mode_name == best_mode:
            line = line + "  ← winner"
        logger.info(line)
    logger.info("│   (* marks each mode's internal best β; ← marks the global winner)")
    logger.info("└─────────────────────────────────────────────────────────┘")


def create_objective(
    df,
    embeddings,
    bnn1_context,   # dict with keys: hidden, input_dim, latent_dim, pipe_wt, pipe_mut, enabled
    substrate_meta,
    config,
    device,
    scope: str,
    reference_modes: List[str],
    objective_metric: str,
    ucb_betas: List[float],
    null_emb: str,
    null_metric: str,
    narrow_map: Optional[Dict[str, dict]],
    acq_sigma: str = "within_epi_ale",
):
    """Return the Optuna objective function (maximise direction).

    Every trial scores predictions under each reference_mode × β combination
    and returns the max. The winning (mode, β) pair is recorded per trial.
    """
    target_substrate = FORMALDEHYDE_SUBSTRATE if scope == "formaldehyde" else None
    # formaldehyde reference mode is invalid when holding out Formaldehyde
    # (aggregator drops self-reference rows → empty mode_df). Skip silently.
    effective_modes = [m for m in reference_modes
                        if not (scope == "formaldehyde" and m == "formaldehyde")]
    if not effective_modes:
        raise RuntimeError("All reference modes were filtered out by scope; "
                           "check --reference-modes and --scope combination.")

    def objective(trial) -> float:
        params, trial_config = build_trial_params(trial, config, narrow_map)

        if not params["features"].get("x_aa", False):
            bnn1_h = None; bnn1_in = 0; latent = 0
            pw = None; pm = None
        else:
            if not bnn1_context["enabled"]:
                # x_aa turned on by the trial but no BNN1 backbone is available
                trial.set_user_attr("status", "missing_bnn1")
                logger.warning("Trial %d: x_aa=True but BNN1 backbone unavailable — aborting trial",
                               trial.number)
                return -math.inf
            bnn1_h = bnn1_context["hidden"]
            bnn1_in = bnn1_context["input_dim"]
            latent = bnn1_context["latent_dim"]
            pw = bnn1_context["pipe_wt"]; pm = bnn1_context["pipe_mut"]

        # ── Train CV folds ──
        try:
            pairwise_df, train_lookup_df, fold_summaries, fold_histories = \
                run_cv_and_collect_predictions(
                    df=df,
                    embeddings=embeddings,
                    bnn1_hidden=bnn1_h, bnn1_input_dim=bnn1_in, latent_dim=latent,
                    bnn1_pipe_wt=pw, bnn1_pipe_mut=pm,
                    params=params, config=trial_config, device=device,
                    split_type="substrate",
                    substrate_meta=substrate_meta,
                    target_substrate=target_substrate,
                    subsample_train_substrates=None,
                    null_embedding_type=null_emb,
                    null_distance_metric=null_metric,
                )
        except Exception as e:
            trial.set_user_attr("status", f"train_error: {type(e).__name__}: {e}")
            logger.exception("Trial %d training failed: %s", trial.number, e)
            return -math.inf

        if len(pairwise_df) == 0:
            trial.set_user_attr("status", "empty_pairwise")
            return -math.inf

        # ── For every reference mode, aggregate + run β-grid; keep the winner ──
        per_mode_results: Dict[str, dict] = {}
        best_bnn = -math.inf
        best_beta = float(ucb_betas[0])
        best_mode = effective_modes[0]
        best_null_for_best_mode = float("nan")

        for mode_name in effective_modes:
            mode_df = aggregate_and_null_per_fold(
                mode=mode_name,
                pairwise_df=pairwise_df,
                train_lookup_df=train_lookup_df,
                embeddings=embeddings,
                substrate_embedding_type=null_emb,
                distance_metric=null_metric,
                distance_weight_temperature=float(params.get("distance_weight_temperature", 1.0)),
                acq_sigma=acq_sigma,
            )
            if len(mode_df) == 0:
                per_mode_results[mode_name] = {
                    "status": "empty", "best_bnn": float("nan"), "best_beta": None,
                    "per_beta": {}, "null": float("nan"), "n_rows": 0,
                }
                continue

            mode_best, mode_beta, mode_per_beta, mode_null = \
                _evaluate_beta_grid_for_mode(objective_metric, mode_df, ucb_betas)
            per_mode_results[mode_name] = {
                "status": "ok",
                "best_bnn": mode_best,
                "best_beta": mode_beta,
                "per_beta": {str(k): v for k, v in mode_per_beta.items()},
                "null": mode_null,
                "n_rows": int(len(mode_df)),
            }
            if not math.isnan(mode_best) and mode_best > best_bnn:
                best_bnn = mode_best
                best_beta = mode_beta
                best_mode = mode_name
                best_null_for_best_mode = mode_null

        if math.isinf(best_bnn):
            trial.set_user_attr("status", "all_modes_empty")
            return -math.inf
        null_val = best_null_for_best_mode

        # ── Median best epoch across folds (for follow-up training) ──
        best_epochs = [getattr(h, "best_epoch", None) for h in fold_histories]
        best_epochs = [e for e in best_epochs if e is not None]
        median_best_epoch = int(np.median(best_epochs)) if best_epochs else params["n_epochs"]

        # ── Collapse penalty ──
        collapse_scores = [
            f["metrics"].get("posterior_collapse_score", 1.0) for f in fold_summaries
        ]
        mean_collapse = float(np.nanmean(collapse_scores)) if collapse_scores else 1.0
        # subtract (not add) — direction is maximise, and we want collapse to hurt
        collapse_penalty = _COLLAPSE_PENALTY if mean_collapse > _COLLAPSE_THRESHOLD else 0.0

        if math.isnan(best_bnn):
            trial.set_user_attr("status", "nan_pred")
            return -math.inf

        raw_objective = best_bnn
        objective_value = raw_objective - collapse_penalty
        delta = raw_objective - null_val if not math.isnan(null_val) else float("nan")

        trial.set_user_attr("status", "ok")
        trial.set_user_attr("raw_objective", raw_objective)
        trial.set_user_attr("null_objective", float(null_val) if not math.isnan(null_val) else None)
        trial.set_user_attr("delta", float(delta) if not math.isnan(delta) else None)
        trial.set_user_attr("best_beta", best_beta)
        trial.set_user_attr("best_reference_mode", best_mode)
        trial.set_user_attr("per_mode_results", per_mode_results)
        # Convenience: β-grid for the winning mode (what older plots expect)
        trial.set_user_attr("per_beta_objective",
                            per_mode_results[best_mode].get("per_beta", {}))
        trial.set_user_attr("median_best_epoch", median_best_epoch)
        trial.set_user_attr("per_fold_best_epoch", best_epochs)
        trial.set_user_attr("mean_collapse_score", mean_collapse)
        trial.set_user_attr("collapse_penalty", collapse_penalty)
        trial.set_user_attr("n_rows_scored", per_mode_results[best_mode].get("n_rows", 0))

        _log_trial_panel(
            trial_number=trial.number,
            objective_metric=objective_metric,
            objective_value=float(objective_value),
            raw_objective=float(raw_objective),
            collapse_penalty=float(collapse_penalty),
            mean_collapse=float(mean_collapse),
            best_mode=best_mode,
            best_beta=float(best_beta),
            best_null=float(null_val) if not math.isnan(null_val) else float("nan"),
            delta=float(delta) if not math.isnan(delta) else float("nan"),
            median_best_epoch=int(median_best_epoch),
            per_fold_best_epoch=best_epochs,
            effective_modes=effective_modes,
            ucb_betas=ucb_betas,
            per_mode_results=per_mode_results,
        )
        return float(objective_value)

    return objective


# ═══════════════════════════════════════════════════════════════════════════
# Working-range export
# ═══════════════════════════════════════════════════════════════════════════

def _is_numeric_like(values) -> bool:
    """True when every observed value can be coerced to float (and none are bool)."""
    try:
        for v in values:
            if isinstance(v, bool):
                return False
            float(v)
        return True
    except (TypeError, ValueError):
        return False


def _is_list_like(values) -> bool:
    return all(isinstance(v, list) for v in values)


def export_working_range(
    study,
    scope: str,
    reference_mode: str,
    objective_metric: str,
    top_frac: float,
    min_good_trials: int,
    narrow_quantiles: Tuple[float, float] = (0.10, 0.90),
) -> dict:
    """Build the working_range.json payload from completed trials."""
    import optuna
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
                 and t.value is not None and not math.isinf(t.value)]
    if not completed:
        return {
            "source_study": study.study_name,
            "scope": scope,
            "reference_mode": reference_mode,
            "objective_metric": objective_metric,
            "n_trials_considered": 0,
            "n_good_trials": 0,
            "confident": False,
            "params": {},
        }

    # Rank by objective descending (study direction is maximise)
    completed.sort(key=lambda t: t.value, reverse=True)
    n_good = max(min_good_trials, int(round(top_frac * len(completed))))
    n_good = min(n_good, len(completed))
    good_trials = completed[:n_good]
    confident = len(good_trials) >= min_good_trials

    # Decode trial params (hidden_dims/x_*_pca are serialised as strings)
    def decode(raw: dict) -> dict:
        out = dict(raw)
        if "hidden_dims" in out:
            try: out["hidden_dims"] = json.loads(out["hidden_dims"])
            except Exception: pass
        for pca_key in ("x_substrate_pca", "esm_wt_pca", "esm_mut_pca"):
            if pca_key in out:
                out[pca_key] = parse_pca_value(out[pca_key])
        return out

    good_param_values: Dict[str, list] = {}
    for t in good_trials:
        for k, v in decode(t.params).items():
            good_param_values.setdefault(k, []).append(v)

    # All observed values across all completed trials (to compute "dropped")
    all_observed: Dict[str, list] = {}
    for t in completed:
        for k, v in decode(t.params).items():
            all_observed.setdefault(k, []).append(v)

    # Collect the Optuna BaseDistribution for each parameter (from any trial
    # where it appears; distributions are invariant within a study).
    from optuna.distributions import (
        FloatDistribution, IntDistribution, CategoricalDistribution,
    )
    param_distributions: Dict[str, object] = {}
    for t in completed:
        for name, dist in t.distributions.items():
            param_distributions.setdefault(name, dist)

    params_out: Dict[str, dict] = {}
    for name, good_vals in good_param_values.items():
        observed_vals = all_observed.get(name, [])
        dist_obj = param_distributions.get(name)

        # ── Continuous / int distribution → numeric_distribution entry ──
        if isinstance(dist_obj, (FloatDistribution, IntDistribution)):
            nums = np.array([float(v) for v in good_vals], dtype=float)
            dist_type = "int" if isinstance(dist_obj, IntDistribution) else (
                "loguniform" if getattr(dist_obj, "log", False) else "uniform")
            # Quantile-based narrowing (trims outlier good trials).
            # For loguniform: compute quantiles in log-space so the trim is
            # symmetric on a multiplicative scale.
            q_lo, q_hi = narrow_quantiles
            if dist_type == "loguniform":
                if (nums > 0).all():
                    logs = np.log(nums)
                    lo_q = float(np.exp(np.quantile(logs, q_lo)))
                    hi_q = float(np.exp(np.quantile(logs, q_hi)))
                    median_val = float(np.exp(np.median(logs)))
                else:
                    lo_q = float(np.quantile(nums, q_lo))
                    hi_q = float(np.quantile(nums, q_hi))
                    median_val = float(np.median(nums))
            else:
                lo_q = float(np.quantile(nums, q_lo))
                hi_q = float(np.quantile(nums, q_hi))
                median_val = float(np.median(nums))
            # Guard: if the quantile trim collapsed the range (all values equal),
            # fall back to raw min/max so we still emit a usable interval.
            if lo_q == hi_q:
                lo_q = float(nums.min())
                hi_q = float(nums.max())
            # Guard: stay within the original distribution bounds.
            lo_q = max(lo_q, float(dist_obj.low))
            hi_q = min(hi_q, float(dist_obj.high))
            params_out[name] = {
                "kind": "numeric_distribution",
                "type": dist_type,
                "orig_low": float(dist_obj.low),
                "orig_high": float(dist_obj.high),
                "min_good": lo_q,
                "max_good": hi_q,
                "median_good": median_val,
                "raw_min_good": float(nums.min()),
                "raw_max_good": float(nums.max()),
                "narrow_quantiles": [float(q_lo), float(q_hi)],
                "n_good": int(nums.size),
                "confident": confident,
            }
            continue

        # Frequency counter (hashable keys)
        freq = {}
        for v in good_vals:
            freq[_value_to_key(v)] = freq.get(_value_to_key(v), 0) + 1

        if _is_list_like(good_vals + observed_vals):
            unique_good = []
            seen = set()
            for v in good_vals:
                k = _value_to_key(v)
                if k not in seen:
                    seen.add(k); unique_good.append(v)
            params_out[name] = {
                "kind": "list",
                "good_values": unique_good,
                "value_frequencies": freq,
                "confident": confident,
            }
            continue

        if _is_numeric_like(good_vals):
            nums = [float(v) for v in good_vals]
            obs_nums = [float(v) for v in observed_vals if not isinstance(v, bool)]
            unique_obs = sorted(set(obs_nums))
            params_out[name] = {
                "kind": "numeric",
                "min_good": float(min(nums)),
                "max_good": float(max(nums)),
                "median_good": float(np.median(nums)),
                "observed_values": unique_obs,
                "value_frequencies": freq,
                "confident": confident,
            }
            continue

        # Plain categorical
        good_unique = []
        seen = set()
        for v in good_vals:
            k = _value_to_key(v)
            if k not in seen:
                seen.add(k); good_unique.append(v)
        params_out[name] = {
            "kind": "categorical",
            "good_values": good_unique,
            "value_frequencies": freq,
            "confident": confident,
        }

    return {
        "source_study": study.study_name,
        "scope": scope,
        "reference_mode": reference_mode,
        "objective_metric": objective_metric,
        "n_trials_considered": len(completed),
        "n_good_trials": len(good_trials),
        "good_trial_cutoff_objective": float(good_trials[-1].value),
        "confident": confident,
        "params": params_out,
    }


def write_narrowed_config_yaml(
    working_range: dict,
    out_path: Path,
    original_config: dict,
):
    """Emit a drop-in YAML with narrowed `search:` entries, annotated."""
    # Only the bnn2 / preprocessing blocks are searchable in config.yaml.
    # We mirror the legacy structure so the user can splice entries in directly.
    lines = ["# Auto-generated from new_opt_05.py — drop into config.yaml",
             f"# Source study: {working_range.get('source_study')}",
             f"# Scope: {working_range.get('scope')}  "
             f"reference_mode: {working_range.get('reference_mode')}  "
             f"metric: {working_range.get('objective_metric')}",
             f"# Good trials: {working_range.get('n_good_trials')}/{working_range.get('n_trials_considered')}"
             f"  confident={working_range.get('confident')}",
             ""]

    def fmt_val(v):
        if isinstance(v, str):
            return f"\"{v}\""
        if v is None:
            return "null"
        if isinstance(v, bool):
            return "true" if v else "false"
        return json.dumps(v)

    def fmt_num(x):
        # Short repr for floats; engineering-looking floats stay compact
        if isinstance(x, int):
            return str(x)
        try:
            fx = float(x)
        except Exception:
            return repr(x)
        if fx == int(fx) and abs(fx) < 1e6:
            return repr(fx)
        return f"{fx:.6g}"

    params = working_range.get("params", {})
    for name in sorted(params.keys()):
        entry = params[name]
        kind = entry.get("kind")

        if kind == "numeric_distribution":
            # Emit dict-form distribution narrowed to the good range.
            t = entry.get("type", "uniform")
            lo = entry.get("min_good")
            hi = entry.get("max_good")
            # Guard against zero-width ranges (single observed value): nudge.
            if lo is not None and hi is not None and lo == hi:
                if t == "int":
                    hi = int(hi) + 1
                else:
                    pad = max(abs(float(lo)), 1e-9) * 0.01
                    lo = float(lo) - pad
                    hi = float(hi) + pad
            if t == "int":
                # Round good-range bounds to ints; preserve original bounds in comment.
                lo_out = int(round(float(lo)))
                hi_out = int(round(float(hi)))
                orig_lo_s = str(int(round(float(entry.get("orig_low")))))
                orig_hi_s = str(int(round(float(entry.get("orig_high")))))
                median_s = f"{entry.get('median_good'):g}"
            else:
                lo_out = fmt_num(lo)
                hi_out = fmt_num(hi)
                orig_lo_s = fmt_num(entry.get("orig_low"))
                orig_hi_s = fmt_num(entry.get("orig_high"))
                median_s = fmt_num(entry.get("median_good"))
            lines.append(f"{name}:")
            lines.append(f"  search:")
            lines.append(f"    type: {t}")
            lines.append(f"    low: {lo_out}")
            lines.append(f"    high: {hi_out}")
            lines.append(
                f"  # orig={orig_lo_s}..{orig_hi_s}"
                f"  median_good={median_s}"
                f"  n_good={entry.get('n_good')}"
                f"  confident={entry.get('confident')}"
            )
            lines.append("")
            continue

        if kind == "numeric":
            values = entry.get("observed_values", [])
            kept = [v for v in values
                    if entry.get("min_good") is not None and
                    entry.get("max_good") is not None and
                    entry["min_good"] <= float(v) <= entry["max_good"]]
            if not kept:
                kept = values
            comment = f"# min={entry['min_good']}  max={entry['max_good']}  median={entry['median_good']}  freq={entry['value_frequencies']}"
        else:
            kept = entry.get("good_values", [])
            comment = f"# freq={entry['value_frequencies']}"
        lines.append(f"{name}:")
        lines.append(f"  search: [{', '.join(fmt_val(v) for v in kept)}]  {comment}")
        lines.append("")

    out_path.write_text("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_objective_trace(trial_results: list, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    completed = [(t["number"], t["value"]) for t in trial_results
                 if t["value"] is not None and not math.isinf(t["value"])]
    if not completed:
        plt.close(fig); return
    xs, ys = zip(*completed)
    ax.plot(xs, ys, ".", alpha=0.5, label="trial value")
    running = np.maximum.accumulate(ys)
    ax.plot(xs, running, "-", color="#d62728", label="best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Objective (maximize)")
    ax.set_title("Optuna trial history")
    ax.legend()
    ax.grid(ls="--", lw=0.3, alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_objective_vs_delta(trial_results: list, out_path: Path):
    pairs = [(t["null_objective"], t["raw_objective"])
             for t in trial_results
             if t["raw_objective"] is not None and t["null_objective"] is not None]
    if not pairs:
        return
    null_vals, raw_vals = zip(*pairs)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(null_vals, raw_vals, alpha=0.5)
    lo = min(min(null_vals), min(raw_vals))
    hi = max(max(null_vals), max(raw_vals))
    ax.plot([lo, hi], [lo, hi], "--", color="#888", label="BNN = null")
    ax.set_xlabel("Null objective")
    ax.set_ylabel("BNN objective")
    ax.set_title("BNN vs matched null per trial")
    ax.legend()
    ax.grid(ls="--", lw=0.3, alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_objective_vs_beta(best_trial_attrs: dict, out_path: Path):
    per_beta = best_trial_attrs.get("per_beta_objective") or {}
    if not per_beta:
        return
    betas = sorted((float(b), v) for b, v in per_beta.items())
    xs, ys = zip(*betas)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, "o-")
    ax.set_xlabel("β (UCB weight on total_std)")
    ax.set_ylabel("Objective")
    ax.set_title("Best trial: objective vs UCB β")
    ax.grid(ls="--", lw=0.3, alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_param_importance(study, out_path: Path):
    try:
        import optuna
        fig = optuna.visualization.matplotlib.plot_param_importances(study).figure
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        logger.warning("Skipping param-importance plot: %s", e)


def plot_working_range_histograms(
    working_range: dict,
    trial_results: list,
    out_path: Path,
):
    params = working_range.get("params", {})
    if not params:
        return
    names = sorted(params.keys())
    ncols = 3
    nrows = math.ceil(len(names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                              squeeze=False)
    for i, name in enumerate(names):
        ax = axes[i // ncols][i % ncols]
        entry = params[name]
        freqs = entry.get("value_frequencies", {})
        if not freqs:
            ax.axis("off"); continue
        labels = list(freqs.keys())
        counts = [freqs[k] for k in labels]
        ax.bar(range(len(labels)), counts)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([str(lab) for lab in labels], rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{name} ({entry['kind']})", fontsize=9)
        ax.grid(axis="y", ls="--", lw=0.3, alpha=0.5)
    for j in range(len(names), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# CLI / main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna hyperopt for BNN2 targeting per-position top-1 recovery.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scope", type=str, default="all-substrates",
                        choices=["all-substrates", "formaldehyde"])
    parser.add_argument("--reference-modes", type=str, nargs="+",
                        default=list(ALL_REFERENCE_MODES),
                        choices=list(ALL_REFERENCE_MODES),
                        help="Reference modes to evaluate per trial. The trial's "
                             "objective is max over (mode, β); the winning mode + β "
                             "are recorded. Default: all four matched-pair modes.")
    parser.add_argument("--objective-metric", type=str, default="per_position_top1_recovery",
                        choices=list(ALL_METRICS))
    parser.add_argument("--ucb-betas", type=str, default="[0.0, 0.5, 1.0, 1.5, 2.0]",
                        help="JSON list of β values to try per trial (UCB = μ + β·σ)")
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None,
                        help="SQLite path for resumable studies")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing study DB and start fresh")
    parser.add_argument("--acq-sigma", type=str, default="within_epi",
                        choices=["within_epi_ale", "within_epi", "total"],
                        help="UCB acquisition σ for the β-grid objective. "
                             "'within_*' drop the between-reference variance so β "
                             "is mode-consistent; 'total' = old tot_std behavior. "
                             "Must match what new_05b_bnn2_score.py uses at CV time.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--bnn1-model-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="Override cv.seed for sampler")
    parser.add_argument("--top-frac", type=float, default=0.2,
                        help="Fraction of trials (by objective rank) treated as "
                             "'good' for working-range export (default 0.2)")
    parser.add_argument("--min-good-trials", type=int, default=5)
    parser.add_argument("--narrow-quantiles", type=float, nargs=2,
                        default=[0.10, 0.90], metavar=("LOW", "HIGH"),
                        help="Lower/upper quantile of good-trial values used as "
                             "narrowed distribution bounds (default 0.10 0.90). "
                             "Tighten (e.g. 0.25 0.75) for more aggressive narrowing; "
                             "loosen (0.0 1.0) = raw min/max.")
    parser.add_argument("--narrow-from", type=str, default=None,
                        help="Path to a prior working_range.json to intersect with")
    return parser.parse_args()


def _default_output_dir(args) -> Path:
    modes_tag = ("all" if sorted(args.reference_modes) == sorted(ALL_REFERENCE_MODES)
                 else "+".join(sorted(args.reference_modes)))
    name = f"{args.scope}_{modes_tag}_{args.objective_metric}"
    return PROJECT_ROOT / "results" / "new_opt_05" / name


def main():
    args = parse_args()
    t_start = time.time()

    # Warn about the scope/reference-modes combo that gets auto-skipped
    if args.scope == "formaldehyde" and "formaldehyde" in args.reference_modes:
        logger.info("scope=formaldehyde → the 'formaldehyde' reference mode will be "
                    "skipped per trial (self-reference filter drops all rows).")

    try:
        ucb_betas = json.loads(args.ucb_betas)
        ucb_betas = [float(b) for b in ucb_betas]
    except Exception as e:
        raise SystemExit(f"--ucb-betas must be a JSON list of floats: {e}")
    if not ucb_betas:
        raise SystemExit("--ucb-betas must contain at least one β (try '[0.0]')")

    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir / "run.log")
    logger.info("=" * 60)
    logger.info("new_opt_05.py — scope=%s reference_modes=%s metric=%s",
                args.scope, args.reference_modes, args.objective_metric)
    logger.info("UCB β grid: %s", ucb_betas)
    logger.info("Output dir: %s", output_dir)
    logger.info("=" * 60)

    config = load_config(args.config)
    if args.seed is not None:
        config.setdefault("cv", {})["seed"] = int(args.seed)
    device = get_device(config, args.device)

    # Data
    processed_dir = PROJECT_ROOT / config["data"]["output_dir"]
    df = load_multi_substrate_data(processed_dir)
    embeddings = load_all_embeddings(processed_dir)
    substrate_meta = load_substrate_metadata(processed_dir)

    # Null / distance metric selection (same as new_05 train does)
    metric_selection = select_best_substrate_metric(df, embeddings)
    null_emb = metric_selection["best_embedding"]
    null_metric = metric_selection["best_metric"]
    logger.info("Null-mode substrate metric: %s / %s (ρ=%.3f vs functional distance)",
                null_emb, null_metric, metric_selection["best_correlation"])

    # BNN1 backbone (only needed when the trial turns x_aa on). Load lazily:
    # if the checkpoint exists, we're good; if it doesn't we still run, and
    # trials that try x_aa=True will fail cleanly.
    bnn1_dir = Path(args.bnn1_model_dir) if args.bnn1_model_dir else (
        PROJECT_ROOT / "results" / "03_formaldehyde_regression" / "models")
    bnn1_context = {"enabled": False, "hidden": None, "input_dim": 0,
                     "latent_dim": 0, "pipe_wt": None, "pipe_mut": None}
    if (bnn1_dir / "final_model.pt").exists():
        try:
            h, in_dim, lat, _ = load_bnn1_backbone(bnn1_dir, device)
            pw, pm = load_bnn1_preprocessing(bnn1_dir)
            bnn1_context.update({"enabled": True, "hidden": h, "input_dim": in_dim,
                                  "latent_dim": lat, "pipe_wt": pw, "pipe_mut": pm})
            logger.info("BNN1 backbone loaded from %s (x_aa trials enabled)", bnn1_dir)
        except Exception as e:
            logger.warning("BNN1 backbone present but load failed (%s) — x_aa trials will abort",
                           e)
    else:
        logger.info("No BNN1 backbone at %s — trials with x_aa=True will abort cleanly",
                    bnn1_dir)

    # Narrow-from
    narrow_map: Optional[Dict[str, dict]] = None
    if args.narrow_from:
        with open(args.narrow_from) as f:
            narrow_payload = json.load(f)
        narrow_map = narrow_payload.get("params", {})
        logger.info("Loaded narrow-from: %d params restricted from %s",
                    len(narrow_map), args.narrow_from)

    # Optuna setup
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    modes_tag = ("all" if sorted(args.reference_modes) == sorted(ALL_REFERENCE_MODES)
                 else "+".join(sorted(args.reference_modes)))
    study_name = args.study_name or f"new_opt_05__{args.scope}__{modes_tag}__{args.objective_metric}"
    if args.storage:
        storage_url = args.storage
        db_path = Path(args.storage.replace("sqlite:///", ""))
    else:
        db_path = output_dir / "study.sqlite"
        storage_url = f"sqlite:///{db_path}"
    if args.fresh and db_path.exists():
        db_path.unlink()
        logger.info("Deleted %s (--fresh)", db_path)

    n_trials_total = args.n_trials or config["cv"].get("n_hyperopt_trials", 50)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config["cv"]["seed"]),
        load_if_exists=True,
    )

    n_completed = len([t for t in study.trials
                       if t.state == optuna.trial.TrialState.COMPLETE])
    n_remaining = max(0, n_trials_total - n_completed)
    logger.info("Study '%s': %d/%d already complete, %d remaining",
                study_name, n_completed, n_trials_total, n_remaining)

    if n_remaining > 0:
        objective = create_objective(
            df=df, embeddings=embeddings, bnn1_context=bnn1_context,
            substrate_meta=substrate_meta, config=config, device=device,
            scope=args.scope,
            reference_modes=args.reference_modes,
            objective_metric=args.objective_metric,
            ucb_betas=ucb_betas,
            null_emb=null_emb, null_metric=null_metric,
            narrow_map=narrow_map,
            acq_sigma=args.acq_sigma,
        )
        study.optimize(objective, n_trials=n_remaining,
                       show_progress_bar=True)

    # ─── Post-process ───
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE
                 and t.value is not None and not math.isinf(t.value)]
    if not completed:
        logger.error("No successful trials — nothing to export.")
        return

    best_trial = study.best_trial
    logger.info("Best trial #%d: objective=%.4f  mode=%s  β=%s  median_ep=%s",
                best_trial.number, best_trial.value,
                best_trial.user_attrs.get("best_reference_mode"),
                best_trial.user_attrs.get("best_beta"),
                best_trial.user_attrs.get("median_best_epoch"))

    # ── Full params for the best trial ──
    # We don't call build_trial_params for the best trial directly (no Optuna
    # context), so reconstruct from raw params + config defaults:
    best_raw = dict(best_trial.params)
    if "hidden_dims" in best_raw:
        try: best_raw["hidden_dims"] = json.loads(best_raw["hidden_dims"])
        except Exception: pass
    for pca_key in ("x_substrate_pca", "esm_wt_pca", "esm_mut_pca"):
        if pca_key in best_raw:
            best_raw[pca_key] = parse_pca_value(best_raw[pca_key])

    # Persist the searched params plus the median best epoch (so the follow-up
    # training run is capped at a sensible budget).
    best_hyperparams = dict(best_raw)
    best_hyperparams["n_epochs"] = int(best_trial.user_attrs.get("median_best_epoch")
                                        or resolve_param(config["bnn2"]["training"]["n_epochs"]))
    best_hyperparams["best_beta"] = best_trial.user_attrs.get("best_beta")
    best_hyperparams["best_reference_mode"] = best_trial.user_attrs.get("best_reference_mode")
    best_hyperparams["objective_metric"] = args.objective_metric
    best_hyperparams["acq_sigma"] = args.acq_sigma
    best_hyperparams["objective_value"] = float(best_trial.value)
    best_hyperparams["raw_objective"] = best_trial.user_attrs.get("raw_objective")
    best_hyperparams["null_objective"] = best_trial.user_attrs.get("null_objective")
    best_hyperparams["delta"] = best_trial.user_attrs.get("delta")
    best_hyperparams["scope"] = args.scope
    best_hyperparams["reference_modes_evaluated"] = list(args.reference_modes)
    best_hyperparams["trial_number"] = best_trial.number
    best_hyperparams["per_beta_objective"] = best_trial.user_attrs.get("per_beta_objective")
    best_hyperparams["per_mode_results"] = best_trial.user_attrs.get("per_mode_results")

    with open(output_dir / "best_hyperparams.json", "w") as f:
        json.dump(best_hyperparams, f, indent=2, default=str)

    # ── Full study dump ──
    trial_results = []
    for t in study.trials:
        raw = dict(t.params)
        if "hidden_dims" in raw:
            try: raw["hidden_dims"] = json.loads(raw["hidden_dims"])
            except Exception: pass
        for pca_key in ("x_substrate_pca", "esm_wt_pca", "esm_mut_pca"):
            if pca_key in raw:
                raw[pca_key] = parse_pca_value(raw[pca_key])
        trial_results.append({
            "number": t.number,
            "state": str(t.state),
            "value": t.value,
            "raw_objective": t.user_attrs.get("raw_objective"),
            "null_objective": t.user_attrs.get("null_objective"),
            "delta": t.user_attrs.get("delta"),
            "best_beta": t.user_attrs.get("best_beta"),
            "best_reference_mode": t.user_attrs.get("best_reference_mode"),
            "per_beta_objective": t.user_attrs.get("per_beta_objective"),
            "per_mode_results": t.user_attrs.get("per_mode_results"),
            "median_best_epoch": t.user_attrs.get("median_best_epoch"),
            "per_fold_best_epoch": t.user_attrs.get("per_fold_best_epoch"),
            "mean_collapse_score": t.user_attrs.get("mean_collapse_score"),
            "collapse_penalty": t.user_attrs.get("collapse_penalty"),
            "status": t.user_attrs.get("status"),
            "params": raw,
        })
    with open(output_dir / "study_results.json", "w") as f:
        json.dump(trial_results, f, indent=2, default=str)

    # ── Best command ──
    target_flag = (f"--target-substrate {FORMALDEHYDE_SUBSTRATE} "
                   if args.scope == "formaldehyde" else "")
    rerun_train = (
        f"python new_05_bnn2_train.py --split substrate {target_flag}"
        f"--hyperparams {output_dir / 'best_hyperparams.json'}"
    )
    rerun_score = (
        f"python new_05b_bnn2_score.py "
        f"--run-dir results/new_05_bnn2/substrate/<run_id>"
    )
    with open(output_dir / "best_command.txt", "w") as f:
        f.write(rerun_train + "\n" + rerun_score + "\n")

    # ── Config snapshot ──
    with open(output_dir / "config_used.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # ── Working range ──
    # When multiple reference modes are evaluated per trial, record the set
    # (sorted) so downstream tooling can see what was considered.
    modes_tag = ("all" if sorted(args.reference_modes) == sorted(ALL_REFERENCE_MODES)
                 else "+".join(sorted(args.reference_modes)))
    q_lo, q_hi = float(args.narrow_quantiles[0]), float(args.narrow_quantiles[1])
    if not (0.0 <= q_lo < q_hi <= 1.0):
        raise SystemExit(f"--narrow-quantiles must satisfy 0 ≤ LOW < HIGH ≤ 1 "
                         f"(got {q_lo}, {q_hi})")
    working_range = export_working_range(
        study=study,
        scope=args.scope,
        reference_mode=modes_tag,
        objective_metric=args.objective_metric,
        top_frac=args.top_frac,
        min_good_trials=args.min_good_trials,
        narrow_quantiles=(q_lo, q_hi),
    )
    with open(output_dir / "working_range.json", "w") as f:
        json.dump(working_range, f, indent=2, default=str)
    write_narrowed_config_yaml(working_range,
                                output_dir / "narrowed_config_suggestion.yaml",
                                config)

    # ── Plots ──
    plot_objective_trace(trial_results, figures_dir / "objective_trace.png")
    plot_objective_vs_delta(trial_results, figures_dir / "pareto_objective_vs_delta.png")
    plot_objective_vs_beta(best_trial.user_attrs, figures_dir / "objective_vs_beta.png")
    try:
        plot_param_importance(study, figures_dir / "param_importance.png")
    except Exception as e:
        logger.warning("param importance skipped: %s", e)
    plot_working_range_histograms(working_range, trial_results,
                                   figures_dir / "working_range_histograms.png")

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("new_opt_05.py done (%.1fs) — %s", elapsed, output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
