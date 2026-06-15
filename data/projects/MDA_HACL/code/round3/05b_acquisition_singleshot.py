#!/usr/bin/env python
"""
05b_acquisition_singleshot.py — Acquisition-Based Singleshot Selection Evaluation
==================================================================================

Compares intelligent acquisition-based selection of singleshot mutations vs random
selection for multi-substrate BNN2 activity prediction.

Phase 1: Train substrate-split BNN (LOSO) using substrate-optimised hyperparams.
          For each held-out substrate, predict all mutations → get mean, UCB, null1, null2.

Phase 2: For each acquisition strategy, select 1 mutation per position on the held-out
          substrate. Train a singleshot BNN (with singleshot-optimised hyperparams)
          using all other substrates + selected singles. Evaluate ranking quality on
          remaining mutations.

Phase 3: Aggregate, compare, and plot results across acquisition methods.

Usage:
    python 05b_acquisition_singleshot.py --device cuda:0
    python 05b_acquisition_singleshot.py --substrates Formaldehyde Acetaldehyde --n-random-repeats 3
    python 05b_acquisition_singleshot.py --ucb-betas 0.5,1.0,2.0 --device cuda:1
"""

import argparse
import copy
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent        # code/round3/
PROJECT_ROOT = SCRIPT_DIR.parent.parent             # MDA_HACL/

sys.path.insert(0, str(SCRIPT_DIR.parent))          # MDA_HACL/code/

# 05_bnn2_common.py and 05_bnn2_multi_substrate.py start with digits —
# use importlib to load them by file path.
from importlib.util import spec_from_file_location, module_from_spec as _mfs

# Load common utilities
_common_spec = spec_from_file_location("bnn2_common", SCRIPT_DIR / "05_bnn2_common.py")
_common = _mfs(_common_spec)
_common_spec.loader.exec_module(_common)

# Config
load_config = _common.load_config
resolve_param = _common.resolve_param
resolve_config_block = _common.resolve_config_block
get_device = _common.get_device
parse_pca_value = _common.parse_pca_value
setup_logging = _common.setup_logging
# Model
BNN2Model = _common.BNN2Model
build_bnn2_model = _common.build_bnn2_model
train_and_evaluate_fold = _common.train_and_evaluate_fold
# Data
load_multi_substrate_data = _common.load_multi_substrate_data
get_supplemental_positions = _common.get_supplemental_positions
_style_supp_ticklabels = _common._style_supp_ticklabels
load_all_embeddings = _common.load_all_embeddings
load_substrate_metadata = _common.load_substrate_metadata
# Features
expand_to_pairwise = _common.expand_to_pairwise
build_bnn1_input = _common.build_bnn1_input
build_other_features = _common.build_other_features
# Preprocessing
preprocess_other_features = _common.preprocess_other_features
# BNN1
load_bnn1_backbone = _common.load_bnn1_backbone
load_bnn1_preprocessing = _common.load_bnn1_preprocessing
# Aggregation
aggregate_pairwise_predictions = _common.aggregate_pairwise_predictions
add_ref_distances = _common.add_ref_distances
# Metrics
compute_pairwise_distances = _common.compute_pairwise_distances
select_best_substrate_metric = _common.select_best_substrate_metric
compute_ndcg = _common.compute_ndcg
compute_selection_regret = _common.compute_selection_regret
compute_active_only_metrics = _common.compute_active_only_metrics
_break_ties = _common._break_ties

# Load multi-substrate script for split functions + null models
_main_spec = spec_from_file_location("bnn2_multi", SCRIPT_DIR / "05_bnn2_multi_substrate.py")
_mod = _mfs(_main_spec)
_main_spec.loader.exec_module(_mod)

make_substrate_folds = _mod.make_substrate_folds
compute_null_predictions = _mod.compute_null_predictions
compute_null2_predictions = _mod.compute_null2_predictions
resolve_all_params = _mod.resolve_all_params
FORMALDEHYDE_SUBSTRATE = _mod.FORMALDEHYDE_SUBSTRATE


# ═══════════════════════════════════════════════════════════════════════════
# Hyperparams Loading
# ═══════════════════════════════════════════════════════════════════════════

def _make_dummy_args():
    """Create a dummy argparse.Namespace with all defaults for resolve_all_params."""
    ns = argparse.Namespace()
    for attr in [
        "hidden_dims", "prior_std", "dropout_rate", "learning_rate",
        "kl_weight", "x_aa_freeze", "substrate_embedding_type",
        "x_substrate_scaler", "x_substrate_pca", "loss_type",
        "null_reg_weight", "log_var_floor", "prediction_floor",
        "floor_threshold", "inference_threshold", "inference_aggregation",
        "use_lds",
    ]:
        setattr(ns, attr, None)
    ns.n_singleshot_repeats = 10
    return ns


def load_and_apply_hyperparams(hp_path: Path, config: dict) -> dict:
    """Load best_hyperparams.json and build a full params dict.

    Starts from config defaults (via resolve_all_params with dummy CLI args),
    then overrides with values from the JSON. This matches the logic in
    05_bnn2_multi_substrate.py lines 1601-1624.

    Args:
        hp_path: Path to best_hyperparams.json.
        config:  Config dict (will be mutated for exclude_self_ref).

    Returns:
        params: Complete hyperparameter dict ready for training.
    """
    dummy_args = _make_dummy_args()
    params = resolve_all_params(dummy_args, config)

    logger.info("Loading hyperparams from %s", hp_path)
    with open(hp_path) as f:
        hp = json.load(f)

    # Keys to skip (metrics/metadata from opt script, not model params)
    _skip = {"optimization_objective", "objective_value", "raw_objective",
             "trial_number"}
    _skip.update(k for k in hp if k.startswith(("mean_cv_", "fold_")))

    for k, v in hp.items():
        if k in _skip:
            continue
        if k.startswith("feat_"):
            feat_name = k[len("feat_"):]
            params.setdefault("features", {})[feat_name] = v
        elif k == "use_lds":
            params.setdefault("lds", {})["use_lds"] = v
        elif k == "exclude_self_ref":
            config.setdefault("bnn2", {}).setdefault("pairwise", {})[k] = v
        else:
            params[k] = v

    return params


# ═══════════════════════════════════════════════════════════════════════════
# Acquisition Strategies
# ═══════════════════════════════════════════════════════════════════════════

def select_by_acquisition(
    agg_df: pd.DataFrame,
    method: str,
    beta: Optional[float] = None,
    rng: Optional[np.random.RandomState] = None,
) -> List[int]:
    """Select one mutation per position according to acquisition strategy.

    Args:
        agg_df: Aggregated predictions with columns: position, _y_pred,
                _tot_std, _null_pred, _null2_pred.
        method: One of 'mean_pred', 'ucb', 'thompson', 'null1', 'null2', 'random'.
        beta:   UCB exploration weight (only for method='ucb').
        rng:    RandomState for stochastic methods.

    Returns:
        List of agg_df index values (one per position).
    """
    selected_idx = []

    for pos, group in agg_df.groupby("position"):
        if method == "mean_pred":
            scores = group["_y_pred"].values.copy()
        elif method == "ucb":
            scores = group["_y_pred"].values + beta * group["_tot_std"].values
        elif method == "thompson":
            means = group["_y_pred"].values
            stds = group["_tot_std"].values
            scores = rng.normal(means, np.maximum(stds, 1e-8))
        elif method == "null1":
            scores = group["_null_pred"].values.copy()
        elif method == "null2":
            scores = group["_null2_pred"].values.copy()
        elif method == "random":
            idx = rng.choice(len(group))
            selected_idx.append(group.index[idx])
            continue
        else:
            raise ValueError(f"Unknown acquisition method: {method}")

        # Deterministic tie-breaking
        scores = _break_ties(scores, seed=int(pos))
        best = np.argmax(scores)
        selected_idx.append(group.index[best])

    return selected_idx


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — Substrate-Split BNN Predictions
# ═══════════════════════════════════════════════════════════════════════════

def run_phase1_substrate_predictions(
    df: pd.DataFrame,
    embeddings: dict,
    bnn1_hidden,
    bnn1_input_dim: int,
    latent_dim: int,
    bnn1_pipe_wt,
    bnn1_pipe_mut,
    substrate_params: dict,
    config: dict,
    device: str,
    substrate_meta: dict,
    null_emb_type: str,
    null_dist_metric: str,
) -> List[Tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Train substrate-split BNN (LOSO), return per-substrate predictions.

    Returns:
        List of (held_out_substrate, agg_df, df_train) per fold.
        agg_df has columns: mutation_string, substrate, position, log_fc,
        _y_pred, _epi_std, _ale_std, _tot_std, _null_pred, _null2_pred.
    """
    folds = make_substrate_folds(df)
    logger.info("Phase 1: %d LOSO folds", len(folds))

    all_fold_results = []

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)
        held_out_sub = df_val["substrate"].unique()[0]

        logger.info("─── Phase 1 Fold %d/%d: held-out %s (train=%d, val=%d) ───",
                     fold_i + 1, len(folds), held_out_sub,
                     len(df_train), len(df_val))

        # Build pairwise training reference lookup (leak prevention)
        train_fc_lookup = {
            (row["mutation_string"], row["substrate"]): row["fold_change"]
            for _, row in df_train.iterrows()
        }
        df_train_exp = expand_to_pairwise(df_train, substrate_meta, config)
        df_val_exp = expand_to_pairwise(df_val, substrate_meta, config,
                                        ref_fc_lookup=train_fc_lookup)

        if len(df_train_exp) == 0 or len(df_val_exp) == 0:
            logger.warning("  Fold %d: empty after expansion, skipping", fold_i + 1)
            continue

        # Add reference distances
        agg_mode = substrate_params.get("inference_aggregation", "nearest")
        if agg_mode in ("nearest", "distance_weighted"):
            sub_emb_type = substrate_params.get("substrate_embedding_type", "morgan")
            sub_emb = embeddings[f"substrate_{sub_emb_type}"].astype(np.float64)
            ref_dist_matrix = compute_pairwise_distances(sub_emb, "cosine")
            substrate_names = embeddings["substrate_names"]
            add_ref_distances(df_train_exp, ref_dist_matrix, substrate_names)
            add_ref_distances(df_val_exp, ref_dist_matrix, substrate_names)

        logger.info("  Expanded: train=%d, val=%d", len(df_train_exp), len(df_val_exp))

        # Build features
        use_bnn1 = substrate_params.get("features", {}).get("x_aa", False)
        if use_bnn1:
            X_bnn1_train = build_bnn1_input(df_train_exp, embeddings, bnn1_pipe_wt, bnn1_pipe_mut)
            X_bnn1_val = build_bnn1_input(df_val_exp, embeddings, bnn1_pipe_wt, bnn1_pipe_mut)

        groups_train = build_other_features(df_train_exp, embeddings, substrate_params, substrate_meta)
        groups_val = build_other_features(df_val_exp, embeddings, substrate_params, substrate_meta)

        X_other_train, X_other_val, _ = preprocess_other_features(
            groups_train, groups_val, substrate_params, config)

        if use_bnn1:
            X_train = np.concatenate([X_bnn1_train, X_other_train], axis=1).astype(np.float32)
            X_val = np.concatenate([X_bnn1_val, X_other_val], axis=1).astype(np.float32)
        else:
            X_train = X_other_train.astype(np.float32)
            X_val = X_other_val.astype(np.float32)

        # Delta targets
        fc_ref_train = df_train_exp["log_fc_ref"].values.astype(np.float32)
        fc_ref_val = df_val_exp["log_fc_ref"].values.astype(np.float32)
        y_train = (df_train_exp["log_fc"].values - fc_ref_train).astype(np.float32)
        y_val = (df_val_exp["log_fc"].values - fc_ref_val).astype(np.float32)

        other_feature_dim = X_other_train.shape[1]

        # Train and evaluate
        fold_metrics, estimates, _ = train_and_evaluate_fold(
            X_train, y_train, X_val, y_val,
            bnn1_hidden if use_bnn1 else None,
            bnn1_input_dim if use_bnn1 else 0,
            latent_dim if use_bnn1 else 0,
            other_feature_dim,
            substrate_params, device, return_predictions=True,
            fc_ref_train=fc_ref_train, fc_ref_val=fc_ref_val,
        )

        # Extract predictions
        y_pred_exp = estimates.mean.cpu().numpy().squeeze(-1)
        epi_std_exp = estimates.epistemic_std.cpu().numpy().squeeze(-1)
        ale_std_exp = estimates.aleatoric_std.cpu().numpy().squeeze(-1)
        tot_std_exp = estimates.total_std.cpu().numpy().squeeze(-1)

        # Aggregate pairwise → per-(mutation, substrate)
        dist_temp = substrate_params.get("distance_weight_temperature", 1.0)
        y_pred_agg, epi_agg, ale_agg, tot_agg, agg_df = aggregate_pairwise_predictions(
            y_pred_exp, epi_std_exp, ale_std_exp, tot_std_exp, df_val_exp,
            aggregation_mode=agg_mode, distance_weight_temperature=dist_temp,
        )

        # Add BNN predictions to agg_df
        agg_df["_y_pred"] = y_pred_agg
        agg_df["_epi_std"] = epi_agg
        agg_df["_ale_std"] = ale_agg
        agg_df["_tot_std"] = tot_agg

        # Null predictions
        null_pred = compute_null_predictions(
            agg_df, df_train, embeddings, "substrate",
            substrate_embedding_type=null_emb_type, distance_metric=null_dist_metric)
        null2_pred = compute_null2_predictions(agg_df, df_train)

        agg_df["_null_pred"] = null_pred
        agg_df["_null2_pred"] = null2_pred

        logger.info("  Phase 1 fold %d: Spearman=%.3f, MAE=%.3f (%s)",
                     fold_i + 1, fold_metrics.get("spearman_rho", float("nan")),
                     fold_metrics.get("mae", float("nan")), held_out_sub)

        all_fold_results.append((held_out_sub, agg_df, df_train))

        # Free GPU memory
        del estimates
        torch.cuda.empty_cache()

    return all_fold_results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — Singleshot Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def _compute_mrr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Reciprocal Rank: 1/rank of true-best mutation in model ranking."""
    if len(y_true) < 2:
        return float("nan")
    true_best = np.argmax(y_true)
    pred_order = np.argsort(_break_ties(y_pred, seed=len(y_pred)))[::-1]
    rank = int(np.where(pred_order == true_best)[0][0]) + 1
    return 1.0 / rank


def _compute_enrichment_factor(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_frac: float = 0.2,
    wt_activity: float = -1.9957,
) -> float:
    """Enrichment factor: fraction of true actives in model's top-k / overall fraction."""
    if len(y_true) < 5:
        return float("nan")
    is_active = y_true > wt_activity
    n_active = int(is_active.sum())
    if n_active == 0 or n_active == len(y_true):
        return float("nan")
    k = max(1, int(len(y_true) * top_frac))
    pred_topk = np.argsort(_break_ties(y_pred, seed=len(y_pred)))[-k:]
    frac_active_in_topk = is_active[pred_topk].mean()
    frac_active_overall = n_active / len(y_true)
    return float(frac_active_in_topk / frac_active_overall)


def _train_and_evaluate_singleshot(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    embeddings: dict,
    bnn1_hidden,
    bnn1_input_dim: int,
    latent_dim: int,
    bnn1_pipe_wt,
    bnn1_pipe_mut,
    params: dict,
    config: dict,
    device: str,
    substrate_meta: dict,
) -> dict:
    """Train a singleshot BNN and compute evaluation metrics.

    Returns dict with metrics + y_true/y_pred arrays.
    """
    # Build pairwise training reference lookup
    train_fc_lookup = {
        (row["mutation_string"], row["substrate"]): row["fold_change"]
        for _, row in df_train.iterrows()
    }
    df_train_exp = expand_to_pairwise(df_train, substrate_meta, config)
    df_val_exp = expand_to_pairwise(df_val, substrate_meta, config,
                                    ref_fc_lookup=train_fc_lookup)

    if len(df_train_exp) == 0 or len(df_val_exp) == 0:
        return {
            "spearman": float("nan"), "kendall_tau": float("nan"),
            "mae": float("nan"), "ndcg_full": float("nan"),
            "ndcg_5": float("nan"), "ndcg_10": float("nan"),
            "ndcg_25": float("nan"), "topk_3": float("nan"),
            "topk_5": float("nan"), "mrr": float("nan"),
            "enrichment": float("nan"),
            "best_epoch": 0, "y_true": np.array([]), "y_pred": np.array([]),
            "mutation_strings": np.array([]), "positions": np.array([]),
        }

    # Add reference distances
    agg_mode = params.get("inference_aggregation", "nearest")
    if agg_mode in ("nearest", "distance_weighted"):
        sub_emb_type = params.get("substrate_embedding_type", "morgan")
        sub_emb = embeddings[f"substrate_{sub_emb_type}"].astype(np.float64)
        ref_dist_matrix = compute_pairwise_distances(sub_emb, "cosine")
        substrate_names = embeddings["substrate_names"]
        add_ref_distances(df_train_exp, ref_dist_matrix, substrate_names)
        add_ref_distances(df_val_exp, ref_dist_matrix, substrate_names)

    # Build features
    use_bnn1 = params.get("features", {}).get("x_aa", False)
    if use_bnn1:
        X_bnn1_train = build_bnn1_input(df_train_exp, embeddings, bnn1_pipe_wt, bnn1_pipe_mut)
        X_bnn1_val = build_bnn1_input(df_val_exp, embeddings, bnn1_pipe_wt, bnn1_pipe_mut)

    groups_train = build_other_features(df_train_exp, embeddings, params, substrate_meta)
    groups_val = build_other_features(df_val_exp, embeddings, params, substrate_meta)

    X_other_train, X_other_val, _ = preprocess_other_features(
        groups_train, groups_val, params, config)

    if use_bnn1:
        X_train = np.concatenate([X_bnn1_train, X_other_train], axis=1).astype(np.float32)
        X_val = np.concatenate([X_bnn1_val, X_other_val], axis=1).astype(np.float32)
    else:
        X_train = X_other_train.astype(np.float32)
        X_val = X_other_val.astype(np.float32)

    # Delta targets
    fc_ref_train = df_train_exp["log_fc_ref"].values.astype(np.float32)
    fc_ref_val = df_val_exp["log_fc_ref"].values.astype(np.float32)
    y_train = (df_train_exp["log_fc"].values - fc_ref_train).astype(np.float32)
    y_val = (df_val_exp["log_fc"].values - fc_ref_val).astype(np.float32)

    other_feature_dim = X_other_train.shape[1]

    # Train
    fold_metrics, estimates, _ = train_and_evaluate_fold(
        X_train, y_train, X_val, y_val,
        bnn1_hidden if use_bnn1 else None,
        bnn1_input_dim if use_bnn1 else 0,
        latent_dim if use_bnn1 else 0,
        other_feature_dim,
        params, device, return_predictions=True,
        fc_ref_train=fc_ref_train, fc_ref_val=fc_ref_val,
    )

    # Extract + aggregate
    y_pred_exp = estimates.mean.cpu().numpy().squeeze(-1)
    epi_std_exp = estimates.epistemic_std.cpu().numpy().squeeze(-1)
    ale_std_exp = estimates.aleatoric_std.cpu().numpy().squeeze(-1)
    tot_std_exp = estimates.total_std.cpu().numpy().squeeze(-1)

    dist_temp = params.get("distance_weight_temperature", 1.0)
    y_pred_agg, epi_agg, ale_agg, tot_agg, agg_df = aggregate_pairwise_predictions(
        y_pred_exp, epi_std_exp, ale_std_exp, tot_std_exp, df_val_exp,
        aggregation_mode=agg_mode, distance_weight_temperature=dist_temp,
    )

    y_true = agg_df["log_fc"].values.astype(np.float64)
    y_pred = y_pred_agg.astype(np.float64)

    # Compute metrics
    spearman = float(stats.spearmanr(y_true, y_pred).statistic) if len(y_true) > 1 else float("nan")
    kendall = float(stats.kendalltau(y_true, y_pred).statistic) if len(y_true) > 1 else float("nan")
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # Top-k recovery (k=3, 5)
    topk_3 = _topk_recovery(y_true, y_pred, k=3)
    topk_5 = _topk_recovery(y_true, y_pred, k=5)

    result = {
        "spearman": spearman,
        "kendall_tau": kendall,
        "mae": mae,
        "ndcg_full": compute_ndcg(y_true, y_pred),
        "ndcg_5": compute_ndcg(y_true, y_pred, k=5),
        "ndcg_10": compute_ndcg(y_true, y_pred, k=10),
        "ndcg_25": compute_ndcg(y_true, y_pred, k=25),
        "topk_3": topk_3,
        "topk_5": topk_5,
        "mrr": _compute_mrr(y_true, y_pred),
        "enrichment": _compute_enrichment_factor(y_true, y_pred),
        "best_epoch": fold_metrics.get("best_epoch", 0),
        "y_true": y_true,
        "y_pred": y_pred,
        "mutation_strings": agg_df["mutation_string"].values,
        "positions": agg_df["position"].values,
    }

    # Free GPU memory
    del estimates
    torch.cuda.empty_cache()

    return result


def _topk_recovery(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
    """Fraction of true top-k in predicted top-k."""
    if len(y_true) < k:
        return float("nan")
    true_topk = set(np.argsort(y_true)[-k:])
    pred_topk = set(np.argsort(_break_ties(y_pred, seed=k))[-k:])
    return len(true_topk & pred_topk) / k


def run_phase2_singleshot_evaluation(
    phase1_results: List[Tuple[str, pd.DataFrame, pd.DataFrame]],
    df: pd.DataFrame,
    embeddings: dict,
    bnn1_hidden,
    bnn1_input_dim: int,
    latent_dim: int,
    bnn1_pipe_wt,
    bnn1_pipe_mut,
    singleshot_params: dict,
    config: dict,
    device: str,
    substrate_meta: dict,
    n_random_repeats: int = 10,
    ucb_betas: Optional[List[float]] = None,
    seed: int = 42,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """For each substrate and acquisition method, train singleshot BNN and evaluate.

    Returns:
        all_results: List of dicts with metrics per (substrate, method, repeat).
        all_selections: List of dicts recording what was selected.
        all_val_predictions: List of dicts with per-mutation validation predictions.
    """
    if ucb_betas is None:
        ucb_betas = [0.5, 1.0, 2.0]

    rng = np.random.RandomState(seed)
    all_results = []
    all_selections = []
    all_val_predictions = []

    for sub_i, (held_out_sub, agg_df, _df_train_p1) in enumerate(phase1_results):
        logger.info("=" * 50)
        logger.info("Phase 2 — Substrate %d/%d: %s (%d mutations)",
                     sub_i + 1, len(phase1_results), held_out_sub, len(agg_df))

        # Get original held-out rows and other substrates from full df
        held_out_mask = df["substrate"] == held_out_sub
        df_held_out = df[held_out_mask].reset_index(drop=True)
        df_others = df[~held_out_mask].reset_index(drop=True)

        # Build list of (method_name, list_of_selection_index_lists)
        method_selections = []

        # Deterministic methods
        sel_idx = select_by_acquisition(agg_df, "mean_pred")
        method_selections.append(("mean_pred", [sel_idx]))

        for beta in ucb_betas:
            sel_idx = select_by_acquisition(agg_df, "ucb", beta=beta)
            method_selections.append((f"ucb_{beta}", [sel_idx]))

        # Thompson sampling (single draw)
        sel_idx = select_by_acquisition(agg_df, "thompson", rng=np.random.RandomState(seed))
        method_selections.append(("thompson", [sel_idx]))

        # Null1
        sel_idx = select_by_acquisition(agg_df, "null1")
        method_selections.append(("null1", [sel_idx]))

        # Null2 (skip for formaldehyde — trivially perfect)
        if held_out_sub != FORMALDEHYDE_SUBSTRATE:
            sel_idx = select_by_acquisition(agg_df, "null2")
            method_selections.append(("null2", [sel_idx]))

        # Random (multiple repeats)
        random_selections = []
        for rep in range(n_random_repeats):
            sel_idx = select_by_acquisition(
                agg_df, "random", rng=np.random.RandomState(seed + rep))
            random_selections.append(sel_idx)
        method_selections.append(("random", random_selections))

        # Evaluate each method
        for method_name, selection_repeats in method_selections:
            for rep_i, sel_idx in enumerate(selection_repeats):
                rep_label = method_name if len(selection_repeats) == 1 else f"{method_name}_r{rep_i}"
                logger.info("  %s: selecting %d mutations...", rep_label, len(sel_idx))

                # Map selected indices to mutation_strings
                selected_rows = agg_df.loc[sel_idx]
                selected_mutations = set(selected_rows["mutation_string"].values)

                # Record what was selected
                for _, row in selected_rows.iterrows():
                    # Compute acquisition score for this selection
                    if method_name == "mean_pred":
                        acq_score = float(row["_y_pred"])
                    elif method_name.startswith("ucb_"):
                        beta = float(method_name.split("_")[1])
                        acq_score = float(row["_y_pred"] + beta * row["_tot_std"])
                    elif method_name == "null1":
                        acq_score = float(row["_null_pred"])
                    elif method_name == "null2":
                        acq_score = float(row["_null2_pred"])
                    else:
                        acq_score = float("nan")

                    # Compute true rank at this position
                    pos_group = agg_df[agg_df["position"] == row["position"]]
                    true_rank = int((pos_group["log_fc"] >= row["log_fc"]).sum())

                    all_selections.append({
                        "substrate": held_out_sub,
                        "method": method_name,
                        "repeat": rep_i,
                        "position": int(row["position"]),
                        "mutation_string": row["mutation_string"],
                        "acquisition_score": acq_score,
                        "true_log_fc": float(row["log_fc"]),
                        "true_rank": true_rank,
                        "n_at_position": len(pos_group),
                    })

                # Build singleshot train/val
                selected_mask = df_held_out["mutation_string"].isin(selected_mutations)
                df_shot = df_held_out[selected_mask]
                df_val = df_held_out[~selected_mask].reset_index(drop=True)

                if len(df_val) == 0:
                    logger.warning("    No validation data for %s/%s, skipping",
                                   held_out_sub, rep_label)
                    continue

                df_train = pd.concat([df_others, df_shot], ignore_index=True)

                logger.info("    Training: %d (others=%d + shots=%d), Val: %d",
                            len(df_train), len(df_others), len(df_shot), len(df_val))

                # Train and evaluate
                result = _train_and_evaluate_singleshot(
                    df_train, df_val, embeddings, bnn1_hidden, bnn1_input_dim,
                    latent_dim, bnn1_pipe_wt, bnn1_pipe_mut, singleshot_params,
                    config, device, substrate_meta,
                )

                result["substrate"] = held_out_sub
                result["method"] = method_name
                result["repeat"] = rep_i
                result["n_selected"] = len(selected_mutations)
                result["n_val"] = len(df_val)
                all_results.append(result)

                # Collect per-mutation validation predictions for acquisition curves
                if len(result.get("y_true", [])) > 0:
                    for k in range(len(result["y_true"])):
                        all_val_predictions.append({
                            "substrate": held_out_sub,
                            "method": method_name,
                            "repeat": rep_i,
                            "mutation_string": result["mutation_strings"][k],
                            "position": int(result["positions"][k]),
                            "y_true": float(result["y_true"][k]),
                            "y_pred": float(result["y_pred"][k]),
                        })

                logger.info("    %s: Spearman=%.3f  NDCG=%.3f  Kendall=%.3f  MAE=%.3f",
                            rep_label,
                            result["spearman"], result["ndcg_full"],
                            result["kendall_tau"], result["mae"])

    return all_results, all_selections, all_val_predictions


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 — Aggregation & Comparison
# ═══════════════════════════════════════════════════════════════════════════

METRIC_COLS = ["spearman", "kendall_tau", "mae", "ndcg_full", "ndcg_5",
               "ndcg_10", "ndcg_25", "topk_3", "topk_5", "mrr", "enrichment"]


def aggregate_results(all_results: List[dict]) -> Tuple[dict, pd.DataFrame]:
    """Aggregate per-substrate per-method results into summary statistics.

    Returns:
        summary: Dict of {method: {per_substrate: DataFrame, mean: Series, std: Series}}.
        results_df: Full results DataFrame (without y_true/y_pred arrays).
    """
    # Build DataFrame, dropping numpy arrays
    rows = []
    for r in all_results:
        row = {k: v for k, v in r.items() if k not in ("y_true", "y_pred")}
        rows.append(row)
    results_df = pd.DataFrame(rows)

    summary = {}
    for method in results_df["method"].unique():
        method_df = results_df[results_df["method"] == method]
        available_cols = [c for c in METRIC_COLS if c in method_df.columns]

        if method == "random":
            # Average across repeats first (per substrate), then across substrates
            per_sub_mean = method_df.groupby("substrate")[available_cols].mean()
            per_sub_std = method_df.groupby("substrate")[available_cols].std()
            summary[method] = {
                "per_substrate_mean": per_sub_mean,
                "per_substrate_std": per_sub_std,
                "mean": per_sub_mean.mean(),
                "std": per_sub_mean.std(),
                "n_repeats": int(method_df["repeat"].nunique()),
            }
        else:
            # One value per substrate
            per_sub = method_df.set_index("substrate")[available_cols]
            summary[method] = {
                "per_substrate_mean": per_sub,
                "mean": per_sub.mean(),
                "std": per_sub.std(),
            }

    return summary, results_df


def _compute_lift(summary: dict) -> dict:
    """Compute % lift over random baseline for each method and metric."""
    if "random" not in summary:
        return {}
    random_mean = summary["random"]["mean"]
    lift = {}
    for method, data in summary.items():
        if method == "random":
            continue
        method_mean = data["mean"]
        lift[method] = {}
        for metric in METRIC_COLS:
            if metric in method_mean and metric in random_mean:
                r_val = random_mean[metric]
                m_val = method_mean[metric]
                if abs(r_val) > 1e-8:
                    # For MAE, lower is better → invert sign
                    if metric == "mae":
                        lift[method][metric] = (r_val - m_val) / abs(r_val) * 100
                    else:
                        lift[method][metric] = (m_val - r_val) / abs(r_val) * 100
                else:
                    lift[method][metric] = float("nan")
    return lift


# ═══════════════════════════════════════════════════════════════════════════
# Plotting Functions
# ═══════════════════════════════════════════════════════════════════════════

# Consistent method colors used across all plots
METHOD_COLORS = {
    "mean_pred": "#2196F3",   # Blue
    "thompson": "#9C27B0",    # Purple
    "null1": "#9E9E9E",       # Gray
    "null2": "#FFC107",       # Amber
    "random": "#4CAF50",      # Green
}
# UCB betas get orange shades
_UCB_SHADES = ["#FF9800", "#FF5722", "#E64A19", "#BF360C"]


def _method_color(method: str) -> str:
    if method in METHOD_COLORS:
        return METHOD_COLORS[method]
    if method.startswith("ucb_"):
        idx = 0
        try:
            beta = float(method.split("_")[1])
            idx = min(int(beta), len(_UCB_SHADES) - 1)
        except (IndexError, ValueError):
            pass
        return _UCB_SHADES[idx]
    return "#795548"  # Brown fallback


def _method_order(methods):
    """Sort methods in a sensible display order."""
    priority = {"mean_pred": 0, "thompson": 3, "null1": 5, "null2": 6, "random": 7}
    def _key(m):
        if m in priority:
            return priority[m]
        if m.startswith("ucb_"):
            try:
                return 1 + float(m.split("_")[1]) * 0.1
            except (IndexError, ValueError):
                return 2
        return 4
    return sorted(methods, key=_key)


def plot_method_comparison_bars(
    summary: dict,
    metric: str,
    output_path: Path,
    title: Optional[str] = None,
):
    """Bar chart comparing methods on a single metric."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = _method_order([m for m in summary if metric in summary[m]["mean"]])
    if not methods:
        return

    vals = [summary[m]["mean"][metric] for m in methods]
    errs = [summary[m]["std"][metric] for m in methods]
    colors = [_method_color(m) for m in methods]

    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.2), 5))
    x = np.arange(len(methods))
    bars = ax.bar(x, vals, yerr=errs, capsize=4, color=colors, edgecolor="none")

    # Random baseline band
    if "random" in summary and metric in summary["random"]["mean"]:
        r_mean = summary["random"]["mean"][metric]
        r_std = summary["random"]["std"][metric]
        ax.axhspan(r_mean - r_std, r_mean + r_std, alpha=0.15, color="#4CAF50")
        ax.axhline(r_mean, color="#4CAF50", ls="--", lw=1.5, alpha=0.7, label="Random mean")

    # Value labels
    for i, (v, e) in enumerate(zip(vals, errs)):
        if not np.isnan(v):
            ax.text(x[i], v + e + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(metric)
    ax.set_title(title or f"Acquisition Method Comparison: {metric}")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_per_substrate_heatmap(
    results_df: pd.DataFrame,
    metric: str,
    output_path: Path,
):
    """Heatmap: substrates × methods, colored by metric value."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # For random, average across repeats
    grouped = results_df.groupby(["substrate", "method"])[metric].mean().reset_index()
    pivot = grouped.pivot(index="substrate", columns="method", values=metric)

    # Reorder columns
    ordered = _method_order([c for c in pivot.columns])
    pivot = pivot[[c for c in ordered if c in pivot.columns]]

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.5),
                                     max(5, len(pivot.index) * 0.8)))
    data = pivot.values
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn" if metric != "mae" else "RdYlGn_r")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > np.nanpercentile(np.abs(data), 75) else "black")

    plt.colorbar(im, ax=ax, label=metric)
    ax.set_title(f"Per-Substrate {metric.capitalize()} by Method")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_lift_over_random(summary: dict, output_path: Path):
    """Bar chart: % improvement over random for each method × metric."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lift = _compute_lift(summary)
    if not lift:
        return

    methods = _method_order(list(lift.keys()))
    metrics = [m for m in METRIC_COLS if any(m in lift[meth] for meth in methods)]

    fig, ax = plt.subplots(figsize=(max(10, len(methods) * len(metrics) * 0.5), 6))

    n_methods = len(methods)
    n_metrics = len(metrics)
    w = 0.8 / n_metrics
    x = np.arange(n_methods)

    cmap = plt.cm.Set2
    for mi, metric in enumerate(metrics):
        offset = mi * w - 0.4 + w / 2
        vals = [lift[m].get(metric, float("nan")) for m in methods]
        colors = [cmap(mi / max(n_metrics - 1, 1))] * n_methods
        ax.bar(x + offset, vals, w, color=colors, label=metric, edgecolor="none")

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("% Improvement over Random")
    ax.set_title("Lift over Random Baseline")
    ax.legend(fontsize=7, ncol=min(4, n_metrics), loc="upper right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_ucb_sensitivity(
    results_df: pd.DataFrame,
    ucb_betas: List[float],
    output_path: Path,
):
    """Line plot: UCB beta vs metrics, one line per substrate + bold average."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ucb_methods = [f"ucb_{b}" for b in ucb_betas]
    ucb_df = results_df[results_df["method"].isin(ucb_methods)].copy()
    if ucb_df.empty:
        return

    ucb_df["beta"] = ucb_df["method"].apply(lambda m: float(m.split("_")[1]))

    metrics_to_plot = ["spearman", "ndcg_full", "kendall_tau"]
    fig, axes = plt.subplots(1, len(metrics_to_plot),
                              figsize=(5 * len(metrics_to_plot), 4))
    if len(metrics_to_plot) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics_to_plot):
        for sub in sorted(ucb_df["substrate"].unique()):
            sub_df = ucb_df[ucb_df["substrate"] == sub].sort_values("beta")
            ax.plot(sub_df["beta"], sub_df[metric], "o-", alpha=0.3, ms=4, lw=1)

        # Bold average
        avg = ucb_df.groupby("beta")[metric].mean().sort_index()
        ax.plot(avg.index, avg.values, "s-", color="black", lw=2.5, ms=8,
                label="Mean", zorder=10)

        ax.set_xlabel("UCB β")
        ax.set_ylabel(metric)
        ax.set_title(f"UCB β Sensitivity: {metric}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_selection_quality(all_selections: List[dict], output_path: Path):
    """Box plot of actual log_fc of selected mutations, grouped by method."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel_df = pd.DataFrame(all_selections)
    if sel_df.empty:
        return

    methods = _method_order(sel_df["method"].unique().tolist())
    data = [sel_df[sel_df["method"] == m]["true_log_fc"].dropna().values for m in methods]
    colors = [_method_color(m) for m in methods]

    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.2), 5))
    bp = ax.boxplot(data, positions=range(len(methods)), widths=0.6,
                    patch_artist=True, showmeans=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("True log_fc of Selected Mutation")
    ax.set_title("Selection Quality: Actual Activity of Chosen Mutations")

    # Add median values as text
    for i, d in enumerate(data):
        if len(d) > 0:
            med = np.median(d)
            ax.text(i, med + 0.05, f"{med:.2f}", ha="center", fontsize=7)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_acquisition_correlation(all_selections: List[dict], output_path: Path):
    """Scatter: acquisition_score vs true_log_fc, one panel per method."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel_df = pd.DataFrame(all_selections)
    # Only deterministic methods with meaningful acquisition scores
    sel_df = sel_df[sel_df["acquisition_score"].notna() & ~sel_df["acquisition_score"].isna()]
    methods = _method_order([m for m in sel_df["method"].unique()
                            if m not in ("random", "thompson")])
    if not methods:
        return

    ncols = min(3, len(methods))
    nrows = (len(methods) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for i, method in enumerate(methods):
        ax = axes[i // ncols, i % ncols]
        mdf = sel_df[sel_df["method"] == method]
        ax.scatter(mdf["acquisition_score"], mdf["true_log_fc"],
                   s=30, alpha=0.6, color=_method_color(method), edgecolor="none")

        if len(mdf) > 2:
            rho, pval = stats.spearmanr(mdf["acquisition_score"], mdf["true_log_fc"])
            ax.set_xlabel("Acquisition Score")
            ax.set_ylabel("True log_fc")
            ax.set_title(f"{method} (ρ={rho:.3f}, p={pval:.3f})")
        else:
            ax.set_title(method)

    # Hide unused
    for i in range(len(methods), nrows * ncols):
        axes[i // ncols, i % ncols].set_visible(False)

    fig.suptitle("Acquisition Score vs Actual Activity", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_agreement_matrix(all_selections: List[dict], output_path: Path):
    """Heatmap: pairwise agreement between methods (same mutation selected)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel_df = pd.DataFrame(all_selections)
    # For random, use repeat 0 only for comparison
    det_df = sel_df[(sel_df["method"] != "random") | (sel_df["repeat"] == 0)].copy()

    methods = _method_order(det_df["method"].unique().tolist())
    n = len(methods)
    agreement = np.zeros((n, n))

    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            df1 = det_df[det_df["method"] == m1]
            df2 = det_df[det_df["method"] == m2]
            # Merge on (substrate, position)
            merged = df1.merge(df2, on=["substrate", "position"],
                               suffixes=("_1", "_2"), how="inner")
            if len(merged) > 0:
                agreement[i, j] = (merged["mutation_string_1"] == merged["mutation_string_2"]).mean()

    fig, ax = plt.subplots(figsize=(max(6, n * 0.9), max(5, n * 0.8)))
    im = ax.imshow(agreement, cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(methods, fontsize=9)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{agreement[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if agreement[i, j] > 0.5 else "black")

    plt.colorbar(im, ax=ax, label="Fraction Agreement")
    ax.set_title("Selection Agreement Between Methods")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_position_selections(
    all_selections: List[dict],
    substrate_meta: dict,
    output_path: Path,
    supp_positions: Optional[set] = None,
):
    """Per-substrate grid: what each method selected at each position."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel_df = pd.DataFrame(all_selections)
    # Deterministic methods + random rep 0
    det_df = sel_df[(sel_df["method"] != "random") | (sel_df["repeat"] == 0)].copy()

    substrates = sorted(det_df["substrate"].unique())
    methods = _method_order(det_df["method"].unique().tolist())
    positions = sorted(det_df["position"].unique())

    n_subs = len(substrates)
    fig, axes = plt.subplots(n_subs, 1, figsize=(max(10, len(positions) * 1.5),
                                                   2.5 * n_subs), squeeze=False)

    for si, sub in enumerate(substrates):
        ax = axes[si, 0]
        sub_df = det_df[det_df["substrate"] == sub]

        # Build matrix: rows=methods, cols=positions, values=true_log_fc
        matrix = np.full((len(methods), len(positions)), float("nan"))
        labels = [[""] * len(positions) for _ in range(len(methods))]

        for mi, method in enumerate(methods):
            for pi, pos in enumerate(positions):
                row = sub_df[(sub_df["method"] == method) & (sub_df["position"] == pos)]
                if len(row) > 0:
                    matrix[mi, pi] = row.iloc[0]["true_log_fc"]
                    # Extract amino acid from mutation_string (e.g., "V83A" → "A")
                    mut_str = row.iloc[0]["mutation_string"]
                    labels[mi][pi] = mut_str[-1] if mut_str else ""

        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn")

        # Annotate with AA letter
        for mi in range(len(methods)):
            for pi in range(len(positions)):
                if labels[mi][pi]:
                    ax.text(pi, mi, labels[mi][pi], ha="center", va="center",
                            fontsize=7, fontweight="bold")

        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels(positions, fontsize=8)
        _style_supp_ticklabels(ax, supp_positions, axis="x")
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods, fontsize=8)
        is_active = substrate_meta.get(sub, {}).get("is_active", True)
        ax.set_title(f"{sub} ({'active' if is_active else 'inactive'})", fontsize=10)

    fig.suptitle("Position Selections by Method (color = true log_fc)", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_metric_distributions(results_df: pd.DataFrame, output_path: Path):
    """Box plots: metric distributions across substrates for each method."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_to_plot = ["spearman", "ndcg_full", "kendall_tau", "mae"]
    nrows = 2
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 9))

    methods = _method_order(results_df["method"].unique().tolist())

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // ncols, idx % ncols]
        # For random: average across repeats per substrate first
        data_per_method = []
        for method in methods:
            mdf = results_df[results_df["method"] == method]
            if method == "random":
                per_sub = mdf.groupby("substrate")[metric].mean()
                data_per_method.append(per_sub.values)
            else:
                data_per_method.append(mdf[metric].dropna().values)

        bp = ax.boxplot(data_per_method, positions=range(len(methods)),
                        widths=0.6, patch_artist=True, showmeans=True)
        for patch, method in zip(bp["boxes"], methods):
            patch.set_facecolor(_method_color(method))
            patch.set_alpha(0.7)

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(metric)
        ax.set_title(metric.capitalize())

    fig.suptitle("Metric Distributions Across Substrates", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_summary_dashboard(summary: dict, output_path: Path):
    """Multi-panel dashboard: one panel per metric showing all methods."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_to_plot = ["spearman", "ndcg_full", "kendall_tau", "mae", "mrr", "topk_5"]
    nrows = 2
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 8))

    methods = _method_order(list(summary.keys()))

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // ncols, idx % ncols]
        vals = []
        errs = []
        colors = []
        labels = []
        for m in methods:
            if metric in summary[m]["mean"]:
                vals.append(summary[m]["mean"][metric])
                errs.append(summary[m]["std"][metric])
                colors.append(_method_color(m))
                labels.append(m)

        if not vals:
            ax.set_visible(False)
            continue

        x = np.arange(len(labels))
        ax.bar(x, vals, yerr=errs, capsize=3, color=colors, edgecolor="none")

        # Random band
        if "random" in summary and metric in summary["random"]["mean"]:
            r_mean = summary["random"]["mean"][metric]
            r_std = summary["random"]["std"][metric]
            ax.axhspan(r_mean - r_std, r_mean + r_std, alpha=0.12, color="#4CAF50")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(metric, fontsize=10)

        for i, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(x[i], v + errs[i] + 0.005, f"{v:.3f}", ha="center", fontsize=6)

    fig.suptitle("Acquisition Method Summary Dashboard", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_selected_rank_distribution(all_selections: List[dict], output_path: Path):
    """Histogram of true rank of selected mutation at each position."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel_df = pd.DataFrame(all_selections)
    # Deterministic methods + random rep 0
    det_df = sel_df[(sel_df["method"] != "random") | (sel_df["repeat"] == 0)].copy()

    methods = _method_order(det_df["method"].unique().tolist())
    ncols = min(4, len(methods))
    nrows = (len(methods) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

    for i, method in enumerate(methods):
        ax = axes[i // ncols, i % ncols]
        mdf = det_df[det_df["method"] == method]
        ranks = mdf["true_rank"].values
        n_total = mdf["n_at_position"].values

        if len(ranks) == 0:
            ax.set_visible(False)
            continue

        ax.hist(ranks, bins=range(1, int(ranks.max()) + 2), color=_method_color(method),
                alpha=0.7, edgecolor="black", lw=0.5)
        ax.axvline(1, color="red", ls="--", lw=1, alpha=0.5, label="Rank 1 (best)")
        frac_rank1 = (ranks == 1).mean()
        ax.set_xlabel("True Rank (1 = best)")
        ax.set_ylabel("Count")
        ax.set_title(f"{method} ({frac_rank1:.0%} rank-1)")
        ax.legend(fontsize=7)

    for i in range(len(methods), nrows * ncols):
        axes[i // ncols, i % ncols].set_visible(False)

    fig.suptitle("Distribution of True Rank of Selected Mutations", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_parity_per_method(
    all_results: List[dict],
    output_path_prefix: Path,
):
    """Per-method parity plots: y_true vs y_pred across substrates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Group results by method
    method_data = {}
    for r in all_results:
        method = r["method"]
        if method == "random":
            continue  # Skip random to keep manageable
        if len(r.get("y_true", [])) == 0:
            continue
        method_data.setdefault(method, {"y_true": [], "y_pred": [], "substrates": []})
        method_data[method]["y_true"].append(r["y_true"])
        method_data[method]["y_pred"].append(r["y_pred"])
        method_data[method]["substrates"].extend([r["substrate"]] * len(r["y_true"]))

    for method, data in method_data.items():
        y_true = np.concatenate(data["y_true"])
        y_pred = np.concatenate(data["y_pred"])
        substrates = np.array(data["substrates"])

        if len(y_true) < 2:
            continue

        rho, _ = stats.spearmanr(y_true, y_pred)
        mae = np.mean(np.abs(y_true - y_pred))

        fig, ax = plt.subplots(figsize=(6, 6))

        # Color by substrate
        unique_subs = sorted(set(substrates))
        cmap = plt.cm.tab10
        for si, sub in enumerate(unique_subs):
            mask = substrates == sub
            ax.scatter(y_true[mask], y_pred[mask], s=15, alpha=0.6,
                       color=cmap(si / max(len(unique_subs) - 1, 1)),
                       label=sub, edgecolor="none")

        lims = [min(y_true.min(), y_pred.min()) - 0.1,
                max(y_true.max(), y_pred.max()) + 0.1]
        ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("True log_fc")
        ax.set_ylabel("Predicted log_fc")
        ax.set_title(f"{method}: ρ={rho:.3f}, MAE={mae:.3f}")
        ax.legend(fontsize=7, loc="upper left", ncol=2)
        ax.set_aspect("equal")

        out = output_path_prefix.parent / f"{output_path_prefix.stem}_{method}.png"
        plt.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", out.name)


def plot_acquisition_curves(
    sel_df: pd.DataFrame,
    val_pred_df: pd.DataFrame,
    output_path: Path,
    top_frac: float = 0.05,
):
    """Acquisition curves: cumulative discovery of top mutations.

    For each acquisition method, the workflow is:
      1. First round: acquire the selected mutations (1 per position).
      2. Subsequent rounds: acquire remaining mutations ranked by singleshot
         model mean prediction (descending).
      3. Track cumulative fraction of true top-X% mutations discovered.

    Curves are averaged across substrates (thin lines per substrate,
    bold line for mean +/- std).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    substrates = sorted(sel_df["substrate"].unique())

    # For random: use repeat 0 only
    sel_use = sel_df[(sel_df["method"] != "random") | (sel_df["repeat"] == 0)].copy()
    pred_use = val_pred_df[(val_pred_df["method"] != "random") | (val_pred_df["repeat"] == 0)].copy()

    methods = _method_order(sel_use["method"].unique().tolist())

    method_curves = {}

    for method in methods:
        substrate_curves = []

        for sub in substrates:
            # Selected mutations for this method/substrate
            sel_sub = sel_use[(sel_use["method"] == method) &
                              (sel_use["substrate"] == sub)]
            if sel_sub.empty:
                continue

            # Val predictions for this method/substrate
            pred_sub = pred_use[(pred_use["method"] == method) &
                                (pred_use["substrate"] == sub)]
            if pred_sub.empty:
                continue

            # Remaining mutations sorted by model mean (descending)
            remaining = pred_sub.sort_values("y_pred", ascending=False)

            # Combine all true activities: selected first, then remaining
            all_true = np.concatenate([
                sel_sub["true_log_fc"].values,
                remaining["y_true"].values,
            ])
            n_total = len(all_true)
            n_top = max(1, int(np.ceil(n_total * top_frac)))

            # Top-X% threshold (n_top-th largest value)
            threshold = np.sort(all_true)[-n_top]

            # Build ordered acquisition: selected batch first, then by prediction
            acquired = np.concatenate([
                sel_sub["true_log_fc"].values,
                remaining["y_true"].values,
            ])
            is_top = acquired >= threshold
            cum_top = np.cumsum(is_top) / n_top

            substrate_curves.append(cum_top)

        if not substrate_curves:
            continue

        # Pad shorter curves to the max length
        max_len = max(len(c) for c in substrate_curves)
        padded = []
        for c in substrate_curves:
            if len(c) < max_len:
                c = np.concatenate([c, np.full(max_len - len(c), c[-1])])
            padded.append(c)

        mean_curve = np.mean(padded, axis=0)
        std_curve = np.std(padded, axis=0)
        x = np.arange(1, max_len + 1)

        method_curves[method] = (x, mean_curve, std_curve, padded)

    if not method_curves:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for method in _method_order(list(method_curves.keys())):
        x, mean, std, per_sub = method_curves[method]
        color = _method_color(method)

        # Thin per-substrate lines
        for curve in per_sub:
            ax.plot(x, curve, color=color, alpha=0.12, lw=0.8)

        # Bold mean +/- std
        ax.plot(x, mean, color=color, label=method, lw=2.2)
        ax.fill_between(x, np.clip(mean - std, 0, 1),
                         np.clip(mean + std, 0, 1), color=color, alpha=0.15)

    # Theoretical random baseline (diagonal)
    n_total = max(len(x) for x, _, _, _ in method_curves.values())
    ax.plot([0, n_total], [0, 1], "k--", lw=1, alpha=0.4, label="Random (theoretical)")

    # Vertical marker at the initial selection batch size (1 per position)
    first_method = _method_order(list(method_curves.keys()))[0]
    n_selected = len(sel_use[(sel_use["method"] == first_method) &
                              (sel_use["substrate"] == substrates[0])])
    ax.axvline(n_selected, color="gray", ls=":", lw=1, alpha=0.6)
    ax.text(n_selected + 2, 0.97,
            f"Initial selection\n(n={n_selected})",
            fontsize=8, color="gray", va="top")

    ax.set_xlabel("Number of Mutations Tested")
    ax.set_ylabel(f"Fraction of Top {top_frac:.0%} Mutations Found")
    ax.set_title(f"Acquisition Curves: Discovery of Top {top_frac:.0%} Mutations\n"
                 f"(first round by acquisition method, remaining by model mean)")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, n_total)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def generate_acquisition_plots(
    summary: dict,
    results_df: pd.DataFrame,
    all_results: List[dict],
    all_selections: List[dict],
    plots_dir: Path,
    ucb_betas: List[float],
    substrate_meta: dict,
    val_pred_df: Optional[pd.DataFrame] = None,
    supp_positions: Optional[set] = None,
):
    """Generate all acquisition comparison plots."""
    plot_funcs = [
        ("Method comparison (Spearman)",
         lambda: plot_method_comparison_bars(summary, "spearman",
                 plots_dir / "method_comparison_spearman.png")),
        ("Method comparison (NDCG)",
         lambda: plot_method_comparison_bars(summary, "ndcg_full",
                 plots_dir / "method_comparison_ndcg.png")),
        ("Method comparison (Kendall)",
         lambda: plot_method_comparison_bars(summary, "kendall_tau",
                 plots_dir / "method_comparison_kendall.png")),
        ("Method comparison (MAE)",
         lambda: plot_method_comparison_bars(summary, "mae",
                 plots_dir / "method_comparison_mae.png")),
        ("Per-substrate heatmap (Spearman)",
         lambda: plot_per_substrate_heatmap(results_df, "spearman",
                 plots_dir / "per_substrate_heatmap_spearman.png")),
        ("Per-substrate heatmap (NDCG)",
         lambda: plot_per_substrate_heatmap(results_df, "ndcg_full",
                 plots_dir / "per_substrate_heatmap_ndcg.png")),
        ("Lift over random",
         lambda: plot_lift_over_random(summary, plots_dir / "lift_over_random.png")),
        ("UCB sensitivity",
         lambda: plot_ucb_sensitivity(results_df, ucb_betas,
                 plots_dir / "ucb_sensitivity.png")),
        ("Selection quality",
         lambda: plot_selection_quality(all_selections, plots_dir / "selection_quality.png")),
        ("Acquisition correlation",
         lambda: plot_acquisition_correlation(all_selections,
                 plots_dir / "acquisition_correlation.png")),
        ("Agreement matrix",
         lambda: plot_agreement_matrix(all_selections, plots_dir / "agreement_matrix.png")),
        ("Position selections",
         lambda: plot_position_selections(all_selections, substrate_meta,
                 plots_dir / "position_selections.png",
                 supp_positions=supp_positions)),
        ("Metric distributions",
         lambda: plot_metric_distributions(results_df, plots_dir / "metric_distributions.png")),
        ("Summary dashboard",
         lambda: plot_summary_dashboard(summary, plots_dir / "summary_dashboard.png")),
        ("Selected rank distribution",
         lambda: plot_selected_rank_distribution(all_selections,
                 plots_dir / "selected_rank_distribution.png")),
        ("Parity per method",
         lambda: plot_parity_per_method(all_results,
                 plots_dir / "parity_grid")),
    ]

    # Acquisition curves require per-mutation val predictions
    if val_pred_df is not None:
        _sel = pd.DataFrame(all_selections)
        plot_funcs.append(
            ("Acquisition curves",
             lambda: plot_acquisition_curves(
                 _sel, val_pred_df, plots_dir / "acquisition_curves.png")),
        )

    for name, func in plot_funcs:
        try:
            func()
        except Exception as e:
            logger.warning("Failed to generate plot '%s': %s", name, e)


# ═══════════════════════════════════════════════════════════════════════════
# CLI & Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Acquisition-based singleshot selection evaluation: "
                    "compare BNN mean, UCB, null1, null2, and random selection "
                    "strategies for choosing singleshot mutations."
    )
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto/cuda:N/cpu)")
    parser.add_argument("--substrate-params", type=str, default=None,
                        help="Path to substrate-split best_hyperparams.json "
                             "(default: results/opt_05_bnn2/substrate/best_hyperparams.json)")
    parser.add_argument("--singleshot-params", type=str, default=None,
                        help="Path to singleshot-split best_hyperparams.json "
                             "(default: results/opt_05_bnn2/singleshot/best_hyperparams.json)")
    parser.add_argument("--n-random-repeats", type=int, default=10,
                        help="Number of random baseline repeats (default: 10)")
    parser.add_argument("--ucb-betas", type=str, default="0.5,1.0,2.0",
                        help="Comma-separated UCB beta values (default: 0.5,1.0,2.0)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Results directory (default: results/05b_acquisition_singleshot)")
    parser.add_argument("--substrates", type=str, nargs="*", default=None,
                        help="Only run on specific substrates (for debugging)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml")
    parser.add_argument("--bnn1-model-dir", type=str, default=None,
                        help="Path to BNN1 model directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip phases 1-2; regenerate plots from saved CSVs "
                             "(requires prior full run in the same output-dir)")
    return parser.parse_args()


def main():
    args = parse_args()
    t_start = time.time()

    # 1. Setup
    results_dir = (Path(args.output_dir) if args.output_dir
                   else PROJECT_ROOT / "results" / "05b_acquisition_singleshot")
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    setup_logging(results_dir / "run.log")

    logger.info("=" * 60)
    logger.info("05b_acquisition_singleshot.py")
    logger.info("=" * 60)

    config = load_config(args.config)

    # ── Plot-only mode: skip phases 1-2, regenerate plots from saved data ──
    if args.plot_only:
        logger.info("--plot-only: loading saved results from %s", results_dir)

        processed_dir = PROJECT_ROOT / config["data"]["output_dir"]
        substrate_meta = load_substrate_metadata(processed_dir)
        _df_for_supp = load_multi_substrate_data(processed_dir)
        supp_positions = get_supplemental_positions(_df_for_supp)
        del _df_for_supp

        # Load saved CSVs
        results_df = pd.read_csv(results_dir / "phase2_results.csv")
        sel_df = pd.read_csv(results_dir / "acquisition_selections.csv")
        all_selections = sel_df.to_dict("records")

        val_pred_path = results_dir / "phase2_val_predictions.csv"
        val_pred_df = pd.read_csv(val_pred_path) if val_pred_path.exists() else None
        if val_pred_df is None:
            logger.warning("phase2_val_predictions.csv not found — "
                           "acquisition curves will be skipped. "
                           "Run without --plot-only once to generate it.")

        # Reconstruct all_results (with y_true/y_pred for parity plots)
        all_results = []
        for _, row in results_df.iterrows():
            result = row.to_dict()
            if val_pred_df is not None:
                pred_rows = val_pred_df[
                    (val_pred_df["substrate"] == row["substrate"]) &
                    (val_pred_df["method"] == row["method"]) &
                    (val_pred_df["repeat"] == row["repeat"])
                ]
                result["y_true"] = pred_rows["y_true"].values
                result["y_pred"] = pred_rows["y_pred"].values
            else:
                result["y_true"] = np.array([])
                result["y_pred"] = np.array([])
            all_results.append(result)

        # Aggregate
        summary, results_df = aggregate_results(all_results)

        # Load saved hyperparams for ucb_betas
        hp_path = results_dir / "hyperparams_used.json"
        with open(hp_path) as f:
            hp_used = json.load(f)
        ucb_betas = hp_used.get("ucb_betas", [1.0])

        # Generate plots
        logger.info("Generating plots...")
        generate_acquisition_plots(
            summary, results_df, all_results, all_selections,
            plots_dir, ucb_betas, substrate_meta,
            val_pred_df=val_pred_df,
            supp_positions=supp_positions,
        )

        elapsed = time.time() - t_start
        logger.info("Plot-only mode complete in %.1f sec", elapsed)
        return

    device = get_device(config, args.device)

    # 2. Load BNN1 backbone
    bnn1_model_dir = (Path(args.bnn1_model_dir) if args.bnn1_model_dir
                      else PROJECT_ROOT / "results" / "03_formaldehyde_regression" / "models")
    bnn1_hidden, bnn1_input_dim, latent_dim, _ = load_bnn1_backbone(bnn1_model_dir, device)
    bnn1_pipe_wt, bnn1_pipe_mut = load_bnn1_preprocessing(bnn1_model_dir)

    # 3. Load data
    processed_dir = PROJECT_ROOT / config["data"]["output_dir"]
    df = load_multi_substrate_data(processed_dir)
    supp_positions = get_supplemental_positions(df)
    embeddings = load_all_embeddings(processed_dir)
    substrate_meta = load_substrate_metadata(processed_dir)

    # 4. Load TWO sets of hyperparams
    substrate_hp_path = (Path(args.substrate_params) if args.substrate_params
                         else PROJECT_ROOT / "results" / "opt_05_bnn2" / "substrate" / "best_hyperparams.json")
    singleshot_hp_path = (Path(args.singleshot_params) if args.singleshot_params
                          else PROJECT_ROOT / "results" / "opt_05_bnn2" / "singleshot" / "best_hyperparams.json")

    config_phase1 = copy.deepcopy(config)
    substrate_params = load_and_apply_hyperparams(substrate_hp_path, config_phase1)

    config_phase2 = copy.deepcopy(config)
    singleshot_params = load_and_apply_hyperparams(singleshot_hp_path, config_phase2)

    logger.info("Substrate-split params:")
    for k, v in substrate_params.items():
        if k != "features":
            logger.info("  %s: %s", k, v)
    logger.info("  features: %s", substrate_params.get("features", {}))

    logger.info("Singleshot-split params:")
    for k, v in singleshot_params.items():
        if k != "features":
            logger.info("  %s: %s", k, v)
    logger.info("  features: %s", singleshot_params.get("features", {}))

    # 5. Auto-select null model metric
    metric_selection = select_best_substrate_metric(df, embeddings)
    null_emb_type = metric_selection["best_embedding"]
    null_dist_metric = metric_selection["best_metric"]
    logger.info("Null model: %s / %s (rho=%.3f)",
                null_emb_type, null_dist_metric, metric_selection["best_correlation"])

    # 6. Filter substrates if requested
    if args.substrates:
        df = df[df["substrate"].isin(args.substrates)].reset_index(drop=True)
        logger.info("Filtered to substrates: %s (%d rows)", args.substrates, len(df))

    ucb_betas = [float(b) for b in args.ucb_betas.split(",")]
    logger.info("UCB betas: %s", ucb_betas)
    logger.info("Random repeats: %d", args.n_random_repeats)

    # 7. Phase 1: Substrate-split BNN predictions
    logger.info("=" * 60)
    logger.info("PHASE 1: Substrate-split BNN predictions")
    logger.info("=" * 60)

    phase1_results = run_phase1_substrate_predictions(
        df, embeddings, bnn1_hidden, bnn1_input_dim, latent_dim,
        bnn1_pipe_wt, bnn1_pipe_mut, substrate_params, config_phase1,
        device, substrate_meta, null_emb_type, null_dist_metric,
    )

    # Save phase1 predictions
    phase1_dfs = []
    for held_out_sub, agg_df, _ in phase1_results:
        agg_df_copy = agg_df.copy()
        agg_df_copy["_held_out_substrate"] = held_out_sub
        phase1_dfs.append(agg_df_copy)
    phase1_df = pd.concat(phase1_dfs, ignore_index=True)
    phase1_df.to_csv(results_dir / "phase1_predictions.csv", index=False)
    logger.info("Saved phase1_predictions.csv (%d rows)", len(phase1_df))

    t_phase1 = time.time()
    logger.info("Phase 1 completed in %.1f min", (t_phase1 - t_start) / 60)

    # 8. Phase 2: Singleshot evaluation
    logger.info("=" * 60)
    logger.info("PHASE 2: Singleshot evaluation per acquisition method")
    logger.info("=" * 60)

    all_results, all_selections, all_val_predictions = run_phase2_singleshot_evaluation(
        phase1_results, df, embeddings, bnn1_hidden, bnn1_input_dim, latent_dim,
        bnn1_pipe_wt, bnn1_pipe_mut, singleshot_params, config_phase2,
        device, substrate_meta,
        n_random_repeats=args.n_random_repeats,
        ucb_betas=ucb_betas,
        seed=args.seed,
    )

    # Save selections
    pd.DataFrame(all_selections).to_csv(results_dir / "acquisition_selections.csv", index=False)
    logger.info("Saved acquisition_selections.csv (%d rows)", len(all_selections))

    # Save per-mutation validation predictions (enables --plot-only acquisition curves)
    val_pred_df = pd.DataFrame(all_val_predictions)
    val_pred_df.to_csv(results_dir / "phase2_val_predictions.csv", index=False)
    logger.info("Saved phase2_val_predictions.csv (%d rows)", len(val_pred_df))
    if val_pred_df.empty:
        val_pred_df = None

    t_phase2 = time.time()
    logger.info("Phase 2 completed in %.1f min", (t_phase2 - t_phase1) / 60)

    # 9. Phase 3: Aggregate and compare
    logger.info("=" * 60)
    logger.info("PHASE 3: Aggregation, comparison, and plotting")
    logger.info("=" * 60)

    summary, results_df = aggregate_results(all_results)

    # Save results
    results_df.to_csv(results_dir / "phase2_results.csv", index=False)

    # Build metrics JSON
    metrics_json = {}
    for method, data in summary.items():
        metrics_json[method] = {
            "mean": data["mean"].to_dict() if hasattr(data["mean"], "to_dict") else dict(data["mean"]),
            "std": data["std"].to_dict() if hasattr(data["std"], "to_dict") else dict(data["std"]),
        }
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2, default=str)

    # Save hyperparams used
    with open(results_dir / "hyperparams_used.json", "w") as f:
        json.dump({
            "substrate_params_path": str(substrate_hp_path),
            "singleshot_params_path": str(singleshot_hp_path),
            "ucb_betas": ucb_betas,
            "n_random_repeats": args.n_random_repeats,
            "seed": args.seed,
        }, f, indent=2, default=str)

    # 10. Log summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    methods = _method_order(list(summary.keys()))
    header = f"{'Method':<20} {'Spearman':>10} {'NDCG':>10} {'Kendall':>10} {'MAE':>10} {'MRR':>10}"
    logger.info(header)
    logger.info("-" * len(header))
    for method in methods:
        m = summary[method]["mean"]
        s = summary[method]["std"]
        logger.info(
            "%-20s %7.3f±%.3f %7.3f±%.3f %7.3f±%.3f %7.3f±%.3f %7.3f±%.3f",
            method,
            m.get("spearman", float("nan")), s.get("spearman", float("nan")),
            m.get("ndcg_full", float("nan")), s.get("ndcg_full", float("nan")),
            m.get("kendall_tau", float("nan")), s.get("kendall_tau", float("nan")),
            m.get("mae", float("nan")), s.get("mae", float("nan")),
            m.get("mrr", float("nan")), s.get("mrr", float("nan")),
        )

    # Lift over random
    lift = _compute_lift(summary)
    if lift:
        logger.info("")
        logger.info("Lift over random (%%): ")
        for method in _method_order(list(lift.keys())):
            parts = []
            for metric in ["spearman", "ndcg_full", "kendall_tau", "mae"]:
                v = lift[method].get(metric, float("nan"))
                parts.append(f"{metric}={v:+.1f}%")
            logger.info("  %-20s %s", method, "  ".join(parts))

    # 11. Generate plots
    logger.info("")
    logger.info("Generating plots...")
    generate_acquisition_plots(
        summary, results_df, all_results, all_selections,
        plots_dir, ucb_betas, substrate_meta,
        val_pred_df=val_pred_df,
        supp_positions=supp_positions,
    )

    elapsed = time.time() - t_start
    logger.info("")
    logger.info("Total runtime: %.1f min (%.1f hours)", elapsed / 60, elapsed / 3600)
    logger.info("Results saved to %s", results_dir)


if __name__ == "__main__":
    main()
