#!/usr/bin/env python
"""
05_bnn2_multi_substrate.py — BNN2 Multi-Substrate Activity Prediction
=====================================================================

Composite BNN2 model: BNN1 backbone (frozen/partial/trainable) + BNN2 head.
Predicts log10(fold_change + epsilon) for mutations across 9 substrates using
pairwise (mutation, ref_substrate, target_substrate) triplets.

Four cross-validation split strategies test generalization:
  --split random      Data for this substrate at all positions; predict unseen AAs
  --split position    Data for this substrate at some positions; predict new positions
  --split substrate   Never tested this substrate; predict from structure + known refs
  --split singleshot  Only 1 mutation per (substrate, position); predict the rest

All hyperparameters are CLI args with defaults from config.yaml bnn2 section.
Use opt_05_bnn2.py for automated hyperopt.

Usage:
    python 05_bnn2_multi_substrate.py --split random --device cuda:1
    python 05_bnn2_multi_substrate.py --split substrate --device cuda:1
    python 05_bnn2_multi_substrate.py --split singleshot --n-singleshot-repeats 10
    python 05_bnn2_multi_substrate.py --split random --substrate-embedding-type molformer
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

# Add code/ to sys.path for BNN module import
sys.path.insert(0, str(SCRIPT_DIR.parent))          # MDA_HACL/code/

# 05_bnn2_common.py starts with a digit and cannot be imported directly.
# Use importlib to load it from its file path.
from importlib.util import spec_from_file_location, module_from_spec as _mfs

_common_spec = spec_from_file_location(
    "bnn2_common",
    SCRIPT_DIR / "05_bnn2_common.py",
)
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
load_all_embeddings = _common.load_all_embeddings
load_substrate_metadata = _common.load_substrate_metadata
# Features
expand_to_pairwise = _common.expand_to_pairwise
build_bnn1_input = _common.build_bnn1_input
build_other_features = _common.build_other_features
get_substrate_embedding = _common.get_substrate_embedding
# Preprocessing
build_preprocessing = _common.build_preprocessing
apply_preprocessing = _common.apply_preprocessing
preprocess_other_features = _common.preprocess_other_features
# BNN1
load_bnn1_backbone = _common.load_bnn1_backbone
load_bnn1_preprocessing = _common.load_bnn1_preprocessing
# LDS weights
compute_lds_weights = _common.compute_lds_weights
plot_lds_weights = _common.plot_lds_weights
# Aggregation
aggregate_pairwise_predictions = _common.aggregate_pairwise_predictions
add_ref_distances = _common.add_ref_distances
# Metrics
compute_nlpd = _common.compute_nlpd
compute_crps_gaussian = _common.compute_crps_gaussian
compute_calibration = _common.compute_calibration
compute_per_group_metrics = _common.compute_per_group_metrics
compute_tanimoto_distances = _common.compute_tanimoto_distances
compute_embedding_distances = _common.compute_embedding_distances
compute_pairwise_distances = _common.compute_pairwise_distances
compute_functional_distances = _common.compute_functional_distances
select_best_substrate_metric = _common.select_best_substrate_metric
range_weighted_mean = _common.range_weighted_mean
# New engineering-value metrics
compute_ndcg = _common.compute_ndcg
compute_active_only_metrics = _common.compute_active_only_metrics
compute_active_substrate_metrics = _common.compute_active_substrate_metrics
compute_classification_metrics = _common.compute_classification_metrics
compute_per_substrate_topk_recovery = _common.compute_per_substrate_topk_recovery
compute_substrate_discrimination = _common.compute_substrate_discrimination
compute_above_floor_metrics = _common.compute_above_floor_metrics
compute_selection_regret = _common.compute_selection_regret
compute_hurdle_metrics = _common.compute_hurdle_metrics
# Plotting
plot_parity = _common.plot_parity
plot_residuals = _common.plot_residuals
plot_calibration = _common.plot_calibration
plot_uncertainty_vs_error = _common.plot_uncertainty_vs_error
plot_uncertainty_decomposition = _common.plot_uncertainty_decomposition
plot_training_curves = _common.plot_training_curves
plot_loss_decomposition = _common.plot_loss_decomposition
plot_per_substrate_metrics = _common.plot_per_substrate_metrics
plot_per_position_metrics = _common.plot_per_position_metrics
plot_substrate_position_heatmap = _common.plot_substrate_position_heatmap
plot_substrate_parity_grid = _common.plot_substrate_parity_grid
plot_substrate_parity_comparison_grid = _common.plot_substrate_parity_comparison_grid
plot_distance_vs_performance = _common.plot_distance_vs_performance
plot_substrate_transfer_matrix = _common.plot_substrate_transfer_matrix
plot_singleshot_distributions = _common.plot_singleshot_distributions
plot_acquisition_recovery = _common.plot_acquisition_recovery
plot_per_substrate_topk_recovery = _common.plot_per_substrate_topk_recovery
plot_per_substrate_acquisition_recovery = _common.plot_per_substrate_acquisition_recovery
plot_engineering_value_summary = _common.plot_engineering_value_summary
plot_hurdle_diagnostics = _common.plot_hurdle_diagnostics
plot_selection_regret = _common.plot_selection_regret


# ═══════════════════════════════════════════════════════════════════════════
# Split Functions
# ═══════════════════════════════════════════════════════════════════════════

def make_random_folds(df: pd.DataFrame, n_folds: int, seed: int):
    """KFold on df index — random split."""
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in kf.split(df):
        folds.append((train_idx, val_idx))
    return folds


def make_position_folds(df: pd.DataFrame, n_folds: int, seed: int):
    """GroupKFold by position — hold out entire positions."""
    from sklearn.model_selection import GroupKFold
    groups = df["position"].values
    gkf = GroupKFold(n_splits=n_folds)
    folds = []
    for train_idx, val_idx in gkf.split(df, groups=groups):
        folds.append((train_idx, val_idx))
    return folds


def make_substrate_folds(df: pd.DataFrame):
    """LeaveOneGroupOut by substrate — 9 folds."""
    from sklearn.model_selection import LeaveOneGroupOut
    groups = df["substrate"].values
    logo = LeaveOneGroupOut()
    folds = []
    for train_idx, val_idx in logo.split(df, groups=groups):
        folds.append((train_idx, val_idx))
    return folds


def make_singleshot_folds(
    df: pd.DataFrame, n_repeats: int, seed: int,
):
    """Leave-one-substrate-out + 1 random mutation per position on the held-out substrate.

    For each substrate S (outer loop) and each repeat (inner loop):
      - train = all data from other substrates
              + 1 randomly chosen mutation per position on S
      - val   = remaining mutations on S

    Returns list of (train_idx, val_idx) — n_substrates * n_repeats folds.
    """
    rng = np.random.RandomState(seed)
    folds = []

    for substrate in sorted(df["substrate"].unique()):
        other_idx = df.index[df["substrate"] != substrate].tolist()
        held_out = df[df["substrate"] == substrate]

        for rep in range(n_repeats):
            shot_idx = []
            val_idx = []
            for pos, group in held_out.groupby("position"):
                indices = group.index.tolist()
                if len(indices) <= 1:
                    shot_idx.extend(indices)
                    continue
                chosen = rng.choice(indices)
                shot_idx.append(chosen)
                val_idx.extend([i for i in indices if i != chosen])

            train_idx = other_idx + shot_idx
            folds.append((np.array(train_idx), np.array(val_idx)))

    return folds


# ═══════════════════════════════════════════════════════════════════════════
# Parameter Resolution
# ═══════════════════════════════════════════════════════════════════════════

def resolve_all_params(args: argparse.Namespace, config: dict) -> dict:
    """Build complete params dict from CLI args + config defaults."""
    bnn2 = config["bnn2"]
    train = bnn2["training"]
    preproc = config["preprocessing"]

    # Parse hidden_dims from CLI string if provided
    hidden_dims_cli = json.loads(args.hidden_dims) if args.hidden_dims else None

    # Parse PCA values
    x_sub_pca_cli = parse_pca_value(args.x_substrate_pca) if args.x_substrate_pca is not None else None
    x_sub_pca_cfg = parse_pca_value(resolve_param(preproc.get("x_substrate", {}).get("pca")))

    params = {
        # Model
        "hidden_dims":             resolve_param(bnn2["hidden_dims"], hidden_dims_cli),
        "prior_std":               resolve_param(bnn2["prior_std"], args.prior_std),
        "dropout_rate":            resolve_param(bnn2["dropout_rate"], args.dropout_rate),
        "activation":              resolve_param(bnn2["activation"]),
        "x_aa_freeze":             resolve_param(bnn2["x_aa_freeze"], args.x_aa_freeze),
        "substrate_embedding_type": resolve_param(bnn2["substrate_embedding_type"],
                                                   args.substrate_embedding_type),
        # Training
        "learning_rate":           resolve_param(train["learning_rate"], args.learning_rate),
        "kl_weight":               resolve_param(train["kl_weight"], args.kl_weight),
        "batch_size":              resolve_param(train["batch_size"]),
        "kl_anneal_epochs":        resolve_param(train["kl_anneal_epochs"]),
        "n_epochs":                resolve_param(train["n_epochs"]),
        "early_stopping_patience": resolve_param(train["early_stopping_patience"]),
        "n_inference_samples":     resolve_param(train["n_inference_samples"]),
        "clip_grad_norm":          resolve_param(train.get("clip_grad_norm", {"value": 1.0})),
        # Feature toggles (resolve value/search dicts to plain booleans)
        "features":                resolve_config_block(bnn2.get("features", {})),
        # LDS config (resolved from config.yaml; use_lds overridden by CLI)
        "lds":                     resolve_config_block(config.get("bnn2", {}).get("lds", {})),
    }

    # Preprocessing for other features
    params["x_substrate_scaler"] = resolve_param(
        preproc.get("x_substrate", {}).get("scaler", "none"), args.x_substrate_scaler)
    params["x_substrate_pca"] = x_sub_pca_cli if args.x_substrate_pca is not None else x_sub_pca_cfg
    params["saprot_zs_scaler"] = resolve_param(
        preproc.get("saprot_zs", {}).get("scaler", "none"))
    # ESM preprocessing (BNN2 applies its own scaling/PCA on ESM features)
    params["esm_wt_scaler"] = resolve_param(preproc.get("esm_wt", {}).get("scaler", "standard"))
    params["esm_wt_pca"] = parse_pca_value(resolve_param(preproc.get("esm_wt", {}).get("pca")))
    params["esm_mut_scaler"] = resolve_param(preproc.get("esm_mut", {}).get("scaler", "standard"))
    params["esm_mut_pca"] = parse_pca_value(resolve_param(preproc.get("esm_mut", {}).get("pca")))

    # Loss type (unified loss control — replaces hurdle.enabled)
    loss_type_cfg = bnn2.get("loss_type", {"value": "gaussian_nll"})
    params["loss_type"] = resolve_param(loss_type_cfg, getattr(args, "loss_type", None))

    # Null regularization
    null_reg_cfg = bnn2.get("training", {}).get("null_reg_weight", {"value": 0.0})
    params["null_reg_weight"] = resolve_param(null_reg_cfg, getattr(args, "null_reg_weight", None))

    # Variance floor (None = off, use default -10 clamp)
    lv_floor_cfg = bnn2.get("training", {}).get("log_var_floor", {"value": None})
    params["log_var_floor"] = resolve_param(lv_floor_cfg, getattr(args, "log_var_floor", None))

    # Prediction floor (clamp mu in absolute space at detection limit)
    pred_floor_cfg = bnn2.get("training", {}).get("prediction_floor", {"value": None})
    params["prediction_floor"] = resolve_param(pred_floor_cfg, getattr(args, "prediction_floor", None))

    # Hurdle sub-parameters (only used when loss_type == "hurdle")
    hurdle_cfg = bnn2.get("hurdle", {})
    params["hurdle"] = {
        "floor_threshold": getattr(args, "floor_threshold", None) or hurdle_cfg.get("floor_threshold", -1.99),
        "floor_value": hurdle_cfg.get("floor_value", -2.0),
        "inference_threshold": getattr(args, "inference_threshold", None) or hurdle_cfg.get("inference_threshold", 0.5),
    }

    # Pairwise aggregation settings (resolve value/search dicts)
    pairwise_cfg = resolve_config_block(bnn2.get("pairwise", {}))
    params["inference_aggregation"] = (
        getattr(args, "inference_aggregation", None)
        or pairwise_cfg.get("inference_aggregation", "nearest")
    )
    params["distance_weight_temperature"] = pairwise_cfg.get("distance_weight_temperature", 1.0)

    return params


# ═══════════════════════════════════════════════════════════════════════════
# Null Models
# ═══════════════════════════════════════════════════════════════════════════

FORMALDEHYDE_SUBSTRATE = "Formaldehyde"

def compute_null_predictions(
    agg_df: pd.DataFrame,
    df_train: pd.DataFrame,
    embeddings: dict,
    split_type: str,
    substrate_embedding_type: str = "molformer",
    distance_metric: str = "cosine",
) -> np.ndarray:
    """Null model predictions for each row in agg_df.

    random / position:  global mean of y_train (constant predictor).
    substrate / singleshot:  log_fc from the nearest training substrate
        (by auto-selected embedding + distance metric), for the same
        mutation_string.  Falls back to y_train mean when the mutation
        is absent on the nearest substrate.

    The (substrate_embedding_type, distance_metric) pair should be chosen
    by select_best_substrate_metric() to maximise correlation with actual
    functional differences between substrates.
    """
    y_train = df_train["log_fc"].values.astype(np.float32)
    mean_pred = float(y_train.mean())

    if split_type in ("random", "position"):
        return np.full(len(agg_df), mean_pred, dtype=np.float32)

    # Substrate / singleshot — nearest-substrate null
    substrate_names = embeddings["substrate_names"]
    emb = embeddings[f"substrate_{substrate_embedding_type}"].astype(np.float64)
    sub_to_idx = {name: i for i, name in enumerate(substrate_names)}

    # Pre-compute full distance matrix
    dist_matrix = compute_pairwise_distances(emb, distance_metric)

    # Build training lookup: (mutation_string, substrate) -> log_fc
    train_lookup: dict = {}
    for _, row in df_train.iterrows():
        train_lookup[(row["mutation_string"], row["substrate"])] = float(row["log_fc"])
    train_substrates = list(df_train["substrate"].unique())

    # Map each unique val substrate → nearest training substrate
    nearest_map: dict = {}
    for val_sub in agg_df["substrate"].unique():
        if val_sub not in sub_to_idx:
            nearest_map[val_sub] = None
            continue
        val_idx = sub_to_idx[val_sub]
        best_dist, best_sub = float("inf"), None
        for train_sub in train_substrates:
            if train_sub == val_sub or train_sub not in sub_to_idx:
                continue
            d = float(dist_matrix[val_idx, sub_to_idx[train_sub]])
            if d < best_dist:
                best_dist, best_sub = d, train_sub
        nearest_map[val_sub] = best_sub
        logger.info("  Null model: %s → nearest = %s (dist=%.3f, %s/%s)",
                    val_sub, best_sub, best_dist,
                    substrate_embedding_type, distance_metric)

    null_preds = np.empty(len(agg_df), dtype=np.float32)
    for i, (_, row) in enumerate(agg_df.iterrows()):
        nearest = nearest_map.get(row["substrate"])
        if nearest is not None:
            null_preds[i] = train_lookup.get((row["mutation_string"], nearest), mean_pred)
        else:
            null_preds[i] = mean_pred
    return null_preds


def compute_null2_predictions(
    agg_df: pd.DataFrame,
    df_train: pd.DataFrame,
) -> np.ndarray:
    """Null2 model: use formaldehyde log_fc for each mutation.

    For every row in agg_df, look up the same mutation_string on the
    Formaldehyde substrate from df_train.  Falls back to the training
    mean when the mutation is absent on Formaldehyde.

    Formaldehyde rows in agg_df are still predicted (using their own
    training values) but should be **excluded from scoring** by the
    caller since that would be a trivially perfect prediction.
    """
    y_train = df_train["log_fc"].values.astype(np.float32)
    mean_pred = float(y_train.mean())

    # Build lookup: mutation_string → formaldehyde log_fc (from training data)
    form_lookup: dict = {}
    form_train = df_train[df_train["substrate"] == FORMALDEHYDE_SUBSTRATE]
    for _, row in form_train.iterrows():
        form_lookup[row["mutation_string"]] = float(row["log_fc"])

    n_found = 0
    null2_preds = np.empty(len(agg_df), dtype=np.float32)
    for i, (_, row) in enumerate(agg_df.iterrows()):
        val = form_lookup.get(row["mutation_string"])
        if val is not None:
            null2_preds[i] = val
            n_found += 1
        else:
            null2_preds[i] = mean_pred

    logger.info("  Null2 (formaldehyde): %d/%d mutations found in training "
                "(formaldehyde train rows: %d)",
                n_found, len(agg_df), len(form_train))

    return null2_preds


# ═══════════════════════════════════════════════════════════════════════════
# Core CV Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_with_cv(
    df: pd.DataFrame,
    embeddings: dict,
    bnn1_hidden,
    bnn1_input_dim: int,
    latent_dim: int,
    bnn1_pipe_wt,
    bnn1_pipe_mut,
    params: dict,
    config: dict,
    device: str,
    split_type: str,
    substrate_meta: dict,
    null_embedding_type: str = "molformer",
    null_distance_metric: str = "cosine",
):
    """Run cross-validation with the specified split strategy.

    Returns:
        metrics: overall metrics dict
        agg_y_true, agg_y_pred, agg_epi, agg_ale, agg_tot: OOF arrays (aggregated)
        agg_df: DataFrame with aggregated predictions + metadata
        fold_histories: list of TrainingHistory objects
        fold_metrics_list: per-fold metric dicts
    """
    seed = config.get("cv", {}).get("seed", 42)
    n_folds = config.get("cv", {}).get("n_folds", 5)

    # Build folds on base data
    if split_type == "random":
        folds = make_random_folds(df, n_folds, seed)
    elif split_type == "position":
        folds = make_position_folds(df, n_folds, seed)
    elif split_type == "substrate":
        folds = make_substrate_folds(df)
    elif split_type == "singleshot":
        n_repeats = params.get("n_singleshot_repeats", 10)
        folds = make_singleshot_folds(df, n_repeats, seed)
    else:
        raise ValueError(f"Unknown split type: {split_type}")

    logger.info("Split strategy: %s → %d folds", split_type, len(folds))

    fold_metrics_list = []
    fold_histories = []
    all_agg_rows = []
    lds_traces = []

    lds_cfg = params.get("lds", {})
    use_lds = lds_cfg.get("use_lds", False)

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        logger.info("─── Fold %d/%d ───", fold_i + 1, len(folds))

        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)

        logger.info("  Base: train=%d, val=%d", len(df_train), len(df_val))

        # Log split info
        if split_type == "position":
            val_positions = sorted(df_val["position"].unique())
            logger.info("  Held-out positions: %s", val_positions)
        elif split_type == "substrate":
            val_subs = sorted(df_val["substrate"].unique())
            logger.info("  Held-out substrate(s): %s", val_subs)
        elif split_type == "singleshot":
            val_subs = sorted(df_val["substrate"].unique())
            logger.info("  Held-out substrate: %s (1-shot per position)", val_subs)

        # Expand to pairwise (independently per fold — no leakage).
        # For substrate/singleshot splits we supply the train FC lookup so
        # that reference substrate values (absent from the held-out split)
        # can be resolved from training data.
        # For position splits, ALL substrates at a position are held out
        # together, so training has no entries for held-out mutations.
        # The val set must use its own internal FC values as references
        # (ref_fc_lookup=None → self-lookup within df_val).
        train_fc_lookup = {
            (row["mutation_string"], row["substrate"]): row["fold_change"]
            for _, row in df_train.iterrows()
        }
        df_train_exp = expand_to_pairwise(df_train, substrate_meta, config)
        val_ref_lookup = train_fc_lookup if split_type != "position" else None
        df_val_exp = expand_to_pairwise(df_val, substrate_meta, config,
                                        ref_fc_lookup=val_ref_lookup)

        if len(df_train_exp) == 0 or len(df_val_exp) == 0:
            logger.warning("  Fold %d: empty after expansion, skipping", fold_i + 1)
            continue

        # Add reference distances for aggregation modes (nearest / distance_weighted)
        agg_mode = params.get("inference_aggregation", "nearest")
        if agg_mode in ("nearest", "distance_weighted"):
            sub_emb_type = params.get("substrate_embedding_type", "morgan")
            sub_emb = embeddings[f"substrate_{sub_emb_type}"].astype(np.float64)
            ref_dist_matrix = compute_pairwise_distances(sub_emb, "cosine")
            substrate_names = embeddings["substrate_names"]
            add_ref_distances(df_train_exp, ref_dist_matrix, substrate_names)
            add_ref_distances(df_val_exp, ref_dist_matrix, substrate_names)

        logger.info("  Expanded: train=%d, val=%d", len(df_train_exp), len(df_val_exp))

        # Build BNN1 input (only when x_aa feature is on)
        use_bnn1 = params.get("features", {}).get("x_aa", False)
        if use_bnn1:
            X_bnn1_train = build_bnn1_input(df_train_exp, embeddings, bnn1_pipe_wt, bnn1_pipe_mut)
            X_bnn1_val = build_bnn1_input(df_val_exp, embeddings, bnn1_pipe_wt, bnn1_pipe_mut)

        # Build other features (includes direct ESM when esm_wt/esm_mut toggled on)
        groups_train = build_other_features(df_train_exp, embeddings, params, substrate_meta)
        groups_val = build_other_features(df_val_exp, embeddings, params, substrate_meta)

        # Preprocess other features (fit on train)
        X_other_train, X_other_val, pipelines = preprocess_other_features(
            groups_train, groups_val, params, config)

        # Concatenate [BNN1_input | other_features] or just [other_features]
        if use_bnn1:
            X_train = np.concatenate([X_bnn1_train, X_other_train], axis=1).astype(np.float32)
            X_val = np.concatenate([X_bnn1_val, X_other_val], axis=1).astype(np.float32)
        else:
            X_train = X_other_train.astype(np.float32)
            X_val = X_other_val.astype(np.float32)
        # Delta targets: y = log_fc - log_fc_ref
        fc_ref_train = df_train_exp["log_fc_ref"].values.astype(np.float32)
        fc_ref_val = df_val_exp["log_fc_ref"].values.astype(np.float32)
        y_train = (df_train_exp["log_fc"].values - fc_ref_train).astype(np.float32)
        y_val = (df_val_exp["log_fc"].values - fc_ref_val).astype(np.float32)

        # ── DIAGNOSTIC: Delta target construction sanity check ──
        log_fc_train = df_train_exp["log_fc"].values
        log_fc_val = df_val_exp["log_fc"].values
        logger.info("  ┌─ DELTA CONSTRUCTION (Fold %d) ─┐", fold_i + 1)
        logger.info("  │ Train: log_fc mean=%.3f std=%.3f | fc_ref mean=%.3f std=%.3f | delta mean=%.3f std=%.3f",
                     log_fc_train.mean(), log_fc_train.std(),
                     fc_ref_train.mean(), fc_ref_train.std(),
                     y_train.mean(), y_train.std())
        logger.info("  │ Val:   log_fc mean=%.3f std=%.3f | fc_ref mean=%.3f std=%.3f | delta mean=%.3f std=%.3f",
                     log_fc_val.mean(), log_fc_val.std(),
                     fc_ref_val.mean(), fc_ref_val.std(),
                     y_val.mean(), y_val.std())
        # Check: how many unique fc_ref values? (if just 1, there's no diversity)
        n_unique_ref_train = len(np.unique(np.round(fc_ref_train, 4)))
        n_unique_ref_val = len(np.unique(np.round(fc_ref_val, 4)))
        logger.info("  │ Unique fc_ref values: train=%d  val=%d", n_unique_ref_train, n_unique_ref_val)
        # Check: fc_ref as feature in X — is it present and at the right scale?
        if "fc_ref" in df_train_exp.columns:
            raw_fc_ref = df_train_exp["fc_ref"].values
            logger.info("  │ Raw fc_ref (fold_change scale): mean=%.4f std=%.4f range=[%.4f, %.4f]",
                         raw_fc_ref.mean(), raw_fc_ref.std(), raw_fc_ref.min(), raw_fc_ref.max())
        bnn1_dim = X_bnn1_train.shape[1] if use_bnn1 else 0
        logger.info("  │ Features: BNN1=%d  other=%d  total=%d",
                     bnn1_dim, X_other_train.shape[1],
                     bnn1_dim + X_other_train.shape[1])
        logger.info("  └──────────────────────────────────────┘")

        # Collect LDS trace for this fold (weights computed inside train_and_evaluate_fold)
        if use_lds:
            fold_weights = compute_lds_weights(y_train, use_lds=True,
                                               n_bins=lds_cfg.get("n_bins", 50),
                                               kernel_size=lds_cfg.get("kernel_size", 5),
                                               sigma=lds_cfg.get("sigma", 2.0))
            lds_traces.append((y_train.copy(), fold_weights.copy(), f"Fold {fold_i + 1}"))

        other_feature_dim = X_other_train.shape[1]

        # Train and evaluate (LDS weights applied inside via params["lds"])
        fold_metrics, estimates, history = train_and_evaluate_fold(
            X_train, y_train, X_val, y_val,
            bnn1_hidden if use_bnn1 else None,
            bnn1_input_dim if use_bnn1 else 0,
            latent_dim if use_bnn1 else 0,
            other_feature_dim,
            params, device, return_predictions=True,
            fc_ref_train=fc_ref_train, fc_ref_val=fc_ref_val,
        )

        fold_metrics_list.append(fold_metrics)
        if history is not None:
            fold_histories.append(history)

        # Get expanded predictions
        from bnns.model import HurdleUncertaintyEstimate
        hurdle_enabled = params.get("loss_type", "gaussian_nll") == "hurdle"

        y_pred_exp = estimates.mean.cpu().numpy().squeeze(-1)
        epi_std_exp = estimates.epistemic_std.cpu().numpy().squeeze(-1)
        ale_std_exp = estimates.aleatoric_std.cpu().numpy().squeeze(-1)
        tot_std_exp = estimates.total_std.cpu().numpy().squeeze(-1)

        # Extract cls_prob for hurdle mode
        cls_prob_exp = None
        if hurdle_enabled and isinstance(estimates, HurdleUncertaintyEstimate):
            cls_prob_exp = estimates.cls_prob.cpu().numpy().squeeze(-1)

        # Aggregate pairwise predictions to per-(mutation, substrate)
        y_pred_agg, epi_agg, ale_agg, tot_agg, agg_df = aggregate_pairwise_predictions(
            y_pred_exp, epi_std_exp, ale_std_exp, tot_std_exp, df_val_exp,
            cls_prob_expanded=cls_prob_exp,
            aggregation_mode=params.get("inference_aggregation", "nearest"),
            distance_weight_temperature=params.get("distance_weight_temperature", 1.0),
        )

        # Null model predictions (using auto-selected best metric)
        null_pred_agg = compute_null_predictions(
            agg_df, df_train, embeddings, split_type,
            substrate_embedding_type=null_embedding_type,
            distance_metric=null_distance_metric,
        )
        agg_df["_null_pred"] = null_pred_agg

        # Null2 model predictions (formaldehyde scores)
        null2_pred_agg = compute_null2_predictions(agg_df, df_train)
        agg_df["_null2_pred"] = null2_pred_agg

        # Fold-level aggregated metrics
        agg_y_true = agg_df["log_fc"].values
        agg_mae_fold = float(np.mean(np.abs(agg_y_true - y_pred_agg)))
        agg_rho_fold, _ = stats.spearmanr(agg_y_true, y_pred_agg)
        null_mae_fold = float(np.mean(np.abs(agg_y_true - null_pred_agg)))
        null_rho_fold, _ = stats.spearmanr(agg_y_true, null_pred_agg)
        # Null2 fold metrics (exclude formaldehyde from scoring)
        null2_mask_fold = agg_df["substrate"].values != FORMALDEHYDE_SUBSTRATE
        if null2_mask_fold.sum() > 1:
            null2_mae_fold = float(np.mean(np.abs(
                agg_y_true[null2_mask_fold] - null2_pred_agg[null2_mask_fold])))
            null2_rho_fold, _ = stats.spearmanr(
                agg_y_true[null2_mask_fold], null2_pred_agg[null2_mask_fold])
        else:
            null2_mae_fold = float("nan")
            null2_rho_fold = float("nan")
        agg_mode = params.get("inference_aggregation", "nearest")
        agg_win = "✓" if agg_mae_fold < null_mae_fold else "✗"

        # ── Unified fold results ──
        logger.info("  ┌─ FOLD %d RESULTS ─────────────────────────────────────────┐", fold_i + 1)
        logger.info("  │ Aggregated (n=%d, from %d expanded, mode=%s):",
                     len(y_pred_agg), len(y_pred_exp), agg_mode)
        logger.info("  │   Overall:  %s model MAE=%.4f ρ=%.4f | null1 MAE=%.4f ρ=%.4f | null2(form) MAE=%.4f ρ=%.4f",
                     agg_win, agg_mae_fold, agg_rho_fold, null_mae_fold, null_rho_fold,
                     null2_mae_fold, null2_rho_fold)

        # Per-substrate breakdown (aggregated level)
        for sub in sorted(agg_df["substrate"].unique()):
            mask = agg_df["substrate"] == sub
            if mask.sum() < 2:
                continue
            sub_true = agg_df.loc[mask, "log_fc"].values
            sub_pred = agg_df.loc[mask, "_y_pred"].values
            sub_null = null_pred_agg[mask.values] if hasattr(mask, 'values') else null_pred_agg[mask]
            sub_mae_m = float(np.mean(np.abs(sub_true - sub_pred)))
            sub_mae_n = float(np.mean(np.abs(sub_true - sub_null)))
            sub_rho_m, _ = stats.spearmanr(sub_true, sub_pred)
            sub_rho_n, _ = stats.spearmanr(sub_true, sub_null)
            # Null2 per-substrate (skip formaldehyde)
            if sub != FORMALDEHYDE_SUBSTRATE:
                sub_null2 = null2_pred_agg[mask.values] if hasattr(mask, 'values') else null2_pred_agg[mask]
                sub_mae_n2 = float(np.mean(np.abs(sub_true - sub_null2)))
                sub_rho_n2, _ = stats.spearmanr(sub_true, sub_null2)
            else:
                sub_mae_n2, sub_rho_n2 = float("nan"), float("nan")
            sw = "✓" if sub_mae_m < sub_mae_n else "✗"
            logger.info("  │   %s %-20s (n=%3d): model MAE=%.4f ρ=%.4f | null1 MAE=%.4f ρ=%.4f | null2 MAE=%.4f ρ=%.4f",
                         sw, sub, mask.sum(), sub_mae_m, sub_rho_m, sub_mae_n, sub_rho_n,
                         sub_mae_n2, sub_rho_n2)

        # Expanded (pairwise) level — secondary diagnostics
        logger.info("  │ Expanded (pairwise, n=%d):", len(y_pred_exp))
        logger.info("  │   model MAE=%.4f ρ=%.4f NLPD=%.4f | null(Δ=0) MAE=%.4f",
                     fold_metrics["mae"], fold_metrics["spearman_rho"],
                     fold_metrics["nlpd"], fold_metrics["null_mae"])
        logger.info("  │   pred: mean=%.3f std=%.3f range=[%.3f, %.3f]",
                     y_pred_agg.mean(), y_pred_agg.std(),
                     y_pred_agg.min(), y_pred_agg.max())
        logger.info("  └───────────────────────────────────────────────────────────┘")

        agg_df["fold"] = fold_i
        all_agg_rows.append(agg_df)

    # ── Combine all folds ──
    if not all_agg_rows:
        raise RuntimeError("All folds were empty — check your data and split strategy.")

    combined_df = pd.concat(all_agg_rows, ignore_index=True)
    all_y_true = combined_df["log_fc"].values
    all_y_pred = combined_df["_y_pred"].values
    all_epi = combined_df["_epi_std"].values
    all_ale = combined_df["_ale_std"].values
    all_tot = combined_df["_tot_std"].values
    all_subs = combined_df["substrate"].values
    all_pos = combined_df["position"].values

    # Overall metrics (on aggregated predictions)
    residuals = all_y_true - all_y_pred
    overall_mae = float(np.mean(np.abs(residuals)))
    overall_rmse = float(np.sqrt(np.mean(residuals**2)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((all_y_true - all_y_true.mean())**2))
    overall_r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
    overall_spearman, overall_spearman_p = stats.spearmanr(all_y_true, all_y_pred)
    overall_nlpd = compute_nlpd(all_y_true, all_y_pred, all_tot)
    overall_crps = compute_crps_gaussian(all_y_true, all_y_pred, all_tot)
    calibration = compute_calibration(all_y_true, all_y_pred, all_tot)
    sharpness = float(np.mean(all_tot))

    # ═══════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC: Cross-fold summary (all numbers are AGGREGATED level)
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("╔═══ CROSS-FOLD SUMMARY (aggregated) ═══╗")
    logger.info("║ Overall: MAE=%.4f  RMSE=%.4f  R²=%.4f  ρ=%.4f",
                 overall_mae, overall_rmse, overall_r2, overall_spearman)
    logger.info("║ Predictions: mean=%.4f  std=%.4f  range=[%.4f, %.4f]",
                 all_y_pred.mean(), all_y_pred.std(), all_y_pred.min(), all_y_pred.max())
    logger.info("║ True values: mean=%.4f  std=%.4f  range=[%.4f, %.4f]",
                 all_y_true.mean(), all_y_true.std(), all_y_true.min(), all_y_true.max())
    logger.info("║ Residuals:   mean=%.4f  std=%.4f  (bias matters!)",
                 residuals.mean(), residuals.std())
    # Per-fold metric consistency (expanded level — from train_and_evaluate_fold)
    fold_maes_exp = [f["mae"] for f in fold_metrics_list]
    fold_rhos_exp = [f["spearman_rho"] for f in fold_metrics_list]
    fold_nulls_exp = [f["null_mae"] for f in fold_metrics_list]
    fold_best_epochs = [f["best_epoch"] for f in fold_metrics_list]
    fold_collapse = [f.get("posterior_collapse_score", float("nan")) for f in fold_metrics_list]
    logger.info("║ Per-fold (expanded): MAEs=%s", ["%.4f" % m for m in fold_maes_exp])
    logger.info("║ Per-fold (expanded): null(Δ=0) MAEs=%s", ["%.4f" % m for m in fold_nulls_exp])
    logger.info("║ Per-fold (expanded): Spearman=%s", ["%.4f" % r for r in fold_rhos_exp])
    logger.info("║ Per-fold best_epoch: %s", fold_best_epochs)
    logger.info("║ Per-fold collapse_score: %s", ["%.3f" % c for c in fold_collapse])
    # Count folds where model beats null (expanded level)
    n_folds_win = sum(1 for m, n in zip(fold_maes_exp, fold_nulls_exp) if m < n)
    logger.info("║ Folds where model beats null (expanded MAE): %d / %d",
                 n_folds_win, len(fold_maes_exp))
    if n_folds_win == 0:
        logger.warning("║ ⚠ MODEL LOSES TO NULL IN ALL FOLDS (expanded)")
    logger.info("╚══════════════════════════════════════╝")

    # Per-group metrics
    per_sub_spearman, per_sub_mae, per_sub_range = compute_per_group_metrics(
        all_y_true, all_y_pred, all_subs)
    per_pos_spearman, per_pos_mae, per_pos_range = compute_per_group_metrics(
        all_y_true, all_y_pred, all_pos)

    # Null model (aggregated, using stored null predictions from each fold)
    all_null_pred = combined_df["_null_pred"].values
    null_mae = float(np.mean(np.abs(all_y_true - all_null_pred)))
    null_spearman, _ = stats.spearmanr(all_y_true, all_null_pred)
    per_sub_null_spearman, per_sub_null_mae, _ = compute_per_group_metrics(
        all_y_true, all_null_pred, all_subs)

    # Null2 model (formaldehyde scores, exclude formaldehyde from scoring)
    all_null2_pred = combined_df["_null2_pred"].values
    null2_score_mask = all_subs != FORMALDEHYDE_SUBSTRATE
    if null2_score_mask.sum() > 1:
        null2_mae = float(np.mean(np.abs(
            all_y_true[null2_score_mask] - all_null2_pred[null2_score_mask])))
        null2_spearman, _ = stats.spearmanr(
            all_y_true[null2_score_mask], all_null2_pred[null2_score_mask])
        per_sub_null2_spearman, per_sub_null2_mae, _ = compute_per_group_metrics(
            all_y_true[null2_score_mask], all_null2_pred[null2_score_mask],
            all_subs[null2_score_mask])
    else:
        null2_mae = float("nan")
        null2_spearman = float("nan")
        per_sub_null2_spearman, per_sub_null2_mae = {}, {}

    # ── Engineering-value metrics ──────────────────────────────────────────
    active_substrates = {s for s, m in substrate_meta.items()
                         if m.get("is_active", True)}

    # WT activity threshold: log10(1.0 + epsilon)
    epsilon = config.get("data", {}).get("epsilon", 0.01)
    wt_activity = float(np.log10(1.0 + epsilon))

    # Active-mutation metrics: only mutations with true log_fc > WT
    # (beneficial mutations, regardless of substrate)
    active_metrics_bnn = compute_active_only_metrics(
        all_y_true, all_y_pred, wt_activity)
    active_metrics_null = compute_active_only_metrics(
        all_y_true, all_null_pred, wt_activity)

    # Null2 engineering metrics (exclude formaldehyde from scoring)
    _n2 = null2_score_mask  # shorthand: True for non-formaldehyde rows
    active_metrics_null2 = compute_active_only_metrics(
        all_y_true[_n2], all_null2_pred[_n2], wt_activity)

    # Active-substrate metrics: all mutations on the 6 active substrates
    # (excludes synthetic all-zero inactive substrates)
    active_sub_metrics_bnn = compute_active_substrate_metrics(
        all_y_true, all_y_pred, all_subs, active_substrates)
    active_sub_metrics_null = compute_active_substrate_metrics(
        all_y_true, all_null_pred, all_subs, active_substrates)
    # Null2 active-substrate (exclude formaldehyde)
    active_sub_metrics_null2 = compute_active_substrate_metrics(
        all_y_true[_n2], all_null2_pred[_n2], all_subs[_n2], active_substrates)

    # Binary classification: active vs inactive mutation discrimination
    classification_bnn = compute_classification_metrics(
        all_y_true, all_y_pred, wt_activity)
    classification_null = compute_classification_metrics(
        all_y_true, all_null_pred, wt_activity)
    classification_null2 = compute_classification_metrics(
        all_y_true[_n2], all_null2_pred[_n2], wt_activity)

    # NDCG at various k values (full dataset and active-substrates-only)
    ndcg_k_values = [10, 25, 50, None]  # None = full ranking
    ndcg_bnn = {}
    ndcg_null = {}
    ndcg_null2 = {}
    for kv in ndcg_k_values:
        label = f"k={kv}" if kv is not None else "full"
        ndcg_bnn[label] = compute_ndcg(all_y_true, all_y_pred, k=kv)
        ndcg_null[label] = compute_ndcg(all_y_true, all_null_pred, k=kv)
        ndcg_null2[label] = compute_ndcg(
            all_y_true[_n2], all_null2_pred[_n2], k=kv)

    # Active-substrates-only NDCG
    active_sub_mask = np.isin(all_subs, list(active_substrates))
    ndcg_active_sub_bnn = {}
    ndcg_active_sub_null = {}
    ndcg_active_sub_null2 = {}
    # Null2 active-sub mask: active AND not formaldehyde
    _n2_active = active_sub_mask & _n2
    for kv in ndcg_k_values:
        label = f"k={kv}" if kv is not None else "full"
        ndcg_active_sub_bnn[label] = compute_ndcg(
            all_y_true[active_sub_mask], all_y_pred[active_sub_mask], k=kv)
        ndcg_active_sub_null[label] = compute_ndcg(
            all_y_true[active_sub_mask], all_null_pred[active_sub_mask], k=kv)
        ndcg_active_sub_null2[label] = compute_ndcg(
            all_y_true[_n2_active], all_null2_pred[_n2_active], k=kv)

    # Per-substrate top-k recovery
    topk_bnn = compute_per_substrate_topk_recovery(
        all_y_true, all_y_pred, all_subs, k=5)
    topk_null = compute_per_substrate_topk_recovery(
        all_y_true, all_null_pred, all_subs, k=5)
    topk_null2 = compute_per_substrate_topk_recovery(
        all_y_true[_n2], all_null2_pred[_n2], all_subs[_n2], k=5)

    # Substrate-level active/inactive discrimination
    sub_discrimination_bnn = compute_substrate_discrimination(
        all_y_pred, all_subs, active_substrates)
    sub_discrimination_null = compute_substrate_discrimination(
        all_null_pred, all_subs, active_substrates)
    sub_discrimination_null2 = compute_substrate_discrimination(
        all_null2_pred[_n2], all_subs[_n2], active_substrates)

    # ── Always-on new metrics ─────────────────────────────────────────────
    floor_threshold = params.get("hurdle", {}).get("floor_threshold", -1.99)

    # Above-floor metrics (BNN + null + null2)
    above_floor_bnn = compute_above_floor_metrics(all_y_true, all_y_pred, floor_threshold)
    above_floor_null = compute_above_floor_metrics(all_y_true, all_null_pred, floor_threshold)
    above_floor_null2 = compute_above_floor_metrics(
        all_y_true[_n2], all_null2_pred[_n2], floor_threshold)

    # Selection regret (BNN + null + null2) — global and active-substrates-only
    regret_bnn = compute_selection_regret(all_y_true, all_y_pred)
    regret_null = compute_selection_regret(all_y_true, all_null_pred)
    regret_null2 = compute_selection_regret(
        all_y_true[_n2], all_null2_pred[_n2])
    active_mask = np.isin(all_subs, list(active_substrates))
    if active_mask.sum() > 0:
        regret_active_bnn = compute_selection_regret(
            all_y_true[active_mask], all_y_pred[active_mask])
        regret_active_null = compute_selection_regret(
            all_y_true[active_mask], all_null_pred[active_mask])
        _n2_active_regret = active_mask & _n2
        regret_active_null2 = compute_selection_regret(
            all_y_true[_n2_active_regret], all_null2_pred[_n2_active_regret])
    else:
        regret_active_bnn, regret_active_null, regret_active_null2 = {}, {}, {}

    # Hurdle-specific metrics (only when loss_type == hurdle and cls_prob available)
    hurdle_enabled = params.get("loss_type", "gaussian_nll") == "hurdle"
    hurdle_met = None
    if hurdle_enabled and "_cls_prob" in combined_df.columns:
        all_cls_prob = combined_df["_cls_prob"].values
        inference_threshold = params["hurdle"].get("inference_threshold", 0.5)
        hurdle_met = compute_hurdle_metrics(
            all_y_true, all_cls_prob, floor_threshold, inference_threshold)

    metrics = {
        "split_type": split_type,
        "n_folds": len(folds),
        "n_samples_base": len(df),
        "n_samples_aggregated": len(combined_df),
        "mae": overall_mae,
        "rmse": overall_rmse,
        "r2": float(overall_r2),
        "spearman_rho": float(overall_spearman),
        "spearman_pvalue": float(overall_spearman_p),
        "nlpd": overall_nlpd,
        "crps": overall_crps,
        "sharpness": sharpness,
        "null_mae": null_mae,
        "null_spearman": float(null_spearman),
        "null2_mae": null2_mae,
        "null2_spearman": float(null2_spearman),
        "mae_improvement_over_null": 1.0 - overall_mae / max(null_mae, 1e-10),
        "mae_improvement_over_null2": 1.0 - overall_mae / max(null2_mae, 1e-10),
        "calibration": calibration,
        "per_substrate_spearman": {str(k): v for k, v in per_sub_spearman.items()},
        "per_substrate_mae": {str(k): v for k, v in per_sub_mae.items()},
        "per_substrate_null_spearman": {str(k): v for k, v in per_sub_null_spearman.items()},
        "per_substrate_null_mae": {str(k): v for k, v in per_sub_null_mae.items()},
        "per_substrate_null2_spearman": {str(k): v for k, v in per_sub_null2_spearman.items()},
        "per_substrate_null2_mae": {str(k): v for k, v in per_sub_null2_mae.items()},
        "per_position_spearman": {str(k): v for k, v in per_pos_spearman.items()},
        "per_position_mae": {str(k): v for k, v in per_pos_mae.items()},
        "per_position_range": {str(k): v for k, v in per_pos_range.items()},
        "per_substrate_range": {str(k): v for k, v in per_sub_range.items()},
        "mean_fold_spearman": float(np.nanmean([f["spearman_rho"] for f in fold_metrics_list])),
        "std_fold_spearman": float(np.nanstd([f["spearman_rho"] for f in fold_metrics_list])),
        "mean_fold_mae": float(np.nanmean([f["mae"] for f in fold_metrics_list])),
        "mean_fold_nlpd": float(np.nanmean([f["nlpd"] for f in fold_metrics_list])),
        "mean_fold_crps": float(np.nanmean([f["crps"] for f in fold_metrics_list])),
        "fold_metrics": fold_metrics_list,
        "weighted_per_substrate_spearman": range_weighted_mean(per_sub_spearman, per_sub_range),
        "weighted_per_position_spearman": range_weighted_mean(per_pos_spearman, per_pos_range),
        # Engineering-value metrics
        "active_mutation_bnn": active_metrics_bnn,
        "active_mutation_null": active_metrics_null,
        "active_mutation_null2": active_metrics_null2,
        "active_substrate_bnn": active_sub_metrics_bnn,
        "active_substrate_null": active_sub_metrics_null,
        "active_substrate_null2": active_sub_metrics_null2,
        "classification_bnn": classification_bnn,
        "classification_null": classification_null,
        "classification_null2": classification_null2,
        "ndcg_bnn": ndcg_bnn,
        "ndcg_null": ndcg_null,
        "ndcg_null2": ndcg_null2,
        "ndcg_active_sub_bnn": ndcg_active_sub_bnn,
        "ndcg_active_sub_null": ndcg_active_sub_null,
        "ndcg_active_sub_null2": ndcg_active_sub_null2,
        "topk_recovery_bnn": {str(k): v for k, v in topk_bnn.items()},
        "topk_recovery_null": {str(k): v for k, v in topk_null.items()},
        "topk_recovery_null2": {str(k): v for k, v in topk_null2.items()},
        "substrate_discrimination_bnn": sub_discrimination_bnn,
        "substrate_discrimination_null": sub_discrimination_null,
        "substrate_discrimination_null2": sub_discrimination_null2,
        # Always-on new metrics
        "above_floor_bnn": above_floor_bnn,
        "above_floor_null": above_floor_null,
        "above_floor_null2": above_floor_null2,
        "selection_regret_bnn": regret_bnn,
        "selection_regret_null": regret_null,
        "selection_regret_null2": regret_null2,
        "selection_regret_active_bnn": regret_active_bnn,
        "selection_regret_active_null": regret_active_null,
        "selection_regret_active_null2": regret_active_null2,
        # Aggregation mode used
        "inference_aggregation": params.get("inference_aggregation", "nearest"),
    }

    # Add hurdle metrics if available
    if hurdle_met is not None:
        metrics["hurdle"] = hurdle_met
        metrics["hurdle_enabled"] = True
    else:
        metrics["hurdle_enabled"] = False

    logger.info("═══ Overall (%s, aggregated) ═══", split_type)
    logger.info("MAE:          %.4f  (null1: %.4f,  %.1f%% impr | null2: %.4f,  %.1f%% impr)",
                overall_mae, null_mae, metrics["mae_improvement_over_null"] * 100,
                null2_mae, metrics["mae_improvement_over_null2"] * 100)
    logger.info("RMSE:         %.4f", overall_rmse)
    logger.info("R²:           %.4f", overall_r2)
    logger.info("Spearman ρ:   %.4f  (null1: %.4f | null2: %.4f)",
                overall_spearman, null_spearman, null2_spearman)
    logger.info("NLPD:         %.4f", overall_nlpd)
    logger.info("CRPS:         %.4f", overall_crps)
    logger.info("Sharpness:    %.4f", sharpness)
    logger.info("Per-substrate Spearman (model): %s",
                "  ".join(f"{k}={v:.3f}" for k, v in sorted(per_sub_spearman.items())))
    logger.info("Per-substrate Spearman (null1): %s",
                "  ".join(f"{k}={v:.3f}" for k, v in sorted(per_sub_null_spearman.items())))
    logger.info("Per-substrate Spearman (null2): %s",
                "  ".join(f"{k}={v:.3f}" for k, v in sorted(per_sub_null2_spearman.items())))

    # Log engineering-value metrics
    logger.info("── Engineering-Value Metrics ──")
    logger.info("Active mutations (log_fc > WT, n=%d):", active_metrics_bnn["active_n"])
    logger.info("  MAE:      %.4f  (null1: %.4f | null2: %.4f)",
                active_metrics_bnn["active_mae"], active_metrics_null["active_mae"],
                active_metrics_null2.get("active_mae", float("nan")))
    logger.info("  Spearman: %.4f  (null1: %.4f | null2: %.4f)",
                active_metrics_bnn["active_spearman"], active_metrics_null["active_spearman"],
                active_metrics_null2.get("active_spearman", float("nan")))
    logger.info("  NDCG:     %.4f  (null1: %.4f | null2: %.4f)",
                active_metrics_bnn["active_ndcg"], active_metrics_null["active_ndcg"],
                active_metrics_null2.get("active_ndcg", float("nan")))
    logger.info("Active substrates (6 subs, n=%d):", active_sub_metrics_bnn["active_sub_n"])
    logger.info("  MAE:      %.4f  (null1: %.4f | null2: %.4f)",
                active_sub_metrics_bnn["active_sub_mae"], active_sub_metrics_null["active_sub_mae"],
                active_sub_metrics_null2.get("active_sub_mae", float("nan")))
    logger.info("  Spearman: %.4f  (null1: %.4f | null2: %.4f)",
                active_sub_metrics_bnn["active_sub_spearman"],
                active_sub_metrics_null["active_sub_spearman"],
                active_sub_metrics_null2.get("active_sub_spearman", float("nan")))
    logger.info("Classification (active vs inactive mutations):")
    logger.info("  BNN:   P=%.3f  R=%.3f  F1=%.3f  FPR=%.3f  (pred %d active / %d true active)",
                classification_bnn["precision"], classification_bnn["recall"],
                classification_bnn["f1"], classification_bnn["fpr"],
                classification_bnn["n_pred_active"], classification_bnn["n_true_active"])
    logger.info("  Null1: P=%.3f  R=%.3f  F1=%.3f  FPR=%.3f  (pred %d active / %d true active)",
                classification_null["precision"], classification_null["recall"],
                classification_null["f1"], classification_null["fpr"],
                classification_null["n_pred_active"], classification_null["n_true_active"])
    logger.info("  Null2: P=%.3f  R=%.3f  F1=%.3f  FPR=%.3f  (pred %d active / %d true active)",
                classification_null2.get("precision", 0), classification_null2.get("recall", 0),
                classification_null2.get("f1", 0), classification_null2.get("fpr", 0),
                classification_null2.get("n_pred_active", 0), classification_null2.get("n_true_active", 0))
    logger.info("NDCG (full):          %s",
                "  ".join(f"{k}=BNN:{ndcg_bnn[k]:.3f}/N1:{ndcg_null[k]:.3f}/N2:{ndcg_null2.get(k, float('nan')):.3f}"
                          for k in ndcg_bnn))
    logger.info("NDCG active-subs:     %s",
                "  ".join(f"{k}=BNN:{ndcg_active_sub_bnn[k]:.3f}/N1:{ndcg_active_sub_null[k]:.3f}/N2:{ndcg_active_sub_null2.get(k, float('nan')):.3f}"
                          for k in ndcg_active_sub_bnn))
    logger.info("Substrate AUROC:      BNN=%.3f  Null1=%.3f  Null2=%.3f",
                sub_discrimination_bnn["auroc"], sub_discrimination_null["auroc"],
                sub_discrimination_null2.get("auroc", float("nan")))
    for sub in sorted(topk_bnn.keys()):
        bnn_r = topk_bnn[sub]["recovery"]
        null_r = topk_null.get(sub, {}).get("recovery", float("nan"))
        null2_r = topk_null2.get(sub, {}).get("recovery", float("nan"))
        logger.info("  Top-5 recovery %s: BNN=%.2f  Null1=%.2f  Null2=%.2f",
                     sub, bnn_r, null_r, null2_r)

    # Log above-floor and selection regret
    logger.info("── Above-Floor Metrics (y > %.2f) ──", floor_threshold)
    logger.info("  BNN:   MAE=%.4f  Spearman=%.4f  n=%d",
                above_floor_bnn.get("mae", float("nan")),
                above_floor_bnn.get("spearman", float("nan")),
                above_floor_bnn.get("n_above_floor", 0))
    logger.info("  Null1: MAE=%.4f  Spearman=%.4f",
                above_floor_null.get("mae", float("nan")),
                above_floor_null.get("spearman", float("nan")))
    logger.info("  Null2: MAE=%.4f  Spearman=%.4f",
                above_floor_null2.get("mae", float("nan")),
                above_floor_null2.get("spearman", float("nan")))
    logger.info("── Selection Regret ──")
    for k in [5, 10, 25, 50]:
        bnn_pct = regret_bnn.get(f"top{k}_pct_of_optimal", float("nan"))
        null_pct = regret_null.get(f"top{k}_pct_of_optimal", float("nan"))
        null2_pct = regret_null2.get(f"top{k}_pct_of_optimal", float("nan"))
        logger.info("  Top-%d: BNN=%.3f  Null1=%.3f  Null2=%.3f  (pct of oracle)",
                     k, bnn_pct, null_pct, null2_pct)

    if hurdle_met is not None:
        logger.info("── Hurdle Metrics ──")
        logger.info("  AUROC=%.3f  Acc=%.3f  P=%.3f  R=%.3f  F1=%.3f",
                    hurdle_met.get("auroc", float("nan")),
                    hurdle_met.get("accuracy", float("nan")),
                    hurdle_met.get("precision", float("nan")),
                    hurdle_met.get("recall", float("nan")),
                    hurdle_met.get("f1", float("nan")))

    logger.info("Aggregation mode: %s", params.get("inference_aggregation", "nearest"))

    # Full-dataset LDS trace (appended after fold loop)
    if use_lds:
        y_all = df["log_fc"].values.astype(np.float32)
        w_all = compute_lds_weights(y_all, use_lds=True,
                                    n_bins=lds_cfg.get("n_bins", 50),
                                    kernel_size=lds_cfg.get("kernel_size", 5),
                                    sigma=lds_cfg.get("sigma", 2.0))
        lds_traces.append((y_all.copy(), w_all.copy(), "All data"))

    return (metrics, all_y_true, all_y_pred, all_epi, all_ale, all_tot,
            combined_df, fold_histories, fold_metrics_list, lds_traces)


# ═══════════════════════════════════════════════════════════════════════════
# Final Model Training (random split only)
# ═══════════════════════════════════════════════════════════════════════════

def train_final_model(
    df: pd.DataFrame,
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
    models_dir: Path,
    fold_histories: Optional[list] = None,
):
    """Train final BNN2 model on all data, save checkpoint + pipelines."""
    import joblib
    from bnns import BNNTrainer, TrainingConfig, HurdleConfig

    models_dir.mkdir(parents=True, exist_ok=True)

    # Determine n_epochs from CV
    if fold_histories:
        best_epochs = [h.best_epoch for h in fold_histories if h.best_epoch is not None]
        if best_epochs:
            median_best = int(np.median(best_epochs))
            n_epochs = min(int(median_best * 1.1) + 1, params["n_epochs"])
            n_epochs = max(n_epochs, params["kl_anneal_epochs"] + 1)
            logger.info("CV best epochs: %s (median=%d) → final n_epochs=%d",
                        best_epochs, median_best, n_epochs)
        else:
            n_epochs = params["n_epochs"]
    else:
        n_epochs = params["n_epochs"]

    # Expand all data to pairwise
    df_exp = expand_to_pairwise(df, substrate_meta, config)

    # Build features
    use_bnn1 = params.get("features", {}).get("x_aa", False)
    if use_bnn1:
        X_bnn1 = build_bnn1_input(df_exp, embeddings, bnn1_pipe_wt, bnn1_pipe_mut)
    groups = build_other_features(df_exp, embeddings, params, substrate_meta)

    # Fit preprocessing on all data
    pipelines = {}
    other_parts = []
    preproc_cfg = config.get("preprocessing", {})

    for group_name in sorted(groups.keys()):
        X_g = groups[group_name]
        if group_name in ("x_target_substrate", "x_ref_substrate"):
            cfg = preproc_cfg.get("x_substrate", {})
        elif group_name == "saprot_zs":
            cfg = preproc_cfg.get("saprot_zs", {})
        else:
            cfg = preproc_cfg.get(group_name, {})
        scaler_type = resolve_param(cfg.get("scaler", "none"),
                                    params.get(f"{group_name}_scaler"))
        pca_val = resolve_param(cfg.get("pca", None),
                                params.get(f"{group_name}_pca"))
        pca_components = parse_pca_value(pca_val)
        if X_g.shape[1] <= 2:
            pca_components = None
        pipe = build_preprocessing(scaler_type, pca_components)
        if pipe is not None:
            X_g_t = pipe.fit_transform(X_g).astype(np.float32)
        else:
            X_g_t = X_g.copy()
        pipelines[group_name] = pipe
        other_parts.append(X_g_t)

    X_other = np.concatenate(other_parts, axis=1)
    if use_bnn1:
        X_all = np.concatenate([X_bnn1, X_other], axis=1).astype(np.float32)
    else:
        X_all = X_other.astype(np.float32)

    # Delta targets: y = log_fc - log_fc_ref
    fc_ref_all = df_exp["log_fc_ref"].values.astype(np.float32)
    y_all = (df_exp["log_fc"].values - fc_ref_all).astype(np.float32)

    other_feature_dim = X_other.shape[1]

    # Build hurdle config (sub-parameters only; loss_type controls activation)
    hurdle_cfg = params.get("hurdle", {})
    hurdle_config = HurdleConfig(
        floor_threshold=hurdle_cfg.get("floor_threshold", -1.99),
        floor_value=hurdle_cfg.get("floor_value", -2.0),
        inference_threshold=hurdle_cfg.get("inference_threshold", 0.5),
    )

    # Build model
    model = build_bnn2_model(
        bnn1_hidden if use_bnn1 else None,
        bnn1_input_dim if use_bnn1 else 0,
        latent_dim if use_bnn1 else 0,
        other_feature_dim,
        params, device)

    training_config = TrainingConfig(
        n_epochs=n_epochs,
        learning_rate=params["learning_rate"],
        batch_size=params["batch_size"],
        kl_anneal_epochs=params["kl_anneal_epochs"],
        kl_weight=params["kl_weight"],
        early_stopping_patience=0,  # no val set
        n_inference_samples=params["n_inference_samples"],
        device=device,
        verbose=True,
        log_interval=50,
        loss_type=params.get("loss_type", "gaussian_nll"),
        hurdle=hurdle_config,
        null_reg_weight=params.get("null_reg_weight", 0.0),
    )

    trainer = BNNTrainer(model, training_config)
    X_t = torch.tensor(X_all, dtype=torch.float32)
    y_t = torch.tensor(y_all, dtype=torch.float32).unsqueeze(-1)

    # Precompute is_floor mask from absolute targets for hurdle
    is_floor_t = None
    if params.get("loss_type", "gaussian_nll") == "hurdle":
        y_abs_all = y_all + fc_ref_all
        is_floor_t = torch.tensor(
            y_abs_all <= hurdle_config.floor_threshold, dtype=torch.float32
        )

    trainer.fit(X_t, y_t, is_floor_train=is_floor_t)

    trainer.save(str(models_dir / "final_model.pt"))

    # Save preprocessing pipelines
    for name, pipe in pipelines.items():
        joblib.dump(pipe, models_dir / f"preprocessing_{name}.joblib")
    joblib.dump(bnn1_pipe_wt, models_dir / "bnn1_preprocessing_wt.joblib")
    joblib.dump(bnn1_pipe_mut, models_dir / "bnn1_preprocessing_mut.joblib")

    # Save model metadata
    meta = {
        "bnn1_input_dim": bnn1_input_dim,
        "latent_dim": latent_dim,
        "other_feature_dim": other_feature_dim,
        "n_training_rows": len(X_all),
        "n_epochs_trained": n_epochs,
        "other_feature_groups": sorted(groups.keys()),
        "other_feature_dims": {k: int(v.shape[1]) for k, v in groups.items()},
    }
    with open(models_dir / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved final model + %d preprocessing pipelines to %s",
                len(pipelines) + 2, models_dir)


# ═══════════════════════════════════════════════════════════════════════════
# Plotting dispatch
# ═══════════════════════════════════════════════════════════════════════════

def generate_plots(
    split_type: str,
    results_dir: Path,
    all_y_true: np.ndarray,
    all_y_pred: np.ndarray,
    all_epi: np.ndarray,
    all_ale: np.ndarray,
    all_tot: np.ndarray,
    combined_df: pd.DataFrame,
    fold_histories: list,
    metrics: dict,
    substrate_meta: dict,
    embeddings: dict,
    wt_activity: float = -1.9957,
    folds_info: Optional[dict] = None,
    repeat_metrics: Optional[list] = None,
    substrate_embedding_type: str = "morgan",
    null_embedding_type: str = "molformer",
    null_distance_metric: str = "cosine",
    params: Optional[dict] = None,
    supp_positions: Optional[set] = None,
):
    """Generate all plots for the given split type."""
    if params is None:
        params = {}
    all_subs = combined_df["substrate"].values
    all_pos = combined_df["position"].values
    all_null_pred = combined_df["_null_pred"].values if "_null_pred" in combined_df else None
    all_null2_pred = combined_df["_null2_pred"].values if "_null2_pred" in combined_df else None

    # ── Shared plots (all splits) ──

    plot_parity(all_y_true, all_y_pred, all_tot, all_subs,
                results_dir / "parity_plot.png")

    plot_residuals(all_y_true, all_y_pred, all_subs,
                   results_dir / "residuals_plot.png")

    plot_calibration(metrics["calibration"],
                     results_dir / "calibration_curve.png")

    plot_uncertainty_vs_error(all_y_true, all_y_pred, all_tot, all_subs,
                              results_dir / "uncertainty_vs_error.png")

    plot_uncertainty_decomposition(all_epi, all_ale, all_subs,
                                   results_dir / "uncertainty_decomposition.png")

    if fold_histories:
        fold_labels = None
        if folds_info and "held_out_substrates" in folds_info:
            fold_labels = folds_info["held_out_substrates"]
        plot_training_curves(fold_histories,
                             results_dir / "training_curves.png",
                             fold_labels=fold_labels)
        plot_loss_decomposition(fold_histories,
                                results_dir / "loss_decomposition.png")

    # ── Per-group metrics (all splits) ──

    per_sub_spearman = {k: v for k, v in metrics.get("per_substrate_spearman", {}).items()}
    per_sub_mae = {k: v for k, v in metrics.get("per_substrate_mae", {}).items()}
    per_sub_null_spearman = metrics.get("per_substrate_null_spearman") or {}
    per_sub_null_mae = metrics.get("per_substrate_null_mae") or {}
    per_sub_null2_spearman = metrics.get("per_substrate_null2_spearman") or {}
    per_sub_null2_mae = metrics.get("per_substrate_null2_mae") or {}
    per_pos_spearman = metrics.get("per_position_spearman", {})
    per_pos_range = metrics.get("per_position_range", {})

    plot_per_substrate_metrics(
        per_sub_spearman, per_sub_mae, substrate_meta,
        results_dir / "per_substrate_metrics.png",
        null_per_sub_spearman=per_sub_null_spearman or None,
        null_per_sub_mae=per_sub_null_mae or None,
        null2_per_sub_spearman=per_sub_null2_spearman or None,
        null2_per_sub_mae=per_sub_null2_mae or None,
    )

    # Convert position keys back to int for plotting
    per_pos_spearman_int = {int(k): v for k, v in per_pos_spearman.items()}
    per_pos_range_int = {int(k): v for k, v in per_pos_range.items()}
    plot_per_position_metrics(per_pos_spearman_int, per_pos_range_int,
                              results_dir / "per_position_metrics.png",
                              supp_positions=supp_positions)

    plot_substrate_position_heatmap(all_y_true, all_y_pred, all_subs, all_pos,
                                    "spearman",
                                    results_dir / "substrate_position_spearman.png",
                                    supp_positions=supp_positions)
    plot_substrate_position_heatmap(all_y_true, all_y_pred, all_subs, all_pos,
                                    "mae",
                                    results_dir / "substrate_position_mae.png",
                                    supp_positions=supp_positions)

    plot_substrate_parity_grid(all_y_true, all_y_pred, all_tot, all_subs,
                               results_dir / "substrate_parity_grid.png")

    # BNN vs null comparison grid (all splits, when null predictions available)
    if all_null_pred is not None:
        plot_substrate_parity_comparison_grid(
            all_y_true, all_y_pred, all_null_pred, all_subs,
            results_dir / "substrate_parity_comparison.png",
            y_pred_null2=all_null2_pred)

    # ── Split-specific plots ──

    if split_type == "substrate" and folds_info is not None:
        # Use the null model's auto-selected metric for the distance plot
        null_emb = embeddings[f"substrate_{null_embedding_type}"].astype(np.float64)
        distance_matrix = compute_pairwise_distances(null_emb, null_distance_metric)
        substrate_names = embeddings["substrate_names"]

        # Build per-substrate metrics for distance plot
        per_sub_metrics = {}
        null_per_sub_metrics = {}
        for sub in folds_info.get("held_out_substrates", []):
            if sub in per_sub_spearman:
                per_sub_metrics[sub] = {
                    "spearman": per_sub_spearman[sub],
                    "mae": per_sub_mae.get(sub, 0),
                }
            if sub in per_sub_null_spearman:
                null_per_sub_metrics[sub] = {
                    "spearman": per_sub_null_spearman.get(sub, 0),
                    "mae": per_sub_null_mae.get(sub, 0),
                }

        plot_distance_vs_performance(
            per_sub_metrics, distance_matrix, substrate_names,
            folds_info["held_out_substrates"],
            folds_info["train_substrates_per_fold"],
            results_dir / "distance_vs_performance.png",
            distance_label=f"{null_distance_metric.title()} Distance ({null_embedding_type})",
            null_per_sub_metrics=null_per_sub_metrics or None,
        )

        # Transfer matrix
        transfer_metrics = {}
        for sub in folds_info["held_out_substrates"]:
            if sub in per_sub_spearman:
                transfer_metrics[sub] = {
                    "spearman": per_sub_spearman[sub],
                    "mae": per_sub_mae.get(sub, 0),
                }
        plot_substrate_transfer_matrix(
            transfer_metrics, substrate_names, "spearman",
            results_dir / "substrate_transfer_spearman.png")
        plot_substrate_transfer_matrix(
            transfer_metrics, substrate_names, "mae",
            results_dir / "substrate_transfer_mae.png")

    if split_type == "singleshot" and repeat_metrics is not None:
        plot_singleshot_distributions(repeat_metrics,
                                      results_dir / "singleshot_distributions.png")

    # Acquisition recovery (all splits) — now with null model overlay
    plot_acquisition_recovery(
        all_y_true, all_y_pred, all_tot,
        wt_activity,
        results_dir / "acquisition_recovery.png",
        null_pred=all_null_pred,
        null2_pred=all_null2_pred,
        title=f"BNN2 {split_type}-split",
    )

    # Per-substrate acquisition recovery (same 3-panel plot, one per substrate)
    plot_per_substrate_acquisition_recovery(
        all_y_true, all_y_pred, all_tot,
        all_subs, substrate_meta,
        wt_activity,
        results_dir / "per_substrate_acquisition",
        null_pred=all_null_pred,
        null2_pred=all_null2_pred,
    )

    # Per-substrate top-k recovery (all splits)
    plot_per_substrate_topk_recovery(
        all_y_true, all_y_pred, all_subs, substrate_meta,
        results_dir / "per_substrate_topk_recovery.png",
        null_pred=all_null_pred,
        null2_pred=all_null2_pred,
        k_values=(3, 5, 10),
    )

    # Engineering value summary dashboard (all splits)
    plot_engineering_value_summary(
        all_y_true, all_y_pred, all_tot, all_subs,
        substrate_meta, results_dir / "engineering_value_summary.png",
        null_pred=all_null_pred,
        null2_pred=all_null2_pred,
        active_only_metrics=metrics.get("active_mutation_bnn"),
        null_active_only_metrics=metrics.get("active_mutation_null"),
        null2_active_only_metrics=metrics.get("active_mutation_null2"),
        ndcg_metrics={
            "bnn": metrics.get("ndcg_active_sub_bnn", {}),
            "null": metrics.get("ndcg_active_sub_null", {}),
            "null2": metrics.get("ndcg_active_sub_null2", {}),
        },
        substrate_discrimination=metrics.get("substrate_discrimination_bnn"),
        classification_bnn=metrics.get("classification_bnn"),
        classification_null=metrics.get("classification_null"),
        classification_null2=metrics.get("classification_null2"),
    )

    # Selection regret plot (always on)
    regret_bnn = metrics.get("selection_regret_bnn")
    regret_null = metrics.get("selection_regret_null")
    regret_null2 = metrics.get("selection_regret_null2")
    if regret_bnn:
        plot_selection_regret(
            regret_bnn, regret_null,
            results_dir / "selection_regret.png",
            regret_active_bnn=metrics.get("selection_regret_active_bnn"),
            regret_active_null=metrics.get("selection_regret_active_null"),
            regret_null2=regret_null2,
            regret_active_null2=metrics.get("selection_regret_active_null2"),
        )

    # Hurdle diagnostics plot (only when hurdle enabled)
    if metrics.get("hurdle_enabled") and "_cls_prob" in combined_df.columns:
        hurdle_met = metrics.get("hurdle", {})
        ft = params.get("hurdle", {}).get("floor_threshold", -1.99)
        plot_hurdle_diagnostics(
            all_y_true,
            combined_df["_cls_prob"].values,
            all_y_pred,
            hurdle_met,
            ft,
            results_dir / "hurdle_diagnostics.png",
        )

    logger.info("All plots saved to %s", results_dir)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BNN2: Multi-Substrate Activity Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 05_bnn2_multi_substrate.py --split random --device cuda:1
  python 05_bnn2_multi_substrate.py --split substrate
  python 05_bnn2_multi_substrate.py --split singleshot --n-singleshot-repeats 10
  python 05_bnn2_multi_substrate.py --split random --substrate-embedding-type molformer
        """,
    )
    parser.add_argument("--split", type=str, required=True,
                        choices=["random", "position", "substrate", "singleshot"],
                        help="CV split strategy")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)

    # BNN2 model
    parser.add_argument("--hidden-dims", type=str, default=None,
                        help="BNN2 head hidden dims as JSON, e.g. '[256, 128, 64]'")
    parser.add_argument("--prior-std", type=float, default=None)
    parser.add_argument("--dropout-rate", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--kl-weight", type=float, default=None)
    parser.add_argument("--substrate-embedding-type", type=str, default=None,
                        choices=["morgan", "maccs", "mordred", "molformer"])
    parser.add_argument("--x-aa-freeze", type=str, default=None,
                        choices=["none", "partial", "full"])
    parser.add_argument("--use-lds", action="store_true",
                        help="Apply Label Distribution Smoothing sample weights during training")

    # Preprocessing
    parser.add_argument("--x-substrate-scaler", type=str, default=None,
                        choices=["none", "standard"])
    parser.add_argument("--x-substrate-pca", type=str, default=None,
                        help="PCA for substrate embeddings: int, float 0<x<1, or 'none'")

    # Loss type
    parser.add_argument("--loss-type", type=str, default=None,
                        choices=["gaussian_nll", "mse", "hurdle"],
                        help="Loss function: gaussian_nll (default), mse, or hurdle")
    parser.add_argument("--null-reg-weight", type=float, default=None,
                        help="L2 penalty on mu toward zero (default: from config)")
    parser.add_argument("--floor-threshold", type=float, default=None,
                        help="Floor threshold for hurdle (default: -1.99)")
    parser.add_argument("--inference-threshold", type=float, default=None,
                        help="P(active) soft mixture center (default: 0.5)")

    # Aggregation
    parser.add_argument("--inference-aggregation", type=str, default=None,
                        choices=["nearest", "distance_weighted", "mean"],
                        help="Pairwise reference aggregation mode (default: nearest)")

    # Flags
    parser.add_argument("--skip-final-model", action="store_true",
                        help="Skip training final model (CV only)")
    parser.add_argument("--n-singleshot-repeats", type=int, default=10,
                        help="Number of repeats for singleshot split")
    parser.add_argument("--hyperparams", type=str, default=None,
                        help="Path to best_hyperparams.json from opt script. "
                             "Overrides config.yaml values with searched params.")
    parser.add_argument("--bnn1-model-dir", type=str, default=None,
                        help="Path to BNN1 model directory (default: results/03_.../models)")

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    t_start = time.time()

    split_type = args.split

    # 1. Setup
    results_dir = PROJECT_ROOT / "results" / "05_bnn2" / split_type
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir = results_dir / "models"

    setup_logging(results_dir / "run.log")

    logger.info("=" * 60)
    logger.info("05_bnn2_multi_substrate.py — %s split", split_type)
    logger.info("=" * 60)

    config = load_config(args.config)
    device = get_device(config, args.device)

    # 2. Load BNN1 backbone
    if args.bnn1_model_dir:
        bnn1_model_dir = Path(args.bnn1_model_dir)
    else:
        bnn1_model_dir = PROJECT_ROOT / "results" / "03_formaldehyde_regression" / "models"
    bnn1_hidden, bnn1_input_dim, latent_dim, bnn1_hp = load_bnn1_backbone(
        bnn1_model_dir, device)
    bnn1_pipe_wt, bnn1_pipe_mut = load_bnn1_preprocessing(bnn1_model_dir)

    # 3. Resolve params
    params = resolve_all_params(args, config)
    params["n_singleshot_repeats"] = args.n_singleshot_repeats
    # CLI --use-lds overrides config value
    if args.use_lds:
        params.setdefault("lds", {})["use_lds"] = True

    # Apply hyperparams from opt script JSON (overrides config + CLI)
    if args.hyperparams:
        hp_path = Path(args.hyperparams)
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
                # feat_fc_ref → features["fc_ref"]
                feat_name = k[len("feat_"):]
                params.setdefault("features", {})[feat_name] = v
            elif k == "use_lds":
                params.setdefault("lds", {})["use_lds"] = v
            elif k == "exclude_self_ref":
                # Pairwise setting — patch config so expand_to_pairwise sees it
                config.setdefault("bnn2", {}).setdefault("pairwise", {})[k] = v
            else:
                params[k] = v

    logger.info("Hyperparameters:")
    for k, v in params.items():
        if k != "features":
            logger.info("  %s: %s", k, v)
    logger.info("Feature toggles: %s", params.get("features", {}))

    # 4. Load data
    processed_dir = PROJECT_ROOT / config["data"]["output_dir"]
    df = load_multi_substrate_data(processed_dir)
    supp_positions = get_supplemental_positions(df)
    embeddings = load_all_embeddings(processed_dir)
    substrate_meta = load_substrate_metadata(processed_dir)

    # 4b. Auto-select best (embedding, distance_metric) for null model
    metric_selection = select_best_substrate_metric(df, embeddings)
    null_emb_type = metric_selection["best_embedding"]
    null_dist_metric = metric_selection["best_metric"]
    logger.info("Null model will use %s / %s (rho=%.3f vs functional distance)",
                null_emb_type, null_dist_metric, metric_selection["best_correlation"])

    # 5. Run CV
    (metrics, all_y_true, all_y_pred, all_epi, all_ale, all_tot,
     combined_df, fold_histories, fold_metrics_list, lds_traces) = evaluate_with_cv(
        df, embeddings, bnn1_hidden, bnn1_input_dim, latent_dim,
        bnn1_pipe_wt, bnn1_pipe_mut, params, config, device,
        split_type, substrate_meta,
        null_embedding_type=null_emb_type,
        null_distance_metric=null_dist_metric,
    )

    # 6. Build split-specific info for plots
    folds_info = None
    repeat_metrics = None

    if split_type == "substrate":
        # Collect held-out substrates and their training sets
        substrate_names = embeddings["substrate_names"]
        folds = make_substrate_folds(df)
        held_out = []
        train_subs_per_fold = {}
        for train_idx, val_idx in folds:
            val_sub = df.iloc[val_idx]["substrate"].unique()[0]
            train_sub = list(df.iloc[train_idx]["substrate"].unique())
            held_out.append(val_sub)
            train_subs_per_fold[val_sub] = train_sub
        folds_info = {
            "held_out_substrates": held_out,
            "train_substrates_per_fold": train_subs_per_fold,
        }

    if split_type == "singleshot":
        repeat_metrics = fold_metrics_list

    # 7. Plots
    epsilon = config["data"]["epsilon"]
    wt_activity = float(np.log10(1.0 + epsilon))
    logger.info("Generating plots...")
    generate_plots(
        split_type, results_dir,
        all_y_true, all_y_pred, all_epi, all_ale, all_tot,
        combined_df, fold_histories, metrics,
        substrate_meta, embeddings,
        wt_activity=wt_activity,
        folds_info=folds_info,
        repeat_metrics=repeat_metrics,
        substrate_embedding_type=params.get("substrate_embedding_type", "morgan"),
        null_embedding_type=null_emb_type,
        null_distance_metric=null_dist_metric,
        params=params,
        supp_positions=supp_positions,
    )
    if lds_traces:
        plot_lds_weights(lds_traces, results_dir / "lds_weights.png")

    # 8. Train final model on all data
    if split_type in ("random", "substrate") and not args.skip_final_model:
        logger.info("Training final model on all data...")
        train_final_model(
            df, embeddings, bnn1_hidden, bnn1_input_dim, latent_dim,
            bnn1_pipe_wt, bnn1_pipe_mut, params, config, device,
            substrate_meta, models_dir, fold_histories,
        )

    # 9. Save results
    metrics["null_model_embedding"] = null_emb_type
    metrics["null_model_distance_metric"] = null_dist_metric
    metrics["null_model_metric_correlation"] = metric_selection["best_correlation"]
    metrics["null_model_metric_selection"] = [
        {"embedding": e, "metric": m, "rho": r, "pvalue": p}
        for e, m, r, p in metric_selection["all_results"]
    ]
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    with open(results_dir / "hyperparams.json", "w") as f:
        json.dump({k: v for k, v in params.items() if k != "features"},
                  f, indent=2, default=str)

    with open(results_dir / "config_used.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Predictions CSV
    pred_cols = ["mutation_string", "substrate", "position",
                 "wt_aa", "mut_aa", "log_fc", "is_active_substrate",
                 "_y_pred", "_epi_std", "_ale_std", "_tot_std",
                 "_null_pred", "_null2_pred",
                 "n_refs", "fold"]
    if "_cls_prob" in combined_df.columns:
        pred_cols.append("_cls_prob")
    pred_df = combined_df[pred_cols].copy()
    rename_map = {
        "_y_pred": "y_pred",
        "_epi_std": "epistemic_std",
        "_ale_std": "aleatoric_std",
        "_tot_std": "total_std",
        "_null_pred": "null_pred",
        "_null2_pred": "null2_pred",
    }
    if "_cls_prob" in pred_df.columns:
        rename_map["_cls_prob"] = "cls_prob"
    pred_df.rename(columns=rename_map, inplace=True)
    pred_df["residual"] = pred_df["log_fc"] - pred_df["y_pred"]
    pred_df.to_csv(results_dir / "predictions.csv", index=False)
    logger.info("Saved predictions.csv (%d rows)", len(pred_df))

    # 10. Summary
    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("BNN2 %s-split Complete (%.1fs)", split_type, elapsed)
    logger.info("=" * 60)
    logger.info("── Standard Metrics ──")
    logger.info("MAE:          %.4f (null1: %.4f, %.1f%% impr | null2: %.4f, %.1f%% impr)",
                metrics["mae"], metrics["null_mae"],
                metrics["mae_improvement_over_null"] * 100,
                metrics["null2_mae"], metrics["mae_improvement_over_null2"] * 100)
    logger.info("R²:           %.4f", metrics["r2"])
    logger.info("Spearman ρ:   %.4f (null1: %.4f | null2: %.4f)",
                metrics["spearman_rho"], metrics["null_spearman"], metrics["null2_spearman"])
    logger.info("NLPD:         %.4f", metrics["nlpd"])
    logger.info("CRPS:         %.4f", metrics["crps"])
    logger.info("Sharpness:    %.4f", metrics["sharpness"])
    logger.info("Wt. sub ρ:    %.4f", metrics["weighted_per_substrate_spearman"])
    logger.info("Wt. pos ρ:    %.4f", metrics["weighted_per_position_spearman"])
    logger.info("── Engineering-Value Metrics ──")
    amb = metrics.get("active_mutation_bnn", {})
    amn = metrics.get("active_mutation_null", {})
    amn2 = metrics.get("active_mutation_null2", {})
    logger.info("Active mutations (log_fc > WT, n=%d):", amb.get("active_n", 0))
    logger.info("  MAE:      %.4f (null1: %.4f | null2: %.4f)",
                amb.get("active_mae", float("nan")), amn.get("active_mae", float("nan")),
                amn2.get("active_mae", float("nan")))
    logger.info("  Spearman: %.4f (null1: %.4f | null2: %.4f)",
                amb.get("active_spearman", float("nan")), amn.get("active_spearman", float("nan")),
                amn2.get("active_spearman", float("nan")))
    asb = metrics.get("active_substrate_bnn", {})
    asn = metrics.get("active_substrate_null", {})
    asn2 = metrics.get("active_substrate_null2", {})
    logger.info("Active substrates (n=%d):", asb.get("active_sub_n", 0))
    logger.info("  MAE:      %.4f (null1: %.4f | null2: %.4f)",
                asb.get("active_sub_mae", float("nan")), asn.get("active_sub_mae", float("nan")),
                asn2.get("active_sub_mae", float("nan")))
    cb = metrics.get("classification_bnn", {})
    cn = metrics.get("classification_null", {})
    cn2 = metrics.get("classification_null2", {})
    logger.info("Classification: BNN P=%.3f R=%.3f F1=%.3f | Null1 P=%.3f R=%.3f F1=%.3f | Null2 P=%.3f R=%.3f F1=%.3f",
                cb.get("precision", 0), cb.get("recall", 0), cb.get("f1", 0),
                cn.get("precision", 0), cn.get("recall", 0), cn.get("f1", 0),
                cn2.get("precision", 0), cn2.get("recall", 0), cn2.get("f1", 0))
    sdb = metrics.get("substrate_discrimination_bnn", {})
    sdn = metrics.get("substrate_discrimination_null", {})
    sdn2 = metrics.get("substrate_discrimination_null2", {})
    logger.info("Substrate AUROC:      %.3f (null1: %.3f | null2: %.3f)",
                sdb.get("auroc", float("nan")), sdn.get("auroc", float("nan")),
                sdn2.get("auroc", float("nan")))
    afb = metrics.get("above_floor_bnn", {})
    afn = metrics.get("above_floor_null", {})
    afn2 = metrics.get("above_floor_null2", {})
    logger.info("Above-floor (n=%d):  MAE BNN=%.4f N1=%.4f N2=%.4f  ρ BNN=%.4f N1=%.4f N2=%.4f",
                afb.get("n_above_floor", 0),
                afb.get("mae", float("nan")), afn.get("mae", float("nan")),
                afn2.get("mae", float("nan")),
                afb.get("spearman", float("nan")), afn.get("spearman", float("nan")),
                afn2.get("spearman", float("nan")))
    srb = metrics.get("selection_regret_bnn", {})
    srn = metrics.get("selection_regret_null", {})
    srn2 = metrics.get("selection_regret_null2", {})
    logger.info("Selection regret (top-10): BNN=%.3f  Null1=%.3f  Null2=%.3f",
                srb.get("top10_pct_of_optimal", float("nan")),
                srn.get("top10_pct_of_optimal", float("nan")),
                srn2.get("top10_pct_of_optimal", float("nan")))
    logger.info("Aggregation mode: %s", metrics.get("inference_aggregation", "nearest"))
    if metrics.get("hurdle_enabled"):
        hm = metrics.get("hurdle", {})
        logger.info("Hurdle: AUROC=%.3f  F1=%.3f", hm.get("auroc", float("nan")), hm.get("f1", float("nan")))
    if split_type == "random" and not args.skip_final_model:
        logger.info("Final model: %s/", models_dir)


if __name__ == "__main__":
    main()
