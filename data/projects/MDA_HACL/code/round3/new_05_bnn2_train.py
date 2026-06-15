#!/usr/bin/env python
"""
new_05_bnn2_train.py — BNN2 CV Training + Raw Pairwise Predictions
===================================================================

Trains the BNN2 composite model under a chosen CV split and writes **raw
pairwise predictions** — one row per (test mutation, reference substrate,
target substrate) — to disk. Aggregation across references, metric
computation, and plotting are deferred to `new_05b_bnn2_score.py`. This
separation lets us sweep metrics and figures without re-training.

Extra knobs not in the legacy script:
  --target-substrate NAME         run a single fold holding out only this
                                   substrate (used by the sensitivity sweep)
  --subsample-train-substrates …  JSON list of training substrates to keep;
                                   everything else in the training set is
                                   dropped prior to pairwise expansion
  --run-id NAME                   directory name under
                                   results/new_05_bnn2/{split}/

Outputs (under results/new_05_bnn2/{split}/{run_id}/):
  train_metadata.json         split type, held-out/training substrate lists,
                              null-embedding + metric, seed, hyperparams
  hyperparams.json            resolved params
  pairwise_predictions.csv    one row per (mut, ref, target) test triplet
  train_lookup.csv            fold, mutation_string, substrate, log_fc (for
                              null lookups at score time)
  training_histories.json     per-fold training curves (lightweight summary)
  models/                     optional final-model artifacts (substrate/random
                              splits only, unless --skip-final-model)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from importlib.util import spec_from_file_location, module_from_spec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution + common module import
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent        # code/round3/
PROJECT_ROOT = SCRIPT_DIR.parent.parent             # MDA_HACL/
sys.path.insert(0, str(SCRIPT_DIR.parent))          # for `from bnns import ...`

_common_spec = spec_from_file_location(
    "bnn2_common", SCRIPT_DIR / "05_bnn2_common.py")
_common = module_from_spec(_common_spec)
_common_spec.loader.exec_module(_common)

# Config / logging / device
load_config = _common.load_config
resolve_param = _common.resolve_param
resolve_config_block = _common.resolve_config_block
get_device = _common.get_device
parse_pca_value = _common.parse_pca_value
setup_logging = _common.setup_logging
# Data / features
load_multi_substrate_data = _common.load_multi_substrate_data
get_supplemental_positions = _common.get_supplemental_positions
load_all_embeddings = _common.load_all_embeddings
load_substrate_metadata = _common.load_substrate_metadata
expand_to_pairwise = _common.expand_to_pairwise
add_ref_distances = _common.add_ref_distances
build_bnn1_input = _common.build_bnn1_input
build_other_features = _common.build_other_features
# Preprocessing
build_preprocessing = _common.build_preprocessing
preprocess_other_features = _common.preprocess_other_features
# BNN1 backbone + training
load_bnn1_backbone = _common.load_bnn1_backbone
load_bnn1_preprocessing = _common.load_bnn1_preprocessing
build_bnn2_model = _common.build_bnn2_model
train_and_evaluate_fold = _common.train_and_evaluate_fold
compute_lds_weights = _common.compute_lds_weights
# Null-mode distance selection
compute_pairwise_distances = _common.compute_pairwise_distances
select_best_substrate_metric = _common.select_best_substrate_metric


# ═══════════════════════════════════════════════════════════════════════════
# Split functions (duplicated from 05_bnn2_multi_substrate.py to keep this
# script self-contained; the legacy script is untouched)
# ═══════════════════════════════════════════════════════════════════════════

def make_random_folds(df: pd.DataFrame, n_folds: int, seed: int):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in kf.split(df)]


def make_position_folds(df: pd.DataFrame, n_folds: int, seed: int):
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=n_folds)
    return [(tr, va) for tr, va in gkf.split(df, groups=df["position"].values)]


def make_substrate_folds(df: pd.DataFrame, substrate_meta: Optional[dict] = None):
    """Leave-one-substrate-out folds. If `substrate_meta` is provided, only
    substrates with `is_active=True` become held-out folds. Inactive
    substrates remain in the training set across every active fold so they
    keep working as ordinary training/reference candidates — they're just
    never the test set (their constant log_fc makes them a degenerate
    target for ranking metrics).
    """
    from sklearn.model_selection import LeaveOneGroupOut
    logo = LeaveOneGroupOut()
    all_folds = [(tr, va) for tr, va in logo.split(df, groups=df["substrate"].values)]
    if substrate_meta is None:
        return all_folds
    out = []
    for tr, va in all_folds:
        held = df.iloc[va]["substrate"].unique()
        if len(held) == 1 and substrate_meta.get(held[0], {}).get("is_active", True):
            out.append((tr, va))
    return out


def make_singleshot_folds(df: pd.DataFrame, n_repeats: int, seed: int):
    rng = np.random.RandomState(seed)
    folds = []
    for substrate in sorted(df["substrate"].unique()):
        other_idx = df.index[df["substrate"] != substrate].tolist()
        held_out = df[df["substrate"] == substrate]
        for _ in range(n_repeats):
            shot_idx, val_idx = [], []
            for _, group in held_out.groupby("position"):
                indices = group.index.tolist()
                if len(indices) <= 1:
                    shot_idx.extend(indices)
                    continue
                chosen = rng.choice(indices)
                shot_idx.append(chosen)
                val_idx.extend([i for i in indices if i != chosen])
            folds.append((np.array(other_idx + shot_idx), np.array(val_idx)))
    return folds


# ═══════════════════════════════════════════════════════════════════════════
# Parameter resolution (subset of legacy resolve_all_params; kept here so the
# training script has no import-cycle with the legacy main script)
# ═══════════════════════════════════════════════════════════════════════════

def resolve_all_params(args: argparse.Namespace, config: dict) -> dict:
    bnn2 = config["bnn2"]
    train = bnn2["training"]
    preproc = config["preprocessing"]

    hidden_dims_cli = json.loads(args.hidden_dims) if args.hidden_dims else None
    x_sub_pca_cli = parse_pca_value(args.x_substrate_pca) if args.x_substrate_pca is not None else None
    x_sub_pca_cfg = parse_pca_value(resolve_param(preproc.get("x_substrate", {}).get("pca")))

    params = {
        "hidden_dims":             resolve_param(bnn2["hidden_dims"], hidden_dims_cli),
        "prior_std":               resolve_param(bnn2["prior_std"], args.prior_std),
        "dropout_rate":            resolve_param(bnn2["dropout_rate"], args.dropout_rate),
        "activation":              resolve_param(bnn2["activation"]),
        "x_aa_freeze":             resolve_param(bnn2["x_aa_freeze"], args.x_aa_freeze),
        "substrate_embedding_type": resolve_param(bnn2["substrate_embedding_type"],
                                                   args.substrate_embedding_type),
        "learning_rate":           resolve_param(train["learning_rate"], args.learning_rate),
        "kl_weight":               resolve_param(train["kl_weight"], args.kl_weight),
        "batch_size":              resolve_param(train["batch_size"]),
        "kl_anneal_epochs":        resolve_param(train["kl_anneal_epochs"]),
        "n_epochs":                resolve_param(train["n_epochs"]),
        "early_stopping_patience": resolve_param(train["early_stopping_patience"]),
        "n_inference_samples":     resolve_param(train["n_inference_samples"]),
        "clip_grad_norm":          resolve_param(train.get("clip_grad_norm", {"value": 1.0})),
        "features":                resolve_config_block(bnn2.get("features", {})),
        "lds":                     resolve_config_block(config.get("bnn2", {}).get("lds", {})),
    }

    params["x_substrate_scaler"] = resolve_param(
        preproc.get("x_substrate", {}).get("scaler", "none"), args.x_substrate_scaler)
    params["x_substrate_pca"] = x_sub_pca_cli if args.x_substrate_pca is not None else x_sub_pca_cfg
    params["saprot_zs_scaler"] = resolve_param(preproc.get("saprot_zs", {}).get("scaler", "none"))
    params["esm_wt_scaler"] = resolve_param(preproc.get("esm_wt", {}).get("scaler", "standard"))
    params["esm_wt_pca"] = parse_pca_value(resolve_param(preproc.get("esm_wt", {}).get("pca")))
    params["esm_mut_scaler"] = resolve_param(preproc.get("esm_mut", {}).get("scaler", "standard"))
    params["esm_mut_pca"] = parse_pca_value(resolve_param(preproc.get("esm_mut", {}).get("pca")))

    loss_type_cfg = bnn2.get("loss_type", {"value": "gaussian_nll"})
    params["loss_type"] = resolve_param(loss_type_cfg, getattr(args, "loss_type", None))

    null_reg_cfg = bnn2.get("training", {}).get("null_reg_weight", {"value": 0.0})
    params["null_reg_weight"] = resolve_param(null_reg_cfg, getattr(args, "null_reg_weight", None))

    lv_floor_cfg = bnn2.get("training", {}).get("log_var_floor", {"value": None})
    params["log_var_floor"] = resolve_param(lv_floor_cfg, getattr(args, "log_var_floor", None))

    pred_floor_cfg = bnn2.get("training", {}).get("prediction_floor", {"value": None})
    params["prediction_floor"] = resolve_param(pred_floor_cfg, getattr(args, "prediction_floor", None))

    hurdle_cfg = bnn2.get("hurdle", {})
    params["hurdle"] = {
        "floor_threshold": getattr(args, "floor_threshold", None) or hurdle_cfg.get("floor_threshold", -1.99),
        "floor_value": hurdle_cfg.get("floor_value", -2.0),
        "inference_threshold": getattr(args, "inference_threshold", None) or hurdle_cfg.get("inference_threshold", 0.5),
    }

    pairwise_cfg = resolve_config_block(bnn2.get("pairwise", {}))
    params["distance_weight_temperature"] = pairwise_cfg.get("distance_weight_temperature", 1.0)

    return params


# ═══════════════════════════════════════════════════════════════════════════
# CV runner — produces raw pairwise predictions
# ═══════════════════════════════════════════════════════════════════════════

# Columns of the expanded DataFrame that we persist alongside predictions.
# Anything the scorer or the null models might look up must be included.
_PAIRWISE_META_COLS = [
    "mutation_string", "position", "wt_aa", "mut_aa",
    "substrate",          # target substrate (retained for grouping)
    "ref_substrate",
    "fold_change", "log_fc",
    "fc_ref", "log_fc_ref",
    "is_active_substrate", "ref_type",
]


def _filter_training_substrates(
    df_train: pd.DataFrame,
    subsample: Optional[List[str]],
    substrate_meta: dict,
) -> pd.DataFrame:
    """Restrict training rows to the requested substrate subsample.

    Treats every substrate (active or inactive) as ordinary. When ``subsample``
    is None the full ``df_train`` is returned (the comprehensive run case).
    When provided, ONLY those names are kept — inactives are no longer
    silently injected.
    """
    if not subsample:
        return df_train
    keep = set(subsample)
    missing = keep - set(df_train["substrate"].unique())
    if missing:
        logger.warning("  Subsample: substrates %s not present in this fold's "
                       "training set; they will be ignored", sorted(missing))
    filtered = df_train[df_train["substrate"].isin(keep)].reset_index(drop=True)
    kept_actives = sorted(s for s in filtered["substrate"].unique()
                          if substrate_meta.get(s, {}).get("is_active", True))
    kept_inactives = sorted(s for s in filtered["substrate"].unique()
                            if not substrate_meta.get(s, {}).get("is_active", True))
    logger.info("  Subsample: kept %d/%d training rows — active=%s, inactive=%s",
                len(filtered), len(df_train), kept_actives, kept_inactives)
    return filtered


def run_cv_and_collect_predictions(
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
    target_substrate: Optional[str],
    subsample_train_substrates: Optional[List[str]],
    null_embedding_type: str,
    null_distance_metric: str,
):
    """Run CV; return (pairwise_df, train_lookup_df, fold_summaries, fold_histories).

    `target_substrate` (only meaningful for split_type='substrate'): if set,
    only the fold holding out THIS substrate is run, and all other folds are
    skipped. For other splits it is ignored.
    """
    from bnns.model import HurdleUncertaintyEstimate
    seed = config.get("cv", {}).get("seed", 42)
    n_folds_cfg = config.get("cv", {}).get("n_folds", 5)

    if split_type == "random":
        folds = make_random_folds(df, n_folds_cfg, seed)
    elif split_type == "position":
        folds = make_position_folds(df, n_folds_cfg, seed)
    elif split_type == "substrate":
        folds = make_substrate_folds(df, substrate_meta=substrate_meta)
    elif split_type == "singleshot":
        n_repeats = params.get("n_singleshot_repeats", 10)
        folds = make_singleshot_folds(df, n_repeats, seed)
    else:
        raise ValueError(f"Unknown split type: {split_type}")

    logger.info("Split strategy: %s → %d folds", split_type, len(folds))

    pairwise_frames: list = []
    train_lookup_frames: list = []
    fold_summaries: list = []
    fold_histories: list = []

    lds_cfg = params.get("lds", {})
    use_lds = lds_cfg.get("use_lds", False)
    hurdle_enabled = params.get("loss_type", "gaussian_nll") == "hurdle"

    # Substrate embedding for _ref_distance (kept identical to the legacy script
    # so BNN aggregation reproduces exactly when the same embeddings are used).
    sub_emb_type = params.get("substrate_embedding_type", "morgan")
    sub_emb = embeddings[f"substrate_{sub_emb_type}"].astype(np.float64)
    ref_dist_matrix = compute_pairwise_distances(sub_emb, "cosine")
    substrate_names = embeddings["substrate_names"]

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        df_val_full = df.iloc[val_idx].reset_index(drop=True)

        # --target-substrate gate (substrate-split only): skip folds that
        # aren't holding out the requested target.
        if target_substrate is not None and split_type == "substrate":
            val_subs = df_val_full["substrate"].unique()
            if target_substrate not in val_subs:
                continue

        logger.info("─── Fold %d/%d ───", fold_i + 1, len(folds))

        df_train_full = df.iloc[train_idx].reset_index(drop=True)
        df_train = _filter_training_substrates(
            df_train_full, subsample_train_substrates, substrate_meta)
        df_val = df_val_full

        if len(df_train) == 0:
            logger.warning("  Fold %d: training set empty after subsampling — skipping",
                           fold_i + 1)
            continue

        logger.info("  Base: train=%d, val=%d", len(df_train), len(df_val))

        if split_type == "substrate":
            logger.info("  Held-out substrate(s): %s",
                        sorted(df_val["substrate"].unique()))
            logger.info("  Training substrates:   %s",
                        sorted(df_train["substrate"].unique()))

        # Pairwise expansion — same rules as the legacy script
        train_fc_lookup = {
            (row["mutation_string"], row["substrate"]): row["fold_change"]
            for _, row in df_train.iterrows()
        }
        df_train_exp = expand_to_pairwise(df_train, substrate_meta, config)
        val_ref_lookup = train_fc_lookup if split_type != "position" else None
        df_val_exp = expand_to_pairwise(df_val, substrate_meta, config,
                                        ref_fc_lookup=val_ref_lookup)

        if len(df_train_exp) == 0 or len(df_val_exp) == 0:
            logger.warning("  Fold %d: empty after expansion — skipping", fold_i + 1)
            continue

        add_ref_distances(df_train_exp, ref_dist_matrix, substrate_names)
        add_ref_distances(df_val_exp, ref_dist_matrix, substrate_names)

        logger.info("  Expanded: train=%d, val=%d", len(df_train_exp), len(df_val_exp))

        # Build features
        use_bnn1 = params.get("features", {}).get("x_aa", False)
        if use_bnn1:
            X_bnn1_train = build_bnn1_input(df_train_exp, embeddings,
                                            bnn1_pipe_wt, bnn1_pipe_mut)
            X_bnn1_val = build_bnn1_input(df_val_exp, embeddings,
                                          bnn1_pipe_wt, bnn1_pipe_mut)
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

        fc_ref_train = df_train_exp["log_fc_ref"].values.astype(np.float32)
        fc_ref_val = df_val_exp["log_fc_ref"].values.astype(np.float32)
        y_train = (df_train_exp["log_fc"].values - fc_ref_train).astype(np.float32)
        y_val = (df_val_exp["log_fc"].values - fc_ref_val).astype(np.float32)

        fold_metrics, estimates, history = train_and_evaluate_fold(
            X_train, y_train, X_val, y_val,
            bnn1_hidden if use_bnn1 else None,
            bnn1_input_dim if use_bnn1 else 0,
            latent_dim if use_bnn1 else 0,
            X_other_train.shape[1],
            params, device, return_predictions=True,
            fc_ref_train=fc_ref_train, fc_ref_val=fc_ref_val,
        )

        fold_summaries.append({"fold": fold_i, "metrics": fold_metrics})
        if history is not None:
            fold_histories.append(history)

        # Extract pairwise predictions on the val set
        y_pred_exp = estimates.mean.cpu().numpy().squeeze(-1)
        epi_std_exp = estimates.epistemic_std.cpu().numpy().squeeze(-1)
        ale_std_exp = estimates.aleatoric_std.cpu().numpy().squeeze(-1)
        tot_std_exp = estimates.total_std.cpu().numpy().squeeze(-1)
        cls_prob_exp = None
        if hurdle_enabled and isinstance(estimates, HurdleUncertaintyEstimate):
            cls_prob_exp = estimates.cls_prob.cpu().numpy().squeeze(-1)

        # Build per-fold pairwise DataFrame
        meta_cols_present = [c for c in _PAIRWISE_META_COLS if c in df_val_exp.columns]
        pairwise_rows = df_val_exp[meta_cols_present].copy()
        pairwise_rows["_ref_distance"] = df_val_exp["_ref_distance"].values
        pairwise_rows["y_pred"] = y_pred_exp
        pairwise_rows["epi_std"] = epi_std_exp
        pairwise_rows["ale_std"] = ale_std_exp
        pairwise_rows["tot_std"] = tot_std_exp
        if cls_prob_exp is not None:
            pairwise_rows["cls_prob"] = cls_prob_exp
        pairwise_rows["fold"] = fold_i
        pairwise_frames.append(pairwise_rows)

        # Per-fold training lookup (only the kept training substrates)
        train_lookup_frames.append(pd.DataFrame({
            "fold": fold_i,
            "mutation_string": df_train["mutation_string"].values,
            "substrate": df_train["substrate"].values,
            "log_fc": df_train["log_fc"].values.astype(np.float32),
        }))

        if use_lds:
            # (Training curve only — LDS diagnostics not persisted here.)
            pass

    if not pairwise_frames:
        raise RuntimeError("No folds produced predictions — check your data, "
                            "split strategy, and subsample list.")

    pairwise_df = pd.concat(pairwise_frames, ignore_index=True)
    train_lookup_df = pd.concat(train_lookup_frames, ignore_index=True)
    return pairwise_df, train_lookup_df, fold_summaries, fold_histories


# ═══════════════════════════════════════════════════════════════════════════
# Final model training (optional; identical to legacy script in effect)
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
    import joblib
    from bnns import BNNTrainer, TrainingConfig, HurdleConfig
    models_dir.mkdir(parents=True, exist_ok=True)

    if fold_histories:
        best_epochs = [h.best_epoch for h in fold_histories if h.best_epoch is not None]
        if best_epochs:
            median_best = int(np.median(best_epochs))
            n_epochs = min(int(median_best * 1.1) + 1, params["n_epochs"])
            n_epochs = max(n_epochs, params["kl_anneal_epochs"] + 1)
        else:
            n_epochs = params["n_epochs"]
    else:
        n_epochs = params["n_epochs"]

    df_exp = expand_to_pairwise(df, substrate_meta, config)
    use_bnn1 = params.get("features", {}).get("x_aa", False)
    if use_bnn1:
        X_bnn1 = build_bnn1_input(df_exp, embeddings, bnn1_pipe_wt, bnn1_pipe_mut)
    groups = build_other_features(df_exp, embeddings, params, substrate_meta)

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
    X_all = np.concatenate([X_bnn1, X_other], axis=1).astype(np.float32) if use_bnn1 else X_other.astype(np.float32)
    fc_ref_all = df_exp["log_fc_ref"].values.astype(np.float32)
    y_all = (df_exp["log_fc"].values - fc_ref_all).astype(np.float32)

    hurdle_cfg = params.get("hurdle", {})
    hurdle_config = HurdleConfig(
        floor_threshold=hurdle_cfg.get("floor_threshold", -1.99),
        floor_value=hurdle_cfg.get("floor_value", -2.0),
        inference_threshold=hurdle_cfg.get("inference_threshold", 0.5),
    )

    model = build_bnn2_model(
        bnn1_hidden if use_bnn1 else None,
        bnn1_input_dim if use_bnn1 else 0,
        latent_dim if use_bnn1 else 0,
        X_other.shape[1],
        params, device,
    )
    training_config = TrainingConfig(
        n_epochs=n_epochs,
        learning_rate=params["learning_rate"],
        batch_size=params["batch_size"],
        kl_anneal_epochs=params["kl_anneal_epochs"],
        kl_weight=params["kl_weight"],
        early_stopping_patience=0,
        n_inference_samples=params["n_inference_samples"],
        device=device, verbose=True, log_interval=50,
        loss_type=params.get("loss_type", "gaussian_nll"),
        hurdle=hurdle_config,
        null_reg_weight=params.get("null_reg_weight", 0.0),
    )

    trainer = BNNTrainer(model, training_config)
    X_t = torch.tensor(X_all, dtype=torch.float32)
    y_t = torch.tensor(y_all, dtype=torch.float32).unsqueeze(-1)
    is_floor_t = None
    if params.get("loss_type", "gaussian_nll") == "hurdle":
        y_abs_all = y_all + fc_ref_all
        is_floor_t = torch.tensor(
            y_abs_all <= hurdle_config.floor_threshold, dtype=torch.float32)
    trainer.fit(X_t, y_t, is_floor_train=is_floor_t)
    trainer.save(str(models_dir / "final_model.pt"))

    for name, pipe in pipelines.items():
        joblib.dump(pipe, models_dir / f"preprocessing_{name}.joblib")
    joblib.dump(bnn1_pipe_wt, models_dir / "bnn1_preprocessing_wt.joblib")
    joblib.dump(bnn1_pipe_mut, models_dir / "bnn1_preprocessing_mut.joblib")

    with open(models_dir / "model_metadata.json", "w") as f:
        json.dump({
            "bnn1_input_dim": bnn1_input_dim,
            "latent_dim": latent_dim,
            "other_feature_dim": X_other.shape[1],
            "n_training_rows": len(X_all),
            "n_epochs_trained": n_epochs,
            "other_feature_groups": sorted(groups.keys()),
            "other_feature_dims": {k: int(v.shape[1]) for k, v in groups.items()},
        }, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BNN2 CV training — saves raw pairwise predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python new_05_bnn2_train.py --split substrate
  python new_05_bnn2_train.py --split substrate --target-substrate Formaldehyde \\
                              --subsample-train-substrates '["Acetaldehyde","Butanal"]'
        """,
    )
    parser.add_argument("--split", type=str, required=True,
                        choices=["random", "position", "substrate", "singleshot"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)

    # BNN2 hyperparameters (same as legacy)
    parser.add_argument("--hidden-dims", type=str, default=None)
    parser.add_argument("--prior-std", type=float, default=None)
    parser.add_argument("--dropout-rate", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--kl-weight", type=float, default=None)
    parser.add_argument("--substrate-embedding-type", type=str, default=None,
                        choices=["morgan", "maccs", "mordred", "molformer"])
    parser.add_argument("--x-aa-freeze", type=str, default=None,
                        choices=["none", "partial", "full"])
    parser.add_argument("--use-lds", action="store_true")
    parser.add_argument("--x-substrate-scaler", type=str, default=None,
                        choices=["none", "standard"])
    parser.add_argument("--x-substrate-pca", type=str, default=None)
    parser.add_argument("--loss-type", type=str, default=None,
                        choices=["gaussian_nll", "mse", "hurdle"])
    parser.add_argument("--null-reg-weight", type=float, default=None)
    parser.add_argument("--floor-threshold", type=float, default=None)
    parser.add_argument("--inference-threshold", type=float, default=None)

    # Training control
    parser.add_argument("--skip-final-model", action="store_true")
    parser.add_argument("--n-singleshot-repeats", type=int, default=10)
    parser.add_argument("--hyperparams", type=str, default=None,
                        help="Path to best_hyperparams.json from opt script")
    parser.add_argument("--bnn1-model-dir", type=str, default=None)

    # New: sensitivity-sweep knobs
    parser.add_argument("--target-substrate", type=str, default=None,
                        help="Substrate split only: restrict CV to the single "
                             "fold that holds out this substrate")
    parser.add_argument("--subsample-train-substrates", type=str, default=None,
                        help="JSON list of training substrates to keep (e.g. "
                             "'[\"Formaldehyde\",\"Acetaldehyde\"]'); everything "
                             "else is dropped before pairwise expansion")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Sub-directory name under results/new_05_bnn2/{split}/"
                             "; auto-generated if omitted")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Override root directory for outputs (default: "
                             "PROJECT_ROOT/results/new_05_bnn2)")

    return parser.parse_args()


def _resolve_run_id(args: argparse.Namespace, subsample: Optional[List[str]]) -> str:
    if args.run_id:
        return args.run_id
    parts = ["run"]
    if args.target_substrate:
        parts.append(f"S={args.target_substrate}")
    if subsample:
        import hashlib
        sub_key = ",".join(sorted(subsample))
        parts.append(f"k={len(subsample)}")
        parts.append(f"h={hashlib.md5(sub_key.encode()).hexdigest()[:6]}")
    if len(parts) == 1:
        parts.append(time.strftime("%Y%m%d_%H%M%S"))
    return "_".join(parts)


def main():
    args = parse_args()
    t_start = time.time()
    split_type = args.split

    # Parse subsample JSON up-front so we can use it in the run_id
    subsample: Optional[List[str]] = None
    if args.subsample_train_substrates:
        subsample = json.loads(args.subsample_train_substrates)
        if not isinstance(subsample, list) or not all(isinstance(s, str) for s in subsample):
            raise ValueError("--subsample-train-substrates must be a JSON list of strings")

    # Output directory
    output_root = Path(args.output_root) if args.output_root else (
        PROJECT_ROOT / "results" / "new_05_bnn2")
    run_id = _resolve_run_id(args, subsample)
    results_dir = output_root / split_type / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir = results_dir / "models"

    setup_logging(results_dir / "run.log")
    logger.info("=" * 60)
    logger.info("new_05_bnn2_train.py — %s split — run=%s", split_type, run_id)
    logger.info("=" * 60)

    config = load_config(args.config)
    device = get_device(config, args.device)

    # Params (resolved first so we can decide whether to load BNN1)
    params = resolve_all_params(args, config)
    params["n_singleshot_repeats"] = args.n_singleshot_repeats
    if args.use_lds:
        params.setdefault("lds", {})["use_lds"] = True

    if args.hyperparams:
        hp_path = Path(args.hyperparams)
        logger.info("Loading hyperparams from %s", hp_path)
        with open(hp_path) as f:
            hp = json.load(f)
        _skip = {"optimization_objective", "objective_value", "raw_objective", "trial_number"}
        _skip.update(k for k in hp if k.startswith(("mean_cv_", "fold_")))
        for k, v in hp.items():
            if k in _skip:
                continue
            if k.startswith("feat_"):
                params.setdefault("features", {})[k[len("feat_"):]] = v
            elif k == "use_lds":
                params.setdefault("lds", {})["use_lds"] = v
            elif k == "exclude_self_ref":
                config.setdefault("bnn2", {}).setdefault("pairwise", {})[k] = v
            else:
                params[k] = v

    logger.info("Hyperparameters:")
    for k, v in params.items():
        if k != "features":
            logger.info("  %s: %s", k, v)
    logger.info("Feature toggles: %s", params.get("features", {}))

    # BNN1 backbone — only required when x_aa feature is enabled.
    use_bnn1 = bool(params.get("features", {}).get("x_aa", False))
    bnn1_model_dir = Path(args.bnn1_model_dir) if args.bnn1_model_dir else (
        PROJECT_ROOT / "results" / "03_formaldehyde_regression" / "models")
    if use_bnn1:
        bnn1_hidden, bnn1_input_dim, latent_dim, _ = load_bnn1_backbone(bnn1_model_dir, device)
        bnn1_pipe_wt, bnn1_pipe_mut = load_bnn1_preprocessing(bnn1_model_dir)
    else:
        logger.info("x_aa feature OFF — skipping BNN1 backbone load")
        bnn1_hidden = None
        bnn1_input_dim = 0
        latent_dim = 0
        bnn1_pipe_wt = None
        bnn1_pipe_mut = None

    # Data
    processed_dir = PROJECT_ROOT / config["data"]["output_dir"]
    df = load_multi_substrate_data(processed_dir)
    _ = get_supplemental_positions(df)  # logged elsewhere; not needed in train phase
    embeddings = load_all_embeddings(processed_dir)
    substrate_meta = load_substrate_metadata(processed_dir)

    # Null-model embedding/metric selection (done on full df so the scorer has
    # a sensible default to use for nearest / distance_weighted nulls)
    metric_selection = select_best_substrate_metric(df, embeddings)
    null_emb_type = metric_selection["best_embedding"]
    null_dist_metric = metric_selection["best_metric"]
    logger.info("Null-mode substrate metric: %s / %s (rho=%.3f)",
                null_emb_type, null_dist_metric, metric_selection["best_correlation"])

    # Active substrate list (for bookkeeping + sweep downstream)
    active_substrates = sorted(
        s for s, m in substrate_meta.items() if m.get("is_active", True))
    inactive_substrates = sorted(
        s for s, m in substrate_meta.items() if not m.get("is_active", True))
    full_substrate_pool = sorted(set(active_substrates) | set(inactive_substrates))
    training_substrates_subsample = (
        sorted(subsample) if subsample is not None else full_substrate_pool
    )

    # CV
    pairwise_df, train_lookup_df, fold_summaries, fold_histories = run_cv_and_collect_predictions(
        df, embeddings, bnn1_hidden, bnn1_input_dim, latent_dim,
        bnn1_pipe_wt, bnn1_pipe_mut, params, config, device,
        split_type, substrate_meta,
        target_substrate=args.target_substrate,
        subsample_train_substrates=subsample,
        null_embedding_type=null_emb_type,
        null_distance_metric=null_dist_metric,
    )

    logger.info("CV complete: %d pairwise rows across %d folds",
                len(pairwise_df), pairwise_df["fold"].nunique())

    # Persist
    pairwise_df.to_csv(results_dir / "pairwise_predictions.csv", index=False)
    train_lookup_df.to_csv(results_dir / "train_lookup.csv", index=False)

    # Final model (optional, only for splits where it makes sense)
    if split_type in ("random", "substrate") and not args.skip_final_model:
        logger.info("Training final model on all data...")
        train_final_model(
            df, embeddings, bnn1_hidden, bnn1_input_dim, latent_dim,
            bnn1_pipe_wt, bnn1_pipe_mut, params, config, device,
            substrate_meta, models_dir, fold_histories,
        )

    # Metadata
    held_out_substrates = sorted(pairwise_df["substrate"].unique().tolist())
    training_substrates_kept = sorted(train_lookup_df["substrate"].unique().tolist())
    train_metadata = {
        "split_type": split_type,
        "run_id": run_id,
        "target_substrate": args.target_substrate,
        "held_out_substrates": held_out_substrates,
        "training_substrates_kept": training_substrates_kept,
        "training_substrates_subsample": training_substrates_subsample,
        "training_pool_includes_inactives": True,
        "active_substrates": active_substrates,
        "inactive_substrates": inactive_substrates,
        "n_folds": int(pairwise_df["fold"].nunique()),
        "n_pairwise_rows": int(len(pairwise_df)),
        "null_model_embedding": null_emb_type,
        "null_model_distance_metric": null_dist_metric,
        "null_model_metric_correlation": float(metric_selection["best_correlation"]),
        "null_model_metric_selection": [
            {"embedding": e, "metric": m, "rho": r, "pvalue": p}
            for e, m, r, p in metric_selection["all_results"]
        ],
        "seed": int(config.get("cv", {}).get("seed", 42)),
        "bnn1_model_dir": str(bnn1_model_dir) if use_bnn1 else None,
        "x_aa_enabled": use_bnn1,
        "n_singleshot_repeats": int(args.n_singleshot_repeats) if split_type == "singleshot" else None,
        "distance_weight_temperature": float(params.get("distance_weight_temperature", 1.0)),
        "substrate_embedding_type": params.get("substrate_embedding_type"),
    }
    with open(results_dir / "train_metadata.json", "w") as f:
        json.dump(train_metadata, f, indent=2, default=str)

    with open(results_dir / "hyperparams.json", "w") as f:
        json.dump({k: v for k, v in params.items() if k != "features"},
                  f, indent=2, default=str)
    # Features kept separately for readability
    with open(results_dir / "features_used.json", "w") as f:
        json.dump(params.get("features", {}), f, indent=2, default=str)

    with open(results_dir / "config_used.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Lightweight fold-level training histories (for loss plots later)
    histories_summary = []
    for h in fold_histories:
        histories_summary.append({
            "train_loss": list(getattr(h, "train_loss", []) or []),
            "val_loss": list(getattr(h, "val_loss", []) or []),
            "train_nll": list(getattr(h, "train_nll", []) or []),
            "val_nll": list(getattr(h, "val_nll", []) or []),
            "train_kl": list(getattr(h, "train_kl", []) or []),
            "kl_weight_schedule": list(getattr(h, "kl_weight_schedule", []) or []),
            "best_epoch": getattr(h, "best_epoch", None),
        })
    with open(results_dir / "training_histories.json", "w") as f:
        json.dump({
            "fold_summaries": fold_summaries,
            "histories": histories_summary,
        }, f, indent=2, default=str)

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("new_05_bnn2_train.py done (%.1fs)  —  %s", elapsed, results_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
