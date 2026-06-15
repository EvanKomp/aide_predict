#!/usr/bin/env python
"""
opt_05_bnn2.py — Optuna hyperopt for BNN2 multi-substrate prediction
=====================================================================

Wraps 05_bnn2_multi_substrate.py: searches over BNN2 hyperparameters
defined in config.yaml search spaces, evaluates each trial via CV
on the specified split strategy, and minimizes mean CV aggregated CRPS.

CRPS (Continuous Ranked Probability Score) is a strictly proper scoring rule
that jointly optimizes accuracy and uncertainty calibration. Computed on
aggregated (per-mutation, per-substrate) predictions — the final scores that
matter for variant selection. Records Spearman, MAE per trial.

After finding best params, re-run 05 with those params:

    python 05_bnn2_multi_substrate.py --split random \\
        --hidden-dims '[256, 128, 64]' --prior-std 0.5 ...

Outputs to results/opt_05_bnn2/{split_name}/:
  - study_results.json, best_hyperparams.json, best_command.txt
  - hyperopt_tradeoffs.png, hyperopt_param_importance.png

Usage:
    python opt_05_bnn2.py --split random --device cuda:1
    python opt_05_bnn2.py --split substrate --n-trials 20
    python opt_05_bnn2.py --split random --n-trials 3 --device cuda:1
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

sys.path.insert(0, str(SCRIPT_DIR.parent))

# 05_bnn2_common.py and 05_bnn2_multi_substrate.py start with digits —
# use importlib to load them by file path.
from importlib.util import spec_from_file_location, module_from_spec as _mfs

# Load common utilities
_common_spec = spec_from_file_location("bnn2_common", SCRIPT_DIR / "05_bnn2_common.py")
_common = _mfs(_common_spec)
_common_spec.loader.exec_module(_common)

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
expand_to_pairwise = _common.expand_to_pairwise
build_bnn1_input = _common.build_bnn1_input
build_other_features = _common.build_other_features
preprocess_other_features = _common.preprocess_other_features
train_and_evaluate_fold = _common.train_and_evaluate_fold
aggregate_pairwise_predictions = _common.aggregate_pairwise_predictions
add_ref_distances = _common.add_ref_distances
compute_pairwise_distances = _common.compute_pairwise_distances
compute_nlpd = _common.compute_nlpd
compute_crps_gaussian = _common.compute_crps_gaussian

# Load split functions from main script
_main_spec = spec_from_file_location("bnn2_multi", SCRIPT_DIR / "05_bnn2_multi_substrate.py")
_mod = _mfs(_main_spec)
_main_spec.loader.exec_module(_mod)

make_random_folds = _mod.make_random_folds
make_position_folds = _mod.make_position_folds
make_substrate_folds = _mod.make_substrate_folds
make_singleshot_folds = _mod.make_singleshot_folds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def search_or_fixed(trial, name, config_entry, is_pca=False):
    """Sample from search space if available, otherwise return fixed value."""
    search_space = None
    if isinstance(config_entry, dict) and "search" in config_entry:
        search_space = config_entry["search"]

    if search_space is None:
        return resolve_param(config_entry)

    if is_pca:
        options = ["none" if v is None else str(v) for v in search_space]
        result_str = trial.suggest_categorical(name, options)
        return parse_pca_value(result_str)

    if search_space and isinstance(search_space[0], list):
        result_str = trial.suggest_categorical(
            name, [json.dumps(d) for d in search_space])
        return json.loads(result_str)

    return trial.suggest_categorical(name, search_space)


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

# Added to objective score when posterior has not moved from initialisation.
_COLLAPSE_PENALTY = 5.0
_COLLAPSE_THRESHOLD = 0.9  # collapse_score > this triggers penalty


_VALID_OBJECTIVES = ("crps", "mae", "spearman", "regret", "hit_rate", "ndcg", "enrichment")


def _break_ties(yp, seed=42):
    """Add tiny deterministic noise to break ties in predictions.

    ``np.argsort`` resolves ties by input order, making top-k selection
    an artifact of row ordering.  We always add negligible noise so that
    tied values are shuffled randomly (but reproducibly).  The noise
    scale is ~10 orders of magnitude below the prediction range, so it
    cannot flip the ordering of genuinely different values.
    """
    rng = np.random.RandomState(seed % (2**31))
    scale = max(np.ptp(yp) * 1e-10, 1e-12)
    return yp + rng.normal(0, scale, size=len(yp))


def _per_substrate_topk(y_true, y_pred, substrates, top_k):
    """Helper: iterate substrates yielding (y_true_sub, y_pred_sub, k) tuples."""
    for sub in np.unique(substrates):
        mask = substrates == sub
        yt = y_true[mask]
        yp = _break_ties(y_pred[mask], seed=len(yt))
        if len(yt) < 2:
            continue
        yield yt, yp, min(top_k, len(yt))


def _compute_regret(y_true, y_pred, substrates, top_k):
    """Mean per-substrate selection regret at top-k.

    Regret = mean(true top-k values) - mean(true values of model's top-k).
    """
    regrets = []
    for yt, yp, k in _per_substrate_topk(y_true, y_pred, substrates, top_k):
        oracle_mean = float(np.mean(yt[np.argsort(yt)[-k:]]))
        sel_mean = float(np.mean(yt[np.argsort(yp)[-k:]]))
        regrets.append(oracle_mean - sel_mean)
    return float(np.mean(regrets)) if regrets else float("nan")


def _compute_hit_rate(y_true, y_pred, substrates, top_k):
    """Mean per-substrate hit rate: fraction of true top-k found in model's top-k."""
    hit_rates = []
    for yt, yp, k in _per_substrate_topk(y_true, y_pred, substrates, top_k):
        true_topk = set(np.argsort(yt)[-k:])
        pred_topk = set(np.argsort(yp)[-k:])
        hit_rates.append(len(true_topk & pred_topk) / k)
    return float(np.mean(hit_rates)) if hit_rates else float("nan")


def _compute_ndcg(y_true, y_pred, substrates, top_k):
    """Mean per-substrate NDCG@k.

    Uses true values as relevance scores (shifted to non-negative via
    2^(y - y_min) - 1 so that all gains are >= 0).
    """
    ndcgs = []
    for yt, yp, k in _per_substrate_topk(y_true, y_pred, substrates, top_k):
        # Relevance: shift so min=0, then use 2^rel - 1 as gain
        rel = yt - yt.min()
        # DCG at k for model ranking
        pred_order = np.argsort(yp)[::-1][:k]
        discounts = 1.0 / np.log2(np.arange(2, k + 2))  # positions 1..k
        dcg = float(np.sum((2.0 ** rel[pred_order] - 1.0) * discounts))
        # Ideal DCG
        ideal_order = np.argsort(rel)[::-1][:k]
        idcg = float(np.sum((2.0 ** rel[ideal_order] - 1.0) * discounts))
        ndcgs.append(dcg / idcg if idcg > 0 else 1.0)
    return float(np.mean(ndcgs)) if ndcgs else float("nan")


def _compute_enrichment(y_true, y_pred, substrates, top_k, threshold=0.0):
    """Mean per-substrate enrichment factor at top-k.

    EF = (fraction active in model's top-k) / (fraction active overall).
    Active = y_true > threshold (default 0 = better than WT).
    """
    efs = []
    for yt, yp, k in _per_substrate_topk(y_true, y_pred, substrates, top_k):
        active = yt > threshold
        base_rate = active.mean()
        if base_rate == 0 or base_rate == 1:
            continue  # EF undefined when all or none are active
        sel_idx = np.argsort(yp)[-k:]
        sel_rate = active[sel_idx].mean()
        efs.append(float(sel_rate / base_rate))
    return float(np.mean(efs)) if efs else float("nan")


def create_objective(
    df, embeddings, bnn1_hidden, bnn1_input_dim, latent_dim,
    bnn1_pipe_wt, bnn1_pipe_mut, config, device, split_type,
    substrate_meta,
):
    """Create Optuna objective: minimize configurable metric + soft collapse penalty.

    The metric is set by ``cv.optimization_objective`` in config:
      - "crps"       — aggregated CRPS (accuracy + calibration)
      - "mae"        — aggregated MAE
      - "spearman"   — negative aggregated Spearman ρ (maximize ranking)
      - "regret"     — mean per-substrate selection regret at top-k
      - "hit_rate"   — negative mean top-k hit rate (maximize overlap with true top-k)
      - "ndcg"       — negative mean NDCG@k (maximize ranking quality at top)
      - "enrichment" — negative mean enrichment factor at top-k (maximize active fraction)

    Evaluates on **aggregated** (per-mutation, per-substrate) predictions —
    these are the final scores that matter for variant selection. Expanded
    pairwise predictions are aggregated using the searched aggregation mode
    before computing metrics.

    A soft collapse penalty is added when mean posterior std / prior_std > 0.9,
    indicating the weight posterior has not updated from its initialisation.
    """
    bnn2 = config["bnn2"]
    train_cfg = bnn2["training"]
    preproc = config["preprocessing"]
    pairwise_cfg_raw = bnn2.get("pairwise", {})
    features_cfg_raw = bnn2.get("features", {})
    cv_config = config["cv"]
    seed = cv_config["seed"]
    n_folds = cv_config["n_folds"]
    opt_objective = cv_config.get("optimization_objective", "crps")
    regret_top_k = cv_config.get("regret_top_k", config.get("selection", {}).get("n_per_substrate", 10))
    enrichment_threshold = cv_config.get("enrichment_threshold", 0.0)
    if opt_objective not in _VALID_OBJECTIVES:
        raise ValueError(f"Unknown optimization_objective {opt_objective!r}; choose from {_VALID_OBJECTIVES}")

    def objective(trial):
        # ── Model architecture ──
        hidden_dims = search_or_fixed(trial, "hidden_dims", bnn2["hidden_dims"])
        prior_std = search_or_fixed(trial, "prior_std", bnn2["prior_std"])
        dropout_rate = search_or_fixed(trial, "dropout_rate", bnn2["dropout_rate"])
        activation = search_or_fixed(trial, "activation", bnn2["activation"])
        x_aa_freeze = search_or_fixed(trial, "x_aa_freeze", bnn2["x_aa_freeze"])
        substrate_embedding_type = search_or_fixed(
            trial, "substrate_embedding_type", bnn2["substrate_embedding_type"])

        # ── Training ──
        learning_rate = search_or_fixed(trial, "learning_rate", train_cfg["learning_rate"])
        kl_weight = search_or_fixed(trial, "kl_weight", train_cfg.get("kl_weight", {"value": 1.0}))
        batch_size = search_or_fixed(trial, "batch_size", train_cfg.get("batch_size", {"value": 32}))
        kl_anneal_epochs = search_or_fixed(trial, "kl_anneal_epochs", train_cfg.get("kl_anneal_epochs", {"value": 30}))
        clip_grad_norm = search_or_fixed(trial, "clip_grad_norm", train_cfg.get("clip_grad_norm", {"value": 5.0}))

        # ── Loss & regularization ──
        loss_type = search_or_fixed(
            trial, "loss_type", bnn2.get("loss_type", {"value": "gaussian_nll"}))
        null_reg_weight = search_or_fixed(
            trial, "null_reg_weight", train_cfg.get("null_reg_weight", {"value": 0.0}))
        log_var_floor = search_or_fixed(
            trial, "log_var_floor", train_cfg.get("log_var_floor", {"value": None}))

        # ── LDS ──
        lds_cfg_raw = bnn2.get("lds", {})
        use_lds = search_or_fixed(trial, "use_lds", lds_cfg_raw.get("use_lds", {"value": False}))
        lds_cfg = {
            "use_lds": use_lds,
            "n_bins": lds_cfg_raw.get("n_bins", 50),
            "kernel_size": lds_cfg_raw.get("kernel_size", 5),
            "sigma": lds_cfg_raw.get("sigma", 2.0),
        }

        # ── Feature toggles ──
        features = {}
        for feat_name in ["fc_ref", "ref_distance", "x_target_substrate",
                          "x_ref_substrate", "x_aa", "esm_wt", "esm_mut", "saprot_zs"]:
            feat_cfg = features_cfg_raw.get(feat_name, {"value": True} if feat_name != "x_aa" else {"value": False})
            features[feat_name] = search_or_fixed(trial, f"feat_{feat_name}", feat_cfg)

        # ── Pairwise / aggregation ──
        inference_aggregation = search_or_fixed(
            trial, "inference_aggregation",
            pairwise_cfg_raw.get("inference_aggregation", {"value": "nearest"}))
        distance_weight_temperature = search_or_fixed(
            trial, "distance_weight_temperature",
            pairwise_cfg_raw.get("distance_weight_temperature", {"value": 1.0}))
        exclude_self_ref = search_or_fixed(
            trial, "exclude_self_ref",
            pairwise_cfg_raw.get("exclude_self_ref", {"value": True}))

        # ── Preprocessing ──
        x_sub_scaler = search_or_fixed(
            trial, "x_substrate_scaler", preproc.get("x_substrate", {}).get("scaler", "none"))
        x_sub_pca = search_or_fixed(
            trial, "x_substrate_pca", preproc.get("x_substrate", {}).get("pca", None), is_pca=True)
        saprot_scaler = search_or_fixed(
            trial, "saprot_zs_scaler", preproc.get("saprot_zs", {}).get("scaler", "none"))
        esm_wt_scaler = search_or_fixed(
            trial, "esm_wt_scaler", preproc.get("esm_wt", {}).get("scaler", "standard"))
        esm_wt_pca = search_or_fixed(
            trial, "esm_wt_pca", preproc.get("esm_wt", {}).get("pca", {"value": 0.99}), is_pca=True)
        esm_mut_scaler = search_or_fixed(
            trial, "esm_mut_scaler", preproc.get("esm_mut", {}).get("scaler", "standard"))
        esm_mut_pca = search_or_fixed(
            trial, "esm_mut_pca", preproc.get("esm_mut", {}).get("pca", {"value": 0.99}), is_pca=True)

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
            # Pairwise / aggregation
            "inference_aggregation": inference_aggregation,
            "distance_weight_temperature": distance_weight_temperature,
            # Preprocessing
            "x_substrate_scaler": x_sub_scaler,
            "x_substrate_pca": x_sub_pca,
            "saprot_zs_scaler": saprot_scaler,
            "esm_wt_scaler": esm_wt_scaler,
            "esm_wt_pca": esm_wt_pca,
            "esm_mut_scaler": esm_mut_scaler,
            "esm_mut_pca": esm_mut_pca,
            # Hurdle sub-params (fixed; only used when loss_type == "hurdle")
            "hurdle": bnn2.get("hurdle", {}),
        }

        # Override pairwise config for expand_to_pairwise (reads from config dict)
        # Temporarily patch config so expand_to_pairwise sees the searched values
        trial_pairwise = {
            "ref_substrates": pairwise_cfg_raw.get("ref_substrates", "all"),
            "exclude_self_ref": exclude_self_ref,
            "inference_aggregation": inference_aggregation,
            "distance_weight_temperature": distance_weight_temperature,
        }
        trial_config = dict(config)
        trial_config["bnn2"] = dict(config["bnn2"])
        trial_config["bnn2"]["pairwise"] = trial_pairwise

        # Build folds
        if split_type == "random":
            folds = make_random_folds(df, n_folds, seed)
        elif split_type == "position":
            folds = make_position_folds(df, n_folds, seed)
        elif split_type == "substrate":
            folds = make_substrate_folds(df)
        elif split_type == "singleshot":
            folds = make_singleshot_folds(df, 1, seed)  # 1 repeat per substrate for speed
        else:
            raise ValueError(f"Unknown split: {split_type}")

        fold_spearmans = []
        fold_maes = []
        fold_crps = []
        fold_regrets = []
        fold_hit_rates = []
        fold_ndcgs = []
        fold_enrichments = []
        fold_collapse_scores = []

        for train_idx, val_idx in folds:
            df_train = df.iloc[train_idx].reset_index(drop=True)
            df_val = df.iloc[val_idx].reset_index(drop=True)

            train_fc_lookup = {
                (row["mutation_string"], row["substrate"]): row["fold_change"]
                for _, row in df_train.iterrows()
            }
            df_train_exp = expand_to_pairwise(df_train, substrate_meta, trial_config)
            # Position splits hold out ALL substrates at a position, so
            # training has no FC entries for held-out mutations.  Let the
            # val set use its own internal references (ref_fc_lookup=None).
            val_ref_lookup = train_fc_lookup if split_type != "position" else None
            df_val_exp = expand_to_pairwise(df_val, substrate_meta, trial_config,
                                            ref_fc_lookup=val_ref_lookup)

            if len(df_train_exp) == 0 or len(df_val_exp) == 0:
                continue

            # Add reference distances (for ref_distance feature + aggregation)
            sub_emb_type = params.get("substrate_embedding_type", "morgan")
            sub_emb = embeddings[f"substrate_{sub_emb_type}"].astype(np.float64)
            ref_dist_matrix = compute_pairwise_distances(sub_emb, "cosine")
            substrate_names = embeddings["substrate_names"]
            add_ref_distances(df_train_exp, ref_dist_matrix, substrate_names)
            add_ref_distances(df_val_exp, ref_dist_matrix, substrate_names)

            use_bnn1 = features.get("x_aa", False)
            if use_bnn1:
                X_bnn1_tr = build_bnn1_input(df_train_exp, embeddings, bnn1_pipe_wt, bnn1_pipe_mut)
                X_bnn1_va = build_bnn1_input(df_val_exp, embeddings, bnn1_pipe_wt, bnn1_pipe_mut)

            groups_tr = build_other_features(df_train_exp, embeddings, params, substrate_meta)
            groups_va = build_other_features(df_val_exp, embeddings, params, substrate_meta)

            X_other_tr, X_other_va, _ = preprocess_other_features(
                groups_tr, groups_va, params, config)

            if use_bnn1:
                X_train = np.concatenate([X_bnn1_tr, X_other_tr], axis=1).astype(np.float32)
                X_val = np.concatenate([X_bnn1_va, X_other_va], axis=1).astype(np.float32)
            else:
                X_train = X_other_tr.astype(np.float32)
                X_val = X_other_va.astype(np.float32)
            # Delta targets: y = log_fc - log_fc_ref
            fc_ref_train = df_train_exp["log_fc_ref"].values.astype(np.float32)
            fc_ref_val = df_val_exp["log_fc_ref"].values.astype(np.float32)
            y_train = (df_train_exp["log_fc"].values - fc_ref_train).astype(np.float32)
            y_val = (df_val_exp["log_fc"].values - fc_ref_val).astype(np.float32)

            other_dim = X_other_tr.shape[1]

            fold_metrics, estimates, _ = train_and_evaluate_fold(
                X_train, y_train, X_val, y_val,
                bnn1_hidden if use_bnn1 else None,
                bnn1_input_dim if use_bnn1 else 0,
                latent_dim if use_bnn1 else 0,
                other_dim,
                params, device, return_predictions=True,
                fc_ref_train=fc_ref_train, fc_ref_val=fc_ref_val,
            )

            # ── Aggregate pairwise → per-(mutation, substrate) ──
            def _to_np(x):
                if x is None:
                    return None
                if hasattr(x, "cpu"):
                    return x.cpu().numpy()
                return np.asarray(x)

            y_pred_exp = _to_np(estimates.mean)
            epistemic_exp = _to_np(estimates.epistemic_std)
            aleatoric_exp = _to_np(estimates.aleatoric_std)
            total_exp = _to_np(estimates.total_std)
            cls_prob_exp = _to_np(getattr(estimates, "cls_prob", None))

            agg_mode = params.get("inference_aggregation", "nearest")
            dist_temp = params.get("distance_weight_temperature", 1.0)

            y_pred_agg, epi_agg, ale_agg, tot_agg, agg_df = aggregate_pairwise_predictions(
                y_pred_exp, epistemic_exp, aleatoric_exp, total_exp,
                df_val_exp,
                cls_prob_expanded=cls_prob_exp,
                aggregation_mode=agg_mode,
                distance_weight_temperature=dist_temp,
            )

            # Compute metrics on aggregated predictions
            y_true_agg = agg_df["log_fc"].values
            subs_agg = agg_df["substrate"].values
            agg_mae = float(np.mean(np.abs(y_true_agg - y_pred_agg)))
            agg_rho = float(stats.spearmanr(y_true_agg, y_pred_agg).statistic)
            agg_crps = float(np.mean(compute_crps_gaussian(y_true_agg, y_pred_agg, tot_agg)))
            agg_regret = _compute_regret(y_true_agg, y_pred_agg, subs_agg, regret_top_k)
            agg_hit_rate = _compute_hit_rate(y_true_agg, y_pred_agg, subs_agg, regret_top_k)
            agg_ndcg = _compute_ndcg(y_true_agg, y_pred_agg, subs_agg, regret_top_k)
            agg_enrichment = _compute_enrichment(
                y_true_agg, y_pred_agg, subs_agg, regret_top_k, enrichment_threshold)

            fold_spearmans.append(agg_rho)
            fold_maes.append(agg_mae)
            fold_crps.append(agg_crps)
            fold_regrets.append(agg_regret)
            fold_hit_rates.append(agg_hit_rate)
            fold_ndcgs.append(agg_ndcg)
            fold_enrichments.append(agg_enrichment)
            fold_collapse_scores.append(fold_metrics.get("posterior_collapse_score", 1.0))

        if not fold_crps:
            return float("inf")

        mean_spearman = float(np.nanmean(fold_spearmans))
        mean_mae = float(np.nanmean(fold_maes))
        mean_crps = float(np.nanmean(fold_crps))
        mean_regret = float(np.nanmean(fold_regrets))
        mean_hit_rate = float(np.nanmean(fold_hit_rates))
        mean_ndcg = float(np.nanmean(fold_ndcgs))
        mean_enrichment = float(np.nanmean(fold_enrichments))
        mean_collapse = float(np.nanmean(fold_collapse_scores))

        collapse_penalty = _COLLAPSE_PENALTY if mean_collapse > _COLLAPSE_THRESHOLD else 0.0

        trial.set_user_attr("fold_spearmans", fold_spearmans)
        trial.set_user_attr("fold_maes", fold_maes)
        trial.set_user_attr("fold_crps", fold_crps)
        trial.set_user_attr("fold_regrets", fold_regrets)
        trial.set_user_attr("fold_hit_rates", fold_hit_rates)
        trial.set_user_attr("fold_ndcgs", fold_ndcgs)
        trial.set_user_attr("fold_enrichments", fold_enrichments)
        trial.set_user_attr("mean_spearman", mean_spearman)
        trial.set_user_attr("mean_mae", mean_mae)
        trial.set_user_attr("mean_crps", mean_crps)
        trial.set_user_attr("mean_regret", mean_regret)
        trial.set_user_attr("mean_hit_rate", mean_hit_rate)
        trial.set_user_attr("mean_ndcg", mean_ndcg)
        trial.set_user_attr("mean_enrichment", mean_enrichment)
        trial.set_user_attr("mean_collapse_score", mean_collapse)
        trial.set_user_attr("collapse_penalty", collapse_penalty)

        # Select objective based on config (Optuna minimizes)
        _obj_map = {
            "crps": mean_crps,
            "mae": mean_mae,
            "spearman": -mean_spearman,        # negate: want high ρ
            "regret": mean_regret,
            "hit_rate": -mean_hit_rate,         # negate: want high hit rate
            "ndcg": -mean_ndcg,                 # negate: want high NDCG
            "enrichment": -mean_enrichment,     # negate: want high EF
        }
        raw_objective = _obj_map[opt_objective]

        objective_value = raw_objective + collapse_penalty
        trial.set_user_attr("raw_objective", raw_objective)

        logger.info(
            "Trial %d [obj=%s]: %.4f (raw=%.4f + penalty=%.1f)  "
            "CRPS=%.4f  MAE=%.4f  ρ=%.4f  regret=%.4f  "
            "hit=%.3f  ndcg=%.3f  EF=%.2f  collapse=%.3f",
            trial.number, opt_objective, objective_value, raw_objective,
            collapse_penalty, mean_crps, mean_mae, mean_spearman,
            mean_regret, mean_hit_rate, mean_ndcg, mean_enrichment,
            mean_collapse,
        )

        return objective_value

    return objective


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def decode_trial_params(raw_params: dict) -> dict:
    """Decode Optuna trial params back into native Python types."""
    params = dict(raw_params)
    if "hidden_dims" in params:
        params["hidden_dims"] = json.loads(params["hidden_dims"])
    if "x_substrate_pca" in params:
        params["x_substrate_pca"] = parse_pca_value(params["x_substrate_pca"])
    return params


def build_full_params(searched_params: dict, config: dict) -> dict:
    """Merge searched params with fixed config defaults."""
    bnn2 = config["bnn2"]
    train_cfg = bnn2["training"]
    preproc = config["preprocessing"]
    pairwise_cfg = resolve_config_block(bnn2.get("pairwise", {}))
    features_cfg = resolve_config_block(bnn2.get("features", {}))
    lds_cfg = resolve_config_block(bnn2.get("lds", {}))

    result = {
        # Model architecture
        "hidden_dims": searched_params.get("hidden_dims", resolve_param(bnn2["hidden_dims"])),
        "prior_std": searched_params.get("prior_std", resolve_param(bnn2["prior_std"])),
        "dropout_rate": searched_params.get("dropout_rate", resolve_param(bnn2["dropout_rate"])),
        "activation": searched_params.get("activation", resolve_param(bnn2["activation"])),
        "x_aa_freeze": searched_params.get("x_aa_freeze", resolve_param(bnn2["x_aa_freeze"])),
        "substrate_embedding_type": searched_params.get(
            "substrate_embedding_type", resolve_param(bnn2["substrate_embedding_type"])),
        # Training
        "learning_rate": searched_params.get("learning_rate", resolve_param(train_cfg["learning_rate"])),
        "kl_weight": searched_params.get("kl_weight", resolve_param(train_cfg.get("kl_weight", {"value": 1.0}))),
        "batch_size": searched_params.get("batch_size", resolve_param(train_cfg.get("batch_size", {"value": 32}))),
        "kl_anneal_epochs": searched_params.get("kl_anneal_epochs", resolve_param(train_cfg.get("kl_anneal_epochs", {"value": 30}))),
        "clip_grad_norm": searched_params.get("clip_grad_norm", resolve_param(train_cfg.get("clip_grad_norm", {"value": 5.0}))),
        # Loss & regularization
        "loss_type": searched_params.get(
            "loss_type", resolve_param(bnn2.get("loss_type", {"value": "gaussian_nll"}))),
        "null_reg_weight": searched_params.get(
            "null_reg_weight", resolve_param(train_cfg.get("null_reg_weight", {"value": 0.0}))),
        "log_var_floor": searched_params.get(
            "log_var_floor", resolve_param(train_cfg.get("log_var_floor", {"value": None}))),
        # LDS
        "use_lds": searched_params.get("use_lds", lds_cfg.get("use_lds", False)),
        # Pairwise / aggregation
        "inference_aggregation": searched_params.get(
            "inference_aggregation", pairwise_cfg.get("inference_aggregation", "nearest")),
        "distance_weight_temperature": searched_params.get(
            "distance_weight_temperature", pairwise_cfg.get("distance_weight_temperature", 1.0)),
        "exclude_self_ref": searched_params.get(
            "exclude_self_ref", pairwise_cfg.get("exclude_self_ref", True)),
        # Preprocessing — substrate
        "x_substrate_scaler": searched_params.get(
            "x_substrate_scaler", resolve_param(preproc.get("x_substrate", {}).get("scaler", "none"))),
        "x_substrate_pca": searched_params.get(
            "x_substrate_pca", parse_pca_value(resolve_param(preproc.get("x_substrate", {}).get("pca")))),
        "saprot_zs_scaler": searched_params.get(
            "saprot_zs_scaler", resolve_param(preproc.get("saprot_zs", {}).get("scaler", "none"))),
        # Preprocessing — ESM
        "esm_wt_scaler": searched_params.get(
            "esm_wt_scaler", resolve_param(preproc.get("esm_wt", {}).get("scaler", "standard"))),
        "esm_wt_pca": searched_params.get(
            "esm_wt_pca", parse_pca_value(resolve_param(preproc.get("esm_wt", {}).get("pca")))),
        "esm_mut_scaler": searched_params.get(
            "esm_mut_scaler", resolve_param(preproc.get("esm_mut", {}).get("scaler", "standard"))),
        "esm_mut_pca": searched_params.get(
            "esm_mut_pca", parse_pca_value(resolve_param(preproc.get("esm_mut", {}).get("pca")))),
    }

    # Feature toggles (feat_* prefix in searched params)
    for feat_name in ["fc_ref", "ref_distance", "x_target_substrate",
                      "x_ref_substrate", "x_aa", "esm_wt", "esm_mut", "saprot_zs"]:
        key = f"feat_{feat_name}"
        result[key] = searched_params.get(key, features_cfg.get(feat_name, feat_name != "x_aa"))

    return result


def build_rerun_command(full_params: dict, split_type: str, device: str,
                        hyperparams_path: str = None) -> str:
    """Build a CLI command to re-run 05 with the best hyperparams."""
    parts = [f"python 05_bnn2_multi_substrate.py --split {split_type}"]
    parts.append(f"--device {device}")
    if hyperparams_path:
        parts.append(f"--hyperparams {hyperparams_path}")
    return " \\\n    ".join(parts)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_hyperopt_tradeoffs(trial_results: list, output_path, opt_objective="crps"):
    """Plot tradeoffs between metrics and objective convergence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    completed = [t for t in trial_results if t["value"] is not None]
    if len(completed) < 2:
        logger.warning("Not enough completed trials for tradeoff plot")
        return

    def _float_or_nan(v):
        """Convert None/inf to nan for safe np.isnan filtering."""
        if v is None or v == float("inf") or v == float("-inf"):
            return float("nan")
        return float(v)

    crps_vals = [_float_or_nan(t.get("mean_crps")) for t in completed]
    objectives = [_float_or_nan(t["value"]) for t in completed]
    spearmans = [_float_or_nan(t.get("mean_spearman")) for t in completed]
    maes = [_float_or_nan(t.get("mean_mae")) for t in completed]
    regrets = [_float_or_nan(t.get("mean_regret")) for t in completed]
    trial_nums = list(range(len(crps_vals)))

    # Best by objective (CRPS + collapse penalty)
    valid_obj = [v if not np.isnan(v) else float("inf") for v in objectives]
    best_idx = int(np.argmin(valid_obj))

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    def _scatter(ax, x, y, xlabel, ylabel, title, best_x, best_y):
        valid = [(xi, yi, n) for xi, yi, n in zip(x, y, trial_nums)
                 if not np.isnan(xi) and not np.isnan(yi)]
        if not valid:
            return
        xv, yv, nv = zip(*valid)
        sc = ax.scatter(xv, yv, c=nv, cmap="viridis", s=30, alpha=0.7, edgecolors="none")
        if not np.isnan(best_x) and not np.isnan(best_y):
            ax.scatter([best_x], [best_y], marker="*", s=200, c="red",
                       zorder=5, label="Best")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        return sc

    _scatter(axes[0], crps_vals, spearmans,
             "CRPS (lower=better)", "Spearman ρ (higher=better)",
             "CRPS vs Spearman", crps_vals[best_idx], spearmans[best_idx])

    _scatter(axes[1], crps_vals, maes,
             "CRPS (lower=better)", "MAE (lower=better)",
             "CRPS vs MAE", crps_vals[best_idx], maes[best_idx])

    sc = _scatter(axes[2], spearmans, maes,
                  "Spearman ρ (higher=better)", "MAE (lower=better)",
                  "Spearman vs MAE", spearmans[best_idx], maes[best_idx])
    if sc is not None:
        plt.colorbar(sc, ax=axes[2], label="Trial number")

    # Panel 4: convergence — objective vs trial number with running best
    valid_conv = [(n, o) for n, o in zip(trial_nums, objectives) if not np.isnan(o)]
    if valid_conv:
        ns, os_ = zip(*valid_conv)
        axes[3].scatter(ns, os_, c="#2196F3", s=20, alpha=0.5, edgecolors="none", label="Trials")
        running_best = np.minimum.accumulate(os_)
        axes[3].step(ns, running_best, where="post", c="red", lw=2, label="Running best")
        axes[3].set_xlabel("Trial number")
        axes[3].set_ylabel(f"Objective ({opt_objective} + penalty)")
        axes[3].set_title("Convergence")
        axes[3].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_param_importance(study, output_path):
    """Plot hyperparameter importance from Optuna."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import optuna
        importance = optuna.importance.get_param_importances(study)
    except Exception as e:
        logger.warning("Could not compute param importance: %s", e)
        return

    if not importance:
        return

    names = list(importance.keys())
    values = list(importance.values())

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.4)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color="#2196F3", edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title("Hyperparameter Importance (fANOVA)")
    ax.invert_yaxis()

    for i, v in enumerate(values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


# ---------------------------------------------------------------------------
# Per-trial callback — save best results as we go
# ---------------------------------------------------------------------------

def _make_progress_callback(results_dir, config, split_type, device, opt_objective):
    """Return an Optuna callback that saves best_command.txt and best_hyperparams.json
    after each completed trial, so results are available even if the run is interrupted."""
    import optuna

    def callback(study, trial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        try:
            best = study.best_trial
        except ValueError:
            return  # no completed trials yet

        searched_params = decode_trial_params(best.params)
        best_params = build_full_params(searched_params, config)

        # best_hyperparams.json (write first so best_command.txt can reference it)
        best_save = dict(best_params)
        best_save["optimization_objective"] = opt_objective
        best_save["objective_value"] = float(best.value)
        best_save["raw_objective"] = best.user_attrs.get("raw_objective")
        best_save["trial_number"] = best.number
        for metric in ["spearman", "mae", "crps", "regret", "hit_rate", "ndcg", "enrichment"]:
            best_save[f"mean_cv_{metric}"] = best.user_attrs.get(f"mean_{metric}")
        for fold_metric in ["spearmans", "maes", "crps", "regrets", "hit_rates", "ndcgs", "enrichments"]:
            best_save[f"fold_{fold_metric}"] = best.user_attrs.get(f"fold_{fold_metric}")
        hp_path = results_dir / "best_hyperparams.json"
        with open(hp_path, "w") as f:
            json.dump(best_save, f, indent=2, default=str)

        # best_command.txt
        rerun_cmd = build_rerun_command(best_params, split_type, device,
                                        hyperparams_path=str(hp_path))
        (results_dir / "best_command.txt").write_text(rerun_cmd + "\n")

    return callback


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna hyperopt for BNN2 multi-substrate prediction",
    )
    parser.add_argument("--split", type=str, required=True,
                        choices=["random", "position", "substrate", "singleshot"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Total number of Optuna trials (default: from config). "
                             "When resuming, only runs remaining trials.")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing study DB and start fresh "
                             "(default: resume from previous trials)")
    parser.add_argument("--bnn1-model-dir", type=str, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    t_start = time.time()

    split_type = args.split

    results_dir = PROJECT_ROOT / "results" / "opt_05_bnn2" / split_type
    results_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(results_dir / "run.log")

    logger.info("=" * 60)
    logger.info("opt_05_bnn2.py — %s split", split_type)
    logger.info("Optuna Hyperopt for BNN2 Multi-Substrate Prediction")
    logger.info("=" * 60)

    config = load_config(args.config)
    device = get_device(config, args.device)

    # Load BNN1
    if args.bnn1_model_dir:
        bnn1_model_dir = Path(args.bnn1_model_dir)
    else:
        bnn1_model_dir = PROJECT_ROOT / "results" / "03_formaldehyde_regression" / "models"
    bnn1_hidden, bnn1_input_dim, latent_dim, _ = load_bnn1_backbone(bnn1_model_dir, device)
    bnn1_pipe_wt, bnn1_pipe_mut = load_bnn1_preprocessing(bnn1_model_dir)

    # Load data
    processed_dir = PROJECT_ROOT / config["data"]["output_dir"]
    df = load_multi_substrate_data(processed_dir)
    embeddings = load_all_embeddings(processed_dir)
    substrate_meta = load_substrate_metadata(processed_dir)

    # Run Optuna
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_trials_total = args.n_trials or config["cv"]["n_hyperopt_trials"]
    opt_objective = config["cv"].get("optimization_objective", "crps")

    # Persistent SQLite storage for resume support
    study_name = f"bnn2_{split_type}"
    db_path = results_dir / "optuna_study.db"
    if args.fresh and db_path.exists():
        db_path.unlink()
        logger.info("Deleted existing study DB (--fresh)")
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config["cv"]["seed"]),
        load_if_exists=True,
    )

    n_completed = len([t for t in study.trials
                       if t.state == optuna.trial.TrialState.COMPLETE])
    n_remaining = max(0, n_trials_total - n_completed)

    if n_completed > 0:
        logger.info("Resuming study: %d/%d trials already completed, %d remaining",
                    n_completed, n_trials_total, n_remaining)
    else:
        logger.info("Starting fresh study (%d trials, %s split, objective=%s)",
                    n_trials_total, split_type, opt_objective)

    if n_remaining == 0:
        logger.info("All %d trials already completed. Use --n-trials N (N > %d) "
                    "to run more, or --fresh to restart.", n_trials_total, n_completed)
    else:
        objective = create_objective(
            df, embeddings, bnn1_hidden, bnn1_input_dim, latent_dim,
            bnn1_pipe_wt, bnn1_pipe_mut, config, device, split_type,
            substrate_meta,
        )
        progress_cb = _make_progress_callback(
            results_dir, config, split_type, device, opt_objective)
        study.optimize(objective, n_trials=n_remaining,
                       show_progress_bar=True, callbacks=[progress_cb])

    # Parse best trial
    best_trial = study.best_trial
    searched_params = decode_trial_params(best_trial.params)
    best_params = build_full_params(searched_params, config)

    logger.info("Best trial: #%d", best_trial.number)
    logger.info("  objective (%s + penalty): %.4f", opt_objective, best_trial.value)
    logger.info("  agg CRPS:       %.4f", best_trial.user_attrs.get("mean_crps", float("nan")))
    logger.info("  agg Spearman:   %.4f", best_trial.user_attrs.get("mean_spearman", float("nan")))
    logger.info("  agg MAE:        %.4f", best_trial.user_attrs.get("mean_mae", float("nan")))
    logger.info("  agg regret:     %.4f", best_trial.user_attrs.get("mean_regret", float("nan")))
    logger.info("  agg hit_rate:   %.4f", best_trial.user_attrs.get("mean_hit_rate", float("nan")))
    logger.info("  agg NDCG:       %.4f", best_trial.user_attrs.get("mean_ndcg", float("nan")))
    logger.info("  agg enrichment: %.4f", best_trial.user_attrs.get("mean_enrichment", float("nan")))
    for k, v in best_params.items():
        searched = "(searched)" if k in best_trial.params else "(fixed)"
        logger.info("  %s: %s  %s", k, v, searched)

    # Save results
    trial_results = []
    for trial in study.trials:
        params = decode_trial_params(trial.params)
        full = build_full_params(params, config)
        trial_results.append({
            "number": trial.number,
            "value": trial.value,
            "objective": opt_objective,
            "raw_objective": trial.user_attrs.get("raw_objective"),
            "params": full,
            "searched_params": params,
            "mean_spearman": trial.user_attrs.get("mean_spearman"),
            "mean_mae": trial.user_attrs.get("mean_mae"),
            "mean_crps": trial.user_attrs.get("mean_crps"),
            "mean_regret": trial.user_attrs.get("mean_regret"),
            "mean_hit_rate": trial.user_attrs.get("mean_hit_rate"),
            "mean_ndcg": trial.user_attrs.get("mean_ndcg"),
            "mean_enrichment": trial.user_attrs.get("mean_enrichment"),
            "fold_spearmans": trial.user_attrs.get("fold_spearmans"),
            "fold_maes": trial.user_attrs.get("fold_maes"),
            "fold_crps": trial.user_attrs.get("fold_crps"),
            "fold_regrets": trial.user_attrs.get("fold_regrets"),
            "fold_hit_rates": trial.user_attrs.get("fold_hit_rates"),
            "fold_ndcgs": trial.user_attrs.get("fold_ndcgs"),
            "fold_enrichments": trial.user_attrs.get("fold_enrichments"),
        })

    with open(results_dir / "study_results.json", "w") as f:
        json.dump(trial_results, f, indent=2, default=str)

    best_save = dict(best_params)
    best_save["optimization_objective"] = opt_objective
    best_save["objective_value"] = float(best_trial.value)
    best_save["raw_objective"] = best_trial.user_attrs.get("raw_objective")
    best_save["trial_number"] = best_trial.number
    for metric in ["spearman", "mae", "crps", "regret", "hit_rate", "ndcg", "enrichment"]:
        best_save[f"mean_cv_{metric}"] = best_trial.user_attrs.get(f"mean_{metric}")
    for fold_metric in ["spearmans", "maes", "crps", "regrets", "hit_rates", "ndcgs", "enrichments"]:
        best_save[f"fold_{fold_metric}"] = best_trial.user_attrs.get(f"fold_{fold_metric}")
    with open(results_dir / "best_hyperparams.json", "w") as f:
        json.dump(best_save, f, indent=2, default=str)

    hp_path = results_dir / "best_hyperparams.json"
    rerun_cmd = build_rerun_command(best_params, split_type, device,
                                    hyperparams_path=str(hp_path))
    with open(results_dir / "best_command.txt", "w") as f:
        f.write(rerun_cmd + "\n")

    with open(results_dir / "config_used.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Plots
    plot_hyperopt_tradeoffs(trial_results, results_dir / "hyperopt_tradeoffs.png",
                            opt_objective=opt_objective)
    plot_param_importance(study, results_dir / "hyperopt_param_importance.png")

    # Summary
    elapsed = time.time() - t_start
    logger.info("=" * 60)
    n_total = len([t for t in study.trials
                   if t.state == optuna.trial.TrialState.COMPLETE])
    logger.info("Hyperopt Complete (%.1fs, %d total trials, %s split, objective=%s)",
                elapsed, n_total, split_type, opt_objective)
    logger.info("=" * 60)
    logger.info("Best objective (%s): %.4f", opt_objective, best_trial.value)
    logger.info("Best agg CRPS:       %.4f", best_trial.user_attrs.get("mean_crps", float("nan")))
    logger.info("Best agg Spearman:   %.4f", best_trial.user_attrs.get("mean_spearman", float("nan")))
    logger.info("Best agg MAE:        %.4f", best_trial.user_attrs.get("mean_mae", float("nan")))
    logger.info("Best agg regret:     %.4f", best_trial.user_attrs.get("mean_regret", float("nan")))
    logger.info("Best agg hit_rate:   %.4f", best_trial.user_attrs.get("mean_hit_rate", float("nan")))
    logger.info("Best agg NDCG:       %.4f", best_trial.user_attrs.get("mean_ndcg", float("nan")))
    logger.info("Best agg enrichment: %.4f", best_trial.user_attrs.get("mean_enrichment", float("nan")))
    logger.info("To reproduce:")
    logger.info("  %s", rerun_cmd)


if __name__ == "__main__":
    main()
