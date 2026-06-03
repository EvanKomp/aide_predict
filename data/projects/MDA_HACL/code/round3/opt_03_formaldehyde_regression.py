#!/usr/bin/env python
"""
opt_03_formaldehyde_regression.py — Optuna hyperopt for formaldehyde regression
================================================================================

Wraps 03_bnn1_formaldehyde_regression.py: searches over all hyperparameters
defined in config.yaml search spaces, evaluates each trial via K-fold CV,
and minimizes mean CV NLPD (Negative Log Predictive Density).

NLPD is a strictly proper scoring rule that jointly optimizes accuracy and
uncertainty calibration — no ad-hoc tradeoff parameter needed. Records
Spearman, MAE, CRPS, and val_loss per trial for tradeoff analysis.

After finding best params, re-run 03 with those params for full evaluation
+ final model training:

    python 03_bnn1_formaldehyde_regression.py \\
        --hidden-dims '[128, 64]' --prior-std 0.5 ...

Outputs to results/opt_03_formaldehyde_regression/:
  - study_results.json: all trial results
  - best_hyperparams.json: best trial params
  - best_command.txt: ready-to-run command for 03 with best params
  - hyperopt_tradeoffs.png: NLPD vs Spearman vs MAE vs CRPS scatter

Usage:
    python opt_03_formaldehyde_regression.py
    python opt_03_formaldehyde_regression.py --n-trials 20
    python opt_03_formaldehyde_regression.py --device cuda:1
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Import core functions from the training script
from importlib.util import spec_from_file_location, module_from_spec

_spec = spec_from_file_location(
    "bnn1_form_reg",
    SCRIPT_DIR / "03_bnn1_formaldehyde_regression.py",
)
_mod = module_from_spec(_spec)

# Add code/ to sys.path for BNN module (needed by 03 script)
sys.path.insert(0, str(SCRIPT_DIR.parent))
_spec.loader.exec_module(_mod)

load_config = _mod.load_config
get_device = _mod.get_device
resolve_param = _mod.resolve_param
load_data_and_features = _mod.load_data_and_features
build_preprocessing = _mod.build_preprocessing
apply_preprocessing = _mod.apply_preprocessing
preprocess_and_concat = _mod.preprocess_and_concat
train_and_evaluate_fold = _mod.train_and_evaluate_fold
parse_pca_value = _mod.parse_pca_value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def search_or_fixed(trial, name, config_entry, is_pca=False):
    """Sample from search space if available, otherwise return fixed value.

    Config entries can be:
        - A dict with {value: ..., search: [...]}  → search over the list
        - A dict with {value: ..., search: null}   → fixed at value
        - A dict with {value: ...}                 → fixed at value
        - A bare value                             → fixed at that value

    For PCA params (is_pca=True), converts to/from string representation
    so Optuna can handle None and mixed int/float types.
    """
    # Determine if there's a search space
    search_space = None
    if isinstance(config_entry, dict) and "search" in config_entry:
        search_space = config_entry["search"]

    if search_space is None:
        # Fixed param — return value, don't register with Optuna
        return resolve_param(config_entry)

    if is_pca:
        options = ["none" if v is None else str(v) for v in search_space]
        result_str = trial.suggest_categorical(name, options)
        return parse_pca_value(result_str)

    # For hidden_dims (lists), serialize to JSON strings for Optuna
    if search_space and isinstance(search_space[0], list):
        result_str = trial.suggest_categorical(
            name, [json.dumps(d) for d in search_space],
        )
        return json.loads(result_str)

    return trial.suggest_categorical(name, search_space)


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

# Added to objective score when posterior has not moved from initialisation.
# Soft (not hard) so collapsed trials are ranked rather than silently dropped.
_COLLAPSE_PENALTY = 5.0
_COLLAPSE_THRESHOLD = 0.9  # collapse_score > this triggers penalty


def create_objective(X_wt, X_mut, y, config, device):
    """Create Optuna objective: minimize mean CV CRPS + soft collapse penalty.

    CRPS (Continuous Ranked Probability Score) is a proper scoring rule that
    rewards both calibration and sharpness — it penalises unnecessarily wide
    intervals more than NLPD does, preventing the model from hiding errors
    behind inflated uncertainty.

    A soft collapse penalty is added when the posterior weight std has not
    moved from its initialisation (ratio of mean_std / prior_std > 0.9),
    indicating the BNN is essentially operating as a MAP model.  The penalty
    is additive so collapsed trials are still ranked against each other rather
    than being silently discarded.

    Uses BOTH WT and mutant features for formaldehyde regression.
    Respects search: null in config — those params are fixed at their value.
    Records NLPD, Spearman, MAE, and collapse score per trial for diagnostics.
    """
    from sklearn.model_selection import KFold

    bnn1 = config["bnn1"]
    cv_config = config["cv"]
    preproc = config["preprocessing"]
    train_cfg = bnn1["training"]

    n_folds = cv_config["n_folds"]
    seed = cv_config["seed"]

    def objective(trial):
        # Sample or fix each hyperparameter based on its search field
        # Model
        hidden_dims    = search_or_fixed(trial, "hidden_dims",    bnn1["hidden_dims"])
        prior_std      = search_or_fixed(trial, "prior_std",      bnn1["prior_std"])
        dropout_rate   = search_or_fixed(trial, "dropout_rate",   bnn1["dropout_rate"])
        activation     = search_or_fixed(trial, "activation",     bnn1["activation"])
        # Training
        learning_rate  = search_or_fixed(trial, "learning_rate",  train_cfg["learning_rate"])
        kl_weight      = search_or_fixed(trial, "kl_weight",      train_cfg["kl_weight"])
        # Preprocessing — WT
        esm_wt_scaler  = search_or_fixed(trial, "esm_wt_scaler",  preproc["esm_wt"]["scaler"])
        esm_wt_pca     = search_or_fixed(trial, "esm_wt_pca",     preproc["esm_wt"]["pca"], is_pca=True)
        # Preprocessing — mutant
        esm_mut_scaler = search_or_fixed(trial, "esm_mut_scaler", preproc["esm_mut"]["scaler"])
        esm_mut_pca    = search_or_fixed(trial, "esm_mut_pca",    preproc["esm_mut"]["pca"], is_pca=True)

        params = {
            "hidden_dims":             hidden_dims,
            "prior_std":               prior_std,
            "dropout_rate":            dropout_rate,
            "activation":              activation,
            "learning_rate":           learning_rate,
            "kl_weight":               kl_weight,
            "esm_wt_scaler":           esm_wt_scaler,
            "esm_wt_pca":             esm_wt_pca,
            "esm_mut_scaler":          esm_mut_scaler,
            "esm_mut_pca":            esm_mut_pca,
            "batch_size":              resolve_param(train_cfg["batch_size"]),
            "kl_anneal_epochs":        resolve_param(train_cfg["kl_anneal_epochs"]),
            "n_epochs":                resolve_param(train_cfg["n_epochs"]),
            "early_stopping_patience": resolve_param(train_cfg["early_stopping_patience"]),
            "n_inference_samples":     resolve_param(train_cfg["n_inference_samples"]),
        }

        # --- K-fold CV ---
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_nlpds = []
        fold_spearmans = []
        fold_maes = []
        fold_crps = []
        fold_val_losses = []
        fold_collapse_scores = []

        for train_idx, val_idx in kf.split(X_wt):
            y_train, y_val = y[train_idx], y[val_idx]

            X_train, X_val, _, _ = preprocess_and_concat(
                X_wt[train_idx], X_wt[val_idx],
                X_mut[train_idx], X_mut[val_idx],
                params,
            )

            fold_metrics, _, _ = train_and_evaluate_fold(
                X_train, y_train, X_val, y_val,
                params, device,
            )
            fold_nlpds.append(fold_metrics["nlpd"])
            fold_spearmans.append(fold_metrics["spearman_rho"])
            fold_maes.append(fold_metrics["mae"])
            fold_crps.append(fold_metrics["crps"])
            fold_val_losses.append(fold_metrics.get("val_loss", float("nan")))
            fold_collapse_scores.append(fold_metrics.get("posterior_collapse_score", 1.0))

        mean_nlpd = float(np.mean(fold_nlpds))
        mean_spearman = float(np.mean(fold_spearmans))
        mean_mae = float(np.mean(fold_maes))
        mean_crps = float(np.mean(fold_crps))
        mean_val_loss = float(np.nanmean(fold_val_losses))
        mean_collapse = float(np.mean(fold_collapse_scores))

        # Soft collapse penalty: added to objective but trials are still ranked
        collapse_penalty = _COLLAPSE_PENALTY if mean_collapse > _COLLAPSE_THRESHOLD else 0.0

        trial.set_user_attr("fold_nlpds", fold_nlpds)
        trial.set_user_attr("fold_spearmans", fold_spearmans)
        trial.set_user_attr("fold_maes", fold_maes)
        trial.set_user_attr("fold_crps", fold_crps)
        trial.set_user_attr("fold_val_losses", fold_val_losses)
        trial.set_user_attr("mean_spearman", mean_spearman)
        trial.set_user_attr("mean_mae", mean_mae)
        trial.set_user_attr("mean_crps", mean_crps)
        trial.set_user_attr("mean_nlpd", mean_nlpd)
        trial.set_user_attr("mean_val_loss", mean_val_loss)
        trial.set_user_attr("mean_collapse_score", mean_collapse)
        trial.set_user_attr("collapse_penalty", collapse_penalty)

        objective_value = mean_crps + collapse_penalty
        logger.info(
            "Trial %d: CRPS=%.4f  collapse=%.3f  penalty=%.1f  obj=%.4f"
            "  NLPD=%.4f  ρ=%.4f  MAE=%.4f  params=%s",
            trial.number, mean_crps, mean_collapse, collapse_penalty, objective_value,
            mean_nlpd, mean_spearman, mean_mae,
            {k: v for k, v in trial.params.items()},
        )

        return objective_value  # MINIMIZE: CRPS + collapse penalty

    return objective


def decode_trial_params(raw_params: dict) -> dict:
    """Decode Optuna trial params back into native Python types."""
    params = dict(raw_params)
    if "hidden_dims" in params:
        params["hidden_dims"] = json.loads(params["hidden_dims"])
    if "esm_wt_pca" in params:
        params["esm_wt_pca"] = parse_pca_value(params["esm_wt_pca"])
    if "esm_mut_pca" in params:
        params["esm_mut_pca"] = parse_pca_value(params["esm_mut_pca"])
    return params


def build_full_params(searched_params: dict, config: dict) -> dict:
    """Merge searched params with fixed config defaults for a complete param set."""
    bnn1 = config["bnn1"]
    train_cfg = bnn1["training"]
    preproc = config["preprocessing"]

    full = {
        "hidden_dims":    searched_params.get("hidden_dims",    resolve_param(bnn1["hidden_dims"])),
        "prior_std":      searched_params.get("prior_std",      resolve_param(bnn1["prior_std"])),
        "dropout_rate":   searched_params.get("dropout_rate",   resolve_param(bnn1["dropout_rate"])),
        "activation":     searched_params.get("activation",     resolve_param(bnn1["activation"])),
        "learning_rate":  searched_params.get("learning_rate",  resolve_param(train_cfg["learning_rate"])),
        "kl_weight":      searched_params.get("kl_weight",      resolve_param(train_cfg["kl_weight"])),
        "esm_wt_scaler":  searched_params.get("esm_wt_scaler",  resolve_param(preproc["esm_wt"]["scaler"])),
        "esm_wt_pca":     searched_params.get("esm_wt_pca",     parse_pca_value(resolve_param(preproc["esm_wt"]["pca"]))),
        "esm_mut_scaler": searched_params.get("esm_mut_scaler", resolve_param(preproc["esm_mut"]["scaler"])),
        "esm_mut_pca":    searched_params.get("esm_mut_pca",    parse_pca_value(resolve_param(preproc["esm_mut"]["pca"]))),
    }
    return full


def build_rerun_command(full_params: dict, device: str) -> str:
    """Build a CLI command to re-run 03 with the best hyperparams."""
    parts = ["python 03_bnn1_formaldehyde_regression.py"]
    parts.append(f"--device {device}")
    parts.append(f"--hidden-dims '{json.dumps(full_params['hidden_dims'])}'")
    parts.append(f"--prior-std {full_params['prior_std']}")
    parts.append(f"--dropout-rate {full_params['dropout_rate']}")
    parts.append(f"--learning-rate {full_params['learning_rate']}")
    parts.append(f"--kl-weight {full_params['kl_weight']}")
    parts.append(f"--esm-wt-scaler {full_params['esm_wt_scaler']}")

    wt_pca = full_params.get("esm_wt_pca")
    parts.append(f"--esm-wt-pca {'none' if wt_pca is None else wt_pca}")

    parts.append(f"--esm-mut-scaler {full_params['esm_mut_scaler']}")

    mut_pca = full_params.get("esm_mut_pca")
    parts.append(f"--esm-mut-pca {'none' if mut_pca is None else mut_pca}")

    return " \\\n    ".join(parts)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_hyperopt_tradeoffs(trial_results: list, output_path):
    """Plot tradeoffs between NLPD (objective), Spearman, MAE, and CRPS across trials."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Extract metrics — value is now NLPD (objective)
    completed = [t for t in trial_results if t["value"] is not None]
    if len(completed) < 2:
        logger.warning("Not enough completed trials for tradeoff plot")
        return

    nlpds = [t["value"] for t in completed]
    spearmans = [t.get("mean_spearman", float("nan")) for t in completed]
    maes = [t.get("mean_mae", float("nan")) for t in completed]
    crps_vals = [t.get("mean_crps", float("nan")) for t in completed]
    trial_nums = list(range(len(nlpds)))

    # Best trial = lowest NLPD
    best_idx = int(np.argmin(nlpds))

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
                       zorder=5, label="Best (NLPD)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        return sc

    # Panel 1: NLPD vs Spearman (key tradeoff: does accuracy come at calibration cost?)
    _scatter(axes[0], nlpds, spearmans,
             "NLPD (objective, lower=better)", "Spearman ρ (higher=better)",
             "NLPD vs Spearman", nlpds[best_idx], spearmans[best_idx])

    # Panel 2: NLPD vs MAE
    _scatter(axes[1], nlpds, maes,
             "NLPD (objective)", "MAE",
             "NLPD vs MAE", nlpds[best_idx], maes[best_idx])

    # Panel 3: NLPD vs CRPS (both proper scoring rules — should correlate)
    _scatter(axes[2], nlpds, crps_vals,
             "NLPD (objective)", "CRPS",
             "NLPD vs CRPS", nlpds[best_idx], crps_vals[best_idx])

    # Panel 4: Spearman vs MAE (classic accuracy tradeoff)
    sc = _scatter(axes[3], spearmans, maes,
                  "Spearman ρ", "MAE",
                  "Spearman vs MAE", spearmans[best_idx], maes[best_idx])
    if sc is not None:
        plt.colorbar(sc, ax=axes[3], label="Trial number")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna hyperopt for BNN1 formaldehyde regression",
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Number of Optuna trials (default: from config)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def setup_logging(results_dir) -> None:
    """Configure logging to both console and file."""
    log_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(log_fmt)

    file_handler = logging.FileHandler(results_dir / "run.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)


def main():
    args = parse_args()
    t_start = time.time()

    # 1. Setup
    results_dir = PROJECT_ROOT / "results" / "opt_03_formaldehyde_regression"
    results_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(results_dir)

    logger.info("=" * 60)
    logger.info("opt_03_formaldehyde_regression.py")
    logger.info("Optuna Hyperopt for Formaldehyde Regression")
    logger.info("=" * 60)

    config = load_config(args.config)
    device = get_device(config, args.device)
    logger.info("Results directory: %s", results_dir)

    # 2. Load data (WT + mutant for regression)
    processed_dir = PROJECT_ROOT / config["data"]["output_dir"]
    X_wt, X_mut, y, positions, mutation_strings = \
        load_data_and_features(config, processed_dir)
    logger.info("Features: WT %s + Mut %s, targets: %s",
                X_wt.shape, X_mut.shape, y.shape)

    # 3. Run Optuna
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_trials = args.n_trials or config["cv"]["n_hyperopt_trials"]
    logger.info("Starting Optuna (%d trials, %d-fold CV, minimize NLPD)...",
                n_trials, config["cv"]["n_folds"])

    objective = create_objective(X_wt, X_mut, y, config, device)
    study = optuna.create_study(
        direction="minimize",  # NLPD: lower is better (proper scoring rule)
        sampler=optuna.samplers.TPESampler(seed=config["cv"]["seed"]),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # 4. Parse best trial
    best_trial = study.best_trial
    searched_params = decode_trial_params(best_trial.params)
    best_params = build_full_params(searched_params, config)

    logger.info("Best trial: #%d", best_trial.number)
    logger.info("  Mean CV NLPD:     %.4f  (objective, lower = better)", best_trial.value)
    logger.info("  Mean CV Spearman: %.4f", best_trial.user_attrs.get("mean_spearman", float("nan")))
    logger.info("  Mean CV MAE:      %.4f", best_trial.user_attrs.get("mean_mae", float("nan")))
    logger.info("  Mean CV CRPS:     %.4f", best_trial.user_attrs.get("mean_crps", float("nan")))
    logger.info("  Mean CV val_loss: %.4f", best_trial.user_attrs.get("mean_val_loss", float("nan")))
    for k, v in best_params.items():
        searched = "(searched)" if k in best_trial.params else "(fixed)"
        logger.info("  %s: %s  %s", k, v, searched)

    # 5. Save results
    # All trials
    trial_results = []
    for trial in study.trials:
        params = decode_trial_params(trial.params)
        full = build_full_params(params, config)
        trial_results.append({
            "number": trial.number,
            "value": trial.value,  # NLPD (objective)
            "params": full,
            "searched_params": params,
            "fold_nlpds": trial.user_attrs.get("fold_nlpds"),
            "fold_spearmans": trial.user_attrs.get("fold_spearmans"),
            "fold_maes": trial.user_attrs.get("fold_maes"),
            "fold_crps": trial.user_attrs.get("fold_crps"),
            "fold_val_losses": trial.user_attrs.get("fold_val_losses"),
            "mean_spearman": trial.user_attrs.get("mean_spearman"),
            "mean_mae": trial.user_attrs.get("mean_mae"),
            "mean_crps": trial.user_attrs.get("mean_crps"),
            "mean_val_loss": trial.user_attrs.get("mean_val_loss"),
        })

    with open(results_dir / "study_results.json", "w") as f:
        json.dump(trial_results, f, indent=2, default=str)
    logger.info("Saved study_results.json (%d trials)", len(trial_results))

    # Best params
    best_save = dict(best_params)
    best_save["mean_cv_nlpd"] = float(best_trial.value)
    best_save["mean_cv_spearman"] = best_trial.user_attrs.get("mean_spearman")
    best_save["mean_cv_mae"] = best_trial.user_attrs.get("mean_mae")
    best_save["mean_cv_crps"] = best_trial.user_attrs.get("mean_crps")
    best_save["mean_cv_val_loss"] = best_trial.user_attrs.get("mean_val_loss")
    best_save["fold_nlpds"] = best_trial.user_attrs.get("fold_nlpds")
    best_save["fold_spearmans"] = best_trial.user_attrs.get("fold_spearmans")
    best_save["fold_maes"] = best_trial.user_attrs.get("fold_maes")
    best_save["trial_number"] = best_trial.number
    with open(results_dir / "best_hyperparams.json", "w") as f:
        json.dump(best_save, f, indent=2, default=str)
    logger.info("Saved best_hyperparams.json")

    # Re-run command
    rerun_cmd = build_rerun_command(best_params, device)
    with open(results_dir / "best_command.txt", "w") as f:
        f.write(rerun_cmd + "\n")
    logger.info("Saved best_command.txt")

    with open(results_dir / "config_used.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # 6. Tradeoff plot
    plot_hyperopt_tradeoffs(trial_results, results_dir / "hyperopt_tradeoffs.png")

    # 7. Summary
    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("Hyperopt Complete (%.1fs, %d trials)", elapsed, n_trials)
    logger.info("=" * 60)
    logger.info("Best NLPD:     %.4f  (objective)", best_trial.value)
    logger.info("Best Spearman: %.4f", best_trial.user_attrs.get("mean_spearman", float("nan")))
    logger.info("Best MAE:      %.4f", best_trial.user_attrs.get("mean_mae", float("nan")))
    logger.info("Best CRPS:     %.4f", best_trial.user_attrs.get("mean_crps", float("nan")))
    logger.info("To reproduce with full evaluation + final model:")
    logger.info("  %s", rerun_cmd)


if __name__ == "__main__":
    main()
