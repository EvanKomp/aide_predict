#!/usr/bin/env python
"""
opt_02_position_classification.py — Optuna hyperopt for position classification
================================================================================

Wraps 02_bnn1_position_classification.py: searches over all hyperparameters
defined in config.yaml search spaces, evaluates each trial via stratified
K-fold CV, and reports the best configuration.

After finding best params, re-run 02 with those params for full evaluation
+ final model training:

    python 02_bnn1_position_classification.py \\
        --hidden-dims '[128, 64]' --prior-std 0.5 ...

Outputs to results/opt_02_position_classification/:
  - study_results.json: all trial results
  - best_hyperparams.json: best trial params
  - best_command.txt: ready-to-run command for 02 with best params

Usage:
    python opt_02_position_classification.py
    python opt_02_position_classification.py --n-trials 20
    python opt_02_position_classification.py --device cuda:1
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
    "bnn1_pos_cls",
    SCRIPT_DIR / "02_bnn1_position_classification.py",
)
_mod = module_from_spec(_spec)

# Add code/ to sys.path for BNN module (needed by 02 script)
sys.path.insert(0, str(SCRIPT_DIR.parent))
_spec.loader.exec_module(_mod)

load_config = _mod.load_config
get_device = _mod.get_device
resolve_param = _mod.resolve_param
load_data_and_features = _mod.load_data_and_features
build_preprocessing = _mod.build_preprocessing
apply_preprocessing = _mod.apply_preprocessing
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

def create_objective(X, y, config, device, n_classes):
    """Create Optuna objective: K-fold CV accuracy for a sampled config.

    Uses mutant-only features (no WT) for position classification.
    Respects search: null in config — those params are fixed at their value.
    """
    from sklearn.model_selection import StratifiedKFold

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
        # Preprocessing (mutant features only)
        esm_mut_scaler = search_or_fixed(trial, "esm_mut_scaler", preproc["esm_mut"]["scaler"])
        esm_mut_pca    = search_or_fixed(trial, "esm_mut_pca",    preproc["esm_mut"]["pca"], is_pca=True)

        params = {
            "hidden_dims":             hidden_dims,
            "prior_std":               prior_std,
            "dropout_rate":            dropout_rate,
            "activation":              activation,
            "learning_rate":           learning_rate,
            "kl_weight":               kl_weight,
            "esm_mut_scaler":          esm_mut_scaler,
            "esm_mut_pca":            esm_mut_pca,
            "batch_size":              resolve_param(train_cfg["batch_size"]),
            "kl_anneal_epochs":        resolve_param(train_cfg["kl_anneal_epochs"]),
            "n_epochs":                resolve_param(train_cfg["n_epochs"]),
            "early_stopping_patience": resolve_param(train_cfg["early_stopping_patience"]),
            "n_inference_samples":     resolve_param(train_cfg["n_inference_samples"]),
        }

        # --- Stratified K-fold CV ---
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_accs = []

        for train_idx, val_idx in skf.split(X, y):
            y_train, y_val = y[train_idx], y[val_idx]

            pipe = build_preprocessing(params["esm_mut_scaler"], params["esm_mut_pca"])
            X_train_p, X_val_p = apply_preprocessing(pipe, X[train_idx], X[val_idx])

            acc, _, _, _ = train_and_evaluate_fold(
                X_train_p, y_train, X_val_p, y_val,
                n_classes, params, device,
            )
            fold_accs.append(acc)

        mean_acc = float(np.mean(fold_accs))
        trial.set_user_attr("fold_accuracies", fold_accs)
        return mean_acc

    return objective


def decode_trial_params(raw_params: dict) -> dict:
    """Decode Optuna trial params back into native Python types.

    Optuna stores list-type params as JSON strings and PCA as string.
    Params that were fixed (search: null) won't appear in trial.params.
    """
    params = dict(raw_params)
    if "hidden_dims" in params:
        params["hidden_dims"] = json.loads(params["hidden_dims"])
    if "esm_mut_pca" in params:
        params["esm_mut_pca"] = parse_pca_value(params["esm_mut_pca"])
    return params


def build_full_params(searched_params: dict, config: dict) -> dict:
    """Merge searched params with fixed config defaults for a complete param set.

    Optuna's trial.params only contains params that were actually searched.
    Params with search: null are fixed and won't appear — fill them from config.
    """
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
        "esm_mut_scaler": searched_params.get("esm_mut_scaler", resolve_param(preproc["esm_mut"]["scaler"])),
        "esm_mut_pca":    searched_params.get("esm_mut_pca",    parse_pca_value(resolve_param(preproc["esm_mut"]["pca"]))),
    }
    return full


def build_rerun_command(full_params: dict, device: str) -> str:
    """Build a CLI command to re-run 02 with the best hyperparams."""
    parts = ["python 02_bnn1_position_classification.py"]
    parts.append(f"--device {device}")
    parts.append(f"--hidden-dims '{json.dumps(full_params['hidden_dims'])}'")
    parts.append(f"--prior-std {full_params['prior_std']}")
    parts.append(f"--dropout-rate {full_params['dropout_rate']}")
    parts.append(f"--learning-rate {full_params['learning_rate']}")
    parts.append(f"--kl-weight {full_params['kl_weight']}")
    parts.append(f"--esm-mut-scaler {full_params['esm_mut_scaler']}")

    mut_pca = full_params.get("esm_mut_pca")
    parts.append(f"--esm-mut-pca {'none' if mut_pca is None else mut_pca}")

    return " \\\n    ".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna hyperopt for BNN1 position classification",
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

def setup_logging(results_dir: Path) -> None:
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
    results_dir = PROJECT_ROOT / "results" / "opt_02_position_classification"
    results_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(results_dir)

    logger.info("=" * 60)
    logger.info("opt_02_position_classification.py")
    logger.info("Optuna Hyperopt for Position Classification")
    logger.info("=" * 60)

    config = load_config(args.config)
    device = get_device(config, args.device)
    logger.info("Results directory: %s", results_dir)

    # 2. Load data (mutant-only for position classification)
    processed_dir = PROJECT_ROOT / config["data"]["output_dir"]
    X_mut, y, position_labels, mutation_strings, n_classes = \
        load_data_and_features(config, processed_dir)
    logger.info("Using mutant-only features: %s", X_mut.shape)

    # 3. Run Optuna
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_trials = args.n_trials or config["cv"]["n_hyperopt_trials"]
    logger.info("Starting Optuna (%d trials, %d-fold CV)...",
                n_trials, config["cv"]["n_folds"])

    objective = create_objective(X_mut, y, config, device, n_classes)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config["cv"]["seed"]),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # 4. Parse best trial
    best_trial = study.best_trial
    searched_params = decode_trial_params(best_trial.params)
    # Merge searched params with fixed config defaults for a complete set
    best_params = build_full_params(searched_params, config)

    logger.info("Best trial: #%d", best_trial.number)
    logger.info("  Mean CV accuracy: %.4f", best_trial.value)
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
            "value": trial.value,
            "params": full,
            "searched_params": params,
            "fold_accuracies": trial.user_attrs.get("fold_accuracies"),
        })

    with open(results_dir / "study_results.json", "w") as f:
        json.dump(trial_results, f, indent=2, default=str)
    logger.info("Saved study_results.json (%d trials)", len(trial_results))

    # Best params
    best_save = dict(best_params)
    best_save["mean_cv_accuracy"] = float(best_trial.value)
    best_save["fold_accuracies"] = best_trial.user_attrs.get("fold_accuracies")
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

    # 6. Summary
    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("Hyperopt Complete (%.1fs, %d trials)", elapsed, n_trials)
    logger.info("=" * 60)
    logger.info("Best accuracy: %.4f", best_trial.value)
    logger.info("To reproduce with full evaluation + final model:")
    logger.info("  %s", rerun_cmd)


if __name__ == "__main__":
    main()
