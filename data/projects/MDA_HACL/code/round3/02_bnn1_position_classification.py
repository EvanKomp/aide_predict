#!/usr/bin/env python
"""
02_bnn1_position_classification.py — BNN1 Phase A: Position Classification
==========================================================================

Sanity check: given ESM2 residue embeddings of MUTANT sequences, classify
which position a mutation is at. 64-class classification over ~1,272
formaldehyde SSM mutations.

Uses ONLY the mutant embedding (not WT) so that every data point has a
unique feature vector. The WT embedding at a given position is identical
for all ~20 AA substitutions there, so including it would let the model
trivially memorize position from the repeated WT vector. By using only
the mutant embedding, we test whether ESM2 natively encodes positional
information in the residue representation — even when the amino acid
identity varies.

Pipeline:
  1. Load data + ESM2 mutant embeddings
  2. Preprocess features (optional scaling → optional PCA)
  3. Stratified K-fold CV → evaluate accuracy
  4. Train final model on ALL data → save model + preprocessing

All hyperparameters are CLI args with defaults read from config.yaml.
Use opt_02_position_classification.py for automated hyperopt.

Outputs to results/02_position_classification/:
  - metrics.json, hyperparams.json, config_used.yaml
  - confusion_matrix.png, per_position_accuracy.png
  - training_curves.png, loss_decomposition.png
  - models/final_model.pt, models/preprocessing.joblib

Usage:
    python 02_bnn1_position_classification.py
    python 02_bnn1_position_classification.py --hidden-dims '[128, 64]' --prior-std 0.5
    python 02_bnn1_position_classification.py --esm-mut-scaler standard --esm-mut-pca 0.95
    python 02_bnn1_position_classification.py --device cuda:1
"""

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent        # code/round3/
PROJECT_ROOT = SCRIPT_DIR.parent.parent             # MDA_HACL/

# Add code/ to sys.path for BNN module import
sys.path.insert(0, str(SCRIPT_DIR.parent))          # MDA_HACL/code/
from bnns import BayesianMLP, BNNTrainer, TrainingConfig, TrainingHistory


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[str] = None) -> dict:
    """Load config.yaml."""
    path = Path(config_path) if config_path else SCRIPT_DIR / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config from %s", path)
    return config


def resolve_param(config_value, cli_override=None):
    """Resolve a parameter value with priority: CLI > config value > first search value.

    Handles three config formats:
        hidden_dims: [128, 64]                                → direct value
        hidden_dims: {search: [[64], [128, 64]]}              → first of search list
        hidden_dims: {value: [128, 64], search: [[64], ...]}  → value field

    CLI override always wins.
    """
    if cli_override is not None:
        return cli_override
    if isinstance(config_value, dict):
        if "value" in config_value:
            return config_value["value"]
        if "search" in config_value:
            return config_value["search"][0]
    return config_value


def get_device(config: dict, override: Optional[str] = None) -> str:
    """Resolve device string. 'auto' selects best available."""
    device = override or config.get("compute", {}).get("device", "cpu")
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    logger.info("Using device: %s", device)
    return device


# ---------------------------------------------------------------------------
# Data loading and feature construction
# ---------------------------------------------------------------------------

def load_data_and_features(
    config: dict,
    processed_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[str], int]:
    """Load formaldehyde SSM data and ESM2 mutant embeddings.

    Only loads mutant residue embeddings — WT embeddings are not used for
    position classification (see module docstring for rationale).

    Returns:
        X_mut: (n_mutations, esm_dim) mutant residue embeddings
        y: (n_mutations,) position class indices
        position_labels_1idx: 1-indexed position labels (length n_classes)
        mutation_strings: mutation string per row
        n_classes: number of unique positions
    """
    form_path = processed_dir / "formaldehyde_ssm.csv"
    form_df = pd.read_csv(form_path)
    logger.info("Loaded formaldehyde SSM: %d mutations at %d positions",
                len(form_df), form_df["position"].nunique())

    emb_dir = processed_dir / "embeddings"
    esm_mutant = np.load(emb_dir / "esm2_mutant_residues.npz")
    logger.info("ESM2 mutant embeddings: %d entries", len(esm_mutant.files))

    unique_positions = sorted(form_df["position"].unique())
    position_to_class = {pos: i for i, pos in enumerate(unique_positions)}
    n_classes = len(unique_positions)
    logger.info("Position classes: %d", n_classes)

    position_offset = config["data"]["position_offset"]
    position_labels_1idx = [pos + position_offset + 1 for pos in unique_positions]

    mut_list = []
    y_list = []
    mutation_strings = []

    for _, row in form_df.iterrows():
        ms = row["mutation_string"]
        pos = row["position"]

        if ms not in esm_mutant:
            warnings.warn(f"Missing ESM2 mutant embedding for {ms}, skipping")
            continue

        mut_list.append(esm_mutant[ms])     # (1280,)
        y_list.append(position_to_class[pos])
        mutation_strings.append(ms)

    X_mut = np.stack(mut_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    counts = np.bincount(y)
    logger.info("Mutant features: %s", X_mut.shape)
    logger.info("Targets: %s, %d classes, min count=%d, max count=%d",
                y.shape, n_classes, counts.min(), counts.max())

    return X_mut, y, position_labels_1idx, mutation_strings, n_classes


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def build_preprocessing(scaler_type: str, pca_components):
    """Build sklearn preprocessing pipeline: optional scaler → optional PCA.

    Args:
        scaler_type: "none", "standard", or "robust"
        pca_components: None (skip), int (n components), or float 0<x<1 (variance)

    Returns a Pipeline, or None if no preprocessing is requested.
    """
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler, StandardScaler

    steps = []
    if scaler_type == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaler_type == "robust":
        steps.append(("scaler", RobustScaler()))

    if pca_components is not None:
        steps.append(("pca", PCA(n_components=pca_components)))

    if not steps:
        return None
    return Pipeline(steps)


def apply_preprocessing(pipeline, X_train: np.ndarray, X_val: np.ndarray):
    """Fit on X_train, transform both. Returns (X_train_t, X_val_t)."""
    if pipeline is None:
        return X_train.copy(), X_val.copy()
    X_train_t = pipeline.fit_transform(X_train).astype(np.float32)
    X_val_t = pipeline.transform(X_val).astype(np.float32)
    return X_train_t, X_val_t


# ---------------------------------------------------------------------------
# Single fold training + evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    params: dict,
    device: str,
    return_predictions: bool = False,
) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray], Optional[TrainingHistory]]:
    """Train BNN on one fold, evaluate on val set.

    Returns:
        accuracy: validation accuracy
        predictions: (n_val,) predicted class indices  (if return_predictions)
        probabilities: (n_val, n_classes) class probs   (if return_predictions)
        history: TrainingHistory with loss curves        (if return_predictions)
    """
    model = BayesianMLP(
        input_dim=X_train.shape[1],
        hidden_dims=params["hidden_dims"],
        output_dim=n_classes,
        task="classification",
        prior_std=params["prior_std"],
        dropout_rate=params["dropout_rate"],
        activation=params.get("activation", "silu"),
    )

    training_config = TrainingConfig(
        n_epochs=params["n_epochs"],
        learning_rate=params["learning_rate"],
        batch_size=params["batch_size"],
        kl_anneal_epochs=params["kl_anneal_epochs"],
        kl_weight=params["kl_weight"],
        early_stopping_patience=params["early_stopping_patience"],
        n_inference_samples=params["n_inference_samples"],
        device=device,
        verbose=return_predictions,
        log_interval=1,
    )

    trainer = BNNTrainer(model, training_config)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    history = trainer.fit(X_train_t, y_train_t, X_val_t, y_val_t)

    estimates = trainer.predict(X_val_t)
    predicted_probs = estimates.mean.cpu().numpy()            # (n_val, n_classes)
    predicted_classes = np.argmax(predicted_probs, axis=1)    # (n_val,)
    accuracy = float((predicted_classes == y_val).mean())

    if return_predictions:
        return accuracy, predicted_classes, predicted_probs, history
    return accuracy, None, None, None


# ---------------------------------------------------------------------------
# K-fold CV evaluation
# ---------------------------------------------------------------------------

def evaluate_with_cv(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    config: dict,
    device: str,
    n_classes: int,
) -> Tuple[dict, np.ndarray, np.ndarray, List]:
    """Run stratified K-fold CV, collecting out-of-fold predictions.

    Preprocessing (scaling → PCA) is fit per fold on training data only.

    Returns:
        metrics: dict of evaluation metrics
        all_y_true: (n_samples,) true labels
        all_y_pred: (n_samples,) predicted labels
        fold_histories: list of TrainingHistory per fold
    """
    from sklearn.model_selection import StratifiedKFold

    cv_config = config["cv"]
    skf = StratifiedKFold(
        n_splits=cv_config["n_folds"],
        shuffle=True,
        random_state=cv_config["seed"],
    )

    all_y_true = np.zeros(len(y), dtype=np.int64)
    all_y_pred = np.zeros(len(y), dtype=np.int64)
    all_y_probs = np.zeros((len(y), n_classes), dtype=np.float32)
    fold_accs = []
    fold_histories = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info("Fold %d/%d", fold_idx + 1, cv_config["n_folds"])
        y_train, y_val = y[train_idx], y[val_idx]

        pipe = build_preprocessing(params["esm_mut_scaler"], params["esm_mut_pca"])
        X_train_p, X_val_p = apply_preprocessing(pipe, X[train_idx], X[val_idx])

        acc, predictions, probabilities, history = train_and_evaluate_fold(
            X_train_p, y_train, X_val_p, y_val,
            n_classes, params, device,
            return_predictions=True,
        )

        all_y_true[val_idx] = y_val
        all_y_pred[val_idx] = predictions
        all_y_probs[val_idx] = probabilities
        fold_accs.append(acc)
        fold_histories.append(history)
        logger.info("  Accuracy: %.4f", acc)

    # --- Compute metrics ---
    overall_acc = float((all_y_true == all_y_pred).mean())

    # Top-5 accuracy
    top5_preds = np.argsort(all_y_probs, axis=1)[:, -5:]
    top5_correct = [all_y_true[i] in top5_preds[i] for i in range(len(all_y_true))]
    top5_acc = float(np.mean(top5_correct))

    # Per-class accuracy
    per_class_acc = {}
    per_class_n = {}
    for c in range(n_classes):
        mask = all_y_true == c
        if mask.sum() > 0:
            per_class_acc[c] = float((all_y_pred[mask] == c).mean())
            per_class_n[c] = int(mask.sum())

    metrics = {
        "overall_accuracy": overall_acc,
        "top5_accuracy": top5_acc,
        "fold_accuracies": [float(a) for a in fold_accs],
        "mean_fold_accuracy": float(np.mean(fold_accs)),
        "std_fold_accuracy": float(np.std(fold_accs)),
        "n_classes": n_classes,
        "n_samples": len(y),
        "random_baseline": float(1.0 / n_classes),
        "per_class_accuracy": {str(k): v for k, v in per_class_acc.items()},
        "per_class_n": {str(k): v for k, v in per_class_n.items()},
        "mean_per_class_accuracy": float(np.mean(list(per_class_acc.values()))),
    }

    logger.info("Overall accuracy:        %.4f", overall_acc)
    logger.info("Top-5 accuracy:          %.4f", top5_acc)
    logger.info("Mean per-class accuracy: %.4f", metrics["mean_per_class_accuracy"])
    logger.info("Random baseline:         %.4f", metrics["random_baseline"])

    return metrics, all_y_true, all_y_pred, fold_histories


# ---------------------------------------------------------------------------
# Train final model on all data
# ---------------------------------------------------------------------------

def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    n_classes: int,
    device: str,
    models_dir: Path,
):
    """Fit preprocessing on all data, train final BNN, save both.

    Saves:
        models_dir/preprocessing.joblib  — preprocessing pipeline
        models_dir/final_model.pt        — BNN checkpoint via trainer.save()
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    # Fit preprocessing on full dataset
    pipe = build_preprocessing(params["esm_mut_scaler"], params["esm_mut_pca"])
    X_processed = pipe.fit_transform(X).astype(np.float32) if pipe else X.copy()

    joblib.dump(pipe, models_dir / "preprocessing.joblib")
    logger.info("Saved preprocessing pipeline -> preprocessing.joblib")

    # Build and train BNN
    model = BayesianMLP(
        input_dim=X_processed.shape[1],
        hidden_dims=params["hidden_dims"],
        output_dim=n_classes,
        task="classification",
        prior_std=params["prior_std"],
        dropout_rate=params["dropout_rate"],
        activation=params.get("activation", "silu"),
    )

    training_config = TrainingConfig(
        n_epochs=params["n_epochs"],
        learning_rate=params["learning_rate"],
        batch_size=params["batch_size"],
        kl_anneal_epochs=params["kl_anneal_epochs"],
        kl_weight=params["kl_weight"],
        early_stopping_patience=params["early_stopping_patience"],
        n_inference_samples=params["n_inference_samples"],
        device=device,
        verbose=True,
        log_interval=50,
    )

    trainer = BNNTrainer(model, training_config)

    X_t = torch.tensor(X_processed, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    # Train on all data (no val set — CV already evaluated generalization)
    trainer.fit(X_t, y_t)

    model_path = str(models_dir / "final_model.pt")
    trainer.save(model_path)
    logger.info("Saved final model -> final_model.pt")
    logger.info("Input dim: %d, architecture: %s, output dim: %d",
                X_processed.shape[1], params["hidden_dims"], n_classes)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    position_labels: List[int],
    output_path: Path,
):
    """Plot normalized confusion matrix (recall per class) as a heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Recall (fraction classified correctly)")

    tick_positions = list(range(0, n_classes, 4))
    tick_labels = [str(position_labels[i]) for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=7)

    ax.set_xlabel("Predicted position (1-indexed)")
    ax.set_ylabel("True position (1-indexed)")
    ax.set_title("Position Classification — Normalized Confusion Matrix")

    diag_acc = float(np.diag(cm_norm).mean())
    ax.text(
        0.02, 0.98, f"Mean diagonal recall: {diag_acc:.3f}",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_per_position_accuracy(
    per_class_acc: dict,
    position_labels: List[int],
    random_baseline: float,
    output_path: Path,
):
    """Bar chart of classification accuracy per position."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = sorted(per_class_acc.keys(), key=int)
    accs = [per_class_acc[c] for c in classes]
    labels = [str(position_labels[int(c)]) for c in classes]
    mean_acc = float(np.mean(accs))

    fig, ax = plt.subplots(figsize=(18, 5))
    colors = ["#2196F3" if a > random_baseline else "#F44336" for a in accs]
    ax.bar(range(len(classes)), accs, color=colors, edgecolor="none", width=0.8)

    ax.axhline(
        random_baseline, color="gray", linestyle="--", linewidth=1,
        label=f"Random baseline ({random_baseline:.4f})",
    )
    ax.axhline(
        mean_acc, color="orange", linestyle="-", linewidth=1.5,
        label=f"Mean accuracy ({mean_acc:.3f})",
    )

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_xlabel("Position (1-indexed)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Position Classification Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_training_curves(
    fold_histories: List,
    output_path: Path,
):
    """Plot training and validation loss curves for all CV folds.

    Two panels:
        Left:  Train loss (all folds overlaid) + best epoch markers
        Right: Validation loss (all folds overlaid) + best epoch markers
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_folds = len(fold_histories)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_folds, 10)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, h in enumerate(fold_histories):
        epochs = range(1, len(h.train_loss) + 1)
        label = f"Fold {i + 1}"
        c = colors[i]

        axes[0].plot(epochs, h.train_loss, color=c, alpha=0.7, label=label)
        if h.val_loss:
            axes[1].plot(
                range(1, len(h.val_loss) + 1), h.val_loss,
                color=c, alpha=0.7, label=label,
            )
            axes[1].axvline(
                h.best_epoch + 1, color=c, linestyle=":", alpha=0.4, linewidth=1,
            )

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("ELBO Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend(fontsize=7)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Loss")
    axes[1].set_title("Validation Loss (dotted = best epoch)")
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_loss_decomposition(
    fold_histories: List,
    output_path: Path,
):
    """Plot NLL vs KL decomposition for fold 1 with KL annealing schedule.

    Three-panel plot showing how the ELBO decomposes during training.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = fold_histories[0]  # first fold
    epochs = range(1, len(h.train_loss) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1: NLL
    axes[0].plot(epochs, h.train_nll, color="#2196F3", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("NLL")
    axes[0].set_title("Negative Log-Likelihood (Fold 1)")

    # Panel 2: KL divergence (raw, before beta scaling)
    axes[1].plot(epochs, h.train_kl, color="#FF9800", linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("KL Divergence")
    axes[1].set_title("KL Divergence (Fold 1)")

    # Panel 3: KL annealing schedule
    axes[2].plot(epochs, h.kl_weight_schedule, color="#4CAF50", linewidth=1.5)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Beta (KL Weight)")
    axes[2].set_title("KL Annealing Schedule")
    axes[2].set_ylim(-0.05, max(h.kl_weight_schedule) * 1.1)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BNN1 Phase A: Position Classification Sanity Check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 02_bnn1_position_classification.py
  python 02_bnn1_position_classification.py --hidden-dims '[128, 64]' --prior-std 0.5
  python 02_bnn1_position_classification.py --esm-mut-scaler standard --esm-mut-pca 0.95
  python 02_bnn1_position_classification.py --device cuda:1
        """,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (cpu, cuda, cuda:1, mps, auto)")

    # Model hyperparams
    parser.add_argument("--hidden-dims", type=str, default=None,
                        help="Hidden layer dims as JSON, e.g. '[128, 64]'")
    parser.add_argument("--prior-std", type=float, default=None)
    parser.add_argument("--dropout-rate", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--kl-weight", type=float, default=None)

    # Preprocessing hyperparams
    parser.add_argument("--esm-mut-scaler", type=str, default=None,
                        choices=["none", "standard", "robust"],
                        help="Scaler for ESM mutant features")
    parser.add_argument("--esm-mut-pca", type=str, default=None,
                        help="PCA for ESM mutant: int (n components), float 0<x<1 "
                             "(variance fraction), or 'none'")

    # Flags
    parser.add_argument("--skip-final-model", action="store_true",
                        help="Skip training final model on all data (CV only)")

    return parser.parse_args()


def parse_pca_value(raw):
    """Parse a PCA n_components value: None, int, or float (variance fraction)."""
    if raw is None:
        return None
    if isinstance(raw, str):
        if raw.lower() == "none":
            return None
        raw = float(raw) if "." in raw else int(raw)
    if isinstance(raw, float) and raw >= 1.0:
        return int(raw)  # e.g. 64.0 → 64
    return raw  # int or float in (0, 1)


def resolve_all_params(args: argparse.Namespace, config: dict) -> dict:
    """Build complete params dict from CLI args + config defaults."""
    bnn1 = config["bnn1"]
    train = bnn1["training"]
    preproc = config["preprocessing"]  # top-level preprocessing section

    # Parse hidden_dims from CLI string if provided
    hidden_dims_cli = json.loads(args.hidden_dims) if args.hidden_dims else None

    # Parse PCA value from CLI string
    esm_mut_pca_cli = parse_pca_value(args.esm_mut_pca) if args.esm_mut_pca is not None else None
    esm_mut_pca_cfg = parse_pca_value(resolve_param(preproc["esm_mut"]["pca"]))

    params = {
        # Model
        "hidden_dims":             resolve_param(bnn1["hidden_dims"], hidden_dims_cli),
        "prior_std":               resolve_param(bnn1["prior_std"], args.prior_std),
        "dropout_rate":            resolve_param(bnn1["dropout_rate"], args.dropout_rate),
        "activation":              resolve_param(bnn1["activation"]),
        # Training
        "learning_rate":           resolve_param(train["learning_rate"], args.learning_rate),
        "kl_weight":               resolve_param(train["kl_weight"], args.kl_weight),
        "batch_size":              resolve_param(train["batch_size"]),
        "kl_anneal_epochs":        resolve_param(train["kl_anneal_epochs"]),
        "n_epochs":                resolve_param(train["n_epochs"]),
        "early_stopping_patience": resolve_param(train["early_stopping_patience"]),
        "n_inference_samples":     resolve_param(train["n_inference_samples"]),
        # Preprocessing (mutant features only for position classification)
        "esm_mut_scaler":         resolve_param(preproc["esm_mut"]["scaler"], args.esm_mut_scaler),
        "esm_mut_pca":            esm_mut_pca_cli if args.esm_mut_pca is not None else esm_mut_pca_cfg,
    }
    return params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def setup_logging(results_dir: Path) -> None:
    """Configure logging to both console and file."""
    log_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(log_fmt)

    # File handler (write to results directory)
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

    # 1. Config, device, params
    # Set up results dir early so logging can write there
    results_dir = PROJECT_ROOT / "results" / "02_position_classification"
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir = results_dir / "models"

    setup_logging(results_dir)

    logger.info("=" * 60)
    logger.info("02_bnn1_position_classification.py")
    logger.info("BNN1 Phase A: Position Classification Sanity Check")
    logger.info("=" * 60)

    config = load_config(args.config)
    device = get_device(config, args.device)
    params = resolve_all_params(args, config)

    logger.info("Hyperparameters:")
    for k, v in params.items():
        logger.info("  %s: %s", k, v)

    logger.info("Results directory: %s", results_dir)

    # 2. Load data and build features (mutant-only for position classification)
    processed_dir = PROJECT_ROOT / config["data"]["output_dir"]
    X_mut, y, position_labels, mutation_strings, n_classes = \
        load_data_and_features(config, processed_dir)

    # 3. K-fold CV evaluation
    logger.info("Running %d-fold stratified CV...", config["cv"]["n_folds"])
    metrics, y_true, y_pred, fold_histories = evaluate_with_cv(
        X_mut, y, params, config, device, n_classes,
    )

    # 4. Train final model on all data
    if not args.skip_final_model:
        logger.info("Training final model on all %d samples...", len(X_mut))
        train_final_model(X_mut, y, params, n_classes, device, models_dir)

    # 5. Plots
    logger.info("Generating plots...")
    plot_confusion_matrix(
        y_true, y_pred, n_classes, position_labels,
        results_dir / "confusion_matrix.png",
    )
    plot_per_position_accuracy(
        metrics["per_class_accuracy"], position_labels,
        metrics["random_baseline"],
        results_dir / "per_position_accuracy.png",
    )
    plot_training_curves(
        fold_histories,
        results_dir / "training_curves.png",
    )
    plot_loss_decomposition(
        fold_histories,
        results_dir / "loss_decomposition.png",
    )

    # 6. Save results
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics.json")

    with open(results_dir / "hyperparams.json", "w") as f:
        json.dump(params, f, indent=2)
    logger.info("Saved hyperparams.json")

    with open(results_dir / "config_used.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info("Saved config_used.yaml")

    # 7. Summary
    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("Position Classification Complete (%.1fs)", elapsed)
    logger.info("=" * 60)
    logger.info("Overall accuracy:        %.4f (random baseline: %.4f)",
                metrics["overall_accuracy"], metrics["random_baseline"])
    logger.info("Top-5 accuracy:          %.4f", metrics["top5_accuracy"])
    logger.info("Mean per-class accuracy: %.4f", metrics["mean_per_class_accuracy"])
    logger.info("Fold accuracies: %s",
                " ".join(f"{a:.3f}" for a in metrics["fold_accuracies"]))
    if not args.skip_final_model:
        logger.info("Final model saved to:    %s/", models_dir)


if __name__ == "__main__":
    main()
