#!/usr/bin/env python
"""
03_bnn1_formaldehyde_regression.py — BNN1 Phase B: Formaldehyde Regression
==========================================================================

Predict log10(fold_change + epsilon) for formaldehyde SSM mutations using
ESM2 residue embeddings from both WT and mutant sequences. The BNN outputs
a predictive mean and heteroscedastic variance, giving full uncertainty
decomposition: epistemic (weight uncertainty) + aleatoric (data noise).

Features: [ESM2_wt_residue, ESM2_mut_residue] at each mutation position.
Both are preprocessed independently (optional scaler → optional PCA), then
concatenated before feeding to the BNN. Unlike position classification (02),
WT embeddings are included here because the regression target (fold-change)
varies across mutations at the same position — no memorization shortcut.

Pipeline:
  1. Load data + both ESM2 embedding sets
  2. K-fold CV → evaluate regression metrics + collect uncertainty
  3. Train final model on ALL data → save model + both preprocessing pipelines

All hyperparameters are CLI args with defaults read from config.yaml.
Use opt_03_formaldehyde_regression.py for automated hyperopt.

Outputs to results/03_formaldehyde_regression/:
  - metrics.json, hyperparams.json, config_used.yaml, predictions.csv
  - parity_plot.png, residuals_plot.png, calibration_curve.png
  - per_position_spearman.png, per_position_mae.png
  - uncertainty_vs_error.png, uncertainty_decomposition.png
  - training_curves.png, loss_decomposition.png
  - models/final_model.pt, models/preprocessing_wt.joblib,
    models/preprocessing_mut.joblib

Usage:
    python 03_bnn1_formaldehyde_regression.py
    python 03_bnn1_formaldehyde_regression.py --hidden-dims '[128, 64]' --prior-std 0.5
    python 03_bnn1_formaldehyde_regression.py --esm-wt-scaler standard --esm-mut-pca 0.95
    python 03_bnn1_formaldehyde_regression.py --device cuda:1
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
from scipy import stats
from scipy.ndimage import convolve1d

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label Distribution Smoothing (LDS) — Yang et al. 2021
# ---------------------------------------------------------------------------

class LDSAttenuatedWeights:
    """Compute per-sample loss weights via Label Distribution Smoothing.

    Fits a smoothed empirical density of training labels, then returns
    inverse-density weights (normalized to mean=1) so that rare label
    values get high weight and over-represented values get low weight.

    Args:
        n_bins:      Number of histogram bins for density estimation.
        kernel_size: Width of the Gaussian smoothing kernel.
        sigma:       Std of the Gaussian kernel (in bin units).
    """

    def __init__(self, n_bins: int = 50, kernel_size: int = 5, sigma: float = 2.0):
        self.n_bins = n_bins
        self.kernel_size = kernel_size
        self.sigma = sigma

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """Compute normalized inverse-density weights for label array y."""
        y = np.asarray(y, dtype=np.float32)
        if y.max() == y.min():
            return np.ones(len(y), dtype=np.float32)

        # Empirical histogram
        counts, edges = np.histogram(y, bins=self.n_bins)
        counts = counts.astype(np.float32)

        # Gaussian kernel
        half = self.kernel_size // 2
        kernel = np.exp(
            -0.5 * (np.arange(self.kernel_size) - half) ** 2 / (self.sigma ** 2)
        )
        kernel /= kernel.sum()

        # Smooth density
        smoothed = convolve1d(counts, kernel, mode="reflect")
        smoothed = np.maximum(smoothed, 1e-8)

        # Map each sample to its bin (0-indexed into counts)
        bin_indices = np.digitize(y, edges[1:-1])  # 0 … n_bins-1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

        # Inverse density, normalized to mean=1
        raw_weights = 1.0 / smoothed[bin_indices]
        weights = raw_weights / raw_weights.mean()
        return weights.astype(np.float32)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent        # code/round3/
PROJECT_ROOT = SCRIPT_DIR.parent.parent             # MDA_HACL/

# Add code/ to sys.path for BNN module import
sys.path.insert(0, str(SCRIPT_DIR.parent))          # MDA_HACL/code/
from bnns import BayesianMLP, BNNTrainer, TrainingConfig, TrainingHistory


# ---------------------------------------------------------------------------
# Config helpers (same as 02)
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[str]]:
    """Load formaldehyde SSM data with BOTH WT and mutant ESM2 embeddings.

    For each mutation at 0-indexed position p:
      - X_wt[i]  = esm2_wt_residues[p]   (1280-dim WT embedding at that position)
      - X_mut[i] = esm2_mutant_residues[mutation_string]  (1280-dim mutant embedding)
      - y[i]     = log_fc  (log10(fold_change + epsilon))

    Returns:
        X_wt: (n_mutations, 1280) WT residue embeddings
        X_mut: (n_mutations, 1280) mutant residue embeddings
        y: (n_mutations,) log_fc targets (float32)
        positions: 0-indexed position per mutation
        mutation_strings: mutation string per row
    """
    form_path = processed_dir / "formaldehyde_ssm.csv"
    form_df = pd.read_csv(form_path)
    logger.info("Loaded formaldehyde SSM: %d mutations at %d positions",
                len(form_df), form_df["position"].nunique())

    emb_dir = processed_dir / "embeddings"
    esm_wt = np.load(emb_dir / "esm2_wt_residues.npy")       # (565, 1280)
    esm_mutant = np.load(emb_dir / "esm2_mutant_residues.npz")
    logger.info("ESM2 WT residues: %s", esm_wt.shape)
    logger.info("ESM2 mutant embeddings: %d entries", len(esm_mutant.files))

    wt_list = []
    mut_list = []
    y_list = []
    positions = []
    mutation_strings = []

    for _, row in form_df.iterrows():
        ms = row["mutation_string"]
        pos = row["position"]

        if ms not in esm_mutant:
            warnings.warn(f"Missing ESM2 mutant embedding for {ms}, skipping")
            continue

        wt_list.append(esm_wt[pos])           # (1280,)
        mut_list.append(esm_mutant[ms])        # (1280,)
        y_list.append(row["log_fc"])
        positions.append(pos)
        mutation_strings.append(ms)

    X_wt = np.stack(wt_list).astype(np.float32)
    X_mut = np.stack(mut_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)

    logger.info("WT features:     %s", X_wt.shape)
    logger.info("Mutant features: %s", X_mut.shape)
    logger.info("Targets: n=%d, min=%.3f, max=%.3f, mean=%.3f, std=%.3f",
                len(y), y.min(), y.max(), y.mean(), y.std())

    return X_wt, X_mut, y, positions, mutation_strings


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


def preprocess_and_concat(
    X_wt_train: np.ndarray, X_wt_val: np.ndarray,
    X_mut_train: np.ndarray, X_mut_val: np.ndarray,
    params: dict,
) -> Tuple[np.ndarray, np.ndarray, Optional[object], Optional[object]]:
    """Build two independent preprocessing pipelines, transform, concatenate.

    Returns:
        X_train_concat: (n_train, d_wt + d_mut)
        X_val_concat:   (n_val, d_wt + d_mut)
        pipe_wt:  fitted Pipeline for WT features (or None)
        pipe_mut: fitted Pipeline for mutant features (or None)
    """
    pipe_wt = build_preprocessing(params["esm_wt_scaler"], params["esm_wt_pca"])
    pipe_mut = build_preprocessing(params["esm_mut_scaler"], params["esm_mut_pca"])

    X_wt_train_t, X_wt_val_t = apply_preprocessing(pipe_wt, X_wt_train, X_wt_val)
    X_mut_train_t, X_mut_val_t = apply_preprocessing(pipe_mut, X_mut_train, X_mut_val)

    X_train = np.concatenate([X_wt_train_t, X_mut_train_t], axis=1)
    X_val = np.concatenate([X_wt_val_t, X_mut_val_t], axis=1)

    logger.debug("Preprocessing: WT %d + Mut %d = %d features",
                 X_wt_train_t.shape[1], X_mut_train_t.shape[1], X_train.shape[1])

    return X_train, X_val, pipe_wt, pipe_mut


# ---------------------------------------------------------------------------
# Single fold training + evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
    device: str,
    return_predictions: bool = False,
    w_train: Optional[torch.Tensor] = None,
) -> Tuple[dict, Optional[object], Optional[TrainingHistory]]:
    """Train BNN regression on one fold, evaluate on val set.

    Returns:
        fold_metrics: dict with mae, rmse, r2, spearman_rho, val_loss
        estimates: UncertaintyEstimate (if return_predictions, else None)
        history: TrainingHistory (if return_predictions, else None)
    """
    model = BayesianMLP(
        input_dim=X_train.shape[1],
        hidden_dims=params["hidden_dims"],
        output_dim=1,
        task="regression",
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
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)  # (n, 1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)      # (n, 1)

    history = trainer.fit(X_train_t, y_train_t, X_val_t, y_val_t, w_train=w_train)

    estimates = trainer.predict(X_val_t)
    y_pred = estimates.mean.cpu().numpy().squeeze(-1)         # (n_val,)
    total_std_np = estimates.total_std.cpu().numpy().squeeze(-1)  # (n_val,)

    # Regression metrics
    residuals = y_val - y_pred
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_val - y_val.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    spearman_rho, spearman_p = stats.spearmanr(y_val, y_pred)

    # Proper scoring rules (jointly evaluate accuracy + calibration)
    nlpd = compute_nlpd(y_val, y_pred, total_std_np)
    crps = compute_crps_gaussian(y_val, y_pred, total_std_np)

    # Null model baseline: always predict training set mean
    null_pred = float(y_train.mean())
    null_residuals = y_val - null_pred
    null_mae = float(np.mean(np.abs(null_residuals)))
    null_rmse = float(np.sqrt(np.mean(null_residuals ** 2)))

    # Get last validation loss from history
    val_loss = history.val_loss[-1] if history.val_loss else float("nan")

    # Posterior collapse diagnostic: ratio of mean posterior std to prior_std.
    # With rho_init = softplus_inverse(prior_std), initialisation starts at 1.0.
    # After training a healthy model should tighten (< 1.0); near 1.0 = no update.
    summary = model.posterior_summary()
    posterior_stds = [v["mean_std"] for v in summary.values()]
    prior_std = params.get("prior_std", 1.0)
    collapse_score = float(np.mean(posterior_stds)) / max(prior_std, 1e-8)

    fold_metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": float(r2),
        "spearman_rho": float(spearman_rho),
        "spearman_pvalue": float(spearman_p),
        "nlpd": nlpd,
        "crps": crps,
        "val_loss": float(val_loss),
        "null_mae": null_mae,
        "null_rmse": null_rmse,
        "posterior_collapse_score": collapse_score,
    }

    if return_predictions:
        return fold_metrics, estimates, history
    return fold_metrics, None, None


# ---------------------------------------------------------------------------
# Calibration and per-position metrics
# ---------------------------------------------------------------------------

def compute_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    total_std: np.ndarray,
    levels: Optional[List[float]] = None,
) -> dict:
    """Compute calibration: does a q% CI contain q% of observations?

    For each confidence level q, computes the interval [pred - z*std, pred + z*std]
    where z = norm.ppf((1+q)/2), and checks what fraction of true values fall within.
    """
    if levels is None:
        levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    observed_coverage = []
    for q in levels:
        z = stats.norm.ppf((1 + q) / 2)
        lower = y_pred - z * total_std
        upper = y_pred + z * total_std
        in_interval = ((y_true >= lower) & (y_true <= upper))
        observed_coverage.append(float(np.mean(in_interval)))

    return {
        "levels": levels,
        "expected_coverage": levels,
        "observed_coverage": observed_coverage,
    }


def compute_per_position_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positions: List[int],
    min_count: int = 5,
) -> Tuple[dict, dict, dict]:
    """Compute within-position Spearman, MAE, and activity range (std).

    The activity range (std of true values) measures how much a position
    can modulate activity. Positions with near-zero range have all mutations
    yielding ~WT activity, so ranking performance there is uninformative.

    Returns:
        per_pos_spearman: {position: rho}
        per_pos_mae: {position: mae}
        per_pos_range: {position: std_of_y_true}
    """
    positions_arr = np.array(positions)
    unique_positions = sorted(set(positions))

    per_pos_spearman = {}
    per_pos_mae = {}
    per_pos_range = {}

    for pos in unique_positions:
        mask = positions_arr == pos
        n = mask.sum()
        if n < min_count:
            continue

        yt = y_true[mask]
        yp = y_pred[mask]

        per_pos_mae[pos] = float(np.mean(np.abs(yt - yp)))
        per_pos_range[pos] = float(np.std(yt))

        # Spearman requires variation in both arrays
        if np.std(yt) > 1e-8 and np.std(yp) > 1e-8:
            rho, _ = stats.spearmanr(yt, yp)
            per_pos_spearman[pos] = float(rho)

    return per_pos_spearman, per_pos_mae, per_pos_range


def compute_nlpd(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    total_std: np.ndarray,
) -> float:
    """Negative Log Predictive Density under Gaussian approximation.

    A strictly proper scoring rule that jointly penalizes poor accuracy
    and miscalibrated uncertainty. For a predictive N(mu, sigma^2):

        NLPD = 0.5 * [log(2*pi) + log(sigma^2) + (y - mu)^2 / sigma^2]

    - Overconfident + wrong: sigma small, error large → (y-mu)^2/sigma^2 explodes
    - Underconfident + right: sigma large, error small → log(sigma^2) penalty
    - Well-calibrated: NLPD is minimized when sigma matches the true noise

    Lower is better.
    """
    variance = np.clip(total_std ** 2, 1e-10, None)
    nlpd = 0.5 * (np.log(2 * np.pi) + np.log(variance) + (y_true - y_pred) ** 2 / variance)
    return float(np.mean(nlpd))


def compute_crps_gaussian(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    total_std: np.ndarray,
) -> float:
    """Continuous Ranked Probability Score for Gaussian predictive distributions.

    Closed-form CRPS for N(mu, sigma^2):
        CRPS = sigma * [z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi)]
    where z = (y - mu) / sigma, Phi = CDF, phi = PDF.

    A strictly proper scoring rule in the same units as the target.
    More robust to outliers than NLPD. Lower is better.
    """
    sigma = np.clip(total_std, 1e-10, None)
    z = (y_true - y_pred) / sigma
    crps = sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    return float(np.mean(crps))


def compute_directional_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    total_std: np.ndarray,
    wt_activity: float,
) -> dict:
    """Uncertainty-aware directional classification: above vs below WT.

    Each mutation is classified as beneficial (above WT) or detrimental (below WT).
    The parity plot has four quadrants divided by WT activity lines:

        TL (false beneficial):  true < WT, pred > WT  — predicted beneficial, actually not
        TR (correct beneficial): true > WT, pred > WT  — correctly identified as beneficial
        BL (correct detrimental): true < WT, pred < WT — correctly identified as detrimental
        BR (false detrimental):  true > WT, pred < WT  — predicted detrimental, actually beneficial

    Uncertainty-adjusted directional accuracy:
        For each mutation, P(prediction on correct side of WT) from N(pred, std²).
        Confident correct → ~1.0, confident wrong → ~0.0, uncertain → ~0.5.
        This forgives misclassifications where the model is genuinely uncertain.
    """
    n = len(y_true)
    true_beneficial = y_true > wt_activity
    pred_beneficial = y_pred > wt_activity

    # Quadrant counts
    tr = int(np.sum(true_beneficial & pred_beneficial))
    bl = int(np.sum(~true_beneficial & ~pred_beneficial))
    tl = int(np.sum(~true_beneficial & pred_beneficial))
    br = int(np.sum(true_beneficial & ~pred_beneficial))

    quadrant_accuracy = (tr + bl) / n

    # Balanced accuracy: average of sensitivity and specificity
    # Sensitivity = TR / (TR + BR)  — how well we detect truly beneficial
    # Specificity = BL / (BL + TL)  — how well we detect truly detrimental
    n_truly_beneficial = tr + br
    n_truly_detrimental = bl + tl
    sensitivity = tr / n_truly_beneficial if n_truly_beneficial > 0 else float("nan")
    specificity = bl / n_truly_detrimental if n_truly_detrimental > 0 else float("nan")
    balanced_quadrant_accuracy = float(np.nanmean([sensitivity, specificity]))

    # P(correct side of WT) for each mutation, using Gaussian CDF
    z = (wt_activity - y_pred) / np.clip(total_std, 1e-8, None)
    p_below_wt = stats.norm.cdf(z)
    p_above_wt = 1.0 - p_below_wt
    p_correct = np.where(true_beneficial, p_above_wt, p_below_wt)
    uncertainty_adjusted_accuracy = float(np.mean(p_correct))

    # Balanced uncertainty-adjusted accuracy: average p_correct per class
    p_correct_beneficial = float(np.mean(p_correct[true_beneficial])) if n_truly_beneficial > 0 else float("nan")
    p_correct_detrimental = float(np.mean(p_correct[~true_beneficial])) if n_truly_detrimental > 0 else float("nan")
    balanced_uncertainty_adjusted_accuracy = float(np.nanmean([p_correct_beneficial, p_correct_detrimental]))

    # Confident misclassifications: wrong quadrant AND ±1σ CI doesn't cross WT
    wrong_quadrant = true_beneficial != pred_beneficial
    ci_lower = y_pred - total_std
    ci_upper = y_pred + total_std
    ci_crosses_wt = (ci_lower < wt_activity) & (ci_upper > wt_activity)
    confident_wrong = wrong_quadrant & ~ci_crosses_wt
    uncertain_wrong = wrong_quadrant & ci_crosses_wt

    return {
        "wt_activity": float(wt_activity),
        "quadrant_accuracy": float(quadrant_accuracy),
        "balanced_quadrant_accuracy": balanced_quadrant_accuracy,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "uncertainty_adjusted_accuracy": uncertainty_adjusted_accuracy,
        "balanced_uncertainty_adjusted_accuracy": balanced_uncertainty_adjusted_accuracy,
        "confident_misclassification_rate": float(np.sum(confident_wrong)) / n,
        "quadrant_counts": {
            "correct_beneficial_TR": tr,
            "correct_detrimental_BL": bl,
            "false_beneficial_TL": tl,
            "false_detrimental_BR": br,
        },
        "quadrant_pct": {
            "correct_beneficial_TR": float(tr / n * 100),
            "correct_detrimental_BL": float(bl / n * 100),
            "false_beneficial_TL": float(tl / n * 100),
            "false_detrimental_BR": float(br / n * 100),
        },
        "n_wrong_quadrant": int(np.sum(wrong_quadrant)),
        "n_confident_wrong": int(np.sum(confident_wrong)),
        "n_uncertain_wrong": int(np.sum(uncertain_wrong)),
    }


def _range_weighted_mean(per_pos_metric: dict, per_pos_range: dict) -> float:
    """Compute range-weighted mean of a per-position metric.

    Weights each position's metric by its activity range (std of y_true).
    Positions where mutations barely change activity contribute less.
    """
    common = set(per_pos_metric) & set(per_pos_range)
    if not common:
        return 0.0
    values = np.array([per_pos_metric[p] for p in common])
    weights = np.array([per_pos_range[p] for p in common])
    total_w = weights.sum()
    if total_w < 1e-10:
        return float(np.mean(values))
    return float(np.average(values, weights=weights))


# ---------------------------------------------------------------------------
# K-fold CV evaluation
# ---------------------------------------------------------------------------

def evaluate_with_cv(
    X_wt: np.ndarray,
    X_mut: np.ndarray,
    y: np.ndarray,
    positions: List[int],
    params: dict,
    config: dict,
    device: str,
    use_lds: bool = False,
) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List, List]:
    """Run K-fold CV, collecting out-of-fold predictions + uncertainty.

    Returns:
        metrics: dict of aggregate + per-fold regression metrics
        all_y_true: (n,) true log_fc
        all_y_pred: (n,) predicted mean log_fc
        all_epistemic_std: (n,) epistemic uncertainty per sample
        all_aleatoric_std: (n,) aleatoric uncertainty per sample
        all_total_std: (n,) total uncertainty per sample
        fold_histories: list of TrainingHistory per fold
        lds_traces: list of (y_train, weights, label) tuples (empty if use_lds=False)
    """
    from sklearn.model_selection import KFold

    cv_config = config["cv"]
    kf = KFold(
        n_splits=cv_config["n_folds"],
        shuffle=True,
        random_state=cv_config["seed"],
    )

    n = len(y)
    all_y_true = np.zeros(n, dtype=np.float32)
    all_y_pred = np.zeros(n, dtype=np.float32)
    all_epistemic_std = np.zeros(n, dtype=np.float32)
    all_aleatoric_std = np.zeros(n, dtype=np.float32)
    all_total_std = np.zeros(n, dtype=np.float32)
    fold_metrics_list = []
    fold_histories = []
    lds_traces: List[Tuple[np.ndarray, np.ndarray, str]] = []

    lds_cfg = params.get("lds", {})

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_wt)):
        logger.info("Fold %d/%d", fold_idx + 1, cv_config["n_folds"])

        y_train, y_val = y[train_idx], y[val_idx]

        X_train, X_val, _, _ = preprocess_and_concat(
            X_wt[train_idx], X_wt[val_idx],
            X_mut[train_idx], X_mut[val_idx],
            params,
        )

        # LDS sample weights — fit on training labels only
        w_train_t: Optional[torch.Tensor] = None
        if use_lds:
            lds = LDSAttenuatedWeights(
                n_bins=lds_cfg.get("n_bins", 50),
                kernel_size=lds_cfg.get("kernel_size", 5),
                sigma=lds_cfg.get("sigma", 2.0),
            )
            fold_weights = lds.fit_transform(y_train)
            w_train_t = torch.tensor(fold_weights, dtype=torch.float32)
            lds_traces.append((y_train.copy(), fold_weights.copy(), f"Fold {fold_idx + 1}"))

        fold_metrics, estimates, history = train_and_evaluate_fold(
            X_train, y_train, X_val, y_val,
            params, device,
            return_predictions=True,
            w_train=w_train_t,
        )

        # Collect out-of-fold predictions and uncertainty
        all_y_true[val_idx] = y_val
        all_y_pred[val_idx] = estimates.mean.cpu().numpy().squeeze(-1)
        all_epistemic_std[val_idx] = estimates.epistemic_std.cpu().numpy().squeeze(-1)
        all_aleatoric_std[val_idx] = estimates.aleatoric_std.cpu().numpy().squeeze(-1)
        all_total_std[val_idx] = estimates.total_std.cpu().numpy().squeeze(-1)

        fold_metrics_list.append(fold_metrics)
        fold_histories.append(history)
        logger.info("  MAE=%.4f  R²=%.4f  Spearman=%.4f  NLPD=%.4f  (null MAE=%.4f)",
                     fold_metrics["mae"], fold_metrics["r2"],
                     fold_metrics["spearman_rho"], fold_metrics["nlpd"],
                     fold_metrics["null_mae"])

    # --- Aggregate metrics ---
    residuals = all_y_true - all_y_pred
    overall_mae = float(np.mean(np.abs(residuals)))
    overall_rmse = float(np.sqrt(np.mean(residuals ** 2)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((all_y_true - all_y_true.mean()) ** 2))
    overall_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    overall_spearman, overall_spearman_p = stats.spearmanr(all_y_true, all_y_pred)

    # Per-position metrics
    per_pos_spearman, per_pos_mae, per_pos_range = compute_per_position_metrics(
        all_y_true, all_y_pred, positions,
    )

    # Calibration
    calibration = compute_calibration(all_y_true, all_y_pred, all_total_std)

    # Sharpness = mean total uncertainty
    sharpness = float(np.mean(all_total_std))

    # Proper scoring rules on pooled OOF predictions
    overall_nlpd = compute_nlpd(all_y_true, all_y_pred, all_total_std)
    overall_crps = compute_crps_gaussian(all_y_true, all_y_pred, all_total_std)

    # Null model baseline: average fold null metrics
    null_mae = float(np.mean([f["null_mae"] for f in fold_metrics_list]))
    null_rmse = float(np.mean([f["null_rmse"] for f in fold_metrics_list]))

    metrics = {
        "mae": overall_mae,
        "rmse": overall_rmse,
        "r2": float(overall_r2),
        "spearman_rho": float(overall_spearman),
        "spearman_pvalue": float(overall_spearman_p),
        "nlpd": overall_nlpd,
        "crps": overall_crps,
        "null_mae": null_mae,
        "null_rmse": null_rmse,
        "mae_improvement_over_null": float(1.0 - overall_mae / null_mae) if null_mae > 0 else 0.0,
        "mean_per_position_spearman": float(np.mean(list(per_pos_spearman.values()))) if per_pos_spearman else 0.0,
        "median_per_position_spearman": float(np.median(list(per_pos_spearman.values()))) if per_pos_spearman else 0.0,
        "weighted_per_position_spearman": _range_weighted_mean(per_pos_spearman, per_pos_range),
        "n_positions_evaluated": len(per_pos_spearman),
        "per_position_spearman": {str(k): v for k, v in per_pos_spearman.items()},
        "per_position_mae": {str(k): v for k, v in per_pos_mae.items()},
        "per_position_range": {str(k): v for k, v in per_pos_range.items()},
        "calibration": calibration,
        "sharpness": sharpness,
        "mean_epistemic_std": float(np.mean(all_epistemic_std)),
        "mean_aleatoric_std": float(np.mean(all_aleatoric_std)),
        "mean_total_std": float(np.mean(all_total_std)),
        "fold_metrics": fold_metrics_list,
        "mean_fold_spearman": float(np.mean([f["spearman_rho"] for f in fold_metrics_list])),
        "std_fold_spearman": float(np.std([f["spearman_rho"] for f in fold_metrics_list])),
        "mean_fold_mae": float(np.mean([f["mae"] for f in fold_metrics_list])),
        "mean_fold_nlpd": float(np.mean([f["nlpd"] for f in fold_metrics_list])),
        "mean_fold_crps": float(np.mean([f["crps"] for f in fold_metrics_list])),
        "n_samples": n,
        "n_positions": len(set(positions)),
        "target_range": {
            "min": float(y.min()),
            "max": float(y.max()),
            "mean": float(y.mean()),
            "std": float(y.std()),
        },
    }

    logger.info("--- Model ---")
    logger.info("MAE:              %.4f", overall_mae)
    logger.info("RMSE:             %.4f", overall_rmse)
    logger.info("R²:               %.4f", overall_r2)
    logger.info("Spearman rho:     %.4f (p=%.2e)", overall_spearman, overall_spearman_p)
    logger.info("NLPD:             %.4f", overall_nlpd)
    logger.info("CRPS:             %.4f", overall_crps)
    logger.info("Mean pos Spearman: %.4f  (range-weighted: %.4f, %d positions)",
                metrics["mean_per_position_spearman"],
                metrics["weighted_per_position_spearman"],
                len(per_pos_spearman))
    logger.info("Sharpness (mean σ): %.4f", sharpness)
    logger.info("--- Null model (predict train mean) ---")
    logger.info("Null MAE:         %.4f", null_mae)
    logger.info("Null RMSE:        %.4f", null_rmse)
    logger.info("MAE improvement:  %.1f%%", metrics["mae_improvement_over_null"] * 100)

    # Full-dataset LDS trace (added after fold loop)
    if use_lds:
        lds_all = LDSAttenuatedWeights(
            n_bins=lds_cfg.get("n_bins", 50),
            kernel_size=lds_cfg.get("kernel_size", 5),
            sigma=lds_cfg.get("sigma", 2.0),
        )
        w_all = lds_all.fit_transform(y)
        lds_traces.append((y.copy(), w_all.copy(), "All data"))

    return (metrics, all_y_true, all_y_pred,
            all_epistemic_std, all_aleatoric_std, all_total_std,
            fold_histories, lds_traces)


# ---------------------------------------------------------------------------
# Train final model on all data
# ---------------------------------------------------------------------------

def train_final_model(
    X_wt: np.ndarray,
    X_mut: np.ndarray,
    y: np.ndarray,
    params: dict,
    device: str,
    models_dir: Path,
    fold_histories: Optional[List] = None,
):
    """Fit both preprocessing pipelines on all data, train final BNN, save everything.

    The number of training epochs is set to the median best_epoch across CV
    folds (with a small buffer), so the final model trains for roughly as
    long as the model actually needs rather than the full n_epochs budget.

    Saves:
        models_dir/preprocessing_wt.joblib   — WT pipeline
        models_dir/preprocessing_mut.joblib  — mutant pipeline
        models_dir/final_model.pt            — BNN checkpoint
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    # Determine n_epochs from CV fold histories
    if fold_histories:
        best_epochs = [h.best_epoch for h in fold_histories]
        # Use median + 10% buffer, clamped to configured max
        median_best = int(np.median(best_epochs))
        n_epochs = min(int(median_best * 1.1) + 1, params["n_epochs"])
        # Ensure at least kl_anneal_epochs so annealing completes
        n_epochs = max(n_epochs, params["kl_anneal_epochs"] + 1)
        logger.info("CV best epochs: %s (median=%d) → final model n_epochs=%d",
                    best_epochs, median_best, n_epochs)
    else:
        n_epochs = params["n_epochs"]

    # Fit preprocessing on full dataset
    pipe_wt = build_preprocessing(params["esm_wt_scaler"], params["esm_wt_pca"])
    pipe_mut = build_preprocessing(params["esm_mut_scaler"], params["esm_mut_pca"])

    X_wt_p = pipe_wt.fit_transform(X_wt).astype(np.float32) if pipe_wt else X_wt.copy()
    X_mut_p = pipe_mut.fit_transform(X_mut).astype(np.float32) if pipe_mut else X_mut.copy()
    X_all = np.concatenate([X_wt_p, X_mut_p], axis=1)

    joblib.dump(pipe_wt, models_dir / "preprocessing_wt.joblib")
    joblib.dump(pipe_mut, models_dir / "preprocessing_mut.joblib")
    logger.info("Saved preprocessing pipelines -> preprocessing_wt.joblib, preprocessing_mut.joblib")

    # Build and train BNN
    model = BayesianMLP(
        input_dim=X_all.shape[1],
        hidden_dims=params["hidden_dims"],
        output_dim=1,
        task="regression",
        prior_std=params["prior_std"],
        dropout_rate=params["dropout_rate"],
        activation=params.get("activation", "silu"),
    )

    training_config = TrainingConfig(
        n_epochs=n_epochs,
        learning_rate=params["learning_rate"],
        batch_size=params["batch_size"],
        kl_anneal_epochs=params["kl_anneal_epochs"],
        kl_weight=params["kl_weight"],
        early_stopping_patience=0,  # no early stopping — no val set
        n_inference_samples=params["n_inference_samples"],
        device=device,
        verbose=True,
        log_interval=50,
    )

    trainer = BNNTrainer(model, training_config)

    X_t = torch.tensor(X_all, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)  # (n, 1)

    # Train on all data (no val set — CV already evaluated generalization)
    trainer.fit(X_t, y_t)

    model_path = str(models_dir / "final_model.pt")
    trainer.save(model_path)
    logger.info("Saved final model -> final_model.pt (%d epochs)", n_epochs)
    logger.info("Input dim: %d (WT %d + Mut %d), architecture: %s, output dim: 1",
                X_all.shape[1], X_wt_p.shape[1], X_mut_p.shape[1], params["hidden_dims"])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    total_std: np.ndarray,
    positions: List[int],
    wt_activity: float,
    directional_metrics: dict,
    output_path: Path,
):
    """Predicted vs actual log_fc with WT reference lines, quadrant %, and error bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))

    # Color by position (cyclic)
    unique_pos = sorted(set(positions))
    pos_to_idx = {p: i for i, p in enumerate(unique_pos)}
    colors = [pos_to_idx[p] for p in positions]

    sc = ax.scatter(
        y_true, y_pred, c=colors, cmap="tab20", s=10, alpha=0.6,
        edgecolors="none", rasterized=True,
    )
    ax.errorbar(
        y_true, y_pred, yerr=total_std,
        fmt="none", ecolor="gray", alpha=0.1, linewidth=0.5,
    )

    # Diagonal
    lims = [min(y_true.min(), y_pred.min()) - 0.2,
            max(y_true.max(), y_pred.max()) + 0.2]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="y = x")

    # WT activity reference lines (solid)
    ax.axhline(wt_activity, color="black", linewidth=1.2, linestyle="-", alpha=0.6,
               label=f"WT ({wt_activity:.3f})")
    ax.axvline(wt_activity, color="black", linewidth=1.2, linestyle="-", alpha=0.6)

    # Quadrant percentage annotations
    pct = directional_metrics["quadrant_pct"]
    mid_right_x = (wt_activity + lims[1]) / 2
    mid_left_x = (lims[0] + wt_activity) / 2
    mid_top_y = (wt_activity + lims[1]) / 2
    mid_bottom_y = (lims[0] + wt_activity) / 2

    quad_font = dict(fontsize=14, fontweight="bold", ha="center", va="center")
    # Correct quadrants (green) — want these to be large
    ax.text(mid_right_x, mid_top_y,
            f'{pct["correct_beneficial_TR"]:.1f}%',
            color="#4CAF50", alpha=0.7, **quad_font)
    ax.text(mid_left_x, mid_bottom_y,
            f'{pct["correct_detrimental_BL"]:.1f}%',
            color="#4CAF50", alpha=0.7, **quad_font)
    # Wrong quadrants (red)
    ax.text(mid_left_x, mid_top_y,
            f'{pct["false_beneficial_TL"]:.1f}%',
            color="#F44336", alpha=0.7, **quad_font)
    ax.text(mid_right_x, mid_bottom_y,
            f'{pct["false_detrimental_BR"]:.1f}%',
            color="#F44336", alpha=0.7, **quad_font)

    # Metrics annotation
    residuals = y_true - y_pred
    r2 = 1 - np.sum(residuals**2) / np.sum((y_true - y_true.mean())**2)
    rho, _ = stats.spearmanr(y_true, y_pred)
    mae = np.mean(np.abs(residuals))

    ax.text(
        0.05, 0.95,
        f"R² = {r2:.3f}\n"
        f"Spearman ρ = {rho:.3f}\n"
        f"MAE = {mae:.3f}\n"
        f"Bal. quad acc = {directional_metrics['balanced_quadrant_accuracy']:.1%}\n"
        f"Bal. uncert-adj = {directional_metrics['balanced_uncertainty_adjusted_accuracy']:.1%}\n"
        f"Sens = {directional_metrics['sensitivity']:.1%}  "
        f"Spec = {directional_metrics['specificity']:.1%}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("True log₁₀(FC + ε)")
    ax.set_ylabel("Predicted log₁₀(FC + ε)")
    ax.set_title("Parity Plot — Formaldehyde Regression (OOF)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
):
    """Residuals (true - pred) vs predicted value."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Residuals vs predicted
    colors = ["#2196F3" if r >= 0 else "#F44336" for r in residuals]
    axes[0].scatter(y_pred, residuals, c=colors, s=8, alpha=0.5, edgecolors="none")
    axes[0].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[0].set_xlabel("Predicted log₁₀(FC + ε)")
    axes[0].set_ylabel("Residual (true − pred)")
    axes[0].set_title("Residuals vs Predicted")
    axes[0].text(
        0.05, 0.95,
        f"Mean residual: {np.mean(residuals):.4f}\nStd: {np.std(residuals):.4f}",
        transform=axes[0].transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Panel 2: Residual histogram
    axes[1].hist(residuals, bins=50, color="#2196F3", edgecolor="none", alpha=0.7)
    axes[1].axvline(0, color="black", linewidth=1, linestyle="--")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_calibration(
    calibration_dict: dict,
    output_path: Path,
):
    """Expected vs observed coverage curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    expected = calibration_dict["expected_coverage"]
    observed = calibration_dict["observed_coverage"]

    fig, ax = plt.subplots(figsize=(7, 7))

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect calibration")

    # Calibration curve
    ax.plot(expected, observed, "o-", color="#2196F3", linewidth=2, markersize=6,
            label="Model calibration")

    # Shade miscalibration
    ax.fill_between(expected, expected, observed, alpha=0.15, color="#2196F3")

    ax.set_xlabel("Expected coverage")
    ax.set_ylabel("Observed coverage")
    ax.set_title("Uncertainty Calibration Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_per_position_spearman(
    per_pos_spearman: dict,
    position_labels_1idx: List[int],
    position_offset: int,
    output_path: Path,
    per_pos_range: Optional[dict] = None,
    weighted_mean: Optional[float] = None,
):
    """Bar chart of within-position Spearman rho, colored by activity range.

    Bar opacity scales with the position's activity range (std of y_true).
    Faded bars = low dynamic range (uninformative). Saturated = high range (matters).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    # Convert 0-indexed positions to 1-indexed labels
    positions_sorted = sorted(per_pos_spearman.keys(), key=int)
    rhos = [per_pos_spearman[p] for p in positions_sorted]
    labels = [str(int(p) + position_offset + 1) for p in positions_sorted]
    median_rho = float(np.median(rhos))

    # Compute alpha from activity range (min 0.25, max 1.0)
    if per_pos_range:
        ranges = np.array([per_pos_range.get(p, 0.0) for p in positions_sorted])
        max_range = ranges.max() if ranges.max() > 0 else 1.0
        alphas = 0.25 + 0.75 * (ranges / max_range)
    else:
        alphas = np.ones(len(positions_sorted))

    colors = [to_rgba("#2196F3" if r >= 0 else "#F44336", alpha=a)
              for r, a in zip(rhos, alphas)]

    fig, ax = plt.subplots(figsize=(18, 5))
    ax.bar(range(len(positions_sorted)), rhos, color=colors, edgecolor="none", width=0.8)

    ax.axhline(median_rho, color="orange", linestyle="-", linewidth=1.5,
               label=f"Median ρ = {median_rho:.3f}")
    if weighted_mean is not None:
        ax.axhline(weighted_mean, color="green", linestyle="--", linewidth=1.5,
                   label=f"Range-weighted mean ρ = {weighted_mean:.3f}")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)

    ax.set_xticks(range(len(positions_sorted)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_xlabel("Position (1-indexed)")
    ax.set_ylabel("Spearman ρ")
    ax.set_title(f"Per-Position Spearman ρ ({len(positions_sorted)} positions, opacity ∝ activity range)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_per_position_mae(
    per_pos_mae: dict,
    position_offset: int,
    global_mae: float,
    output_path: Path,
):
    """Bar chart of MAE per position."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions_sorted = sorted(per_pos_mae.keys(), key=int)
    maes = [per_pos_mae[p] for p in positions_sorted]
    labels = [str(int(p) + position_offset + 1) for p in positions_sorted]

    fig, ax = plt.subplots(figsize=(18, 5))

    ax.bar(range(len(positions_sorted)), maes, color="#2196F3", edgecolor="none", width=0.8)
    ax.axhline(global_mae, color="orange", linestyle="-", linewidth=1.5,
               label=f"Global MAE = {global_mae:.3f}")

    ax.set_xticks(range(len(positions_sorted)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_xlabel("Position (1-indexed)")
    ax.set_ylabel("MAE (log₁₀ FC)")
    ax.set_title(f"Per-Position MAE ({len(positions_sorted)} positions)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_uncertainty_vs_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    total_std: np.ndarray,
    output_path: Path,
):
    """Scatter of total_std vs |residual|."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    abs_error = np.abs(y_true - y_pred)
    rho, _ = stats.spearmanr(total_std, abs_error)

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(total_std, abs_error, s=8, alpha=0.4, color="#2196F3", edgecolors="none")

    # Reference line: std = |error|
    lim = max(total_std.max(), abs_error.max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.4, label="σ = |error|")

    ax.text(
        0.05, 0.95,
        f"Spearman ρ(σ, |error|) = {rho:.3f}",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Predicted uncertainty (total σ)")
    ax.set_ylabel("|Residual|")
    ax.set_title("Uncertainty vs Prediction Error")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_uncertainty_decomposition(
    epistemic_std: np.ndarray,
    aleatoric_std: np.ndarray,
    output_path: Path,
):
    """Epistemic vs aleatoric uncertainty: scatter + stacked histogram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Scatter
    axes[0].scatter(epistemic_std, aleatoric_std, s=8, alpha=0.4,
                    color="#2196F3", edgecolors="none")
    axes[0].set_xlabel("Epistemic σ (model uncertainty)")
    axes[0].set_ylabel("Aleatoric σ (data noise)")
    axes[0].set_title("Uncertainty Decomposition")

    # Equal axes
    lim = max(epistemic_std.max(), aleatoric_std.max()) * 1.1
    axes[0].set_xlim(0, lim)
    axes[0].set_ylim(0, lim)
    axes[0].plot([0, lim], [0, lim], "k--", linewidth=0.5, alpha=0.3)

    axes[0].text(
        0.05, 0.95,
        f"Mean epistemic: {epistemic_std.mean():.3f}\nMean aleatoric: {aleatoric_std.mean():.3f}",
        transform=axes[0].transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Panel 2: Overlapping histograms
    bins = np.linspace(0, lim, 50)
    axes[1].hist(epistemic_std, bins=bins, alpha=0.6, color="#FF9800",
                 label=f"Epistemic (mean={epistemic_std.mean():.3f})", edgecolor="none")
    axes[1].hist(aleatoric_std, bins=bins, alpha=0.6, color="#4CAF50",
                 label=f"Aleatoric (mean={aleatoric_std.mean():.3f})", edgecolor="none")
    axes[1].set_xlabel("Standard deviation")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Uncertainty Component Distributions")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_training_curves(
    fold_histories: List,
    output_path: Path,
):
    """Plot training and validation loss curves for all CV folds."""
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
    """Plot NLL vs KL decomposition for fold 1 with KL annealing schedule."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = fold_histories[0]  # first fold
    epochs = range(1, len(h.train_loss) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1: NLL
    axes[0].plot(epochs, h.train_nll, color="#2196F3", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("NLL (Gaussian)")
    axes[0].set_title("Negative Log-Likelihood (Fold 1)")

    # Panel 2: KL divergence
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


def plot_lds_weights(
    traces: List[Tuple[np.ndarray, np.ndarray, str]],
    output_path: Path,
) -> None:
    """Plot LDS weight distributions across multiple folds + full dataset.

    Args:
        traces: List of (y_train, weights, label) tuples, one per fold + full dataset.
                Convention: last entry is the full-dataset trace (bold black).
        output_path: Where to save the figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    if not traces:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    n_traces = len(traces)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_traces, 1)))

    all_y = np.concatenate([t[0] for t in traces])
    y_min, y_max = float(all_y.min()), float(all_y.max())
    x_range = np.linspace(y_min - 0.1, y_max + 0.1, 300)

    # ---- Left panel: label histogram + density curves ----
    ax_l = axes[0]
    y_all_data, _, _ = traces[-1]
    ax_l.hist(y_all_data, bins=50, color="gray", alpha=0.35, density=True, label="Full data")
    ax_l.axvline(-2.0, color="red", linestyle="--", alpha=0.7, linewidth=1.2, label="FC=0 (log_fc=−2)")
    ax_l.axvline(0.0, color="green", linestyle="--", alpha=0.7, linewidth=1.2, label="FC=1 (log_fc=0)")
    for i, (y_tr, _, label) in enumerate(traces):
        is_last = (i == n_traces - 1)
        lw = 2.5 if is_last else 1.2
        alpha = 0.9 if is_last else 0.5
        color = "black" if is_last else colors[i]
        try:
            kde = gaussian_kde(y_tr)
            ax_l.plot(x_range, kde(x_range), color=color, lw=lw, alpha=alpha, label=label)
        except Exception:
            pass
    ax_l.set_xlabel("log_fc")
    ax_l.set_ylabel("Density")
    ax_l.set_title("Label Distribution")
    ax_l.legend(fontsize=8)

    # ---- Right panel: LDS weight vs log_fc ----
    ax_r = axes[1]
    ax_r.axhline(1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1, label="Uniform (w=1)")
    for i, (y_tr, w_tr, label) in enumerate(traces):
        is_last = (i == n_traces - 1)
        lw = 2.5 if is_last else 1.0
        alpha = 0.8 if is_last else 0.5
        color = "black" if is_last else colors[i]
        sort_idx = np.argsort(y_tr)
        y_sorted = y_tr[sort_idx]
        w_sorted = w_tr[sort_idx]
        win = max(3, len(y_sorted) // 20)
        w_smooth = pd.Series(w_sorted).rolling(win, center=True, min_periods=1).mean().values
        ax_r.plot(y_sorted, w_smooth, color=color, lw=lw, alpha=alpha, label=label)

    # Annotate mean weight at FC=0 cluster and at FC>1
    y_last, w_last, _ = traces[-1]
    fc0_mask = np.abs(y_last - (-2.0)) < 0.05
    fc1_mask = y_last > 0.0
    if fc0_mask.sum() > 0:
        w_fc0 = float(w_last[fc0_mask].mean())
        ax_r.annotate(
            f"w̄(FC=0)={w_fc0:.2f}", xy=(-2.0, w_fc0),
            xytext=(-1.5, w_fc0 + 0.1), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.8),
        )
    if fc1_mask.sum() > 0:
        w_fc1 = float(w_last[fc1_mask].mean())
        y_fc1_rep = float(y_last[fc1_mask].mean())
        ax_r.annotate(
            f"w̄(FC>1)={w_fc1:.2f}", xy=(y_fc1_rep, w_fc1),
            xytext=(y_fc1_rep + 0.05, w_fc1 + 0.1), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.8),
        )
    ax_r.set_xlabel("log_fc")
    ax_r.set_ylabel("LDS Weight")
    ax_r.set_title("Per-Sample LDS Weights")
    ax_r.legend(fontsize=8)

    n_folds = max(0, n_traces - 1)
    fig.suptitle(f"LDS Weight Distribution — {n_folds} fold(s) + full dataset", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_acquisition_recovery(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    total_std: np.ndarray,
    wt_activity: float,
    output_path: Path,
    top_k_frac: float = 0.05,
    budget_fracs: tuple = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50),
    ucb_kappas: tuple = (1.0, 2.0),
    n_thompson: int = 20,
    seed: int = 42,
    title: str = "",
) -> None:
    """Plot acquisition function performance for prospective top-mutation recovery.

    Three panels:
      Left:   Recovery curves — cumulative recall of true top-k as more variants
              are tested, ranked by acquisition score. Compared to random.
      Middle: Hit rate vs screening budget — for each budget (top N%), what
              fraction of selected variants have log_fc > WT activity.
      Right:  Activity distribution (KDE) of selected top-5% variants vs all data.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    rng = np.random.default_rng(seed)
    n = len(y_true)

    k = max(5, min(int(round(top_k_frac * n)), int(0.20 * n)))
    actual_frac = k / n

    true_top_idx = set(np.argsort(y_true)[-k:])
    bg_hit_rate = float(np.mean(y_true > wt_activity))
    n_beneficial = int(np.sum(y_true > wt_activity))

    acq_scores = {"Mean": y_pred}
    for kappa in ucb_kappas:
        acq_scores[f"UCB(κ={kappa:.0f})"] = y_pred + kappa * total_std
    thompson_repeats = rng.normal(
        y_pred[None, :], total_std[None, :], size=(n_thompson, n)
    )
    acq_scores["Thompson"] = thompson_repeats.mean(axis=0)

    _colors = ["tab:blue", "tab:orange", "tab:red", "tab:green"]
    acq_colors = {name: _colors[i % len(_colors)]
                  for i, name in enumerate(acq_scores)}

    n_pts = min(n, 300)
    checkpoints = np.unique(np.round(np.linspace(0, n, n_pts)).astype(int))

    def _recovery_curve(scores):
        order = np.argsort(scores)[::-1]
        fracs, recalls = [], []
        for m in checkpoints:
            fracs.append(m / n)
            recalls.append(len(set(order[:int(m)]) & true_top_idx) / k)
        return np.array(fracs), np.array(recalls)

    def _delta_auc(fracs, recalls):
        return float(np.trapz(recalls, fracs)) - 0.5

    def _hit_rate(scores, budget_frac):
        m = max(1, int(round(budget_frac * n)))
        top_idx = np.argsort(scores)[::-1][:m]
        return float(np.mean(y_true[top_idx] > wt_activity))

    curves = {}
    for name, scores in acq_scores.items():
        if name == "Thompson":
            all_r = np.array([_recovery_curve(thompson_repeats[i])[1]
                               for i in range(n_thompson)])
            fracs = _recovery_curve(thompson_repeats[0])[0]
            r_mean, r_std = all_r.mean(axis=0), all_r.std(axis=0)
        else:
            fracs, r_mean = _recovery_curve(scores)
            r_std = None
        curves[name] = (fracs, r_mean, r_std, _delta_auc(fracs, r_mean))

    hit_rates = {}
    for name, scores in acq_scores.items():
        if name == "Thompson":
            hit_rates[name] = np.array([
                float(np.mean([_hit_rate(thompson_repeats[i], bf)
                               for i in range(n_thompson)]))
                for bf in budget_fracs
            ])
        else:
            hit_rates[name] = np.array([_hit_rate(scores, bf)
                                        for bf in budget_fracs])

    budget_pcts = np.array([b * 100 for b in budget_fracs])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Recovery curves
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.35, lw=1.2, label="Random")
    for name, (fracs, r_mean, r_std, dauc) in curves.items():
        color = acq_colors[name]
        ax.plot(fracs, r_mean, color=color, lw=2,
                label=f"{name}  (ΔAUC={dauc:+.3f})")
        if r_std is not None:
            ax.fill_between(fracs, r_mean - r_std, r_mean + r_std,
                            alpha=0.15, color=color)
    ax.axvline(actual_frac, color="gray", linestyle=":", alpha=0.5, lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Fraction of dataset screened")
    ax.set_ylabel(f"Recall of true top {actual_frac*100:.0f}%  ({k} variants)")
    ax.set_title(f"Top-{actual_frac*100:.0f}% mutation recovery")
    ax.legend(fontsize=8, loc="lower right")

    # Panel 2: Hit rate vs budget
    ax = axes[1]
    ax.axhline(bg_hit_rate * 100, color="gray", linestyle="--", lw=1.2,
               alpha=0.7,
               label=f"Background  ({bg_hit_rate*100:.1f}%,  n={n_beneficial})")
    for name, rates in hit_rates.items():
        color = acq_colors[name]
        ax.plot(budget_pcts, rates * 100, color=color, lw=2,
                marker="o", ms=4, label=name)
    ax.set_xscale("log")
    ax.set_xticks(budget_pcts)
    ax.set_xticklabels([f"{b:.0f}%" for b in budget_pcts], fontsize=8)
    ax.set_xlabel("Screening budget (top N% of dataset)")
    ax.set_ylabel("Hit rate  (% variants with log_fc > WT)")
    ax.set_title("Hit rate vs screening budget")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(fontsize=8)

    # Panel 3: Activity distributions
    ax = axes[2]
    dist_budget = 0.05
    k_dist = max(5, int(round(dist_budget * n)))
    x_range = np.linspace(y_true.min() - 0.3, y_true.max() + 0.3, 300)
    try:
        from scipy.stats import gaussian_kde
        kde_all = gaussian_kde(y_true)
        ax.fill_between(x_range, kde_all(x_range), alpha=0.2, color="gray",
                        label="All data")
        ax.plot(x_range, kde_all(x_range), color="gray", lw=1.5, alpha=0.5)
    except Exception:
        pass
    for name, scores in acq_scores.items():
        scores_use = thompson_repeats.mean(axis=0) if name == "Thompson" else scores
        top_idx = np.argsort(scores_use)[::-1][:k_dist]
        y_sel = y_true[top_idx]
        color = acq_colors[name]
        if len(y_sel) >= 3:
            try:
                kde_sel = gaussian_kde(y_sel)
                ax.plot(x_range, kde_sel(x_range), color=color, lw=2,
                        label=f"{name}  top {dist_budget*100:.0f}%")
            except Exception:
                pass
    ax.axvline(wt_activity, color="black", linestyle="--", lw=1.2, alpha=0.7,
               label=f"WT  ({wt_activity:.3f})")
    ax.set_xlabel("log_fc")
    ax.set_ylabel("Density")
    ax.set_title(f"Activity distribution: top {dist_budget*100:.0f}% by score vs all data")
    ax.legend(fontsize=8)

    suptitle = (f"Acquisition Recovery — {n} variants, "
                f"{n_beneficial} beneficial ({bg_hit_rate*100:.1f}%)")
    if title:
        suptitle = f"{title}  |  {suptitle}"
    fig.suptitle(suptitle, fontsize=11)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BNN1 Phase B: Formaldehyde Fold-Change Regression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 03_bnn1_formaldehyde_regression.py
  python 03_bnn1_formaldehyde_regression.py --hidden-dims '[128, 64]' --prior-std 0.5
  python 03_bnn1_formaldehyde_regression.py --esm-wt-scaler standard --esm-mut-pca 0.95
  python 03_bnn1_formaldehyde_regression.py --device cuda:1
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

    # Preprocessing hyperparams — WT
    parser.add_argument("--esm-wt-scaler", type=str, default=None,
                        choices=["none", "standard", "robust"],
                        help="Scaler for ESM WT features")
    parser.add_argument("--esm-wt-pca", type=str, default=None,
                        help="PCA for ESM WT: int, float 0<x<1, or 'none'")

    # Preprocessing hyperparams — mutant
    parser.add_argument("--esm-mut-scaler", type=str, default=None,
                        choices=["none", "standard", "robust"],
                        help="Scaler for ESM mutant features")
    parser.add_argument("--esm-mut-pca", type=str, default=None,
                        help="PCA for ESM mutant: int, float 0<x<1, or 'none'")

    # Flags
    parser.add_argument("--skip-final-model", action="store_true",
                        help="Skip training final model on all data (CV only)")
    parser.add_argument("--use-lds", action="store_true",
                        help="Apply Label Distribution Smoothing sample weights during training")

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
    preproc = config["preprocessing"]

    # Parse hidden_dims from CLI string if provided
    hidden_dims_cli = json.loads(args.hidden_dims) if args.hidden_dims else None

    # Parse PCA values from CLI strings
    esm_wt_pca_cli = parse_pca_value(args.esm_wt_pca) if args.esm_wt_pca is not None else None
    esm_wt_pca_cfg = parse_pca_value(resolve_param(preproc["esm_wt"]["pca"]))
    esm_mut_pca_cli = parse_pca_value(args.esm_mut_pca) if args.esm_mut_pca is not None else None
    esm_mut_pca_cfg = parse_pca_value(resolve_param(preproc["esm_mut"]["pca"]))

    # LDS config (read from config.yaml bnn1.lds if present; use_lds overridden by CLI)
    lds_cfg = config.get("bnn1", {}).get("lds", {})

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
        # Preprocessing — WT
        "esm_wt_scaler":          resolve_param(preproc["esm_wt"]["scaler"], args.esm_wt_scaler),
        "esm_wt_pca":             esm_wt_pca_cli if args.esm_wt_pca is not None else esm_wt_pca_cfg,
        # Preprocessing — mutant
        "esm_mut_scaler":         resolve_param(preproc["esm_mut"]["scaler"], args.esm_mut_scaler),
        "esm_mut_pca":            esm_mut_pca_cli if args.esm_mut_pca is not None else esm_mut_pca_cfg,
        # LDS
        "lds": {
            "use_lds":      lds_cfg.get("use_lds", False),
            "n_bins":       lds_cfg.get("n_bins", 50),
            "kernel_size":  lds_cfg.get("kernel_size", 5),
            "sigma":        lds_cfg.get("sigma", 2.0),
        },
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
    results_dir = PROJECT_ROOT / "results" / "03_formaldehyde_regression"
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir = results_dir / "models"

    setup_logging(results_dir)

    logger.info("=" * 60)
    logger.info("03_bnn1_formaldehyde_regression.py")
    logger.info("BNN1 Phase B: Formaldehyde Fold-Change Regression")
    logger.info("=" * 60)

    config = load_config(args.config)
    device = get_device(config, args.device)
    params = resolve_all_params(args, config)
    # CLI --use-lds overrides config; config bnn1.lds.use_lds is the default
    use_lds = args.use_lds or params["lds"].get("use_lds", False)

    logger.info("Hyperparameters:")
    for k, v in params.items():
        logger.info("  %s: %s", k, v)
    logger.info("  use_lds: %s", use_lds)

    logger.info("Results directory: %s", results_dir)

    # 2. Load data and build features (WT + mutant for regression)
    processed_dir = PROJECT_ROOT / config["data"]["output_dir"]
    X_wt, X_mut, y, positions, mutation_strings = \
        load_data_and_features(config, processed_dir)

    # 3. K-fold CV evaluation
    logger.info("Running %d-fold CV%s...", config["cv"]["n_folds"],
                " with LDS weights" if use_lds else "")
    (metrics, y_true, y_pred,
     epistemic_std, aleatoric_std, total_std,
     fold_histories, lds_traces) = evaluate_with_cv(
        X_wt, X_mut, y, positions, params, config, device,
        use_lds=use_lds,
    )

    # 4. Directional metrics (above/below WT)
    epsilon = config["data"]["epsilon"]
    wt_activity = float(np.log10(1.0 + epsilon))
    directional = compute_directional_metrics(y_true, y_pred, total_std, wt_activity)
    metrics["directional"] = directional

    logger.info("--- Directional (above/below WT = %.4f) ---", wt_activity)
    logger.info("Quadrant accuracy:          %.1f%%", directional["quadrant_accuracy"] * 100)
    logger.info("Balanced quadrant accuracy: %.1f%%", directional["balanced_quadrant_accuracy"] * 100)
    logger.info("  Sensitivity (beneficial): %.1f%%", directional["sensitivity"] * 100)
    logger.info("  Specificity (detrimental):%.1f%%", directional["specificity"] * 100)
    logger.info("Bal. uncert-adjusted:       %.1f%%", directional["balanced_uncertainty_adjusted_accuracy"] * 100)
    logger.info("Confident misclass:         %.1f%% (%d/%d)",
                directional["confident_misclassification_rate"] * 100,
                directional["n_confident_wrong"], len(y_true))
    pct = directional["quadrant_pct"]
    logger.info("Quadrants: TR=%.1f%%  BL=%.1f%%  TL=%.1f%%  BR=%.1f%%",
                pct["correct_beneficial_TR"], pct["correct_detrimental_BL"],
                pct["false_beneficial_TL"], pct["false_detrimental_BR"])

    # 5. Train final model on all data
    if not args.skip_final_model:
        logger.info("Training final model on all %d samples...", len(y))
        train_final_model(X_wt, X_mut, y, params, device, models_dir,
                          fold_histories=fold_histories)

    # 6. Plots
    logger.info("Generating plots...")
    position_offset = config["data"]["position_offset"]

    plot_parity(
        y_true, y_pred, total_std, positions,
        wt_activity, directional,
        results_dir / "parity_plot.png",
    )
    plot_residuals(
        y_true, y_pred,
        results_dir / "residuals_plot.png",
    )
    plot_calibration(
        metrics["calibration"],
        results_dir / "calibration_curve.png",
    )
    plot_per_position_spearman(
        metrics["per_position_spearman"],
        [],  # not used directly — labels computed inside from position_offset
        position_offset,
        results_dir / "per_position_spearman.png",
        per_pos_range=metrics.get("per_position_range"),
        weighted_mean=metrics.get("weighted_per_position_spearman"),
    )
    plot_per_position_mae(
        metrics["per_position_mae"],
        position_offset,
        metrics["mae"],
        results_dir / "per_position_mae.png",
    )
    plot_uncertainty_vs_error(
        y_true, y_pred, total_std,
        results_dir / "uncertainty_vs_error.png",
    )
    plot_uncertainty_decomposition(
        epistemic_std, aleatoric_std,
        results_dir / "uncertainty_decomposition.png",
    )
    plot_training_curves(
        fold_histories,
        results_dir / "training_curves.png",
    )
    plot_loss_decomposition(
        fold_histories,
        results_dir / "loss_decomposition.png",
    )
    if use_lds and lds_traces:
        plot_lds_weights(lds_traces, results_dir / "lds_weights.png")
    plot_acquisition_recovery(
        y_true, y_pred, total_std,
        wt_activity,
        results_dir / "acquisition_recovery.png",
        title="BNN1 Formaldehyde",
    )

    # 7. Save results
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Saved metrics.json")

    with open(results_dir / "hyperparams.json", "w") as f:
        json.dump(params, f, indent=2, default=str)
    logger.info("Saved hyperparams.json")

    with open(results_dir / "config_used.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info("Saved config_used.yaml")

    # Save predictions CSV
    pred_df = pd.DataFrame({
        "mutation_string": mutation_strings,
        "position": positions,
        "y_true": y_true,
        "y_pred": y_pred,
        "epistemic_std": epistemic_std,
        "aleatoric_std": aleatoric_std,
        "total_std": total_std,
        "residual": y_true - y_pred,
    })
    pred_df.to_csv(results_dir / "predictions.csv", index=False)
    logger.info("Saved predictions.csv (%d rows)", len(pred_df))

    # 8. Summary
    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("Formaldehyde Regression Complete (%.1fs)", elapsed)
    logger.info("=" * 60)
    logger.info("MAE:               %.4f  (null: %.4f, %.1f%% improvement)",
                metrics["mae"], metrics["null_mae"],
                metrics["mae_improvement_over_null"] * 100)
    logger.info("RMSE:              %.4f  (null: %.4f)", metrics["rmse"], metrics["null_rmse"])
    logger.info("R²:                %.4f", metrics["r2"])
    logger.info("Spearman ρ:        %.4f  (null: 0.000)", metrics["spearman_rho"])
    logger.info("NLPD:              %.4f  (proper scoring rule, lower = better)", metrics["nlpd"])
    logger.info("CRPS:              %.4f  (proper scoring rule, lower = better)", metrics["crps"])
    logger.info("Mean pos Spearman: %.4f  (range-weighted: %.4f, %d positions)",
                metrics["mean_per_position_spearman"],
                metrics["weighted_per_position_spearman"],
                metrics["n_positions_evaluated"])
    logger.info("Sharpness (mean σ): %.4f", metrics["sharpness"])
    logger.info("Bal. quad acc:     %.1f%%  (sens: %.1f%%, spec: %.1f%%)",
                directional["balanced_quadrant_accuracy"] * 100,
                directional["sensitivity"] * 100,
                directional["specificity"] * 100)
    logger.info("Bal. uncert-adj:   %.1f%%", directional["balanced_uncertainty_adjusted_accuracy"] * 100)
    logger.info("Fold Spearman: %s",
                " ".join(f"{f['spearman_rho']:.3f}" for f in metrics["fold_metrics"]))
    if not args.skip_final_model:
        logger.info("Final model saved to: %s/", models_dir)


if __name__ == "__main__":
    main()
