#!/usr/bin/env python
"""
06_predict.py — BNN2 Multi-Substrate Inference
===============================================

Load a trained BNN2 model and produce predictions with uncertainties for
mutations on one or more substrates.

Outputs:
  - CSV with predictions including null model baselines
    (formaldehyde, nearest-substrate) and acquisition function scores
    (Mean, UCB1, UCB2, Thompson)
  - Comprehensive plots: mutation heatmaps, activity distributions,
    uncertainty decomposition, acquisition function comparisons,
    top mutation rankings, and position-level summaries.

Three input modes:
  --input CSV              Pairwise: CSV with mutation_string, substrate,
                           ref_substrate, fc_ref columns.
  --mutations FILE         + --substrate NAME: predict one target substrate.
  --mutations FILE         (no --substrate): predict all substrates.

New substrates can be specified via --substrate-smiles (JSON or file) and
will have embeddings computed from SMILES (Morgan, MACCS, Mordred, or
MoLFormer — matching the trained model's substrate_embedding_type).
New mutations require pre-computed ESM2 embeddings (--extra-esm2-npz).

Usage:
    # All-substrates mode (expand all training substrates × refs)
    python 06_predict.py --mutations muts.txt --model-dir results/05_bnn2/random/models

    # Single-substrate mode
    python 06_predict.py --mutations muts.txt --substrate Formaldehyde \\
        --model-dir results/05_bnn2/random/models

    # Pairwise mode (full control)
    python 06_predict.py --input pairwise.csv --model-dir results/05_bnn2/random/models

    # New substrate from SMILES
    python 06_predict.py --mutations muts.txt --substrate Propanal \\
        --substrate-smiles '{"Propanal": "CCC=O"}' \\
        --model-dir results/05_bnn2/random/models

    # Custom output directory
    python 06_predict.py --mutations muts.txt --model-dir results/05_bnn2/random/models \\
        --output-dir results/my_run

    # Skip plots
    python 06_predict.py --mutations muts.txt --model-dir results/05_bnn2/random/models \\
        --no-plots
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

# Import shared utilities from 05_bnn2_common.py (starts with digit)
from importlib.util import spec_from_file_location, module_from_spec as _mfs

_common_spec = spec_from_file_location(
    "bnn2_common",
    SCRIPT_DIR / "05_bnn2_common.py",
)
_common = _mfs(_common_spec)
_common_spec.loader.exec_module(_common)

# Import from 01_embeddings.py for fingerprint computation
_emb_spec = spec_from_file_location(
    "embeddings_01",
    SCRIPT_DIR / "01_embeddings.py",
)
_emb_mod = _mfs(_emb_spec)
_emb_spec.loader.exec_module(_emb_mod)

# Config
load_config = _common.load_config
get_device = _common.get_device
setup_logging = _common.setup_logging

# Model
BNN2Model = _common.BNN2Model
build_bnn2_model = _common.build_bnn2_model

# Data loading
load_bnn1_backbone = _common.load_bnn1_backbone
load_all_embeddings = _common.load_all_embeddings
load_substrate_metadata = _common.load_substrate_metadata
load_multi_substrate_data = _common.load_multi_substrate_data

# Feature assembly
build_bnn1_input = _common.build_bnn1_input
build_other_features = _common.build_other_features
get_substrate_embedding = _common.get_substrate_embedding

# Aggregation
aggregate_pairwise_predictions = _common.aggregate_pairwise_predictions

# Fingerprint / embedding computation
compute_substrate_morgan = _emb_mod.compute_substrate_morgan
compute_substrate_maccs = _emb_mod.compute_substrate_maccs
compute_substrate_mordred = _emb_mod.compute_substrate_mordred
compute_substrate_molformer = _emb_mod.compute_substrate_molformer

# Substrate distance computation
compute_pairwise_distances = _common.compute_pairwise_distances


# ═══════════════════════════════════════════════════════════════════════════
# Mutation string parsing
# ═══════════════════════════════════════════════════════════════════════════

def parse_mutation_string(ms: str, position_offset: int) -> Tuple[str, int, str]:
    """Parse 'V83A' → (wt_aa, pos_0idx, mut_aa).

    Mutation strings use 1-indexed experimental positions.
    pos_0idx = pos_1idx - position_offset - 1
    """
    wt_aa = ms[0]
    mut_aa = ms[-1]
    pos_1idx = int(ms[1:-1])
    pos_0idx = pos_1idx - position_offset - 1
    return wt_aa, pos_0idx, mut_aa


# ═══════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════

def load_trained_model(
    models_dir: Path,
    bnn1_model_dir: Path,
    device: str,
) -> Tuple[BNN2Model, dict, dict]:
    """Load a trained BNN2 model from saved artifacts.

    Returns:
        model: BNN2Model with loaded weights
        params: hyperparameters dict (from hyperparams.json)
        metadata: model_metadata.json contents
    """
    # Validate required files
    required = [
        models_dir / "final_model.pt",
        models_dir / "model_metadata.json",
    ]
    hp_path = models_dir.parent / "hyperparams.json"
    required.append(hp_path)

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifacts:\n  " + "\n  ".join(missing)
        )

    # Load metadata
    with open(models_dir / "model_metadata.json") as f:
        metadata = json.load(f)
    with open(hp_path) as f:
        params = json.load(f)

    # Load BNN1 backbone (architecture only — weights will be overwritten
    # by the BNN2 checkpoint which contains the full model including BNN1)
    bnn1_hidden, bnn1_input_dim, latent_dim, _ = load_bnn1_backbone(
        bnn1_model_dir, device)

    # Respect the x_aa feature flag: if the model was trained without BNN1,
    # don't include it when reconstructing for inference.
    use_bnn1 = params.get("features", {}).get("x_aa", False)
    if not use_bnn1:
        bnn1_hidden = None
        bnn1_input_dim = 0
        latent_dim = 0

    # Construct BNN2 model with same architecture
    model = build_bnn2_model(
        bnn1_hidden, bnn1_input_dim, latent_dim,
        metadata["other_feature_dim"], params, device,
    )

    # Load trained weights (overwrites both BNN1 backbone and BNN2 head)
    checkpoint = torch.load(
        models_dir / "final_model.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Loaded trained BNN2 model from %s", models_dir / "final_model.pt")

    return model, params, metadata


def load_preprocessing_pipelines(models_dir: Path) -> Tuple[object, object, dict]:
    """Load all saved preprocessing pipelines from the model directory.

    Returns:
        bnn1_pipe_wt: fitted BNN1 WT preprocessing pipeline (or None)
        bnn1_pipe_mut: fitted BNN1 mutant preprocessing pipeline (or None)
        other_pipelines: dict of {group_name: fitted pipeline or None}
    """
    def _load_or_none(path):
        if path.exists():
            obj = joblib.load(path)
            # joblib files that are just None (identity transform)
            return obj if obj is not None else None
        return None

    bnn1_pipe_wt = _load_or_none(models_dir / "bnn1_preprocessing_wt.joblib")
    bnn1_pipe_mut = _load_or_none(models_dir / "bnn1_preprocessing_mut.joblib")

    # Discover every preprocessing_*.joblib in the model dir — training
    # saves one per feature group that was actually used (esm_wt, esm_mut,
    # ref_distance, etc. may or may not be present depending on hyperparams).
    other_pipelines = {}
    for p in sorted(models_dir.glob("preprocessing_*.joblib")):
        group = p.stem[len("preprocessing_"):]
        other_pipelines[group] = _load_or_none(p)

    logger.info("Loaded preprocessing pipelines from %s", models_dir)
    return bnn1_pipe_wt, bnn1_pipe_mut, other_pipelines


# ═══════════════════════════════════════════════════════════════════════════
# Feature preprocessing (inference — transform only)
# ═══════════════════════════════════════════════════════════════════════════

def apply_saved_preprocessing(
    groups: dict,
    pipelines: dict,
) -> np.ndarray:
    """Apply fitted preprocessing pipelines and concatenate in sorted key order.

    Must match the training-time concatenation order from
    preprocess_other_features (sorted group keys).
    """
    parts = []
    for group_name in sorted(groups.keys()):
        X = groups[group_name]
        pipe = pipelines.get(group_name)
        raw_dim = X.shape[1]
        if pipe is not None:
            X = pipe.transform(X).astype(np.float32)
        else:
            X = X.astype(np.float32)
        logger.info("  preproc[%s]: %d → %d", group_name, raw_dim, X.shape[1])
        parts.append(X)
    return np.concatenate(parts, axis=1)


# ═══════════════════════════════════════════════════════════════════════════
# New substrate handling
# ═══════════════════════════════════════════════════════════════════════════

def parse_substrate_smiles(arg: str) -> dict:
    """Parse --substrate-smiles from JSON string or file path.

    Returns: dict of {name: SMILES}
    """
    path = Path(arg)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    # Try parsing as JSON string
    try:
        return json.loads(arg)
    except json.JSONDecodeError:
        raise ValueError(
            f"--substrate-smiles must be a JSON string or path to a .json file. "
            f"Got: {arg}"
        )


def inject_new_substrates(
    embeddings: dict,
    substrate_meta: dict,
    new_smiles: dict,
    embedding_type: str,
    config: dict,
) -> None:
    """Compute fingerprints for new substrates and inject into embeddings/metadata.

    Modifies embeddings and substrate_meta in place.
    """
    # Filter out substrates that already exist
    existing = set(embeddings["substrate_names"])
    truly_new = {k: v for k, v in new_smiles.items() if k not in existing}
    if not truly_new:
        logger.info("All substrates in --substrate-smiles already exist; nothing to inject")
        return

    emb_cfg = config.get("embeddings", {})

    # Compute substrate embeddings matching the training embedding type
    if embedding_type == "morgan":
        new_fps = compute_substrate_morgan(
            truly_new,
            radius=emb_cfg.get("morgan_radius", 2),
            n_bits=emb_cfg.get("morgan_bits", 2048),
        )
    elif embedding_type == "maccs":
        new_fps = compute_substrate_maccs(truly_new)
    elif embedding_type == "mordred":
        feat_names_path = (
            PROJECT_ROOT / config["data"]["output_dir"]
            / "embeddings" / "mordred_feature_names.json"
        )
        existing_names = None
        if feat_names_path.exists():
            with open(feat_names_path) as f:
                existing_names = json.load(f)
        new_fps, _ = compute_substrate_mordred(
            truly_new, existing_feature_names=existing_names)
    elif embedding_type == "molformer":
        new_fps = compute_substrate_molformer(truly_new)
    else:
        raise ValueError(f"Unsupported embedding_type for new substrates: {embedding_type}")

    # Inject into embeddings dict
    emb_key = f"substrate_{embedding_type}"
    embeddings[emb_key] = np.concatenate(
        [embeddings[emb_key], new_fps.astype(np.float32)], axis=0)

    for name in truly_new:
        embeddings["substrate_names"].append(name)
        substrate_meta[name] = {
            "name": name,
            "smiles": truly_new[name],
            "is_active": False,
            "ref_type": "wt",
        }

    logger.info("Injected %d new substrate(s): %s", len(truly_new), list(truly_new.keys()))


# ═══════════════════════════════════════════════════════════════════════════
# Null model baselines
# ═══════════════════════════════════════════════════════════════════════════

FORMALDEHYDE_SUBSTRATE = "Formaldehyde"


def compute_formaldehyde_baseline(
    out_df: pd.DataFrame,
    df_train: pd.DataFrame,
) -> np.ndarray:
    """Formaldehyde null: predict each mutation's log_fc from formaldehyde training data.

    Falls back to training mean when the mutation wasn't tested on formaldehyde.
    """
    mean_pred = float(df_train["log_fc"].mean())

    form_train = df_train[df_train["substrate"] == FORMALDEHYDE_SUBSTRATE]
    form_lookup = dict(zip(form_train["mutation_string"], form_train["log_fc"].astype(float)))

    preds = out_df["mutation_string"].map(form_lookup).fillna(mean_pred).values.astype(np.float32)
    n_found = out_df["mutation_string"].isin(form_lookup).sum()
    logger.info("Formaldehyde baseline: %d/%d mutations found in training data",
                n_found, len(out_df))
    return preds


def compute_nearest_substrate_baseline(
    out_df: pd.DataFrame,
    df_train: pd.DataFrame,
    embeddings: dict,
    embedding_type: str,
    distance_metric: str = "cosine",
) -> np.ndarray:
    """Nearest-substrate null: for each target substrate, find the chemically
    nearest training substrate and use its measured log_fc for each mutation.

    Falls back to training mean when the mutation is absent on the nearest
    substrate.
    """
    mean_pred = float(df_train["log_fc"].mean())

    # Build (mutation_string, substrate) → log_fc lookup from training data
    train_lookup: dict = {}
    for _, row in df_train.iterrows():
        train_lookup[(row["mutation_string"], row["substrate"])] = float(row["log_fc"])
    train_substrates = set(df_train["substrate"].unique())

    # Substrate embeddings and distance matrix
    substrate_names = embeddings["substrate_names"]
    emb_key = f"substrate_{embedding_type}"
    emb = embeddings[emb_key].astype(np.float64)
    sub_to_idx = {name: i for i, name in enumerate(substrate_names)}
    dist_matrix = compute_pairwise_distances(emb, distance_metric)

    # Map each target substrate → nearest training substrate
    nearest_map: dict = {}
    for target_sub in out_df["substrate"].unique():
        if target_sub not in sub_to_idx:
            nearest_map[target_sub] = None
            continue
        target_idx = sub_to_idx[target_sub]
        best_dist, best_sub = float("inf"), None
        for train_sub in train_substrates:
            if train_sub == target_sub or train_sub not in sub_to_idx:
                continue
            d = float(dist_matrix[target_idx, sub_to_idx[train_sub]])
            if d < best_dist:
                best_dist, best_sub = d, train_sub
        nearest_map[target_sub] = best_sub
        logger.info("  Nearest-substrate null: %s → %s (dist=%.4f, %s/%s)",
                    target_sub, best_sub, best_dist, embedding_type, distance_metric)

    # Look up predictions
    preds = np.empty(len(out_df), dtype=np.float32)
    for i, (_, row) in enumerate(out_df.iterrows()):
        nearest = nearest_map.get(row["substrate"])
        if nearest is not None:
            preds[i] = train_lookup.get((row["mutation_string"], nearest), mean_pred)
        else:
            preds[i] = mean_pred
    return preds


# ═══════════════════════════════════════════════════════════════════════════
# FC reference lookup
# ═══════════════════════════════════════════════════════════════════════════

def build_fc_lookup(processed_dir: Path) -> dict:
    """Build (mutation_string, substrate) → fold_change lookup from training data."""
    df = load_multi_substrate_data(processed_dir)
    lookup = {}
    for _, row in df.iterrows():
        lookup[(row["mutation_string"], row["substrate"])] = row["fold_change"]
    logger.info("Built fc_ref lookup: %d entries from training data", len(lookup))
    return lookup


# ═══════════════════════════════════════════════════════════════════════════
# Input parsing & pairwise DataFrame construction
# ═══════════════════════════════════════════════════════════════════════════

def read_mutations_file(path: Path) -> List[str]:
    """Read mutation strings from a file.

    Accepts:
      - One mutation per line (plain text)
      - CSV with a 'mutation_string' column
    """
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        if "mutation_string" not in df.columns:
            raise ValueError(
                f"CSV file {path} must have a 'mutation_string' column. "
                f"Found: {list(df.columns)}"
            )
        return df["mutation_string"].dropna().unique().tolist()
    else:
        # Plain text: one per line
        with open(path) as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines


def build_inference_df(
    mutations: List[str],
    target_substrates: List[str],
    substrate_meta: dict,
    fc_lookup: dict,
    config: dict,
    position_offset: int,
    esm2_wt_len: int,
    esm2_mut_keys: set,
) -> Tuple[pd.DataFrame, dict]:
    """Build pairwise DataFrame for inference.

    For each (mutation, target_substrate), finds all valid reference
    substrates and looks up fc_ref from the training data.

    Returns:
        df: Pairwise DataFrame with required columns
        report: dict with skip/coverage statistics
    """
    pairwise_cfg = config.get("bnn2", {}).get("pairwise", {})
    ref_mode = pairwise_cfg.get("ref_substrates", "active_only")
    exclude_self = pairwise_cfg.get("exclude_self_ref", True)

    # Determine valid reference substrates
    if ref_mode == "active_only":
        valid_refs = [n for n, m in substrate_meta.items() if m.get("is_active", False)]
    else:
        valid_refs = list(substrate_meta.keys())

    rows = []
    skipped_no_esm = []
    skipped_no_refs = []
    ref_counts = {}  # (mutation, substrate) → n_refs

    for ms in mutations:
        # Parse mutation string
        try:
            wt_aa, pos_0idx, mut_aa = parse_mutation_string(ms, position_offset)
        except (ValueError, IndexError):
            logger.warning("Cannot parse mutation string '%s', skipping", ms)
            continue

        # Validate position bounds
        if pos_0idx < 0 or pos_0idx >= esm2_wt_len:
            logger.warning(
                "Position %d (from '%s') out of bounds [0, %d), skipping",
                pos_0idx, ms, esm2_wt_len)
            continue

        # Check ESM2 embedding exists
        if ms not in esm2_mut_keys:
            skipped_no_esm.append(ms)
            continue

        for target_sub in target_substrates:
            ref_type = substrate_meta.get(target_sub, {}).get("ref_type", "wt")
            is_active = substrate_meta.get(target_sub, {}).get("is_active", False)
            n_refs = 0

            for ref_sub in valid_refs:
                if exclude_self and ref_sub == target_sub:
                    continue
                fc_ref = fc_lookup.get((ms, ref_sub))
                if fc_ref is None:
                    continue

                rows.append({
                    "mutation_string": ms,
                    "position": pos_0idx,
                    "wt_aa": wt_aa,
                    "mut_aa": mut_aa,
                    "substrate": target_sub,
                    "ref_substrate": ref_sub,
                    "fc_ref": fc_ref,
                    "ref_type": ref_type,
                    "is_active_substrate": is_active,
                    "fold_change": 0.0,  # placeholder (not known at inference)
                    "log_fc": 0.0,       # placeholder
                })
                n_refs += 1

            if n_refs == 0:
                skipped_no_refs.append((ms, target_sub))
            else:
                ref_counts[(ms, target_sub)] = n_refs

    # Build report
    report = {
        "n_pairwise_rows": len(rows),
        "n_predictions": len(ref_counts),
        "skipped_no_esm": skipped_no_esm,
        "skipped_no_refs": skipped_no_refs,
        "ref_counts": ref_counts,
    }

    if not rows:
        return pd.DataFrame(), report

    df = pd.DataFrame(rows)
    return df, report


def load_pairwise_input(
    path: Path,
    substrate_meta: dict,
    position_offset: int,
    esm2_wt_len: int,
    esm2_mut_keys: set,
) -> Tuple[pd.DataFrame, dict]:
    """Load and validate a user-provided pairwise CSV.

    Required columns: mutation_string, substrate, ref_substrate, fc_ref
    Optional: position, wt_aa, mut_aa, ref_type (derived if missing)
    """
    df = pd.read_csv(path)
    required = ["mutation_string", "substrate", "ref_substrate", "fc_ref"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Pairwise CSV missing required columns: {missing_cols}. "
            f"Found: {list(df.columns)}"
        )

    # Derive missing columns
    if "position" not in df.columns or "wt_aa" not in df.columns or "mut_aa" not in df.columns:
        positions, wt_aas, mut_aas = [], [], []
        for ms in df["mutation_string"]:
            wt, pos, mut = parse_mutation_string(ms, position_offset)
            positions.append(pos)
            wt_aas.append(wt)
            mut_aas.append(mut)
        if "position" not in df.columns:
            df["position"] = positions
        if "wt_aa" not in df.columns:
            df["wt_aa"] = wt_aas
        if "mut_aa" not in df.columns:
            df["mut_aa"] = mut_aas

    if "ref_type" not in df.columns:
        df["ref_type"] = df["substrate"].map(
            lambda s: substrate_meta.get(s, {}).get("ref_type", "wt"))

    if "is_active_substrate" not in df.columns:
        df["is_active_substrate"] = df["substrate"].map(
            lambda s: substrate_meta.get(s, {}).get("is_active", False))

    if "fold_change" not in df.columns:
        df["fold_change"] = 0.0
    if "log_fc" not in df.columns:
        df["log_fc"] = 0.0

    # Validate ESM2 availability
    available = df["mutation_string"].isin(esm2_mut_keys)
    skipped = df.loc[~available, "mutation_string"].unique().tolist()
    df = df[available].reset_index(drop=True)

    # Validate positions
    valid_pos = (df["position"] >= 0) & (df["position"] < esm2_wt_len)
    bad_pos = df.loc[~valid_pos, "mutation_string"].unique().tolist()
    df = df[valid_pos].reset_index(drop=True)

    ref_counts = df.groupby(["mutation_string", "substrate"]).size().to_dict()

    report = {
        "n_pairwise_rows": len(df),
        "n_predictions": len(ref_counts),
        "skipped_no_esm": skipped,
        "skipped_no_refs": [],
        "skipped_bad_position": bad_pos,
        "ref_counts": ref_counts,
    }

    return df, report


# ═══════════════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_batched(
    model: BNN2Model,
    X: np.ndarray,
    n_samples: int,
    batch_size: int,
    device: str,
):
    """Run MC inference in batches, return concatenated UncertaintyEstimate."""
    from bnns.model import UncertaintyEstimate

    n = len(X)
    all_mean, all_epi, all_ale, all_tot = [], [], [], []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        X_batch = torch.tensor(X[start:end], dtype=torch.float32).to(device)

        est = model.predict_with_uncertainty(X_batch, n_samples=n_samples)

        all_mean.append(est.mean.cpu())
        all_epi.append(est.epistemic_std.cpu())
        all_ale.append(est.aleatoric_std.cpu())
        all_tot.append(est.total_std.cpu())

    return UncertaintyEstimate(
        mean=torch.cat(all_mean, dim=0),
        epistemic_std=torch.cat(all_epi, dim=0),
        aleatoric_std=torch.cat(all_ale, dim=0),
        total_std=torch.cat(all_tot, dim=0),
        samples=None,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Acquisition functions
# ═══════════════════════════════════════════════════════════════════════════

def compute_acquisition_scores(
    out_df: pd.DataFrame,
    n_thompson: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Add acquisition function columns to the output DataFrame.

    Computes:
        acq_mean:     Predictive mean (same as predicted_log_fc)
        acq_ucb1:     Mean + 1.0 * total_std
        acq_ucb2:     Mean + 2.0 * total_std
        acq_thompson: Mean of n_thompson samples from N(mean, total_std)

    Returns the DataFrame with added columns (modified in place).
    """
    rng = np.random.default_rng(seed)

    mean = out_df["predicted_log_fc"].values
    std = out_df["total_std"].values

    out_df["acq_mean"] = mean
    out_df["acq_ucb1"] = mean + 1.0 * std
    out_df["acq_ucb2"] = mean + 2.0 * std

    # Thompson: average of multiple draws from posterior predictive
    thompson_samples = rng.normal(
        mean[None, :], std[None, :], size=(n_thompson, len(mean))
    )
    out_df["acq_thompson"] = thompson_samples.mean(axis=0)

    return out_df


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

# Canonical amino acid order for heatmaps
_AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")

ACQ_NAMES = ["acq_mean", "acq_ucb1", "acq_ucb2", "acq_thompson"]
ACQ_LABELS = {
    "acq_mean": "Mean",
    "acq_ucb1": "UCB ($\\kappa$=1)",
    "acq_ucb2": "UCB ($\\kappa$=2)",
    "acq_thompson": "Thompson",
}
ACQ_COLORS = {
    "acq_mean": "#2196F3",
    "acq_ucb1": "#FF9800",
    "acq_ucb2": "#F44336",
    "acq_thompson": "#4CAF50",
}


def _ensure_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    return plt, mcolors


def plot_mutation_heatmap(
    out_df: pd.DataFrame,
    substrate: str,
    score_col: str,
    score_label: str,
    output_path: Path,
    position_offset: int,
):
    """Heatmap of scores: positions (x) × amino acid mutations (y).

    Each cell shows the score for mutating the WT residue at that position
    to the given amino acid. WT residues are marked with a dot.
    """
    plt, mcolors = _ensure_mpl()

    sub_df = out_df[out_df["substrate"] == substrate].copy()
    if sub_df.empty:
        return

    positions = sorted(sub_df["position"].unique())
    pos_labels = [str(p + position_offset + 1) for p in positions]

    # Build matrix: rows=amino acids, cols=positions
    matrix = np.full((len(_AA_ORDER), len(positions)), np.nan)
    wt_residues = {}

    for _, row in sub_df.iterrows():
        aa_idx = _AA_ORDER.index(row["mut_aa"]) if row["mut_aa"] in _AA_ORDER else None
        pos_idx = positions.index(row["position"]) if row["position"] in positions else None
        if aa_idx is not None and pos_idx is not None:
            matrix[aa_idx, pos_idx] = row[score_col]
        wt_residues[row["position"]] = row["wt_aa"]

    fig, ax = plt.subplots(figsize=(max(8, len(positions) * 0.7), 7))

    # Diverging colormap centered at 0 for mean/log_fc, sequential for UCB
    vals_flat = matrix[~np.isnan(matrix)]
    if len(vals_flat) == 0:
        plt.close(fig)
        return

    if "ucb" in score_col:
        cmap = "YlOrRd"
        vmin, vmax = np.nanmin(matrix), np.nanmax(matrix)
    else:
        abs_max = max(abs(np.nanmin(vals_flat)), abs(np.nanmax(vals_flat)), 0.1)
        cmap = "RdBu_r"
        vmin, vmax = -abs_max, abs_max

    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
                   interpolation="nearest")

    # Mark WT residues
    for j, pos in enumerate(positions):
        wt_aa = wt_residues.get(pos)
        if wt_aa and wt_aa in _AA_ORDER:
            i = _AA_ORDER.index(wt_aa)
            ax.plot(j, i, "ko", markersize=5, markerfacecolor="none", markeredgewidth=1.5)

    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels(pos_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(_AA_ORDER)))
    ax.set_yticklabels(_AA_ORDER, fontsize=8, family="monospace")
    ax.set_xlabel("Position (1-indexed)")
    ax.set_ylabel("Mutant Amino Acid")
    ax.set_title(f"{substrate} — {score_label}")

    plt.colorbar(im, ax=ax, label=score_label, shrink=0.8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_activity_distributions(
    out_df: pd.DataFrame,
    output_path: Path,
):
    """Per-substrate KDE/histogram of predicted log_fc values."""
    plt, _ = _ensure_mpl()
    from scipy.stats import gaussian_kde

    substrates = sorted(out_df["substrate"].unique())
    n_subs = len(substrates)
    ncols = min(3, n_subs)
    nrows = (n_subs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = np.atleast_2d(axes)

    for idx, sub in enumerate(substrates):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        vals = out_df.loc[out_df["substrate"] == sub, "predicted_log_fc"].values

        ax.hist(vals, bins=30, color="#2196F3", edgecolor="none", alpha=0.6,
                density=True, label="Histogram")

        if len(vals) >= 3:
            try:
                kde = gaussian_kde(vals)
                x_range = np.linspace(vals.min() - 0.3, vals.max() + 0.3, 200)
                ax.plot(x_range, kde(x_range), color="#1565C0", lw=2, label="KDE")
            except Exception:
                pass

        # WT reference line at 0 (log10(1 + eps) ~ 0)
        ax.axvline(0, color="black", linestyle="--", lw=1.2, alpha=0.7, label="WT (FC=1)")

        ax.set_xlabel("Predicted log$_{10}$(FC + $\\epsilon$)")
        ax.set_ylabel("Density")
        ax.set_title(sub, fontsize=10)
        ax.text(0.95, 0.95,
                f"n={len(vals)}\nmean={np.mean(vals):.3f}\nstd={np.std(vals):.3f}",
                transform=ax.transAxes, fontsize=7, va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        if idx == 0:
            ax.legend(fontsize=7)

    # Hide unused axes
    for idx in range(n_subs, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle("Predicted Activity Distributions per Substrate", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_uncertainty_decomposition_inference(
    out_df: pd.DataFrame,
    output_path: Path,
):
    """Epistemic vs aleatoric uncertainty scatter + fraction histogram."""
    plt, _ = _ensure_mpl()
    import matplotlib.cm as cm

    substrates = out_df["substrate"].values
    unique_subs = sorted(set(substrates))
    cmap = cm.get_cmap("tab10", len(unique_subs))
    color_map = {s: cmap(i) for i, s in enumerate(unique_subs)}

    epi = out_df["epistemic_std"].values
    ale = out_df["aleatoric_std"].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter
    for sub in unique_subs:
        mask = substrates == sub
        axes[0].scatter(epi[mask], ale[mask], c=[color_map[sub]], s=8,
                        alpha=0.4, label=sub, edgecolors="none")
    max_val = max(epi.max(), ale.max()) if len(epi) > 0 else 1
    axes[0].plot([0, max_val], [0, max_val], "k--", alpha=0.3)
    axes[0].set_xlabel("Epistemic $\\sigma$")
    axes[0].set_ylabel("Aleatoric $\\sigma$")
    axes[0].set_title("Uncertainty Decomposition")
    axes[0].legend(fontsize=6, ncol=2)

    # Fraction histogram
    total_var = epi**2 + ale**2
    epi_frac = epi**2 / np.clip(total_var, 1e-10, None)
    axes[1].hist(epi_frac, bins=50, color="#2196F3", edgecolor="none", alpha=0.7)
    axes[1].axvline(0.5, color="black", linewidth=1, linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Epistemic Fraction ($\\sigma^2_{epi}$ / $\\sigma^2_{total}$)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Epistemic Fraction Distribution")
    axes[1].text(0.05, 0.95,
                 f"Mean: {np.mean(epi_frac):.3f}\nMedian: {np.median(epi_frac):.3f}",
                 transform=axes[1].transAxes, fontsize=9, va="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_acquisition_comparison(
    out_df: pd.DataFrame,
    output_path: Path,
):
    """Compare rankings across acquisition functions.

    Left: Rank correlation heatmap across acquisition functions.
    Right: Score distributions (violin/box) per acquisition function.
    """
    plt, _ = _ensure_mpl()
    from scipy import stats

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # -- Left: rank correlation heatmap --
    n_acq = len(ACQ_NAMES)
    rank_corr = np.ones((n_acq, n_acq))
    for i, a1 in enumerate(ACQ_NAMES):
        for j, a2 in enumerate(ACQ_NAMES):
            if i < j:
                rho, _ = stats.spearmanr(out_df[a1], out_df[a2])
                rank_corr[i, j] = rho
                rank_corr[j, i] = rho

    im = axes[0].imshow(rank_corr, cmap="YlGnBu", vmin=0.5, vmax=1.0, aspect="equal")
    labels = [ACQ_LABELS[a] for a in ACQ_NAMES]
    axes[0].set_xticks(range(n_acq))
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    axes[0].set_yticks(range(n_acq))
    axes[0].set_yticklabels(labels, fontsize=9)
    for i in range(n_acq):
        for j in range(n_acq):
            axes[0].text(j, i, f"{rank_corr[i, j]:.3f}", ha="center", va="center",
                         fontsize=9, color="white" if rank_corr[i, j] > 0.8 else "black")
    plt.colorbar(im, ax=axes[0], label="Spearman $\\rho$", shrink=0.8)
    axes[0].set_title("Rank Correlation Across Acquisition Functions")

    # -- Right: score distributions --
    data_for_box = [out_df[a].values for a in ACQ_NAMES]
    bp = axes[1].boxplot(data_for_box, patch_artist=True, labels=labels)
    for patch, acq_name in zip(bp["boxes"], ACQ_NAMES):
        patch.set_facecolor(ACQ_COLORS[acq_name])
        patch.set_alpha(0.6)
    axes[1].set_ylabel("Score")
    axes[1].set_title("Acquisition Score Distributions")
    axes[1].axhline(0, color="black", linestyle="--", lw=0.8, alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_top_mutations_per_substrate(
    out_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 15,
):
    """Bar chart of top-N mutations per substrate for each acquisition function."""
    plt, _ = _ensure_mpl()

    substrates = sorted(out_df["substrate"].unique())
    n_subs = len(substrates)
    n_acq = len(ACQ_NAMES)

    fig, axes = plt.subplots(n_subs, n_acq,
                             figsize=(5 * n_acq, max(4, 0.4 * top_n) * n_subs))
    if n_subs == 1:
        axes = axes[np.newaxis, :]
    if n_acq == 1:
        axes = axes[:, np.newaxis]

    for i, sub in enumerate(substrates):
        sub_df = out_df[out_df["substrate"] == sub]
        for j, acq in enumerate(ACQ_NAMES):
            ax = axes[i, j]
            top = sub_df.nlargest(min(top_n, len(sub_df)), acq)
            bars = ax.barh(
                range(len(top)),
                top[acq].values,
                color=ACQ_COLORS[acq],
                alpha=0.7,
                edgecolor="none",
            )
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(top["mutation_string"].values, fontsize=7,
                               family="monospace")
            ax.invert_yaxis()
            ax.axvline(0, color="black", lw=0.8, alpha=0.5)

            if i == 0:
                ax.set_title(ACQ_LABELS[acq], fontsize=10)
            if j == 0:
                ax.set_ylabel(sub, fontsize=10, fontweight="bold")
            if i == n_subs - 1:
                ax.set_xlabel("Score")

    fig.suptitle(f"Top-{top_n} Mutations by Acquisition Function", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_position_summary(
    out_df: pd.DataFrame,
    output_path: Path,
    position_offset: int,
):
    """Per-position aggregated statistics: mean predicted_log_fc and uncertainty."""
    plt, _ = _ensure_mpl()
    import matplotlib.cm as cm

    substrates = sorted(out_df["substrate"].unique())
    cmap_colors = cm.get_cmap("tab10", len(substrates))
    sub_colors = {s: cmap_colors(i) for i, s in enumerate(substrates)}

    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(out_df["position"].unique()) * 0.5), 10),
                             sharex=True)

    # Group by substrate and position
    grouped = out_df.groupby(["substrate", "position"])

    positions_all = sorted(out_df["position"].unique())
    pos_labels = [str(p + position_offset + 1) for p in positions_all]
    x_ticks = np.arange(len(positions_all))
    bar_width = 0.8 / max(len(substrates), 1)

    for s_idx, sub in enumerate(substrates):
        means = []
        stds = []
        xs = []
        for p_idx, pos in enumerate(positions_all):
            sub_pos = out_df[(out_df["substrate"] == sub) & (out_df["position"] == pos)]
            if len(sub_pos) > 0:
                means.append(sub_pos["predicted_log_fc"].mean())
                stds.append(sub_pos["total_std"].mean())
                xs.append(p_idx + s_idx * bar_width)

        offset = s_idx * bar_width - (len(substrates) - 1) * bar_width / 2

        # Mean activity
        axes[0].bar(
            [p_idx + offset for p_idx, _ in enumerate(positions_all)
             if (out_df[(out_df["substrate"] == sub) & (out_df["position"] == _)]).shape[0] > 0],
            means, width=bar_width, color=sub_colors[sub], alpha=0.7,
            label=sub if len(positions_all) > 0 else None, edgecolor="none",
        )

        # Mean uncertainty
        axes[1].bar(
            [p_idx + offset for p_idx, _ in enumerate(positions_all)
             if (out_df[(out_df["substrate"] == sub) & (out_df["position"] == _)]).shape[0] > 0],
            stds, width=bar_width, color=sub_colors[sub], alpha=0.7,
            edgecolor="none",
        )

    axes[0].axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Mean Predicted log$_{10}$(FC + $\\epsilon$)")
    axes[0].set_title("Per-Position Mean Predicted Activity")
    axes[0].legend(fontsize=7, ncol=min(4, len(substrates)), loc="upper right")

    axes[1].set_ylabel("Mean Total $\\sigma$")
    axes[1].set_title("Per-Position Mean Uncertainty")
    axes[1].set_xticks(range(len(positions_all)))
    axes[1].set_xticklabels(pos_labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_xlabel("Position (1-indexed)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_uncertainty_vs_score(
    out_df: pd.DataFrame,
    output_path: Path,
):
    """Scatter of total uncertainty vs predicted mean, colored by substrate.

    Highlights the exploration-exploitation tradeoff: high-mean + high-uncertainty
    variants are promising exploration targets.
    """
    plt, _ = _ensure_mpl()
    import matplotlib.cm as cm

    substrates = out_df["substrate"].values
    unique_subs = sorted(set(substrates))
    cmap = cm.get_cmap("tab10", len(unique_subs))
    color_map = {s: cmap(i) for i, s in enumerate(unique_subs)}

    fig, ax = plt.subplots(figsize=(9, 7))

    for sub in unique_subs:
        mask = substrates == sub
        ax.scatter(
            out_df.loc[mask, "predicted_log_fc"],
            out_df.loc[mask, "total_std"],
            c=[color_map[sub]], s=12, alpha=0.5, label=sub, edgecolors="none",
        )

    ax.set_xlabel("Predicted log$_{10}$(FC + $\\epsilon$) [Exploitation]")
    ax.set_ylabel("Total $\\sigma$ [Exploration]")
    ax.set_title("Exploration-Exploitation Landscape")
    ax.legend(fontsize=7, ncol=2)
    ax.axvline(0, color="black", lw=0.8, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_acq_scatter_grid(
    out_df: pd.DataFrame,
    output_path: Path,
):
    """Pairwise scatter of acquisition function scores.

    Upper triangle: per-substrate colored scatter.
    Lower triangle: Spearman rho annotation.
    """
    plt, _ = _ensure_mpl()
    from scipy import stats
    import matplotlib.cm as cm

    n_acq = len(ACQ_NAMES)
    substrates = out_df["substrate"].values
    unique_subs = sorted(set(substrates))
    cmap = cm.get_cmap("tab10", len(unique_subs))
    color_map = {s: cmap(i) for i, s in enumerate(unique_subs)}

    fig, axes = plt.subplots(n_acq, n_acq, figsize=(4 * n_acq, 4 * n_acq))

    for i, a1 in enumerate(ACQ_NAMES):
        for j, a2 in enumerate(ACQ_NAMES):
            ax = axes[i, j]
            if i == j:
                # Diagonal: histogram
                ax.hist(out_df[a1], bins=40, color=ACQ_COLORS[a1],
                        edgecolor="none", alpha=0.7)
                ax.set_title(ACQ_LABELS[a1], fontsize=9)
            elif i < j:
                # Upper triangle: scatter
                for sub in unique_subs:
                    mask = substrates == sub
                    ax.scatter(out_df.loc[mask, a2], out_df.loc[mask, a1],
                               c=[color_map[sub]], s=4, alpha=0.3, edgecolors="none")
                rho, _ = stats.spearmanr(out_df[a1], out_df[a2])
                ax.text(0.05, 0.95, f"$\\rho$={rho:.3f}",
                        transform=ax.transAxes, fontsize=9, va="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            else:
                # Lower triangle: hide
                ax.set_visible(False)

            if i == n_acq - 1 and j < n_acq:
                ax.set_xlabel(ACQ_LABELS.get(ACQ_NAMES[j], ""), fontsize=8)
            if j == 0 and i < n_acq:
                ax.set_ylabel(ACQ_LABELS.get(ACQ_NAMES[i], ""), fontsize=8)

    fig.suptitle("Acquisition Function Pairwise Comparison", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_rank_stability(
    out_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 20,
):
    """For each substrate, show how the top-N mutations by mean score are
    ranked by each acquisition function (bump chart / parallel coordinates).
    """
    plt, _ = _ensure_mpl()

    substrates = sorted(out_df["substrate"].unique())
    n_subs = len(substrates)
    ncols = min(3, n_subs)
    nrows = (n_subs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = np.atleast_2d(axes)

    for idx, sub in enumerate(substrates):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

        sub_df = out_df[out_df["substrate"] == sub].copy()
        if len(sub_df) == 0:
            ax.set_visible(False)
            continue

        # Get top-N by mean
        top_by_mean = sub_df.nlargest(min(top_n, len(sub_df)), "acq_mean")
        top_mutations = top_by_mean["mutation_string"].values

        # Compute ranks within this substrate for each acq function
        rank_data = {}
        for acq in ACQ_NAMES:
            sub_df[f"_rank_{acq}"] = sub_df[acq].rank(ascending=False, method="min")
            rank_data[acq] = sub_df.set_index("mutation_string").loc[
                top_mutations, f"_rank_{acq}"
            ].values

        # Plot parallel coordinates
        x_pos = np.arange(len(ACQ_NAMES))
        for m_idx, mut in enumerate(top_mutations):
            ranks = [rank_data[acq][m_idx] for acq in ACQ_NAMES]
            alpha = 0.7 if m_idx < 5 else 0.3
            lw = 2 if m_idx < 5 else 1
            ax.plot(x_pos, ranks, "-o", markersize=3, alpha=alpha, lw=lw)
            if m_idx < 5:
                ax.annotate(mut, (x_pos[-1], ranks[-1]),
                            textcoords="offset points", xytext=(5, 0),
                            fontsize=6, family="monospace", alpha=0.8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([ACQ_LABELS[a] for a in ACQ_NAMES], fontsize=8, rotation=15)
        ax.set_ylabel("Rank (lower = better)")
        ax.invert_yaxis()
        ax.set_title(sub, fontsize=10)

    for idx in range(n_subs, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle(f"Top-{top_n} Mutation Rank Stability Across Acquisition Functions",
                 fontsize=13)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_beneficial_fraction(
    out_df: pd.DataFrame,
    output_path: Path,
    budget_fracs: tuple = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50),
):
    """For each substrate, show the fraction of top-N% variants predicted
    as beneficial (predicted_log_fc > 0, i.e. FC > 1) per acquisition function.
    """
    plt, _ = _ensure_mpl()

    substrates = sorted(out_df["substrate"].unique())
    n_subs = len(substrates)
    ncols = min(3, n_subs)
    nrows = (n_subs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = np.atleast_2d(axes)

    budget_pcts = [b * 100 for b in budget_fracs]

    for idx, sub in enumerate(substrates):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        sub_df = out_df[out_df["substrate"] == sub]
        n = len(sub_df)

        if n == 0:
            ax.set_visible(False)
            continue

        bg_rate = float(np.mean(sub_df["predicted_log_fc"].values > 0))
        ax.axhline(bg_rate * 100, color="gray", linestyle="--", lw=1.2, alpha=0.7,
                    label=f"All ({bg_rate*100:.1f}%)")

        for acq in ACQ_NAMES:
            rates = []
            for bf in budget_fracs:
                m = max(1, int(round(bf * n)))
                top_idx = sub_df[acq].nlargest(m).index
                rate = float(np.mean(sub_df.loc[top_idx, "predicted_log_fc"].values > 0))
                rates.append(rate * 100)
            ax.plot(budget_pcts, rates, "-o", ms=4, lw=2,
                    color=ACQ_COLORS[acq], label=ACQ_LABELS[acq])

        ax.set_xscale("log")
        ax.set_xticks(budget_pcts)
        ax.set_xticklabels([f"{b:.0f}%" for b in budget_pcts], fontsize=7)
        ax.set_xlabel("Budget (top N%)")
        ax.set_ylabel("% Predicted Beneficial")
        ax.set_title(sub, fontsize=10)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        if idx == 0:
            ax.legend(fontsize=7)

    for idx in range(n_subs, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle("Predicted Beneficial Fraction vs Screening Budget", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


# ═══════════════════════════════════════════════════════════════════════════
# Null-vs-model comparison plots
# ═══════════════════════════════════════════════════════════════════════════

NULL_COLS = [
    ("null_formaldehyde", "Formaldehyde null"),
    ("null_nearest_substrate", "Nearest-substrate null"),
]


def _substrate_color_map(substrates):
    import matplotlib.cm as cm
    cmap = cm.get_cmap("tab10", max(len(substrates), 1))
    return {s: cmap(i) for i, s in enumerate(substrates)}


def plot_null_vs_model_parity(out_df: pd.DataFrame, output_path: Path):
    """3-panel parity grid: model vs each null, and null vs null.

    Each panel shows a scatter of all (mutation, substrate) points colored by
    substrate with a y=x diagonal. Spearman rho and MAE are reported in titles.
    """
    plt, _ = _ensure_mpl()
    from scipy.stats import spearmanr

    substrates = sorted(out_df["substrate"].unique())
    color_map = _substrate_color_map(substrates)

    pairs = [
        ("predicted_log_fc", "null_formaldehyde", "Model", "Formaldehyde null"),
        ("predicted_log_fc", "null_nearest_substrate", "Model", "Nearest-substrate null"),
        ("null_formaldehyde", "null_nearest_substrate", "Formaldehyde null", "Nearest-substrate null"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, (xc, yc, xl, yl) in zip(axes, pairs):
        x = out_df[xc].values.astype(float)
        y = out_df[yc].values.astype(float)
        for sub in substrates:
            mask = out_df["substrate"].values == sub
            ax.scatter(x[mask], y[mask], s=10, alpha=0.55,
                       color=color_map[sub], edgecolors="none", label=sub)
        lo = float(min(x.min(), y.min())) - 0.1
        hi = float(max(x.max(), y.max())) + 0.1
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="y = x")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        rho, _ = spearmanr(x, y)
        mae = float(np.mean(np.abs(x - y)))
        ax.set_title(f"{xl} vs {yl}\nSpearman ρ = {rho:.3f}    MAE = {mae:.3f}",
                     fontsize=10)
        ax.grid(alpha=0.25)

    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                    fontsize=7, frameon=True, title="Substrate")
    fig.suptitle("Parity: BNN2 model predictions vs null baselines", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_null_disagreement_distributions(out_df: pd.DataFrame, output_path: Path):
    """Per-substrate violins of (model − null) for both nulls.

    Substrates ordered by median model−formaldehyde disagreement so the user
    can see at a glance which substrates the model thinks differ most from
    the formaldehyde anchor. Horizontal line at 0 = perfect agreement.
    """
    plt, _ = _ensure_mpl()

    df = out_df.copy()
    df["delta_form"] = df["predicted_log_fc"] - df["null_formaldehyde"]
    df["delta_near"] = df["predicted_log_fc"] - df["null_nearest_substrate"]

    median_form = df.groupby("substrate")["delta_form"].median().sort_values()
    sub_order = list(median_form.index)

    fig, axes = plt.subplots(1, 2, figsize=(max(10, 0.7 * len(sub_order) + 4), 6),
                             sharey=False)
    for ax, col, title in [
        (axes[0], "delta_form", "Model − Formaldehyde null"),
        (axes[1], "delta_near", "Model − Nearest-substrate null"),
    ]:
        data = [df.loc[df["substrate"] == s, col].values for s in sub_order]
        parts = ax.violinplot(data, positions=range(len(sub_order)),
                              showmedians=True, showextrema=False, widths=0.85)
        for body in parts["bodies"]:
            body.set_facecolor("#4C9CD9"); body.set_edgecolor("#1f4e79"); body.set_alpha(0.7)
        if "cmedians" in parts:
            parts["cmedians"].set_color("black")
        # Strip overlay
        for i, s in enumerate(sub_order):
            vals = df.loc[df["substrate"] == s, col].values
            jitter = (np.random.RandomState(0).rand(len(vals)) - 0.5) * 0.2
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       s=4, color="black", alpha=0.25, edgecolors="none")

        ax.axhline(0, color="red", ls="--", lw=1, alpha=0.7, label="agreement")
        ax.set_xticks(range(len(sub_order)))
        ax.set_xticklabels(sub_order, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Δ log$_{10}$(FC)")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Model vs null disagreement per substrate "
                 "(>0 → model says more active than null)", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_top_disagreement_mutations(out_df: pd.DataFrame, output_path: Path,
                                    top_n: int = 20):
    """Per-substrate × per-null grid: top-N mutations by |model − null|.

    Bars colored by sign of disagreement (red = model says less active, green =
    more active). Error bars show model `total_std` so the user can see which
    disagreements are confident vs uncertain.
    """
    plt, _ = _ensure_mpl()

    substrates = sorted(out_df["substrate"].unique())
    n_subs = len(substrates)
    fig, axes = plt.subplots(n_subs, 2,
                             figsize=(14, max(2.5 * n_subs, 4)),
                             sharex=False)
    if n_subs == 1:
        axes = np.atleast_2d(axes)

    for r, sub in enumerate(substrates):
        sub_df = out_df[out_df["substrate"] == sub]
        for c, (null_col, null_label) in enumerate(NULL_COLS):
            ax = axes[r, c]
            delta = sub_df["predicted_log_fc"].values - sub_df[null_col].values
            order = np.argsort(-np.abs(delta))[:top_n]
            sel = sub_df.iloc[order]
            d = delta[order]
            std = sel["total_std"].values
            colors = ["#2ca02c" if v > 0 else "#d62728" for v in d]
            y = np.arange(len(d))
            ax.barh(y, d, color=colors, alpha=0.85,
                    xerr=std, error_kw=dict(ecolor="black", lw=0.8, capsize=2))
            ax.set_yticks(y)
            ax.set_yticklabels(sel["mutation_string"].values, fontsize=7)
            ax.invert_yaxis()
            ax.axvline(0, color="black", lw=0.7)
            if r == 0:
                ax.set_title(f"vs {null_label}", fontsize=10)
            if c == 0:
                ax.set_ylabel(sub, fontsize=10, fontweight="bold")
            ax.grid(axis="x", alpha=0.25)

    axes[-1, 0].set_xlabel("Model − null  (Δ log$_{10}$ FC, ±total std)")
    axes[-1, 1].set_xlabel("Model − null  (Δ log$_{10}$ FC, ±total std)")
    fig.suptitle(f"Top-{top_n} mutations where model most disagrees with each null",
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_null_vs_model_rank_agreement(out_df: pd.DataFrame, output_path: Path,
                                      top_k: int = 25):
    """Per-substrate bump chart of top-K mutations under each scoring method.

    Methods compared: model mean, model UCB2, formaldehyde null, nearest-substrate
    null. A mutation that is top-K under the model but rank >>K under both nulls
    is one the nulls would never select — exactly the case where the model adds
    value over baselines.
    """
    plt, _ = _ensure_mpl()
    import matplotlib.cm as cm

    substrates = sorted(out_df["substrate"].unique())
    score_cols = [
        ("predicted_log_fc", "Model mean"),
        ("acq_ucb2", "Model UCB2"),
        ("null_formaldehyde", "Formaldehyde null"),
        ("null_nearest_substrate", "Nearest-substrate null"),
    ]

    n_subs = len(substrates)
    ncols = min(2, n_subs)
    nrows = (n_subs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 4.5 * nrows),
                             squeeze=False)

    for idx, sub in enumerate(substrates):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        sub_df = out_df[out_df["substrate"] == sub].copy()

        # rank: 1 = best (highest score)
        ranks = {}
        for col, _ in score_cols:
            ranks[col] = sub_df[col].rank(ascending=False, method="min").astype(int).values
        muts = sub_df["mutation_string"].values

        # Identify mutations in top-K under any method
        top_mask = np.zeros(len(sub_df), dtype=bool)
        for col, _ in score_cols:
            top_mask |= ranks[col] <= top_k
        idx_keep = np.where(top_mask)[0]

        # Color by model rank
        cmap = cm.get_cmap("plasma")
        model_rank = ranks["predicted_log_fc"]
        norm_rank = np.clip(model_rank[idx_keep] / top_k, 0, 1)

        x = np.arange(len(score_cols))
        for j, ki in enumerate(idx_keep):
            ys = [ranks[col][ki] for col, _ in score_cols]
            color = cmap(norm_rank[j])
            ax.plot(x, ys, "-", color=color, lw=0.9, alpha=0.7)
            ax.scatter(x, ys, color=color, s=18, zorder=3)
            # Label mutations that are top-5 under model
            if model_rank[ki] <= 5:
                ax.text(0 - 0.05, model_rank[ki], muts[ki],
                        fontsize=6, ha="right", va="center")

        ax.invert_yaxis()
        ax.set_ylim(top_k * 1.5 + 1, 0)
        ax.axhline(top_k + 0.5, color="grey", ls="--", lw=0.7,
                   label=f"top-{top_k} cutoff")
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in score_cols], rotation=20, ha="right",
                           fontsize=8)
        ax.set_ylabel("Rank (1 = best)")
        ax.set_title(sub, fontsize=10)
        ax.grid(axis="y", alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=7, loc="lower right")

    for idx in range(n_subs, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle(f"Rank agreement: top-{top_k} mutations across model & null scoring",
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_uncertainty_vs_disagreement(out_df: pd.DataFrame, output_path: Path):
    """Scatter: |model − null| vs model total_std, per null.

    A well-calibrated model should be MORE uncertain where it disagrees with
    baselines (positive trend → calibration sanity check). Spearman rho between
    |Δ| and total_std is reported.
    """
    plt, _ = _ensure_mpl()
    from scipy.stats import spearmanr

    substrates = sorted(out_df["substrate"].unique())
    color_map = _substrate_color_map(substrates)
    std = out_df["total_std"].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, (null_col, null_label) in zip(axes, NULL_COLS):
        delta = np.abs(out_df["predicted_log_fc"].values - out_df[null_col].values)
        for sub in substrates:
            mask = out_df["substrate"].values == sub
            ax.scatter(delta[mask], std[mask], s=10, alpha=0.5,
                       color=color_map[sub], edgecolors="none", label=sub)
        rho, _ = spearmanr(delta, std)
        ax.set_xlabel(f"|Model − {null_label}|")
        ax.set_ylabel("Model total std")
        ax.set_title(f"vs {null_label}    Spearman ρ(|Δ|, std) = {rho:.3f}",
                     fontsize=10)
        ax.grid(alpha=0.25)

    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7,
                    title="Substrate", frameon=True)
    fig.suptitle("Is the model more uncertain when it disagrees with the nulls?",
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def generate_all_plots(
    out_df: pd.DataFrame,
    plot_dir: Path,
    position_offset: int,
):
    """Generate all inference plots and save to plot_dir."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating plots in %s ...", plot_dir)

    substrates = sorted(out_df["substrate"].unique())

    # 1. Mutation heatmaps — per substrate, per acquisition function
    heatmap_dir = plot_dir / "heatmaps"
    heatmap_dir.mkdir(exist_ok=True)
    for sub in substrates:
        sub_safe = sub.replace(" ", "_").replace("/", "_")
        # Predicted mean heatmap
        plot_mutation_heatmap(
            out_df, sub, "predicted_log_fc", "Predicted log$_{10}$(FC + $\\epsilon$)",
            heatmap_dir / f"heatmap_{sub_safe}_mean.png",
            position_offset,
        )
        # Uncertainty heatmap
        plot_mutation_heatmap(
            out_df, sub, "total_std", "Total $\\sigma$",
            heatmap_dir / f"heatmap_{sub_safe}_uncertainty.png",
            position_offset,
        )
        # UCB2 heatmap
        plot_mutation_heatmap(
            out_df, sub, "acq_ucb2", "UCB ($\\kappa$=2)",
            heatmap_dir / f"heatmap_{sub_safe}_ucb2.png",
            position_offset,
        )

    # 2. Activity distributions
    plot_activity_distributions(out_df, plot_dir / "activity_distributions.png")

    # 3. Uncertainty decomposition
    plot_uncertainty_decomposition_inference(
        out_df, plot_dir / "uncertainty_decomposition.png")

    # 4. Exploration-exploitation landscape
    plot_uncertainty_vs_score(out_df, plot_dir / "exploration_exploitation.png")

    # 5. Acquisition function comparison
    plot_acquisition_comparison(out_df, plot_dir / "acquisition_comparison.png")

    # 6. Acquisition pairwise scatter grid
    plot_acq_scatter_grid(out_df, plot_dir / "acquisition_scatter_grid.png")

    # 7. Top mutations per substrate per acquisition function
    plot_top_mutations_per_substrate(out_df, plot_dir / "top_mutations.png")

    # 8. Rank stability across acquisition functions
    plot_rank_stability(out_df, plot_dir / "rank_stability.png")

    # 9. Beneficial fraction vs screening budget
    plot_beneficial_fraction(out_df, plot_dir / "beneficial_fraction.png")

    # 10. Position-level summary
    plot_position_summary(out_df, plot_dir / "position_summary.png", position_offset)

    # 11. Null-vs-model comparison plots
    if all(c in out_df.columns for c in ("null_formaldehyde", "null_nearest_substrate")):
        null_dir = plot_dir / "null_comparison"
        null_dir.mkdir(parents=True, exist_ok=True)
        plot_null_vs_model_parity(out_df, null_dir / "parity_grid.png")
        plot_null_disagreement_distributions(out_df, null_dir / "disagreement_distributions.png")
        plot_top_disagreement_mutations(out_df, null_dir / "top_disagreements.png", top_n=20)
        plot_null_vs_model_rank_agreement(out_df, null_dir / "rank_agreement.png", top_k=25)
        plot_uncertainty_vs_disagreement(out_df, null_dir / "uncertainty_vs_disagreement.png")
    else:
        logger.info("Null columns missing — skipping null comparison plots.")

    logger.info("All plots saved to %s", plot_dir)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BNN2 Multi-Substrate Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input modes:
  Pairwise:         --input pairwise.csv
  Single-substrate: --mutations muts.txt --substrate Formaldehyde
  All-substrates:   --mutations muts.txt

Examples:
  python 06_predict.py --mutations muts.txt --model-dir results/05_bnn2/random/models
  python 06_predict.py --mutations muts.txt --substrate Formaldehyde \\
      --model-dir results/05_bnn2/random/models --device cuda:0
  python 06_predict.py --input pairwise.csv --model-dir results/05_bnn2/random/models
  python 06_predict.py --mutations muts.txt --substrate Propanal \\
      --substrate-smiles '{"Propanal": "CCC=O"}' \\
      --model-dir results/05_bnn2/random/models
        """,
    )
    # Input (mutually exclusive groups)
    input_grp = parser.add_mutually_exclusive_group(required=False)
    input_grp.add_argument("--input", type=str,
                           help="Pairwise CSV with mutation_string, substrate, "
                                "ref_substrate, fc_ref columns")
    input_grp.add_argument("--mutations", type=str,
                           help="File with mutation strings (one per line or CSV "
                                "with mutation_string column). If omitted, all "
                                "mutations from multi_substrate_ssm.csv are used.")

    parser.add_argument("--substrate", type=str, default=None,
                        help="Target substrate name (single-substrate mode)")

    # Model
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to BNN2 models/ directory")
    parser.add_argument("--bnn1-model-dir", type=str, default=None,
                        help="Path to BNN1 models/ directory "
                             "(default: results/03_formaldehyde_regression/models)")

    # Data
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--substrate-smiles", type=str, default=None,
                        help="New substrate SMILES as JSON string or path to .json file")
    parser.add_argument("--extra-esm2-npz", type=str, default=None,
                        help="Additional ESM2 mutant embeddings NPZ file")
    parser.add_argument("--extra-saprot-json", type=str, default=None,
                        help="Additional SaProt scores JSON file")

    # Inference
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=None,
                        help="MC inference samples (default: from hyperparams.json)")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-thompson", type=int, default=50,
                        help="Number of Thompson sampling draws (default: 50)")

    # Output
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for all outputs: CSV, plots, log "
                             "(default: <model-dir>/../predict_<timestamp>)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--null-distance-metric", type=str, default="cosine",
                        choices=["cosine", "euclidean", "manhattan", "correlation"],
                        help="Distance metric for nearest-substrate null model (default: cosine)")

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    t_start = time.time()

    models_dir = Path(args.model_dir)

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = models_dir.parent / f"predict_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    log_path = output_dir / "predict.log"
    setup_logging(log_path)

    logger.info("=" * 60)
    logger.info("06_predict.py — BNN2 Inference")
    logger.info("=" * 60)

    # ── 1. Load config ──
    config = load_config(args.config)
    device = get_device(config, args.device)
    position_offset = config["data"]["position_offset"]

    # ── 2. Load BNN1 backbone ──
    if args.bnn1_model_dir:
        bnn1_model_dir = Path(args.bnn1_model_dir)
    else:
        bnn1_model_dir = PROJECT_ROOT / "results" / "03_formaldehyde_regression" / "models"

    # ── 3. Load trained BNN2 model ──
    model, params, metadata = load_trained_model(models_dir, bnn1_model_dir, device)

    n_samples = args.n_samples or params.get("n_inference_samples", 200)
    embedding_type = params.get("substrate_embedding_type", "morgan")
    logger.info("Inference: n_samples=%d, embedding_type=%s", n_samples, embedding_type)

    # ── 4. Load preprocessing pipelines ──
    bnn1_pipe_wt, bnn1_pipe_mut, other_pipelines = load_preprocessing_pipelines(models_dir)

    # ── 5. Load embeddings + metadata ──
    processed_dir = PROJECT_ROOT / config["data"]["output_dir"]
    embeddings = load_all_embeddings(processed_dir)
    substrate_meta = load_substrate_metadata(processed_dir)

    esm2_wt_len = embeddings["esm2_wt"].shape[0]
    esm2_mut_keys = set(embeddings["esm2_mut"].keys())

    # ── 6. Inject extra ESM2 embeddings ──
    if args.extra_esm2_npz:
        extra = dict(np.load(args.extra_esm2_npz))
        n_before = len(esm2_mut_keys)
        embeddings["esm2_mut"].update(extra)
        esm2_mut_keys = set(embeddings["esm2_mut"].keys())
        logger.info("Injected %d extra ESM2 embeddings (%d → %d total)",
                    len(extra), n_before, len(esm2_mut_keys))

    # ── 7. Inject extra SaProt scores ──
    if args.extra_saprot_json:
        with open(args.extra_saprot_json) as f:
            extra_saprot = json.load(f)
        embeddings["saprot"].update(extra_saprot)
        logger.info("Injected %d extra SaProt scores", len(extra_saprot))

    # ── 8. Inject new substrates ──
    if args.substrate_smiles:
        new_smiles = parse_substrate_smiles(args.substrate_smiles)
        inject_new_substrates(embeddings, substrate_meta, new_smiles,
                              embedding_type, config)

    # ── 9. Build pairwise DataFrame ──
    if args.input:
        # Pairwise mode
        logger.info("Mode: pairwise (reading from %s)", args.input)
        df_pairwise, report = load_pairwise_input(
            Path(args.input), substrate_meta, position_offset,
            esm2_wt_len, esm2_mut_keys)
    else:
        # Read mutations from file or default to all multi-substrate mutations
        if args.mutations:
            mutations = read_mutations_file(Path(args.mutations))
            logger.info("Read %d unique mutations from %s", len(mutations), args.mutations)
        else:
            multi_csv = processed_dir / "multi_substrate_ssm.csv"
            multi_df = pd.read_csv(multi_csv)
            mutations = sorted(multi_df["mutation_string"].unique().tolist())
            logger.info("No --mutations provided; using all %d unique mutations "
                        "from %s", len(mutations), multi_csv.name)

        # Build fc_ref lookup from training data
        fc_lookup = build_fc_lookup(processed_dir)

        # Determine target substrates
        if args.substrate:
            target_substrates = [args.substrate]
            # Validate substrate exists
            all_known = set(embeddings["substrate_names"])
            if args.substrate not in all_known:
                raise ValueError(
                    f"Substrate '{args.substrate}' not found. "
                    f"Known substrates: {sorted(all_known)}. "
                    f"Use --substrate-smiles to add new substrates."
                )
            logger.info("Mode: single-substrate (%s)", args.substrate)
        else:
            target_substrates = list(embeddings["substrate_names"])
            logger.info("Mode: all-substrates (%d substrates)", len(target_substrates))

        df_pairwise, report = build_inference_df(
            mutations, target_substrates, substrate_meta,
            fc_lookup, config, position_offset, esm2_wt_len, esm2_mut_keys)

    # ── 10. Report coverage ──
    if report["skipped_no_esm"]:
        logger.warning(
            "Skipped %d mutations with missing ESM2 embeddings: %s",
            len(report["skipped_no_esm"]),
            report["skipped_no_esm"][:10],
        )
        if len(report["skipped_no_esm"]) > 10:
            logger.warning("  ... and %d more", len(report["skipped_no_esm"]) - 10)

    if report["skipped_no_refs"]:
        logger.warning(
            "Skipped %d (mutation, substrate) pairs with no reference data: %s",
            len(report["skipped_no_refs"]),
            report["skipped_no_refs"][:10],
        )
        if len(report["skipped_no_refs"]) > 10:
            logger.warning("  ... and %d more", len(report["skipped_no_refs"]) - 10)

    if len(df_pairwise) == 0:
        logger.error(
            "No valid prediction rows after filtering. "
            "Check that mutations have ESM2 embeddings and reference "
            "substrate data in training."
        )
        sys.exit(1)

    ref_count_vals = list(report["ref_counts"].values())
    logger.info(
        "Prediction summary: %d unique (mutation, substrate) pairs, "
        "%d pairwise rows, refs per prediction: min=%d, max=%d, mean=%.1f",
        report["n_predictions"], report["n_pairwise_rows"],
        min(ref_count_vals), max(ref_count_vals), np.mean(ref_count_vals),
    )

    # ── 11. Build features ──
    logger.info("Building features...")
    use_bnn1 = params.get("features", {}).get("x_aa", False)
    # Align params["features"] with what the checkpoint actually contains.
    # hyperparams.json can be stale (Optuna search flags collapsed to defaults);
    # metadata["other_feature_groups"] is the ground truth of what was trained.
    trained_groups = set(metadata.get("other_feature_groups", []))
    if trained_groups:
        feats = dict(params.get("features", {}))
        for g in ("fc_ref", "x_target_substrate", "x_ref_substrate",
                  "ref_distance", "saprot_zs", "esm_wt", "esm_mut"):
            feats[g] = g in trained_groups
        params["features"] = feats
        logger.info("Aligned params['features'] to checkpoint groups: %s",
                    sorted(trained_groups))
    groups = build_other_features(df_pairwise, embeddings, params, substrate_meta)
    X_other = apply_saved_preprocessing(groups, other_pipelines)
    if use_bnn1:
        X_bnn1 = build_bnn1_input(df_pairwise, embeddings, bnn1_pipe_wt, bnn1_pipe_mut)
        X = np.concatenate([X_bnn1, X_other], axis=1).astype(np.float32)
        logger.info("Feature matrix: %s (BNN1: %d, other: %d)",
                    X.shape, X_bnn1.shape[1], X_other.shape[1])
    else:
        X = X_other.astype(np.float32)
        logger.info("Feature matrix: %s (BNN1 disabled, other: %d)",
                    X.shape, X_other.shape[1])

    # Validate dimensions match model. Note: metadata["bnn1_input_dim"] is
    # saved unconditionally by train_final_model even when BNN1 is disabled,
    # so only include it when use_bnn1.
    if use_bnn1:
        expected_dim = metadata["bnn1_input_dim"] + metadata["other_feature_dim"]
    else:
        expected_dim = metadata["other_feature_dim"]
    if X.shape[1] != expected_dim:
        raise ValueError(
            f"Feature dimension mismatch: built {X.shape[1]}, "
            f"model expects {expected_dim} "
            f"(bnn1={'on' if use_bnn1 else 'off'}, "
            f"other={metadata['other_feature_dim']})"
        )

    # ── 12. Run inference ──
    logger.info("Running MC inference (%d samples, batch_size=%d)...",
                n_samples, args.batch_size)
    estimates = predict_batched(model, X, n_samples, args.batch_size, device)

    y_pred = estimates.mean.numpy().squeeze(-1)
    epi_std = estimates.epistemic_std.numpy().squeeze(-1)
    ale_std = estimates.aleatoric_std.numpy().squeeze(-1)
    tot_std = estimates.total_std.numpy().squeeze(-1)

    # ── 13. Aggregate across references ──
    logger.info("Aggregating predictions across reference substrates...")
    _, _, _, _, agg_df = aggregate_pairwise_predictions(
        y_pred, epi_std, ale_std, tot_std, df_pairwise)

    # ── 14. Build output ──
    out_df = agg_df[[
        "mutation_string", "substrate", "position", "wt_aa", "mut_aa",
        "_y_pred", "_epi_std", "_ale_std", "_tot_std", "n_refs",
    ]].copy()
    out_df.rename(columns={
        "_y_pred": "predicted_log_fc",
        "_epi_std": "epistemic_std",
        "_ale_std": "aleatoric_std",
        "_tot_std": "total_std",
    }, inplace=True)

    # ── 14b. Null model baselines ──
    logger.info("Computing null model baselines...")
    df_train = load_multi_substrate_data(processed_dir)
    out_df["null_formaldehyde"] = compute_formaldehyde_baseline(out_df, df_train)
    out_df["null_nearest_substrate"] = compute_nearest_substrate_baseline(
        out_df, df_train, embeddings, embedding_type, args.null_distance_metric)

    # ── 15. Compute acquisition function scores ──
    logger.info("Computing acquisition function scores...")
    out_df = compute_acquisition_scores(out_df, n_thompson=args.n_thompson)

    # Sort for readability
    out_df = out_df.sort_values(
        ["substrate", "position", "mutation_string"]
    ).reset_index(drop=True)

    # Write output
    output_path = output_dir / "predictions.csv"
    out_df.to_csv(output_path, index=False)

    # ── 16. Generate plots ──
    if not args.no_plots:
        plot_dir = output_dir / "plots"
        generate_all_plots(out_df, plot_dir, position_offset)

    # ── 17. Summary ──
    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("Inference complete (%.1fs)", elapsed)
    logger.info("=" * 60)
    logger.info("Predictions: %d (mutation, substrate) pairs", len(out_df))
    logger.info("  predicted_log_fc: mean=%.4f, std=%.4f",
                out_df["predicted_log_fc"].mean(), out_df["predicted_log_fc"].std())
    logger.info("  total_std:        mean=%.4f, std=%.4f",
                out_df["total_std"].mean(), out_df["total_std"].std())
    logger.info("  acq_ucb2:         mean=%.4f, std=%.4f",
                out_df["acq_ucb2"].mean(), out_df["acq_ucb2"].std())
    logger.info("  refs per prediction: min=%d, max=%d",
                out_df["n_refs"].min(), out_df["n_refs"].max())
    logger.info("Output dir: %s", output_dir)
    logger.info("  CSV:    %s", output_path)
    if not args.no_plots:
        logger.info("  Plots:  %s", output_dir / "plots")

    # ── 18. Per-acquisition top-5 summary ──
    for acq in ACQ_NAMES:
        logger.info("--- Top 5 by %s ---", ACQ_LABELS[acq])
        for sub in sorted(out_df["substrate"].unique()):
            sub_top = out_df[out_df["substrate"] == sub].nlargest(5, acq)
            muts = ", ".join(
                f"{r['mutation_string']}({r[acq]:.3f})"
                for _, r in sub_top.iterrows()
            )
            logger.info("  %s: %s", sub, muts)


if __name__ == "__main__":
    main()
