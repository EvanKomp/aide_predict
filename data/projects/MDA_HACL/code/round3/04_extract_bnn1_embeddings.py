#!/usr/bin/env python
"""
04_extract_bnn1_embeddings.py — Extract Learned Latent Embeddings from BNN1
============================================================================

After BNN1 (formaldehyde regression) is trained in script 03, this script:
  1. Loads the final BNN1 model and preprocessing pipelines
  2. Forward-passes all unique mutations through the hidden layers
  3. Extracts penultimate hidden layer activations as learned AA-position embeddings
  4. Saves as bnn1_latent.npz for use as X_aa features in BNN2

The latent embeddings capture position-aware, activity-informed representations
of amino acid substitutions, learned from the formaldehyde regression task.

MC averaging over multiple stochastic forward passes yields the posterior mean
representation, reducing noise from weight sampling and dropout. Optionally,
the std across MC samples is saved as a measure of embedding uncertainty.

Outputs to processed/embeddings/:
  - bnn1_latent.npz — keyed by mutation_string, each entry (latent_dim,)
  - bnn1_latent_std.npz — MC std per mutation (embedding uncertainty)
  - bnn1_latent_meta.json — metadata (model path, n_samples, latent_dim, etc.)

Usage:
    python 04_extract_bnn1_embeddings.py
    python 04_extract_bnn1_embeddings.py --n-samples 200
    python 04_extract_bnn1_embeddings.py --device cuda:1
    python 04_extract_bnn1_embeddings.py --model-dir results/03_formaldehyde_regression/models
"""

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
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
from bnns import BayesianLinear, BayesianMLP, BNNTrainer, TrainingConfig


# ---------------------------------------------------------------------------
# Config helpers (same as 02/03)
# ---------------------------------------------------------------------------

def load_config(config_path=None) -> dict:
    path = Path(config_path) if config_path else SCRIPT_DIR / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config from %s", path)
    return config


def resolve_param(config_value, cli_override=None):
    if cli_override is not None:
        return cli_override
    if isinstance(config_value, dict):
        if "value" in config_value:
            return config_value["value"]
        if "search" in config_value:
            return config_value["search"][0]
    return config_value


def get_device(config: dict, override=None) -> str:
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
# Load BNN1 model from checkpoint
# ---------------------------------------------------------------------------

def load_bnn1_model(
    model_dir: Path,
    device: str,
    hyperparams: dict = None,
) -> BayesianMLP:
    """Reconstruct BNN1 architecture and load trained weights.

    The checkpoint (final_model.pt) contains model_state_dict and TrainingConfig
    but NOT the model architecture params. Those come from hyperparams.json
    (saved by script 03) or from `hyperparams` dict if provided.

    Args:
        model_dir: Directory containing final_model.pt and hyperparams.json
        device: Device string
        hyperparams: Optional dict with hidden_dims, prior_std, etc.
            If None, loads from hyperparams.json in model_dir's parent.

    Returns:
        BayesianMLP with trained weights loaded.
    """
    model_path = model_dir / "final_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run script 03 first."
        )

    # Load hyperparams to reconstruct architecture
    if hyperparams is None:
        hp_path = model_dir.parent / "hyperparams.json"
        if not hp_path.exists():
            raise FileNotFoundError(
                f"No hyperparams.json found at {hp_path}. "
                "Need architecture params to reconstruct model."
            )
        with open(hp_path) as f:
            hyperparams = json.load(f)
        logger.info("Loaded hyperparams from %s", hp_path)

    # Infer input_dim from the first layer's weight shape in the checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    # Find the first BayesianLinear layer's weight_mu to get input_dim
    first_weight = state_dict["hidden.0.weight_mu"]
    input_dim = first_weight.shape[1]
    logger.info("Inferred input_dim=%d from checkpoint", input_dim)

    # Reconstruct model
    model = BayesianMLP(
        input_dim=input_dim,
        hidden_dims=hyperparams["hidden_dims"],
        output_dim=1,
        task="regression",
        prior_std=hyperparams["prior_std"],
        dropout_rate=hyperparams["dropout_rate"],
        activation=hyperparams.get("activation", "silu"),
    )

    model.load_state_dict(state_dict)
    model.to(device)
    logger.info("Loaded BNN1 model: input=%d, hidden=%s, latent=%d",
                input_dim, hyperparams["hidden_dims"],
                hyperparams["hidden_dims"][-1])

    return model


# ---------------------------------------------------------------------------
# Load preprocessing pipelines
# ---------------------------------------------------------------------------

def load_preprocessing(model_dir: Path):
    """Load fitted WT and mutant preprocessing pipelines from script 03.

    Returns:
        pipe_wt: fitted sklearn Pipeline (or None)
        pipe_mut: fitted sklearn Pipeline (or None)
    """
    pipe_wt_path = model_dir / "preprocessing_wt.joblib"
    pipe_mut_path = model_dir / "preprocessing_mut.joblib"

    if not pipe_wt_path.exists() or not pipe_mut_path.exists():
        raise FileNotFoundError(
            f"Preprocessing pipelines not found in {model_dir}. Run script 03 first."
        )

    pipe_wt = joblib.load(pipe_wt_path)
    pipe_mut = joblib.load(pipe_mut_path)
    logger.info("Loaded preprocessing: wt=%s, mut=%s",
                type(pipe_wt).__name__, type(pipe_mut).__name__)
    return pipe_wt, pipe_mut


# ---------------------------------------------------------------------------
# Load all unique mutations
# ---------------------------------------------------------------------------

def load_all_mutations(
    processed_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    """Load ESM2 embeddings for ALL unique mutations across datasets.

    Combines formaldehyde_ssm.csv and multi_substrate_ssm.csv to get every
    unique mutation_string with its 0-indexed position, then fetches
    corresponding WT and mutant ESM2 embeddings.

    Returns:
        X_wt: (n_unique, 1280) WT residue embeddings
        X_mut: (n_unique, 1280) mutant residue embeddings
        mutation_strings: list of mutation_string identifiers
        positions: list of 0-indexed positions
    """
    import pandas as pd

    # Build mutation_string → position mapping from both datasets
    ms_to_pos = {}

    form_path = processed_dir / "formaldehyde_ssm.csv"
    if form_path.exists():
        form_df = pd.read_csv(form_path)
        for _, row in form_df.iterrows():
            ms_to_pos[row["mutation_string"]] = int(row["position"])
        logger.info("Formaldehyde SSM: %d mutations at %d positions",
                    len(form_df), form_df["position"].nunique())

    ms_path = processed_dir / "multi_substrate_ssm.csv"
    if ms_path.exists():
        ms_df = pd.read_csv(ms_path)
        # Multi-substrate has rows per (mutation, substrate) — deduplicate
        for _, row in ms_df.drop_duplicates("mutation_string").iterrows():
            if row["mutation_string"] not in ms_to_pos:
                ms_to_pos[row["mutation_string"]] = int(row["position"])
        logger.info("Multi-substrate SSM: added %d new mutations",
                    len(ms_to_pos) - len(form_df) if form_path.exists() else len(ms_to_pos))

    logger.info("Total unique mutations: %d", len(ms_to_pos))

    # Load ESM2 embeddings
    emb_dir = processed_dir / "embeddings"
    esm_wt = np.load(emb_dir / "esm2_wt_residues.npy")       # (565, 1280)
    esm_mutant = np.load(emb_dir / "esm2_mutant_residues.npz")
    logger.info("ESM2 WT residues: %s", esm_wt.shape)
    logger.info("ESM2 mutant embeddings: %d entries", len(esm_mutant.files))

    # Build feature arrays for all unique mutations
    wt_list = []
    mut_list = []
    mutation_strings = []
    positions = []

    for ms, pos in sorted(ms_to_pos.items()):
        if ms not in esm_mutant:
            warnings.warn(f"Missing ESM2 mutant embedding for {ms}, skipping")
            continue
        wt_list.append(esm_wt[pos])
        mut_list.append(esm_mutant[ms])
        mutation_strings.append(ms)
        positions.append(pos)

    X_wt = np.stack(wt_list).astype(np.float32)
    X_mut = np.stack(mut_list).astype(np.float32)

    logger.info("Loaded features for %d mutations: WT %s, Mut %s",
                len(mutation_strings), X_wt.shape, X_mut.shape)

    return X_wt, X_mut, mutation_strings, positions


# ---------------------------------------------------------------------------
# Extract latent embeddings
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_latents(
    model: BayesianMLP,
    X: torch.Tensor,
    n_samples: int = 100,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract penultimate hidden layer activations via MC averaging.

    Runs X through the hidden layers only (not the output head) for n_samples
    stochastic forward passes. Returns the mean and std of activations across
    samples.

    Dropout is kept active during extraction (model.train()) so that the MC
    average integrates over both weight uncertainty and dropout masks, matching
    inference behavior.

    Args:
        model: Trained BayesianMLP (already on device).
        X: Input tensor (n, input_dim) on same device.
        n_samples: MC forward passes to average.
        batch_size: Process this many mutations at once (memory management).

    Returns:
        latent_mean: (n, latent_dim) posterior mean latent embedding
        latent_std: (n, latent_dim) std across MC samples (embedding uncertainty)
    """
    model.train()  # keep dropout active for MC sampling
    device = next(model.parameters()).device
    n = X.shape[0]

    # Determine latent dim from the last hidden layer
    # hidden_dims[-1] = last hidden layer width = latent dim
    latent_dim = model.hidden[-1 if not hasattr(model.hidden[-1], 'out_features')
                              else -1].out_features if hasattr(model.hidden[-1], 'out_features') else None

    # Safer: just run one sample to get the dim
    with torch.no_grad():
        x_probe = X[:1].to(device)
        h = x_probe
        for layer in model.hidden:
            if isinstance(layer, BayesianLinear):
                h, _ = layer(h)
            else:
                h = layer(h)
        latent_dim = h.shape[-1]

    logger.info("Latent dimension: %d", latent_dim)

    # Accumulate running mean and M2 (Welford's online algorithm) for memory efficiency
    latent_sum = torch.zeros(n, latent_dim, device=device)
    latent_sq_sum = torch.zeros(n, latent_dim, device=device)

    for s in range(n_samples):
        # Process in batches
        sample_latents = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x_batch = X[start:end].to(device)

            h = x_batch
            for layer in model.hidden:
                if isinstance(layer, BayesianLinear):
                    h, _ = layer(h)
                else:
                    h = layer(h)
            sample_latents.append(h)

        sample_all = torch.cat(sample_latents, dim=0)  # (n, latent_dim)
        latent_sum += sample_all
        latent_sq_sum += sample_all ** 2

        if (s + 1) % 50 == 0 or s == 0:
            logger.info("  MC sample %d/%d", s + 1, n_samples)

    # Compute mean and std
    latent_mean = latent_sum / n_samples
    latent_var = (latent_sq_sum / n_samples) - (latent_mean ** 2)
    latent_std = torch.sqrt(latent_var.clamp(min=0.0))

    return latent_mean.cpu().numpy(), latent_std.cpu().numpy()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract BNN1 latent embeddings for all mutations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 04_extract_bnn1_embeddings.py
  python 04_extract_bnn1_embeddings.py --n-samples 200
  python 04_extract_bnn1_embeddings.py --device cuda:1
  python 04_extract_bnn1_embeddings.py --model-dir results/03_formaldehyde_regression/models
        """,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (cpu, cuda, cuda:1, mps, auto)")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Directory with final_model.pt + preprocessing .joblib files. "
                             "Default: results/03_formaldehyde_regression/models")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="MC samples for latent extraction. Default: config n_inference_samples")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size for forward passes (memory). Default: 512")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    log_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(log_fmt)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)


def main():
    args = parse_args()
    t_start = time.time()
    setup_logging()

    logger.info("=" * 60)
    logger.info("04_extract_bnn1_embeddings.py")
    logger.info("Extract BNN1 Latent Embeddings")
    logger.info("=" * 60)

    # 1. Load config and resolve paths
    config = load_config(args.config)
    device = get_device(config, args.device)

    model_dir = (
        Path(args.model_dir) if args.model_dir
        else PROJECT_ROOT / "results" / "03_formaldehyde_regression" / "models"
    )
    processed_dir = PROJECT_ROOT / config["data"]["output_dir"]
    output_dir = processed_dir / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_samples = args.n_samples or resolve_param(
        config["bnn1"]["training"]["n_inference_samples"]
    )
    logger.info("Model directory:  %s", model_dir)
    logger.info("Output directory: %s", output_dir)
    logger.info("MC samples:       %d", n_samples)

    # 2. Load trained BNN1
    model = load_bnn1_model(model_dir, device)

    # Log model summary
    param_counts = model.trainable_parameter_count()
    logger.info("Model parameters: %s", param_counts)
    posterior = model.posterior_summary()
    for name, stats in posterior.items():
        logger.info("  %s: mean_abs_mu=%.4f, mean_std=%.4f",
                    name, stats["mean_abs_mu"], stats["mean_std"])

    # 3. Load preprocessing pipelines
    pipe_wt, pipe_mut = load_preprocessing(model_dir)

    # 4. Load all unique mutations
    X_wt_raw, X_mut_raw, mutation_strings, positions = load_all_mutations(processed_dir)

    # 5. Preprocess (transform only — pipelines already fitted on full data in script 03)
    if pipe_wt is not None:
        X_wt_p = pipe_wt.transform(X_wt_raw).astype(np.float32)
    else:
        X_wt_p = X_wt_raw.copy()

    if pipe_mut is not None:
        X_mut_p = pipe_mut.transform(X_mut_raw).astype(np.float32)
    else:
        X_mut_p = X_mut_raw.copy()

    X_all = np.concatenate([X_wt_p, X_mut_p], axis=1)
    logger.info("Preprocessed features: WT %s + Mut %s = %s",
                X_wt_p.shape, X_mut_p.shape, X_all.shape)

    # 6. Extract latent embeddings
    logger.info("Extracting latent embeddings (%d samples, %d mutations)...",
                n_samples, len(mutation_strings))
    X_tensor = torch.tensor(X_all, dtype=torch.float32)

    latent_mean, latent_std = extract_latents(
        model, X_tensor,
        n_samples=n_samples,
        batch_size=args.batch_size,
    )

    logger.info("Latent embeddings: %s (mean norm=%.3f)",
                latent_mean.shape, np.linalg.norm(latent_mean, axis=1).mean())
    logger.info("Latent std: mean=%.4f, max=%.4f",
                latent_std.mean(), latent_std.max())

    # 7. Save as .npz keyed by mutation_string
    latent_dict = {ms: latent_mean[i] for i, ms in enumerate(mutation_strings)}
    latent_std_dict = {ms: latent_std[i] for i, ms in enumerate(mutation_strings)}

    np.savez(output_dir / "bnn1_latent.npz", **latent_dict)
    np.savez(output_dir / "bnn1_latent_std.npz", **latent_std_dict)
    logger.info("Saved bnn1_latent.npz (%d entries, %d-dim each)",
                len(latent_dict), latent_mean.shape[1])
    logger.info("Saved bnn1_latent_std.npz (%d entries)", len(latent_std_dict))

    # 8. Save metadata
    meta = {
        "model_dir": str(model_dir),
        "n_mutations": len(mutation_strings),
        "latent_dim": int(latent_mean.shape[1]),
        "n_mc_samples": n_samples,
        "latent_mean_norm": float(np.linalg.norm(latent_mean, axis=1).mean()),
        "latent_mean_std": float(latent_mean.std()),
        "latent_uncertainty_mean": float(latent_std.mean()),
        "mutations": mutation_strings,
        "positions": positions,
    }
    with open(output_dir / "bnn1_latent_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved bnn1_latent_meta.json")

    # 9. Summary
    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("BNN1 Latent Extraction Complete (%.1fs)", elapsed)
    logger.info("=" * 60)
    logger.info("Mutations:   %d", len(mutation_strings))
    logger.info("Latent dim:  %d", latent_mean.shape[1])
    logger.info("MC samples:  %d", n_samples)
    logger.info("Output:      %s/bnn1_latent.npz", output_dir)


if __name__ == "__main__":
    main()
