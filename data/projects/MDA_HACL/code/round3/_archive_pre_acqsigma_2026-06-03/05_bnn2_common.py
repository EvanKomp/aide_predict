"""
05_bnn2_common.py — Shared Utilities for BNN2 Multi-Substrate Prediction
=========================================================================

Provides the composite BNN2 model (BNN1 backbone + BNN2 head), data loading,
pairwise data expansion, feature assembly, preprocessing, training, metrics,
and plotting functions shared by 05_bnn2_multi_substrate.py and opt_05_bnn2.py.

Architecture:
    Raw ESM features ──→ [BNN1 backbone] ──→ latent
                                                 ↓
    Other features ─────────────────────→ concat ──→ [BNN2 head] ──→ mu, log_var
    (fc_ref, substrate embeddings, saprot)

BNN1's hidden layers are loaded from a trained checkpoint (script 03) and
embedded directly into BNN2. This preserves BNN1's stochastic forward pass
(Bayesian weight sampling) so its epistemic uncertainty propagates into
BNN2's predictions. The x_aa_freeze parameter controls whether BNN1's
layers are frozen, partially trainable, or fully trainable.
"""

from __future__ import annotations

import copy
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy import stats
from scipy.ndimage import convolve1d, gaussian_filter1d

logger = logging.getLogger(__name__)

# Talk-ready matplotlib defaults — applied at import so every downstream
# plotting helper inherits bold lines, thick axes, and large fonts.
try:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _plot_style import apply_talk_style as _apply_talk_style
    _apply_talk_style()
except Exception as _e:
    logger.debug("apply_talk_style not applied: %s", _e)


# ═══════════════════════════════════════════════════════════════════════════
# Config helpers (reused from script 03)
# ═══════════════════════════════════════════════════════════════════════════

def load_config(config_path: Optional[str] = None) -> dict:
    """Load config.yaml."""
    SCRIPT_DIR = Path(__file__).resolve().parent
    path = Path(config_path) if config_path else SCRIPT_DIR / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config from %s", path)
    return config


def resolve_param(config_value, cli_override=None):
    """Resolve parameter: CLI > config value > first search value."""
    if cli_override is not None:
        return cli_override
    if isinstance(config_value, dict):
        if "value" in config_value:
            return config_value["value"]
        if "search" in config_value:
            return config_value["search"][0]
    return config_value


def resolve_config_block(block: dict) -> dict:
    """Resolve a config block: convert all value/search dicts to plain values.

    Handles nested dicts that use the {value: ..., search: ...} convention.
    Non-dict leaves and dicts without 'value' key are passed through as-is.
    """
    resolved = {}
    for k, v in block.items():
        if isinstance(v, dict) and "value" in v:
            resolved[k] = v["value"]
        elif isinstance(v, dict) and "search" in v and "value" not in v:
            # Only search key, no value — shouldn't happen but handle gracefully
            resolved[k] = v["search"][0] if v["search"] else None
        else:
            resolved[k] = v
    return resolved


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


def parse_pca_value(val):
    """Parse PCA components from string/int/float/None."""
    if val is None or val == "none" or val == "None" or val == "null":
        return None
    try:
        v = float(val)
        if v == int(v) and v > 1:
            return int(v)
        return v
    except (ValueError, TypeError):
        return None


def setup_logging(log_file: Path, level=logging.INFO):
    """Configure logging to file + console."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)
    # Remove existing handlers
    for h in root.handlers[:]:
        root.removeHandler(h)
    # File handler
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(fh)
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(ch)


# ═══════════════════════════════════════════════════════════════════════════
# BNN2 Composite Model
# ═══════════════════════════════════════════════════════════════════════════

# Import BNN components (deferred to allow flexible sys.path setup by callers)
def _import_bnns():
    """Import BNN components. Caller must ensure sys.path includes code/."""
    from bnns import BayesianMLP, BNNTrainer, TrainingConfig, TrainingHistory
    from bnns.layers import BayesianLinear
    from bnns.model import UncertaintyEstimate, HurdleUncertaintyEstimate
    from bnns.trainer import HurdleConfig
    return BayesianMLP, BNNTrainer, TrainingConfig, TrainingHistory, BayesianLinear, UncertaintyEstimate, HurdleUncertaintyEstimate, HurdleConfig


class BNN2Model(nn.Module):
    """Composite Bayesian model: BNN1 backbone + BNN2 head.

    The BNN1 backbone processes ESM2 WT+mutant features and produces a learned
    latent representation. This is concatenated with other features (fc_ref,
    substrate embeddings, SaProt) and fed through new BNN2 head layers.

    The model accepts a single input tensor where the first `bnn1_input_dim`
    columns are the BNN1 input (preprocessed ESM features) and the remaining
    columns are other features. This keeps the interface compatible with
    BNNTrainer which expects model(x) → (output, kl).

    Args:
        bnn1_hidden: nn.ModuleList of BNN1's hidden layers (BayesianLinear + act + dropout)
        bnn1_input_dim: Dimension of BNN1 input (preprocessed ESM wt+mut)
        latent_dim: Output dimension of BNN1 backbone (last hidden layer width)
        other_feature_dim: Dimension of non-ESM features
        hidden_dims: BNN2 head hidden layer widths
        output_dim: Number of outputs (1 for regression)
        task: 'regression' or 'classification'
        prior_std: Prior std for BNN2 head layers
        dropout_rate: Dropout between BNN2 head layers
        activation: Activation function name
        freeze_mode: 'full', 'partial', or 'none' for BNN1 backbone
    """

    ACTIVATIONS = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "silu": nn.SiLU,
    }

    def __init__(
        self,
        bnn1_hidden: Optional[nn.ModuleList],
        bnn1_input_dim: int,
        latent_dim: int,
        other_feature_dim: int,
        hidden_dims: list[int],
        output_dim: int = 1,
        task: str = "regression",
        prior_mu: float = 0.0,
        prior_std: float = 1.0,
        rho_init: float = -4.0,
        dropout_rate: float = 0.1,
        activation: str = "silu",
        bias: bool = True,
        freeze_mode: str = "full",
        hurdle: bool = False,
    ):
        super().__init__()
        _, _, _, _, BayesianLinear, _, _, _ = _import_bnns()

        self.bnn1_input_dim = bnn1_input_dim
        self.latent_dim = latent_dim
        self.other_feature_dim = other_feature_dim
        self.output_dim = output_dim
        self.task = task
        self.freeze_mode = freeze_mode
        self.hurdle = hurdle
        self.use_bnn1 = bnn1_hidden is not None

        # ---- BNN1 backbone (optional — only when x_aa feature is on) ----
        if self.use_bnn1:
            self.bnn1_hidden = bnn1_hidden

            # Apply freeze
            if freeze_mode == "full":
                for layer in self.bnn1_hidden:
                    for p in layer.parameters():
                        p.requires_grad = False
            elif freeze_mode == "partial":
                last_bl_idx = -1
                for i, layer in enumerate(self.bnn1_hidden):
                    if isinstance(layer, BayesianLinear):
                        last_bl_idx = i
                for i, layer in enumerate(self.bnn1_hidden):
                    if i < last_bl_idx:
                        for p in layer.parameters():
                            p.requires_grad = False

            # Make frozen BayesianLinear layers deterministic during training.
            for layer in self.bnn1_hidden:
                if isinstance(layer, BayesianLinear):
                    is_frozen = not any(p.requires_grad for p in layer.parameters())
                    layer.deterministic = is_frozen
        else:
            self.bnn1_hidden = nn.ModuleList()  # empty, no params

        # ---- BNN2 head layers ----
        layer_kwargs = dict(prior_mu=prior_mu, prior_std=prior_std,
                            rho_init=rho_init, bias=bias)
        act_cls = self.ACTIVATIONS.get(activation, nn.SiLU)

        combined_dim = (latent_dim if self.use_bnn1 else 0) + other_feature_dim
        head_layers = []
        dims = [combined_dim] + hidden_dims
        for i in range(len(dims) - 1):
            head_layers.append(BayesianLinear(dims[i], dims[i + 1], **layer_kwargs))
            head_layers.append(act_cls())
            if dropout_rate > 0.0:
                head_layers.append(nn.Dropout(p=dropout_rate))
        self.bnn2_hidden = nn.ModuleList(head_layers)

        # Output head: regression → 2*output_dim (mu + log_var)
        # Hurdle adds output_dim extra outputs for classification logit(s)
        head_out = 2 * output_dim if task == "regression" else output_dim
        if hurdle:
            head_out += output_dim  # (mu, log_var, cls_logit)
        self.head = BayesianLinear(dims[-1], head_out, **layer_kwargs)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: optional BNN1 backbone → concat → BNN2 head.

        Args:
            x: (batch, bnn1_input_dim + other_feature_dim) when use_bnn1=True,
               or (batch, other_feature_dim) when use_bnn1=False.

        Returns:
            output: (batch, 2*output_dim) for regression, (batch, output_dim) for classification
            total_kl: scalar KL divergence from all trainable BayesianLinear layers
        """
        _, _, _, _, BayesianLinear, _, _, _ = _import_bnns()

        total_kl = torch.tensor(0.0, device=x.device)

        if self.use_bnn1:
            x_bnn1 = x[:, :self.bnn1_input_dim]
            x_other = x[:, self.bnn1_input_dim:]

            # ---- BNN1 backbone ----
            h = x_bnn1
            for layer in self.bnn1_hidden:
                if isinstance(layer, BayesianLinear):
                    h, kl = layer(h)
                    if layer.weight_mu.requires_grad:
                        total_kl = total_kl + kl
                else:
                    h = layer(h)

            combined = torch.cat([h, x_other], dim=-1)
        else:
            combined = x

        # ---- BNN2 head ----
        for layer in self.bnn2_hidden:
            if isinstance(layer, BayesianLinear):
                combined, kl = layer(combined)
                total_kl = total_kl + kl
            else:
                combined = layer(combined)

        output, kl = self.head(combined)
        total_kl = total_kl + kl

        return output, total_kl

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 100,
        hurdle_config=None,
        fc_ref=None,
        prediction_floor: float = None,
    ):
        """MC-averaged prediction with decomposed uncertainty.

        When fc_ref is provided, the model is assumed to output deltas
        (y_pred = log_fc - log_fc_ref).  After MC sampling, fc_ref is added
        to the regression mean samples to reconstruct absolute predictions.
        Variance is unaffected (constant shift).

        When hurdle mode is active (hurdle_config provided), returns a
        HurdleUncertaintyEstimate with soft-mixture predictions:
            mean = p * reg_mean + (1 - p) * floor_value
        where p = mean(sigmoid(cls_logit)) across MC samples.

        Otherwise returns a standard UncertaintyEstimate.
        """
        _, _, _, _, BayesianLinear, UncertaintyEstimate, HurdleUncertaintyEstimate, _ = _import_bnns()

        self.train()  # keep dropout active for MC estimation

        # Temporarily re-enable stochastic sampling on frozen BNN1 layers
        # so MC integration captures full epistemic uncertainty from BNN1.
        _det_layers = []
        if self.use_bnn1:
            for layer in self.bnn1_hidden:
                if isinstance(layer, BayesianLinear) and layer.deterministic:
                    layer.deterministic = False
                    _det_layers.append(layer)

        try:
            raw_outputs = torch.stack([self(x)[0] for _ in range(n_samples)], dim=0)
        finally:
            for layer in _det_layers:
                layer.deterministic = True

        od = self.output_dim

        if self.hurdle and hurdle_config is not None:
            # ── Hurdle mode: split into regression + classification ──
            mu_samples = raw_outputs[..., :od]
            if fc_ref is not None:
                mu_samples = mu_samples + fc_ref.unsqueeze(0)
            if prediction_floor is not None:
                mu_samples = mu_samples.clamp(min=prediction_floor)
            log_var_samples = raw_outputs[..., od:2 * od].clamp(-10.0, 10.0)
            var_samples = torch.exp(log_var_samples)
            cls_logit_samples = raw_outputs[..., 2 * od:]  # (S, B, od)
            cls_prob_samples = torch.sigmoid(cls_logit_samples)  # (S, B, od)

            # Regression uncertainty (raw, from all MC samples)
            reg_mean = mu_samples.mean(dim=0)  # (B, od)
            reg_epi_var = mu_samples.var(dim=0, unbiased=True)
            reg_ale_var = var_samples.mean(dim=0)
            reg_total_var = reg_epi_var + reg_ale_var

            # Classification uncertainty
            cls_prob = cls_prob_samples.mean(dim=0)  # (B, od)
            cls_prob_std = cls_prob_samples.std(dim=0, unbiased=True)

            # Soft mixture: mean = p * reg_mean + (1-p) * floor
            floor = hurdle_config.floor_value
            p = cls_prob  # (B, od)
            mix_mean = p * reg_mean + (1.0 - p) * floor

            # Mixture variance: E[X^2] - E[X]^2
            # E[X^2] = p * (reg_var + reg_mean^2) + (1-p) * floor^2
            mix_var = (
                p * (reg_total_var + reg_mean ** 2)
                + (1.0 - p) * (floor ** 2)
                - mix_mean ** 2
            ).clamp(min=0.0)

            return HurdleUncertaintyEstimate(
                cls_prob=cls_prob,
                cls_prob_std=cls_prob_std,
                reg_mean=reg_mean,
                reg_epistemic_std=reg_epi_var.sqrt(),
                reg_aleatoric_std=reg_ale_var.sqrt(),
                reg_total_std=reg_total_var.sqrt(),
                reg_samples=mu_samples,
                mean=mix_mean,
                epistemic_std=reg_epi_var.sqrt() * p,  # scale by p
                aleatoric_std=reg_ale_var.sqrt() * p,
                total_std=mix_var.sqrt(),
                samples=mu_samples,
            )

        # ── Standard regression ──
        mu_samples = raw_outputs[..., :od]
        if fc_ref is not None:
            mu_samples = mu_samples + fc_ref.unsqueeze(0)
        if prediction_floor is not None:
            mu_samples = mu_samples.clamp(min=prediction_floor)
        log_var_samples = raw_outputs[..., od:2 * od].clamp(-10.0, 10.0)
        var_samples = torch.exp(log_var_samples)

        mean = mu_samples.mean(dim=0)
        epistemic_var = mu_samples.var(dim=0, unbiased=True)
        aleatoric_var = var_samples.mean(dim=0)
        total_var = epistemic_var + aleatoric_var

        return UncertaintyEstimate(
            mean=mean,
            epistemic_std=epistemic_var.sqrt(),
            aleatoric_std=aleatoric_var.sqrt(),
            total_std=total_var.sqrt(),
            samples=mu_samples,
        )

    def trainable_parameter_count(self) -> dict[str, int]:
        """Count trainable vs frozen parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}

    def posterior_summary(self) -> dict:
        """Mean posterior std for trainable BayesianLinear layers (BNN2 head only).

        A ratio of mean_std / prior_std near 1.0 means the posterior has not moved
        from initialisation (no learning about weight uncertainty from data).
        """
        _, _, _, _, BayesianLinear, _, _, _ = _import_bnns()
        import torch.nn.functional as _F
        summary = {}
        for name, module in self.named_modules():
            if isinstance(module, BayesianLinear) and any(
                p.requires_grad for p in module.parameters()
            ):
                std = _F.softplus(module.weight_rho).detach()
                summary[name] = {
                    "mean_abs_mu": module.weight_mu.detach().abs().mean().item(),
                    "mean_std": std.mean().item(),
                }
        return summary


# ═══════════════════════════════════════════════════════════════════════════
# BNN1 Backbone Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_bnn1_backbone(
    model_dir: Path,
    device: str,
) -> Tuple[nn.ModuleList, int, int, dict]:
    """Load BNN1 model and extract its hidden layers as a backbone.

    Args:
        model_dir: Directory containing final_model.pt and hyperparams.json
        device: Device string

    Returns:
        bnn1_hidden: nn.ModuleList of BNN1's hidden layers
        bnn1_input_dim: Input dimension of BNN1
        latent_dim: Output dimension of BNN1's last hidden layer
        hyperparams: BNN1 hyperparameters dict
    """
    BayesianMLP, _, _, _, _, _, _, _ = _import_bnns()

    model_path = model_dir / "final_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained BNN1 model at {model_path}. Run script 03 first."
        )

    # Load hyperparams
    hp_path = model_dir.parent / "hyperparams.json"
    if not hp_path.exists():
        raise FileNotFoundError(f"No hyperparams.json at {hp_path}")
    with open(hp_path) as f:
        hyperparams = json.load(f)
    logger.info("BNN1 hyperparams: hidden=%s, prior_std=%.2f",
                hyperparams["hidden_dims"], hyperparams["prior_std"])

    # Infer input_dim from checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    first_weight = state_dict["hidden.0.weight_mu"]
    bnn1_input_dim = first_weight.shape[1]

    # Reconstruct full model to get properly initialized hidden layers
    model = BayesianMLP(
        input_dim=bnn1_input_dim,
        hidden_dims=hyperparams["hidden_dims"],
        output_dim=1,
        task="regression",
        prior_std=hyperparams["prior_std"],
        dropout_rate=hyperparams["dropout_rate"],
        activation=hyperparams.get("activation", "silu"),
    )
    model.load_state_dict(state_dict)
    model.to(device)

    # Extract hidden layers (backbone only, not the output head)
    bnn1_hidden = model.hidden
    latent_dim = hyperparams["hidden_dims"][-1]

    logger.info("Loaded BNN1 backbone: input=%d, hidden=%s, latent=%d",
                bnn1_input_dim, hyperparams["hidden_dims"], latent_dim)

    return bnn1_hidden, bnn1_input_dim, latent_dim, hyperparams


def load_bnn1_preprocessing(model_dir: Path):
    """Load BNN1's fitted preprocessing pipelines (transform-only for BNN2).

    Returns:
        pipe_wt: fitted sklearn Pipeline (or None)
        pipe_mut: fitted sklearn Pipeline (or None)
    """
    pipe_wt_path = model_dir / "preprocessing_wt.joblib"
    pipe_mut_path = model_dir / "preprocessing_mut.joblib"
    if not pipe_wt_path.exists() or not pipe_mut_path.exists():
        raise FileNotFoundError(
            f"BNN1 preprocessing pipelines not found in {model_dir}. Run script 03 first."
        )
    pipe_wt = joblib.load(pipe_wt_path)
    pipe_mut = joblib.load(pipe_mut_path)
    logger.info("Loaded BNN1 preprocessing: wt=%s, mut=%s",
                type(pipe_wt).__name__, type(pipe_mut).__name__)
    return pipe_wt, pipe_mut


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_multi_substrate_data(processed_dir: Path) -> pd.DataFrame:
    """Load multi-substrate SSM data (1,798 rows)."""
    path = processed_dir / "multi_substrate_ssm.csv"
    df = pd.read_csv(path)
    logger.info("Loaded multi-substrate data: %d rows, %d substrates, %d positions",
                len(df), df["substrate"].nunique(), df["position"].nunique())
    return df


def get_supplemental_positions(df: pd.DataFrame) -> set:
    """Return set of 0-indexed positions that came from supplemental data."""
    if "is_supplemental" not in df.columns:
        return set()
    return set(df.loc[df["is_supplemental"], "position"].unique())


def _style_supp_ticklabels(ax, supp_positions, axis="x"):
    """Bold + red tick labels for supplemental positions."""
    if not supp_positions:
        return
    getter = ax.get_xticklabels if axis == "x" else ax.get_yticklabels
    for label in getter():
        text = label.get_text()
        digits = "".join(c for c in text if c.isdigit())
        if digits and int(digits) in supp_positions:
            label.set_color("#d62728")
            label.set_fontweight("bold")


def load_all_embeddings(processed_dir: Path) -> dict:
    """Load all precomputed embeddings needed for BNN2.

    Returns dict with keys:
        esm2_wt, esm2_v83p, esm2_mut, saprot,
        substrate_morgan, substrate_maccs, substrate_mordred, substrate_molformer,
        substrate_names
    """
    emb_dir = processed_dir / "embeddings"
    embeddings = {}

    # ESM2
    embeddings["esm2_wt"] = np.load(emb_dir / "esm2_wt_residues.npy")
    embeddings["esm2_v83p"] = np.load(emb_dir / "esm2_v83p_residues.npy")
    embeddings["esm2_mut"] = dict(np.load(emb_dir / "esm2_mutant_residues.npz"))

    # SaProt
    with open(emb_dir / "saprot_scores.json") as f:
        embeddings["saprot"] = json.load(f)

    # Substrate embeddings
    for emb_type in ["morgan", "maccs", "mordred", "molformer"]:
        embeddings[f"substrate_{emb_type}"] = np.load(
            emb_dir / f"substrate_{emb_type}.npy"
        )

    # Substrate name ordering (matches row order of substrate embedding arrays)
    with open(emb_dir / "substrate_names.json") as f:
        embeddings["substrate_names"] = json.load(f)

    logger.info("Loaded embeddings: ESM2 wt=%s, mutants=%d, substrates=%d",
                embeddings["esm2_wt"].shape, len(embeddings["esm2_mut"]),
                len(embeddings["substrate_names"]))

    return embeddings


def load_substrate_metadata(processed_dir: Path) -> dict:
    """Load substrate metadata (SMILES, is_active, etc.)."""
    path = processed_dir / "substrate_metadata.json"
    with open(path) as f:
        meta = json.load(f)
    logger.info("Loaded substrate metadata: %d substrates", len(meta))
    return meta


# ═══════════════════════════════════════════════════════════════════════════
# Pairwise Data Expansion
# ═══════════════════════════════════════════════════════════════════════════

def expand_to_pairwise(
    df: pd.DataFrame,
    substrate_meta: dict,
    config: dict,
    ref_fc_lookup: Optional[dict] = None,
) -> pd.DataFrame:
    """Expand base data to (mutation, ref_substrate, target_substrate) triplets.

    For each row (a mutation on a target substrate), creates one triplet per
    valid reference substrate. The reference substrate's fold_change for the
    same mutation is looked up from the base data.

    Args:
        df: Base multi-substrate DataFrame (1,798 rows)
        substrate_meta: Substrate metadata dict
        config: Config dict (for pairwise settings)
        ref_fc_lookup: Optional external FC lookup dict
            {(mutation_string, substrate) → fold_change} for reference
            substrates. When provided (e.g. for validation in substrate-split
            CV), reference FC values are taken from this dict instead of from
            df itself, allowing references that are absent from df (the held-out
            split) to be resolved from training data.

    Returns:
        Expanded DataFrame with additional columns:
            ref_substrate: name of reference substrate
            fc_ref: fold_change of this mutation on the reference substrate
            log_fc_ref: log10(fc_ref + epsilon) for delta target computation
            base_idx: index into original df for tracking
    """
    pairwise_cfg = resolve_config_block(config.get("bnn2", {}).get("pairwise", {}))
    ref_substrates_mode = pairwise_cfg.get("ref_substrates", "active_only")
    exclude_self = pairwise_cfg.get("exclude_self_ref", True)
    epsilon = config.get("data", {}).get("epsilon", 0.01)

    # Determine valid reference substrates
    if ref_substrates_mode == "active_only":
        valid_refs = [name for name, meta in substrate_meta.items() if meta["is_active"]]
    else:
        valid_refs = list(substrate_meta.keys())

    # Build FC lookup from df itself; caller may supply an external one for refs
    self_fc_lookup = {}
    for _, row in df.iterrows():
        self_fc_lookup[(row["mutation_string"], row["substrate"])] = row["fold_change"]

    # Use external ref lookup if provided, otherwise fall back to self
    lookup = ref_fc_lookup if ref_fc_lookup is not None else self_fc_lookup

    expanded_rows = []
    for base_idx, row in df.iterrows():
        target_sub = row["substrate"]
        mut_str = row["mutation_string"]

        for ref_sub in valid_refs:
            if exclude_self and ref_sub == target_sub:
                continue

            # Look up FC on reference substrate
            fc_ref = lookup.get((mut_str, ref_sub))
            if fc_ref is None:
                continue  # mutation not measured on this ref substrate

            expanded_rows.append({
                **row.to_dict(),
                "ref_substrate": ref_sub,
                "fc_ref": fc_ref,
                "log_fc_ref": float(np.log10(fc_ref + epsilon)),
                "base_idx": base_idx,
            })

    expanded_df = pd.DataFrame(expanded_rows)
    logger.info("Pairwise expansion: %d base rows → %d triplets (%.1fx)",
                len(df), len(expanded_df), len(expanded_df) / len(df))

    return expanded_df


def add_ref_distances(
    expanded_df: pd.DataFrame,
    distance_matrix: np.ndarray,
    substrate_names: list,
) -> pd.DataFrame:
    """Add ``_ref_distance`` column to an expanded pairwise DataFrame.

    For each row, ``_ref_distance`` is the chemical distance between the
    target substrate and the reference substrate, looked up from a
    precomputed distance matrix.

    Args:
        expanded_df:     DataFrame with ``substrate`` and ``ref_substrate`` columns.
        distance_matrix: (n_substrates, n_substrates) symmetric distance matrix.
        substrate_names: Ordered list of substrate names matching matrix rows/cols.

    Returns:
        The same DataFrame with ``_ref_distance`` column added (in-place).
    """
    sub_to_idx = {name: i for i, name in enumerate(substrate_names)}
    dists = np.empty(len(expanded_df), dtype=np.float32)
    for i, (_, row) in enumerate(expanded_df.iterrows()):
        t_idx = sub_to_idx.get(row["substrate"])
        r_idx = sub_to_idx.get(row["ref_substrate"])
        if t_idx is not None and r_idx is not None:
            dists[i] = distance_matrix[t_idx, r_idx]
        else:
            dists[i] = np.inf  # unknown substrate → infinite distance
    expanded_df["_ref_distance"] = dists
    return expanded_df


# ═══════════════════════════════════════════════════════════════════════════
# Feature Assembly
# ═══════════════════════════════════════════════════════════════════════════

def get_substrate_embedding(
    substrate_name: str,
    embedding_type: str,
    embeddings: dict,
) -> np.ndarray:
    """Get substrate embedding vector by name and type.

    Args:
        substrate_name: e.g. "Formaldehyde"
        embedding_type: "morgan", "maccs", "mordred", or "molformer"
        embeddings: dict from load_all_embeddings

    Returns:
        (dim,) embedding array
    """
    names = embeddings["substrate_names"]
    idx = names.index(substrate_name)
    return embeddings[f"substrate_{embedding_type}"][idx].astype(np.float32)


def build_bnn1_input(
    df_subset: pd.DataFrame,
    embeddings: dict,
    pipe_wt,
    pipe_mut,
) -> np.ndarray:
    """Build BNN1 input features: preprocessed ESM2 WT + mutant concatenated.

    Uses BNN1's saved preprocessing pipelines (transform-only, already fitted).
    Handles pyruvate (ref_type='v83p') by using V83P WT embeddings.

    Returns:
        (n, bnn1_input_dim) float32 array
    """
    esm_wt = embeddings["esm2_wt"]
    esm_v83p = embeddings["esm2_v83p"]
    esm_mut = embeddings["esm2_mut"]

    wt_list = []
    mut_list = []

    for _, row in df_subset.iterrows():
        pos = row["position"]
        ms = row["mutation_string"]
        ref_type = row.get("ref_type", "wt")

        # WT embedding: use V83P for pyruvate
        if ref_type == "v83p":
            wt_list.append(esm_v83p[pos])
        else:
            wt_list.append(esm_wt[pos])

        # Mutant embedding
        if ms in esm_mut:
            mut_list.append(esm_mut[ms])
        else:
            warnings.warn(f"Missing ESM2 mutant embedding for {ms}")
            mut_list.append(np.zeros(esm_wt.shape[1], dtype=np.float32))

    X_wt = np.stack(wt_list).astype(np.float32)
    X_mut = np.stack(mut_list).astype(np.float32)

    # Apply BNN1's preprocessing (transform-only)
    if pipe_wt is not None:
        X_wt = pipe_wt.transform(X_wt).astype(np.float32)
    if pipe_mut is not None:
        X_mut = pipe_mut.transform(X_mut).astype(np.float32)

    return np.concatenate([X_wt, X_mut], axis=1)


def build_other_features(
    df_subset: pd.DataFrame,
    embeddings: dict,
    params: dict,
    substrate_meta: dict,
) -> dict:
    """Build non-ESM feature groups for BNN2.

    Returns dict of {group_name: (n, dim) array}. Respects feature toggles
    from bnn2.features config.

    For pairwise data: df_subset must have 'ref_substrate' and 'fc_ref' columns.
    """
    features_raw = params.get("features", {})
    # Resolve value/search dicts to plain booleans
    features_cfg = resolve_config_block(features_raw) if any(
        isinstance(v, dict) for v in features_raw.values()) else features_raw
    substrate_emb_type = params.get("substrate_embedding_type", "morgan")
    n = len(df_subset)

    groups = {}

    # fc_ref: fold-change on reference substrate (scalar)
    if features_cfg.get("fc_ref", True) and "fc_ref" in df_subset.columns:
        groups["fc_ref"] = df_subset["fc_ref"].values.astype(np.float32).reshape(-1, 1)

    # x_target_substrate: target substrate embedding
    if features_cfg.get("x_target_substrate", True):
        target_embs = []
        for _, row in df_subset.iterrows():
            target_embs.append(
                get_substrate_embedding(row["substrate"], substrate_emb_type, embeddings)
            )
        groups["x_target_substrate"] = np.stack(target_embs)

    # x_ref_substrate: reference substrate embedding (pairwise)
    if features_cfg.get("x_ref_substrate", True) and "ref_substrate" in df_subset.columns:
        ref_embs = []
        for _, row in df_subset.iterrows():
            ref_embs.append(
                get_substrate_embedding(row["ref_substrate"], substrate_emb_type, embeddings)
            )
        groups["x_ref_substrate"] = np.stack(ref_embs)

    # ref_distance: chemical distance between target and reference substrate (scalar)
    if features_cfg.get("ref_distance", True) and "_ref_distance" in df_subset.columns:
        groups["ref_distance"] = df_subset["_ref_distance"].values.astype(np.float32).reshape(-1, 1)

    # saprot_zs: SaProt zero-shot score (scalar)
    if features_cfg.get("saprot_zs", True):
        saprot = embeddings["saprot"]
        scores = []
        for _, row in df_subset.iterrows():
            ms = row["mutation_string"]
            scores.append(float(saprot.get(ms, 0.0)))
        groups["saprot_zs"] = np.array(scores, dtype=np.float32).reshape(-1, 1)

    # esm_wt: WT ESM2 embedding at mutated position (1280-dim, preprocessed independently of BNN1)
    if features_cfg.get("esm_wt", False):
        esm_wt_emb = embeddings["esm2_wt"]
        esm_v83p = embeddings["esm2_v83p"]
        wt_list = []
        for _, row in df_subset.iterrows():
            pos = row["position"]
            ref_type = row.get("ref_type", "wt")
            if ref_type == "v83p":
                wt_list.append(esm_v83p[pos])
            else:
                wt_list.append(esm_wt_emb[pos])
        groups["esm_wt"] = np.stack(wt_list).astype(np.float32)

    # esm_mut: Mutant ESM2 embedding (1280-dim, preprocessed independently of BNN1)
    if features_cfg.get("esm_mut", False):
        esm_mut_dict = embeddings["esm2_mut"]
        mut_list = []
        for _, row in df_subset.iterrows():
            ms = row["mutation_string"]
            if ms in esm_mut_dict:
                mut_list.append(esm_mut_dict[ms])
            else:
                mut_list.append(np.zeros(embeddings["esm2_wt"].shape[1], dtype=np.float32))
        groups["esm_mut"] = np.stack(mut_list).astype(np.float32)

    return groups


# ═══════════════════════════════════════════════════════════════════════════
# Preprocessing (for "other" features — BNN1 preprocessing is fixed)
# ═══════════════════════════════════════════════════════════════════════════

def build_preprocessing(scaler_type: str, pca_components):
    """Build sklearn preprocessing pipeline: optional scaler → optional PCA."""
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


def preprocess_other_features(
    groups_train: dict,
    groups_val: dict,
    params: dict,
    config: dict,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Preprocess other feature groups independently, then concatenate.

    Each group gets its own scaler/PCA from config.preprocessing settings.
    Groups with 1-2 dimensions (fc_ref, saprot_zs) skip PCA.

    Returns:
        X_train_concat: (n_train, total_dim) preprocessed + concatenated
        X_val_concat: (n_val, total_dim) preprocessed + concatenated
        pipelines: dict of {group_name: fitted Pipeline}
    """
    preproc_cfg = config.get("preprocessing", {})
    pipelines = {}

    train_parts = []
    val_parts = []

    for group_name in sorted(groups_train.keys()):
        X_tr = groups_train[group_name]
        X_va = groups_val[group_name]

        # Determine preprocessing for this group
        if group_name in ("x_target_substrate", "x_ref_substrate"):
            cfg = preproc_cfg.get("x_substrate", {})
        elif group_name == "saprot_zs":
            cfg = preproc_cfg.get("saprot_zs", {})
        elif group_name == "fc_ref":
            # FC_ref is a scalar; just pass through (optionally scale)
            cfg = {}
        else:
            cfg = preproc_cfg.get(group_name, {})

        scaler_type = resolve_param(cfg.get("scaler", "none"),
                                    params.get(f"{group_name}_scaler"))
        pca_val = resolve_param(cfg.get("pca", None),
                                params.get(f"{group_name}_pca"))
        pca_components = parse_pca_value(pca_val)

        # Skip PCA for low-dim features
        if X_tr.shape[1] <= 2:
            pca_components = None

        pipe = build_preprocessing(scaler_type, pca_components)
        X_tr_t, X_va_t = apply_preprocessing(pipe, X_tr, X_va)

        pipelines[group_name] = pipe
        train_parts.append(X_tr_t)
        val_parts.append(X_va_t)

        logger.debug("Preprocessed %s: %d → %d dims (scaler=%s, pca=%s)",
                      group_name, X_tr.shape[1], X_tr_t.shape[1],
                      scaler_type, pca_components)

    X_train_concat = np.concatenate(train_parts, axis=1)
    X_val_concat = np.concatenate(val_parts, axis=1)

    logger.info("Other features preprocessed: %d dims total", X_train_concat.shape[1])
    return X_train_concat, X_val_concat, pipelines


# ═══════════════════════════════════════════════════════════════════════════
# Label Distribution Smoothing (LDS) — per-sample loss weights
# ═══════════════════════════════════════════════════════════════════════════
#
# Adapted from Yang et al. 2021 "Delving into Deep Imbalanced Regression"
# and Gado 2024 https://github.com/jafetgado/EpHod


class LDSAttenuatedWeights:
    """Compute per-sample loss weights using Label Distribution Smoothing.

    Fits a smoothed histogram of label values and returns per-sample weights
    proportional to 1 / smoothed_density.  Rare label values (e.g. high-FC
    beneficial mutations) get high weights; common values (e.g. FC=0 cluster)
    get low weights.

    With normalize=True (default) the weights are scaled so mean=1, which
    preserves the overall loss magnitude regardless of the label distribution.
    """

    def __init__(
        self,
        n_bins: int = 100,
        kernel_size: int = 5,
        sigma: float = 2.0,
        min_weight: float = 0.0,
        max_weight: float = np.inf,
        normalize: bool = True,
        eps: float = 1e-1,
    ):
        self.n_bins = n_bins
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.normalize = normalize
        self.eps = eps
        self.bin_edges_: Optional[np.ndarray] = None
        self.bin_densities_: Optional[np.ndarray] = None

    def _compute_kernel(self) -> np.ndarray:
        half_ks = (self.kernel_size - 1) // 2
        base = [0.0] * half_ks + [1.0] + [0.0] * half_ks
        kernel = gaussian_filter1d(base, sigma=self.sigma)
        return kernel / kernel.max()

    def fit(self, X: np.ndarray) -> "LDSAttenuatedWeights":
        X = np.asarray(X).reshape(-1)
        valid = X[~np.isnan(X)]
        if len(valid) == 0:
            raise ValueError("No valid values provided to LDSAttenuatedWeights.fit()")
        hist, self.bin_edges_ = np.histogram(valid, bins=self.n_bins, density=True)
        kernel = self._compute_kernel()
        self.bin_densities_ = convolve1d(hist, weights=kernel, mode="constant")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.bin_densities_ is None:
            raise RuntimeError("Call fit() before transform()")
        X = np.asarray(X).reshape(-1)
        weights = np.zeros_like(X, dtype=float)
        valid_mask = ~np.isnan(X)
        if not np.any(valid_mask):
            return weights
        bin_centers = (self.bin_edges_[:-1] + self.bin_edges_[1:]) / 2
        densities = np.interp(X[valid_mask], bin_centers, self.bin_densities_)
        w = 1.0 / (densities + self.eps)
        w = np.clip(w, self.min_weight, self.max_weight)
        if self.normalize:
            w /= np.std(w) if np.std(w) > 0 else 1.0
            w /= np.mean(w) if np.mean(w) > 0 else 1.0
        weights[valid_mask] = w
        return weights

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def compute_lds_weights(
    y_train: np.ndarray,
    use_lds: bool,
    n_bins: int = 50,
    kernel_size: int = 5,
    sigma: float = 2.0,
) -> Optional[np.ndarray]:
    """Return LDS per-sample weights (mean=1) or None for uniform weighting.

    Args:
        y_train: Training label array (log_fc values).
        use_lds: If False, returns None (all samples weighted equally).
        n_bins: Number of histogram bins for density estimation.
        kernel_size: Gaussian kernel window size for density smoothing.
        sigma: Gaussian kernel standard deviation.

    Returns:
        np.ndarray of shape (n,) with mean ≈ 1.0, or None.
    """
    if not use_lds:
        return None
    lds = LDSAttenuatedWeights(n_bins=n_bins, kernel_size=kernel_size, sigma=sigma)
    weights = lds.fit_transform(y_train)
    logger.info(
        "LDS weights: min=%.3f  max=%.3f  mean=%.3f  "
        "(n_negative_delta=%d, n_positive_delta=%d / %d total)",
        weights.min(), weights.max(), weights.mean(),
        (y_train < 0.0).sum(),
        (y_train > 0.0).sum(),
        len(y_train),
    )
    return weights


def plot_lds_weights(
    traces: List[Tuple[np.ndarray, np.ndarray, str]],
    output_path: Path,
) -> None:
    """Plot LDS weight distributions for multiple folds and/or the full dataset.

    Args:
        traces: List of (y_values, weights, label) tuples, one per fold/dataset.
                The last trace is treated as the "full dataset" reference and
                drawn with a thicker, darker line.
        output_path: PNG path to save the figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Colour palette — folds get lighter colours, full dataset is bold black
    n_traces = len(traces)
    cmap = cm.get_cmap("tab10", max(n_traces, 1))
    colors = [cmap(i) for i in range(n_traces - 1)] + ["black"]
    linewidths = [1.2] * (n_traces - 1) + [2.2]
    alphas = [0.6] * (n_traces - 1) + [1.0]

    # ---- Left panel: label distribution + smoothed density curves ----
    ax_left = axes[0]

    # Background histogram from the last (full-dataset) trace
    if traces:
        y_all = traces[-1][0]
        ax_left.hist(y_all, bins=50, density=True, color="lightgray",
                     edgecolor="none", label="Full data histogram", zorder=0)

    for (y_vals, _, label), color, lw, alpha in zip(traces, colors, linewidths, alphas):
        lds = LDSAttenuatedWeights(n_bins=50, kernel_size=5, sigma=2.0)
        lds.fit(y_vals)
        bin_centers = (lds.bin_edges_[:-1] + lds.bin_edges_[1:]) / 2
        ax_left.plot(bin_centers, lds.bin_densities_, color=color, lw=lw,
                     alpha=alpha, label=label)

    ax_left.axvline(0.0, color="green", ls="--", lw=0.8, alpha=0.7, label="Δ=0 (no change)")
    ax_left.set_xlabel("Δ log_fc (delta)")
    ax_left.set_ylabel("Smoothed density")
    ax_left.set_title("Delta Target Distribution (smoothed)")
    ax_left.legend(fontsize=7)

    # ---- Right panel: weight vs log_fc ----
    ax_right = axes[1]

    for (y_vals, weights, label), color, lw, alpha in zip(traces, colors, linewidths, alphas):
        order = np.argsort(y_vals)
        y_sorted = y_vals[order]
        w_sorted = weights[order]
        # Rolling mean for readability
        window = max(1, len(y_sorted) // 80)
        w_smooth = np.convolve(w_sorted, np.ones(window) / window, mode="same")
        ax_right.plot(y_sorted, w_smooth, color=color, lw=lw, alpha=alpha, label=label)

    ax_right.axhline(1.0, color="gray", ls=":", lw=1.0, label="weight=1 (baseline)")
    ax_right.axvline(0.0, color="green", ls="--", lw=0.8, alpha=0.7)

    # Annotate mean weights at key tiers using the last (full-dataset) trace
    if traces:
        y_all, w_all, _ = traces[-1]
        w_neg = w_all[y_all < 0.0].mean() if (y_all < 0.0).any() else float("nan")
        w_pos = w_all[y_all > 0.0].mean() if (y_all > 0.0).any() else float("nan")
        ax_right.text(0.02, 0.97,
                      f"Mean weight Δ<0: {w_neg:.2f}\nMean weight Δ>0: {w_pos:.2f}",
                      transform=ax_right.transAxes, va="top", fontsize=8,
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    ax_right.set_xlabel("Δ log_fc (delta)")
    ax_right.set_ylabel("LDS weight")
    ax_right.set_title("Per-Sample Loss Weights")
    ax_right.legend(fontsize=7)

    n_folds = n_traces - 1
    fig.suptitle(f"LDS Weight Distribution — {n_folds} fold(s) + full dataset")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved LDS weight plot: %s", output_path)


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def build_bnn2_model(
    bnn1_hidden: Optional[nn.ModuleList],
    bnn1_input_dim: int,
    latent_dim: int,
    other_feature_dim: int,
    params: dict,
    device: str,
) -> BNN2Model:
    """Construct a BNN2Model with optional BNN1 backbone + BNN2 head.

    When bnn1_hidden is None (x_aa feature off), the model has no BNN1 backbone
    and the BNN2 head receives only other_features directly.
    """
    use_bnn1 = bnn1_hidden is not None
    bnn1_clone = copy.deepcopy(bnn1_hidden) if use_bnn1 else None

    hurdle_enabled = params.get("loss_type", "gaussian_nll") == "hurdle"
    model = BNN2Model(
        bnn1_hidden=bnn1_clone,
        bnn1_input_dim=bnn1_input_dim if use_bnn1 else 0,
        latent_dim=latent_dim if use_bnn1 else 0,
        other_feature_dim=other_feature_dim,
        hidden_dims=params["hidden_dims"],
        output_dim=1,
        task="regression",
        prior_std=params["prior_std"],
        dropout_rate=params["dropout_rate"],
        activation=params.get("activation", "silu"),
        freeze_mode=params.get("x_aa_freeze", "full") if use_bnn1 else "full",
        hurdle=hurdle_enabled,
    )
    model.to(device)

    counts = model.trainable_parameter_count()
    logger.info("BNN2 model: %d trainable, %d frozen, %d total params (freeze=%s)",
                counts["trainable"], counts["frozen"], counts["total"],
                params.get("x_aa_freeze", "full"))

    return model


def train_and_evaluate_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    bnn1_hidden: nn.ModuleList,
    bnn1_input_dim: int,
    latent_dim: int,
    other_feature_dim: int,
    params: dict,
    device: str,
    return_predictions: bool = False,
    fc_ref_train: Optional[np.ndarray] = None,
    fc_ref_val: Optional[np.ndarray] = None,
) -> Tuple[dict, Optional[object], Optional[object]]:
    """Train BNN2 on one fold, evaluate on val set.

    X_train/X_val are already preprocessed and concatenated:
        [BNN1_input | other_features_preprocessed]

    y_train/y_val are **delta** targets: log_fc - log_fc_ref.
    fc_ref_train/fc_ref_val are the log_fc_ref values used to reconstruct
    absolute predictions for metrics and hurdle floor detection.

    Returns:
        fold_metrics: dict with mae, rmse, r2, spearman_rho, nlpd, crps, val_loss
        estimates: UncertaintyEstimate (if return_predictions, else None)
        history: TrainingHistory (if return_predictions, else None)
    """
    _, BNNTrainer, TrainingConfig, _, _, _, _, HurdleConfig = _import_bnns()

    model = build_bnn2_model(
        bnn1_hidden, bnn1_input_dim, latent_dim, other_feature_dim,
        params, device,
    )

    # Build hurdle config (sub-parameters only; loss_type controls activation)
    hurdle_cfg = params.get("hurdle", {})
    hurdle_config = HurdleConfig(
        floor_threshold=hurdle_cfg.get("floor_threshold", -1.99),
        floor_value=hurdle_cfg.get("floor_value", -2.0),
        inference_threshold=hurdle_cfg.get("inference_threshold", 0.5),
    )

    training_config = TrainingConfig(
        n_epochs=params["n_epochs"],
        learning_rate=params["learning_rate"],
        batch_size=params["batch_size"],
        kl_anneal_epochs=params["kl_anneal_epochs"],
        kl_weight=params["kl_weight"],
        early_stopping_patience=params["early_stopping_patience"],
        n_inference_samples=params["n_inference_samples"],
        clip_grad_norm=params.get("clip_grad_norm", 1.0),
        device=device,
        verbose=return_predictions,
        log_interval=1,
        loss_type=params.get("loss_type", "gaussian_nll"),
        hurdle=hurdle_config,
        null_reg_weight=params.get("null_reg_weight", 0.0),
        log_var_floor=params.get("log_var_floor", None),
    )

    trainer = BNNTrainer(model, training_config)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)

    # LDS per-sample weights (fitted on training labels only; val always unweighted)
    lds_cfg = params.get("lds", {})
    weights = compute_lds_weights(
        y_train,
        use_lds=lds_cfg.get("use_lds", False),
        n_bins=lds_cfg.get("n_bins", 50),
        kernel_size=lds_cfg.get("kernel_size", 5),
        sigma=lds_cfg.get("sigma", 2.0),
    )
    w_train_t = torch.tensor(weights, dtype=torch.float32) if weights is not None else None

    # Precompute is_floor masks from absolute targets for hurdle loss.
    # Delta targets shift y per-reference, so y <= threshold is meaningless;
    # we compute is_floor in absolute space: (y_delta + log_fc_ref) <= threshold.
    is_floor_train_t = None
    is_floor_val_t = None
    hurdle_active = params.get("loss_type", "gaussian_nll") == "hurdle"
    if hurdle_active and fc_ref_train is not None:
        floor_thresh = hurdle_config.floor_threshold
        y_abs_train = y_train + fc_ref_train
        is_floor_train_t = torch.tensor(
            y_abs_train <= floor_thresh, dtype=torch.float32
        )
    if hurdle_active and fc_ref_val is not None:
        floor_thresh = hurdle_config.floor_threshold
        y_abs_val = y_val + fc_ref_val
        is_floor_val_t = torch.tensor(
            y_abs_val <= floor_thresh, dtype=torch.float32
        )

    # fc_ref tensor for delta → absolute reconstruction at inference
    fc_ref_val_t = None
    if fc_ref_val is not None:
        fc_ref_val_t = torch.tensor(
            fc_ref_val, dtype=torch.float32
        ).unsqueeze(-1)  # (B,) → (B, 1) to match output_dim

    history = trainer.fit(
        X_train_t, y_train_t, X_val_t, y_val_t,
        w_train=w_train_t,
        is_floor_train=is_floor_train_t,
        is_floor_val=is_floor_val_t,
    )

    prediction_floor = params.get("prediction_floor", None)
    estimates = trainer.predict(
        X_val_t, hurdle_config=hurdle_config, fc_ref=fc_ref_val_t,
        prediction_floor=prediction_floor,
    )
    y_pred = estimates.mean.cpu().numpy().squeeze(-1)
    total_std_np = estimates.total_std.cpu().numpy().squeeze(-1)

    # Metrics — always in absolute space.
    # y_pred is already absolute (fc_ref added in predict_with_uncertainty).
    # Reconstruct absolute y_val for comparison.
    if fc_ref_val is not None:
        y_val_abs = y_val + fc_ref_val  # delta + ref = absolute
    else:
        y_val_abs = y_val  # fallback: y_val is already absolute

    # ═══════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC: Prediction analysis (delta space + absolute space)
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("┌─── FOLD PREDICTION DIAGNOSTICS ───┐")

    # Delta space analysis: what did the model actually predict before fc_ref?
    if fc_ref_val is not None:
        y_pred_delta = y_pred - fc_ref_val  # undo reconstruction to see raw delta
        logger.info("│ Delta space (model raw output):")
        logger.info("│   y_pred_delta: mean=%.4f  std=%.4f  range=[%.4f, %.4f]",
                     y_pred_delta.mean(), y_pred_delta.std(),
                     y_pred_delta.min(), y_pred_delta.max())
        logger.info("│   y_val_delta:  mean=%.4f  std=%.4f  range=[%.4f, %.4f]",
                     y_val.mean(), y_val.std(), y_val.min(), y_val.max())
        delta_resid = y_val - y_pred_delta
        logger.info("│   Delta residual: mean=%.4f  MAE=%.4f  RMSE=%.4f",
                     delta_resid.mean(),
                     np.abs(delta_resid).mean(),
                     np.sqrt(np.mean(delta_resid**2)))
        null_delta_mae = float(np.mean(np.abs(y_val)))  # null = predict 0
        model_delta_mae = float(np.mean(np.abs(delta_resid)))
        logger.info("│   Null MAE (Δ=0):  %.4f", null_delta_mae)
        logger.info("│   Model MAE (Δ):   %.4f", model_delta_mae)
        if model_delta_mae > null_delta_mae:
            logger.warning(
                "│   ⚠ MODEL WORSE THAN NULL in delta space! ratio=%.2f",
                model_delta_mae / max(null_delta_mae, 1e-10))
        delta_rho, _ = stats.spearmanr(y_val, y_pred_delta)
        logger.info("│   Delta Spearman: %.4f", delta_rho)

        # Correlation between fc_ref and prediction — is model just copying fc_ref?
        ref_rho, _ = stats.spearmanr(fc_ref_val, y_pred)
        target_rho, _ = stats.spearmanr(y_val_abs, fc_ref_val)
        logger.info("│ fc_ref analysis:")
        logger.info("│   fc_ref: mean=%.4f  std=%.4f  range=[%.4f, %.4f]",
                     fc_ref_val.mean(), fc_ref_val.std(),
                     fc_ref_val.min(), fc_ref_val.max())
        logger.info("│   corr(y_pred_abs, fc_ref): %.4f  (=1 means model just copies ref)",
                     ref_rho)
        logger.info("│   corr(y_true_abs, fc_ref): %.4f  (baseline ref explains this much)",
                     target_rho)

    # Absolute space analysis
    logger.info("│ Absolute space:")
    logger.info("│   y_pred_abs: mean=%.4f  std=%.4f  range=[%.4f, %.4f]",
                 y_pred.mean(), y_pred.std(), y_pred.min(), y_pred.max())
    logger.info("│   y_val_abs:  mean=%.4f  std=%.4f  range=[%.4f, %.4f]",
                 y_val_abs.mean(), y_val_abs.std(), y_val_abs.min(), y_val_abs.max())

    # Uncertainty analysis
    logger.info("│ Uncertainty:")
    logger.info("│   total_std: mean=%.4f  std=%.4f  range=[%.4f, %.4f]",
                 total_std_np.mean(), total_std_np.std(),
                 total_std_np.min(), total_std_np.max())
    epi_np = estimates.epistemic_std.cpu().numpy().squeeze(-1)
    ale_np = estimates.aleatoric_std.cpu().numpy().squeeze(-1)
    logger.info("│   epistemic_std: mean=%.4f  aleatoric_std: mean=%.4f",
                 epi_np.mean(), ale_np.mean())
    logger.info("│   epi/total ratio: %.2f%%",
                 100.0 * epi_np.mean() / max(total_std_np.mean(), 1e-10))

    # MC sample analysis
    samples_np = estimates.samples.cpu().numpy().squeeze(-1)  # (S, B)
    mc_std_per_sample = samples_np.std(axis=0)
    logger.info("│ MC samples (%d):", samples_np.shape[0])
    logger.info("│   per-sample MC std: mean=%.4f  min=%.4f  max=%.4f",
                 mc_std_per_sample.mean(), mc_std_per_sample.min(), mc_std_per_sample.max())
    if mc_std_per_sample.mean() < 0.01:
        logger.warning("│   ⚠ Very low MC variance — possible posterior collapse!")

    logger.info("└──────────────────────────────────────┘")

    residuals = y_val_abs - y_pred
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_val_abs - y_val_abs.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rho, rho_p = stats.spearmanr(y_val_abs, y_pred)

    nlpd = compute_nlpd(y_val_abs, y_pred, total_std_np)
    crps = compute_crps_gaussian(y_val_abs, y_pred, total_std_np)

    # Null model in delta space: predict zero (no change from reference).
    # In absolute space: null_pred = fc_ref_val, so null_mae = mean(|y_abs - fc_ref|).
    if fc_ref_val is not None:
        null_mae = float(np.mean(np.abs(y_val_abs - fc_ref_val)))
    else:
        null_pred = float(y_train.mean())
        null_mae = float(np.mean(np.abs(y_val_abs - null_pred)))

    val_nll = history.val_nll[-1] if history.val_nll else float("nan")
    val_elbo = history.val_loss[-1] if history.val_loss else float("nan")

    # Posterior collapse diagnostic: ratio of mean posterior std to prior_std.
    # With rho_init = softplus_inverse(prior_std), initialisation starts at 1.0.
    # After training a healthy model should tighten (< 1.0); near 1.0 = no update.
    summary = model.posterior_summary()
    posterior_stds = [v["mean_std"] for v in summary.values()]
    prior_std = params.get("prior_std", 1.0)
    collapse_score = (
        float(np.mean(posterior_stds)) / max(prior_std, 1e-8)
        if posterior_stds else 1.0
    )

    fold_metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": float(r2),
        "spearman_rho": float(rho),
        "spearman_pvalue": float(rho_p),
        "nlpd": nlpd,
        "crps": crps,
        "val_nll": float(val_nll),
        "val_elbo": float(val_elbo),
        "null_mae": null_mae,
        "posterior_collapse_score": collapse_score,
        "best_epoch": history.best_epoch,
    }

    if return_predictions:
        return fold_metrics, estimates, history
    return fold_metrics, None, None


# ═══════════════════════════════════════════════════════════════════════════
# Pairwise Aggregation
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_pairwise_predictions(
    y_pred_expanded: np.ndarray,
    epistemic_std_expanded: np.ndarray,
    aleatoric_std_expanded: np.ndarray,
    total_std_expanded: np.ndarray,
    expanded_df: pd.DataFrame,
    cls_prob_expanded: Optional[np.ndarray] = None,
    hurdle_config: Optional[dict] = None,
    aggregation_mode: str = "nearest",
    distance_weight_temperature: float = 1.0,
    formaldehyde_substrate: str = "Formaldehyde",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Aggregate pairwise predictions back to per-(mutation, target_substrate).

    Four aggregation modes:
      - "mean":              Equal-weight average across all reference substrates.
                             Uses law of total variance for uncertainty.
      - "nearest":           Select only the closest reference substrate (by
                             ``_ref_distance`` column in expanded_df). Single
                             prediction, no averaging.
      - "distance_weighted": Weight each reference by softmax(-dist / temperature).
                             Weighted mean + weighted law of total variance.
      - "formaldehyde":      Keep only rows where ``ref_substrate ==
                             formaldehyde_substrate``. Groups whose target is
                             Formaldehyde (self-reference) or whose mutation is
                             missing on Formaldehyde are dropped from the output.

    When ``cls_prob_expanded`` is provided (hurdle mode), cls_prob is aggregated
    with the same weighting scheme and stored as ``_cls_prob`` in the output.
    The hurdle soft mixture is NOT applied here — it was already applied during
    inference (in predict_with_uncertainty).

    Args:
        y_pred_expanded:   Predictions per expanded row (soft-mixture mean if hurdle).
        epistemic_std_expanded: Epistemic std per expanded row.
        aleatoric_std_expanded: Aleatoric std per expanded row.
        total_std_expanded:     Total std per expanded row.
        expanded_df:            DataFrame with pairwise rows (must have ``_ref_distance``
                                column for "nearest" and "distance_weighted" modes).
        cls_prob_expanded:      Optional P(active) per expanded row (hurdle only).
        hurdle_config:          Optional hurdle config dict (for floor_value, etc.).
        aggregation_mode:       "mean", "nearest", or "distance_weighted".
        distance_weight_temperature: Temperature for distance-weighted softmax.

    Returns:
        y_pred_agg, epistemic_agg, aleatoric_agg, total_std_agg, base_df
    """
    expanded_df = expanded_df.copy()
    expanded_df["_y_pred"] = y_pred_expanded
    expanded_df["_epi_std"] = epistemic_std_expanded
    expanded_df["_ale_std"] = aleatoric_std_expanded
    expanded_df["_tot_std"] = total_std_expanded
    if cls_prob_expanded is not None:
        expanded_df["_cls_prob"] = cls_prob_expanded

    group_cols = ["mutation_string", "substrate"]
    grouped = expanded_df.groupby(group_cols, sort=False)

    agg_rows = []
    for (mut, sub), grp in grouped:
        # Formaldehyde mode: drop self-reference targets and filter to formaldehyde-as-ref.
        if aggregation_mode == "formaldehyde":
            if sub == formaldehyde_substrate:
                continue
            grp = grp[grp["ref_substrate"] == formaldehyde_substrate]
            if len(grp) == 0:
                continue

        means = grp["_y_pred"].values
        epi_stds = grp["_epi_std"].values
        ale_stds = grp["_ale_std"].values
        n_refs = len(grp)

        # ── Compute weights based on aggregation mode ──
        if aggregation_mode == "nearest" and "_ref_distance" in grp.columns:
            # Select only the closest reference
            best_idx = int(grp["_ref_distance"].values.argmin())
            w = np.zeros(n_refs)
            w[best_idx] = 1.0
        elif aggregation_mode == "distance_weighted" and "_ref_distance" in grp.columns:
            dists = grp["_ref_distance"].values.astype(float)
            # softmax(-dist / temperature)
            logits = -dists / max(distance_weight_temperature, 1e-8)
            logits = logits - logits.max()  # numerical stability
            exp_logits = np.exp(logits)
            w = exp_logits / exp_logits.sum()
        else:
            # "mean" / "formaldehyde" modes, or fallback without distance column
            w = np.ones(n_refs) / n_refs

        # ── Weighted aggregation ──
        agg_mean = float(np.dot(w, means))

        # Weighted law of total variance
        agg_epi_var_within = float(np.dot(w, epi_stds**2))
        agg_ale_var = float(np.dot(w, ale_stds**2))
        var_means = float(np.dot(w, (means - agg_mean)**2))
        agg_epi_var = agg_epi_var_within + var_means
        agg_total_var = agg_epi_var + agg_ale_var

        # Hurdle: aggregate cls_prob with same weights
        agg_cls_prob = None
        if "_cls_prob" in grp.columns:
            agg_cls_prob = float(np.dot(w, grp["_cls_prob"].values))

        # Take first row for metadata (or nearest row if applicable)
        if aggregation_mode == "nearest" and "_ref_distance" in grp.columns:
            first_row = grp.iloc[int(grp["_ref_distance"].values.argmin())]
        else:
            first_row = grp.iloc[0]

        row_dict = {
            "mutation_string": mut,
            "substrate": sub,
            "position": first_row["position"],
            "wt_aa": first_row["wt_aa"],
            "mut_aa": first_row["mut_aa"],
            "fold_change": first_row["fold_change"],
            "log_fc": first_row["log_fc"],
            "is_active_substrate": first_row["is_active_substrate"],
            "ref_type": first_row.get("ref_type", "wt"),
            "_y_pred": agg_mean,
            "_epi_std": np.sqrt(max(agg_epi_var, 0)),
            "_ale_std": np.sqrt(max(agg_ale_var, 0)),
            "_tot_std": np.sqrt(max(agg_total_var, 0)),
            "n_refs": n_refs,
        }
        if agg_cls_prob is not None:
            row_dict["_cls_prob"] = agg_cls_prob
        agg_rows.append(row_dict)

    agg_df = pd.DataFrame(agg_rows)

    if len(agg_df) == 0:
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty, empty, agg_df

    return (
        agg_df["_y_pred"].values,
        agg_df["_epi_std"].values,
        agg_df["_ale_std"].values,
        agg_df["_tot_std"].values,
        agg_df,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_nlpd(y_true: np.ndarray, y_pred: np.ndarray, total_std: np.ndarray) -> float:
    """Negative Log Predictive Density (Gaussian). Lower is better."""
    variance = np.clip(total_std ** 2, 1e-10, None)
    nlpd = 0.5 * (np.log(2 * np.pi) + np.log(variance) + (y_true - y_pred) ** 2 / variance)
    return float(np.mean(nlpd))


def compute_crps_gaussian(y_true: np.ndarray, y_pred: np.ndarray, total_std: np.ndarray) -> float:
    """Continuous Ranked Probability Score (Gaussian). Lower is better."""
    sigma = np.clip(total_std, 1e-10, None)
    z = (y_true - y_pred) / sigma
    crps = sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    return float(np.mean(crps))


def compute_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    total_std: np.ndarray,
    levels: Optional[List[float]] = None,
) -> dict:
    """Compute calibration: does a q% CI contain q% of observations?"""
    if levels is None:
        levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    observed_coverage = []
    for q in levels:
        z = stats.norm.ppf((1 + q) / 2)
        lower = y_pred - z * total_std
        upper = y_pred + z * total_std
        in_interval = (y_true >= lower) & (y_true <= upper)
        observed_coverage.append(float(np.mean(in_interval)))

    return {
        "levels": levels,
        "expected_coverage": levels,
        "observed_coverage": observed_coverage,
    }


def compute_per_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_labels: np.ndarray,
    min_count: int = 5,
) -> Tuple[dict, dict, dict]:
    """Compute Spearman, MAE, and activity range per group.

    Returns:
        per_group_spearman, per_group_mae, per_group_range
    """
    unique_groups = sorted(set(group_labels))
    per_group_spearman = {}
    per_group_mae = {}
    per_group_range = {}

    for grp in unique_groups:
        mask = group_labels == grp
        n = mask.sum()
        if n < min_count:
            continue

        yt = y_true[mask]
        yp = y_pred[mask]

        per_group_mae[grp] = float(np.mean(np.abs(yt - yp)))
        per_group_range[grp] = float(np.std(yt))

        if np.std(yt) > 1e-8 and np.std(yp) > 1e-8:
            rho, _ = stats.spearmanr(yt, yp)
            per_group_spearman[grp] = float(rho)

    return per_group_spearman, per_group_mae, per_group_range


def compute_tanimoto_distances(embeddings: dict) -> np.ndarray:
    """Compute pairwise Tanimoto distances using Morgan fingerprints (fixed).

    Returns (n_substrates, n_substrates) distance matrix.
    """
    morgan = embeddings["substrate_morgan"]  # (n_sub, 2048)
    n = morgan.shape[0]
    dist = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = morgan[i], morgan[j]
            intersection = np.sum(np.minimum(a, b))
            union = np.sum(np.maximum(a, b))
            sim = intersection / union if union > 0 else 0.0
            dist[i, j] = 1.0 - sim
            dist[j, i] = dist[i, j]

    return dist


def compute_embedding_distances(
    embeddings: dict,
    embedding_type: str = "morgan",
) -> np.ndarray:
    """Pairwise cosine distances using the specified substrate embedding type.

    Returns (n_substrates, n_substrates) distance matrix where
    d[i,j] = 1 - cosine_similarity(emb[i], emb[j]).
    """
    emb = embeddings[f"substrate_{embedding_type}"].astype(np.float64)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    normed = emb / np.maximum(norms, 1e-10)
    cos_sim = normed @ normed.T
    np.clip(cos_sim, -1.0, 1.0, out=cos_sim)
    dist = (1.0 - cos_sim).astype(np.float32)
    np.fill_diagonal(dist, 0.0)
    return dist


def compute_pairwise_distances(
    emb: np.ndarray,
    metric: str = "cosine",
) -> np.ndarray:
    """Compute pairwise distance matrix using the specified metric.

    Supported metrics: cosine, euclidean, manhattan, correlation.

    Returns (n, n) float32 distance matrix with zeros on diagonal.
    """
    from scipy.spatial.distance import pdist, squareform

    emb = emb.astype(np.float64)

    if metric == "cosine":
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        normed = emb / np.maximum(norms, 1e-10)
        cos_sim = normed @ normed.T
        np.clip(cos_sim, -1.0, 1.0, out=cos_sim)
        dist = 1.0 - cos_sim
    elif metric in ("euclidean", "manhattan", "correlation"):
        scipy_name = {"manhattan": "cityblock"}.get(metric, metric)
        dist = squareform(pdist(emb, metric=scipy_name))
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

    dist = dist.astype(np.float32)
    np.fill_diagonal(dist, 0.0)
    return dist


def compute_functional_distances(
    df: pd.DataFrame,
    substrate_names: list,
) -> np.ndarray:
    """Compute pairwise functional distances between substrates.

    For each pair (i, j), computes the mean |Δlog_fc| across all mutations
    measured on both substrates.  Returns (n, n) float32 matrix with NaN
    for pairs with no shared mutations.
    """
    n = len(substrate_names)
    dist = np.full((n, n), np.nan, dtype=np.float32)
    sub_to_idx = {name: i for i, name in enumerate(substrate_names)}

    # Build mutation -> {substrate: log_fc} lookup
    mut_sub_fc: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        mut = row["mutation_string"]
        sub = row["substrate"]
        if mut not in mut_sub_fc:
            mut_sub_fc[mut] = {}
        mut_sub_fc[mut][sub] = float(row["log_fc"])

    for i in range(n):
        for j in range(i + 1, n):
            si, sj = substrate_names[i], substrate_names[j]
            deltas = []
            for mut, subs in mut_sub_fc.items():
                if si in subs and sj in subs:
                    deltas.append(abs(subs[si] - subs[sj]))
            if deltas:
                d = float(np.mean(deltas))
                dist[i, j] = d
                dist[j, i] = d

    np.fill_diagonal(dist, 0.0)
    return dist


def select_best_substrate_metric(
    df: pd.DataFrame,
    embeddings: dict,
    embedding_types: Optional[List[str]] = None,
    distance_metrics: Optional[List[str]] = None,
) -> dict:
    """Auto-select the (embedding_type, distance_metric) combination whose
    pairwise distances best correlate with actual functional distances
    between substrates.

    Functional distance = mean |Δlog_fc| across shared mutations.

    Returns dict with keys:
        best_embedding: str
        best_metric: str
        best_correlation: float
        all_results: list of (embedding, metric, spearman_rho, pvalue) tuples
        functional_distances: np.ndarray
    """
    if embedding_types is None:
        embedding_types = ["morgan", "maccs", "mordred", "molformer"]
    if distance_metrics is None:
        distance_metrics = ["cosine", "euclidean", "manhattan", "correlation"]

    substrate_names = embeddings["substrate_names"]

    # Compute functional distances from experimental data
    func_dist = compute_functional_distances(df, substrate_names)

    # Extract upper triangle, excluding NaN pairs
    n = len(substrate_names)
    triu_i, triu_j = np.triu_indices(n, k=1)
    func_vec = func_dist[triu_i, triu_j]
    valid = ~np.isnan(func_vec)
    func_valid = func_vec[valid]

    if len(func_valid) < 3:
        logger.warning("Too few substrate pairs with shared mutations for metric "
                        "selection; defaulting to molformer/cosine")
        return {
            "best_embedding": "molformer",
            "best_metric": "cosine",
            "best_correlation": 0.0,
            "all_results": [],
            "functional_distances": func_dist,
        }

    results = []
    best_corr = -1.0
    best_emb = "molformer"
    best_met = "cosine"

    for emb_type in embedding_types:
        key = f"substrate_{emb_type}"
        if key not in embeddings:
            continue
        emb = embeddings[key].astype(np.float64)

        for metric in distance_metrics:
            try:
                dist = compute_pairwise_distances(emb, metric)
                dist_vec = dist[triu_i, triu_j][valid]
                rho, pval = stats.spearmanr(func_valid, dist_vec)
                rho = float(rho)
                results.append((emb_type, metric, rho, float(pval)))
                if rho > best_corr:
                    best_corr = rho
                    best_emb = emb_type
                    best_met = metric
            except Exception as e:
                logger.warning("Metric selection failed for %s/%s: %s",
                               emb_type, metric, e)

    logger.info("Substrate metric auto-selection (Spearman vs functional distance):")
    for emb, met, rho, pval in sorted(results, key=lambda x: -x[2]):
        marker = " <-- best" if emb == best_emb and met == best_met else ""
        logger.info("  %-10s / %-12s: rho=%.3f (p=%.3e)%s", emb, met, rho, pval, marker)

    return {
        "best_embedding": best_emb,
        "best_metric": best_met,
        "best_correlation": best_corr,
        "all_results": results,
        "functional_distances": func_dist,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Matched-pair Null Models (per-mode)
#
# Each function below maps one row of ``agg_df`` (a test (mutation, target) pair
# from the held-out validation set) to a scalar null prediction in log_fc space,
# using only information contained in ``df_train`` (the training slice of the
# current fold). Every null falls back to the global training mean when the
# per-row lookup is not resolvable.
# ═══════════════════════════════════════════════════════════════════════════

FORMALDEHYDE_SUBSTRATE = "Formaldehyde"


def _build_mutation_substrate_lookup(df_train: pd.DataFrame) -> dict:
    """(mutation_string, substrate) → log_fc lookup over training rows."""
    lookup: dict = {}
    for _, row in df_train.iterrows():
        lookup[(row["mutation_string"], row["substrate"])] = float(row["log_fc"])
    return lookup


def compute_null_formaldehyde(
    agg_df: pd.DataFrame,
    df_train: pd.DataFrame,
    formaldehyde_substrate: str = FORMALDEHYDE_SUBSTRATE,
) -> np.ndarray:
    """Null: look up each test mutation's log_fc on Formaldehyde in training data.

    Rows whose target substrate IS Formaldehyde return NaN — callers must mask
    these out before scoring (otherwise the null becomes trivially perfect).
    Falls back to the global training mean when the mutation is absent on
    Formaldehyde training rows.
    """
    mean_pred = float(df_train["log_fc"].values.mean())
    form_lookup: dict = {}
    for _, row in df_train[df_train["substrate"] == formaldehyde_substrate].iterrows():
        form_lookup[row["mutation_string"]] = float(row["log_fc"])

    preds = np.empty(len(agg_df), dtype=np.float32)
    n_found = 0
    for i, (_, row) in enumerate(agg_df.iterrows()):
        if row["substrate"] == formaldehyde_substrate:
            preds[i] = np.nan
            continue
        val = form_lookup.get(row["mutation_string"])
        if val is None:
            preds[i] = mean_pred
        else:
            preds[i] = val
            n_found += 1
    logger.info("  Null(formaldehyde): %d/%d test rows matched on formaldehyde "
                "(fallback to training mean for the rest)", n_found, len(agg_df))
    return preds


def compute_null_nearest(
    agg_df: pd.DataFrame,
    df_train: pd.DataFrame,
    embeddings: dict,
    substrate_embedding_type: str = "molformer",
    distance_metric: str = "cosine",
) -> np.ndarray:
    """Null: look up each test mutation on the nearest training substrate.

    Nearest is computed once per unique target substrate, in the substrate
    embedding space chosen by the caller (via ``select_best_substrate_metric``).
    Falls back to the global training mean when the mutation is absent on the
    nearest training substrate.
    """
    mean_pred = float(df_train["log_fc"].values.mean())

    substrate_names = embeddings["substrate_names"]
    sub_to_idx = {name: i for i, name in enumerate(substrate_names)}

    emb = embeddings[f"substrate_{substrate_embedding_type}"].astype(np.float64)
    dist_matrix = compute_pairwise_distances(emb, distance_metric)

    train_lookup = _build_mutation_substrate_lookup(df_train)
    train_subs = list(df_train["substrate"].unique())

    nearest_map: dict = {}
    for val_sub in agg_df["substrate"].unique():
        if val_sub not in sub_to_idx:
            nearest_map[val_sub] = None
            continue
        val_idx = sub_to_idx[val_sub]
        best_dist, best_sub = float("inf"), None
        for train_sub in train_subs:
            if train_sub == val_sub or train_sub not in sub_to_idx:
                continue
            d = float(dist_matrix[val_idx, sub_to_idx[train_sub]])
            if d < best_dist:
                best_dist, best_sub = d, train_sub
        nearest_map[val_sub] = best_sub
        if best_sub is not None:
            logger.info("  Null(nearest): %s → %s (dist=%.3f, %s/%s)",
                        val_sub, best_sub, best_dist,
                        substrate_embedding_type, distance_metric)

    preds = np.empty(len(agg_df), dtype=np.float32)
    for i, (_, row) in enumerate(agg_df.iterrows()):
        nearest = nearest_map.get(row["substrate"])
        if nearest is None:
            preds[i] = mean_pred
        else:
            preds[i] = train_lookup.get(
                (row["mutation_string"], nearest), mean_pred)
    return preds


def compute_null_avg_all(
    agg_df: pd.DataFrame,
    df_train: pd.DataFrame,
) -> np.ndarray:
    """Null: mean of log_fc across training substrates where mutation is observed.

    For each test row (mutation, target), collect the log_fc values of the same
    mutation on every training substrate (excluding the target itself, though
    in substrate-split the target is already absent from training). Average
    over the substrates where the mutation was actually observed. Fall back to
    the global training mean if the mutation is absent on every training
    substrate.
    """
    mean_pred = float(df_train["log_fc"].values.mean())

    mut_sub_logfc: Dict[str, Dict[str, float]] = {}
    for _, row in df_train.iterrows():
        mut = row["mutation_string"]
        sub = row["substrate"]
        mut_sub_logfc.setdefault(mut, {})[sub] = float(row["log_fc"])

    preds = np.empty(len(agg_df), dtype=np.float32)
    n_found = 0
    for i, (_, row) in enumerate(agg_df.iterrows()):
        mut = row["mutation_string"]
        target = row["substrate"]
        if mut in mut_sub_logfc:
            observed = {
                sub: val for sub, val in mut_sub_logfc[mut].items()
                if sub != target
            }
            if observed:
                preds[i] = float(np.mean(list(observed.values())))
                n_found += 1
                continue
        preds[i] = mean_pred
    logger.info("  Null(avg_all): %d/%d test rows had observed refs "
                "(fallback to training mean for the rest)", n_found, len(agg_df))
    return preds


def compute_null_distance_weighted(
    agg_df: pd.DataFrame,
    df_train: pd.DataFrame,
    embeddings: dict,
    substrate_embedding_type: str = "molformer",
    distance_metric: str = "cosine",
    temperature: float = 1.0,
) -> np.ndarray:
    """Null: softmax(-d/τ)-weighted mean of log_fc over training substrates
    where the mutation is observed.

    Weights are computed in substrate-embedding space using the same distance
    as used elsewhere. Missing refs (where the mutation is not observed) are
    dropped before renormalisation, so the returned prediction averages ONLY
    over substrates that actually contain the mutation. Fall back to the
    global training mean when the mutation is absent on every training
    substrate.
    """
    mean_pred = float(df_train["log_fc"].values.mean())

    substrate_names = embeddings["substrate_names"]
    sub_to_idx = {name: i for i, name in enumerate(substrate_names)}

    emb = embeddings[f"substrate_{substrate_embedding_type}"].astype(np.float64)
    dist_matrix = compute_pairwise_distances(emb, distance_metric)

    mut_sub_logfc: Dict[str, Dict[str, float]] = {}
    for _, row in df_train.iterrows():
        mut = row["mutation_string"]
        sub = row["substrate"]
        mut_sub_logfc.setdefault(mut, {})[sub] = float(row["log_fc"])

    preds = np.empty(len(agg_df), dtype=np.float32)
    n_found = 0
    tau = max(float(temperature), 1e-8)
    for i, (_, row) in enumerate(agg_df.iterrows()):
        mut = row["mutation_string"]
        target = row["substrate"]
        target_idx = sub_to_idx.get(target)
        if mut not in mut_sub_logfc or target_idx is None:
            preds[i] = mean_pred
            continue
        observed = [
            (sub, val) for sub, val in mut_sub_logfc[mut].items()
            if sub != target and sub in sub_to_idx
        ]
        if not observed:
            preds[i] = mean_pred
            continue
        subs, vals = zip(*observed)
        dists = np.array(
            [dist_matrix[target_idx, sub_to_idx[s]] for s in subs],
            dtype=np.float64,
        )
        logits = -dists / tau
        logits = logits - logits.max()
        w = np.exp(logits)
        w = w / w.sum()
        preds[i] = float(np.dot(w, np.array(vals, dtype=np.float64)))
        n_found += 1
    logger.info("  Null(distance_weighted, τ=%.3f): %d/%d test rows had observed refs "
                "(fallback to training mean for the rest)", tau, n_found, len(agg_df))
    return preds


def compute_null_for_mode(
    mode: str,
    agg_df: pd.DataFrame,
    df_train: pd.DataFrame,
    embeddings: dict,
    substrate_embedding_type: str = "molformer",
    distance_metric: str = "cosine",
    distance_weight_temperature: float = 1.0,
    formaldehyde_substrate: str = FORMALDEHYDE_SUBSTRATE,
) -> np.ndarray:
    """Dispatch to the matched null for a given BNN reference mode."""
    if mode == "formaldehyde":
        return compute_null_formaldehyde(
            agg_df, df_train, formaldehyde_substrate=formaldehyde_substrate)
    if mode == "nearest":
        return compute_null_nearest(
            agg_df, df_train, embeddings,
            substrate_embedding_type=substrate_embedding_type,
            distance_metric=distance_metric)
    if mode == "avg_all" or mode == "mean":
        return compute_null_avg_all(agg_df, df_train)
    if mode == "distance_weighted":
        return compute_null_distance_weighted(
            agg_df, df_train, embeddings,
            substrate_embedding_type=substrate_embedding_type,
            distance_metric=distance_metric,
            temperature=distance_weight_temperature)
    raise ValueError(f"Unknown null mode: {mode}")


def compute_ndcg(y_true: np.ndarray, y_pred: np.ndarray, k: Optional[int] = None) -> float:
    """Normalized Discounted Cumulative Gain — emphasises getting top ranks right.

    Treats y_true as relevance scores (higher = better). y_true is shifted to
    non-negative (rel = y_true - y_true.min()) before computing DCG/IDCG, so
    the metric is well-defined on continuous data with negative values (e.g.
    log_fc where most mutations are deleterious). Ordering is preserved by
    the shift. When k is given, computes NDCG@k (only top-k predicted
    positions matter). When k is None, uses the full ranking.

    Returns a value in [0, 1]; 1.0 means the predicted ranking is perfect.
    NaN when n<2 or when all y_true values are identical (idcg=0 after shift).
    """
    if len(y_true) < 2:
        return float("nan")

    if k is None:
        k = len(y_true)
    k = min(k, len(y_true))

    rel = y_true - y_true.min()

    y_pred_tb = _break_ties(y_pred, seed=len(y_pred))
    pred_order = np.argsort(y_pred_tb)[::-1][:k]
    ideal_order = np.argsort(rel)[::-1][:k]

    discounts = 1.0 / np.log2(np.arange(2, k + 2))  # 1/log2(2), 1/log2(3), ...

    dcg = float(np.sum(rel[pred_order] * discounts))
    idcg = float(np.sum(rel[ideal_order] * discounts))

    if idcg < 1e-12:
        return float("nan")
    return dcg / idcg


def compute_active_only_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    wt_activity: float,
) -> dict:
    """Compute MAE, Spearman, NDCG on truly active mutations only.

    Filters to mutations where y_true > wt_activity (i.e. the mutation is
    beneficial, regardless of which substrate it's on). This focuses
    evaluation on the mutations that matter for engineering: does the model
    get the magnitude and ranking right for mutations worth pursuing?

    Returns dict with active_mae, active_spearman, active_ndcg, active_n.
    """
    mask = y_true > wt_activity
    yt = y_true[mask]
    yp = y_pred[mask]
    n = int(mask.sum())

    if n < 5:
        return {"active_mae": float("nan"), "active_spearman": float("nan"),
                "active_ndcg": float("nan"), "active_n": n}

    mae = float(np.mean(np.abs(yt - yp)))
    rho, _ = stats.spearmanr(yt, yp) if np.std(yt) > 1e-8 and np.std(yp) > 1e-8 else (float("nan"), 1.0)
    ndcg = compute_ndcg(yt, yp)

    return {"active_mae": mae, "active_spearman": float(rho),
            "active_ndcg": ndcg, "active_n": n}


def compute_active_substrate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    substrate_labels: np.ndarray,
    active_substrates: set,
) -> dict:
    """Compute MAE, Spearman, NDCG on active substrates only (all mutations).

    Excludes the 3 synthetic all-zero inactive substrates but keeps ALL
    mutations on active substrates (including inactive mutations on active
    substrates). Complements compute_active_only_metrics which filters by
    mutation activity.

    Returns dict with active_sub_mae, active_sub_spearman, active_sub_ndcg,
    active_sub_n.
    """
    mask = np.isin(substrate_labels, list(active_substrates))
    yt = y_true[mask]
    yp = y_pred[mask]
    n = int(mask.sum())

    if n < 5:
        return {"active_sub_mae": float("nan"), "active_sub_spearman": float("nan"),
                "active_sub_ndcg": float("nan"), "active_sub_n": n}

    mae = float(np.mean(np.abs(yt - yp)))
    rho, _ = stats.spearmanr(yt, yp) if np.std(yt) > 1e-8 and np.std(yp) > 1e-8 else (float("nan"), 1.0)
    ndcg = compute_ndcg(yt, yp)

    return {"active_sub_mae": mae, "active_sub_spearman": float(rho),
            "active_sub_ndcg": ndcg, "active_sub_n": n}


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    wt_activity: float,
) -> dict:
    """Binary classification metrics: can the model distinguish active from inactive mutations?

    Thresholds both y_true and y_pred at wt_activity. Reports precision, recall,
    F1, and the false positive rate (fraction of truly inactive mutations the
    model calls active).

    Returns dict with precision, recall, f1, fpr, n_true_active, n_pred_active.
    """
    true_active = y_true > wt_activity
    pred_active = y_pred > wt_activity

    tp = int(np.sum(true_active & pred_active))
    fp = int(np.sum(~true_active & pred_active))
    fn = int(np.sum(true_active & ~pred_active))
    tn = int(np.sum(~true_active & ~pred_active))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    fpr = fp / max(fp + tn, 1)  # false positive rate

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "n_true_active": int(np.sum(true_active)),
        "n_pred_active": int(np.sum(pred_active)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


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


def compute_per_substrate_topk_recovery(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    substrate_labels: np.ndarray,
    k: int = 5,
) -> dict:
    """Per-substrate top-k overlap: what fraction of true top-k are in predicted top-k?

    Returns {substrate: {"recovery": float, "n": int, "k_used": int}}.
    Only substrates with enough variance and at least k samples are included.
    """
    results = {}
    for sub in sorted(set(substrate_labels)):
        mask = substrate_labels == sub
        yt = y_true[mask]
        yp = _break_ties(y_pred[mask], seed=int(mask.sum()))
        n = int(mask.sum())

        if n < k or np.std(yt) < 1e-8:
            continue

        k_used = min(k, n)
        true_topk = set(np.argsort(yt)[-k_used:])
        pred_topk = set(np.argsort(yp)[-k_used:])
        overlap = len(true_topk & pred_topk) / k_used

        results[sub] = {"recovery": overlap, "n": n, "k_used": k_used}

    return results


def compute_substrate_discrimination(
    y_pred: np.ndarray,
    substrate_labels: np.ndarray,
    active_substrates: set,
) -> dict:
    """Can the model's mean prediction per substrate distinguish active vs inactive?

    Returns dict with auroc, per-substrate mean predictions, and labels.
    """
    sub_means = {}
    for sub in sorted(set(substrate_labels)):
        mask = substrate_labels == sub
        sub_means[sub] = float(np.mean(y_pred[mask]))

    subs = sorted(sub_means.keys())
    if len(subs) < 2:
        return {"auroc": float("nan"), "per_substrate_mean_pred": sub_means,
                "n_active": 0, "n_inactive": 0}

    labels = np.array([1 if s in active_substrates else 0 for s in subs])
    scores = np.array([sub_means[s] for s in subs])

    n_active = int(labels.sum())
    n_inactive = len(labels) - n_active

    if n_active == 0 or n_inactive == 0:
        return {"auroc": float("nan"), "per_substrate_mean_pred": sub_means,
                "n_active": n_active, "n_inactive": n_inactive}

    # Manual AUROC (no sklearn dependency): fraction of (active, inactive) pairs
    # where active has higher predicted mean
    n_correct = 0
    n_pairs = 0
    for i, s_i in enumerate(subs):
        for j, s_j in enumerate(subs):
            if labels[i] == 1 and labels[j] == 0:
                n_pairs += 1
                if scores[i] > scores[j]:
                    n_correct += 1
                elif scores[i] == scores[j]:
                    n_correct += 0.5

    auroc = n_correct / max(n_pairs, 1)

    return {"auroc": auroc, "per_substrate_mean_pred": sub_means,
            "n_active": n_active, "n_inactive": n_inactive}


def range_weighted_mean(per_group_metric: dict, per_group_range: dict) -> float:
    """Compute range-weighted mean: weight each group's metric by its activity range."""
    common = set(per_group_metric) & set(per_group_range)
    if not common:
        return 0.0
    values = np.array([per_group_metric[g] for g in common])
    weights = np.array([per_group_range[g] for g in common])
    total_w = weights.sum()
    if total_w < 1e-10:
        return float(np.mean(values))
    return float(np.average(values, weights=weights))


def compute_above_floor_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    floor_threshold: float = -1.99,
) -> dict:
    """Compute MAE and Spearman on non-floor samples only.

    Filters to samples where y_true > floor_threshold, then computes:
    - MAE
    - Spearman correlation (if enough samples with variance)
    - NDCG@10 (if enough samples)
    - Count / fraction of above-floor samples

    Always returned (no config gating).
    """
    mask = y_true > floor_threshold
    n_above = int(mask.sum())
    n_total = len(y_true)

    result = {
        "n_above_floor": n_above,
        "n_total": n_total,
        "frac_above_floor": n_above / max(n_total, 1),
    }

    if n_above < 2:
        result["mae"] = float("nan")
        result["spearman"] = float("nan")
        result["ndcg10"] = float("nan")
        return result

    yt = y_true[mask]
    yp = y_pred[mask]

    result["mae"] = float(np.mean(np.abs(yt - yp)))

    if np.std(yt) > 1e-8 and np.std(yp) > 1e-8:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(yt, yp)
        result["spearman"] = float(rho)
    else:
        result["spearman"] = float("nan")

    # NDCG@10
    if n_above >= 10:
        k = min(10, n_above)
        true_order = np.argsort(yt)[::-1]
        pred_order = np.argsort(_break_ties(yp, seed=len(yp)))[::-1]
        # Relevance: rank-based — top item gets highest relevance
        relevance = np.zeros(n_above)
        for rank, idx in enumerate(true_order):
            relevance[idx] = n_above - rank  # higher = more relevant
        # DCG
        dcg = sum(relevance[pred_order[i]] / np.log2(i + 2) for i in range(k))
        idcg = sum(relevance[true_order[i]] / np.log2(i + 2) for i in range(k))
        result["ndcg10"] = float(dcg / max(idcg, 1e-10))
    else:
        result["ndcg10"] = float("nan")

    return result


def compute_selection_regret(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k_values: Optional[List[int]] = None,
) -> dict:
    """Fraction of optimal cumulative activity captured by model's top-k selections.

    For each k, computes:
      - sum of true activities for model's top-k by predicted value
      - sum of true activities for oracle's top-k by true value
      - pct_of_optimal = model_sum / oracle_sum

    Always returned (no config gating).
    """
    if k_values is None:
        k_values = [5, 10, 25, 50]

    n = len(y_true)
    oracle_order = np.argsort(y_true)[::-1]
    pred_order = np.argsort(_break_ties(y_pred, seed=len(y_pred)))[::-1]

    result = {}
    for k in k_values:
        k_used = min(k, n)
        if k_used == 0:
            continue
        oracle_sum = float(y_true[oracle_order[:k_used]].sum())
        model_sum = float(y_true[pred_order[:k_used]].sum())
        pct = model_sum / oracle_sum if abs(oracle_sum) > 1e-10 else float("nan")
        result[f"top{k}_pct_of_optimal"] = pct
        result[f"top{k}_model_sum"] = model_sum
        result[f"top{k}_oracle_sum"] = oracle_sum

    return result


def compute_hurdle_metrics(
    y_true: np.ndarray,
    cls_prob: np.ndarray,
    floor_threshold: float = -1.99,
    inference_threshold: float = 0.5,
) -> dict:
    """Classification metrics for the hurdle component.

    Computes accuracy, AUROC, precision, recall for floor vs active prediction.
    Only meaningful when hurdle is enabled.

    Args:
        y_true:              True log_fc values.
        cls_prob:            Predicted P(active) from hurdle model.
        floor_threshold:     y <= this is floor.
        inference_threshold: P(active) > this is classified as active.
    """
    is_active_true = (y_true > floor_threshold).astype(int)
    is_active_pred = (cls_prob > inference_threshold).astype(int)

    n = len(y_true)
    tp = int(((is_active_pred == 1) & (is_active_true == 1)).sum())
    fp = int(((is_active_pred == 1) & (is_active_true == 0)).sum())
    fn = int(((is_active_pred == 0) & (is_active_true == 1)).sum())
    tn = int(((is_active_pred == 0) & (is_active_true == 0)).sum())

    accuracy = (tp + tn) / max(n, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    result = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_floor_true": int(is_active_true.sum() == 0) and tn + fp or tn + fp,
        "n_active_true": tp + fn,
    }

    # Manual AUROC for floor/active
    n_pos = int(is_active_true.sum())
    n_neg = n - n_pos
    if n_pos > 0 and n_neg > 0:
        # Sort by descending cls_prob, count concordant pairs
        sorted_idx = np.argsort(-cls_prob)
        sorted_labels = is_active_true[sorted_idx]
        # Efficient AUROC via rank sum
        ranks = np.empty(n)
        ranks[sorted_idx] = np.arange(1, n + 1, dtype=float)
        # Handle ties
        pos_rank_sum = ranks[is_active_true == 1].sum()
        auroc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        result["auroc"] = float(auroc)
    else:
        result["auroc"] = float("nan")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def _get_substrate_colors(substrates, substrate_names=None):
    """Consistent color mapping for substrates."""
    if substrate_names is None:
        substrate_names = sorted(set(substrates))
    color_map = {}
    import matplotlib.cm as cm
    cmap = cm.get_cmap("tab10", len(substrate_names))
    for i, name in enumerate(substrate_names):
        color_map[name] = cmap(i)
    return color_map


def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    total_std: np.ndarray,
    substrates: np.ndarray,
    output_path: Path,
):
    """Predicted vs actual, colored by substrate, with error bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    color_map = _get_substrate_colors(substrates)

    for sub in sorted(set(substrates)):
        mask = substrates == sub
        ax.scatter(y_true[mask], y_pred[mask], c=[color_map[sub]], s=28,
                   alpha=0.7, label=sub,
                   edgecolors="black", linewidths=0.6, rasterized=True)
    ax.errorbar(y_true, y_pred, yerr=total_std, fmt="none",
                ecolor="gray", alpha=0.12, linewidth=0.8)

    lims = [min(y_true.min(), y_pred.min()) - 0.2,
            max(y_true.max(), y_pred.max()) + 0.2]
    ax.plot(lims, lims, "k--", linewidth=2.0, alpha=0.7, label="y = x")

    residuals = y_true - y_pred
    r2 = 1 - np.sum(residuals**2) / max(np.sum((y_true - y_true.mean())**2), 1e-10)
    rho, _ = stats.spearmanr(y_true, y_pred)
    mae = np.mean(np.abs(residuals))

    ax.text(0.05, 0.95,
            f"R² = {r2:.3f}\nSpearman ρ = {rho:.3f}\nMAE = {mae:.3f}",
            transform=ax.transAxes, fontsize=14, fontweight="bold", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85,
                      edgecolor="black", linewidth=1.5))

    ax.set_xlabel("True log₁₀(FC + ε)")
    ax.set_ylabel("Predicted log₁₀(FC + ε)")
    ax.set_title("Parity Plot — Multi-Substrate BNN2 (OOF)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", ncol=2)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    substrates: np.ndarray,
    output_path: Path,
):
    """Residuals vs predicted, colored by substrate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    residuals = y_true - y_pred
    color_map = _get_substrate_colors(substrates)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for sub in sorted(set(substrates)):
        mask = substrates == sub
        axes[0].scatter(y_pred[mask], residuals[mask], c=[color_map[sub]],
                        s=8, alpha=0.4, label=sub, edgecolors="none")
    axes[0].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[0].set_xlabel("Predicted log₁₀(FC + ε)")
    axes[0].set_ylabel("Residual (true − pred)")
    axes[0].set_title("Residuals vs Predicted")
    axes[0].legend(fontsize=6, ncol=2)

    axes[1].hist(residuals, bins=50, color="#2196F3", edgecolor="none", alpha=0.7)
    axes[1].axvline(0, color="black", linewidth=1, linestyle="--")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")
    axes[1].text(0.05, 0.95,
                 f"Mean: {np.mean(residuals):.4f}\nStd: {np.std(residuals):.4f}",
                 transform=axes[1].transAxes, fontsize=9, va="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_calibration(calibration_dict: dict, output_path: Path):
    """Expected vs observed coverage curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    expected = calibration_dict["expected_coverage"]
    observed = calibration_dict["observed_coverage"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Ideal")
    ax.plot(expected, observed, "o-", color="#2196F3", markersize=6)

    for e, o in zip(expected, observed):
        ax.annotate(f"{o:.2f}", (e, o), textcoords="offset points",
                    xytext=(5, 5), fontsize=7)

    ax.set_xlabel("Expected Coverage")
    ax.set_ylabel("Observed Coverage")
    ax.set_title("Calibration Curve")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_uncertainty_vs_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    total_std: np.ndarray,
    substrates: np.ndarray,
    output_path: Path,
):
    """Total uncertainty vs absolute error scatter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    abs_error = np.abs(y_true - y_pred)
    color_map = _get_substrate_colors(substrates)

    fig, ax = plt.subplots(figsize=(7, 6))
    for sub in sorted(set(substrates)):
        mask = substrates == sub
        ax.scatter(total_std[mask], abs_error[mask], c=[color_map[sub]],
                   s=8, alpha=0.4, label=sub, edgecolors="none")

    rho, p = stats.spearmanr(total_std, abs_error)
    ax.text(0.05, 0.95,
            f"Spearman ρ = {rho:.3f} (p={p:.1e})",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("Total σ (predicted uncertainty)")
    ax.set_ylabel("|Error| = |true − pred|")
    ax.set_title("Uncertainty vs Error")
    ax.legend(fontsize=6, ncol=2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_uncertainty_decomposition(
    epistemic_std: np.ndarray,
    aleatoric_std: np.ndarray,
    substrates: np.ndarray,
    output_path: Path,
):
    """Epistemic vs aleatoric uncertainty: scatter + histogram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color_map = _get_substrate_colors(substrates)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter
    for sub in sorted(set(substrates)):
        mask = substrates == sub
        axes[0].scatter(epistemic_std[mask], aleatoric_std[mask],
                        c=[color_map[sub]], s=8, alpha=0.4, label=sub, edgecolors="none")
    axes[0].plot([0, max(epistemic_std.max(), aleatoric_std.max())],
                 [0, max(epistemic_std.max(), aleatoric_std.max())],
                 "k--", alpha=0.3)
    axes[0].set_xlabel("Epistemic σ")
    axes[0].set_ylabel("Aleatoric σ")
    axes[0].set_title("Uncertainty Decomposition")
    axes[0].legend(fontsize=6, ncol=2)

    # Stacked histogram
    total_var = epistemic_std**2 + aleatoric_std**2
    epi_frac = epistemic_std**2 / np.clip(total_var, 1e-10, None)
    axes[1].hist(epi_frac, bins=50, color="#2196F3", edgecolor="none", alpha=0.7)
    axes[1].axvline(0.5, color="black", linewidth=1, linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Epistemic Fraction (σ²_epi / σ²_total)")
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


def plot_training_curves(
    fold_histories: list,
    output_path: Path,
    fold_labels: Optional[List[str]] = None,
):
    """Training + validation loss across folds with dual y-axes.

    Left panel: train ELBO (solid, left axis) vs val NLL (dashed, right axis).
    Val NLL is the early-stopping monitor; val ELBO shown faintly for reference.
    Right panel: KL annealing schedule.

    Args:
        fold_histories: List of TrainingHistory objects, one per fold.
        output_path: Save path.
        fold_labels: Optional names for each fold (e.g. held-out substrate).
            Falls back to "Fold 0", "Fold 1", etc.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Color cycle — same fold = same color, train=solid, val=dashed
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(fold_histories))]

    ax_train = axes[0]
    ax_val = ax_train.twinx()

    for i, h in enumerate(fold_histories):
        label = fold_labels[i] if fold_labels and i < len(fold_labels) else f"Fold {i}"
        c = colors[i]
        ax_train.plot(h.train_loss, color=c, lw=2.5, alpha=0.85, label=f"{label} train")

        # Val NLL (early-stop monitor)
        has_val_nll = hasattr(h, "val_nll") and h.val_nll
        if has_val_nll:
            ax_val.plot(h.val_nll, "--", color=c, lw=2.0, alpha=0.85, label=f"{label} val NLL")
        elif h.val_loss:
            ax_val.plot(h.val_loss, "--", color=c, lw=2.0, alpha=0.85, label=f"{label} val")

        if h.best_epoch is not None:
            ax_train.axvline(h.best_epoch, color=c, linewidth=1.5,
                             linestyle=":", alpha=0.6)

    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Train ELBO (solid)")
    ax_val.set_ylabel("Val NLL (dashed)")
    ax_train.set_title("Training Curves")

    lines_t, labels_t = ax_train.get_legend_handles_labels()
    lines_v, labels_v = ax_val.get_legend_handles_labels()
    ax_train.legend(lines_t + lines_v, labels_t + labels_v, ncol=2,
                    loc="upper right")

    # KL weight schedule (same across folds, show fold 0)
    if fold_histories and fold_histories[0].kl_weight_schedule:
        axes[1].plot(fold_histories[0].kl_weight_schedule, color="#FF9800", lw=3.0)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("KL Weight (β)")
        axes[1].set_title("KL Annealing Schedule")
    else:
        axes[1].text(0.5, 0.5, "No KL schedule data", ha="center", va="center",
                     transform=axes[1].transAxes)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_loss_decomposition(fold_histories: list, output_path: Path):
    """NLL vs KL components across training (fold 0)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not fold_histories:
        return

    h = fold_histories[0]
    fig, ax = plt.subplots(figsize=(8, 5))

    if h.train_nll and h.train_kl:
        epochs = range(len(h.train_nll))
        ax.plot(epochs, h.train_nll, label="NLL", color="#2196F3")
        ax.plot(epochs, h.train_kl, label="KL (raw)", color="#F44336")
        if h.kl_weight_schedule:
            weighted_kl = [kl * beta for kl, beta in zip(h.train_kl, h.kl_weight_schedule)]
            ax.plot(epochs, weighted_kl, label="KL (weighted)", color="#FF9800", linestyle="--")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss Component")
    ax.set_title("Loss Decomposition (Fold 0)")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_per_substrate_metrics(
    per_sub_spearman: dict,
    per_sub_mae: dict,
    substrate_meta: dict,
    output_path: Path,
    null_per_sub_spearman: Optional[dict] = None,
    null_per_sub_mae: Optional[dict] = None,
    null2_per_sub_spearman: Optional[dict] = None,
    null2_per_sub_mae: Optional[dict] = None,
):
    """Grouped bar chart: Spearman + MAE per substrate, active vs inactive highlighted.

    If null_per_sub_spearman / null_per_sub_mae are provided, a lighter hatched
    bar for the null model is shown alongside each model bar.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    substrates = sorted(per_sub_spearman.keys())
    if not substrates:
        return

    show_null = null_per_sub_spearman is not None or null_per_sub_mae is not None
    show_null2 = null2_per_sub_spearman is not None or null2_per_sub_mae is not None
    null_per_sub_spearman = null_per_sub_spearman or {}
    null_per_sub_mae = null_per_sub_mae or {}
    null2_per_sub_spearman = null2_per_sub_spearman or {}
    null2_per_sub_mae = null2_per_sub_mae or {}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Bar widths depend on how many series we have
    n_series = 1 + int(show_null) + int(show_null2)
    if n_series == 1:
        width = 0.6
    elif n_series == 2:
        width = 0.35
    else:
        width = 0.25

    x = np.arange(len(substrates))
    # Center the bar groups
    offsets = np.linspace(-(n_series - 1) * width / 2, (n_series - 1) * width / 2, n_series)
    x_model = x + offsets[0]
    x_null = x + offsets[1] if n_series >= 2 else x
    x_null2 = x + offsets[2] if n_series >= 3 else x

    # Colors: green for active, gray for inactive
    model_colors, null_colors, null2_colors = [], [], []
    for s in substrates:
        is_active = substrate_meta.get(s, {}).get("is_active", True)
        model_colors.append("#4CAF50" if is_active else "#9E9E9E")
        null_colors.append("#A5D6A7" if is_active else "#BDBDBD")
        null2_colors.append("#FFF59D" if is_active else "#E0E0E0")

    # ── Spearman ──
    spearman_vals = [per_sub_spearman.get(s, float("nan")) for s in substrates]
    null_spearman_vals = [null_per_sub_spearman.get(s, float("nan")) for s in substrates]
    null2_spearman_vals = [null2_per_sub_spearman.get(s, float("nan")) for s in substrates]

    axes[0].bar(x_model, spearman_vals, width, color=model_colors,
                edgecolor="black", linewidth=1.2, label="Model")
    if show_null:
        axes[0].bar(x_null, null_spearman_vals, width, color=null_colors,
                    edgecolor="black", linewidth=1.2, hatch="//", label="Null1 (nearest)")
    if show_null2:
        axes[0].bar(x_null2, null2_spearman_vals, width, color=null2_colors,
                    edgecolor="black", linewidth=1.2, hatch="\\\\", label="Null2 (form.)")
    if show_null or show_null2:
        axes[0].legend()

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(substrates, rotation=45, ha="right")
    axes[0].set_ylabel("Spearman ρ")
    axes[0].set_title("Per-Substrate Spearman")
    axes[0].axhline(0, color="black", linewidth=1.5)
    for i, v in enumerate(spearman_vals):
        if not np.isnan(v):
            axes[0].text(x_model[i], v + 0.02, f"{v:.2f}", ha="center",
                         fontsize=10, fontweight="bold")
    if show_null:
        for i, v in enumerate(null_spearman_vals):
            if not np.isnan(v):
                axes[0].text(x_null[i], v + 0.02, f"{v:.2f}", ha="center",
                             fontsize=10, fontweight="bold", color="#333")
    if show_null2:
        for i, v in enumerate(null2_spearman_vals):
            if not np.isnan(v):
                axes[0].text(x_null2[i], v + 0.02, f"{v:.2f}", ha="center",
                             fontsize=10, fontweight="bold", color="#665500")

    # ── MAE ──
    mae_vals = [per_sub_mae.get(s, float("nan")) for s in substrates]
    null_mae_vals = [null_per_sub_mae.get(s, float("nan")) for s in substrates]
    null2_mae_vals = [null2_per_sub_mae.get(s, float("nan")) for s in substrates]

    axes[1].bar(x_model, mae_vals, width, color=model_colors,
                edgecolor="black", linewidth=1.2, label="Model")
    if show_null:
        axes[1].bar(x_null, null_mae_vals, width, color=null_colors,
                    edgecolor="black", linewidth=1.2, hatch="//", label="Null1 (nearest)")
    if show_null2:
        axes[1].bar(x_null2, null2_mae_vals, width, color=null2_colors,
                    edgecolor="black", linewidth=1.2, hatch="\\\\", label="Null2 (form.)")
    if show_null or show_null2:
        axes[1].legend()

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(substrates, rotation=45, ha="right")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("Per-Substrate MAE")
    for i, v in enumerate(mae_vals):
        if not np.isnan(v):
            axes[1].text(x_model[i], v + 0.01, f"{v:.3f}", ha="center",
                         fontsize=9, fontweight="bold")
    if show_null:
        for i, v in enumerate(null_mae_vals):
            if not np.isnan(v):
                axes[1].text(x_null[i], v + 0.01, f"{v:.3f}", ha="center",
                             fontsize=9, fontweight="bold", color="#333")
    if show_null2:
        for i, v in enumerate(null2_mae_vals):
            if not np.isnan(v):
                axes[1].text(x_null2[i], v + 0.01, f"{v:.3f}", ha="center",
                             fontsize=9, fontweight="bold", color="#665500")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_per_position_metrics(
    per_pos_spearman: dict,
    per_pos_range: dict,
    output_path: Path,
    supp_positions: Optional[set] = None,
):
    """Bar chart of per-position Spearman, colored by activity range."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    positions = sorted(per_pos_spearman.keys())
    if not positions:
        return

    spearman_vals = [per_pos_spearman[p] for p in positions]
    range_vals = [per_pos_range.get(p, 0) for p in positions]

    # Normalize range for color mapping
    range_arr = np.array(range_vals)
    if range_arr.max() > range_arr.min():
        norm_range = (range_arr - range_arr.min()) / (range_arr.max() - range_arr.min())
    else:
        norm_range = np.ones_like(range_arr) * 0.5
    cmap = cm.get_cmap("YlOrRd")
    colors = [cmap(v) for v in norm_range]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(positions))
    ax.bar(x, spearman_vals, color=colors, edgecolor="black", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in positions], rotation=45, ha="right")
    _style_supp_ticklabels(ax, supp_positions, axis="x")
    ax.set_xlabel("Position (0-indexed)")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Per-Position Spearman (color = activity range)")
    ax.axhline(0, color="black", linewidth=1.5)

    for i, (v, r) in enumerate(zip(spearman_vals, range_vals)):
        ax.text(i, v + 0.02, f"{v:.2f}\n(σ={r:.2f})", ha="center",
                fontsize=9, fontweight="bold")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(range_arr.min(), range_arr.max()))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Activity Range (std of y_true)")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_substrate_position_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    substrates: np.ndarray,
    positions: np.ndarray,
    metric_name: str,
    output_path: Path,
    supp_positions: Optional[set] = None,
):
    """Heatmap of per-(substrate, position) metric."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    unique_subs = sorted(set(substrates))
    unique_pos = sorted(set(positions))
    matrix = np.full((len(unique_subs), len(unique_pos)), np.nan)

    for i, sub in enumerate(unique_subs):
        for j, pos in enumerate(unique_pos):
            mask = (substrates == sub) & (positions == pos)
            if mask.sum() < 3:
                continue
            yt = y_true[mask]
            yp = y_pred[mask]
            if metric_name == "spearman":
                if np.std(yt) > 1e-8 and np.std(yp) > 1e-8:
                    rho, _ = stats.spearmanr(yt, yp)
                    matrix[i, j] = rho
            elif metric_name == "mae":
                matrix[i, j] = np.mean(np.abs(yt - yp))

    fig, ax = plt.subplots(figsize=(12, 6))
    if metric_name == "spearman":
        cmap = "RdYlGn"
        vmin, vmax = -1, 1
    else:
        cmap = "YlOrRd"
        vmin = np.nanmin(matrix) if not np.all(np.isnan(matrix)) else 0
        vmax = np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 1

    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(unique_pos)))
    ax.set_xticklabels([str(p) for p in unique_pos], rotation=45, ha="right")
    _style_supp_ticklabels(ax, supp_positions, axis="x")
    ax.set_yticks(range(len(unique_subs)))
    ax.set_yticklabels(unique_subs)
    ax.set_xlabel("Position")
    ax.set_ylabel("Substrate")
    ax.set_title(f"Per-(Substrate, Position) {metric_name.title()}")

    # Annotate cells
    for i in range(len(unique_subs)):
        for j in range(len(unique_pos)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_substrate_parity_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    total_std: np.ndarray,
    substrates: np.ndarray,
    output_path: Path,
):
    """3×3 grid of parity plots, one per substrate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unique_subs = sorted(set(substrates))
    n_subs = len(unique_subs)
    ncols = 3
    nrows = (n_subs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.atleast_2d(axes)

    for idx, sub in enumerate(unique_subs):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        mask = substrates == sub

        yt = y_true[mask]
        yp = y_pred[mask]
        ts = total_std[mask]

        ax.scatter(yt, yp, s=10, alpha=0.5, edgecolors="none", rasterized=True)
        ax.errorbar(yt, yp, yerr=ts, fmt="none", ecolor="gray", alpha=0.1, linewidth=0.5)

        lims = [min(yt.min(), yp.min()) - 0.2, max(yt.max(), yp.max()) + 0.2]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5)

        residuals = yt - yp
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        rho = stats.spearmanr(yt, yp)[0] if len(yt) > 2 else 0

        ax.text(0.05, 0.95, f"R²={r2:.2f}\nρ={rho:.2f}\nn={mask.sum()}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_title(sub, fontsize=10)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")

    # Hide unused axes
    for idx in range(n_subs, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.supxlabel("True log₁₀(FC + ε)", fontsize=12)
    fig.supylabel("Predicted log₁₀(FC + ε)", fontsize=12)
    fig.suptitle("Per-Substrate Parity Plots", fontsize=14, y=1.01)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_substrate_parity_comparison_grid(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_null: np.ndarray,
    substrates: np.ndarray,
    output_path: Path,
    y_pred_null2: Optional[np.ndarray] = None,
):
    """3×3 grid comparing BNN predictions vs null model(s) per substrate.

    Each true data point shows markers connected by a vertical line:
    a blue dot for BNN, red open circle for null1, and (optionally) a
    green diamond for null2 (formaldehyde).  Spearman ρ and MAE are shown.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unique_subs = sorted(set(substrates))
    n_subs = len(unique_subs)
    ncols = 3
    nrows = (n_subs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.atleast_2d(axes)

    for idx, sub in enumerate(unique_subs):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        mask = substrates == sub

        yt = y_true[mask]
        yp_m = y_pred_model[mask]
        yp_n = y_pred_null[mask]
        yp_n2 = y_pred_null2[mask] if y_pred_null2 is not None else None

        # Connecting lines (vertical segments from null → model)
        for xi, ym, yn in zip(yt, yp_m, yp_n):
            ax.plot([xi, xi], [yn, ym], color="gray", linewidth=0.6, alpha=0.4,
                    zorder=1)

        # Null2 dots (open green diamonds) — draw first so they're behind
        if yp_n2 is not None:
            ax.scatter(yt, yp_n2, s=20, facecolors="none", edgecolors="#FF9800",
                       linewidths=0.8, alpha=0.7, marker="D",
                       label="Null2(form)", zorder=2, rasterized=True)
        # Null dots (open red circles)
        ax.scatter(yt, yp_n, s=18, facecolors="none", edgecolors="#F44336",
                   linewidths=0.8, alpha=0.7, label="Null1", zorder=2,
                   rasterized=True)
        # Model dots (filled blue)
        ax.scatter(yt, yp_m, s=18, c="#2196F3", edgecolors="none", alpha=0.6,
                   label="BNN", zorder=3, rasterized=True)

        # Diagonal
        all_vals_list = [yt, yp_m, yp_n]
        if yp_n2 is not None:
            all_vals_list.append(yp_n2)
        all_vals = np.concatenate(all_vals_list)
        lims = [all_vals.min() - 0.2, all_vals.max() + 0.2]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5)

        # Metrics
        rho_m = stats.spearmanr(yt, yp_m)[0] if len(yt) > 2 else float("nan")
        rho_n = stats.spearmanr(yt, yp_n)[0] if len(yt) > 2 else float("nan")
        mae_m = float(np.mean(np.abs(yt - yp_m)))
        mae_n = float(np.mean(np.abs(yt - yp_n)))

        info_text = (f"BNN  ρ={rho_m:.2f}  MAE={mae_m:.3f}\n"
                     f"N1   ρ={rho_n:.2f}  MAE={mae_n:.3f}")
        if yp_n2 is not None:
            rho_n2 = stats.spearmanr(yt, yp_n2)[0] if len(yt) > 2 else float("nan")
            mae_n2 = float(np.mean(np.abs(yt - yp_n2)))
            info_text += f"\nN2   ρ={rho_n2:.2f}  MAE={mae_n2:.3f}"
        info_text += f"\nn={mask.sum()}"

        ax.text(0.05, 0.95, info_text,
                transform=ax.transAxes, fontsize=7, va="top", family="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        ax.set_title(sub, fontsize=10)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")

        if idx == 0:
            ax.legend(fontsize=7, loc="lower right")

    # Hide unused axes
    for idx in range(n_subs, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.supxlabel("True log₁₀(FC + ε)", fontsize=12)
    fig.supylabel("Predicted log₁₀(FC + ε)", fontsize=12)
    fig.suptitle("BNN vs Null — Per-Substrate Parity", fontsize=14, y=1.01)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_distance_vs_performance(
    per_sub_metrics: dict,
    distance_matrix: np.ndarray,
    substrate_names: list,
    held_out_substrates: list,
    train_substrates_per_fold: dict,
    output_path: Path,
    distance_label: str = "Cosine Distance",
    null_per_sub_metrics: Optional[dict] = None,
):
    """Embedding distance to nearest training substrate vs prediction quality.

    Only for substrate split: each held-out substrate gets one point.
    If *null_per_sub_metrics* is provided, null model markers are shown
    alongside the BNN markers connected by a vertical line.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    distances = []
    spearman_vals = []
    mae_vals = []
    null_spearman_vals = []
    null_mae_vals = []
    labels = []

    for sub in held_out_substrates:
        if sub not in per_sub_metrics:
            continue
        sub_idx = substrate_names.index(sub)
        train_subs = train_substrates_per_fold.get(sub, [])
        if not train_subs:
            continue
        train_indices = [substrate_names.index(s) for s in train_subs]
        min_dist = min(distance_matrix[sub_idx, ti] for ti in train_indices)
        distances.append(min_dist)
        spearman_vals.append(per_sub_metrics[sub].get("spearman", 0))
        mae_vals.append(per_sub_metrics[sub].get("mae", 0))
        labels.append(sub)
        if null_per_sub_metrics:
            null_spearman_vals.append(null_per_sub_metrics.get(sub, {}).get("spearman", 0))
            null_mae_vals.append(null_per_sub_metrics.get(sub, {}).get("mae", 0))

    if not distances:
        return

    show_null = bool(null_per_sub_metrics and null_spearman_vals)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Spearman vs distance ──
    if show_null:
        for d, sm, sn in zip(distances, spearman_vals, null_spearman_vals):
            axes[0].plot([d, d], [sn, sm], color="gray", linewidth=1, alpha=0.5,
                         zorder=1)
        axes[0].scatter(distances, null_spearman_vals, s=60, marker="D",
                        facecolors="none", edgecolors="#F44336", linewidths=1.2,
                        label="Null", zorder=2)
    axes[0].scatter(distances, spearman_vals, s=80, c="#2196F3",
                    edgecolors="black", label="BNN", zorder=3)
    for d, s, lab in zip(distances, spearman_vals, labels):
        axes[0].annotate(lab, (d, s), textcoords="offset points",
                         xytext=(8, 4), fontsize=7)
    axes[0].set_xlabel(f"Min {distance_label} to Training Substrates")
    axes[0].set_ylabel("Spearman ρ")
    axes[0].set_title("Generalization vs Substrate Novelty")
    if show_null:
        axes[0].legend(fontsize=8)

    # ── MAE vs distance ──
    if show_null:
        for d, mm, mn in zip(distances, mae_vals, null_mae_vals):
            axes[1].plot([d, d], [mn, mm], color="gray", linewidth=1, alpha=0.5,
                         zorder=1)
        axes[1].scatter(distances, null_mae_vals, s=60, marker="D",
                        facecolors="none", edgecolors="#9E9E9E", linewidths=1.2,
                        label="Null", zorder=2)
    axes[1].scatter(distances, mae_vals, s=80, c="#F44336",
                    edgecolors="black", label="BNN", zorder=3)
    for d, m, lab in zip(distances, mae_vals, labels):
        axes[1].annotate(lab, (d, m), textcoords="offset points",
                         xytext=(8, 4), fontsize=7)
    axes[1].set_xlabel(f"Min {distance_label} to Training Substrates")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("Error vs Substrate Novelty")
    if show_null:
        axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_substrate_transfer_matrix(
    transfer_metrics: dict,
    substrate_names: list,
    metric_name: str,
    output_path: Path,
):
    """Heatmap showing cross-substrate prediction quality.

    transfer_metrics[held_out_sub] = {metric_name: value}
    Diagonal is NaN (can't predict self from self).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(substrate_names)
    matrix = np.full((n, n), np.nan)

    for held_out_sub, metrics in transfer_metrics.items():
        if held_out_sub in substrate_names:
            j = substrate_names.index(held_out_sub)
            val = metrics.get(metric_name, np.nan)
            # Fill the held-out column (all training substrates predict this one)
            for i in range(n):
                if i != j:
                    matrix[i, j] = val  # trained without j, predicting j

    fig, ax = plt.subplots(figsize=(10, 8))
    if metric_name == "spearman":
        cmap = "RdYlGn"
        vmin, vmax = -1, 1
    else:
        cmap = "YlOrRd_r"
        vmin = np.nanmin(matrix) if not np.all(np.isnan(matrix)) else 0
        vmax = np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 1

    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(substrate_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(substrate_names, fontsize=8)
    ax.set_xlabel("Held-Out (Predicted) Substrate")
    ax.set_ylabel("Training Substrates")
    ax.set_title(f"Substrate Transfer Matrix ({metric_name.title()})")

    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_singleshot_distributions(
    repeat_metrics: list,
    output_path: Path,
):
    """Box/violin plot of metrics across single-shot repeats."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_names = ["spearman_rho", "mae", "nlpd", "crps"]
    metric_labels = ["Spearman ρ", "MAE", "NLPD", "CRPS"]

    fig, axes = plt.subplots(1, len(metric_names), figsize=(4 * len(metric_names), 5))

    for i, (mname, mlabel) in enumerate(zip(metric_names, metric_labels)):
        values = [rm.get(mname, 0) for rm in repeat_metrics]
        ax = axes[i]
        bp = ax.boxplot(values, patch_artist=True)
        bp["boxes"][0].set_facecolor("#2196F3")
        bp["boxes"][0].set_alpha(0.5)
        ax.set_ylabel(mlabel)
        ax.set_title(f"{mlabel}\n(n={len(values)} repeats)")
        ax.text(0.5, 0.02,
                f"Mean: {np.mean(values):.3f}\nStd: {np.std(values):.3f}",
                transform=ax.transAxes, ha="center", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.suptitle("Single-Shot Performance Distribution", fontsize=12, y=1.02)
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
    null_pred: Optional[np.ndarray] = None,
    null2_pred: Optional[np.ndarray] = None,
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
              If null_pred is provided, the null model's recovery curve is overlaid.
      Middle: Hit rate vs screening budget — for each budget (top N%), what
              fraction of selected variants have log_fc > WT activity.
      Right:  Activity distribution (KDE) of selected top-5% variants vs all data.

    Args:
        y_true:       True log_fc values (OOF CV predictions)
        y_pred:       Predicted means
        total_std:    Predicted total uncertainties
        wt_activity:  WT log_fc threshold (FC=1; beneficial = above this)
        output_path:  Where to save the figure
        null_pred:    Null model predictions (optional). If provided, null model
                      acquisition curves are overlaid with dashed lines.
        top_k_frac:   Fraction defining "true top" variants for recovery curve
        budget_fracs: Screening budgets for hit rate panel (fractions of dataset)
        ucb_kappas:   UCB exploration parameters (one curve per kappa)
        n_thompson:   Number of Thompson sampling repeats for uncertainty bands
        seed:         RNG seed for Thompson sampling
        title:        Suptitle prefix
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    rng = np.random.default_rng(seed)
    n = len(y_true)

    # Ensure at least 5 samples in the "true top" group
    k = max(5, min(int(round(top_k_frac * n)), int(0.20 * n)))
    actual_frac = k / n

    true_top_idx = set(np.argsort(y_true)[-k:])
    bg_hit_rate = float(np.mean(y_true > wt_activity))
    n_beneficial = int(np.sum(y_true > wt_activity))

    # ── Acquisition scores ─────────────────────────────────────────────────
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

    # ── Helpers ────────────────────────────────────────────────────────────
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
        return float(np.trapz(recalls, fracs)) - 0.5  # vs random diagonal

    def _hit_rate(scores, budget_frac):
        m = max(1, int(round(budget_frac * n)))
        top_idx = np.argsort(scores)[::-1][:m]
        return float(np.mean(y_true[top_idx] > wt_activity))

    # ── Pre-compute BNN acquisition curves ─────────────────────────────────
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

    # ── Pre-compute null model curve (if provided) ─────────────────────────
    null_curve = None
    null_hit_rates = None
    if null_pred is not None:
        null_pred_tb = _break_ties(null_pred, seed=len(null_pred))
        null_fracs, null_recalls = _recovery_curve(null_pred_tb)
        null_dauc = _delta_auc(null_fracs, null_recalls)
        null_curve = (null_fracs, null_recalls, null_dauc)
        null_hit_rates = np.array([_hit_rate(null_pred_tb, bf) for bf in budget_fracs])

    # ── Pre-compute null2 model curve (if provided) ────────────────────────
    null2_curve = None
    null2_hit_rates = None
    if null2_pred is not None:
        null2_pred_tb = _break_ties(null2_pred, seed=len(null2_pred) + 1)
        null2_fracs, null2_recalls = _recovery_curve(null2_pred_tb)
        null2_dauc = _delta_auc(null2_fracs, null2_recalls)
        null2_curve = (null2_fracs, null2_recalls, null2_dauc)
        null2_hit_rates = np.array([_hit_rate(null2_pred_tb, bf) for bf in budget_fracs])

    budget_pcts = np.array([b * 100 for b in budget_fracs])

    # ── Figure ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── Panel 1: Recovery curves ───────────────────────────────────────────
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.35, lw=1.2, label="Random")

    # Null model (dashed, behind BNN curves)
    if null_curve is not None:
        nf, nr, nd = null_curve
        ax.plot(nf, nr, color="tab:purple", lw=2.5, linestyle="--",
                alpha=0.8, label=f"Null1  (ΔAUC={nd:+.3f})")
    if null2_curve is not None:
        n2f, n2r, n2d = null2_curve
        ax.plot(n2f, n2r, color="tab:brown", lw=2.5, linestyle=":",
                alpha=0.8, label=f"Null2(form)  (ΔAUC={n2d:+.3f})")

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

    # ── Panel 2: Hit rate vs screening budget ──────────────────────────────
    ax = axes[1]
    ax.axhline(bg_hit_rate * 100, color="gray", linestyle="--", lw=1.2,
               alpha=0.7,
               label=f"Background  ({bg_hit_rate*100:.1f}%,  n={n_beneficial})")

    # Null model hit rate
    if null_hit_rates is not None:
        ax.plot(budget_pcts, null_hit_rates * 100, color="tab:purple", lw=2.5,
                linestyle="--", marker="s", ms=4, alpha=0.8, label="Null1")
    if null2_hit_rates is not None:
        ax.plot(budget_pcts, null2_hit_rates * 100, color="tab:brown", lw=2.5,
                linestyle=":", marker="^", ms=4, alpha=0.8, label="Null2(form)")

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

    # ── Panel 3: Activity distributions ────────────────────────────────────
    ax = axes[2]
    dist_budget = 0.05
    k_dist = max(5, int(round(dist_budget * n)))
    x_range = np.linspace(y_true.min() - 0.3, y_true.max() + 0.3, 300)

    try:
        kde_all = gaussian_kde(y_true)
        ax.fill_between(x_range, kde_all(x_range), alpha=0.2, color="gray",
                        label="All data")
        ax.plot(x_range, kde_all(x_range), color="gray", lw=1.5, alpha=0.5)
    except Exception:
        pass

    # Null model distribution
    if null_pred is not None:
        null_top_idx = np.argsort(null_pred_tb)[::-1][:k_dist]
        y_null_sel = y_true[null_top_idx]
        if len(y_null_sel) >= 3:
            try:
                kde_null = gaussian_kde(y_null_sel)
                ax.plot(x_range, kde_null(x_range), color="tab:purple", lw=2.5,
                        linestyle="--", alpha=0.8,
                        label=f"Null1  top {dist_budget*100:.0f}%")
            except Exception:
                pass
    if null2_pred is not None:
        null2_top_idx = np.argsort(null2_pred_tb)[::-1][:k_dist]
        y_null2_sel = y_true[null2_top_idx]
        if len(y_null2_sel) >= 3:
            try:
                kde_null2 = gaussian_kde(y_null2_sel)
                ax.plot(x_range, kde_null2(x_range), color="tab:brown", lw=2.5,
                        linestyle=":", alpha=0.8,
                        label=f"Null2(form)  top {dist_budget*100:.0f}%")
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


def plot_per_substrate_acquisition_recovery(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    total_std: np.ndarray,
    substrate_labels: np.ndarray,
    substrate_meta: dict,
    wt_activity: float,
    output_dir: Path,
    null_pred: Optional[np.ndarray] = None,
    null2_pred: Optional[np.ndarray] = None,
    min_samples: int = 20,
    **kwargs,
) -> None:
    """Generate per-substrate acquisition recovery plots.

    Calls ``plot_acquisition_recovery`` once per substrate that has at least
    ``min_samples`` data points and non-zero variance in y_true.  Results are
    saved into *output_dir* (created if needed), one PNG per substrate.

    Inactive substrates (all y_true == floor) are skipped because there are no
    "true top-k" variants to recover.

    Args:
        y_true, y_pred, total_std: Arrays over all (mutation, substrate) pairs.
        substrate_labels:          Substrate name for each row.
        substrate_meta:            Mapping substrate name -> metadata dict.
        wt_activity:               WT log_fc threshold.
        output_dir:                Directory to save per-substrate PNGs.
        null_pred:                 Optional null model predictions.
        min_samples:               Skip substrates with fewer rows.
        **kwargs:                  Forwarded to ``plot_acquisition_recovery``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    active_substrates = {
        s for s, meta in substrate_meta.items() if meta.get("active", True)
    }

    substrates = sorted(set(substrate_labels))
    n_plotted = 0

    for sub in substrates:
        mask = substrate_labels == sub
        n_sub = int(mask.sum())
        if n_sub < min_samples:
            logger.debug("Skipping acquisition plot for %s (n=%d < %d)", sub, n_sub, min_samples)
            continue

        yt = y_true[mask]
        # Skip if no variance (all-inactive substrates)
        if yt.std() < 1e-8:
            logger.debug("Skipping acquisition plot for %s (no variance in y_true)", sub)
            continue

        yp = y_pred[mask]
        ts = total_std[mask]
        np_sub = null_pred[mask] if null_pred is not None else None
        np2_sub = null2_pred[mask] if null2_pred is not None else None

        tag = "active" if sub in active_substrates else "inactive"
        safe_name = sub.replace(" ", "_").replace("/", "_")

        plot_acquisition_recovery(
            yt, yp, ts,
            wt_activity,
            output_dir / f"acquisition_{safe_name}.png",
            null_pred=np_sub,
            null2_pred=np2_sub,
            title=f"{sub} ({tag}, n={n_sub})",
            **kwargs,
        )
        n_plotted += 1

    logger.info("Per-substrate acquisition plots: %d/%d substrates in %s",
                n_plotted, len(substrates), output_dir)


def plot_per_substrate_topk_recovery(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    substrate_labels: np.ndarray,
    substrate_meta: dict,
    output_path: Path,
    null_pred: Optional[np.ndarray] = None,
    null2_pred: Optional[np.ndarray] = None,
    k_values: tuple = (3, 5, 10),
) -> None:
    """Per-substrate top-k recovery for BNN vs null model(s).

    For each substrate with sufficient variance, plots the fraction of the
    true top-k mutations that appear in the model's top-k. Grouped bars
    show BNN (solid) vs null1 (hatched) vs null2 (dotted) at multiple k values.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Collect results per substrate per k
    substrates_with_data = []
    bnn_results = {}
    null_results = {}
    null2_results = {}

    for sub in sorted(set(substrate_labels)):
        mask = substrate_labels == sub
        yt = y_true[mask]
        yp = y_pred[mask]
        n = int(mask.sum())

        if n < max(k_values) or np.std(yt) < 1e-8:
            continue

        substrates_with_data.append(sub)
        bnn_results[sub] = {}
        null_results[sub] = {}
        null2_results[sub] = {}

        for kv in k_values:
            kv_use = min(kv, n)
            true_topk = set(np.argsort(yt)[-kv_use:])
            yp_tb = _break_ties(yp, seed=n)
            pred_topk = set(np.argsort(yp_tb)[-kv_use:])
            bnn_results[sub][kv] = len(true_topk & pred_topk) / kv_use

            if null_pred is not None:
                np_sub = _break_ties(null_pred[mask], seed=n)
                null_topk = set(np.argsort(np_sub)[-kv_use:])
                null_results[sub][kv] = len(true_topk & null_topk) / kv_use

            if null2_pred is not None:
                np2_sub = _break_ties(null2_pred[mask], seed=n + 1)
                null2_topk = set(np.argsort(np2_sub)[-kv_use:])
                null2_results[sub][kv] = len(true_topk & null2_topk) / kv_use

    if not substrates_with_data:
        return

    n_subs = len(substrates_with_data)
    n_k = len(k_values)
    has_null = null_pred is not None
    has_null2 = null2_pred is not None
    n_per_k = 1 + int(has_null) + int(has_null2)

    fig, ax = plt.subplots(figsize=(max(12, n_subs * 1.5), 6))

    # Bar positioning
    total_bar_groups = n_k * n_per_k
    bar_width = 0.8 / total_bar_groups
    x = np.arange(n_subs)

    cmap = plt.cm.Set2
    for ki, kv in enumerate(k_values):
        offset = ki * n_per_k * bar_width - 0.4 + bar_width / 2
        bnn_vals = [bnn_results[sub][kv] for sub in substrates_with_data]
        color = cmap(ki / max(n_k - 1, 1))

        ax.bar(x + offset, bnn_vals, bar_width, color=color,
               edgecolor="black", linewidth=1.2, label=f"BNN top-{kv}")

        if has_null:
            null_vals = [null_results[sub].get(kv, 0) for sub in substrates_with_data]
            ax.bar(x + offset + bar_width, null_vals, bar_width, color=color,
                   edgecolor="black", linewidth=1.2, hatch="//", alpha=0.6,
                   label=f"N1 top-{kv}")

        if has_null2:
            n2_offset = bar_width * (1 + int(has_null))
            null2_vals = [null2_results[sub].get(kv, 0) for sub in substrates_with_data]
            ax.bar(x + offset + n2_offset, null2_vals, bar_width, color=color,
                   edgecolor="black", linewidth=1.2, hatch="\\\\", alpha=0.4,
                   label=f"N2 top-{kv}")

    # Highlight active vs inactive
    for i, sub in enumerate(substrates_with_data):
        is_active = substrate_meta.get(sub, {}).get("is_active", True)
        if not is_active:
            ax.axvspan(i - 0.45, i + 0.45, color="gray", alpha=0.08)

    ax.set_xticks(x)
    ax.set_xticklabels(substrates_with_data, rotation=45, ha="right")
    ax.set_ylabel("Recovery (fraction of true top-k in predicted top-k)")
    ax.set_title("Per-Substrate Top-k Mutation Recovery")
    ax.set_ylim(0, 1.05)
    ax.axhline(0, color="black", lw=1.5)
    ax.legend(ncol=n_k, loc="upper right")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_engineering_value_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    total_std: np.ndarray,
    substrate_labels: np.ndarray,
    substrate_meta: dict,
    output_path: Path,
    null_pred: Optional[np.ndarray] = None,
    null2_pred: Optional[np.ndarray] = None,
    active_only_metrics: Optional[dict] = None,
    null_active_only_metrics: Optional[dict] = None,
    null2_active_only_metrics: Optional[dict] = None,
    ndcg_metrics: Optional[dict] = None,
    substrate_discrimination: Optional[dict] = None,
    classification_bnn: Optional[dict] = None,
    classification_null: Optional[dict] = None,
    classification_null2: Optional[dict] = None,
) -> None:
    """Summary dashboard: engineering-relevant metrics, BNN vs null.

    Five panels (2x3 grid):
      1. Active-mutation metrics comparison (BNN vs null)
      2. Classification (precision, recall, F1, FPR)
      3. NDCG at various k (BNN vs null)
      4. Substrate discrimination (mean pred by substrate, active vs inactive)
      5. Per-substrate Spearman comparison (BNN vs null)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    # ── Panel 1: Active-mutation metrics bar chart ─────────────────────────
    ax = axes[0, 0]
    metric_names = ["MAE", "Spearman", "NDCG"]
    bnn_vals = [
        active_only_metrics.get("active_mae", 0) if active_only_metrics else 0,
        active_only_metrics.get("active_spearman", 0) if active_only_metrics else 0,
        active_only_metrics.get("active_ndcg", 0) if active_only_metrics else 0,
    ]
    null_vals = [
        null_active_only_metrics.get("active_mae", 0) if null_active_only_metrics else 0,
        null_active_only_metrics.get("active_spearman", 0) if null_active_only_metrics else 0,
        null_active_only_metrics.get("active_ndcg", 0) if null_active_only_metrics else 0,
    ]
    null2_vals_p1 = [
        null2_active_only_metrics.get("active_mae", 0) if null2_active_only_metrics else 0,
        null2_active_only_metrics.get("active_spearman", 0) if null2_active_only_metrics else 0,
        null2_active_only_metrics.get("active_ndcg", 0) if null2_active_only_metrics else 0,
    ]

    x_m = np.arange(len(metric_names))
    n_series_p1 = 1 + int(bool(null_active_only_metrics)) + int(bool(null2_active_only_metrics))
    w = 0.8 / n_series_p1
    offsets_p1 = np.linspace(-(n_series_p1 - 1) * w / 2, (n_series_p1 - 1) * w / 2, n_series_p1)
    ax.bar(x_m + offsets_p1[0], bnn_vals, w, color="#2196F3",
           edgecolor="black", linewidth=1.2, label="BNN")
    si = 1
    if null_active_only_metrics:
        ax.bar(x_m + offsets_p1[si], null_vals, w, color="#9E9E9E",
               edgecolor="black", linewidth=1.2, hatch="//", label="Null1")
        si += 1
    if null2_active_only_metrics:
        ax.bar(x_m + offsets_p1[si], null2_vals_p1, w, color="#FFF59D",
               edgecolor="black", linewidth=1.2, hatch="\\\\", label="Null2(form)")
    for i, v in enumerate(bnn_vals):
        if not np.isnan(v):
            ax.text(x_m[i] - w/2, v + 0.02, f"{v:.3f}", ha="center",
                    fontsize=10, fontweight="bold")
    if null_active_only_metrics:
        for i, v in enumerate(null_vals):
            if not np.isnan(v):
                ax.text(x_m[i] + w/2, v + 0.02, f"{v:.3f}", ha="center",
                        fontsize=10, fontweight="bold", color="#333")
    ax.set_xticks(x_m)
    ax.set_xticklabels(metric_names)
    ax.set_title("Active Mutations Only (log_fc > WT)")
    ax.legend()
    n_active = active_only_metrics.get("active_n", "?") if active_only_metrics else "?"
    ax.set_xlabel(f"(n={n_active} truly beneficial mutations)")

    # ── Panel 2: Classification metrics ────────────────────────────────────
    ax = axes[0, 1]
    if classification_bnn:
        clf_names = ["Precision", "Recall", "F1", "FPR"]
        clf_bnn = [classification_bnn.get(k, 0) for k in ["precision", "recall", "f1", "fpr"]]
        clf_null = [classification_null.get(k, 0) for k in ["precision", "recall", "f1", "fpr"]] if classification_null else [0]*4
        clf_null2 = [classification_null2.get(k, 0) for k in ["precision", "recall", "f1", "fpr"]] if classification_null2 else None
        x_c = np.arange(len(clf_names))
        n_clf = 1 + int(bool(classification_null)) + int(clf_null2 is not None)
        w_c = 0.8 / n_clf
        off_c = np.linspace(-(n_clf - 1) * w_c / 2, (n_clf - 1) * w_c / 2, n_clf)
        ax.bar(x_c + off_c[0], clf_bnn, w_c, color="#2196F3",
               edgecolor="black", linewidth=1.2, label="BNN")
        ci = 1
        if classification_null:
            ax.bar(x_c + off_c[ci], clf_null, w_c, color="#9E9E9E",
                   edgecolor="black", linewidth=1.2, hatch="//", label="Null1")
            ci += 1
        if clf_null2 is not None:
            ax.bar(x_c + off_c[ci], clf_null2, w_c, color="#FFF59D",
                   edgecolor="black", linewidth=1.2, hatch="\\\\", label="Null2(form)")
        for i, v in enumerate(clf_bnn):
            ax.text(x_c[i] + off_c[0], v + 0.02, f"{v:.3f}", ha="center",
                    fontsize=10, fontweight="bold")
        ax.set_xticks(x_c)
        ax.set_xticklabels(clf_names)
        ax.set_ylim(0, 1.15)
        ax.legend()
        tp, fp = classification_bnn.get("tp", 0), classification_bnn.get("fp", 0)
        fn, tn = classification_bnn.get("fn", 0), classification_bnn.get("tn", 0)
        ax.set_xlabel(f"BNN: TP={tp} FP={fp} FN={fn} TN={tn}")
    ax.set_title("Active/Inactive Classification")

    # ── Panel 3: NDCG at various k ────────────────────────────────────────
    ax = axes[0, 2]
    if ndcg_metrics:
        k_vals = sorted(ndcg_metrics.get("bnn", {}).keys())
        bnn_ndcg = [ndcg_metrics["bnn"][kk] for kk in k_vals]
        x_k = np.arange(len(k_vals))
        ax.plot(x_k, bnn_ndcg, "o-", color="#2196F3", lw=3.0, ms=10, label="BNN")

        if "null" in ndcg_metrics:
            null_ndcg = [ndcg_metrics["null"][kk] for kk in k_vals]
            ax.plot(x_k, null_ndcg, "s--", color="#9E9E9E", lw=2.5, ms=10, label="Null1")

        if "null2" in ndcg_metrics and ndcg_metrics["null2"]:
            null2_ndcg = [ndcg_metrics["null2"].get(kk, float("nan")) for kk in k_vals]
            ax.plot(x_k, null2_ndcg, "^:", color="#FFC107", lw=2.5, ms=10, label="Null2(form)")

        ax.set_xticks(x_k)
        ax.set_xticklabels([f"k={kk}" for kk in k_vals])
        ax.set_ylabel("NDCG@k")
        ax.set_ylim(0, 1.05)
        ax.legend()
    ax.set_title("Ranking Quality (NDCG)")

    # ── Panel 4: Substrate discrimination ──────────────────────────────────
    ax = axes[1, 0]
    if substrate_discrimination:
        sub_means = substrate_discrimination.get("per_substrate_mean_pred", {})
        auroc = substrate_discrimination.get("auroc", float("nan"))
        subs_sorted = sorted(sub_means.keys())
        colors = []
        for s in subs_sorted:
            is_active = substrate_meta.get(s, {}).get("is_active", True)
            colors.append("#4CAF50" if is_active else "#F44336")
        vals = [sub_means[s] for s in subs_sorted]

        ax.barh(range(len(subs_sorted)), vals, color=colors,
                edgecolor="black", linewidth=1.2)
        ax.set_yticks(range(len(subs_sorted)))
        ax.set_yticklabels(subs_sorted)
        ax.set_xlabel("Mean predicted log_fc")
        ax.axvline(0, color="black", lw=1.5)

        # Add null model means if available
        legend_extras = []
        if null_pred is not None:
            null_disc = compute_substrate_discrimination(
                null_pred, substrate_labels,
                {s for s, m in substrate_meta.items() if m.get("is_active", True)}
            )
            null_means = null_disc.get("per_substrate_mean_pred", {})
            for i, s in enumerate(subs_sorted):
                if s in null_means:
                    ax.plot(null_means[s], i, "D", color="tab:purple", ms=10,
                            markeredgecolor="black", markeredgewidth=1.5, zorder=5)
            legend_extras.append(
                plt.Line2D([0], [0], marker="D", color="tab:purple", ls="",
                           ms=10, markeredgecolor="black", markeredgewidth=1.5,
                           label="Null1 mean"))

        if null2_pred is not None:
            null2_disc = compute_substrate_discrimination(
                null2_pred, substrate_labels,
                {s for s, m in substrate_meta.items() if m.get("is_active", True)}
            )
            null2_means = null2_disc.get("per_substrate_mean_pred", {})
            for i, s in enumerate(subs_sorted):
                if s in null2_means:
                    ax.plot(null2_means[s], i, "*", color="#FFC107", ms=14,
                            markeredgecolor="black", markeredgewidth=1.5, zorder=5)
            legend_extras.append(
                plt.Line2D([0], [0], marker="*", color="#FFC107", ls="",
                           ms=14, markeredgecolor="black", markeredgewidth=1.5,
                           label="Null2 mean"))

        ax.set_title(f"Substrate Discrimination (AUROC={auroc:.3f})")
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor="#4CAF50", edgecolor="black", linewidth=1.2, label="Active"),
            Patch(facecolor="#F44336", edgecolor="black", linewidth=1.2, label="Inactive"),
        ] + legend_extras,
            loc="lower right")
    else:
        ax.set_title("Substrate Discrimination (no data)")

    # ── Panel 5: Per-substrate Spearman, BNN vs null ───────────────────────
    ax = axes[1, 1]
    active_subs = {s for s, m in substrate_meta.items() if m.get("is_active", True)}
    per_sub_rho_bnn = {}
    per_sub_rho_null = {}
    per_sub_rho_null2 = {}
    for sub in sorted(set(substrate_labels)):
        mask = substrate_labels == sub
        yt = y_true[mask]
        if np.std(yt) < 1e-8 or mask.sum() < 5:
            continue
        yp = y_pred[mask]
        rho_bnn, _ = stats.spearmanr(yt, yp)
        per_sub_rho_bnn[sub] = float(rho_bnn)
        if null_pred is not None:
            np_sub = null_pred[mask]
            if np.std(np_sub) > 1e-8:
                rho_null, _ = stats.spearmanr(yt, np_sub)
                per_sub_rho_null[sub] = float(rho_null)
        if null2_pred is not None:
            np2_sub = null2_pred[mask]
            if np.std(np2_sub) > 1e-8:
                rho_null2, _ = stats.spearmanr(yt, np2_sub)
                per_sub_rho_null2[sub] = float(rho_null2)

    subs_plot = sorted(per_sub_rho_bnn.keys())
    if subs_plot:
        x_s = np.arange(len(subs_plot))
        n_series_p5 = 1 + int(bool(per_sub_rho_null)) + int(bool(per_sub_rho_null2))
        w_s = 0.8 / n_series_p5
        offsets_p5 = np.linspace(-(n_series_p5 - 1) * w_s / 2,
                                  (n_series_p5 - 1) * w_s / 2, n_series_p5)
        bnn_rho = [per_sub_rho_bnn[s] for s in subs_plot]
        colors_bnn = ["#4CAF50" if s in active_subs else "#9E9E9E" for s in subs_plot]
        ax.bar(x_s + offsets_p5[0], bnn_rho, w_s, color=colors_bnn,
               edgecolor="black", linewidth=1.2, label="BNN")

        si = 1
        if per_sub_rho_null:
            null_rho = [per_sub_rho_null.get(s, float("nan")) for s in subs_plot]
            colors_null = ["#A5D6A7" if s in active_subs else "#BDBDBD" for s in subs_plot]
            ax.bar(x_s + offsets_p5[si], null_rho, w_s, color=colors_null,
                   edgecolor="black", linewidth=1.2, hatch="//", label="Null1")
            si += 1

        if per_sub_rho_null2:
            null2_rho = [per_sub_rho_null2.get(s, float("nan")) for s in subs_plot]
            ax.bar(x_s + offsets_p5[si], null2_rho, w_s, color="#FFF59D",
                   edgecolor="black", linewidth=1.2, hatch="\\\\", label="Null2(form)")

        for i, v in enumerate(bnn_rho):
            if not np.isnan(v):
                ax.text(x_s[i] + offsets_p5[0], v + 0.02, f"{v:.2f}",
                        ha="center", fontsize=10, fontweight="bold")

        ax.set_xticks(x_s)
        ax.set_xticklabels(subs_plot, rotation=45, ha="right")
        ax.axhline(0, color="black", lw=1.5)
        ax.legend()
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Per-Substrate Ranking (active subs only have data)")

    axes[1, 2].set_visible(False)

    fig.suptitle("Engineering Value Summary — BNN vs Null", fontsize=18, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_hurdle_diagnostics(
    y_true: np.ndarray,
    cls_prob: np.ndarray,
    y_pred: np.ndarray,
    hurdle_metrics: dict,
    floor_threshold: float,
    output_path: Path,
):
    """4-panel hurdle diagnostic plot.

    Panel 1: P(active) histogram, colored by true floor/active.
    Panel 2: P(active) vs y_true scatter.
    Panel 3: Parity plot colored by floor (blue) / active (orange).
    Panel 4: Metrics summary text.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    is_floor = y_true <= floor_threshold
    is_active = ~is_floor

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # ── Panel 1: P(active) histogram ─────────────────────────────────────
    ax = axes[0, 0]
    bins = np.linspace(0, 1, 31)
    if is_floor.any():
        ax.hist(cls_prob[is_floor], bins=bins, alpha=0.6, color="#2196F3",
                label=f"Floor (n={is_floor.sum()})", density=True)
    if is_active.any():
        ax.hist(cls_prob[is_active], bins=bins, alpha=0.6, color="#FF9800",
                label=f"Active (n={is_active.sum()})", density=True)
    ax.set_xlabel("P(active)")
    ax.set_ylabel("Density")
    ax.set_title("Hurdle Classification Distribution")
    ax.legend(fontsize=8)

    # ── Panel 2: P(active) vs y_true ─────────────────────────────────────
    ax = axes[0, 1]
    ax.scatter(y_true[is_floor], cls_prob[is_floor], s=8, alpha=0.4,
               color="#2196F3", label="Floor", zorder=2)
    ax.scatter(y_true[is_active], cls_prob[is_active], s=8, alpha=0.4,
               color="#FF9800", label="Active", zorder=2)
    ax.axhline(0.5, color="grey", ls="--", lw=0.8, alpha=0.7)
    ax.axvline(floor_threshold, color="grey", ls=":", lw=0.8, alpha=0.7)
    ax.set_xlabel("True log_fc")
    ax.set_ylabel("P(active)")
    ax.set_title("Classifier Calibration")
    ax.legend(fontsize=8)

    # ── Panel 3: Parity colored by classification ────────────────────────
    ax = axes[1, 0]
    ax.scatter(y_true[is_floor], y_pred[is_floor], s=8, alpha=0.4,
               color="#2196F3", label="Floor", zorder=2)
    ax.scatter(y_true[is_active], y_pred[is_active], s=8, alpha=0.4,
               color="#FF9800", label="Active", zorder=2)
    lims = [min(y_true.min(), y_pred.min()) - 0.1,
            max(y_true.max(), y_pred.max()) + 0.1]
    ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("True log_fc")
    ax.set_ylabel("Predicted log_fc (mixture)")
    ax.set_title("Parity (hurdle mixture)")
    ax.legend(fontsize=8)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # ── Panel 4: Metrics text ────────────────────────────────────────────
    ax = axes[1, 1]
    ax.axis("off")
    lines = [
        f"AUROC:     {hurdle_metrics.get('auroc', float('nan')):.3f}",
        f"Accuracy:  {hurdle_metrics.get('accuracy', float('nan')):.3f}",
        f"Precision: {hurdle_metrics.get('precision', float('nan')):.3f}",
        f"Recall:    {hurdle_metrics.get('recall', float('nan')):.3f}",
        f"F1:        {hurdle_metrics.get('f1', float('nan')):.3f}",
        "",
        f"TP={hurdle_metrics.get('tp', 0)}  FP={hurdle_metrics.get('fp', 0)}",
        f"FN={hurdle_metrics.get('fn', 0)}  TN={hurdle_metrics.get('tn', 0)}",
        "",
        f"Floor threshold: {floor_threshold:.2f}",
    ]
    ax.text(0.1, 0.9, "\n".join(lines), transform=ax.transAxes,
            fontsize=11, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#f5f5f5", alpha=0.8))
    ax.set_title("Hurdle Metrics")

    fig.suptitle("Hurdle Model Diagnostics", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def _plot_selection_regret_panel(
    ax,
    regret_bnn: dict,
    regret_null: Optional[dict],
    title: str,
    regret_null2: Optional[dict] = None,
):
    """Draw a single selection-regret bar chart on the given axes."""
    k_vals = sorted(
        int(key.replace("top", "").replace("_pct_of_optimal", ""))
        for key in regret_bnn if key.endswith("_pct_of_optimal")
    )
    if not k_vals:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return

    bnn_pcts = [regret_bnn.get(f"top{k}_pct_of_optimal", float("nan")) for k in k_vals]
    x = np.arange(len(k_vals))
    n_series = 1 + int(bool(regret_null)) + int(bool(regret_null2))
    w = 0.8 / n_series
    offsets = np.linspace(-(n_series - 1) * w / 2, (n_series - 1) * w / 2, n_series)

    ax.bar(x + offsets[0], bnn_pcts, w, color="#2196F3", label="BNN", edgecolor="none")

    si = 1
    null_pcts = None
    if regret_null:
        null_pcts = [regret_null.get(f"top{k}_pct_of_optimal", float("nan"))
                     for k in k_vals]
        ax.bar(x + offsets[si], null_pcts, w, color="#9E9E9E", label="Null1",
               edgecolor="none", hatch="//")
        si += 1

    null2_pcts = None
    if regret_null2:
        null2_pcts = [regret_null2.get(f"top{k}_pct_of_optimal", float("nan"))
                      for k in k_vals]
        ax.bar(x + offsets[si], null2_pcts, w, color="#FFF59D", label="Null2(form)",
               edgecolor="none", hatch="\\\\")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Top-{k}" for k in k_vals])
    ax.set_ylabel("% of Optimal Cumulative Activity")

    # Auto-scale y to show negative values; keep oracle line visible
    all_vals = [v for v in bnn_pcts if not np.isnan(v)]
    if null_pcts:
        all_vals += [v for v in null_pcts if not np.isnan(v)]
    if null2_pcts:
        all_vals += [v for v in null2_pcts if not np.isnan(v)]
    if all_vals:
        ymin = min(min(all_vals) - 0.1, -0.1)
        ymax = max(max(all_vals) + 0.15, 1.15)
        ax.set_ylim(ymin, ymax)

    ax.axhline(1.0, color="black", ls="--", lw=0.8, alpha=0.5, label="Oracle")
    ax.axhline(0.0, color="gray", ls="-", lw=0.5, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(title)

    # Value labels on bars
    for i, v in enumerate(bnn_pcts):
        if not np.isnan(v):
            offset = 0.02 if v >= 0 else -0.06
            ax.text(x[i] + offsets[0], v + offset, f"{v:.2f}", ha="center", fontsize=7)
    if null_pcts:
        for i, v in enumerate(null_pcts):
            if not np.isnan(v):
                offset = 0.02 if v >= 0 else -0.06
                ax.text(x[i] + offsets[1], v + offset, f"{v:.2f}", ha="center",
                        fontsize=7, color="#555")
    if null2_pcts:
        for i, v in enumerate(null2_pcts):
            if not np.isnan(v):
                offset = 0.02 if v >= 0 else -0.06
                ax.text(x[i] + offsets[-1], v + offset, f"{v:.2f}", ha="center",
                        fontsize=7, color="#8B6914")


def plot_selection_regret(
    regret_bnn: dict,
    regret_null: Optional[dict],
    output_path: Path,
    regret_active_bnn: Optional[dict] = None,
    regret_active_null: Optional[dict] = None,
    regret_null2: Optional[dict] = None,
    regret_active_null2: Optional[dict] = None,
):
    """Bar chart of pct_of_optimal at various k for BNN vs null.

    Two panels: global (all substrates) and active-substrates-only.

    Args:
        regret_bnn:  Dict from compute_selection_regret for BNN (all substrates).
        regret_null: Dict from compute_selection_regret for null (all substrates).
        regret_active_bnn:  Same, filtered to active substrates only.
        regret_active_null: Same, filtered to active substrates only.
        regret_null2: Dict from compute_selection_regret for null2 (all substrates).
        regret_active_null2: Same for null2, filtered to active substrates only.
        output_path: Save path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_active = regret_active_bnn and any(
        k.endswith("_pct_of_optimal") for k in regret_active_bnn)
    ncols = 2 if has_active else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    _plot_selection_regret_panel(
        axes[0], regret_bnn, regret_null, "Selection Regret — All Substrates",
        regret_null2=regret_null2)

    if has_active:
        _plot_selection_regret_panel(
            axes[1], regret_active_bnn, regret_active_null,
            "Selection Regret — Active Substrates Only",
            regret_null2=regret_active_null2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path.name)
