"""
bnn/model.py
============
Bayesian MLP with full uncertainty decomposition.

Supports two tasks:
  - 'regression': Models heteroscedastic aleatoric uncertainty by predicting both
    a mean (mu) and log-variance (log_sigma^2) for each output dimension. Combines
    with epistemic uncertainty from weight sampling to give a full uncertainty budget:

        Var_total = Var_epistemic + E[Var_aleatoric]

  - 'classification': Outputs class logits. Epistemic uncertainty is captured by
    the variance of predicted probabilities across MC samples. Aleatoric uncertainty
    is captured by the mean predictive entropy.

The key method for both tasks is `predict_with_uncertainty`, which runs N stochastic
forward passes and returns decomposed uncertainty estimates.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import BayesianLinear


@dataclass
class UncertaintyEstimate:
    """
    Container for decomposed uncertainty from a BayesianMLP forward pass.

    For regression:
        mean:             Predictive mean, shape (batch, output_dim)
        epistemic_std:    Std from weight uncertainty (reducible), shape (batch, output_dim)
        aleatoric_std:    Std from data noise (irreducible), shape (batch, output_dim)
        total_std:        Combined std = sqrt(epistemic_var + aleatoric_var), shape (batch, output_dim)
        samples:          Raw mean predictions across MC samples, shape (n_samples, batch, output_dim)

    For classification:
        mean:             Mean predicted probabilities, shape (batch, n_classes)
        epistemic_std:    Std of predicted probs across samples (model disagreement), shape (batch, n_classes)
        aleatoric_std:    Mean entropy per sample (expected data uncertainty), shape (batch,) scalar per example
        total_std:        Predictive entropy (total uncertainty), shape (batch,) scalar per example
        samples:          Raw probability predictions across MC samples, shape (n_samples, batch, n_classes)
    """

    mean: torch.Tensor
    epistemic_std: torch.Tensor
    aleatoric_std: torch.Tensor
    total_std: torch.Tensor
    samples: torch.Tensor


@dataclass
class HurdleUncertaintyEstimate:
    """Uncertainty estimates from a hurdle (classification + regression) model.

    The hurdle model produces three raw outputs: (mu, log_var, cls_logit).
    At inference, a soft mixture combines regression and floor predictions:
        mean = p * reg_mean + (1 - p) * floor_value
    where p = cls_prob = mean(sigmoid(cls_logit)) across MC samples.

    This dataclass stores both raw regression outputs and the combined
    soft-mixture outputs. The combined fields (mean, epistemic_std,
    aleatoric_std, total_std, samples) provide backward compatibility
    with code that expects an UncertaintyEstimate.

    Classification fields:
        cls_prob:       Mean P(active) across MC samples, shape (batch, 1)
        cls_prob_std:   Epistemic uncertainty on P(active), shape (batch, 1)

    Raw regression fields (before soft mixture):
        reg_mean:           Predictive mean from regression head, shape (batch, output_dim)
        reg_epistemic_std:  Epistemic std from regression head
        reg_aleatoric_std:  Aleatoric std from regression head
        reg_total_std:      Total std from regression head
        reg_samples:        Regression mu samples, shape (n_samples, batch, output_dim)

    Combined fields (after soft mixture):
        mean:           Soft mixture prediction, shape (batch, output_dim)
        epistemic_std:  Mixture epistemic std
        aleatoric_std:  Mixture aleatoric std
        total_std:      Mixture total std (nonzero when cls_prob intermediate)
        samples:        Alias for reg_samples (backward compat for aggregation)
    """

    # Classification
    cls_prob: torch.Tensor
    cls_prob_std: torch.Tensor
    # Raw regression
    reg_mean: torch.Tensor
    reg_epistemic_std: torch.Tensor
    reg_aleatoric_std: torch.Tensor
    reg_total_std: torch.Tensor
    reg_samples: torch.Tensor
    # Combined (soft mixture)
    mean: torch.Tensor
    epistemic_std: torch.Tensor
    aleatoric_std: torch.Tensor
    total_std: torch.Tensor
    samples: torch.Tensor


class BayesianMLP(nn.Module):
    """
    Bayesian MLP using mean-field variational inference for weight uncertainty.

    For regression, the network also models aleatoric (data) uncertainty by
    predicting both a mean and a log-variance for each output. This gives a
    complete decomposition of predictive uncertainty:
        - Epistemic: from weight posterior spread (reducible with more data)
        - Aleatoric: from predicted output variance (irreducible noise floor)
        - Total:     sqrt(epistemic_var + aleatoric_var)

    For classification, epistemic uncertainty comes from variance of predicted
    probabilities across MC samples, and aleatoric from mean predictive entropy.

    All BayesianLinear layers share the same prior hyperparameters. Dropout
    is added between hidden layers for additional regularization (it does NOT
    serve as the uncertainty source here — that comes from weight sampling).

    Args:
        input_dim (int): Number of input features.
        hidden_dims (list[int]): Width of each hidden layer. Deeper/wider = more
            expressive but more parameters and slower inference.
            Example: [256, 128, 64]
        output_dim (int): Number of output dimensions (regression targets or classes).
        task (str): 'regression' or 'classification'. Controls output head and
            uncertainty decomposition strategy.
        prior_mu (float): Mean of weight prior. Default 0.0.
        prior_std (float): Std of weight prior. Key regularization hyperparameter.
            Smaller = stronger regularization. Try 0.1–1.0.
        rho_init (float): Initial posterior log-std via softplus(rho_init).
            More negative = tighter initialization around the mean. Default -2.5
            (sigma≈0.082). Fixed regardless of prior_std — coupling to prior_std
            causes first-layer noise to dominate signal for large fan_in.
        dropout_rate (float): Dropout between hidden layers. Helps generalization
            independent of Bayesian weight uncertainty. Set 0.0 to disable.
        activation (str): Hidden layer activation. 'relu', 'tanh', 'elu', or 'silu'.
        bias (bool): Whether layers include bias terms.
    """

    ACTIVATIONS = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "silu": nn.SiLU,
    }

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        task: Literal["regression", "classification"] = "regression",
        prior_mu: float = 0.0,
        prior_std: float = 1.0,
        rho_init: Optional[float] = None,
        dropout_rate: float = 0.1,
        activation: str = "relu",
        bias: bool = True,
    ):
        super().__init__()

        if task not in ("regression", "classification"):
            raise ValueError(f"task must be 'regression' or 'classification', got '{task}'")
        if activation not in self.ACTIVATIONS:
            raise ValueError(f"activation must be one of {list(self.ACTIVATIONS)}, got '{activation}'")
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer.")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task = task
        self.prior_std = prior_std

        # Fixed rho_init independent of prior_std.
        # softplus(-2.5) ≈ 0.082 — small enough for numerical stability (first-layer
        # noise variance ≈ 0.082² × fan_in ≪ typical signal variance), yet large enough
        # that the KL gradient can move rho during training (avoids posterior collapse
        # that occurs when rho is stuck far from zero, e.g. at -4.0 giving sigma≈0.018).
        if rho_init is None:
            rho_init = -2.5

        layer_kwargs = dict(prior_mu=prior_mu, prior_std=prior_std, rho_init=rho_init, bias=bias)
        act_cls = self.ACTIVATIONS[activation]

        # ---- Hidden layers (shared backbone) ----
        hidden_layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            hidden_layers.append(BayesianLinear(dims[i], dims[i + 1], **layer_kwargs))
            hidden_layers.append(act_cls())
            if dropout_rate > 0.0:
                hidden_layers.append(nn.Dropout(p=dropout_rate))
        self.hidden = nn.ModuleList(hidden_layers)

        # ---- Output head ----
        # Regression: predict (mu, log_var) per output => 2 * output_dim outputs
        # Classification: predict logits => output_dim outputs
        head_out = 2 * output_dim if task == "regression" else output_dim
        self.head = BayesianLinear(hidden_dims[-1], head_out, **layer_kwargs)

    # ------------------------------------------------------------------
    # Internal forward (returns raw output + total KL)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single stochastic forward pass with weight sampling.

        Accumulates KL divergence across all Bayesian layers. During training,
        call this inside the ELBO loss function. For inference, prefer
        `predict_with_uncertainty` which aggregates multiple samples.

        Args:
            x (torch.Tensor): Input of shape (batch, input_dim).

        Returns:
            output (torch.Tensor): Raw head output. For regression this is
                shape (batch, 2 * output_dim) — first half is mu, second is
                log_var. For classification it is (batch, output_dim) logits.
            total_kl (torch.Tensor): Scalar sum of KL from all layers.
        """
        total_kl = torch.tensor(0.0, device=x.device)

        for layer in self.hidden:
            if isinstance(layer, BayesianLinear):
                x, kl = layer(x)
                total_kl = total_kl + kl
            else:
                x = layer(x)

        output, kl = self.head(x)
        total_kl = total_kl + kl

        return output, total_kl

    # ------------------------------------------------------------------
    # Uncertainty-aware inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 100,
        fc_ref: Optional[torch.Tensor] = None,
        prediction_floor: Optional[float] = None,
    ) -> UncertaintyEstimate:
        """
        Generate predictions with fully decomposed uncertainty via MC weight sampling.

        Runs `n_samples` stochastic forward passes, each drawing a fresh set of
        weights from the learned posterior q(w). The resulting predictive distribution
        is then summarized into epistemic and aleatoric components.

        For REGRESSION:
            epistemic_var  = Var_w[E[y|x,w]] = variance of predicted means across samples
            aleatoric_var  = E_w[Var[y|x,w]] = mean of predicted variances across samples
            total_var      = epistemic_var + aleatoric_var  (law of total variance)

        For CLASSIFICATION:
            epistemic_std  = std of predicted class probabilities across samples
                             (high when different weight draws disagree on the prediction)
            aleatoric_std  = mean per-sample predictive entropy
                             (high when the model is confident but the problem is ambiguous)
            total_std      = entropy of the mean probability vector
                             (standard predictive uncertainty)

        Args:
            x (torch.Tensor): Input of shape (batch, input_dim).
            n_samples (int): Number of MC weight samples. More samples = better
                uncertainty estimates at the cost of compute. 50–200 is typical.
                Use ~20 for quick checks, ~500 for final reporting.
            fc_ref (torch.Tensor, optional): Reference log_fc values (N, 1) for
                delta learning reconstruction. When provided, model predictions
                (deltas) are shifted to absolute scale: pred_abs = pred_delta + fc_ref.
            prediction_floor (float, optional): Minimum predicted value in absolute
                space. MC mu samples are clamped at this floor after the fc_ref
                shift. None = no floor.

        Returns:
            UncertaintyEstimate: Dataclass containing mean, epistemic_std,
                aleatoric_std, total_std, and raw samples.
        """
        self.train()  # keep dropout active if used (not the main source of uncertainty here)

        raw_outputs = torch.stack([self(x)[0] for _ in range(n_samples)], dim=0)
        # raw_outputs: (n_samples, batch, head_out)

        if self.task == "regression":
            # Apply delta-learning shift and floor before uncertainty decomposition
            if fc_ref is not None or prediction_floor is not None:
                # Split into mu and log_var before modification
                mu_part = raw_outputs[..., : self.output_dim]
                rest = raw_outputs[..., self.output_dim :]
                if fc_ref is not None:
                    # fc_ref: (batch, 1) → broadcast over n_samples
                    mu_part = mu_part + fc_ref.unsqueeze(0)
                if prediction_floor is not None:
                    mu_part = mu_part.clamp(min=prediction_floor)
                raw_outputs = torch.cat([mu_part, rest], dim=-1)
            return self._decompose_regression_uncertainty(raw_outputs)
        else:
            return self._decompose_classification_uncertainty(raw_outputs)

    def _decompose_regression_uncertainty(
        self, raw_outputs: torch.Tensor
    ) -> UncertaintyEstimate:
        """
        Decompose predictive uncertainty for regression via law of total variance.

        Args:
            raw_outputs (torch.Tensor): Shape (n_samples, batch, 2 * output_dim).
                First output_dim columns are predicted means, last are log-variances.

        Returns:
            UncertaintyEstimate with per-output uncertainty components.
        """
        # Split head output into mean and log-variance predictions
        mu_samples = raw_outputs[..., : self.output_dim]          # (n_samples, batch, D)
        log_var_samples = raw_outputs[..., self.output_dim :]     # (n_samples, batch, D)

        # Clamp log-var for numerical stability (prevents exp overflow/underflow)
        log_var_samples = log_var_samples.clamp(-10.0, 10.0)
        var_samples = torch.exp(log_var_samples)                   # predicted data variance

        # Predictive mean: E_w[mu(x, w)]
        mean = mu_samples.mean(dim=0)                              # (batch, D)

        # Epistemic variance: Var_w[mu(x, w)] — spread of the mean predictions
        epistemic_var = mu_samples.var(dim=0, unbiased=True)      # (batch, D)

        # Aleatoric variance: E_w[sigma^2(x, w)] — expected predicted data noise
        aleatoric_var = var_samples.mean(dim=0)                   # (batch, D)

        # Total variance by law of total variance
        total_var = epistemic_var + aleatoric_var                  # (batch, D)

        return UncertaintyEstimate(
            mean=mean,
            epistemic_std=epistemic_var.sqrt(),
            aleatoric_std=aleatoric_var.sqrt(),
            total_std=total_var.sqrt(),
            samples=mu_samples,
        )

    def _decompose_classification_uncertainty(
        self, raw_outputs: torch.Tensor
    ) -> UncertaintyEstimate:
        """
        Decompose predictive uncertainty for classification.

        Uses mutual information decomposition:
            Total uncertainty    = H[E_w[p(y|x,w)]]    (entropy of mean probs)
            Aleatoric uncertainty = E_w[H[p(y|x,w)]]   (mean entropy per sample)
            Epistemic uncertainty = Total - Aleatoric   (mutual information)

        Args:
            raw_outputs (torch.Tensor): Shape (n_samples, batch, n_classes). Logits.

        Returns:
            UncertaintyEstimate where total_std/aleatoric_std are per-example
            scalars (shape batch,) and epistemic_std is per-class (batch, n_classes).
        """
        prob_samples = torch.softmax(raw_outputs, dim=-1)          # (n_samples, batch, C)

        mean_probs = prob_samples.mean(dim=0)                      # (batch, C)
        epistemic_std = prob_samples.std(dim=0, unbiased=True)     # (batch, C)

        # Entropy of the mean: H[E[p(y|x,w)]]
        total_entropy = self._entropy(mean_probs)                  # (batch,)

        # Mean entropy: E[H[p(y|x,w)]]
        per_sample_entropy = self._entropy(prob_samples)           # (n_samples, batch)
        mean_entropy = per_sample_entropy.mean(dim=0)              # (batch,)

        # Mutual information = total - aleatoric
        # (Not directly a "std" but consistent interface for inspection)
        epistemic_entropy = (total_entropy - mean_entropy).clamp(min=0.0)

        return UncertaintyEstimate(
            mean=mean_probs,
            epistemic_std=epistemic_std,
            aleatoric_std=mean_entropy,          # shape (batch,)
            total_std=total_entropy,             # shape (batch,)
            samples=prob_samples,
        )

    @staticmethod
    def _entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Compute Shannon entropy H = -sum(p * log(p)) along the last dimension.

        Args:
            probs (torch.Tensor): Probability tensor, any shape (..., C).
            eps (float): Small constant for numerical stability in log.

        Returns:
            torch.Tensor: Entropy values, shape (...).
        """
        return -(probs * (probs + eps).log()).sum(dim=-1)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def freeze_hidden(self) -> None:
        """
        Freeze all hidden layers, leaving only the output head trainable.

        Useful for transfer learning: load pretrained weights, freeze the
        backbone, and fine-tune only the head on the new task. Remember to
        reinitialize the optimizer after calling this.
        """
        for layer in self.hidden:
            for param in layer.parameters():
                param.requires_grad = False

    def unfreeze_hidden(self) -> None:
        """Unfreeze all hidden layers. Reinitialize optimizer after calling."""
        for layer in self.hidden:
            for param in layer.parameters():
                param.requires_grad = True

    def trainable_parameter_count(self) -> dict[str, int]:
        """
        Count trainable vs frozen parameters.

        Returns:
            dict with keys 'trainable', 'frozen', 'total'.
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}

    def posterior_summary(self) -> dict[str, dict[str, float]]:
        """
        Summarize the learned posterior over weights for diagnostic purposes.

        Reports the mean absolute weight mean and mean weight std for each
        BayesianLinear layer. Useful for checking that the posterior hasn't
        collapsed (std near zero = point estimate) or exploded.

        Returns:
            dict mapping layer names to {'mean_abs_mu', 'mean_std'}.
        """
        summary = {}
        for name, module in self.named_modules():
            if isinstance(module, BayesianLinear):
                std = F.softplus(module.weight_rho).detach()
                summary[name] = {
                    "mean_abs_mu": module.weight_mu.detach().abs().mean().item(),
                    "mean_std": std.mean().item(),
                }
        return summary