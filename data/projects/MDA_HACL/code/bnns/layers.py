"""
bnn/layers.py
=============
Bayesian linear layer using mean-field variational inference.

Each weight and bias is parameterized as an independent Gaussian:
    q(w) = N(mu, softplus(rho)^2)

Weights are sampled at every forward pass via the reparameterization trick,
making the layer differentiable w.r.t. the variational parameters mu and rho.
The KL divergence KL[q(w) || p(w)] is computed analytically and returned
alongside the layer output so it can be accumulated and added to the ELBO loss.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence as torch_kl


class BayesianLinear(nn.Module):
    """
    A linear layer with weight distributions instead of point estimates.

    Variational parameters (mu, rho) are learned. At each forward pass, weights
    are sampled as:
        w = mu + softplus(rho) * epsilon,  epsilon ~ N(0, I)

    The KL divergence against a zero-mean Gaussian prior is computed analytically
    and returned for inclusion in the ELBO loss.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        prior_mu (float): Mean of the weight prior. Almost always 0.
        prior_std (float): Standard deviation of the weight prior N(prior_mu, prior_std^2).
            Controls regularization strength — smaller = stronger pull toward zero.
            Typical range: 0.1 to 1.0. Treat as a hyperparameter.
        bias (bool): Whether to include a bias term. Default True.
        rho_init (float): Initial value for rho parameters. Converted to std via
            softplus: std_init = log(1 + exp(rho_init)). Default -4.0 gives
            std ~0.018, which starts the weights near their means.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_mu: float = 0.0,
        prior_std: float = 1.0,
        bias: bool = True,
        rho_init: float = -4.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_mu = prior_mu
        self.prior_std = prior_std

        # Variational parameters for weights
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), rho_init))

        # Variational parameters for bias
        self.bias = bias
        if bias:
            self.bias_mu = nn.Parameter(torch.zeros(out_features))
            self.bias_rho = nn.Parameter(torch.full((out_features,), rho_init))
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_rho", None)

        self._init_weights()

        # When True, forward() uses mean weights (no sampling, no KL).
        # Set on frozen layers in composite models (e.g. BNN2) so the
        # backbone acts as a deterministic feature extractor during training.
        # At inference time this can be toggled back to False for full MC.
        self.deterministic = False

    def _init_weights(self) -> None:
        """Initialize weight means with Kaiming uniform (same as nn.Linear default)."""
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        if self.bias_mu is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_mu, -bound, bound)

    @staticmethod
    def _rho_to_std(rho: torch.Tensor) -> torch.Tensor:
        """Convert unconstrained rho to positive std via softplus: sigma = log(1 + exp(rho))."""
        return F.softplus(rho)

    def _compute_kl(self, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        Compute analytic KL divergence KL[ N(mu, std^2) || N(prior_mu, prior_std^2) ].

        Summed over all elements of the tensor.

        Args:
            mu (torch.Tensor): Variational mean.
            std (torch.Tensor): Variational standard deviation (must be positive).

        Returns:
            torch.Tensor: Scalar KL value.
        """
        posterior = Normal(mu, std)
        prior = Normal(
            torch.tensor(self.prior_mu, device=mu.device),
            torch.tensor(self.prior_std, device=mu.device),
        )
        return torch_kl(posterior, prior).sum()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample weights and compute a linear forward pass.

        A fresh weight sample is drawn at every call, so repeated calls with
        the same input will give different outputs. This is the desired behavior
        for Monte Carlo integration over the posterior during inference.

        When ``self.deterministic`` is True, uses the posterior mean weights
        directly (no sampling) and returns KL = 0. This is used for frozen
        backbone layers during training to eliminate gradient noise.

        Args:
            x (torch.Tensor): Input tensor of shape (..., in_features).

        Returns:
            output (torch.Tensor): Result of shape (..., out_features).
            kl (torch.Tensor): Scalar KL divergence for this layer. Sum across
                all layers in the network before adding to the ELBO loss.
        """
        if self.deterministic:
            bias = self.bias_mu if self.bias else None
            return F.linear(x, self.weight_mu, bias), torch.tensor(0.0, device=x.device)

        weight_std = self._rho_to_std(self.weight_rho)
        weight = self.weight_mu + weight_std * torch.randn_like(weight_std)
        kl = self._compute_kl(self.weight_mu, weight_std)

        if self.bias:
            bias_std = self._rho_to_std(self.bias_rho)
            bias = self.bias_mu + bias_std * torch.randn_like(bias_std)
            kl = kl + self._compute_kl(self.bias_mu, bias_std)
        else:
            bias = None

        return F.linear(x, weight, bias), kl

    def extra_repr(self) -> str:
        """String representation showing layer configuration."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"prior_std={self.prior_std}, bias={self.bias is not None}"
        )