"""
bnn/trainer.py
==============
Training loop for BayesianMLP with ELBO loss and KL annealing.

The ELBO (Evidence Lower BOund) objective is:
    ELBO = E_q[log p(y | x, w)] - KL[q(w) || p(w)]

Maximizing ELBO is equivalent to minimizing:
    L = NLL(y | f(x, w)) + (beta / N) * KL[q(w) || p(w)]

where:
    NLL  = negative log-likelihood (MSE or Gaussian NLL for regression,
           cross-entropy for classification)
    N    = total number of training examples (KL scaling)
    beta = KL weight, often annealed from 0 → 1 to prevent posterior collapse
           early in training ("beta annealing" / "KL warm-up")

KL Annealing:
    When beta is too large early in training, the KL term dominates and forces
    the posterior toward the prior before the likelihood has had a chance to
    shape it. Starting beta at 0 and linearly warming up to 1 over the first
    `kl_anneal_epochs` epochs avoids this.

    Important: Validation loss for early stopping is always computed at the
    final kl_weight (not the annealed beta). This prevents checkpointing from
    favoring under-regularized states during the warm-up phase, where low beta
    mechanically produces lower loss without better generalization.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .model import BayesianMLP, UncertaintyEstimate, HurdleUncertaintyEstimate

logger = logging.getLogger(__name__)


@dataclass
class HurdleConfig:
    """Sub-parameters for hurdle (classification + regression) loss.

    Only used when ``TrainingConfig.loss_type == "hurdle"``.

    When active, the model output is (batch, 2*output_dim + output_dim):
      - [:, :output_dim]            = mu (regression mean)
      - [:, output_dim:2*output_dim] = log_var (regression log-variance)
      - [:, 2*output_dim:]          = cls_logit (floor vs non-floor logit)

    Loss is the proper hurdle log-likelihood:
      For y <= floor_threshold:  -log(1 - sigmoid(cls_logit)) = softplus(cls_logit)
      For y > floor_threshold:   -log(sigmoid(cls_logit)) + gaussian_nll(y, mu, var)

    Args:
        floor_threshold:     y values <= this are classified as floor.
        floor_value:         Prediction for samples classified as floor at inference.
        inference_threshold: Not used in loss; used at inference for soft mixture center.
    """
    floor_threshold: float = -1.99
    floor_value: float = -2.0
    inference_threshold: float = 0.5

    @property
    def enabled(self) -> bool:
        """Backward-compat shim — always returns False.

        Loss selection is controlled by ``TrainingConfig.loss_type``.
        Code that checks ``hurdle_config.enabled`` should migrate to
        checking ``loss_type == "hurdle"`` instead.
        """
        return False


@dataclass
class TrainingConfig:
    """
    Hyperparameter configuration for BNNTrainer.

    Args:
        learning_rate (float): Adam optimizer learning rate. Typical: 1e-3.
        weight_decay (float): L2 regularization on variational parameters.
            Note: the KL term already acts as regularization; this is supplemental.
        batch_size (int): Mini-batch size. Larger = more stable gradients but
            less frequent updates.
        n_epochs (int): Total training epochs.
        n_elbo_samples (int): Weight samples per forward pass during training.
            1 is standard (and cheap). Use 3–5 for lower-variance gradient estimates
            at the cost of proportionally more compute per step.
        kl_weight (float): Final KL weight beta after annealing completes.
            beta=1.0 is the standard ELBO. Values < 1 (e.g. 0.1) downweight the
            KL, allowing a more expressive posterior at the risk of overfitting.
        kl_anneal_epochs (int): Number of epochs to linearly ramp beta from 0
            to `kl_weight`. Set to 0 to disable annealing (use fixed kl_weight).
        n_inference_samples (int): MC samples used in `predict_with_uncertainty`
            during validation. More = better estimates, slower validation.
        early_stopping_patience (int): Stop if validation loss doesn't improve
            for this many epochs. Set to 0 to disable early stopping.
        n_val_samples (int): Number of weight samples to average when computing
            validation loss. A single sample is extremely noisy (stochastic
            weights + dropout); 10 gives ~3x noise reduction.
        clip_grad_norm (float): Max gradient norm for clipping. Helps with
            occasional large gradients from sampled weights. Set to 0 to disable.
        device (str): 'cuda', 'mps', or 'cpu'. Autodetected if 'auto'.
        verbose (bool): Print training progress each epoch.
        log_interval (int): Print progress every N epochs (if verbose=True).
    """
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    n_epochs: int = 100
    n_elbo_samples: int = 1
    kl_weight: float = 1.0
    kl_anneal_epochs: int = 20
    n_inference_samples: int = 50
    early_stopping_patience: int = 10
    n_val_samples: int = 10
    clip_grad_norm: float = 1.0
    device: str = "auto"
    verbose: bool = True
    log_interval: int = 10
    loss_type: str = "gaussian_nll"  # "gaussian_nll", "mse", or "hurdle"
    hurdle: "HurdleConfig" = None  # sub-params for hurdle loss (thresholds, floor value)
    null_reg_weight: float = 0.0  # L2 penalty on mu toward zero (null hypothesis: delta=0)
    log_var_floor: Optional[float] = None  # min log_var (e.g. -4 → σ_min≈0.14). None = use default -10 clamp.

    _VALID_LOSS_TYPES = ("gaussian_nll", "mse", "hurdle")

    def __post_init__(self):
        if self.hurdle is None:
            self.hurdle = HurdleConfig()
        if self.loss_type not in self._VALID_LOSS_TYPES:
            raise ValueError(
                f"loss_type must be one of {self._VALID_LOSS_TYPES}, got '{self.loss_type}'"
            )


@dataclass
class TrainingHistory:
    """
    Record of losses across training epochs.

    Attributes:
        train_loss (list[float]): Total ELBO loss per epoch.
        train_nll (list[float]): NLL component of the loss per epoch.
        train_kl (list[float]): KL component (before beta scaling) per epoch.
        val_loss (list[float]): Validation ELBO per epoch (NLL + KL; logged only).
        val_nll (list[float]): Validation NLL per epoch (used for early stopping).
        kl_weight (list[float]): Beta value used each epoch.
        best_epoch (int): Epoch index of best validation NLL.
    """
    train_loss: list[float] = field(default_factory=list)
    train_nll: list[float] = field(default_factory=list)
    train_kl: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_nll: list[float] = field(default_factory=list)
    kl_weight_schedule: list[float] = field(default_factory=list)
    best_epoch: int = 0


class BNNTrainer:
    """
    Trainer for BayesianMLP with ELBO loss, KL annealing, and early stopping.

    Handles the full training loop including:
        - Mini-batch ELBO optimization
        - KL weight annealing schedule
        - Optional validation loop with uncertainty estimates
        - Gradient clipping
        - Early stopping with best-model checkpointing
        - Training history logging

    Args:
        model (BayesianMLP): The Bayesian MLP to train.
        config (TrainingConfig): Training hyperparameters.

    Example:
        >>> model = BayesianMLP(input_dim=10, hidden_dims=[64, 32], output_dim=1)
        >>> config = TrainingConfig(n_epochs=100, kl_anneal_epochs=20)
        >>> trainer = BNNTrainer(model, config)
        >>> history = trainer.fit(X_train, y_train, X_val, y_val)
        >>> estimates = trainer.predict(X_test)
    """

    def __init__(self, model: BayesianMLP, config: TrainingConfig):
        self.model = model
        self.config = config
        self.device = self._resolve_device(config.device)
        self.model.to(self.device)
        self.history = TrainingHistory()
        self._best_val_loss = float("inf")
        self._best_state: Optional[dict] = None
        self._no_improve_count = 0

        self.optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """
        Resolve 'auto' to the best available device.

        Args:
            device (str): 'auto', 'cuda', 'mps', or 'cpu'.

        Returns:
            torch.device: Resolved device.
        """
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    def _kl_beta(self, epoch: int) -> float:
        """
        Compute the KL annealing weight for the current epoch.

        Linearly ramps from 0 to `config.kl_weight` over `kl_anneal_epochs`.

        Args:
            epoch (int): Current epoch (0-indexed).

        Returns:
            float: Beta value for this epoch.
        """
        if self.config.kl_anneal_epochs <= 0:
            return self.config.kl_weight
        progress = min(1.0, (epoch + 1) / self.config.kl_anneal_epochs)
        return progress * self.config.kl_weight

    def _hurdle_nll(
        self,
        output: torch.Tensor,
        y: torch.Tensor,
        w: Optional[torch.Tensor] = None,
        is_floor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute hurdle negative log-likelihood (summed over batch).

        The hurdle model factors the likelihood as:
            P(y | x) = (1 - p) * delta(floor)   if y is floor
                      = p * N(y | mu, sigma^2)   if y is active
        where p = sigmoid(cls_logit).

        The NLL decomposes cleanly:
            Floor samples:  -log(1 - p) = softplus(cls_logit)
            Active samples: -log(p) + gaussian_nll = softplus(-cls_logit) + gnll

        Classification and regression gradients are independent — the cls_logit
        term doesn't involve (mu, log_var) and vice versa.

        Args:
            output:   (batch, 2*output_dim + output_dim) — [mu, log_var, cls_logit].
            y:        (batch, output_dim) targets (may be deltas).
            w:        (batch,) optional per-sample weights.
            is_floor: (batch,) precomputed floor mask (bool or float 0/1).
                      When provided, used directly instead of comparing y to
                      floor_threshold. Required for delta learning where y is
                      not in absolute scale.

        Returns:
            Scalar total NLL (summed over batch).
        """
        od = self.model.output_dim
        mu = output[:, :od]
        lv_floor = self.config.log_var_floor if self.config.log_var_floor is not None else -10.0
        log_var = output[:, od:2 * od].clamp(lv_floor, 10.0)
        cls_logit = output[:, 2 * od:].squeeze(-1)  # (batch,)

        if is_floor is not None:
            is_floor = is_floor.bool()
        else:
            is_floor = y.squeeze(-1) <= self.config.hurdle.floor_threshold
        nll_per = torch.zeros(y.shape[0], device=y.device)

        # Floor samples: -log(1 - sigmoid(logit)) = softplus(logit)
        if is_floor.any():
            nll_per[is_floor] = F.softplus(cls_logit[is_floor])

        # Active samples: -log(sigmoid(logit)) + gaussian_nll
        active = ~is_floor
        if active.any():
            cls_nll = F.softplus(-cls_logit[active])
            reg_nll = F.gaussian_nll_loss(
                mu[active], y[active], log_var[active].exp(),
                full=False, reduction="none",
            ).squeeze(-1)
            nll_per[active] = cls_nll + reg_nll

        if w is not None:
            nll_per = nll_per * w
        return nll_per.sum()

    def _compute_nll(
        self,
        output: torch.Tensor,
        y: torch.Tensor,
        w: Optional[torch.Tensor] = None,
        is_floor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute NLL for regression or classification.

        Routes to the appropriate loss based on ``config.loss_type``:
          - ``"gaussian_nll"``: Heteroscedastic Gaussian NLL (learns variance)
          - ``"mse"``:          Mean squared error on mu (ignores log_var in loss)
          - ``"hurdle"``:       Classification + regression hurdle likelihood

        Returns a scalar (summed over batch).
        """
        if self.model.task == "regression":
            if self.config.loss_type == "hurdle":
                return self._hurdle_nll(output, y, w, is_floor=is_floor)

            mu = output[:, : self.model.output_dim]

            if self.config.loss_type == "mse":
                # Pure MSE on mu; log_var is ignored during training.
                # Epistemic uncertainty still comes from MC weight sampling.
                nll_per = (mu - y).pow(2).squeeze(-1)
            else:
                # gaussian_nll: learn heteroscedastic variance
                lv_floor = self.config.log_var_floor if self.config.log_var_floor is not None else -10.0
                log_var = output[:, self.model.output_dim :].clamp(lv_floor, 10.0)
                nll_per = F.gaussian_nll_loss(
                    mu, y, log_var.exp(), full=False, reduction="none"
                ).squeeze(-1)

            if w is not None:
                nll_per = nll_per * w
            return nll_per.sum()
        else:
            return F.cross_entropy(output, y, reduction="sum")

    def _elbo_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n_data: int,
        beta: float,
        w: Optional[torch.Tensor] = None,
        is_floor: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the negative ELBO loss for a mini-batch.

        Averages over `config.n_elbo_samples` weight samples to reduce gradient
        variance. KL is scaled by beta/N_total to balance against the NLL.

        Args:
            x (torch.Tensor): Input batch of shape (batch, input_dim).
            y (torch.Tensor): Target batch of shape (batch, output_dim) for
                regression, or (batch,) integer class indices for classification.
            n_data (int): Total number of training examples (for KL scaling).
            beta (float): Current KL annealing weight.
            w (torch.Tensor, optional): Per-sample loss weights of shape (batch,).
                When provided, each sample's NLL is multiplied by its weight before
                summing. Weights should have mean ≈ 1.0 to preserve the overall
                loss scale. If None, uniform weights (all 1.0) are assumed.
            is_floor (torch.Tensor, optional): Precomputed floor mask for hurdle loss.

        Returns:
            total_loss (torch.Tensor): Scalar ELBO loss.
            nll_val (torch.Tensor): NLL component (for logging).
            kl_val (torch.Tensor): KL component before scaling (for logging).
        """
        nll_accum = torch.tensor(0.0, device=self.device)
        kl_accum = torch.tensor(0.0, device=self.device)
        mu_sq_accum = torch.tensor(0.0, device=self.device)

        for _ in range(self.config.n_elbo_samples):
            output, kl = self.model(x)
            nll = self._compute_nll(output, y, w, is_floor=is_floor)
            nll_accum = nll_accum + nll
            kl_accum = kl_accum + kl

            # Accumulate mu^2 for null regularization
            if self.config.null_reg_weight > 0.0:
                mu = output[:, :self.model.output_dim]
                mu_sq_accum = mu_sq_accum + mu.pow(2).mean()

        nll_accum = nll_accum / self.config.n_elbo_samples
        kl_accum = kl_accum / self.config.n_elbo_samples
        total_loss = nll_accum + (beta / n_data) * kl_accum

        # Null regularization: bias predictions toward zero (delta = 0)
        if self.config.null_reg_weight > 0.0:
            null_reg = self.config.null_reg_weight * (mu_sq_accum / self.config.n_elbo_samples)
            total_loss = total_loss + null_reg

        return total_loss, nll_accum.detach(), kl_accum.detach()

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        w_train: Optional[torch.Tensor] = None,
        is_floor_train: Optional[torch.Tensor] = None,
        is_floor_val: Optional[torch.Tensor] = None,
    ) -> TrainingHistory:
        """
        Train the model on the provided data.

        Tensors are moved to the configured device automatically. If validation
        data is provided, early stopping and best-model checkpointing are applied
        against the validation loss (otherwise against training loss).

        Args:
            X_train (torch.Tensor): Training inputs, shape (N, input_dim).
            y_train (torch.Tensor): Training targets. Shape (N, output_dim) for
                regression or (N,) integer labels for classification.
            X_val (torch.Tensor, optional): Validation inputs.
            y_val (torch.Tensor, optional): Validation targets.
            w_train (torch.Tensor, optional): Per-sample loss weights of shape (N,).
                Weights are applied to each sample's NLL contribution. Use mean ≈ 1.0
                to preserve the overall loss scale (e.g. from LDSAttenuatedWeights
                with normalize=True). If None, all samples are weighted equally.
            is_floor_train (torch.Tensor, optional): Precomputed floor mask (N,)
                for hurdle loss. Required for delta learning where y values are
                not in absolute scale.
            is_floor_val (torch.Tensor, optional): Precomputed floor mask for
                validation set. Stored for use in _validation_loss.

        Returns:
            TrainingHistory: Loss curves and metadata from training.
        """
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val = X_val.to(self.device)
            y_val = y_val.to(self.device)

        # Store is_floor_val for validation loss computation
        self._is_floor_val = None
        if is_floor_val is not None:
            self._is_floor_val = is_floor_val.to(self.device)

        # ═══════════════════════════════════════════════════════════════════
        # DIAGNOSTIC: Pre-training data statistics
        # ═══════════════════════════════════════════════════════════════════
        y_np = y_train.cpu().numpy().squeeze()
        logger.info("┌─── PRE-TRAINING DATA DIAGNOSTICS ───┐")
        logger.info("│ Training set: %d samples", len(X_train))
        logger.info("│ Input dim: %d", X_train.shape[1])
        logger.info("│ y_train (delta targets):")
        logger.info("│   mean=%.4f  std=%.4f  min=%.4f  max=%.4f  median=%.4f",
                     y_np.mean(), y_np.std(), y_np.min(), y_np.max(), np.median(y_np))
        logger.info("│   pct near zero (|y|<0.1): %.1f%%",
                     100.0 * np.mean(np.abs(y_np) < 0.1))
        logger.info("│   pct negative: %.1f%%  pct positive: %.1f%%",
                     100.0 * np.mean(y_np < 0), 100.0 * np.mean(y_np > 0))
        # Feature statistics (check for degenerate inputs)
        x_np = X_train.cpu().numpy()
        x_stds = x_np.std(axis=0)
        n_const = int((x_stds < 1e-8).sum())
        n_large = int((np.abs(x_np).max(axis=0) > 100).sum())
        logger.info("│ X_train feature stats:")
        logger.info("│   mean_of_means=%.4f  mean_of_stds=%.4f",
                     x_np.mean(axis=0).mean(), x_stds.mean())
        logger.info("│   constant features (std<1e-8): %d / %d", n_const, x_np.shape[1])
        logger.info("│   large-magnitude features (|max|>100): %d / %d", n_large, x_np.shape[1])
        if has_val:
            yv_np = y_val.cpu().numpy().squeeze()
            logger.info("│ Validation set: %d samples", len(X_val))
            logger.info("│ y_val (delta targets):")
            logger.info("│   mean=%.4f  std=%.4f  min=%.4f  max=%.4f  median=%.4f",
                         yv_np.mean(), yv_np.std(), yv_np.min(), yv_np.max(), np.median(yv_np))
        if is_floor_train is not None:
            n_floor = int(is_floor_train.sum().item())
            logger.info("│ Hurdle floor mask: %d / %d = %.1f%% floor samples",
                         n_floor, len(is_floor_train),
                         100.0 * n_floor / len(is_floor_train))
        logger.info("│ Model: %s  loss_type=%s",
                     type(self.model).__name__, self.config.loss_type)
        logger.info("│ Optimizer: lr=%.1e  weight_decay=%.1e",
                     self.config.learning_rate, self.config.weight_decay)
        logger.info("│ KL: anneal_epochs=%d  kl_weight=%.4f  prior_std=%.4f",
                     self.config.kl_anneal_epochs, self.config.kl_weight,
                     getattr(self.model, 'prior_std', float('nan')))
        logger.info("│ Early stopping: monitors val NLL (not ELBO)")
        if self.config.null_reg_weight > 0.0:
            logger.info("│ Null reg: weight=%.4f", self.config.null_reg_weight)
        if self.config.log_var_floor is not None:
            logger.info("│ log_var floor: %.1f (σ_min=%.4f)",
                         self.config.log_var_floor,
                         float(np.exp(self.config.log_var_floor / 2)))
        logger.info("└──────────────────────────────────────┘")

        # Always include weights in the DataLoader (default: uniform ones).
        # This avoids conditional unpacking in the batch loop.
        if w_train is None:
            w_train = torch.ones(len(X_train), dtype=torch.float32)
        w_train = w_train.to(self.device)

        if w_train is not None:
            w_np = w_train.cpu().numpy()
            logger.info("  LDS weights: mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
                         w_np.mean(), w_np.std(), w_np.min(), w_np.max())

        # Include is_floor in dataset when provided (for hurdle + delta learning)
        if is_floor_train is not None:
            is_floor_train = is_floor_train.to(self.device)
            dataset = TensorDataset(X_train, y_train, w_train, is_floor_train)
        else:
            dataset = TensorDataset(X_train, y_train, w_train)

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        n_data = len(X_train)

        # ── DIAGNOSTIC: Initial forward pass (before any training) ──
        self.model.eval()
        with torch.no_grad():
            init_out, init_kl = self.model(X_train[:min(256, len(X_train))])
            y_subset = y_train[:min(256, len(X_train))]
            od = self.model.output_dim
            init_mu = init_out[:, :od]
            init_logvar = init_out[:, od:2*od].clamp(-10, 10)
            logger.info("┌─── INITIAL FORWARD PASS (epoch 0, no training) ───┐")
            logger.info("│ mu:      mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
                         init_mu.mean().item(), init_mu.std().item(),
                         init_mu.min().item(), init_mu.max().item())
            logger.info("│ log_var: mean=%.4f  std=%.4f",
                         init_logvar.mean().item(), init_logvar.std().item())
            logger.info("│ sigma:   mean=%.4f",
                         init_logvar.exp().sqrt().mean().item())
            y_subset_f = y_subset.float()
            logger.info("│ y_true:  mean=%.4f  std=%.4f",
                         y_subset_f.mean().item(), y_subset_f.std().item())
            logger.info("│ Initial residual (mu - y): mean=%.4f  std=%.4f  MAE=%.4f",
                         (init_mu.squeeze() - y_subset.squeeze()).mean().item(),
                         (init_mu.squeeze() - y_subset.squeeze()).std().item(),
                         (init_mu.squeeze() - y_subset.squeeze()).abs().mean().item())
            logger.info("│ KL: %.4f", init_kl.item())
            if self.config.loss_type == "hurdle":
                cls_logit = init_out[:, 2*od:]
                cls_prob = torch.sigmoid(cls_logit)
                logger.info("│ cls_logit: mean=%.4f  std=%.4f",
                             cls_logit.mean().item(), cls_logit.std().item())
                logger.info("│ P(active): mean=%.4f  std=%.4f",
                             cls_prob.mean().item(), cls_prob.std().item())
            logger.info("└──────────────────────────────────────────────────────┘")
        self.model.train()

        # Track gradient norms for diagnostics
        _grad_norms = []

        for epoch in range(self.config.n_epochs):
            beta = self._kl_beta(epoch)
            self.model.train()

            epoch_loss = epoch_nll = epoch_kl = 0.0
            epoch_grad_norm = 0.0
            n_batches = 0
            for batch in loader:
                if len(batch) == 4:
                    x_batch, y_batch, w_batch, floor_batch = batch
                else:
                    x_batch, y_batch, w_batch = batch
                    floor_batch = None
                self.optimizer.zero_grad()
                loss, nll, kl = self._elbo_loss(x_batch, y_batch, n_data, beta,
                                                w=w_batch, is_floor=floor_batch)
                loss.backward()

                # Track gradient norms
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
                epoch_grad_norm += total_norm
                n_batches += 1

                if self.config.clip_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)

                self.optimizer.step()
                epoch_loss += loss.item()
                epoch_nll += nll.item()
                epoch_kl += kl.item()

            epoch_loss /= n_data
            epoch_nll /= len(loader)
            epoch_kl /= len(loader)
            avg_grad_norm = epoch_grad_norm / max(n_batches, 1)
            _grad_norms.append(avg_grad_norm)

            self.history.train_loss.append(epoch_loss)
            self.history.train_nll.append(epoch_nll)
            self.history.train_kl.append(epoch_kl)
            self.history.kl_weight_schedule.append(beta)

            # ---- Validation ----
            # Early stopping monitors val NLL only (not full ELBO). The KL
            # term measures posterior divergence from the prior — not predictive
            # quality. Including KL in the monitor biases checkpointing toward
            # the prior (epoch 0), preventing the model from improving.
            # Val ELBO is still logged for diagnostics.
            monitor_loss = epoch_loss
            if has_val:
                val_nll, val_elbo = self._validation_loss(
                    X_val, y_val, n_data,
                    self.config.kl_weight,
                    self.config.n_val_samples)
                self.history.val_nll.append(val_nll)
                self.history.val_loss.append(val_elbo)
                monitor_loss = val_nll  # early stopping on NLL only

            # ---- Early stopping + checkpointing ----
            if monitor_loss < self._best_val_loss:
                self._best_val_loss = monitor_loss
                self._best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                self.history.best_epoch = epoch
                self._no_improve_count = 0
            else:
                self._no_improve_count += 1

            if self.config.verbose and (epoch % self.config.log_interval == 0 or epoch == self.config.n_epochs - 1):
                val_str = ""
                if has_val:
                    val_str = f"  val_nll={val_nll:.4f}  val_elbo={val_elbo:.4f}"
                logger.info(
                    "Epoch %4d/%d | loss=%.4f  nll=%.4f  kl=%.2f  beta=%.3f  grad=%.2f%s",
                    epoch + 1, self.config.n_epochs,
                    epoch_loss, epoch_nll, epoch_kl, beta, avg_grad_norm, val_str,
                )

            # ── DIAGNOSTIC: Detailed snapshot at key epochs ──
            # Log at epochs 1, 5, 10, 25, 50, 100, and every 100 thereafter
            if epoch + 1 in (1, 5, 10, 25, 50, 100) or (epoch + 1) % 100 == 0:
                self.model.eval()
                with torch.no_grad():
                    # Quick prediction on val set (or train subset)
                    diag_X = X_val if has_val else X_train[:min(256, len(X_train))]
                    diag_y = y_val if has_val else y_train[:min(256, len(X_train))]
                    out_diag, kl_diag = self.model(diag_X)
                    mu_diag = out_diag[:, :od]
                    logvar_diag = out_diag[:, od:2*od].clamp(-10, 10)
                    resid = mu_diag.squeeze() - diag_y.squeeze()
                    mae_diag = resid.abs().mean().item()
                    bias_diag = resid.mean().item()
                    sigma_diag = logvar_diag.exp().sqrt().mean().item()

                    # Posterior summary
                    from .model import BayesianLinear
                    post_stds = []
                    post_mus = []
                    for name, mod in self.model.named_modules():
                        if isinstance(mod, BayesianLinear) and any(p.requires_grad for p in mod.parameters()):
                            post_stds.append(F.softplus(mod.weight_rho).detach().mean().item())
                            post_mus.append(mod.weight_mu.detach().abs().mean().item())

                    set_name = "val" if has_val else "train_subset"
                    logger.info(
                        "  ┌─ SNAPSHOT epoch %d (%s, n=%d) ─┐", epoch + 1, set_name, len(diag_X))
                    logger.info(
                        "  │ mu: mean=%.4f std=%.4f | sigma: %.4f | MAE=%.4f bias=%.4f",
                        mu_diag.mean().item(), mu_diag.std().item(),
                        sigma_diag, mae_diag, bias_diag)
                    if post_stds:
                        mean_post_std = float(np.mean(post_stds))
                        mean_post_mu = float(np.mean(post_mus))
                        logger.info(
                            "  │ posterior: mean_|μ|=%.4f  mean_σ=%.4f  collapse_ratio=%.4f",
                            mean_post_mu, mean_post_std,
                            mean_post_std / max(getattr(self.model, 'prior_std', 1.0), 1e-8))
                    logger.info(
                        "  │ grad_norm=%.2f  KL=%.1f  beta=%.3f  scaled_KL=%.4f",
                        avg_grad_norm, kl_diag.item(), beta,
                        (beta / n_data) * kl_diag.item())
                    # Check if predictions are degenerately constant
                    pred_range = mu_diag.max().item() - mu_diag.min().item()
                    pred_std = mu_diag.std().item()
                    if pred_range < 0.01:
                        logger.warning(
                            "  │ ⚠ DEGENERATE: prediction range=%.6f (nearly constant!)", pred_range)
                    logger.info(
                        "  │ pred_range=%.4f  pred_std=%.4f  y_range=%.4f  y_std=%.4f",
                        pred_range, pred_std,
                        (diag_y.max() - diag_y.min()).item(), diag_y.std().item())
                    if self.config.loss_type == "hurdle":
                        cls_logit = out_diag[:, 2*od:]
                        cls_prob = torch.sigmoid(cls_logit)
                        logger.info(
                            "  │ cls_logit: mean=%.4f std=%.4f | P(active): mean=%.3f std=%.3f",
                            cls_logit.mean().item(), cls_logit.std().item(),
                            cls_prob.mean().item(), cls_prob.std().item())
                    logger.info("  └──────────────────────────────────┘")
                self.model.train()

            if (
                self.config.early_stopping_patience > 0
                and self._no_improve_count >= self.config.early_stopping_patience
            ):
                if self.config.verbose:
                    logger.info("Early stopping at epoch %d. Best epoch: %d",
                                epoch + 1, self.history.best_epoch + 1)
                break

        # ═══════════════════════════════════════════════════════════════════
        # DIAGNOSTIC: Post-training summary
        # ═══════════════════════════════════════════════════════════════════
        n_epochs_actual = len(self.history.train_loss)
        logger.info("┌─── POST-TRAINING SUMMARY ───┐")
        logger.info("│ Epochs trained: %d / %d (best: %d)",
                     n_epochs_actual, self.config.n_epochs, self.history.best_epoch + 1)
        logger.info("│ Train loss: start=%.4f  end=%.4f  best=%.4f",
                     self.history.train_loss[0],
                     self.history.train_loss[-1],
                     min(self.history.train_loss))
        logger.info("│ Train NLL:  start=%.4f  end=%.4f",
                     self.history.train_nll[0], self.history.train_nll[-1])
        logger.info("│ Train KL:   start=%.1f  end=%.1f",
                     self.history.train_kl[0], self.history.train_kl[-1])
        if self.history.val_nll:
            logger.info("│ Val NLL (early-stop monitor): start=%.4f  end=%.4f  best=%.4f",
                         self.history.val_nll[0],
                         self.history.val_nll[-1],
                         min(self.history.val_nll))
        if self.history.val_loss:
            logger.info("│ Val ELBO (logged only):       start=%.4f  end=%.4f  best=%.4f",
                         self.history.val_loss[0],
                         self.history.val_loss[-1],
                         min(self.history.val_loss))
        logger.info("│ Grad norms: start=%.2f  end=%.2f  max=%.2f  min=%.2f",
                     _grad_norms[0], _grad_norms[-1],
                     max(_grad_norms), min(_grad_norms))
        # Loss improvement ratio
        if len(self.history.train_loss) > 1:
            loss_ratio = self.history.train_loss[-1] / max(self.history.train_loss[0], 1e-10)
            logger.info("│ Loss ratio (end/start): %.4f", loss_ratio)
            if loss_ratio > 0.95:
                logger.warning("│ ⚠ Loss barely decreased — model may not be learning!")

        # Restore best weights
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)

        # DIAGNOSTIC: Prediction on val set with best model
        if has_val:
            self.model.eval()
            with torch.no_grad():
                out_final, _ = self.model(X_val)
                mu_final = out_final[:, :od]
                resid_final = mu_final.squeeze() - y_val.squeeze()
                logger.info("│ Best-model val predictions (raw delta, no fc_ref):")
                logger.info("│   mu: mean=%.4f  std=%.4f  range=[%.4f, %.4f]",
                             mu_final.mean().item(), mu_final.std().item(),
                             mu_final.min().item(), mu_final.max().item())
                logger.info("│   residual: mean=%.4f  MAE=%.4f  RMSE=%.4f",
                             resid_final.mean().item(),
                             resid_final.abs().mean().item(),
                             (resid_final**2).mean().sqrt().item())
                # Compare to null model (predict zero delta)
                null_mae = y_val.squeeze().abs().mean().item()
                model_mae = resid_final.abs().mean().item()
                logger.info("│   Null MAE (predict Δ=0): %.4f", null_mae)
                logger.info("│   Model MAE (raw delta):  %.4f", model_mae)
                if model_mae > null_mae:
                    logger.warning(
                        "│   ⚠ MODEL WORSE THAN NULL in delta space! "
                        "(model_mae=%.4f > null_mae=%.4f, ratio=%.2f)",
                        model_mae, null_mae, model_mae / max(null_mae, 1e-10))
                else:
                    logger.info(
                        "│   ✓ Model beats null: improvement=%.1f%%",
                        100.0 * (1.0 - model_mae / max(null_mae, 1e-10)))

        logger.info("└──────────────────────────────┘")

        return self.history

    @torch.no_grad()
    def _validation_loss(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        n_data: int,
        beta: float,
        n_samples: int = 10,
    ) -> tuple[float, float]:
        """
        Compute average NLL and ELBO on the validation set.

        Averages over multiple weight samples (and dropout masks) to produce
        a stable estimate. Early stopping uses val NLL (not the full ELBO)
        because the KL term measures posterior divergence from the prior, not
        predictive quality. Including KL in the monitor biases early stopping
        toward the prior (epoch 0) and prevents the model from improving.

        Args:
            X_val (torch.Tensor): Validation inputs.
            y_val (torch.Tensor): Validation targets.
            n_data (int): Training set size (for consistent KL scaling).
            beta (float): KL weight for the ELBO computation (logged only).
            n_samples (int): Number of weight samples to average over.

        Returns:
            (val_nll, val_elbo): Per-example averages. val_nll is used for
            early stopping; val_elbo is logged for diagnostics.
        """
        self.model.train()  # keep dropout active for MC estimation
        nll_accum = 0.0
        elbo_accum = 0.0

        for _ in range(n_samples):
            output, kl = self.model(X_val)
            nll = self._compute_nll(output, y_val,
                                    is_floor=getattr(self, "_is_floor_val", None))
            nll_accum += nll.item()
            elbo_accum += (nll + (beta / n_data) * kl).item()

        val_nll = nll_accum / (n_samples * len(X_val))
        val_elbo = elbo_accum / (n_samples * len(X_val))
        return val_nll, val_elbo

    def predict(
        self,
        X: torch.Tensor,
        n_samples: int = None,
        hurdle_config: Optional[HurdleConfig] = None,
        fc_ref: Optional[torch.Tensor] = None,
        prediction_floor: Optional[float] = None,
    ) -> UncertaintyEstimate:
        """
        Run inference with full uncertainty decomposition.

        Args:
            X (torch.Tensor): Input tensor of shape (N, input_dim).
            n_samples (int, optional): MC weight samples. Defaults to
                config.n_inference_samples. Override for faster/slower inference.
            hurdle_config (HurdleConfig, optional): Sub-parameters for hurdle
                inference (thresholds, floor value). Only used when
                ``loss_type == "hurdle"``. Falls back to self.config.hurdle.
            fc_ref (torch.Tensor, optional): Reference log_fc values (N, 1) for
                delta learning reconstruction. When provided, model predictions
                (deltas) are shifted to absolute scale: pred_abs = pred_delta + fc_ref.
            prediction_floor (float, optional): Minimum predicted value in absolute
                space. MC mu samples are clamped at this floor after the fc_ref
                shift. None = no floor.

        Returns:
            UncertaintyEstimate or HurdleUncertaintyEstimate.
        """
        n_samples = n_samples or self.config.n_inference_samples
        X = X.to(self.device)
        if fc_ref is not None:
            fc_ref = fc_ref.to(self.device)
        hc = hurdle_config if hurdle_config is not None else self.config.hurdle
        if self.config.loss_type == "hurdle":
            return self.model.predict_with_uncertainty(
                X, n_samples=n_samples, hurdle_config=hc, fc_ref=fc_ref,
                prediction_floor=prediction_floor)
        return self.model.predict_with_uncertainty(
            X, n_samples=n_samples, fc_ref=fc_ref,
            prediction_floor=prediction_floor)

    def save(self, path: str) -> None:
        """
        Save model weights and training config to disk.

        Args:
            path (str): File path (e.g. 'checkpoints/bnn_model.pt').
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "history": self.history,
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Load model weights from a checkpoint saved with `save`.

        Note: The model architecture must match the checkpoint. Create an
        identically configured BayesianMLP before calling this.

        Args:
            path (str): File path to the checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]