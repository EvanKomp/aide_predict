"""
bnn — Bayesian Neural Network with Full Uncertainty Decomposition
=================================================================
A clean, modular implementation of a Bayesian MLP using mean-field variational
inference. Supports both regression (with heteroscedastic aleatoric uncertainty)
and classification.

Public API
----------
    BayesianMLP       : The model. Configure architecture and priors here.
    BNNTrainer        : Training loop with ELBO loss and KL annealing.
    TrainingConfig    : Dataclass of all training hyperparameters.
    TrainingHistory   : Training loss curves returned by trainer.fit().
    UncertaintyEstimate : Container for decomposed uncertainty from inference.
    BayesianLinear    : The underlying Bayesian linear layer (if you need it directly).

Quickstart — Regression
-----------------------
    import torch
    from bnn import BayesianMLP, BNNTrainer, TrainingConfig

    model = BayesianMLP(
        input_dim=10,
        hidden_dims=[128, 64],
        output_dim=1,
        task="regression",
        prior_std=0.5,
    )

    config = TrainingConfig(
        n_epochs=200,
        learning_rate=1e-3,
        kl_anneal_epochs=30,
        kl_weight=1.0,
        n_inference_samples=100,
    )

    trainer = BNNTrainer(model, config)
    history = trainer.fit(X_train, y_train, X_val, y_val)

    # Inference with uncertainty decomposition
    estimates = trainer.predict(X_test)
    print(estimates.mean)           # predictive mean
    print(estimates.epistemic_std)  # uncertainty from weight posterior
    print(estimates.aleatoric_std)  # uncertainty from data noise
    print(estimates.total_std)      # combined uncertainty

Quickstart — Classification
---------------------------
    model = BayesianMLP(
        input_dim=10,
        hidden_dims=[64, 32],
        output_dim=5,
        task="classification",
        prior_std=1.0,
    )
    # Same training API. estimates.mean is mean class probabilities.
    # estimates.aleatoric_std and total_std are per-example entropy scalars.
"""

from .layers import BayesianLinear
from .model import BayesianMLP, UncertaintyEstimate
from .trainer import BNNTrainer, TrainingConfig, TrainingHistory, HurdleConfig

__all__ = [
    "BayesianMLP",
    "BayesianLinear",
    "BNNTrainer",
    "TrainingConfig",
    "TrainingHistory",
    "HurdleConfig",
    "UncertaintyEstimate",
]