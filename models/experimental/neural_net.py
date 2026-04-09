"""
NeuralNetModel -- MLP-based local model for the generic EM pipeline.

Implements the ``LocalModel`` protocol using a small multi-layer perceptron
(MLP) per cluster. The network learns local X -> Y mappings around each
cluster center with weighted MSE loss, Adam optimization, and early stopping.
This is a good choice for high-dimensional data where polynomial lifting
would be prohibitively expensive (monomial count grows combinatorially with
dimension, while MLP cost is linear in dimension).

For discrete models (``predict_offset=True``, the default), the network
learns the offset: ``net(X - center) ~ Y - center``, and prediction adds
center back. For continuous models (``predict_offset=False``), it learns
the velocity directly: ``net(X - center) ~ F``.

Usage
-----
Standalone::

    import torch
    from residual_aware_clustering.models.experimental import NeuralNetModel

    model = NeuralNetModel(
        hidden_dims=(128, 128),   # two hidden layers of 128 units
        lr=1e-3,                  # Adam learning rate
        n_epochs=80,              # max training epochs per EM iteration
        patience=10,              # early stopping patience
        predict_offset=True,      # discrete mode (learn X_next - center)
    )

    X = torch.randn(500, 20)        # (P, d) current states
    Y = torch.randn(500, 20)        # (P, d) next states
    weights = torch.ones(500)        # uniform weights
    center = X.mean(dim=0)

    model.fit(X, Y, weights, center)
    Y_pred = model.predict(X, center)  # (500, 20) predicted next states

With the generic EM pipeline::

    from residual_aware_clustering.models.experimental import generic_em, NeuralNetModel

    prototype = NeuralNetModel(hidden_dims=(64, 64), n_epochs=50)
    state, r, history = generic_em.fit(
        X, X_next, N=5, hp=hp, model_prototype=prototype,
    )

Key concepts
------------
- **Architecture**: ``nn.Sequential`` with ``nn.Linear`` layers and ``nn.SiLU``
  activations. Input and output dimensions match the state dimension d.
  Configure depth and width via ``hidden_dims`` (e.g. ``(64, 64)`` for two
  layers of 64 units).
- **Weighted loss**: The fit uses weighted MSE where each sample's contribution
  is scaled by its soft-assignment weight from the EM E-step. This ensures the
  network focuses on points belonging to its cluster.
- **Early stopping**: Training halts when the loss has not improved for
  ``patience`` consecutive epochs. The best-loss network state is restored.
- **Float32 training**: The MLP trains in float32 for efficiency (float64 is
  wasteful for SGD), then converts back to the EM dtype (float64) for
  prediction consistency.
- **Fallback**: When the effective sample size is below ``min_samples``, the
  network is initialized but not trained, producing random (but finite)
  predictions. This prevents NaN from empty-cluster updates.
"""

from __future__ import annotations
from typing import Any
import copy

import torch
import torch.nn as nn


class NeuralNetModel:
    """Neural network local model for use with the generic EM pipeline."""

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (64, 64),
        lr: float = 1e-3,
        n_epochs: int = 50,
        patience: int = 10,
        predict_offset: bool = True,
        min_samples: int = 10,
    ):
        """Initialize an MLP-based local model.

        Parameters
        ----------
        hidden_dims : tuple[int, ...]
            Sizes of hidden layers in the MLP.
        lr : float
            Adam learning rate.
        n_epochs : int
            Maximum training epochs per fit call.
        patience : int
            Early stopping patience (epochs without improvement).
        predict_offset : bool
            If True (discrete mode), learn ``Y - center``; if False
            (continuous mode), learn the velocity directly.
        min_samples : int
            Minimum effective sample size to attempt training.
        """
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.n_epochs = n_epochs
        self.patience = patience
        self.predict_offset = predict_offset
        self.min_samples = min_samples
        self._net: nn.Module | None = None
        self._d: int | None = None

    @property
    def min_points(self) -> int:
        """Minimum effective sample size required to fit.

        Returns
        -------
        int
            The configured ``min_samples`` threshold.
        """
        return self.min_samples

    def _build_net(self, d: int) -> nn.Module:
        """Construct the MLP architecture.

        Parameters
        ----------
        d : int
            Input and output dimension (state-space dimension).

        Returns
        -------
        nn.Module
            Sequential MLP with SiLU activations.
        """
        layers = []
        in_dim = d
        for h in self.hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.SiLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, d))
        return nn.Sequential(*layers)

    def fit(self, X: torch.Tensor, Y: torch.Tensor,
            weights: torch.Tensor, center: torch.Tensor) -> None:
        """Train the MLP on weighted data with early stopping.

        Parameters
        ----------
        X : torch.Tensor
            Input states, shape ``(P, d)``.
        Y : torch.Tensor
            Targets (next states or velocities), shape ``(P, d)``.
        weights : torch.Tensor
            Per-sample soft-assignment weights, shape ``(P,)``.
        center : torch.Tensor
            Cluster center, shape ``(d,)``.
        """
        d = X.shape[1]
        self._d = d
        dev, dt = X.device, X.dtype

        # Work in float32 for NN training (float64 is wasteful for SGD)
        U = (X - center).float()
        if self.predict_offset:
            target = (Y - center).float()
        else:
            target = Y.float()
        w = weights.float()

        # Effective sample size — skip if too small
        if w.sum().item() < self.min_samples:
            self._net = self._build_net(d).to(U.device)
            return

        self._net = self._build_net(d).to(U.device)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)

        best_loss = float("inf")
        best_state = None
        no_improve = 0

        self._net.train()
        for _ in range(self.n_epochs):
            pred = self._net(U)
            residual = (pred - target) ** 2
            loss = (w.unsqueeze(1) * residual).sum() / (w.sum() * d)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)

        self._net = self._net.to(dtype=dt)

    def predict(self, X: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        """Predict outputs using the trained MLP.

        Parameters
        ----------
        X : torch.Tensor
            Input states, shape ``(P, d)``.
        center : torch.Tensor
            Cluster center, shape ``(d,)``.

        Returns
        -------
        torch.Tensor
            Predictions, shape ``(P, d)``. Includes center offset if discrete mode.
        """
        self._net.eval()
        with torch.no_grad():
            U = X - center
            out = self._net(U)
        if self.predict_offset:
            return center + out
        return out

    def state_dict(self) -> dict[str, Any]:
        """Return a serializable dictionary of model parameters.

        Returns
        -------
        dict[str, Any]
            Contains 'net_state', 'd', and hyperparameters.
        """
        return {
            'net_state': self._net.state_dict() if self._net else None,
            'd': self._d,
            'hidden_dims': self.hidden_dims,
            'lr': self.lr,
            'n_epochs': self.n_epochs,
            'predict_offset': self.predict_offset,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore model from a state dictionary.

        Parameters
        ----------
        state : dict[str, Any]
            Dictionary previously returned by ``state_dict()``.
        """
        self._d = state['d']
        if state['net_state'] is not None:
            self._net = self._build_net(self._d)
            self._net.load_state_dict(state['net_state'])

    def clone(self) -> NeuralNetModel:
        """Return a fresh instance with the same hyperparameters but no fitted state.

        Returns
        -------
        NeuralNetModel
            New unfitted model with identical configuration.
        """
        return NeuralNetModel(
            hidden_dims=self.hidden_dims,
            lr=self.lr,
            n_epochs=self.n_epochs,
            patience=self.patience,
            predict_offset=self.predict_offset,
            min_samples=self.min_samples,
        )

    def fallback_init(self, d: int, device: torch.device, dtype: torch.dtype) -> None:
        """Initialize the network without training as a safe fallback.

        Parameters
        ----------
        d : int
            State-space dimension.
        device : torch.device
            Target device.
        dtype : torch.dtype
            Target dtype.
        """
        self._d = d
        self._net = self._build_net(d).to(device=device, dtype=dtype)

    def to(self, device: torch.device, dtype: torch.dtype) -> NeuralNetModel:
        """Move model parameters to the given device and dtype.

        Parameters
        ----------
        device : torch.device
            Target device.
        dtype : torch.dtype
            Target dtype.

        Returns
        -------
        NeuralNetModel
            Self, for method chaining.
        """
        if self._net is not None:
            self._net = self._net.to(device=device, dtype=dtype)
        return self
