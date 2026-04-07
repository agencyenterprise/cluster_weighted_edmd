"""
Neural network local model wrapper for the generic EM pipeline.

Trains a small MLP per cluster to learn X -> Y mappings locally around
each cluster center. For discrete models (predict_offset=True), the network
learns the offset: net(X - center) ≈ Y - center. For continuous models
(predict_offset=False), it learns the velocity directly: net(X - center) ≈ F.
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
        return self.min_samples

    def _build_net(self, d: int) -> nn.Module:
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
        self._net.eval()
        with torch.no_grad():
            U = X - center
            out = self._net(U)
        if self.predict_offset:
            return center + out
        return out

    def state_dict(self) -> dict[str, Any]:
        return {
            'net_state': self._net.state_dict() if self._net else None,
            'd': self._d,
            'hidden_dims': self.hidden_dims,
            'lr': self.lr,
            'n_epochs': self.n_epochs,
            'predict_offset': self.predict_offset,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._d = state['d']
        if state['net_state'] is not None:
            self._net = self._build_net(self._d)
            self._net.load_state_dict(state['net_state'])

    def clone(self) -> NeuralNetModel:
        return NeuralNetModel(
            hidden_dims=self.hidden_dims,
            lr=self.lr,
            n_epochs=self.n_epochs,
            patience=self.patience,
            predict_offset=self.predict_offset,
            min_samples=self.min_samples,
        )

    def fallback_init(self, d: int, device: torch.device, dtype: torch.dtype) -> None:
        self._d = d
        self._net = self._build_net(d).to(device=device, dtype=dtype)

    def to(self, device: torch.device, dtype: torch.dtype) -> NeuralNetModel:
        if self._net is not None:
            self._net = self._net.to(device=device, dtype=dtype)
        return self
