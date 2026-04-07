"""
LocalModel protocol — the interface between the EM pipeline and any local dynamical model.

Each cluster k in the residual-aware clustering has its own LocalModel instance.
The model learns a mapping X -> Y locally around a cluster center, where:
  - Y = X_next for discrete-time models
  - Y = F (velocity) for continuous-time models
"""

from __future__ import annotations
from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class LocalModel(Protocol):
    """Protocol for local dynamical models used within the generic EM pipeline."""

    @property
    def min_points(self) -> int:
        """Minimum effective sample size to fit. Below this, fallback_init is used."""
        ...

    def fit(self, X: torch.Tensor, Y: torch.Tensor,
            weights: torch.Tensor, center: torch.Tensor) -> None:
        """Fit the local model on weighted data.

        Args:
            X: (P, d) input states
            Y: (P, d) targets (X_next for discrete, F for continuous)
            weights: (P,) soft assignment weights for this cluster
            center: (d,) cluster center (model should work in centered coords)
        """
        ...

    def predict(self, X: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        """Predict Y for all points.

        Args:
            X: (P, d) input states
            center: (d,) cluster center

        Returns:
            (P, d) predicted targets. For discrete models this is the predicted
            next state (including center offset). For continuous models this is
            the predicted velocity (no center offset).
        """
        ...

    def state_dict(self) -> dict[str, Any]:
        """Return serializable model state."""
        ...

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore model from a state dict."""
        ...

    def clone(self) -> LocalModel:
        """Return a fresh instance with same config but no fitted state."""
        ...

    def fallback_init(self, d: int, device: torch.device, dtype: torch.dtype) -> None:
        """Initialize to a safe fallback when too few points are available."""
        ...

    def to(self, device: torch.device, dtype: torch.dtype) -> LocalModel:
        """Move model parameters to device/dtype. Returns self."""
        ...
