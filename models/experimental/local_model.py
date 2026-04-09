"""
LocalModel protocol -- the interface that all local surrogate models must implement.

This is the key abstraction enabling pluggable model types (polynomial EDMD,
neural networks, transformers, or any future architecture) into the generic
EM pipeline. Each cluster k in the residual-aware clustering holds its own
LocalModel instance, which learns a mapping X -> Y locally around a cluster
center, where Y = X_next for discrete-time models or Y = F (velocity) for
continuous-time models.

Usage
-----
Implementing a custom LocalModel::

    from residual_aware_clustering.models.experimental.local_model import LocalModel

    class MyCustomModel:
        \"\"\"Must satisfy the LocalModel protocol (runtime-checkable).\"\"\"

        @property
        def min_points(self) -> int:
            return 5  # minimum effective sample size to attempt fitting

        def fit(self, X, Y, weights, center):
            # Fit on weighted data centered around `center`.
            # X: (P, d) inputs, Y: (P, d) targets
            # weights: (P,) soft-assignment weights for this cluster
            # center: (d,) cluster center
            ...

        def predict(self, X, center):
            # Return (P, d) predictions for all points.
            # Discrete models: return predicted next state (include center offset).
            # Continuous models: return predicted velocity (no center offset).
            ...

        def clone(self):
            # Return a fresh instance with same config but NO fitted state.
            # The generic EM calls clone() N times to create one model per cluster.
            ...

        def fallback_init(self, d, device, dtype):
            # Initialize to a safe fallback (e.g. identity) when too few points
            # are assigned to the cluster (effective weight < min_points).
            ...

        def state_dict(self):
            # Return a serializable dict of fitted parameters.
            ...

        def load_state_dict(self, state):
            # Restore fitted parameters from a state dict.
            ...

        def to(self, device, dtype):
            # Move model parameters to the given device/dtype. Return self.
            ...

    # Verify at runtime:
    assert isinstance(MyCustomModel(), LocalModel)

Plugging into the EM pipeline::

    from residual_aware_clustering.models.experimental import generic_em

    prototype = MyCustomModel()
    state, r, history = generic_em.fit(
        X, Y, N=8, hp=hp, model_prototype=prototype,
    )

Key concepts
------------
- **Protocol, not base class**: LocalModel is a ``typing.Protocol`` decorated
  with ``@runtime_checkable``. You do NOT inherit from it -- just implement
  all the required methods and the structural subtype check passes.
- **clone()**: The EM pipeline receives a single ``model_prototype`` and calls
  ``clone()`` N times to create independent model instances, one per cluster.
  ``clone()`` must return the same configuration but with no fitted state.
- **Centered coordinates**: ``fit()`` and ``predict()`` receive a ``center``
  argument. Models should work in centered coordinates (X - center) internally
  to improve numerical conditioning.
- **fallback_init()**: When a cluster has fewer than ``min_points`` effective
  samples, the pipeline calls ``fallback_init()`` instead of ``fit()``. This
  should set the model to a safe default (e.g. identity operator for discrete,
  zeros for continuous).
- **Device/dtype portability**: The EM pipeline may move computations between
  devices (e.g. MPS -> CPU for float64 EM). ``to(device, dtype)`` must move
  all internal tensors accordingly.

Method contracts
----------------
``min_points`` (property)
    Return the minimum effective sample size needed to fit. If the weighted
    count falls below this, ``fallback_init`` is called instead of ``fit``.

``fit(X, Y, weights, center)``
    Fit the model on soft-weighted data. ``weights`` sums to the effective
    sample count for this cluster. The model should use centered inputs
    (X - center) for numerical stability.

``predict(X, center)``
    Return (P, d) predictions. Discrete models add center back to the output;
    continuous models return raw velocity (no offset).

``state_dict() / load_state_dict(state)``
    Round-trip serialization of fitted parameters.

``clone()``
    Fresh copy with identical hyperparameters but no fitted state.

``fallback_init(d, device, dtype)``
    Safe initialization when data is insufficient.

``to(device, dtype)``
    Move all internal tensors. Return ``self``.
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
