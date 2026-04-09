"""
PolynomialDiscreteEDMD -- discrete EDMD local model with polynomial lifting.

Implements the ``LocalModel`` protocol for use with the generic EM pipeline.
Fits a discrete Koopman operator K that maps polynomial-lifted states at time
t to lifted states at time t+1, using SVD-based weighted least squares
(``_lstsq_svd``) for GPU compatibility. Supports FULL multivariate monomials
and DIAGONAL (univariate-only) monomials via the ``ObservableType`` enum.

Usage
-----
Standalone (single cluster, uniform weights)::

    import torch
    from residual_aware_clustering.models.experimental import PolynomialDiscreteEDMD
    from residual_aware_clustering.models.em_local_edmd import ObservableType

    model = PolynomialDiscreteEDMD(degree=2, observable_type=ObservableType.FULL)

    X = torch.randn(500, 4)         # (P, d) current states
    Y = torch.randn(500, 4)         # (P, d) next states
    weights = torch.ones(500)        # uniform weights
    center = X.mean(dim=0)           # cluster center

    model.fit(X, Y, weights, center)
    Y_pred = model.predict(X, center)  # (500, 4) predicted next states

With the generic EM pipeline::

    from residual_aware_clustering.models.experimental import (
        generic_em, PolynomialDiscreteEDMD, ObservableType,
    )

    prototype = PolynomialDiscreteEDMD(
        degree=3,
        observable_type=ObservableType.DIAGONAL,  # cheaper for high-d
        rcond=1e-10,
    )
    state, r, history = generic_em.fit(
        X, X_next, N=8, hp=hp, model_prototype=prototype,
    )

Key concepts
------------
- **Polynomial lifting**: Input states (X - center) are lifted to a
  monomial feature space Phi(X - center) of dimension M. The Koopman
  operator K is an (M, M) matrix such that Phi(X_next - center) ~ K @ Phi(X - center).
- **Observable types**: ``ObservableType.FULL`` uses all multivariate
  monomials up to the given degree (M grows combinatorially with d).
  ``ObservableType.DIAGONAL`` uses only univariate monomials (M = 1 + d * degree),
  which is far cheaper for high-dimensional data.
- **SVD solve**: Uses ``_lstsq_svd`` (from the GPU EDMD module) instead of
  ``torch.linalg.lstsq`` for reliable behavior on CUDA and MPS backends.
- **Prediction**: ``predict()`` extracts the state-space columns (indices 1:d+1)
  from the lifted prediction and adds the center back, returning the
  predicted next state in the original coordinate frame.
- **Fallback**: When too few points are assigned, ``fallback_init()`` sets K
  to the identity matrix (predicting no change in lifted space).
"""

from __future__ import annotations
from typing import Any

import torch

from ..em_local_edmd import make_exponents, monomials, ObservableType
from ..em_local_edmd_discrete_gpu import _lstsq_svd


class PolynomialDiscreteEDMD:
    """Discrete EDMD local model using polynomial lifting.

    Args:
        degree: Maximum polynomial degree.
        observable_type: ObservableType enum or string ('full', 'diagonal').
        rcond: Cutoff for SVD pseudoinverse.
    """

    def __init__(self, degree: int = 2,
                 observable_type: ObservableType | str = ObservableType.FULL,
                 rcond: float = 1e-10):
        """Initialize a discrete polynomial EDMD local model.

        Parameters
        ----------
        degree : int
            Maximum polynomial degree for the monomial lifting.
        observable_type : ObservableType or str
            Type of monomial observables ('full' or 'diagonal').
        rcond : float
            Cutoff for SVD pseudoinverse in the least-squares solve.
        """
        self.degree = degree
        self.observable_type = ObservableType(observable_type) if isinstance(observable_type, str) else observable_type
        self.rcond = rcond
        self._K: torch.Tensor | None = None
        self._exps: list | None = None
        self._d: int | None = None

    @property
    def min_points(self) -> int:
        """Minimum effective sample size required to fit.

        Returns
        -------
        int
            Number of monomial terms if exponents are initialized, else 1.
        """
        if self._exps is not None:
            return len(self._exps)
        return 1

    def _ensure_exps(self, d: int):
        """Build or rebuild monomial exponents when the dimension changes.

        Parameters
        ----------
        d : int
            State-space dimension.
        """
        if self._exps is None or self._d != d:
            self._d = d
            self._exps = make_exponents(d, self.degree, self.observable_type)

    def fit(self, X: torch.Tensor, Y: torch.Tensor,
            weights: torch.Tensor, center: torch.Tensor) -> None:
        """Fit the discrete Koopman operator via weighted least squares.

        Parameters
        ----------
        X : torch.Tensor
            Current states, shape ``(P, d)``.
        Y : torch.Tensor
            Next states, shape ``(P, d)``.
        weights : torch.Tensor
            Per-sample soft-assignment weights, shape ``(P,)``.
        center : torch.Tensor
            Cluster center, shape ``(d,)``.
        """
        d = X.shape[1]
        self._ensure_exps(d)

        U_curr = X - center
        U_next = Y - center
        Phi_curr = monomials(U_curr, self._exps)
        Phi_next = monomials(U_next, self._exps)

        sqrt_w = torch.sqrt(weights).unsqueeze(1)
        A = Phi_curr * sqrt_w
        B = Phi_next * sqrt_w

        self._K = _lstsq_svd(A, B).T

    def predict(self, X: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        """Predict next states using the fitted Koopman operator.

        Parameters
        ----------
        X : torch.Tensor
            Input states, shape ``(P, d)``.
        center : torch.Tensor
            Cluster center, shape ``(d,)``.

        Returns
        -------
        torch.Tensor
            Predicted next states, shape ``(P, d)``.
        """
        U = X - center
        Phi = monomials(U, self._exps)
        Phi_next = Phi @ self._K.T
        return center + Phi_next[:, 1:self._d + 1]

    def state_dict(self) -> dict[str, Any]:
        """Return a serializable dictionary of fitted model parameters.

        Returns
        -------
        dict[str, Any]
            Contains 'K', 'exps', 'd', 'degree', 'observable_type', 'rcond'.
        """
        return {
            'K': self._K,
            'exps': self._exps,
            'd': self._d,
            'degree': self.degree,
            'observable_type': self.observable_type.value,
            'rcond': self.rcond,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore fitted parameters from a state dictionary.

        Parameters
        ----------
        state : dict[str, Any]
            Dictionary previously returned by ``state_dict()``.
        """
        self._K = state['K']
        self._exps = state['exps']
        self._d = state['d']

    def clone(self) -> PolynomialDiscreteEDMD:
        """Return a fresh instance with the same hyperparameters but no fitted state.

        Returns
        -------
        PolynomialDiscreteEDMD
            New unfitted model with identical configuration.
        """
        return PolynomialDiscreteEDMD(
            degree=self.degree,
            observable_type=self.observable_type,
            rcond=self.rcond,
        )

    def fallback_init(self, d: int, device: torch.device, dtype: torch.dtype) -> None:
        """Initialize K to the identity matrix as a safe fallback.

        Parameters
        ----------
        d : int
            State-space dimension.
        device : torch.device
            Target device.
        dtype : torch.dtype
            Target dtype.
        """
        self._ensure_exps(d)
        Mdim = len(self._exps)
        self._K = torch.eye(Mdim, dtype=dtype, device=device)

    def to(self, device: torch.device, dtype: torch.dtype) -> PolynomialDiscreteEDMD:
        """Move model parameters to the given device and dtype.

        Parameters
        ----------
        device : torch.device
            Target device.
        dtype : torch.dtype
            Target dtype.

        Returns
        -------
        PolynomialDiscreteEDMD
            Self, for method chaining.
        """
        if self._K is not None:
            self._K = self._K.to(device=device, dtype=dtype)
        return self
