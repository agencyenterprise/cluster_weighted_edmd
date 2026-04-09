"""
PolynomialContinuousEDMD -- continuous EDMD local model with polynomial lifting.

Implements the ``LocalModel`` protocol for use with the generic EM pipeline.
Instead of fitting a discrete map x_{t+1} = K @ Phi(x_t), this model fits a
continuous Koopman generator M that approximates the velocity field (vector
field) f(x) from data. Given polynomial-lifted features Phi and their
gradients grad(Phi), the generator M satisfies:

    d/dt Phi(x) = grad(Phi) @ f(x) ~ M @ Phi(x)

The velocity prediction extracts the state-space components (indices 1:d+1)
from M @ Phi(X - center). Fitting uses ridge-regularized least squares.

Usage
-----
Standalone (single cluster, uniform weights)::

    import torch
    from residual_aware_clustering.models.experimental import PolynomialContinuousEDMD

    model = PolynomialContinuousEDMD(degree=2, ridge=1e-4)

    X = torch.randn(500, 4)         # (P, d) states
    F = torch.randn(500, 4)         # (P, d) velocity / vector field at X
    weights = torch.ones(500)        # uniform weights
    center = X.mean(dim=0)           # cluster center

    model.fit(X, F, weights, center)
    F_pred = model.predict(X, center)  # (500, 4) predicted velocities

With the generic EM pipeline::

    from residual_aware_clustering.models.experimental import (
        generic_em, PolynomialContinuousEDMD,
    )

    prototype = PolynomialContinuousEDMD(degree=2, ridge=1e-4)
    state, r, history = generic_em.fit(
        X, F, N=6, hp=hp, model_prototype=prototype,
    )

Key concepts
------------
- **Continuous vs. discrete**: The discrete variant (``PolynomialDiscreteEDMD``)
  fits x_{t+1} from x_t. This continuous variant fits the velocity field f(x)
  directly, which is appropriate when you have time-derivative data or want to
  model a continuous-time ODE dx/dt = f(x).
- **Polynomial lifting**: Same monomial feature space as the discrete variant.
  Uses ``monomial_exponents`` / ``monomials`` / ``monomials_grad`` from the
  shared EDMD utilities.
- **Ridge regularization**: The ``ridge`` parameter adds Tikhonov regularization
  to the Gram matrix solve, preventing ill-conditioning when the monomial
  basis is near-singular.
- **Prediction convention**: Unlike the discrete model, ``predict()`` returns
  the raw velocity (no center offset is added), since velocities are
  translation-invariant.
- **Fallback**: ``fallback_init()`` sets M to zeros, predicting zero velocity
  (stationary dynamics) when data is insufficient.
"""

from __future__ import annotations
from typing import Any

import torch

from ..em_local_edmd import monomial_exponents, monomials, monomials_grad


class PolynomialContinuousEDMD:
    """Continuous EDMD local model using polynomial lifting."""

    def __init__(self, degree: int = 2, ridge: float = 1e-4):
        """Initialize a continuous polynomial EDMD local model.

        Parameters
        ----------
        degree : int
            Maximum polynomial degree for the monomial lifting.
        ridge : float
            Tikhonov regularization parameter for the Gram matrix solve.
        """
        self.degree = degree
        self.ridge = ridge
        self._M: torch.Tensor | None = None
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
            self._exps = monomial_exponents(d, self.degree)

    def fit(self, X: torch.Tensor, Y: torch.Tensor,
            weights: torch.Tensor, center: torch.Tensor) -> None:
        """Fit the continuous Koopman generator via ridge-regularized least squares.

        Parameters
        ----------
        X : torch.Tensor
            Input states, shape ``(P, d)``.
        Y : torch.Tensor
            Velocity field values ``f(x_i)``, shape ``(P, d)`` (not next states).
        weights : torch.Tensor
            Per-sample soft-assignment weights, shape ``(P,)``.
        center : torch.Tensor
            Cluster center, shape ``(d,)``.
        """
        d = X.shape[1]
        self._ensure_exps(d)

        U = X - center
        Phi = monomials(U, self._exps)
        grad = monomials_grad(U, self._exps)
        Phi_dot = (grad @ Y.unsqueeze(-1)).squeeze(-1)

        Mdim = len(self._exps)
        W = weights.unsqueeze(1)
        G = (Phi * W).T @ Phi
        A = (Phi_dot * W).T @ Phi
        self._M = torch.linalg.solve(
            G + self.ridge * torch.eye(Mdim, dtype=X.dtype, device=X.device),
            A.T,
        ).T

    def predict(self, X: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        """Predict the velocity field at the given states.

        Parameters
        ----------
        X : torch.Tensor
            Input states, shape ``(P, d)``.
        center : torch.Tensor
            Cluster center, shape ``(d,)``.

        Returns
        -------
        torch.Tensor
            Predicted velocities, shape ``(P, d)``. No center offset is added.
        """
        U = X - center
        Phi = monomials(U, self._exps)
        Phi_dot = Phi @ self._M.T
        return Phi_dot[:, 1:self._d + 1]

    def state_dict(self) -> dict[str, Any]:
        """Return a serializable dictionary of fitted model parameters.

        Returns
        -------
        dict[str, Any]
            Contains 'M', 'exps', 'd', 'degree', 'ridge'.
        """
        return {
            'M': self._M,
            'exps': self._exps,
            'd': self._d,
            'degree': self.degree,
            'ridge': self.ridge,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore fitted parameters from a state dictionary.

        Parameters
        ----------
        state : dict[str, Any]
            Dictionary previously returned by ``state_dict()``.
        """
        self._M = state['M']
        self._exps = state['exps']
        self._d = state['d']

    def clone(self) -> PolynomialContinuousEDMD:
        """Return a fresh instance with the same hyperparameters but no fitted state.

        Returns
        -------
        PolynomialContinuousEDMD
            New unfitted model with identical configuration.
        """
        return PolynomialContinuousEDMD(degree=self.degree, ridge=self.ridge)

    def fallback_init(self, d: int, device: torch.device, dtype: torch.dtype) -> None:
        """Initialize M to zeros (predicting zero velocity) as a safe fallback.

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
        self._M = torch.zeros(Mdim, Mdim, dtype=dtype, device=device)

    def to(self, device: torch.device, dtype: torch.dtype) -> PolynomialContinuousEDMD:
        """Move model parameters to the given device and dtype.

        Parameters
        ----------
        device : torch.device
            Target device.
        dtype : torch.dtype
            Target dtype.

        Returns
        -------
        PolynomialContinuousEDMD
            Self, for method chaining.
        """
        if self._M is not None:
            self._M = self._M.to(device=device, dtype=dtype)
        return self
