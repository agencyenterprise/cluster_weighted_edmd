"""
Continuous EDMD local model implemented as a LocalModel.

Fits a continuous Koopman generator M via ridge-regularized solve on
polynomial-lifted features and their gradients. Prediction returns
the velocity field: F_pred = [M @ Phi(X - center)]_{1:d}.
"""

from __future__ import annotations
from typing import Any

import torch

from ..em_local_edmd import monomial_exponents, monomials, monomials_grad


class PolynomialContinuousEDMD:
    """Continuous EDMD local model using polynomial lifting."""

    def __init__(self, degree: int = 2, ridge: float = 1e-4):
        self.degree = degree
        self.ridge = ridge
        self._M: torch.Tensor | None = None
        self._exps: list | None = None
        self._d: int | None = None

    @property
    def min_points(self) -> int:
        if self._exps is not None:
            return len(self._exps)
        return 1

    def _ensure_exps(self, d: int):
        if self._exps is None or self._d != d:
            self._d = d
            self._exps = monomial_exponents(d, self.degree)

    def fit(self, X: torch.Tensor, Y: torch.Tensor,
            weights: torch.Tensor, center: torch.Tensor) -> None:
        """Y is F (velocity field), not X_next."""
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
        """Returns predicted velocity (no center offset)."""
        U = X - center
        Phi = monomials(U, self._exps)
        Phi_dot = Phi @ self._M.T
        return Phi_dot[:, 1:self._d + 1]

    def state_dict(self) -> dict[str, Any]:
        return {
            'M': self._M,
            'exps': self._exps,
            'd': self._d,
            'degree': self.degree,
            'ridge': self.ridge,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._M = state['M']
        self._exps = state['exps']
        self._d = state['d']

    def clone(self) -> PolynomialContinuousEDMD:
        return PolynomialContinuousEDMD(degree=self.degree, ridge=self.ridge)

    def fallback_init(self, d: int, device: torch.device, dtype: torch.dtype) -> None:
        self._ensure_exps(d)
        Mdim = len(self._exps)
        self._M = torch.zeros(Mdim, Mdim, dtype=dtype, device=device)

    def to(self, device: torch.device, dtype: torch.dtype) -> PolynomialContinuousEDMD:
        if self._M is not None:
            self._M = self._M.to(device=device, dtype=dtype)
        return self
