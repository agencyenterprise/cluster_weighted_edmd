"""
Discrete EDMD local model implemented as a LocalModel.

Fits a discrete Koopman operator K via SVD-based least squares on
polynomial-lifted features. Supports full multivariate monomials
or diagonal (univariate-only) monomials.
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
        self.degree = degree
        self.observable_type = ObservableType(observable_type) if isinstance(observable_type, str) else observable_type
        self.rcond = rcond
        self._K: torch.Tensor | None = None
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
            self._exps = make_exponents(d, self.degree, self.observable_type)

    def fit(self, X: torch.Tensor, Y: torch.Tensor,
            weights: torch.Tensor, center: torch.Tensor) -> None:
        d = X.shape[1]
        self._ensure_exps(d)

        U_curr = X - center
        U_next = Y - center
        Phi_curr = monomials(U_curr, self._exps)
        Phi_next = monomials(U_next, self._exps)

        sqrt_w = torch.sqrt(weights).unsqueeze(1)
        A = Phi_curr * sqrt_w
        B = Phi_next * sqrt_w

        self._K = _lstsq_svd(A, B, rcond=self.rcond).T

    def predict(self, X: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        U = X - center
        Phi = monomials(U, self._exps)
        Phi_next = Phi @ self._K.T
        return center + Phi_next[:, 1:self._d + 1]

    def state_dict(self) -> dict[str, Any]:
        return {
            'K': self._K,
            'exps': self._exps,
            'd': self._d,
            'degree': self.degree,
            'observable_type': self.observable_type.value,
            'rcond': self.rcond,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._K = state['K']
        self._exps = state['exps']
        self._d = state['d']

    def clone(self) -> PolynomialDiscreteEDMD:
        return PolynomialDiscreteEDMD(
            degree=self.degree,
            observable_type=self.observable_type,
            rcond=self.rcond,
        )

    def fallback_init(self, d: int, device: torch.device, dtype: torch.dtype) -> None:
        self._ensure_exps(d)
        Mdim = len(self._exps)
        self._K = torch.eye(Mdim, dtype=dtype, device=device)

    def to(self, device: torch.device, dtype: torch.dtype) -> PolynomialDiscreteEDMD:
        if self._K is not None:
            self._K = self._K.to(device=device, dtype=dtype)
        return self
