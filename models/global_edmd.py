"""
Global (single-cluster) continuous EDMD baseline.

Fits one polynomial Koopman generator M over the entire dataset using
the same ``weighted_continuous_edmd`` solver as the local (per-cluster)
EDMD, but with N=1 and uniform weights.  This ensures a fair
comparison: any improvement from local clustering is due to
partitioning, not a different solver or regularization.

Functions
---------
- ``fit(X, F, degree, ridge=1e-4)`` -- fit a single global operator.
  Returns a dict with ``'M'`` (generator matrix), ``'c'`` (data
  centroid), ``'exps'`` (monomial exponents), and ``'d'`` (state dim).
- ``predict_f(X, model)`` -- predict the vector field f(x) at new
  points using the fitted operator.

Usage
-----
::

    from residual_aware_clustering.models.global_edmd import fit, predict_f
    import torch

    X = torch.tensor(data['X'], dtype=torch.float64)
    F = torch.tensor(data['F'], dtype=torch.float64)

    model = fit(X, F, degree=2)
    F_hat = predict_f(X, model)          # (P, d)
    rmse  = (F - F_hat).pow(2).mean().sqrt()

Key concepts
------------
- **Fair baseline**: uses identical code path (``weighted_continuous_edmd``
  with ridge regularization) as the local EDMD clusters -- only the
  number of clusters differs (N=1 vs N>1).
- **Polynomial lift**: monomials up to ``degree`` are constructed
  relative to the data centroid ``c = X.mean(dim=0)``.
"""

import torch
from .em_local_edmd import (
    weighted_continuous_edmd,
    monomial_exponents,
    monomials,
)


def fit(X, F, degree, ridge=1e-4):
    """
    Fit a single global continuous EDMD operator.

    Uses the same solver as local EDMD (``weighted_continuous_edmd`` with
    uniform weights), ensuring a fair comparison.

    Parameters
    ----------
    X : torch.Tensor
        Training state vectors, shape ``(P, d)``.
    F : torch.Tensor
        Vector field observations, shape ``(P, d)``.
    degree : int
        Polynomial lift degree.
    ridge : float, optional
        Ridge regularization, by default 1e-4.

    Returns
    -------
    dict
        Model dict with keys ``'M'`` (generator), ``'c'`` (centroid),
        ``'exps'`` (monomial exponents), ``'d'`` (state dimension).
    """
    d = X.shape[1]
    exps = monomial_exponents(d, degree)
    c = X.mean(dim=0)
    r = torch.ones(X.shape[0], dtype=X.dtype)
    M = weighted_continuous_edmd(X, F, r, c, exps, ridge=ridge)
    return {'M': M, 'c': c, 'exps': exps, 'd': d}


def predict_f(X, model):
    """
    Predict f(x) from the global EDMD operator.

    Parameters
    ----------
    X : torch.Tensor
        State points, shape ``(P, d)``.
    model : dict
        Fitted model returned by :func:`fit`.

    Returns
    -------
    torch.Tensor
        Predicted vector field, shape ``(P, d)``.
    """
    d = model['d']
    U = X - model['c']
    Phi = monomials(U, model['exps'])
    Phi_dot = Phi @ model['M'].T
    return Phi_dot[:, 1:d+1]
