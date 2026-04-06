"""
Global continuous EDMD using our own solver.

Uses the same weighted_continuous_edmd as local EDMD (same ridge, same code path)
with N=1 and uniform weights, so the comparison local-vs-global is fair.
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

    Uses the same solver as local EDMD (weighted_continuous_edmd with
    uniform weights), ensuring a fair comparison.

    Args:
        X: (P, d) training state vectors
        F: (P, d) vector field observations
        degree: polynomial lift degree
        ridge: regularization (same default as local EDMD)

    Returns dict with 'M', 'c', 'exps', 'd'.
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

    Returns: (P, d) predicted vector field.
    """
    d = model['d']
    U = X - model['c']
    Phi = monomials(U, model['exps'])
    Phi_dot = Phi @ model['M'].T
    return Phi_dot[:, 1:d+1]
