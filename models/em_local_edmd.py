"""
Residual-aware clustering with piecewise continuous Koopman generators.

Fits N local continuous-time Koopman generators M_k via Expectation-Maximization.
Each cluster k owns a polynomial-lifted operator that approximates the vector
field (dx/dt = f(x)) locally around its center c_k:

    f(x) ~ [M_k . Phi(x - c_k)]_{1:d}

where Phi(u) collects monomials up to a given total degree. This is the
**continuous-time** counterpart of ``em_local_edmd_discrete.py``, which
instead maps x_t to x_{t+1} directly.

Continuous vs discrete
~~~~~~~~~~~~~~~~~~~~~~
- **Continuous** (this module): input is (X, F) where F[i] = f(x_i) is the
  vector field evaluated at x_i. The operator M_k is a Koopman *generator*
  that acts on the time-derivative of the lifted observables.
- **Discrete** (``em_local_edmd_discrete``): input is (X, X_next) consecutive
  state pairs. The operator K_k is a discrete Koopman *operator*.

Use the continuous variant when you have access to f(x) (e.g. from an ODE
right-hand side), and the discrete variant when you only have trajectory
snapshots.

Usage
-----
Fit piecewise continuous Koopman generators::

    import torch
    from residual_aware_clustering.models.em_local_edmd import (
        fit, predict_f_all_clusters, ObservableType, make_exponents,
    )

    # State points and vector field evaluations
    X = torch.tensor(x_data, dtype=torch.float64)    # (P, d)
    F = torch.tensor(f_data, dtype=torch.float64)    # (P, d)  f(x_i) at each point

    d = X.shape[1]
    hp = {
        'alpha0':  1.0,
        'mu0':     torch.zeros(d, dtype=torch.float64),
        'Lambda0': 0.01 * torch.eye(d, dtype=torch.float64),
        'kappa0':  0.01,
        'Psi0':    torch.eye(d, dtype=torch.float64),
        'nu0':     float(d + 2),
        'sigma2':  'auto',
    }

    state, responsibilities, elbos = fit(
        X, F, N=5, hp=hp, degree=2, n_iter=100,
    )

    # Predict f(x) under all clusters
    F_pred = predict_f_all_clusters(
        X, state['centers'], state['M_ops'], state['exps'], d,
    )  # (P, N_active, d)

Observable types (monomial bases)
---------------------------------
Two monomial bases are available via ``ObservableType``:

- ``FULL``: all multivariate monomials up to total degree p, including cross
  terms (e.g. x1*x2). Count = C(d+p, p). Use for low-dimensional systems.
- ``DIAGONAL``: only univariate powers x_i^k, no cross terms.
  Count = 1 + d*p. Scales better to high-dimensional state spaces.

Select the basis via ``make_exponents``::

    exps_full = make_exponents(d=3, degree=2, observable_type=ObservableType.FULL)
    # 10 monomials: 1, x1, x2, x3, x1^2, x1*x2, x1*x3, x2^2, x2*x3, x3^2

    exps_diag = make_exponents(d=3, degree=2, observable_type=ObservableType.DIAGONAL)
    # 7 monomials: 1, x1, x2, x3, x1^2, x2^2, x3^2

Key concepts
------------
- **Koopman generator**: M_k acts on Phi-dot (the chain rule derivative of
  the lifted observables), solved via weighted continuous EDMD with ridge
  regularization.
- **Residual**: eps_k(x) = f(x) - [M_k . Phi(x - c_k)]_{1:d}. Points
  where the local generator is accurate get higher responsibility.
- **Same EM scaffold**: E-step, M-step, ELBO, dead-cluster pruning, and
  multiple restarts follow the same structure as the discrete variant
  and the Taylor-analytic baseline (``em.py``).

Also provides monomial utilities: ``monomial_exponents``, ``monomials``,
``monomials_grad``, ``diagonal_monomial_exponents``, ``make_exponents``.
"""

from enum import Enum
from itertools import combinations_with_replacement

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from .distributions import mvn_logpdf_batch
from .elbo import check_monotone


class ObservableType(Enum):
    """Type of observable basis for EDMD lifting."""
    FULL = "full"              # All multivariate monomials (cross terms included)
    DIAGONAL = "diagonal"      # Only univariate terms: x_i^k, no cross terms


# ─────────────────────────────────────────────────────────────────────────────
# Monomial lifting
# ─────────────────────────────────────────────────────────────────────────────

def monomial_exponents(d: int, degree: int) -> list:
    """
    Generate all monomial exponent tuples up to a given total degree.

    Ordering: bias (constant) first, then by degree, lexicographic within
    each degree. For d=3, degree=2 this yields 10 monomials.

    Parameters
    ----------
    d : int
        State-space dimension.
    degree : int
        Maximum total degree of monomials.

    Returns
    -------
    list of tuple
        Each tuple has length ``d`` with non-negative integer exponents.
    """
    exps = []
    for p in range(degree + 1):
        for combo in combinations_with_replacement(range(d), p):
            exp = [0] * d
            for i in combo:
                exp[i] += 1
            exps.append(tuple(exp))
    return exps


def diagonal_monomial_exponents(d: int, degree: int) -> list:
    """
    Generate univariate monomial exponents (no cross terms).

    For d=3, degree=2 this yields 7 monomials. Count: ``1 + d * degree``.

    Parameters
    ----------
    d : int
        State-space dimension.
    degree : int
        Maximum univariate degree.

    Returns
    -------
    list of tuple
        Each tuple has length ``d`` with at most one nonzero entry.
    """
    exps = [tuple([0] * d)]  # constant term
    for p in range(1, degree + 1):
        for i in range(d):
            exp = [0] * d
            exp[i] = p
            exps.append(tuple(exp))
    return exps


def make_exponents(d: int, degree: int,
                   observable_type: ObservableType = ObservableType.FULL) -> list:
    """
    Create monomial exponent tuples for the chosen observable basis.

    Parameters
    ----------
    d : int
        State-space dimension.
    degree : int
        Maximum polynomial degree.
    observable_type : ObservableType, optional
        ``FULL`` for all multivariate monomials, ``DIAGONAL`` for univariate
        only, by default ``ObservableType.FULL``.

    Returns
    -------
    list of tuple
        Monomial exponent tuples.
    """
    if observable_type == ObservableType.FULL:
        return monomial_exponents(d, degree)
    elif observable_type == ObservableType.DIAGONAL:
        return diagonal_monomial_exponents(d, degree)
    else:
        raise ValueError(f"Unknown observable type: {observable_type}")


def monomials(U: torch.Tensor, exps: list) -> torch.Tensor:
    """
    Evaluate monomial basis at given points.

    Parameters
    ----------
    U : torch.Tensor
        Shifted state points, shape ``(P, d)``.
    exps : list of tuple
        Monomial exponent tuples of length ``d``.

    Returns
    -------
    torch.Tensor
        Monomial values, shape ``(P, M)`` where ``M = len(exps)``.
    """
    P, d = U.shape
    M = len(exps)
    Phi = torch.ones(P, M, dtype=U.dtype, device=U.device)
    for j, e in enumerate(exps):
        for i in range(d):
            if e[i] > 0:
                Phi[:, j] = Phi[:, j] * (U[:, i] ** e[i])
    return Phi


def monomials_grad(U: torch.Tensor, exps: list) -> torch.Tensor:
    """
    Compute the gradient of each monomial with respect to each coordinate.

    Parameters
    ----------
    U : torch.Tensor
        Shifted state points, shape ``(P, d)``.
    exps : list of tuple
        Monomial exponent tuples of length ``d``.

    Returns
    -------
    torch.Tensor
        Gradient tensor, shape ``(P, M, d)`` where entry ``[i, j, k]`` is
        ``dPhi_j / du_k`` evaluated at ``U[i]``.
    """
    P, d = U.shape
    M = len(exps)
    grad = torch.zeros(P, M, d, dtype=U.dtype, device=U.device)
    for j, e in enumerate(exps):
        for i in range(d):
            if e[i] == 0:
                continue
            coef = float(e[i])
            term = torch.full((P,), coef, dtype=U.dtype, device=U.device)
            for k in range(d):
                ek = e[k] - 1 if k == i else e[k]
                if ek > 0:
                    term = term * (U[:, k] ** ek)
            grad[:, j, i] = term
    return grad


# ─────────────────────────────────────────────────────────────────────────────
# Local EDMD fit + prediction
# ─────────────────────────────────────────────────────────────────────────────

def weighted_continuous_edmd(
    X:      torch.Tensor,   # (P, d)
    F:      torch.Tensor,   # (P, d)
    r_k:    torch.Tensor,   # (P,)
    c_k:    torch.Tensor,   # (d,)
    exps:   list,
    ridge:  float = 1e-4,
) -> torch.Tensor:
    """
    Fit M_k minimizing  Σᵢ rᵢ‖Φ̇(xᵢ−c_k) − M_k · Φ(xᵢ−c_k)‖²
    where Φ̇(x−c_k) = (∇_u Φ)(x−c_k) · f(x).

    Closed-form weighted LS:
        M_k = (Σ r_i y_i φ_iᵀ) (Σ r_i φ_i φ_iᵀ + ridge·I)⁻¹
    """
    U       = X - c_k                                      # (P, d)
    Phi     = monomials(U, exps)                           # (P, M)
    grad    = monomials_grad(U, exps)                      # (P, M, d)
    Phi_dot = (grad @ F.unsqueeze(-1)).squeeze(-1)         # (P, M)

    M = len(exps)
    W = r_k.unsqueeze(1)                                   # (P, 1)
    G = (Phi * W).T @ Phi                                  # (M, M)
    A = (Phi_dot * W).T @ Phi                              # (M, M)
    M_k = torch.linalg.solve(
        G + ridge * torch.eye(M, dtype=X.dtype),
        A.T,
    ).T                                                    # (M, M)
    return M_k


def predict_f_all_clusters(
    X:           torch.Tensor,   # (P, d)
    centers:     torch.Tensor,   # (N, d)
    M_ops:       torch.Tensor,   # (N, M, M)
    exps:        list,
    d:           int,
) -> torch.Tensor:                # (P, N, d)
    """
    Predict f(x_i) under each cluster's local EDMD generator.
    """
    P, _ = X.shape
    N    = centers.shape[0]
    F_pred = torch.zeros(P, N, d, dtype=X.dtype)
    for k in range(N):
        U_k     = X - centers[k]
        Phi_k   = monomials(U_k, exps)                     # (P, M)
        Phi_dot = Phi_k @ M_ops[k].T                       # (P, M)
        # Entries 1..d of Phi_dot = d u_i / dt = f_i(x)
        F_pred[:, k, :] = Phi_dot[:, 1:d + 1]
    return F_pred


# ─────────────────────────────────────────────────────────────────────────────
# E-step
# ─────────────────────────────────────────────────────────────────────────────

def residual_logpdf_local_edmd(
    X:       torch.Tensor,   # (P, d)
    F:       torch.Tensor,   # (P, d)
    centers: torch.Tensor,   # (N, d)
    M_ops:   torch.Tensor,   # (N, M, M)
    sigma2,                  # float or (N,) tensor — per-cluster residual scale
    exps:    list,
    d:       int,
) -> torch.Tensor:            # (P, N)
    """
    Compute residual log-pdf under each cluster's local EDMD generator.

    Parameters
    ----------
    X : torch.Tensor
        State points, shape ``(P, d)``.
    F : torch.Tensor
        Vector field evaluations, shape ``(P, d)``.
    centers : torch.Tensor
        Cluster centers, shape ``(N, d)``.
    M_ops : torch.Tensor
        Koopman generator matrices, shape ``(N, M, M)``.
    sigma2 : float or torch.Tensor
        Per-cluster residual variance. Scalar or tensor of shape ``(N,)``.
    exps : list
        Monomial exponent tuples.
    d : int
        State-space dimension.

    Returns
    -------
    torch.Tensor
        Log-pdf values, shape ``(P, N)``.
    """
    F_pred  = predict_f_all_clusters(X, centers, M_ops, exps, d)
    eps     = F.unsqueeze(1) - F_pred                      # (P, N, d)
    sq_norm = (eps ** 2).sum(dim=2)                        # (P, N)
    N = centers.shape[0]
    if isinstance(sigma2, (int, float)):
        s2 = torch.full((N,), float(sigma2), dtype=X.dtype)
    else:
        s2 = sigma2
    log_norm = -(d / 2.0) * torch.log(2.0 * torch.pi * s2).unsqueeze(0)  # (1, N)
    return log_norm - sq_norm / (2.0 * s2.unsqueeze(0))                    # (P, N)


def e_step(
    X:     torch.Tensor,
    F:     torch.Tensor,
    state: dict,
    hp:    dict,
) -> torch.Tensor:
    """
    Compute soft assignments using proximity and local EDMD residual likelihoods.

    Parameters
    ----------
    X : torch.Tensor
        State points, shape ``(P, d)``.
    F : torch.Tensor
        Vector field evaluations, shape ``(P, d)``.
    state : dict
        Current EM state containing centers, covariances, M_ops, etc.
    hp : dict
        Hyperparameters (alpha0, mu0, Lambda0, etc.).

    Returns
    -------
    torch.Tensor
        Responsibility matrix, shape ``(P, N)``.
    """
    log_prox  = mvn_logpdf_batch(X, state['centers'], state['covariances'])
    log_resid = residual_logpdf_local_edmd(
        X, F, state['centers'], state['M_ops'], state['sigma2'],
        state['exps'], state['d'],
    )
    log_pi   = torch.log(state['pi']).unsqueeze(0)
    log_r_un = log_pi + log_prox + log_resid
    log_r    = log_r_un - torch.logsumexp(log_r_un, dim=1, keepdim=True)
    return torch.exp(log_r)


# ─────────────────────────────────────────────────────────────────────────────
# M-step: c_k (proximity-style) → M_k (weighted EDMD at new c_k) → Σ_k → π
# ─────────────────────────────────────────────────────────────────────────────

def m_step(
    X:     torch.Tensor,
    F:     torch.Tensor,
    r:     torch.Tensor,
    state: dict,
    hp:    dict,
) -> dict:
    """
    Update all parameters given soft assignments.

    Updates centers (proximity-style), local EDMD operators, covariances,
    mixing weights, and per-cluster residual variances.

    Parameters
    ----------
    X : torch.Tensor
        State points, shape ``(P, d)``.
    F : torch.Tensor
        Vector field evaluations, shape ``(P, d)``.
    r : torch.Tensor
        Responsibility matrix, shape ``(P, N)``.
    state : dict
        Current EM state.
    hp : dict
        Hyperparameters.

    Returns
    -------
    dict
        Updated EM state.
    """
    N, d, P = state['N'], state['d'], state['P']
    alpha0  = hp['alpha0']
    mu0     = hp['mu0']
    Lambda0 = hp['Lambda0']
    Psi0    = hp['Psi0']
    nu0     = hp['nu0']
    exps    = state['exps']
    Mdim    = len(exps)

    R         = r.sum(dim=0)                               # (N,)
    Sigma_inv = torch.linalg.inv(state['covariances'])    # (N, d, d)
    eye_d     = torch.eye(d, dtype=torch.float64)

    centers_new     = torch.zeros(N, d, dtype=torch.float64)
    covariances_new = torch.zeros(N, d, d, dtype=torch.float64)
    M_ops_new       = torch.zeros(N, Mdim, Mdim, dtype=torch.float64)

    for k in range(N):
        r_k = r[:, k]
        R_k = R[k].item()

        # ── c_k: proximity-style (same as em.py) — ignores residual gradient.
        hat_Lambda = Lambda0 + R_k * Sigma_inv[k]
        rhs        = Lambda0 @ mu0 + Sigma_inv[k] @ (r_k @ X)
        c_k_new    = torch.linalg.solve(hat_Lambda, rhs)
        centers_new[k] = c_k_new

        # ── M_k: weighted continuous EDMD at new c_k ─────────────────────────
        M_ops_new[k] = weighted_continuous_edmd(X, F, r_k, c_k_new, exps)

        # ── Σ_k: NIW posterior mode at new c_k ───────────────────────────────
        diff        = X - c_k_new
        scatter     = (r_k.unsqueeze(1) * diff).T @ diff
        Sigma_k_new = (Psi0 + scatter) / (nu0 + R_k + d + 1)
        covariances_new[k] = Sigma_k_new + 1e-6 * eye_d

    pi_new = (R + alpha0 - 1.0) / (P + N * (alpha0 - 1.0))
    pi_new = torch.clamp(pi_new, min=1e-10)
    pi_new = pi_new / pi_new.sum()

    # ── per-cluster sigma2_k update (only if learned) ──────────────────────
    if state.get('learn_sigma2', True):
        F_pred = predict_f_all_clusters(X, centers_new, M_ops_new, exps, d)
        sigma2_new = torch.zeros(N, dtype=torch.float64)
        for k in range(N):
            eps_k = F - F_pred[:, k]
            sq    = (eps_k ** 2).sum(dim=1)
            sigma2_new[k] = max((r[:, k] * sq).sum().item() / (d * R[k].item()), 1e-3)
    else:
        sigma2_new = state['sigma2']

    return {
        'pi':          pi_new,
        'centers':     centers_new,
        'covariances': covariances_new,
        'M_ops':       M_ops_new,
        'sigma2':      sigma2_new,
        'learn_sigma2': state.get('learn_sigma2', True),
        'N':           N, 'd': d, 'P': P,
        'exps':        exps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ELBO (uses same 6-term structure as elbo.py but with local-EDMD residual)
# ─────────────────────────────────────────────────────────────────────────────

def compute_elbo_local(
    X:     torch.Tensor, F: torch.Tensor, r: torch.Tensor,
    state: dict, hp: dict,
) -> torch.Tensor:
    """
    Compute the ELBO for the local EDMD model.

    Uses the same six-term structure as ``elbo.compute_elbo`` but substitutes
    the Taylor residual term with the local EDMD residual log-likelihood.

    Parameters
    ----------
    X : torch.Tensor
        State points, shape ``(P, d)``.
    F : torch.Tensor
        Vector field evaluations, shape ``(P, d)``.
    r : torch.Tensor
        Responsibility matrix, shape ``(P, N)``.
    state : dict
        Current EM state with M_ops, centers, covariances, etc.
    hp : dict
        Hyperparameters.

    Returns
    -------
    torch.Tensor
        Scalar ELBO value.
    """
    from .distributions import dirichlet_logpdf, niw_logpdf
    N = state['N']

    log_pi = torch.log(state['pi'])
    term1  = (r * log_pi.unsqueeze(0)).sum()

    log_prox = mvn_logpdf_batch(X, state['centers'], state['covariances'])
    term2    = (r * log_prox).sum()

    log_resid = residual_logpdf_local_edmd(
        X, F, state['centers'], state['M_ops'], state['sigma2'], state['exps'], state['d']
    )
    term3 = (r * log_resid).sum()

    term4 = -(r * torch.log(r + 1e-300)).sum()
    term5 = dirichlet_logpdf(state['pi'], hp['alpha0'])
    term6 = sum(
        niw_logpdf(state['centers'][k], state['covariances'][k],
                   hp['mu0'], hp['kappa0'], hp['Psi0'], hp['nu0'])
        for k in range(N)
    )
    return term1 + term2 + term3 + term4 + term5 + term6


# ─────────────────────────────────────────────────────────────────────────────
# Initialization (GMM warm start + fit local EDMD per initial cluster)
# ─────────────────────────────────────────────────────────────────────────────

def initialize(
    X: torch.Tensor, F: torch.Tensor,
    N: int, hp: dict, degree: int = 2, seed: int = 42,
) -> dict:
    """
    Initialize EM state with GMM warm start and per-cluster EDMD operators.

    Parameters
    ----------
    X : torch.Tensor
        State points, shape ``(P, d)``.
    F : torch.Tensor
        Vector field evaluations, shape ``(P, d)``.
    N : int
        Number of initial clusters.
    hp : dict
        Hyperparameters. If ``hp['sigma2'] == 'auto'``, per-cluster residual
        variances are calibrated from initial EDMD residuals.
    degree : int, optional
        Polynomial degree for monomial lifting, by default 2.
    seed : int, optional
        Random seed for GMM initialization, by default 42.

    Returns
    -------
    dict
        Initial EM state with keys ``'pi'``, ``'centers'``, ``'covariances'``,
        ``'M_ops'``, ``'sigma2'``, ``'exps'``, etc.
    """
    P, d = X.shape
    exps = monomial_exponents(d, degree)
    Mdim = len(exps)

    gmm = GaussianMixture(n_components=N, covariance_type='full',
                          n_init=5, random_state=seed)
    gmm.fit(X.numpy())
    labels = gmm.predict(X.numpy())

    centers     = torch.tensor(gmm.means_,       dtype=torch.float64)
    covariances = torch.tensor(gmm.covariances_, dtype=torch.float64) \
                  + 1e-6 * torch.eye(d, dtype=torch.float64).unsqueeze(0)
    pi          = torch.tensor(gmm.weights_,     dtype=torch.float64)

    # Fit initial M_k using hard cluster assignments
    M_ops = torch.zeros(N, Mdim, Mdim, dtype=torch.float64)
    for k in range(N):
        mask = torch.tensor(labels == k)
        r_k  = mask.to(torch.float64)
        if r_k.sum() < Mdim:
            # too few points — fall back to identity in lifted space (trivial model)
            M_ops[k] = torch.eye(Mdim, dtype=torch.float64) * 0.0
        else:
            M_ops[k] = weighted_continuous_edmd(X, F, r_k, centers[k], exps)

    # Calibrate per-cluster sigma2 from initial residuals
    if hp.get('sigma2', 'auto') == 'auto':
        F_pred   = predict_f_all_clusters(X, centers, M_ops, exps, d)
        sigma2   = torch.full((N,), 1e-3, dtype=torch.float64)
        for k in range(N):
            mask = labels == k
            if mask.sum() == 0:
                continue
            eps_k = F[mask] - F_pred[mask, k]
            sq    = (eps_k ** 2).sum(dim=1)
            sigma2[k] = max(sq.median().item() / d, 1e-3)
        print(f"    sigma2 calibrated per cluster: mean={sigma2.mean().item():.4f}, "
              f"range=[{sigma2.min().item():.4f}, {sigma2.max().item():.4f}]")
        learn_sigma2 = True
    else:
        s = hp['sigma2']
        if isinstance(s, (int, float)):
            sigma2 = torch.full((N,), float(s), dtype=torch.float64)
        else:
            sigma2 = s.clone()
        learn_sigma2 = False

    return {
        'pi': pi, 'centers': centers, 'covariances': covariances,
        'M_ops': M_ops, 'sigma2': sigma2, 'learn_sigma2': learn_sigma2,
        'exps': exps, 'N': N, 'd': d, 'P': P,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dead-cluster pruning
# ─────────────────────────────────────────────────────────────────────────────

def prune_dead(state, r, X, F, hp, threshold=1.0, min_N=2):
    """
    Remove clusters whose effective mass falls below a threshold.

    Parameters
    ----------
    state : dict
        Current EM state.
    r : torch.Tensor
        Responsibility matrix, shape ``(P, N)``.
    X : torch.Tensor
        State points, shape ``(P, d)``.
    F : torch.Tensor
        Vector field evaluations, shape ``(P, d)``.
    hp : dict
        Hyperparameters.
    threshold : float, optional
        Minimum effective mass to keep a cluster, by default 1.0.
    min_N : int, optional
        Never reduce below this many clusters, by default 2.

    Returns
    -------
    tuple of (dict, torch.Tensor)
        Updated state and recomputed responsibilities.
    """
    R    = r.sum(dim=0)
    dead = (R < threshold).nonzero(as_tuple=True)[0].tolist()
    if not dead:
        return state, r
    for k in sorted(dead, reverse=True):
        if state['N'] <= min_N:
            break
        print(f"  Pruning cluster {k} (R_k={R[k].item():.2f})")
        keep = [j for j in range(state['N']) if j != k]
        state = {
            'pi': state['pi'][keep] / state['pi'][keep].sum(),
            'centers':     state['centers'][keep],
            'covariances': state['covariances'][keep],
            'M_ops':       state['M_ops'][keep],
            'sigma2':      state['sigma2'][keep],
            'learn_sigma2': state.get('learn_sigma2', True),
            'exps':        state['exps'],
            'N': state['N'] - 1, 'd': state['d'], 'P': state['P'],
        }
    r = e_step(X, F, state, hp)
    return state, r


# ─────────────────────────────────────────────────────────────────────────────
# Full EM loop
# ─────────────────────────────────────────────────────────────────────────────

def fit(
    X: torch.Tensor, F: torch.Tensor,
    N: int, hp: dict, degree: int = 2,
    n_iter: int = 100, tol: float = 1e-4, n_restarts: int = 3, verbose: bool = True,
) -> tuple:
    """
    Run multi-restart EM with local continuous EDMD operators.

    Parameters
    ----------
    X : torch.Tensor
        State points, shape ``(P, d)``.
    F : torch.Tensor
        Vector field evaluations, shape ``(P, d)``.
    N : int
        Initial number of clusters.
    hp : dict
        Hyperparameters.
    degree : int, optional
        Polynomial degree for monomial lifting, by default 2.
    n_iter : int, optional
        Maximum EM iterations per restart, by default 100.
    tol : float, optional
        ELBO convergence tolerance, by default 1e-4.
    n_restarts : int, optional
        Number of random restarts, by default 3.
    verbose : bool, optional
        Print progress, by default True.

    Returns
    -------
    tuple of (dict, torch.Tensor, list)
        Best state, best responsibilities ``(P, N_active)``, and ELBO history.
    """
    best_elbo = -torch.inf
    best_state, best_r, best_history = None, None, None

    for restart in range(n_restarts):
        if verbose:
            print(f"\n  Restart {restart+1}/{n_restarts}  (local EDMD, deg={degree})")

        state = initialize(X, F, N, hp, degree=degree, seed=restart * 17)
        history = []

        for t in range(n_iter):
            r = e_step(X, F, state, hp)
            state, r = prune_dead(state, r, X, F, hp)
            if state['N'] == 0:
                break

            elbo_val = compute_elbo_local(X, F, r, state, hp).item()
            history.append(elbo_val)

            if verbose and t % 10 == 0:
                print(f"    iter {t:3d} | ELBO = {elbo_val:.2f} | N_active = {state['N']}")

            if t > 0 and abs(history[-1] - history[-2]) < tol:
                if verbose:
                    print(f"    Converged at iteration {t}")
                break

            state = m_step(X, F, r, state, hp)

        if not check_monotone(history):
            print("  [NOTE] ELBO non-monotone (likely pruning artifact — c_k update also partial)")

        r = e_step(X, F, state, hp)
        if history and history[-1] > best_elbo:
            best_elbo    = history[-1]
            best_state   = state
            best_r       = r
            best_history = history

    return best_state, best_r, best_history
