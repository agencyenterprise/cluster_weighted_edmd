"""
Marginal likelihood and BIC for model selection.

Provides closed-form log marginal likelihood by integrating out
(c_k, Sigma_k) under the Normal-Inverse-Wishart (NIW) conjugate prior,
plus a residual penalty term that accounts for linearization error.
Used to compare different cluster counts N and select the best model.

Functions
---------
- ``cluster_log_marginal(X_k, F_k, c_k, J_k, f_ck, hp)`` -- exact
  log marginal for a single cluster (five terms, see docstring).
- ``total_log_marginal(X, F, r, state, hp)`` -- full log p(X, F | N)
  summing the Dirichlet-Categorical marginal and all per-cluster terms.
- ``bic(log_ml, N, P, d)`` -- Bayesian Information Criterion from the
  log marginal likelihood.  Higher is better.

Usage
-----
::

    from residual_aware_clustering.models.marginal_likelihood import (
        total_log_marginal, bic,
    )

    log_ml = total_log_marginal(X, F, r, state, hp)

    # Compare cluster counts
    scores = {}
    for N in [3, 5, 8, 10]:
        state_N, r_N, _ = fit_taylor(X, F, f, J, N=N, hp=hp)
        ml_N = total_log_marginal(X, F, r_N, state_N, hp).item()
        scores[N] = bic(ml_N, N, P=X.shape[0], d=X.shape[1])

    best_N = max(scores, key=scores.get)

Key concepts
------------
- **NIW conjugacy**: the proximity integral has a closed-form solution
  because the Gaussian likelihood is conjugate to the NIW prior.
- **Residual penalty** (term 5, novel): a Gaussian penalty on
  linearization residuals eps_k(x_i) = F_i - [f(c_k) + J_k (x_i - c_k)]
  that prevents overly large clusters where the linear approximation
  breaks down.
- **BIC**: penalizes model complexity with D_N free parameters,
  where D_N = N * (d*(d+3)/2 + 1) - 1.
"""

import torch
from torch.special import gammaln
from .distributions import _mvlgamma


def cluster_log_marginal(
    X_k:  torch.Tensor,   # (R_k, d)
    F_k:  torch.Tensor,   # (R_k, d)
    c_k:  torch.Tensor,   # (d,)  current center
    J_k:  torch.Tensor,   # (d, d)
    f_ck: torch.Tensor,   # (d,)  f(c_k)
    hp:   dict,
) -> torch.Tensor:         # scalar
    """
    Exact log marginal likelihood for cluster k, integrating out
    (c_k, Sigma_k) under NIW prior via conjugacy.

    Five terms derived in Rung 4:
      1. Center uncertainty:    (d/2) log(kappa0 / kappa_n)
      2. Covariance uncertainty: (nu0/2) log|Psi0| - (nu_n/2) log|Psi_n|
      3. Degrees of freedom:    log Gamma_d(nu_n/2) - log Gamma_d(nu0/2)
      4. Volume normalization:  -(R_k * d / 2) log(pi)
      5. Residual penalty:      -(1/2sigma2) sum ||eps_k(x_i)||^2   [novel]
    """
    R_k    = X_k.shape[0]
    d      = X_k.shape[1]
    mu0    = hp['mu0']
    kappa0 = hp['kappa0']
    Psi0   = hp['Psi0']
    nu0    = hp['nu0']
    sigma2 = hp.get('_sigma2_k', hp.get('sigma2', 1.0))  # per-cluster or scalar

    if R_k == 0:
        return torch.tensor(0.0, dtype=torch.float64)

    # ── NIW posterior parameters ──────────────────────────────────────────────

    kappa_n = kappa0 + R_k
    nu_n    = nu0 + R_k
    x_bar   = X_k.mean(dim=0)                            # (d,)
    mu_n    = (kappa0 * mu0 + R_k * x_bar) / kappa_n    # (d,)

    diff    = X_k - x_bar                                # (R_k, d)
    S_k     = diff.T @ diff                              # (d, d)

    conflict = (kappa0 * R_k / kappa_n) * torch.outer(
        x_bar - mu0, x_bar - mu0
    )
    Psi_n   = Psi0 + S_k + conflict                     # (d, d)

    # ── Five terms ────────────────────────────────────────────────────────────

    # Term 1 — center uncertainty
    term1 = (d / 2.0) * torch.log(
        torch.tensor(kappa0 / kappa_n, dtype=torch.float64)
    )

    # Term 2 — covariance uncertainty
    _, log_det_Psi0 = torch.linalg.slogdet(Psi0)
    _, log_det_Psi_n = torch.linalg.slogdet(Psi_n)
    term2 = (nu0 / 2.0) * log_det_Psi0 - (nu_n / 2.0) * log_det_Psi_n

    # Term 3 — degrees of freedom
    term3 = _mvlgamma(nu_n / 2.0, d) - _mvlgamma(nu0 / 2.0, d)

    # Term 4 — volume normalization
    term4 = -(R_k * d / 2.0) * torch.log(
        torch.tensor(torch.pi, dtype=torch.float64)
    )

    # Term 5 — residual penalty (novel term)
    delta       = X_k - c_k                              # (R_k, d)
    linear_pred = f_ck + (J_k @ delta.T).T              # (R_k, d)
    eps         = F_k - linear_pred                      # (R_k, d)
    term5       = -0.5 / sigma2 * (eps ** 2).sum()

    return term1 + term2 + term3 + term4 + term5


def total_log_marginal(
    X:     torch.Tensor,   # (P, d)
    F:     torch.Tensor,   # (P, d)
    r:     torch.Tensor,   # (P, N)
    state: dict,
    hp:    dict,
) -> torch.Tensor:          # scalar
    """
    Full log marginal likelihood p(X, F | N).

    log p(X, F | N) = log p(Z marginal from Dirichlet-Categorical)
                    + sum_k log p(X_k, F_k | N, Z)
    """
    N      = state['N']
    P      = X.shape[0]
    alpha0 = hp['alpha0']

    assignments = r.argmax(dim=1)   # (P,) hard assignments

    # Effective cluster counts
    R = torch.zeros(N, dtype=torch.float64)
    for k in range(N):
        R[k] = float((assignments == k).sum())

    # ── Dirichlet-Categorical marginal ────────────────────────────────────────
    # log B(R + alpha0) / B(alpha0 * 1_N)
    log_dir = (
          gammaln(torch.tensor(N * alpha0, dtype=torch.float64))
        - N * gammaln(torch.tensor(alpha0, dtype=torch.float64))
        + gammaln(R + alpha0).sum()
        - gammaln(torch.tensor(P + N * alpha0, dtype=torch.float64))
    )

    # ── Per-cluster NIW marginals ─────────────────────────────────────────────
    log_clusters = torch.tensor(0.0, dtype=torch.float64)

    for k in range(N):
        mask = assignments == k
        if mask.sum() == 0:
            continue
        # Pass per-cluster sigma2_k via hp override
        hp_k = dict(hp)
        hp_k['_sigma2_k'] = state['sigma2'][k].item() if 'sigma2' in state else hp.get('sigma2', 1.0)
        log_clusters = log_clusters + cluster_log_marginal(
            X_k=X[mask],
            F_k=F[mask],
            c_k=state['centers'][k],
            J_k=state['jacobians'][k],
            f_ck=state['f_centers'][k],
            hp=hp_k,
        )

    return log_dir + log_clusters


def bic(
    log_ml: float,
    N:      int,
    P:      int,
    d:      int,
) -> float:
    """
    BIC score given log marginal likelihood.
    D_N = N * (d*(d+3)/2 + 1) - 1  free parameters.
    Higher BIC is better.
    """
    import math
    D_N = N * (d * (d + 3) // 2 + 1) - 1
    return log_ml - (D_N / 2.0) * math.log(P)
