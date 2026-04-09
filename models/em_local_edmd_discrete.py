"""
Residual-aware clustering with piecewise discrete Koopman operators (CPU).

Fits N local discrete-time Koopman operators K_k via Expectation-Maximization,
where each cluster k owns a polynomial-lifted least-squares operator that maps
the current lifted state to the next lifted state:

    Phi(x_{t+1} - c_k) = K_k . Phi(x_t - c_k)

The prediction for the raw state is recovered from the linear monomial entries:

    x_{t+1} = c_k + [K_k . Phi(x_t - c_k)]_{1:d}

This is a *discrete-time* formulation -- no time derivative or Euler step is
involved. The residual that drives cluster assignments is:

    eps_k(x_t) = x_{t+1} - [K_k . Phi(x_t - c_k)]_{1:d}

Points with small residuals under cluster k receive higher responsibility,
causing the EM to partition state-space into regions where a single
polynomial Koopman operator is accurate.

Usage
-----
Fit piecewise discrete Koopman operators on consecutive state pairs::

    import torch
    from residual_aware_clustering.models.em_local_edmd_discrete import (
        fit, predict_next_all_clusters,
    )

    # Consecutive state snapshots: (P, d) tensors, float64
    X      = torch.tensor(x_data[:-1], dtype=torch.float64)   # current
    X_next = torch.tensor(x_data[1:],  dtype=torch.float64)   # next

    d = X.shape[1]

    # Hyperparameters (Bayesian priors for the EM)
    hp = {
        'alpha0':  1.0,                                  # Dirichlet concentration
        'mu0':     torch.zeros(d, dtype=torch.float64),  # NIW prior mean
        'Lambda0': 0.01 * torch.eye(d, dtype=torch.float64),
        'kappa0':  0.01,
        'Psi0':    torch.eye(d, dtype=torch.float64),
        'nu0':     float(d + 2),
        'sigma2':  'auto',   # calibrate residual variance from data
    }

    # Fit with N=5 initial clusters, degree-2 polynomial lift
    state, responsibilities, elbos = fit(
        X, X_next, N=5, hp=hp, degree=2, n_iter=100,
    )

    # state['K_ops']   — (N_active, M, M) local Koopman operators
    # state['centers'] — (N_active, d)    cluster centers
    # state['exps']    — list of monomial exponent tuples

    # Predict next states under all clusters
    X_pred = predict_next_all_clusters(
        X, state['centers'], state['K_ops'], state['exps'], d,
    )  # (P, N_active, d)

    # Use responsibilities to pick best cluster per point
    labels = responsibilities.argmax(dim=1)
    X_pred_best = X_pred[torch.arange(len(X)), labels]  # (P, d)

Key concepts
------------
- **Discrete Koopman**: operates on (x_t, x_{t+1}) pairs directly, unlike
  the continuous variant in ``em_local_edmd.py`` which fits f(x) = dx/dt.
- **Polynomial lifting**: each state is lifted via monomials up to a given
  total degree (e.g. degree=2 for quadratic). The operator K_k acts in the
  lifted space, capturing nonlinear dynamics as a linear map on observables.
- **Weighted pseudoinverse**: each cluster's K_k is solved via
  ``torch.linalg.lstsq`` with responsibility-weighted rows -- no ridge
  regularization.
- **Dead-cluster pruning**: clusters with negligible total responsibility
  (R_k < threshold) are removed mid-EM, so N shrinks automatically.
- **Multiple restarts**: the best run (highest final ELBO) is returned.

Also provides ``fit_global`` / ``predict_next_global`` for a single global
Koopman operator (N=1 baseline, same solver, no EM).
"""

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from .distributions import mvn_logpdf_batch
from .elbo import check_monotone
from .em_local_edmd import monomial_exponents, monomials


# ─────────────────────────────────────────────────────────────────────────────
# Discrete EDMD fit
# ─────────────────────────────────────────────────────────────────────────────

def weighted_discrete_edmd(
    X:      torch.Tensor,   # (P, d) current states
    X_next: torch.Tensor,   # (P, d) next states
    r_k:    torch.Tensor,   # (P,) weights for cluster k
    c_k:    torch.Tensor,   # (d,) center
    exps:   list,
) -> torch.Tensor:
    """Fit a local discrete Koopman operator via weighted least squares.

    Minimizes  sum_i r_i ||Phi(x_{i+1} - c_k) - K_k . Phi(x_i - c_k)||^2
    using ``torch.linalg.lstsq`` (no ridge regularisation).

    Parameters
    ----------
    X : torch.Tensor, shape (P, d)
        Current states.
    X_next : torch.Tensor, shape (P, d)
        Next states.
    r_k : torch.Tensor, shape (P,)
        Responsibility weights for cluster *k*.
    c_k : torch.Tensor, shape (d,)
        Cluster center.
    exps : list of tuple
        Monomial exponent tuples produced by ``monomial_exponents``.

    Returns
    -------
    torch.Tensor, shape (M, M)
        Discrete Koopman operator K_k in the lifted space.
    """
    U_curr = X - c_k                                        # (P, d)
    U_next = X_next - c_k                                   # (P, d)
    Phi_curr = monomials(U_curr, exps)                      # (P, M)
    Phi_next = monomials(U_next, exps)                      # (P, M)

    # Apply sqrt(weights) to both sides for weighted lstsq
    sqrt_w = torch.sqrt(r_k).unsqueeze(1)                   # (P, 1)
    A = Phi_curr * sqrt_w                                    # (P, M)
    B = Phi_next * sqrt_w                                    # (P, M)

    # lstsq solves A @ X = B, giving X = K^T (since Phi_next = Phi_curr @ K^T)
    result = torch.linalg.lstsq(A, B)
    K_k = result.solution.T                                  # (M, M) — transpose!
    return K_k


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_next_all_clusters(
    X:       torch.Tensor,   # (P, d) current states
    centers: torch.Tensor,   # (N, d)
    K_ops:   torch.Tensor,   # (N, M, M)
    exps:    list,
    d:       int,
) -> torch.Tensor:            # (P, N, d)
    """Predict x_{t+1} under each cluster's discrete Koopman operator.

    Applies x_{t+1} = c_k + [K_k . Phi(x_t - c_k)]_{1:d} for every cluster.

    Parameters
    ----------
    X : torch.Tensor, shape (P, d)
        Current states.
    centers : torch.Tensor, shape (N, d)
        Cluster centers.
    K_ops : torch.Tensor, shape (N, M, M)
        Per-cluster Koopman operators in the lifted space.
    exps : list of tuple
        Monomial exponent tuples.
    d : int
        Raw state dimension.

    Returns
    -------
    torch.Tensor, shape (P, N, d)
        Predicted next states under each cluster.
    """
    P = X.shape[0]
    N = centers.shape[0]
    X_pred = torch.zeros(P, N, d, dtype=X.dtype)
    for k in range(N):
        U_k = X - centers[k]
        Phi_k = monomials(U_k, exps)                        # (P, M)
        Phi_next = Phi_k @ K_ops[k].T                       # (P, M)
        # Entries 1:d+1 of Phi are the linear monomials u_1,...,u_d
        # After K maps them, entries 1:d+1 give the next-state offset from c_k
        X_pred[:, k, :] = centers[k] + Phi_next[:, 1:d+1]
    return X_pred


# ─────────────────────────────────────────────────────────────────────────────
# Residual log-pdf
# ─────────────────────────────────────────────────────────────────────────────

def residual_logpdf_discrete(
    X:       torch.Tensor,   # (P, d) current states
    X_next:  torch.Tensor,   # (P, d) next states (target)
    centers: torch.Tensor,   # (N, d)
    K_ops:   torch.Tensor,   # (N, M, M)
    sigma2,                  # float or (N,) tensor
    exps:    list,
    d:       int,
) -> torch.Tensor:            # (P, N)
    """Compute log-pdf of the prediction residual under an isotropic Gaussian.

    For each cluster k the residual is eps_k = x_{t+1} - pred_k(x_t), and
    the log-pdf is evaluated as N(eps_k; 0, sigma2_k I).

    Parameters
    ----------
    X : torch.Tensor, shape (P, d)
        Current states.
    X_next : torch.Tensor, shape (P, d)
        Next states (targets).
    centers : torch.Tensor, shape (N, d)
        Cluster centers.
    K_ops : torch.Tensor, shape (N, M, M)
        Per-cluster Koopman operators.
    sigma2 : float or torch.Tensor, shape (N,)
        Isotropic residual variance per cluster.
    exps : list of tuple
        Monomial exponent tuples.
    d : int
        Raw state dimension.

    Returns
    -------
    torch.Tensor, shape (P, N)
        Log-pdf of the residual for each point under each cluster.
    """
    X_pred = predict_next_all_clusters(X, centers, K_ops, exps, d)
    eps    = X_next.unsqueeze(1) - X_pred                    # (P, N, d)
    sq_norm = (eps ** 2).sum(dim=2)                          # (P, N)
    N = centers.shape[0]
    if isinstance(sigma2, (int, float)):
        s2 = torch.full((N,), float(sigma2), dtype=X.dtype)
    else:
        s2 = sigma2
    log_norm = -(d / 2.0) * torch.log(2.0 * torch.pi * s2).unsqueeze(0)
    return log_norm - sq_norm / (2.0 * s2.unsqueeze(0))


# ─────────────────────────────────────────────────────────────────────────────
# E-step
# ─────────────────────────────────────────────────────────────────────────────

def e_step(
    X:      torch.Tensor,    # (P, d) current states
    X_next: torch.Tensor,    # (P, d) next states
    state:  dict,
    hp:     dict,
) -> torch.Tensor:
    """Compute cluster responsibilities (E-step).

    Combines mixing weights, proximity log-pdf, and residual log-pdf in
    log-space, then normalises via log-sum-exp.

    Parameters
    ----------
    X : torch.Tensor, shape (P, d)
        Current states.
    X_next : torch.Tensor, shape (P, d)
        Next states.
    state : dict
        Current EM state containing ``pi``, ``centers``, ``covariances``,
        ``K_ops``, ``sigma2``, ``exps``, and ``d``.
    hp : dict
        Hyperparameters (unused directly but kept for API consistency).

    Returns
    -------
    torch.Tensor, shape (P, N)
        Normalised cluster responsibilities.
    """
    log_prox  = mvn_logpdf_batch(X, state['centers'], state['covariances'])
    log_resid = residual_logpdf_discrete(
        X, X_next, state['centers'], state['K_ops'], state['sigma2'],
        state['exps'], state['d'],
    )
    log_pi   = torch.log(state['pi']).unsqueeze(0)
    log_r_un = log_pi + log_prox + log_resid
    log_r    = log_r_un - torch.logsumexp(log_r_un, dim=1, keepdim=True)
    return torch.exp(log_r)


# ─────────────────────────────────────────────────────────────────────────────
# M-step
# ─────────────────────────────────────────────────────────────────────────────

def m_step(
    X:      torch.Tensor,
    X_next: torch.Tensor,
    r:      torch.Tensor,
    state:  dict,
    hp:     dict,
) -> dict:
    """Update all cluster parameters given responsibilities (M-step).

    Updates centers (NIW posterior), covariances (NIW posterior), Koopman
    operators (weighted discrete EDMD), mixing weights (Dirichlet MAP), and
    per-cluster residual variances.

    Parameters
    ----------
    X : torch.Tensor, shape (P, d)
        Current states.
    X_next : torch.Tensor, shape (P, d)
        Next states.
    r : torch.Tensor, shape (P, N)
        Cluster responsibilities from the E-step.
    state : dict
        Current EM state.
    hp : dict
        Hyperparameters (``alpha0``, ``mu0``, ``Lambda0``, ``Psi0``, ``nu0``).

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

    R         = r.sum(dim=0)
    Sigma_inv = torch.linalg.inv(state['covariances'])
    eye_d     = torch.eye(d, dtype=torch.float64)

    centers_new     = torch.zeros(N, d, dtype=torch.float64)
    covariances_new = torch.zeros(N, d, d, dtype=torch.float64)
    K_ops_new       = torch.zeros(N, Mdim, Mdim, dtype=torch.float64)

    for k in range(N):
        r_k = r[:, k]
        R_k = R[k].item()

        # c_k: proximity-only update
        hat_Lambda = Lambda0 + R_k * Sigma_inv[k] + 1e-6 * eye_d
        rhs        = Lambda0 @ mu0 + Sigma_inv[k] @ (r_k @ X)
        c_k_new    = torch.linalg.solve(hat_Lambda, rhs)
        centers_new[k] = c_k_new

        # K_k: weighted discrete EDMD at new c_k
        K_ops_new[k] = weighted_discrete_edmd(X, X_next, r_k, c_k_new, exps)

        # Sigma_k: NIW posterior mode
        diff    = X - c_k_new
        scatter = (r_k.unsqueeze(1) * diff).T @ diff
        covariances_new[k] = (Psi0 + scatter) / (nu0 + R_k + d + 1) + 1e-6 * eye_d

    pi_new = (R + alpha0 - 1.0) / (P + N * (alpha0 - 1.0))
    pi_new = torch.clamp(pi_new, min=1e-10)
    pi_new = pi_new / pi_new.sum()

    # Per-cluster sigma2 update
    if state.get('learn_sigma2', True):
        X_pred = predict_next_all_clusters(X, centers_new, K_ops_new, exps, d)
        sigma2_new = torch.zeros(N, dtype=torch.float64)
        for k in range(N):
            eps_k = X_next - X_pred[:, k]
            sq    = (eps_k ** 2).sum(dim=1)
            sigma2_new[k] = max((r[:, k] * sq).sum().item() / (d * R[k].item()), 1e-6)
    else:
        sigma2_new = state['sigma2']

    return {
        'pi':          pi_new,
        'centers':     centers_new,
        'covariances': covariances_new,
        'K_ops':       K_ops_new,
        'sigma2':      sigma2_new,
        'learn_sigma2': state.get('learn_sigma2', True),
        'N':           N, 'd': d, 'P': P,
        'exps':        exps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ELBO
# ─────────────────────────────────────────────────────────────────────────────

def compute_elbo(
    X: torch.Tensor, X_next: torch.Tensor, r: torch.Tensor,
    state: dict, hp: dict,
) -> torch.Tensor:
    """Compute the evidence lower bound (ELBO) for the current EM state.

    Sums the expected log-likelihood terms (mixing weights, proximity,
    residual), the entropy of responsibilities, and the Dirichlet/NIW priors.

    Parameters
    ----------
    X : torch.Tensor, shape (P, d)
        Current states.
    X_next : torch.Tensor, shape (P, d)
        Next states.
    r : torch.Tensor, shape (P, N)
        Cluster responsibilities.
    state : dict
        Current EM state.
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

    log_resid = residual_logpdf_discrete(
        X, X_next, state['centers'], state['K_ops'], state['sigma2'],
        state['exps'], state['d']
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
# Dead-cluster pruning
# ─────────────────────────────────────────────────────────────────────────────

def prune_dead(state, r, X, X_next, hp, threshold=1.0, min_N=2):
    """Remove clusters whose total responsibility falls below a threshold.

    After pruning, responsibilities are recomputed via a fresh E-step.

    Parameters
    ----------
    state : dict
        Current EM state.
    r : torch.Tensor, shape (P, N)
        Cluster responsibilities.
    X : torch.Tensor, shape (P, d)
        Current states.
    X_next : torch.Tensor, shape (P, d)
        Next states.
    hp : dict
        Hyperparameters.
    threshold : float, optional
        Minimum total responsibility to keep a cluster (default 1.0).
    min_N : int, optional
        Minimum number of clusters to retain (default 2).

    Returns
    -------
    state : dict
        Updated EM state with dead clusters removed.
    r : torch.Tensor, shape (P, N_active)
        Recomputed responsibilities.
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
            'K_ops':       state['K_ops'][keep],
            'sigma2':      state['sigma2'][keep],
            'learn_sigma2': state.get('learn_sigma2', True),
            'exps':        state['exps'],
            'N': state['N'] - 1, 'd': state['d'], 'P': state['P'],
        }
    r = e_step(X, X_next, state, hp)
    return state, r


# ─────────────────────────────────────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────────────────────────────────────

def initialize(
    X: torch.Tensor, X_next: torch.Tensor,
    N: int, hp: dict, degree: int = 2, seed: int = 42,
) -> dict:
    """Create the initial EM state using a GMM fit on the current states.

    Fits a ``GaussianMixture`` with *N* components, then builds initial
    Koopman operators per cluster via hard-assignment weighted EDMD.
    Optionally calibrates per-cluster residual variance from data.

    Parameters
    ----------
    X : torch.Tensor, shape (P, d)
        Current states.
    X_next : torch.Tensor, shape (P, d)
        Next states.
    N : int
        Number of initial clusters.
    hp : dict
        Hyperparameters. If ``hp['sigma2'] == 'auto'``, sigma2 is calibrated
        from median squared residuals.
    degree : int, optional
        Total polynomial degree for the monomial lift (default 2).
    seed : int, optional
        Random seed for the GMM initialisation (default 42).

    Returns
    -------
    dict
        Initial EM state with keys ``pi``, ``centers``, ``covariances``,
        ``K_ops``, ``sigma2``, ``learn_sigma2``, ``exps``, ``N``, ``d``, ``P``.
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

    K_ops = torch.zeros(N, Mdim, Mdim, dtype=torch.float64)
    for k in range(N):
        mask = torch.tensor(labels == k)
        r_k  = mask.to(torch.float64)
        if r_k.sum() < Mdim:
            K_ops[k] = torch.eye(Mdim, dtype=torch.float64)
        else:
            K_ops[k] = weighted_discrete_edmd(X, X_next, r_k, centers[k], exps)

    # Calibrate sigma2
    if hp.get('sigma2', 'auto') == 'auto':
        X_pred = predict_next_all_clusters(X, centers, K_ops, exps, d)
        sigma2 = torch.full((N,), 1e-6, dtype=torch.float64)
        for k in range(N):
            mask = labels == k
            if mask.sum() == 0:
                continue
            eps_k = X_next[mask] - X_pred[mask, k]
            sq = (eps_k ** 2).sum(dim=1)
            sigma2[k] = max(sq.median().item() / d, 1e-6)
        print(f"    sigma2 calibrated per cluster: mean={sigma2.mean().item():.6f}, "
              f"range=[{sigma2.min().item():.6f}, {sigma2.max().item():.6f}]")
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
        'K_ops': K_ops, 'sigma2': sigma2, 'learn_sigma2': learn_sigma2,
        'exps': exps, 'N': N, 'd': d, 'P': P,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full EM loop
# ─────────────────────────────────────────────────────────────────────────────

def fit(
    X: torch.Tensor, X_next: torch.Tensor,
    N: int, hp: dict, degree: int = 2,
    n_iter: int = 100, tol: float = 1e-4, n_restarts: int = 3, verbose: bool = True,
) -> tuple:
    """Fit piecewise discrete Koopman operators via EM with multiple restarts.

    Parameters
    ----------
    X : torch.Tensor, shape (P, d)
        Current states.
    X_next : torch.Tensor, shape (P, d)
        Next states.
    N : int
        Initial number of clusters (may shrink due to pruning).
    hp : dict
        Hyperparameters (``alpha0``, ``mu0``, ``Lambda0``, ``kappa0``,
        ``Psi0``, ``nu0``, ``sigma2``).
    degree : int, optional
        Total polynomial degree for the monomial lift (default 2).
    n_iter : int, optional
        Maximum EM iterations per restart (default 100).
    tol : float, optional
        ELBO convergence tolerance (default 1e-4).
    n_restarts : int, optional
        Number of random restarts; best ELBO wins (default 3).
    verbose : bool, optional
        Print progress every 10 iterations (default True).

    Returns
    -------
    best_state : dict
        EM state from the best restart.
    best_r : torch.Tensor, shape (P, N_active)
        Final responsibilities from the best restart.
    best_history : list of float
        ELBO history from the best restart.
    """
    best_elbo = -torch.inf
    best_state, best_r, best_history = None, None, None

    for restart in range(n_restarts):
        if verbose:
            print(f"\n  Restart {restart+1}/{n_restarts}  (discrete local EDMD, deg={degree})")

        state = initialize(X, X_next, N, hp, degree=degree, seed=restart * 17)
        history = []

        for t in range(n_iter):
            r = e_step(X, X_next, state, hp)
            state, r = prune_dead(state, r, X, X_next, hp)
            if state['N'] == 0:
                break

            elbo_val = compute_elbo(X, X_next, r, state, hp).item()
            history.append(elbo_val)

            if verbose and t % 10 == 0:
                print(f"    iter {t:3d} | ELBO = {elbo_val:.2f} | N_active = {state['N']}")

            if t > 0 and abs(history[-1] - history[-2]) < tol:
                if verbose:
                    print(f"    Converged at iteration {t}")
                break

            state = m_step(X, X_next, r, state, hp)

        if not check_monotone(history):
            print("  [NOTE] ELBO non-monotone")

        r = e_step(X, X_next, state, hp)
        if history and history[-1] > best_elbo:
            best_elbo    = history[-1]
            best_state   = state
            best_r       = r
            best_history = history

    return best_state, best_r, best_history


# ─────────────────────────────────────────────────────────────────────────────
# Global discrete EDMD (N=1, for fair comparison)
# ─────────────────────────────────────────────────────────────────────────────

def fit_global(X, X_next, degree=2):
    """Fit a single global discrete Koopman operator (N=1 baseline).

    Uses the same weighted least-squares solver as the local version with
    uniform weights, centred at the data mean.

    Parameters
    ----------
    X : torch.Tensor, shape (P, d)
        Current states.
    X_next : torch.Tensor, shape (P, d)
        Next states.
    degree : int, optional
        Total polynomial degree for the monomial lift (default 2).

    Returns
    -------
    dict
        Model with keys ``K`` (M, M), ``c`` (d,), ``exps``, ``d``.
    """
    d = X.shape[1]
    exps = monomial_exponents(d, degree)
    c = X.mean(dim=0)
    r = torch.ones(X.shape[0], dtype=X.dtype)
    K = weighted_discrete_edmd(X, X_next, r, c, exps)
    return {'K': K, 'c': c, 'exps': exps, 'd': d}


def predict_next_global(X, model):
    """Predict x_{t+1} using the global discrete Koopman operator.

    Parameters
    ----------
    X : torch.Tensor, shape (P, d)
        Current states.
    model : dict
        Model returned by ``fit_global``.

    Returns
    -------
    torch.Tensor, shape (P, d)
        Predicted next states.
    """
    d = model['d']
    U = X - model['c']
    Phi = monomials(U, model['exps'])
    Phi_next = Phi @ model['K'].T
    return model['c'] + Phi_next[:, 1:d+1]
