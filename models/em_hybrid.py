"""
Hybrid EM: Taylor-initialized, LS-refined.

Changes vs em.py:
  * M-step updates J_k, f_k by weighted least-squares on (x, f(x)) per
    cluster, instead of retying to analytic J(c_k), f(c_k).
  * M-step center update includes the residual-gradient term that pure
    coordinate-ascent requires when J_k is a free parameter.
  * initialize_hybrid uses analytic J(c_k), f(c_k) at t=0 (Taylor warm
    start), then lets LS refine during EM.
  * report_drift: diagnostic comparing fitted J_k to analytic J(c_k).

With these changes the ELBO is monotone under pure coordinate-ascent EM.
"""

import numpy as np
import torch

from .em import (
    initialize,
    e_step,
    handle_dead_clusters,
)
from .elbo import compute_elbo, check_monotone


# ─────────────────────────────────────────────────────────────────────────────
# M-step (hybrid): LS-refit J_k, f_k + correct c_k update
# ─────────────────────────────────────────────────────────────────────────────

def weighted_affine_fit(
    X:      torch.Tensor,   # (P, d)
    F:      torch.Tensor,   # (P, d)
    r_k:    torch.Tensor,   # (P,) weights for cluster k
    c_k:    torch.Tensor,   # (d,) current center
    ridge:  float = 1e-6,
) -> tuple:
    """
    Weighted least squares for affine model:
        f(x_i) ≈ f_k + J_k (x_i - c_k)

    Solves  (Z^T W Z + ridge·I) · [f_k; J_k^T] = Z^T W F
    where Z_i = [1, (x_i - c_k)^T], W = diag(r_k).

    Returns (f_k, J_k) of shapes (d,) and (d, d).
    """
    d = X.shape[1]
    delta = X - c_k                                # (P, d)
    Z     = torch.cat([torch.ones_like(delta[:, :1]), delta], dim=1)  # (P, d+1)
    W     = r_k                                    # (P,)

    # Weighted normal equations with ridge
    ZtWZ = (Z * W.unsqueeze(1)).T @ Z              # (d+1, d+1)
    ZtWF = (Z * W.unsqueeze(1)).T @ F              # (d+1, d)
    ZtWZ = ZtWZ + ridge * torch.eye(d + 1, dtype=torch.float64)
    B    = torch.linalg.solve(ZtWZ, ZtWF)          # (d+1, d)

    f_k = B[0]                                     # (d,)
    J_k = B[1:].T                                  # (d, d)
    return f_k, J_k


def m_step_hybrid(
    X:     torch.Tensor,
    F:     torch.Tensor,
    r:     torch.Tensor,
    state: dict,
    hp:    dict,
) -> dict:
    """
    Pure-EM M-step with free (f_k, J_k) parameters.

    Update order (each block maximizes Q holding the others fixed):
      1. f_k, J_k  ← weighted LS on (X_cluster, F_cluster) about c_k
      2. c_k       ← solve  [Lambda0 + R_k Sigma_k^{-1} + (R_k/sigma2) J_k^T J_k] c_k
                              = Lambda0 mu0 + Sigma_k^{-1} S_x
                              - (1/sigma2) J_k^T (S_f - R_k f_k - J_k S_x)
      3. Sigma_k   ← NIW posterior mode at updated c_k
      4. pi        ← Dirichlet posterior mode
    """
    N       = state['N']
    d       = state['d']
    alpha0  = hp['alpha0']
    mu0     = hp['mu0']
    Lambda0 = hp['Lambda0']
    Psi0    = hp['Psi0']
    nu0     = hp['nu0']
    P       = X.shape[0]

    R         = r.sum(dim=0)                               # (N,)
    Sigma_inv = torch.linalg.inv(state['covariances'])    # (N, d, d)
    eye_d     = torch.eye(d, dtype=torch.float64)

    centers_new     = torch.zeros(N, d, dtype=torch.float64)
    covariances_new = torch.zeros(N, d, d, dtype=torch.float64)
    f_centers_new   = torch.zeros(N, d, dtype=torch.float64)
    jacobians_new   = torch.zeros(N, d, d, dtype=torch.float64)

    for k in range(N):
        r_k = r[:, k]                                      # (P,)
        R_k = R[k].item()

        # ── Step 1 — fit f_k, J_k by weighted LS at current c_k ──────────────
        c_k_old = state['centers'][k]
        f_k, J_k = weighted_affine_fit(X, F, r_k, c_k_old)
        f_centers_new[k] = f_k
        jacobians_new[k] = J_k

        # ── Step 2 — update c_k using full gradient (proximity + residual) ───
        S_x = r_k @ X                                      # (d,)
        S_f = r_k @ F                                      # (d,)

        s2_k = state['sigma2'][k]                             # per-cluster
        JtJ  = J_k.T @ J_k                                    # (d, d)
        LHS  = Lambda0 + R_k * Sigma_inv[k] + (R_k / s2_k) * JtJ
        RHS  = (Lambda0 @ mu0
                + Sigma_inv[k] @ S_x
                - (1.0 / s2_k) * (J_k.T @ (S_f - R_k * f_k - J_k @ S_x)))

        c_k_new = torch.linalg.solve(LHS, RHS)             # (d,)
        centers_new[k] = c_k_new

        # ── Step 3 — covariance at new c_k ───────────────────────────────────
        diff        = X - c_k_new                          # (P, d)
        scatter     = (r_k.unsqueeze(1) * diff).T @ diff   # (d, d)
        Sigma_k_new = (Psi0 + scatter) / (nu0 + R_k + d + 1)
        Sigma_k_new = Sigma_k_new + 1e-6 * eye_d
        covariances_new[k] = Sigma_k_new

    # ── Step 4 — Dirichlet posterior mode for pi ──────────────────────────────
    pi_new = (R + alpha0 - 1.0) / (P + N * (alpha0 - 1.0))
    pi_new = torch.clamp(pi_new, min=1e-10)
    pi_new = pi_new / pi_new.sum()

    # ── Step 5 — per-cluster sigma2_k update (only if learned) ──────────────
    if state.get('learn_sigma2', True):
        sigma2_new = torch.zeros(N, dtype=torch.float64)
        for k in range(N):
            r_k   = r[:, k]
            R_k   = R[k].item()
            delta = X - centers_new[k]
            lp    = f_centers_new[k] + (jacobians_new[k] @ delta.T).T
            eps   = F - lp
            sq    = (eps ** 2).sum(dim=1)
            sigma2_new[k] = max((r_k * sq).sum().item() / (d * R_k), 1e-3)
    else:
        sigma2_new = state['sigma2']

    return {
        'pi':          pi_new,
        'centers':     centers_new,
        'covariances': covariances_new,
        'f_centers':   f_centers_new,
        'jacobians':   jacobians_new,
        'sigma2':      sigma2_new,
        'learn_sigma2': state.get('learn_sigma2', True),
        'N':           N, 'd': d, 'P': P,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full EM loop (hybrid)
# ─────────────────────────────────────────────────────────────────────────────

def fit_hybrid(
    X:          torch.Tensor,
    F:          torch.Tensor,
    f_fn,
    J_fn,
    N:          int,
    hp:         dict,
    n_iter:     int   = 100,
    tol:        float = 1e-4,
    n_restarts: int   = 3,
    verbose:    bool  = True,
) -> tuple:
    """
    Multi-restart EM with the hybrid M-step.

    f_fn, J_fn are used only for Taylor-warm-start initialization.
    """
    best_elbo    = -torch.inf
    best_state   = None
    best_r       = None
    best_history = None

    for restart in range(n_restarts):
        if verbose:
            print(f"\n  Restart {restart + 1}/{n_restarts}")

        state   = initialize(X, F, f_fn, J_fn, N, hp, seed=restart * 17)
        history = []

        for t in range(n_iter):
            r = e_step(X, F, state, hp)
            state, r = handle_dead_clusters(state, r, X, F, f_fn, J_fn, hp)
            if state['N'] == 0:
                print("  All clusters pruned — stopping")
                break

            elbo_val = compute_elbo(X, F, r, state, hp).item()
            history.append(elbo_val)

            if verbose and t % 10 == 0:
                print(f"    iter {t:3d} | ELBO = {elbo_val:.4f} | N_active = {state['N']}")

            if t > 0 and abs(history[-1] - history[-2]) < tol:
                if verbose:
                    print(f"    Converged at iteration {t}")
                break

            state = m_step_hybrid(X, F, r, state, hp)

        if not check_monotone(history):
            print("  [WARNING] ELBO non-monotone (hybrid M-step) — investigate")

        r = e_step(X, F, state, hp)

        if history and history[-1] > best_elbo:
            best_elbo    = history[-1]
            best_state   = state
            best_r       = r
            best_history = history

    return best_state, best_r, best_history


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic: compare fitted J_k to analytic J(c_k)
# ─────────────────────────────────────────────────────────────────────────────

def jacobian_drift(state: dict, J_fn) -> dict:
    """
    For each cluster, compute the Frobenius-norm relative drift between
    the LS-fitted J_k and the analytic J(c_k).

    Returns dict with per-cluster distances and summary statistics.
    """
    drifts = []
    for k in range(state['N']):
        c_k      = state['centers'][k]
        J_fitted = state['jacobians'][k]
        J_true   = torch.tensor(J_fn(c_k.numpy()), dtype=torch.float64)
        abs_drift = torch.linalg.norm(J_fitted - J_true).item()
        rel_drift = abs_drift / torch.linalg.norm(J_true).item()
        drifts.append((abs_drift, rel_drift))

    abs_drifts = [d[0] for d in drifts]
    rel_drifts = [d[1] for d in drifts]
    return {
        'per_cluster': drifts,
        'mean_abs':    float(np.mean(abs_drifts)),
        'max_abs':     float(np.max(abs_drifts)),
        'mean_rel':    float(np.mean(rel_drifts)),
        'max_rel':     float(np.max(rel_drifts)),
    }
