"""
GPU-compatible local discrete-time EDMD per cluster.

Drop-in replacement for em_local_edmd_discrete.py that works on any device
(CPU, CUDA, MPS). All tensor creation calls propagate device from the input
data. sklearn GaussianMixture (CPU-only) is called with explicit .cpu().numpy()
and results are moved back to the target device.
"""

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from .distributions_gpu import mvn_logpdf_batch, dirichlet_logpdf, niw_logpdf
from .em_local_edmd import monomial_exponents, monomials


# ─────────────────────────────────────────────────────────────────────────────
# Discrete EDMD fit
# ─────────────────────────────────────────────────────────────────────────────

def _lstsq_svd(A, B, rcond=1e-10):
    """SVD-based least squares: solves A @ X = B via truncated pseudoinverse.

    Equivalent to gelsd: uses SVD, thresholds small singular values.
    Works reliably on CPU, CUDA, and MPS.
    """
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    # Threshold: ignore singular values below rcond * max(S)
    cutoff = rcond * S[0]
    S_inv = torch.where(S > cutoff, 1.0 / S, torch.zeros_like(S))
    # pseudoinverse solution: V @ diag(1/s) @ U^T @ B
    return Vh.T @ (S_inv.unsqueeze(1) * (U.T @ B))


def weighted_discrete_edmd(X, X_next, r_k, c_k, exps):
    U_curr = X - c_k
    U_next = X_next - c_k
    Phi_curr = monomials(U_curr, exps)
    Phi_next = monomials(U_next, exps)

    sqrt_w = torch.sqrt(r_k).unsqueeze(1)
    A = Phi_curr * sqrt_w
    B = Phi_next * sqrt_w

    return _lstsq_svd(A, B).T


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_next_all_clusters(X, centers, K_ops, exps, d):
    P = X.shape[0]
    N = centers.shape[0]
    X_pred = torch.zeros(P, N, d, dtype=X.dtype, device=X.device)
    for k in range(N):
        U_k = X - centers[k]
        Phi_k = monomials(U_k, exps)
        Phi_next = Phi_k @ K_ops[k].T
        X_pred[:, k, :] = centers[k] + Phi_next[:, 1:d+1]
    return X_pred


# ─────────────────────────────────────────────────────────────────────────────
# Residual log-pdf
# ─────────────────────────────────────────────────────────────────────────────

def residual_logpdf_discrete(X, X_next, centers, K_ops, sigma2, exps, d):
    X_pred = predict_next_all_clusters(X, centers, K_ops, exps, d)
    eps    = X_next.unsqueeze(1) - X_pred
    sq_norm = (eps ** 2).sum(dim=2)
    N = centers.shape[0]
    if isinstance(sigma2, (int, float)):
        s2 = torch.full((N,), float(sigma2), dtype=X.dtype, device=X.device)
    else:
        s2 = sigma2
    log_norm = -(d / 2.0) * torch.log(2.0 * torch.pi * s2).unsqueeze(0)
    return log_norm - sq_norm / (2.0 * s2.unsqueeze(0))


# ─────────────────────────────────────────────────────────────────────────────
# E-step
# ─────────────────────────────────────────────────────────────────────────────

def e_step(X, X_next, state, hp):
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

def m_step(X, X_next, r, state, hp):
    N, d, P = state['N'], state['d'], state['P']
    dev = X.device
    alpha0  = hp['alpha0']
    mu0     = hp['mu0']
    Lambda0 = hp['Lambda0']
    Psi0    = hp['Psi0']
    nu0     = hp['nu0']
    exps    = state['exps']
    Mdim    = len(exps)

    R         = r.sum(dim=0)
    Sigma_inv = torch.linalg.inv(state['covariances'])
    dt = X.dtype
    eye_d     = torch.eye(d, dtype=dt, device=dev)

    centers_new     = torch.zeros(N, d, dtype=dt, device=dev)
    covariances_new = torch.zeros(N, d, d, dtype=dt, device=dev)
    K_ops_new       = torch.zeros(N, Mdim, Mdim, dtype=dt, device=dev)

    for k in range(N):
        r_k = r[:, k]
        R_k = R[k].item()

        hat_Lambda = Lambda0 + R_k * Sigma_inv[k] + 1e-6 * eye_d
        rhs        = Lambda0 @ mu0 + Sigma_inv[k] @ (r_k @ X)
        c_k_new    = torch.linalg.solve(hat_Lambda, rhs)
        centers_new[k] = c_k_new

        K_ops_new[k] = weighted_discrete_edmd(X, X_next, r_k, c_k_new, exps)

        diff    = X - c_k_new
        scatter = (r_k.unsqueeze(1) * diff).T @ diff
        covariances_new[k] = (Psi0 + scatter) / (nu0 + R_k + d + 1) + 1e-6 * eye_d

    pi_new = (R + alpha0 - 1.0) / (P + N * (alpha0 - 1.0))
    pi_new = torch.clamp(pi_new, min=1e-10)
    pi_new = pi_new / pi_new.sum()

    if state.get('learn_sigma2', True):
        X_pred = predict_next_all_clusters(X, centers_new, K_ops_new, exps, d)
        sigma2_new = torch.zeros(N, dtype=dt, device=dev)
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

def compute_elbo(X, X_next, r, state, hp):
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
# Initialization (sklearn on CPU, results moved to target device)
# ─────────────────────────────────────────────────────────────────────────────

def initialize(X, X_next, N, hp, degree=2, seed=42):
    P, d = X.shape
    dev = X.device
    dt = X.dtype
    exps = monomial_exponents(d, degree)
    Mdim = len(exps)

    # GMM is CPU-only — move data there and back
    # .float() for MPS (which doesn't support float64 numpy conversion)
    gmm = GaussianMixture(n_components=N, covariance_type='full',
                          n_init=5, random_state=seed)
    X_np = X.detach().cpu().to(torch.float64).numpy()
    gmm.fit(X_np)
    labels = gmm.predict(X_np)

    # sklearn may find fewer clusters than requested (duplicate points in low-d)
    n_found = len(set(labels))
    if n_found < N:
        print(f"    GMM found {n_found} clusters (requested {N}), reducing N")
        N = n_found
        # Re-index labels to be contiguous 0..N-1
        unique_labels = sorted(set(labels))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels])
        # Keep only the active GMM components
        active = [unique_labels[i] for i in range(N)]
        gmm.means_ = gmm.means_[active]
        gmm.covariances_ = gmm.covariances_[active]
        gmm.weights_ = gmm.weights_[active]
        gmm.weights_ /= gmm.weights_.sum()

    centers     = torch.tensor(gmm.means_,       dtype=dt, device=dev)
    covariances = torch.tensor(gmm.covariances_, dtype=dt, device=dev) \
                  + 1e-6 * torch.eye(d, dtype=dt, device=dev).unsqueeze(0)
    pi          = torch.tensor(gmm.weights_,     dtype=dt, device=dev)

    Mdim = len(exps)
    K_ops = torch.zeros(N, Mdim, Mdim, dtype=dt, device=dev)
    for k in range(N):
        mask = torch.tensor(labels == k, device=dev)
        r_k  = mask.to(dt)
        if r_k.sum() < Mdim:
            K_ops[k] = torch.eye(Mdim, dtype=dt, device=dev)
        else:
            K_ops[k] = weighted_discrete_edmd(X, X_next, r_k, centers[k], exps)

    # Calibrate sigma2
    if hp.get('sigma2', 'auto') == 'auto':
        X_pred = predict_next_all_clusters(X, centers, K_ops, exps, d)
        sigma2 = torch.full((N,), 1e-6, dtype=dt, device=dev)
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
            sigma2 = torch.full((N,), float(s), dtype=dt, device=dev)
        else:
            sigma2 = s.clone().to(dev)
        learn_sigma2 = False

    return {
        'pi': pi, 'centers': centers, 'covariances': covariances,
        'K_ops': K_ops, 'sigma2': sigma2, 'learn_sigma2': learn_sigma2,
        'exps': exps, 'N': N, 'd': d, 'P': P,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full EM loop
# ─────────────────────────────────────────────────────────────────────────────

def check_monotone(history, tol=1.0):
    if len(history) < 2:
        return True
    violations = [
        (t, history[t] - history[t-1])
        for t in range(1, len(history))
        if history[t] - history[t-1] < -tol
    ]
    if violations:
        print(f"  [WARNING] ELBO decreased at {len(violations)} steps — "
              f"largest: {min(d for _, d in violations):.4f} nats")
        return False
    return True


def _to_device_dtype(state, r, device, dtype):
    """Move state dict and responsibilities to target device/dtype."""
    out = dict(state)
    for key in ['pi', 'centers', 'covariances', 'K_ops', 'sigma2']:
        if isinstance(out[key], torch.Tensor):
            out[key] = out[key].to(device=device, dtype=dtype)
    return out, r.to(device=device, dtype=dtype)


def fit(X, X_next, N, hp, degree=2,
        n_iter=100, tol=1e-4, n_restarts=3, verbose=True):
    """
    Fit discrete local EDMD via EM. GPU-compatible version.

    Works on CPU, CUDA, and MPS. For MPS (float32-only), the EM runs
    internally at float64 on CPU to avoid numerical issues in
    log-probability computations, then casts results back.
    """
    orig_device = X.device
    orig_dtype = X.dtype

    # EM always runs at float64. MPS can't do float64 so we fall back to CPU.
    # CUDA stays on CUDA. CPU stays on CPU.
    if orig_device.type == "mps":
        em_device = torch.device("cpu")
    else:
        em_device = orig_device

    if orig_dtype != torch.float64 or em_device != orig_device:
        X_em = X.cpu().to(dtype=torch.float64, device=em_device)
        X_next_em = X_next.cpu().to(dtype=torch.float64, device=em_device)
        hp_em = {}
        for k, v in hp.items():
            if isinstance(v, torch.Tensor):
                hp_em[k] = v.cpu().to(dtype=torch.float64, device=em_device)
            else:
                hp_em[k] = v
    else:
        X_em, X_next_em, hp_em = X, X_next, hp

    best_elbo = -torch.inf
    best_state, best_r, best_history = None, None, None

    for restart in range(n_restarts):
        if verbose:
            print(f"\n  Restart {restart+1}/{n_restarts}  (discrete local EDMD, deg={degree})")

        state = initialize(X_em, X_next_em, N, hp_em, degree=degree, seed=restart * 17)
        history = []

        for t in range(n_iter):
            r = e_step(X_em, X_next_em, state, hp_em)
            state, r = prune_dead(state, r, X_em, X_next_em, hp_em)
            if state['N'] == 0:
                break

            elbo_val = compute_elbo(X_em, X_next_em, r, state, hp_em).item()
            history.append(elbo_val)

            if verbose and t % 10 == 0:
                print(f"    iter {t:3d} | ELBO = {elbo_val:.2f} | N_active = {state['N']}")

            if t > 0 and abs(history[-1] - history[-2]) < tol:
                if verbose:
                    print(f"    Converged at iteration {t}")
                break

            state = m_step(X_em, X_next_em, r, state, hp_em)

        if not check_monotone(history):
            print("  [NOTE] ELBO non-monotone")

        r = e_step(X_em, X_next_em, state, hp_em)
        if history and history[-1] > best_elbo:
            best_elbo    = history[-1]
            best_state   = state
            best_r       = r
            best_history = history

    # Move results back to original device/dtype
    if best_state is not None and (orig_device != X_em.device or orig_dtype != X_em.dtype):
        best_state, best_r = _to_device_dtype(best_state, best_r, orig_device, orig_dtype)

    return best_state, best_r, best_history


# ─────────────────────────────────────────────────────────────────────────────
# Global discrete EDMD
# ─────────────────────────────────────────────────────────────────────────────

def fit_global(X, X_next, degree=2):
    d = X.shape[1]
    exps = monomial_exponents(d, degree)
    c = X.mean(dim=0)
    r = torch.ones(X.shape[0], dtype=X.dtype, device=X.device)
    K = weighted_discrete_edmd(X, X_next, r, c, exps)
    return {'K': K, 'c': c, 'exps': exps, 'd': d}


def predict_next_global(X, model):
    d = model['d']
    U = X - model['c']
    Phi = monomials(U, model['exps'])
    Phi_next = Phi @ model['K'].T
    return model['c'] + Phi_next[:, 1:d+1]
