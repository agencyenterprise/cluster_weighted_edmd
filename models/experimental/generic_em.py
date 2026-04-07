"""
Generic EM pipeline for residual-aware clustering with any LocalModel.

Single implementation that works with polynomial EDMD, neural networks,
or any model satisfying the LocalModel protocol. Device/dtype aware
(CPU, CUDA, MPS).
"""

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from ..distributions_gpu import mvn_logpdf_batch, dirichlet_logpdf, niw_logpdf
from .local_model import LocalModel


# ─────────────────────────────────────────────────────────────────────────────
# Prediction + residual
# ─────────────────────────────────────────────────────────────────────────────

def predict_all(X, centers, models, d):
    """Predict Y for all points under each cluster's local model. Returns (P, N, d)."""
    P = X.shape[0]
    N = len(models)
    Y_pred = torch.zeros(P, N, d, dtype=X.dtype, device=X.device)
    for k in range(N):
        Y_pred[:, k, :] = models[k].predict(X, centers[k])
    return Y_pred


def residual_logpdf(X, Y, centers, models, sigma2, d):
    """Log p(eps_k(x_i) | 0, sigma2_k * I) for all i, k. Returns (P, N)."""
    Y_pred = predict_all(X, centers, models, d)
    eps = Y.unsqueeze(1) - Y_pred
    sq_norm = (eps ** 2).sum(dim=2)
    N = len(models)
    if isinstance(sigma2, (int, float)):
        s2 = torch.full((N,), float(sigma2), dtype=X.dtype, device=X.device)
    else:
        s2 = sigma2
    log_norm = -(d / 2.0) * torch.log(2.0 * torch.pi * s2).unsqueeze(0)
    return log_norm - sq_norm / (2.0 * s2.unsqueeze(0))


# ─────────────────────────────────────────────────────────────────────────────
# E-step
# ─────────────────────────────────────────────────────────────────────────────

def e_step(X, Y, state, hp):
    log_prox = mvn_logpdf_batch(X, state['centers'], state['covariances'])
    log_resid = residual_logpdf(
        X, Y, state['centers'], state['models'], state['sigma2'], state['d'])
    log_pi = torch.log(state['pi']).unsqueeze(0)
    log_r_un = log_pi + log_prox + log_resid
    log_r = log_r_un - torch.logsumexp(log_r_un, dim=1, keepdim=True)
    return torch.exp(log_r)


# ─────────────────────────────────────────────────────────────────────────────
# M-step
# ─────────────────────────────────────────────────────────────────────────────

def m_step(X, Y, r, state, hp):
    N, d, P = state['N'], state['d'], state['P']
    dev = X.device
    dt = X.dtype
    alpha0 = hp['alpha0']
    mu0 = hp['mu0']
    Lambda0 = hp['Lambda0']
    Psi0 = hp['Psi0']
    nu0 = hp['nu0']
    models = state['models']

    R = r.sum(dim=0)
    Sigma_inv = torch.linalg.inv(state['covariances'])
    eye_d = torch.eye(d, dtype=dt, device=dev)

    centers_new = torch.zeros(N, d, dtype=dt, device=dev)
    covariances_new = torch.zeros(N, d, d, dtype=dt, device=dev)

    for k in range(N):
        r_k = r[:, k]
        R_k = R[k].item()

        # Center update (proximity-driven)
        hat_Lambda = Lambda0 + R_k * Sigma_inv[k] + 1e-6 * eye_d
        rhs = Lambda0 @ mu0 + Sigma_inv[k] @ (r_k @ X)
        c_k_new = torch.linalg.solve(hat_Lambda, rhs)
        centers_new[k] = c_k_new

        # Refit local model at new center
        models[k].fit(X, Y, r_k, c_k_new)

        # Covariance update (NIW posterior)
        diff = X - c_k_new
        scatter = (r_k.unsqueeze(1) * diff).T @ diff
        covariances_new[k] = (Psi0 + scatter) / (nu0 + R_k + d + 1) + 1e-6 * eye_d

    # Mixing weights
    pi_new = (R + alpha0 - 1.0) / (P + N * (alpha0 - 1.0))
    pi_new = torch.clamp(pi_new, min=1e-10)
    pi_new = pi_new / pi_new.sum()

    # Per-cluster sigma2
    if state.get('learn_sigma2', True):
        Y_pred = predict_all(X, centers_new, models, d)
        sigma2_new = torch.zeros(N, dtype=dt, device=dev)
        for k in range(N):
            eps_k = Y - Y_pred[:, k]
            sq = (eps_k ** 2).sum(dim=1)
            sigma2_new[k] = max((r[:, k] * sq).sum().item() / (d * R[k].item()), 1e-6)
    else:
        sigma2_new = state['sigma2']

    return {
        'pi': pi_new,
        'centers': centers_new,
        'covariances': covariances_new,
        'models': models,
        'sigma2': sigma2_new,
        'learn_sigma2': state.get('learn_sigma2', True),
        'N': N, 'd': d, 'P': P,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ELBO
# ─────────────────────────────────────────────────────────────────────────────

def compute_elbo(X, Y, r, state, hp):
    N = state['N']

    log_pi = torch.log(state['pi'])
    term1 = (r * log_pi.unsqueeze(0)).sum()

    log_prox = mvn_logpdf_batch(X, state['centers'], state['covariances'])
    term2 = (r * log_prox).sum()

    log_resid = residual_logpdf(
        X, Y, state['centers'], state['models'], state['sigma2'], state['d'])
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

def prune_dead(state, r, X, Y, hp, threshold=1.0, min_N=2):
    R = r.sum(dim=0)
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
            'centers': state['centers'][keep],
            'covariances': state['covariances'][keep],
            'models': [state['models'][j] for j in keep],
            'sigma2': state['sigma2'][keep],
            'learn_sigma2': state.get('learn_sigma2', True),
            'N': state['N'] - 1, 'd': state['d'], 'P': state['P'],
        }
    r = e_step(X, Y, state, hp)
    return state, r


# ─────────────────────────────────────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────────────────────────────────────

def initialize(X, Y, N, hp, model_prototype, seed=42):
    P, d = X.shape
    dev = X.device
    dt = X.dtype

    # GMM warm-start (CPU-only)
    gmm = GaussianMixture(n_components=N, covariance_type='full',
                          n_init=5, random_state=seed)
    X_np = X.detach().cpu().to(torch.float64).numpy()
    gmm.fit(X_np)
    labels = gmm.predict(X_np)

    # Handle fewer clusters than requested
    n_found = len(set(labels))
    if n_found < N:
        print(f"    GMM found {n_found} clusters (requested {N}), reducing N")
        N = n_found
        unique_labels = sorted(set(labels))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels])
        active = [unique_labels[i] for i in range(N)]
        gmm.means_ = gmm.means_[active]
        gmm.covariances_ = gmm.covariances_[active]
        gmm.weights_ = gmm.weights_[active]
        gmm.weights_ /= gmm.weights_.sum()

    centers = torch.tensor(gmm.means_, dtype=dt, device=dev)
    covariances = torch.tensor(gmm.covariances_, dtype=dt, device=dev) \
                  + 1e-6 * torch.eye(d, dtype=dt, device=dev).unsqueeze(0)
    pi = torch.tensor(gmm.weights_, dtype=dt, device=dev)

    # Create N model instances
    models = []
    for k in range(N):
        model_k = model_prototype.clone()
        mask = torch.tensor(labels == k, device=dev)
        r_k = mask.to(dt)
        if r_k.sum() < model_k.min_points:
            model_k.fallback_init(d, dev, dt)
        else:
            model_k.fit(X, Y, r_k, centers[k])
        models.append(model_k)

    # Calibrate sigma2
    if hp.get('sigma2', 'auto') == 'auto':
        Y_pred = predict_all(X, centers, models, d)
        sigma2 = torch.full((N,), 1e-6, dtype=dt, device=dev)
        for k in range(N):
            mask = labels == k
            if mask.sum() == 0:
                continue
            eps_k = Y[mask] - Y_pred[mask, k]
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
        'models': models, 'sigma2': sigma2, 'learn_sigma2': learn_sigma2,
        'N': N, 'd': d, 'P': P,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Convergence check
# ─────────────────────────────────────────────────────────────────────────────

def check_monotone(history, tol=1.0):
    if len(history) < 2:
        return True
    violations = [
        (t, history[t] - history[t - 1])
        for t in range(1, len(history))
        if history[t] - history[t - 1] < -tol
    ]
    if violations:
        print(f"  [WARNING] ELBO decreased at {len(violations)} steps — "
              f"largest: {min(d for _, d in violations):.4f} nats")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Full EM loop
# ─────────────────────────────────────────────────────────────────────────────

def _models_to_device_dtype(models, device, dtype):
    for m in models:
        m.to(device, dtype)


def fit(X, Y, N, hp, model_prototype,
        n_iter=100, tol=1e-4, n_restarts=3, verbose=True):
    """
    Generic EM fit with any LocalModel.

    Args:
        X: (P, d) input states
        Y: (P, d) targets (X_next for discrete, F for continuous)
        N: initial cluster count
        hp: hyperparameters dict
        model_prototype: LocalModel instance used as template (cloned N times)
    """
    orig_device = X.device
    orig_dtype = X.dtype

    # EM always runs at float64. MPS falls back to CPU.
    if orig_device.type == "mps":
        em_device = torch.device("cpu")
    else:
        em_device = orig_device

    if orig_dtype != torch.float64 or em_device != orig_device:
        X_em = X.cpu().to(dtype=torch.float64, device=em_device)
        Y_em = Y.cpu().to(dtype=torch.float64, device=em_device)
        hp_em = {}
        for k, v in hp.items():
            if isinstance(v, torch.Tensor):
                hp_em[k] = v.cpu().to(dtype=torch.float64, device=em_device)
            else:
                hp_em[k] = v
    else:
        X_em, Y_em, hp_em = X, Y, hp

    best_elbo = -torch.inf
    best_state, best_r, best_history = None, None, None

    for restart in range(n_restarts):
        if verbose:
            print(f"\n  Restart {restart + 1}/{n_restarts}  (generic EM)")

        state = initialize(X_em, Y_em, N, hp_em, model_prototype,
                           seed=restart * 17)
        history = []

        for t in range(n_iter):
            r = e_step(X_em, Y_em, state, hp_em)
            state, r = prune_dead(state, r, X_em, Y_em, hp_em)
            if state['N'] == 0:
                break

            elbo_val = compute_elbo(X_em, Y_em, r, state, hp_em).item()
            history.append(elbo_val)

            if verbose and t % 10 == 0:
                print(f"    iter {t:3d} | ELBO = {elbo_val:.2f} | N_active = {state['N']}")

            if t > 0 and abs(history[-1] - history[-2]) < tol:
                if verbose:
                    print(f"    Converged at iteration {t}")
                break

            state = m_step(X_em, Y_em, r, state, hp_em)

        if not check_monotone(history):
            print("  [NOTE] ELBO non-monotone")

        r = e_step(X_em, Y_em, state, hp_em)
        if history and history[-1] > best_elbo:
            best_elbo = history[-1]
            best_state = state
            best_r = r
            best_history = history

    # Move results back to original device/dtype
    if best_state is not None and (orig_device != em_device or orig_dtype != torch.float64):
        for key in ['pi', 'centers', 'covariances', 'sigma2']:
            if isinstance(best_state[key], torch.Tensor):
                best_state[key] = best_state[key].cpu().to(
                    device=orig_device, dtype=orig_dtype)
        _models_to_device_dtype(best_state['models'], orig_device, orig_dtype)
        best_r = best_r.cpu().to(device=orig_device, dtype=orig_dtype)

    return best_state, best_r, best_history
