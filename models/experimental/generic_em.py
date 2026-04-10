"""
Generic EM pipeline for residual-aware clustering with any LocalModel.

This is the main entry point for fitting cluster-weighted models with
arbitrary local surrogates. It accepts ANY object satisfying the
``LocalModel`` protocol -- polynomial EDMD, neural networks, transformers,
or custom implementations -- and runs the full Bayesian EM loop:
initialization (GMM warm-start), E-step (soft assignment), M-step
(center/covariance/model refit), ELBO tracking, dead-cluster pruning,
and multi-restart selection. Device/dtype aware (CPU, CUDA, MPS; EM
always runs in float64, with automatic device migration).

Usage
-----
Fit with polynomial EDMD::

    from residual_aware_clustering.models.experimental import (
        generic_em, PolynomialDiscreteEDMD, ObservableType,
    )

    prototype = PolynomialDiscreteEDMD(degree=2, observable_type=ObservableType.FULL)
    state, r, history = generic_em.fit(
        X, X_next,          # (P, d) input/target pairs
        N=8,                # initial number of clusters
        hp=hp,              # hyperparameters dict (alpha0, mu0, Lambda0, Psi0, nu0, ...)
        model_prototype=prototype,
        n_iter=100,
        n_restarts=3,
    )

    # state['centers']      — (N_final, d) cluster centers
    # state['models']       — list of N_final fitted LocalModel instances
    # state['pi']           — (N_final,) mixing weights
    # state['covariances']  — (N_final, d, d) proximity covariances
    # state['sigma2']       — (N_final,) per-cluster residual variances
    # r                     — (P, N_final) soft assignment matrix
    # history               — list of ELBO values per iteration

Fit with a neural network model::

    from residual_aware_clustering.models.experimental import generic_em, NeuralNetModel

    prototype = NeuralNetModel(hidden_dims=(128, 128), n_epochs=80)
    state, r, history = generic_em.fit(X, X_next, N=5, hp=hp, model_prototype=prototype)

Fit with a transformer model::

    from residual_aware_clustering.models.experimental import generic_em, TransformerNetModel

    prototype = TransformerNetModel(d_model=64, n_heads=4, n_layers=2)
    state, r, history = generic_em.fit(X, X_next, N=5, hp=hp, model_prototype=prototype)

Key concepts
------------
- **model_prototype**: A single unfitted ``LocalModel`` instance. The pipeline
  calls ``clone()`` on it N times to create one independent model per cluster.
- **Hyperparameters (hp)**: Dict with Bayesian prior parameters: ``alpha0``
  (Dirichlet concentration), ``mu0`` / ``Lambda0`` (prior mean / precision for
  centers), ``Psi0`` / ``nu0`` / ``kappa0`` (NIW prior on covariances), and
  optionally ``sigma2`` (set to ``'auto'`` for per-cluster calibration).
- **Multi-restart**: Runs ``n_restarts`` independent EM chains and returns the
  result with the highest ELBO, reducing sensitivity to initialization.
- **Dead-cluster pruning**: Clusters with effective weight below a threshold
  are automatically removed during EM, so the final N may be smaller than the
  initial N.
- **Device handling**: EM internally runs on CPU in float64 (MPS does not
  support float64). Results are automatically moved back to the original
  device/dtype after fitting.

Pipeline functions
------------------
``fit(X, Y, N, hp, model_prototype, ...)``
    Full EM with multi-restart. Returns ``(state, r, history)``.

``initialize(X, Y, N, hp, model_prototype, ...)``
    GMM warm-start + initial model fitting. Returns the initial state dict.

``e_step(X, Y, state, hp)``
    Compute soft assignments. Returns (P, N) responsibility matrix.

``m_step(X, Y, r, state, hp)``
    Update centers, covariances, models, mixing weights, and sigma2.

``compute_elbo(X, Y, r, state, hp)``
    Evaluate the evidence lower bound (ELBO).

``prune_dead(state, r, X, Y, hp, ...)``
    Remove clusters with negligible effective weight.
"""

import time

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

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
    """Log p(eps_k(x_i) | 0, sigma2_k * I) for all i, k. Returns (P, N).

    Streams over clusters to avoid materializing the full (P, N, d) prediction
    tensor — important when N is large (e.g. 1000+ clusters).
    """
    P = X.shape[0]
    N = len(models)
    sq_norm = torch.zeros(P, N, dtype=X.dtype, device=X.device)
    for k in range(N):
        eps_k = Y - models[k].predict(X, centers[k])
        sq_norm[:, k] = (eps_k ** 2).sum(dim=1)

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
    """Compute soft assignments (responsibilities) for all points and clusters.

    Parameters
    ----------
    X : torch.Tensor
        Input states, shape ``(P, d)``.
    Y : torch.Tensor
        Target states, shape ``(P, d)``.
    state : dict
        Current EM state containing ``centers``, ``covariances``, ``models``,
        ``sigma2``, ``pi``, and ``d``.
    hp : dict
        Hyperparameters (unused directly, kept for API consistency).

    Returns
    -------
    torch.Tensor
        Responsibility matrix of shape ``(P, N)`` where each row sums to 1.
    """
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
    """Update cluster parameters given current responsibilities.

    Updates centers (MAP with NIW prior), refits each local model,
    recomputes proximity covariances, mixing weights, and per-cluster
    residual variances.

    Parameters
    ----------
    X : torch.Tensor
        Input states, shape ``(P, d)``.
    Y : torch.Tensor
        Target states, shape ``(P, d)``.
    r : torch.Tensor
        Responsibility matrix, shape ``(P, N)``.
    state : dict
        Current EM state dict.
    hp : dict
        Hyperparameters with keys ``alpha0``, ``mu0``, ``Lambda0``,
        ``Psi0``, ``nu0``.

    Returns
    -------
    dict
        Updated state dict with keys ``pi``, ``centers``, ``covariances``,
        ``models``, ``sigma2``, ``learn_sigma2``, ``N``, ``d``, ``P``.
    """
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

    # Per-cluster sigma2 (streamed to avoid materializing (P, N, d))
    if state.get('learn_sigma2', True):
        sigma2_new = torch.zeros(N, dtype=dt, device=dev)
        for k in range(N):
            eps_k = Y - models[k].predict(X, centers_new[k])
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
    """Evaluate the evidence lower bound (ELBO) of the current model.

    Combines expected log-likelihood terms (mixing weights, proximity,
    residual), the entropy of responsibilities, and Bayesian prior terms
    (Dirichlet on pi, NIW on centers/covariances).

    Parameters
    ----------
    X : torch.Tensor
        Input states, shape ``(P, d)``.
    Y : torch.Tensor
        Target states, shape ``(P, d)``.
    r : torch.Tensor
        Responsibility matrix, shape ``(P, N)``.
    state : dict
        Current EM state dict.
    hp : dict
        Hyperparameters with keys ``alpha0``, ``mu0``, ``kappa0``,
        ``Psi0``, ``nu0``.

    Returns
    -------
    torch.Tensor
        Scalar ELBO value (in nats).
    """
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
    """Remove clusters whose effective weight falls below a threshold.

    After removal, responsibilities are recomputed via ``e_step``.

    Parameters
    ----------
    state : dict
        Current EM state dict.
    r : torch.Tensor
        Responsibility matrix, shape ``(P, N)``.
    X : torch.Tensor
        Input states, shape ``(P, d)``.
    Y : torch.Tensor
        Target states, shape ``(P, d)``.
    hp : dict
        Hyperparameters (forwarded to ``e_step``).
    threshold : float, optional
        Minimum effective cluster weight ``R_k`` to keep, by default 1.0.
    min_N : int, optional
        Minimum number of clusters to retain, by default 2.

    Returns
    -------
    state : dict
        Updated state with dead clusters removed.
    r : torch.Tensor
        Recomputed responsibility matrix, shape ``(P, N_new)``.
    """
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

def initialize(X, Y, N, hp, model_prototype, seed=42, max_gmm_samples=10000):
    """Create the initial EM state via GMM warm-start and local model fitting.

    Fits a scikit-learn ``GaussianMixture`` to ``X`` to obtain initial
    centers, covariances, and mixing weights, then clones the model
    prototype N times and fits each clone on its assigned cluster.
    Optionally calibrates per-cluster residual variances.

    Parameters
    ----------
    X : torch.Tensor
        Input states, shape ``(P, d)``.
    Y : torch.Tensor
        Target states, shape ``(P, d)``.
    N : int
        Requested number of clusters (may be reduced if GMM finds fewer).
    hp : dict
        Hyperparameters. If ``hp['sigma2']`` is ``'auto'``, per-cluster
        residual variances are calibrated from median squared residuals.
    model_prototype : LocalModel
        Unfitted model instance; ``clone()`` is called N times.
    seed : int, optional
        Random seed for GMM and subsampling, by default 42.
    max_gmm_samples : int or None, optional
        Maximum number of points for GMM fitting. If ``None`` or the
        dataset is smaller, all points are used. By default 10000.

    Returns
    -------
    dict
        Initial state dict with keys ``pi``, ``centers``, ``covariances``,
        ``models``, ``sigma2``, ``learn_sigma2``, ``N``, ``d``, ``P``.
    """
    P, d = X.shape
    dev = X.device
    dt = X.dtype

    # GMM warm-start (CPU-only)
    gmm = GaussianMixture(n_components=N, covariance_type='full',
                          n_init=5, random_state=seed)
    X_np = X.detach().cpu().to(torch.float64).numpy()

    # Subsample for GMM fitting if dataset is large (GMM is O(N*K*d^2))
    if max_gmm_samples and X_np.shape[0] > max_gmm_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X_np.shape[0], max_gmm_samples, replace=False)
        gmm.fit(X_np[idx])
    else:
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

    # Calibrate sigma2 (streamed to avoid materializing (P, N, d))
    if hp.get('sigma2', 'auto') == 'auto':
        sigma2 = torch.full((N,), 1e-6, dtype=dt, device=dev)
        for k in range(N):
            mask = labels == k
            if mask.sum() == 0:
                continue
            X_k = X[mask]
            Y_k = Y[mask]
            eps_k = Y_k - models[k].predict(X_k, centers[k])
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
    """Check whether the ELBO history is monotonically non-decreasing.

    Prints a warning if any iteration shows a decrease larger than
    ``tol`` nats.

    Parameters
    ----------
    history : list of float
        ELBO values recorded at each EM iteration.
    tol : float, optional
        Tolerance for acceptable decrease (in nats), by default 1.0.

    Returns
    -------
    bool
        ``True`` if no violations exceed ``tol``, ``False`` otherwise.
    """
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
        n_iter=100, tol=1e-4, n_restarts=3, max_gmm_samples=10000, verbose=True):
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
            print(f"\n  Restart {restart + 1}/{n_restarts}  (generic EM)", flush=True)
            print(f"    Initializing (GMM on {X_em.shape[0]} points, d={X_em.shape[1]})...", flush=True)

        t_start = time.time()
        state = initialize(X_em, Y_em, N, hp_em, model_prototype,
                           seed=restart * 17, max_gmm_samples=max_gmm_samples)
        if verbose:
            print(f"    Initialized in {time.time() - t_start:.1f}s | N_active = {state['N']}", flush=True)

        history = []
        pbar = tqdm(range(n_iter), desc=f"    Restart {restart+1}/{n_restarts}",
                    disable=not verbose, leave=True)

        for t in pbar:
            r = e_step(X_em, Y_em, state, hp_em)
            state, r = prune_dead(state, r, X_em, Y_em, hp_em)
            if state['N'] == 0:
                pbar.set_postfix_str("all clusters pruned")
                break

            elbo_val = compute_elbo(X_em, Y_em, r, state, hp_em).item()
            history.append(elbo_val)

            delta_str = ""
            if t > 0:
                delta = history[-1] - history[-2]
                delta_str = f"dE={delta:+.2f}"

            pbar.set_postfix_str(
                f"ELBO={elbo_val:.2f} {delta_str} N={state['N']}")

            if t > 0 and abs(history[-1] - history[-2]) < tol:
                pbar.set_postfix_str(
                    f"ELBO={elbo_val:.2f} converged N={state['N']}")
                break

            state = m_step(X_em, Y_em, r, state, hp_em)

        pbar.close()

        if not check_monotone(history):
            print("  [NOTE] ELBO non-monotone")

        r = e_step(X_em, Y_em, state, hp_em)
        if history and history[-1] > best_elbo:
            best_elbo = history[-1]
            best_state = state
            best_r = r
            best_history = history

        if verbose and history:
            total_time = time.time() - t_start
            print(f"    Done | ELBO={history[-1]:.2f} | {len(history)} iters | {total_time:.1f}s")

    # Move results back to original device/dtype
    if best_state is not None and (orig_device != em_device or orig_dtype != torch.float64):
        for key in ['pi', 'centers', 'covariances', 'sigma2']:
            if isinstance(best_state[key], torch.Tensor):
                best_state[key] = best_state[key].cpu().to(
                    device=orig_device, dtype=orig_dtype)
        _models_to_device_dtype(best_state['models'], orig_device, orig_dtype)
        best_r = best_r.cpu().to(device=orig_device, dtype=orig_dtype)

    return best_state, best_r, best_history
