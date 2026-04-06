import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from .distributions import mvn_logpdf_batch, residual_logpdf_batch
from .elbo import compute_elbo, check_monotone
from .marginal_likelihood import total_log_marginal


# ── Initialization ────────────────────────────────────────────────────────────

def compute_sigma2_from_residuals(
    X:        torch.Tensor,
    F:        torch.Tensor,
    centers:  torch.Tensor,
    f_centers: torch.Tensor,
    jacobians: torch.Tensor,
    labels:   np.ndarray,
) -> torch.Tensor:
    """
    Calibrate per-cluster sigma2_k from actual within-cluster Taylor residuals.

    sigma2_k = median(||eps_k(x_i)||^2) / d  for points assigned to cluster k.

    Returns a tensor of shape (N,) — one residual variance per cluster.
    """
    d          = X.shape[1]
    N          = centers.shape[0]
    sigma2     = torch.full((N,), 1e-2, dtype=torch.float64)

    for k in range(N):
        mask = labels == k
        if mask.sum() == 0:
            continue
        X_k   = X[mask]
        F_k   = F[mask]
        delta = X_k - centers[k]
        lp    = f_centers[k] + (jacobians[k] @ delta.T).T
        eps   = F_k - lp
        res_sq = (eps ** 2).sum(dim=1)
        sigma2[k] = max(res_sq.median().item() / d, 1e-2)

    return sigma2


def initialize(
    X:    torch.Tensor,
    F:    torch.Tensor,
    f_fn,
    J_fn,
    N:    int,
    hp:   dict,
    seed: int = 42,
) -> dict:
    """
    Initialize EM state using sklearn GMM (k-means warm start).

    Also calibrates sigma2 from within-cluster residuals if hp['sigma2']
    is not set ('auto'), so the residual likelihood is on the right scale.

    Returns state dict with all parameters as float64 torch tensors.
    """
    P, d = X.shape

    gmm = GaussianMixture(
        n_components=N,
        covariance_type='full',
        n_init=5,
        random_state=seed,
    )
    gmm.fit(X.numpy())
    labels = gmm.predict(X.numpy())

    centers     = torch.tensor(gmm.means_,       dtype=torch.float64)  # (N, d)
    covariances = torch.tensor(gmm.covariances_, dtype=torch.float64)  # (N, d, d)
    pi          = torch.tensor(gmm.weights_,     dtype=torch.float64)  # (N,)

    # Regularize initial covariances
    covariances = covariances + 1e-6 * torch.eye(d, dtype=torch.float64).unsqueeze(0)

    f_centers = torch.stack([
        torch.tensor(f_fn(centers[k].numpy()), dtype=torch.float64)
        for k in range(N)
    ])                                                                   # (N, d)

    jacobians = torch.stack([
        torch.tensor(J_fn(centers[k].numpy()), dtype=torch.float64)
        for k in range(N)
    ])                                                                   # (N, d, d)

    # Calibrate sigma2 from actual within-cluster residuals
    if hp.get('sigma2', 'auto') == 'auto':
        sigma2 = compute_sigma2_from_residuals(
            X, F, centers, f_centers, jacobians, labels
        )
        print(f"    sigma2 calibrated per cluster: mean={sigma2.mean().item():.4f}, "
              f"range=[{sigma2.min().item():.4f}, {sigma2.max().item():.4f}]")
        learn_sigma2 = True
    else:
        # Fixed sigma2: either scalar or pre-set tensor
        s = hp['sigma2']
        if isinstance(s, (int, float)):
            sigma2 = torch.full((N,), float(s), dtype=torch.float64)
        else:
            sigma2 = s.clone()
        learn_sigma2 = False

    return {
        'pi':          pi,
        'centers':     centers,
        'covariances': covariances,
        'f_centers':   f_centers,
        'jacobians':   jacobians,
        'sigma2':      sigma2,            # (N,) per-cluster residual variance
        'learn_sigma2': learn_sigma2,     # if False, sigma2 stays fixed in M-step
        'N':           N,
        'd':           d,
        'P':           P,
    }


# ── E-step ────────────────────────────────────────────────────────────────────

def e_step(
    X:     torch.Tensor,   # (P, d)
    F:     torch.Tensor,   # (P, d)
    state: dict,
    hp:    dict,
) -> torch.Tensor:          # (P, N)
    """
    Compute soft assignments r_ik.

    log r_ik  ∝  log pi_k
               + log N(x_i; c_k, Sigma_k)           [proximity]
               + log N(eps_k(x_i); 0, sigma2*I)     [residual — novel]

    All computation in log space. Normalize with logsumexp.
    """
    log_prox  = mvn_logpdf_batch(X, state['centers'], state['covariances'])
    log_resid = residual_logpdf_batch(
        X, F,
        state['centers'],
        state['f_centers'],
        state['jacobians'],
        state['sigma2'],
    )
    log_pi    = torch.log(state['pi']).unsqueeze(0)  # (1, N)

    log_r_un  = log_pi + log_prox + log_resid        # (P, N)
    log_r     = log_r_un - torch.logsumexp(log_r_un, dim=1, keepdim=True)

    return torch.exp(log_r)                           # (P, N)


# ── M-step ────────────────────────────────────────────────────────────────────

def m_step(
    X:     torch.Tensor,   # (P, d)
    F:     torch.Tensor,   # (P, d)
    r:     torch.Tensor,   # (P, N)
    state: dict,
    f_fn,
    J_fn,
    hp:    dict,
) -> dict:
    """
    Update all parameters given soft assignments r.

    Center update has three terms (Rung 3, M-step derivation):
      hat_Lambda_k @ c_k = Lambda0 @ mu0          [prior]
                         + Sigma_k_inv @ sum r_ik x_i   [data]
                         + (1/sigma2) sum r_ik J_k^T eps_k(x_i)  [residual — novel]
    """
    N       = state['N']
    d       = state['d']
    alpha0  = hp['alpha0']
    mu0     = hp['mu0']
    Lambda0 = hp['Lambda0']
    Psi0    = hp['Psi0']
    nu0     = hp['nu0']
    P       = X.shape[0]

    R           = r.sum(dim=0)                    # (N,) effective masses
    Sigma_inv   = torch.linalg.inv(state['covariances'])  # (N, d, d)

    centers_new     = torch.zeros(N, d, dtype=torch.float64)
    covariances_new = torch.zeros(N, d, d, dtype=torch.float64)
    f_centers_new   = torch.zeros(N, d, dtype=torch.float64)
    jacobians_new   = torch.zeros(N, d, d, dtype=torch.float64)

    for k in range(N):
        r_k = r[:, k]       # (P,)
        R_k = R[k].item()

        # ── Posterior precision ───────────────────────────────────────────────
        hat_Lambda_k = Lambda0 + R_k * Sigma_inv[k]   # (d, d)

        # ── Term 1: prior ─────────────────────────────────────────────────────
        prior_term = Lambda0 @ mu0                     # (d,)

        # ── Term 2: data (responsibility-weighted mean) ───────────────────────
        # This is where the residual likelihood acts: r_ik is already reweighted
        # by exp(-||eps_k(x_i)||^2 / 2sigma^2) in the E-step.
        # Points with large Taylor residuals under cluster k get lower r_ik,
        # so they contribute less to this weighted mean.
        # This pulls c_k toward low-curvature regions without an explicit
        # correction term — the E-step does the work.
        data_term = Sigma_inv[k] @ (r_k @ X)          # (d,)

        # Note: the residual correction term J_k^T @ sum(r_ik * eps_k(x_i))
        # derived in Rung 3 is INCORRECT. With J_k treated as fixed in the
        # M-step (standard EM coordinate ascent), d(eps_k)/dc_k = -J_k + J_k = 0.
        # The residual enters only through r_ik in the E-step.

        # ── Solve for new center ──────────────────────────────────────────────
        rhs      = prior_term + data_term             # (d,)
        c_k_new  = torch.linalg.solve(hat_Lambda_k, rhs)        # (d,)
        centers_new[k] = c_k_new

        # ── Recompute f(c_k) and J_k at new center ────────────────────────────
        f_centers_new[k] = torch.tensor(
            f_fn(c_k_new.numpy()), dtype=torch.float64
        )
        jacobians_new[k] = torch.tensor(
            J_fn(c_k_new.numpy()), dtype=torch.float64
        )

        # ── Covariance update ─────────────────────────────────────────────────
        diff    = X - c_k_new                                    # (P, d)
        scatter = (r_k.unsqueeze(1) * diff).T @ diff             # (d, d)
        Sigma_k_new = (Psi0 + scatter) / (nu0 + R_k + d + 1)
        Sigma_k_new = Sigma_k_new + 1e-6 * torch.eye(d, dtype=torch.float64)
        covariances_new[k] = Sigma_k_new

    # ── Update pi ─────────────────────────────────────────────────────────────
    pi_new = (R + alpha0 - 1.0) / (P + N * (alpha0 - 1.0))
    pi_new = torch.clamp(pi_new, min=1e-10)
    pi_new = pi_new / pi_new.sum()

    # ── Update per-cluster sigma2_k (only if learned, not fixed) ─────────────
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
        'N':           N,
        'd':           d,
        'P':           P,
    }


# ── Dead cluster handling ─────────────────────────────────────────────────────

def handle_dead_clusters(
    state:     dict,
    r:         torch.Tensor,
    X:         torch.Tensor,
    F:         torch.Tensor,
    f_fn,
    J_fn,
    hp:        dict,
    threshold: float = 1.0,
    min_N:     int   = 2,
) -> tuple:
    """
    Remove clusters whose effective mass R_k < threshold.

    Theory: with alpha0 < 1 the Dirichlet prior drives pi_k → 0 for
    unused clusters. Once R_k < threshold the cluster explains nothing
    and should be permanently removed.

    We never reinitialize mid-EM — doing so injects randomness that
    breaks ELBO monotonicity. The user starts with N larger than needed
    and lets pruning find the right number.

    min_N: never reduce below this many clusters.
    """
    R    = r.sum(dim=0)
    dead = (R < threshold).nonzero(as_tuple=True)[0].tolist()

    if not dead:
        return state, r

    for k in sorted(dead, reverse=True):   # remove from high index down
        if state['N'] <= min_N:
            break
        print(f"  Pruning cluster {k} (R_k={R[k].item():.2f})")
        state = _remove_cluster(state, k)

    # Recompute responsibilities after removal
    r = e_step(X, F, state, hp)
    return state, r


def _remove_cluster(state: dict, k: int) -> dict:
    keep = [j for j in range(state['N']) if j != k]
    s = {
        'pi':          state['pi'][keep],
        'centers':     state['centers'][keep],
        'covariances': state['covariances'][keep],
        'f_centers':   state['f_centers'][keep],
        'jacobians':   state['jacobians'][keep],
        'sigma2':      state['sigma2'][keep],
        'learn_sigma2': state.get('learn_sigma2', True),
        'N':           state['N'] - 1,
        'd':           state['d'],
        'P':           state['P'],
    }
    s['pi'] = s['pi'] / s['pi'].sum()
    return s


def _reinitialize_to_worst_point(
    state: dict,
    k:     int,
    X:     torch.Tensor,
    F:     torch.Tensor,
    r:     torch.Tensor,
    f_fn,
    J_fn,
    hp:    dict,
) -> dict:
    """Place new center at the phase point with highest residual
    under its current best cluster — where the method needs help most."""
    best_k    = r.argmax(dim=1)
    residuals = torch.zeros(X.shape[0], dtype=torch.float64)

    for j in range(state['N']):
        mask = best_k == j
        if mask.sum() == 0:
            continue
        X_j   = X[mask]
        F_j   = F[mask]
        delta = X_j - state['centers'][j]
        lp    = state['f_centers'][j] + (state['jacobians'][j] @ delta.T).T
        eps   = F_j - lp
        residuals[mask] = (eps ** 2).sum(dim=1)

    idx = residuals.argmax().item()
    state['centers'][k]    = X[idx].clone()
    state['f_centers'][k]  = torch.tensor(f_fn(X[idx].numpy()), dtype=torch.float64)
    state['jacobians'][k]  = torch.tensor(J_fn(X[idx].numpy()), dtype=torch.float64)
    state['covariances'][k] = hp['Psi0'].clone()
    state['pi'][k]          = 1.0 / state['N']
    state['pi']             = state['pi'] / state['pi'].sum()
    return state


# ── Full EM loop ──────────────────────────────────────────────────────────────

def fit(
    X:           torch.Tensor,
    F:           torch.Tensor,
    f_fn,
    J_fn,
    N:           int,
    hp:          dict,
    n_iter:      int   = 100,
    tol:         float = 1e-4,
    n_restarts:  int   = 3,
    verbose:     bool  = True,
) -> tuple:
    """
    Run EM with multiple random restarts.
    Returns (best_state, best_r, best_elbo_history).
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
            # E-step: compute soft assignments given current params
            r        = e_step(X, F, state, hp)

            # Prune dead clusters (Dirichlet prior drives pi_k → 0)
            state, r = handle_dead_clusters(state, r, X, F, f_fn, J_fn, hp)
            if state['N'] == 0:
                print("  All clusters pruned — stopping")
                break

            # Compute ELBO before M-step using current q and current θ
            # This preserves the EM monotonicity guarantee:
            # ELBO(q_t, θ_t) ≤ ELBO(q_t, θ_{t+1}) by M-step optimality
            elbo_val = compute_elbo(X, F, r, state, hp).item()
            history.append(elbo_val)

            if verbose and t % 10 == 0:
                print(f"    iter {t:3d} | ELBO = {elbo_val:.4f} | N_active = {state['N']}")

            if t > 0 and abs(history[-1] - history[-2]) < tol:
                if verbose:
                    print(f"    Converged at iteration {t}")
                break

            # M-step: update θ given current q
            state = m_step(X, F, r, state, f_fn, J_fn, hp)

        if not check_monotone(history):
            print("  [WARNING] ELBO non-monotone — possible M-step bug")

        # Final E-step after last M-step so best_r is consistent with best_state
        r = e_step(X, F, state, hp)

        if history and history[-1] > best_elbo:
            best_elbo    = history[-1]
            best_state   = state
            best_r       = r
            best_history = history

    return best_state, best_r, best_history
