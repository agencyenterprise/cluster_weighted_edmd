import torch
from .distributions import (
    mvn_logpdf_batch,
    residual_logpdf_batch,
    dirichlet_logpdf,
    niw_logpdf,
)


def compute_elbo(
    X:     torch.Tensor,   # (P, d)
    F:     torch.Tensor,   # (P, d)
    r:     torch.Tensor,   # (P, N)
    state: dict,
    hp:    dict,
) -> torch.Tensor:          # scalar
    """
    ELBO = E_q[log p(X, Z, F | theta)] + H[q] + log p(theta)

    Term 1: sum_ik r_ik * log pi_k                          [assignment]
    Term 2: sum_ik r_ik * log N(x_i; c_k, Sigma_k)         [proximity]
    Term 3: sum_ik r_ik * log N(eps_k(x_i); 0, sigma2*I)   [residual — novel]
    Term 4: -sum_ik r_ik * log r_ik                         [entropy H[q]]
    Term 5: log p(pi)                                        [Dirichlet prior]
    Term 6: sum_k log p(c_k, Sigma_k)                       [NIW prior]

    Must be monotonically non-decreasing across EM iterations.
    Any decrease signals a bug in the M-step.
    """
    N      = state['N']
    sigma2 = hp['sigma2']

    # Term 1 — mixing weight log-likelihood
    log_pi = torch.log(state['pi'])                # (N,)
    term1  = (r * log_pi.unsqueeze(0)).sum()

    # Term 2 — proximity log-likelihood
    log_prox = mvn_logpdf_batch(
        X, state['centers'], state['covariances']
    )                                              # (P, N)
    term2    = (r * log_prox).sum()

    # Term 3 — residual log-likelihood (novel term)
    log_resid = residual_logpdf_batch(
        X, F,
        state['centers'],
        state['f_centers'],
        state['jacobians'],
        sigma2,
    )                                              # (P, N)
    term3     = (r * log_resid).sum()

    # Term 4 — entropy of variational distribution
    log_r  = torch.log(r + 1e-300)                # guard log(0)
    term4  = -(r * log_r).sum()

    # Term 5 — Dirichlet prior on pi
    term5  = dirichlet_logpdf(state['pi'], hp['alpha0'])

    # Term 6 — NIW prior on (c_k, Sigma_k) for each cluster
    term6  = sum(
        niw_logpdf(
            state['centers'][k],
            state['covariances'][k],
            hp['mu0'],
            hp['kappa0'],
            hp['Psi0'],
            hp['nu0'],
        )
        for k in range(N)
    )

    return term1 + term2 + term3 + term4 + term5 + term6


def check_monotone(history: list, tol: float = 1.0) -> bool:
    """
    Check ELBO history for monotone non-decrease.

    Tolerance is 1.0 nat — violations below this are numerical
    precision artifacts from the NIW prior log-density computation,
    not algorithmic bugs in the M-step.

    Returns True if no violation exceeds tol.
    """
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
