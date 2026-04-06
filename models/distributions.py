import torch
from torch.distributions import MultivariateNormal, Dirichlet


# ── Multivariate log-gamma ────────────────────────────────────────────────────

def _mvlgamma(a: float, d: int) -> torch.Tensor:
    """
    Log of multivariate gamma function:
    log Gamma_d(a) = d(d-1)/4 * log(pi) + sum_{j=1}^{d} log Gamma(a + (1-j)/2)
    """
    result = (d * (d - 1) / 4.0) * torch.log(torch.tensor(torch.pi, dtype=torch.float64))
    for j in range(1, d + 1):
        result = result + torch.lgamma(torch.tensor(a + (1.0 - j) / 2.0, dtype=torch.float64))
    return result


# ── Batched Gaussian log-density ──────────────────────────────────────────────

def mvn_logpdf_batch(
    X:           torch.Tensor,   # (P, d)
    centers:     torch.Tensor,   # (N, d)
    covariances: torch.Tensor,   # (N, d, d)
) -> torch.Tensor:               # (P, N)
    """
    Log p(x_i | c_k, Sigma_k) for all i, k simultaneously.
    Uses torch.distributions.MultivariateNormal internally for
    stable Cholesky decomposition and log-determinant computation.
    """
    P, d = X.shape
    N    = centers.shape[0]
    out  = torch.zeros(P, N, dtype=torch.float64)

    for k in range(N):
        dist      = MultivariateNormal(
            loc=centers[k],
            covariance_matrix=covariances[k]
        )
        out[:, k] = dist.log_prob(X)

    return out


# ── Residual log-density — novel likelihood factor ────────────────────────────

def residual_logpdf_batch(
    X:         torch.Tensor,   # (P, d)
    F:         torch.Tensor,   # (P, d)
    centers:   torch.Tensor,   # (N, d)
    f_centers: torch.Tensor,   # (N, d)  f(c_k) precomputed
    jacobians: torch.Tensor,   # (N, d, d)
    sigma2:    float,
) -> torch.Tensor:             # (P, N)
    """
    Log p(eps_k(x_i) | 0, sigma2*I) for all i, k.

    eps_k(x_i) = f(x_i) - f(c_k) - J_k @ (x_i - c_k)

    This is the novel likelihood factor that makes cluster assignments
    sensitive to linearization quality, not just geometric proximity.
    """
    P, d = X.shape
    N    = centers.shape[0]
    out  = torch.zeros(P, N, dtype=torch.float64)

    noise_dist = MultivariateNormal(
        loc=torch.zeros(d, dtype=torch.float64),
        covariance_matrix=sigma2 * torch.eye(d, dtype=torch.float64)
    )

    for k in range(N):
        delta       = X - centers[k]                             # (P, d)
        linear_pred = f_centers[k] + (jacobians[k] @ delta.T).T # (P, d)
        eps         = F - linear_pred                            # (P, d)
        out[:, k]   = noise_dist.log_prob(eps)                   # (P,)

    return out


# ── Dirichlet log-density ─────────────────────────────────────────────────────

def dirichlet_logpdf(pi: torch.Tensor, alpha0: float) -> torch.Tensor:
    """
    Log p(pi) under Dir(alpha0 * 1_N).
    pi: (N,) — mixing weights
    returns: scalar
    """
    N    = pi.shape[0]
    dist = Dirichlet(alpha0 * torch.ones(N, dtype=torch.float64))
    return dist.log_prob(pi)


# ── NIW log-density ───────────────────────────────────────────────────────────

def niw_logpdf(
    c:      torch.Tensor,   # (d,)
    Sigma:  torch.Tensor,   # (d, d)
    mu0:    torch.Tensor,   # (d,)
    kappa0: float,
    Psi0:   torch.Tensor,   # (d, d)
    nu0:    float,
) -> torch.Tensor:           # scalar
    """
    Log p(c, Sigma) under NIW(mu0, kappa0, Psi0, nu0).
    Factorizes as log p(c | Sigma) + log p(Sigma).
    """
    d = c.shape[0]

    # log p(c | Sigma) = log N(mu0, Sigma / kappa0)
    c_dist = MultivariateNormal(
        loc=mu0,
        covariance_matrix=Sigma / kappa0
    )
    log_pc = c_dist.log_prob(c)

    # log p(Sigma) = Inverse-Wishart log density
    _, log_det_Psi0  = torch.linalg.slogdet(Psi0)
    _, log_det_Sigma = torch.linalg.slogdet(Sigma)
    Sigma_inv        = torch.linalg.inv(Sigma)
    log_gamma_d      = _mvlgamma(nu0 / 2.0, d)

    log_pSigma = (
          (nu0 / 2.0) * log_det_Psi0
        - (nu0 * d / 2.0) * torch.log(torch.tensor(2.0, dtype=torch.float64))
        - log_gamma_d
        - ((nu0 + d + 1) / 2.0) * log_det_Sigma
        - 0.5 * torch.trace(Psi0 @ Sigma_inv)
    )

    return log_pc + log_pSigma


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_mvn_logpdf() -> bool:
    """
    Compare mvn_logpdf_batch against scipy for known inputs.
    """
    from scipy.stats import multivariate_normal
    import numpy as np

    d   = 3
    P   = 100
    N   = 4
    rng = np.random.default_rng(0)

    X_np  = rng.standard_normal((P, d))
    c_np  = rng.standard_normal((N, d))
    A     = rng.standard_normal((N, d, d))
    S_np  = np.array([a @ a.T + np.eye(d) for a in A])

    X   = torch.tensor(X_np, dtype=torch.float64)
    c   = torch.tensor(c_np, dtype=torch.float64)
    Sig = torch.tensor(S_np, dtype=torch.float64)

    ours   = mvn_logpdf_batch(X, c, Sig).numpy()
    scipy_ = np.array([
        multivariate_normal(mean=c_np[k], cov=S_np[k]).logpdf(X_np)
        for k in range(N)
    ]).T

    max_err = float(np.max(np.abs(ours - scipy_)))
    passed  = max_err < 1e-10
    status  = "PASSED" if passed else "FAILED"
    print(f"[{status}] test_mvn_logpdf — max error vs scipy: {max_err:.2e}  (threshold: 1e-10)")
    return passed


def test_residual_logpdf_zero() -> bool:
    """
    When residual is exactly zero, log-density should equal
    log N(0; 0, sigma2*I) = -d/2 * log(2*pi*sigma2).
    """
    import numpy as np

    d      = 3
    P      = 50
    N      = 2
    sigma2 = 1.0
    rng    = np.random.default_rng(1)

    centers   = torch.tensor(rng.standard_normal((N, d)), dtype=torch.float64)
    jacobians = torch.tensor(rng.standard_normal((N, d, d)), dtype=torch.float64)
    X         = torch.tensor(rng.standard_normal((P, d)), dtype=torch.float64)
    f_centers = torch.tensor(rng.standard_normal((N, d)), dtype=torch.float64)

    # Construct F so residual = 0 exactly for cluster 0
    F = f_centers[0] + (jacobians[0] @ (X - centers[0]).T).T

    out = residual_logpdf_batch(X, F, centers, f_centers, jacobians, sigma2)

    expected = MultivariateNormal(
        torch.zeros(d, dtype=torch.float64),
        sigma2 * torch.eye(d, dtype=torch.float64)
    ).log_prob(torch.zeros(d, dtype=torch.float64)).item()

    max_err = (out[:, 0] - expected).abs().max().item()
    passed  = max_err < 1e-10
    status  = "PASSED" if passed else "FAILED"
    print(f"[{status}] test_residual_logpdf_zero — max error: {max_err:.2e}  (threshold: 1e-10)")
    return passed


def test_responsibilities_sum_to_one() -> bool:
    """
    Responsibilities from E-step must sum to 1 across clusters for each point.
    """
    import numpy as np

    d      = 3
    P      = 200
    N      = 4
    sigma2 = 5.0
    rng    = np.random.default_rng(2)

    centers   = torch.tensor(rng.standard_normal((N, d)) * 10, dtype=torch.float64)
    A         = rng.standard_normal((N, d, d))
    covs      = torch.tensor(np.array([a @ a.T + np.eye(d) for a in A]), dtype=torch.float64)
    f_centers = torch.tensor(rng.standard_normal((N, d)), dtype=torch.float64)
    jacobians = torch.tensor(rng.standard_normal((N, d, d)), dtype=torch.float64)
    pi        = torch.ones(N, dtype=torch.float64) / N
    X         = torch.tensor(rng.standard_normal((P, d)) * 5, dtype=torch.float64)
    F         = torch.tensor(rng.standard_normal((P, d)), dtype=torch.float64)

    log_prox  = mvn_logpdf_batch(X, centers, covs)
    log_resid = residual_logpdf_batch(X, F, centers, f_centers, jacobians, sigma2)
    log_pi    = torch.log(pi).unsqueeze(0)
    log_r_un  = log_pi + log_prox + log_resid
    log_r     = log_r_un - torch.logsumexp(log_r_un, dim=1, keepdim=True)
    r         = torch.exp(log_r)

    row_sums  = r.sum(dim=1)
    max_err   = (row_sums - 1.0).abs().max().item()
    passed    = max_err < 1e-12
    status    = "PASSED" if passed else "FAILED"
    print(f"[{status}] test_responsibilities_sum_to_one — max deviation: {max_err:.2e}  (threshold: 1e-12)")
    return passed


if __name__ == "__main__":
    test_mvn_logpdf()
    test_residual_logpdf_zero()
    test_responsibilities_sum_to_one()
