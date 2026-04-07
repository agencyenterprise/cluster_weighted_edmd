"""
GPU-compatible versions of distribution functions.

Drop-in replacements for distributions.py that work on any device
(CPU, CUDA, MPS) by propagating device and dtype from input tensors.
"""

import torch
from torch.distributions import MultivariateNormal, Dirichlet


def _mvlgamma(a: float, d: int, device=None, dtype=torch.float64):
    result = (d * (d - 1) / 4.0) * torch.log(
        torch.tensor(torch.pi, dtype=dtype, device=device))
    for j in range(1, d + 1):
        result = result + torch.lgamma(
            torch.tensor(a + (1.0 - j) / 2.0, dtype=dtype, device=device))
    return result


def mvn_logpdf_batch(X, centers, covariances):
    P = X.shape[0]
    N = centers.shape[0]
    out = torch.zeros(P, N, dtype=X.dtype, device=X.device)
    for k in range(N):
        dist = MultivariateNormal(loc=centers[k], covariance_matrix=covariances[k])
        out[:, k] = dist.log_prob(X)
    return out


def residual_logpdf_batch(X, F, centers, f_centers, jacobians, sigma2):
    P, d = X.shape
    N = centers.shape[0]
    out = torch.zeros(P, N, dtype=X.dtype, device=X.device)

    if isinstance(sigma2, (int, float)):
        s2 = torch.full((N,), float(sigma2), dtype=X.dtype, device=X.device)
    else:
        s2 = sigma2

    for k in range(N):
        delta = X - centers[k]
        linear_pred = f_centers[k] + (jacobians[k] @ delta.T).T
        eps = F - linear_pred
        sq_norm = (eps ** 2).sum(dim=1)
        out[:, k] = -(d / 2.0) * torch.log(2 * torch.pi * s2[k]) \
                    - sq_norm / (2.0 * s2[k])
    return out


def dirichlet_logpdf(pi, alpha0):
    N = pi.shape[0]
    dist = Dirichlet(alpha0 * torch.ones(N, dtype=pi.dtype, device=pi.device))
    return dist.log_prob(pi)


def niw_logpdf(c, Sigma, mu0, kappa0, Psi0, nu0):
    d = c.shape[0]
    dev = c.device
    dt = c.dtype

    c_dist = MultivariateNormal(loc=mu0, covariance_matrix=Sigma / kappa0)
    log_pc = c_dist.log_prob(c)

    _, log_det_Psi0 = torch.linalg.slogdet(Psi0)
    _, log_det_Sigma = torch.linalg.slogdet(Sigma)
    Sigma_inv = torch.linalg.inv(Sigma)
    log_gamma_d = _mvlgamma(nu0 / 2.0, d, device=dev, dtype=dt)

    log_pSigma = (
          (nu0 / 2.0) * log_det_Psi0
        - (nu0 * d / 2.0) * torch.log(torch.tensor(2.0, dtype=dt, device=dev))
        - log_gamma_d
        - ((nu0 + d + 1) / 2.0) * log_det_Sigma
        - 0.5 * torch.trace(Psi0 @ Sigma_inv)
    )

    return log_pc + log_pSigma
