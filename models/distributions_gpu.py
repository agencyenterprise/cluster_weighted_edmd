"""
Probability distributions for the residual-aware EM pipeline (GPU/MPS).

Drop-in replacement for ``distributions.py`` that works on any device
(CPU, CUDA, MPS). Every function propagates ``device`` and ``dtype`` from
the input tensors, so no manual device management is needed.

Functions
---------
``mvn_logpdf_batch(X, centers, covariances)``
    Batched multivariate normal log-pdf. Same semantics as the CPU version.

``residual_logpdf_batch(X, F, centers, f_centers, jacobians, sigma2)``
    Residual (Taylor remainder) log-pdf for the oracle EM (``em.py``).

``dirichlet_logpdf(pi, alpha0)``
    Dirichlet prior log-density on mixing weights.

``niw_logpdf(c, Sigma, mu0, kappa0, Psi0, nu0)``
    Normal-Inverse-Wishart prior log-density on (center, covariance).

Usage
-----
Used internally by ``em_local_edmd_discrete_gpu.py``. Import path mirrors
the CPU module::

    from residual_aware_clustering.models.distributions_gpu import (
        mvn_logpdf_batch, dirichlet_logpdf, niw_logpdf,
    )

    # All inputs are GPU tensors -- outputs stay on the same device
    log_prox = mvn_logpdf_batch(X_cuda, centers_cuda, covariances_cuda)

Key difference from CPU version
--------------------------------
Tensor creation (zeros, full, lgamma constants) uses ``device=`` and
``dtype=`` drawn from the input arguments, avoiding implicit CPU fallbacks
that would trigger costly device transfers on CUDA/MPS.

See ``distributions.py`` for full docstrings and mathematical details on
each distribution.
"""

import torch
from torch.distributions import MultivariateNormal, Dirichlet


def _mvlgamma(a: float, d: int, device=None, dtype=torch.float64):
    """
    Log of multivariate gamma function (device-aware).

    Parameters
    ----------
    a : float
        Argument of the multivariate gamma.
    d : int
        Dimension.
    device : torch.device or None, optional
        Target device for tensor creation.
    dtype : torch.dtype, optional
        Data type, by default ``torch.float64``.

    Returns
    -------
    torch.Tensor
        Scalar log Gamma_d(a).
    """
    result = (d * (d - 1) / 4.0) * torch.log(
        torch.tensor(torch.pi, dtype=dtype, device=device))
    for j in range(1, d + 1):
        result = result + torch.lgamma(
            torch.tensor(a + (1.0 - j) / 2.0, dtype=dtype, device=device))
    return result


def mvn_logpdf_batch(X, centers, covariances):
    """
    Batched multivariate normal log-pdf (device-aware).

    Parameters
    ----------
    X : torch.Tensor
        Data points, shape ``(P, d)``.
    centers : torch.Tensor
        Cluster centers, shape ``(N, d)``.
    covariances : torch.Tensor
        Cluster covariance matrices, shape ``(N, d, d)``.

    Returns
    -------
    torch.Tensor
        Log-pdf values, shape ``(P, N)``.
    """
    P = X.shape[0]
    N = centers.shape[0]
    out = torch.zeros(P, N, dtype=X.dtype, device=X.device)
    for k in range(N):
        dist = MultivariateNormal(loc=centers[k], covariance_matrix=covariances[k])
        out[:, k] = dist.log_prob(X)
    return out


def residual_logpdf_batch(X, F, centers, f_centers, jacobians, sigma2):
    """
    Residual (Taylor remainder) log-pdf for all points and clusters (device-aware).

    Parameters
    ----------
    X : torch.Tensor
        State points, shape ``(P, d)``.
    F : torch.Tensor
        Vector field evaluations, shape ``(P, d)``.
    centers : torch.Tensor
        Cluster centers, shape ``(N, d)``.
    f_centers : torch.Tensor
        Vector field at centers, shape ``(N, d)``.
    jacobians : torch.Tensor
        Jacobians at centers, shape ``(N, d, d)``.
    sigma2 : float or torch.Tensor
        Per-cluster residual variance. Scalar or tensor of shape ``(N,)``.

    Returns
    -------
    torch.Tensor
        Log-pdf values, shape ``(P, N)``.
    """
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
    """
    Log-density of the Dirichlet prior on mixing weights (device-aware).

    Parameters
    ----------
    pi : torch.Tensor
        Mixing weights, shape ``(N,)``.
    alpha0 : float
        Concentration parameter (symmetric Dirichlet).

    Returns
    -------
    torch.Tensor
        Scalar log-density.
    """
    N = pi.shape[0]
    dist = Dirichlet(alpha0 * torch.ones(N, dtype=pi.dtype, device=pi.device))
    return dist.log_prob(pi)


def niw_logpdf(c, Sigma, mu0, kappa0, Psi0, nu0):
    """
    Log-density of the Normal-Inverse-Wishart prior (device-aware).

    Parameters
    ----------
    c : torch.Tensor
        Cluster center, shape ``(d,)``.
    Sigma : torch.Tensor
        Covariance matrix, shape ``(d, d)``.
    mu0 : torch.Tensor
        Prior mean, shape ``(d,)``.
    kappa0 : float
        Prior precision scaling.
    Psi0 : torch.Tensor
        Prior scale matrix, shape ``(d, d)``.
    nu0 : float
        Prior degrees of freedom.

    Returns
    -------
    torch.Tensor
        Scalar log-density.
    """
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
