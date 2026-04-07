"""
Residual-Aware Bayesian Clustering for Local Dynamical Models.

A Cluster-Weighted Model framework for partitioning phase space into
regions where local dynamical models (Taylor, LS-fit, or EDMD) accurately
predict the vector field.

Quick start::

    from residual_aware_clustering import fit_taylor, make_hp
    from residual_aware_clustering.simulators.lorenz import generate_data, f, J
    import torch

    data = generate_data(n_steps=5000, dt=0.01, warmup=1000)
    X = torch.tensor(data['X'], dtype=torch.float64)
    F = torch.tensor(data['F'], dtype=torch.float64)

    hp = make_hp(X, d=3)
    state, responsibilities, elbo_history = fit_taylor(X, F, f, J, N=5, hp=hp)

    # state['centers']     — cluster centers (N, d)
    # state['covariances'] — cluster covariances (N, d, d)
    # state['f_centers']   — f(c_k) at each center (N, d)
    # state['jacobians']   — J(c_k) at each center (N, d, d)
    # state['sigma2']      — per-cluster residual variance (N,)
    # state['pi']          — mixing weights (N,)

Available fitting functions:

- ``fit_taylor``: Taylor-analytic (requires analytic f, J)
- ``fit_hybrid``: Taylor-LS (LS-refit J_k, f_k per cluster)
- ``fit_local_edmd``: Local continuous EDMD per cluster
- ``fit_local_edmd_discrete``: Local discrete EDMD per cluster
- ``fit_global_edmd_discrete``: Global discrete EDMD (fair baseline)
"""

# Auto-patch pykoopman if installed (sklearn compat fix)
def _patch_pykoopman():
    try:
        import pykoopman.observables._polynomial as _poly
        import numpy as _np
        obs = _poly.Polynomial(degree=2, include_bias=True)
        obs.fit(_np.ones((3, 2)))
        obs.transform(_np.ones((1, 2)))
    except ImportError:
        pass  # pykoopman not installed, nothing to patch
    except Exception:
        try:
            import pykoopman
            import os
            path = os.path.join(os.path.dirname(pykoopman.__file__),
                                "observables", "_polynomial.py")
            with open(path, "r") as f:
                src = f.read()
            marker = "# PATCH: sklearn compat"
            if marker not in src:
                target = "y_poly_out = super(Polynomial, self).fit(x.real, y)\n"
                line = f"        self.n_input_features_ = self.n_features_in_  {marker}\n"
                src = src.replace(target, target + line)
                with open(path, "w") as f:
                    f.write(src)
                # Reload
                import importlib
                importlib.reload(_poly)
        except Exception:
            pass  # best-effort, fail silently

_patch_pykoopman()
del _patch_pykoopman

from .models.em import fit as fit_taylor
from .models.em_hybrid import fit_hybrid
from .models.em_local_edmd import fit as fit_local_edmd
from .models.em_local_edmd_discrete import (
    fit as fit_local_edmd_discrete,
    fit_global as fit_global_edmd_discrete,
    predict_next_global as predict_global_edmd_discrete,
)
from .models.distributions import mvn_logpdf_batch, residual_logpdf_batch
from .models.elbo import compute_elbo
from .models.marginal_likelihood import total_log_marginal

import torch


def make_hp(X, d, alpha0=0.5, kappa0=1.0, psi0_scale=10.0, nu0=None, lambda0_scale=0.01):
    """
    Create default hyperparameter dict for EM fitting.

    Args:
        X: (P, d) training data tensor (used to compute prior mean)
        d: state dimension
        alpha0: Dirichlet concentration (< 1 = sparse, encourages pruning)
        kappa0: prior precision scaling for center
        psi0_scale: scale for Psi0 = psi0_scale * I
        nu0: degrees of freedom (default: d + 2)
        lambda0_scale: scale for Lambda0 = lambda0_scale * I

    Returns:
        dict with keys: alpha0, mu0, Lambda0, kappa0, Psi0, nu0, sigma2
    """
    if nu0 is None:
        nu0 = float(d + 2)
    return {
        'alpha0': alpha0,
        'mu0': X.mean(dim=0),
        'Lambda0': lambda0_scale * torch.eye(d, dtype=torch.float64),
        'kappa0': kappa0,
        'Psi0': psi0_scale * torch.eye(d, dtype=torch.float64),
        'nu0': nu0,
        'sigma2': 'auto',
    }


__version__ = "0.1.0"

__all__ = [
    "fit_taylor",
    "fit_hybrid",
    "fit_local_edmd",
    "fit_local_edmd_discrete",
    "fit_global_edmd_discrete",
    "predict_global_edmd_discrete",
    "mvn_logpdf_batch",
    "residual_logpdf_batch",
    "compute_elbo",
    "total_log_marginal",
    "make_hp",
]
