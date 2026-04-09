"""
Models sub-package for residual-aware Bayesian clustering.

Contains all EM fitting variants, probability distributions, the ELBO
objective, marginal likelihood for model selection, observable
transformations, and the global EDMD baseline.

Modules
-------
- ``em`` -- Taylor-analytic EM (analytic Jacobian, zero free local
  parameters).  The purest variant; requires closed-form f and J.
- ``em_hybrid`` -- Taylor-LS EM (least-squares refit of J_k, f_k per
  cluster each M-step).
- ``em_local_edmd`` -- Local continuous-time EDMD per cluster.
- ``em_local_edmd_discrete`` -- Local discrete-time EDMD per cluster.
- ``em_local_edmd_discrete_gpu`` -- GPU-accelerated discrete EDMD.
- ``global_edmd`` -- Global (single-cluster) continuous EDMD baseline.
- ``distributions`` -- Log-density computations (MVN, residual
  Gaussian, Dirichlet, NIW).
- ``elbo`` -- Evidence Lower Bound computation with residual term.
- ``marginal_likelihood`` -- NIW marginal likelihood and BIC for
  model selection across cluster counts.
- ``observables`` -- Time-delay embedding (block Hankel) for
  state-space augmentation.
"""
