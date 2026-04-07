"""
Model variants for residual-aware Bayesian clustering.

- ``em``: Taylor-analytic (analytic Jacobian, zero free local params)
- ``em_hybrid``: Taylor-LS (LS-refit J_k, f_k per cluster)
- ``em_local_edmd``: Local continuous-time EDMD per cluster
- ``em_local_edmd_discrete``: Local discrete-time EDMD per cluster
- ``global_edmd``: Global continuous-time EDMD (baseline)
- ``distributions``: Log-density computations
- ``elbo``: Evidence Lower Bound
- ``marginal_likelihood``: NIW marginal likelihood for model selection
"""
