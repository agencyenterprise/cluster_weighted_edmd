# Residual-Aware Bayesian Clustering for Local Dynamical Models

A unified framework for partitioning a dynamical system's phase space
into regions within which a simple local model accurately predicts the
vector field. Clusters form by joint likelihood of position AND
linearization quality, not geometry alone.

## Three local-model variants

| Variant | Module | Local model | Needs analytic J? |
|---|---|---|---|
| Taylor-analytic | `models/em.py` | `f(c_k) + J(c_k)(x - c_k)` | Yes |
| Taylor-LS | `models/em_hybrid.py` | `f_k + J_k(x - c_k)` (LS-fit) | No |
| Local EDMD | `models/em_local_edmd.py` | `[M_k Phi(x - c_k)]_{1:d}` | No |

All share the same E-step; they differ only in the M-step.

## Project structure

    residual_aware_clustering/
    ├── simulators/                          Dynamical system definitions
    │   ├── lorenz.py                          Lorenz attractor (d=3, polynomial)
    │   └── pendulum.py                        Damped pendulum (d=2, non-polynomial)
    │
    ├── models/                              EM framework and shared components
    │   ├── em.py                              Taylor-analytic EM
    │   ├── em_hybrid.py                       Taylor-LS EM (LS-refit J_k, f_k)
    │   ├── em_local_edmd.py                   Local EDMD EM (Koopman per cluster)
    │   ├── distributions.py                   Stable log-densities
    │   ├── elbo.py                            ELBO computation + monotonicity
    │   └── marginal_likelihood.py             Exact NIW marginal likelihood
    │
    ├── validation/                          Experiments and comparisons
    │   ├── validation_lorenz.py               Lorenz: full experiment (sanity + fit + plots)
    │   ├── validation_lorenz_vs_edmd.py       Lorenz: one-step + rollout vs global EDMD
    │   ├── validation_lorenz_sweep_N.py       Lorenz: sweep N for all methods
    │   ├── validation_lorenz_hybrid.py        Lorenz: Taylor-analytic vs Taylor-LS
    │   ├── validation_lorenz_local_edmd.py    Lorenz: local EDMD vs global EDMD
    │   └── validation_pendulum.py             Pendulum: all four families compared
    │
    ├── utils/                               Shared utilities
    │   ├── viz.py                             Visualization / plotting
    │   └── paths.py                           Output paths (figures, data)
    │
    ├── papers/                              Manuscripts and outputs
    │   ├── paper.tex                          Main paper
    │   ├── derivations.tex                    Detailed derivations from first principles
    │   ├── figures/                           Generated plots (git-tracked)
    │   └── data/                              Raw experiment data JSON (git-ignored)
    │
    ├── requirements.txt
    └── README.md

## Setup

    pip install -r requirements.txt

## Quick start

    # Lorenz full experiment (sanity tests + EM fit + plots)
    python -m validation.validation_lorenz

    # Pendulum comparison — the key result
    python -m validation.validation_pendulum

    # Lorenz: sweep cluster count N
    python -m validation.validation_lorenz_sweep_N

    # Lorenz: compare Taylor-analytic vs Taylor-LS vs GMM
    python -m validation.validation_lorenz_hybrid
