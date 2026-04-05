# Residual-Aware Bayesian Linearization on the Lorenz Attractor

Implementation of a novel Bayesian mixture model for partitioning
a dynamical system's phase space into linearization regions.

## What this implements

A Gaussian Mixture Model augmented with a **linearization residual
likelihood factor**:

    p(x_i, f(x_i) | z_i=k) = N(x_i; c_k, Sigma_k)        [proximity]
                             * N(eps_k(x_i); 0, sigma2*I)  [novel]

where eps_k(x_i) = f(x_i) - f(c_k) - J_k @ (x_i - c_k) is the
Taylor remainder. This makes cluster assignments sensitive to
linearization quality, not just geometric proximity.

## Files

    lorenz.py              System definition, data generation, Jacobian test
    distributions.py       Stable log-densities via torch.distributions
    marginal_likelihood.py Exact NIW marginal likelihood (Rung 4 derivation)
    elbo.py                ELBO computation for convergence monitoring
    em.py                  Full EM loop with residual-corrected M-step
    viz.py                 All visualization
    run.py                 Main experiment entry point

## Setup

    pip install -r requirements.txt

## Run

    python run.py

## Expected output

    All tests passed.
    ...
    Mean linearization error — GMM:   [value]
    Mean linearization error — Ours:  [value]  (should be lower)
    Improvement:                       [X]%

    Best N by log marginal likelihood: 2 or 3

## Plots generated

    elbo_gmm.png              ELBO convergence — baseline
    elbo_ours.png             ELBO convergence — our method
    clusters_gmm.png          3D attractor, GMM cluster coloring
    clusters_ours.png         3D attractor, our method cluster coloring
    comparison.png            Side-by-side center locations
    residuals_per_cluster.png Residual vs distance for each cluster
    responsibilities.png      Soft assignment heatmap
    model_selection.png       ELBO / BIC / log ML vs N

## Key sanity checks

1. ELBO must be monotonically non-decreasing (checked automatically)
2. test_jacobian error < 1e-7
3. test_mvn_logpdf error < 1e-10
4. test_residual_logpdf_zero error < 1e-10
5. Responsibilities sum to 1 across clusters for each point
6. Our method should show lower mean linearization error than GMM
7. Model selection should peak at N=2 or N=3 for Lorenz
