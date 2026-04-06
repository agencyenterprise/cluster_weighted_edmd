import argparse
import numpy as np
import torch
from utils.paths import fig_path, data_path

parser = argparse.ArgumentParser(description="Lorenz full experiment")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--n-steps', type=int, default=5000)
parser.add_argument('--dt', type=float, default=0.01)
parser.add_argument('--warmup', type=int, default=1000)
parser.add_argument('--N', type=int, default=5, help="Cluster count for main fit")
parser.add_argument('--n-iter', type=int, default=100)
parser.add_argument('--n-restarts', type=int, default=3)
parser.add_argument('--ms-range', type=int, nargs=2, default=[2, 13],
                    help="Model selection N range [start, stop)")
parser.add_argument('--ms-restarts', type=int, default=2,
                    help="Restarts for model selection sweep")
args = parser.parse_args()

from simulators.lorenz import generate_data, f, J, test_jacobian
from models.distributions import (
    test_mvn_logpdf,
    test_residual_logpdf_zero,
    test_responsibilities_sum_to_one,
)
from models.em import fit, e_step
from models.marginal_likelihood import total_log_marginal, bic
from utils.viz import (
    plot_elbo,
    plot_attractor_clusters,
    plot_comparison,
    plot_residuals_per_cluster,
    plot_model_selection,
    plot_responsibilities,
)

# ═════════════════════════════════════════════════════════════════════════════
# 1. SANITY TESTS — all must pass before proceeding
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("SANITY TESTS")
print("=" * 60)

results = [
    test_jacobian(),
    test_mvn_logpdf(),
    test_residual_logpdf_zero(),
    test_responsibilities_sum_to_one(),
]

if not all(results):
    raise RuntimeError("One or more sanity tests failed. Fix before running EM.")

print("\nAll tests passed.\n")

# ═════════════════════════════════════════════════════════════════════════════
# 2. DATA GENERATION
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("DATA GENERATION")
print("=" * 60)

data  = generate_data(n_steps=args.n_steps, dt=args.dt, warmup=args.warmup, seed=args.seed)
X     = torch.tensor(data['X'],     dtype=torch.float64)   # (5000, 3)
F_obs = torch.tensor(data['F'],     dtype=torch.float64)   # (5000, 3)
P     = X.shape[0]

print(f"X shape:     {X.shape}")
print(f"F shape:     {F_obs.shape}")
print(f"X range:     [{X.min():.2f}, {X.max():.2f}]")
print()

print("=" * 60)
print("HYPERPARAMETERS")
print("=" * 60)

d     = X.shape[1]
X_mean = X.mean(dim=0)

hp = {
    'alpha0':  0.5,
    'mu0':     X_mean,
    'Lambda0': 0.01 * torch.eye(d, dtype=torch.float64),
    'kappa0':  1.0,
    'Psi0':    10.0 * torch.eye(d, dtype=torch.float64),
    'nu0':     float(d + 2),
    'sigma2':  'auto',   # calibrated from within-cluster residuals at init
}

hp_gmm = {**hp, 'sigma2': 1e10}   # sigma2 → inf removes residual term

print("sigma2: will be calibrated per restart from within-cluster residuals")
print()

print()

# ═════════════════════════════════════════════════════════════════════════════
# 4. FIT STANDARD GMM BASELINE (sigma2 = inf)
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("BASELINE: Standard GMM (no residual term, sigma2 → inf)")
print("=" * 60)

state_gmm, r_gmm, history_gmm = fit(
    X, F_obs, f, J,
    N=args.N,
    hp=hp_gmm,
    n_iter=args.n_iter,
    n_restarts=args.n_restarts,
    verbose=True,
)

print(f"\nGMM final ELBO: {history_gmm[-1]:.4f}")
print(f"GMM active clusters: {state_gmm['N']}")
print()

# ═════════════════════════════════════════════════════════════════════════════
# 5. FIT RESIDUAL-AWARE MODEL (our method)
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("OUR METHOD: Residual-Aware EM")
print("=" * 60)

state_ours, r_ours, history_ours = fit(
    X, F_obs, f, J,
    N=args.N,
    hp=hp,
    n_iter=args.n_iter,
    n_restarts=args.n_restarts,
    verbose=True,
)

print(f"\nOurs final ELBO: {history_ours[-1]:.4f}")
print(f"Ours active clusters: {state_ours['N']}")
print()

# ═════════════════════════════════════════════════════════════════════════════
# 6. QUANTITATIVE COMPARISON
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("QUANTITATIVE COMPARISON")
print("=" * 60)


def mean_linearization_error(X, F, r, state):
    assignments = r.argmax(dim=1)
    total = 0.0
    n     = 0
    for k in range(state['N']):
        mask = assignments == k
        if mask.sum() == 0:
            continue
        X_k   = X[mask]
        F_k   = F[mask]
        delta = X_k - state['centers'][k]
        lp    = state['f_centers'][k] + (state['jacobians'][k] @ delta.T).T
        eps   = F_k - lp
        total += (eps ** 2).sum(dim=1).sum().item()
        n     += mask.sum().item()
    return total / n


err_gmm  = mean_linearization_error(X, F_obs, r_gmm,  state_gmm)
err_ours = mean_linearization_error(X, F_obs, r_ours, state_ours)
improvement = 100.0 * (err_gmm - err_ours) / err_gmm

print(f"Mean linearization error — GMM:   {err_gmm:.6f}")
print(f"Mean linearization error — Ours:  {err_ours:.6f}")
print(f"Improvement:                       {improvement:.1f}%")
print()

# Log marginal likelihoods
ml_gmm  = total_log_marginal(X, F_obs, r_gmm,  state_gmm,  hp_gmm).item()
ml_ours = total_log_marginal(X, F_obs, r_ours, state_ours, hp).item()
print(f"Log marginal likelihood — GMM:   {ml_gmm:.4f}")
print(f"Log marginal likelihood — Ours:  {ml_ours:.4f}")
print()

# ═════════════════════════════════════════════════════════════════════════════
# 7. MODEL SELECTION OVER N
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("MODEL SELECTION  (N = 2 … 12)")
print("=" * 60)

elbo_by_N = {}
bic_by_N  = {}
ml_by_N   = {}

for N in range(args.ms_range[0], args.ms_range[1]):
    print(f"\n  N = {N}")
    s, r, hist = fit(
        X, F_obs, f, J,
        N=N, hp=hp,
        n_iter=args.n_iter, n_restarts=args.ms_restarts,
        verbose=False,
    )
    if s is None:
        continue

    ml_val       = total_log_marginal(X, F_obs, r, s, hp).item()
    bic_val      = bic(ml_val, s['N'], P, d)
    elbo_val     = hist[-1]

    elbo_by_N[N] = elbo_val
    bic_by_N[N]  = bic_val
    ml_by_N[N]   = ml_val

    print(f"    ELBO={elbo_val:.2f}  log ML={ml_val:.2f}  "
          f"BIC={bic_val:.2f}  active={s['N']}")

best_N_ml   = max(ml_by_N,   key=ml_by_N.get)
best_N_bic  = max(bic_by_N,  key=bic_by_N.get)
best_N_elbo = max(elbo_by_N, key=elbo_by_N.get)
print(f"\n  Best N by log marginal likelihood: {best_N_ml}")
print(f"  Best N by BIC:                     {best_N_bic}")
print(f"  Best N by ELBO:                    {best_N_elbo}")
print(f"\n  Note: ELBO uses soft assignments (more conservative).")
print(f"  BIC/log-ML use hard assignments (biased toward larger N).")
print()

# ═════════════════════════════════════════════════════════════════════════════
# 8. SAVE RAW DATA
# ═════════════════════════════════════════════════════════════════════════════

import json

raw = {
    'data_seed': args.seed,
    'err_gmm': err_gmm, 'err_ours': err_ours, 'improvement': improvement,
    'ml_gmm': ml_gmm, 'ml_ours': ml_ours,
    'elbo_by_N': {str(k): v for k, v in elbo_by_N.items()},
    'bic_by_N': {str(k): v for k, v in bic_by_N.items()},
    'ml_by_N': {str(k): v for k, v in ml_by_N.items()},
    'best_N_ml': best_N_ml, 'best_N_bic': best_N_bic, 'best_N_elbo': best_N_elbo,
}
with open(data_path("lorenz_results.json"), "w") as fp:
    json.dump(raw, fp, indent=2)
print(f"Raw data saved to {data_path('lorenz_results.json')}")

# ═════════════════════════════════════════════════════════════════════════════
# 9. PLOTS
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("GENERATING PLOTS")
print("=" * 60)

plot_elbo(history_gmm,  title="ELBO_GMM",   save=fig_path("elbo_gmm.png"))
plot_elbo(history_ours, title="ELBO_Ours",  save=fig_path("elbo_ours.png"))

plot_attractor_clusters(X, r_gmm,  state_gmm,
                        title="Standard GMM Clusters",
                        save=fig_path("clusters_gmm.png"))
plot_attractor_clusters(X, r_ours, state_ours,
                        title="Residual-Aware Clusters",
                        save=fig_path("clusters_ours.png"))

plot_comparison(X, r_gmm, state_gmm, r_ours, state_ours,
                save=fig_path("comparison.png"))

plot_residuals_per_cluster(X, F_obs, r_ours, state_ours,
                           save=fig_path("residuals_per_cluster.png"))

plot_responsibilities(r_ours, save=fig_path("responsibilities.png"))

if elbo_by_N:
    plot_model_selection(elbo_by_N, bic_by_N, ml_by_N,
                         save=fig_path("model_selection.png"))

print("\nDone.")
