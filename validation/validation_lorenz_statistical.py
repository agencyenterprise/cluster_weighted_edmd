"""
Lorenz validation with statistical rigor.

Runs each method across N_SEEDS random seeds, reports:
  - Mean ± 95% CI for one-step error and rollout error
  - Paired t-tests between key method pairs with p-values
  - Saves full per-seed data for reproducibility

Seeds vary: (i) trajectory data generation, (ii) EM initialization.
"""

import numpy as np
import torch
import json
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from utils.paths import fig_path, data_path
from utils.stats import confidence_interval, paired_test

from simulators.lorenz import generate_data, f as lorenz_f, J as lorenz_J
from models.em import fit as fit_taylor
from models.distributions import mvn_logpdf_batch

import pykoopman as pk
from models.global_edmd import fit as fit_global_ours, predict_f as predict_f_global_ours

torch.set_default_dtype(torch.float64)

import argparse

parser = argparse.ArgumentParser(description="Lorenz statistical validation")
parser.add_argument('--seeds', type=int, nargs='+', required=True,
                    help="List of random seeds to run")
args = parser.parse_args()
seeds = args.seeds
N_SEEDS = len(seeds)
d = 3
dt_lorenz = 0.01

# ── Helpers ──────────────────────────────────────────────────────────────────

def pick_cluster(x, state):
    log_pi   = torch.log(state['pi']).unsqueeze(0)
    log_prox = mvn_logpdf_batch(x, state['centers'], state['covariances'])
    return (log_pi + log_prox).argmax(dim=1)

def piecewise_f(x, state):
    k  = pick_cluster(x, state)
    c  = state['centers'][k]
    fc = state['f_centers'][k]
    J  = state['jacobians'][k]
    return fc + (J @ (x - c).unsqueeze(-1)).squeeze(-1)

def one_step_err(state, X_te_in, X_te_next):
    pred = X_te_in + dt_lorenz * piecewise_f(X_te_in, state)
    return torch.linalg.norm(pred - X_te_next, dim=1).mean().item()

def rollout_truth(x0, n_steps):
    sol = solve_ivp(lambda t, y: lorenz_f(y),
                    (0.0, n_steps * dt_lorenz), x0.numpy(),
                    t_eval=np.linspace(0.0, n_steps * dt_lorenz, n_steps + 1),
                    method='RK45', rtol=1e-10, atol=1e-10)
    return torch.tensor(sol.y.T, dtype=torch.float64)

def rollout_err_at(state, inits, step=500):
    errs = []
    for x0 in inits:
        tru = rollout_truth(x0, step)
        traj = torch.zeros(step + 1, d, dtype=torch.float64)
        traj[0] = x0
        for t in range(step):
            traj[t+1] = traj[t] + dt_lorenz * piecewise_f(traj[t:t+1], state)[0]
        errs.append(torch.linalg.norm(traj[step] - tru[step]).item())
    return float(np.mean(errs))


# ── Configs ──────────────────────────────────────────────────────────────────

N_values = [5, 12, 20, 50]

# ── Single-seed run ──────────────────────────────────────────────────────────

def run_seed(seed):
    data = generate_data(n_steps=5000, dt=0.01, warmup=1000, seed=seed)
    X_all = torch.tensor(data['X'], dtype=torch.float64)
    F_all = torch.tensor(data['F'], dtype=torch.float64)

    X_tr, X_te = X_all[:4000], X_all[4000:]
    F_tr, F_te = F_all[:4000], F_all[4000:]
    X_te_in  = X_all[4000:4999]
    X_te_next = X_all[4001:5000]
    step_baseline = torch.linalg.norm(X_te_next - X_te_in, dim=1).mean().item()

    inits = [X_te[i] for i in (0, 300, 700)]

    results = {}

    # EDMD baselines (pykoopman)
    edmd2 = pk.Koopman(observables=pk.observables.Polynomial(degree=2, include_bias=True),
                       regressor=pk.regression.EDMD())
    edmd2.fit(X_tr.numpy(), dt=dt_lorenz)
    edmd2_one = torch.linalg.norm(
        torch.tensor(edmd2.predict(X_te_in.numpy())) - X_te_next, dim=1).mean().item()
    results['EDMD-pk deg-2'] = {'one_step': edmd2_one, 'rel_pct': 100 * edmd2_one / step_baseline}

    edmd3 = pk.Koopman(observables=pk.observables.Polynomial(degree=3, include_bias=True),
                       regressor=pk.regression.EDMD())
    edmd3.fit(X_tr.numpy(), dt=dt_lorenz)
    edmd3_one = torch.linalg.norm(
        torch.tensor(edmd3.predict(X_te_in.numpy())) - X_te_next, dim=1).mean().item()
    results['EDMD-pk deg-3'] = {'one_step': edmd3_one, 'rel_pct': 100 * edmd3_one / step_baseline}

    # EDMD baselines (ours)
    edmd_ours2 = fit_global_ours(X_tr, F_tr, degree=2, ridge=1e-4)
    edmd_ours3 = fit_global_ours(X_tr, F_tr, degree=3, ridge=1e-4)

    f_hat2 = predict_f_global_ours(X_te_in, edmd_ours2)
    edmd_ours2_one = torch.linalg.norm(
        X_te_in + dt_lorenz * f_hat2 - X_te_next, dim=1).mean().item()
    results['EDMD-ours deg-2'] = {'one_step': edmd_ours2_one, 'rel_pct': 100 * edmd_ours2_one / step_baseline}

    f_hat3 = predict_f_global_ours(X_te_in, edmd_ours3)
    edmd_ours3_one = torch.linalg.norm(
        X_te_in + dt_lorenz * f_hat3 - X_te_next, dim=1).mean().item()
    results['EDMD-ours deg-3'] = {'one_step': edmd_ours3_one, 'rel_pct': 100 * edmd_ours3_one / step_baseline}

    hp_base = {
        'alpha0': 0.5, 'mu0': X_tr.mean(dim=0),
        'Lambda0': 0.01 * torch.eye(d, dtype=torch.float64),
        'kappa0': 1.0, 'Psi0': 10.0 * torch.eye(d, dtype=torch.float64),
        'nu0': float(d + 2), 'sigma2': 'auto',
    }

    for N in N_values:
        # Ours (residual-aware)
        hp = dict(hp_base); hp['sigma2'] = 'auto'
        state_o, _, _ = fit_taylor(X_tr, F_tr, lorenz_f, lorenz_J,
                                   N=N, hp=hp, n_iter=100, n_restarts=2, verbose=False)
        one_o = one_step_err(state_o, X_te_in, X_te_next)
        r500_o = rollout_err_at(state_o, inits, step=500)

        # GMM baseline
        hp_g = dict(hp_base); hp_g['sigma2'] = 1e10
        state_g, _, _ = fit_taylor(X_tr, F_tr, lorenz_f, lorenz_J,
                                   N=N, hp=hp_g, n_iter=100, n_restarts=2, verbose=False)
        one_g = one_step_err(state_g, X_te_in, X_te_next)
        r500_g = rollout_err_at(state_g, inits, step=500)

        results[f'Ours N={N}'] = {
            'one_step': one_o, 'rel_pct': 100 * one_o / step_baseline,
            'rollout_500': r500_o,
        }
        results[f'GMM N={N}'] = {
            'one_step': one_g, 'rel_pct': 100 * one_g / step_baseline,
            'rollout_500': r500_g,
        }

    return results


# ── Run all seeds ────────────────────────────────────────────────────────────

# seeds already defined from env at top of file
all_runs = {}

for i, seed in enumerate(seeds):
    print(f"\n{'='*60}")
    print(f"SEED {seed} ({i+1}/{N_SEEDS})")
    print(f"{'='*60}")
    results = run_seed(seed)
    for name, metrics in results.items():
        if name not in all_runs:
            all_runs[name] = {m: [] for m in metrics}
        for m, v in metrics.items():
            all_runs[name][m].append(v)

# ── Report ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 100)
print(f"LORENZ — {N_SEEDS} seeds, mean ± 95% CI")
print("=" * 100)
print(f"{'method':<20} {'one-step':>22s} {'rel%':>18s} {'rollout@500':>22s}")
print("-" * 100)

for name in ['EDMD-pk deg-2', 'EDMD-pk deg-3',
             'EDMD-ours deg-2', 'EDMD-ours deg-3'] + \
            [f'Ours N={N}' for N in N_values] + \
            [f'GMM N={N}' for N in N_values]:
    m_one, hw_one, _, _ = confidence_interval(all_runs[name]['one_step'])
    m_rel, hw_rel, _, _ = confidence_interval(all_runs[name]['rel_pct'])
    if 'rollout_500' in all_runs[name]:
        m_r, hw_r, _, _ = confidence_interval(all_runs[name]['rollout_500'])
        r_str = f"{m_r:>10.3f} +/- {hw_r:.3f}"
    else:
        r_str = f"{'—':>22s}"
    print(f"{name:<20} {m_one:>10.5f} +/- {hw_one:.5f}   {m_rel:>7.2f} +/- {hw_rel:.2f}%   {r_str}")

# ── Paired tests ─────────────────────────────────────────────────────────────

print("\n" + "=" * 100)
print("PAIRED T-TESTS (p-values)")
print("=" * 100)

for N in N_values:
    a = np.array(all_runs[f'Ours N={N}']['one_step'])
    b = np.array(all_runs[f'GMM N={N}']['one_step'])
    t = paired_test(a, b, alternative='less')
    sig = "***" if t['p_value'] < 0.001 else "**" if t['p_value'] < 0.01 else "*" if t['p_value'] < 0.05 else "ns"
    pct_better = 100 * (b.mean() - a.mean()) / b.mean()
    print(f"  Ours vs GMM N={N:>2}: ours {a.mean():.5f} vs gmm {b.mean():.5f} "
          f"({pct_better:+.1f}% better), p={t['p_value']:.4f} {sig}")

print()
for N in N_values:
    a_one = np.array(all_runs[f'Ours N={N}']['one_step'])
    b_one = np.array(all_runs['EDMD-pk deg-2']['one_step'])
    t = paired_test(a_one, b_one)
    print(f"  Ours N={N:>2} vs EDMD-pk deg-2: diff={t['mean_diff']:+.5f}, p={t['p_value']:.4f}")

print()
for N in N_values:
    a_one = np.array(all_runs[f'Ours N={N}']['one_step'])
    b_one = np.array(all_runs['EDMD-ours deg-2']['one_step'])
    t = paired_test(a_one, b_one)
    print(f"  Ours N={N:>2} vs EDMD-ours deg-2: diff={t['mean_diff']:+.5f}, p={t['p_value']:.4f}")

# ── Save ─────────────────────────────────────────────────────────────────────

raw = {
    'n_seeds': N_SEEDS, 'seeds': seeds,
    'results': {name: {m: vals for m, vals in metrics.items()}
                for name, metrics in all_runs.items()},
}
with open(data_path("lorenz_statistical.json"), "w") as fp:
    json.dump(raw, fp, indent=2)
print(f"\nRaw data saved to {data_path('lorenz_statistical.json')}")

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

ours_means = [confidence_interval(all_runs[f'Ours N={N}']['rel_pct'])[0] for N in N_values]
ours_cis   = [confidence_interval(all_runs[f'Ours N={N}']['rel_pct'])[1] for N in N_values]
gmm_means  = [confidence_interval(all_runs[f'GMM N={N}']['rel_pct'])[0] for N in N_values]
gmm_cis    = [confidence_interval(all_runs[f'GMM N={N}']['rel_pct'])[1] for N in N_values]

x = np.arange(len(N_values))
w = 0.35
ax.bar(x - w/2, ours_means, w, yerr=ours_cis, label='Ours', capsize=4, color='C0', alpha=0.8)
ax.bar(x + w/2, gmm_means,  w, yerr=gmm_cis,  label='GMM',  capsize=4, color='C1', alpha=0.8)

edmd2_pk_m = confidence_interval(all_runs['EDMD-pk deg-2']['rel_pct'])[0]
ax.axhline(edmd2_pk_m, ls='--', color='C3', label=f'EDMD-pk deg-2 ({edmd2_pk_m:.1f}%)')
edmd2_ours_m = confidence_interval(all_runs['EDMD-ours deg-2']['rel_pct'])[0]
ax.axhline(edmd2_ours_m, ls=':', color='C3', label=f'EDMD-ours deg-2 ({edmd2_ours_m:.1f}%)')

ax.set_xticks(x)
ax.set_xticklabels([f'N={N}' for N in N_values])
ax.set_ylabel('Relative one-step error (%)')
ax.set_title(f'Lorenz: Ours vs GMM (mean ± 95% CI, n={N_SEEDS})')
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(fig_path("lorenz_statistical.png"), dpi=120)
print(f"Plot saved to {fig_path('lorenz_statistical.png')}")
