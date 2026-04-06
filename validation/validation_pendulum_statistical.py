"""
Pendulum validation with statistical rigor.

Runs each method across N_SEEDS random seeds, reports:
  - Mean ± 95% CI for all metrics
  - Paired t-tests between key method pairs with p-values
  - Saves full per-seed data for reproducibility

Seeds vary: (i) training/test data sampling, (ii) EM initialization.
"""

import numpy as np
import torch
import json
import matplotlib.pyplot as plt

from utils.paths import fig_path, data_path
from utils.stats import confidence_interval, paired_test, summarize, compare

from simulators.pendulum import (
    f as pendulum_f, J as pendulum_J,
    sample_phase_space, generate_trajectory,
    wrap_theta, angular_dist,
)
from models.em import fit as fit_taylor
from models.em_hybrid import fit_hybrid
from models.em_local_edmd import (
    fit as fit_local_edmd, predict_f_all_clusters,
    monomial_exponents, monomials, monomials_grad,
    weighted_continuous_edmd,
)
from models.distributions import mvn_logpdf_batch

torch.set_default_dtype(torch.float64)

import argparse

parser = argparse.ArgumentParser(description="Pendulum statistical validation")
parser.add_argument('--seeds', type=int, nargs='+', required=True,
                    help="List of random seeds to run")
args = parser.parse_args()
seeds = args.seeds
N_SEEDS = len(seeds)
d = 2
dt = 0.05

# ── Helpers (same as validation_pendulum.py) ─────────────────────────────────

def fit_global_edmd(X_tr, F_tr, degree, ridge=1e-6):
    exps = monomial_exponents(d, degree)
    c = X_tr.mean(dim=0)
    r_weights = torch.ones(X_tr.shape[0], dtype=torch.float64)
    M = weighted_continuous_edmd(X_tr, F_tr, r_weights, c, exps, ridge=ridge)
    return {'M': M, 'c': c, 'exps': exps}

def predict_global_edmd(X, g):
    U = X - g['c']
    Phi = monomials(U, g['exps'])
    Phi_dot = Phi @ g['M'].T
    return Phi_dot[:, 1:d+1]

def pick_cluster(x, state):
    log_pi   = torch.log(state['pi']).unsqueeze(0)
    log_prox = mvn_logpdf_batch(x, state['centers'], state['covariances'])
    return (log_pi + log_prox).argmax(dim=1)

def predict_f_local(x, state):
    k = pick_cluster(x, state)
    F_all = predict_f_all_clusters(x, state['centers'], state['M_ops'],
                                   state['exps'], d)
    return F_all[torch.arange(x.shape[0]), k]

def predict_f_taylor(x, state):
    k  = pick_cluster(x, state)
    c  = state['centers'][k]
    fc = state['f_centers'][k]
    J  = state['jacobians'][k]
    return fc + (J @ (x - c).unsqueeze(-1)).squeeze(-1)

def euler_step_wrap(x, f_val):
    x_new = x + dt * f_val
    return wrap_theta(x_new)

def rollout(x0, predict_fn, model, n_steps):
    traj = torch.zeros(n_steps + 1, d, dtype=torch.float64)
    traj[0] = x0
    for t in range(n_steps):
        f_hat = predict_fn(traj[t:t+1], model)[0]
        traj[t+1] = euler_step_wrap(traj[t], f_hat)
    return traj

def rollout_global(x0, g, n_steps):
    traj = torch.zeros(n_steps + 1, d, dtype=torch.float64)
    traj[0] = x0
    for t in range(n_steps):
        f_hat = predict_global_edmd(traj[t:t+1], g)[0]
        traj[t+1] = euler_step_wrap(traj[t], f_hat)
    return traj

# Fixed rollout initial conditions (same across seeds)
rollout_inits = [
    torch.tensor([0.3, 0.0]), torch.tensor([1.5, 0.0]),
    torch.tensor([2.8, 0.0]), torch.tensor([0.0, 2.5]),
    torch.tensor([-2.0, 1.0]),
]
n_roll = 200

def eval_rollout(predict_fn, model, is_global=False):
    errs = []
    for x0 in rollout_inits:
        tru = torch.tensor(generate_trajectory(x0.numpy(), n_steps=n_roll, dt=dt),
                           dtype=torch.float64)
        if is_global:
            sim = rollout_global(x0, model, n_roll)
        else:
            sim = rollout(x0, predict_fn, model, n_roll)
        errs.append(angular_dist(sim[200], tru[200]).item())
    return float(np.mean(errs))


# ── Single-seed experiment function ──────────────────────────────────────────

def run_seed(seed, configs):
    """
    Run all method configs for one seed.
    Returns dict of {config_name: {metric: value}}.
    """
    # Generate data with this seed
    train = sample_phase_space(n_samples=4000, seed=seed)
    test  = sample_phase_space(n_samples=1000, seed=seed + 10000)
    X_tr = torch.tensor(train['X'], dtype=torch.float64)
    F_tr = torch.tensor(train['F'], dtype=torch.float64)
    X_te = torch.tensor(test['X'],  dtype=torch.float64)
    F_te = torch.tensor(test['F'],  dtype=torch.float64)

    hp_base = {
        'alpha0': 0.5, 'mu0': X_tr.mean(dim=0),
        'Lambda0': 0.01 * torch.eye(d, dtype=torch.float64),
        'kappa0': 1.0, 'Psi0': 1.0 * torch.eye(d, dtype=torch.float64),
        'nu0': float(d + 2), 'sigma2': 'auto',
    }

    results = {}

    for cfg in configs:
        name = cfg['name']
        kind = cfg['kind']

        if kind == 'global_edmd':
            g = fit_global_edmd(X_tr, F_tr, degree=cfg['degree'])
            F_pred = predict_global_edmd(X_te, g)
            one = torch.linalg.norm(F_pred - F_te, dim=1).mean().item()
            r200 = eval_rollout(None, g, is_global=True)
            results[name] = {'one_step': one, 'rollout_10s': r200}

        elif kind == 'local_edmd':
            hp = dict(hp_base); hp['sigma2'] = 'auto'
            state, _, _ = fit_local_edmd(X_tr, F_tr, N=cfg['N'], hp=hp,
                                         degree=cfg['degree'], n_iter=60,
                                         n_restarts=2, verbose=False)
            F_pred = predict_f_local(X_te, state)
            one = torch.linalg.norm(F_pred - F_te, dim=1).mean().item()
            r200 = eval_rollout(predict_f_local, state)
            results[name] = {'one_step': one, 'rollout_10s': r200}

        elif kind == 'taylor_analytic':
            hp = dict(hp_base); hp['sigma2'] = 'auto'
            state, _, _ = fit_taylor(X_tr, F_tr, pendulum_f, pendulum_J,
                                     N=cfg['N'], hp=hp, n_iter=60,
                                     n_restarts=2, verbose=False)
            F_pred = predict_f_taylor(X_te, state)
            one = torch.linalg.norm(F_pred - F_te, dim=1).mean().item()
            r200 = eval_rollout(predict_f_taylor, state)
            results[name] = {'one_step': one, 'rollout_10s': r200}

        elif kind == 'taylor_ls':
            hp = dict(hp_base); hp['sigma2'] = 'auto'
            state, _, _ = fit_hybrid(X_tr, F_tr, pendulum_f, pendulum_J,
                                     N=cfg['N'], hp=hp, n_iter=60,
                                     n_restarts=2, verbose=False)
            F_pred = predict_f_taylor(X_te, state)
            one = torch.linalg.norm(F_pred - F_te, dim=1).mean().item()
            r200 = eval_rollout(predict_f_taylor, state)
            results[name] = {'one_step': one, 'rollout_10s': r200}

    return results


# ── Configuration ────────────────────────────────────────────────────────────

configs = [
    {'name': 'Global EDMD deg=2',    'kind': 'global_edmd', 'degree': 2},
    {'name': 'Global EDMD deg=4',    'kind': 'global_edmd', 'degree': 4},
    {'name': 'Global EDMD deg=6',    'kind': 'global_edmd', 'degree': 6},
    {'name': 'Global EDMD deg=8',    'kind': 'global_edmd', 'degree': 8},
    {'name': 'local-EDMD d2 N=2',    'kind': 'local_edmd',  'degree': 2, 'N': 2},
    {'name': 'local-EDMD d2 N=4',    'kind': 'local_edmd',  'degree': 2, 'N': 4},
    {'name': 'local-EDMD d2 N=8',    'kind': 'local_edmd',  'degree': 2, 'N': 8},
    {'name': 'Taylor-analytic N=2',  'kind': 'taylor_analytic', 'N': 2},
    {'name': 'Taylor-analytic N=4',  'kind': 'taylor_analytic', 'N': 4},
    {'name': 'Taylor-analytic N=8',  'kind': 'taylor_analytic', 'N': 8},
    {'name': 'Taylor-analytic N=16', 'kind': 'taylor_analytic', 'N': 16},
    {'name': 'Taylor-LS N=2',       'kind': 'taylor_ls', 'N': 2},
    {'name': 'Taylor-LS N=4',       'kind': 'taylor_ls', 'N': 4},
    {'name': 'Taylor-LS N=8',       'kind': 'taylor_ls', 'N': 8},
    {'name': 'Taylor-LS N=16',      'kind': 'taylor_ls', 'N': 16},
]

# ── Run all seeds ────────────────────────────────────────────────────────────

# seeds already defined from env at top of file
all_runs = {}  # {config_name: {metric: [values_per_seed]}}

for i, seed in enumerate(seeds):
    print(f"\n{'='*60}")
    print(f"SEED {seed} ({i+1}/{N_SEEDS})")
    print(f"{'='*60}")
    results = run_seed(seed, configs)
    for name, metrics in results.items():
        if name not in all_runs:
            all_runs[name] = {m: [] for m in metrics}
        for m, v in metrics.items():
            all_runs[name][m].append(v)

# ── Report: mean ± 95% CI ────────────────────────────────────────────────────

print("\n" + "=" * 100)
print(f"PENDULUM — {N_SEEDS} seeds, mean ± 95% CI")
print("=" * 100)
print(f"{'method':<28} {'one-step':>22s} {'rollout@10s':>22s}")
print("-" * 100)

for cfg in configs:
    name = cfg['name']
    one_m, one_hw, _, _ = confidence_interval(all_runs[name]['one_step'])
    r_m,   r_hw,   _, _ = confidence_interval(all_runs[name]['rollout_10s'])
    print(f"{name:<28} {one_m:>10.4f} +/- {one_hw:.4f}   {r_m:>10.3f} +/- {r_hw:.3f}")

# ── Statistical tests ────────────────────────────────────────────────────────

print("\n" + "=" * 100)
print("PAIRED T-TESTS (p-values)")
print("=" * 100)

test_pairs = [
    ("Taylor-analytic N=8",  "Global EDMD deg=8",   "Taylor-ana N=8 vs Global deg=8"),
    ("Taylor-analytic N=2",  "Taylor-LS N=2",       "Taylor-ana N=2 vs Taylor-LS N=2 (crossover)"),
    ("Taylor-analytic N=8",  "Taylor-LS N=8",       "Taylor-ana N=8 vs Taylor-LS N=8 (crossover)"),
    ("local-EDMD d2 N=2",   "Global EDMD deg=6",   "local-EDMD N=2 vs Global deg=6"),
    ("Taylor-analytic N=16", "local-EDMD d2 N=8",   "Taylor-ana N=16 vs local-EDMD N=8"),
]

for name_a, name_b, label in test_pairs:
    for metric in ['one_step', 'rollout_10s']:
        a = np.array(all_runs[name_a][metric])
        b = np.array(all_runs[name_b][metric])
        t = paired_test(a, b)
        sig = "***" if t['p_value'] < 0.001 else "**" if t['p_value'] < 0.01 else "*" if t['p_value'] < 0.05 else "ns"
        print(f"  {label} [{metric}]: diff={t['mean_diff']:+.4f}, p={t['p_value']:.4f} {sig}")
    print()

# ── Save raw data ────────────────────────────────────────────────────────────

raw = {
    'n_seeds': N_SEEDS,
    'seeds': seeds,
    'results': {name: {m: vals for m, vals in metrics.items()}
                for name, metrics in all_runs.items()},
}
with open(data_path("pendulum_statistical.json"), "w") as fp:
    json.dump(raw, fp, indent=2)
print(f"\nRaw data saved to {data_path('pendulum_statistical.json')}")

# ── Plot with error bars ─────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, metric, ylabel, title in [
    (axes[0], 'one_step',    r'mean $\|f_{pred}-f_{true}\|$', 'One-step f-error (mean ± 95% CI)'),
    (axes[1], 'rollout_10s', r'angular dist at t=10s',         'Rollout @ 10s (mean ± 95% CI)'),
]:
    # Group by kind
    for kind_label, prefix, color, marker in [
        ('Global EDMD',     'Global EDMD',     'C3', 'D'),
        ('local-EDMD d2',   'local-EDMD d2',   'C0', 'o'),
        ('Taylor-analytic', 'Taylor-analytic',  'C2', '^'),
        ('Taylor-LS',       'Taylor-LS',        'C4', 'v'),
    ]:
        names = [c['name'] for c in configs if c['name'].startswith(prefix)]
        if not names:
            continue
        means = [confidence_interval(all_runs[n][metric])[0] for n in names]
        cis   = [confidence_interval(all_runs[n][metric])[1] for n in names]
        xs    = range(len(names))
        ax.errorbar(xs, means, yerr=cis, fmt=f'{marker}-', color=color,
                    label=kind_label, capsize=4, markersize=6)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([n.replace(prefix + ' ', '') for n in names],
                           rotation=45, ha='right', fontsize=7)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(fig_path("pendulum_statistical.png"), dpi=120)
print(f"Plot saved to {fig_path('pendulum_statistical.png')}")
