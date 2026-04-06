"""
Complete statistical validation for both Lorenz and Pendulum.

Runs all method families across multiple seeds, reports:
  - Mean ± 95% CI for all metrics
  - Paired t-tests between key method pairs
  - Saves full per-seed data to JSON
  - Generates error-bar plots

This is the single authoritative script for paper results.
"""

import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

torch.set_default_dtype(torch.float64)

from utils.paths import fig_path, data_path
from utils.stats import confidence_interval, paired_test

from simulators.lorenz import generate_data as lorenz_generate, f as lorenz_f, J as lorenz_J
from simulators.pendulum import (
    f as pendulum_f, J as pendulum_J,
    sample_phase_space, generate_trajectory, wrap_theta, angular_dist,
)
from models.em import fit as fit_taylor
from models.em_hybrid import fit_hybrid
from models.em_local_edmd import (
    fit as fit_local_edmd_cont,
    predict_f_all_clusters, monomial_exponents, monomials,
)
from models.em_local_edmd_discrete import (
    fit as fit_local_edmd_disc,
    fit_global as fit_global_disc,
    predict_next_global as predict_next_disc,
    predict_next_all_clusters as predict_next_all_disc,
)
from models.distributions import mvn_logpdf_batch

import pykoopman as pk


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Complete statistical validation (Lorenz + Pendulum)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# Seeds
parser.add_argument('--seeds', type=int, nargs='+',
                    default=[1, 42, 101, 307, 1001, 7789, 13245, 11, 103, 13],
                    help="Random seeds for statistical runs")

# Lorenz data
parser.add_argument('--lorenz-n-steps', type=int, default=5000)
parser.add_argument('--lorenz-dt', type=float, default=0.01)
parser.add_argument('--lorenz-warmup', type=int, default=1000)
parser.add_argument('--lorenz-n-train', type=int, default=4000)

# Pendulum data
parser.add_argument('--pendulum-n-train', type=int, default=4000)
parser.add_argument('--pendulum-n-test', type=int, default=1000)
parser.add_argument('--pendulum-dt', type=float, default=0.05)
parser.add_argument('--pendulum-rollout-steps', type=int, default=200)

# EM fitting
parser.add_argument('--n-iter', type=int, default=100)
parser.add_argument('--n-restarts', type=int, default=2)

# Cluster counts to sweep
parser.add_argument('--lorenz-N', type=int, nargs='+', default=[5, 12, 20, 50])
parser.add_argument('--pendulum-N', type=int, nargs='+', default=[2, 4, 8, 16])

# EDMD degrees
parser.add_argument('--edmd-degrees', type=int, nargs='+', default=[2, 3])
parser.add_argument('--pendulum-edmd-degrees', type=int, nargs='+', default=[2, 4, 6, 8])

# Rollout
parser.add_argument('--lorenz-rollout-steps', type=int, default=500)

# Systems to run
parser.add_argument('--skip-lorenz', action='store_true')
parser.add_argument('--skip-pendulum', action='store_true')

# Model saving (on by default for visualization)
parser.add_argument('--no-save-models', action='store_true',
                    help="Skip saving fitted model states (saves disk space)")

args = parser.parse_args()

seeds = args.seeds
N_SEEDS = len(seeds)

print("=" * 80)
print("  Statistical Validation")
print(f"  Seeds ({N_SEEDS}): {seeds}")
print("=" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def pick_cluster(x, state):
    log_pi = torch.log(state['pi']).unsqueeze(0)
    log_prox = mvn_logpdf_batch(x, state['centers'], state['covariances'])
    return (log_pi + log_prox).argmax(dim=1)


def make_hp(X_tr, d):
    return {
        'alpha0': 0.5, 'mu0': X_tr.mean(dim=0),
        'Lambda0': 0.01 * torch.eye(d, dtype=torch.float64),
        'kappa0': 1.0, 'Psi0': 10.0 * torch.eye(d, dtype=torch.float64),
        'nu0': float(d + 2), 'sigma2': 'auto',
    }


def make_hp_pendulum(X_tr, d):
    return {
        'alpha0': 0.5, 'mu0': X_tr.mean(dim=0),
        'Lambda0': 0.01 * torch.eye(d, dtype=torch.float64),
        'kappa0': 1.0, 'Psi0': 1.0 * torch.eye(d, dtype=torch.float64),
        'nu0': float(d + 2), 'sigma2': 'auto',
    }


# ─────────────────────────────────────────────────────────────────────────────
# LORENZ per-seed
# ─────────────────────────────────────────────────────────────────────────────

def lorenz_piecewise_f(x, state):
    k = pick_cluster(x, state)
    c = state['centers'][k]
    fc = state['f_centers'][k]
    J = state['jacobians'][k]
    return fc + (J @ (x - c).unsqueeze(-1)).squeeze(-1)


def lorenz_one_step_err(state, X_te_in, X_te_next, dt):
    pred = X_te_in + dt * lorenz_piecewise_f(X_te_in, state)
    return torch.linalg.norm(pred - X_te_next, dim=1).mean().item()


def lorenz_rollout_truth(x0, n_steps, dt):
    sol = solve_ivp(lambda t, y: lorenz_f(y),
                    (0.0, n_steps * dt), x0.numpy(),
                    t_eval=np.linspace(0.0, n_steps * dt, n_steps + 1),
                    method='RK45', rtol=1e-10, atol=1e-10)
    return torch.tensor(sol.y.T, dtype=torch.float64)


def lorenz_rollout_err(state, inits, n_steps, dt):
    errs = []
    for x0 in inits:
        tru = lorenz_rollout_truth(x0, n_steps, dt)
        traj = torch.zeros(n_steps + 1, 3, dtype=torch.float64)
        traj[0] = x0
        for t in range(n_steps):
            traj[t + 1] = traj[t] + dt * lorenz_piecewise_f(traj[t:t + 1], state)[0]
        errs.append(torch.linalg.norm(traj[n_steps] - tru[n_steps]).item())
    return float(np.mean(errs))


def lorenz_disc_rollout_err(model, inits, n_steps, dt):
    errs = []
    for x0 in inits:
        tru = lorenz_rollout_truth(x0, n_steps, dt)
        traj = torch.zeros(n_steps + 1, 3, dtype=torch.float64)
        traj[0] = x0
        for t in range(n_steps):
            traj[t + 1] = predict_next_disc(traj[t:t + 1], model)[0]
        errs.append(torch.linalg.norm(traj[n_steps] - tru[n_steps]).item())
    return float(np.mean(errs))


def lorenz_disc_local_rollout_err(state, inits, n_steps, d):
    errs = []
    for x0 in inits:
        tru = lorenz_rollout_truth(x0, n_steps, args.lorenz_dt)
        traj = torch.zeros(n_steps + 1, d, dtype=torch.float64)
        traj[0] = x0
        for t in range(n_steps):
            k = pick_cluster(traj[t:t + 1], state)
            preds = predict_next_all_disc(traj[t:t + 1], state['centers'],
                                          state['K_ops'], state['exps'], d)
            traj[t + 1] = preds[0, k[0]]
        errs.append(torch.linalg.norm(traj[n_steps] - tru[n_steps]).item())
    return float(np.mean(errs))


def run_lorenz_seed(seed):
    dt = args.lorenz_dt
    d = 3
    data = lorenz_generate(n_steps=args.lorenz_n_steps, dt=dt,
                           warmup=args.lorenz_warmup, seed=seed)
    X_all = torch.tensor(data['X'], dtype=torch.float64)
    F_all = torch.tensor(data['F'], dtype=torch.float64)

    nt = args.lorenz_n_train
    X_tr, X_te = X_all[:nt], X_all[nt:]
    F_tr = F_all[:nt]
    X_te_in = X_all[nt:X_all.shape[0] - 1]
    X_te_next = X_all[nt + 1:]
    step_bl = torch.linalg.norm(X_te_next - X_te_in, dim=1).mean().item()

    # Consecutive pairs for discrete EDMD
    X_tr_curr = X_all[:nt - 1]
    X_tr_next = X_all[1:nt]

    inits = [X_te[i] for i in (0, min(300, len(X_te) - 1), min(700, len(X_te) - 1))]
    rs = args.lorenz_rollout_steps

    results = {}

    # ── pykoopman baselines ──────────────────────────────────────────────
    for deg in args.edmd_degrees:
        model = pk.Koopman(
            observables=pk.observables.Polynomial(degree=deg, include_bias=True),
            regressor=pk.regression.EDMD())
        model.fit(X_tr.numpy(), dt=dt)
        one = torch.linalg.norm(
            torch.tensor(model.predict(X_te_in.numpy())) - X_te_next, dim=1
        ).mean().item()
        results[f'EDMD-pk deg-{deg}'] = {
            'one_step': one, 'rel_pct': 100 * one / step_bl,
        }

    # ── Discrete global EDMD (our solver, fair baseline) ─────────────────
    for deg in args.edmd_degrees:
        g = fit_global_disc(X_tr_curr, X_tr_next, degree=deg)
        pred = predict_next_disc(X_te_in, g)
        one = torch.linalg.norm(pred - X_te_next, dim=1).mean().item()
        r500 = lorenz_disc_rollout_err(g, inits, rs, dt)
        results[f'EDMD-disc deg-{deg}'] = {
            'one_step': one, 'rel_pct': 100 * one / step_bl, 'rollout': r500,
        }

    # ── Taylor-analytic (ours) + GMM ─────────────────────────────────────
    hp = make_hp(X_tr, d)
    for N in args.lorenz_N:
        # Ours
        s_o, _, _ = fit_taylor(X_tr, F_tr, lorenz_f, lorenz_J,
                               N=N, hp={**hp, 'sigma2': 'auto'},
                               n_iter=args.n_iter, n_restarts=args.n_restarts,
                               verbose=False)
        one_o = lorenz_one_step_err(s_o, X_te_in, X_te_next, dt)
        r500_o = lorenz_rollout_err(s_o, inits, rs, dt)
        results[f'Taylor N={N}'] = {
            'one_step': one_o, 'rel_pct': 100 * one_o / step_bl, 'rollout': r500_o,
        }

        # GMM
        s_g, _, _ = fit_taylor(X_tr, F_tr, lorenz_f, lorenz_J,
                               N=N, hp={**hp, 'sigma2': 1e10},
                               n_iter=args.n_iter, n_restarts=args.n_restarts,
                               verbose=False)
        one_g = lorenz_one_step_err(s_g, X_te_in, X_te_next, dt)
        r500_g = lorenz_rollout_err(s_g, inits, rs, dt)
        results[f'GMM N={N}'] = {
            'one_step': one_g, 'rel_pct': 100 * one_g / step_bl, 'rollout': r500_g,
        }

    # ── Local discrete EDMD ──────────────────────────────────────────────
    for N in args.lorenz_N:
        s_ld, _, _ = fit_local_edmd_disc(
            X_tr_curr, X_tr_next, N=N, hp={**hp, 'sigma2': 'auto'},
            degree=2, n_iter=args.n_iter, n_restarts=args.n_restarts,
            verbose=False)
        k = pick_cluster(X_te_in, s_ld)
        pred = predict_next_all_disc(
            X_te_in, s_ld['centers'], s_ld['K_ops'], s_ld['exps'], d)
        one_ld = torch.linalg.norm(
            pred[torch.arange(len(X_te_in)), k] - X_te_next, dim=1
        ).mean().item()
        r500_ld = lorenz_disc_local_rollout_err(s_ld, inits, rs, d)
        results[f'Local-EDMD-disc N={N}'] = {
            'one_step': one_ld, 'rel_pct': 100 * one_ld / step_bl, 'rollout': r500_ld,
        }

    # Collect model states for visualization
    models = {}
    # Re-fit best N for each family to store (use median N)
    best_N = args.lorenz_N[len(args.lorenz_N) // 2]
    s_taylor, _, _ = fit_taylor(X_tr, F_tr, lorenz_f, lorenz_J,
                                N=best_N, hp={**hp, 'sigma2': 'auto'},
                                n_iter=args.n_iter, n_restarts=args.n_restarts,
                                verbose=False)
    models['taylor'] = s_taylor

    s_gmm, _, _ = fit_taylor(X_tr, F_tr, lorenz_f, lorenz_J,
                             N=best_N, hp={**hp, 'sigma2': 1e10},
                             n_iter=args.n_iter, n_restarts=args.n_restarts,
                             verbose=False)
    models['gmm'] = s_gmm

    s_disc, _, _ = fit_local_edmd_disc(
        X_tr_curr, X_tr_next, N=best_N, hp={**hp, 'sigma2': 'auto'},
        degree=2, n_iter=args.n_iter, n_restarts=args.n_restarts,
        verbose=False)
    models['local_edmd_disc'] = s_disc

    g_disc = fit_global_disc(X_tr_curr, X_tr_next, degree=2)
    models['global_edmd_disc'] = g_disc

    models['X_all'] = X_all
    models['F_all'] = F_all

    return results, models


# ─────────────────────────────────────────────────────────────────────────────
# PENDULUM per-seed
# ─────────────────────────────────────────────────────────────────────────────

def pendulum_predict_f_taylor(x, state):
    k = pick_cluster(x, state)
    c = state['centers'][k]
    fc = state['f_centers'][k]
    J = state['jacobians'][k]
    return fc + (J @ (x - c).unsqueeze(-1)).squeeze(-1)


def pendulum_predict_f_local(x, state, d):
    k = pick_cluster(x, state)
    F_all = predict_f_all_clusters(
        x, state['centers'], state['M_ops'], state['exps'], d)
    return F_all[torch.arange(x.shape[0]), k]


def pendulum_euler_step(x, f_val, dt):
    return wrap_theta(x + dt * f_val)


def pendulum_rollout(x0, predict_fn, model, n_steps, dt, d):
    traj = torch.zeros(n_steps + 1, d, dtype=torch.float64)
    traj[0] = x0
    for t in range(n_steps):
        f_hat = predict_fn(traj[t:t + 1], model)[0]
        traj[t + 1] = pendulum_euler_step(traj[t], f_hat, dt)
    return traj


def pendulum_eval_rollout(predict_fn, model, inits, n_roll, dt, d):
    errs = []
    for x0 in inits:
        tru = torch.tensor(generate_trajectory(x0.numpy(), n_steps=n_roll, dt=dt),
                           dtype=torch.float64)
        sim = pendulum_rollout(x0, predict_fn, model, n_roll, dt, d)
        errs.append(angular_dist(sim[n_roll], tru[n_roll]).item())
    return float(np.mean(errs))


def pendulum_fit_global_edmd(X_tr, F_tr, degree, d):
    exps = monomial_exponents(d, degree)
    c = X_tr.mean(dim=0)
    r = torch.ones(X_tr.shape[0], dtype=torch.float64)
    from models.em_local_edmd import weighted_continuous_edmd
    M = weighted_continuous_edmd(X_tr, F_tr, r, c, exps, ridge=1e-6)
    return {'M': M, 'c': c, 'exps': exps}


def pendulum_predict_global(X, g, d):
    U = X - g['c']
    Phi = monomials(U, g['exps'])
    Phi_dot = Phi @ g['M'].T
    return Phi_dot[:, 1:d + 1]


def pendulum_rollout_global(x0, g, n_steps, dt, d):
    traj = torch.zeros(n_steps + 1, d, dtype=torch.float64)
    traj[0] = x0
    for t in range(n_steps):
        f_hat = pendulum_predict_global(traj[t:t + 1], g, d)[0]
        traj[t + 1] = pendulum_euler_step(traj[t], f_hat, dt)
    return traj


def pendulum_eval_rollout_global(g, inits, n_roll, dt, d):
    errs = []
    for x0 in inits:
        tru = torch.tensor(generate_trajectory(x0.numpy(), n_steps=n_roll, dt=dt),
                           dtype=torch.float64)
        sim = pendulum_rollout_global(x0, g, n_roll, dt, d)
        errs.append(angular_dist(sim[n_roll], tru[n_roll]).item())
    return float(np.mean(errs))


def run_pendulum_seed(seed):
    dt = args.pendulum_dt
    d = 2
    n_roll = args.pendulum_rollout_steps

    train = sample_phase_space(n_samples=args.pendulum_n_train, seed=seed)
    test = sample_phase_space(n_samples=args.pendulum_n_test, seed=seed + 10000)
    X_tr = torch.tensor(train['X'], dtype=torch.float64)
    F_tr = torch.tensor(train['F'], dtype=torch.float64)
    X_te = torch.tensor(test['X'], dtype=torch.float64)
    F_te = torch.tensor(test['F'], dtype=torch.float64)

    hp = make_hp_pendulum(X_tr, d)

    rollout_inits = [
        torch.tensor([0.3, 0.0]), torch.tensor([1.5, 0.0]),
        torch.tensor([2.8, 0.0]), torch.tensor([0.0, 2.5]),
        torch.tensor([-2.0, 1.0]),
    ]

    results = {}

    # ── Global EDMD ──────────────────────────────────────────────────────
    for deg in args.pendulum_edmd_degrees:
        g = pendulum_fit_global_edmd(X_tr, F_tr, deg, d)
        F_pred = pendulum_predict_global(X_te, g, d)
        one = torch.linalg.norm(F_pred - F_te, dim=1).mean().item()
        r10s = pendulum_eval_rollout_global(g, rollout_inits, n_roll, dt, d)
        results[f'Global EDMD deg={deg}'] = {'one_step': one, 'rollout_10s': r10s}

    # ── Local EDMD (continuous) ──────────────────────────────────────────
    for N in args.pendulum_N:
        s, _, _ = fit_local_edmd_cont(
            X_tr, F_tr, N=N, hp={**hp, 'sigma2': 'auto'},
            degree=2, n_iter=args.n_iter, n_restarts=args.n_restarts,
            verbose=False)
        F_pred = pendulum_predict_f_local(X_te, s, d)
        one = torch.linalg.norm(F_pred - F_te, dim=1).mean().item()
        r10s = pendulum_eval_rollout(
            lambda x, m: pendulum_predict_f_local(x, m, d), s,
            rollout_inits, n_roll, dt, d)
        results[f'local-EDMD d2 N={N}'] = {'one_step': one, 'rollout_10s': r10s}

    # ── Taylor-analytic ──────────────────────────────────────────────────
    for N in args.pendulum_N:
        s, _, _ = fit_taylor(
            X_tr, F_tr, pendulum_f, pendulum_J,
            N=N, hp={**hp, 'sigma2': 'auto'},
            n_iter=args.n_iter, n_restarts=args.n_restarts, verbose=False)
        F_pred = pendulum_predict_f_taylor(X_te, s)
        one = torch.linalg.norm(F_pred - F_te, dim=1).mean().item()
        r10s = pendulum_eval_rollout(
            pendulum_predict_f_taylor, s, rollout_inits, n_roll, dt, d)
        results[f'Taylor-analytic N={N}'] = {'one_step': one, 'rollout_10s': r10s}

    # ── Taylor-LS ────────────────────────────────────────────────────────
    for N in args.pendulum_N:
        s, _, _ = fit_hybrid(
            X_tr, F_tr, pendulum_f, pendulum_J,
            N=N, hp={**hp, 'sigma2': 'auto'},
            n_iter=args.n_iter, n_restarts=args.n_restarts, verbose=False)
        F_pred = pendulum_predict_f_taylor(X_te, s)
        one = torch.linalg.norm(F_pred - F_te, dim=1).mean().item()
        r10s = pendulum_eval_rollout(
            pendulum_predict_f_taylor, s, rollout_inits, n_roll, dt, d)
        results[f'Taylor-LS N={N}'] = {'one_step': one, 'rollout_10s': r10s}

    # Collect model states for visualization
    models = {}
    best_N = args.pendulum_N[len(args.pendulum_N) // 2]

    s_taylor, _, _ = fit_taylor(
        X_tr, F_tr, pendulum_f, pendulum_J,
        N=best_N, hp={**hp, 'sigma2': 'auto'},
        n_iter=args.n_iter, n_restarts=args.n_restarts, verbose=False)
    models['taylor'] = s_taylor

    s_ls, _, _ = fit_hybrid(
        X_tr, F_tr, pendulum_f, pendulum_J,
        N=best_N, hp={**hp, 'sigma2': 'auto'},
        n_iter=args.n_iter, n_restarts=args.n_restarts, verbose=False)
    models['taylor_ls'] = s_ls

    s_local, _, _ = fit_local_edmd_cont(
        X_tr, F_tr, N=best_N, hp={**hp, 'sigma2': 'auto'},
        degree=2, n_iter=args.n_iter, n_restarts=args.n_restarts, verbose=False)
    models['local_edmd'] = s_local

    best_deg = args.pendulum_edmd_degrees[-1]
    g_global = pendulum_fit_global_edmd(X_tr, F_tr, best_deg, d)
    models['global_edmd'] = g_global
    models['global_edmd_degree'] = best_deg

    models['X_tr'] = X_tr
    models['F_tr'] = F_tr
    models['X_te'] = X_te
    models['F_te'] = F_te

    return results, models


# ─────────────────────────────────────────────────────────────────────────────
# Run all seeds
# ─────────────────────────────────────────────────────────────────────────────

def run_system(name, run_fn):
    all_runs = {}
    best_models = None
    best_seed = None
    best_total_err = float('inf')

    for i, seed in enumerate(seeds):
        print(f"\n  {name} — seed {seed} ({i + 1}/{N_SEEDS})")
        results, models = run_fn(seed)
        for method, metrics in results.items():
            if method not in all_runs:
                all_runs[method] = {m: [] for m in metrics}
            for m, v in metrics.items():
                all_runs[method][m].append(v)

        # Track best seed by average one_step error across all methods
        total_err = np.mean([m['one_step'] for m in results.values() if 'one_step' in m])
        if total_err < best_total_err:
            best_total_err = total_err
            best_models = models
            best_seed = seed

    print(f"\n  Best seed for {name}: {best_seed} (avg one-step err: {best_total_err:.6f})")
    return all_runs, best_models, best_seed


def report(name, all_runs, metrics_list):
    print(f"\n{'=' * 100}")
    print(f"{name} — {N_SEEDS} seeds, mean ± 95% CI")
    print("=" * 100)
    header = f"{'method':<30}"
    for m in metrics_list:
        header += f" {m:>22s}"
    print(header)
    print("-" * 100)
    for method in all_runs:
        line = f"{method:<30}"
        for m in metrics_list:
            if m in all_runs[method]:
                mean, hw, _, _ = confidence_interval(all_runs[method][m])
                line += f" {mean:>10.5f} ± {hw:.5f}"
            else:
                line += f" {'—':>22s}"
        print(line)


def paired_tests(name, all_runs, pairs):
    print(f"\n  Paired t-tests ({name}):")
    for na, nb, metric, label in pairs:
        if na not in all_runs or nb not in all_runs:
            continue
        if metric not in all_runs[na] or metric not in all_runs[nb]:
            continue
        a = np.array(all_runs[na][metric])
        b = np.array(all_runs[nb][metric])
        t = paired_test(a, b)
        sig = "***" if t['p_value'] < 0.001 else "**" if t['p_value'] < 0.01 \
            else "*" if t['p_value'] < 0.05 else "ns"
        print(f"    {label}: {a.mean():.5f} vs {b.mean():.5f}, "
              f"diff={t['mean_diff']:+.5f}, p={t['p_value']:.4f} {sig}")


# ─────────────────────────────────────────────────────────────────────────────
# LORENZ
# ─────────────────────────────────────────────────────────────────────────────

if not args.skip_lorenz:
    print("\n" + "=" * 80)
    print("  LORENZ SYSTEM")
    print("=" * 80)

    lorenz_runs, lorenz_models, lorenz_best_seed = run_system("Lorenz", run_lorenz_seed)
    report("LORENZ", lorenz_runs, ['one_step', 'rel_pct', 'rollout'])

    lorenz_pairs = []
    for N in args.lorenz_N:
        lorenz_pairs.append(
            (f'Taylor N={N}', f'GMM N={N}', 'one_step', f'Taylor vs GMM N={N}'))
    for N in args.lorenz_N:
        lorenz_pairs.append(
            (f'Taylor N={N}', 'EDMD-disc deg-2', 'one_step', f'Taylor N={N} vs EDMD-disc deg-2'))
    for N in args.lorenz_N:
        lorenz_pairs.append(
            (f'Local-EDMD-disc N={N}', 'EDMD-disc deg-2', 'one_step',
             f'Local-EDMD N={N} vs EDMD-disc deg-2'))
    paired_tests("Lorenz", lorenz_runs, lorenz_pairs)

    with open(data_path("statistical_lorenz.json"), "w") as fp:
        json.dump({'n_seeds': N_SEEDS, 'seeds': seeds, 'args': vars(args),
                   'best_seed': lorenz_best_seed,
                   'results': lorenz_runs}, fp, indent=2)
    print(f"\n  Saved: {data_path('statistical_lorenz.json')}")

    if not args.no_save_models:
        torch.save(lorenz_models, data_path("lorenz_models.pt"))
        print(f"  Saved models (best seed {lorenz_best_seed}): {data_path('lorenz_models.pt')}")

# ─────────────────────────────────────────────────────────────────────────────
# PENDULUM
# ─────────────────────────────────────────────────────────────────────────────

if not args.skip_pendulum:
    print("\n" + "=" * 80)
    print("  PENDULUM SYSTEM")
    print("=" * 80)

    pendulum_runs, pendulum_models, pendulum_best_seed = run_system("Pendulum", run_pendulum_seed)
    report("PENDULUM", pendulum_runs, ['one_step', 'rollout_10s'])

    pendulum_pairs = [
        ('Taylor-analytic N=8', 'Global EDMD deg=8', 'rollout_10s',
         'Taylor-ana N=8 vs Global deg=8'),
        ('Taylor-analytic N=8', 'Taylor-LS N=8', 'rollout_10s',
         'Taylor-ana N=8 vs Taylor-LS N=8'),
        ('local-EDMD d2 N=2', 'Global EDMD deg=6', 'rollout_10s',
         'local-EDMD N=2 vs Global deg=6'),
    ]
    paired_tests("Pendulum", pendulum_runs, pendulum_pairs)

    with open(data_path("statistical_pendulum.json"), "w") as fp:
        json.dump({'n_seeds': N_SEEDS, 'seeds': seeds, 'args': vars(args),
                   'best_seed': pendulum_best_seed,
                   'results': pendulum_runs}, fp, indent=2)
    print(f"\n  Saved: {data_path('statistical_pendulum.json')}")

    if not args.no_save_models:
        torch.save(pendulum_models, data_path("pendulum_models.pt"))
        print(f"  Saved models (best seed {pendulum_best_seed}): {data_path('pendulum_models.pt')}")

print("\n" + "=" * 80)
print("  Done.")
print("=" * 80)
