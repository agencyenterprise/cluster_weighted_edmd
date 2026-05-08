"""
Duffing validation with statistical rigor.

System: unforced double-well Duffing oscillator,

    x_ddot + DELTA*x_dot - x + x**3 = 0,

with stable foci at (+/-1, 0) and an index-1 saddle at the origin.

Runs each method across N_SEEDS random seeds, varying both training/test
data sampling and EM initialization.  Reports:

  - Mean +/- 95% CI for one-step f-error and rollout error at 5/10/20 s.
  - Paired t-tests between key method pairs with p-values and significance
    stars (* p<0.05, ** p<0.01, *** p<0.001).
  - Per-seed raw data saved to papers/data/duffing_statistical.json.
  - Error-bar comparison plot saved to papers/figures/duffing_statistical.png.

Mirrors validation_pendulum_statistical.py and validation_lorenz_statistical.py
in CLI shape and output conventions.

Usage
-----
Default seed set used by the rest of the package::

    python -m validation.validation_duffing_statistical \\
        --seeds 1 42 101 307 1001 7789 13245 11 103 13

Trajectory-ensemble sampling instead of uniform-on-box::

    python -m validation.validation_duffing_statistical \\
        --seeds 1 42 101 --sampling trajectory --n-traj 200 --traj-steps 50

Smaller / faster sweep for development::

    python -m validation.validation_duffing_statistical \\
        --seeds 1 42 101 --n-iter 30 --N-list 4 8
"""

import argparse
import json

import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.paths import fig_path, data_path
from utils.stats import confidence_interval, paired_test

from simulators.duffing import (
    f as duffing_f,
    J as duffing_J,
    sample_phase_space,
    sample_trajectory_ensemble,
    generate_trajectory,
    DELTA,
)
from models.em import fit as fit_taylor
from models.em_local_edmd import (
    fit as fit_local_edmd,
    predict_f_all_clusters,
    monomial_exponents,
    monomials,
    weighted_continuous_edmd,
)
from models.distributions import mvn_logpdf_batch

torch.set_default_dtype(torch.float64)


# -- CLI ----------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Duffing statistical validation (multi-seed prediction-error sweep)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seeds', type=int, nargs='+', required=True,
                    help="random seeds to run (e.g. 1 42 101 307 1001 7789 13245 11 103 13)")
# Data configuration
parser.add_argument('--sampling', choices=['uniform', 'trajectory'], default='uniform',
                    help="uniform: sample (x, x_dot) uniformly on a box. "
                         "trajectory: ensemble of short trajectories from random ICs "
                         "(damping concentrates density near the foci).")
parser.add_argument('--n-train',   type=int,   default=4000,
                    help="(uniform) training-set size")
parser.add_argument('--n-test',    type=int,   default=1000,
                    help="(uniform) test-set size")
parser.add_argument('--box-x',     type=float, default=2.0,
                    help="(uniform) half-width of x range")
parser.add_argument('--box-xdot',  type=float, default=2.0,
                    help="(uniform) half-width of x_dot range")
parser.add_argument('--n-traj',    type=int,   default=200,
                    help="(trajectory) number of trajectories")
parser.add_argument('--traj-steps',type=int,   default=50,
                    help="(trajectory) integration steps per trajectory")
parser.add_argument('--ic-x',      type=float, default=2.0,
                    help="(trajectory) IC half-width for x")
parser.add_argument('--ic-xdot',   type=float, default=2.5,
                    help="(trajectory) IC half-width for x_dot")
# Fit configuration
parser.add_argument('--n-iter',    type=int,   default=80)
parser.add_argument('--n-restarts',type=int,   default=2)
# Rollout configuration
parser.add_argument('--dt',           type=float, default=0.05)
parser.add_argument('--rollout-steps',type=int,   default=400,
                    help="rollout horizon in dt units (default 400 -> 20 s at dt=0.05)")
# Method sweep
parser.add_argument('--N-list',     type=int, nargs='+', default=[2, 4, 8, 16],
                    help="cluster counts to sweep")
parser.add_argument('--edmd-degs',  type=int, nargs='+', default=[2, 3, 4, 5],
                    help="global EDMD degrees")
parser.add_argument('--le2-N-list', type=int, nargs='+', default=[2, 4, 8, 16],
                    help="local EDMD deg-2 cluster counts")
parser.add_argument('--le3-N-list', type=int, nargs='+', default=[2, 4, 8],
                    help="local EDMD deg-3 cluster counts")
args = parser.parse_args()

seeds   = args.seeds
N_SEEDS = len(seeds)
d       = 2
dt      = args.dt
n_roll  = args.rollout_steps

print("=" * 80)
print(f"  Duffing statistical validation -- {N_SEEDS} seeds: {seeds}")
print(f"  Sampling: {args.sampling}   ", end='')
if args.sampling == 'uniform':
    print(f"n_train={args.n_train}, n_test={args.n_test}, box=[{-args.box_x},{args.box_x}] x "
          f"[{-args.box_xdot},{args.box_xdot}]")
else:
    print(f"n_traj={args.n_traj}, traj_steps={args.traj_steps}, "
          f"IC box=[{-args.ic_x},{args.ic_x}] x [{-args.ic_xdot},{args.ic_xdot}]")
print(f"  Methods: global EDMD deg in {args.edmd_degs}; "
      f"local-EDMD deg-2 N in {args.le2_N_list}; local-EDMD deg-3 N in {args.le3_N_list}; "
      f"Taylor-analytic N in {args.N_list}; GMM-baseline N in {args.N_list}")
print("=" * 80)


# -- Helpers (match validation_duffing.py / validation_pendulum_statistical.py) ----

def fit_global_edmd(X_tr_, F_tr_, degree, ridge=1e-6):
    exps = monomial_exponents(d, degree)
    c    = X_tr_.mean(dim=0)
    r    = torch.ones(X_tr_.shape[0], dtype=torch.float64)
    M    = weighted_continuous_edmd(X_tr_, F_tr_, r, c, exps, ridge=ridge)
    return {'M': M, 'c': c, 'exps': exps}


def predict_global_edmd(X_pts, g):
    U   = X_pts - g['c']
    Phi = monomials(U, g['exps'])
    return (Phi @ g['M'].T)[:, 1:d+1]


def pick_cluster(X_pts, state):
    log_pi   = torch.log(state['pi']).unsqueeze(0)
    log_prox = mvn_logpdf_batch(X_pts, state['centers'], state['covariances'])
    return (log_pi + log_prox).argmax(dim=1)


def predict_f_local(X_pts, state):
    k     = pick_cluster(X_pts, state)
    F_all = predict_f_all_clusters(X_pts, state['centers'], state['M_ops'],
                                   state['exps'], d)
    return F_all[torch.arange(X_pts.shape[0]), k]


def predict_f_taylor(X_pts, state):
    k  = pick_cluster(X_pts, state)
    c  = state['centers'][k]
    fc = state['f_centers'][k]
    Jk = state['jacobians'][k]
    return fc + (Jk @ (X_pts - c).unsqueeze(-1)).squeeze(-1)


def euler_step(x, f_val):
    return x + dt * f_val


def rollout_traj(x0, predict_fn, model, n_steps, is_global=False):
    traj    = torch.zeros(n_steps + 1, d, dtype=torch.float64)
    traj[0] = x0
    for t in range(n_steps):
        if is_global:
            f_hat = predict_global_edmd(traj[t:t+1], model)[0]
        else:
            f_hat = predict_fn(traj[t:t+1], model)[0]
        traj[t + 1] = euler_step(traj[t], f_hat)
    return traj


# Fixed diagnostic ICs used across all seeds (covers both basins, saddle, transient)
ROLLOUT_INITS = [
    torch.tensor([+1.5, 0.0]),
    torch.tensor([-1.5, 0.0]),
    torch.tensor([+0.3, 0.0]),
    torch.tensor([-0.3, 0.0]),
    torch.tensor([+0.5, 1.5]),
]


def eval_rollout(predict_fn, model, is_global=False):
    """Return mean rollout L2 error at t = 5 s, 10 s, 20 s averaged over inits."""
    indices = {5: int(round(5.0  / dt)),
               10: int(round(10.0 / dt)),
               20: int(round(20.0 / dt))}
    errs = {h: [] for h in indices}
    for x0 in ROLLOUT_INITS:
        tru = torch.tensor(generate_trajectory(x0.numpy(), n_steps=n_roll, dt=dt),
                           dtype=torch.float64)
        sim = rollout_traj(x0, predict_fn, model, n_roll, is_global=is_global)
        diff = torch.linalg.norm(sim - tru, dim=1)
        for h, idx in indices.items():
            errs[h].append(diff[idx].item())
    return {h: float(np.mean(errs[h])) for h in errs}


# -- Per-seed runner ----------------------------------------------------------

def make_data(seed):
    """Generate train / test data with the given seed."""
    if args.sampling == 'uniform':
        train = sample_phase_space(n_samples=args.n_train, x_max=args.box_x,
                                   xdot_max=args.box_xdot, seed=seed)
        test  = sample_phase_space(n_samples=args.n_test,  x_max=args.box_x,
                                   xdot_max=args.box_xdot, seed=seed + 10000)
    else:
        train = sample_trajectory_ensemble(
            n_traj=args.n_traj, n_steps=args.traj_steps,
            dt=dt, ic_x_max=args.ic_x, ic_xdot_max=args.ic_xdot, seed=seed,
        )
        test  = sample_trajectory_ensemble(
            n_traj=max(args.n_traj // 4, 20),
            n_steps=args.traj_steps,
            dt=dt, ic_x_max=args.ic_x, ic_xdot_max=args.ic_xdot,
            seed=seed + 10000,
        )
    return train, test


def run_seed(seed, configs):
    """Fit every config on the train set drawn at this seed and evaluate.

    Returns
    -------
    dict
        Mapping name -> {'one_step': float, 'r05s': float, 'r10s': float, 'r20s': float}.
    """
    train, test = make_data(seed)
    X_tr = torch.tensor(train['X'], dtype=torch.float64)
    F_tr = torch.tensor(train['F'], dtype=torch.float64)
    X_te = torch.tensor(test ['X'], dtype=torch.float64)
    F_te = torch.tensor(test ['F'], dtype=torch.float64)

    hp_base = {
        'alpha0':  0.5, 'mu0':     X_tr.mean(dim=0),
        'Lambda0': 0.01 * torch.eye(d, dtype=torch.float64),
        'kappa0':  1.0, 'Psi0':    1.0  * torch.eye(d, dtype=torch.float64),
        'nu0':     float(d + 2),
    }

    results = {}
    for cfg in configs:
        name = cfg['name']
        kind = cfg['kind']

        if kind == 'global_edmd':
            g = fit_global_edmd(X_tr, F_tr, degree=cfg['degree'])
            F_pred = predict_global_edmd(X_te, g)
            one    = torch.linalg.norm(F_pred - F_te, dim=1).mean().item()
            roll   = eval_rollout(None, g, is_global=True)

        elif kind == 'local_edmd':
            hp = {**hp_base, 'sigma2': 'auto'}
            state, _, _ = fit_local_edmd(
                X_tr, F_tr, N=cfg['N'], hp=hp, degree=cfg['degree'],
                n_iter=args.n_iter, n_restarts=args.n_restarts, verbose=False,
            )
            F_pred = predict_f_local(X_te, state)
            one    = torch.linalg.norm(F_pred - F_te, dim=1).mean().item()
            roll   = eval_rollout(predict_f_local, state)

        elif kind == 'taylor_analytic':
            hp = {**hp_base, 'sigma2': 'auto'}
            state, _, _ = fit_taylor(
                X_tr, F_tr, duffing_f, duffing_J, N=cfg['N'], hp=hp,
                n_iter=args.n_iter, n_restarts=args.n_restarts, verbose=False,
            )
            F_pred = predict_f_taylor(X_te, state)
            one    = torch.linalg.norm(F_pred - F_te, dim=1).mean().item()
            roll   = eval_rollout(predict_f_taylor, state)

        elif kind == 'gmm_baseline':
            hp = {**hp_base, 'sigma2': 1e10}      # residual term -> 0
            state, _, _ = fit_taylor(
                X_tr, F_tr, duffing_f, duffing_J, N=cfg['N'], hp=hp,
                n_iter=args.n_iter, n_restarts=args.n_restarts, verbose=False,
            )
            F_pred = predict_f_taylor(X_te, state)
            one    = torch.linalg.norm(F_pred - F_te, dim=1).mean().item()
            roll   = eval_rollout(predict_f_taylor, state)
        else:
            raise ValueError(f"unknown kind: {kind}")

        results[name] = {
            'one_step': one,
            'r05s':     roll[5],
            'r10s':     roll[10],
            'r20s':     roll[20],
        }
    return results


# -- Configuration ------------------------------------------------------------

configs = []
for deg in args.edmd_degs:
    configs.append({'name': f"Global EDMD deg={deg}",
                    'kind': 'global_edmd', 'degree': deg})
for N in args.le2_N_list:
    configs.append({'name': f"local-EDMD d2 N={N}",
                    'kind': 'local_edmd', 'degree': 2, 'N': N})
for N in args.le3_N_list:
    configs.append({'name': f"local-EDMD d3 N={N}",
                    'kind': 'local_edmd', 'degree': 3, 'N': N})
for N in args.N_list:
    configs.append({'name': f"Taylor-analytic N={N}",
                    'kind': 'taylor_analytic', 'N': N})
for N in args.N_list:
    configs.append({'name': f"GMM-baseline N={N}",
                    'kind': 'gmm_baseline', 'N': N})


# -- Run all seeds ------------------------------------------------------------

all_runs = {}
for i, seed in enumerate(seeds):
    print(f"\n{'-'*60}")
    print(f"  SEED {seed} ({i+1}/{N_SEEDS})")
    print(f"{'-'*60}")
    seed_results = run_seed(seed, configs)
    for name, metrics in seed_results.items():
        if name not in all_runs:
            all_runs[name] = {m: [] for m in metrics}
        for m, v in metrics.items():
            all_runs[name][m].append(v)
    # progress: a few key numbers
    for cn in ['Taylor-analytic N=8', 'GMM-baseline N=8',
               f"Global EDMD deg={args.edmd_degs[0]}"]:
        if cn in seed_results:
            r = seed_results[cn]
            print(f"    {cn:<26} one={r['one_step']:.4f}  r10s={r['r10s']:.3f}")


# -- Report: mean +/- 95% CI --------------------------------------------------

print("\n" + "=" * 110)
print(f"DUFFING -- {N_SEEDS} seeds, mean +/- 95% CI")
print("=" * 110)
print(f"{'method':<26} {'one-step':>22}  {'r@5s':>22}  {'r@10s':>22}  {'r@20s':>22}")
print("-" * 110)
for cfg in configs:
    name = cfg['name']
    one_m,  one_hw,  *_ = confidence_interval(all_runs[name]['one_step'])
    r05_m,  r05_hw,  *_ = confidence_interval(all_runs[name]['r05s'])
    r10_m,  r10_hw,  *_ = confidence_interval(all_runs[name]['r10s'])
    r20_m,  r20_hw,  *_ = confidence_interval(all_runs[name]['r20s'])
    print(f"{name:<26} "
          f"{one_m:>10.5f} +/- {one_hw:.5f}  "
          f"{r05_m:>10.4f} +/- {r05_hw:.4f}  "
          f"{r10_m:>10.4f} +/- {r10_hw:.4f}  "
          f"{r20_m:>10.4f} +/- {r20_hw:.4f}")


# -- Paired t-tests (key comparisons) -----------------------------------------

print("\n" + "=" * 110)
print("PAIRED T-TESTS")
print("=" * 110)

test_pairs = []
# Residual-aware Taylor vs geometric GMM at matched N
for N in args.N_list:
    test_pairs.append((f"Taylor-analytic N={N}", f"GMM-baseline N={N}",
                       f"Taylor-ana vs GMM at N={N}"))
# Taylor-analytic at the largest N vs every Global EDMD degree
for deg in args.edmd_degs:
    test_pairs.append((f"Taylor-analytic N={args.N_list[-1]}",
                       f"Global EDMD deg={deg}",
                       f"Taylor N={args.N_list[-1]} vs Global deg={deg}"))
# Local-EDMD deg-2 at largest N vs Global EDMD deg=2
if args.le2_N_list and args.edmd_degs:
    test_pairs.append((f"local-EDMD d2 N={args.le2_N_list[-1]}",
                       f"Global EDMD deg={args.edmd_degs[0]}",
                       f"local-EDMD d2 N={args.le2_N_list[-1]} vs Global deg={args.edmd_degs[0]}"))


def _sig(p):
    return ("***" if p < 0.001 else "**" if p < 0.01 else
            "*"   if p < 0.05  else "ns")


for name_a, name_b, label in test_pairs:
    if name_a not in all_runs or name_b not in all_runs:
        continue
    print(f"\n  {label}:")
    for metric in ('one_step', 'r10s', 'r20s'):
        a = np.array(all_runs[name_a][metric])
        b = np.array(all_runs[name_b][metric])
        t = paired_test(a, b)
        print(f"    [{metric:>8}]  {a.mean():>10.4f}  vs  {b.mean():>10.4f}  "
              f"diff={t['mean_diff']:+10.4f}  p={t['p_value']:.4f}  {_sig(t['p_value'])}")


# -- Save raw per-seed data ---------------------------------------------------

raw = {
    'system':  'duffing_unforced_2d',
    'parameters': {'DELTA': DELTA, 'ALPHA': -1.0, 'BETA': 1.0},
    'n_seeds': N_SEEDS,
    'seeds':   seeds,
    'args':    {k: (v if isinstance(v, (int, float, str, bool, list, tuple, type(None)))
                    else str(v))
                for k, v in vars(args).items()},
    'configs': configs,
    'results': {name: {m: vals for m, vals in metrics.items()}
                for name, metrics in all_runs.items()},
}
with open(data_path("duffing_statistical.json"), "w") as fp:
    json.dump(raw, fp, indent=2)
print(f"\n-> saved raw per-seed data: {data_path('duffing_statistical.json')}")


# -- Plot with error bars -----------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))
groups = [
    ('Global EDMD',     'Global EDMD',     'C3', 'D'),
    ('local-EDMD d2',   'local-EDMD d2',   'C0', 'o'),
    ('local-EDMD d3',   'local-EDMD d3',   'C1', 's'),
    ('Taylor-analytic', 'Taylor-analytic', 'C2', '^'),
    ('GMM-baseline',    'GMM-baseline',    'C4', 'v'),
]

def _params(cfg):
    """Approximate parameter count for sorting the x-axis."""
    if cfg['kind'] == 'global_edmd':
        m = len(monomial_exponents(d, cfg['degree']))
        return m * m
    if cfg['kind'] == 'local_edmd':
        m = len(monomial_exponents(d, cfg['degree']))
        return cfg['N'] * m * m
    return cfg['N'] * (d * d + d)        # Taylor / GMM (Taylor local model)


for ax, metric, ylabel, title in [
    (axes[0], 'one_step', r'mean $\|f_{pred}-f_{true}\|$ on test set',
     f"One-step f-error (mean +/- 95% CI, n={N_SEEDS})"),
    (axes[1], 'r10s',     r'rollout L2 error at $t=10$ s',
     f"Rollout @ 10 s (mean +/- 95% CI, n={N_SEEDS})"),
]:
    for label, prefix, color, marker in groups:
        members = [c for c in configs if c['name'].startswith(prefix)]
        if not members:
            continue
        members = sorted(members, key=_params)
        xs    = [_params(c) for c in members]
        means = [confidence_interval(all_runs[c['name']][metric])[0] for c in members]
        cis   = [confidence_interval(all_runs[c['name']][metric])[1] for c in members]
        ax.errorbar(xs, means, yerr=cis, fmt=f'{marker}-', color=color,
                    label=label, capsize=4, markersize=6)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('# parameters')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(fig_path("duffing_statistical.png"), dpi=130)
print(f"-> saved figure: {fig_path('duffing_statistical.png')}")

print("\n" + "=" * 80)
print("  Done.")
print("=" * 80)
