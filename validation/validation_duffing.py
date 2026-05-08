"""
Duffing validation -- prediction-error comparison.

System: unforced double-well Duffing oscillator,

    x_ddot + DELTA*x_dot - x + x**3 = 0,

with stable foci at (+/-1, 0) and an index-1 saddle at the origin.
The cubic term makes the vector field smoothly nonlinear; a single
global EDMD model needs high polynomial degree to track the dynamics
across the box, while local Koopman / Taylor models can do it with
far fewer parameters per cluster.

Comparisons (mirroring the Lorenz / pendulum precedent):
  * Global EDMD at degrees 2, 3, 4, 5 (single operator over the whole
    domain, fit by weighted continuous EDMD with N=1).
  * Local EDMD deg-2 at N in {2, 4, 8, 16}.
  * Local EDMD deg-3 at N in {2, 4, 8}.
  * Taylor-analytic at N in {2, 4, 8, 16} -- residual-aware, physics
    knowledge of f, J.
  * GMM baseline (Taylor-analytic local model with sigma2 -> inf so
    the residual term vanishes; same EM machinery, geometric partition).

Metrics:
  * One-step f-prediction error on a held-out test set sampled
    independently from the same distribution.
  * Rollout error in (x, x_dot) at t = 5, 10, 20 from a small set of
    diagnostic initial conditions covering both basins and the saddle
    region.

Outputs
-------
- papers/figures/duffing_comparison.png    -- error vs # parameters log-log
- papers/figures/duffing_rollout.png       -- sample phase-plane rollouts
- papers/data/validation_duffing.json      -- raw results table
"""

import argparse

parser = argparse.ArgumentParser(
    description="Duffing (unforced double-well) -- prediction error vs parameters"
)
parser.add_argument('--train-seed', type=int, default=42)
parser.add_argument('--test-seed',  type=int, default=17)
parser.add_argument('--n-train',    type=int, default=4000)
parser.add_argument('--n-test',     type=int, default=1000)
parser.add_argument('--n-iter',     type=int, default=80)
parser.add_argument('--n-restarts', type=int, default=2)
parser.add_argument('--dt',         type=float, default=0.05)
parser.add_argument('--rollout-steps', type=int, default=400)
args = parser.parse_args()

import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.paths import fig_path, data_path
from simulators.duffing import (
    f as duffing_f,
    J as duffing_J,
    sample_phase_space,
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
np.random.seed(0)


# -- Training and test data ---------------------------------------------------

train = sample_phase_space(n_samples=args.n_train, x_max=2.0, xdot_max=2.0,
                           seed=args.train_seed)
test  = sample_phase_space(n_samples=args.n_test,  x_max=2.0, xdot_max=2.0,
                           seed=args.test_seed)

X_tr = torch.tensor(train['X'], dtype=torch.float64)
F_tr = torch.tensor(train['F'], dtype=torch.float64)
X_te = torch.tensor(test ['X'], dtype=torch.float64)
F_te = torch.tensor(test ['F'], dtype=torch.float64)

d  = 2
dt = args.dt

hp_base = {
    'alpha0':  0.5,
    'mu0':     X_tr.mean(dim=0),
    'Lambda0': 0.01 * torch.eye(d, dtype=torch.float64),
    'kappa0':  1.0,
    'Psi0':    1.0  * torch.eye(d, dtype=torch.float64),
    'nu0':     float(d + 2),
}
hp_our = lambda: {**hp_base, 'sigma2': 'auto'}
hp_gmm = lambda: {**hp_base, 'sigma2': 1e10}     # residual term -> 0


# -- Global continuous EDMD (N=1, single operator over whole domain) ---------

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


# -- One-step f-error helpers (same machinery as validation_pendulum.py) -----

def pick_cluster(X_pts, state):
    log_pi   = torch.log(state['pi']).unsqueeze(0)
    log_prox = mvn_logpdf_batch(X_pts, state['centers'], state['covariances'])
    return (log_pi + log_prox).argmax(dim=1)


def predict_f_local_edmd(X_pts, state):
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


def onestep_err(F_pred):
    return torch.linalg.norm(F_pred - F_te, dim=1).mean().item()


# -- Rollout helpers (Euler integration of predicted f) ----------------------

def euler_step(x, f_val):
    return x + dt * f_val


def rollout_global(x0, g, n_steps):
    traj    = torch.zeros(n_steps + 1, d, dtype=torch.float64)
    traj[0] = x0
    for t in range(n_steps):
        f_hat       = predict_global_edmd(traj[t:t+1], g)[0]
        traj[t + 1] = euler_step(traj[t], f_hat)
    return traj


def rollout_local(x0, state, n_steps):
    traj    = torch.zeros(n_steps + 1, d, dtype=torch.float64)
    traj[0] = x0
    for t in range(n_steps):
        f_hat       = predict_f_local_edmd(traj[t:t+1], state)[0]
        traj[t + 1] = euler_step(traj[t], f_hat)
    return traj


def rollout_taylor(x0, state, n_steps):
    traj    = torch.zeros(n_steps + 1, d, dtype=torch.float64)
    traj[0] = x0
    for t in range(n_steps):
        f_hat       = predict_f_taylor(traj[t:t+1], state)[0]
        traj[t + 1] = euler_step(traj[t], f_hat)
    return traj


# Diagnostic initial conditions covering both basins and the saddle region
rollout_inits = [
    torch.tensor([+1.5, 0.0]),     # start in right basin, settle at +1
    torch.tensor([-1.5, 0.0]),     # start in left  basin, settle at -1
    torch.tensor([+0.3, 0.0]),     # near saddle, right side
    torch.tensor([-0.3, 0.0]),     # near saddle, left side
    torch.tensor([+0.5, 1.5]),     # high-velocity transient
]
n_roll = args.rollout_steps        # 20 s at dt=0.05


def rollout_metrics(rollout_fn, model):
    """Mean Euclidean rollout error at three horizons, averaged over inits."""
    errs = {100: [], 200: [], 400: []}
    for x0 in rollout_inits:
        tru = torch.tensor(generate_trajectory(x0.numpy(), n_steps=n_roll, dt=dt),
                           dtype=torch.float64)
        sim = rollout_fn(x0, model, n_roll)
        err = torch.linalg.norm(sim - tru, dim=1)
        for k in errs:
            errs[k].append(err[k].item())
    return tuple(np.mean(errs[k]) for k in (100, 200, 400))


# -- Run all experiments ------------------------------------------------------

rows = []

# Global EDMD at various degrees (single operator over the whole domain)
print("\nGlobal EDMD (single operator, no clustering)")
for deg in (2, 3, 4, 5):
    g    = fit_global_edmd(X_tr, F_tr, degree=deg)
    Mn   = len(g['exps'])
    nP   = Mn * Mn
    one  = onestep_err(predict_global_edmd(X_te, g))
    r    = rollout_metrics(rollout_global, g)
    rows.append(dict(name=f"global EDMD deg={deg}", method='global',
                     N=1, params=nP, one=one, r100=r[0], r200=r[1], r400=r[2]))
    print(f"  deg={deg}: params={nP:4d}, one-step={one:.4f}, "
          f"rollout@5s={r[0]:.3f}, @10s={r[1]:.3f}, @20s={r[2]:.3f}")

# Local EDMD
print("\nLocal EDMD (residual-aware)")
for deg, N_list in [(2, [2, 4, 8, 16]), (3, [2, 4, 8])]:
    Mn = len(monomial_exponents(d, deg))
    for N in N_list:
        print(f"  deg={deg} N={N} ...")
        state, _, _ = fit_local_edmd(
            X_tr, F_tr, N=N, hp=hp_our(), degree=deg,
            n_iter=args.n_iter, n_restarts=args.n_restarts, verbose=False,
        )
        nP  = state['N'] * Mn * Mn
        one = onestep_err(predict_f_local_edmd(X_te, state))
        r   = rollout_metrics(rollout_local, state)
        rows.append(dict(name=f"local-EDMD deg={deg} N={N}", method='local',
                         N=state['N'], params=nP,
                         one=one, r100=r[0], r200=r[1], r400=r[2]))
        print(f"    active={state['N']}, params={nP}, one-step={one:.4f}, "
              f"rollout@10s={r[1]:.3f}")

# Taylor-analytic (residual-aware, exact f and J at centers)
print("\nTaylor-analytic (residual-aware, exact J)")
for N in (2, 4, 8, 16):
    print(f"  N={N} ...")
    state, _, _ = fit_taylor(
        X_tr, F_tr, duffing_f, duffing_J, N=N, hp=hp_our(),
        n_iter=args.n_iter, n_restarts=args.n_restarts, verbose=False,
    )
    nP  = state['N'] * (d * d + d)        # J_k (d x d) + f(c_k) (d)
    one = onestep_err(predict_f_taylor(X_te, state))
    r   = rollout_metrics(rollout_taylor, state)
    rows.append(dict(name=f"Taylor-analytic N={N}", method='taylor',
                     N=state['N'], params=nP,
                     one=one, r100=r[0], r200=r[1], r400=r[2]))
    print(f"    active={state['N']}, params={nP}, one-step={one:.4f}, "
          f"rollout@10s={r[1]:.3f}")

# GMM baseline (Taylor local model + sigma2 -> inf removes residual term)
print("\nGMM baseline (Taylor local model, sigma2 -> inf)")
for N in (2, 4, 8, 16):
    print(f"  N={N} ...")
    state, _, _ = fit_taylor(
        X_tr, F_tr, duffing_f, duffing_J, N=N, hp=hp_gmm(),
        n_iter=args.n_iter, n_restarts=args.n_restarts, verbose=False,
    )
    nP  = state['N'] * (d * d + d)
    one = onestep_err(predict_f_taylor(X_te, state))
    r   = rollout_metrics(rollout_taylor, state)
    rows.append(dict(name=f"GMM-baseline N={N}", method='gmm',
                     N=state['N'], params=nP,
                     one=one, r100=r[0], r200=r[1], r400=r[2]))
    print(f"    active={state['N']}, params={nP}, one-step={one:.4f}, "
          f"rollout@10s={r[1]:.3f}")


# -- Report -------------------------------------------------------------------

print("\n" + "=" * 100)
print("DUFFING -- one-step f-error + rollout error (5 ICs)")
print("=" * 100)
print(f"{'method':<28} {'N':>3} {'params':>7}  {'one-step':>10}  "
      f"{'r@5s':>8} {'r@10s':>8} {'r@20s':>8}")
print("-" * 100)
for r in rows:
    print(f"{r['name']:<28} {r['N']:>3} {r['params']:>7}  "
          f"{r['one']:>10.4f}  {r['r100']:>8.3f} {r['r200']:>8.3f} {r['r400']:>8.3f}")
print("=" * 100)


# -- Plot 1: error vs # parameters (log-log) ---------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, metric, ylabel, title in [
    (axes[0], 'one',  r'mean $\|f_{pred} - f_{true}\|$ on test set',
     'One-step f-error vs # parameters'),
    (axes[1], 'r200', r'rollout L2 error at $t=10$',
     'Rollout error @ 10 s vs # parameters'),
]:
    grp = lambda key: [r for r in rows if r['method'] == key]
    ax.loglog([r['params'] for r in grp('global')],
              [r[metric]   for r in grp('global')],
              'D-', color='C3', markersize=8, label='global EDMD')
    le2 = [r for r in grp('local') if 'deg=2' in r['name']]
    le3 = [r for r in grp('local') if 'deg=3' in r['name']]
    ax.loglog([r['params'] for r in le2], [r[metric] for r in le2],
              'o-', color='C0', label='local-EDMD deg-2')
    ax.loglog([r['params'] for r in le3], [r[metric] for r in le3],
              's-', color='C1', label='local-EDMD deg-3')
    ax.loglog([r['params'] for r in grp('taylor')],
              [r[metric]   for r in grp('taylor')],
              '^-', color='C2', label='Taylor-analytic')
    ax.loglog([r['params'] for r in grp('gmm')],
              [r[metric]   for r in grp('gmm')],
              'v-', color='C4', label='GMM baseline (geometric)')
    ax.set_xlabel('# parameters')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(fig_path("duffing_comparison.png"), dpi=130)
print(f"\n-> saved {fig_path('duffing_comparison.png')}")


# -- Plot 2: phase-plane rollouts from a few diagnostic initial conditions ---

fig, ax = plt.subplots(1, 1, figsize=(7.0, 6.4))
x0  = rollout_inits[4]                          # high-velocity transient
tru = torch.tensor(generate_trajectory(x0.numpy(), n_steps=n_roll, dt=dt),
                   dtype=torch.float64)
ax.plot(tru[:, 0], tru[:, 1], 'k-', lw=2.0, label='truth')

g3  = fit_global_edmd(X_tr, F_tr, degree=3)
sim = rollout_global(x0, g3, n_roll)
ax.plot(sim[:, 0], sim[:, 1], '-', color='C3', alpha=0.85, label='global EDMD deg-3')

g5  = fit_global_edmd(X_tr, F_tr, degree=5)
sim = rollout_global(x0, g5, n_roll)
ax.plot(sim[:, 0], sim[:, 1], '--', color='C3', alpha=0.85, label='global EDMD deg-5')

st_le, _, _ = fit_local_edmd(X_tr, F_tr, N=4, hp=hp_our(), degree=2,
                             n_iter=args.n_iter, n_restarts=args.n_restarts, verbose=False)
sim = rollout_local(x0, st_le, n_roll)
ax.plot(sim[:, 0], sim[:, 1], '-', color='C0', alpha=0.95,
        label=f"local-EDMD deg-2 N={st_le['N']}")

st_t, _, _  = fit_taylor(X_tr, F_tr, duffing_f, duffing_J, N=8, hp=hp_our(),
                          n_iter=args.n_iter, n_restarts=args.n_restarts, verbose=False)
sim = rollout_taylor(x0, st_t, n_roll)
ax.plot(sim[:, 0], sim[:, 1], '-', color='C2', alpha=0.95,
        label=f"Taylor-analytic N={st_t['N']}")

ax.scatter(*x0.numpy(), s=110, marker='*', color='black', zorder=5, label='start')
ax.plot([-1, +1], [0, 0], 'o', color='white', mec='black', mew=1.2, ms=8,
        label='stable foci')
ax.plot([0], [0], 'X', color='red', mec='black', mew=1.0, ms=10, label='saddle')

ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$\dot x$')
ax.set_title(rf"Duffing rollout from $(x, \dot x) = ({x0[0]:.1f}, {x0[1]:.1f})$  "
             rf"($t = {n_roll * dt:.0f}$ s)")
ax.legend(fontsize=9, loc='best')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(fig_path("duffing_rollout.png"), dpi=130)
print(f"-> saved {fig_path('duffing_rollout.png')}")


# -- Save raw data ------------------------------------------------------------

raw = {
    'system': 'duffing_unforced_2d',
    'parameters': {'DELTA': DELTA, 'ALPHA': -1.0, 'BETA': 1.0},
    'config': {
        'train_seed':  args.train_seed,
        'test_seed':   args.test_seed,
        'n_train':     args.n_train,
        'n_test':      args.n_test,
        'n_iter':      args.n_iter,
        'n_restarts':  args.n_restarts,
        'dt':          args.dt,
        'rollout_steps': args.rollout_steps,
    },
    'rows': rows,
}
with open(data_path("validation_duffing.json"), "w") as fp:
    json.dump(raw, fp, indent=2)
print(f"-> saved {data_path('validation_duffing.json')}")
