"""
Pendulum validation — THE test where local-EDMD should beat global-EDMD.

System: damped pendulum, 2D, f(θ, θ̇) = (θ̇, -sin(θ) - γ·θ̇).
The sin(θ) is non-polynomial, so polynomial EDMD must approximate it by
Taylor — which degrades badly over wide θ ranges. Local EDMD with
clusters covering sub-intervals of θ can each do Taylor locally.

Comparisons:
  * Global EDMD at degrees 2, 4, 6, 8 (via custom continuous EDMD,
    using the same vector-field-regression formulation as local EDMD)
  * Local EDMD deg-2 at N ∈ {2, 4, 8, 16}
  * Local EDMD deg-4 at N ∈ {2, 4, 8}
  * Taylor piecewise-linear at matched N

Metrics:
  * One-step f-prediction error on held-out test set
  * Trajectory rollout error at t = 2s, 5s, 10s (angular distance metric)
"""

import argparse

parser = argparse.ArgumentParser(description="Pendulum validation: local-EDMD vs global-EDMD vs Taylor")
parser.add_argument('--train-seed', type=int, default=42)
parser.add_argument('--test-seed', type=int, default=17)
parser.add_argument('--n-train', type=int, default=4000)
parser.add_argument('--n-test', type=int, default=1000)
parser.add_argument('--n-iter', type=int, default=60)
parser.add_argument('--n-restarts', type=int, default=2)
parser.add_argument('--dt', type=float, default=0.05)
parser.add_argument('--rollout-steps', type=int, default=200)
args = parser.parse_args()

import numpy as np
from utils.paths import fig_path, data_path
import torch
import matplotlib.pyplot as plt

from simulators.pendulum import (
    f as pendulum_f,
    J as pendulum_J,
    sample_phase_space,
    generate_trajectory,
    wrap_theta,
    angular_dist,
)
from models.em import fit as fit_taylor
from models.em_hybrid import fit_hybrid
from models.em_local_edmd import (
    fit as fit_local_edmd,
    predict_f_all_clusters,
    monomial_exponents,
    monomials,
    monomials_grad,
    weighted_continuous_edmd,
)
from models.distributions import mvn_logpdf_batch

torch.set_default_dtype(torch.float64)
np.random.seed(0)

# ── Training + test data ─────────────────────────────────────────────────────
train = sample_phase_space(n_samples=args.n_train, seed=args.train_seed)
test  = sample_phase_space(n_samples=args.n_test, seed=args.test_seed)

X_tr = torch.tensor(train['X'], dtype=torch.float64)
F_tr = torch.tensor(train['F'], dtype=torch.float64)
X_te = torch.tensor(test ['X'], dtype=torch.float64)
F_te = torch.tensor(test ['F'], dtype=torch.float64)

d  = 2
dt = args.dt

hp_base = {
    'alpha0': 0.5, 'mu0': X_tr.mean(dim=0),
    'Lambda0': 0.01 * torch.eye(d, dtype=torch.float64),
    'kappa0': 1.0, 'Psi0': 1.0 * torch.eye(d, dtype=torch.float64),
    'nu0': float(d + 2), 'sigma2': 'auto',
}

# ── Global continuous EDMD (implemented identically to local, with N=1) ──────
def fit_global_edmd(X_tr, F_tr, degree, ridge=1e-6):
    """Single global operator M in lifted space, no clustering."""
    exps = monomial_exponents(d, degree)
    c = X_tr.mean(dim=0)
    r = torch.ones(X_tr.shape[0], dtype=torch.float64)
    M = weighted_continuous_edmd(X_tr, F_tr, r, c, exps, ridge=ridge)
    return {'M': M, 'c': c, 'exps': exps}

def predict_global_edmd(X, g):
    U       = X - g['c']
    Phi     = monomials(U, g['exps'])
    Phi_dot = Phi @ g['M'].T
    return Phi_dot[:, 1:d+1]

# ── One-step f-error helpers ─────────────────────────────────────────────────
def onestep_err_global(g):
    F_pred = predict_global_edmd(X_te, g)
    return torch.linalg.norm(F_pred - F_te, dim=1).mean().item()

def pick_cluster(x, state):
    log_pi   = torch.log(state['pi']).unsqueeze(0)
    log_prox = mvn_logpdf_batch(x, state['centers'], state['covariances'])
    return (log_pi + log_prox).argmax(dim=1)

def predict_f_local_edmd(x, state):
    k = pick_cluster(x, state)
    F_all = predict_f_all_clusters(x, state['centers'], state['M_ops'],
                                   state['exps'], d)
    return F_all[torch.arange(x.shape[0]), k]

def onestep_err_local(state):
    F_pred = predict_f_local_edmd(X_te, state)
    return torch.linalg.norm(F_pred - F_te, dim=1).mean().item()

def predict_f_taylor(x, state):
    k  = pick_cluster(x, state)
    c  = state['centers'][k]
    fc = state['f_centers'][k]
    J  = state['jacobians'][k]
    return fc + (J @ (x - c).unsqueeze(-1)).squeeze(-1)

def onestep_err_taylor(state):
    F_pred = predict_f_taylor(X_te, state)
    return torch.linalg.norm(F_pred - F_te, dim=1).mean().item()

# ── Rollout helpers with angular wrapping ────────────────────────────────────
def euler_step_wrap(x, f_val):
    x_new = x + dt * f_val
    x_new = wrap_theta(x_new)
    return x_new

def rollout_global(x0, g, n_steps):
    traj = torch.zeros(n_steps + 1, d, dtype=torch.float64)
    traj[0] = x0
    for t in range(n_steps):
        f_hat = predict_global_edmd(traj[t:t+1], g)[0]
        traj[t+1] = euler_step_wrap(traj[t], f_hat)
    return traj

def rollout_local(x0, state, n_steps):
    traj = torch.zeros(n_steps + 1, d, dtype=torch.float64)
    traj[0] = x0
    for t in range(n_steps):
        f_hat = predict_f_local_edmd(traj[t:t+1], state)[0]
        traj[t+1] = euler_step_wrap(traj[t], f_hat)
    return traj

def rollout_taylor(x0, state, n_steps):
    traj = torch.zeros(n_steps + 1, d, dtype=torch.float64)
    traj[0] = x0
    for t in range(n_steps):
        f_hat = predict_f_taylor(traj[t:t+1], state)[0]
        traj[t+1] = euler_step_wrap(traj[t], f_hat)
    return traj

# Starting points for rollout — covering different dynamical regimes
rollout_inits = [
    torch.tensor([ 0.3, 0.0]),  # small oscillation
    torch.tensor([ 1.5, 0.0]),  # large oscillation
    torch.tensor([ 2.8, 0.0]),  # near inverted, will swing down
    torch.tensor([ 0.0, 2.5]),  # fast spin, energy decays to oscillation
    torch.tensor([-2.0, 1.0]),
]
n_roll = args.rollout_steps   # 10 seconds at dt=0.05

def rollout_metrics(rollout_fn, model):
    errs = {50: [], 100: [], 200: []}
    for x0 in rollout_inits:
        tru = torch.tensor(generate_trajectory(x0.numpy(), n_steps=n_roll, dt=dt),
                           dtype=torch.float64)
        if model is None:
            continue
        sim = rollout_fn(x0, model, n_roll)
        err = angular_dist(sim, tru)
        errs[50 ].append(err[50 ].item())
        errs[100].append(err[100].item())
        errs[200].append(err[200].item())
    return tuple(np.mean(errs[k]) for k in (50, 100, 200))

# ── Run all experiments ──────────────────────────────────────────────────────
rows = []

# Global EDMD at various degrees
for deg in (2, 4, 6, 8):
    g = fit_global_edmd(X_tr, F_tr, degree=deg)
    M = len(g['exps'])
    n_params = M * M
    one = onestep_err_global(g)
    r = rollout_metrics(rollout_global, g)
    rows.append(dict(name=f"Global EDMD deg={deg}", N=1, params=n_params,
                     one=one, r50=r[0], r100=r[1], r200=r[2]))
    print(f"  global deg={deg}: params={n_params:4d}, one-step={one:.4f}, "
          f"rollout@10s={r[2]:.3f}")

# Local EDMD
for deg, N_list in [(2, [2, 4, 8, 16]), (4, [2, 4, 8])]:
    M = len(monomial_exponents(d, deg))
    for N in N_list:
        hp = dict(hp_base); hp['sigma2'] = 'auto'
        print(f"\n  local deg={deg} N={N} ...")
        state, _, _ = fit_local_edmd(X_tr, F_tr, N=N, hp=hp, degree=deg,
                                     n_iter=args.n_iter, n_restarts=args.n_restarts, verbose=False)
        one = onestep_err_local(state)
        r   = rollout_metrics(rollout_local, state)
        rows.append(dict(name=f"local-EDMD deg={deg} N={N}", N=state['N'],
                         params=state['N'] * M * M,
                         one=one, r50=r[0], r100=r[1], r200=r[2]))
        print(f"    params={state['N']*M*M}, one-step={one:.4f}, rollout@10s={r[2]:.3f}")

# Taylor piecewise (analytic J(c_k), f(c_k))
print()
for N in (2, 4, 8, 16):
    hp = dict(hp_base); hp['sigma2'] = 'auto'
    print(f"  Taylor-analytic N={N} ...")
    state, _, _ = fit_taylor(X_tr, F_tr, pendulum_f, pendulum_J,
                             N=N, hp=hp, n_iter=args.n_iter, n_restarts=args.n_restarts, verbose=False)
    one = onestep_err_taylor(state)
    r   = rollout_metrics(rollout_taylor, state)
    rows.append(dict(name=f"Taylor-analytic N={N}", N=state['N'],
                     params=state['N'] * (d*d + d),
                     one=one, r50=r[0], r100=r[1], r200=r[2]))
    print(f"    params={state['N']*(d*d+d)}, one-step={one:.4f}, rollout@10s={r[2]:.3f}")

# Hybrid — same state dict shape as Taylor, but J_k and f_k are LS-fit per cluster
print()
for N in (2, 4, 8, 16):
    hp = dict(hp_base); hp['sigma2'] = 'auto'
    print(f"  Taylor-LS N={N} ...")
    state, _, _ = fit_hybrid(X_tr, F_tr, pendulum_f, pendulum_J,
                             N=N, hp=hp, n_iter=args.n_iter, n_restarts=args.n_restarts, verbose=False)
    one = onestep_err_taylor(state)   # predict uses same state shape (f_centers, jacobians)
    r   = rollout_metrics(rollout_taylor, state)
    rows.append(dict(name=f"Taylor-LS N={N}", N=state['N'],
                     params=state['N'] * (d*d + d),
                     one=one, r50=r[0], r100=r[1], r200=r[2]))
    print(f"    params={state['N']*(d*d+d)}, one-step={one:.4f}, rollout@10s={r[2]:.3f}")

# ── Report ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 100)
print("PENDULUM — one-step f-error + rollout error (angular distance, avg 5 ICs)")
print("=" * 100)
print(f"{'method':<28} {'N':>3} {'params':>7}  {'one-step':>10}  "
      f"{'r@2.5s':>8} {'r@5s':>8} {'r@10s':>8}")
print("-" * 100)
for r in rows:
    print(f"{r['name']:<28} {r['N']:>3} {r['params']:>7}  "
          f"{r['one']:>10.4f}  {r['r50']:>8.3f} {r['r100']:>8.3f} {r['r200']:>8.3f}")
print("=" * 100)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

for ax, metric, ylabel, title in [
    (axes[0], 'one',  r'mean $\|f_{pred} - f_{true}\|$ on test set', 'One-step f-error vs params'),
    (axes[1], 'r200', r'angular dist at t=10s',                     'Rollout error @ t=10s vs params'),
]:
    globals_r  = [r for r in rows if r['name'].startswith('Global')]
    local2     = [r for r in rows if 'local-EDMD deg=2' in r['name']]
    local4     = [r for r in rows if 'local-EDMD deg=4' in r['name']]
    taylor_ana = [r for r in rows if 'Taylor-analytic' in r['name']]
    taylor_ls  = [r for r in rows if 'Taylor-LS'       in r['name']]
    ax.loglog([r['params'] for r in globals_r],  [r[metric] for r in globals_r],
              'D-', label='global EDMD',        color='C3', markersize=8)
    ax.loglog([r['params'] for r in local2],    [r[metric] for r in local2],
              'o-', label='local-EDMD deg-2',   color='C0')
    ax.loglog([r['params'] for r in local4],    [r[metric] for r in local4],
              's-', label='local-EDMD deg-4',   color='C1')
    ax.loglog([r['params'] for r in taylor_ana],[r[metric] for r in taylor_ana],
              '^-', label='Taylor-analytic',    color='C2')
    ax.loglog([r['params'] for r in taylor_ls], [r[metric] for r in taylor_ls],
              'v-', label='Taylor-LS (data-only)', color='C4')
    ax.set_xlabel('# parameters')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(fig_path("pendulum_comparison.png"), dpi=120)
print("\n→ saved pendulum_comparison.png")

# ── Also plot a sample rollout visually ──────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(7, 6))
x0 = rollout_inits[2]
tru = torch.tensor(generate_trajectory(x0.numpy(), n_steps=n_roll, dt=dt), dtype=torch.float64)
ax.plot(tru[:, 0].numpy(), tru[:, 1].numpy(), 'k-', lw=2, label='truth')

# Re-fit best-of-each for the plot
g2 = fit_global_edmd(X_tr, F_tr, degree=2); sim = rollout_global(x0, g2, n_roll)
ax.plot(sim[:, 0].numpy(), sim[:, 1].numpy(), '-', color='C3', alpha=0.7, label='global deg-2')
g8 = fit_global_edmd(X_tr, F_tr, degree=8); sim = rollout_global(x0, g8, n_roll)
ax.plot(sim[:, 0].numpy(), sim[:, 1].numpy(), '--', color='C3', alpha=0.7, label='global deg-8')

hp = dict(hp_base); hp['sigma2'] = 'auto'
st, _, _ = fit_local_edmd(X_tr, F_tr, N=4, hp=hp, degree=2, n_iter=args.n_iter, n_restarts=args.n_restarts, verbose=False)
sim = rollout_local(x0, st, n_roll)
ax.plot(sim[:, 0].numpy(), sim[:, 1].numpy(), '-', color='C0', alpha=0.9, label='local deg-2 N=4')

ax.scatter(*x0.numpy(), s=80, marker='*', color='k', zorder=5, label='start')
ax.set_xlabel(r'$\theta$'); ax.set_ylabel(r'$\dot\theta$')
ax.set_title(f"Rollout from θ={x0[0]:.1f}, θ̇={x0[1]:.1f}  (10 seconds)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(fig_path("pendulum_rollout.png"), dpi=120)
print("→ saved pendulum_rollout.png")

# ── Save raw data ────────────────────────────────────────────────────────────
import json
raw = {
    "train_seed": args.train_seed,
    "test_seed": args.test_seed,
    "rows": rows,
}
with open(data_path("validation_pendulum.json"), "w") as fp:
    json.dump(raw, fp, indent=2)
print(f"Raw data saved to {data_path('validation_pendulum.json')}")
