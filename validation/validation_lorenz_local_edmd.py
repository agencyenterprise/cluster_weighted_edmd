"""
Test the hypothesis:
  "Local EDMD per cluster should give better rollouts on Lorenz, because
   each region has its own good-enough local approximation."

Comparison:
  * Global EDMD deg-2, deg-3  (pykoopman)
  * Local EDMD deg-2 at N = 2, 3, 5, 10, 20
  * Local EDMD deg-3 at N = 2, 3, 5
  * Taylor piecewise-linear at matched N (for reference)

Metrics:
  * One-step state error on held-out test set
  * Rollout error @ 50 / 200 / 500 steps from 5 test initial conditions
  * Parameter count
"""

import argparse

parser = argparse.ArgumentParser(description="Local EDMD vs global EDMD vs Taylor piecewise-linear")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--n-steps', type=int, default=5000)
parser.add_argument('--dt', type=float, default=0.01)
parser.add_argument('--warmup', type=int, default=1000)
parser.add_argument('--n-train', type=int, default=4000)
parser.add_argument('--n-iter', type=int, default=80)
parser.add_argument('--n-restarts', type=int, default=2)
args = parser.parse_args()

import numpy as np
from utils.paths import fig_path, data_path
import torch
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import pykoopman as pk
from models.global_edmd import fit as fit_global_ours, predict_f as predict_f_global_ours

from simulators.lorenz import generate_data, f as lorenz_f, J as lorenz_J
from models.em import fit as fit_taylor
from models.em_local_edmd import (
    fit as fit_local_edmd,
    predict_f_all_clusters,
    monomial_exponents,
)
from models.distributions import mvn_logpdf_batch

torch.set_default_dtype(torch.float64)

# ── Data ─────────────────────────────────────────────────────────────────────
data  = generate_data(n_steps=args.n_steps, dt=args.dt, warmup=args.warmup, seed=args.seed)
X_all = torch.tensor(data['X'], dtype=torch.float64)
F_all = torch.tensor(data['F'], dtype=torch.float64)
dt    = args.dt
d     = 3

X_tr, X_te = X_all[:args.n_train], X_all[args.n_train:]
F_tr, F_te = F_all[:args.n_train], F_all[args.n_train:]
X_te_in    = X_all[args.n_train:args.n_steps-1]
X_te_next  = X_all[args.n_train+1:args.n_steps]
step_baseline = torch.linalg.norm(X_te_next - X_te_in, dim=1).mean().item()

hp_base = {
    'alpha0': 0.5, 'mu0': X_tr.mean(dim=0),
    'Lambda0': 0.01 * torch.eye(d, dtype=torch.float64),
    'kappa0': 1.0, 'Psi0': 10.0 * torch.eye(d, dtype=torch.float64),
    'nu0': float(d + 2), 'sigma2': 'auto',
}

# ── Switching piecewise integrator (local EDMD) ──────────────────────────────

def pick_cluster(x, state):
    log_pi   = torch.log(state['pi']).unsqueeze(0)
    log_prox = mvn_logpdf_batch(x, state['centers'], state['covariances'])
    return (log_pi + log_prox).argmax(dim=1)

def predict_f_local_edmd(x, state):
    """Vector-field prediction from local EDMD, picking cluster by proximity."""
    k = pick_cluster(x, state)                              # (P,)
    F_all_k = predict_f_all_clusters(
        x, state['centers'], state['M_ops'], state['exps'], d
    )                                                        # (P, N, d)
    P = x.shape[0]
    return F_all_k[torch.arange(P), k]                       # (P, d)

def rollout_local_edmd(x0, state, n_steps):
    traj = torch.zeros(n_steps + 1, d, dtype=torch.float64)
    traj[0] = x0
    for t in range(n_steps):
        f_hat = predict_f_local_edmd(traj[t:t+1], state)
        traj[t+1] = traj[t] + dt * f_hat[0]
    return traj

# ── Taylor piecewise (existing em.py state) ──────────────────────────────────

def predict_f_taylor(x, state):
    k  = pick_cluster(x, state)
    c  = state['centers'][k]
    fc = state['f_centers'][k]
    J  = state['jacobians'][k]
    return fc + (J @ (x - c).unsqueeze(-1)).squeeze(-1)

def rollout_taylor(x0, state, n_steps):
    traj = torch.zeros(n_steps + 1, d, dtype=torch.float64)
    traj[0] = x0
    for t in range(n_steps):
        traj[t+1] = traj[t] + dt * predict_f_taylor(traj[t:t+1], state)[0]
    return traj

# ── Truth (RK45) ─────────────────────────────────────────────────────────────

def rollout_truth(x0, n_steps):
    sol = solve_ivp(lambda t, y: lorenz_f(y),
                    (0.0, n_steps*dt), x0.numpy(),
                    t_eval=np.linspace(0.0, n_steps*dt, n_steps+1),
                    method='RK45', rtol=1e-10, atol=1e-10)
    return torch.tensor(sol.y.T, dtype=torch.float64)

inits = [X_te[i] for i in (0, 200, 400, 600, 800)]

def rollout_metrics(rollout_fn, state_or_model, n_steps=500):
    errs = [[], [], []]  # at 50, 200, 500
    for x0 in inits:
        tru = rollout_truth(x0, n_steps)
        if state_or_model is None:
            continue
        traj = rollout_fn(x0, state_or_model, n_steps)
        e = torch.linalg.norm(traj - tru, dim=1)
        errs[0].append(e[50].item())
        errs[1].append(e[200].item())
        errs[2].append(e[500].item())
    return tuple(np.mean(x) for x in errs)

# ── Global EDMD baselines ────────────────────────────────────────────────────
print("Fitting global EDMD ...")
edmd2 = pk.Koopman(observables=pk.observables.Polynomial(degree=2, include_bias=True),
                   regressor=pk.regression.EDMD()); edmd2.fit(X_tr.numpy(), dt=dt)
edmd3 = pk.Koopman(observables=pk.observables.Polynomial(degree=3, include_bias=True),
                   regressor=pk.regression.EDMD()); edmd3.fit(X_tr.numpy(), dt=dt)

edmd2_one = torch.linalg.norm(torch.tensor(edmd2.predict(X_te_in.numpy())) - X_te_next, dim=1).mean().item()
edmd3_one = torch.linalg.norm(torch.tensor(edmd3.predict(X_te_in.numpy())) - X_te_next, dim=1).mean().item()

print("Fitting EDMD-ours baselines ...")
edmd_ours2 = fit_global_ours(X_tr, F_tr, degree=2, ridge=1e-4)
edmd_ours3 = fit_global_ours(X_tr, F_tr, degree=3, ridge=1e-4)

f_hat2 = predict_f_global_ours(X_te_in, edmd_ours2)
edmd_ours2_one = torch.linalg.norm(
    X_te_in + dt * f_hat2 - X_te_next, dim=1).mean().item()
f_hat3 = predict_f_global_ours(X_te_in, edmd_ours3)
edmd_ours3_one = torch.linalg.norm(
    X_te_in + dt * f_hat3 - X_te_next, dim=1).mean().item()

def rollout_edmd_ours(x0, edmd_model, n_steps):
    traj = torch.zeros(n_steps + 1, d, dtype=torch.float64)
    traj[0] = x0
    for t in range(n_steps):
        f_hat = predict_f_global_ours(traj[t:t+1], edmd_model)
        traj[t + 1] = traj[t] + dt * f_hat[0]
    return traj

edmd_ours2_rollout = rollout_metrics(rollout_edmd_ours, edmd_ours2)
edmd_ours3_rollout = rollout_metrics(rollout_edmd_ours, edmd_ours3)

print(f"  EDMD-ours deg-2: one-step {edmd_ours2_one:.5f} ({100*edmd_ours2_one/step_baseline:.2f}%)  "
      f"rollout {edmd_ours2_rollout[0]:.3f}/{edmd_ours2_rollout[1]:.3f}/{edmd_ours2_rollout[2]:.3f}")
print(f"  EDMD-ours deg-3: one-step {edmd_ours3_one:.5f} ({100*edmd_ours3_one/step_baseline:.2f}%)  "
      f"rollout {edmd_ours3_rollout[0]:.3f}/{edmd_ours3_rollout[1]:.3f}/{edmd_ours3_rollout[2]:.3f}")

def rollout_edmd(x0, model, n_steps):
    sim = torch.tensor(model.simulate(x0.numpy().reshape(1,-1), n_steps=n_steps),
                       dtype=torch.float64)
    return torch.cat([x0.unsqueeze(0), sim], dim=0)

edmd2_rollout = rollout_metrics(rollout_edmd, edmd2)
edmd3_rollout = rollout_metrics(rollout_edmd, edmd3)

print(f"  global EDMD deg-2: one-step {edmd2_one:.5f} ({100*edmd2_one/step_baseline:.2f}%)  "
      f"rollout {edmd2_rollout[0]:.3f}/{edmd2_rollout[1]:.3f}/{edmd2_rollout[2]:.3f}")
print(f"  global EDMD deg-3: one-step {edmd3_one:.5f} ({100*edmd3_one/step_baseline:.2f}%)  "
      f"rollout {edmd3_rollout[0]:.3f}/{edmd3_rollout[1]:.3f}/{edmd3_rollout[2]:.3f}")

# ── One-step prediction for local EDMD ───────────────────────────────────────
def onestep_local_edmd(state):
    f_hat = predict_f_local_edmd(X_te_in, state)
    x_pred = X_te_in + dt * f_hat
    return torch.linalg.norm(x_pred - X_te_next, dim=1).mean().item()

def onestep_taylor(state):
    f_hat = predict_f_taylor(X_te_in, state)
    x_pred = X_te_in + dt * f_hat
    return torch.linalg.norm(x_pred - X_te_next, dim=1).mean().item()

# ── Sweep N and degree for local EDMD ────────────────────────────────────────
rows = []
configs = [
    ('local-EDMD deg=2', 2, 2),
    ('local-EDMD deg=2', 2, 3),
    ('local-EDMD deg=2', 2, 5),
    ('local-EDMD deg=2', 2, 10),
    ('local-EDMD deg=2', 2, 20),
    ('local-EDMD deg=3', 3, 2),
    ('local-EDMD deg=3', 3, 3),
    ('local-EDMD deg=3', 3, 5),
]

for name, degree, N in configs:
    print(f"\n{name}, N={N} ...", flush=True)
    hp = dict(hp_base); hp['sigma2'] = 'auto'
    state, _, _ = fit_local_edmd(X_tr, F_tr, N=N, hp=hp, degree=degree,
                                 n_iter=args.n_iter, n_restarts=args.n_restarts, verbose=False)
    one = onestep_local_edmd(state)
    r50, r200, r500 = rollout_metrics(rollout_local_edmd, state)

    M = len(monomial_exponents(d, degree))
    n_params = state['N'] * M * M   # M_k only (centers/covs not counted)

    rows.append({
        'name': f"{name} N={N}", 'N': N, 'degree': degree, 'active': state['N'],
        'params': n_params, 'sigma2': state['sigma2'].mean().item(),
        'one': one, 'r50': r50, 'r200': r200, 'r500': r500,
    })

# Taylor reference at matching N
taylor_rows = []
for N in (2, 3, 5, 10, 20):
    print(f"\nTaylor piecewise N={N} ...", flush=True)
    hp = dict(hp_base); hp['sigma2'] = 'auto'
    state, _, _ = fit_taylor(X_tr, F_tr, lorenz_f, lorenz_J, N=N, hp=hp,
                             n_iter=args.n_iter, n_restarts=args.n_restarts, verbose=False)
    one = onestep_taylor(state)
    r50, r200, r500 = rollout_metrics(rollout_taylor, state)
    n_params = state['N'] * (d*d + d)   # J_k + f_k
    taylor_rows.append({
        'name': f"Taylor N={N}", 'N': N, 'active': state['N'],
        'params': n_params, 'one': one, 'r50': r50, 'r200': r200, 'r500': r500,
    })

# ── Report ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 110)
print(f"ONE-STEP + ROLLOUT  (baseline ‖Δx‖ = {step_baseline:.4f}, attractor ø ≈ 76)")
print("=" * 110)
print(f"{'method':<26} {'active':>6} {'params':>7} {'one-step':>10} {'rel%':>7}  "
      f"{'r@50':>8} {'r@200':>8} {'r@500':>8}")
print("-" * 110)

for r in rows:
    print(f"{r['name']:<26} {r['active']:>6} {r['params']:>7} "
          f"{r['one']:>10.5f} {100*r['one']/step_baseline:>6.2f}%  "
          f"{r['r50']:>8.3f} {r['r200']:>8.3f} {r['r500']:>8.3e}")

print("-" * 110)
for r in taylor_rows:
    print(f"{r['name']:<26} {r['active']:>6} {r['params']:>7} "
          f"{r['one']:>10.5f} {100*r['one']/step_baseline:>6.2f}%  "
          f"{r['r50']:>8.3f} {r['r200']:>8.3f} {r['r500']:>8.3e}")

print("-" * 110)
M2, M3 = len(monomial_exponents(d, 2)), len(monomial_exponents(d, 3))
print(f"{'EDMD-pk deg-2 (global)':<26} {'-':>6} {M2*M2:>7} "
      f"{edmd2_one:>10.5f} {100*edmd2_one/step_baseline:>6.2f}%  "
      f"{edmd2_rollout[0]:>8.3f} {edmd2_rollout[1]:>8.3f} {edmd2_rollout[2]:>8.3f}")
print(f"{'EDMD-pk deg-3 (global)':<26} {'-':>6} {M3*M3:>7} "
      f"{edmd3_one:>10.5f} {100*edmd3_one/step_baseline:>6.2f}%  "
      f"{edmd3_rollout[0]:>8.3f} {edmd3_rollout[1]:>8.3f} {edmd3_rollout[2]:>8.3f}")
print(f"{'EDMD-ours deg-2 (global)':<26} {'-':>6} {M2*M2:>7} "
      f"{edmd_ours2_one:>10.5f} {100*edmd_ours2_one/step_baseline:>6.2f}%  "
      f"{edmd_ours2_rollout[0]:>8.3f} {edmd_ours2_rollout[1]:>8.3f} {edmd_ours2_rollout[2]:>8.3f}")
print(f"{'EDMD-ours deg-3 (global)':<26} {'-':>6} {M3*M3:>7} "
      f"{edmd_ours3_one:>10.5f} {100*edmd_ours3_one/step_baseline:>6.2f}%  "
      f"{edmd_ours3_rollout[0]:>8.3f} {edmd_ours3_rollout[1]:>8.3f} {edmd_ours3_rollout[2]:>8.3f}")
print("=" * 110)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

for ax, metric, ylabel, title in [
    (axes[0], 'one', r'mean $\|\hat x_{t+1}-x_{t+1}\|$', 'One-step state error vs params'),
    (axes[1], 'r500', r'$\|\hat x_{500}-x_{500}\|$',     'Rollout error @ t=5s vs params'),
]:
    deg2 = [r for r in rows if r['degree'] == 2]
    deg3 = [r for r in rows if r['degree'] == 3]
    if deg2:
        ax.loglog([r['params'] for r in deg2], [r[metric] for r in deg2],
                  'o-', label='local-EDMD deg-2')
    if deg3:
        ax.loglog([r['params'] for r in deg3], [r[metric] for r in deg3],
                  's-', label='local-EDMD deg-3')
    ax.loglog([r['params'] for r in taylor_rows], [r[metric] for r in taylor_rows],
              '^-', label='Taylor piecewise')
    ax.axhline(edmd2_one if metric == 'one' else edmd2_rollout[2],
               ls='--', color='C3', label='EDMD-pk deg-2')
    ax.axhline(edmd3_one if metric == 'one' else edmd3_rollout[2],
               ls='--', color='C4', label='EDMD-pk deg-3')
    ax.axhline(edmd_ours2_one if metric == 'one' else edmd_ours2_rollout[2],
               ls=':', color='C3', label='EDMD-ours deg-2')
    ax.axhline(edmd_ours3_one if metric == 'one' else edmd_ours3_rollout[2],
               ls=':', color='C4', label='EDMD-ours deg-3')
    ax.set_xlabel('# parameters')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(fig_path("local_edmd_comparison.png"), dpi=120)
print("\n→ saved local_edmd_comparison.png")

# ── Save raw data ────────────────────────────────────────────────────────────
import json
raw = {
    "data_seed": args.seed,
    "step_baseline": step_baseline,
    "rows": rows,
    "taylor_rows": taylor_rows,
    "edmd_pk_baselines": {
        "deg2": {
            "one_step": edmd2_one,
            "rollout": list(edmd2_rollout),
        },
        "deg3": {
            "one_step": edmd3_one,
            "rollout": list(edmd3_rollout),
        },
    },
    "edmd_ours_baselines": {
        "deg2": {
            "one_step": edmd_ours2_one,
            "rollout": list(edmd_ours2_rollout),
        },
        "deg3": {
            "one_step": edmd_ours3_one,
            "rollout": list(edmd_ours3_rollout),
        },
    },
}
with open(data_path("validation_lorenz_local_edmd.json"), "w") as fp:
    json.dump(raw, fp, indent=2)
print(f"Raw data saved to {data_path('validation_lorenz_local_edmd.json')}")
