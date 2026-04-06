"""
Sweep N for the residual-aware piecewise-linear model.
Compare against GMM (same N) and EDMD deg-2/3 baselines.

Reports:
  * one-step state error on held-out test set
  * rollout error @ 50 / 200 / 500 steps from 3 test initial conditions
  * effective N after dead-cluster pruning
"""

import numpy as np
from utils.paths import fig_path, data_path
import torch
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import pykoopman as pk

from simulators.lorenz import generate_data, f as lorenz_f, J as lorenz_J
from models.em import fit
from models.distributions import mvn_logpdf_batch

torch.set_default_dtype(torch.float64)

# ── Data (identical to validation.py) ─────────────────────────────────────────
data  = generate_data(n_steps=5000, dt=0.01, warmup=1000, seed=42)
X_all = torch.tensor(data['X'], dtype=torch.float64)
F_all = torch.tensor(data['F'], dtype=torch.float64)
dt    = 0.01
d     = 3

X_tr, X_te = X_all[:4000], X_all[4000:]
F_tr, F_te = F_all[:4000], F_all[4000:]
X_te_in    = X_all[4000:4999]
X_te_next  = X_all[4001:5000]
step_baseline = torch.linalg.norm(X_te_next - X_te_in, dim=1).mean().item()

hp = {
    'alpha0': 0.5, 'mu0': X_tr.mean(dim=0),
    'Lambda0': 0.01 * torch.eye(d, dtype=torch.float64),
    'kappa0': 1.0, 'Psi0': 10.0 * torch.eye(d, dtype=torch.float64),
    'nu0': float(d + 2), 'sigma2': 'auto',
}
hp_gmm = {**hp, 'sigma2': 1e10}

# ── Helpers ───────────────────────────────────────────────────────────────────

def pick_cluster(x, state):
    log_pi   = torch.log(state['pi']).unsqueeze(0)
    log_prox = mvn_logpdf_batch(x, state['centers'], state['covariances'])
    return (log_pi + log_prox).argmax(dim=1)

def piecewise_f(x, state):
    k = pick_cluster(x, state)
    c  = state['centers'][k]
    fc = state['f_centers'][k]
    J  = state['jacobians'][k]
    delta = (x - c).unsqueeze(-1)
    return fc + (J @ delta).squeeze(-1)

def one_step(X_in, state):
    return X_in + dt * piecewise_f(X_in, state)

def rollout_ours(x0, state, n_steps):
    traj = torch.zeros(n_steps + 1, d, dtype=torch.float64)
    traj[0] = x0
    for t in range(n_steps):
        traj[t+1] = one_step(traj[t:t+1], state)[0]
    return traj

def rollout_truth(x0, n_steps):
    sol = solve_ivp(lambda t, y: lorenz_f(y),
                    (0.0, n_steps*dt), x0.numpy(),
                    t_eval=np.linspace(0.0, n_steps*dt, n_steps+1),
                    method='RK45', rtol=1e-10, atol=1e-10)
    return torch.tensor(sol.y.T, dtype=torch.float64)

def mean_onestep_err(state):
    pred = one_step(X_te_in, state)
    return torch.linalg.norm(pred - X_te_next, dim=1).mean().item()

def rollout_errs(state, inits, n_steps=500):
    """Return err at [50, 200, 500] averaged over inits."""
    errs_50, errs_200, errs_500 = [], [], []
    for x0 in inits:
        tru = rollout_truth(x0, n_steps)
        our = rollout_ours(x0, state, n_steps)
        e = torch.linalg.norm(our - tru, dim=1)
        errs_50.append(e[50].item())
        errs_200.append(e[200].item())
        errs_500.append(e[500].item())
    return (np.mean(errs_50), np.mean(errs_200), np.mean(errs_500))

# 3 test initial conditions
inits = [X_te[i] for i in (0, 300, 700)]

# ── EDMD baselines (N-independent) ────────────────────────────────────────────
print("Fitting EDMD baselines ...")
edmd2 = pk.Koopman(observables=pk.observables.Polynomial(degree=2, include_bias=True),
                   regressor=pk.regression.EDMD())
edmd2.fit(X_tr.numpy(), dt=dt)
edmd3 = pk.Koopman(observables=pk.observables.Polynomial(degree=3, include_bias=True),
                   regressor=pk.regression.EDMD())
edmd3.fit(X_tr.numpy(), dt=dt)

edmd2_onestep = torch.linalg.norm(
    torch.tensor(edmd2.predict(X_te_in.numpy())) - X_te_next, dim=1).mean().item()
edmd3_onestep = torch.linalg.norm(
    torch.tensor(edmd3.predict(X_te_in.numpy())) - X_te_next, dim=1).mean().item()

def rollout_edmd(x0, model, n_steps):
    sim = torch.tensor(model.simulate(x0.numpy().reshape(1,-1), n_steps=n_steps),
                       dtype=torch.float64)
    return torch.cat([x0.unsqueeze(0), sim], dim=0)

def rollout_errs_edmd(model, inits, n_steps=500):
    errs_50, errs_200, errs_500 = [], [], []
    for x0 in inits:
        tru = rollout_truth(x0, n_steps)
        sim = rollout_edmd(x0, model, n_steps)
        e = torch.linalg.norm(sim - tru, dim=1)
        errs_50.append(e[50].item())
        errs_200.append(e[200].item())
        errs_500.append(e[500].item())
    return (np.mean(errs_50), np.mean(errs_200), np.mean(errs_500))

edmd2_rollout = rollout_errs_edmd(edmd2, inits)
edmd3_rollout = rollout_errs_edmd(edmd3, inits)

# ── Sweep N ───────────────────────────────────────────────────────────────────
N_values = [3, 5, 8, 12, 20, 30, 50]
rows = []

for N in N_values:
    # copies to keep hp clean across runs (sigma2 gets set in-place)
    hp_o = dict(hp);    hp_o['sigma2'] = 'auto'
    hp_g = dict(hp_gmm)
    print(f"\nN={N} residual-aware ...", flush=True)
    state_o, _, _ = fit(X_tr, F_tr, lorenz_f, lorenz_J,
                        N=N, hp=hp_o, n_iter=100, n_restarts=2, verbose=False)
    print(f"N={N} gmm ...", flush=True)
    state_g, _, _ = fit(X_tr, F_tr, lorenz_f, lorenz_J,
                        N=N, hp=hp_g, n_iter=100, n_restarts=2, verbose=False)

    one_o = mean_onestep_err(state_o)
    one_g = mean_onestep_err(state_g)
    r_o   = rollout_errs(state_o, inits)
    r_g   = rollout_errs(state_g, inits)

    rows.append({
        'N': N,
        'active_o': state_o['N'], 'active_g': state_g['N'],
        'sigma2':   hp_o['sigma2'],
        'one_o': one_o, 'one_g': one_g,
        'r50_o': r_o[0], 'r50_g': r_g[0],
        'r200_o': r_o[1], 'r200_g': r_g[1],
        'r500_o': r_o[2], 'r500_g': r_g[2],
    })

# ── Report ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 108)
print(f"SWEEP — one-step state err  (baseline ‖x_{{t+1}}-x_t‖ = {step_baseline:.4f})")
print("=" * 108)
print(f"{'N_req':>6} {'act_o':>6} {'act_g':>6} {'sigma2':>8} "
      f"{'one-step ours':>14} {'rel% ours':>10} "
      f"{'one-step gmm':>14} {'rel% gmm':>10}")
for r in rows:
    print(f"{r['N']:>6} {r['active_o']:>6} {r['active_g']:>6} "
          f"{r['sigma2']:>8.2f} "
          f"{r['one_o']:>14.5f} {100*r['one_o']/step_baseline:>10.2f} "
          f"{r['one_g']:>14.5f} {100*r['one_g']/step_baseline:>10.2f}")

print("\n" + "=" * 108)
print(f"SWEEP — rollout error (mean over 3 inits)")
print("=" * 108)
print(f"{'N_req':>6}  {'t=50 ours':>12} {'t=200 ours':>12} {'t=500 ours':>12}   "
      f"{'t=50 gmm':>12} {'t=200 gmm':>12} {'t=500 gmm':>12}")
for r in rows:
    print(f"{r['N']:>6}  {r['r50_o']:>12.3f} {r['r200_o']:>12.3f} {r['r500_o']:>12.3e}   "
          f"{r['r50_g']:>12.3f} {r['r200_g']:>12.3f} {r['r500_g']:>12.3e}")

print("\nEDMD baselines:")
print(f"  deg-2: one-step {edmd2_onestep:.5f} ({100*edmd2_onestep/step_baseline:.2f}%)  "
      f"rollout {edmd2_rollout[0]:.3f} / {edmd2_rollout[1]:.3f} / {edmd2_rollout[2]:.3f}")
print(f"  deg-3: one-step {edmd3_onestep:.5f} ({100*edmd3_onestep/step_baseline:.2f}%)  "
      f"rollout {edmd3_rollout[0]:.3f} / {edmd3_rollout[1]:.3f} / {edmd3_rollout[2]:.3f}")

# ── Plot one-step and rollout-500 error vs N ──────────────────────────────────
Ns = [r['N'] for r in rows]
one_o = [r['one_o'] for r in rows]
one_g = [r['one_g'] for r in rows]
r500_o = [r['r500_o'] for r in rows]
r500_g = [r['r500_g'] for r in rows]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].loglog(Ns, one_o, 'o-', label='ours')
axes[0].loglog(Ns, one_g, 's-', label='gmm')
axes[0].axhline(edmd2_onestep, ls='--', color='C2', label=f'EDMD deg-2')
axes[0].axhline(edmd3_onestep, ls='--', color='C3', label=f'EDMD deg-3')
axes[0].set_xlabel("N (requested)")
axes[0].set_ylabel(r"mean $\|\hat x_{t+1}-x_{t+1}\|$ on test set")
axes[0].set_title("One-step state error vs N")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].loglog(Ns, r500_o, 'o-', label='ours')
axes[1].loglog(Ns, r500_g, 's-', label='gmm')
axes[1].axhline(edmd2_rollout[2], ls='--', color='C2', label='EDMD deg-2')
axes[1].axhline(edmd3_rollout[2], ls='--', color='C3', label='EDMD deg-3')
axes[1].set_xlabel("N (requested)")
axes[1].set_ylabel(r"$\|\hat x_{500}-x_{500}\|$")
axes[1].set_title("Rollout error at t=5s vs N")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(fig_path("sweep_N.png"), dpi=120)
print("\n→ saved sweep_N.png")

# ── Save raw data ────────────────────────────────────────────────────────────
import json
raw = {
    "step_baseline": step_baseline,
    "rows": rows,
    "edmd_baselines": {
        "deg2": {
            "one_step": edmd2_onestep,
            "rollout": list(edmd2_rollout),
        },
        "deg3": {
            "one_step": edmd3_onestep,
            "rollout": list(edmd3_rollout),
        },
    },
}
with open(data_path("validation_lorenz_sweep_N.json"), "w") as fp:
    json.dump(raw, fp, indent=2)
print(f"Raw data saved to {data_path('validation_lorenz_sweep_N.json')}")
