"""
Compare four methods across N:
  1. Taylor-tied (original em.py): J_k = J(c_k) analytic at every M-step
  2. Hybrid (em_hybrid.py): Taylor-initialized, LS-refined
  3. GMM baseline (original em.py, sigma2 = inf)
  4. EDMD deg-2, deg-3 (pykoopman)

Reports:
  * one-step state error (held-out test set)
  * rollout error @ 50 / 500 steps from 3 test initial conditions
  * ELBO monotonicity flag (hybrid should be monotone)
  * Jacobian drift: fitted J_k vs analytic J(c_k) for hybrid
"""

import numpy as np
import torch
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import pykoopman as pk

from lorenz import generate_data, f as lorenz_f, J as lorenz_J
from em import fit               as fit_taylor
from em_hybrid import fit_hybrid, jacobian_drift
from distributions import mvn_logpdf_batch
from elbo import check_monotone

torch.set_default_dtype(torch.float64)

# ── Data ─────────────────────────────────────────────────────────────────────
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

hp_base = {
    'alpha0': 0.5, 'mu0': X_tr.mean(dim=0),
    'Lambda0': 0.01 * torch.eye(d, dtype=torch.float64),
    'kappa0': 1.0, 'Psi0': 10.0 * torch.eye(d, dtype=torch.float64),
    'nu0': float(d + 2), 'sigma2': 'auto',
}
def hp_our():  return {**hp_base, 'sigma2': 'auto'}
def hp_gmm_(): return {**hp_base, 'sigma2': 1e10}

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

def onestep_err(state):
    pred = one_step(X_te_in, state)
    return torch.linalg.norm(pred - X_te_next, dim=1).mean().item()

def rollout_errs(state, inits, n_steps=500):
    errs_50, errs_500 = [], []
    for x0 in inits:
        tru = rollout_truth(x0, n_steps)
        our = rollout_ours(x0, state, n_steps)
        e = torch.linalg.norm(our - tru, dim=1)
        errs_50.append(e[50].item()); errs_500.append(e[500].item())
    return (np.mean(errs_50), np.mean(errs_500))

inits = [X_te[i] for i in (0, 300, 700)]

# ── EDMD baselines ───────────────────────────────────────────────────────────
print("Fitting EDMD baselines ...")
edmd2 = pk.Koopman(observables=pk.observables.Polynomial(degree=2, include_bias=True),
                   regressor=pk.regression.EDMD()); edmd2.fit(X_tr.numpy(), dt=dt)
edmd3 = pk.Koopman(observables=pk.observables.Polynomial(degree=3, include_bias=True),
                   regressor=pk.regression.EDMD()); edmd3.fit(X_tr.numpy(), dt=dt)

edmd2_one = torch.linalg.norm(torch.tensor(edmd2.predict(X_te_in.numpy())) - X_te_next, dim=1).mean().item()
edmd3_one = torch.linalg.norm(torch.tensor(edmd3.predict(X_te_in.numpy())) - X_te_next, dim=1).mean().item()

def rollout_edmd_errs(model, inits, n_steps=500):
    e50, e500 = [], []
    for x0 in inits:
        tru = rollout_truth(x0, n_steps)
        sim = torch.tensor(model.simulate(x0.numpy().reshape(1,-1), n_steps=n_steps), dtype=torch.float64)
        sim = torch.cat([x0.unsqueeze(0), sim], dim=0)
        e = torch.linalg.norm(sim - tru, dim=1)
        e50.append(e[50].item()); e500.append(e[500].item())
    return np.mean(e50), np.mean(e500)

edmd2_r50, edmd2_r500 = rollout_edmd_errs(edmd2, inits)
edmd3_r50, edmd3_r500 = rollout_edmd_errs(edmd3, inits)

# ── Sweep N ──────────────────────────────────────────────────────────────────
N_values = [5, 12, 20, 30, 50]
rows = []

for N in N_values:
    print(f"\n{'='*70}\nN = {N}\n{'='*70}")

    print("  Fitting Taylor-tied ...", flush=True)
    st_t, _, hist_t = fit_taylor(X_tr, F_tr, lorenz_f, lorenz_J,
                                 N=N, hp=hp_our(), n_iter=100, n_restarts=2, verbose=False)
    mono_t = check_monotone(hist_t) if hist_t else True

    print("  Fitting hybrid ...", flush=True)
    st_h, _, hist_h = fit_hybrid(X_tr, F_tr, lorenz_f, lorenz_J,
                                 N=N, hp=hp_our(), n_iter=100, n_restarts=2, verbose=False)
    mono_h = check_monotone(hist_h) if hist_h else True
    drift = jacobian_drift(st_h, lorenz_J)

    print("  Fitting GMM ...", flush=True)
    st_g, _, _ = fit_taylor(X_tr, F_tr, lorenz_f, lorenz_J,
                            N=N, hp=hp_gmm_(), n_iter=100, n_restarts=2, verbose=False)

    one_t, one_h, one_g = onestep_err(st_t), onestep_err(st_h), onestep_err(st_g)
    r50_t, r500_t = rollout_errs(st_t, inits)
    r50_h, r500_h = rollout_errs(st_h, inits)
    r50_g, r500_g = rollout_errs(st_g, inits)

    rows.append({
        'N': N,
        'act_t': st_t['N'], 'act_h': st_h['N'], 'act_g': st_g['N'],
        'one_t': one_t, 'one_h': one_h, 'one_g': one_g,
        'r50_t': r50_t, 'r500_t': r500_t,
        'r50_h': r50_h, 'r500_h': r500_h,
        'r50_g': r50_g, 'r500_g': r500_g,
        'mono_t': mono_t, 'mono_h': mono_h,
        'drift_mean': drift['mean_rel'], 'drift_max': drift['max_rel'],
    })
    print(f"  drift mean/max rel: {drift['mean_rel']:.3f} / {drift['max_rel']:.3f}")
    print(f"  ELBO monotone — taylor: {mono_t}, hybrid: {mono_h}")

# ── Report ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 100)
print(f"ONE-STEP STATE ERROR  (baseline ‖Δx‖ = {step_baseline:.4f})")
print("=" * 100)
print(f"{'N':>4} {'act(t/h/g)':>12}  "
      f"{'Taylor':>10} {'Hybrid':>10} {'GMM':>10}  "
      f"{'rel% T':>8} {'rel% H':>8} {'rel% G':>8}")
for r in rows:
    actstr = f"{r['act_t']}/{r['act_h']}/{r['act_g']}"
    print(f"{r['N']:>4} {actstr:>12}  "
          f"{r['one_t']:>10.5f} {r['one_h']:>10.5f} {r['one_g']:>10.5f}  "
          f"{100*r['one_t']/step_baseline:>7.2f}% {100*r['one_h']/step_baseline:>7.2f}% "
          f"{100*r['one_g']/step_baseline:>7.2f}%")
print(f"\nEDMD deg-2: {edmd2_one:.5f}  ({100*edmd2_one/step_baseline:.2f}%)")
print(f"EDMD deg-3: {edmd3_one:.5f}  ({100*edmd3_one/step_baseline:.2f}%)")

print("\n" + "=" * 100)
print("ROLLOUT ERROR @ 500 steps (t=5s)")
print("=" * 100)
print(f"{'N':>4}  {'Taylor':>14} {'Hybrid':>14} {'GMM':>14}")
for r in rows:
    print(f"{r['N']:>4}  {r['r500_t']:>14.3e} {r['r500_h']:>14.3e} {r['r500_g']:>14.3e}")
print(f"\nEDMD deg-2: {edmd2_r500:.3f}")
print(f"EDMD deg-3: {edmd3_r500:.3f}")

print("\n" + "=" * 100)
print("DIAGNOSTICS")
print("=" * 100)
print(f"{'N':>4}  {'mono T':>8} {'mono H':>8}  {'J drift mean':>14} {'J drift max':>14}")
for r in rows:
    print(f"{r['N']:>4}  {str(r['mono_t']):>8} {str(r['mono_h']):>8}  "
          f"{r['drift_mean']:>14.3f} {r['drift_max']:>14.3f}")

# ── Plot ─────────────────────────────────────────────────────────────────────
Ns = [r['N'] for r in rows]
one_t = [r['one_t'] for r in rows]
one_h = [r['one_h'] for r in rows]
one_g = [r['one_g'] for r in rows]

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].loglog(Ns, one_t, 'o-', label='Taylor-tied')
ax[0].loglog(Ns, one_h, 's-', label='Hybrid (LS)')
ax[0].loglog(Ns, one_g, '^-', label='GMM')
ax[0].axhline(edmd2_one, ls='--', color='C3', label='EDMD deg-2')
ax[0].axhline(edmd3_one, ls='--', color='C4', label='EDMD deg-3')
ax[0].set_xlabel('N'); ax[0].set_ylabel(r'mean $\|\hat x_{t+1}-x_{t+1}\|$')
ax[0].set_title('One-step state error vs N')
ax[0].legend(); ax[0].grid(alpha=0.3)

drift_mean = [r['drift_mean'] for r in rows]
drift_max  = [r['drift_max']  for r in rows]
ax[1].semilogx(Ns, drift_mean, 'o-', label='mean')
ax[1].semilogx(Ns, drift_max,  's-', label='max')
ax[1].set_xlabel('N')
ax[1].set_ylabel(r'relative drift $\|J_k^{LS} - J(c_k)\| / \|J(c_k)\|$')
ax[1].set_title('Hybrid J drift vs analytic (Taylor validity)')
ax[1].legend(); ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("hybrid_sweep.png", dpi=120)
print("\n→ saved hybrid_sweep.png")
