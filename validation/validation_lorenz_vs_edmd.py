"""
Honest validation of the residual-aware piecewise-linear Lorenz model.

Three experiments:
  1. Trajectory rollout vs RK45 truth — how far can the piecewise-linear
     switching integrator go before diverging?
  2. Local residual vs radius around each fixed point C+, C-.
  3. One-step state-space error on a held-out test set, compared to
     EDMD (via pykoopman) with polynomial lifting.

Design note: at test time our model picks the active cluster k from
proximity alone (argmax of log pi_k + log N(x; c_k, Sigma_k)) — the
residual term is used only during training.
"""

import argparse
import numpy as np
from utils.paths import fig_path, data_path
import torch
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import pykoopman as pk
from models.global_edmd import fit as fit_global_ours, predict_f as predict_f_global_ours

parser = argparse.ArgumentParser(description="Lorenz: one-step + rollout vs global EDMD")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--n-steps', type=int, default=5000)
parser.add_argument('--dt', type=float, default=0.01)
parser.add_argument('--warmup', type=int, default=1000)
parser.add_argument('--n-train', type=int, default=4000)
parser.add_argument('--N', type=int, default=5)
parser.add_argument('--n-iter', type=int, default=100)
parser.add_argument('--n-restarts', type=int, default=3)
parser.add_argument('--rollout-steps', type=int, default=500)
args = parser.parse_args()

from simulators.lorenz import generate_data, f as lorenz_f, J as lorenz_J, SIGMA, RHO, BETA
from models.em import fit, initialize
from models.distributions import mvn_logpdf_batch

torch.set_default_dtype(torch.float64)

# ═════════════════════════════════════════════════════════════════════════════
# Data
# ═════════════════════════════════════════════════════════════════════════════

data = generate_data(n_steps=args.n_steps, dt=args.dt, warmup=args.warmup, seed=args.seed)
X_all = torch.tensor(data['X'], dtype=torch.float64)
F_all = torch.tensor(data['F'], dtype=torch.float64)
dt    = args.dt

# train / test split
X_tr, X_te = X_all[:args.n_train], X_all[args.n_train:]
F_tr, F_te = F_all[:args.n_train], F_all[args.n_train:]

d = 3
hp = {
    'alpha0':  0.5,
    'mu0':     X_tr.mean(dim=0),
    'Lambda0': 0.01 * torch.eye(d, dtype=torch.float64),
    'kappa0':  1.0,
    'Psi0':    10.0 * torch.eye(d, dtype=torch.float64),
    'nu0':     float(d + 2),
    'sigma2':  'auto',
}
hp_gmm = {**hp, 'sigma2': 1e10}

# ═════════════════════════════════════════════════════════════════════════════
# Fit our method (residual-aware) and GMM baseline
# ═════════════════════════════════════════════════════════════════════════════

print(f"Fitting residual-aware model (N={args.N}) ...")
state_ours, r_ours, _ = fit(X_tr, F_tr, lorenz_f, lorenz_J,
                            N=args.N, hp=hp, n_iter=args.n_iter, n_restarts=args.n_restarts, verbose=False)
print(f"  active N = {state_ours['N']}, sigma2 = {state_ours['sigma2'].mean().item():.4f} (mean)")

print(f"Fitting GMM baseline    (N={args.N}) ...")
state_gmm, r_gmm, _ = fit(X_tr, F_tr, lorenz_f, lorenz_J,
                          N=args.N, hp=hp_gmm, n_iter=args.n_iter, n_restarts=args.n_restarts, verbose=False)
print(f"  active N = {state_gmm['N']}")

# ═════════════════════════════════════════════════════════════════════════════
# Fit EDMD (pykoopman) with degree-2 and degree-3 polynomial lifting
# ═════════════════════════════════════════════════════════════════════════════

print("Fitting EDMD deg-2 ...")
edmd2 = pk.Koopman(
    observables=pk.observables.Polynomial(degree=2, include_bias=True),
    regressor=pk.regression.EDMD(),
)
edmd2.fit(X_tr.numpy(), dt=dt)

print("Fitting EDMD deg-3 ...")
edmd3 = pk.Koopman(
    observables=pk.observables.Polynomial(degree=3, include_bias=True),
    regressor=pk.regression.EDMD(),
)
edmd3.fit(X_tr.numpy(), dt=dt)

print("Fitting EDMD-ours deg-2 ...")
edmd_ours2 = fit_global_ours(X_tr, F_tr, degree=2, ridge=1e-4)
print("Fitting EDMD-ours deg-3 ...")
edmd_ours3 = fit_global_ours(X_tr, F_tr, degree=3, ridge=1e-4)

# ═════════════════════════════════════════════════════════════════════════════
# Active-cluster picker (proximity only, for test-time use)
# ═════════════════════════════════════════════════════════════════════════════

def pick_cluster(x: torch.Tensor, state: dict) -> torch.Tensor:
    """Return argmax_k [log pi_k + log N(x; c_k, Sigma_k)] for each x."""
    log_pi   = torch.log(state['pi']).unsqueeze(0)                # (1, N)
    log_prox = mvn_logpdf_batch(x, state['centers'], state['covariances'])  # (P, N)
    return (log_pi + log_prox).argmax(dim=1)                       # (P,)


def piecewise_linear_f(x: torch.Tensor, state: dict) -> torch.Tensor:
    """
    Evaluate the piecewise-linear vector-field surrogate at each x.
      f_hat(x) = f(c_k) + J_k (x - c_k)     with k = pick_cluster(x)
    """
    k     = pick_cluster(x, state)                                 # (P,)
    c     = state['centers'][k]                                    # (P, d)
    fc    = state['f_centers'][k]                                  # (P, d)
    J     = state['jacobians'][k]                                  # (P, d, d)
    delta = (x - c).unsqueeze(-1)                                  # (P, d, 1)
    return fc + (J @ delta).squeeze(-1)                            # (P, d)


def piecewise_euler_step(x: torch.Tensor, state: dict) -> torch.Tensor:
    return x + dt * piecewise_linear_f(x, state)


# ═════════════════════════════════════════════════════════════════════════════
# Experiment 1 — Trajectory rollout vs RK45 truth
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("EXPERIMENT 1 — Trajectory rollout (3 init conditions, 500 steps)")
print("=" * 70)

n_steps_rollout = args.rollout_steps
init_indices    = [0, 300, 700]   # pick 3 points from test set

def rollout_ours(x0, state, n_steps):
    traj = torch.zeros(n_steps + 1, d, dtype=torch.float64)
    traj[0] = x0
    for t in range(n_steps):
        traj[t + 1] = piecewise_euler_step(traj[t:t+1], state)[0]
    return traj

def rollout_edmd(x0, model, n_steps):
    return torch.tensor(model.simulate(x0.numpy().reshape(1, -1), n_steps=n_steps),
                        dtype=torch.float64)

def rollout_edmd_ours(x0, edmd_model, n_steps):
    traj = torch.zeros(n_steps + 1, d, dtype=torch.float64)
    traj[0] = x0
    for t in range(n_steps):
        f_hat = predict_f_global_ours(traj[t:t+1], edmd_model)
        traj[t + 1] = traj[t] + dt * f_hat[0]
    return traj

def rollout_truth(x0, n_steps):
    sol = solve_ivp(
        fun=lambda t, y: lorenz_f(y),
        t_span=(0.0, n_steps * dt),
        y0=x0.numpy(),
        method='RK45', t_eval=np.linspace(0.0, n_steps * dt, n_steps + 1),
        rtol=1e-10, atol=1e-10,
    )
    return torch.tensor(sol.y.T, dtype=torch.float64)   # (n_steps+1, d)

rollouts = {'truth': [], 'ours': [], 'gmm': [], 'edmd2': [], 'edmd3': [],
            'edmd_ours2': [], 'edmd_ours3': []}
for idx in init_indices:
    x0 = X_te[idx]
    rollouts['truth'].append(rollout_truth(x0, n_steps_rollout))
    rollouts['ours' ].append(rollout_ours (x0, state_ours, n_steps_rollout))
    rollouts['gmm'  ].append(rollout_ours (x0, state_gmm,  n_steps_rollout))
    # EDMD simulate returns (n_steps, d) — prepend x0 to align shapes with truth (n_steps+1, d)
    sim2 = rollout_edmd(x0, edmd2, n_steps_rollout)
    sim3 = rollout_edmd(x0, edmd3, n_steps_rollout)
    rollouts['edmd2'].append(torch.cat([x0.unsqueeze(0), sim2], dim=0))
    rollouts['edmd3'].append(torch.cat([x0.unsqueeze(0), sim3], dim=0))
    rollouts['edmd_ours2'].append(rollout_edmd_ours(x0, edmd_ours2, n_steps_rollout))
    rollouts['edmd_ours3'].append(rollout_edmd_ours(x0, edmd_ours3, n_steps_rollout))

# Per-step error ‖x̂_t − x_t‖ averaged across the 3 initial conditions
def avg_err(method):
    errs = []
    for i in range(len(init_indices)):
        e = torch.linalg.norm(rollouts[method][i] - rollouts['truth'][i], dim=1)
        errs.append(e)
    return torch.stack(errs).mean(dim=0)

err_curves = {m: avg_err(m) for m in ('ours', 'gmm', 'edmd2', 'edmd3', 'edmd_ours2', 'edmd_ours3')}

attractor_diameter = (X_all.max(dim=0).values - X_all.min(dim=0).values).norm().item()
print(f"attractor diameter ≈ {attractor_diameter:.1f}")
print()
rollout_labels = {
    'ours': 'ours', 'gmm': 'gmm',
    'edmd2': 'EDMD-pk deg-2', 'edmd3': 'EDMD-pk deg-3',
    'edmd_ours2': 'EDMD-ours deg-2', 'edmd_ours3': 'EDMD-ours deg-3',
}
print(f"{'method':<20} {'err @ 50 steps':>16} {'err @ 200 steps':>17} {'err @ 500 steps':>17}")
for m, e in err_curves.items():
    label = rollout_labels.get(m, m)
    print(f"{label:<20} {e[50].item():>16.3f} {e[200].item():>17.3f} {e[500].item():>17.3f}")

# ═════════════════════════════════════════════════════════════════════════════
# Experiment 2 — Local residual vs radius near each fixed point
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("EXPERIMENT 2 — Local vector-field residual near fixed points")
print("=" * 70)

fp_xy = float(np.sqrt(BETA * (RHO - 1.0)))        # ≈ 8.485
C_plus  = torch.tensor([ fp_xy,  fp_xy, RHO - 1.0], dtype=torch.float64)
C_minus = torch.tensor([-fp_xy, -fp_xy, RHO - 1.0], dtype=torch.float64)
print(f"C+ = {C_plus.tolist()},  C- = {C_minus.tolist()}")

def local_metrics(center, radius, state):
    """
    Over all training points within `radius` of `center`:
      mean ‖ε_k(x)‖  where k is the proximity-selected active cluster.
    Returns (n_points, mean_eps_norm, mean_f_norm, relative_pct).
    """
    dists = torch.linalg.norm(X_tr - center, dim=1)
    mask  = dists < radius
    if mask.sum() == 0:
        return (0, float('nan'), float('nan'), float('nan'))
    X_loc = X_tr[mask]
    F_loc = F_tr[mask]
    F_hat = piecewise_linear_f(X_loc, state)
    eps   = F_loc - F_hat
    eps_n = torch.linalg.norm(eps, dim=1).mean().item()
    f_n   = torch.linalg.norm(F_loc, dim=1).mean().item()
    return (int(mask.sum()), eps_n, f_n, 100.0 * eps_n / f_n if f_n > 0 else float('nan'))

print()
print(f"{'fp':<4} {'r':>5} {'n':>6}   "
      f"{'mean‖ε‖ ours':>14} {'mean‖f‖':>10} {'rel% ours':>10}   "
      f"{'mean‖ε‖ gmm':>14} {'rel% gmm':>10}")
for label, fp in [('C+', C_plus), ('C-', C_minus)]:
    for r in (1.0, 2.0, 5.0, 10.0):
        n_o, eo, fo, ro = local_metrics(fp, r, state_ours)
        n_g, eg, fg, rg = local_metrics(fp, r, state_gmm)
        print(f"{label:<4} {r:>5.1f} {n_o:>6}   "
              f"{eo:>14.3f} {fo:>10.3f} {ro:>10.2f}   "
              f"{eg:>14.3f} {rg:>10.2f}")

# ═════════════════════════════════════════════════════════════════════════════
# Experiment 3 — One-step state-space error on held-out test set
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("EXPERIMENT 3 — One-step state prediction error on held-out test set")
print("=" * 70)

# True next state from test trajectory (they are consecutive solve_ivp samples)
X_te_next = X_all[4001:5000]       # x_{t+1}  (shape 999, 3)
X_te_in   = X_all[4000:4999]       # x_t

def one_step_ours(X_in, state):
    return X_in + dt * piecewise_linear_f(X_in, state)

def one_step_edmd(X_in, model):
    # predict() maps x_t -> x_{t+1} in batched fashion
    return torch.tensor(model.predict(X_in.numpy()), dtype=torch.float64)

def one_step_metrics(name, X_pred):
    err = torch.linalg.norm(X_pred - X_te_next, dim=1)
    mean_err    = err.mean().item()
    median_err  = err.median().item()
    rel = mean_err / torch.linalg.norm(X_te_next - X_te_in, dim=1).mean().item()
    print(f"  {name:<14} mean ‖Δ‖ = {mean_err:.5f}   "
          f"median = {median_err:.5f}   "
          f"relative to ‖x_{{t+1}}-x_t‖ = {rel*100:.2f}%")
    return mean_err

onestep_ours = one_step_metrics("ours  (N=5)",   one_step_ours(X_te_in, state_ours))
onestep_gmm  = one_step_metrics("gmm   (N=5)",   one_step_ours(X_te_in, state_gmm))
# EDMD simulate starting from each X_te_in[i] for 1 step
edmd2_pred = one_step_edmd(X_te_in, edmd2)
edmd3_pred = one_step_edmd(X_te_in, edmd3)
onestep_edmd2 = one_step_metrics("EDMD-pk deg-2",  edmd2_pred)
onestep_edmd3 = one_step_metrics("EDMD-pk deg-3",  edmd3_pred)
# Our global EDMD one-step: x_{t+1} = x_t + dt * predict_f(x_t)
f_hat2 = predict_f_global_ours(X_te_in, edmd_ours2)
edmd_ours2_pred = X_te_in + dt * f_hat2
f_hat3 = predict_f_global_ours(X_te_in, edmd_ours3)
edmd_ours3_pred = X_te_in + dt * f_hat3
onestep_edmd_ours2 = one_step_metrics("EDMD-ours deg-2", edmd_ours2_pred)
onestep_edmd_ours3 = one_step_metrics("EDMD-ours deg-3", edmd_ours3_pred)

print(f"\n  baseline ‖x_{{t+1}}-x_t‖ mean = {torch.linalg.norm(X_te_next - X_te_in, dim=1).mean().item():.5f}")

# ═════════════════════════════════════════════════════════════════════════════
# Plot: rollout errors vs time, and one rollout in 3D
# ═════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(14, 5))
ax1 = fig.add_subplot(1, 2, 1)
ts = np.arange(n_steps_rollout + 1) * dt
for m, e in err_curves.items():
    ax1.semilogy(ts, e.numpy(), label=m, lw=1.5)
ax1.axhline(attractor_diameter, color='k', ls=':', lw=1, label=f'attractor ø ≈ {attractor_diameter:.0f}')
ax1.set_xlabel("time (s)")
ax1.set_ylabel(r"$\|\hat x_t - x_t\|$")
ax1.set_title("Rollout error vs truth (mean over 3 initial conditions)")
ax1.legend()
ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
tru = rollouts['truth'][0].numpy()
our = rollouts['ours' ][0].numpy()
ed  = rollouts['edmd3'][0].numpy()
ax2.plot(*tru.T, lw=0.6, color='k',    label='truth')
ax2.plot(*our.T, lw=0.6, color='C0',   label='ours')
ax2.plot(*ed .T, lw=0.6, color='C3',   label='edmd3')
ax2.set_title("Rollout trajectories (first init)")
ax2.legend()

plt.tight_layout()
plt.savefig(fig_path("validation_rollouts.png"), dpi=120)
print("\n→ saved validation_rollouts.png")

# ── Save raw data ────────────────────────────────────────────────────────────
import json
step_baseline = torch.linalg.norm(X_te_next - X_te_in, dim=1).mean().item()
raw = {
    "data_seed": args.seed,
    "attractor_diameter": attractor_diameter,
    "step_baseline": step_baseline,
    "rollout_err_at_step": {
        m: {"50": e[50].item(), "200": e[200].item(), "500": e[500].item()}
        for m, e in err_curves.items()
    },
    "one_step_mean_err": {
        "ours": onestep_ours,
        "gmm": onestep_gmm,
        "edmd_pk_deg2": onestep_edmd2,
        "edmd_pk_deg3": onestep_edmd3,
        "edmd_ours_deg2": onestep_edmd_ours2,
        "edmd_ours_deg3": onestep_edmd_ours3,
    },
}
with open(data_path("validation_lorenz_vs_edmd.json"), "w") as fp:
    json.dump(raw, fp, indent=2)
print(f"Raw data saved to {data_path('validation_lorenz_vs_edmd.json')}")
