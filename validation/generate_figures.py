"""
Generate paper-quality figures from saved model states.

Requires: papers/data/lorenz_models.pt, pendulum_models.pt,
          statistical_lorenz.json, statistical_pendulum.json

Generates:
  - Phase-space rollout comparisons (best model per family)
  - Cluster density regions with centers
  - Statistical comparison bar charts with error bars

Run: python -m validation.generate_figures
"""

import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.integrate import solve_ivp

from utils.paths import fig_path, data_path
from utils.stats import confidence_interval
from models.distributions import mvn_logpdf_batch
from models.em_local_edmd import predict_f_all_clusters, monomials
from models.em_local_edmd_discrete import (
    predict_next_global, predict_next_all_clusters as predict_next_all_disc,
)
from simulators.lorenz import f as lorenz_f
from simulators.pendulum import (
    f as pendulum_f, generate_trajectory, wrap_theta, angular_dist,
)

torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser(description="Generate paper figures from saved data")
parser.add_argument('--skip-lorenz', action='store_true')
parser.add_argument('--skip-pendulum', action='store_true')
args = parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def pick_cluster(x, state):
    log_pi = torch.log(state['pi']).unsqueeze(0)
    log_prox = mvn_logpdf_batch(x, state['centers'], state['covariances'])
    return (log_pi + log_prox).argmax(dim=1)


def draw_ellipses(ax, centers, covariances, pi, n_std=2.0, dims=(0, 1), **kwargs):
    """Draw 2D ellipses for cluster covariances projected onto dims."""
    for k in range(centers.shape[0]):
        c = centers[k, list(dims)].numpy()
        cov = covariances[k][np.ix_(list(dims), list(dims))].numpy()
        vals, vecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        w, h = 2 * n_std * np.sqrt(vals)
        alpha = min(float(pi[k]) * 3, 0.4)
        ell = Ellipse(xy=c, width=w, height=h, angle=angle,
                      alpha=alpha, **kwargs)
        ax.add_patch(ell)
        ax.plot(*c, 'k*', markersize=8, zorder=5)


# ─────────────────────────────────────────────────────────────────────────────
# LORENZ FIGURES
# ─────────────────────────────────────────────────────────────────────────────

if not args.skip_lorenz:
    print("Loading Lorenz data...")
    lm = torch.load(data_path("lorenz_models.pt"), weights_only=False)
    with open(data_path("statistical_lorenz.json")) as f:
        ls = json.load(f)

    X_all = lm['X_all']
    dt = 0.01

    # ── Fig 1: 3D attractor with cluster regions ─────────────────────────
    fig = plt.figure(figsize=(14, 5))

    for idx, (name, state, title) in enumerate([
        ('taylor', lm['taylor'], 'Taylor-analytic clusters'),
        ('local_edmd_disc', lm['local_edmd_disc'], 'Local discrete EDMD clusters'),
        ('gmm', lm['gmm'], 'GMM clusters'),
    ]):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        labels = pick_cluster(X_all, state)
        N = state['N']
        for k in range(N):
            mask = labels == k
            pts = X_all[mask].numpy()
            ax.scatter(pts[::3, 0], pts[::3, 1], pts[::3, 2],
                       s=1, alpha=0.3, label=f'C{k} (n={mask.sum()})')
        # Plot centers
        c = state['centers'].numpy()
        ax.scatter(c[:, 0], c[:, 1], c[:, 2], c='k', s=80, marker='*', zorder=10)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    plt.tight_layout()
    plt.savefig(fig_path("lorenz_clusters_3d.png"), dpi=150)
    print(f"  Saved: {fig_path('lorenz_clusters_3d.png')}")

    # ── Fig 2: Cluster density regions (2D projections) ──────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    projections = [(0, 1, 'x', 'y'), (0, 2, 'x', 'z'), (1, 2, 'y', 'z')]
    state = lm['local_edmd_disc']
    labels = pick_cluster(X_all, state)

    for ax, (d0, d1, xl, yl) in zip(axes, projections):
        # Data colored by cluster
        for k in range(state['N']):
            mask = labels == k
            pts = X_all[mask].numpy()
            ax.scatter(pts[::2, d0], pts[::2, d1], s=1, alpha=0.2)

        # Density ellipses
        draw_ellipses(ax, state['centers'], state['covariances'], state['pi'],
                      n_std=2.0, dims=(d0, d1), edgecolor='black',
                      facecolor='none', linewidth=1.5)
        draw_ellipses(ax, state['centers'], state['covariances'], state['pi'],
                      n_std=1.0, dims=(d0, d1), edgecolor='black',
                      facecolor='gray', linewidth=0.5)

        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f'Local EDMD clusters ({xl}-{yl} plane)')

    plt.tight_layout()
    plt.savefig(fig_path("lorenz_cluster_density.png"), dpi=150)
    print(f"  Saved: {fig_path('lorenz_cluster_density.png')}")

    # ── Fig 3: Rollout comparison ────────────────────────────────────────
    x0 = X_all[4000]
    n_roll = 500

    def lorenz_truth(x0, n):
        sol = solve_ivp(lambda t, y: lorenz_f(y), (0, n * dt), x0.numpy(),
                        t_eval=np.linspace(0, n * dt, n + 1),
                        method='RK45', rtol=1e-10, atol=1e-10)
        return torch.tensor(sol.y.T, dtype=torch.float64)

    def lorenz_rollout_taylor(x0, state, n):
        traj = torch.zeros(n + 1, 3, dtype=torch.float64)
        traj[0] = x0
        for t in range(n):
            k = pick_cluster(traj[t:t+1], state)
            c = state['centers'][k]; fc = state['f_centers'][k]; J = state['jacobians'][k]
            f_hat = fc + (J @ (traj[t:t+1] - c).unsqueeze(-1)).squeeze(-1)
            traj[t+1] = traj[t] + dt * f_hat[0]
        return traj

    def lorenz_rollout_disc(x0, state, n, d=3):
        traj = torch.zeros(n + 1, d, dtype=torch.float64)
        traj[0] = x0
        for t in range(n):
            k = pick_cluster(traj[t:t+1], state)
            preds = predict_next_all_disc(traj[t:t+1], state['centers'],
                                          state['K_ops'], state['exps'], d)
            traj[t+1] = preds[0, k[0]]
        return traj

    def lorenz_rollout_global_disc(x0, model, n):
        traj = torch.zeros(n + 1, 3, dtype=torch.float64)
        traj[0] = x0
        for t in range(n):
            traj[t+1] = predict_next_global(traj[t:t+1], model)[0]
        return traj

    truth = lorenz_truth(x0, n_roll)
    traj_taylor = lorenz_rollout_taylor(x0, lm['taylor'], n_roll)
    traj_disc = lorenz_rollout_disc(x0, lm['local_edmd_disc'], n_roll)
    traj_global = lorenz_rollout_global_disc(x0, lm['global_edmd_disc'], n_roll)

    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(*truth.numpy().T, 'k-', lw=0.8, label='Truth (RK45)')
    ax1.plot(*traj_taylor.numpy().T, '-', color='C0', lw=0.6, alpha=0.8, label='Taylor-analytic')
    ax1.plot(*traj_disc.numpy().T, '-', color='C2', lw=0.6, alpha=0.8, label='Local EDMD-disc')
    ax1.plot(*traj_global.numpy().T, '--', color='C3', lw=0.6, alpha=0.8, label='Global EDMD-disc')
    ax1.set_title('Rollout trajectories (500 steps)')
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(1, 2, 2)
    ts = np.arange(n_roll + 1) * dt
    ax2.semilogy(ts, (traj_taylor - truth).norm(dim=1).numpy(), label='Taylor-analytic', color='C0')
    ax2.semilogy(ts, (traj_disc - truth).norm(dim=1).numpy(), label='Local EDMD-disc', color='C2')
    ax2.semilogy(ts, (traj_global - truth).norm(dim=1).numpy(), label='Global EDMD-disc', color='C3')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('||error||')
    ax2.set_title('Rollout error vs time')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_path("lorenz_rollout_comparison.png"), dpi=150)
    print(f"  Saved: {fig_path('lorenz_rollout_comparison.png')}")

    # ── Fig 4: Statistical bar chart ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    methods = [k for k in ls['results'].keys() if 'one_step' in ls['results'][k]]
    means = [confidence_interval(ls['results'][m]['one_step'])[0] for m in methods]
    cis = [confidence_interval(ls['results'][m]['one_step'])[1] for m in methods]

    colors = []
    for m in methods:
        if 'Taylor' in m: colors.append('C0')
        elif 'GMM' in m: colors.append('C1')
        elif 'Local' in m: colors.append('C2')
        elif 'disc' in m: colors.append('C3')
        elif 'pk' in m: colors.append('C4')
        else: colors.append('C5')

    x = np.arange(len(methods))
    ax.bar(x, means, yerr=cis, capsize=3, color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('One-step state error')
    ax.set_title(f'Lorenz: all methods (mean ± 95% CI, n={ls["n_seeds"]})')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(fig_path("lorenz_statistical_all.png"), dpi=150)
    print(f"  Saved: {fig_path('lorenz_statistical_all.png')}")

    print("  Lorenz figures done.")


# ─────────────────────────────────────────────────────────────────────────────
# PENDULUM FIGURES
# ─────────────────────────────────────────────────────────────────────────────

if not args.skip_pendulum:
    print("\nLoading Pendulum data...")
    pm = torch.load(data_path("pendulum_models.pt"), weights_only=False)
    with open(data_path("statistical_pendulum.json")) as f:
        ps = json.load(f)

    X_tr = pm['X_tr']
    F_tr = pm['F_tr']
    dt = 0.05
    d = 2

    # ── Fig 5: Phase-space with cluster density regions ──────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (name, state, title) in zip(axes, [
        ('taylor', pm['taylor'], 'Taylor-analytic clusters'),
        ('local_edmd', pm['local_edmd'], 'Local EDMD clusters'),
        ('taylor_ls', pm['taylor_ls'], 'Taylor-LS clusters'),
    ]):
        labels = pick_cluster(X_tr, state)
        for k in range(state['N']):
            mask = labels == k
            pts = X_tr[mask].numpy()
            ax.scatter(pts[::2, 0], pts[::2, 1], s=2, alpha=0.3)

        draw_ellipses(ax, state['centers'], state['covariances'], state['pi'],
                      n_std=2.0, dims=(0, 1), edgecolor='black',
                      facecolor='none', linewidth=1.5)
        draw_ellipses(ax, state['centers'], state['covariances'], state['pi'],
                      n_std=1.0, dims=(0, 1), edgecolor='black',
                      facecolor='gray', linewidth=0.5)

        ax.set_xlabel(r'$\theta$'); ax.set_ylabel(r'$\dot{\theta}$')
        ax.set_title(title, fontsize=10)

    plt.tight_layout()
    plt.savefig(fig_path("pendulum_cluster_density.png"), dpi=150)
    print(f"  Saved: {fig_path('pendulum_cluster_density.png')}")

    # ── Fig 6: Phase-space rollout comparison ────────────────────────────
    rollout_inits = [
        (torch.tensor([0.3, 0.0]),  'small osc'),
        (torch.tensor([2.8, 0.0]),  'near inverted'),
        (torch.tensor([-2.0, 1.0]), 'large swing'),
    ]
    n_roll = 200

    def pendulum_rollout_taylor(x0, state, n):
        traj = torch.zeros(n + 1, d, dtype=torch.float64)
        traj[0] = x0
        for t in range(n):
            k = pick_cluster(traj[t:t+1], state)
            c = state['centers'][k]; fc = state['f_centers'][k]; J = state['jacobians'][k]
            f_hat = fc + (J @ (traj[t:t+1] - c).unsqueeze(-1)).squeeze(-1)
            traj[t+1] = wrap_theta(traj[t] + dt * f_hat[0])
        return traj

    def pendulum_rollout_local(x0, state, n):
        traj = torch.zeros(n + 1, d, dtype=torch.float64)
        traj[0] = x0
        for t in range(n):
            k = pick_cluster(traj[t:t+1], state)
            F_all = predict_f_all_clusters(
                traj[t:t+1], state['centers'], state['M_ops'], state['exps'], d)
            f_hat = F_all[0, k[0]]
            traj[t+1] = wrap_theta(traj[t] + dt * f_hat)
        return traj

    def pendulum_rollout_global(x0, g, n):
        traj = torch.zeros(n + 1, d, dtype=torch.float64)
        traj[0] = x0
        for t in range(n):
            U = traj[t:t+1] - g['c']
            Phi = monomials(U, g['exps'])
            Phi_dot = Phi @ g['M'].T
            f_hat = Phi_dot[0, 1:d+1]
            traj[t+1] = wrap_theta(traj[t] + dt * f_hat)
        return traj

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (x0, label) in zip(axes, rollout_inits):
        tru = torch.tensor(generate_trajectory(x0.numpy(), n_steps=n_roll, dt=dt),
                           dtype=torch.float64)

        t_taylor = pendulum_rollout_taylor(x0, pm['taylor'], n_roll)
        t_local = pendulum_rollout_local(x0, pm['local_edmd'], n_roll)
        t_global = pendulum_rollout_global(x0, pm['global_edmd'], n_roll)

        ax.plot(tru[:, 0].numpy(), tru[:, 1].numpy(), 'k-', lw=2, label='Truth')
        ax.plot(t_taylor[:, 0].numpy(), t_taylor[:, 1].numpy(), '-',
                color='C0', alpha=0.8, lw=1, label='Taylor-analytic')
        ax.plot(t_local[:, 0].numpy(), t_local[:, 1].numpy(), '-',
                color='C2', alpha=0.8, lw=1, label='Local EDMD')
        ax.plot(t_global[:, 0].numpy(), t_global[:, 1].numpy(), '--',
                color='C3', alpha=0.8, lw=1, label=f'Global EDMD deg={pm["global_edmd_degree"]}')
        ax.scatter(*x0.numpy(), s=60, marker='*', c='k', zorder=10)
        ax.set_xlabel(r'$\theta$'); ax.set_ylabel(r'$\dot{\theta}$')
        ax.set_title(f'{label}  (θ₀={x0[0]:.1f})', fontsize=10)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_path("pendulum_rollout_comparison.png"), dpi=150)
    print(f"  Saved: {fig_path('pendulum_rollout_comparison.png')}")

    # ── Fig 7: Statistical comparison ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, ylabel, title in [
        (axes[0], 'one_step', 'One-step f-error', 'Pendulum: one-step error'),
        (axes[1], 'rollout_10s', 'Rollout @ 10s (rad)', 'Pendulum: rollout error'),
    ]:
        methods = [k for k in ps['results'] if metric in ps['results'][k]]
        means = [confidence_interval(ps['results'][m][metric])[0] for m in methods]
        cis = [confidence_interval(ps['results'][m][metric])[1] for m in methods]

        colors = []
        for m in methods:
            if 'Global' in m: colors.append('C3')
            elif 'local' in m: colors.append('C0')
            elif 'Taylor-a' in m: colors.append('C2')
            elif 'Taylor-L' in m: colors.append('C4')
            else: colors.append('C5')

        x = np.arange(len(methods))
        ax.bar(x, means, yerr=cis, capsize=3, color=colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=6)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title} (n={ps["n_seeds"]})')
        ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(fig_path("pendulum_statistical_all.png"), dpi=150)
    print(f"  Saved: {fig_path('pendulum_statistical_all.png')}")

    print("  Pendulum figures done.")

print("\nAll figures saved to papers/figures/")
