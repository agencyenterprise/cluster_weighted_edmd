import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch


COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


# ── ELBO convergence ──────────────────────────────────────────────────────────

def plot_elbo(history: list, title: str = "ELBO Convergence", save: str = None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history, color='steelblue', linewidth=2)

    diffs = np.diff(history)
    bad   = np.where(diffs < -1e-4)[0]
    if len(bad):
        ax.scatter(bad + 1, np.array(history)[bad + 1],
                   color='red', zorder=5, s=40, label='ELBO decreased (bug)')
        ax.legend()

    ax.set_xlabel("EM Iteration")
    ax.set_ylabel("ELBO")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = save or f"{title.replace(' ', '_').lower()}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── 3D attractor with cluster coloring ───────────────────────────────────────

def plot_attractor_clusters(
    X:     torch.Tensor,
    r:     torch.Tensor,
    state: dict,
    title: str  = "Lorenz — Linearization Regions",
    save:  str  = None,
):
    assignments = r.argmax(dim=1).numpy()
    X_np        = X.numpy()
    centers_np  = state['centers'].numpy()
    N           = state['N']

    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111, projection='3d')

    for k in range(N):
        mask = assignments == k
        if mask.sum() == 0:
            continue
        ax.scatter(
            X_np[mask, 0], X_np[mask, 1], X_np[mask, 2],
            c=[COLORS[k % 10]], s=0.5, alpha=0.3,
            label=f'Cluster {k}  (n={mask.sum()})'
        )
        ax.scatter(
            *centers_np[k],
            c=[COLORS[k % 10]], s=250, marker='*',
            edgecolors='black', linewidths=0.8, zorder=10
        )

    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_title(title)
    ax.legend(markerscale=4, fontsize=8)
    plt.tight_layout()

    path = save or f"{title.replace(' ', '_').lower()}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Side-by-side comparison ───────────────────────────────────────────────────

def plot_comparison(
    X:              torch.Tensor,
    r_gmm:          torch.Tensor,
    state_gmm:      dict,
    r_ours:         torch.Tensor,
    state_ours:     dict,
    save:           str = "comparison.png",
):
    fig = plt.figure(figsize=(18, 8))

    for col, (r, state, title) in enumerate([
        (r_gmm,  state_gmm,  "Standard GMM\n(no residual term)"),
        (r_ours, state_ours, "Residual-Aware\n(our method)"),
    ]):
        ax          = fig.add_subplot(1, 2, col + 1, projection='3d')
        assignments = r.argmax(dim=1).numpy()
        X_np        = X.numpy()
        centers_np  = state['centers'].numpy()
        N           = state['N']

        for k in range(N):
            mask = assignments == k
            if mask.sum() == 0:
                continue
            ax.scatter(
                X_np[mask, 0], X_np[mask, 1], X_np[mask, 2],
                c=[COLORS[k % 10]], s=0.3, alpha=0.2
            )
            ax.scatter(
                *centers_np[k], c=[COLORS[k % 10]],
                s=300, marker='*',
                edgecolors='black', linewidths=1.0, zorder=10
            )

        ax.set_title(title, fontsize=13)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    plt.suptitle("Center Locations: GMM vs Residual-Aware", fontsize=14)
    plt.tight_layout()
    plt.savefig(save, dpi=150)
    plt.close()
    print(f"  Saved: {save}")


# ── Residual magnitude vs distance ────────────────────────────────────────────

def plot_residuals_per_cluster(
    X:     torch.Tensor,
    F:     torch.Tensor,
    r:     torch.Tensor,
    state: dict,
    save:  str = "residuals_per_cluster.png",
):
    N           = state['N']
    assignments = r.argmax(dim=1).numpy()

    fig, axes = plt.subplots(1, N, figsize=(5 * N, 4), squeeze=False)

    for k in range(N):
        ax   = axes[0, k]
        mask = assignments == k
        if mask.sum() == 0:
            ax.set_title(f"Cluster {k} (empty)")
            continue

        X_k   = X[mask]
        F_k   = F[mask]
        c_k   = state['centers'][k]
        f_ck  = state['f_centers'][k]
        J_k   = state['jacobians'][k]

        delta       = X_k - c_k
        linear_pred = f_ck + (J_k @ delta.T).T
        eps         = F_k - linear_pred

        residuals = (eps ** 2).sum(dim=1).sqrt().numpy()
        distances = (delta ** 2).sum(dim=1).sqrt().numpy()

        ax.scatter(distances, residuals, s=2, alpha=0.4, color=COLORS[k % 10])
        ax.set_xlabel(r'$\|x_i - c_k\|$')
        ax.set_ylabel(r'$\|\varepsilon_k(x_i)\|$')
        ax.set_title(f'Cluster {k}  (n={mask.sum()})')
        ax.grid(True, alpha=0.3)

    plt.suptitle("Linearization Residual vs Distance from Center", fontsize=13)
    plt.tight_layout()
    plt.savefig(save, dpi=150)
    plt.close()
    print(f"  Saved: {save}")


# ── Model selection curves ────────────────────────────────────────────────────

def plot_model_selection(
    elbo_by_N:  dict,
    bic_by_N:   dict,
    ml_by_N:    dict,
    save:       str = "model_selection.png",
):
    Ns    = sorted(elbo_by_N.keys())
    elbos = [elbo_by_N[n] for n in Ns]
    bics  = [bic_by_N[n]  for n in Ns]
    mls   = [ml_by_N[n]   for n in Ns]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    for ax, vals, label, color in zip(
        axes,
        [elbos, bics, mls],
        ['ELBO', 'BIC', 'Log Marginal Likelihood'],
        ['steelblue', 'darkorange', 'seagreen'],
    ):
        ax.plot(Ns, vals, 'o-', color=color, linewidth=2)
        best_n = Ns[int(np.argmax(vals))]
        ax.axvline(best_n, color='red', linestyle='--',
                   label=f'Best N={best_n}')
        ax.set_xlabel("N (number of clusters)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Model Selection", fontsize=14)
    plt.tight_layout()
    plt.savefig(save, dpi=150)
    plt.close()
    print(f"  Saved: {save}")


# ── Responsibility heatmap ────────────────────────────────────────────────────

def plot_responsibilities(
    r:    torch.Tensor,
    save: str = "responsibilities.png",
):
    r_np = r.numpy()
    P, N = r_np.shape

    # Subsample for readability
    idx  = np.random.choice(P, min(P, 500), replace=False)
    r_sub = r_np[idx]

    fig, ax = plt.subplots(figsize=(max(6, N * 1.5), 8))
    im = ax.imshow(r_sub, aspect='auto', cmap='hot', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Responsibility $r_{ik}$')
    ax.set_xlabel("Cluster k")
    ax.set_ylabel("Phase point i (subsample)")
    ax.set_title("Soft Assignment Responsibilities")
    ax.set_xticks(range(N))
    plt.tight_layout()
    plt.savefig(save, dpi=150)
    plt.close()
    print(f"  Saved: {save}")
