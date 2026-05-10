"""Re-render all three Pareto figures: B + E + F with polish for publication.

Fixes:
- Proper axis labels (no underscores, no piped raw metric strings)
- Publication-style titles (system + metric + horizon, formatted prose)
- Larger label offsets and longer leader lines to clear dense data clusters
- Boxed label backgrounds with stronger borders for readability
"""
import csv, re, statistics
from collections import defaultdict
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = Path('/Users/lorenzoalencartomaz/projects/client/fractal/bayesian_cluster_linear_approximation/residual_aware_clustering/analysis_2026-05-09_05-47-40_train_cap_4k_rollout_steps_cap_200/all_results.csv')
OUT_DIR  = Path('/Users/lorenzoalencartomaz/projects/client/fractal/bayesian_cluster_linear_approximation/residual_aware_clustering/papers/scml2026')

_FAMILIES = [
    ('EDMD',           '#DE8F05',  '^',    9,  4, 0),
    ('CW-EDMD',        '#0173B2',  'o',    8,  5, 1),
    ('GMM-EDMD',       '#949494',  'x',    8,  3, 2),
    ('CW-Taylor',      '#029E73',  'D',    7,  3, 3),
    ('GMM-Taylor',     '#D55E00',  's',    7,  3, 4),
    ('Other',          '#7f7f7f',  '.',    6,  1, 99),
]
_FAM_COLOR  = {n: c for (n, c, _, _, _, _) in _FAMILIES}
_FAM_MARKER = {n: m for (n, _, m, _, _, _) in _FAMILIES}
_FAM_SIZE   = {n: s for (n, _, _, s, _, _) in _FAMILIES}
_FAM_ZORDER = {n: z for (n, _, _, _, z, _) in _FAMILIES}
_FAM_ORDER  = {n: o for (n, _, _, _, _, o) in _FAMILIES}

D_BY_SYS = {'lorenz': 3, 'pendulum': 2, 'duffing': 2}
SYSTEM_DISPLAY = {'lorenz': 'Lorenz attractor',
                  'pendulum': 'Damped pendulum',
                  'duffing': 'Duffing oscillator'}
METRIC_DISPLAY = {'one_step': 'one-step prediction error',
                  'r1s': '1 s rollout error',
                  'r2s': '2 s rollout error',
                  'r5s': '5 s rollout error',
                  'r10s': '10 s rollout error'}

def family(name):
    n = name.lower().replace(' ', '').replace('_', '-')
    if 'gmm-edmd' in n or 'gmmedmd' in n: return 'GMM-EDMD'
    if 'gmm-taylor' in n or 'gmmtaylor' in n: return 'GMM-Taylor'
    if 'cw-edmd' in n or 'cwedmd' in n: return 'CW-EDMD'
    if 'cw-taylor' in n or 'cwtaylor' in n: return 'CW-Taylor'
    if 'edmd-pk' in n or 'edmd-pykoopman' in n: return 'EDMD-pykoopman'
    if 'gmm' in n and ('local-edmd' in n or 'localedmd' in n): return 'GMM-EDMD'
    if 'gmm' in n: return 'GMM-Taylor'
    if 'taylor' in n: return 'CW-Taylor'
    if 'local-edmd' in n or 'localedmd' in n: return 'CW-EDMD'
    if 'edmd-disc' in n or 'edmddisc' in n or n.startswith('edmd'): return 'EDMD'
    return 'Other'

def params(name, d):
    nl = name.lower().replace(' ', '').replace('_', '-')
    fam = family(name)
    K_match = re.search(r'k=(\d+)', nl)
    n_match = re.search(r'n=(\d+)', nl)
    N = int(K_match.group(1)) if K_match else (int(n_match.group(1)) if n_match else None)
    deg = None
    q_match = re.search(r'q=(\d+)', nl)
    if q_match: deg = int(q_match.group(1))
    else:
        m = re.search(r'deg[-=](\d+)', nl)
        if m: deg = int(m.group(1))
        elif re.search(r'\bd(\d+)', nl): deg = int(re.search(r'\bd(\d+)', nl).group(1))
    if fam in ('CW-Taylor', 'GMM-Taylor') and N is not None:
        return N * (d * d + d), f"K={N}"
    if fam in ('CW-EDMD', 'GMM-EDMD') and N is not None:
        d_eff = deg or 2
        M = comb(d + d_eff, d_eff)
        return N * M * M, f"q={d_eff}, K={N}"
    if fam == 'EDMD' and deg is not None:
        M = comb(d + deg, deg)
        return M * M, f"q={deg}"
    return None, ''

rows = list(csv.DictReader(open(CSV_PATH)))

def render(system, metric, out_name):
    d = D_BY_SYS[system]
    sys_rows = [r for r in rows if r['system'] == system and r['metric'] == metric
                and family(r['method']) != 'EDMD-pykoopman']

    groups = defaultdict(list)
    for r in sys_rows:
        groups[(r['config_name'], r['method'])].append(float(r['value']))

    agg = []
    for (cfg, m), vals in groups.items():
        if len(vals) < 2: continue
        mean = statistics.mean(vals)
        sd   = statistics.stdev(vals)
        ci   = 1.96 * sd / np.sqrt(len(vals))
        p, ps = params(m, d)
        if p is None or not np.isfinite(mean) or mean <= 0: continue
        agg.append({'config': cfg, 'method': m, 'family': family(m),
                    'params': p, 'mean': mean, 'ci': ci, 'param_str': ps})

    if not agg:
        return

    # Pareto frontier (deduplicated by method)
    pts_sorted = sorted(agg, key=lambda x: x['params'])
    pareto_pts = []
    best_y = float('inf')
    for a in pts_sorted:
        if a['mean'] < best_y:
            pareto_pts.append(a); best_y = a['mean']
    # Dedup by displayed label text (e.g. "q=5, K=2"): when two distinct methods
    # land on the frontier at essentially the same (params, error) — typical at
    # high q where CW-EDMD and GMM-EDMD become numerically indistinguishable —
    # we keep the lowest-mean representative so the figure carries a single
    # label per text, not duplicates.
    dedup = {}
    for a in pareto_pts:
        key = a['param_str']
        if not key: key = a['method']
        if key not in dedup or a['mean'] < dedup[key]['mean']:
            dedup[key] = a
    frontier_label_pts = sorted(dedup.values(), key=lambda a: a['params'])

    fig, ax = plt.subplots(figsize=(10.5, 6.7))

    families_present = sorted({a['family'] for a in agg},
                              key=lambda f: _FAM_ORDER.get(f, 99))
    for fam in families_present:
        sub = [a for a in agg if a['family'] == fam]
        if not sub: continue
        xs = np.array([a['params'] for a in sub])
        ys = np.array([a['mean'] for a in sub])
        cs = np.array([a['ci'] for a in sub])
        y_floor = max(ys.min() * 0.05, 1e-12) if len(ys) else 1e-12
        lo = np.clip(ys - cs, y_floor, None)
        err_low  = ys - lo
        err_high = cs
        color = _FAM_COLOR.get(fam, '#7f7f7f')
        marker = _FAM_MARKER.get(fam, 'o')
        msize = _FAM_SIZE.get(fam, 7)
        z = _FAM_ZORDER.get(fam, 3)
        ax.errorbar(xs, ys, yerr=[err_low, err_high],
                    fmt=marker, color=color, markersize=msize,
                    markeredgecolor='black', markeredgewidth=0.6,
                    elinewidth=0.7, capsize=2.5, zorder=z, alpha=0.92,
                    label=fam)

    # Pareto frontier line
    pareto_pts.sort(key=lambda a: a['params'])
    px = [a['params'] for a in pareto_pts]
    py = [a['mean']   for a in pareto_pts]
    ax.plot(px, py, '--', color='black', alpha=0.55, linewidth=1.4,
            zorder=2, label='Pareto frontier')

    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.canvas.draw()

    # Two-axis staggering: alternate above/below AND left/right of the marker so
    # adjacent frontier labels can never stack. Each entry is (dx_mult, dy_mult)
    # on the underlying log-scale data. After placing each label we also repel
    # it from previously-placed labels if their bboxes would otherwise overlap.
    base_offsets = [
        (0.78, 5.0),    # high-left
        (1.30, 0.16),   # low-right
        (0.62, 9.0),    # higher-left
        (1.65, 0.09),   # lower-right
        (0.50, 16.0),   # taller-left
        (2.10, 0.05),   # deeper-right
        (0.42, 28.0),   # tallest-left
        (2.70, 0.030),  # deepest-right
    ]
    placed = []
    for i, a in enumerate(frontier_label_pts):
        ps = a['param_str']
        if not ps: continue
        dx, dy = base_offsets[i % len(base_offsets)]
        text_x = a['params'] * dx
        text_y = a['mean']   * dy
        # Repel from previously-placed labels in log space
        for (px, py) in placed:
            if px <= 0 or py <= 0: continue
            if abs(np.log10(text_x) - np.log10(px)) < 0.18 \
               and abs(np.log10(text_y) - np.log10(py)) < 0.45:
                text_x *= (0.55 if dx < 1 else 1.85)
                text_y *= (0.55 if dy < 1 else 1.60)
        placed.append((text_x, text_y))
        va = 'bottom' if dy > 1 else 'top'
        ha = 'right' if dx < 1 else 'left'
        ax.annotate(ps, xy=(a['params'], a['mean']),
                    xytext=(text_x, text_y),
                    textcoords='data', ha=ha, va=va,
                    fontsize=9.5, color='#111111', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.32', fc='white',
                              ec='#444444', alpha=0.97, linewidth=0.8),
                    arrowprops=dict(arrowstyle='-|>', color='#444444',
                                    linewidth=0.85, shrinkA=2, shrinkB=7,
                                    mutation_scale=11),
                    zorder=10)

    # Publication-style axis labels and title
    metric_label = METRIC_DISPLAY.get(metric, metric.replace('_', ' '))
    ax.set_xlabel('Number of parameters (log scale)', fontsize=11)
    ax.set_ylabel(f'Mean {metric_label} (log scale)', fontsize=11)
    ax.grid(which='both', alpha=0.3, zorder=1)
    ax.set_axisbelow(True)
    n_configs = len({a['config'] for a in agg})
    sys_disp = SYSTEM_DISPLAY.get(system, system.capitalize())
    ax.set_title(f"{sys_disp}: parameters vs. {metric_label}\n"
                 f"(across {n_configs} configurations; mean ± 95% CI; "
                 "lower-left is better)",
                 fontsize=11)
    ax.legend(fontsize=10, loc='best', framealpha=0.94)
    plt.tight_layout()
    out_path = OUT_DIR / out_name
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path}")

render('lorenz',   'one_step', 'pareto_lorenz.png')
render('pendulum', 'one_step', 'pareto_pendulum.png')
render('duffing',  'one_step', 'pareto_duffing.png')
