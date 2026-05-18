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

CSV_PATH = Path('/Users/lorenzoalencartomaz/projects/client/fractal/bayesian_cluster_linear_approximation/residual_aware_clustering/analysis_2026-05-11_16-18-16_train_cap_4k_rollout_steps_cap_200/all_results.csv')
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

    # Step 1: per-(config, method) mean across seeds.
    seed_groups = defaultdict(list)
    for r in sys_rows:
        seed_groups[(r['config_name'], r['method'])].append(float(r['value']))
    cfg_method_means = {}
    for (cfg, m), vals in seed_groups.items():
        if len(vals) < 2: continue
        cfg_method_means[(cfg, m)] = statistics.mean(vals)

    # Step 2: collapse the 12-config cluster per method into a single point at
    # its best-case (smallest mean error) configuration. The previous version
    # plotted all 12 config points per method, which was visually dense and
    # mostly redundant since labels and Pareto frontier already operate per
    # method. One point per (method, hyperparameter) keeps the Pareto envelope
    # clear without losing any information that the figure was actually using.
    by_method = defaultdict(list)
    for (cfg, m), mean in cfg_method_means.items():
        by_method[m].append(mean)

    # Step 2b: restrict EDMD and GMM-EDMD to lift degrees we also ran CW-EDMD
    # at. Including higher-q EDMD (e.g. q=8 on Pendulum) when CW-EDMD was only
    # run at q=2,4 would render an unfair cross-degree comparison: the paper's
    # claim is matched-degree CW-EDMD-q beats EDMD-q, so the figure should
    # show only the lift degrees where that comparison is actually defined.
    cw_qs = set()
    for m in by_method:
        if family(m) == 'CW-EDMD':
            q_m = re.search(r'q=(\d+)', m.lower())
            if q_m: cw_qs.add(int(q_m.group(1)))

    # Plot only the two families the matched-q story is about: EDMD baseline
    # and CW-EDMD (the focus). Ablation variants (GMM-EDMD, CW-Taylor,
    # GMM-Taylor) are covered quantitatively in Table 5 and Appendix E and
    # would otherwise crowd the figure with five marker shapes for two
    # conceptually distinct comparisons.
    agg = []
    for m, means in by_method.items():
        fam = family(m)
        if fam not in ('EDMD', 'CW-EDMD'): continue
        if fam == 'EDMD':
            q_m = re.search(r'q=(\d+)', m.lower())
            if q_m and int(q_m.group(1)) not in cw_qs: continue
        best = min(means)
        p, ps = params(m, d)
        if p is None or not np.isfinite(best) or best <= 0: continue
        agg.append({'method': m, 'family': fam, 'params': p,
                    'mean': best, 'param_str': ps,
                    'n_configs_seen': len(means)})

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
    # Label both EDMD and CW-EDMD frontier points so each marker carries its
    # ($q$, $K$) configuration. Since the families are at distinct parameter
    # counts (EDMD has at most one point per $q$; CW-EDMD has one per ($q,K$)
    # combination), the labels do not collide along the x-axis.
    frontier_label_pts = sorted(dedup.values(), key=lambda a: a['params'])

    fig, ax = plt.subplots(figsize=(10.5, 6.7))

    # Plot only points on the Pareto frontier. Non-frontier configurations
    # would just clutter the figure: they are dominated by some plotted point
    # in both parameter count and error, so showing them adds no information
    # about the tradeoff curve.
    families_present = sorted({a['family'] for a in pareto_pts},
                              key=lambda f: _FAM_ORDER.get(f, 99))
    for fam in families_present:
        sub = [a for a in pareto_pts if a['family'] == fam]
        if not sub: continue
        xs = np.array([a['params'] for a in sub])
        ys = np.array([a['mean'] for a in sub])
        color = _FAM_COLOR.get(fam, '#7f7f7f')
        marker = _FAM_MARKER.get(fam, 'o')
        msize = _FAM_SIZE.get(fam, 7)
        z = _FAM_ZORDER.get(fam, 3)
        ax.plot(xs, ys, marker, color=color, markersize=msize,
                markeredgecolor='black', markeredgewidth=0.6,
                linestyle='none', zorder=z, alpha=0.92, label=fam)

    # Pareto frontier line
    pareto_pts.sort(key=lambda a: a['params'])
    px = [a['params'] for a in pareto_pts]
    py = [a['mean']   for a in pareto_pts]
    ax.plot(px, py, '--', color='black', alpha=0.55, linewidth=1.4,
            zorder=2, label='Tradeoff frontier')

    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.canvas.draw()

    # Two-axis staggering: alternate above/below AND left/right of the marker so
    # adjacent frontier labels can never stack. Each entry is (dx_mult, dy_mult)
    # on the underlying log-scale data. After placing each label we also repel
    # it from previously-placed labels if their bboxes would otherwise overlap.
    # Pure-vertical labels, all placed ABOVE their markers (never below — placing
    # below the rightmost low-y points pushed labels into the x-axis text). The
    # offsets alternate between two heights so adjacent labels at similar x do
    # not stack on the rendered y-axis. All multipliers are >1.
    dy_offsets = [1.8, 4.0, 2.4, 7.0, 3.2, 12.0, 4.8, 20.0]
    placed = []
    for i, a in enumerate(frontier_label_pts):
        ps = a['param_str']
        if not ps: continue
        dy = dy_offsets[i % len(dy_offsets)]
        text_x = a['params']
        text_y = a['mean'] * dy
        # Repel from previously-placed labels: if too close in log-x AND log-y,
        # push current label further upward.
        for (px, py) in placed:
            if px <= 0 or py <= 0: continue
            if abs(np.log10(text_x) - np.log10(px)) < 0.10 \
               and abs(np.log10(text_y) - np.log10(py)) < 0.25:
                text_y *= 1.55
        placed.append((text_x, text_y))
        va = 'bottom'
        ha = 'center'
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
    n_configs = max((a['n_configs_seen'] for a in agg), default=0)
    sys_disp = SYSTEM_DISPLAY.get(system, system.capitalize())
    ax.set_title(f"{sys_disp}: accuracy-parameter tradeoff at matched lift degree\n"
                 f"(best {metric_label} over {n_configs} configurations per method; "
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
