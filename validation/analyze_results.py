"""
Compile multi-config statistical-validation outputs into paper-ready artifacts.

Reads every per-config JSON written by ``run_statistical.py --config <yaml>``
under ``papers/data/`` and emits, under ``papers/analysis/``:

- ``all_results.csv``         tidy long-form DataFrame
                              ``(system, config_name, method, seed, metric, value)``
- ``summary_<system>.md``     per-system markdown tables
                              (mean +/- 95% CI per method x config, one table per metric)
- ``summary_<system>.tex``    LaTeX-ready versions of the same tables
- ``paired_tests.md``         residual-aware vs baseline, aggregated across configs
                              (count of significant configs + median p-value + mean effect)
- ``ablation_<system>.png``   per-system ablation heatmap (rows=method, cols=config)
- ``robustness_<system>.png`` cross-config robustness lines / bars per method

Usage
-----
After running any subset of configs::

    ./run_all.sh                 # produces 36 per-config JSONs
    python -m validation.analyze_results

Or restrict to one system::

    python -m validation.analyze_results --systems duffing

Conventions
-----------
- A JSON is included in the corpus iff it contains a ``config_name`` field
  (= produced by the YAML-driven run mode). Legacy single-config outputs
  written without ``--config`` are skipped silently.
- The "headline metric" is the largest configured rollout horizon for that
  system; everything else (one_step, intermediate horizons) is also reported.
"""

import argparse
import json
import re
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from utils.paths import data_path
from utils.stats import confidence_interval, paired_test


# -- CLI ----------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Compile multi-config statistical results into paper artifacts.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--systems', nargs='+', default=['lorenz', 'pendulum', 'duffing'],
                    help="restrict the analysis to a subset of systems")
parser.add_argument('--data-dir', type=str, default=None,
                    help="override input directory (default: papers/data/)")
parser.add_argument('--out-dir',  type=str, default=None,
                    help="override output directory (default: papers/analysis/)")
parser.add_argument('--metrics-table', nargs='+',
                    default=['one_step', 'r5s', 'r10s', 'r20s'],
                    help="metrics to include in the per-system summary tables")
parser.add_argument('--from-csv', type=str, default=None,
                    help="Skip JSON-corpus loading and read the long-form "
                         "DataFrame directly from this CSV. Useful for "
                         "re-rendering figures after changing the plotting "
                         "code without re-running the full sweep.")
args = parser.parse_args()


# -- Paths --------------------------------------------------------------------

DATA_DIR = Path(args.data_dir) if args.data_dir else Path(data_path('.')).resolve()
OUT_DIR  = (Path(args.out_dir) if args.out_dir
            else (DATA_DIR.parent / 'analysis').resolve())
OUT_DIR.mkdir(exist_ok=True, parents=True)


# -- Corpus loader ------------------------------------------------------------

def _system_of(config_name: str, args_dict: dict) -> str:
    """Best-effort recovery of the system name from a config JSON."""
    for sys in ('lorenz', 'pendulum', 'duffing'):
        if sys in config_name.lower():
            return sys
    # Fallback: scan args namespace for system-specific keys
    for key in args_dict or {}:
        for sys in ('lorenz', 'pendulum', 'duffing'):
            if key.startswith(f"{sys}_"):
                return sys
    return 'unknown'


def load_corpus(data_dir: Path, systems_filter):
    """Return a tidy long-form DataFrame of every per-config JSON in ``data_dir``."""
    rows = []
    n_loaded = 0
    n_skipped = 0
    for jp in sorted(data_dir.glob('*.json')):
        try:
            payload = json.loads(jp.read_text())
        except Exception:
            n_skipped += 1
            continue

        config_name = payload.get('config_name')
        if not config_name:
            n_skipped += 1
            continue

        system = _system_of(config_name, payload.get('args', {}) or {})
        if system not in systems_filter:
            continue

        seeds   = payload.get('seeds', [])
        results = payload.get('results', {})
        for method, metrics in results.items():
            for metric, values in metrics.items():
                if not isinstance(values, list):
                    continue
                for seed_idx, val in enumerate(values):
                    if val is None or not np.isfinite(val):
                        continue
                    rows.append({
                        'system':      system,
                        'config_name': config_name,
                        'method':      method,
                        'seed':        seeds[seed_idx] if seed_idx < len(seeds) else seed_idx,
                        'metric':      metric,
                        'value':       float(val),
                    })
        n_loaded += 1

    print(f"[corpus] loaded {n_loaded} config JSON(s) from {data_dir}; "
          f"skipped {n_skipped} (no config_name or unparseable)")
    return pd.DataFrame(rows)


# -- Aggregation --------------------------------------------------------------

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per ``(system, config_name, method, metric)``: mean +/- 95% CI."""
    if df.empty:
        return pd.DataFrame(columns=['system', 'config_name', 'method', 'metric',
                                     'n', 'mean', 'ci'])

    def _agg(v):
        m, hw, _, _ = confidence_interval(v.tolist())
        return pd.Series({'n': len(v), 'mean': m, 'ci': hw})

    agg = (df.groupby(['system', 'config_name', 'method', 'metric'])['value']
             .apply(_agg).unstack(-1).reset_index())
    return agg


# -- Tables -------------------------------------------------------------------

def _fmt(mean: float, ci: float) -> str:
    """Format a mean and CI half-width with sensible precision."""
    if not np.isfinite(mean):
        return "nan"
    if abs(mean) < 1e-3:
        return f"{mean:.2e} +/- {ci:.1e}"
    return f"{mean:.4f} +/- {ci:.4f}"


def _system_metrics(agg: pd.DataFrame, system: str, preferred):
    """Return the per-system metric list, intersected with the preferred order.

    The corpus may have system-specific metrics (e.g., Lorenz uses r0_5s, r1s
    while Pendulum uses r5s, r10s). Filtering globally to the intersection
    silently drops every rollout metric for systems that don't share it.
    Instead, take the per-system metrics that are present, ordered by the
    user's preferred list with any leftover metrics appended at the end.
    """
    sys_metrics  = set(agg[agg.system == system]['metric'].unique())
    head = [m for m in preferred if m in sys_metrics]
    tail = [m for m in sorted(sys_metrics) if m not in head and m != 'rel_pct']
    return head + tail


def summary_tables(agg: pd.DataFrame, metrics, out_dir: Path):
    """Write per-system markdown + latex tables (one per metric, columns=configs)."""
    written = []
    for system in sorted(agg['system'].unique()):
        sys_metrics = _system_metrics(agg, system, metrics)
        sys_md  = [f"# {system.capitalize()} -- multi-config summary",
                   "",
                   f"Mean +/- 95% CI across seeds for each (method, config, metric).",
                   f"Metrics included: {', '.join(f'`{m}`' for m in sys_metrics)}.",
                   ""]
        sys_tex = []

        for metric in sys_metrics:
            block = agg[(agg.system == system) & (agg.metric == metric)]
            if block.empty:
                continue
            # Pivot: rows=method, cols=config_name
            pivot = block.pivot_table(
                index='method', columns='config_name',
                values=['mean', 'ci'], aggfunc='first'
            )
            # Format each cell as "mean +/- ci"
            methods = sorted(pivot.index.tolist())
            configs = sorted({c for _, c in pivot.columns.tolist()})

            sys_md.append(f"## metric: `{metric}`")
            sys_md.append("")
            sys_md.append("| method | " + " | ".join(configs) + " |")
            sys_md.append("|---" * (len(configs) + 1) + "|")
            for m in methods:
                cells = []
                for c in configs:
                    try:
                        mean = pivot.loc[m, ('mean', c)]
                        ci   = pivot.loc[m, ('ci',   c)]
                        cells.append(_fmt(mean, ci))
                    except KeyError:
                        cells.append("--")
                sys_md.append(f"| {m} | " + " | ".join(cells) + " |")
            sys_md.append("")

            # LaTeX (booktabs)
            sys_tex.append(r"\begin{table}[h]\centering")
            sys_tex.append(r"\small")
            sys_tex.append(r"\caption{" + f"{system}: {metric} (mean $\\pm$ 95\\% CI)" + "}")
            sys_tex.append(r"\begin{tabular}{l" + "r" * len(configs) + "}")
            sys_tex.append(r"\toprule")
            sys_tex.append("method & " + " & ".join(c.replace('_', r'\_') for c in configs) + r" \\")
            sys_tex.append(r"\midrule")
            for m in methods:
                cells = []
                for c in configs:
                    try:
                        mean = pivot.loc[m, ('mean', c)]
                        ci   = pivot.loc[m, ('ci',   c)]
                        cells.append(_fmt(mean, ci).replace('+/-', r'$\pm$'))
                    except KeyError:
                        cells.append('--')
                sys_tex.append(m.replace('_', r'\_') + " & " + " & ".join(cells) + r" \\")
            sys_tex.append(r"\bottomrule")
            sys_tex.append(r"\end{tabular}")
            sys_tex.append(r"\end{table}" + "\n")

        md_path  = out_dir / f"summary_{system}.md"
        tex_path = out_dir / f"summary_{system}.tex"
        md_path.write_text("\n".join(sys_md))
        tex_path.write_text("\n".join(sys_tex))
        print(f"[tables] wrote {md_path.name}, {tex_path.name}")
        written.append(system)
    return written


# -- Paired tests aggregated across configs -----------------------------------

def _candidate_pairs(methods, system):
    """Heuristic pairs of (residual-aware, baseline) to compare.

    Pair categories built here:
      - CW-Taylor vs GMM-Taylor at matched K   (within-Taylor ablation)
      - CW-EDMD  vs GMM-EDMD  at matched (q, K) (within-EDMD ablation)
      - CW-Taylor vs EDMD                       (largest K vs lowest q)
      - CW-EDMD  vs EDMD                        (largest K vs lowest q)
    """
    pairs = []

    def _find(pattern):
        rx = re.compile(pattern, re.IGNORECASE)
        return [m for m in methods if rx.search(m)]

    # Within-Taylor ablation: CW-Taylor vs GMM-Taylor at matched K
    for K in (2, 4, 8, 16, 5, 12, 20, 50):
        a = _find(rf"^CW-Taylor[ ,]*K=\s*{K}\b")
        b = _find(rf"^GMM-Taylor[ ,]*K=\s*{K}\b")
        if not a:
            # Legacy fallback
            a = [m for m in _find(rf"Taylor[ -].*N=\s*{K}\b") if 'GMM' not in m]
        if not b:
            b = [m for m in _find(rf"GMM.*N=\s*{K}\b") if 'EDMD' not in m]
        if a and b:
            pairs.append((a[0], b[0], f"CW-Taylor vs GMM-Taylor at K={K}"))

    # Within-EDMD ablation: CW-EDMD vs GMM-EDMD at matched (q, K)
    for q in (2, 3, 4, 5):
        for K in (2, 4, 8, 16, 5, 12, 20):
            a = _find(rf"^CW-EDMD[ ,]*q=\s*{q}[ ,]*K=\s*{K}\b")
            b = _find(rf"^GMM-EDMD[ ,]*q=\s*{q}[ ,]*K=\s*{K}\b")
            if a and b:
                pairs.append((a[0], b[0], f"CW-EDMD vs GMM-EDMD at q={q}, K={K}"))

    # CW-Taylor vs EDMD
    taylors = _find(r"^CW-Taylor[ ,]*K=\s*\d+")
    if not taylors:
        taylors = [m for m in _find(r"Taylor.*N=\s*\d+") if 'GMM' not in m]
    globals_ = _find(r"^EDMD\s+q=\s*\d+|EDMD-disc\s+deg")
    if taylors and globals_:
        pairs.append((taylors[-1], globals_[0],
                      "CW-Taylor (largest K) vs EDMD (lowest q)"))

    # CW-EDMD vs EDMD
    cw_edmds = _find(r"^CW-EDMD[ ,]*q=\s*\d+[ ,]*K=\s*\d+")
    if not cw_edmds:
        cw_edmds = [m for m in _find(r"local-EDMD.*N=\s*\d+|Local-EDMD.*N=\s*\d+")
                    if 'GMM' not in m]
    if cw_edmds and globals_:
        pairs.append((cw_edmds[-1], globals_[0],
                      "CW-EDMD (largest K) vs EDMD (lowest q)"))

    return pairs


def cross_config_paired_tests(df: pd.DataFrame, metrics, out_dir: Path):
    """For each system + key method pair, paired t-test per config + summary."""
    out = ["# Cross-config paired tests",
           "",
           "Per-config paired t-test (seed-paired) between residual-aware and "
           "baseline methods, then summarized across configs.",
           ""]

    for system in sorted(df['system'].unique()):
        sys_df = df[df.system == system]
        methods = sorted(sys_df['method'].unique())
        pairs   = _candidate_pairs(methods, system)
        if not pairs:
            continue

        out.append(f"## {system.capitalize()}")
        out.append("")
        out.append("| comparison | metric | configs | wins (p<.05) | median p | mean diff |")
        out.append("|---|---|---|---|---|---|")

        for a, b, label in pairs:
            for metric in metrics:
                rows = []
                for cfg_name in sorted(sys_df.config_name.unique()):
                    cfg_df = sys_df[sys_df.config_name == cfg_name]
                    a_seeds = cfg_df[(cfg_df.method == a) & (cfg_df.metric == metric)] \
                              .sort_values('seed')
                    b_seeds = cfg_df[(cfg_df.method == b) & (cfg_df.metric == metric)] \
                              .sort_values('seed')
                    common = sorted(set(a_seeds.seed) & set(b_seeds.seed))
                    if len(common) < 3:
                        continue
                    av = a_seeds[a_seeds.seed.isin(common)] \
                         .sort_values('seed').value.values
                    bv = b_seeds[b_seeds.seed.isin(common)] \
                         .sort_values('seed').value.values
                    if not (np.isfinite(av).all() and np.isfinite(bv).all()):
                        continue
                    t = paired_test(av, bv)
                    rows.append((cfg_name, t['p_value'], t['mean_diff']))
                if not rows:
                    continue
                ps     = np.array([r[1] for r in rows])
                diffs  = np.array([r[2] for r in rows])
                # "win" for residual-aware = a's mean < b's mean (lower error is better)
                wins   = int(((diffs < 0) & (ps < 0.05)).sum())
                out.append(
                    f"| {a} vs {b} ({label}) | `{metric}` | "
                    f"{len(rows)} | {wins} / {len(rows)} | "
                    f"{np.median(ps):.4f} | {np.mean(diffs):+.4f} |"
                )

        out.append("")

    md_path = out_dir / "paired_tests.md"
    md_path.write_text("\n".join(out))
    print(f"[paired-tests] wrote {md_path.name}")


# -- Figures ------------------------------------------------------------------

# Method-family taxonomy. Order determines color/legend order in plots.
# Names match the paper terminology: EDMD (global), CW-EDMD (residual-aware
# partitioning + EDMD per cluster), GMM-EDMD (within-EDMD ablation:
# geometry-only responsibilities + EDMD per cluster), CW-Taylor (Taylor variant
# of CW-EDMD), GMM-Taylor (within-Taylor ablation = GMM-clustered local model
# of CWM literature).
_FAMILIES = [
    ('EDMD',                   '#ff7f0e'),    # orange -- single global operator
    ('CW-EDMD',                '#1f77b4'),    # blue -- our method
    ('GMM-EDMD',               '#8c564b'),    # brown -- within-EDMD ablation
    ('CW-Taylor',              '#2ca02c'),    # green -- Taylor variant (Appendix E)
    ('GMM-Taylor',             '#e377c2'),    # pink -- within-Taylor ablation
    ('EDMD-pykoopman',         '#999999'),    # external pykoopman (legacy)
    ('Other',                  '#7f7f7f'),
]
_FAMILY_COLOR = dict(_FAMILIES)
_FAMILY_ORDER = {name: i for i, (name, _) in enumerate(_FAMILIES)}

# State dimension per system (used to derive parameter counts from method labels).
_SYSTEM_D = {'lorenz': 3, 'pendulum': 2, 'duffing': 2}


def _method_family(name: str) -> str:
    """Map a method name to its family bucket for grouping/color."""
    n = name.lower().replace(' ', '').replace('_', '-')
    # Order matters: more specific prefixes first
    if 'gmm-edmd' in n or 'gmmedmd' in n:
        return 'GMM-EDMD'
    if 'gmm-taylor' in n or 'gmmtaylor' in n:
        return 'GMM-Taylor'
    if 'cw-edmd' in n or 'cwedmd' in n:
        return 'CW-EDMD'
    if 'cw-taylor' in n or 'cwtaylor' in n:
        return 'CW-Taylor'
    if 'edmd-pk' in n or 'edmd-pykoopman' in n:
        return 'EDMD-pykoopman'
    # Legacy fallback labels (pre-rename corpus compatibility)
    if 'gmm' in n and ('local-edmd' in n or 'localedmd' in n):
        return 'GMM-EDMD'
    if 'gmm' in n:
        return 'GMM-Taylor'
    if 'taylor-ls' in n or 'taylorls' in n:
        return 'Other'
    if 'taylor' in n:
        return 'CW-Taylor'
    if 'local-edmd' in n or 'localedmd' in n:
        return 'CW-EDMD'
    if 'edmd-disc' in n or 'edmddisc' in n or re.match(r'^edmd\s*q', n.replace('-', '')):
        return 'EDMD'
    if n.startswith('edmd'):
        return 'EDMD'
    return 'Other'


def _method_params(name: str, d: int):
    """Best-effort parameter count parsed from a method label.

    Returns ``(params, label_suffix)`` for plot annotation, or ``(None, '')``
    if the label can't be parsed.

    New label format (paper-aligned):
      'EDMD q=2'                 -> deg=2,        params = M_q^2
      'CW-EDMD q=2, K=8'         -> deg=2, K=8,   params = K * M_q^2
      'GMM-EDMD q=2, K=8'        -> deg=2, K=8,   params = K * M_q^2
      'CW-Taylor K=8'            -> K=8,          params = K * (d^2 + d)
      'GMM-Taylor K=8'           -> K=8,          params = K * (d^2 + d)
    Legacy formats also recognised for backward compat with older corpora.
    """
    nl = name.lower().replace(' ', '').replace('_', '-')
    fam = _method_family(name)
    from math import comb

    # Cluster count: prefer K=, fall back to N= for legacy labels
    K_match = re.search(r'k=(\d+)', nl)
    n_match = re.search(r'n=(\d+)', nl)
    N = int(K_match.group(1)) if K_match else (int(n_match.group(1)) if n_match else None)

    # Polynomial degree: prefer q=, fall back to deg=k or dN
    deg = None
    q_match = re.search(r'q=(\d+)', nl)
    if q_match:
        deg = int(q_match.group(1))
    else:
        deg_match = re.search(r'deg[-=](\d+)', nl)
        if deg_match:
            deg = int(deg_match.group(1))
        elif re.search(r'\bd(\d+)', nl):
            deg = int(re.search(r'\bd(\d+)', nl).group(1))

    if fam in ('CW-Taylor', 'GMM-Taylor') and N is not None:
        return N * (d * d + d), f"K={N}"
    if fam in ('CW-EDMD', 'GMM-EDMD') and N is not None:
        d_eff = deg or 2
        M = comb(d + d_eff, d_eff)
        return N * M * M, f"q={d_eff}, K={N}"
    if fam in ('EDMD', 'EDMD-pykoopman') and deg is not None:
        M = comb(d + deg, deg)
        return M * M, f"q={deg}"
    return None, ''


def _short_config(config_name: str, system: str) -> str:
    return config_name.replace(f"{system}_", "")


def _plot_method_bars(agg: pd.DataFrame, df: pd.DataFrame, system: str,
                      metric: str, config: str, out_dir: Path):
    """Per (system, config, metric) horizontal forest plot, sorted by mean.

    Forest-plot style (dot + whisker) is preferred over bars on a log axis
    because (a) bars don't have a meaningful "zero" on a log scale and
    (b) asymmetric CI clipping is visually clean.
    """
    block = agg[(agg.system == system) & (agg.metric == metric)
                & (agg.config_name == config)].copy()
    if block.empty:
        return
    block = block[np.isfinite(block['mean']) & (block['mean'] > 0)]
    if block.empty:
        return

    d = _SYSTEM_D.get(system, 2)
    block['family']    = block['method'].apply(_method_family)
    block['params']    = block['method'].apply(lambda m: _method_params(m, d)[0])
    block['param_str'] = block['method'].apply(lambda m: _method_params(m, d)[1])
    # Sort: best (lowest mean) at the TOP for fast scanning
    block = block.sort_values('mean', ascending=False).reset_index(drop=True)

    n_seeds = int(df[(df.system == system) & (df.metric == metric)
                     & (df.config_name == config)].groupby('method')['seed']
                     .nunique().max() or 0)

    fig_h = max(3.0, 0.36 * len(block) + 1.4)
    fig, ax = plt.subplots(figsize=(11.5, fig_h))

    means = block['mean'].values
    cis   = block['ci'].values
    means_pos = means[means > 0]
    if not means_pos.size:
        plt.close(fig); return

    # Asymmetric error bars, clipped so lower bound stays positive
    lower_floor = max(np.nanmin(means_pos) * 0.2, 1e-12)
    lo  = np.clip(means - cis, lower_floor, None)
    hi  = means + cis
    err_low  = means - lo
    err_high = hi - means

    colors = [_FAMILY_COLOR.get(fam, _FAMILY_COLOR['Other']) for fam in block['family']]
    y_pos  = np.arange(len(block))

    # Light row stripes for readability
    for i in range(len(block)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color='#f7f7f7', zorder=0)

    # Dot + whisker per method
    ax.errorbar(means, y_pos, xerr=[err_low, err_high], fmt='none',
                ecolor='#444444', elinewidth=1.0, capsize=3, zorder=2)
    ax.scatter(means, y_pos, c=colors, s=80, edgecolor='black', linewidth=0.7,
               zorder=3)

    # Compact annotation on the far right (column-aligned)
    x_anno = hi.max() * 1.6
    for i, (m, ci, ps, params) in enumerate(zip(
            means, cis, block['param_str'], block['params'])):
        if not np.isfinite(m):
            continue
        prec = max(0, -int(np.floor(np.log10(max(m, 1e-12)))) + 2)
        prec = min(prec, 5)
        ann_main = f"{m:.{prec}f} ± {ci:.{prec}f}"
        ann_meta = f"{ps}, {params}p" if (ps and params) else (ps or '')
        ax.text(x_anno, i, ann_main, va='center', ha='left',
                fontsize=8.5, family='monospace', color='black')
        if ann_meta:
            ax.text(x_anno * 4.5, i, ann_meta, va='center', ha='left',
                    fontsize=7.5, family='monospace', color='#555555')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(block['method'].values, fontsize=9)
    ax.invert_yaxis()                    # best at top
    ax.set_xscale('log')

    # X-limits: leave room for the annotation column
    ax.set_xlim(left=lower_floor, right=x_anno * 30)
    ax.set_xlabel(f"mean {metric}  (lower is better, log scale)")
    ax.set_title(f"{system.capitalize()} | {_short_config(config, system)} | "
                 f"{metric}  (mean ± 95% CI, n={n_seeds} seeds; "
                 "best methods at top)",
                 fontsize=11)
    ax.grid(axis='x', which='both', alpha=0.25, zorder=1)
    ax.set_axisbelow(True)

    # Legend: one entry per family present, ordered by family hierarchy
    seen_fams = []
    for fam in block['family']:
        if fam not in seen_fams:
            seen_fams.append(fam)
    seen_fams.sort(key=lambda f: _FAMILY_ORDER.get(f, 99))
    handles = [plt.Line2D([0], [0], marker='o', linestyle='',
                          markerfacecolor=_FAMILY_COLOR.get(f, _FAMILY_COLOR['Other']),
                          markeredgecolor='black', markersize=8)
               for f in seen_fams]
    ax.legend(handles, seen_fams, loc='lower right', fontsize=8,
              framealpha=0.92, ncol=1)

    plt.tight_layout()
    fp = out_dir / f"bars_{system}_{_short_config(config, system)}_{metric}.png"
    plt.savefig(fp, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"[fig] wrote {fp.name}")


def _plot_pareto(agg: pd.DataFrame, system: str, metric: str, out_dir: Path):
    """Pareto plot: x = #parameters (log), y = mean metric (log).

    One scatter point per (method, config) pair, colored by family. Error
    bars on y-axis show 95% CI (clipped to stay positive on log scale).
    Pareto frontier (best error per param budget) is highlighted.
    Each point labeled with its method's distinguishing parameter
    (``N=k`` for clustered methods, ``deg=k`` for global EDMD).
    """
    block = agg[(agg.system == system) & (agg.metric == metric)].copy()
    if block.empty:
        return
    d = _SYSTEM_D.get(system, 2)
    block['family']    = block['method'].apply(_method_family)
    block['params']    = block['method'].apply(lambda m: _method_params(m, d)[0])
    block['param_str'] = block['method'].apply(lambda m: _method_params(m, d)[1])
    block = block.dropna(subset=['params'])
    block = block[np.isfinite(block['mean']) & (block['mean'] > 0)]
    if block.empty:
        return

    fig, ax = plt.subplots(figsize=(10.0, 6.5))

    # Lower-bound floor for log-scale CI clipping
    y_floor = max(block['mean'].min() * 0.05, 1e-12)

    for fam in sorted(block['family'].unique(),
                      key=lambda f: _FAMILY_ORDER.get(f, 99)):
        sub = block[block.family == fam]
        color = _FAMILY_COLOR.get(fam, _FAMILY_COLOR['Other'])
        means = sub['mean'].values
        cis   = sub['ci'].values
        lo    = np.clip(means - cis, y_floor, None)
        err_low  = means - lo
        err_high = cis
        ax.errorbar(sub['params'].values, means,
                    yerr=[err_low, err_high],
                    fmt='o', color=color, markersize=8,
                    markeredgecolor='black', markeredgewidth=0.6,
                    elinewidth=0.8, capsize=3, zorder=3, alpha=0.9,
                    label=fam)

    # Per-point text label with the parameter (N or deg)
    label_offset_y = 1.07          # multiplicative on log axis
    for _, row in block.iterrows():
        ps = row['param_str']
        if not ps:
            continue
        # Compact label: "N=8" or "deg=2"
        compact = (ps.split(',')[0]).replace(' ', '')
        ax.annotate(compact, xy=(row['params'], row['mean']),
                    xytext=(row['params'], row['mean'] * label_offset_y),
                    textcoords='data', ha='center', va='bottom',
                    fontsize=7.5, color='#333333',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white',
                              ec='none', alpha=0.6))

    # Pareto frontier
    pts = block[['params', 'mean', 'method']].sort_values('params').values
    pareto = []
    best_y = float('inf')
    for x, y, m in pts:
        if y < best_y:
            pareto.append((x, y, m))
            best_y = y
    if pareto:
        px, py, _ = zip(*pareto)
        ax.plot(px, py, '--', color='black', alpha=0.55, linewidth=1.4,
                zorder=2, label='Pareto frontier')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('# parameters (log)')
    ax.set_ylabel(f'mean {metric} (log)')
    ax.grid(which='both', alpha=0.3, zorder=1)
    ax.set_axisbelow(True)

    n_configs = block['config_name'].nunique()
    ax.set_title(f"{system.capitalize()} | {metric} | parameters vs error  "
                 f"(across {n_configs} config(s); mean ± 95% CI; "
                 "lower-left is better)",
                 fontsize=11)
    ax.legend(fontsize=8, loc='best', framealpha=0.92)

    plt.tight_layout()
    fp = out_dir / f"pareto_{system}_{metric}.png"
    plt.savefig(fp, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"[fig] wrote {fp.name}")


def _plot_cross_config_sensitivity(agg: pd.DataFrame, system: str,
                                   metric: str, out_dir: Path):
    """Small-multiples: one panel per family, x = config, y = mean ± CI."""
    block = agg[(agg.system == system) & (agg.metric == metric)].copy()
    if block.empty:
        return
    block['family'] = block['method'].apply(_method_family)
    families = [f for f in
                sorted(block['family'].unique(), key=lambda f: _FAMILY_ORDER.get(f, 99))
                if (block.family == f).sum() > 0]
    if not families:
        return

    configs = sorted(block['config_name'].unique())
    if len(configs) < 2:
        return

    n_panels = len(families)
    n_cols = min(3, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.0 * n_rows),
                             sharex=True, squeeze=False)

    for idx, fam in enumerate(families):
        ax = axes[idx // n_cols][idx % n_cols]
        fam_block = block[block.family == fam]
        methods = sorted(fam_block['method'].unique())
        cmap = plt.get_cmap('tab20')
        for j, method in enumerate(methods):
            m_block = fam_block[fam_block.method == method].set_index('config_name')
            means = [m_block.loc[c, 'mean'] if c in m_block.index else np.nan
                     for c in configs]
            cis   = [m_block.loc[c, 'ci']   if c in m_block.index else 0.0
                     for c in configs]
            ax.errorbar(range(len(configs)), means, yerr=cis,
                        fmt='o-', color=cmap(j % 20), markersize=4,
                        elinewidth=0.7, capsize=2, label=method, alpha=0.9)
        ax.set_yscale('log')
        ax.grid(alpha=0.3, which='both')
        ax.set_title(fam, fontsize=10, color=_FAMILY_COLOR.get(fam, 'black'))
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([_short_config(c, system) for c in configs],
                           rotation=45, ha='right', fontsize=7)
        if len(methods) <= 6:
            ax.legend(fontsize=6, loc='best', framealpha=0.9)
        else:
            ax.legend(fontsize=5, loc='upper left', bbox_to_anchor=(1.0, 1.0),
                      framealpha=0.9, ncol=1)

    # Hide unused panels
    for idx in range(n_panels, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle(f"{system.capitalize()} | {metric} across {len(configs)} configs"
                 "  (mean ± 95% CI; one panel per method family)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    fp = out_dir / f"sensitivity_{system}_{metric}.png"
    plt.savefig(fp, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"[fig] wrote {fp.name}")


def aggregate_figures(agg: pd.DataFrame, df: pd.DataFrame, metrics, out_dir: Path):
    for system in sorted(agg['system'].unique()):
        configs     = sorted(agg[agg.system == system]['config_name'].unique())
        sys_metrics = _system_metrics(agg, system, metrics)
        for metric in sys_metrics:
            for config in configs:
                _plot_method_bars(agg, df, system, metric, config, out_dir)
            _plot_pareto(agg, system, metric, out_dir)
            if len(configs) >= 2:
                _plot_cross_config_sensitivity(agg, system, metric, out_dir)


# -- Main ---------------------------------------------------------------------

print("=" * 72)
print(f"  Multi-config analysis")
print(f"  data dir:  {DATA_DIR}")
print(f"  out  dir:  {OUT_DIR}")
print(f"  systems:   {args.systems}")
print(f"  metrics:   {args.metrics_table}")
print("=" * 72)

if args.from_csv:
    print(f"[corpus] reading long-form CSV: {args.from_csv}")
    df = pd.read_csv(args.from_csv)
    df = df[df['system'].isin(set(args.systems))]
else:
    df = load_corpus(DATA_DIR, set(args.systems))
if df.empty:
    print("\nNo per-config rows in corpus. "
          "Run `./run_all.sh` (or one of the per-system scripts) first.")
    raise SystemExit(0)

# Master CSV
csv_path = OUT_DIR / "all_results.csv"
df.to_csv(csv_path, index=False)
print(f"\n[csv] wrote {csv_path}  ({len(df):,} rows)")

# Per-(system, config, method, metric) summary
agg = summarize(df)
agg_csv = OUT_DIR / "summary.csv"
agg.to_csv(agg_csv, index=False)
print(f"[csv] wrote {agg_csv}  ({len(agg):,} rows)")

# Per-system metrics: each system shows whatever it has (the user's
# --metrics-table is a *preferred order*, not a corpus-wide filter).
metrics_per_system = {sys: sorted(df[df.system == sys]['metric'].unique())
                      for sys in sorted(df['system'].unique())}
metrics_to_use = list(args.metrics_table) if args.metrics_table else \
                 ['one_step'] + sorted({m for s in metrics_per_system.values()
                                        for m in s if m.startswith('r')})
print(f"[metrics] per-system available:")
for sys, ms in metrics_per_system.items():
    print(f"  {sys}: {ms}")
print(f"[metrics] preferred order: {metrics_to_use}")

# Tables, paired-tests, figures
summary_tables(agg, metrics_to_use, OUT_DIR)
cross_config_paired_tests(df, metrics_to_use, OUT_DIR)
aggregate_figures(agg, df, metrics_to_use, OUT_DIR)

print("\n" + "=" * 72)
print(f"  Done. Outputs in {OUT_DIR}")
print("=" * 72)
