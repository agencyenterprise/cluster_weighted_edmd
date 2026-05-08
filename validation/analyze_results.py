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


def summary_tables(agg: pd.DataFrame, metrics, out_dir: Path):
    """Write per-system markdown + latex tables (one per metric, columns=configs)."""
    written = []
    for system in sorted(agg['system'].unique()):
        sys_md  = [f"# {system.capitalize()} -- multi-config summary",
                   "",
                   f"Mean +/- 95% CI across seeds for each (method, config, metric).",
                   ""]
        sys_tex = []

        for metric in metrics:
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
    """Heuristic pairs of (residual-aware, baseline) to compare."""
    pairs = []

    def _find(pattern):
        rx = re.compile(pattern)
        return [m for m in methods if rx.search(m)]

    # Taylor-analytic vs GMM-baseline at matched N
    for N in (2, 4, 8, 16, 5, 12, 20, 50):
        a = [m for m in _find(rf"Taylor[ -].*N=\s*{N}\b") if 'Taylor' in m]
        b = [m for m in _find(rf"GMM.*N=\s*{N}\b") if 'GMM' in m]
        if a and b:
            pairs.append((a[0], b[0], "residual-aware Taylor vs GMM-baseline"))

    # Taylor-analytic vs Global EDMD
    taylors = _find(r"Taylor.*N=\s*\d+")
    globals_ = _find(r"Global EDMD deg=\s*\d+|EDMD-disc deg")
    if taylors and globals_:
        pairs.append((taylors[-1], globals_[0],
                      "Taylor-analytic (largest N) vs Global EDMD (lowest deg)"))

    # Local EDMD vs Global EDMD
    locals_  = _find(r"local-EDMD.*N=\s*\d+|Local-EDMD.*N=\s*\d+")
    if locals_ and globals_:
        pairs.append((locals_[-1], globals_[0],
                      "local-EDMD (largest N) vs Global EDMD (lowest deg)"))

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

def _plot_ablation_grid(agg: pd.DataFrame, system: str, metric: str, out_dir: Path):
    """Heatmap of mean metric: rows=method, cols=config_name."""
    block = agg[(agg.system == system) & (agg.metric == metric)]
    if block.empty:
        return
    pivot = block.pivot_table(index='method', columns='config_name', values='mean')
    if pivot.empty:
        return
    methods = pivot.index.tolist()
    configs = pivot.columns.tolist()
    M = pivot.values

    fig_h = max(4.5, 0.35 * len(methods) + 1.5)
    fig_w = max(7.0, 0.6  * len(configs) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Use log color norm if values span several decades and are positive
    finite = M[np.isfinite(M) & (M > 0)]
    if finite.size and finite.max() / max(finite.min(), 1e-12) > 100:
        norm = LogNorm(vmin=max(finite.min(), 1e-12), vmax=finite.max())
    else:
        norm = None

    im = ax.imshow(M, aspect='auto', cmap='viridis', norm=norm)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([c.replace(f"{system}_", "") for c in configs],
                       rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=8)

    for i, m in enumerate(methods):
        for j, c in enumerate(configs):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3g}", ha='center', va='center',
                        fontsize=6, color='white' if (v > np.nanmedian(M)) else 'black')

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(f"mean {metric}")
    ax.set_title(f"{system.capitalize()} - {metric} across configurations "
                 f"(mean across seeds)")
    plt.tight_layout()
    fp = out_dir / f"ablation_{system}_{metric}.png"
    plt.savefig(fp, dpi=130)
    plt.close(fig)
    print(f"[fig] wrote {fp.name}")


def _plot_robustness(agg: pd.DataFrame, system: str, metric: str, out_dir: Path):
    """Per-method line across configs (x = config, y = mean ± CI)."""
    block = agg[(agg.system == system) & (agg.metric == metric)]
    if block.empty:
        return
    methods = sorted(block['method'].unique())
    configs = sorted(block['config_name'].unique())

    fig, ax = plt.subplots(figsize=(max(8.0, 0.5 * len(configs) + 3.0), 5.5))

    cmap = plt.get_cmap('tab20')
    for i, method in enumerate(methods):
        m_block = block[block.method == method].set_index('config_name')
        means = [m_block.loc[c, 'mean'] if c in m_block.index else np.nan
                 for c in configs]
        cis   = [m_block.loc[c, 'ci']   if c in m_block.index else 0.0
                 for c in configs]
        ax.errorbar(range(len(configs)), means, yerr=cis,
                    fmt='o-', color=cmap(i % 20),
                    label=method, markersize=4, alpha=0.85, linewidth=1.0)

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([c.replace(f"{system}_", "") for c in configs],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(f"mean {metric}")
    ax.set_yscale('log')
    ax.grid(alpha=0.3, which='both')
    ax.set_title(f"{system.capitalize()} - {metric} robustness across configs "
                 "(mean +/- 95% CI per method)")
    ax.legend(fontsize=7, loc='center left', bbox_to_anchor=(1.0, 0.5),
              frameon=False, ncol=1)
    plt.tight_layout()
    fp = out_dir / f"robustness_{system}_{metric}.png"
    plt.savefig(fp, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"[fig] wrote {fp.name}")


def aggregate_figures(agg: pd.DataFrame, metrics, out_dir: Path):
    for system in sorted(agg['system'].unique()):
        for metric in metrics:
            _plot_ablation_grid(agg, system, metric, out_dir)
            _plot_robustness(agg, system, metric, out_dir)


# -- Main ---------------------------------------------------------------------

print("=" * 72)
print(f"  Multi-config analysis")
print(f"  data dir:  {DATA_DIR}")
print(f"  out  dir:  {OUT_DIR}")
print(f"  systems:   {args.systems}")
print(f"  metrics:   {args.metrics_table}")
print("=" * 72)

df = load_corpus(DATA_DIR, set(args.systems))
if df.empty:
    print("\nNo per-config JSONs found (empty corpus). "
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

# Available metrics in the corpus (intersection with requested)
available_metrics = sorted(df['metric'].unique())
metrics_to_use = [m for m in args.metrics_table if m in available_metrics]
if not metrics_to_use:
    metrics_to_use = available_metrics
print(f"[metrics] available={available_metrics}; using={metrics_to_use}")

# Tables, paired-tests, figures
summary_tables(agg, metrics_to_use, OUT_DIR)
cross_config_paired_tests(df, metrics_to_use, OUT_DIR)
aggregate_figures(agg, metrics_to_use, OUT_DIR)

print("\n" + "=" * 72)
print(f"  Done. Outputs in {OUT_DIR}")
print("=" * 72)
