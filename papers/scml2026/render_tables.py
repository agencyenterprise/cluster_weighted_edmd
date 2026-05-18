"""Regenerate paper tables directly from the analysis CSV.

Writes complete LaTeX ``tabular`` blocks for:
  * the §2 headline summary (``tab_headline.tex``);
  * the per-system §D detail tables (``tab_pendulum.tex``, ``tab_duffing.tex``,
    ``tab_lorenz.tex``).

Each generated file is a self-contained ``\\begin{tabular}{...}...\\end{tabular}``
block: column spec, ``\\toprule``, multi-column header bands (where used),
``\\cmidrule``s, sub-header, ``\\midrule``s between blocks, data rows, and
``\\bottomrule`` are all baked in. The surrounding paper.tex only owns the
``\\begin{table}[H]\\caption{...}\\label{...}...\\end{table}`` wrapper plus
font/spacing tweaks; ``\\input{tab_<name>.tex}`` inserts the tabular itself.

Values are the median across the 12 configurations of the per-configuration
mean across 10 seeds -- the same aggregation used by ``render_paretos.py``.
Headline ``wins`` columns are paired Wilcoxon signed-rank tests at p<0.05
with lower CW-EDMD mean. Bolding marks the column-wise minimum within each
§D detail table.

Usage::

    python render_tables.py             # writes all four tables
    python render_tables.py pendulum    # only one
"""
import csv, re, statistics, sys
from collections import defaultdict
from math import comb, log10, floor
from pathlib import Path

CSV_PATH = Path('/Users/lorenzoalencartomaz/projects/client/fractal/'
                'bayesian_cluster_linear_approximation/residual_aware_clustering/'
                'analysis_2026-05-11_16-18-16_train_cap_4k_rollout_steps_cap_200/'
                'all_results.csv')
OUT_DIR = Path(__file__).parent

D_BY_SYS = {'lorenz': 3, 'pendulum': 2, 'duffing': 2}

# Each entry: (csv_method_name, optional_label_suffix_in_parens)
PENDULUM_ROWS = [
    ('EDMD q=2',            'low lift'),
    ('EDMD q=4',            'matched'),
    ('CW-EDMD q=4, K=4',    None),
    ('CW-EDMD q=4, K=8',    None),
    ('CW-EDMD q=4, K=16',   None),
]
DUFFING_ROWS = [
    ('EDMD q=2',            'low lift'),
    ('EDMD q=3',            'matched'),
    ('EDMD q=4',            'matched'),
    ('EDMD q=5',            'matched'),
    ('CW-EDMD q=3, K=8',    None),
    ('CW-EDMD q=4, K=4',    None),
    ('CW-EDMD q=5, K=4',    None),
]
LORENZ_ROWS = [
    ('EDMD q=2',            None),
    ('EDMD q=3',            None),
    ('CW-EDMD q=2, K=5',    None),
    ('CW-EDMD q=2, K=12',   None),
    ('CW-EDMD q=2, K=20',   None),
    ('CW-EDMD q=3, K=5',    None),
    ('CW-EDMD q=3, K=12',   None),
    ('CW-EDMD q=3, K=20',   None),
]
TABLES = {'pendulum': PENDULUM_ROWS, 'duffing': DUFFING_ROWS, 'lorenz': LORENZ_ROWS}

# One headline row per system. Each row's W/L/T is the paired-Wilcoxon tally
# across every CW-EDMD $(q, G)$ variant and every configuration in that
# system's corpus, against EDMD at the matched lift degree $q$.
HEADLINE_SYSTEMS = [
    ('Pendulum', 'pendulum', [(4, 4), (4, 8), (4, 16)]),
    ('Duffing',  'duffing',  [(3, 8), (4, 4), (5, 4)]),
    ('Lorenz',   'lorenz',   [(2, 5), (2, 12), (2, 20),
                              (3, 5), (3, 12), (3, 20)]),
]

# Per-table tabular structure. ``col_spec`` is the {...} after \begin{tabular};
# ``header_lines`` are the lines between \toprule and \midrule.
DETAIL_HEADER = [r'Method & Params & one-step & 5\,s \\']
TABLE_SPECS = {
    'headline': {
        'col_spec': 'lcccc',
        'header_lines': [
            r'& \multicolumn{2}{c}{W / L / T} & \multicolumn{2}{c}{ratio} \\',
            r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
            r'System & 1-step & 5\,s & 1-step & 5\,s \\',
        ],
    },
    'pendulum': {'col_spec': 'lrrr', 'header_lines': DETAIL_HEADER},
    'duffing':  {'col_spec': 'lrrr', 'header_lines': DETAIL_HEADER},
    'lorenz':   {'col_spec': 'lrrr', 'header_lines': DETAIL_HEADER},
    'lorenz_per_config': {
        'col_spec': 'lcc',
        'header_lines': [r'Configuration & one-step (E/CW) & 5\,s (E/CW) \\'],
    },
}


def family(name):
    n = name.lower().replace(' ', '').replace('_', '-')
    if 'cw-edmd' in n:  return 'CW-EDMD'
    if 'gmm-edmd' in n: return 'GMM-EDMD'
    if n.startswith('edmd') or 'edmd-disc' in n: return 'EDMD'
    return 'Other'


def params(name, d):
    nl = name.lower().replace(' ', '').replace('_', '-')
    fam = family(name)
    K_m = re.search(r'k=(\d+)', nl)
    q_m = re.search(r'q=(\d+)', nl)
    K = int(K_m.group(1)) if K_m else None
    q = int(q_m.group(1)) if q_m else None
    if fam in ('CW-EDMD', 'GMM-EDMD') and K is not None:
        d_eff = q or 2
        M = comb(d + d_eff, d_eff)
        return K * M * M
    if fam == 'EDMD' and q is not None:
        M = comb(d + q, q)
        return M * M
    return None


def fmt_label(method_name, suffix):
    """``EDMD q=3`` -> ``EDMD $q{=}3$``; ``CW-EDMD q=4, K=8`` -> ``CW-EDMD $q{=}4, K{=}8$``."""
    s = re.sub(r'q=(\d+)', r'$q{=}\1$', method_name)
    s = re.sub(r'K=(\d+)', r'$G{=}\1$', s)  # rename cluster count K -> G in displayed labels (CSV identifier "K=..." stays unchanged)
    s = s.replace('$, $', ', ')
    if suffix:
        s += f' ({suffix})'
    return s


def fmt_sci(x, sig=2):
    """Format ``x`` as ``M\\!\\times\\!10^{E}`` (no surrounding ``$``)."""
    if x == 0: return '0'
    exp = int(floor(log10(abs(x))))
    mant = x / 10**exp
    # Round mantissa to (sig-1) decimals
    mant_str = f'{mant:.{sig-1}f}'
    # If rounding pushed mantissa to >= 10, renormalise
    if float(mant_str) >= 10:
        exp += 1
        mant_str = f'{mant / 10:.{sig-1}f}'
    return f'{mant_str}\\!\\times\\!10^{{{exp}}}'


def aggregate(rows, system, method, metric):
    """Median across configurations of per-config mean across seeds."""
    by_cfg = defaultdict(list)
    for r in rows:
        if r['system'] == system and r['metric'] == metric and r['method'] == method:
            try: by_cfg[r['config_name']].append(float(r['value']))
            except ValueError: pass
    if not by_cfg: return None
    cfg_means = [sum(v) / len(v) for v in by_cfg.values()]
    return statistics.median(cfg_means)


def _block_key(method_name):
    """Identify a block for \\midrule grouping: (family, q-or-None)."""
    fam = family(method_name)
    q_m = re.search(r'q=(\d+)', method_name.lower())
    q = int(q_m.group(1)) if q_m else None
    if fam == 'EDMD':
        return ('EDMD', None)  # all EDMD rows in one block
    return (fam, q)            # CW/GMM blocks split by q


def build_rows(rows, system, table_rows):
    d = D_BY_SYS[system]
    data = []
    for method, suffix in table_rows:
        data.append({
            'method':   method,
            'label':    fmt_label(method, suffix),
            'params':   params(method, d),
            'one_step': aggregate(rows, system, method, 'one_step'),
            'r5s':      aggregate(rows, system, method, 'r5s'),
        })

    def col_min(col):
        vals = [r[col] for r in data if r[col] is not None]
        return min(vals) if vals else None
    min_os = col_min('one_step')
    min_r5 = col_min('r5s')

    # Block sizes for midrule decisions: cross-family midrule always; within-
    # family midrule only when both adjacent sub-blocks are multi-row, so that
    # tables like Duffing (one row per q) don't get cluttered with rules
    # between consecutive single rows.
    block_seq  = [_block_key(r['method']) for r in data]
    block_size = {b: block_seq.count(b) for b in set(block_seq)}
    out_lines = []
    prev_block = None
    for r, block in zip(data, block_seq):
        if prev_block is not None and block != prev_block:
            cross_family = (prev_block[0] != block[0])
            if cross_family or (block_size[prev_block] >= 2 and block_size[block] >= 2):
                out_lines.append('\\midrule')
        prev_block = block
        p_str = str(r['params']) if r['params'] is not None else '--'
        def cell(val, is_min):
            if val is None: return '--'
            body = fmt_sci(val)
            return f'$\\mathbf{{{body}}}$' if is_min else f'${body}$'
        os_cell = cell(r['one_step'], r['one_step'] is not None and r['one_step'] == min_os)
        r5_cell = cell(r['r5s'],      r['r5s']      is not None and r['r5s']      == min_r5)
        out_lines.append(f"{r['label']} & {p_str} & {os_cell} & {r5_cell} \\\\")
    return out_lines


def _wilcoxon_wlt(rows, system, q, K, metric):
    """Per-config paired Wilcoxon CW-EDMD vs EDMD. Returns (wins, losses, ties)."""
    from scipy import stats
    import numpy as np
    cw = f'CW-EDMD q={q}, K={K}'
    ed = f'EDMD q={q}'
    by_cfg_seed_cw, by_cfg_seed_ed = defaultdict(dict), defaultdict(dict)
    for r in rows:
        if r['system'] != system or r['metric'] != metric: continue
        seed = r.get('seed', '?')
        if r['method'] == cw: by_cfg_seed_cw[r['config_name']][seed] = float(r['value'])
        elif r['method'] == ed: by_cfg_seed_ed[r['config_name']][seed] = float(r['value'])
    configs = sorted(set(by_cfg_seed_cw) & set(by_cfg_seed_ed))
    W = L = T = 0
    for cfg in configs:
        seeds = sorted(set(by_cfg_seed_cw[cfg]) & set(by_cfg_seed_ed[cfg]))
        if len(seeds) < 2: continue
        cw_v = np.array([by_cfg_seed_cw[cfg][s] for s in seeds])
        ed_v = np.array([by_cfg_seed_ed[cfg][s] for s in seeds])
        try: _, p = stats.wilcoxon(cw_v, ed_v)
        except ValueError: p = 1.0
        if not np.isfinite(p): p = 1.0
        if p < 0.05 and cw_v.mean() < ed_v.mean(): W += 1
        elif p < 0.05 and cw_v.mean() > ed_v.mean(): L += 1
        else: T += 1
    return W, L, T


def _median_of_ratios(rows, system, qK_list, metric):
    """Aggregated median-of-ratios over the headline corpus for a system:
    for each $(q, G, \text{config})$ tuple, form the per-config seed-mean ratio
    $\\text{EDMD}/\\text{CW-EDMD}$; return the median across all such tuples."""
    ratios = []
    for q, K in qK_list:
        cw = f'CW-EDMD q={q}, K={K}'
        ed = f'EDMD q={q}'
        by_cfg_cw, by_cfg_ed = defaultdict(list), defaultdict(list)
        for r in rows:
            if r['system'] != system or r['metric'] != metric: continue
            if r['method'] == cw: by_cfg_cw[r['config_name']].append(float(r['value']))
            elif r['method'] == ed: by_cfg_ed[r['config_name']].append(float(r['value']))
        for c in set(by_cfg_cw) & set(by_cfg_ed):
            cw_mean = sum(by_cfg_cw[c]) / len(by_cfg_cw[c])
            ed_mean = sum(by_cfg_ed[c]) / len(by_cfg_ed[c])
            if cw_mean > 0:
                ratios.append(ed_mean / cw_mean)
    return statistics.median(ratios) if ratios else None


# Human-readable descriptions for each Lorenz configuration. The raw config
# names (e.g. ``lorenz_attractor_small_data``) are internal YAML identifiers
# and should not appear in paper bodies; we map them here to short prose
# descriptions of what the configuration varies relative to the baseline.
LORENZ_CONFIG_DESCRIPTION = {
    'lorenz_attractor_baseline':   'attractor sampling, baseline settings',
    'lorenz_attractor_heavy_fit':  'attractor sampling, heavy fit budget',
    'lorenz_attractor_large_data': 'attractor sampling, large training set',
    'lorenz_attractor_small_data': 'attractor sampling, small training set',
    'lorenz_dt_fast':              'short integrator step',
    'lorenz_dt_slow':              'long integrator step',
    'lorenz_gaussian':             'Gaussian sampling',
    'lorenz_gaussian_mixture':     'Gaussian-mixture sampling',
    'lorenz_periodic_noise':       'periodic-noise sampling',
    'lorenz_trajectory':           'trajectory-ensemble sampling',
    'lorenz_uniform_narrow_box':   'uniform sampling, narrow box',
    'lorenz_uniform_wide_box':     'uniform sampling, wide box',
}


def _lorenz_per_config_rows(rows):
    """Per-configuration breakdown for the Lorenz $(q{=}3, K{=}12)$ headline cell.

    For each of the 12 Lorenz configurations, computes the paired Wilcoxon
    outcome (W / T / L at p<0.05) and per-config error ratio EDMD/CW-EDMD on
    both one-step and 5\\,s rollout metrics. This is the Tier-2 per-config
    table that backs the headline ``$10/12$ on 5\\,s'' claim by making the
    two non-wins visible (and tying them to specific configurations).
    """
    from scipy import stats
    import numpy as np
    cw_method = 'CW-EDMD q=3, K=12'
    ed_method = 'EDMD q=3'
    by_cfg = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in rows:
        if r['system'] != 'lorenz': continue
        if r['method'] not in (cw_method, ed_method): continue
        if r['metric'] not in ('one_step', 'r5s'): continue
        seed = r.get('seed', '?')
        by_cfg[r['config_name']][r['metric']][r['method']][seed] = float(r['value'])

    def verdict_cell(cw_d, ed_d):
        seeds = sorted(set(cw_d) & set(ed_d))
        if len(seeds) < 2: return ('--', float('nan'))
        cw_v = np.array([cw_d[s] for s in seeds])
        ed_v = np.array([ed_d[s] for s in seeds])
        try: _, p = stats.wilcoxon(cw_v, ed_v)
        except ValueError: p = float('nan')
        if not np.isfinite(p): return ('T', float('nan'))
        if cw_v.mean() <= 0: return ('--', float('nan'))
        ratio = ed_v.mean() / cw_v.mean()
        if p < 0.05 and cw_v.mean() < ed_v.mean(): verdict = 'W'
        elif p < 0.05:                              verdict = 'L'
        else:                                       verdict = 'T'
        return (verdict, ratio)

    out_rows = []
    for cfg in sorted(by_cfg):
        label = LORENZ_CONFIG_DESCRIPTION.get(cfg, cfg)
        v1, r1 = verdict_cell(by_cfg[cfg]['one_step'][cw_method],
                              by_cfg[cfg]['one_step'][ed_method])
        v5, r5 = verdict_cell(by_cfg[cfg]['r5s'][cw_method],
                              by_cfg[cfg]['r5s'][ed_method])
        def fmt(v, r):
            if v == '--': return '--'
            if not (isinstance(r, float) and r == r and r > 0): return v
            if r >= 100: return f'{int(round(r/10)*10)}$\\times$~({v})'
            if r >= 10:  return f'{int(round(r))}$\\times$~({v})'
            return f'{r:.1f}$\\times$~({v})'
        out_rows.append(f"{label} & {fmt(v1, r1)} & {fmt(v5, r5)} \\\\")
    return out_rows


def _fmt_ratio(r):
    """2-sig-fig rounding so the headline ratios are consistent with the
    rounded values in the §D detail tables (avoids a precise-looking number
    like 2620 that conflicts with what a reader computes from the rounded
    2-sig-fig per-method values in Appendix D)."""
    if r is None: return '--'
    if r >= 1000: return f'{int(round(r/100)*100)}$\\times$'
    if r >= 100:  return f'{int(round(r/10)*10)}$\\times$'
    if r >= 10:   return f'{int(round(r))}$\\times$'
    return f'{r:.1f}$\\times$'


def build_headline_rows(rows):
    out = []
    for label, system, qK_list in HEADLINE_SYSTEMS:
        W = {'one_step': 0, 'r5s': 0}
        L = {'one_step': 0, 'r5s': 0}
        T = {'one_step': 0, 'r5s': 0}
        for q, K in qK_list:
            for metric in ('one_step', 'r5s'):
                w, l, t = _wilcoxon_wlt(rows, system, q, K, metric)
                W[metric] += w; L[metric] += l; T[metric] += t
        os_wlt = f"{W['one_step']}/{L['one_step']}/{T['one_step']}"
        r5_wlt = f"{W['r5s']}/{L['r5s']}/{T['r5s']}"
        rat_os = _median_of_ratios(rows, system, qK_list, 'one_step')
        rat_r5 = _median_of_ratios(rows, system, qK_list, 'r5s')
        out.append(f"{label} & {os_wlt} & {r5_wlt} & "
                   f"{_fmt_ratio(rat_os)} & {_fmt_ratio(rat_r5)} \\\\")
    return out


def emit_tabular(name, body_lines):
    """Wrap body data rows in a complete \\begin{tabular}{...}...\\end{tabular}
    block including column spec, top/mid/bottomrules, and the per-table header
    lines (multi-column header band for the headline table; single-row header
    for §D detail tables)."""
    spec = TABLE_SPECS[name]
    out  = [f"\\begin{{tabular}}{{{spec['col_spec']}}}", r'\toprule']
    out += spec['header_lines']
    out += [r'\midrule']
    out += body_lines
    out += [r'\bottomrule', r'\end{tabular}']
    return out


def main(argv):
    rows = list(csv.DictReader(open(CSV_PATH)))
    requested = argv[1:] or ['headline'] + list(TABLES.keys()) + ['lorenz_per_config']
    for name in requested:
        if name == 'headline':
            body_lines = build_headline_rows(rows)
        elif name == 'lorenz_per_config':
            body_lines = _lorenz_per_config_rows(rows)
        elif name in TABLES:
            body_lines = build_rows(rows, name, TABLES[name])
        else:
            print(f'% unknown table: {name}', file=sys.stderr); continue
        block = emit_tabular(name, body_lines)
        out_path = OUT_DIR / f'tab_{name}.tex'
        header = (f'% Auto-generated by render_tables.py from {CSV_PATH.name}.\n'
                  f'% Do not edit by hand -- rerun the script if results change.\n')
        out_path.write_text(header + '\n'.join(block) + '\n')
        print(f'wrote {out_path}')


if __name__ == '__main__':
    main(sys.argv)
