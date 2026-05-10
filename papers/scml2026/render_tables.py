"""Regenerate paper tables directly from the analysis CSV.

Writes LaTeX tabular row content for:
  * the §2 headline summary (``tab_headline.tex``);
  * the per-system §D detail tables (``tab_pendulum.tex``, ``tab_duffing.tex``,
    ``tab_lorenz.tex``).

The paper imports each via ``\\input{tab_<name>.tex}`` inside the surrounding
``tabular``; ``\\midrule`` block separators and the closing ``\\bottomrule``
are baked into the generated files (so the surrounding ``tabular`` only owns
``\\toprule`` + the column header).

Values are the median across the 12 configurations of the per-configuration
mean across 10 seeds -- the same aggregation used by ``render_paretos.py``.
Headline ``wins`` columns are paired Wilcoxon signed-rank tests at p<0.05
with lower CW-EDMD mean. Bolding marks the column-wise minimum within each
detail table.

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
                'analysis_2026-05-09_05-47-40_train_cap_4k_rollout_steps_cap_200/'
                'all_results.csv')
OUT_DIR = Path(__file__).parent

D_BY_SYS = {'lorenz': 3, 'pendulum': 2, 'duffing': 2}

# Each entry: (csv_method_name, optional_label_suffix_in_parens)
PENDULUM_ROWS = [
    ('EDMD q=2',            None),
    ('EDMD q=4',            None),
    ('EDMD q=8',            None),
    ('CW-EDMD q=4, K=4',    None),
    ('CW-EDMD q=4, K=8',    None),
    ('CW-EDMD q=4, K=16',   None),
]
DUFFING_ROWS = [
    ('EDMD q=2',            None),
    ('EDMD q=3',            'matches RHS deg'),
    ('EDMD q=4',            'higher lift'),
    ('EDMD q=5',            'higher lift'),
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

# Headline rows: (display_name, system_csv_key, q, K)
HEADLINE_ROWS = [
    ('Pendulum (4, 8)',  'pendulum', 4, 8),
    ('Duffing  (3, 8)',  'duffing',  3, 8),
    ('Lorenz   (3, 12)', 'lorenz',   3, 12),
]


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
    s = re.sub(r'K=(\d+)', r'$K{=}\1$', s)
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
    out_lines.append('\\bottomrule')
    return out_lines


def _wilcoxon_wins(rows, system, q, K, metric):
    """Per-config paired Wilcoxon CW-EDMD vs EDMD. Returns wins / n."""
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
    wins = n = 0
    for cfg in configs:
        seeds = sorted(set(by_cfg_seed_cw[cfg]) & set(by_cfg_seed_ed[cfg]))
        if len(seeds) < 2: continue
        n += 1
        cw_v = np.array([by_cfg_seed_cw[cfg][s] for s in seeds])
        ed_v = np.array([by_cfg_seed_ed[cfg][s] for s in seeds])
        try: _, p = stats.wilcoxon(cw_v, ed_v)
        except ValueError: continue
        if not np.isfinite(p): continue
        if p < 0.05 and cw_v.mean() < ed_v.mean(): wins += 1
    return wins, n


def _ratio_median(rows, system, q, K, metric):
    """Ratio of per-method medians: median(EDMD per-cfg-mean) / median(CW per-cfg-mean).

    We use ratio-of-medians (not median-of-ratios) so the headline value matches
    what a reader gets back-computing from the per-method medians displayed in
    the §D detail tables. The two statistics can differ when the per-config
    ratio distribution is skewed (e.g. Duffing one-step: 2.67 vs 3.04)."""
    cw = f'CW-EDMD q={q}, K={K}'
    ed = f'EDMD q={q}'
    by_cfg_cw, by_cfg_ed = defaultdict(list), defaultdict(list)
    for r in rows:
        if r['system'] != system or r['metric'] != metric: continue
        if r['method'] == cw: by_cfg_cw[r['config_name']].append(float(r['value']))
        elif r['method'] == ed: by_cfg_ed[r['config_name']].append(float(r['value']))
    common = set(by_cfg_cw) & set(by_cfg_ed)
    if not common: return None
    med_cw = statistics.median(sum(by_cfg_cw[c]) / len(by_cfg_cw[c]) for c in common)
    med_ed = statistics.median(sum(by_cfg_ed[c]) / len(by_cfg_ed[c]) for c in common)
    return (med_ed / med_cw) if med_cw > 0 else None


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
    for label, system, q, K in HEADLINE_ROWS:
        w_os, n_os = _wilcoxon_wins(rows, system, q, K, 'one_step')
        w_r5, n_r5 = _wilcoxon_wins(rows, system, q, K, 'r5s')
        rat_os = _ratio_median(rows, system, q, K, 'one_step')
        rat_r5 = _ratio_median(rows, system, q, K, 'r5s')
        out.append(f"{label} & {w_os}/{n_os} & {w_r5}/{n_r5} & "
                   f"{_fmt_ratio(rat_os)} & {_fmt_ratio(rat_r5)} \\\\")
    out.append('\\bottomrule')
    return out


def main(argv):
    rows = list(csv.DictReader(open(CSV_PATH)))
    requested = argv[1:] or ['headline'] + list(TABLES.keys())
    for name in requested:
        if name == 'headline':
            body_lines = build_headline_rows(rows)
        elif name in TABLES:
            body_lines = build_rows(rows, name, TABLES[name])
        else:
            print(f'% unknown table: {name}', file=sys.stderr); continue
        out_path = OUT_DIR / f'tab_{name}.tex'
        header = (f'% Auto-generated by render_tables.py from {CSV_PATH.name}.\n'
                  f'% Do not edit by hand -- rerun the script if results change.\n')
        out_path.write_text(header + '\n'.join(body_lines) + '\n')
        print(f'wrote {out_path}')


if __name__ == '__main__':
    main(sys.argv)
