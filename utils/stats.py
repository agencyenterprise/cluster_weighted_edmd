"""
Statistical utilities for experiment reproducibility.

Provides:
  - multi_seed_run: run an experiment across multiple seeds, collect metrics
  - confidence_interval: mean ± 95% CI from samples
  - paired_test: paired t-test or Wilcoxon signed-rank with p-value
  - results_table: format multi-seed results as a printable table
"""

import numpy as np
from scipy import stats as sp_stats


def confidence_interval(samples, confidence=0.95):
    """
    Compute mean and symmetric CI from an array of samples.

    Uses t-distribution (appropriate for small n).
    Returns (mean, ci_half_width, lo, hi).
    """
    samples = np.asarray(samples)
    n = len(samples)
    if n < 2:
        m = samples[0] if n == 1 else float('nan')
        return m, 0.0, m, m
    m = np.mean(samples)
    se = sp_stats.sem(samples)
    t_crit = sp_stats.t.ppf((1 + confidence) / 2, df=n - 1)
    hw = t_crit * se
    return m, hw, m - hw, m + hw


def paired_test(a, b, method='t-test', alternative='two-sided'):
    """
    Test whether method A differs from method B (paired samples).

    a, b: arrays of same length (one value per seed).
    method: 't-test' (parametric) or 'wilcoxon' (non-parametric).
    alternative: 'two-sided', 'less' (a < b), 'greater' (a > b).

    Returns dict with 'statistic', 'p_value', 'method', 'mean_diff', 'ci'.
    """
    a, b = np.asarray(a), np.asarray(b)
    diff = a - b

    if method == 't-test':
        stat, p = sp_stats.ttest_rel(a, b, alternative=alternative)
    elif method == 'wilcoxon':
        if np.all(diff == 0):
            stat, p = 0.0, 1.0
        else:
            stat, p = sp_stats.wilcoxon(diff, alternative=alternative)
    else:
        raise ValueError(f"Unknown method: {method}")

    m, hw, lo, hi = confidence_interval(diff)
    return {
        'statistic': stat,
        'p_value': p,
        'method': method,
        'mean_diff': m,
        'ci': (lo, hi),
        'n': len(a),
    }


def multi_seed_run(run_fn, seeds, **kwargs):
    """
    Run an experiment function across multiple seeds.

    run_fn(seed, **kwargs) -> dict of scalar metrics
    seeds: list of int seeds

    Returns dict of {metric_name: [values_per_seed]}.
    """
    all_results = []
    for seed in seeds:
        result = run_fn(seed=seed, **kwargs)
        all_results.append(result)

    # Transpose: list of dicts -> dict of lists
    keys = all_results[0].keys()
    collected = {k: [r[k] for r in all_results] for k in keys}
    return collected


def summarize(collected, label=""):
    """
    Print summary statistics for multi-seed results.

    collected: dict of {metric_name: [values_per_seed]}
    """
    if label:
        print(f"\n  {label}")
    for key, values in collected.items():
        m, hw, lo, hi = confidence_interval(values)
        print(f"    {key:<30s} {m:>10.5f} +/- {hw:.5f}  "
              f"[{lo:.5f}, {hi:.5f}]  (n={len(values)})")


def compare(name_a, results_a, name_b, results_b, metrics=None, method='t-test'):
    """
    Compare two methods across seeds with paired tests.

    results_a, results_b: dicts from multi_seed_run (same seeds).
    metrics: list of metric names to compare (default: all shared keys).
    """
    if metrics is None:
        metrics = [k for k in results_a if k in results_b]

    print(f"\n  {name_a} vs {name_b} (paired {method}, n={len(results_a[metrics[0]])})")
    print(f"  {'metric':<30s} {'mean_A':>10s} {'mean_B':>10s} {'diff':>10s} {'p-value':>10s} {'sig':>5s}")
    print("  " + "-" * 78)

    for metric in metrics:
        a = np.array(results_a[metric])
        b = np.array(results_b[metric])
        t = paired_test(a, b, method=method, alternative='two-sided')
        ma, mb = np.mean(a), np.mean(b)
        sig = "***" if t['p_value'] < 0.001 else "**" if t['p_value'] < 0.01 else "*" if t['p_value'] < 0.05 else "ns"
        print(f"  {metric:<30s} {ma:>10.5f} {mb:>10.5f} {t['mean_diff']:>+10.5f} {t['p_value']:>10.4f} {sig:>5s}")
