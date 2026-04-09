"""
Statistical utilities for multi-seed experiment reproducibility.

Provides helpers to run experiments across multiple random seeds,
compute confidence intervals, perform paired significance tests, and
print formatted comparison tables.  Used by the validation scripts
to report mean +/- CI results and to test whether the residual-aware
method differs significantly from baselines.

Functions
---------
- ``confidence_interval(samples)`` -- mean and 95 % CI (t-distribution).
- ``paired_test(a, b)`` -- paired t-test or Wilcoxon signed-rank test.
- ``multi_seed_run(run_fn, seeds)`` -- call ``run_fn(seed=s)`` for each
  seed and collect per-metric arrays.
- ``summarize(collected)`` -- pretty-print mean +/- CI for every metric.
- ``compare(name_a, results_a, name_b, results_b)`` -- side-by-side
  paired comparison with p-values and significance stars.

Usage
-----
Run an experiment over five seeds and summarize::

    from residual_aware_clustering.utils.stats import (
        multi_seed_run, summarize, compare,
    )

    def my_experiment(seed, **kw):
        # ... fit model, evaluate ...
        return {'rmse': 0.12, 'nll': -3.4}   # scalar metrics

    results = multi_seed_run(my_experiment, seeds=[0, 1, 2, 3, 4])
    summarize(results, label="My method")

    # Compare two methods (same seeds) with paired t-test
    results_baseline = multi_seed_run(baseline_experiment, seeds=[0, 1, 2, 3, 4])
    compare("Ours", results, "Baseline", results_baseline)

Key concepts
------------
- **Small-sample CI**: ``confidence_interval`` uses the t-distribution,
  which is appropriate when the number of seeds is small (typically 5-10).
- **Paired testing**: ``paired_test`` accepts ``'t-test'`` (parametric)
  or ``'wilcoxon'`` (non-parametric) and supports one-sided alternatives.
- **Transpose trick**: ``multi_seed_run`` collects a list of dicts and
  transposes it into a dict of lists for easy per-metric analysis.
"""

import numpy as np
from scipy import stats as sp_stats


def confidence_interval(samples, confidence=0.95):
    """Compute mean and symmetric confidence interval from an array of samples.

    Uses the t-distribution, which is appropriate for small sample sizes.

    Parameters
    ----------
    samples : array_like
        1-D array of scalar observations.
    confidence : float
        Confidence level (default 0.95 for a 95% CI).

    Returns
    -------
    mean : float
        Sample mean.
    ci_half_width : float
        Half-width of the confidence interval.
    lo : float
        Lower bound of the CI.
    hi : float
        Upper bound of the CI.
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
    """Paired significance test between two matched sample arrays.

    Parameters
    ----------
    a : array_like
        Metric values for method A, one per seed.
    b : array_like
        Metric values for method B, same length as *a*.
    method : {'t-test', 'wilcoxon'}
        Parametric paired t-test or non-parametric Wilcoxon signed-rank.
    alternative : {'two-sided', 'less', 'greater'}
        Direction of the alternative hypothesis.

    Returns
    -------
    dict
        Keys: 'statistic', 'p_value', 'method', 'mean_diff', 'ci', 'n'.
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
    """Run an experiment function across multiple random seeds.

    Parameters
    ----------
    run_fn : callable
        Function with signature ``run_fn(seed=int, **kwargs) -> dict``,
        returning a dictionary of scalar metrics.
    seeds : list[int]
        Random seeds to iterate over.
    **kwargs
        Extra keyword arguments forwarded to *run_fn*.

    Returns
    -------
    dict[str, list]
        Mapping from metric name to list of per-seed values.
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
    """Print mean +/- CI for each metric in multi-seed results.

    Parameters
    ----------
    collected : dict[str, list]
        Mapping from metric name to list of per-seed values
        (as returned by ``multi_seed_run``).
    label : str
        Optional header label printed before the table.
    """
    if label:
        print(f"\n  {label}")
    for key, values in collected.items():
        m, hw, lo, hi = confidence_interval(values)
        print(f"    {key:<30s} {m:>10.5f} +/- {hw:.5f}  "
              f"[{lo:.5f}, {hi:.5f}]  (n={len(values)})")


def compare(name_a, results_a, name_b, results_b, metrics=None, method='t-test'):
    """Print a side-by-side paired comparison of two methods with p-values.

    Parameters
    ----------
    name_a : str
        Display name for method A.
    results_a : dict[str, list]
        Per-seed results for method A (from ``multi_seed_run``).
    name_b : str
        Display name for method B.
    results_b : dict[str, list]
        Per-seed results for method B (same seeds as *results_a*).
    metrics : list[str] or None
        Metric names to compare. Defaults to all shared keys.
    method : {'t-test', 'wilcoxon'}
        Statistical test to use.
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
