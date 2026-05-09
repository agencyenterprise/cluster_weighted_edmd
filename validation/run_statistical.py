"""
Complete statistical validation for Lorenz, Pendulum, and Duffing.

Runs all method families across multiple seeds, reports:
  - Mean ± 95% CI for all metrics
  - Paired t-tests between key method pairs
  - Saves full per-seed data to JSON (config-name-suffixed if --config used)
  - Generates error-bar plots

This is the single authoritative script for paper results.

Two operating modes:
  1. CLI mode (legacy): pass flags like --lorenz-N, --pendulum-N, etc.
     Outputs go to ``papers/data/statistical_<system>.json``.
  2. Config mode: ``--config config/<system>/<name>.yaml`` loads a YAML
     config that specifies system, seeds, data, fit, rollout, and method
     parameters. Outputs go to ``papers/data/<config_name>.json`` and
     ``papers/figures/<config_name>.png``, so multiple configs do NOT
     overwrite each other.

Run a single config::

    python -m validation.run_statistical --config config/duffing/uniform_baseline.yaml

Loop over all configs for a system (typical runpod batch)::

    for cfg in config/duffing/*.yaml; do
        python -m validation.run_statistical --config "$cfg"
    done
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

torch.set_default_dtype(torch.float64)
# Cap intra-op torch threads so per-seed processes don't oversubscribe the
# CPU when run in a process pool (each process otherwise tries to use all
# cores for BLAS, which thrashes when N processes also fight for cores).
torch.set_num_threads(1)
# Multiprocessing start method:
#   - Linux: default 'fork' is fast (no module re-import per worker) and fine.
#   - macOS: default 'spawn' is required (Objective-C runtime is NOT fork-safe;
#     forking + PyTorch crashes the worker process). Spawn is slower because
#     each worker re-imports this module (re-parses argparse, re-loads the
#     YAML config, re-imports torch), but the per-config wall-clock benefit
#     of N parallel seeds still dominates the startup cost.
# Leave the default in place rather than forcing 'fork' globally.
import sys as _sys
if _sys.platform.startswith('linux'):
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass

from utils.paths import fig_path, data_path
from utils.stats import confidence_interval, paired_test

from simulators.lorenz import (
    generate_data as lorenz_generate, f as lorenz_f, J as lorenz_J,
    sample_uniform           as lorenz_sample_uniform,
    sample_gaussian          as lorenz_sample_gaussian,
    sample_gaussian_mixture  as lorenz_sample_gaussian_mixture,
    sample_periodic_noise    as lorenz_sample_periodic_noise,
    sample_trajectory_ensemble as lorenz_sample_trajectory_ensemble,
)
from simulators.pendulum import (
    f as pendulum_f, J as pendulum_J,
    sample_phase_space, generate_trajectory, wrap_theta, angular_dist,
    sample_gaussian           as pendulum_sample_gaussian,
    sample_gaussian_mixture   as pendulum_sample_gaussian_mixture,
    sample_periodic_noise     as pendulum_sample_periodic_noise,
    sample_trajectory_ensemble as pendulum_sample_trajectory_ensemble,
)
from simulators.duffing import (
    f as duffing_f, J as duffing_J,
    sample_phase_space         as duffing_sample_phase_space,
    sample_trajectory_ensemble as duffing_sample_trajectory_ensemble,
    sample_gaussian            as duffing_sample_gaussian,
    sample_gaussian_mixture    as duffing_sample_gaussian_mixture,
    sample_periodic_noise      as duffing_sample_periodic_noise,
    generate_trajectory        as duffing_generate_trajectory,
    DELTA                       as DUFFING_DELTA,
)
from models.em import fit as fit_taylor
# Discrete EDMD is the only EDMD variant used here -- no derivatives, no
# Euler steps. The continuous fitters (weighted_continuous_edmd /
# fit_local_edmd_cont) are intentionally not imported.
from models.em_local_edmd_discrete import (
    fit as fit_local_edmd_disc,
    fit_global as fit_global_disc,
    predict_next_global as predict_next_disc,
    predict_next_all_clusters as predict_next_all_disc,
)
from models.distributions import mvn_logpdf_batch

import pykoopman as pk


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Complete statistical validation (Lorenz + Pendulum)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# Seeds
parser.add_argument('--seeds', type=int, nargs='+',
                    default=[1, 42, 101, 307, 1001, 7789, 13245, 11, 103, 13],
                    help="Random seeds for statistical runs")

def _json_arg(s):
    return json.loads(s) if isinstance(s, str) else s


# Lorenz data
parser.add_argument('--lorenz-n-steps', type=int, default=5000)
parser.add_argument('--lorenz-dt', type=float, default=0.01)
parser.add_argument('--lorenz-warmup', type=int, default=1000)
parser.add_argument('--lorenz-n-train', type=int, default=4000)
parser.add_argument('--lorenz-distribution', default='attractor',
                    choices=['attractor', 'uniform', 'gaussian',
                             'gaussian_mixture', 'periodic_noise', 'trajectory'])
parser.add_argument('--lorenz-distribution-params', type=_json_arg, default={})

# Pendulum data
parser.add_argument('--pendulum-n-train', type=int, default=4000)
parser.add_argument('--pendulum-n-test', type=int, default=1000)
parser.add_argument('--pendulum-dt', type=float, default=0.05)
parser.add_argument('--pendulum-rollout-steps', type=int, default=200)
parser.add_argument('--pendulum-n-trajs', type=int, default=100,
                    help="Number of trajectories for discrete EDMD pairs")
parser.add_argument('--pendulum-traj-len', type=int, default=50,
                    help="Steps per trajectory for discrete EDMD pairs")
parser.add_argument('--pendulum-distribution', default='uniform',
                    choices=['uniform', 'gaussian', 'gaussian_mixture',
                             'periodic_noise', 'trajectory'])
parser.add_argument('--pendulum-distribution-params', type=_json_arg, default={})

# Local-EDMD-disc degree sweep (per-system). Each system runs Local-EDMD-disc
# at every degree in this list, with the configured N_list.  Default keeps
# legacy degree=2 only behavior.  Setting e.g. [2, 3] on Lorenz also tests the
# polynomial-exact-degree partitioned variant, which is the apples-to-apples
# comparison vs global EDMD-disc deg=3.
parser.add_argument('--lorenz-le-degrees',   type=int, nargs='+', default=[2])
parser.add_argument('--pendulum-le-degrees', type=int, nargs='+', default=[2])

# EM fitting
parser.add_argument('--n-iter', type=int, default=100)
parser.add_argument('--n-restarts', type=int, default=2)

# Cluster counts to sweep
parser.add_argument('--lorenz-N', type=int, nargs='+', default=[5, 12, 20, 50])
parser.add_argument('--pendulum-N', type=int, nargs='+', default=[2, 4, 8, 16])

# EDMD degrees
parser.add_argument('--edmd-degrees', type=int, nargs='+', default=[2, 3])
parser.add_argument('--pendulum-edmd-degrees', type=int, nargs='+', default=[2, 4, 6, 8])

# Rollout horizons (in seconds; per-system metric is reported at each horizon)
parser.add_argument('--lorenz-rollout-steps', type=int, default=500)
parser.add_argument('--lorenz-horizons',  type=float, nargs='+',
                    default=[0.5, 1.0, 2.0, 5.0])
parser.add_argument('--pendulum-horizons', type=float, nargs='+',
                    default=[1.0, 2.5, 5.0, 10.0])

# Duffing data
parser.add_argument('--duffing-distribution', default='uniform',
                    choices=['uniform', 'gaussian', 'gaussian_mixture',
                             'periodic_noise', 'trajectory'])
parser.add_argument('--duffing-distribution-params', type=_json_arg, default={})
parser.add_argument('--duffing-n-train',    type=int, default=4000)
parser.add_argument('--duffing-n-test',     type=int, default=1000)
parser.add_argument('--duffing-test-box-x',    type=float, default=2.0,
                    help="Held-out test set is uniform on a box; this sets its half-width")
parser.add_argument('--duffing-test-box-xdot', type=float, default=2.0)
parser.add_argument('--duffing-dt',         type=float, default=0.05)
parser.add_argument('--duffing-rollout-steps', type=int, default=400)
parser.add_argument('--duffing-horizons',  type=float, nargs='+',
                    default=[1.0, 2.0, 5.0, 10.0, 20.0])
parser.add_argument('--duffing-N',          type=int, nargs='+', default=[2, 4, 8, 16])
parser.add_argument('--duffing-edmd-degrees', type=int, nargs='+', default=[2, 3, 4, 5])
parser.add_argument('--duffing-le2-N',      type=int, nargs='+', default=[2, 4, 8, 16])
parser.add_argument('--duffing-le3-N',      type=int, nargs='+', default=[2, 4, 8])
# Higher-degree Local-EDMD-disc for polynomial-exactness comparison.  Empty
# by default; set in YAML's methods.le4_N_list / le5_N_list to enable.
# At deg=5 (M=21, M^2=441) the per-cluster fit is much heavier; only set
# small N (<=4) unless you have abundant compute.
parser.add_argument('--duffing-le4-N',      type=int, nargs='+', default=[])
parser.add_argument('--duffing-le5-N',      type=int, nargs='+', default=[])

# Systems to run
parser.add_argument('--skip-lorenz',   action='store_true')
parser.add_argument('--skip-pendulum', action='store_true')
parser.add_argument('--skip-duffing',  action='store_true')

# Model saving (on by default for visualization)
parser.add_argument('--no-save-models', action='store_true',
                    help="Skip saving fitted model states (saves disk space)")

# YAML config (overrides CLI defaults; sets output filenames; restricts to one system)
parser.add_argument('--config', type=str, default=None,
                    help="Path to a YAML experiment config (see config/<system>/*.yaml). "
                         "If provided, it sets the system, seeds, data, fit, rollout, and "
                         "method parameters from the YAML; other systems are skipped; "
                         "outputs are named after the config's `name` field.")

# Speed knobs
parser.add_argument('--max-workers', type=int, default=0,
                    help="Number of seed-level parallel workers. 0 (default) "
                         "auto-picks min(os.cpu_count(), len(seeds)). 1 = sequential.")
parser.add_argument('--n-iter-cap', type=int, default=None,
                    help="If set, cap n_iter at this value (overrides YAML/CLI default).")
parser.add_argument('--n-restarts-cap', type=int, default=None,
                    help="If set, cap n_restarts at this value (overrides YAML/CLI default).")
parser.add_argument('--n-train-cap', type=int, default=None,
                    help="If set, cap every system's n_train at this value. "
                         "EM cost is ~linear in n_train; capping is the simplest "
                         "way to trade quality for speed across the corpus without "
                         "rewriting per-config YAMLs.")
parser.add_argument('--n-test-cap', type=int, default=None,
                    help="If set, cap every system's n_test at this value.")
parser.add_argument('--rollout-steps-cap', type=int, default=None,
                    help="If set, cap every system's rollout_steps at this value. "
                         "Rollout cost is linear in this; horizons beyond cap*dt "
                         "will saturate at the final reachable step.")

parser.add_argument('--run-id', type=str, default=None,
                    help="Optional run identifier (e.g. a timestamp). When set, "
                         "all per-config outputs are written under "
                         "papers/data/<run-id>/ and papers/figures/<run-id>/, "
                         "so multiple invocations can coexist without "
                         "overwriting each other. Empty (default) keeps the "
                         "legacy flat layout.")

args = parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# YAML config loader (config-driven mode)
# ─────────────────────────────────────────────────────────────────────────────

_DATA_TOP_KEYS = {'distribution', 'n_train', 'n_test'}


def _split_data_block(d):
    """Split a YAML data block into (distribution, n_train, n_test, params)."""
    distribution = d.get('distribution')
    n_train      = d.get('n_train')
    n_test       = d.get('n_test')
    params       = {k: v for k, v in d.items() if k not in _DATA_TOP_KEYS}
    return distribution, n_train, n_test, params


def _apply_lorenz_config(cfg, args):
    d = cfg.get('data', {})
    distribution, n_train, n_test, params = _split_data_block(d)
    args.lorenz_distribution        = distribution if distribution else args.lorenz_distribution
    args.lorenz_distribution_params = params
    args.lorenz_n_train             = n_train if n_train is not None else args.lorenz_n_train
    # Legacy attractor-trajectory fields (kept for backward compat with the
    # default attractor sampler; ignored when a non-attractor distribution
    # is selected)
    args.lorenz_n_steps = params.get('n_steps', args.lorenz_n_steps)
    args.lorenz_dt      = params.get('dt',      args.lorenz_dt)
    args.lorenz_warmup  = params.get('warmup',  args.lorenz_warmup)
    f_ = cfg.get('fit', {})
    args.n_iter        = f_.get('n_iter',     args.n_iter)
    args.n_restarts    = f_.get('n_restarts', args.n_restarts)
    r = cfg.get('rollout', {})
    args.lorenz_dt            = r.get('dt',            args.lorenz_dt)
    args.lorenz_rollout_steps = r.get('rollout_steps', args.lorenz_rollout_steps)
    args.lorenz_horizons      = r.get('horizons',      args.lorenz_horizons)
    m = cfg.get('methods', {})
    args.lorenz_N           = m.get('N_list',     args.lorenz_N)
    args.edmd_degrees       = m.get('edmd_degs',  args.edmd_degrees)
    args.lorenz_le_degrees  = m.get('le_degrees', args.lorenz_le_degrees)
    args.skip_pendulum = True
    args.skip_duffing  = True


def _apply_pendulum_config(cfg, args):
    d = cfg.get('data', {})
    distribution, n_train, n_test, params = _split_data_block(d)
    args.pendulum_distribution        = distribution if distribution else args.pendulum_distribution
    args.pendulum_distribution_params = params
    args.pendulum_n_train             = n_train if n_train is not None else args.pendulum_n_train
    args.pendulum_n_test              = n_test  if n_test  is not None else args.pendulum_n_test
    # discrete-EDMD trajectory pairs (independent of the train-set distribution)
    args.pendulum_n_trajs  = params.get('n_traj_pairs', args.pendulum_n_trajs)
    args.pendulum_traj_len = params.get('traj_steps_pairs', args.pendulum_traj_len)
    f_ = cfg.get('fit', {})
    args.n_iter     = f_.get('n_iter',     args.n_iter)
    args.n_restarts = f_.get('n_restarts', args.n_restarts)
    r = cfg.get('rollout', {})
    args.pendulum_dt            = r.get('dt',            args.pendulum_dt)
    args.pendulum_rollout_steps = r.get('rollout_steps', args.pendulum_rollout_steps)
    args.pendulum_horizons      = r.get('horizons',      args.pendulum_horizons)
    m = cfg.get('methods', {})
    args.pendulum_N             = m.get('N_list',     args.pendulum_N)
    args.pendulum_edmd_degrees  = m.get('edmd_degs',  args.pendulum_edmd_degrees)
    args.pendulum_le_degrees    = m.get('le_degrees', args.pendulum_le_degrees)
    args.skip_lorenz  = True
    args.skip_duffing = True


def _apply_duffing_config(cfg, args):
    d = cfg.get('data', {})
    distribution, n_train, n_test, params = _split_data_block(d)
    args.duffing_distribution        = distribution if distribution else args.duffing_distribution
    args.duffing_distribution_params = params
    args.duffing_n_train             = n_train if n_train is not None else args.duffing_n_train
    args.duffing_n_test              = n_test  if n_test  is not None else args.duffing_n_test
    args.duffing_test_box_x          = params.get('test_box_x',    args.duffing_test_box_x)
    args.duffing_test_box_xdot       = params.get('test_box_xdot', args.duffing_test_box_xdot)
    f_ = cfg.get('fit', {})
    args.n_iter     = f_.get('n_iter',     args.n_iter)
    args.n_restarts = f_.get('n_restarts', args.n_restarts)
    r = cfg.get('rollout', {})
    args.duffing_dt            = r.get('dt',            args.duffing_dt)
    args.duffing_rollout_steps = r.get('rollout_steps', args.duffing_rollout_steps)
    args.duffing_horizons      = r.get('horizons',      args.duffing_horizons)
    m = cfg.get('methods', {})
    args.duffing_N             = m.get('N_list',     args.duffing_N)
    args.duffing_edmd_degrees  = m.get('edmd_degs',  args.duffing_edmd_degrees)
    args.duffing_le2_N         = m.get('le2_N_list', args.duffing_le2_N)
    args.duffing_le3_N         = m.get('le3_N_list', args.duffing_le3_N)
    args.duffing_le4_N         = m.get('le4_N_list', args.duffing_le4_N)
    args.duffing_le5_N         = m.get('le5_N_list', args.duffing_le5_N)
    args.skip_lorenz  = True
    args.skip_pendulum = True


# -- Distribution dispatch tables ---------------------------------------------
#
# Each entry maps a distribution name to a callable
# ``(n_samples, params, seed) -> {'X', 'F', 'J_all'}`` that uses sensible
# system-specific defaults when ``params`` does not specify them.

DUFFING_TRAIN_SAMPLERS = {
    'uniform': lambda n, p, seed: duffing_sample_phase_space(
        n_samples=n,
        x_max   =p.get('box_x',    2.0),
        xdot_max=p.get('box_xdot', 2.0),
        seed=seed),
    'gaussian': lambda n, p, seed: duffing_sample_gaussian(
        n_samples=n,
        mean =p.get('mean'),
        sigma=p.get('sigma', 1.0),
        seed=seed),
    'gaussian_mixture': lambda n, p, seed: duffing_sample_gaussian_mixture(
        n_samples=n,
        centers=p.get('centers'),
        sigmas =p.get('sigmas', 0.4),
        weights=p.get('weights'),
        seed=seed),
    'periodic_noise': lambda n, p, seed: duffing_sample_periodic_noise(
        n_samples=n,
        amplitudes=p.get('amplitudes'),
        frequency =p.get('frequency'),
        center    =p.get('center'),
        noise_std =p.get('noise_std', 0.1),
        seed=seed),
    'trajectory': lambda n, p, seed: duffing_sample_trajectory_ensemble(
        n_traj      =p.get('n_traj',    200),
        n_steps     =p.get('traj_steps', 50),
        dt          =p.get('dt',        0.05),
        ic_x_max    =p.get('ic_x',       2.0),
        ic_xdot_max =p.get('ic_xdot',    2.5),
        seed=seed),
}


PENDULUM_TRAIN_SAMPLERS = {
    'uniform': lambda n, p, seed: sample_phase_space(
        n_samples=n,
        theta_max   =p.get('box_x',    np.pi),
        thetadot_max=p.get('box_xdot', 3.0),
        seed=seed),
    'gaussian': lambda n, p, seed: pendulum_sample_gaussian(
        n_samples=n,
        mean =p.get('mean'),
        sigma=p.get('sigma', 1.0),
        seed=seed),
    'gaussian_mixture': lambda n, p, seed: pendulum_sample_gaussian_mixture(
        n_samples=n,
        centers=p.get('centers'),
        sigmas =p.get('sigmas', 0.5),
        weights=p.get('weights'),
        seed=seed),
    'periodic_noise': lambda n, p, seed: pendulum_sample_periodic_noise(
        n_samples=n,
        amplitudes=p.get('amplitudes'),
        frequency =p.get('frequency', 1.0),
        center    =p.get('center'),
        noise_std =p.get('noise_std', 0.1),
        seed=seed),
    'trajectory': lambda n, p, seed: pendulum_sample_trajectory_ensemble(
        n_traj         =p.get('n_traj',          200),
        n_steps        =p.get('traj_steps',       50),
        dt             =p.get('dt',             0.05),
        ic_theta_max   =p.get('ic_theta',     np.pi),
        ic_thetadot_max=p.get('ic_thetadot',    3.0),
        seed=seed),
}


LORENZ_TRAIN_SAMPLERS = {
    'attractor':  None,             # special-cased: uses lorenz_generate
    'uniform': lambda n, p, seed: lorenz_sample_uniform(
        n_samples=n,
        box_x   =p.get('box_x',    25.0),
        box_y   =p.get('box_y',    30.0),
        box_z   =p.get('box_z',    25.0),
        z_offset=p.get('z_offset', 25.0),
        seed=seed),
    'gaussian': lambda n, p, seed: lorenz_sample_gaussian(
        n_samples=n,
        mean =p.get('mean'),
        sigma=p.get('sigma', 10.0),
        seed=seed),
    'gaussian_mixture': lambda n, p, seed: lorenz_sample_gaussian_mixture(
        n_samples=n,
        centers=p.get('centers'),
        sigmas =p.get('sigmas', 6.0),
        weights=p.get('weights'),
        seed=seed),
    'periodic_noise': lambda n, p, seed: lorenz_sample_periodic_noise(
        n_samples=n,
        amplitudes=p.get('amplitudes'),
        frequency =p.get('frequency', 1.0),
        center    =p.get('center'),
        noise_std =p.get('noise_std', 1.0),
        seed=seed),
    'trajectory': lambda n, p, seed: lorenz_sample_trajectory_ensemble(
        n_traj     =p.get('n_traj',         100),
        n_steps    =p.get('traj_steps',      50),
        dt         =p.get('dt',            0.01),
        ic_box     =p.get('ic_box',         20.0),
        ic_z_offset=p.get('ic_z_offset',    25.0),
        seed=seed),
}


def _horizon_key(h: float) -> str:
    """Stable string key for a rollout horizon (e.g., 5.0 -> 'r5s', 0.5 -> 'r0_5s')."""
    if float(h).is_integer():
        return f"r{int(h)}s"
    return "r" + ("%.2f" % h).rstrip('0').rstrip('.').replace('.', '_') + "s"


CONFIG_NAME = None
CONFIG_DESC = None
if args.config is not None:
    with open(args.config) as _fp:
        _cfg = yaml.safe_load(_fp)
    for _req in ('name', 'system', 'seeds'):
        if _req not in _cfg:
            raise ValueError(f"config {args.config!r} missing required field: {_req!r}")
    CONFIG_NAME = _cfg['name']
    CONFIG_DESC = _cfg.get('description', '')
    args.seeds  = _cfg['seeds']

    _system = _cfg['system'].lower()
    if   _system == 'lorenz':   _apply_lorenz_config(_cfg, args)
    elif _system == 'pendulum': _apply_pendulum_config(_cfg, args)
    elif _system == 'duffing':  _apply_duffing_config(_cfg, args)
    else:
        raise ValueError(f"unknown system in config {args.config!r}: {_system!r}")


def _out_paths(default_stem: str):
    """Return (json_path, fig_path, models_path) for this run.

    Layout:
      - With ``--run-id <id>``, every output is written under
        ``papers/data/<id>/`` (and ``papers/figures/<id>/``), so multiple
        invocations of ``run_all.sh`` (each with a distinct timestamp run-id)
        coexist without overwriting.
      - With ``--config <name>``, the per-file stem is the config's
        ``name:`` field; without it, the legacy ``statistical_<system>``
        stem is used.
    """
    stem   = CONFIG_NAME if CONFIG_NAME is not None else default_stem
    prefix = f"{args.run_id}/" if getattr(args, 'run_id', None) else ""
    return (data_path(f"{prefix}{stem}.json"),
            fig_path(f"{prefix}{stem}.png"),
            data_path(f"{prefix}{stem}_models.pt"))


seeds = args.seeds
N_SEEDS = len(seeds)

# Apply CLI caps after YAML has been merged into args
if args.n_iter_cap is not None:
    args.n_iter = min(args.n_iter, args.n_iter_cap)
if args.n_restarts_cap is not None:
    args.n_restarts = min(args.n_restarts, args.n_restarts_cap)
if args.n_train_cap is not None:
    args.lorenz_n_train   = min(args.lorenz_n_train,   args.n_train_cap)
    args.pendulum_n_train = min(args.pendulum_n_train, args.n_train_cap)
    args.duffing_n_train  = min(args.duffing_n_train,  args.n_train_cap)
if args.n_test_cap is not None:
    args.pendulum_n_test = min(args.pendulum_n_test, args.n_test_cap)
    args.duffing_n_test  = min(args.duffing_n_test,  args.n_test_cap)
if args.rollout_steps_cap is not None:
    args.lorenz_rollout_steps   = min(args.lorenz_rollout_steps,   args.rollout_steps_cap)
    args.pendulum_rollout_steps = min(args.pendulum_rollout_steps, args.rollout_steps_cap)
    args.duffing_rollout_steps  = min(args.duffing_rollout_steps,  args.rollout_steps_cap)

# Resolve worker count
if args.max_workers <= 0:
    args.max_workers = max(1, min(os.cpu_count() or 1, N_SEEDS))

# Multiprocessing safety on macOS: 'spawn' (the only safe start method for
# PyTorch on macOS) re-imports the module in each worker, but the child's
# argparse runs with a stripped sys.argv, so the YAML-loaded `args` does
# NOT propagate. Workers would silently fall back to defaults. Until the
# per-seed runners take args explicitly, force sequential on macOS.
if _sys.platform == 'darwin' and args.max_workers > 1:
    print(f"[warning] macOS detected; PyTorch+multiprocessing isn't reliably "
          f"safe under either fork (Objective-C runtime) or spawn (argparse "
          f"state isn't inherited). Forcing --max-workers=1.")
    print("[warning] Run on Linux (e.g., runpod) for the {n}x parallel speedup."
          .format(n=args.max_workers))
    args.max_workers = 1

print("=" * 80)
print("  Statistical Validation")
if CONFIG_NAME is not None:
    print(f"  Config: {CONFIG_NAME}  ({args.config})")
    if CONFIG_DESC:
        print(f"    {CONFIG_DESC.strip()}")
print(f"  Seeds ({N_SEEDS}): {seeds}")
print("=" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def pick_cluster(x, state):
    log_pi = torch.log(state['pi']).unsqueeze(0)
    log_prox = mvn_logpdf_batch(x, state['centers'], state['covariances'])
    return (log_pi + log_prox).argmax(dim=1)


def make_hp(X_tr, d):
    return {
        'alpha0': 0.5, 'mu0': X_tr.mean(dim=0),
        'Lambda0': 0.01 * torch.eye(d, dtype=torch.float64),
        'kappa0': 1.0, 'Psi0': 10.0 * torch.eye(d, dtype=torch.float64),
        'nu0': float(d + 2), 'sigma2': 'auto',
    }


def make_hp_pendulum(X_tr, d):
    return {
        'alpha0': 0.5, 'mu0': X_tr.mean(dim=0),
        'Lambda0': 0.01 * torch.eye(d, dtype=torch.float64),
        'kappa0': 1.0, 'Psi0': 1.0 * torch.eye(d, dtype=torch.float64),
        'nu0': float(d + 2), 'sigma2': 'auto',
    }


# ─────────────────────────────────────────────────────────────────────────────
# LORENZ per-seed
# ─────────────────────────────────────────────────────────────────────────────

def lorenz_piecewise_f(x, state):
    k = pick_cluster(x, state)
    c = state['centers'][k]
    fc = state['f_centers'][k]
    J = state['jacobians'][k]
    return fc + (J @ (x - c).unsqueeze(-1)).squeeze(-1)


def lorenz_one_step_err(state, X_te_in, X_te_next, dt):
    pred = X_te_in + dt * lorenz_piecewise_f(X_te_in, state)
    return torch.linalg.norm(pred - X_te_next, dim=1).mean().item()


def lorenz_rollout_truth(x0, n_steps, dt):
    sol = solve_ivp(lambda t, y: lorenz_f(y),
                    (0.0, n_steps * dt), x0.numpy(),
                    t_eval=np.linspace(0.0, n_steps * dt, n_steps + 1),
                    method='RK45', rtol=1e-10, atol=1e-10)
    return torch.tensor(sol.y.T, dtype=torch.float64)


def _lorenz_rollout_inner(x0_list, dt, n_steps, horizons, step_fn):
    """Generic rollout-error evaluator: ``step_fn(traj[t:t+1])`` -> next state.

    Defensive against divergence: if a step produces non-finite values
    (or ``step_fn`` itself errors on a non-finite input -- e.g. a Koopman
    operator with unstable eigenvalues that drives the rollout to NaN),
    the trajectory is filled with NaN from that step onward and per-init
    errors at affected horizons are reported as NaN (not propagated as
    an exception). Aggregation then uses ``np.nanmean`` so partially-
    diverged inits still contribute.
    """
    indices = {h: min(int(round(h / dt)), n_steps) for h in horizons}
    errs    = {h: [] for h in horizons}
    for x0 in x0_list:
        tru  = lorenz_rollout_truth(x0, n_steps, dt)
        traj = torch.zeros(n_steps + 1, 3, dtype=torch.float64)
        traj[0]     = x0
        diverged_at = n_steps + 1
        for t in range(n_steps):
            cur = traj[t:t + 1]
            if not torch.isfinite(cur).all():
                diverged_at  = t
                traj[t:]     = float('nan')
                break
            try:
                nxt = step_fn(cur)
            except (ValueError, RuntimeError):
                diverged_at    = t + 1
                traj[t + 1:]   = float('nan')
                break
            if not torch.isfinite(nxt).all():
                diverged_at    = t + 1
                traj[t + 1:]   = float('nan')
                break
            traj[t + 1] = nxt
        diff = torch.linalg.norm(traj - tru, dim=1)
        for h, idx in indices.items():
            v = diff[idx].item() if idx < diverged_at else float('nan')
            errs[h].append(v if np.isfinite(v) else float('nan'))
    out = {}
    for h in horizons:
        vals = np.array(errs[h], dtype=float)
        out[_horizon_key(h)] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float('nan')
    return out


def lorenz_rollout_err(state, inits, n_steps, dt, horizons):
    return _lorenz_rollout_inner(
        inits, dt, n_steps, horizons,
        lambda x: x[0] + dt * lorenz_piecewise_f(x, state)[0])


def lorenz_disc_rollout_err(model, inits, n_steps, dt, horizons):
    return _lorenz_rollout_inner(
        inits, dt, n_steps, horizons,
        lambda x: predict_next_disc(x, model)[0])


def lorenz_disc_local_rollout_err(state, inits, n_steps, d, dt, horizons):
    def _step(x):
        k     = pick_cluster(x, state)
        preds = predict_next_all_disc(x, state['centers'],
                                      state['K_ops'], state['exps'], d)
        return preds[0, k[0]]
    return _lorenz_rollout_inner(inits, dt, n_steps, horizons, _step)


def _make_lorenz_data(seed):
    """Build train/test data for Lorenz using args.lorenz_distribution.

    For ``attractor`` (default), the legacy single-trajectory pattern is used:
    one long ``lorenz_generate`` call, split into train + test by index.

    For other distributions, the train set is drawn from the requested
    distribution while the test set is a fresh attractor sample (so the
    held-out metric measures generalization to the actual invariant set).

    Returns ``(X_all, F_all, X_tr, F_tr, X_te_in, X_te_next, X_tr_curr,
    X_tr_next, step_bl)`` where the last several items are required by
    the Lorenz runner's downstream steps.
    """
    dt           = args.lorenz_dt
    distribution = args.lorenz_distribution
    params       = args.lorenz_distribution_params
    nt           = args.lorenz_n_train

    if distribution == 'attractor':
        data  = lorenz_generate(n_steps=args.lorenz_n_steps, dt=dt,
                                warmup=args.lorenz_warmup, seed=seed)
        X_all = torch.tensor(data['X'], dtype=torch.float64)
        F_all = torch.tensor(data['F'], dtype=torch.float64)
        X_tr      = X_all[:nt]
        F_tr      = F_all[:nt]
        X_te_in   = X_all[nt:X_all.shape[0] - 1]
        X_te_next = X_all[nt + 1:]
        X_tr_curr = X_all[:nt - 1]
        X_tr_next = X_all[1:nt]
    else:
        sampler = LORENZ_TRAIN_SAMPLERS.get(distribution)
        if sampler is None:
            raise ValueError(
                f"unknown lorenz distribution: {distribution!r}; "
                f"valid: {list(LORENZ_TRAIN_SAMPLERS)}")
        train = sampler(nt, params, seed)
        X_tr = torch.tensor(train['X'], dtype=torch.float64)
        F_tr = torch.tensor(train['F'], dtype=torch.float64)

        # Test set + discrete-EDMD pairs come from a fresh attractor trajectory
        # so prediction-error metrics measure generalization to the invariant
        # set independently of the training distribution.
        ref = lorenz_generate(n_steps=args.lorenz_n_steps, dt=dt,
                              warmup=args.lorenz_warmup, seed=seed + 10000)
        X_all = torch.tensor(ref['X'], dtype=torch.float64)
        F_all = torch.tensor(ref['F'], dtype=torch.float64)
        n_pair = max(nt, 4000)
        X_te_in   = X_all[:X_all.shape[0] - 1]
        X_te_next = X_all[1:]
        X_tr_curr = X_all[:n_pair - 1]
        X_tr_next = X_all[1:n_pair]

    step_bl = torch.linalg.norm(X_te_next - X_te_in, dim=1).mean().item()
    return (X_all, F_all, X_tr, F_tr,
            X_te_in, X_te_next, X_tr_curr, X_tr_next, step_bl)


def run_lorenz_seed(seed):
    dt = args.lorenz_dt
    d  = 3
    horizons = list(args.lorenz_horizons)

    (X_all, F_all, X_tr, F_tr, X_te_in, X_te_next,
     X_tr_curr, X_tr_next, step_bl) = _make_lorenz_data(seed)

    # Use a few attractor points as rollout initial conditions
    inits = [X_te_in[i] for i in (0,
                                  min(300, len(X_te_in) - 1),
                                  min(700, len(X_te_in) - 1))]
    rs = args.lorenz_rollout_steps

    results = {}

    # ── pykoopman baselines ──────────────────────────────────────────────
    for deg in args.edmd_degrees:
        model = pk.Koopman(
            observables=pk.observables.Polynomial(degree=deg, include_bias=True),
            regressor=pk.regression.EDMD())
        model.fit(X_tr.numpy(), dt=dt)
        one = torch.linalg.norm(
            torch.tensor(model.predict(X_te_in.numpy())) - X_te_next, dim=1
        ).mean().item()
        results[f'EDMD-pk deg-{deg}'] = {
            'one_step': one, 'rel_pct': 100 * one / step_bl,
        }

    # ── Discrete global EDMD (our solver, fair baseline) ─────────────────
    for deg in args.edmd_degrees:
        g = fit_global_disc(X_tr_curr, X_tr_next, degree=deg)
        pred = predict_next_disc(X_te_in, g)
        one = torch.linalg.norm(pred - X_te_next, dim=1).mean().item()
        roll = lorenz_disc_rollout_err(g, inits, rs, dt, horizons)
        results[f'EDMD-disc deg-{deg}'] = {
            'one_step': one, 'rel_pct': 100 * one / step_bl, **roll,
        }

    # ── Taylor-analytic (ours) + GMM ─────────────────────────────────────
    hp = make_hp(X_tr, d)
    for N in args.lorenz_N:
        # Ours
        s_o, _, _ = fit_taylor(X_tr, F_tr, lorenz_f, lorenz_J,
                               N=N, hp={**hp, 'sigma2': 'auto'},
                               n_iter=args.n_iter, n_restarts=args.n_restarts,
                               verbose=False)
        one_o = lorenz_one_step_err(s_o, X_te_in, X_te_next, dt)
        roll_o = lorenz_rollout_err(s_o, inits, rs, dt, horizons)
        results[f'Taylor N={N}'] = {
            'one_step': one_o, 'rel_pct': 100 * one_o / step_bl, **roll_o,
        }

        # GMM
        s_g, _, _ = fit_taylor(X_tr, F_tr, lorenz_f, lorenz_J,
                               N=N, hp={**hp, 'sigma2': 1e10},
                               n_iter=args.n_iter, n_restarts=args.n_restarts,
                               verbose=False)
        one_g = lorenz_one_step_err(s_g, X_te_in, X_te_next, dt)
        roll_g = lorenz_rollout_err(s_g, inits, rs, dt, horizons)
        results[f'GMM N={N}'] = {
            'one_step': one_g, 'rel_pct': 100 * one_g / step_bl, **roll_g,
        }

    # ── Local discrete EDMD (loop over configured degrees) ──────────────
    for deg in args.lorenz_le_degrees:
        for N in args.lorenz_N:
            s_ld, _, _ = fit_local_edmd_disc(
                X_tr_curr, X_tr_next, N=N, hp={**hp, 'sigma2': 'auto'},
                degree=deg, n_iter=args.n_iter, n_restarts=args.n_restarts,
                verbose=False)
            k = pick_cluster(X_te_in, s_ld)
            pred = predict_next_all_disc(
                X_te_in, s_ld['centers'], s_ld['K_ops'], s_ld['exps'], d)
            one_ld = torch.linalg.norm(
                pred[torch.arange(len(X_te_in)), k] - X_te_next, dim=1
            ).mean().item()
            roll_ld = lorenz_disc_local_rollout_err(s_ld, inits, rs, d, dt, horizons)
            # Backwards-compat label: degree 2 keeps the legacy
            # "Local-EDMD-disc N=k" name; higher degrees use the disambiguated
            # "Local-EDMD-disc d{deg} N=k" form (same convention as Duffing).
            label = (f'Local-EDMD-disc N={N}' if deg == 2
                     else f'Local-EDMD-disc d{deg} N={N}')
            results[label] = {
                'one_step': one_ld, 'rel_pct': 100 * one_ld / step_bl, **roll_ld,
            }

    # Collect model states for visualization
    models = {}
    # Re-fit best N for each family to store (use median N)
    best_N = args.lorenz_N[len(args.lorenz_N) // 2]
    s_taylor, _, _ = fit_taylor(X_tr, F_tr, lorenz_f, lorenz_J,
                                N=best_N, hp={**hp, 'sigma2': 'auto'},
                                n_iter=args.n_iter, n_restarts=args.n_restarts,
                                verbose=False)
    models['taylor'] = s_taylor

    s_gmm, _, _ = fit_taylor(X_tr, F_tr, lorenz_f, lorenz_J,
                             N=best_N, hp={**hp, 'sigma2': 1e10},
                             n_iter=args.n_iter, n_restarts=args.n_restarts,
                             verbose=False)
    models['gmm'] = s_gmm

    s_disc, _, _ = fit_local_edmd_disc(
        X_tr_curr, X_tr_next, N=best_N, hp={**hp, 'sigma2': 'auto'},
        degree=2, n_iter=args.n_iter, n_restarts=args.n_restarts,
        verbose=False)
    models['local_edmd_disc'] = s_disc

    g_disc = fit_global_disc(X_tr_curr, X_tr_next, degree=2)
    models['global_edmd_disc'] = g_disc

    models['X_all'] = X_all
    models['F_all'] = F_all

    return results, models


# ─────────────────────────────────────────────────────────────────────────────
# PENDULUM per-seed
# ─────────────────────────────────────────────────────────────────────────────

def pendulum_predict_f_taylor(x, state):
    k = pick_cluster(x, state)
    c = state['centers'][k]
    fc = state['f_centers'][k]
    J = state['jacobians'][k]
    return fc + (J @ (x - c).unsqueeze(-1)).squeeze(-1)


def pendulum_euler_step(x, f_val, dt):
    return wrap_theta(x + dt * f_val)


def pendulum_rollout(x0, predict_fn, model, n_steps, dt, d):
    traj = torch.zeros(n_steps + 1, d, dtype=torch.float64)
    traj[0] = x0
    for t in range(n_steps):
        f_hat = predict_fn(traj[t:t + 1], model)[0]
        traj[t + 1] = pendulum_euler_step(traj[t], f_hat, dt)
    return traj


def pendulum_generate_trajectory_pairs(n_trajs, traj_len, dt, seed):
    """Generate consecutive (x_t, x_{t+1}) pairs from multiple pendulum trajectories."""
    rng = np.random.default_rng(seed)
    X_curr, X_next = [], []
    for _ in range(n_trajs):
        x0 = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-3.0, 3.0)])
        traj = generate_trajectory(x0, n_steps=traj_len, dt=dt, wrap=True)
        X_curr.append(traj[:-1])
        X_next.append(traj[1:])
    return (torch.tensor(np.concatenate(X_curr), dtype=torch.float64),
            torch.tensor(np.concatenate(X_next), dtype=torch.float64))


def _pendulum_rollout_inner(inits, n_steps, dt, d, horizons, step_fn):
    """Generic pendulum rollout-error eval with angular distance.

    Defensive against divergence (see ``_lorenz_rollout_inner`` for the
    same NaN-handling protocol).
    """
    indices = {h: min(int(round(h / dt)), n_steps) for h in horizons}
    errs    = {h: [] for h in horizons}
    for x0 in inits:
        tru = torch.tensor(generate_trajectory(x0.numpy(), n_steps=n_steps, dt=dt),
                           dtype=torch.float64)
        traj = torch.zeros(n_steps + 1, d, dtype=torch.float64)
        traj[0]     = x0
        diverged_at = n_steps + 1
        for t in range(n_steps):
            cur = traj[t:t + 1]
            if not torch.isfinite(cur).all():
                diverged_at  = t
                traj[t:]     = float('nan')
                break
            try:
                nxt = step_fn(cur)
            except (ValueError, RuntimeError):
                diverged_at    = t + 1
                traj[t + 1:]   = float('nan')
                break
            if not torch.isfinite(nxt).all():
                diverged_at    = t + 1
                traj[t + 1:]   = float('nan')
                break
            traj[t + 1] = nxt
        diff = angular_dist(traj, tru)
        for h, idx in indices.items():
            v = diff[idx].item() if idx < diverged_at else float('nan')
            errs[h].append(v if np.isfinite(v) else float('nan'))
    out = {}
    for h in horizons:
        vals = np.array(errs[h], dtype=float)
        out[_horizon_key(h)] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float('nan')
    return out


def pendulum_disc_rollout_err(model, inits, n_steps, dt, d, horizons):
    return _pendulum_rollout_inner(
        inits, n_steps, dt, d, horizons,
        lambda x: wrap_theta(predict_next_disc(x, model)))


def pendulum_disc_local_rollout_err(state, inits, n_steps, dt, d, horizons):
    def _step(x):
        k     = pick_cluster(x, state)
        preds = predict_next_all_disc(x, state['centers'],
                                      state['K_ops'], state['exps'], d)
        return wrap_theta(preds[0, k[0]])
    return _pendulum_rollout_inner(inits, n_steps, dt, d, horizons, _step)


def pendulum_eval_rollout(predict_fn, model, inits, n_roll, dt, d, horizons):
    def _step(x):
        return pendulum_euler_step(x[0], predict_fn(x, model)[0], dt)
    return _pendulum_rollout_inner(inits, n_roll, dt, d, horizons, _step)


def _make_pendulum_data(seed):
    """Build train / test data for Pendulum using args.pendulum_distribution."""
    sampler = PENDULUM_TRAIN_SAMPLERS.get(args.pendulum_distribution)
    if sampler is None:
        raise ValueError(
            f"unknown pendulum distribution: {args.pendulum_distribution!r}; "
            f"valid: {list(PENDULUM_TRAIN_SAMPLERS)}")
    train = sampler(args.pendulum_n_train, args.pendulum_distribution_params, seed)
    # Test set is uniform-on-box for a clean prediction-error metric (parity
    # with Duffing).
    test = sample_phase_space(n_samples=args.pendulum_n_test, seed=seed + 10000)
    return train, test


def run_pendulum_seed(seed):
    dt        = args.pendulum_dt
    d         = 2
    n_roll    = args.pendulum_rollout_steps
    horizons  = list(args.pendulum_horizons)

    # -- training and test data (distribution-dispatched) --------------------
    train, test = _make_pendulum_data(seed)
    X_tr = torch.tensor(train['X'], dtype=torch.float64)
    F_tr = torch.tensor(train['F'], dtype=torch.float64)
    X_te = torch.tensor(test ['X'], dtype=torch.float64)
    F_te = torch.tensor(test ['F'], dtype=torch.float64)

    hp = make_hp_pendulum(X_tr, d)

    rollout_inits = [
        torch.tensor([0.3, 0.0]), torch.tensor([1.5, 0.0]),
        torch.tensor([2.8, 0.0]), torch.tensor([0.0, 2.5]),
        torch.tensor([-2.0, 1.0]),
    ]

    results = {}

    # Generate trajectory pairs for discrete EDMD (independent of training
    # distribution; needed because discrete-EDMD baselines fit on (x_t, x_{t+1}))
    X_tr_curr, X_tr_next = pendulum_generate_trajectory_pairs(
        args.pendulum_n_trajs, args.pendulum_traj_len, dt, seed)

    # Discrete test pairs: integrate from test inits for one step
    X_te_disc_curr = X_te
    X_te_disc_next = torch.stack([
        torch.tensor(generate_trajectory(x.numpy(), n_steps=1, dt=dt)[1],
                     dtype=torch.float64)
        for x in X_te
    ])

    # ── Global discrete EDMD ─────────────────────────────────────────────
    for deg in args.pendulum_edmd_degrees:
        g = fit_global_disc(X_tr_curr, X_tr_next, degree=deg)
        pred = predict_next_disc(X_te_disc_curr, g)
        one  = angular_dist(pred, X_te_disc_next).mean().item()
        roll = pendulum_disc_rollout_err(g, rollout_inits, n_roll, dt, d, horizons)
        results[f'EDMD-disc deg={deg}'] = {'one_step': one, **roll}

    # ── Local discrete EDMD (loop over configured degrees) ──────────────
    hp_disc = make_hp_pendulum(X_tr_curr, d)
    for deg in args.pendulum_le_degrees:
        for N in args.pendulum_N:
            s_ld, _, _ = fit_local_edmd_disc(
                X_tr_curr, X_tr_next, N=N, hp={**hp_disc, 'sigma2': 'auto'},
                degree=deg, n_iter=args.n_iter, n_restarts=args.n_restarts,
                verbose=False)
            k = pick_cluster(X_te_disc_curr, s_ld)
            preds = predict_next_all_disc(
                X_te_disc_curr, s_ld['centers'], s_ld['K_ops'], s_ld['exps'], d)
            one  = angular_dist(
                preds[torch.arange(len(X_te_disc_curr)), k], X_te_disc_next
            ).mean().item()
            roll = pendulum_disc_local_rollout_err(s_ld, rollout_inits, n_roll, dt, d, horizons)
            label = (f'Local-EDMD-disc N={N}' if deg == 2
                     else f'Local-EDMD-disc d{deg} N={N}')
            results[label] = {'one_step': one, **roll}

    # ── Taylor-analytic ──────────────────────────────────────────────────
    for N in args.pendulum_N:
        s, _, _ = fit_taylor(
            X_tr, F_tr, pendulum_f, pendulum_J,
            N=N, hp={**hp, 'sigma2': 'auto'},
            n_iter=args.n_iter, n_restarts=args.n_restarts, verbose=False)
        F_pred = pendulum_predict_f_taylor(X_te, s)
        one  = torch.linalg.norm(F_pred - F_te, dim=1).mean().item()
        roll = pendulum_eval_rollout(
            pendulum_predict_f_taylor, s, rollout_inits, n_roll, dt, d, horizons)
        results[f'Taylor-analytic N={N}'] = {'one_step': one, **roll}

    # ── GMM-baseline (geometry-only Taylor: sigma2=1e10 disables residual term) ──
    for N in args.pendulum_N:
        s, _, _ = fit_taylor(
            X_tr, F_tr, pendulum_f, pendulum_J,
            N=N, hp={**hp, 'sigma2': 1e10},
            n_iter=args.n_iter, n_restarts=args.n_restarts, verbose=False)
        F_pred = pendulum_predict_f_taylor(X_te, s)
        one  = torch.linalg.norm(F_pred - F_te, dim=1).mean().item()
        roll = pendulum_eval_rollout(
            pendulum_predict_f_taylor, s, rollout_inits, n_roll, dt, d, horizons)
        results[f'GMM-baseline N={N}'] = {'one_step': one, **roll}

    # Collect model states for visualization
    models = {}
    best_N = args.pendulum_N[len(args.pendulum_N) // 2]

    s_taylor, _, _ = fit_taylor(
        X_tr, F_tr, pendulum_f, pendulum_J,
        N=best_N, hp={**hp, 'sigma2': 'auto'},
        n_iter=args.n_iter, n_restarts=args.n_restarts, verbose=False)
    models['taylor'] = s_taylor

    hp_disc = make_hp_pendulum(X_tr_curr, d)
    s_disc, _, _ = fit_local_edmd_disc(
        X_tr_curr, X_tr_next, N=best_N, hp={**hp_disc, 'sigma2': 'auto'},
        degree=2, n_iter=args.n_iter, n_restarts=args.n_restarts, verbose=False)
    models['local_edmd_disc'] = s_disc

    g_global_disc = fit_global_disc(X_tr_curr, X_tr_next, degree=2)
    models['global_edmd_disc'] = g_global_disc

    models['X_tr'] = X_tr
    models['F_tr'] = F_tr
    models['X_te'] = X_te
    models['F_te'] = F_te

    return results, models


# ─────────────────────────────────────────────────────────────────────────────
# Run all seeds
# ─────────────────────────────────────────────────────────────────────────────

def _accumulate_seed(all_runs, results):
    """Fold a single seed's results dict into the shared all_runs dict."""
    for method, metrics in results.items():
        if method not in all_runs:
            all_runs[method] = {m: [] for m in metrics}
        for m, v in metrics.items():
            all_runs[method][m].append(v)


def run_system(name, run_fn):
    all_runs       = {}
    best_models    = None
    best_seed      = None
    best_total_err = float('inf')
    n_workers      = max(1, min(args.max_workers, N_SEEDS))

    print(f"\n  [{name}] running {N_SEEDS} seeds across "
          f"{n_workers} worker process(es)")
    t0 = time.time()

    if n_workers == 1:
        for i, seed in enumerate(seeds):
            print(f"\n  {name} — seed {seed} ({i + 1}/{N_SEEDS})")
            results, models = run_fn(seed)
            _accumulate_seed(all_runs, results)
            total_err = np.mean(
                [m['one_step'] for m in results.values() if 'one_step' in m])
            if total_err < best_total_err:
                best_total_err = total_err
                best_models    = models
                best_seed      = seed
    else:
        # Parallel: submit one task per seed, collect as they finish.
        # `fork` start method (set at module load) lets each worker inherit
        # the parsed `args` and torch state without re-running argparse / YAML.
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            future_to_seed = {ex.submit(run_fn, s): s for s in seeds}
            n_done = 0
            for fut in as_completed(future_to_seed):
                seed = future_to_seed[fut]
                try:
                    results, models = fut.result()
                except Exception as exc:
                    print(f"  {name} seed {seed} FAILED: {exc!r}")
                    raise
                n_done += 1
                _accumulate_seed(all_runs, results)
                total_err = np.mean(
                    [m['one_step'] for m in results.values() if 'one_step' in m])
                elapsed = time.time() - t0
                eta     = elapsed / n_done * (N_SEEDS - n_done)
                print(f"  {name} — seed {seed} done "
                      f"({n_done}/{N_SEEDS}, "
                      f"avg one-step={total_err:.5f}, "
                      f"elapsed={elapsed:.0f}s, eta={eta:.0f}s)")
                if total_err < best_total_err:
                    best_total_err = total_err
                    best_models    = models
                    best_seed      = seed

    print(f"\n  Best seed for {name}: {best_seed} "
          f"(avg one-step err: {best_total_err:.6f}, total {time.time()-t0:.0f}s)")
    return all_runs, best_models, best_seed


def report(name, all_runs, metrics_list):
    print(f"\n{'=' * 100}")
    print(f"{name} — {N_SEEDS} seeds, mean ± 95% CI")
    print("=" * 100)
    header = f"{'method':<30}"
    for m in metrics_list:
        header += f" {m:>22s}"
    print(header)
    print("-" * 100)
    for method in all_runs:
        line = f"{method:<30}"
        for m in metrics_list:
            if m in all_runs[method]:
                mean, hw, _, _ = confidence_interval(all_runs[method][m])
                line += f" {mean:>10.5f} ± {hw:.5f}"
            else:
                line += f" {'—':>22s}"
        print(line)


def paired_tests(name, all_runs, pairs):
    print(f"\n  Paired t-tests ({name}):")
    for na, nb, metric, label in pairs:
        if na not in all_runs or nb not in all_runs:
            continue
        if metric not in all_runs[na] or metric not in all_runs[nb]:
            continue
        a = np.array(all_runs[na][metric])
        b = np.array(all_runs[nb][metric])
        t = paired_test(a, b)
        sig = "***" if t['p_value'] < 0.001 else "**" if t['p_value'] < 0.01 \
            else "*" if t['p_value'] < 0.05 else "ns"
        print(f"    {label}: {a.mean():.5f} vs {b.mean():.5f}, "
              f"diff={t['mean_diff']:+.5f}, p={t['p_value']:.4f} {sig}")


# ─────────────────────────────────────────────────────────────────────────────
# LORENZ
# ─────────────────────────────────────────────────────────────────────────────

if not args.skip_lorenz:
    print("\n" + "=" * 80)
    print("  LORENZ SYSTEM")
    print("=" * 80)

    lorenz_runs, lorenz_models, lorenz_best_seed = run_system("Lorenz", run_lorenz_seed)
    lorenz_metrics = ['one_step', 'rel_pct'] + [_horizon_key(h) for h in args.lorenz_horizons]
    report("LORENZ", lorenz_runs, lorenz_metrics)

    lorenz_pairs = []
    for N in args.lorenz_N:
        lorenz_pairs.append(
            (f'Taylor N={N}', f'GMM N={N}', 'one_step', f'Taylor vs GMM N={N}'))
    for N in args.lorenz_N:
        lorenz_pairs.append(
            (f'Taylor N={N}', 'EDMD-disc deg-2', 'one_step', f'Taylor N={N} vs EDMD-disc deg-2'))
    for N in args.lorenz_N:
        lorenz_pairs.append(
            (f'Local-EDMD-disc N={N}', 'EDMD-disc deg-2', 'one_step',
             f'Local-EDMD N={N} vs EDMD-disc deg-2'))
    paired_tests("Lorenz", lorenz_runs, lorenz_pairs)

    _json, _fig, _models = _out_paths("statistical_lorenz")
    with open(_json, "w") as fp:
        json.dump({'n_seeds': N_SEEDS, 'seeds': seeds, 'args': vars(args),
                   'config_name': CONFIG_NAME, 'config_description': CONFIG_DESC,
                   'best_seed': lorenz_best_seed,
                   'results': lorenz_runs}, fp, indent=2)
    print(f"\n  Saved: {_json}")

    if not args.no_save_models:
        torch.save(lorenz_models, _models)
        print(f"  Saved models (best seed {lorenz_best_seed}): {_models}")

# ─────────────────────────────────────────────────────────────────────────────
# PENDULUM
# ─────────────────────────────────────────────────────────────────────────────

if not args.skip_pendulum:
    print("\n" + "=" * 80)
    print("  PENDULUM SYSTEM")
    print("=" * 80)

    pendulum_runs, pendulum_models, pendulum_best_seed = run_system("Pendulum", run_pendulum_seed)
    pendulum_metrics = ['one_step'] + [_horizon_key(h) for h in args.pendulum_horizons]
    report("PENDULUM", pendulum_runs, pendulum_metrics)

    pendulum_headline_h_key = (_horizon_key(args.pendulum_horizons[-1])
                               if args.pendulum_horizons else 'one_step')
    pendulum_headline_h_label = (f"rollout {args.pendulum_horizons[-1]}s"
                                 if args.pendulum_horizons else 'one_step')
    biggest_pend_deg = (args.pendulum_edmd_degrees[-1]
                        if args.pendulum_edmd_degrees else 2)
    biggest_pend_N   = args.pendulum_N[-1] if args.pendulum_N else 8
    pendulum_pairs = [
        (f'Taylor-analytic N={biggest_pend_N}',
         f'EDMD-disc deg={biggest_pend_deg}', pendulum_headline_h_key,
         f'Taylor-ana N={biggest_pend_N} vs EDMD-disc deg={biggest_pend_deg} '
         f'({pendulum_headline_h_label})'),
    ]
    for N in args.pendulum_N:
        pendulum_pairs.append(
            (f'Local-EDMD-disc N={N}', f'EDMD-disc deg=2', 'one_step',
             f'Local-EDMD-disc N={N} vs EDMD-disc deg=2'))
    for N in args.pendulum_N:
        pendulum_pairs.append(
            (f'Local-EDMD-disc N={N}', f'EDMD-disc deg=2', pendulum_headline_h_key,
             f'Local-EDMD-disc N={N} vs EDMD-disc deg=2 ({pendulum_headline_h_label})'))
    paired_tests("Pendulum", pendulum_runs, pendulum_pairs)

    _json, _fig, _models = _out_paths("statistical_pendulum")
    with open(_json, "w") as fp:
        json.dump({'n_seeds': N_SEEDS, 'seeds': seeds, 'args': vars(args),
                   'config_name': CONFIG_NAME, 'config_description': CONFIG_DESC,
                   'best_seed': pendulum_best_seed,
                   'results': pendulum_runs}, fp, indent=2)
    print(f"\n  Saved: {_json}")

    if not args.no_save_models:
        torch.save(pendulum_models, _models)
        print(f"  Saved models (best seed {pendulum_best_seed}): {_models}")

# ─────────────────────────────────────────────────────────────────────────────
# DUFFING per-seed
# ─────────────────────────────────────────────────────────────────────────────

DUFFING_ROLLOUT_INITS = [
    torch.tensor([+1.5, 0.0]),
    torch.tensor([-1.5, 0.0]),
    torch.tensor([+0.3, 0.0]),
    torch.tensor([-0.3, 0.0]),
    torch.tensor([+0.5, 1.5]),
]


def duffing_predict_f_taylor(x, state):
    """Taylor-analytic local-linear prediction of the velocity field f(x)."""
    k  = pick_cluster(x, state)
    c  = state['centers'][k]
    fc = state['f_centers'][k]
    Jk = state['jacobians'][k]
    return fc + (Jk @ (x - c).unsqueeze(-1)).squeeze(-1)


def _duffing_one_step_pairs(X, dt):
    """Integrate each row of ``X`` forward by ``dt`` to produce x_{t+1}.

    Vectorized RK4: a single step of fourth-order Runge-Kutta evaluates the
    Duffing vector field 4 times across the whole batch (numpy array ops),
    producing all next-state predictions in O(N) total time. This replaces
    a previous per-point ``solve_ivp`` loop that was O(N) calls (each with
    its own adaptive-step setup), which dominated runtime when ``N_train``
    was a few thousand.
    """
    Xnp = X.cpu().numpy() if hasattr(X, 'cpu') else np.asarray(X)
    # duffing_f operates on a single state; vectorise over rows
    def _f_batch(Xb):
        # f(state) for Duffing is cheap; batch via numpy broadcasting
        x  = Xb[:, 0]
        xd = Xb[:, 1]
        return np.column_stack((
            xd,
            -DUFFING_DELTA * xd + x - x ** 3,
        ))
    k1 = _f_batch(Xnp)
    k2 = _f_batch(Xnp + 0.5 * dt * k1)
    k3 = _f_batch(Xnp + 0.5 * dt * k2)
    k4 = _f_batch(Xnp + dt       * k3)
    Xn = Xnp + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return torch.tensor(Xn, dtype=torch.float64)


def duffing_eval_rollout_taylor(predict_fn, model, dt, n_steps, d, horizons):
    """Taylor / Euler-stepped rollout error per horizon.

    Used only for Variant A (Taylor-analytic / GMM-baseline), which models
    the velocity field. The discrete-EDMD methods (global and local) use
    ``duffing_disc_rollout_err`` / ``duffing_disc_local_rollout_err``
    instead -- no Euler step, no derivatives.
    """
    indices = {h: min(int(round(h / dt)), n_steps) for h in horizons}
    errs    = {h: [] for h in horizons}
    for x0 in DUFFING_ROLLOUT_INITS:
        tru = torch.tensor(duffing_generate_trajectory(
            x0.numpy(), n_steps=n_steps, dt=dt), dtype=torch.float64)
        traj = torch.zeros(n_steps + 1, d, dtype=torch.float64)
        traj[0]     = x0
        diverged_at = n_steps + 1
        for t in range(n_steps):
            cur = traj[t:t + 1]
            if not torch.isfinite(cur).all():
                diverged_at = t; traj[t:] = float('nan'); break
            try:
                f_hat = predict_fn(cur, model)[0]
            except (ValueError, RuntimeError):
                diverged_at = t + 1; traj[t + 1:] = float('nan'); break
            nxt = traj[t] + dt * f_hat
            if not torch.isfinite(nxt).all():
                diverged_at = t + 1; traj[t + 1:] = float('nan'); break
            traj[t + 1] = nxt
        diff = torch.linalg.norm(traj - tru, dim=1)
        for h, idx in indices.items():
            v = diff[idx].item() if idx < diverged_at else float('nan')
            errs[h].append(v if np.isfinite(v) else float('nan'))
    out = {}
    for h in horizons:
        vals = np.array(errs[h], dtype=float)
        out[_horizon_key(h)] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float('nan')
    return out


def duffing_disc_rollout_err(model, inits, n_steps, dt, d, horizons):
    """Discrete-EDMD rollout error (global): iterate Koopman matrix without Euler."""
    indices = {h: min(int(round(h / dt)), n_steps) for h in horizons}
    errs    = {h: [] for h in horizons}
    for x0 in inits:
        tru = torch.tensor(duffing_generate_trajectory(
            x0.numpy(), n_steps=n_steps, dt=dt), dtype=torch.float64)
        traj = torch.zeros(n_steps + 1, d, dtype=torch.float64)
        traj[0]     = x0
        diverged_at = n_steps + 1
        for t in range(n_steps):
            cur = traj[t:t + 1]
            if not torch.isfinite(cur).all():
                diverged_at = t; traj[t:] = float('nan'); break
            try:
                nxt = predict_next_disc(cur, model)[0]
            except (ValueError, RuntimeError):
                diverged_at = t + 1; traj[t + 1:] = float('nan'); break
            if not torch.isfinite(nxt).all():
                diverged_at = t + 1; traj[t + 1:] = float('nan'); break
            traj[t + 1] = nxt
        diff = torch.linalg.norm(traj - tru, dim=1)
        for h, idx in indices.items():
            v = diff[idx].item() if idx < diverged_at else float('nan')
            errs[h].append(v if np.isfinite(v) else float('nan'))
    out = {}
    for h in horizons:
        vals = np.array(errs[h], dtype=float)
        out[_horizon_key(h)] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float('nan')
    return out


def duffing_disc_local_rollout_err(state, inits, n_steps, dt, d, horizons):
    """Discrete-EDMD rollout error (local per-cluster Koopman operators)."""
    indices = {h: min(int(round(h / dt)), n_steps) for h in horizons}
    errs    = {h: [] for h in horizons}
    for x0 in inits:
        tru = torch.tensor(duffing_generate_trajectory(
            x0.numpy(), n_steps=n_steps, dt=dt), dtype=torch.float64)
        traj = torch.zeros(n_steps + 1, d, dtype=torch.float64)
        traj[0]     = x0
        diverged_at = n_steps + 1
        for t in range(n_steps):
            cur = traj[t:t + 1]
            if not torch.isfinite(cur).all():
                diverged_at = t; traj[t:] = float('nan'); break
            try:
                k     = pick_cluster(cur, state)
                preds = predict_next_all_disc(cur, state['centers'],
                                              state['K_ops'], state['exps'], d)
                nxt   = preds[0, k[0]]
            except (ValueError, RuntimeError):
                diverged_at = t + 1; traj[t + 1:] = float('nan'); break
            if not torch.isfinite(nxt).all():
                diverged_at = t + 1; traj[t + 1:] = float('nan'); break
            traj[t + 1] = nxt
        diff = torch.linalg.norm(traj - tru, dim=1)
        for h, idx in indices.items():
            v = diff[idx].item() if idx < diverged_at else float('nan')
            errs[h].append(v if np.isfinite(v) else float('nan'))
    out = {}
    for h in horizons:
        vals = np.array(errs[h], dtype=float)
        out[_horizon_key(h)] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float('nan')
    return out


def _make_duffing_data(seed):
    """Build train / test data for Duffing using args.duffing_distribution.

    Returns ``(train, test, X_tr_curr, X_tr_next, X_te_curr, X_te_next)``.
    The first two are the standard ``{X, F, J_all}`` dicts used by the
    Taylor-analytic / GMM-baseline (velocity-field) variants. The latter
    four are consecutive-state pair tensors derived from those samples by
    integrating each point one ``dt`` forward; they feed the discrete-
    EDMD baselines (global and local) which require ``(x_t, x_{t+1})``
    pairs and never touch derivative data.
    """
    sampler = DUFFING_TRAIN_SAMPLERS.get(args.duffing_distribution)
    if sampler is None:
        raise ValueError(
            f"unknown duffing distribution: {args.duffing_distribution!r}; "
            f"valid: {list(DUFFING_TRAIN_SAMPLERS)}")
    train = sampler(args.duffing_n_train, args.duffing_distribution_params, seed)
    # Test set is always uniform-on-box for a clean prediction-error metric
    test = duffing_sample_phase_space(
        n_samples=args.duffing_n_test,
        x_max   =args.duffing_test_box_x,
        xdot_max=args.duffing_test_box_xdot,
        seed=seed + 10000)
    dt = args.duffing_dt
    X_tr_curr = torch.tensor(train['X'], dtype=torch.float64)
    X_te_curr = torch.tensor(test ['X'], dtype=torch.float64)
    X_tr_next = _duffing_one_step_pairs(X_tr_curr, dt)
    X_te_next = _duffing_one_step_pairs(X_te_curr, dt)
    return train, test, X_tr_curr, X_tr_next, X_te_curr, X_te_next


def run_duffing_seed(seed):
    dt        = args.duffing_dt
    d         = 2
    n_roll    = args.duffing_rollout_steps
    horizons  = list(args.duffing_horizons)

    # -- training and test data (distribution-dispatched) --------------------
    train, test, X_tr_curr, X_tr_next, X_te_curr, X_te_next = _make_duffing_data(seed)
    X_tr = X_tr_curr
    F_tr = torch.tensor(train['F'], dtype=torch.float64)
    X_te = X_te_curr
    F_te = torch.tensor(test ['F'], dtype=torch.float64)

    hp_base = {
        'alpha0':  0.5, 'mu0': X_tr.mean(dim=0),
        'Lambda0': 0.01 * torch.eye(d, dtype=torch.float64),
        'kappa0':  1.0, 'Psi0': 1.0  * torch.eye(d, dtype=torch.float64),
        'nu0':     float(d + 2),
    }
    hp_disc = {
        'alpha0':  0.5, 'mu0': X_tr_curr.mean(dim=0),
        'Lambda0': 0.01 * torch.eye(d, dtype=torch.float64),
        'kappa0':  1.0, 'Psi0': 1.0  * torch.eye(d, dtype=torch.float64),
        'nu0':     float(d + 2),
    }

    results = {}

    # ── Global discrete EDMD ────────────────────────────────────────────────
    for deg in args.duffing_edmd_degrees:
        g    = fit_global_disc(X_tr_curr, X_tr_next, degree=deg)
        pred = predict_next_disc(X_te_curr, g)
        one  = torch.linalg.norm(pred - X_te_next, dim=1).mean().item()
        roll = duffing_disc_rollout_err(g, DUFFING_ROLLOUT_INITS, n_roll, dt, d, horizons)
        results[f"EDMD-disc deg={deg}"] = {'one_step': one, **roll}

    # ── Local discrete EDMD (deg-2 and deg-3) ──────────────────────────────
    for deg, N_list in [(2, args.duffing_le2_N),
                        (3, args.duffing_le3_N),
                        (4, args.duffing_le4_N),
                        (5, args.duffing_le5_N)]:
        for N in N_list:
            state, _, _ = fit_local_edmd_disc(
                X_tr_curr, X_tr_next, N=N,
                hp={**hp_disc, 'sigma2': 'auto'}, degree=deg,
                n_iter=args.n_iter, n_restarts=args.n_restarts, verbose=False)
            k    = pick_cluster(X_te_curr, state)
            pred = predict_next_all_disc(
                X_te_curr, state['centers'], state['K_ops'], state['exps'], d)
            one  = torch.linalg.norm(
                pred[torch.arange(len(X_te_curr)), k] - X_te_next, dim=1
            ).mean().item()
            roll = duffing_disc_local_rollout_err(
                state, DUFFING_ROLLOUT_INITS, n_roll, dt, d, horizons)
            results[f"Local-EDMD-disc d{deg} N={N}"] = {'one_step': one, **roll}

    # ── Taylor-analytic (Variant A: physics-aware, uses J) ─────────────────
    for N in args.duffing_N:
        state, _, _ = fit_taylor(
            X_tr, F_tr, duffing_f, duffing_J, N=N,
            hp={**hp_base, 'sigma2': 'auto'},
            n_iter=args.n_iter, n_restarts=args.n_restarts, verbose=False)
        F_pred = duffing_predict_f_taylor(X_te, state)
        one  = torch.linalg.norm(F_pred - F_te, dim=1).mean().item()
        roll = duffing_eval_rollout_taylor(
            duffing_predict_f_taylor, state, dt, n_roll, d, horizons)
        results[f"Taylor-analytic N={N}"] = {'one_step': one, **roll}

    # ── GMM baseline (Taylor local model + sigma2 -> inf removes residual) ─
    for N in args.duffing_N:
        state, _, _ = fit_taylor(
            X_tr, F_tr, duffing_f, duffing_J, N=N,
            hp={**hp_base, 'sigma2': 1e10},
            n_iter=args.n_iter, n_restarts=args.n_restarts, verbose=False)
        F_pred = duffing_predict_f_taylor(X_te, state)
        one  = torch.linalg.norm(F_pred - F_te, dim=1).mean().item()
        roll = duffing_eval_rollout_taylor(
            duffing_predict_f_taylor, state, dt, n_roll, d, horizons)
        results[f"GMM-baseline N={N}"] = {'one_step': one, **roll}

    # ── Representative models for visualization ────────────────────────────
    models = {}
    best_N = args.duffing_N[len(args.duffing_N) // 2]
    s_taylor, _, _ = fit_taylor(
        X_tr, F_tr, duffing_f, duffing_J, N=best_N,
        hp={**hp_base, 'sigma2': 'auto'},
        n_iter=args.n_iter, n_restarts=args.n_restarts, verbose=False)
    models['taylor'] = s_taylor
    s_gmm, _, _ = fit_taylor(
        X_tr, F_tr, duffing_f, duffing_J, N=best_N,
        hp={**hp_base, 'sigma2': 1e10},
        n_iter=args.n_iter, n_restarts=args.n_restarts, verbose=False)
    models['gmm'] = s_gmm
    if args.duffing_le2_N:
        s_le2, _, _ = fit_local_edmd_disc(
            X_tr_curr, X_tr_next, N=args.duffing_le2_N[len(args.duffing_le2_N) // 2],
            hp={**hp_disc, 'sigma2': 'auto'}, degree=2,
            n_iter=args.n_iter, n_restarts=args.n_restarts, verbose=False)
        models['local_edmd_disc_d2'] = s_le2
    if args.duffing_edmd_degrees:
        models['edmd_disc'] = fit_global_disc(
            X_tr_curr, X_tr_next, degree=args.duffing_edmd_degrees[0])
    models['X_tr'] = X_tr; models['F_tr'] = F_tr
    models['X_te'] = X_te; models['F_te'] = F_te
    models['X_tr_curr'] = X_tr_curr; models['X_tr_next'] = X_tr_next
    models['X_te_curr'] = X_te_curr; models['X_te_next'] = X_te_next

    return results, models


# ─────────────────────────────────────────────────────────────────────────────
# DUFFING dispatch
# ─────────────────────────────────────────────────────────────────────────────

if not args.skip_duffing:
    print("\n" + "=" * 80)
    print("  DUFFING SYSTEM")
    print(f"  distribution={args.duffing_distribution}  "
          f"params={args.duffing_distribution_params}")
    print(f"  n_train={args.duffing_n_train}  n_test={args.duffing_n_test}  "
          f"dt={args.duffing_dt}  rollout_steps={args.duffing_rollout_steps}  "
          f"horizons={args.duffing_horizons}")
    print(f"  fit: n_iter={args.n_iter} n_restarts={args.n_restarts}")
    print("=" * 80)

    duffing_runs, duffing_models, duffing_best_seed = run_system(
        "Duffing", run_duffing_seed)
    duffing_metrics = ['one_step'] + [_horizon_key(h) for h in args.duffing_horizons]
    report("DUFFING", duffing_runs, duffing_metrics)

    # The headline rollout horizon for paired tests; pick the largest configured
    headline_h_key = _horizon_key(args.duffing_horizons[-1]) if args.duffing_horizons else 'one_step'

    duffing_pairs = []
    for N in args.duffing_N:
        duffing_pairs.append(
            (f"Taylor-analytic N={N}", f"GMM-baseline N={N}", 'one_step',
             f"Taylor-ana vs GMM at N={N}"))
        duffing_pairs.append(
            (f"Taylor-analytic N={N}", f"GMM-baseline N={N}", headline_h_key,
             f"Taylor-ana vs GMM at N={N} (rollout {args.duffing_horizons[-1]}s)"))
    if args.duffing_edmd_degrees:
        deg0 = args.duffing_edmd_degrees[0]
        duffing_pairs.append(
            (f"Taylor-analytic N={args.duffing_N[-1]}",
             f"EDMD-disc deg={deg0}", headline_h_key,
             f"Taylor N={args.duffing_N[-1]} vs EDMD-disc deg={deg0} "
             f"(rollout {args.duffing_horizons[-1]}s)"))
    if args.duffing_le2_N and args.duffing_edmd_degrees:
        deg0 = args.duffing_edmd_degrees[0]
        duffing_pairs.append(
            (f"Local-EDMD-disc d2 N={args.duffing_le2_N[-1]}",
             f"EDMD-disc deg={deg0}", 'one_step',
             f"Local-EDMD-disc d2 N={args.duffing_le2_N[-1]} vs EDMD-disc deg={deg0}"))
    paired_tests("Duffing", duffing_runs, duffing_pairs)

    _json, _fig, _models = _out_paths("statistical_duffing")
    with open(_json, "w") as fp:
        json.dump({'n_seeds': N_SEEDS, 'seeds': seeds, 'args': vars(args),
                   'config_name': CONFIG_NAME, 'config_description': CONFIG_DESC,
                   'best_seed': duffing_best_seed,
                   'system_parameters': {'DELTA': DUFFING_DELTA,
                                         'ALPHA': -1.0, 'BETA': 1.0},
                   'results': duffing_runs}, fp, indent=2)
    print(f"\n  Saved: {_json}")

    if not args.no_save_models:
        torch.save(duffing_models, _models)
        print(f"  Saved models (best seed {duffing_best_seed}): {_models}")

print("\n" + "=" * 80)
print("  Done.")
print("=" * 80)
