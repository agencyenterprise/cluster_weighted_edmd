# Residual-Aware Bayesian Clustering for Local Dynamical Models

Code and configs for **Cluster-Weighted EDMD (CW-EDMD)** — a framework that
partitions a dynamical system's phase space into regions where a simple local
Koopman operator accurately predicts the next state. Clusters form by joint
likelihood of geometric proximity AND prediction accuracy, not geometry alone.

> **Paper:** `papers/scml2026/paper.pdf` (Tomaz, Rosenblatt, Kicis, Jones,
> Schwerz de Lucena — AE Studio). See `papers/scml2026/relavant_papers/` for
> local copies of every cited work.

---

## TL;DR

CW-EDMD fits a separate Koopman operator per cluster via EM on a
cluster-weighted-model joint density. Each cluster's responsibility for a
training transition combines two factors:

```
r_{ig} ∝ π_g · N(x_t; c_g, Σ_g) · N(Δx_g; 0, σ²_g I)
         └── proximity ─┘   └── prediction accuracy ──┘
```

The residual factor is what distinguishes CW-EDMD from a standard
geometry-only Gaussian mixture: a cluster earns responsibility only if it both
*lives near* the data and *predicts it well*.

The paper shows that on three classical systems (Lorenz, damped pendulum,
Duffing) and 36 configurations × 10 seeds, CW-EDMD outperforms global EDMD
at the matched polynomial lift, including in regimes where global EDMD
saturates near machine precision.

---

## Local-model variants

CW-EDMD instantiates a general "per-cluster predictor" plug-in. The repo
provides five variants — all share the same E-step, differ only in the M-step:

| Variant                            | Module                                | Local model                          | Headline use      |
|------------------------------------|---------------------------------------|--------------------------------------|-------------------|
| **CW-EDMD (discrete)** *(paper)*   | `models/em_local_edmd_discrete.py`    | per-cluster discrete Koopman matrix  | Lorenz / Duffing  |
| CW-EDMD (continuous)               | `models/em_local_edmd.py`             | per-cluster continuous Koopman gen.  | ablation          |
| CW-Taylor (analytic)               | `models/em.py`                        | `f(c_k) + J(c_k)(x − c_k)`           | pendulum          |
| CW-Taylor (LS-fit)                 | `models/em_hybrid.py`                 | `f_k + J_k(x − c_k)`, LS per cluster | ablation          |
| Global EDMD (baseline)             | `models/em_local_edmd_discrete.py`*   | single global Koopman operator       | apples-to-apples  |
| GPU variants                       | `models/*_gpu.py`                     | same as above, PyTorch GPU/MPS       | large sweeps      |

\* `fit_global_edmd_discrete` in the same module.

The paper also reports two ablation baselines (GMM-EDMD, GMM-Taylor) — the
same M-step but with the residual factor disabled in the E-step. See the
paper's Appendix D for the within-EDMD ablation table.

---

## Project structure

```
residual_aware_clustering/
├── simulators/                    # Dynamical systems
│   ├── lorenz.py                    # d=3, polynomial RHS
│   ├── pendulum.py                  # d=2, non-polynomial (sin θ)
│   └── duffing.py                   # d=2, polynomial, two basins
│
├── models/                        # EM framework + variants
│   ├── em.py                        # CW-Taylor (analytic J)
│   ├── em_hybrid.py                 # CW-Taylor (LS-fit)
│   ├── em_local_edmd.py             # CW-EDMD (continuous-time)
│   ├── em_local_edmd_discrete.py    # CW-EDMD (discrete) ← paper headline
│   ├── em_local_edmd_discrete_gpu.py# Same on PyTorch device
│   ├── distributions.py             # Stable log-densities (CPU)
│   ├── distributions_gpu.py         # Stable log-densities (GPU)
│   ├── elbo.py                      # ELBO computation + monotonicity check
│   ├── marginal_likelihood.py       # Exact NIW marginal (model selection)
│   ├── observables.py               # Polynomial lift Φ
│   └── global_edmd.py               # Global EDMD baseline
│
├── validation/                    # Experiments and runners
│   ├── run_statistical.py           # Single-config statistical driver
│   ├── validation_{lorenz,pendulum,duffing}_statistical.py
│   ├── validation_*.py              # Per-system one-off scripts
│   ├── analyze_results.py           # Aggregate JSONs into long-form CSV
│   └── generate_figures.py          # Paper figures from CSV
│
├── config/                        # YAML configs (one per experiment)
│   ├── lorenz/{baseline,dt_fast,attractor_*,...}.yaml
│   ├── pendulum/...yaml
│   ├── duffing/...yaml
│   └── smoke/                       # Sub-minute smoke tests
│
├── utils/                         # paths, viz, statistical helpers
├── patches/                       # pykoopman/sklearn compat patch
├── papers/                        # paper, figures/, data/, scml2026/
│   ├── scml2026/                    # SCML 2026 submission package
│   │   ├── paper.tex, paper.pdf
│   │   ├── references.bib
│   │   └── relavant_papers/         # local copies of every citation
│   └── data/, figures/, analysis/   # generated outputs (git-ignored)
│
├── run_all.sh                     # Full paper-grade suite (36 configs)
├── run_{lorenz,pendulum,duffing}.sh
├── run_all_smoke.sh
├── run_matched_degree.sh
├── setup.sh
├── pyproject.toml, requirements.txt
└── README.md (this file)
```

---

## Setup

The repo is a standard Python package. Python ≥ 3.10. Recommended toolchain
is [uv](https://github.com/astral-sh/uv) but plain `pip` works.

```bash
# Option 1: uv (recommended)
uv venv
uv pip install -e .
uv pip install -e ".[pykoopman]"   # optional: pykoopman/pydmd baselines

# Option 2: pip
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e ".[pykoopman]"

# Option 3: scripted (uses pip)
./setup.sh
```

The package import name is `residual_aware_clustering`. After install, all
`run_*.sh` scripts and `python -m validation.*` commands work from the repo
root.

**GPU (optional):** the `*_discrete_gpu` variants run on any PyTorch device
(CUDA, MPS, CPU). Device is inferred from the input tensors — pass
`X.to('cuda')` or `X.to('mps')` and use `make_hp_gpu` in place of `make_hp`.

**pykoopman patch:** if `pykoopman` is installed, the package auto-patches a
small sklearn-compat issue on first import. No action needed.

---

## Reproducing the paper

The paper's claims rest on a YAML-config-driven statistical suite — 12
configurations per system × 3 systems × 10 seeds × multiple methods.

**Full suite (~hours, paper-grade):**
```bash
./run_all.sh
# Outputs:
#   papers/data/<RUN_ID>/<config_name>.json    (per-seed metrics)
#   papers/figures/<RUN_ID>/<config_name>.png  (per-config plots)
#
# Then aggregate:
python -m validation.analyze_results \
  --data-dir papers/data/<RUN_ID> \
  --out-dir  papers/analysis/<RUN_ID>
# Aggregates 36 JSONs into a long-form CSV + the paper's headline tables.
```

**Per-system (~30 min each):**
```bash
./run_lorenz.sh
./run_pendulum.sh
./run_duffing.sh
```

**Single config (minutes):**
```bash
python -m validation.run_statistical \
  --config config/lorenz/attractor_baseline.yaml
```

**Speed knobs** (env vars consumed by `run_all.sh`):
```bash
MAX_WORKERS=8 N_ITER_CAP=40 N_RESTARTS_CAP=2 \
  N_TRAIN_CAP=4000 ROLLOUT_STEPS_CAP=200 \
  ./run_all.sh lorenz
```

**Smoke test (~1 min, sanity only):**
```bash
./run_all_smoke.sh
```

### What's in a config file

Each YAML config in `config/<system>/` declares a single experiment cell:

```yaml
name: lorenz_attractor_baseline
system: lorenz
seeds: [1, 42, 101, 307, 1001, 7789, 13245, 11, 103, 13]

data:
  distribution: attractor       # or uniform / gaussian / gmm / periodic_noise
  n_train: 4000
  dt: 0.01

fit:     { n_iter: 100, n_restarts: 2 }
rollout: { dt: 0.01, rollout_steps: 500, horizons: [0.5, 1.0, 2.0, 5.0] }

methods:
  N_list:    [5, 12, 20, 50]   # cluster counts to sweep
  edmd_degs: [2, 3]            # global EDMD lift degrees
  le_degrees: [2, 3]           # CW-EDMD lift degrees
```

The 36-config paper sweep is one baseline + 11 single-axis variations per
system (vary data size, dt, sampling distribution, fit budget, etc.).

---

## Using CW-EDMD on your own system

Define a simulator (any function `f(x) → \dot{x}` for analytic variants,
or sampled `(x_t, x_{t+1})` pairs for EDMD variants), then call one fitter.

**Discrete CW-EDMD (no analytic derivatives needed):**
```python
import torch
from residual_aware_clustering import fit_local_edmd_discrete, make_hp

# X: (P, d) current states, Y: (P, d) next states
X = torch.tensor(my_data["X"], dtype=torch.float64)
Y = torch.tensor(my_data["Y"], dtype=torch.float64)

hp = make_hp(X, d=X.shape[1])
state, elbos, labels = fit_local_edmd_discrete(
    X, Y, N=10, hp=hp, degree=2, n_iter=100,
)

# state['centers']   : (N, d)
# state['K']         : (N, M, M)   per-cluster Koopman operators on lifted basis
# state['Sigma']     : (N, d, d)   per-cluster covariances
# state['sigma2']    : (N,)        per-cluster residual variances
# state['pi']        : (N,)        mixture weights
```

**Taylor-analytic CW (requires analytic `f`, `J`):**
```python
from residual_aware_clustering import fit_taylor, make_hp
from residual_aware_clustering.simulators.lorenz import generate_data, f, J

data = generate_data(n_steps=5000, dt=0.01, warmup=1000)
X = torch.tensor(data["X"], dtype=torch.float64)
F = torch.tensor(data["F"], dtype=torch.float64)

hp = make_hp(X, d=3)
state, responsibilities, elbo_history = fit_taylor(X, F, f, J, N=5, hp=hp)
```

**GPU:** swap `fit_local_edmd_discrete` → `fit_local_edmd_discrete_gpu`,
`make_hp` → `make_hp_gpu`, and move tensors to a CUDA/MPS device.

Use `simulators/lorenz.py` as a template for adding your own system.

---

## Citation

```bibtex
@unpublished{tomaz2026cwedmd,
  author = {Tomaz, Lorenzo and Rosenblatt, Judd and Kicis, Flavio and
            Jones, Thomas B. and Schwerz de Lucena, Diogo},
  title  = {Cluster-Weighted {EDMD}},
  note   = {Manuscript, AE Studio. See papers/scml2026/paper.pdf in this repo.},
  year   = {2026}
}
```

---

## License

MIT. See `pyproject.toml`.
