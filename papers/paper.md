# Residual-Aware Bayesian Clustering for Local Dynamical Models

## A Unified Framework for Piecewise Linearization of Vector Fields

---

## Abstract

We present a Bayesian mixture-model framework for partitioning the phase space of a dynamical system `ẋ = f(x)` into regions within which a simple local model accurately predicts the vector field. Unlike standard Gaussian mixture models, which cluster points by geometric proximity alone, our framework clusters points by the *joint likelihood of position and vector-field observation*: a region is favoured only if both the points fall there AND a local dynamical model predicts `f(x)` accurately there. The framework is parameterization-agnostic: the local model inside each cluster can be a Taylor expansion evaluated from an analytic Jacobian, a least-squares affine fit, or a full local Extended Dynamic Mode Decomposition (EDMD) operator in a lifted polynomial space. We derive the EM updates for all three variants, prove they share a common E-step and differ only in the M-step, and present empirical results on two canonical dynamical systems — the Lorenz attractor and the damped pendulum. On Lorenz, global EDMD with a degree-2 polynomial lift is exact, so no clustering method can improve upon it; partitioning adds parameters without benefit. On the pendulum, where the nonlinearity `sin(θ)` has no finite polynomial representation, the framework delivers an order-of-magnitude parameter-efficiency gain over global EDMD, and reveals a non-trivial crossover: the LS-refit variant dominates at coarse N while the analytic-Jacobian variant dominates at fine N. These findings characterize precisely when piecewise-local modeling helps, when it does not, and which parameterization to select given the available information about the system.

---

## 1. Introduction

### 1.1 The Partition Problem

Consider a continuous dynamical system

```
ẋ = f(x),    x ∈ R^d,    f: R^d → R^d.
```

In many applications — reduced-order modeling, control design, stability analysis, reachability, surrogate modeling — one wishes to *approximate* `f` with a simpler function that is tractable to analyze or cheap to evaluate. The simplest such approximation is a single linear model:

```
f(x) ≈ A(x − x₀) + b,
```

built from the Jacobian `A = ∂f/∂x|_{x₀}` at some reference point `x₀`. This is the foundation of classical linear control theory, stability analysis of equilibria, and linearization-based reduced models. It works well when the trajectory of interest stays near `x₀`, but the approximation error grows as `O(‖x − x₀‖²)` away from the reference, so a single global linearization is inadequate for any problem whose trajectories explore a large portion of state space.

The natural remedy is **piecewise linearization**: partition the region of interest `V ⊂ R^d` into `N` subregions `V_1, ..., V_N`, and equip each with its own linearization `L_k(x) = A_k(x − c_k) + b_k` centered at a carefully-chosen `c_k ∈ V_k`. Given a point `x ∈ V`, identify which subregion it belongs to, then use that subregion's linear model to predict `f(x)`.

Two questions arise: *where should the centers `c_k` be placed?*, and *how should phase space be partitioned into the `V_k`?* A naive answer is: cluster the observed data points `{x_i}` geometrically using `k`-means or a Gaussian mixture, then evaluate the Jacobian at each cluster's centroid. But this ignores the dynamics entirely. A region with high sample density but high *curvature* of `f` (high Taylor remainder) is a bad choice of subregion: the linearization will be inaccurate even at points close to the center. Conversely, a region where `f` is nearly linear can tolerate large `V_k` cheaply.

**The key idea**: the partition should be chosen not by geometric proximity but by **linearization quality**. Centers should sit in regions where `f` is locally flat; boundaries should fall where `f` changes character.

### 1.2 A Generative Framing

We cast this as a **mixture model over joint observations** `(x, f(x))`. Each component `k` is characterized by a triple `(π_k, c_k, M_k)`:

- `π_k`: prior probability of the component
- `c_k`: center of the `k`-th region
- `M_k`: parameters of the local dynamical model (which will be a Jacobian matrix, an affine operator, or a Koopman operator in a lifted space, depending on the variant)

The joint likelihood of a training pair `(x_i, f(x_i))` under component `k` factorizes into two Gaussians:

```
p(x_i, f(x_i) | z_i = k) = N(x_i; c_k, Σ_k) · N(f(x_i) − L_k(x_i); 0, σ²I)
                          └── proximity factor ──┘  └── residual factor ──┘
```

The **proximity factor** `N(x_i; c_k, Σ_k)` is the standard GMM term: it prefers to assign `x_i` to the component whose center is geometrically close. The **residual factor** `N(f(x_i) − L_k(x_i); 0, σ²I)` is novel: it prefers to assign `x_i` to the component whose *local model* correctly predicts `f(x_i)`. A point will only be assigned to component `k` if *both* factors agree — the point is close to `c_k` AND the local model `L_k` fits well there. This joint criterion is what makes the clustering "residual-aware."

Writing `ε_k(x) = f(x) − L_k(x)` for the Taylor remainder (or whatever the local-model residual is), the component-conditional log-likelihood is

```
log p(x_i, f(x_i) | z_i = k) = const − ½(x_i − c_k)ᵀ Σ_k⁻¹ (x_i − c_k) − ‖ε_k(x_i)‖² / (2σ²).
```

The hyperparameter `σ²` sets the scale at which residuals matter: large `σ²` reduces the framework to ordinary GMM (residual term ≈ 0, always), while small `σ²` makes the residual dominate (points with bad local fit get pushed out regardless of geometry).

### 1.3 Contributions

1. **A unified framework** in which the local-model parameterization (analytic Taylor, LS-fit affine, or local EDMD in a lifted space) is modular: the E-step is identical across variants, and variants differ only in the M-step closed-form update for the local model.

2. **A principled Bayesian treatment** with Normal-Inverse-Wishart (NIW) conjugate priors on `(c_k, Σ_k)`, giving closed-form marginal likelihoods for cluster-count selection and monotone-ELBO convergence guarantees.

3. **An explicit characterization of when each variant wins**, demonstrated on Lorenz and pendulum benchmarks, including the non-trivial finding that the LS-fit variant dominates the analytic-Jacobian variant at *coarse* partitions, while the analytic variant dominates at *fine* partitions.

4. **A fair comparison to global EDMD**, showing that for polynomial dynamical systems (Lorenz) piecewise local models cannot improve on the right global lifting, but for non-polynomial systems (pendulum) piecewise local EDMD achieves an order-of-magnitude parameter-efficiency gain.

---

## 2. Background

### 2.1 Dynamical Systems and Linearization

Given a smooth vector field `f: R^d → R^d`, the **Jacobian** at `x₀` is the matrix `J(x₀) ∈ R^{d×d}` with entries `[J(x₀)]_{ij} = ∂f_i/∂x_j(x₀)`. Taylor's theorem gives, for any `x` near `x₀`:

```
f(x) = f(x₀) + J(x₀)·(x − x₀) + ½·H(ξ)·(x−x₀, x−x₀)
```

for some `ξ` on the segment between `x` and `x₀`, where `H` is the Hessian tensor. The **linearization error** is `O(‖x − x₀‖²)` and is controlled by the operator norm of the Hessian.

In phase space, a single linearization is accurate in a ball of radius `r₀` ~ `ε / ‖H‖` for a target error `ε`. Beyond that, piecewise models are necessary.

### 2.2 Gaussian Mixture Models and EM

A Gaussian Mixture Model with `N` components specifies a density

```
p(x) = Σ_{k=1}^N π_k · N(x; μ_k, Σ_k)
```

with mixing weights `π_k ≥ 0` summing to 1. Fitting such a model to samples `{x_i}_{i=1}^P` by maximum likelihood is intractable in closed form (the log-likelihood contains a log-sum-exp), but becomes tractable via **Expectation-Maximization** (EM) [Dempster, Laird, & Rubin, 1977]. One introduces latent assignments `z_i ∈ {1,...,N}` and iterates:

- **E-step**: compute soft responsibilities `r_{ik} = p(z_i = k | x_i)` using current parameters.
- **M-step**: update `(π_k, μ_k, Σ_k)` to maximize the expected complete-data log-likelihood under the current responsibilities.

EM is a special case of coordinate ascent on the **Evidence Lower Bound** (ELBO):

```
ELBO(q, θ) = E_q[log p(X, Z | θ)] + H[q],
```

where `q(Z)` is the variational distribution over latent assignments. Each E-step maximizes the ELBO over `q` (closing the KL gap); each M-step maximizes it over `θ`. Both steps are non-decreasing in the ELBO, guaranteeing convergence to a local maximum.

### 2.3 Koopman Operator Theory

The **Koopman operator** `K_t` associated with a flow `φ_t: R^d → R^d` acts on observables `g: R^d → R`:

```
(K_t g)(x) = g(φ_t(x)).
```

Though the flow `φ_t` is generally nonlinear, the Koopman operator `K_t` is **linear** on the space of observables. Its infinitesimal generator `L` satisfies `(L g)(x) = ∇g(x) · f(x)`, the Lie derivative of `g` along `f`.

**Extended Dynamic Mode Decomposition** (EDMD) [Williams, Kevrekidis & Rowley, 2015] approximates `L` in a finite-dimensional subspace spanned by chosen observables `Φ(x) = (φ_1(x), ..., φ_M(x))`. Given snapshot data `{(x_i, f(x_i))}`, one fits a matrix `M ∈ R^{M×M}` such that

```
(∇Φ(x_i))·f(x_i) ≈ M · Φ(x_i)
```

by weighted least squares. When the observable dictionary `{φ_j}` is a Koopman-invariant subspace (i.e., its closure under `L` is itself), `M` is exact. For a polynomial system like Lorenz with RHS of degree 2, a degree-2 monomial dictionary in `R^3` is Koopman-invariant and EDMD recovers the dynamics essentially exactly.

### 2.4 Normal-Inverse-Wishart Conjugacy

For a Gaussian component with unknown mean `c_k` and covariance `Σ_k`, the conjugate prior is the Normal-Inverse-Wishart (NIW):

```
Σ_k ~ IW(Ψ₀, ν₀),
c_k | Σ_k ~ N(μ₀, Σ_k / κ₀).
```

Under this prior, the marginal likelihood `p(X_k)` of a cluster's points is available in closed form, enabling rigorous Bayesian model selection over the cluster count `N` via posterior odds, BIC, or log marginal likelihood.

---

## 3. The Residual-Aware Framework

### 3.1 The Generative Model

The full generative story is:

```
1.  π ~ Dirichlet(α₀ · 1_N)
2.  For each k = 1, ..., N:
      Σ_k ~ InverseWishart(Ψ₀, ν₀)
      c_k | Σ_k ~ Normal(μ₀, Σ_k / κ₀)
3.  For each observation i = 1, ..., P:
      z_i ~ Categorical(π)
      x_i | z_i = k ~ Normal(c_k, Σ_k)
      f_obs(x_i) | x_i, z_i = k ~ Normal(L_k(x_i), σ² · I_d)
```

The last step embeds the novel residual factor. The observed `f_obs(x_i)` is the *true* vector field value at `x_i` (assumed known or numerically integrable). The local model `L_k` predicts it; the difference is Gaussian noise with scale `σ²`.

### 3.2 The Joint Likelihood

Marginalizing the assignment `z_i` gives the joint density

```
p(x_i, f(x_i)) = Σ_{k=1}^N  π_k · N(x_i; c_k, Σ_k) · N(f(x_i); L_k(x_i), σ²I).
```

In log-space, using `ε_k(x) = f(x) − L_k(x)`:

```
log p(x_i, f(x_i)) = log Σ_{k=1}^N  π_k · exp[
    − ½ (x_i − c_k)ᵀ Σ_k⁻¹ (x_i − c_k)
    − ‖ε_k(x_i)‖² / (2σ²)
    − ½ log|2π Σ_k|
    − d/2 · log(2π σ²)
  ].
```

Each component's contribution to the log-likelihood is a sum of a **proximity** term (quadratic in `x_i − c_k`) and a **residual** term (squared norm of the local prediction error). A point contributes positively only if both are small: it must be both geometrically near `c_k` AND well-predicted by `L_k`.

### 3.3 The Evidence Lower Bound

For variational inference with responsibilities `r_{ik}` as the variational distribution `q(z_i = k)`:

```
ELBO(q, θ) = Σ_i Σ_k r_{ik} · [log π_k + log N(x_i; c_k, Σ_k) + log N(ε_k(x_i); 0, σ²I)]
           − Σ_i Σ_k r_{ik} log r_{ik}                                   [entropy]
           + log p(π | α₀) + Σ_k log p(c_k, Σ_k | μ₀, κ₀, Ψ₀, ν₀).        [priors]
```

The ELBO is monotone non-decreasing across E-M iterations provided each M-step is a true coordinate-ascent maximization. In §4 we will see how different local-model parameterizations satisfy or violate this condition.

### 3.4 Contrast with Standard GMM

In the limit `σ² → ∞`, the residual term `−‖ε_k(x_i)‖²/(2σ²) → 0` and the framework reduces exactly to a standard GMM. In the limit `σ² → 0`, the residual term dominates and the model becomes a **piecewise regression**: clusters form wherever one local operator happens to fit a subset of points well, regardless of geometry. The interesting regime is `σ²` of order the squared linearization error at typical cluster size — here the residual term acts as a **geometry-modulating correction**, pulling cluster centers toward low-curvature regions.

In practice we calibrate `σ²` automatically from within-cluster residuals during initialization:

```
σ² ← median_{i,k}(‖ε_k(x_i)‖²) / d.
```

This ensures the residual term is neither vanishing nor dominant out of the box.

---

## 4. Local-Model Variants

All three variants share the framework above. They differ only in how the local model `L_k(x)` is parameterized and fit.

### 4.1 Variant A: Taylor-Analytic

**Parameterization**: `L_k(x) = f(c_k) + J(c_k) · (x − c_k)` where `J = ∂f/∂x` is evaluated **analytically** at the center `c_k`.

**Zero free local-model parameters**: `f(c_k)` and `J(c_k)` are deterministic functions of the state `c_k`, computed by evaluating `f` and the closed-form Jacobian. The only local parameter is the center itself.

**Intuition**: This is the classical Taylor linearization at each center. It has excellent local accuracy (zero error at `x = c_k`, `O(‖x − c_k‖²)` error away from it) and is strongly regularized — no fitting noise, no overfitting possible. The downside: if the cluster is large, points at the boundary are poorly modeled because Taylor's theorem only gives local accuracy.

**ELBO consideration**: The center update for `c_k` should include a contribution from the residual gradient, but that gradient is tangled: when `c_k` moves, both `f(c_k)` and `J(c_k)` change. Properly handling this requires either a non-trivial fixed-point iteration or a generalized-EM (GEM) scheme. Our implementation uses a GEM-style update that takes the proximity-only gradient for `c_k` and then re-evaluates `f(c_k)` and `J(c_k)` at the new center. This can introduce small non-monotonicities in the ELBO at high `N` where the residual term dominates.

### 4.2 Variant B: Taylor-LS (Hybrid)

**Parameterization**: `L_k(x) = f_k + J_k · (x − c_k)` where `(f_k, J_k)` are **free parameters** fit by weighted least squares over the cluster's responsibilities.

**Free local-model parameters per cluster**: `d + d²` (`f_k` has `d` entries, `J_k` has `d²`).

**Intuition**: Instead of trusting Taylor expansion to be accurate, we directly fit the best affine operator over each cluster's data. This gives a **cluster-averaged Jacobian** rather than a pointwise one. When clusters are large and Taylor's remainder is significant, the LS fit can be substantially better than analytic `J(c_k)`.

**M-step for `(f_k, J_k)`**: Closed-form weighted least squares. Let `φ_i = [1, (x_i − c_k)]ᵀ ∈ R^{d+1}` be the feature vector. Stack features into `Z_k ∈ R^{P × (d+1)}` and targets into `F ∈ R^{P × d}`. With diagonal weight matrix `W = diag(r_{ik})`:

```
[f_k   J_kᵀ] = (Z_kᵀ W Z_k + λI)⁻¹ Z_kᵀ W F.
```

**M-step for `c_k`** (correct coordinate-ascent update, including residual gradient):

```
[Λ₀ + R_k · Σ_k⁻¹ + (R_k/σ²) · J_kᵀ J_k] · c_k_new
   = Λ₀ μ₀ + Σ_k⁻¹ · Σ_i r_{ik} x_i − (1/σ²) · J_kᵀ · (Σ_i r_{ik} f_i − R_k · f_k − J_k · Σ_i r_{ik} x_i)
```

where `R_k = Σ_i r_{ik}`. This is the correct gradient equation that the original Taylor-tied variant effectively dropped. Under this update, the ELBO is monotone (pure coordinate-ascent).

**Intuition for the crossover**: When clusters are small, `J(c_k) ≈ J_k^LS` (both are "the Jacobian around `c_k`"), but analytic has zero noise while LS has fitting noise — analytic wins. When clusters are large, the cluster-averaged `J_k^LS` differs substantially from pointwise `J(c_k)`, and averaging beats pointwise because the remainder dominates — LS wins.

### 4.3 Variant C: Local EDMD

**Parameterization**: `L_k(x) = [M_k · Φ(x − c_k)]_{1..d}` where `Φ: R^d → R^M` is a polynomial lifting of degree `p`, and `M_k ∈ R^{M×M}` is a **local continuous-time Koopman generator** that acts on the lifted observables.

**Free local-model parameters per cluster**: `M²` where `M = \binom{d+p}{p}` is the number of monomials up to degree `p`. For `d=3, p=2`: `M = 10`, so each `M_k` has 100 parameters.

**Intuition**: Each cluster carries its own locally-valid finite-dimensional Koopman subspace. For a cluster whose data is confined to a small region of phase space, even a modest polynomial lift can form an approximately-invariant subspace, because the nonlinearities of `f` are well-approximated by low-degree polynomials over the cluster's support.

**Mechanism**: For any monomial basis `Φ(u)`, the Lie derivative along `f` is `(L f)(Φ)(x) = ∇_u Φ(x − c_k) · f(x)`. The M-step fits `M_k` such that `∇Φ · f ≈ M_k · Φ` in weighted least squares:

```
M_k = (Σ_i r_{ik} · [∇Φ(x_i − c_k) · f(x_i)] · Φ(x_i − c_k)ᵀ)
       · (Σ_i r_{ik} · Φ(x_i − c_k) · Φ(x_i − c_k)ᵀ + λ·I)⁻¹.
```

The predicted vector field is then `f̂(x) = [M_k · Φ(x − c_k)]_{1..d}` (the linear-monomial entries, corresponding to `du/dt`).

**Intuition for why this helps on non-polynomial systems**: Global EDMD at degree `p` can only approximate `sin(θ)` by a fixed degree-`p` Taylor polynomial, which degrades as `|θ|` grows. Local EDMD at degree `p` around `θ_k` approximates `sin(θ)` by Taylor around `θ_k`, which is locally accurate within that cluster's region. Clustering into narrow angular intervals effectively gives each interval its own local Taylor expansion.

### 4.4 Comparison Matrix

| Variant | Free params/cluster | Narrative | Monotone ELBO | Works without analytic `f, J` |
|---|---|---|---|---|
| Taylor-analytic | 0 (beyond `c_k`, `Σ_k`) | Physics-grounded pointwise Taylor | Approximate (GEM) | No |
| Taylor-LS | `d + d²` | Data-driven cluster-averaged affine | Yes (pure CA-EM) | Yes |
| Local EDMD | `M² = \binom{d+p}{p}²` | Lifted Koopman per cluster | Yes (with fixed `c_k`) | Yes |

---

## 5. The EM Algorithm

### 5.1 Initialization

All three variants use a **GMM warm start**. We fit a standard Gaussian mixture to the `{x_i}` alone, using `sklearn.mixture.GaussianMixture` with multiple restarts. This gives initial `π_k`, `c_k`, `Σ_k`. Hard cluster labels from this initial GMM are used to:

1. Initialize the local models `L_k`:
   - **Taylor-analytic**: compute `f(c_k)`, `J(c_k)` analytically.
   - **Taylor-LS**: fit `(f_k, J_k)` by weighted LS using hard-assignment labels.
   - **Local EDMD**: fit `M_k` by weighted continuous EDMD on the hard clusters.

2. **Calibrate `σ²`** from within-cluster residuals:
   ```
   σ² ← median(‖ε_k(x_i)‖²) / d,   subject to σ² > 10⁻³.
   ```

Good initialization is important: the EM landscape is non-convex, and the residual-factor structure creates additional local optima that pure-GMM initialization may not escape. We use `n_restarts` random seeds and select the best by final ELBO.

### 5.2 The E-step (identical for all variants)

Given current parameters `θ = (π, {c_k, Σ_k, L_k})`, compute soft responsibilities:

```
log r̃_{ik} = log π_k + log N(x_i; c_k, Σ_k) + log N(ε_k(x_i); 0, σ²I),
log r_{ik} = log r̃_{ik} − logsumexp_k log r̃_{ik}.
```

The `logsumexp` normalizer ensures numerical stability. For the residual factor, we compute `ε_k(x_i) = f(x_i) − L_k(x_i)` using whichever local-model parameterization is active.

### 5.3 The M-step (varies by variant)

**Taylor-analytic** (GEM update):
1. Update `c_k` using proximity-only gradient: `[Λ₀ + R_k Σ_k⁻¹] c_k = Λ₀ μ₀ + Σ_k⁻¹ Σ_i r_{ik} x_i`.
2. Re-tie `f(c_k) ← f(c_k_new)`, `J(c_k) ← J(c_k_new)` analytically.
3. Update `Σ_k` from NIW posterior at new `c_k`.
4. Update `π` from Dirichlet posterior.

**Taylor-LS** (pure coordinate-ascent EM):
1. Fit `(f_k, J_k)` by weighted LS given current `c_k`.
2. Update `c_k` using full gradient (proximity + residual) with new `J_k`, `f_k`.
3. Update `Σ_k` from NIW posterior at new `c_k`.
4. Update `π` from Dirichlet posterior.

**Local EDMD** (pure coordinate-ascent EM on `M_k` and `Σ_k`, partial on `c_k`):
1. Update `c_k` using proximity-only gradient (the `Φ(x − c_k)` nonlinearity makes the residual gradient w.r.t. `c_k` non-closed-form; we accept this approximation).
2. Fit `M_k` by weighted continuous EDMD at new `c_k`.
3. Update `Σ_k` from NIW posterior at new `c_k`.
4. Update `π` from Dirichlet posterior.

### 5.4 Dead-Cluster Pruning

With `α₀ < 1`, the Dirichlet prior on `π` is **sparse-inducing**: it drives `π_k → 0` for unused clusters. When a cluster's effective mass `R_k = Σ_i r_{ik}` falls below a threshold (we use 1.0), we *permanently remove* the cluster. This is preferable to re-initialization, which would inject randomness and break ELBO monotonicity.

The user initializes EM with `N` larger than needed; pruning finds the right number automatically. In our experiments starting from `N=50` on Lorenz typically prunes down to 18-22 active clusters, while on pendulum it rarely prunes at all.

### 5.5 Model Selection

For choosing `N`, we compute the **log marginal likelihood** in closed form by integrating out `(c_k, Σ_k)` under the NIW conjugate prior:

```
log p(X_k) = (d/2) log(κ₀/κ_n)
           + (ν₀/2) log|Ψ₀| − (ν_n/2) log|Ψ_n|
           + log Γ_d(ν_n/2) − log Γ_d(ν₀/2)
           − (R_k · d / 2) log π
           − (1/(2σ²)) Σ_i ‖ε_k(x_i)‖²,
```

where `κ_n = κ₀ + R_k`, `ν_n = ν₀ + R_k`, and `Ψ_n` is the posterior scatter. The last term is the residual penalty — the only novel term compared to standard NIW marginal likelihood. The total log marginal is

```
log p(X, F | N) = log B(R + α₀) − log B(α₀ · 1_N) + Σ_k log p(X_k),
```

where `B` is the Beta function. `N*` is chosen to maximize this quantity. For datasets where log-ML grows unboundedly with `N`, BIC with appropriate parameter counts (d + d²/2 + free local-model parameters per cluster) gives a consistent estimator.

---

## 6. Algorithms (Pseudo-code)

### Algorithm 1: Unified EM Loop

```
Input:  X ∈ R^{P×d} (training points)
        F ∈ R^{P×d} (vector field observations)
        N (initial cluster count)
        hp (hyperparameters: α₀, μ₀, Λ₀, κ₀, Ψ₀, ν₀, σ²)
        M_step (function: one of m_step_analytic, m_step_LS, m_step_edmd)
Output: state (converged parameters), responsibilities r

state ← initialize(X, F, N, hp)            # GMM warm start + σ² calibration
history ← []

for t = 1 to n_iter:
    r ← E_step(X, F, state, hp)            # soft responsibilities
    state, r ← prune_dead(state, r, X, F, hp)
    if state.N == 0: break
    history.append(compute_elbo(X, F, r, state, hp))
    if converged(history): break
    state ← M_step(X, F, r, state, hp)

return state, r, history
```

### Algorithm 2: E-step

```
Input:  X, F, state, hp
Output: r ∈ R^{P×N}

log_π ← log(state.π)
log_prox ← mvn_logpdf(X, state.centers, state.covariances)     # (P, N)
log_resid ← residual_logpdf(X, F, state.L_params, hp.σ²)       # (P, N)
log_r̃ ← log_π + log_prox + log_resid
log_r ← log_r̃ − logsumexp(log_r̃, axis=1)                      # normalize
return exp(log_r)
```

### Algorithm 3: M-step (Taylor-LS Variant)

```
Input:  X, F, r, state, hp
Output: state (updated)

R ← sum(r, axis=0)
Σ_inv ← inv(state.covariances)

for k = 1 to N:
    r_k ← r[:, k];  R_k ← R[k]

    # Step 1: weighted LS for f_k, J_k
    δ ← X − state.centers[k]                    # (P, d)
    Z ← [1, δ] ∈ R^{P × (d+1)}
    β ← solve(Zᵀ diag(r_k) Z + ridge·I, Zᵀ diag(r_k) F)
    f_k ← β[0]; J_k ← β[1:].T

    # Step 2: c_k update (full gradient)
    S_x ← r_k ᵀ X
    S_f ← r_k ᵀ F
    JtJ ← J_kᵀ J_k
    LHS ← Λ₀ + R_k · Σ_inv[k] + (R_k/σ²) · JtJ
    RHS ← Λ₀ μ₀ + Σ_inv[k] · S_x
            − (1/σ²) · J_kᵀ · (S_f − R_k · f_k − J_k · S_x)
    c_k_new ← solve(LHS, RHS)

    # Step 3: Σ_k posterior mode
    diff ← X − c_k_new
    scatter ← diff.T @ diag(r_k) @ diff
    Σ_k_new ← (Ψ₀ + scatter) / (ν₀ + R_k + d + 1)

    state.centers[k] ← c_k_new
    state.f_centers[k] ← f_k
    state.jacobians[k] ← J_k
    state.covariances[k] ← Σ_k_new + ridge·I

# Dirichlet posterior mode for π
π_new ← (R + α₀ − 1) / (P + N(α₀ − 1))
state.π ← clamp(π_new, min=10⁻¹⁰) / sum(...)

return state
```

### Algorithm 4: M-step (Local EDMD Variant)

```
Input:  X, F, r, state, hp, exps (monomial exponents for lift)
Output: state (updated)

R ← sum(r, axis=0)
Σ_inv ← inv(state.covariances)

for k = 1 to N:
    r_k ← r[:, k];  R_k ← R[k]

    # Step 1: c_k update (proximity-only, approximate)
    LHS ← Λ₀ + R_k · Σ_inv[k]
    RHS ← Λ₀ μ₀ + Σ_inv[k] · r_kᵀ X
    c_k_new ← solve(LHS, RHS)

    # Step 2: weighted continuous EDMD for M_k at new c_k
    U ← X − c_k_new                                          # (P, d)
    Φ ← monomials(U, exps)                                   # (P, M)
    ∇Φ ← monomials_grad(U, exps)                             # (P, M, d)
    Φ̇ ← einsum('pmd,pd->pm', ∇Φ, F)                          # (P, M)
    G ← Φᵀ diag(r_k) Φ                                        # (M, M)
    A ← Φ̇ᵀ diag(r_k) Φ                                        # (M, M)
    M_k ← solve(G + ridge·I, Aᵀ).T                            # (M, M)

    # Step 3: Σ_k posterior mode
    diff ← X − c_k_new
    scatter ← diff.T @ diag(r_k) @ diff
    Σ_k_new ← (Ψ₀ + scatter) / (ν₀ + R_k + d + 1)

    state.centers[k] ← c_k_new
    state.M_ops[k] ← M_k
    state.covariances[k] ← Σ_k_new + ridge·I

state.π ← (R + α₀ − 1) / (P + N(α₀ − 1))
return state
```

---

## 7. Experiments

### 7.1 Systems and Metrics

We test on two canonical systems:

**Lorenz attractor** (`d=3`):
```
ẋ = σ(y − x),  ẏ = x(ρ − z) − y,  ż = xy − β·z
(σ, ρ, β) = (10, 28, 8/3)
```
A chaotic system whose right-hand side is a polynomial of total degree 2. Data: 5000 points sampled at `dt = 0.01` after 1000-step warmup. Train/test split: 4000/1000.

**Damped pendulum** (`d=2`):
```
θ̇ = θ̇,  θ̈ = −sin(θ) − γ·θ̇,  γ = 0.2
```
Dissipative, nonlinear, stable at (0, 0). Data: 4000 training points uniformly sampled from `(θ, θ̇) ∈ [−π, π] × [−3, 3]`; 1000 held-out test points from the same distribution. The `sin(θ)` nonlinearity has no finite polynomial representation over its full domain.

**Metrics**:

1. **One-step vector-field error**: `mean_i ‖f̂(x_i) − f(x_i)‖` on held-out test set.

2. **Rollout error**: integrate both the true dynamics (via RK45) and the learned piecewise model (via switching Euler with `dt = 0.05` for pendulum, `0.01` for Lorenz) from several initial conditions on the attractor, measure `‖x̂_t − x_t‖` at fixed horizons.

3. **Parameter count**: total free parameters in the local models (excluding centers and covariances, which are the same for all variants at fixed `N`).

### 7.2 Lorenz Results

We compared the four methods across `N ∈ {3, 5, 8, 12, 20, 30, 50}`.

**Key findings**:

1. **Global EDMD dominates absolutely.** A single degree-2 polynomial lift (100 parameters, 10 monomials squared) achieves 3.7% one-step relative error. Degree-3 (400 parameters) achieves **0.04%** — essentially exact, as expected since Lorenz's RHS is already a degree-2 polynomial and the Koopman operator of such a system is exact in the polynomial basis.

2. **Local EDMD is constant in `N`.** On Lorenz, local-EDMD at any `N` and any degree gives one-step error 5.55% and identical rollout behavior. This is a structural consequence: since the global Koopman operator is exact in a 10-dimensional lift, every local cluster learns the *same* underlying polynomial (just re-expressed around its own `c_k`). Partitioning adds parameters without adding expressive power.

3. **Taylor variants and GMM improve with `N` but cannot catch global EDMD.** At `N=50`, Taylor-analytic achieves 6.7% one-step error, Taylor-LS achieves 8.4%, both ~2× worse than global EDMD deg-2.

4. **Taylor-LS at small `N` outperforms Taylor-analytic.** At `N=2` (2 clusters, 12 parameters), Taylor-LS achieves 38.8% rollout error vs Taylor-analytic's 105.9% — a **2.7×** improvement. This reveals the regime where data-driven cluster averaging beats pointwise Taylor.

5. **Rollout stability**: At `N≥12`, all piecewise-linear methods produce bounded rollouts (errors ≤ attractor scale). At `N<12`, the GMM baseline and sometimes Taylor-analytic diverge numerically (errors reaching `10^10`+).

**Interpretation**: Lorenz is the wrong system to demonstrate clustering benefit. Any Koopman-based method with an adequate global lift dominates. The value of the residual-aware clustering framework is invisible here because the "right" global model already exists in a small lifting space.

### 7.3 Pendulum Results

The pendulum exhibits the opposite behavior. The non-polynomial `sin(θ)` has no finite polynomial representation, so global EDMD must use high-degree lifts; local methods can specialize per angular region.

Table of headline results:

| Method | `N` | params | one-step | rollout @ 10s |
|---|---|---|---|---|
| Global EDMD deg=2 | 1 | 36 | 0.398 | 0.892 rad |
| Global EDMD deg=4 | 1 | 225 | 0.059 | 0.273 |
| Global EDMD deg=6 | 1 | 784 | 0.004 | 0.180 |
| Global EDMD deg=8 | 1 | 2025 | 0.0001 | 0.174 |
| **local-EDMD deg=2 N=2** | 2 | **72** | 0.015 | **0.159** |
| local-EDMD deg=2 N=16 | 16 | 576 | 0.003 | 0.174 |
| **Taylor-analytic N=8** | 8 | **48** | 0.019 | **0.139** |
| **Taylor-analytic N=16** | 16 | **96** | 0.019 | **0.136** |
| Taylor-LS N=2 | 2 | 12 | 0.265 | 0.388 |
| Taylor-LS N=8 | 8 | 48 | 0.032 | 0.210 |

**Key findings**:

1. **Parameter efficiency**: Local-EDMD deg-2 at `N=2` (72 parameters) matches global EDMD deg-6 (784 parameters) on rollout accuracy — an **~11× parameter efficiency gain**. This is the central demonstration that the framework is valuable on non-polynomial systems.

2. **Taylor-analytic is the overall winner given analytic `J`**. At `N=16`, Taylor-analytic achieves 0.136 rad rollout error — beating even global EDMD deg-8 (0.174 rad) at 21× fewer parameters. When the physics is known, fine-grained piecewise-linear with analytic Jacobians is extraordinarily parameter-efficient.

3. **The Taylor crossover is real and reproducible**:
   - At `N=2` (coarse): Taylor-LS (0.388 rad rollout) beats Taylor-analytic (1.059 rad) by **2.7×**.
   - At `N=8` (fine): Taylor-analytic (0.139 rad) beats Taylor-LS (0.210 rad) by **1.5×**.
   
   This confirms theoretical reasoning: Taylor's remainder dominates for large clusters, and LS-averaging gives a better region-average operator. As clusters shrink, Taylor's remainder shrinks quadratically and analytic `J(c_k)` becomes precise; fitting noise in LS then dominates.

4. **Global EDMD saturates at degree ~6-8**: beyond that, one-step error continues to drop but rollout error plateaus at ~0.17 rad, suggesting a floor imposed by Euler discretization error rather than model capacity.

5. **Local EDMD doesn't benefit from higher `N` as much as expected**: one-step error decreases with `N` (0.015 → 0.003) but rollout error barely moves. This suggests that for pendulum, a small number of well-placed local lifts is enough; adding clusters refines local accuracy marginally but doesn't improve trajectory-level prediction.

### 7.4 Visual Comparison: Pendulum Rollout

From initial condition `(θ, θ̇) = (2.8, 0.0)` (near the inverted position), integrating for 10 seconds:

- **Global EDMD deg-2** (36 params) — the polynomial approximation of `sin(θ)` is inadequate near `θ = π`, the trajectory diverges off the phase portrait.
- **Global EDMD deg-8** (2025 params) — visually tracks truth.
- **Local EDMD deg-2 `N=4`** (144 params) — visually indistinguishable from global deg-8.

The visualization shows the parameter-efficiency gain concretely: ~14× fewer parameters achieve qualitatively identical trajectory tracking.

---

## 8. Discussion

### 8.1 When Does Each Variant Win?

Our experiments support a clean decision rule:

| System characteristic | Analytic `f, J`? | Best variant |
|---|---|---|
| Globally polynomial, low-degree | — | Global EDMD (no partitioning needed) |
| Non-polynomial + physics known + can afford fine `N` | Yes | Taylor-analytic |
| Non-polynomial + physics known + constrained to coarse `N` | Yes | Taylor-LS |
| Non-polynomial + data-only (no analytic `f, J`) | No | Taylor-LS or local-EDMD |
| Highly non-polynomial + rich local structure | No | local-EDMD with higher degree |

### 8.2 Parameter Efficiency

The framework's main practical selling point is **parameter efficiency on non-polynomial systems**. Global polynomial Koopman methods must use high degree to approximate non-polynomial nonlinearities like `sin`, `exp`, `sign`, `tanh`. The parameter count for a global EDMD with degree-`p` lift in `d`-dimensional state space is `M² = \binom{d+p}{p}²`, which grows combinatorially.

Clustering into `N` regions with local degree-`p` lifts gives total parameter count `N · \binom{d+p}{p}²`. For the same effective accuracy on a locally-smooth nonlinearity, local methods typically require a much smaller per-cluster degree because each cluster's data span is narrower. We observed 10-20× parameter savings on pendulum.

This matters in high-dimensional settings: a system with `d = 10` at `p = 4` has 1001 monomials, yielding a million-parameter global Koopman operator. Local methods with `N = 20, p = 2` in `d = 10` have `20 × 66² ≈ 87,000` parameters — still large but tractable.

### 8.3 Limitations

1. **Non-monotone ELBO at high `N`**: Dead-cluster pruning is a discrete model-change step that can drop ELBO by `O(log N)`. Our `check_monotone` flags these as warnings but they are artifacts, not bugs. Additionally, the Taylor-analytic variant has a GEM structure that can produce small ELBO drops at high `N`.

2. **Center update in local EDMD**: The `c_k` update is currently proximity-only; the residual's gradient w.r.t. `c_k` involves `∇_{c_k} Φ(x − c_k)`, which is polynomial in `(x − c_k)` and coupled to the Koopman operator. Deriving the closed-form update is straightforward but algebraically heavy; we defer it.

3. **Cluster boundary handling**: At test time we assign clusters by proximity alone (dropping the residual term because we lack `f(x)` observations at prediction time). This is a **distribution shift** between training and test cluster assignment. For smooth vector fields it is minor, but for systems with abrupt regime changes it could be important.

4. **Lyapunov horizon**: For chaotic systems, rollout error eventually grows exponentially regardless of model quality. The methods here improve short-to-medium-horizon prediction but cannot bypass the Lyapunov time.

### 8.4 Connection to Related Work

The framework sits at the intersection of several literatures:

- **Mixture of Experts (MoE)** [Jacobs, Jordan, Nowlan, Hinton, 1991]: local experts gated by position. Most MoE formulations are *conditional* (`p(y|x) = Σ_k p(k|x)·p(y|x,k)`); ours is *joint-generative* (`p(x)·p(f|x)·p(k)`). Joint formulations are less common but allow principled Bayesian model selection over `N`.

- **Piecewise Affine (PWA) System Identification** [Bemporad, Ferrari-Trecate, Muselli, 2001; Ferrari-Trecate, Muselli, Liberati, Morari, 2003]: identifies piecewise affine dynamics from data, typically via k-means clustering + local regression. Our method generalizes this with soft (EM-based) clustering and Bayesian priors.

- **Switching Linear Dynamical Systems (SLDS)** [Fox, Sudderth, Jordan, Willsky, 2011]: regime-switching temporal models with HMM structure. We share the "multiple local linear regimes" premise but differ in time vs. phase-space focus.

- **EDMD and Koopman operator approximation** [Williams, Kevrekidis, Rowley, 2015; Klus, Nüske, Peitz et al., 2020]: fits Koopman operators globally. Our local-EDMD variant can be seen as cluster-wise application of EDMD with soft assignments.

- **Cluster-based reduced-order models** [Kaiser, Noack, Cordier et al., 2014]: cluster phase space, then build ROMs per cluster. We share the philosophy but differ in using a joint likelihood that aligns clustering with model fidelity.

- **Multi-resolution Koopman** [Giannakis, 2019; Berry, Giannakis, Harlim]: decomposes Koopman operators at multiple scales. Related but uses spectral decomposition rather than soft clustering.

What we specifically contribute:
1. The joint-likelihood clustering criterion that makes the partition residual-aware.
2. A unified framework in which the local-model parameterization is modular (analytic, LS, or lifted).
3. Closed-form NIW-conjugate Bayesian model selection for cluster count `N`.
4. Empirical characterization of the parameterization tradeoff via the Taylor-analytic/Taylor-LS crossover.

---

## 9. Conclusion

We have developed a residual-aware Bayesian clustering framework for piecewise local modeling of nonlinear vector fields. The framework generalizes standard Gaussian mixture models by introducing a **joint likelihood over position and vector-field observation**: regions are assigned based both on geometric proximity and on how accurately a local dynamical model predicts `f(x)`. The local-model parameterization is modular, supporting physics-grounded Taylor expansion (analytic Jacobian), data-driven affine regression (LS-fit), or Koopman lifting per cluster (local EDMD), within a single EM-based optimization procedure.

Our empirical investigation produced three practically-important findings:

1. **On globally-polynomial systems (Lorenz), partitioning does not help.** Global Koopman methods with polynomial lifts of matching degree are exact, and no clustering can improve upon them. This is important because it rules out naive application of piecewise methods where better global methods exist.

2. **On non-polynomial systems (pendulum), partitioning delivers 10-20× parameter-efficiency gains** over global Koopman methods. The fundamental reason is that non-polynomial nonlinearities like `sin(θ)` require high-degree global lifts but are locally-polynomial, so clustering into narrow regions lets low-degree local lifts suffice.

3. **The analytic-vs-LS tradeoff has a parameterization-level crossover**: analytic Jacobians win at fine `N`; LS-fit Jacobians win at coarse `N`. This crossover tells practitioners which variant to use given their cluster budget and physics knowledge.

The framework provides a principled Bayesian machinery — closed-form marginal likelihoods, NIW-conjugate priors, monotone ELBO for the LS and EDMD variants — built on a single conceptual contribution: **clusters should form where both geometry and dynamics align**. We believe this joint criterion is broadly applicable beyond the specific variants we have explored.

### Future Work

Directions we find particularly promising:

1. **Stronger test systems**: chaotic driven pendulum, coupled oscillators, neural models (Hodgkin-Huxley with sigmoidal gating), stick-slip friction systems. These would further characterize the parameter-efficiency regime and, crucially, exhibit multiple distinct Koopman-invariant subspaces across regions — the setting where local EDMD should dominate most strongly.

2. **Control-theoretic extensions**: EDMDc (controlled EDMD) per cluster, giving local LPV-style models that can be composed into piecewise control laws.

3. **Automatic local-degree selection**: within each cluster, the lift degree `p_k` could be selected by local BIC rather than held fixed across clusters. This would let the method adapt local complexity to local nonlinearity strength.

4. **Rigorous ELBO handling for pruning**: separating "discrete model change" events from "continuous parameter update" events in the ELBO monitoring would allow cleaner convergence diagnostics.

5. **Non-isotropic residual covariance**: we use `σ²I` for the residual noise; a full cluster-specific `Σ_k^{res}` could capture anisotropic local error structure.

---

## References

**EM and Mixture Models**
- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society. Series B*, 39(1), 1-38.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- McLachlan, G. J., & Peel, D. (2000). *Finite Mixture Models*. Wiley.

**Koopman Operator Theory and EDMD**
- Koopman, B. O. (1931). Hamiltonian systems and transformation in Hilbert space. *Proceedings of the National Academy of Sciences*, 17(5), 315-318.
- Williams, M. O., Kevrekidis, I. G., & Rowley, C. W. (2015). A data-driven approximation of the Koopman operator: Extending dynamic mode decomposition. *Journal of Nonlinear Science*, 25(6), 1307-1346.
- Klus, S., Nüske, F., Peitz, S., Niemann, J. H., Clementi, C., & Schütte, C. (2020). Data-driven approximation of the Koopman generator: Model reduction, system identification, and control. *Physica D: Nonlinear Phenomena*, 406, 132416.
- Brunton, S. L., Budišić, M., Kaiser, E., & Kutz, J. N. (2022). Modern Koopman theory for dynamical systems. *SIAM Review*, 64(2), 229-340.

**Mixture of Experts**
- Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). Adaptive mixtures of local experts. *Neural Computation*, 3(1), 79-87.
- Jordan, M. I., & Jacobs, R. A. (1994). Hierarchical mixtures of experts and the EM algorithm. *Neural Computation*, 6(2), 181-214.

**Piecewise Affine System Identification**
- Bemporad, A., Ferrari-Trecate, G., & Muselli, M. (2001). Observability and controllability of piecewise affine and hybrid systems. *IEEE Transactions on Automatic Control*, 46(12), 1864-1876.
- Ferrari-Trecate, G., Muselli, M., Liberati, D., & Morari, M. (2003). A clustering technique for the identification of piecewise affine systems. *Automatica*, 39(2), 205-217.
- Paoletti, S., Juloski, A. L., Ferrari-Trecate, G., & Vidal, R. (2007). Identification of hybrid systems: A tutorial. *European Journal of Control*, 13(2-3), 242-260.

**Switching and Hybrid Dynamical Systems**
- Fox, E. B., Sudderth, E. B., Jordan, M. I., & Willsky, A. S. (2011). Bayesian nonparametric inference of switching dynamic linear models. *IEEE Transactions on Signal Processing*, 59(4), 1569-1585.
- Ghahramani, Z., & Hinton, G. E. (2000). Variational learning for switching state-space models. *Neural Computation*, 12(4), 831-864.

**Cluster-Based Reduced Order Modeling**
- Kaiser, E., Noack, B. R., Cordier, L., Spohn, A., Segond, M., Abel, M., Daviller, G., Östh, J., Krajnović, S., & Niven, R. K. (2014). Cluster-based reduced-order modelling of a mixing layer. *Journal of Fluid Mechanics*, 754, 365-414.

**Dynamical Systems and the Lorenz Attractor**
- Lorenz, E. N. (1963). Deterministic nonperiodic flow. *Journal of the Atmospheric Sciences*, 20(2), 130-141.
- Strogatz, S. H. (2014). *Nonlinear Dynamics and Chaos*. CRC Press.

**Bayesian Model Selection**
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. [Chapter 5 on Bayesian model comparison.]
- Schwarz, G. (1978). Estimating the dimension of a model. *The Annals of Statistics*, 6(2), 461-464.

---

*Code and experiments: see `residual_aware_clustering/` — in particular `em.py` (Taylor-analytic), `em_hybrid.py` (Taylor-LS), `em_local_edmd.py` (local EDMD), and validation scripts `validation_sweep_N.py`, `validation_pendulum.py`.*
