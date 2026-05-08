"""
Shared phase-space sampling primitives.

Low-level utilities used by every simulator
(``simulators/duffing.py``, ``simulators/pendulum.py``, ``simulators/lorenz.py``)
to expose a unified set of input distributions for paper experiments:

- ``uniform_box``         -- uniform sampling on a (hyper)rectangle.
- ``gaussian``            -- multivariate Gaussian with diagonal covariance.
- ``gaussian_mixture``    -- weighted mixture of Gaussians.
- ``periodic_noise``      -- points sampled along a Lissajous-style
                              periodic curve plus additive Gaussian noise.
- ``trajectory_ensemble`` -- ensemble of short forward trajectories
                              integrated from random initial conditions
                              (yields non-uniform density that reflects
                              the system's flow, e.g., concentrated near
                              attractors).
- ``evaluate_field``      -- given an array of states ``X`` and a system's
                              ``f``, ``J`` callables, return the standard
                              ``{'X', 'F', 'J_all'}`` dict expected by the
                              EM fitters.

Every sampler returns *just* the state array ``X`` (shape ``(n, d)``); the
caller wraps it with ``evaluate_field`` to produce the final data dict.
This keeps the primitives system-agnostic and side-effect-free.
"""

import numpy as np
from scipy.integrate import solve_ivp


# -- State samplers (return X only) ------------------------------------------

def uniform_box(n_samples: int, box: np.ndarray, seed: int) -> np.ndarray:
    """Uniform sampling on the (hyper)rectangle ``[-box, +box]``.

    Parameters
    ----------
    n_samples : int
    box : array_like, shape ``(d,)``
        Half-widths per axis.
    seed : int

    Returns
    -------
    np.ndarray of shape ``(n_samples, d)``.
    """
    rng = np.random.default_rng(seed)
    box = np.asarray(box, dtype=np.float64)
    return rng.uniform(-box, +box, size=(n_samples, len(box)))


def gaussian(n_samples: int, mean: np.ndarray, sigma, seed: int) -> np.ndarray:
    """Multivariate Gaussian with diagonal covariance.

    Parameters
    ----------
    n_samples : int
    mean : array_like, shape ``(d,)``
    sigma : float or array_like, shape ``(d,)``
        Per-axis standard deviations (or scalar broadcast over all axes).
    seed : int

    Returns
    -------
    np.ndarray of shape ``(n_samples, d)``.
    """
    rng  = np.random.default_rng(seed)
    mean = np.asarray(mean, dtype=np.float64)
    d    = len(mean)
    sig  = np.asarray(np.full(d, sigma) if np.isscalar(sigma) else sigma,
                      dtype=np.float64)
    return rng.normal(loc=mean, scale=sig, size=(n_samples, d))


def gaussian_mixture(
    n_samples: int,
    centers:   np.ndarray,
    sigmas,
    weights:   np.ndarray = None,
    seed:      int = 42,
) -> np.ndarray:
    """Weighted mixture of diagonal-covariance Gaussians.

    Parameters
    ----------
    n_samples : int
    centers : array_like, shape ``(K, d)``
        Component means.
    sigmas : array_like, shape ``(K, d)`` or ``(K,)`` or scalar
        Per-component standard deviations (broadcast over axes if 1-D).
    weights : array_like or None
        Component weights (sum to 1). Equal weights if None.
    seed : int

    Returns
    -------
    np.ndarray of shape ``(n_samples, d)``.
    """
    rng     = np.random.default_rng(seed)
    centers = np.asarray(centers, dtype=np.float64)
    K, d    = centers.shape
    sigmas  = np.asarray(sigmas, dtype=np.float64)
    if sigmas.ndim == 0:
        sigmas = np.full((K, d), float(sigmas))
    elif sigmas.ndim == 1:
        sigmas = np.tile(sigmas[:, None], (1, d))
    if weights is None:
        weights = np.ones(K) / K
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()
    components = rng.choice(K, size=n_samples, p=weights)
    X = np.empty((n_samples, d), dtype=np.float64)
    for k in range(K):
        mask = components == k
        if mask.any():
            X[mask] = rng.normal(loc=centers[k], scale=sigmas[k],
                                 size=(int(mask.sum()), d))
    return X


def periodic_noise(
    n_samples:     int,
    amplitudes:    np.ndarray,
    frequency:     float,
    center:        np.ndarray = None,
    phase_offsets: np.ndarray = None,
    noise_std:     float = 0.05,
    seed:          int = 42,
) -> np.ndarray:
    """Sample states from a Lissajous-style periodic curve plus Gaussian noise.

    Each axis evolves as ``x_i(t) = center_i + amplitudes_i *
    cos(frequency * t + phase_offsets_i)`` with ``t`` drawn uniformly
    on ``[0, 2*pi)``; isotropic Gaussian noise of std ``noise_std`` is
    added.  With the default ``phase_offsets = [0, -pi/2, 0, ...]`` the
    first two axes trace a circle of radius ``amplitudes[0]`` (= the
    natural orbit topology for a 2-D oscillator near a focus).

    Parameters
    ----------
    n_samples : int
    amplitudes : array_like, shape ``(d,)``
        Per-axis amplitudes.
    frequency : float
        Common angular frequency.
    center : array_like, shape ``(d,)`` or None
        Orbit center (default zeros).
    phase_offsets : array_like, shape ``(d,)`` or None
        Per-axis phase offsets (default ``[0, -pi/2, 0, ...]``).
    noise_std : float
        Isotropic Gaussian noise std added to all axes.
    seed : int

    Returns
    -------
    np.ndarray of shape ``(n_samples, d)``.
    """
    rng        = np.random.default_rng(seed)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    d          = len(amplitudes)
    if center is None:
        center = np.zeros(d)
    else:
        center = np.asarray(center, dtype=np.float64)
    if phase_offsets is None:
        phase_offsets = np.zeros(d)
        if d >= 2:
            phase_offsets[1] = -np.pi / 2.0
    else:
        phase_offsets = np.asarray(phase_offsets, dtype=np.float64)

    t = rng.uniform(0.0, 2.0 * np.pi, n_samples)
    X = np.empty((n_samples, d), dtype=np.float64)
    for i in range(d):
        X[:, i] = center[i] + amplitudes[i] * np.cos(frequency * t + phase_offsets[i])
    X += rng.normal(0.0, noise_std, X.shape)
    return X


def trajectory_ensemble(
    n_traj:    int,
    n_steps:   int,
    dt:        float,
    f_fn,
    initial_condition_sampler,
    seed:      int = 42,
    rtol:      float = 1e-9,
    atol:      float = 1e-9,
) -> np.ndarray:
    """Sample states by integrating an ensemble of forward trajectories.

    Returns the concatenated trajectory points (shape
    ``((n_steps + 1) * n_traj, d)``).  Density is determined by the flow
    of ``f_fn``: dissipative systems accumulate near attractors,
    conservative systems sweep level sets uniformly, etc.

    Parameters
    ----------
    n_traj : int
    n_steps : int
        Integration steps per trajectory; ``n_steps + 1`` points are
        returned per trajectory (including the IC).
    dt : float
    f_fn : callable
        ``f_fn(state) -> dx/dt``, accepting and returning numpy arrays.
    initial_condition_sampler : callable
        ``ic(rng) -> state`` -- returns one IC each call.
    seed : int
    rtol, atol : float
        ``solve_ivp`` tolerances.

    Returns
    -------
    np.ndarray of shape ``((n_steps + 1) * n_traj, d)``.
    """
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(n_traj):
        x0 = initial_condition_sampler(rng)
        sol = solve_ivp(
            fun=lambda t, y: f_fn(y),
            t_span=(0.0, n_steps * dt),
            y0=np.asarray(x0, dtype=np.float64),
            method='RK45',
            t_eval=np.linspace(0.0, n_steps * dt, n_steps + 1),
            rtol=rtol, atol=atol,
        )
        pts.append(sol.y.T)
    return np.concatenate(pts, axis=0)


# -- Field evaluation --------------------------------------------------------

def evaluate_field(X: np.ndarray, f_fn, J_fn=None) -> dict:
    """Evaluate the vector field (and optionally the Jacobian) at every state.

    Parameters
    ----------
    X : np.ndarray, shape ``(n, d)``
    f_fn : callable
        ``f_fn(state) -> dx/dt``.
    J_fn : callable or None
        ``J_fn(state) -> (d, d) Jacobian``.  If None, ``J_all`` is omitted.

    Returns
    -------
    dict
        ``{'X': X, 'F': F, 'J_all': J_all}`` (``J_all`` only if ``J_fn`` given).
    """
    F = np.array([f_fn(x) for x in X])
    out = {'X': X, 'F': F}
    if J_fn is not None:
        out['J_all'] = np.array([J_fn(x) for x in X])
    return out
