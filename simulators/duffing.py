"""
Forced double-well Duffing oscillator simulator.

Implements the canonical Duffing equation in its double-well form,

    x_ddot + delta * x_dot - x + x**3 = gamma * cos(omega * t)

in two flavors that share the same constants and conventions:

- **Unforced (autonomous, 2D)**: ``gamma = 0``.  The phase plane has a
  saddle at the origin and stable foci at ``(+/- 1, 0)``.  The saddle's
  stable manifold is the basin boundary -- the cleanest visual demo of
  "linearizability contours not density contours" for the residual-aware
  clustering pipeline.
- **Forced (non-autonomous, 3D phase-augmented)**: ``gamma > 0``.  We
  augment the state with the forcing phase ``phi = omega * t`` so that
  the system becomes autonomous in ``(x, x_dot, phi)``.  At standard
  Holmes/Moon parameters ``(delta=0.25, gamma=0.3, omega=1.0)`` the system
  is chaotic with a fractal basin boundary.

This is the Mauroy-Mezic-Moehlis lineage testbed: residual-aware clusters
should approximate isostable / stable-manifold level sets, not density
blobs.

Provides the same core objects as the Lorenz and pendulum simulators so
the existing EM pipeline runs without modification:

- ``f(state)``           -- 2D unforced vector field, shape ``(2,)``.
- ``J(state)``           -- 2D unforced Jacobian, shape ``(2, 2)``.
- ``f_forced(state)``    -- 3D phase-augmented vector field, shape ``(3,)``.
- ``J_forced(state)``    -- 3D phase-augmented Jacobian, shape ``(3, 3)``.
- ``sample_phase_space(...)``     -- uniform 2D phase-space sampling.
- ``sample_phase_space_forced(...)`` -- uniform 3D phase-space sampling.
- ``generate_data(...)``          -- integrate unforced ODE, return X, F, J_all.
- ``generate_data_forced(...)``   -- integrate forced ODE, return X, F, J_all.
- ``generate_trajectory(...)``    -- integrate unforced ODE from x0.
- ``generate_trajectory_forced(...)`` -- integrate forced ODE from x0.
- ``classify_basin(...)``  -- ground-truth basin label by long integration
  (for plotting basin partitions and computing partition-vs-basin agreement).
- ``test_jacobian()``      -- finite-difference check against analytic J / J_forced.

Usage
-----
Generate phase-space training data for the unforced double-well::

    from residual_aware_clustering.simulators.duffing import sample_phase_space
    data = sample_phase_space(n_samples=4000, x_max=2.0, xdot_max=2.0, seed=42)

    # data['X'] -- (4000, 2) sampled (x, x_dot)
    # data['F'] -- (4000, 2) f(x_i) at each sample
    # data['J_all'] -- (4000, 2, 2) exact Jacobian at each sample

Generate phase-space training data for the forced/chaotic regime::

    from residual_aware_clustering.simulators.duffing import sample_phase_space_forced
    data = sample_phase_space_forced(n_samples=8000, x_max=2.0, xdot_max=2.0, seed=42)

    # data['X'] -- (8000, 3) sampled (x, x_dot, phi)
    # data['F'] -- (8000, 3) f_forced(x_i) at each sample

Compute ground-truth basin labels for the unforced system::

    import numpy as np
    from residual_aware_clustering.simulators.duffing import classify_basin

    grid = np.stack(np.meshgrid(np.linspace(-2, 2, 100),
                                np.linspace(-2, 2, 100)), axis=-1).reshape(-1, 2)
    basin = classify_basin(grid, t_max=200.0)
    # basin in {-1, 0, +1}: which well the trajectory settles into
    # (0 = unresolved within t_max)

Verify analytic Jacobians against finite differences::

    from residual_aware_clustering.simulators.duffing import test_jacobian
    test_jacobian()  # tests both J and J_forced

Key concepts
------------
- **Two-well topology**: the unforced system has stable foci at
  ``(+/- 1, 0)`` (well centers) and an index-1 saddle at the origin.
  The saddle's stable manifold separates the two basins.  Residual-aware
  EM should place cluster boundaries on this manifold; GMM will not.
- **Phase augmentation**: for the forced system, we lift to
  ``(x, x_dot, phi)`` with ``d phi / dt = omega`` so that the vector
  field is time-independent and ``J`` is a well-defined 3x3 matrix.
  Trajectories live on the cylinder ``R^2 x S^1`` (we do *not* wrap
  ``phi`` after integration -- downstream code can mod-2pi if needed).
- **Standard parameters**: ``DELTA = 0.25``, ``GAMMA = 0.3``,
  ``OMEGA = 1.0`` are the Holmes/Moon parameters that produce the
  classical chaotic strange attractor and fractal basin boundary in the
  forced case.
"""

import numpy as np
from scipy.integrate import solve_ivp

from . import _sampling

# -- System parameters --------------------------------------------------------

DELTA = 0.25   # damping
ALPHA = -1.0   # linear stiffness (negative -> double-well topology)
BETA  = 1.0    # cubic stiffness
GAMMA = 0.3    # forcing amplitude (used by *_forced variants)
OMEGA = 1.0    # forcing angular frequency (used by *_forced variants)


# -- Unforced (2D autonomous) -------------------------------------------------

def f(state: np.ndarray) -> np.ndarray:
    """Evaluate the unforced (autonomous) Duffing vector field.

    Parameters
    ----------
    state : np.ndarray
        State vector ``[x, x_dot]``, shape ``(2,)``.

    Returns
    -------
    np.ndarray
        Time derivatives ``[x_dot, x_ddot]``, shape ``(2,)``.
    """
    x, xdot = state
    return np.array([
        xdot,
        -DELTA * xdot - ALPHA * x - BETA * x ** 3,
    ])


def J(state: np.ndarray) -> np.ndarray:
    """Compute the exact Jacobian of the unforced Duffing vector field.

    Parameters
    ----------
    state : np.ndarray
        State vector ``[x, x_dot]``, shape ``(2,)``.

    Returns
    -------
    np.ndarray
        Jacobian matrix, shape ``(2, 2)``.
    """
    x, _ = state
    return np.array([
        [0.0,                          1.0   ],
        [-ALPHA - 3.0 * BETA * x ** 2, -DELTA],
    ])


# -- Forced (3D phase-augmented autonomous) -----------------------------------

def f_forced(state: np.ndarray) -> np.ndarray:
    """Evaluate the forced Duffing vector field in phase-augmented form.

    The state is lifted to ``(x, x_dot, phi)`` with ``d phi / dt = omega``
    so the system is autonomous.  ``phi`` is *not* wrapped here; mod-2pi
    if needed downstream.

    Parameters
    ----------
    state : np.ndarray
        State vector ``[x, x_dot, phi]``, shape ``(3,)``.

    Returns
    -------
    np.ndarray
        Time derivatives ``[x_dot, x_ddot, omega]``, shape ``(3,)``.
    """
    x, xdot, phi = state
    return np.array([
        xdot,
        -DELTA * xdot - ALPHA * x - BETA * x ** 3 + GAMMA * np.cos(phi),
        OMEGA,
    ])


def J_forced(state: np.ndarray) -> np.ndarray:
    """Compute the exact Jacobian of the phase-augmented forced Duffing field.

    Parameters
    ----------
    state : np.ndarray
        State vector ``[x, x_dot, phi]``, shape ``(3,)``.

    Returns
    -------
    np.ndarray
        Jacobian matrix, shape ``(3, 3)``.
    """
    x, _, phi = state
    return np.array([
        [0.0,                           1.0,    0.0                  ],
        [-ALPHA - 3.0 * BETA * x ** 2, -DELTA, -GAMMA * np.sin(phi)  ],
        [0.0,                           0.0,    0.0                  ],
    ])


# -- Phase-space sampling -----------------------------------------------------

def sample_phase_space(
    n_samples:   int   = 4000,
    x_max:       float = 2.0,
    xdot_max:    float = 2.0,
    seed:        int   = 42,
) -> dict:
    """Uniformly sample 2D phase points and evaluate the unforced field.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw.
    x_max : float
        Half-width of the position sampling range ``[-x_max, x_max]``.
        The default ``2.0`` covers both wells (centered at ``+/-1``)
        plus generous boundary regions.
    xdot_max : float
        Half-width of the velocity sampling range.
    seed : int
        Random seed.

    Returns
    -------
    dict
        'X' : np.ndarray, shape ``(n_samples, 2)`` -- sampled states.
        'F' : np.ndarray, shape ``(n_samples, 2)`` -- vector field values.
        'J_all' : np.ndarray, shape ``(n_samples, 2, 2)`` -- exact Jacobians.
    """
    rng    = np.random.default_rng(seed)
    x      = rng.uniform(-x_max,    x_max,    n_samples)
    xdot   = rng.uniform(-xdot_max, xdot_max, n_samples)
    X      = np.stack([x, xdot], axis=1)
    F_arr  = np.array([f(s) for s in X])
    J_all  = np.array([J(s) for s in X])
    return {'X': X, 'F': F_arr, 'J_all': J_all}


def sample_phase_space_forced(
    n_samples:   int   = 8000,
    x_max:       float = 2.0,
    xdot_max:    float = 2.0,
    seed:        int   = 42,
) -> dict:
    """Uniformly sample 3D phase points and evaluate the forced field.

    The phase ``phi`` is sampled uniformly on ``[0, 2 pi)``.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw.
    x_max : float
        Half-width of the position sampling range.
    xdot_max : float
        Half-width of the velocity sampling range.
    seed : int
        Random seed.

    Returns
    -------
    dict
        'X' : np.ndarray, shape ``(n_samples, 3)`` -- sampled ``(x, x_dot, phi)``.
        'F' : np.ndarray, shape ``(n_samples, 3)`` -- vector field values.
        'J_all' : np.ndarray, shape ``(n_samples, 3, 3)`` -- exact Jacobians.
    """
    rng    = np.random.default_rng(seed)
    x      = rng.uniform(-x_max,    x_max,    n_samples)
    xdot   = rng.uniform(-xdot_max, xdot_max, n_samples)
    phi    = rng.uniform(0.0, 2.0 * np.pi, n_samples)
    X      = np.stack([x, xdot, phi], axis=1)
    F_arr  = np.array([f_forced(s) for s in X])
    J_all  = np.array([J_forced(s) for s in X])
    return {'X': X, 'F': F_arr, 'J_all': J_all}


# -- Trajectory integration ---------------------------------------------------

def sample_gaussian(
    n_samples: int = 4000,
    mean:      np.ndarray = None,
    sigma                 = 1.0,
    seed:      int = 42,
) -> dict:
    """Sample Duffing phase points from a 2D Gaussian.

    Parameters
    ----------
    n_samples : int
    mean : array_like, shape ``(2,)`` or None
        Default ``(0, 0)`` (centered between the two foci).
    sigma : float or array_like, shape ``(2,)``
        Per-axis std (default 1.0).
    seed : int

    Returns
    -------
    dict with 'X', 'F', 'J_all'.
    """
    if mean is None:
        mean = np.zeros(2)
    X = _sampling.gaussian(n_samples, mean, sigma, seed)
    return _sampling.evaluate_field(X, f, J)


def sample_gaussian_mixture(
    n_samples: int = 4000,
    centers:   np.ndarray = None,
    sigmas                 = 0.4,
    weights:   np.ndarray = None,
    seed:      int = 42,
) -> dict:
    """Sample Duffing phase points from a Gaussian mixture.

    The default centers ``[(+1, 0), (-1, 0)]`` are the two stable foci
    -- the natural multimodal density on this system's basin structure.

    Parameters
    ----------
    n_samples : int
    centers : array_like, shape ``(K, 2)`` or None
        Default ``[(+1, 0), (-1, 0)]``.
    sigmas : float or array_like
        Per-component std (default 0.4).
    weights : array_like or None
        Component weights (default equal).
    seed : int

    Returns
    -------
    dict with 'X', 'F', 'J_all'.
    """
    if centers is None:
        centers = np.array([[+1.0, 0.0], [-1.0, 0.0]])
    X = _sampling.gaussian_mixture(n_samples, centers, sigmas, weights, seed)
    return _sampling.evaluate_field(X, f, J)


def sample_periodic_noise(
    n_samples:  int = 4000,
    amplitudes: np.ndarray = None,
    frequency:  float = None,
    center:     np.ndarray = None,
    noise_std:  float = 0.1,
    seed:       int   = 42,
) -> dict:
    """Sample Duffing phase points from a small-amplitude orbit + noise.

    Defaults trace the linearized small-amplitude oscillation around the
    right focus ``(+1, 0)`` with frequency ``sqrt(2)`` (the local natural
    frequency of the well).  This is a "physically meaningful" non-uniform
    distribution that emphasizes one basin's orbital structure.

    Parameters
    ----------
    n_samples : int
    amplitudes : array_like, shape ``(2,)`` or None
        Default ``(0.5, 0.5*sqrt(2))`` (proportional to natural frequency).
    frequency : float or None
        Default ``sqrt(2)`` (linearization eigenfrequency at the focus).
    center : array_like, shape ``(2,)`` or None
        Default ``(+1, 0)`` (the right focus).
    noise_std : float
        Isotropic Gaussian noise std (default 0.1).
    seed : int

    Returns
    -------
    dict with 'X', 'F', 'J_all'.
    """
    if frequency is None:
        frequency = float(np.sqrt(2.0))
    if amplitudes is None:
        amplitudes = np.array([0.5, 0.5 * frequency])
    if center is None:
        center = np.array([+1.0, 0.0])
    X = _sampling.periodic_noise(
        n_samples=n_samples, amplitudes=amplitudes, frequency=frequency,
        center=center, noise_std=noise_std, seed=seed,
    )
    return _sampling.evaluate_field(X, f, J)


def sample_trajectory_ensemble(
    n_traj:     int   = 200,
    n_steps:    int   = 80,
    dt:         float = 0.05,
    ic_x_max:   float = 2.0,
    ic_xdot_max:float = 2.5,
    seed:       int   = 42,
) -> dict:
    """Sample non-uniform phase-space data from an ensemble of short trajectories.

    Generates ``n_traj`` initial conditions uniformly on
    ``[-ic_x_max, ic_x_max] x [-ic_xdot_max, ic_xdot_max]``, integrates
    the unforced ODE forward for ``n_steps * dt`` time units from each,
    and returns *all* trajectory points as the training set.  This is
    the appropriate sampling for demonstrating "linearizability contours
    not density contours": damping concentrates density near the foci
    (lots of points settled at +/- 1, 0), while transient orbits trace
    paths through the saddle region that are sparser but dynamically
    informative.  GMM placed on this distribution will follow density
    (clusters at the foci); residual-aware methods can place clusters
    along basin-boundary structure where linearizability changes.

    Parameters
    ----------
    n_traj : int
        Number of trajectories.
    n_steps : int
        Number of integration steps per trajectory (the trajectory
        therefore contributes ``n_steps + 1`` points).
    dt : float
        Time step.
    ic_x_max, ic_xdot_max : float
        Half-widths of the initial-condition box.
    seed : int
        Random seed for the initial conditions.

    Returns
    -------
    dict
        'X' : np.ndarray, shape ``((n_steps + 1) * n_traj, 2)``.
        'F' : np.ndarray, same shape.
        'J_all' : np.ndarray, shape ``((n_steps + 1) * n_traj, 2, 2)``.
    """
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(n_traj):
        x0 = np.array([
            rng.uniform(-ic_x_max,    ic_x_max),
            rng.uniform(-ic_xdot_max, ic_xdot_max),
        ])
        sol = solve_ivp(
            fun=lambda t, y: f(y),
            t_span=(0.0, n_steps * dt),
            y0=x0,
            method='RK45',
            t_eval=np.linspace(0.0, n_steps * dt, n_steps + 1),
            rtol=1e-9, atol=1e-9,
        )
        pts.append(sol.y.T)
    X     = np.concatenate(pts, axis=0)
    F_arr = np.array([f(x) for x in X])
    J_all = np.array([J(x) for x in X])
    return {'X': X, 'F': F_arr, 'J_all': J_all}


def generate_data(
    n_steps: int        = 5000,
    dt:      float      = 0.01,
    x0:      np.ndarray = None,
    warmup:  int        = 500,
    seed:    int        = 42,
) -> dict:
    """Integrate the unforced Duffing system and return phase points.

    Note: with ``DELTA > 0`` (damping) the unforced system is dissipative
    and trajectories converge to one of the two stable foci, so a single
    long trajectory is *not* a good training source -- prefer
    ``sample_phase_space`` (uniform) or ``sample_trajectory_ensemble``
    (non-uniform, density concentrated near foci) for the unforced
    case.  This function is provided for API parity with the other
    simulators.

    Parameters
    ----------
    n_steps : int
        Number of post-warmup time steps to return.
    dt : float
        Integration time step.
    x0 : np.ndarray or None
        Initial condition, shape ``(2,)``.  Random in
        ``[-2, 2] x [-2, 2]`` if None.
    warmup : int
        Transient steps to discard before sampling.
    seed : int
        Random seed for the initial condition.

    Returns
    -------
    dict
        'X' : np.ndarray, shape ``(n_steps, 2)`` -- phase points.
        'F' : np.ndarray, shape ``(n_steps, 2)`` -- vector field values.
        'J_all' : np.ndarray, shape ``(n_steps, 2, 2)`` -- exact Jacobians.
    """
    rng = np.random.default_rng(seed)
    if x0 is None:
        x0 = rng.uniform(-2.0, 2.0, size=2)

    total_steps = n_steps + warmup
    t_span      = (0.0, total_steps * dt)
    t_eval      = np.linspace(*t_span, total_steps)

    sol = solve_ivp(
        fun=lambda t, y: f(y),
        t_span=t_span,
        y0=x0,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-9,
    )

    X     = sol.y[:, warmup:].T
    F_arr = np.array([f(x) for x in X])
    J_all = np.array([J(x) for x in X])

    return {'X': X, 'F': F_arr, 'J_all': J_all}


def generate_data_forced(
    n_steps: int        = 5000,
    dt:      float      = 0.05,
    x0:      np.ndarray = None,
    warmup:  int        = 500,
    seed:    int        = 42,
) -> dict:
    """Integrate the forced (chaotic) Duffing system and return phase points.

    The forced system is non-dissipative-trivially: at the standard
    Holmes parameters it has a chaotic strange attractor and a single
    long trajectory *is* a useful training source (analogous to Lorenz).

    Parameters
    ----------
    n_steps : int
        Number of post-warmup time steps to return.
    dt : float
        Integration time step.  Default ``0.05`` is well below the
        forcing period ``T = 2 pi / OMEGA ~= 6.28``, giving ~125 samples
        per drive cycle.
    x0 : np.ndarray or None
        Initial condition, shape ``(3,)`` ``(x, x_dot, phi)``.  Random
        in ``[-1, 1] x [-1, 1] x [0, 2 pi)`` if None.
    warmup : int
        Transient steps to discard before sampling.
    seed : int
        Random seed for the initial condition.

    Returns
    -------
    dict
        'X' : np.ndarray, shape ``(n_steps, 3)``.
        'F' : np.ndarray, shape ``(n_steps, 3)``.
        'J_all' : np.ndarray, shape ``(n_steps, 3, 3)``.
    """
    rng = np.random.default_rng(seed)
    if x0 is None:
        x0 = np.array([
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.0, 1.0),
            rng.uniform(0.0, 2.0 * np.pi),
        ])

    total_steps = n_steps + warmup
    t_span      = (0.0, total_steps * dt)
    t_eval      = np.linspace(*t_span, total_steps)

    sol = solve_ivp(
        fun=lambda t, y: f_forced(y),
        t_span=t_span,
        y0=x0,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-10,
    )

    X     = sol.y[:, warmup:].T
    F_arr = np.array([f_forced(x) for x in X])
    J_all = np.array([J_forced(x) for x in X])

    return {'X': X, 'F': F_arr, 'J_all': J_all}


def generate_trajectory(x0, n_steps, dt=0.05):
    """Integrate the unforced Duffing ODE from an initial condition.

    Parameters
    ----------
    x0 : np.ndarray
        Initial state ``[x, x_dot]``, shape ``(2,)``.
    n_steps : int
        Number of integration steps.
    dt : float
        Time step.

    Returns
    -------
    np.ndarray
        Trajectory of shape ``(n_steps + 1, 2)``.
    """
    t_span = (0.0, n_steps * dt)
    t_eval = np.linspace(0.0, n_steps * dt, n_steps + 1)
    sol = solve_ivp(lambda t, y: f(y), t_span, x0,
                    method='RK45', t_eval=t_eval, rtol=1e-10, atol=1e-10)
    return sol.y.T


def generate_trajectory_forced(x0, n_steps, dt=0.05):
    """Integrate the forced phase-augmented Duffing ODE from an initial condition.

    Parameters
    ----------
    x0 : np.ndarray
        Initial state ``[x, x_dot, phi]``, shape ``(3,)``.
    n_steps : int
        Number of integration steps.
    dt : float
        Time step.

    Returns
    -------
    np.ndarray
        Trajectory of shape ``(n_steps + 1, 3)``.  Phase is *not* wrapped.
    """
    t_span = (0.0, n_steps * dt)
    t_eval = np.linspace(0.0, n_steps * dt, n_steps + 1)
    sol = solve_ivp(lambda t, y: f_forced(y), t_span, x0,
                    method='RK45', t_eval=t_eval, rtol=1e-10, atol=1e-10)
    return sol.y.T


# -- Ground-truth basin classification ---------------------------------------

def classify_basin(
    states:    np.ndarray,
    t_max:     float = 100.0,
    tol:       float = 0.1,
) -> np.ndarray:
    """Label each unforced-system state by the well it converges to.

    Integrates the *unforced* damped dynamics from each state once to
    ``t_max`` and inspects the final position:

    - ``+1`` if the final state is within radius ``tol`` of ``(+1, 0)``,
    - ``-1`` if within ``tol`` of ``(-1, 0)``,
    - ``0``  if neither (unresolved, e.g. the saddle itself).

    With ``DELTA > 0`` the system is dissipative and ``t_max = 100``
    is comfortably long enough for any non-saddle initial condition
    to settle.  This is the ground-truth basin label used to overlay
    the saddle separatrix on partition figures and to score partition-
    versus-basin agreement.

    Parameters
    ----------
    states : np.ndarray
        States to classify, shape ``(P, 2)`` or ``(2,)``.
    t_max : float
        Integration horizon.
    tol : float
        Radius around each well center within which the final state is
        declared "settled".

    Returns
    -------
    np.ndarray
        Integer labels, shape ``(P,)`` in ``{-1, 0, +1}``.
    """
    states = np.atleast_2d(states)
    P      = states.shape[0]
    labels = np.zeros(P, dtype=np.int8)

    well_pos = np.array([+1.0, 0.0])
    well_neg = np.array([-1.0, 0.0])

    for i in range(P):
        sol = solve_ivp(
            fun=lambda t, z: f(z),
            t_span=(0.0, t_max),
            y0=states[i].astype(np.float64),
            method='RK45',
            rtol=1e-7,
            atol=1e-7,
        )
        y_final = sol.y[:, -1]
        if np.linalg.norm(y_final - well_pos) < tol:
            labels[i] = +1
        elif np.linalg.norm(y_final - well_neg) < tol:
            labels[i] = -1
        # else: stays 0 (unresolved)

    return labels


# -- Tests --------------------------------------------------------------------

def test_jacobian(n_tests: int = 20, eps: float = 1e-5) -> bool:
    """Verify both analytic Jacobians (J and J_forced) against finite differences.

    Parameters
    ----------
    n_tests : int
        Number of random test points per variant.
    eps : float
        Finite-difference step size.

    Returns
    -------
    bool
        True if all element-wise errors are below 1e-7.
    """
    rng = np.random.default_rng(0)

    # Unforced 2D
    errs_2d = []
    for _ in range(n_tests):
        x  = rng.standard_normal(2) * 2.0
        Je = J(x)
        Jf = np.zeros((2, 2))
        for j in range(2):
            ej     = np.zeros(2); ej[j] = eps
            Jf[:, j] = (f(x + ej) - f(x - ej)) / (2.0 * eps)
        errs_2d.append(np.max(np.abs(Je - Jf)))
    max_err_2d = float(np.max(errs_2d))
    pass_2d    = max_err_2d < 1e-7

    # Forced 3D
    errs_3d = []
    for _ in range(n_tests):
        x      = rng.standard_normal(3) * 2.0
        x[2]   = rng.uniform(0.0, 2.0 * np.pi)
        Je     = J_forced(x)
        Jf     = np.zeros((3, 3))
        for j in range(3):
            ej     = np.zeros(3); ej[j] = eps
            Jf[:, j] = (f_forced(x + ej) - f_forced(x - ej)) / (2.0 * eps)
        errs_3d.append(np.max(np.abs(Je - Jf)))
    max_err_3d = float(np.max(errs_3d))
    pass_3d    = max_err_3d < 1e-7

    print(f"[{'PASS' if pass_2d else 'FAIL'}] duffing J (unforced 2D): max err {max_err_2d:.2e}")
    print(f"[{'PASS' if pass_3d else 'FAIL'}] duffing J_forced (3D)  : max err {max_err_3d:.2e}")
    return pass_2d and pass_3d


if __name__ == "__main__":
    ok = test_jacobian()
    assert ok, "Jacobian check failed"

    data = sample_phase_space(n_samples=200)
    print(f"unforced: X {data['X'].shape}  F {data['F'].shape}  J_all {data['J_all'].shape}")

    data_f = sample_phase_space_forced(n_samples=200)
    print(f"forced:   X {data_f['X'].shape}  F {data_f['F'].shape}  J_all {data_f['J_all'].shape}")

    traj = generate_trajectory(np.array([1.5, 0.0]), n_steps=400, dt=0.05)
    print(f"unforced trajectory: {traj.shape}, settled at {traj[-1]}")

    traj_f = generate_trajectory_forced(np.array([0.5, 0.0, 0.0]), n_steps=400, dt=0.05)
    print(f"forced trajectory:   {traj_f.shape}, last state {traj_f[-1]}")

    grid = np.array([[+1.5, 0.0], [-1.5, 0.0], [0.0, 0.0]])
    print(f"basin labels for sample points: {classify_basin(grid)}")
