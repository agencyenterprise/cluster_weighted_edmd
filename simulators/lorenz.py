"""
Lorenz-63 chaotic attractor simulator.

Implements the classic Lorenz system with parameters (sigma=10, rho=28,
beta=8/3) as a validation testbed for residual-aware clustering.  The
right-hand side is polynomial, so polynomial EDMD can represent it exactly
at sufficient degree -- making it a best-case sanity check for the local
Koopman pipeline.

Provides three core objects:

- ``f(state)`` -- vector field dx/dt at a single point (3,).
- ``J(state)`` -- analytic Jacobian (3, 3) at a single point.
- ``generate_data(...)`` -- integrate the ODE, discard transient, and
  return phase points, vector-field evaluations, and exact Jacobians
  ready for the EM fitting functions.

Usage
-----
Generate training data for the clustering pipeline::

    from residual_aware_clustering.simulators.lorenz import generate_data, f, J
    import torch

    data = generate_data(n_steps=5000, dt=0.01, warmup=1000, seed=42)

    # data['X']     -- (5000, 3)   phase points on the attractor
    # data['F']     -- (5000, 3)   f(x_i) at each point
    # data['J_all'] -- (5000, 3, 3) exact Jacobian at each point

    X = torch.tensor(data['X'], dtype=torch.float64)
    F = torch.tensor(data['F'], dtype=torch.float64)

Evaluate the vector field and Jacobian at a single point::

    import numpy as np
    x = np.array([1.0, 2.0, 3.0])
    dxdt = f(x)    # (3,)
    jac  = J(x)    # (3, 3)

Verify analytic Jacobian against finite differences::

    from residual_aware_clustering.simulators.lorenz import test_jacobian
    test_jacobian()  # prints PASSED/FAILED

Key concepts
------------
- **Warmup period**: the first ``warmup`` integration steps are discarded
  so that the trajectory has settled onto the attractor before sampling.
- **Exact Jacobian**: because the RHS is polynomial, ``J`` is exact
  (no numerical differentiation), which lets the Taylor-analytic EM
  variant (``fit_taylor``) run without any approximation error in the
  local models.
"""

import numpy as np
from scipy.integrate import solve_ivp

from . import _sampling

# ── System parameters ─────────────────────────────────────────────────────────

SIGMA = 10.0
RHO   = 28.0
BETA  = 8.0 / 3.0


def f(state: np.ndarray) -> np.ndarray:
    """Evaluate the Lorenz-63 vector field.

    Parameters
    ----------
    state : np.ndarray
        State vector ``[x, y, z]``, shape ``(3,)``.

    Returns
    -------
    np.ndarray
        Time derivatives ``[dx/dt, dy/dt, dz/dt]``, shape ``(3,)``.
    """
    x, y, z = state
    return np.array([
        SIGMA * (y - x),
        x * (RHO - z) - y,
        x * y - BETA * z
    ])


def J(state: np.ndarray) -> np.ndarray:
    """Compute the exact Jacobian of the Lorenz vector field.

    Parameters
    ----------
    state : np.ndarray
        State vector ``[x, y, z]``, shape ``(3,)``.

    Returns
    -------
    np.ndarray
        Jacobian matrix, shape ``(3, 3)``.
    """
    x, y, z = state
    return np.array([
        [-SIGMA,    SIGMA,  0.0  ],
        [RHO - z,  -1.0,   -x   ],
        [y,         x,     -BETA]
    ])


# ── Data generation ───────────────────────────────────────────────────────────

def generate_data(
    n_steps: int        = 5000,
    dt:      float      = 0.01,
    x0:      np.ndarray = None,
    warmup:  int        = 1000,
    seed:    int        = 42,
) -> dict:
    """Integrate the Lorenz system and return phase points with derivatives.

    Parameters
    ----------
    n_steps : int
        Number of post-warmup time steps to return.
    dt : float
        Integration time step.
    x0 : np.ndarray or None
        Initial condition, shape ``(3,)``. Random if None.
    warmup : int
        Transient steps to discard before sampling.
    seed : int
        Random seed for the initial condition.

    Returns
    -------
    dict
        'X' : np.ndarray, shape ``(n_steps, 3)`` -- phase points.
        'F' : np.ndarray, shape ``(n_steps, 3)`` -- vector field values.
        'J_all' : np.ndarray, shape ``(n_steps, 3, 3)`` -- exact Jacobians.
    """
    rng = np.random.default_rng(seed)

    if x0 is None:
        x0 = rng.standard_normal(3)

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

    X     = sol.y[:, warmup:].T                  # (n_steps, 3)
    F_arr = np.array([f(x) for x in X])         # (n_steps, 3)
    J_all = np.array([J(x) for x in X])         # (n_steps, 3, 3)

    return {'X': X, 'F': F_arr, 'J_all': J_all}


# ── Distribution samplers ─────────────────────────────────────────────────────
#
# These complement ``generate_data`` (which samples a single long trajectory
# on the attractor) with distribution-controllable phase-space sampling that
# is used by config-driven paper experiments.  All return the standard
# ``{'X', 'F', 'J_all'}`` dict.

LORENZ_FIXED_POINTS = (
    np.array([+np.sqrt(BETA * (RHO - 1.0)),
              +np.sqrt(BETA * (RHO - 1.0)), RHO - 1.0]),   # C+
    np.array([-np.sqrt(BETA * (RHO - 1.0)),
              -np.sqrt(BETA * (RHO - 1.0)), RHO - 1.0]),   # C-
)


def sample_uniform(
    n_samples: int = 4000,
    box_x:     float = 25.0,
    box_y:     float = 30.0,
    box_z:     float = 25.0,
    z_offset:  float = 25.0,
    seed:      int = 42,
) -> dict:
    """Uniform sampling on a (xyz) box covering the Lorenz attractor.

    Defaults give roughly the attractor's bounding box for the standard
    parameters: ``x in [-25, 25]``, ``y in [-30, 30]``,
    ``z in [0, 50]`` (centered at ``z_offset``).

    Parameters
    ----------
    n_samples : int
    box_x, box_y, box_z : float
        Half-widths per axis (z range is ``z_offset +/- box_z``).
    z_offset : float
        Center of the z range (Lorenz z is non-negative, so we offset).
    seed : int
    """
    rng = np.random.default_rng(seed)
    X = np.column_stack([
        rng.uniform(-box_x, +box_x, n_samples),
        rng.uniform(-box_y, +box_y, n_samples),
        rng.uniform(z_offset - box_z, z_offset + box_z, n_samples),
    ])
    return _sampling.evaluate_field(X, f, J)


def sample_gaussian(
    n_samples: int = 4000,
    mean:      np.ndarray = None,
    sigma                 = 10.0,
    seed:      int = 42,
) -> dict:
    """Sample Lorenz phase points from a 3D Gaussian.

    Default mean is the attractor's approximate centroid (around the
    center of the two lobes' enclosing region).
    """
    if mean is None:
        mean = np.array([0.0, 0.0, RHO - 1.0])
    X = _sampling.gaussian(n_samples, mean, sigma, seed)
    return _sampling.evaluate_field(X, f, J)


def sample_gaussian_mixture(
    n_samples: int = 4000,
    centers:   np.ndarray = None,
    sigmas                 = 6.0,
    weights:   np.ndarray = None,
    seed:      int = 42,
) -> dict:
    """Sample from a Gaussian mixture centered on the two unstable foci.

    The default centers ``C+`` and ``C-`` are the system's non-trivial
    fixed points -- a multimodal density tied to the system's geometry.
    """
    if centers is None:
        centers = np.stack(LORENZ_FIXED_POINTS, axis=0)
    X = _sampling.gaussian_mixture(n_samples, centers, sigmas, weights, seed)
    return _sampling.evaluate_field(X, f, J)


def sample_periodic_noise(
    n_samples:  int = 4000,
    amplitudes: np.ndarray = None,
    frequency:  float = 1.0,
    center:     np.ndarray = None,
    noise_std:  float = 1.0,
    seed:       int   = 42,
) -> dict:
    """Sample states along a 3D Lissajous-style curve plus noise.

    The default curve traces an ellipse in the ``(x, y)`` plane with a
    z-axis modulation -- a periodic curve sweeping through both lobes
    of the attractor's neighborhood.
    """
    if amplitudes is None:
        amplitudes = np.array([10.0, 12.0, 8.0])
    if center is None:
        center = np.array([0.0, 0.0, RHO - 1.0])
    X = _sampling.periodic_noise(
        n_samples=n_samples, amplitudes=amplitudes, frequency=frequency,
        center=center, noise_std=noise_std, seed=seed,
    )
    return _sampling.evaluate_field(X, f, J)


def sample_trajectory_ensemble(
    n_traj:        int   = 100,
    n_steps:       int   = 50,
    dt:            float = 0.01,
    ic_box:        float = 20.0,
    ic_z_offset:   float = 25.0,
    seed:          int   = 42,
) -> dict:
    """Sample Lorenz states from an ensemble of forward trajectories.

    Trajectories converge to the attractor; ICs are uniform in a box.
    Density is highest along the attractor itself.
    """
    def _ic(rng):
        return np.array([
            rng.uniform(-ic_box, +ic_box),
            rng.uniform(-ic_box, +ic_box),
            rng.uniform(ic_z_offset - ic_box, ic_z_offset + ic_box),
        ])
    X = _sampling.trajectory_ensemble(
        n_traj=n_traj, n_steps=n_steps, dt=dt,
        f_fn=f, initial_condition_sampler=_ic, seed=seed,
    )
    return _sampling.evaluate_field(X, f, J)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_jacobian(n_tests: int = 20, eps: float = 1e-5) -> bool:
    """Verify the analytic Jacobian against central finite differences.

    Parameters
    ----------
    n_tests : int
        Number of random test points.
    eps : float
        Finite-difference step size.

    Returns
    -------
    bool
        True if all element-wise errors are below 1e-7.
    """
    rng    = np.random.default_rng(0)
    errors = []

    for _ in range(n_tests):
        x       = rng.standard_normal(3) * 5
        J_exact = J(x)
        J_fd    = np.zeros((3, 3))

        for j in range(3):
            e_j        = np.zeros(3)
            e_j[j]     = eps
            J_fd[:, j] = (f(x + e_j) - f(x - e_j)) / (2 * eps)

        errors.append(np.max(np.abs(J_exact - J_fd)))

    max_err = float(np.max(errors))
    passed  = max_err < 1e-7
    status  = "PASSED" if passed else "FAILED"
    print(f"[{status}] test_jacobian — max elementwise error: {max_err:.2e}  (threshold: 1e-7)")
    return passed


if __name__ == "__main__":
    test_jacobian()
    data = generate_data()
    print(f"X shape:     {data['X'].shape}")
    print(f"F shape:     {data['F'].shape}")
    print(f"J_all shape: {data['J_all'].shape}")
