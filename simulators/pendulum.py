"""
Damped pendulum (undriven) simulator.

Implements the equations of motion for an undriven damped pendulum:

    theta_dot  = theta_dot
    theta_ddot = -sin(theta) - gamma * theta_dot

State: (theta, theta_dot) in R^2.  The nonlinearity sin(theta) makes this
a genuine stress test for local-vs-global Koopman approximation: polynomial
EDMD can only approximate sin(theta) via Taylor expansion, which degrades
badly over wide theta ranges.  This is the second validation system
alongside the Lorenz attractor.

Provides five core objects:

- ``f(state)`` -- vector field dx/dt at a single point (2,).
- ``J(state)`` -- analytic Jacobian (2, 2) at a single point.
- ``sample_phase_space(...)`` -- uniformly sample (theta, theta_dot) and
  evaluate f.  This is the primary training data generator (phase-space
  regression, not a single trajectory).
- ``generate_trajectory(...)`` -- integrate the true ODE from an initial
  condition, returning wrapped trajectory.
- ``angular_dist(x_pred, x_true)`` -- Euclidean distance metric that
  handles angular wrapping on the theta component.

Usage
-----
Generate training data (phase-space sampling)::

    from residual_aware_clustering.simulators.pendulum import sample_phase_space
    import torch

    data = sample_phase_space(n_samples=4000, theta_max=np.pi,
                              thetadot_max=3.0, seed=42)

    # data['X'] -- (4000, 2) uniformly sampled (theta, theta_dot)
    # data['F'] -- (4000, 2) f(x_i) at each sample

    X = torch.tensor(data['X'], dtype=torch.float64)
    F = torch.tensor(data['F'], dtype=torch.float64)

Generate a test trajectory for prediction evaluation::

    import numpy as np
    from residual_aware_clustering.simulators.pendulum import generate_trajectory

    x0   = np.array([2.5, 0.0])        # large initial angle
    traj = generate_trajectory(x0, n_steps=200, dt=0.05)
    # traj.shape == (201, 2), theta wrapped to [-pi, pi)

Compute prediction error with angular wrapping::

    from residual_aware_clustering.simulators.pendulum import angular_dist

    err = angular_dist(predicted_states, true_states)  # works batched

Key concepts
------------
- **Phase-space sampling vs trajectory**: unlike the Lorenz simulator
  which produces a single long trajectory, the pendulum uses uniform
  phase-space sampling (``sample_phase_space``) for training.  This
  avoids the density bias of trajectory data near the fixed point.
- **Angular wrapping**: ``wrap_theta`` maps theta to [-pi, pi) after
  integration, and ``angular_dist`` uses atan2 to compute the shortest
  angular difference.  Both accept numpy arrays and torch tensors.
- **Damping coefficient**: ``GAMMA = 0.2`` (module-level constant).
"""

import numpy as np
from scipy.integrate import solve_ivp

from . import _sampling


GAMMA = 0.2   # damping


def f(state: np.ndarray) -> np.ndarray:
    """Evaluate the damped pendulum vector field.

    Parameters
    ----------
    state : np.ndarray
        State vector ``[theta, theta_dot]``, shape ``(2,)``.

    Returns
    -------
    np.ndarray
        Time derivatives ``[theta_dot, theta_ddot]``, shape ``(2,)``.
    """
    theta, theta_dot = state
    return np.array([
        theta_dot,
        -np.sin(theta) - GAMMA * theta_dot,
    ])


def J(state: np.ndarray) -> np.ndarray:
    """Compute the exact Jacobian of the pendulum vector field.

    Parameters
    ----------
    state : np.ndarray
        State vector ``[theta, theta_dot]``, shape ``(2,)``.

    Returns
    -------
    np.ndarray
        Jacobian matrix, shape ``(2, 2)``.
    """
    theta, _ = state
    return np.array([
        [0.0,           1.0  ],
        [-np.cos(theta), -GAMMA],
    ])


def wrap_theta(x):
    """Wrap the theta (first) component to [-pi, pi).

    Parameters
    ----------
    x : np.ndarray or torch.Tensor
        State array with theta in ``x[..., 0]``.

    Returns
    -------
    np.ndarray or torch.Tensor
        Copy of *x* with theta wrapped to [-pi, pi).
    """
    import torch
    if isinstance(x, torch.Tensor):
        y = x.clone()
        y[..., 0] = torch.remainder(y[..., 0] + np.pi, 2 * np.pi) - np.pi
        return y
    y = x.copy()
    y[..., 0] = (y[..., 0] + np.pi) % (2 * np.pi) - np.pi
    return y


def angular_dist(x_pred, x_true):
    """Euclidean distance metric with angular wrapping on the theta component.

    Parameters
    ----------
    x_pred : np.ndarray or torch.Tensor
        Predicted states, shape ``(..., 2)``.
    x_true : np.ndarray or torch.Tensor
        True states, shape ``(..., 2)``.

    Returns
    -------
    np.ndarray or torch.Tensor
        Per-sample distance, shape ``(...)``.
    """
    import torch
    if isinstance(x_pred, torch.Tensor):
        dth = torch.atan2(torch.sin(x_pred[..., 0] - x_true[..., 0]),
                          torch.cos(x_pred[..., 0] - x_true[..., 0]))
        dtd = x_pred[..., 1] - x_true[..., 1]
        return torch.sqrt(dth**2 + dtd**2)
    dth = np.arctan2(np.sin(x_pred[..., 0] - x_true[..., 0]),
                     np.cos(x_pred[..., 0] - x_true[..., 0]))
    dtd = x_pred[..., 1] - x_true[..., 1]
    return np.sqrt(dth**2 + dtd**2)


def sample_phase_space(n_samples=4000, theta_max=np.pi, thetadot_max=3.0, seed=42):
    """Uniformly sample phase-space points and evaluate the vector field.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw.
    theta_max : float
        Half-width of the theta sampling range ``[-theta_max, theta_max]``.
    thetadot_max : float
        Half-width of the theta_dot sampling range.
    seed : int
        Random seed.

    Returns
    -------
    dict
        'X' : np.ndarray, shape ``(n_samples, 2)`` -- sampled states.
        'F' : np.ndarray, shape ``(n_samples, 2)`` -- vector field values.
    """
    rng = np.random.default_rng(seed)
    theta     = rng.uniform(-theta_max,    theta_max,    n_samples)
    theta_dot = rng.uniform(-thetadot_max, thetadot_max, n_samples)
    X = np.stack([theta, theta_dot], axis=1)
    F = np.array([f(x) for x in X])
    return {'X': X, 'F': F}


def sample_gaussian(
    n_samples: int = 4000,
    mean:      np.ndarray = None,
    sigma                 = 1.0,
    seed:      int = 42,
) -> dict:
    """Sample pendulum phase points from a 2D Gaussian.

    Default mean is ``(0, 0)`` (small-amplitude oscillation regime).

    Parameters
    ----------
    n_samples : int
    mean : array_like, shape ``(2,)`` or None
    sigma : float or array_like, shape ``(2,)``
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
    sigmas                 = 0.5,
    weights:   np.ndarray = None,
    seed:      int = 42,
) -> dict:
    """Sample pendulum phase points from a Gaussian mixture.

    Default centers cover the small-oscillation regime around the down
    equilibrium and a moderate-energy regime: ``[(0, 0), (0, 2)]``.

    Parameters
    ----------
    n_samples : int
    centers : array_like, shape ``(K, 2)`` or None
    sigmas : float or array_like
    weights : array_like or None
    seed : int
    """
    if centers is None:
        centers = np.array([[0.0, 0.0], [0.0, 2.0]])
    X = _sampling.gaussian_mixture(n_samples, centers, sigmas, weights, seed)
    return _sampling.evaluate_field(X, f, J)


def sample_periodic_noise(
    n_samples:  int = 4000,
    amplitudes: np.ndarray = None,
    frequency:  float = 1.0,
    center:     np.ndarray = None,
    noise_std:  float = 0.1,
    seed:       int   = 42,
) -> dict:
    """Sample pendulum phase points from a small-amplitude orbit + noise.

    Defaults trace the linearized small-amplitude orbit
    ``(theta, theta_dot) = (A cos t, -A sin t)`` (frequency 1 = the
    pendulum's small-oscillation eigenfrequency).

    Parameters
    ----------
    n_samples : int
    amplitudes : array_like, shape ``(2,)`` or None
        Default ``(1.0, 1.0)``.
    frequency : float
    center : array_like, shape ``(2,)`` or None
        Default ``(0, 0)``.
    noise_std : float
    seed : int
    """
    if amplitudes is None:
        amplitudes = np.array([1.0, 1.0])
    if center is None:
        center = np.zeros(2)
    X = _sampling.periodic_noise(
        n_samples=n_samples, amplitudes=amplitudes, frequency=frequency,
        center=center, noise_std=noise_std, seed=seed,
    )
    return _sampling.evaluate_field(X, f, J)


def sample_trajectory_ensemble(
    n_traj:        int   = 200,
    n_steps:       int   = 50,
    dt:            float = 0.05,
    ic_theta_max:  float = np.pi,
    ic_thetadot_max: float = 3.0,
    seed:          int   = 42,
) -> dict:
    """Sample pendulum phase points from an ensemble of forward trajectories.

    Damping concentrates trajectory density near ``(0, 0)`` over time;
    high-energy ICs spend most of their time in the rotational regime.

    Parameters
    ----------
    n_traj : int
    n_steps : int
        Steps per trajectory; each contributes ``n_steps + 1`` points.
    dt : float
    ic_theta_max, ic_thetadot_max : float
    seed : int
    """
    def _ic(rng):
        return np.array([
            rng.uniform(-ic_theta_max,    ic_theta_max),
            rng.uniform(-ic_thetadot_max, ic_thetadot_max),
        ])
    X = _sampling.trajectory_ensemble(
        n_traj=n_traj, n_steps=n_steps, dt=dt,
        f_fn=f, initial_condition_sampler=_ic, seed=seed,
    )
    return _sampling.evaluate_field(X, f, J)


def generate_trajectory(x0, n_steps, dt=0.05, wrap=True):
    """Integrate the true pendulum ODE from an initial condition.

    Parameters
    ----------
    x0 : np.ndarray
        Initial state ``[theta, theta_dot]``, shape ``(2,)``.
    n_steps : int
        Number of integration steps.
    dt : float
        Time step.
    wrap : bool
        If True, wrap theta to [-pi, pi) after integration.

    Returns
    -------
    np.ndarray
        Trajectory of shape ``(n_steps + 1, 2)``.
    """
    t_span = (0.0, n_steps * dt)
    t_eval = np.linspace(0.0, n_steps * dt, n_steps + 1)
    sol = solve_ivp(lambda t, y: f(y), t_span, x0,
                    method='RK45', t_eval=t_eval, rtol=1e-10, atol=1e-10)
    traj = sol.y.T
    if wrap:
        traj = wrap_theta(traj)
    return traj


# ── Tests ────────────────────────────────────────────────────────────────────

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
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(n_tests):
        x = rng.standard_normal(2) * 2
        Je = J(x)
        Jf = np.zeros((2, 2))
        for j in range(2):
            ej = np.zeros(2); ej[j] = eps
            Jf[:, j] = (f(x + ej) - f(x - ej)) / (2 * eps)
        errs.append(np.max(np.abs(Je - Jf)))
    max_err = float(np.max(errs))
    passed = max_err < 1e-7
    print(f"[{'PASS' if passed else 'FAIL'}] pendulum Jacobian: max err {max_err:.2e}")
    return passed


if __name__ == "__main__":
    test_jacobian()
    data = sample_phase_space(n_samples=100)
    print("data shapes:", data['X'].shape, data['F'].shape)
    traj = generate_trajectory(np.array([2.5, 0.0]), n_steps=200)
    print("trajectory shape:", traj.shape)
    print(f"final state: θ={traj[-1, 0]:.3f}, θ̇={traj[-1, 1]:.3f}")
