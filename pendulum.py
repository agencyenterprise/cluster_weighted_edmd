"""
Damped pendulum (undriven).

    θ̇  = θ̇
    θ̈  = -sin(θ) - γ·θ̇

State: (θ, θ̇) ∈ R². The nonlinearity sin(θ) makes this a genuine test
for local-vs-global Koopman: polynomial EDMD can only approximate sin(θ)
by Taylor, which degrades badly over wide θ ranges.
"""

import numpy as np
from scipy.integrate import solve_ivp


GAMMA = 0.2   # damping


def f(state: np.ndarray) -> np.ndarray:
    theta, theta_dot = state
    return np.array([
        theta_dot,
        -np.sin(theta) - GAMMA * theta_dot,
    ])


def J(state: np.ndarray) -> np.ndarray:
    theta, _ = state
    return np.array([
        [0.0,           1.0  ],
        [-np.cos(theta), -GAMMA],
    ])


def wrap_theta(x):
    """Wrap angular component to [-pi, pi). Accepts arrays or tensors."""
    import torch
    if isinstance(x, torch.Tensor):
        y = x.clone()
        y[..., 0] = torch.remainder(y[..., 0] + np.pi, 2 * np.pi) - np.pi
        return y
    y = x.copy()
    y[..., 0] = (y[..., 0] + np.pi) % (2 * np.pi) - np.pi
    return y


def angular_dist(x_pred, x_true):
    """Euclidean norm with angular distance on theta. Works batched."""
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
    """
    Uniformly sample (theta, theta_dot) from a box and evaluate f.
    This is the TRAINING data: phase-space regression, not a trajectory.
    """
    rng = np.random.default_rng(seed)
    theta     = rng.uniform(-theta_max,    theta_max,    n_samples)
    theta_dot = rng.uniform(-thetadot_max, thetadot_max, n_samples)
    X = np.stack([theta, theta_dot], axis=1)
    F = np.array([f(x) for x in X])
    return {'X': X, 'F': F}


def generate_trajectory(x0, n_steps, dt=0.05, wrap=True):
    """
    Integrate true pendulum from x0 for n_steps at dt.
    Returns array of shape (n_steps+1, 2). Wraps theta after integration.
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
