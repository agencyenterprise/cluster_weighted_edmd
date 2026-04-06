import numpy as np
from scipy.integrate import solve_ivp

# ── System parameters ─────────────────────────────────────────────────────────

SIGMA = 10.0
RHO   = 28.0
BETA  = 8.0 / 3.0


def f(state: np.ndarray) -> np.ndarray:
    """
    Lorenz vector field.
    state: (3,) array [x, y, z]
    returns: (3,) array [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = state
    return np.array([
        SIGMA * (y - x),
        x * (RHO - z) - y,
        x * y - BETA * z
    ])


def J(state: np.ndarray) -> np.ndarray:
    """
    Exact Jacobian of f at state.
    state: (3,) array
    returns: (3, 3) array
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
    """
    Integrate the Lorenz system and return phase points,
    vector field values, and exact Jacobians.

    Returns dict with keys:
        X     : (n_steps, 3) phase points on attractor
        F     : (n_steps, 3) f(x_i) at each point
        J_all : (n_steps, 3, 3) exact Jacobian at each point
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


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_jacobian(n_tests: int = 20, eps: float = 1e-5) -> bool:
    """
    Verify exact Jacobian against central finite differences.
    Returns True if all errors are below 1e-7.
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
