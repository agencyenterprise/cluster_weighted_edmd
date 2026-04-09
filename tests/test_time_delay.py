"""
Integration tests: verify our TimeDelay matches pykoopman's TimeDelay.
"""

import numpy as np
import torch
import pytest

from residual_aware_clustering.models.observables import TimeDelay


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def trajectory_1d():
    """Simple 1D trajectory."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((50, 1))


@pytest.fixture
def trajectory_2d():
    """2D trajectory (like a 2-state dynamical system)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((50, 2))


@pytest.fixture
def trajectory_high_d():
    """High-dimensional trajectory (like LLM hidden states)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((100, 16))


# ── Test against pykoopman ───────────────────────────────────────────────────

from pykoopman.observables import TimeDelay as PyKoopmanTD


@pytest.mark.parametrize("delay,n_delays", [(1, 1), (1, 2), (1, 3), (2, 2), (3, 1)])
def test_matches_pykoopman_1d(trajectory_1d, delay, n_delays):
    x = trajectory_1d

    pk_td = PyKoopmanTD(delay=delay, n_delays=n_delays)
    pk_td.fit(x)
    pk_out = pk_td.transform(x)

    # ours
    our_td = TimeDelay(delay=delay, n_delays=n_delays)
    our_out = our_td.transform(x)

    assert pk_out.shape == our_out.shape, \
        f"Shape mismatch: pykoopman={pk_out.shape}, ours={our_out.shape}"
    assert np.allclose(pk_out, our_out, atol=1e-12), \
        f"Max diff: {np.abs(pk_out - our_out).max():.2e}"


@pytest.mark.parametrize("delay,n_delays", [(1, 1), (1, 2), (1, 3), (2, 2), (3, 1)])
def test_matches_pykoopman_2d(trajectory_2d, delay, n_delays):
    x = trajectory_2d

    pk_td = PyKoopmanTD(delay=delay, n_delays=n_delays)
    pk_td.fit(x)
    pk_out = pk_td.transform(x)

    our_td = TimeDelay(delay=delay, n_delays=n_delays)
    our_out = our_td.transform(x)

    assert pk_out.shape == our_out.shape
    assert np.allclose(pk_out, our_out, atol=1e-12), \
        f"Max diff: {np.abs(pk_out - our_out).max():.2e}"


@pytest.mark.parametrize("delay,n_delays", [(1, 2), (1, 5), (2, 3)])
def test_matches_pykoopman_high_d(trajectory_high_d, delay, n_delays):
    x = trajectory_high_d

    pk_td = PyKoopmanTD(delay=delay, n_delays=n_delays)
    pk_td.fit(x)
    pk_out = pk_td.transform(x)

    our_td = TimeDelay(delay=delay, n_delays=n_delays)
    our_out = our_td.transform(x)

    assert pk_out.shape == our_out.shape
    assert np.allclose(pk_out, our_out, atol=1e-12), \
        f"Max diff: {np.abs(pk_out - our_out).max():.2e}"


# ── Shape tests ──────────────────────────────────────────────────────────────

def test_output_shape_1d(trajectory_1d):
    td = TimeDelay(delay=1, n_delays=3)
    out = td.transform(trajectory_1d)
    T, d = trajectory_1d.shape
    assert out.shape == (T - 3, d * 4)


def test_output_shape_2d(trajectory_2d):
    td = TimeDelay(delay=2, n_delays=2)
    out = td.transform(trajectory_2d)
    T, d = trajectory_2d.shape
    assert out.shape == (T - 4, d * 3)


def test_consumed_samples():
    td = TimeDelay(delay=3, n_delays=4)
    assert td.n_consumed_samples == 12


def test_n_output_features():
    td = TimeDelay(delay=1, n_delays=2)
    assert td.n_output_features(5) == 15  # 5 * (1 + 2)


# ── Content tests ────────────────────────────────────────────────────────────

def test_first_column_is_current_state(trajectory_2d):
    td = TimeDelay(delay=1, n_delays=2)
    out = td.transform(trajectory_2d)
    consumed = td.n_consumed_samples
    np.testing.assert_array_equal(out[:, :2], trajectory_2d[consumed:])


def test_delay_values_correct(trajectory_1d):
    td = TimeDelay(delay=1, n_delays=2)
    out = td.transform(trajectory_1d)
    # out[t] should be [x(t+2), x(t+1), x(t)] for 1D with consumed=2
    for t in range(len(out)):
        assert out[t, 0] == trajectory_1d[t + 2, 0]  # current
        assert out[t, 1] == trajectory_1d[t + 1, 0]  # 1 delay back
        assert out[t, 2] == trajectory_1d[t + 0, 0]  # 2 delays back


def test_delay_spacing(trajectory_1d):
    td = TimeDelay(delay=3, n_delays=2)
    out = td.transform(trajectory_1d)
    consumed = 6  # 3 * 2
    for t in range(len(out)):
        assert out[t, 0] == trajectory_1d[t + consumed, 0]    # current
        assert out[t, 1] == trajectory_1d[t + consumed - 3, 0]  # 3 steps back
        assert out[t, 2] == trajectory_1d[t + consumed - 6, 0]  # 6 steps back


# ── Torch compatibility ──────────────────────────────────────────────────────

def test_torch_input(trajectory_2d):
    td = TimeDelay(delay=1, n_delays=2)
    x_torch = torch.tensor(trajectory_2d, dtype=torch.float64)
    out = td.transform(x_torch)
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float64

    out_np = td.transform(trajectory_2d)
    assert np.allclose(out.numpy(), out_np, atol=1e-12)


def test_torch_device_preserved():
    td = TimeDelay(delay=1, n_delays=2)
    x = torch.randn(30, 4, dtype=torch.float32)
    out = td.transform(x)
    assert out.device == x.device
    assert out.dtype == x.dtype


# ── Pair building ────────────────────────────────────────────────────────────

def test_build_pairs_shapes(trajectory_2d):
    td = TimeDelay(delay=1, n_delays=2)
    X, Y = td.build_pairs(trajectory_2d)
    T, d = trajectory_2d.shape
    d_out = d * 3
    expected_n = T - td.n_consumed_samples - 1  # -1 for the Y shift
    assert X.shape == (expected_n, d_out)
    assert Y.shape == (expected_n, d_out)


def test_build_pairs_consecutive(trajectory_2d):
    td = TimeDelay(delay=1, n_delays=2)
    X, Y = td.build_pairs(trajectory_2d)
    embedded = td.transform(trajectory_2d)
    np.testing.assert_array_equal(X, embedded[:-1])
    np.testing.assert_array_equal(Y, embedded[1:])


# ── Multiple trajectories ───────────────────────────────────────────────────

def test_trajectories_no_cross_boundary():
    rng = np.random.default_rng(99)
    traj1 = rng.standard_normal((20, 3))
    traj2 = rng.standard_normal((25, 3))

    td = TimeDelay(delay=1, n_delays=2)
    X, Y = td.build_pairs_from_trajectories([traj1, traj2])

    X1, Y1 = td.build_pairs(traj1)
    X2, Y2 = td.build_pairs(traj2)

    X_manual = np.concatenate([X1, X2], axis=0)
    Y_manual = np.concatenate([Y1, Y2], axis=0)

    np.testing.assert_array_equal(X, X_manual)
    np.testing.assert_array_equal(Y, Y_manual)


def test_short_trajectory_skipped():
    td = TimeDelay(delay=1, n_delays=5)
    short = np.random.randn(3, 2)  # too short
    long = np.random.randn(20, 2)
    X, Y = td.build_pairs_from_trajectories([short, long])
    assert X.shape[0] > 0  # only long trajectory contributes


# ── Error handling ───────────────────────────────────────────────────────────

def test_too_short_raises():
    td = TimeDelay(delay=1, n_delays=10)
    x = np.random.randn(5, 2)
    with pytest.raises(ValueError, match="too short"):
        td.transform(x)


def test_invalid_delay():
    with pytest.raises(ValueError):
        TimeDelay(delay=0, n_delays=2)


def test_invalid_n_delays():
    with pytest.raises(ValueError):
        TimeDelay(delay=1, n_delays=0)


# ── End-to-end: TimeDelay + EDMD vs pykoopman full pipeline ──────────────────

@pytest.mark.parametrize("d,n_delays", [(1, 2), (2, 2), (3, 3)])
def test_end_to_end_vs_pykoopman(d, n_delays):
    """Full pipeline: our TimeDelay + PolynomialDiscreteEDMD vs pykoopman Koopman."""
    import pykoopman as pk
    from pykoopman.regression import EDMD as PyKoopmanEDMD

    from residual_aware_clustering.models.experimental.polynomial_discrete import PolynomialDiscreteEDMD
    from residual_aware_clustering.models.em_local_edmd import ObservableType

    # Generate stable linear dynamics (damped, so EDMD works well)
    rng = np.random.default_rng(42)
    T = 500
    A = np.eye(d) * 0.9 + rng.standard_normal((d, d)) * 0.05
    # Ensure stable (spectral radius < 1)
    eigvals = np.linalg.eigvals(A)
    A = A / (np.max(np.abs(eigvals)) + 0.1)
    x = np.zeros((T, d))
    x[0] = rng.standard_normal(d)
    for t in range(1, T):
        x[t] = A @ x[t - 1] + rng.standard_normal(d) * 0.01

    delay = 1

    from pykoopman.observables import Polynomial as PyKoopmanPoly
    from residual_aware_clustering.models.experimental.polynomial_discrete import PolynomialDiscreteEDMD

    # --- pykoopman pipeline: raw x → TimeDelay → Polynomial(deg=1) → EDMD ---
    pk_td = PyKoopmanTD(delay=delay, n_delays=n_delays)
    pk_td.fit(x)
    x_delayed = pk_td.transform(x)  # (T - consumed, d_delayed)

    pk_model = pk.Koopman(
        observables=PyKoopmanPoly(degree=1),
        regressor=PyKoopmanEDMD()
    )
    pk_model.fit(x_delayed, dt=1)
    K_pk = pk_model._pipeline.named_steps['regressor'].coef_

    # --- our pipeline: raw x → our TimeDelay → build_pairs → PolynomialDiscreteEDMD ---
    our_td = TimeDelay(delay=delay, n_delays=n_delays)
    X, Y = our_td.build_pairs(x)

    # pykoopman internally does X1=x_delayed[:-1], X2=x_delayed[1:]
    # Our build_pairs does the same: X=embedded[:-1], Y=embedded[1:]
    # Verify the delayed data matches before comparing K
    assert np.allclose(x_delayed[:-1], X, atol=1e-12), "Delay embedding X mismatch"
    assert np.allclose(x_delayed[1:], Y, atol=1e-12), "Delay embedding Y mismatch"

    model = PolynomialDiscreteEDMD(degree=1)
    X_t = torch.tensor(X, dtype=torch.float64)
    Y_t = torch.tensor(Y, dtype=torch.float64)
    center = torch.zeros(X_t.shape[1], dtype=torch.float64)
    weights = torch.ones(X_t.shape[0], dtype=torch.float64)
    model.fit(X_t, Y_t, weights, center)

    # K matrices are in different bases (ours: polynomial, pykoopman: SVD)
    # but predictions must match: U @ K_svd @ U' = K_lstsq
    our_pred = model.predict(X_t, center).numpy()

    # pykoopman prediction on same delayed data
    pk_pred = pk_model.predict(x_delayed[:-1])

    min_len = min(len(our_pred), len(pk_pred))
    assert np.allclose(our_pred[:min_len, :d], pk_pred[:min_len, :d], atol=1e-6), \
        f"Predictions differ: max diff={np.abs(our_pred[:min_len, :d] - pk_pred[:min_len, :d]).max():.2e}"


@pytest.mark.parametrize("d,n_delays", [(2, 2), (3, 3)])
def test_end_to_end_prediction_quality(d, n_delays):
    """Verify our TimeDelay + EDMD produces reasonable predictions."""
    from residual_aware_clustering.models.experimental.polynomial_discrete import PolynomialDiscreteEDMD

    rng = np.random.default_rng(42)
    T = 500
    # Simple linear dynamics with noise
    A = np.eye(d) * 0.95 + rng.standard_normal((d, d)) * 0.02
    x = np.zeros((T, d))
    x[0] = rng.standard_normal(d)
    for t in range(1, T):
        x[t] = A @ x[t - 1] + rng.standard_normal(d) * 0.01

    td = TimeDelay(delay=1, n_delays=n_delays)
    X, Y = td.build_pairs(x)

    X_t = torch.tensor(X, dtype=torch.float64)
    Y_t = torch.tensor(Y, dtype=torch.float64)
    center = torch.zeros(X_t.shape[1], dtype=torch.float64)
    weights = torch.ones(X_t.shape[0], dtype=torch.float64)

    model = PolynomialDiscreteEDMD(degree=1)
    model.fit(X_t, Y_t, weights, center)
    Y_pred = model.predict(X_t, center)

    # Compare against pykoopman predictions (via predict(), which handles basis correctly)
    import pykoopman as pk
    from pykoopman.observables import Polynomial as PkPoly
    from pykoopman.regression import EDMD as PkEDMD

    pk_model = pk.Koopman(observables=PkPoly(degree=1), regressor=PkEDMD())
    pk_model.fit(X, y=Y, dt=1)
    pk_pred = pk_model.predict(X[:-1])  # pykoopman returns d_delayed dims

    # Compare first d_delayed dimensions (both should predict the delay-embedded state)
    min_len = min(len(Y_pred), len(pk_pred))
    d_delayed = X.shape[1]
    assert np.allclose(Y_pred.numpy()[:min_len, :d_delayed], pk_pred[:min_len], atol=1e-6), \
        f"Predictions differ from pykoopman: max diff={np.abs(Y_pred.numpy()[:min_len, :d_delayed] - pk_pred[:min_len]).max():.2e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



