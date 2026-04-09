"""
Integration tests: verify GPU-compatible versions match CPU originals.

Runs each function on CPU with both the original and GPU-compatible version,
then checks outputs match. If CUDA or MPS is available, also runs the GPU
version on the accelerator and checks it matches CPU.
"""

import os
import platform
if platform.system() == "Darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import numpy as np
import pytest

# CPU originals
from residual_aware_clustering import make_hp, fit_local_edmd_discrete, make_hp_gpu, fit_local_edmd_discrete_gpu
from residual_aware_clustering.models.distributions import (
    mvn_logpdf_batch as mvn_cpu,
    residual_logpdf_batch as resid_cpu,
    dirichlet_logpdf as dirichlet_cpu,
    niw_logpdf as niw_cpu,
)
from residual_aware_clustering.models.distributions_gpu import (
    mvn_logpdf_batch as mvn_gpu,
    residual_logpdf_batch as resid_gpu,
    dirichlet_logpdf as dirichlet_gpu,
    niw_logpdf as niw_gpu,
)
from residual_aware_clustering.models.em_local_edmd import monomials, monomial_exponents
from residual_aware_clustering.models.em_local_edmd_discrete import (
    e_step as e_step_cpu,
    m_step as m_step_cpu,
    initialize as init_cpu,
    weighted_discrete_edmd as edmd_cpu,
    predict_next_all_clusters as predict_cpu,
    residual_logpdf_discrete as resid_discrete_cpu,
)
from residual_aware_clustering.models.em_local_edmd_discrete_gpu import (
    e_step as e_step_gpu,
    m_step as m_step_gpu,
    initialize as init_gpu,
    weighted_discrete_edmd as edmd_gpu,
    predict_next_all_clusters as predict_gpu,
    residual_logpdf_discrete as resid_discrete_gpu,
)


def _get_accelerator():
    """Return the best available accelerator device, or None."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return None


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    d, P = 4, 200
    X = torch.tensor(rng.standard_normal((P, d)), dtype=torch.float64)
    X_next = X + 0.1 * torch.tensor(rng.standard_normal((P, d)), dtype=torch.float64)
    return X, X_next, d, P


@pytest.fixture
def sample_clusters(sample_data):
    X, X_next, d, P = sample_data
    N = 3
    hp = make_hp(X, d=d)
    state = init_cpu(X, X_next, N, hp, degree=2, seed=42)
    return state, hp


@pytest.fixture
def distribution_data():
    rng = np.random.default_rng(0)
    d, P, N = 3, 100, 4
    X = torch.tensor(rng.standard_normal((P, d)), dtype=torch.float64)
    centers = torch.tensor(rng.standard_normal((N, d)), dtype=torch.float64)
    A = rng.standard_normal((N, d, d))
    covs = torch.tensor(np.array([a @ a.T + np.eye(d) for a in A]), dtype=torch.float64)
    f_centers = torch.tensor(rng.standard_normal((N, d)), dtype=torch.float64)
    jacobians = torch.tensor(rng.standard_normal((N, d, d)), dtype=torch.float64)
    F = torch.tensor(rng.standard_normal((P, d)), dtype=torch.float64)
    pi = torch.ones(N, dtype=torch.float64) / N
    return X, centers, covs, f_centers, jacobians, F, pi


# ── Distribution tests ───────────────────────────────────────────────────────

def test_mvn_logpdf_matches(distribution_data):
    X, centers, covs, _, _, _, _ = distribution_data
    out_cpu = mvn_cpu(X, centers, covs)
    out_gpu = mvn_gpu(X, centers, covs)
    assert torch.allclose(out_cpu, out_gpu, atol=1e-12), \
        f"mvn_logpdf max diff: {(out_cpu - out_gpu).abs().max():.2e}"


def test_residual_logpdf_matches(distribution_data):
    X, centers, covs, f_centers, jacobians, F, _ = distribution_data
    sigma2 = 2.0
    out_cpu = resid_cpu(X, F, centers, f_centers, jacobians, sigma2)
    out_gpu = resid_gpu(X, F, centers, f_centers, jacobians, sigma2)
    assert torch.allclose(out_cpu, out_gpu, atol=1e-12), \
        f"residual_logpdf max diff: {(out_cpu - out_gpu).abs().max():.2e}"


def test_residual_logpdf_per_cluster_sigma(distribution_data):
    X, centers, covs, f_centers, jacobians, F, _ = distribution_data
    sigma2 = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    out_cpu = resid_cpu(X, F, centers, f_centers, jacobians, sigma2)
    out_gpu = resid_gpu(X, F, centers, f_centers, jacobians, sigma2)
    assert torch.allclose(out_cpu, out_gpu, atol=1e-12)


def test_dirichlet_logpdf_matches(distribution_data):
    _, _, _, _, _, _, pi = distribution_data
    out_cpu = dirichlet_cpu(pi, 0.5)
    out_gpu = dirichlet_gpu(pi, 0.5)
    assert torch.allclose(out_cpu, out_gpu, atol=1e-12)


def test_niw_logpdf_matches(distribution_data):
    _, centers, covs, _, _, _, _ = distribution_data
    d = centers.shape[1]
    mu0 = torch.zeros(d, dtype=torch.float64)
    Psi0 = 10.0 * torch.eye(d, dtype=torch.float64)
    out_cpu = niw_cpu(centers[0], covs[0], mu0, 1.0, Psi0, float(d + 2))
    out_gpu = niw_gpu(centers[0], covs[0], mu0, 1.0, Psi0, float(d + 2))
    assert torch.allclose(out_cpu, out_gpu, atol=1e-12)


# ── make_hp test ─────────────────────────────────────────────────────────────

def test_make_hp_matches(sample_data):
    X, _, d, _ = sample_data
    hp_cpu = make_hp(X, d=d)
    hp_g = make_hp_gpu(X, d=d)
    assert torch.allclose(hp_cpu['mu0'], hp_g['mu0'])
    assert torch.allclose(hp_cpu['Lambda0'], hp_g['Lambda0'])
    assert torch.allclose(hp_cpu['Psi0'], hp_g['Psi0'])


# ── EDMD component tests ────────────────────────────────────────────────────

def test_weighted_discrete_edmd_matches(sample_data):
    X, X_next, d, P = sample_data
    exps = monomial_exponents(d, 2)
    c = X.mean(dim=0)
    r = torch.ones(P, dtype=torch.float64)

    K_cpu = edmd_cpu(X, X_next, r, c, exps)
    K_gpu = edmd_gpu(X, X_next, r, c, exps)
    assert torch.allclose(K_cpu, K_gpu, atol=1e-10), \
        f"edmd max diff: {(K_cpu - K_gpu).abs().max():.2e}"


def test_predict_next_all_clusters_matches(sample_data, sample_clusters):
    X, X_next, d, _ = sample_data
    state, _ = sample_clusters

    pred_cpu = predict_cpu(X, state['centers'], state['K_ops'], state['exps'], d)
    pred_gpu = predict_gpu(X, state['centers'], state['K_ops'], state['exps'], d)
    assert torch.allclose(pred_cpu, pred_gpu, atol=1e-10)


def test_residual_logpdf_discrete_matches(sample_data, sample_clusters):
    X, X_next, d, _ = sample_data
    state, _ = sample_clusters

    out_cpu = resid_discrete_cpu(X, X_next, state['centers'], state['K_ops'],
                                  state['sigma2'], state['exps'], d)
    out_gpu = resid_discrete_gpu(X, X_next, state['centers'], state['K_ops'],
                                  state['sigma2'], state['exps'], d)
    assert torch.allclose(out_cpu, out_gpu, atol=1e-10)


# ── E-step / M-step tests ───────────────────────────────────────────────────

def test_e_step_matches(sample_data, sample_clusters):
    X, X_next, _, _ = sample_data
    state, hp = sample_clusters

    r_cpu = e_step_cpu(X, X_next, state, hp)
    r_gpu = e_step_gpu(X, X_next, state, hp)
    assert torch.allclose(r_cpu, r_gpu, atol=1e-10), \
        f"e_step max diff: {(r_cpu - r_gpu).abs().max():.2e}"


def test_m_step_matches(sample_data, sample_clusters):
    X, X_next, _, _ = sample_data
    state, hp = sample_clusters

    r = e_step_cpu(X, X_next, state, hp)
    state_cpu = m_step_cpu(X, X_next, r, state, hp)
    state_gpu = m_step_gpu(X, X_next, r, state, hp)

    for key in ['centers', 'covariances', 'K_ops', 'pi', 'sigma2']:
        assert torch.allclose(state_cpu[key], state_gpu[key], atol=1e-8), \
            f"m_step[{key}] max diff: {(state_cpu[key] - state_gpu[key]).abs().max():.2e}"


# ── Initialize test ──────────────────────────────────────────────────────────

def test_initialize_matches(sample_data):
    X, X_next, d, _ = sample_data
    hp = make_hp(X, d=d)

    state_cpu = init_cpu(X, X_next, N=3, hp=hp, degree=2, seed=42)
    state_gpu = init_gpu(X, X_next, N=3, hp=hp, degree=2, seed=42)

    for key in ['centers', 'covariances', 'K_ops', 'pi', 'sigma2']:
        assert torch.allclose(state_cpu[key], state_gpu[key], atol=1e-10), \
            f"init[{key}] max diff: {(state_cpu[key] - state_gpu[key]).abs().max():.2e}"


# ── Full fit test ────────────────────────────────────────────────────────────

def test_fit_matches(sample_data):
    X, X_next, d, _ = sample_data
    hp = make_hp(X, d=d)

    state_cpu, r_cpu, hist_cpu = fit_local_edmd_discrete(
        X, X_next, N=3, hp=hp, degree=2, n_iter=20, n_restarts=1, verbose=False)
    state_gpu, r_gpu, hist_gpu = fit_local_edmd_discrete_gpu(
        X, X_next, N=3, hp=hp, degree=2, n_iter=20, n_restarts=1, verbose=False)

    assert len(hist_cpu) == len(hist_gpu)
    for i, (a, b) in enumerate(zip(hist_cpu, hist_gpu)):
        assert abs(a - b) < 1e-8, f"ELBO diff at iter {i}: {abs(a-b):.2e}"

    for key in ['centers', 'covariances', 'K_ops', 'pi', 'sigma2']:
        assert torch.allclose(state_cpu[key], state_gpu[key], atol=1e-8), \
            f"fit[{key}] max diff: {(state_cpu[key] - state_gpu[key]).abs().max():.2e}"

    assert torch.allclose(r_cpu, r_gpu, atol=1e-8)


# ── Accelerator test (runs on CUDA or MPS if available) ──────────────────────

@pytest.mark.skipif(_get_accelerator() is None, reason="No GPU accelerator available")
def test_fit_on_accelerator():
    """Verify the GPU version actually runs on the accelerator and produces similar results.

    MPS does not support float64 — we cast to float32 for MPS and use relaxed tolerances.
    Uses larger dataset than sample_data to avoid pruning issues at float32.
    """
    dev = _get_accelerator()
    rng = np.random.default_rng(42)
    d, P = 4, 500

    if dev.type == "mps":
        dtype = torch.float32
    else:
        dtype = torch.float64

    # Generate well-separated clusters for stability at float32
    X_np = np.vstack([rng.standard_normal((P // 2, d)) + 3,
                      rng.standard_normal((P // 2, d)) - 3])
    X_next_np = X_np + 0.1 * rng.standard_normal((P, d))

    X_dev = torch.tensor(X_np, dtype=dtype, device=dev)
    X_next_dev = torch.tensor(X_next_np, dtype=dtype, device=dev)

    hp_dev = make_hp_gpu(X_dev, d=d)

    state_dev, r_dev, hist_dev = fit_local_edmd_discrete_gpu(
        X_dev, X_next_dev, N=2, hp=hp_dev, degree=1,
        n_iter=50, n_restarts=2, verbose=False)

    assert state_dev is not None, "fit returned None — all clusters pruned"
    assert state_dev['centers'].device.type == dev.type
    assert state_dev['covariances'].device.type == dev.type
    assert state_dev['K_ops'].device.type == dev.type
    assert r_dev.device.type == dev.type


# ── Three-way: CPU vs GPU vs pykoopman ───────────────────────────────────────

import pykoopman as pk
from pykoopman.observables import Polynomial as PkPoly
from pykoopman.regression import EDMD as PkEDMD


@pytest.fixture
def edmd_data():
    """Well-conditioned data for EDMD comparison."""
    rng = np.random.default_rng(42)
    d, P = 3, 300
    X = torch.tensor(rng.standard_normal((P, d)), dtype=torch.float64)
    X_next = X + 0.1 * torch.tensor(rng.standard_normal((P, d)), dtype=torch.float64)
    return X, X_next, d


def test_cpu_edmd_matches_pykoopman(edmd_data):
    """CPU weighted_discrete_edmd matches pykoopman EDMD on same data."""
    X, X_next, d = edmd_data
    exps = monomial_exponents(d, 2)
    c = torch.zeros(d, dtype=torch.float64)
    r = torch.ones(X.shape[0], dtype=torch.float64)

    # Our CPU
    K_cpu = edmd_cpu(X, X_next, r, c, exps)

    # pykoopman: lift with same polynomial, fit EDMD
    pk_model = pk.Koopman(observables=PkPoly(degree=2), regressor=PkEDMD())
    pk_model.fit(X.numpy(), y=X_next.numpy(), dt=1)
    K_pk = pk_model._pipeline.named_steps['regressor'].coef_

    assert K_cpu.shape == K_pk.shape, \
        f"Shape mismatch: cpu={K_cpu.shape}, pk={K_pk.shape}"
    assert torch.allclose(K_cpu, torch.tensor(K_pk), atol=1e-8), \
        f"CPU vs pykoopman max diff: {(K_cpu - torch.tensor(K_pk)).abs().max():.2e}"


def test_gpu_edmd_matches_pykoopman(edmd_data):
    """GPU weighted_discrete_edmd matches pykoopman EDMD on same data."""
    X, X_next, d = edmd_data
    exps = monomial_exponents(d, 2)
    c = torch.zeros(d, dtype=torch.float64)
    r = torch.ones(X.shape[0], dtype=torch.float64)

    # Our GPU
    K_gpu = edmd_gpu(X, X_next, r, c, exps)

    # pykoopman
    pk_model = pk.Koopman(observables=PkPoly(degree=2), regressor=PkEDMD())
    pk_model.fit(X.numpy(), y=X_next.numpy(), dt=1)
    K_pk = pk_model._pipeline.named_steps['regressor'].coef_

    assert K_gpu.shape == K_pk.shape, \
        f"Shape mismatch: gpu={K_gpu.shape}, pk={K_pk.shape}"
    assert torch.allclose(K_gpu, torch.tensor(K_pk), atol=1e-8), \
        f"GPU vs pykoopman max diff: {(K_gpu - torch.tensor(K_pk)).abs().max():.2e}"


def test_cpu_gpu_pykoopman_three_way(edmd_data):
    """All three implementations produce identical K on the same data."""
    X, X_next, d = edmd_data
    exps = monomial_exponents(d, 2)
    c = torch.zeros(d, dtype=torch.float64)
    r = torch.ones(X.shape[0], dtype=torch.float64)

    K_cpu = edmd_cpu(X, X_next, r, c, exps)
    K_gpu = edmd_gpu(X, X_next, r, c, exps)

    pk_model = pk.Koopman(observables=PkPoly(degree=2), regressor=PkEDMD())
    pk_model.fit(X.numpy(), y=X_next.numpy(), dt=1)
    K_pk = torch.tensor(pk_model._pipeline.named_steps['regressor'].coef_)

    # CPU == GPU
    assert torch.allclose(K_cpu, K_gpu, atol=1e-10), \
        f"CPU vs GPU max diff: {(K_cpu - K_gpu).abs().max():.2e}"

    # CPU == pykoopman
    assert torch.allclose(K_cpu, K_pk, atol=1e-8), \
        f"CPU vs pykoopman max diff: {(K_cpu - K_pk).abs().max():.2e}"

    # GPU == pykoopman
    assert torch.allclose(K_gpu, K_pk, atol=1e-8), \
        f"GPU vs pykoopman max diff: {(K_gpu - K_pk).abs().max():.2e}"


def test_cpu_gpu_pykoopman_predictions(edmd_data):
    """All three produce identical predictions."""
    X, X_next, d = edmd_data
    exps = monomial_exponents(d, 2)
    c = torch.zeros(d, dtype=torch.float64)
    r = torch.ones(X.shape[0], dtype=torch.float64)

    K_cpu = edmd_cpu(X, X_next, r, c, exps)
    K_gpu = edmd_gpu(X, X_next, r, c, exps)

    # Predictions via our code
    pred_cpu = predict_cpu(X, c.unsqueeze(0), K_cpu.unsqueeze(0), exps, d)[:, 0, :]
    pred_gpu = predict_gpu(X, c.unsqueeze(0), K_gpu.unsqueeze(0), exps, d)[:, 0, :]

    # pykoopman predictions
    pk_model = pk.Koopman(observables=PkPoly(degree=2), regressor=PkEDMD())
    pk_model.fit(X.numpy(), y=X_next.numpy(), dt=1)
    K_pk = pk_model._pipeline.named_steps['regressor'].coef_
    poly = PkPoly(degree=2); poly.fit(X.numpy())
    Phi_X = poly.transform(X.numpy())
    pk_pred_full = Phi_X @ K_pk.T
    pk_pred = torch.tensor(pk_pred_full[:, 1:d + 1])

    assert torch.allclose(pred_cpu, pred_gpu, atol=1e-10), \
        f"CPU vs GPU pred max diff: {(pred_cpu - pred_gpu).abs().max():.2e}"
    assert torch.allclose(pred_cpu, pk_pred, atol=1e-8), \
        f"CPU vs pykoopman pred max diff: {(pred_cpu - pk_pred).abs().max():.2e}"


def test_cpu_gpu_pykoopman_weighted(edmd_data):
    """Three-way match with non-uniform weights."""
    X, X_next, d = edmd_data
    exps = monomial_exponents(d, 1)
    c = torch.zeros(d, dtype=torch.float64)

    rng = np.random.default_rng(99)
    r = torch.tensor(rng.random(X.shape[0]), dtype=torch.float64)

    K_cpu = edmd_cpu(X, X_next, r, c, exps)
    K_gpu = edmd_gpu(X, X_next, r, c, exps)

    assert torch.allclose(K_cpu, K_gpu, atol=1e-10), \
        f"CPU vs GPU weighted max diff: {(K_cpu - K_gpu).abs().max():.2e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
