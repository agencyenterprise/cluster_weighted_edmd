"""
Integration tests: verify generic EM pipeline with polynomial models
matches the existing hardcoded pipelines exactly.
"""

import torch
import numpy as np
import pytest

from residual_aware_clustering import make_hp
from residual_aware_clustering.models.em_local_edmd import monomial_exponents, monomials
from residual_aware_clustering.models.em_local_edmd_discrete_gpu import (
    fit as fit_discrete_gpu,
    weighted_discrete_edmd as edmd_ref,
    predict_next_all_clusters as predict_ref,
    e_step as e_step_ref,
    m_step as m_step_ref,
    initialize as init_ref,
)
from residual_aware_clustering.models.em_local_edmd import (
    weighted_continuous_edmd as cont_edmd_ref,
    predict_f_all_clusters as predict_f_ref,
)
from residual_aware_clustering.models.em_local_edmd import fit as fit_continuous_ref
from residual_aware_clustering.models.experimental.polynomial_discrete import PolynomialDiscreteEDMD
from residual_aware_clustering.models.experimental.polynomial_continuous import PolynomialContinuousEDMD
from residual_aware_clustering.models.experimental.generic_em import (
    fit as generic_fit,
    predict_all,
    residual_logpdf,
    e_step as e_step_generic,
    m_step as m_step_generic,
    initialize as init_generic,
    compute_elbo as elbo_generic,
)
from residual_aware_clustering.models.experimental.local_model import LocalModel


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    d, P = 4, 200
    X = torch.tensor(rng.standard_normal((P, d)), dtype=torch.float64)
    X_next = X + 0.1 * torch.tensor(rng.standard_normal((P, d)), dtype=torch.float64)
    return X, X_next, d, P


@pytest.fixture
def larger_data():
    rng = np.random.default_rng(99)
    d, P = 4, 500
    X = torch.tensor(rng.standard_normal((P, d)), dtype=torch.float64)
    X_next = X + 0.1 * torch.tensor(rng.standard_normal((P, d)), dtype=torch.float64)
    return X, X_next, d, P


# ── Protocol compliance ──────────────────────────────────────────────────────

def test_polynomial_discrete_is_local_model():
    model = PolynomialDiscreteEDMD(degree=2)
    assert isinstance(model, LocalModel)


# ── Component: fit matches ───────────────────────────────────────────────────

def test_discrete_fit_matches(sample_data):
    X, X_next, d, P = sample_data
    exps = monomial_exponents(d, 2)
    c = X.mean(dim=0)
    r = torch.ones(P, dtype=torch.float64)

    K_ref = edmd_ref(X, X_next, r, c, exps)

    model = PolynomialDiscreteEDMD(degree=2)
    model.fit(X, X_next, r, c)
    K_new = model.state_dict()['K']

    assert torch.allclose(K_ref, K_new, atol=1e-12), \
        f"K max diff: {(K_ref - K_new).abs().max():.2e}"


def test_discrete_fit_weighted(sample_data):
    X, X_next, d, P = sample_data
    exps = monomial_exponents(d, 2)
    c = X[:50].mean(dim=0)
    rng = np.random.default_rng(7)
    r = torch.tensor(rng.random(P), dtype=torch.float64)

    K_ref = edmd_ref(X, X_next, r, c, exps)

    model = PolynomialDiscreteEDMD(degree=2)
    model.fit(X, X_next, r, c)
    K_new = model.state_dict()['K']

    assert torch.allclose(K_ref, K_new, atol=1e-12)


# ── Component: predict matches ───────────────────────────────────────────────

def test_discrete_predict_matches(sample_data):
    X, X_next, d, P = sample_data
    hp = make_hp(X, d=d)
    state_ref = init_ref(X, X_next, N=3, hp=hp, degree=2, seed=42)

    pred_ref = predict_ref(X, state_ref['centers'], state_ref['K_ops'],
                           state_ref['exps'], d)

    # Build equivalent generic models
    models = []
    for k in range(3):
        m = PolynomialDiscreteEDMD(degree=2)
        m._d = d
        m._exps = state_ref['exps']
        m._K = state_ref['K_ops'][k]
        models.append(m)

    pred_gen = predict_all(X, state_ref['centers'], models, d)
    assert torch.allclose(pred_ref, pred_gen, atol=1e-12)


# ── Component: e_step matches ────────────────────────────────────────────────

def test_e_step_matches(sample_data):
    X, X_next, d, P = sample_data
    hp = make_hp(X, d=d)
    state_ref = init_ref(X, X_next, N=3, hp=hp, degree=2, seed=42)

    r_ref = e_step_ref(X, X_next, state_ref, hp)

    # Build generic state from reference
    models = []
    for k in range(state_ref['N']):
        m = PolynomialDiscreteEDMD(degree=2)
        m._d = d
        m._exps = state_ref['exps']
        m._K = state_ref['K_ops'][k]
        models.append(m)

    state_gen = dict(state_ref)
    state_gen['models'] = models

    r_gen = e_step_generic(X, X_next, state_gen, hp)
    assert torch.allclose(r_ref, r_gen, atol=1e-10), \
        f"e_step max diff: {(r_ref - r_gen).abs().max():.2e}"


# ── Component: m_step matches ────────────────────────────────────────────────

def test_m_step_matches(sample_data):
    X, X_next, d, P = sample_data
    hp = make_hp(X, d=d)
    state_ref = init_ref(X, X_next, N=3, hp=hp, degree=2, seed=42)
    r = e_step_ref(X, X_next, state_ref, hp)

    state_ref_new = m_step_ref(X, X_next, r, state_ref, hp)

    # Build generic state
    models = []
    for k in range(state_ref['N']):
        m = PolynomialDiscreteEDMD(degree=2)
        m._d = d
        m._exps = state_ref['exps']
        m._K = state_ref['K_ops'][k]
        models.append(m)

    state_gen = dict(state_ref)
    state_gen['models'] = models
    state_gen_new = m_step_generic(X, X_next, r, state_gen, hp)

    for key in ['centers', 'covariances', 'pi', 'sigma2']:
        assert torch.allclose(state_ref_new[key], state_gen_new[key], atol=7e-6), \
            f"m_step[{key}] max diff: {(state_ref_new[key] - state_gen_new[key]).abs().max():.2e}"

    # Compare K_ops
    K_ref = state_ref_new['K_ops']
    K_gen = torch.stack([m.state_dict()['K'] for m in state_gen_new['models']])
    assert torch.allclose(K_ref, K_gen, atol=7e-6), \
        f"m_step[K_ops] max diff: {(K_ref - K_gen).abs().max():.2e}"


# ── Initialize matches ───────────────────────────────────────────────────────

def test_initialize_matches(sample_data):
    X, X_next, d, P = sample_data
    hp = make_hp(X, d=d)

    state_ref = init_ref(X, X_next, N=3, hp=hp, degree=2, seed=42)

    model_proto = PolynomialDiscreteEDMD(degree=2)
    state_gen = init_generic(X, X_next, N=3, hp=hp,
                             model_prototype=model_proto, seed=42)

    for key in ['centers', 'covariances', 'pi', 'sigma2']:
        assert torch.allclose(state_ref[key], state_gen[key], atol=1e-10), \
            f"init[{key}] max diff: {(state_ref[key] - state_gen[key]).abs().max():.2e}"

    K_ref = state_ref['K_ops']
    K_gen = torch.stack([m.state_dict()['K'] for m in state_gen['models']])
    assert torch.allclose(K_ref, K_gen, atol=1e-10), \
        f"init[K_ops] max diff: {(K_ref - K_gen).abs().max():.2e}"

    assert state_ref['N'] == state_gen['N']


# ── Full pipeline matches ────────────────────────────────────────────────────

def test_full_pipeline_discrete_matches(sample_data):
    X, X_next, d, _ = sample_data
    hp = make_hp(X, d=d)

    state_ref, r_ref, hist_ref = fit_discrete_gpu(
        X, X_next, N=3, hp=hp, degree=2,
        n_iter=20, n_restarts=1, verbose=False)

    model_proto = PolynomialDiscreteEDMD(degree=2)
    state_gen, r_gen, hist_gen = generic_fit(
        X, X_next, N=3, hp=hp, model_prototype=model_proto,
        n_iter=20, n_restarts=1, verbose=False)

    # ELBO histories
    assert len(hist_ref) == len(hist_gen), \
        f"History lengths differ: {len(hist_ref)} vs {len(hist_gen)}"
    for i, (a, b) in enumerate(zip(hist_ref, hist_gen)):
        assert abs(a - b) < 7e-6, f"ELBO diff at iter {i}: {abs(a - b):.2e}"

    # State
    for key in ['centers', 'covariances', 'pi', 'sigma2']:
        assert torch.allclose(state_ref[key], state_gen[key], atol=7e-6), \
            f"fit[{key}] max diff: {(state_ref[key] - state_gen[key]).abs().max():.2e}"

    # Responsibilities
    assert torch.allclose(r_ref, r_gen, atol=7e-6)

    # K_ops
    K_ref = state_ref['K_ops']
    K_gen = torch.stack([m.state_dict()['K'] for m in state_gen['models']])
    assert torch.allclose(K_ref, K_gen, atol=7e-6), \
        f"fit[K_ops] max diff: {(K_ref - K_gen).abs().max():.2e}"


def test_full_pipeline_multiple_restarts(larger_data):
    X, X_next, d, _ = larger_data
    hp = make_hp(X, d=d)

    state_ref, r_ref, hist_ref = fit_discrete_gpu(
        X, X_next, N=3, hp=hp, degree=2,
        n_iter=30, n_restarts=3, verbose=False)

    model_proto = PolynomialDiscreteEDMD(degree=2)
    state_gen, r_gen, hist_gen = generic_fit(
        X, X_next, N=3, hp=hp, model_prototype=model_proto,
        n_iter=30, n_restarts=3, verbose=False)

    assert len(hist_ref) == len(hist_gen)
    for key in ['centers', 'covariances', 'pi', 'sigma2']:
        assert torch.allclose(state_ref[key], state_gen[key], atol=7e-6)


# ── Clone and state_dict ─────────────────────────────────────────────────────

def test_clone_produces_independent_instance(sample_data):
    X, X_next, d, P = sample_data
    c = X.mean(dim=0)
    r = torch.ones(P, dtype=torch.float64)

    model = PolynomialDiscreteEDMD(degree=3)
    model.fit(X, X_next, r, c)

    clone = model.clone()
    assert clone._K is None  # fresh, not fitted
    assert clone.degree == model.degree


def test_state_dict_roundtrip(sample_data):
    X, X_next, d, P = sample_data
    c = X.mean(dim=0)
    r = torch.ones(P, dtype=torch.float64)

    model = PolynomialDiscreteEDMD(degree=2)
    model.fit(X, X_next, r, c)
    pred1 = model.predict(X, c)

    state = model.state_dict()
    model2 = PolynomialDiscreteEDMD(degree=2)
    model2.load_state_dict(state)
    pred2 = model2.predict(X, c)

    assert torch.allclose(pred1, pred2, atol=1e-14)


def test_to_device_dtype(sample_data):
    X, X_next, d, P = sample_data
    c = X.mean(dim=0)
    r = torch.ones(P, dtype=torch.float64)

    model = PolynomialDiscreteEDMD(degree=2)
    model.fit(X, X_next, r, c)

    model.to(torch.device('cpu'), torch.float32)
    assert model._K.dtype == torch.float32


# ── Continuous EDMD tests ────────────────────────────────────────────────────

@pytest.fixture
def continuous_data():
    """Data for continuous EDMD: X (states) and F (velocities)."""
    rng = np.random.default_rng(55)
    d, P = 3, 300
    X = torch.tensor(rng.standard_normal((P, d)), dtype=torch.float64)
    F = torch.tensor(rng.standard_normal((P, d)), dtype=torch.float64)
    return X, F, d, P


def test_continuous_fit_matches(continuous_data):
    X, F, d, P = continuous_data
    exps = monomial_exponents(d, 2)
    c = X.mean(dim=0)
    r = torch.ones(P, dtype=torch.float64)

    M_ref = cont_edmd_ref(X, F, r, c, exps, ridge=1e-4)

    model = PolynomialContinuousEDMD(degree=2, ridge=1e-4)
    model.fit(X, F, r, c)
    M_new = model.state_dict()['M']

    assert torch.allclose(M_ref, M_new, atol=1e-12), \
        f"M max diff: {(M_ref - M_new).abs().max():.2e}"


def test_continuous_predict_matches(continuous_data):
    X, F, d, P = continuous_data
    exps = monomial_exponents(d, 2)
    hp = make_hp(X, d=d)

    # Fit reference models
    from residual_aware_clustering.models.em_local_edmd import initialize as init_cont_ref
    state_ref = init_cont_ref(X, F, N=3, hp=hp, degree=2, seed=42)

    F_pred_ref = predict_f_ref(X, state_ref['centers'], state_ref['M_ops'],
                                state_ref['exps'], d)

    # Build equivalent generic models
    models = []
    for k in range(3):
        m = PolynomialContinuousEDMD(degree=2, ridge=1e-4)
        m._d = d
        m._exps = state_ref['exps']
        m._M = state_ref['M_ops'][k]
        models.append(m)

    F_pred_gen = predict_all(X, state_ref['centers'], models, d)
    assert torch.allclose(F_pred_ref, F_pred_gen, atol=1e-12)


def test_continuous_protocol():
    model = PolynomialContinuousEDMD(degree=2)
    assert isinstance(model, LocalModel)


def test_full_pipeline_continuous_matches(continuous_data):
    X, F, d, _ = continuous_data
    hp = make_hp(X, d=d)

    state_ref, r_ref, hist_ref = fit_continuous_ref(
        X, F, N=3, hp=hp, degree=2,
        n_iter=20, n_restarts=1, verbose=False)

    model_proto = PolynomialContinuousEDMD(degree=2, ridge=1e-4)
    state_gen, r_gen, hist_gen = generic_fit(
        X, F, N=3, hp=hp, model_prototype=model_proto,
        n_iter=20, n_restarts=1, verbose=False)

    # ELBO histories should match
    assert len(hist_ref) == len(hist_gen), \
        f"History lengths differ: {len(hist_ref)} vs {len(hist_gen)}"
    for i, (a, b) in enumerate(zip(hist_ref, hist_gen)):
        assert abs(a - b) < 7e-6, f"ELBO diff at iter {i}: {abs(a - b):.2e}"

    # State
    for key in ['centers', 'covariances', 'pi', 'sigma2']:
        assert torch.allclose(state_ref[key], state_gen[key], atol=7e-6), \
            f"fit[{key}] max diff: {(state_ref[key] - state_gen[key]).abs().max():.2e}"

    # M_ops
    M_ref = state_ref['M_ops']
    M_gen = torch.stack([m.state_dict()['M'] for m in state_gen['models']])
    assert torch.allclose(M_ref, M_gen, atol=7e-6), \
        f"fit[M_ops] max diff: {(M_ref - M_gen).abs().max():.2e}"


# ── Sparse E-step tests ─────────────────────────────────────────────────────

@pytest.fixture
def sparse_data():
    """Larger dataset with well-separated clusters for sparse testing."""
    rng = np.random.default_rng(123)
    d, P = 4, 500
    N = 5
    # Create well-separated cluster centers
    centers = torch.tensor(rng.standard_normal((N, d)) * 5, dtype=torch.float64)
    # Generate points near each center
    labels = rng.integers(0, N, size=P)
    X = centers[labels] + torch.tensor(rng.standard_normal((P, d)) * 0.5,
                                        dtype=torch.float64)
    X_next = X + 0.1 * torch.tensor(rng.standard_normal((P, d)),
                                     dtype=torch.float64)
    return X, X_next, d, P, N


@pytest.fixture
def sparse_state(sparse_data):
    """Fitted generic EM state for sparse tests."""
    X, X_next, d, P, N = sparse_data
    hp = make_hp(X, d=d)
    model_proto = PolynomialDiscreteEDMD(degree=1)
    state = init_generic(X, X_next, N=N, hp=hp, model_prototype=model_proto,
                         seed=42, max_gmm_samples=500)
    return state


def test_sparse_residual_logpdf_matches_full(sparse_data, sparse_state):
    """Sparse residual_logpdf with top_k=N should exactly match full."""
    X, X_next, d, P, N = sparse_data
    state = sparse_state

    from residual_aware_clustering.models.distributions import mvn_logpdf_batch
    log_prox = mvn_logpdf_batch(X, state['centers'], state['covariances'])

    # Full mode
    log_resid_full = residual_logpdf(
        X, X_next, state['centers'], state['models'], state['sigma2'], d)

    # Sparse with top_k = N (should be identical)
    log_resid_sparse = residual_logpdf(
        X, X_next, state['centers'], state['models'], state['sigma2'], d,
        log_prox=log_prox, sparse_top_k=N)

    assert torch.allclose(log_resid_full, log_resid_sparse, atol=1e-12), \
        f"Max diff: {(log_resid_full - log_resid_sparse).abs().max():.2e}"


def test_sparse_e_step_assignments_match(sparse_data, sparse_state):
    """Sparse E-step assignments should match full for well-separated clusters."""
    X, X_next, d, P, N = sparse_data
    state = sparse_state

    hp_full = make_hp(X, d=d)
    hp_sparse = make_hp(X, d=d)
    hp_sparse['sparse_top_k'] = 3

    r_full = e_step_generic(X, X_next, state, hp_full)
    r_sparse = e_step_generic(X, X_next, state, hp_sparse)

    # Hard assignments should match for well-separated data
    assign_full = r_full.argmax(dim=1)
    assign_sparse = r_sparse.argmax(dim=1)
    match_rate = (assign_full == assign_sparse).float().mean().item()

    assert match_rate > 0.99, \
        f"Assignment match rate: {match_rate:.4f} (expected >0.99)"


def test_sparse_top_k_equals_N_is_exact(sparse_data, sparse_state):
    """top_k=N should give numerically identical responsibilities as full."""
    X, X_next, d, P, N = sparse_data
    state = sparse_state

    hp_full = make_hp(X, d=d)
    hp_sparse = make_hp(X, d=d)
    hp_sparse['sparse_top_k'] = N

    r_full = e_step_generic(X, X_next, state, hp_full)
    r_sparse = e_step_generic(X, X_next, state, hp_sparse)

    assert torch.allclose(r_full, r_sparse, atol=1e-12), \
        f"Max diff: {(r_full - r_sparse).abs().max():.2e}"


def test_sparse_fit_converges(sparse_data):
    """Sparse EM should converge (ELBO increases)."""
    X, X_next, d, P, N = sparse_data
    hp = make_hp(X, d=d)
    hp['sparse_top_k'] = 3

    model_proto = PolynomialDiscreteEDMD(degree=1)
    state, r, elbos = generic_fit(
        X, X_next, N=N, hp=hp, model_prototype=model_proto,
        n_iter=20, n_restarts=1, verbose=False)

    assert elbos is not None, "fit returned None history"
    assert len(elbos) > 0
    assert state['N'] > 0
    # ELBO should generally increase (allow small dips from sparse approximation)
    if len(elbos) > 5:
        assert elbos[-1] > elbos[2], \
            f"ELBO did not increase: {elbos[2]:.2f} → {elbos[-1]:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
