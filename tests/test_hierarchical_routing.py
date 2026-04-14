"""
Validate that hierarchical routing does NOT mix σ² levels.

Creates a scenario with:
  - Good parent clusters (low σ²)
  - Bad parent clusters (high σ²) that get refined into children
  - Children have σ²≈0 (overfit)

Tests that:
  1. Points in good parents use parent model, report parent σ²
  2. Points in bad parents use child model, report PARENT σ² (not child's)
  3. σ²=0 from parent level vs σ²=0 from child level are never mixed
  4. Predictions for good-parent points are identical before/after refinement
"""

import torch
import numpy as np
import pytest

from residual_aware_clustering import make_hp
from residual_aware_clustering.models.experimental.polynomial_discrete import (
    PolynomialDiscreteEDMD,
)
from residual_aware_clustering.models.experimental.generic_em import (
    fit as generic_fit,
    e_step,
)
from residual_aware_clustering.models.experimental.hierarchical_refine import (
    identify_refinement_targets,
    refine_clusters,
    HierarchicalState,
)


@pytest.fixture
def two_tier_data():
    """Create data with clearly separated good and bad clusters.

    Good clusters: tight, low noise → will have low σ²
    Bad clusters: spread, high noise → will have high σ²
    """
    rng = np.random.default_rng(42)
    d = 4
    N_good = 300  # points in good clusters
    N_bad = 300   # points in bad clusters

    # Good clusters: 2 tight clusters, very predictable dynamics
    center_g1 = torch.tensor([10.0, 0, 0, 0], dtype=torch.float64)
    center_g2 = torch.tensor([0, 10.0, 0, 0], dtype=torch.float64)
    X_g1 = center_g1 + torch.tensor(rng.standard_normal((N_good // 2, d)) * 0.1,
                                     dtype=torch.float64)
    X_g2 = center_g2 + torch.tensor(rng.standard_normal((N_good // 2, d)) * 0.1,
                                     dtype=torch.float64)
    # Simple linear dynamics for good clusters
    Y_g1 = X_g1 * 0.99 + 0.01
    Y_g2 = X_g2 * 0.99 + 0.01

    # Bad clusters: 2 spread clusters, noisy dynamics
    center_b1 = torch.tensor([0, 0, 10.0, 0], dtype=torch.float64)
    center_b2 = torch.tensor([0, 0, 0, 10.0], dtype=torch.float64)
    X_b1 = center_b1 + torch.tensor(rng.standard_normal((N_bad // 2, d)) * 2.0,
                                     dtype=torch.float64)
    X_b2 = center_b2 + torch.tensor(rng.standard_normal((N_bad // 2, d)) * 2.0,
                                     dtype=torch.float64)
    # Noisy dynamics for bad clusters
    Y_b1 = X_b1 * 0.8 + torch.tensor(rng.standard_normal((N_bad // 2, d)) * 1.0,
                                       dtype=torch.float64)
    Y_b2 = X_b2 * 0.8 + torch.tensor(rng.standard_normal((N_bad // 2, d)) * 1.0,
                                       dtype=torch.float64)

    X = torch.cat([X_g1, X_g2, X_b1, X_b2])
    Y = torch.cat([Y_g1, Y_g2, Y_b1, Y_b2])

    # Labels: 0,1 = good, 2,3 = bad
    labels = torch.cat([
        torch.zeros(N_good // 2, dtype=torch.long),
        torch.ones(N_good // 2, dtype=torch.long),
        torch.full((N_bad // 2,), 2, dtype=torch.long),
        torch.full((N_bad // 2,), 3, dtype=torch.long),
    ])

    return X, Y, d, labels


@pytest.fixture
def fitted_state(two_tier_data):
    """Fit a 4-cluster model on the two-tier data."""
    X, Y, d, labels = two_tier_data
    hp = make_hp(X, d=d)
    model_proto = PolynomialDiscreteEDMD(degree=1)

    state, r, elbos = generic_fit(
        X, Y, N=4, hp=hp, model_prototype=model_proto,
        n_iter=50, n_restarts=1, verbose=False)

    return state, r


def test_good_clusters_have_low_sigma2(two_tier_data, fitted_state):
    """Verify that fitting produces clusters with different sigma2 levels."""
    X, Y, d, labels = two_tier_data
    state, r = fitted_state

    sigma2 = state["sigma2"]
    print(f"\nCluster sigma2 values: {sigma2}")

    # Should have at least some variation in sigma2
    assert sigma2.max() > sigma2.min() * 2, \
        f"sigma2 range too narrow: [{sigma2.min():.4f}, {sigma2.max():.4f}]"


def test_refinement_targets_are_high_sigma2(two_tier_data, fitted_state):
    """Identify refinement targets — should be the high-sigma2 clusters."""
    state, r = fitted_state

    sigma2 = state["sigma2"]
    median_sigma = sigma2.median().item()

    targets = identify_refinement_targets(state, sigma2_threshold=median_sigma)
    assert len(targets) > 0, "No refinement targets found"

    for t in targets:
        assert sigma2[t].item() >= median_sigma, \
            f"Target {t} has sigma2={sigma2[t]:.4f} < threshold={median_sigma:.4f}"

    print(f"\nTargets: {targets}")
    print(f"Target sigma2: {[sigma2[t].item() for t in targets]}")


def test_unrefined_predictions_unchanged_after_refinement(two_tier_data, fitted_state):
    """Predictions for unrefined-parent points must be IDENTICAL before/after."""
    X, Y, d, labels = two_tier_data
    state, r = fitted_state

    sigma2 = state["sigma2"]
    median_sigma = sigma2.median().item()
    targets = identify_refinement_targets(state, sigma2_threshold=median_sigma)

    if not targets:
        pytest.skip("No refinement targets")

    hp = make_hp(X, d=d)
    model_proto = PolynomialDiscreteEDMD(degree=1)

    # Flat predictions (before refinement)
    from residual_aware_clustering.models.distributions import mvn_logpdf_batch

    log_prox = mvn_logpdf_batch(X, state["centers"], state["covariances"])
    log_pi = torch.log(state["pi"].clamp(min=1e-30)).unsqueeze(0)
    flat_assignments = (log_prox + log_pi).argmax(dim=1)

    flat_pred = torch.zeros_like(X)
    for k in range(state["N"]):
        mask = flat_assignments == k
        if mask.any():
            flat_pred[mask] = state["models"][k].predict(X[mask], state["centers"][k])

    # Refine
    h_state = refine_clusters(
        X, Y, state, r, targets,
        hp=hp, model_prototype=model_proto,
        n_subclusters=3, n_iter=20, n_restarts=1, verbose=False)

    # Hierarchical predictions
    hier_pred, parent_k, child_k, sigma2_eff = h_state.predict(X)

    # Points in unrefined parents
    refined_set = set(h_state.refined_clusters)
    is_unrefined = torch.tensor([pk.item() not in refined_set for pk in parent_k])

    n_unrefined = is_unrefined.sum().item()
    assert n_unrefined > 0, "No unrefined points"

    # Predictions must be IDENTICAL for unrefined parents
    diff = (flat_pred[is_unrefined] - hier_pred[is_unrefined]).abs().max().item()
    assert diff < 1e-12, \
        f"Unrefined predictions differ! Max diff={diff:.2e}"

    print(f"\nUnrefined points: {n_unrefined}")
    print(f"Max prediction diff for unrefined: {diff:.2e}")


def test_sigma2_eff_uses_parent_for_unrefined(two_tier_data, fitted_state):
    """sigma2_eff for unrefined parents must equal the parent's sigma2."""
    X, Y, d, labels = two_tier_data
    state, r = fitted_state

    sigma2 = state["sigma2"]
    median_sigma = sigma2.median().item()
    targets = identify_refinement_targets(state, sigma2_threshold=median_sigma)

    if not targets:
        pytest.skip("No refinement targets")

    hp = make_hp(X, d=d)
    model_proto = PolynomialDiscreteEDMD(degree=1)

    h_state = refine_clusters(
        X, Y, state, r, targets,
        hp=hp, model_prototype=model_proto,
        n_subclusters=3, n_iter=20, n_restarts=1, verbose=False)

    parent_k, child_k, sigma2_eff = h_state.assign(X)

    refined_set = set(h_state.refined_clusters)

    for i in range(len(X)):
        pk = parent_k[i].item()
        ck = child_k[i].item()
        s2 = sigma2_eff[i].item()
        parent_s2 = state["sigma2"][pk].item()

        if pk not in refined_set:
            # Unrefined: sigma2_eff must equal parent sigma2
            assert abs(s2 - parent_s2) < 1e-12, \
                f"Point {i}: unrefined parent {pk}, sigma2_eff={s2:.6f} " \
                f"!= parent sigma2={parent_s2:.6f}"
            # child_k must be -1
            assert ck == -1, \
                f"Point {i}: unrefined parent {pk} but child_k={ck}"


def test_sigma2_eff_for_refined_uses_child(two_tier_data, fitted_state):
    """sigma2_eff for refined parents uses child's sigma2 (current behavior).

    NOTE: This is what h_state.assign() currently does. The gating code
    should NOT use this — it should use parent sigma2 instead.
    """
    X, Y, d, labels = two_tier_data
    state, r = fitted_state

    sigma2 = state["sigma2"]
    median_sigma = sigma2.median().item()
    targets = identify_refinement_targets(state, sigma2_threshold=median_sigma)

    if not targets:
        pytest.skip("No refinement targets")

    hp = make_hp(X, d=d)
    model_proto = PolynomialDiscreteEDMD(degree=1)

    h_state = refine_clusters(
        X, Y, state, r, targets,
        hp=hp, model_prototype=model_proto,
        n_subclusters=3, n_iter=20, n_restarts=1, verbose=False)

    parent_k, child_k, sigma2_eff = h_state.assign(X)

    refined_set = set(h_state.refined_clusters)

    for i in range(len(X)):
        pk = parent_k[i].item()
        ck = child_k[i].item()
        s2 = sigma2_eff[i].item()
        parent_s2 = state["sigma2"][pk].item()

        if pk in refined_set:
            # Refined: child_k should be >= 0
            assert ck >= 0, \
                f"Point {i}: refined parent {pk} but child_k={ck}"
            # sigma2_eff comes from child (may differ from parent)
            child_s2 = h_state.children[pk]["sigma2"][ck].item()
            assert abs(s2 - child_s2) < 1e-12, \
                f"Point {i}: sigma2_eff={s2:.6f} != child sigma2={child_s2:.6f}"


def test_parent_sigma2_always_available_for_gating(two_tier_data, fitted_state):
    """Gating must always have access to parent sigma2, never child sigma2.

    Simulates the gating logic: for each point, get parent_k from
    h_state.predict(), then look up surrogate_state["sigma2"][parent_k].
    This must return the PARENT's original sigma2, not the child's.
    """
    X, Y, d, labels = two_tier_data
    state, r = fitted_state

    sigma2 = state["sigma2"]
    median_sigma = sigma2.median().item()
    targets = identify_refinement_targets(state, sigma2_threshold=median_sigma)

    if not targets:
        pytest.skip("No refinement targets")

    hp = make_hp(X, d=d)
    model_proto = PolynomialDiscreteEDMD(degree=1)

    h_state = refine_clusters(
        X, Y, state, r, targets,
        hp=hp, model_prototype=model_proto,
        n_subclusters=3, n_iter=20, n_restarts=1, verbose=False)

    _, parent_k, child_k, _ = h_state.predict(X)

    refined_set = set(h_state.refined_clusters)

    n_good_parent_zero = 0
    n_bad_parent_reported_as_zero = 0

    for i in range(len(X)):
        pk = parent_k[i].item()
        ck = child_k[i].item()

        # Gating signal: ALWAYS parent sigma2
        gate_sigma2 = state["sigma2"][pk].item()

        if pk not in refined_set and gate_sigma2 < 1e-6:
            n_good_parent_zero += 1
        elif pk in refined_set and gate_sigma2 < 1e-6:
            # BUG: a refined parent should NOT have sigma2 ≈ 0
            # because it was selected for refinement due to HIGH sigma2
            n_bad_parent_reported_as_zero += 1

    print(f"\nGood parents with σ²≈0: {n_good_parent_zero}")
    print(f"Refined parents falsely reporting σ²≈0: {n_bad_parent_reported_as_zero}")

    assert n_bad_parent_reported_as_zero == 0, \
        f"BUG: {n_bad_parent_reported_as_zero} refined-parent points " \
        f"have gate_sigma2 ≈ 0 (should have high sigma2)"


def test_no_level_mixing_in_sigma2_groups(two_tier_data, fitted_state):
    """When grouping by parent sigma2, unrefined and refined points must
    land in DIFFERENT groups (because refined parents have high sigma2).

    This catches the bug where child sigma2=0 gets mixed with parent sigma2=0.
    """
    X, Y, d, labels = two_tier_data
    state, r = fitted_state

    sigma2 = state["sigma2"]
    median_sigma = sigma2.median().item()
    targets = identify_refinement_targets(state, sigma2_threshold=median_sigma)

    if not targets:
        pytest.skip("No refinement targets")

    hp = make_hp(X, d=d)
    model_proto = PolynomialDiscreteEDMD(degree=1)

    h_state = refine_clusters(
        X, Y, state, r, targets,
        hp=hp, model_prototype=model_proto,
        n_subclusters=3, n_iter=20, n_restarts=1, verbose=False)

    _, parent_k, child_k, _ = h_state.predict(X)
    refined_set = set(h_state.refined_clusters)

    # Group points by parent sigma2
    parent_sigma2_per_point = state["sigma2"][parent_k]

    # Sort and split into 2 groups (low and high sigma2)
    sorted_idx = parent_sigma2_per_point.argsort()
    mid = len(X) // 2
    low_group = sorted_idx[:mid]
    high_group = sorted_idx[mid:]

    # Check: low group should be ALL unrefined parents
    n_refined_in_low = sum(
        1 for i in low_group if parent_k[i].item() in refined_set)
    n_unrefined_in_high = sum(
        1 for i in high_group if parent_k[i].item() not in refined_set)

    print(f"\nLow-sigma2 group: {len(low_group)} points, "
          f"{n_refined_in_low} from refined parents")
    print(f"High-sigma2 group: {len(high_group)} points, "
          f"{n_unrefined_in_high} from unrefined parents")

    # The low-sigma2 group should have NO refined-parent points
    # (because refined parents have high sigma2 by definition)
    assert n_refined_in_low == 0, \
        f"BUG: {n_refined_in_low} refined-parent points in low-sigma2 group " \
        f"(level mixing!)"


def test_state_dict_roundtrip_preserves_routing(two_tier_data, fitted_state):
    """Save/load hierarchical state and verify routing is identical."""
    X, Y, d, labels = two_tier_data
    state, r = fitted_state

    sigma2 = state["sigma2"]
    median_sigma = sigma2.median().item()
    targets = identify_refinement_targets(state, sigma2_threshold=median_sigma)

    if not targets:
        pytest.skip("No refinement targets")

    hp = make_hp(X, d=d)
    model_proto = PolynomialDiscreteEDMD(degree=1)

    h_state = refine_clusters(
        X, Y, state, r, targets,
        hp=hp, model_prototype=model_proto,
        n_subclusters=3, n_iter=20, n_restarts=1, verbose=False)

    # Predict before save
    pred_before, pk_before, ck_before, s2_before = h_state.predict(X)

    # Save and reload
    saved = h_state.state_dict()
    h_state2 = HierarchicalState.from_state_dict(saved, state, model_proto)
    h_state2.to(torch.device("cpu"), torch.float64)

    # Predict after reload
    pred_after, pk_after, ck_after, s2_after = h_state2.predict(X)

    # Must be identical
    assert torch.allclose(pred_before, pred_after, atol=1e-12), \
        f"Predictions differ after roundtrip: max diff={( pred_before - pred_after).abs().max():.2e}"
    assert (pk_before == pk_after).all(), "Parent assignments differ"
    assert (ck_before == ck_after).all(), "Child assignments differ"
    assert torch.allclose(s2_before, s2_after, atol=1e-12), "sigma2_eff differs"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
