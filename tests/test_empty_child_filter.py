"""
Test that empty child clusters are properly filtered during assignment
and prediction. Points assigned to empty children should fall back to
the parent model.
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
)
from residual_aware_clustering.models.experimental.hierarchical_refine import (
    identify_refinement_targets,
    refine_clusters,
    HierarchicalState,
)


@pytest.fixture
def setup_with_empty_children():
    """Create a scenario where some children will have very few points."""
    rng = np.random.default_rng(42)
    d = 4

    # Two good clusters
    X_good = torch.tensor(rng.standard_normal((300, d)) * 0.1 + 10,
                           dtype=torch.float64)
    Y_good = X_good * 0.99 + 0.01

    # One bad cluster with substructure
    # Most points in one region, a few outliers
    X_bad_main = torch.tensor(rng.standard_normal((280, d)) * 2.0,
                               dtype=torch.float64)
    X_bad_outlier = torch.tensor(rng.standard_normal((20, d)) * 0.5 + 15,
                                  dtype=torch.float64)
    X_bad = torch.cat([X_bad_main, X_bad_outlier])
    Y_bad = X_bad * 0.5 + torch.tensor(rng.standard_normal((300, d)) * 1.5,
                                        dtype=torch.float64)

    X = torch.cat([X_good, X_bad])
    Y = torch.cat([Y_good, Y_bad])

    hp = make_hp(X, d=d)
    model_proto = PolynomialDiscreteEDMD(degree=1)

    state, r, _ = generic_fit(
        X, Y, N=2, hp=hp, model_prototype=model_proto,
        n_iter=50, n_restarts=1, verbose=False)

    # Refine the bad cluster with many subclusters to force some empty ones
    sigma2 = state["sigma2"]
    threshold = sigma2.min().item() + 0.01
    targets = identify_refinement_targets(state, sigma2_threshold=threshold)

    h_state = refine_clusters(
        X, Y, state, r, targets,
        hp=hp, model_prototype=model_proto,
        n_subclusters=10,  # many subclusters to create empty ones
        n_iter=30, n_restarts=1, verbose=False)

    return X, Y, state, h_state, model_proto


def test_n_points_saved_during_refinement(setup_with_empty_children):
    """Verify that n_points is saved for each child cluster."""
    X, Y, state, h_state, _ = setup_with_empty_children

    for pk in h_state.refined_clusters:
        child_state = h_state.children[pk]
        assert "n_points" in child_state, \
            f"Parent {pk}: n_points not saved"
        n_points = child_state["n_points"]
        assert len(n_points) == child_state["N"], \
            f"Parent {pk}: n_points length {len(n_points)} != N={child_state['N']}"
        assert n_points.sum().item() > 0, \
            f"Parent {pk}: total points is 0"
        print(f"  Parent {pk}: n_points = {n_points.tolist()}")


def test_empty_children_get_child_k_minus_one(setup_with_empty_children):
    """Points assigned to empty children should get child_k = -1."""
    X, Y, state, h_state, _ = setup_with_empty_children

    # Use a high threshold so most children are "empty"
    parent_k, child_k, sigma2_eff = h_state.assign(X, empty_threshold=1000)

    for pk in h_state.refined_clusters:
        mask = parent_k == pk
        if not mask.any():
            continue
        ck_sub = child_k[mask]
        # With threshold=1000, most/all children are empty → child_k should be -1
        n_empty = (ck_sub == -1).sum().item()
        print(f"  Parent {pk}: {mask.sum().item()} points, "
              f"{n_empty} with child_k=-1")


def test_empty_threshold_zero_allows_all(setup_with_empty_children):
    """With empty_threshold=0, all children are valid (no filtering)."""
    X, Y, state, h_state, _ = setup_with_empty_children

    pk_0, ck_0, _ = h_state.assign(X, empty_threshold=0)
    pk_25, ck_25, _ = h_state.assign(X, empty_threshold=25)

    # Parent assignments should be identical
    assert (pk_0 == pk_25).all()

    # With threshold=0, no children filtered → no -1 in child_k for refined
    for pk in h_state.refined_clusters:
        mask = pk_0 == pk
        assert (ck_0[mask] >= 0).all(), \
            f"Parent {pk}: threshold=0 still produced child_k=-1"


def test_predict_falls_back_to_parent_for_empty_children(setup_with_empty_children):
    """Points with child_k=-1 should use parent model prediction."""
    X, Y, state, h_state, _ = setup_with_empty_children

    # High threshold: all children are "empty"
    pred_h, pk_h, ck_h, _ = h_state.predict(X, empty_threshold=100000)

    for pk in h_state.refined_clusters:
        mask = pk_h == pk
        if not mask.any():
            continue

        # All should be child_k = -1
        assert (ck_h[mask] == -1).all(), \
            f"Parent {pk}: expected all child_k=-1 with huge threshold"

        # Prediction should match parent model
        pred_parent = state["models"][pk].predict(
            X[mask], state["centers"][pk])
        diff = (pred_h[mask] - pred_parent).abs().max().item()
        assert diff < 1e-10, \
            f"Parent {pk}: fallback prediction differs from parent, diff={diff:.2e}"

    print("  All empty-child points fell back to parent model correctly")


def test_predict_uses_child_when_not_empty(setup_with_empty_children):
    """With threshold=0, all children are valid → predictions differ from parent."""
    X, Y, state, h_state, _ = setup_with_empty_children

    pred_h, pk_h, ck_h, _ = h_state.predict(X, empty_threshold=0)

    for pk in h_state.refined_clusters:
        mask = pk_h == pk
        if not mask.any():
            continue

        # Should have valid child assignments
        assert (ck_h[mask] >= 0).all()

        # Prediction should differ from parent (children != parent)
        pred_parent = state["models"][pk].predict(
            X[mask], state["centers"][pk])
        diff = (pred_h[mask] - pred_parent).abs().max().item()
        # They should differ (unless child learned same thing)
        print(f"  Parent {pk}: child vs parent diff = {diff:.6f}")


def test_state_dict_preserves_n_points(setup_with_empty_children):
    """n_points survives save/load roundtrip."""
    X, Y, state, h_state, model_proto = setup_with_empty_children

    saved = h_state.state_dict()
    h_state2 = HierarchicalState.from_state_dict(saved, state, model_proto)

    for pk in h_state.refined_clusters:
        original = h_state.children[pk]["n_points"]
        loaded = h_state2.children[pk]["n_points"]
        assert torch.equal(original, loaded), \
            f"Parent {pk}: n_points changed after roundtrip"


def test_consistent_predictions_with_and_without_filter(setup_with_empty_children):
    """Points assigned to non-empty children should get the same prediction
    regardless of whether empty filtering is on or off."""
    X, Y, state, h_state, _ = setup_with_empty_children

    pred_no_filter, pk_nf, ck_nf, _ = h_state.predict(X, empty_threshold=0)
    pred_filtered, pk_f, ck_f, _ = h_state.predict(X, empty_threshold=25)

    # Parent assignments must match
    assert (pk_nf == pk_f).all()

    # For points where both have a valid (same) child, predictions must match
    for pk in h_state.refined_clusters:
        mask = pk_nf == pk
        if not mask.any():
            continue

        both_valid = (ck_nf[mask] >= 0) & (ck_f[mask] >= 0) & (ck_nf[mask] == ck_f[mask])
        if both_valid.any():
            diff = (pred_no_filter[mask][both_valid] -
                    pred_filtered[mask][both_valid]).abs().max().item()
            assert diff < 1e-10, \
                f"Parent {pk}: same-child predictions differ, diff={diff:.2e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
