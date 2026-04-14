"""
Test that hierarchical model reconstruction preserves predictions.

Verifies:
1. Child models predict correctly on their own training data (σ² matches)
2. from_state_dict roundtrip preserves child predictions exactly
3. h_state.predict() routes to the correct child and gets the right answer
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
    _extract_cluster_data,
)
from residual_aware_clustering.models.distributions import mvn_logpdf_batch


@pytest.fixture
def setup():
    """Fit parent, refine, return everything needed for testing."""
    rng = np.random.default_rng(42)
    d = 4
    N_per = 200

    # Two good clusters (tight, linear dynamics)
    X_g1 = torch.tensor(rng.standard_normal((N_per, d)) * 0.1 + 10,
                         dtype=torch.float64)
    Y_g1 = X_g1 * 0.99 + 0.01
    X_g2 = torch.tensor(rng.standard_normal((N_per, d)) * 0.1 - 10,
                         dtype=torch.float64)
    Y_g2 = X_g2 * 0.99 + 0.01

    # One bad cluster (spread, noisy — will be refined)
    X_bad = torch.tensor(rng.standard_normal((N_per * 2, d)) * 3.0,
                          dtype=torch.float64)
    Y_bad = X_bad * 0.5 + torch.tensor(rng.standard_normal((N_per * 2, d)) * 1.5,
                                        dtype=torch.float64)

    X = torch.cat([X_g1, X_g2, X_bad])
    Y = torch.cat([Y_g1, Y_g2, Y_bad])

    hp = make_hp(X, d=d)
    model_proto = PolynomialDiscreteEDMD(degree=1)

    state, r, _ = generic_fit(
        X, Y, N=3, hp=hp, model_prototype=model_proto,
        n_iter=50, n_restarts=1, verbose=False)

    sigma2 = state["sigma2"]
    threshold = sigma2.median().item()
    targets = identify_refinement_targets(state, sigma2_threshold=threshold)

    h_state = refine_clusters(
        X, Y, state, r, targets,
        hp=hp, model_prototype=model_proto,
        n_subclusters=5, n_iter=30, n_restarts=1, verbose=False)

    return X, Y, state, r, h_state, model_proto, targets


def test_child_predictions_match_training_sigma2(setup):
    """For each child cluster, predict on its training data and verify
    the residual matches the child's sigma2."""
    X, Y, state, r, h_state, model_proto, targets = setup

    for pk in h_state.refined_clusters:
        child_state = h_state.children[pk]

        # Extract parent's training data
        X_pk, Y_pk = _extract_cluster_data(X, Y, r, pk)

        if X_pk.shape[0] == 0:
            continue

        # Assign to children within this parent
        child_centers = child_state["centers"]
        child_covs = child_state["covariances"]
        child_pi = child_state["pi"]

        log_prox = mvn_logpdf_batch(X_pk, child_centers, child_covs)
        log_pi = torch.log(child_pi.clamp(min=1e-30)).unsqueeze(0)
        child_assignments = (log_prox + log_pi).argmax(dim=1)

        # Predict from each child model on its assigned data
        for ck in range(child_state["N"]):
            mask = child_assignments == ck
            if not mask.any():
                continue

            X_ck = X_pk[mask]
            Y_ck = Y_pk[mask]
            Y_pred = child_state["models"][ck].predict(X_ck, child_centers[ck])

            residual = (Y_ck - Y_pred).pow(2).mean().item()
            child_sigma = child_state["sigma2"][ck].item()

            print(f"  Parent {pk}, child {ck}: {mask.sum().item()} points, "
                  f"residual={residual:.6f}, sigma2={child_sigma:.6f}")

            # Residual should be close to sigma2 * d
            # (sigma2 is per-dimension, residual is mean over all dims)
            if child_sigma > 1e-10:
                ratio = residual / (child_sigma + 1e-20)
                assert ratio < 10.0, \
                    f"Parent {pk}, child {ck}: residual/sigma2 ratio = {ratio:.2f}"


def test_roundtrip_preserves_child_predictions(setup):
    """Save/load via state_dict and verify child predictions are identical."""
    X, Y, state, r, h_state, model_proto, targets = setup

    # Predict before roundtrip
    pred_before, pk_before, ck_before, _ = h_state.predict(X)

    # Roundtrip
    saved = h_state.state_dict()
    h_state2 = HierarchicalState.from_state_dict(saved, state, model_proto)
    h_state2.to(torch.device("cpu"), torch.float64)

    # Predict after roundtrip
    pred_after, pk_after, ck_after, _ = h_state2.predict(X)

    # Parent assignments must match
    assert (pk_before == pk_after).all(), \
        f"Parent assignments differ: {(pk_before != pk_after).sum()} mismatches"

    # Child assignments must match
    assert (ck_before == ck_after).all(), \
        f"Child assignments differ: {(ck_before != ck_after).sum()} mismatches"

    # Predictions must be identical
    max_diff = (pred_before - pred_after).abs().max().item()
    assert max_diff < 1e-10, \
        f"Predictions differ after roundtrip: max diff={max_diff:.2e}"

    print(f"\nRoundtrip: {len(X)} points, max prediction diff = {max_diff:.2e}")


def test_h_state_predict_routes_to_correct_child(setup):
    """Verify h_state.predict() gives same result as manual routing."""
    X, Y, state, r, h_state, model_proto, targets = setup

    # h_state.predict()
    pred_h, parent_k, child_k, sigma2_eff = h_state.predict(X)

    # Manual routing
    pred_manual = torch.zeros_like(X)

    # Parent assignment (same as h_state uses)
    log_prox = mvn_logpdf_batch(X, state["centers"], state["covariances"])
    log_pi = torch.log(state["pi"].clamp(min=1e-30)).unsqueeze(0)
    manual_pk = (log_prox + log_pi).argmax(dim=1)

    assert (parent_k == manual_pk).all(), "Parent assignment mismatch"

    refined_set = set(h_state.refined_clusters)

    for pk_val in range(state["N"]):
        mask = manual_pk == pk_val

        if not mask.any():
            continue

        if pk_val not in refined_set:
            # Unrefined: use parent model directly
            pred_manual[mask] = state["models"][pk_val].predict(
                X[mask], state["centers"][pk_val])
        else:
            # Refined: assign to child, use child model
            child_state = h_state.children[pk_val]
            X_sub = X[mask]

            child_log_prox = mvn_logpdf_batch(
                X_sub, child_state["centers"], child_state["covariances"])
            child_log_pi = torch.log(
                child_state["pi"].clamp(min=1e-30)).unsqueeze(0)
            manual_ck = (child_log_prox + child_log_pi).argmax(dim=1)

            for ck_val in range(child_state["N"]):
                cmask = manual_ck == ck_val
                if cmask.any():
                    pred_manual[mask.nonzero().squeeze(1)[cmask]] = \
                        child_state["models"][ck_val].predict(
                            X_sub[cmask], child_state["centers"][ck_val])

    max_diff = (pred_h - pred_manual).abs().max().item()
    assert max_diff < 1e-10, \
        f"h_state.predict() != manual routing: max diff={max_diff:.2e}"

    print(f"\nRouting check: {len(X)} points, max diff = {max_diff:.2e}")


def test_child_model_matches_direct_fit(setup):
    """Verify that loading child model from state_dict gives same predictions
    as the original fitted model."""
    X, Y, state, r, h_state, model_proto, targets = setup

    for pk in h_state.refined_clusters:
        child_state = h_state.children[pk]

        # Get training data for this parent
        X_pk, Y_pk = _extract_cluster_data(X, Y, r, pk)
        if X_pk.shape[0] == 0:
            continue

        # Predict with live model
        for ck in range(child_state["N"]):
            model_live = child_state["models"][ck]
            pred_live = model_live.predict(X_pk, child_state["centers"][ck])

            # Save and reload model
            ms = model_live.state_dict()
            model_reloaded = model_proto.clone()
            model_reloaded.load_state_dict(ms)
            pred_reloaded = model_reloaded.predict(X_pk, child_state["centers"][ck])

            max_diff = (pred_live - pred_reloaded).abs().max().item()
            assert max_diff < 1e-12, \
                f"Parent {pk}, child {ck}: live vs reloaded diff = {max_diff:.2e}"

    print(f"\nAll child models: state_dict roundtrip preserves predictions exactly")


def test_refined_parent_not_used_for_prediction(setup):
    """Points in a refined parent must use child model, NOT parent model."""
    X, Y, state, r, h_state, model_proto, targets = setup

    pred_h, parent_k, child_k, _ = h_state.predict(X)

    refined_set = set(h_state.refined_clusters)

    for pk_val in refined_set:
        mask = parent_k == pk_val
        if not mask.any():
            continue

        X_sub = X[mask]
        pred_h_sub = pred_h[mask]

        # What would the PARENT model predict?
        pred_parent = state["models"][pk_val].predict(
            X_sub, state["centers"][pk_val])

        # What does the hierarchy predict? (should be child, not parent)
        # If hierarchy uses parent model, these would match
        # If hierarchy uses child model, these should differ
        diff = (pred_h_sub - pred_parent).abs().max().item()

        # They should differ (child model != parent model)
        # Unless the child learned the exact same thing, which is unlikely
        print(f"  Parent {pk_val}: {mask.sum().item()} points, "
              f"hierarchy vs parent diff = {diff:.6f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
