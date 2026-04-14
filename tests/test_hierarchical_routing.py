"""
Unit tests for hierarchical routing logic.

Tests assign() and predict() directly with manually constructed states,
not dependent on EM fitting producing specific cluster structures.
"""

import torch
import pytest

from residual_aware_clustering.models.experimental.polynomial_discrete import (
    PolynomialDiscreteEDMD,
)
from residual_aware_clustering.models.experimental.hierarchical_refine import (
    HierarchicalState,
)
from residual_aware_clustering.models.distributions import mvn_logpdf_batch


def _make_parent_state(n_clusters, d, centers, sigma2_values):
    """Build a minimal parent state dict with fitted linear models."""
    models = []
    for k in range(n_clusters):
        m = PolynomialDiscreteEDMD(degree=1)
        m.fallback_init(d, torch.device("cpu"), torch.float64)
        models.append(m)

    return {
        "N": n_clusters,
        "d": d,
        "P": 1000,
        "centers": centers,
        "covariances": torch.eye(d, dtype=torch.float64).unsqueeze(0).repeat(n_clusters, 1, 1),
        "pi": torch.ones(n_clusters, dtype=torch.float64) / n_clusters,
        "sigma2": torch.tensor(sigma2_values, dtype=torch.float64),
        "models": models,
    }


def _make_child_state(n_children, d, centers, sigma2_values, n_points_list):
    """Build a minimal child state dict."""
    models = []
    for k in range(n_children):
        m = PolynomialDiscreteEDMD(degree=1)
        m.fallback_init(d, torch.device("cpu"), torch.float64)
        models.append(m)

    return {
        "N": n_children,
        "d": d,
        "centers": centers,
        "covariances": torch.eye(d, dtype=torch.float64).unsqueeze(0).repeat(n_children, 1, 1),
        "pi": torch.ones(n_children, dtype=torch.float64) / n_children,
        "sigma2": torch.tensor(sigma2_values, dtype=torch.float64),
        "n_points": torch.tensor(n_points_list, dtype=torch.long),
        "models": models,
    }


# ── assign() tests ──────────────────────────────────────────────────────────

def test_assign_unrefined_returns_parent_sigma2():
    """For unrefined parents, sigma2_eff must equal parent sigma2."""
    d = 3
    parent = _make_parent_state(
        n_clusters=2, d=d,
        centers=torch.tensor([[10.0, 0, 0], [0, 10.0, 0]], dtype=torch.float64),
        sigma2_values=[0.001, 5.0],
    )
    h_state = HierarchicalState(parent=parent, children={})

    X = torch.tensor([[10.0, 0, 0], [0, 10.0, 0]], dtype=torch.float64)
    pk, ck, s2 = h_state.assign(X)

    assert pk[0] == 0
    assert pk[1] == 1
    assert ck[0] == -1  # unrefined
    assert ck[1] == -1
    assert abs(s2[0].item() - 0.001) < 1e-10
    assert abs(s2[1].item() - 5.0) < 1e-10


def test_assign_refined_returns_child_sigma2():
    """For refined parents, sigma2_eff comes from the child."""
    d = 3
    parent = _make_parent_state(
        n_clusters=2, d=d,
        centers=torch.tensor([[10.0, 0, 0], [0, 0, 0]], dtype=torch.float64),
        sigma2_values=[0.001, 200.0],
    )
    child = _make_child_state(
        n_children=2, d=d,
        centers=torch.tensor([[1.0, 0, 0], [-1.0, 0, 0]], dtype=torch.float64),
        sigma2_values=[0.5, 50.0],
        n_points_list=[500, 500],
    )
    h_state = HierarchicalState(parent=parent, children={1: child})

    # Point near parent 0 (unrefined) → parent sigma2
    # Point near parent 1 child 0 → child sigma2
    X = torch.tensor([[10.0, 0, 0], [1.0, 0, 0]], dtype=torch.float64)
    pk, ck, s2 = h_state.assign(X)

    assert pk[0] == 0  # unrefined parent
    assert ck[0] == -1
    assert abs(s2[0].item() - 0.001) < 1e-10

    assert pk[1] == 1  # refined parent
    assert ck[1] == 0  # child 0 (closer to [1,0,0])
    assert abs(s2[1].item() - 0.5) < 1e-10


def test_assign_empty_child_gets_minus_one():
    """Children below empty_threshold get child_k = -1."""
    d = 3
    parent = _make_parent_state(
        n_clusters=1, d=d,
        centers=torch.tensor([[0.0, 0, 0]], dtype=torch.float64),
        sigma2_values=[100.0],
    )
    child = _make_child_state(
        n_children=3, d=d,
        centers=torch.tensor([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0]],
                              dtype=torch.float64),
        sigma2_values=[0.0, 0.0, 0.0],
        n_points_list=[500, 10, 3],  # child 1 and 2 are "empty"
    )
    h_state = HierarchicalState(parent=parent, children={0: child})

    X = torch.tensor([
        [1.0, 0, 0],   # nearest to child 0 (500 points, valid)
        [-1.0, 0, 0],  # nearest to child 1 (10 points, empty at threshold=25)
        [0, 1.0, 0],   # nearest to child 2 (3 points, empty)
    ], dtype=torch.float64)

    pk, ck, s2 = h_state.assign(X, empty_threshold=25)

    assert ck[0] == 0   # valid child (500 points)
    # Points 1,2 nearest empty children, but reassigned to best non-empty child (0)
    assert ck[1] == 0   # reassigned to child 0 (only non-empty)
    assert ck[2] == 0   # reassigned to child 0


def test_assign_empty_threshold_zero_allows_all():
    """With threshold=0, no filtering — all children valid."""
    d = 3
    parent = _make_parent_state(
        n_clusters=1, d=d,
        centers=torch.tensor([[0.0, 0, 0]], dtype=torch.float64),
        sigma2_values=[100.0],
    )
    child = _make_child_state(
        n_children=2, d=d,
        centers=torch.tensor([[1.0, 0, 0], [-1.0, 0, 0]], dtype=torch.float64),
        sigma2_values=[0.0, 0.0],
        n_points_list=[5, 3],  # both would be "empty" at threshold=25
    )
    h_state = HierarchicalState(parent=parent, children={0: child})

    X = torch.tensor([[1.0, 0, 0], [-1.0, 0, 0]], dtype=torch.float64)
    pk, ck, s2 = h_state.assign(X, empty_threshold=0)

    assert ck[0] >= 0  # no filtering
    assert ck[1] >= 0


# ── predict() tests ─────────────────────────────────────────────────────────

def test_predict_unrefined_uses_parent_model():
    """Unrefined parent → parent model prediction."""
    d = 3
    parent = _make_parent_state(
        n_clusters=2, d=d,
        centers=torch.tensor([[10.0, 0, 0], [0, 10.0, 0]], dtype=torch.float64),
        sigma2_values=[0.0, 0.0],
    )
    h_state = HierarchicalState(parent=parent, children={})

    X = torch.tensor([[10.0, 0, 0]], dtype=torch.float64)
    pred_h, pk, ck, _ = h_state.predict(X)

    pred_parent = parent["models"][0].predict(X, parent["centers"][0])
    assert torch.allclose(pred_h, pred_parent, atol=1e-12)


def test_predict_refined_uses_child_model():
    """Refined parent → child model prediction (not parent)."""
    d = 3
    parent = _make_parent_state(
        n_clusters=1, d=d,
        centers=torch.tensor([[0.0, 0, 0]], dtype=torch.float64),
        sigma2_values=[100.0],
    )
    child = _make_child_state(
        n_children=2, d=d,
        centers=torch.tensor([[1.0, 0, 0], [-1.0, 0, 0]], dtype=torch.float64),
        sigma2_values=[0.0, 0.0],
        n_points_list=[500, 500],
    )
    h_state = HierarchicalState(parent=parent, children={0: child})

    X = torch.tensor([[1.0, 0, 0]], dtype=torch.float64)
    pred_h, pk, ck, _ = h_state.predict(X)

    # Should use child 0, not parent
    pred_child = child["models"][0].predict(X, child["centers"][0])
    pred_parent = parent["models"][0].predict(X, parent["centers"][0])

    assert torch.allclose(pred_h, pred_child, atol=1e-12)
    # Child and parent predictions should differ (different models)
    # (unless fallback_init produces same identity, which it does)
    # So just verify the routing happened correctly
    assert ck[0] == 0


def test_predict_empty_child_falls_back_to_parent():
    """When ALL children are empty → falls back to parent model."""
    d = 3
    parent = _make_parent_state(
        n_clusters=1, d=d,
        centers=torch.tensor([[0.0, 0, 0]], dtype=torch.float64),
        sigma2_values=[100.0],
    )
    child = _make_child_state(
        n_children=2, d=d,
        centers=torch.tensor([[1.0, 0, 0], [-1.0, 0, 0]], dtype=torch.float64),
        sigma2_values=[0.0, 0.0],
        n_points_list=[3, 5],  # BOTH children are empty
    )
    h_state = HierarchicalState(parent=parent, children={0: child})

    X = torch.tensor([[-1.0, 0, 0]], dtype=torch.float64)
    pred_h, pk, ck, _ = h_state.predict(X, empty_threshold=25)

    assert ck[0] == -1  # all children empty → -1
    pred_parent = parent["models"][0].predict(X, parent["centers"][0])
    assert torch.allclose(pred_h, pred_parent, atol=1e-12)


def test_predict_empty_child_reassigns_to_non_empty():
    """When SOME children are empty, point reassigned to best non-empty child."""
    d = 3
    parent = _make_parent_state(
        n_clusters=1, d=d,
        centers=torch.tensor([[0.0, 0, 0]], dtype=torch.float64),
        sigma2_values=[100.0],
    )
    child = _make_child_state(
        n_children=2, d=d,
        centers=torch.tensor([[1.0, 0, 0], [-1.0, 0, 0]], dtype=torch.float64),
        sigma2_values=[0.0, 0.0],
        n_points_list=[500, 3],  # child 1 is empty, child 0 is valid
    )
    h_state = HierarchicalState(parent=parent, children={0: child})

    # Point near child 1 (empty) → should go to child 0 (valid)
    X = torch.tensor([[-1.0, 0, 0]], dtype=torch.float64)
    pred_h, pk, ck, _ = h_state.predict(X, empty_threshold=25)

    assert ck[0] == 0  # reassigned to child 0
    pred_child = child["models"][0].predict(X, child["centers"][0])
    assert torch.allclose(pred_h, pred_child, atol=1e-12)


# ── Level mixing tests ──────────────────────────────────────────────────────

def test_no_sigma2_level_mixing():
    """Parent σ²=0 (unrefined) must never mix with child σ²=0 (refined)
    when grouping by sigma2_eff."""
    d = 3
    parent = _make_parent_state(
        n_clusters=3, d=d,
        centers=torch.tensor([
            [10.0, 0, 0],   # good, σ²=0
            [-10.0, 0, 0],  # good, σ²=0
            [0, 0, 0],      # bad, σ²=200
        ], dtype=torch.float64),
        sigma2_values=[0.0, 0.0, 200.0],
    )
    child = _make_child_state(
        n_children=2, d=d,
        centers=torch.tensor([[1.0, 0, 0], [-1.0, 0, 0]], dtype=torch.float64),
        sigma2_values=[0.0, 50.0],  # child 0 has σ²=0 (overfit)
        n_points_list=[500, 500],
    )
    h_state = HierarchicalState(parent=parent, children={2: child})

    X = torch.tensor([
        [10.0, 0, 0],   # → parent 0, unrefined, σ²=0
        [-10.0, 0, 0],  # → parent 1, unrefined, σ²=0
        [1.0, 0, 0],    # → parent 2, child 0, child σ²=0
        [-1.0, 0, 0],   # → parent 2, child 1, child σ²=50
    ], dtype=torch.float64)

    pk, ck, s2 = h_state.assign(X)

    # Unrefined parents: s2 = parent σ² = 0
    assert abs(s2[0].item() - 0.0) < 1e-10
    assert abs(s2[1].item() - 0.0) < 1e-10

    # Refined children: s2 = child σ²
    assert abs(s2[2].item() - 0.0) < 1e-10   # child 0
    assert abs(s2[3].item() - 50.0) < 1e-10  # child 1

    # The mixing: points 0,1 (unrefined σ²=0) and point 2 (child σ²=0)
    # have the same sigma2_eff. To distinguish them, check parent_k:
    assert pk[0].item() not in h_state.refined_clusters  # unrefined
    assert pk[1].item() not in h_state.refined_clusters  # unrefined
    assert pk[2].item() in h_state.refined_clusters       # refined
    assert pk[3].item() in h_state.refined_clusters       # refined

    # Gating should use PARENT σ² to avoid mixing:
    parent_s2 = parent["sigma2"]
    gate_signal = parent_s2[pk]  # always parent σ²

    assert abs(gate_signal[0].item() - 0.0) < 1e-10    # parent 0
    assert abs(gate_signal[1].item() - 0.0) < 1e-10    # parent 1
    assert abs(gate_signal[2].item() - 200.0) < 1e-10  # parent 2 (high!)
    assert abs(gate_signal[3].item() - 200.0) < 1e-10  # parent 2 (high!)

    # Parent σ² cleanly separates: unrefined (0) vs refined (200)
    # No mixing possible when gating on parent σ²


def test_parent_sigma2_always_separates_levels():
    """Using parent σ² for gating, refined children NEVER appear in
    the same group as unrefined parents (regardless of child σ²)."""
    d = 3
    parent = _make_parent_state(
        n_clusters=4, d=d,
        centers=torch.tensor([
            [20.0, 0, 0],   # σ²=0 (good)
            [0, 20.0, 0],   # σ²=0.5 (good)
            [0, 0, 20.0],   # σ²=150 (bad, refined)
            [0, 0, -20.0],  # σ²=300 (bad, refined)
        ], dtype=torch.float64),
        sigma2_values=[0.0, 0.5, 150.0, 300.0],
    )
    child_2 = _make_child_state(
        n_children=2, d=d,
        centers=torch.tensor([[0, 0, 21.0], [0, 0, 19.0]], dtype=torch.float64),
        sigma2_values=[0.0, 0.0],  # both children overfit to σ²=0
        n_points_list=[300, 300],
    )
    child_3 = _make_child_state(
        n_children=2, d=d,
        centers=torch.tensor([[0, 0, -21.0], [0, 0, -19.0]], dtype=torch.float64),
        sigma2_values=[0.0, 0.0],
        n_points_list=[300, 300],
    )
    h_state = HierarchicalState(parent=parent, children={2: child_2, 3: child_3})

    # Points assigned to each parent
    X = torch.tensor([
        [20.0, 0, 0],    # parent 0, unrefined
        [0, 20.0, 0],    # parent 1, unrefined
        [0, 0, 21.0],    # parent 2, child 0
        [0, 0, -21.0],   # parent 3, child 0
    ], dtype=torch.float64)

    pk, ck, s2 = h_state.assign(X)

    # Gate signal = parent σ²
    gate = parent["sigma2"][pk]

    # Unrefined: low gate values
    assert gate[0].item() < 1.0
    assert gate[1].item() < 1.0

    # Refined: high gate values (even though child σ²=0)
    assert gate[2].item() >= 150.0
    assert gate[3].item() >= 300.0

    # Any threshold between 1.0 and 150.0 cleanly separates the levels
    threshold = 100.0
    unrefined_pass = gate[:2] < threshold  # both True
    refined_pass = gate[2:] < threshold     # both False

    assert unrefined_pass.all()
    assert not refined_pass.any()


# ── state_dict roundtrip ────────────────────────────────────────────────────

def test_state_dict_roundtrip_preserves_routing():
    """Save/load preserves all routing behavior."""
    d = 3
    parent = _make_parent_state(
        n_clusters=2, d=d,
        centers=torch.tensor([[10.0, 0, 0], [0, 0, 0]], dtype=torch.float64),
        sigma2_values=[0.0, 200.0],
    )
    child = _make_child_state(
        n_children=2, d=d,
        centers=torch.tensor([[1.0, 0, 0], [-1.0, 0, 0]], dtype=torch.float64),
        sigma2_values=[0.0, 50.0],
        n_points_list=[500, 500],
    )
    h_state = HierarchicalState(parent=parent, children={1: child})

    X = torch.tensor([[10.0, 0, 0], [1.0, 0, 0], [-1.0, 0, 0]],
                      dtype=torch.float64)

    pred_before, pk_before, ck_before, s2_before = h_state.predict(X)

    # Roundtrip
    saved = h_state.state_dict()
    model_proto = PolynomialDiscreteEDMD(degree=1)
    h_state2 = HierarchicalState.from_state_dict(saved, parent, model_proto)

    pred_after, pk_after, ck_after, s2_after = h_state2.predict(X)

    assert (pk_before == pk_after).all()
    assert (ck_before == ck_after).all()
    assert torch.allclose(s2_before, s2_after, atol=1e-12)
    assert torch.allclose(pred_before, pred_after, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
