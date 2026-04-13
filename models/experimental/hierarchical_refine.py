"""
Hierarchical cluster refinement for high-residual clusters.

Takes a fitted CWM state, identifies clusters with high residual variance
(sigma2_k above a threshold), and runs a second round of CWM fitting on
each one's assigned data. The result is a two-level hierarchy: low-sigma2
parent clusters are kept as-is, high-sigma2 parents are replaced by their
child sub-clusters.

At prediction time, assignment cascades: if a point lands in a refined
parent, it's reassigned among the parent's children.

Usage::

    from residual_aware_clustering.models.experimental.hierarchical_refine import (
        identify_refinement_targets, refine_clusters, HierarchicalState,
    )

    targets = identify_refinement_targets(state, sigma2_threshold=100.0)
    h_state = refine_clusters(
        X, Y, state, responsibilities, targets,
        hp=hp, model_prototype=model_proto, n_subclusters=20,
    )

    # Prediction: cascades to children for refined parents
    z_pred, assignment = h_state.predict(z_input)
"""

from __future__ import annotations
from dataclasses import dataclass, field

import torch

from .generic_em import (
    fit as generic_fit,
    e_step,
)
from residual_aware_clustering.models.distributions import mvn_logpdf_batch


def identify_refinement_targets(state: dict, sigma2_threshold: float,
                                min_points: int = 100) -> list[int]:
    """Find clusters whose sigma2_k exceeds the threshold.

    Parameters
    ----------
    state : dict
        Fitted CWM state with 'sigma2', 'pi' keys.
    sigma2_threshold : float
        Clusters with sigma2 >= this are refinement targets.
    min_points : int
        Skip clusters with fewer effective points than this
        (estimated from mixing weight * total data).

    Returns
    -------
    list of int
        Indices of clusters to refine.
    """
    sigma2 = state["sigma2"]
    pi = state["pi"]
    P = state["P"]

    targets = []
    for k in range(state["N"]):
        if sigma2[k].item() >= sigma2_threshold:
            n_eff = pi[k].item() * P
            if n_eff >= min_points:
                targets.append(k)

    return targets


def _extract_cluster_data(X, Y, responsibilities, k, hard=True):
    """Extract data assigned to cluster k.

    Parameters
    ----------
    X, Y : torch.Tensor
        Full dataset, shape (P, d).
    responsibilities : torch.Tensor
        Shape (P, N).
    k : int
        Cluster index.
    hard : bool
        If True, use hard assignments (argmax). If False, use all points
        with r_k > 0.01.

    Returns
    -------
    X_k, Y_k : torch.Tensor
        Data for cluster k.
    """
    if hard:
        mask = responsibilities.argmax(dim=1) == k
    else:
        mask = responsibilities[:, k] > 0.01
    return X[mask], Y[mask]


def refine_clusters(X, Y, parent_state, responsibilities, targets,
                    hp, model_prototype, n_subclusters=20,
                    n_iter=50, n_restarts=1, max_gmm_samples=5000,
                    verbose=True):
    """Refine high-sigma2 clusters by fitting sub-clusters within each.

    Parameters
    ----------
    X, Y : torch.Tensor
        Full dataset.
    parent_state : dict
        Fitted parent CWM state.
    responsibilities : torch.Tensor
        Parent responsibilities, shape (P, N).
    targets : list of int
        Parent cluster indices to refine.
    hp : dict
        Hyperparameters for child fitting.
    model_prototype : LocalModel
        Template for child local models.
    n_subclusters : int
        Number of sub-clusters per refined parent.
    n_iter, n_restarts, max_gmm_samples : int
        EM parameters for child fitting.
    verbose : bool

    Returns
    -------
    HierarchicalState
        Combined parent + children state for prediction.
    """
    children = {}

    for k in targets:
        X_k, Y_k = _extract_cluster_data(X, Y, responsibilities, k)
        n_k = X_k.shape[0]

        if verbose:
            parent_sigma = parent_state["sigma2"][k].item()
            print(f"\n  Refining cluster {k}: {n_k} points, "
                  f"parent sigma2={parent_sigma:.2f}")

        if n_k < n_subclusters * 2:
            if verbose:
                print(f"    Skipping: too few points ({n_k}) for "
                      f"{n_subclusters} subclusters")
            continue

        # Build hp for the subset
        hp_child = dict(hp)
        hp_child["mu0"] = X_k.mean(dim=0)

        child_state, child_r, child_history = generic_fit(
            X_k, Y_k,
            N=n_subclusters,
            hp=hp_child,
            model_prototype=model_prototype,
            n_iter=n_iter,
            n_restarts=n_restarts,
            max_gmm_samples=max_gmm_samples,
            verbose=verbose,
        )

        if child_state is not None:
            child_sigma_mean = child_state["sigma2"].mean().item()
            child_sigma_min = child_state["sigma2"].min().item()
            n_active = child_state["N"]
            if verbose:
                print(f"    Result: {n_active} subclusters, "
                      f"sigma2 mean={child_sigma_mean:.2f}, "
                      f"min={child_sigma_min:.4f}")
            children[k] = child_state

    return HierarchicalState(parent=parent_state, children=children)


@dataclass
class HierarchicalState:
    """Two-level cluster hierarchy for prediction.

    Parent clusters with low sigma2 are used directly.
    Refined parent clusters cascade to their child sub-clusters.
    """
    parent: dict
    children: dict = field(default_factory=dict)

    @property
    def refined_clusters(self) -> list[int]:
        """Parent cluster indices that have been refined."""
        return list(self.children.keys())

    @property
    def n_total_clusters(self) -> int:
        """Total clusters: unrefinned parents + all children."""
        n = self.parent["N"] - len(self.children)
        for child_state in self.children.values():
            n += child_state["N"]
        return n

    def get_sigma2(self, parent_k: int, child_k: int = None) -> float:
        """Get sigma2 for a (parent, child) assignment.

        If parent_k is refined and child_k is given, returns child sigma2.
        Otherwise returns parent sigma2.
        """
        if parent_k in self.children and child_k is not None:
            return self.children[parent_k]["sigma2"][child_k].item()
        return self.parent["sigma2"][parent_k].item()

    def assign(self, z):
        """Assign a single point or batch to (parent_k, child_k).

        Parameters
        ----------
        z : torch.Tensor
            Shape (d,) or (N, d).

        Returns
        -------
        parent_k : torch.Tensor
            Parent cluster indices, shape (N,).
        child_k : torch.Tensor or None
            Child cluster indices (N,). -1 for unrefined parents.
        sigma2_effective : torch.Tensor
            Effective sigma2 for each point, shape (N,).
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        N = z.shape[0]

        # Parent assignment
        parent_centers = self.parent["centers"]
        parent_covs = self.parent["covariances"]
        parent_pi = self.parent["pi"]

        log_prox = mvn_logpdf_batch(z, parent_centers, parent_covs)
        log_pi = torch.log(parent_pi.clamp(min=1e-30)).unsqueeze(0)
        parent_k = (log_prox + log_pi).argmax(dim=1)

        # Child assignment for refined parents
        child_k = torch.full((N,), -1, dtype=torch.long)
        sigma2_eff = self.parent["sigma2"][parent_k]

        for pk in self.children:
            mask = parent_k == pk
            if not mask.any():
                continue
            child_state = self.children[pk]
            z_sub = z[mask]

            child_centers = child_state["centers"]
            child_covs = child_state["covariances"]
            child_pi = child_state["pi"]

            child_log_prox = mvn_logpdf_batch(z_sub, child_centers, child_covs)
            child_log_pi = torch.log(child_pi.clamp(min=1e-30)).unsqueeze(0)
            ck = (child_log_prox + child_log_pi).argmax(dim=1)

            child_k[mask] = ck
            sigma2_eff[mask] = child_state["sigma2"][ck]

        return parent_k, child_k, sigma2_eff

    def predict(self, z):
        """Predict next state for a batch of points.

        Parameters
        ----------
        z : torch.Tensor
            Shape (N, d).

        Returns
        -------
        z_pred : torch.Tensor
            Predicted next state, shape (N, d).
        parent_k : torch.Tensor
            Parent assignments.
        child_k : torch.Tensor
            Child assignments (-1 if unrefined).
        sigma2_eff : torch.Tensor
            Effective sigma2 per point.
        """
        N = z.shape[0]
        parent_k, child_k, sigma2_eff = self.assign(z)

        z_pred = torch.zeros_like(z)

        # Unrefined parents: predict directly
        for pk in range(self.parent["N"]):
            if pk in self.children:
                continue
            mask = parent_k == pk
            if mask.any():
                z_pred[mask] = self.parent["models"][pk].predict(
                    z[mask], self.parent["centers"][pk])

        # Refined parents: predict via child models
        for pk, child_state in self.children.items():
            mask = parent_k == pk
            if not mask.any():
                continue
            z_sub = z[mask]
            ck_sub = child_k[mask]

            pred_sub = torch.zeros_like(z_sub)
            for ck in range(child_state["N"]):
                cmask = ck_sub == ck
                if cmask.any():
                    pred_sub[cmask] = child_state["models"][ck].predict(
                        z_sub[cmask], child_state["centers"][ck])
            z_pred[mask] = pred_sub

        return z_pred, parent_k, child_k, sigma2_eff

    def state_dict(self) -> dict:
        """Serialize for saving."""
        child_dicts = {}
        for pk, cs in self.children.items():
            child_dicts[pk] = {
                "centers": cs["centers"],
                "covariances": cs["covariances"],
                "pi": cs["pi"],
                "sigma2": cs["sigma2"],
                "N": cs["N"],
                "d": cs["d"],
                "model_states": [m.state_dict() for m in cs["models"]],
            }
        return {
            "parent": self.parent,
            "children": child_dicts,
            "refined_clusters": self.refined_clusters,
        }

    @classmethod
    def from_state_dict(cls, saved: dict, parent_state: dict,
                        model_prototype) -> HierarchicalState:
        """Reconstruct from a saved state_dict.

        The saved children have 'model_states' (serialized dicts) instead
        of live 'models'. This method rebuilds them using the prototype.

        Parameters
        ----------
        saved : dict
            Output of ``state_dict()``, with keys 'children', 'refined_clusters'.
        parent_state : dict
            The parent CWM state (with live models).
        model_prototype : LocalModel
            Template to clone and load child model states into.
        """
        children = {}
        for pk_str, child_dict in saved.get("children", {}).items():
            pk = int(pk_str) if isinstance(pk_str, str) else pk_str
            # Rebuild live models from saved states
            models = []
            for ms in child_dict["model_states"]:
                m = model_prototype.clone()
                m.load_state_dict(ms)
                models.append(m)
            child_state = {
                "centers": child_dict["centers"],
                "covariances": child_dict["covariances"],
                "pi": child_dict["pi"],
                "sigma2": child_dict["sigma2"],
                "N": child_dict["N"],
                "d": child_dict["d"],
                "models": models,
            }
            children[pk] = child_state
        return cls(parent=parent_state, children=children)

    def to(self, device, dtype):
        """Move all tensors to device/dtype."""
        for key in ["centers", "covariances", "pi", "sigma2"]:
            if isinstance(self.parent.get(key), torch.Tensor):
                self.parent[key] = self.parent[key].to(device=device, dtype=dtype)
        if "models" in self.parent:
            for m in self.parent["models"]:
                m.to(device, dtype)

        for cs in self.children.values():
            for key in ["centers", "covariances", "pi", "sigma2"]:
                if isinstance(cs.get(key), torch.Tensor):
                    cs[key] = cs[key].to(device=device, dtype=dtype)
            if "models" in cs:
                for m in cs["models"]:
                    m.to(device, dtype)
        return self
