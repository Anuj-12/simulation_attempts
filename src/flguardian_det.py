"""
FLGuardian Contamination Detector  —  PyTorch implementation
=============================================================
Exact implementation of the FLGuardian layer-wise malicious-client detection
mechanism from:

    Zhou et al., "FLGuardian: Defending Against Model Poisoning Attacks via
    Fine-Grained Detection in Federated Learning,"
    IEEE Transactions on Information Forensics and Security, Vol. 20, 2025.
    DOI: 10.1109/TIFS.2025.3570119

All distance arithmetic and clustering run entirely in PyTorch — no NumPy or
scikit-learn is required in the hot path.  The module is therefore safe to
use inside a CUDA training loop without CPU round-trips.

Public interface (unchanged from the NumPy version)
----------------------------------------------------
    FLGuardianDetector(beta, k, device, seed)
    detector.fit(client_updates)              — one FL round
    detector.detect_contamination(client_id)  — returns lambda_j ∈ [0, 1]
    detector.contamination_scores()           — {client_id: lambda_j}
    detector.trust_scores()                   — {client_id: raw score}
    detector.top_k_clients()                  — list of k safest client ids
    detector.benign_sets()                    — {layer_name: set of client ids}

Input format
------------
client_updates : dict[any, dict[str, torch.Tensor]]
    {client_id: {layer_name: 1-D float tensor of the model update for that layer}}

    Callers coming from a PyTorch model can build this with:
        {cid: {name: (new_w - old_w).detach().flatten()
               for name, new_w in model.named_parameters()}}

    Tensors may live on any device; all computation is done on the device
    specified at construction time (default: "cpu").

Assumptions / approximations (marked ⚑)
-----------------------------------------
⚑ A1  k-means with k=2 (paper §IV-B).
        Implemented as Lloyd's algorithm in PyTorch.  We run `n_init` random
        restarts and `max_iter` iterations per restart (defaults: 5 and 300),
        keeping the result with the lowest inertia — matching the behaviour of
        sklearn.KMeans used in the original NumPy version.

⚑ A2  "Larger cluster = benign candidate" (paper §IV-B).
        Tie-breaking: equal-size clusters → return all indices (maximum
        uncertainty; intersection then equals the full set).

⚑ A3  Layer enumeration.
        All clients must supply identical layer names in identical order.
        The detector raises ValueError otherwise.

⚑ A4  Depth ordering.
        Layer weight β^l uses 1-based l, preserving the dict iteration order
        supplied by the caller as canonical depth ordering.

⚑ A5  Trust-score → contamination score mapping.
        The PPO governance framework expects λ_j ∈ [0,1] where higher = more
        suspicious.  FLGuardian produces trust scores (higher = safer), so we
        invert after min-max normalisation:
            λ_j = 1 − (score_j − min) / (max − min)
        When all scores are identical, λ_j = 0.5 (maximum uncertainty).

⚑ A6  Single-client rounds.
        k-means on a 1×1 distance matrix is undefined.  If fewer than 2
        clients are supplied the detector returns λ_j = 0.0 and emits a
        RuntimeWarning.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal PyTorch helpers
# ---------------------------------------------------------------------------

def _pairwise_cosine_distances(
    layer_updates: torch.Tensor,
) -> torch.Tensor:
    """
    Equation (9): C[i,j] = 1 − cos(g_i^(l), g_j^(l))

    Parameters
    ----------
    layer_updates : Tensor of shape (n, d_l)

    Returns
    -------
    C : Tensor of shape (n, n), values in [0, 2]
    """
    norms = layer_updates.norm(dim=1, keepdim=True)          # (n, 1)
    # Zero-norm rows: treat as unit vector in a neutral direction so
    # cosine distance to every other vector = 1  (⚑ A3 zero-update guard)
    safe_norms = norms.clamp(min=1e-12)
    normalised = layer_updates / safe_norms                   # (n, d_l)
    cosine_sim = normalised @ normalised.T                    # (n, n)
    cosine_sim = cosine_sim.clamp(-1.0, 1.0)
    return 1.0 - cosine_sim


def _pairwise_euclidean_distances(
    layer_updates: torch.Tensor,
) -> torch.Tensor:
    """
    Equation (11): E[i,j] = ||g_i^(l) − g_j^(l)||_2

    Uses the identity ||a-b||² = ||a||² + ||b||² - 2⟨a,b⟩ for efficiency.

    Parameters
    ----------
    layer_updates : Tensor of shape (n, d_l)

    Returns
    -------
    E : Tensor of shape (n, n), values ≥ 0
    """
    sq_norms = (layer_updates ** 2).sum(dim=1)               # (n,)
    sq_dist = (
        sq_norms.unsqueeze(1)                                 # (n, 1)
        + sq_norms.unsqueeze(0)                               # (1, n)
        - 2.0 * (layer_updates @ layer_updates.T)            # (n, n)
    )
    return sq_dist.clamp(min=0.0).sqrt()


def _kmeans_2(
    features: torch.Tensor,
    n_init: int = 5,
    max_iter: int = 300,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Lloyd's k-means with k=2, pure PyTorch.

    Runs `n_init` random restarts; returns labels from the restart with the
    lowest inertia (sum of squared distances to assigned centroid).

    Parameters
    ----------
    features : Tensor of shape (n, d)
        One row per client; typically the row of the pairwise distance matrix.
    n_init : int
        Number of independent random restarts.
    max_iter : int
        Maximum Lloyd iterations per restart.
    generator : torch.Generator or None
        Optional RNG for reproducibility.

    Returns
    -------
    labels : LongTensor of shape (n,) with values in {0, 1}
    """
    n, d = features.shape
    device = features.device
    best_labels = torch.zeros(n, dtype=torch.long, device=device)
    best_inertia = float("inf")

    for _ in range(n_init):
        # Random initialisation: pick 2 distinct rows as initial centroids
        perm = torch.randperm(n, generator=generator, device=device)
        centroids = features[perm[:2]].clone()                # (2, d)

        labels = torch.zeros(n, dtype=torch.long, device=device)
        for _ in range(max_iter):
            # Assignment step
            d0 = ((features - centroids[0]) ** 2).sum(dim=1)
            d1 = ((features - centroids[1]) ** 2).sum(dim=1)
            new_labels = (d1 < d0).long()                    # 0 if closer to c0

            if (new_labels == labels).all():
                labels = new_labels
                break
            labels = new_labels

            # Update step
            for k in range(2):
                mask = labels == k
                if mask.any():
                    centroids[k] = features[mask].mean(dim=0)
                # If a cluster becomes empty its centroid stays put (rare edge case)

        # Compute inertia
        d0 = ((features - centroids[0]) ** 2).sum(dim=1)
        d1 = ((features - centroids[1]) ** 2).sum(dim=1)
        assigned_dist = torch.where(labels == 0, d0, d1)
        inertia = assigned_dist.sum().item()

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.clone()

    return best_labels


def _larger_cluster_indices(
    dist_matrix: torch.Tensor,
    generator: Optional[torch.Generator],
    n_init: int,
    max_iter: int,
) -> torch.Tensor:
    """
    Run k-means (k=2) on the distance-matrix rows and return indices of the
    larger cluster.

    ⚑ A1: distance-matrix row used as per-client feature vector.
    ⚑ A2: tie → return all indices.
    """
    n = dist_matrix.shape[0]
    if n < 2:                                                  # ⚑ A6
        return torch.arange(n, device=dist_matrix.device)

    labels = _kmeans_2(
        dist_matrix,
        n_init=n_init,
        max_iter=max_iter,
        generator=generator,
    )

    count0 = (labels == 0).sum().item()
    count1 = (labels == 1).sum().item()

    if count0 > count1:
        return (labels == 0).nonzero(as_tuple=False).squeeze(1)
    elif count1 > count0:
        return (labels == 1).nonzero(as_tuple=False).squeeze(1)
    else:                                                      # ⚑ A2 tie
        logger.debug(
            "k-means tie: both clusters size %d — returning all indices.", count0
        )
        return torch.arange(n, device=dist_matrix.device)


def _benign_set_for_layer(
    layer_updates: torch.Tensor,
    generator: Optional[torch.Generator],
    n_init: int,
    max_iter: int,
) -> torch.Tensor:
    """
    §IV-B:
    1. Pairwise cosine distances  → k-means → larger cluster (C-candidates)
    2. Pairwise Euclidean distances → k-means → larger cluster (E-candidates)
    3. Benign set = C-candidates ∩ E-candidates

    Returns
    -------
    benign_indices : LongTensor of sorted client indices in the benign set
    """
    C = _pairwise_cosine_distances(layer_updates)
    E = _pairwise_euclidean_distances(layer_updates)

    # Use two independent generators so the two k-means runs are not identical
    cos_idx = _larger_cluster_indices(C, generator, n_init, max_iter)
    euc_idx = _larger_cluster_indices(E, generator, n_init, max_iter)

    cos_set = set(cos_idx.tolist())
    euc_set = set(euc_idx.tolist())
    benign = sorted(cos_set & euc_set)
    return torch.tensor(benign, dtype=torch.long, device=layer_updates.device)


# ---------------------------------------------------------------------------
# Public detector class
# ---------------------------------------------------------------------------

class FLGuardianDetector:
    """
    Layer-wise malicious-client detector (FLGuardian, Zhou et al. 2025).
    Pure PyTorch — no scikit-learn dependency.

    Parameters
    ----------
    beta : float
        Layer-depth weight (Eq. 12).  Paper default = 2.
    k : int
        Clients selected for aggregation (Eq. 14).
    device : str | torch.device
        All tensors are moved here before computation.  Default "cpu".
    seed : int | None
        Seed for the internal torch.Generator used by k-means.
        None → non-deterministic.
    n_init : int
        k-means random restarts per layer per distance type.  Default 5.
    max_iter : int
        Maximum Lloyd iterations per k-means restart.  Default 300.
    """

    def __init__(
        self,
        beta: float = 2.0,
        k: int = 8,
        device: str | torch.device = "cpu",
        seed: Optional[int] = 42,
        n_init: int = 5,
        max_iter: int = 300,
    ) -> None:
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        self.beta = beta
        self.k = k
        self.device = torch.device(device)
        self.seed = seed
        self.n_init = n_init
        self.max_iter = max_iter

        self._generator: Optional[torch.Generator] = None
        if seed is not None:
            self._generator = torch.Generator(device=self.device)
            self._generator.manual_seed(seed)

        # Populated by fit()
        self._client_ids: List = []
        self._trust_scores: Dict = {}
        self._contamination: Dict = {}
        self._benign_sets: Dict = {}
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Core algorithm — Algorithm 2 (Lines 8-21)
    # ------------------------------------------------------------------

    def fit(
        self,
        client_updates: Dict[object, Dict[str, torch.Tensor]],
    ) -> "FLGuardianDetector":
        """
        Run FLGuardian layer-wise detection for one FL round.

        Parameters
        ----------
        client_updates : dict
            {client_id: {layer_name: 1-D torch.Tensor}}

            Each tensor is the model *update* (w_i^t − w^{t-1}) for that
            layer, already flattened to 1-D.  Tensors may be on any device;
            they are moved to self.device internally.

            ⚑ A3: All clients must have identical layer names in identical order.

        Returns
        -------
        self
        """
        if len(client_updates) == 0:
            raise ValueError("client_updates is empty.")

        self._client_ids = list(client_updates.keys())
        n = len(self._client_ids)

        if n < 2:                                              # ⚑ A6
            warnings.warn(
                "FLGuardianDetector.fit() received fewer than 2 clients. "
                "Contamination detection is undefined; returning λ=0.0 for all.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._trust_scores = {cid: 0.0 for cid in self._client_ids}
            self._contamination = {cid: 0.0 for cid in self._client_ids}
            self._fitted = True
            return self

        # Validate consistent layer names across clients
        reference_layers = list(next(iter(client_updates.values())).keys())
        for cid, layers in client_updates.items():
            if list(layers.keys()) != reference_layers:
                raise ValueError(
                    f"Client {cid} has layer names {list(layers.keys())} "
                    f"but expected {reference_layers}."
                )

        L = len(reference_layers)
        # ⚑ A4: β^l, l = 1 … L  (1-based depth index)
        layer_weights = {
            name: self.beta ** (idx + 1)
            for idx, name in enumerate(reference_layers)
        }

        trust = {cid: 0.0 for cid in self._client_ids}
        benign_sets: Dict[str, set] = {}

        # Algorithm 2, Lines 9-18
        for layer_name in reference_layers:
            # Stack updates for this layer → (n, d_l) on self.device
            layer_matrix = torch.stack(
                [
                    client_updates[cid][layer_name]
                    .to(dtype=torch.float64, device=self.device)
                    .flatten()
                    for cid in self._client_ids
                ],
                dim=0,
            )

            benign_indices = _benign_set_for_layer(
                layer_matrix,
                generator=self._generator,
                n_init=self.n_init,
                max_iter=self.max_iter,
            )
            benign_client_ids = {
                self._client_ids[i] for i in benign_indices.tolist()
            }
            benign_sets[layer_name] = benign_client_ids

            w_l = layer_weights[layer_name]
            for cid in benign_client_ids:
                trust[cid] += w_l

        self._trust_scores = trust
        self._benign_sets = benign_sets
        self._contamination = self._trust_to_contamination(trust)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    def detect_contamination(self, client_id: object) -> float:
        """
        Return contamination score λ_j ∈ [0, 1] for *client_id*.

        Higher → more suspicious.  Lower → more trustworthy.
        Must call fit() first.
        """
        self._require_fitted()
        if client_id not in self._contamination:
            raise KeyError(
                f"client_id={client_id!r} was not present in the last fit() call."
            )
        return float(self._contamination[client_id])

    def contamination_scores(self) -> Dict[object, float]:
        """Return {client_id: λ_j} for all clients in the last fit() call."""
        self._require_fitted()
        return dict(self._contamination)

    def trust_scores(self) -> Dict[object, float]:
        """
        Return raw trust scores (Eq. 12) — higher is safer.
        Use contamination_scores() / detect_contamination() for the PPO input.
        """
        self._require_fitted()
        return dict(self._trust_scores)

    def top_k_clients(self) -> List[object]:
        """
        Return the k client ids with the highest trust scores (set S, Eq. 14).
        Ties broken by original dict order.
        """
        self._require_fitted()
        return sorted(
            self._client_ids,
            key=lambda cid: self._trust_scores[cid],
            reverse=True,
        )[: self.k]

    def benign_sets(self) -> Dict[str, set]:
        """Return {layer_name: set_of_benign_client_ids} from the last fit()."""
        self._require_fitted()
        return {k: set(v) for k, v in self._benign_sets.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trust_to_contamination(
        self, trust: Dict[object, float]
    ) -> Dict[object, float]:
        """
        ⚑ A5: λ_j = 1 − (score_j − min) / (max − min)
        Identical scores → λ_j = 0.5.
        """
        scores = torch.tensor(
            [trust[cid] for cid in self._client_ids],
            dtype=torch.float64,
            device=self.device,
        )
        s_min, s_max = scores.min(), scores.max()
        if s_max == s_min:
            normalised = torch.full_like(scores, 0.5)
        else:
            normalised = 1.0 - (scores - s_min) / (s_max - s_min)
        normalised = normalised.clamp(0.0, 1.0)

        return {
            cid: float(normalised[i].item())
            for i, cid in enumerate(self._client_ids)
        }

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "FLGuardianDetector has not been fitted yet. "
                "Call fit(client_updates) first."
            )

    def __repr__(self) -> str:
        return (
            f"FLGuardianDetector(beta={self.beta}, k={self.k}, "
            f"device='{self.device}', fitted={self._fitted})"
        )


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def build_flguardian(
    beta: float = 2.0,
    k: int = 8,
    device: str | torch.device = "cpu",
    seed: Optional[int] = 42,
) -> FLGuardianDetector:
    """
    Factory function.

    Example
    -------
    detector = build_flguardian(beta=2, k=8, device="cuda")
    detector.fit(round_updates)
    lambda_j = detector.detect_contamination(edge_uav_id)
    """
    return FLGuardianDetector(beta=beta, k=k, device=device, seed=seed)


# ---------------------------------------------------------------------------
# ReCon HFL integration — coalition-level φ(g_j) adapter
# ---------------------------------------------------------------------------


def extract_layer_updates(
    model: torch.nn.Module,
    reference_weights: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Build FLGuardian client_updates from a local model and broadcast weights.

    g_j = w_j - w^{t-1}  (Eq. edge_grad, weight-delta form used in GTG-Shapley §4.1)
    """
    updates: Dict[str, torch.Tensor] = {}
    state = model.state_dict()
    for name, local_w in state.items():
        ref = reference_weights[name].to(device=local_w.device, dtype=local_w.dtype)
        updates[name] = (local_w - ref).detach().flatten()
    return updates


class FLGuardianHFLAdapter:
    """
    Coalition-level contamination detector for ReCon HFL (Sec. 3, Eq. lamda_def).

    Each fog UAV runs φ on its coalition's active edge UAVs after local training.
    Scores are cached and returned via __call__(uav) for reputation / PPO input.
    """

    def __init__(
        self,
        beta: float = 2.0,
        k: Optional[int] = None,
        device: str | torch.device = "cpu",
        seed: Optional[int] = 42,
    ) -> None:
        self.beta = beta
        self.k = k
        self.device = torch.device(device)
        self.seed = seed
        self._reference_weights: Dict[str, torch.Tensor] = {}
        self._scores: Dict[object, float] = {}
        self._last_trust: Dict[object, float] = {}

    def set_reference_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Store global model w^{t-1} broadcast at the start of the round."""
        self._reference_weights = {
            name: tensor.detach().clone() for name, tensor in weights.items()
        }

    def score_coalition(self, uavs: Sequence) -> Dict[str, float]:
        """
        Run FLGuardian on one coalition's active edge UAVs.

        Parameters
        ----------
        uavs:
            Iterable of edge UAV objects with uav_id and .model attributes.
        """
        uav_list = list(uavs)
        if not uav_list:
            return {}

        if len(uav_list) < 2:
            for uav in uav_list:
                self._scores[uav.uav_id] = 0.0
                self._last_trust[uav.uav_id] = 0.0
            return {uav.uav_id: 0.0 for uav in uav_list}

        client_updates = {
            uav.uav_id: extract_layer_updates(uav.model, self._reference_weights)
            for uav in uav_list
        }

        if len(uav_list) == 2:
            scores = self._pairwise_anomaly_scores(client_updates)
            self._scores.update(scores)
            self._last_trust.update({cid: 1.0 - lam for cid, lam in scores.items()})
            return scores

        k = self.k if self.k is not None else len(uav_list)
        detector = FLGuardianDetector(
            beta=self.beta,
            k=min(k, len(uav_list)),
            device=self.device,
            seed=self.seed,
        )
        detector.fit(client_updates)
        scores = detector.contamination_scores()
        trust = detector.trust_scores()
        self._scores.update(scores)
        self._last_trust.update(trust)
        return scores

    def __call__(self, uav) -> float:
        """Return cached λ_j for PPO observation (Eq. obs_vector)."""
        return float(self._scores.get(uav.uav_id, 0.0))

    def contamination_scores(self) -> Dict[object, float]:
        return dict(self._scores)

    def trust_scores(self) -> Dict[object, float]:
        return dict(self._last_trust)

    def clear_scores(self) -> None:
        self._scores.clear()
        self._last_trust.clear()

    @staticmethod
    def _pairwise_anomaly_scores(
        client_updates: Dict[object, Dict[str, torch.Tensor]],
    ) -> Dict[object, float]:
        """
        Fallback for |c_k| = 2 where k-means tie-breaking marks both clients benign.
        Uses mean cosine distance to the peer update as λ_j.
        """
        client_ids = list(client_updates.keys())
        if len(client_ids) != 2:
            return {cid: 0.0 for cid in client_ids}

        cid_a, cid_b = client_ids
        vecs_a = torch.cat([client_updates[cid_a][layer] for layer in client_updates[cid_a]])
        vecs_b = torch.cat([client_updates[cid_b][layer] for layer in client_updates[cid_b]])
        dist_ab = 1.0 - torch.nn.functional.cosine_similarity(
            vecs_a.unsqueeze(0), vecs_b.unsqueeze(0), dim=1
        ).clamp(0.0, 2.0).item()
        return {cid_a: dist_ab / 2.0, cid_b: dist_ab / 2.0}


def build_flguardian_hfl_adapter(
    beta: float = 2.0,
    k: Optional[int] = None,
    device: str | torch.device = "cpu",
    seed: Optional[int] = 42,
) -> FLGuardianHFLAdapter:
    """Factory for the ReCon HFL coalition-level FLGuardian adapter."""
    return FLGuardianHFLAdapter(beta=beta, k=k, device=device, seed=seed)
