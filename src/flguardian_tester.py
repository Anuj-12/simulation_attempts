"""
Tests for FLGuardianDetector  —  PyTorch implementation
========================================================
Mirrors the original NumPy test suite exactly, with all array/numpy calls
replaced by torch equivalents.  Every test is labelled with the paper
section / equation it validates.

Run with:  python -m pytest test_flguardian_detector_torch.py -v
"""

import warnings
import pytest
import torch

from flguardian_det import (
    FLGuardianDetector,
    build_flguardian,
    _pairwise_cosine_distances,
    _pairwise_euclidean_distances,
    _benign_set_for_layer,
)

# Shared generator for reproducible random tensors in tests
GEN = torch.Generator().manual_seed(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand(*shape, seed=0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.empty(*shape, dtype=torch.float64).normal_(generator=g)


def make_updates(n_clients: int, n_layers: int, d: int, seed: int = 0):
    """Return {client_id: {layer_name: 1-D tensor}} with random updates."""
    layer_names = [f"layer{i}" for i in range(n_layers)]
    return {
        f"client{c}": {
            ln: _rand(d, seed=seed * 1000 + c * 10 + li)
            for li, ln in enumerate(layer_names)
        }
        for c in range(n_clients)
    }


def make_poisoned_updates(
    n_clients: int,
    n_malicious: int,
    n_layers: int,
    d: int,
    poison_scale: float = 50.0,
    seed: int = 42,
):
    layer_names = [f"layer{i}" for i in range(n_layers)]
    updates = {}
    for c in range(n_clients):
        scale = poison_scale if c < n_malicious else 1.0
        updates[f"client{c}"] = {
            ln: _rand(d, seed=seed * 1000 + c * 10 + li) * scale
            for li, ln in enumerate(layer_names)
        }
    return updates


# ---------------------------------------------------------------------------
# §IV-B  Distance functions  (Eqs. 9, 11)
# ---------------------------------------------------------------------------

class TestDistanceFunctions:

    def test_cosine_distance_self_is_zero(self):
        """cos(v,v) = 1 → distance = 0"""
        v = _rand(4, 20)
        C = _pairwise_cosine_distances(v)
        assert C.diagonal().abs().max().item() < 1e-10

    def test_cosine_distance_symmetric(self):
        v = _rand(5, 15, seed=1)
        C = _pairwise_cosine_distances(v)
        assert (C - C.T).abs().max().item() < 1e-10

    def test_cosine_distance_range(self):
        """Cosine distance ∈ [0, 2]"""
        v = _rand(6, 10, seed=2)
        C = _pairwise_cosine_distances(v)
        assert C.min().item() >= -1e-10
        assert C.max().item() <= 2.0 + 1e-10

    def test_euclidean_distance_self_is_zero(self):
        # ||a-a||² can accumulate ~1e-7 float drift; tolerate 1e-6
        v = _rand(4, 20, seed=3)
        E = _pairwise_euclidean_distances(v)
        assert E.diagonal().abs().max().item() < 1e-6

    def test_euclidean_distance_symmetric(self):
        v = _rand(5, 15, seed=4)
        E = _pairwise_euclidean_distances(v)
        assert (E - E.T).abs().max().item() < 1e-10

    def test_euclidean_distance_nonnegative(self):
        v = _rand(6, 10, seed=5)
        E = _pairwise_euclidean_distances(v)
        assert E.min().item() >= -1e-10

    def test_euclidean_distance_triangle_inequality(self):
        v = _rand(3, 8, seed=6)
        E = _pairwise_euclidean_distances(v)
        assert E[0, 2].item() <= E[0, 1].item() + E[1, 2].item() + 1e-10

    def test_appendix_relationship_unit_vectors(self):
        """
        Appendix A (Eq. 22): for unit vectors,
        Euclidean(x,y) = sqrt(2 * Cosine(x,y))
        """
        v = _rand(4, 16, seed=7)
        v = v / v.norm(dim=1, keepdim=True)
        C = _pairwise_cosine_distances(v)
        E = _pairwise_euclidean_distances(v)
        expected = (2.0 * C).clamp(min=0.0).sqrt()
        assert (E - expected).abs().max().item() < 1e-8

    def test_zero_norm_vector_handled(self):
        """Zero-norm update should not raise; cosine distance = 1."""
        v = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
        C = _pairwise_cosine_distances(v)
        assert abs(C[0, 1].item() - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# §IV-B  Benign set intersection
# ---------------------------------------------------------------------------

class TestBenignSet:

    def test_benign_set_subset_of_clients(self):
        n = 8
        layer = _rand(n, 20, seed=10)
        g = torch.Generator().manual_seed(0)
        idx = _benign_set_for_layer(layer, generator=g, n_init=5, max_iter=300)
        assert set(idx.tolist()).issubset(set(range(n)))

    def test_benign_set_majority_benign(self):
        """
        With 2 clearly malicious (100× scale) out of 10, the intersection
        benign set should contain most of the 8 genuine clients.
        """
        layer = _rand(10, 50, seed=11)
        layer[0] = layer[0] * 100
        layer[1] = layer[1] * 100
        g = torch.Generator().manual_seed(0)
        idx = _benign_set_for_layer(layer, generator=g, n_init=5, max_iter=300)
        benign_survivors = set(range(2, 10)) & set(idx.tolist())
        assert len(benign_survivors) >= 4, \
            f"Only {len(benign_survivors)} benign clients survived: {idx.tolist()}"


# ---------------------------------------------------------------------------
# Eq. (12): Trust score formula
# ---------------------------------------------------------------------------

class TestTrustScores:

    def test_trust_score_range(self):
        det = build_flguardian(beta=2, k=6)
        updates = make_updates(8, 4, 32)
        det.fit(updates)
        for cid, score in det.trust_scores().items():
            assert score >= 0.0, f"Negative trust score for {cid}: {score}"

    def test_trust_score_beta_effect(self):
        """Max trust score ≤ Σ β^l for l=1..L"""
        n_layers, beta = 4, 2.0
        max_possible = sum(beta ** l for l in range(1, n_layers + 1))
        det = build_flguardian(beta=beta, k=6)
        det.fit(make_updates(8, n_layers, 32))
        for cid, score in det.trust_scores().items():
            assert score <= max_possible + 1e-9, \
                f"{cid}: score {score} > max {max_possible}"

    def test_trust_score_beta_1_equal_layers(self):
        """beta=1 → each layer weight = 1; max score = L"""
        n_layers = 5
        det = build_flguardian(beta=1, k=4)
        det.fit(make_updates(8, n_layers, 16))
        for score in det.trust_scores().values():
            assert score <= n_layers + 1e-9


# ---------------------------------------------------------------------------
# §IV-C  Layer-biased selection — top_k_clients (Eq. 14)
# ---------------------------------------------------------------------------

class TestTopKClients:

    def test_top_k_returns_k_clients(self):
        det = build_flguardian(beta=2, k=6)
        det.fit(make_updates(10, 3, 20))
        assert len(det.top_k_clients()) == 6

    def test_top_k_clients_are_subset(self):
        updates = make_updates(8, 3, 20)
        det = build_flguardian(beta=2, k=4)
        det.fit(updates)
        assert set(det.top_k_clients()).issubset(set(updates.keys()))

    def test_top_k_malicious_excluded(self):
        """Clearly poisoned clients should not appear in the top-k."""
        updates = make_poisoned_updates(10, 2, 3, 64)
        det = build_flguardian(beta=2, k=8)
        det.fit(updates)
        selected = det.top_k_clients()
        assert "client0" not in selected
        assert "client1" not in selected


# ---------------------------------------------------------------------------
# ⚑ A5  Contamination score mapping
# ---------------------------------------------------------------------------

class TestContaminationScores:

    def test_contamination_range(self):
        """λ_j ∈ [0, 1]"""
        det = build_flguardian(beta=2, k=6)
        det.fit(make_updates(8, 4, 20))
        for cid, lam in det.contamination_scores().items():
            assert 0.0 <= lam <= 1.0, f"λ out of range for {cid}: {lam}"

    def test_contamination_inversion(self):
        """Highest trust → lowest contamination."""
        updates = make_poisoned_updates(10, 2, 3, 64)
        det = build_flguardian(beta=2, k=8)
        det.fit(updates)
        trust = det.trust_scores()
        contamination = det.contamination_scores()
        best_trusted = max(trust, key=trust.get)
        lowest_contamination = min(contamination, key=contamination.get)
        assert best_trusted == lowest_contamination, (
            f"Highest trust {best_trusted} ≠ lowest contamination {lowest_contamination}"
        )

    def test_contamination_identical_scores(self):
        """⚑ A5: all equal trust → λ = 0.5"""
        det = FLGuardianDetector(beta=2, k=2)
        det._client_ids = ["a", "b", "c"]
        det._trust_scores = {"a": 5.0, "b": 5.0, "c": 5.0}
        det._contamination = det._trust_to_contamination(det._trust_scores)
        for lam in det._contamination.values():
            assert lam == pytest.approx(0.5)

    def test_detect_contamination_single_client(self):
        det = build_flguardian(beta=2, k=1)
        updates = {"uav_0": {"layer0": torch.tensor([1.0, 2.0, 3.0])}}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            det.fit(updates)
            assert len(w) == 1
            assert issubclass(w[0].category, RuntimeWarning)
        assert det.detect_contamination("uav_0") == 0.0


# ---------------------------------------------------------------------------
# Interface / integration tests
# ---------------------------------------------------------------------------

class TestInterface:

    def test_fit_returns_self(self):
        det = FLGuardianDetector()
        assert det.fit(make_updates(6, 2, 16)) is det

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError, match="not been fitted"):
            FLGuardianDetector().detect_contamination("x")

    def test_unknown_client_raises(self):
        det = FLGuardianDetector()
        det.fit(make_updates(4, 2, 8))
        with pytest.raises(KeyError):
            det.detect_contamination("ghost")

    def test_inconsistent_layers_raises(self):
        updates = {
            "c0": {"layer0": torch.ones(4), "layer1": torch.ones(4)},
            "c1": {"layer0": torch.ones(4), "layer2": torch.ones(4)},
        }
        with pytest.raises(ValueError, match="layer names"):
            FLGuardianDetector().fit(updates)

    def test_build_flguardian_factory(self):
        det = build_flguardian(beta=1.5, k=5, seed=7)
        assert det.beta == 1.5
        assert det.k == 5
        assert det.seed == 7

    def test_repr(self):
        r = repr(FLGuardianDetector(beta=2, k=8))
        assert "FLGuardianDetector" in r
        assert "fitted=False" in r

    def test_multiple_rounds(self):
        det = build_flguardian(beta=2, k=4)
        for i in range(3):
            det.fit(make_updates(6, 3, 20, seed=i))
        assert len(det.contamination_scores()) == 6

    def test_integer_uav_ids(self):
        """Client IDs can be integers (UAV indices)."""
        updates = {
            i: {"layer0": torch.randn(16), "layer1": torch.randn(8)}
            for i in range(6)
        }
        det = build_flguardian(beta=2, k=4)
        det.fit(updates)
        for i in range(6):
            lam = det.detect_contamination(i)
            assert 0.0 <= lam <= 1.0

    def test_benign_sets_exposed(self):
        det = build_flguardian(beta=2, k=4)
        det.fit(make_updates(6, 3, 16))
        bs = det.benign_sets()
        assert set(bs.keys()) == {"layer0", "layer1", "layer2"}

    def test_gpu_tensors_accepted(self):
        """Tensors on CUDA move to detector device without error."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        updates = {
            f"c{i}": {"layer0": torch.randn(32).cuda(), "layer1": torch.randn(16).cuda()}
            for i in range(6)
        }
        det = build_flguardian(device="cuda")
        det.fit(updates)
        for lam in det.contamination_scores().values():
            assert 0.0 <= lam <= 1.0

    def test_mixed_precision_tensors(self):
        """float32 input tensors are safely cast to float64 internally."""
        updates = {
            f"c{i}": {"layer0": torch.randn(32, dtype=torch.float32)}
            for i in range(6)
        }
        det = build_flguardian(beta=2, k=4)
        det.fit(updates)
        for lam in det.contamination_scores().values():
            assert 0.0 <= lam <= 1.0

    def test_beta_invalid(self):
        with pytest.raises(ValueError, match="beta must be positive"):
            FLGuardianDetector(beta=-1)

    def test_k_invalid(self):
        with pytest.raises(ValueError, match="k must be"):
            FLGuardianDetector(k=0)

    def test_named_parameters_workflow(self):
        """
        Simulate the real caller workflow: extract updates from two torch
        nn.Linear layers and pass them directly to the detector.
        """
        import torch.nn as nn

        def get_update(model_new, model_old):
            return {
                name: (p_new - p_old).detach().flatten()
                for (name, p_new), (_, p_old)
                in zip(model_new.named_parameters(), model_old.named_parameters())
            }

        n = 6
        global_model = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2))
        updates = {}
        for i in range(n):
            local = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2))
            updates[f"uav_{i}"] = get_update(local, global_model)

        det = build_flguardian(beta=2, k=4)
        det.fit(updates)
        for uid in updates:
            lam = det.detect_contamination(uid)
            assert 0.0 <= lam <= 1.0


# ---------------------------------------------------------------------------
# Paper-level claim: deep-layer attack detected via β weighting
# ---------------------------------------------------------------------------

class TestPaperClaimLayerSpaceAttack:

    def test_deep_layer_attack_detected(self):
        """
        Synthetic LPattack: only the deepest layer is poisoned.
        β > 1 means the deep-layer signal dominates → malicious clients
        receive higher contamination scores than benign ones.
        """
        n_total, n_mal, n_layers, d = 10, 2, 5, 128
        layer_names = [f"layer{i}" for i in range(n_layers)]
        updates = {}
        g = torch.Generator().manual_seed(99)
        for c in range(n_total):
            update = {ln: torch.empty(d, dtype=torch.float64).normal_(generator=g)
                      for ln in layer_names}
            if c < n_mal:
                # Poison only the deepest layer — mimics LPattack
                update[layer_names[-1]] = (
                    torch.empty(d, dtype=torch.float64).normal_(generator=g) * 80
                )
            updates[f"client{c}"] = update

        det = build_flguardian(beta=2, k=8)
        det.fit(updates)
        scores = det.contamination_scores()

        mal_mean = sum(scores[f"client{c}"] for c in range(n_mal)) / n_mal
        benign_mean = sum(scores[f"client{c}"] for c in range(n_mal, n_total)) / (n_total - n_mal)
        assert mal_mean > benign_mean, (
            f"Malicious mean {mal_mean:.3f} not > benign mean {benign_mean:.3f}"
        )


if __name__ == "__main__":
    import sys, pytest as _pt
    sys.exit(_pt.main([__file__, "-v"]))
