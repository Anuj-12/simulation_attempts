"""
FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping
(Cao, Fang, Liu, Gong; NDSS 2021).

Implements the paper's server-side trust-bootstrapping mechanism as a
ReCon contamination detector phi(g_j) -> [0, 1] (ReCon.tex Eq. 6), pluggable
into hfl_rl.HFLRLStation / hfl_recovery.HFLRecoveryStation via the
coalition-adapter protocol already used by _run_contamination_detection:
    set_reference_weights(w)   - called once per round with w^(t-1)
    clear_scores()             - called once per round before scoring
    score_coalition(active)    - called once per coalition, returns
                                  {uav_id: contamination_score}

Paper mechanics reproduced here (IV-A, IV-B, Algorithm 1-2):
  - Root dataset D0: a small, clean, server-held dataset (Sec. III,
    |D0| ~ 100 examples by default, Table I), sampled IID from the
    overall training distribution (Case I).
  - Server model update g0 = ModelUpdate(w, D0, b, beta, Rl)  (Algorithm 1):
    plain local SGD from the current global weights w, discarded and
    retrained from scratch each round (no state carried across rounds).
  - Trust score TS_i = ReLU(cosine_similarity(g_i, g0))            (Eq. 2)
  - Magnitude-normalized update ḡ_i = (||g0|| / ||g_i||) · g_i     (Eq. 3)
  - FLTrust's own aggregation g = (1/sum_j TS_j) * sum_i TS_i * ḡ_i (Eq. 4)

ReCon does not use FLTrust's aggregation rule (Eq. 3-4) here — ReCon's
hierarchical FedAvg (hfl_base/hfl_rl) already owns aggregation, and
FLGuardianHFLAdapter-style detectors only ever contribute a per-client
score. So this module implements Eq. 2 (trust score) faithfully and
repurposes it, plus a signal drawn from the magnitude idea behind Eq. 3,
to produce phi. Polarity note: FLTrust's TS_i is a *trust* score (higher
= more trustworthy, 0..1). ReCon's phi is a *contamination* score (higher
= more suspicious, 0..1) — see CONTAMINATION SCORE below for the mapping
and why the magnitude term is necessary for this codebase specifically.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from hfl_common import FashionMNISTNet, HFLConfig, load_fashion_mnist

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Core FLTrust math (Algorithm 1, Eq. 2-3)
# ---------------------------------------------------------------------------


def _flatten(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in state_dict.values()])


def model_update(
    weights: Dict[str, torch.Tensor],
    dataset: Dataset,
    config: HFLConfig,
    local_epochs: int,
) -> Dict[str, torch.Tensor]:
    """
    Algorithm 1, ModelUpdate(w, D, b, beta, R):
        w' <- w
        for r = 1..R: sample batch, w' <- w' - beta * grad Loss(batch; w')
        return w' - w

    Used both for clients' local model updates (already computed elsewhere
    in ReCon by EdgeUAV.train_local) and for the server's own model update
    on the root dataset (this function, called from set_reference_weights).
    """
    model = FashionMNISTNet().to(config.device)
    model.load_state_dict(copy.deepcopy(weights))
    model.train()
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

    for _ in range(max(local_epochs, 1)):
        for images, labels in loader:
            images = images.to(config.device)
            labels = labels.to(config.device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()

    updated = model.state_dict()
    return {
        name: updated[name] - weights[name].to(config.device)
        for name in weights
    }


def cosine_similarity(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    va, vb = _flatten(a), _flatten(b)
    denom = va.norm() * vb.norm()
    if float(denom) <= _EPS:
        return 0.0
    return float(torch.dot(va, vb) / denom)


def trust_score(local_update: Dict[str, torch.Tensor], server_update: Dict[str, torch.Tensor]) -> float:
    """TS_i = ReLU(cosine(g_i, g0))  (Eq. 2)."""
    return max(cosine_similarity(local_update, server_update), 0.0)


# ---------------------------------------------------------------------------
# Root dataset (Sec. III "Defender's knowledge and capability")
# ---------------------------------------------------------------------------


def sample_root_dataset(data_dir: str, root_size: int = 100, seed: int = 1337) -> Dataset:
    """
    Case I sampling (Sec. VI-A-4): root dataset drawn IID/uniformly from
    the overall training distribution. |D0| = 100 is the paper's default
    across all six evaluated datasets (Table I) and is shown (Fig. 4) to
    already be sufficient; we keep it as the default here too.

    Note: for simplicity this samples from the full FashionMNIST training
    set independently of hfl_base/hfl_rl's edge-UAV shard partition, so
    the root dataset may overlap with some edge UAVs' shards. The paper
    assumes the server collects D0 itself (e.g. via manual labeling), so
    in a real deployment it would not be drawn from client data at all;
    this is a simulation convenience, not a paper requirement.
    """
    train_set, _ = load_fashion_mnist(data_dir)
    generator = torch.Generator().manual_seed(seed)
    n = min(root_size, len(train_set))
    indices = torch.randperm(len(train_set), generator=generator)[:n].tolist()
    return Subset(train_set, indices)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


@dataclass
class FLTrustDetector:
    """
    Coalition-level phi adapter. One server model / one g0 is computed
    per round (not per coalition, matching the paper's single centralized
    server) and reused across every coalition's score_coalition() call
    that round.

    CONTAMINATION SCORE (not in the paper — see module docstring):
    ReCon's simulated attack (RLEdgeUAV.apply_poison in hfl_rl.py) is a
    pure magnitude-scaling attack: g_i' = g_i + (g_i - w_ref) * (poison_scale
    - 1), i.e. it stretches the client's own honest gradient by a large
    factor (default 50x) without changing its direction. Cosine similarity
    is scale-invariant, so TS_i for such an update is identical to the
    unscaled, honest TS_i — Eq. 2's trust score alone is blind to this
    attack by construction. In the original paper this isn't a gap: FLTrust
    neutralizes magnitude attacks not via TS but via the separate
    normalization step (Eq. 3), which is part of its aggregation rule, not
    its scoring. Since ReCon's phi is only ever used as a *score* (feeding
    reputation/PPO, not aggregation), we fold a magnitude-deviation signal
    into phi as well, so this detector is actually useful against this
    codebase's default attack:

        magnitude_ratio_i  = ||g_i|| / max(||g0||, eps)
        magnitude_penalty  = max(0, 1 - ||g0|| / ||g_i||)   in [0, 1),
                              i.e. 0 when ||g_i|| <= ||g0||, saturating
                              toward 1 as ||g_i|| grows past ||g0||.
        contamination_i    = max(1 - TS_i, magnitude_penalty_i)

    Set include_magnitude_signal=False to fall back to the paper-exact
    "contamination = 1 - TS" mapping (direction only).
    """

    config: HFLConfig
    root_dataset: Dataset
    local_epochs: int
    include_magnitude_signal: bool = True

    _reference_weights: Optional[Dict[str, torch.Tensor]] = field(default=None, repr=False)
    _server_update: Optional[Dict[str, torch.Tensor]] = field(default=None, repr=False)
    _server_update_norm: float = field(default=0.0, repr=False)

    # Diagnostics from the most recent scoring pass, keyed by uav_id.
    last_trust_scores: Dict[str, float] = field(default_factory=dict, repr=False)
    last_magnitude_ratios: Dict[str, float] = field(default_factory=dict, repr=False)
    last_scores: Dict[str, float] = field(default_factory=dict, repr=False)

    def set_reference_weights(self, reference_weights: Dict[str, torch.Tensor]) -> None:
        """w^(t-1); trains g0 = ModelUpdate(w, D0, ...) fresh each round (Algorithm 2, line 12)."""
        self._reference_weights = reference_weights
        self._server_update = model_update(
            reference_weights, self.root_dataset, self.config, self.local_epochs
        )
        self._server_update_norm = float(_flatten(self._server_update).norm())

    def clear_scores(self) -> None:
        self.last_trust_scores.clear()
        self.last_magnitude_ratios.clear()
        self.last_scores.clear()

    def score_coalition(self, active_uavs: Iterable) -> Dict[str, float]:
        if self._reference_weights is None or self._server_update is None:
            raise RuntimeError(
                "FLTrustDetector.score_coalition() called before "
                "set_reference_weights(); the station must call it once "
                "per round before scoring any coalition."
            )

        scores: Dict[str, float] = {}
        for uav in active_uavs:
            local_update = {
                name: tensor.to(self.config.device) - self._reference_weights[name].to(self.config.device)
                for name, tensor in uav.state_dict().items()
            }
            ts = trust_score(local_update, self._server_update)

            update_norm = float(_flatten(local_update).norm())
            magnitude_ratio = update_norm / max(self._server_update_norm, _EPS)
            magnitude_penalty = max(0.0, 1.0 - 1.0 / max(magnitude_ratio, _EPS)) if magnitude_ratio > 1.0 else 0.0

            contamination = (1.0 - ts)
            if self.include_magnitude_signal:
                contamination = max(contamination, magnitude_penalty)
            contamination = min(max(contamination, 0.0), 1.0)

            scores[uav.uav_id] = contamination
            self.last_trust_scores[uav.uav_id] = ts
            self.last_magnitude_ratios[uav.uav_id] = magnitude_ratio
            self.last_scores[uav.uav_id] = contamination

        return scores


def build_fltrust_hfl_adapter(
    config: Optional[HFLConfig] = None,
    device: str = "cpu",
    data_dir: str = "./data",
    root_size: int = 100,
    local_epochs: Optional[int] = None,
    root_dataset: Optional[Dataset] = None,
    include_magnitude_signal: bool = True,
) -> FLTrustDetector:
    """
    Build a coalition-level FLTrust adapter for main.py's --detector flag.

    Mirrors flguardian_det.build_flguardian_hfl_adapter's role so main.py
    can select between --detector flguardian|fltrust|none. Prefer passing
    `config` (the run's actual HFLConfig) so the server model is trained
    with the same batch size / learning rate / local-epoch count as the
    edge UAVs (Algorithm 2 uses the same b, beta, Rl for clients and
    server) — falling back to device/data_dir alone reconstructs a default
    HFLConfig, which may not match a run invoked with non-default
    --batch-size/--lr/--local-epochs.
    """
    cfg = config or HFLConfig(data_dir=data_dir, device=device)
    epochs = local_epochs if local_epochs is not None else cfg.local_epochs
    root = root_dataset if root_dataset is not None else sample_root_dataset(cfg.data_dir, root_size)
    return FLTrustDetector(
        config=cfg,
        root_dataset=root,
        local_epochs=epochs,
        include_magnitude_signal=include_magnitude_signal,
    )
