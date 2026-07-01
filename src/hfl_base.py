"""
Base HFL UAV structure for ReCon-style hierarchical federated learning.

Topology (ReCon Sec. 3):
  Edge UAVs (U)  ->  Fog UAVs (H) per coalition (C)  ->  Base station

This module covers local training and hierarchical aggregation only.
RL state detection and checkpoint recovery are intentionally omitted.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from hfl_common import FashionMNISTNet, HFLConfig, load_fashion_mnist, partition_dataset


# ---------------------------------------------------------------------------
# Edge UAV
# ---------------------------------------------------------------------------


@dataclass
class EdgeUAV:
    """
    Edge UAV u_j: collects data via sensors and trains a local model (Eq. 1–3).

    All edge UAVs are active in this base implementation.
    """

    uav_id: str
    coalition_id: str
    dataset: Dataset
    model: FashionMNISTNet = field(default_factory=FashionMNISTNet)
    sensors: List[str] = field(default_factory=list)

    @property
    def num_samples(self) -> int:
        return len(self.dataset)

    def train_local(self, config: HFLConfig) -> float:
        """Run local SGD on this UAV's FashionMNIST shard; return average loss."""
        self.model.train()
        loader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.SGD(self.model.parameters(), lr=config.learning_rate)
        total_loss = 0.0
        num_batches = 0

        for _ in range(config.local_epochs):
            for images, labels in loader:
                images = images.to(config.device)
                labels = labels.to(config.device)
                optimizer.zero_grad()
                loss = F.cross_entropy(self.model(images), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def evaluate(self, dataset: Dataset, config: HFLConfig) -> Tuple[float, float]:
        """Return (loss, accuracy) on a held-out dataset."""
        self.model.eval()
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(config.device)
                labels = labels.to(config.device)
                logits = self.model(images)
                total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

        return total_loss / max(total, 1), correct / max(total, 1)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return copy.deepcopy(self.model.state_dict())

    def load_state_dict(self, weights: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(copy.deepcopy(weights))


# ---------------------------------------------------------------------------
# Coalition
# ---------------------------------------------------------------------------


@dataclass
class Coalition:
    """
    Coalition c_k: group of edge UAVs whose updates are aggregated locally (Eq. 4).
    """

    coalition_id: str
    edge_uavs: List[EdgeUAV] = field(default_factory=list)
    fog_uav: Optional["FogUAV"] = None

    @property
    def total_samples(self) -> int:
        return sum(uav.num_samples for uav in self.edge_uavs)

    def aggregate_weights(self) -> Dict[str, torch.Tensor]:
        """
        FedAvg over edge models in this coalition:
          w_k^F = sum_j (n_j / N_k) * w_j   (Eq. 4a, weight form)
        """
        if not self.edge_uavs:
            raise RuntimeError(f"Coalition {self.coalition_id} has no edge UAVs.")

        total = self.total_samples
        aggregated: Dict[str, torch.Tensor] = {}

        for uav in self.edge_uavs:
            weight = uav.num_samples / total
            for key, tensor in uav.state_dict().items():
                aggregated[key] = aggregated.get(key, torch.zeros_like(tensor)) + weight * tensor

        return aggregated

    def distribute_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        for uav in self.edge_uavs:
            uav.load_state_dict(weights)


# ---------------------------------------------------------------------------
# Fog UAV
# ---------------------------------------------------------------------------


@dataclass
class FogUAV:
    """
    Fog UAV h_i: one per coalition; aggregates edge updates before the base
    station (Eq. 4).
    """

    fog_id: str
    coalition: Coalition

    def aggregate(self) -> Dict[str, torch.Tensor]:
        return self.coalition.aggregate_weights()

    def distribute(self, weights: Dict[str, torch.Tensor]) -> None:
        self.coalition.distribute_weights(weights)


# ---------------------------------------------------------------------------
# Base station
# ---------------------------------------------------------------------------


class BaseStation:
    """
    Central node performing global aggregation across fog/coalition models (Eq. 5).
    """

    def __init__(self, config: Optional[HFLConfig] = None) -> None:
        self.config = config or HFLConfig()
        self.global_model = FashionMNISTNet().to(self.config.device)
        self.edge_uavs: Dict[str, EdgeUAV] = {}
        self.fog_uavs: Dict[str, FogUAV] = {}
        self.coalitions: Dict[str, Coalition] = {}

    @property
    def all_edge_uavs(self) -> List[EdgeUAV]:
        return list(self.edge_uavs.values())

    def register_coalition(self, coalition: Coalition) -> None:
        self.coalitions[coalition.coalition_id] = coalition

    def register_fog_uav(self, fog_uav: FogUAV) -> None:
        self.fog_uavs[fog_uav.fog_id] = fog_uav
        coalition = fog_uav.coalition
        coalition.fog_uav = fog_uav
        self.register_coalition(coalition)
        for uav in coalition.edge_uavs:
            self.edge_uavs[uav.uav_id] = uav

    def global_aggregate(self) -> Dict[str, torch.Tensor]:
        """
        Global FedAvg across coalitions:
          w^G = sum_k (N_k / N) * w_k^F   (Eq. 5a, weight form)
        """
        fog_weights: List[Tuple[int, Dict[str, torch.Tensor]]] = []
        for fog in self.fog_uavs.values():
            fog_weights.append((fog.coalition.total_samples, fog.aggregate()))

        total_samples = sum(n for n, _ in fog_weights)
        global_weights: Dict[str, torch.Tensor] = {}

        for n_k, weights in fog_weights:
            coef = n_k / total_samples
            for key, tensor in weights.items():
                global_weights[key] = global_weights.get(key, torch.zeros_like(tensor)) + coef * tensor

        return global_weights

    def distribute_global_model(self) -> None:
        weights = copy.deepcopy(self.global_model.state_dict())
        for uav in self.all_edge_uavs:
            uav.load_state_dict(weights)

    def train_round(self, round_idx: int) -> Dict[str, float]:
        """One HFL round: broadcast -> local train -> coalition agg -> global agg."""
        self.distribute_global_model()

        local_losses: Dict[str, float] = {}
        for uav in self.all_edge_uavs:
            local_losses[uav.uav_id] = uav.train_local(self.config)

        coalition_weights = {cid: c.aggregate_weights() for cid, c in self.coalitions.items()}
        for coalition in self.coalitions.values():
            coalition.distribute_weights(coalition_weights[coalition.coalition_id])

        global_weights = self.global_aggregate()
        self.global_model.load_state_dict(global_weights)
        self.distribute_global_model()

        return local_losses

    def run(self, test_dataset: Dataset) -> List[Dict[str, float]]:
        history: List[Dict[str, float]] = []
        for round_idx in range(1, self.config.num_rounds + 1):
            losses = self.train_round(round_idx)
            loss, acc = self._evaluate_global(test_dataset)
            avg_local_loss = sum(losses.values()) / len(losses)
            history.append({
                "round": round_idx,
                "global_loss": loss,
                "global_accuracy": acc,
                "avg_local_loss": avg_local_loss,
            })
            print(
                f"Round {round_idx}/{self.config.num_rounds} | "
                f"global acc={acc:.4f} loss={loss:.4f} | "
                f"avg local loss={avg_local_loss:.4f}"
            )
        return history

    def _evaluate_global(self, test_dataset: Dataset) -> Tuple[float, float]:
        self.global_model.eval()
        loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.config.device)
                labels = labels.to(self.config.device)
                logits = self.global_model(images)
                total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

        return total_loss / max(total, 1), correct / max(total, 1)


# ---------------------------------------------------------------------------
# System builder
# ---------------------------------------------------------------------------


def build_hfl_system(
    coalition_specs: Sequence[Tuple[str, Sequence[str]]],
    config: Optional[HFLConfig] = None,
) -> BaseStation:
    """
    Build an HFL system with FashionMNIST shards assigned to edge UAVs.

    Parameters
    ----------
    coalition_specs:
        Sequence of (coalition_id, [edge_uav_id, ...]).
        Each edge_uav_id receives a unique data shard.
    """
    cfg = config or HFLConfig()
    all_edge_ids = [eid for _, members in coalition_specs for eid in members]
    train_set, _ = load_fashion_mnist(cfg.data_dir)
    shards = partition_dataset(train_set, len(all_edge_ids))
    shard_map = dict(zip(all_edge_ids, shards))

    station = BaseStation(cfg)
    for coalition_id, edge_ids in coalition_specs:
        edge_uavs = [
            EdgeUAV(
                uav_id=eid,
                coalition_id=coalition_id,
                dataset=shard_map[eid],
                sensors=[f"{eid}_cam"],
            )
            for eid in edge_ids
        ]
        coalition = Coalition(coalition_id=coalition_id, edge_uavs=edge_uavs)
        fog = FogUAV(fog_id=f"fog_{coalition_id}", coalition=coalition)
        station.register_fog_uav(fog)

    return station
