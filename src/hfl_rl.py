"""
PPO-based UAV state management for ReCon HFL (Sec. 4.2, Algorithm 1).

Builds on hfl_base.py and implements:
  - Observation o_j = (lambda_j, q_j, rho_j)           (Eq. 20)
  - Action space {Allow, Quarantine, Exclude}        (Eq. 23)
  - Reputation update                                 (Eq. 7)
  - Quarantine duration T_j^Q                         (Eq. 28)
  - Reward R_j                                        (Eq. 26–27)
  - PPO clipped objective + value + entropy losses    (Eq. 21, 29–33)

Contamination detection phi(g_j) is injectable; a zero-score stub is used
by default until an external detector is provided.

Recovery / checkpoint mechanisms (Sec. 4.3) are intentionally omitted.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset

from hfl_base import BaseStation, Coalition, EdgeUAV, FogUAV
from hfl_common import HFLConfig, load_fashion_mnist, partition_dataset


# ---------------------------------------------------------------------------
# Enums and protocols
# ---------------------------------------------------------------------------


class ParticipationState(Enum):
    """Edge UAV partitions U^A, U^Q, U^E."""

    ACTIVE = auto()
    QUARANTINED = auto()
    EXCLUDED = auto()


# Explicit values because PPO outputs 0, 1, and 2
class UAVAction(Enum):
    """Discrete PPO action space A (Eq. 23)."""

    ALLOW = 0
    QUARANTINE = 1
    EXCLUDE = 2


# Protocol could have other methods that have been listed ahead with hasattr()
class ContaminationDetector(Protocol):
    """Black-box phi: g_j -> [0, 1] (Eq. 6). Provided by caller later."""

    def __call__(self, uav: "RLEdgeUAV") -> float:
        ...


def zero_contamination_detector(_: "RLEdgeUAV") -> float:
    """Default stub until a real phi is supplied."""
    return 0.0


# ---------------------------------------------------------------------------
# RL configuration
# ---------------------------------------------------------------------------


@dataclass
class RLConfig:
    """Hyper-parameters for reputation and PPO (ReCon Sec. 4.1–4.2)."""

    initial_reputation: float = 0.5          # rho_0
    reputation_lr: float = 0.1               # eta
    penalty_tuning: float = 1.0              # tau
    initial_energy: float = 1000.0
    default_contribution: float = 1.0        # phi_j placeholder
    retraining_energy_scale: float = 10.0  # simplified E_j^ret

    # PPO
    ppo_lr: float = 3e-4
    gamma: float = 0.99
    clip_epsilon: float = 0.2
    value_coef: float = 0.5                  # l1
    entropy_coef: float = 0.01               # l2
    ppo_epochs: int = 4
    hidden_dim: int = 64


# ---------------------------------------------------------------------------
# RL-enriched edge UAV
# ---------------------------------------------------------------------------


@dataclass
class RLEdgeUAV(EdgeUAV):
    """Edge UAV extended with reputation, flags, and participation state."""

    participation: ParticipationState = ParticipationState.ACTIVE
    reputation: float = 0.5
    flag_count: int = 0
    contamination_score: float = 0.0
    quarantine_rounds_remaining: int = 0
    residual_energy: float = 1000.0
    retraining_energy: float = 0.0
    model_contribution: float = 1.0
    is_malicious: bool = False
    poison_scale: float = 50.0

    @property
    def is_active(self) -> bool:
        return self.participation == ParticipationState.ACTIVE

    def apply_poison(self, reference_weights: Dict[str, torch.Tensor]) -> None:
        """Scale local update for simulated model-poisoning attacks."""
        if not self.is_malicious:
            return
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                ref = reference_weights[name].to(param.device)
                delta = param.data - ref
                param.data.copy_(ref + delta * self.poison_scale)

    def observation(self) -> torch.Tensor:
        """o_j = (lambda_j, q_j, rho_j)  (Eq. 20)."""
        return torch.tensor(
            [
                self.contamination_score,
                float(self.flag_count) / 5.0,
                self.reputation / 10.0,
            ],
            dtype=torch.float32,
        )

    def update_reputation(self, eta: float, tau: float) -> float:
        """Eq. 7 (energy terms normalized to keep ρ in a stable range)."""
        lam = self.contamination_score
        q = self.flag_count
        e_res = self.residual_energy / 1000.0
        e_ret = max(self.retraining_energy, 0.0) / 100.0
        phi = max(self.model_contribution, 1e-8)

        reward_term = (1.0 - lam) * (e_res / (1.0 + q)) * phi
        penalty_term = lam * (e_ret / phi) * math.exp(tau * q)
        self.reputation = self.reputation + eta * (reward_term - penalty_term)
        self.reputation = max(min(self.reputation, 10.0), 0.0)
        return self.reputation

    def assign_quarantine(self, tau: float) -> int:
        """T_j^Q = exp(tau * q_j) / (1 + rho_j)  (Eq. 28)."""
        self.flag_count += 1
        duration = int(math.exp(tau * self.flag_count) / (1.0 + max(self.reputation, 0.0)))
        self.quarantine_rounds_remaining = max(duration, 1)
        self.participation = ParticipationState.QUARANTINED
        return self.quarantine_rounds_remaining

    def exclude(self) -> None:
        self.participation = ParticipationState.EXCLUDED

    def tick_quarantine(self) -> bool:
        """Decrement T_j^Q; return True when UAV rejoins U^A."""
        if self.participation != ParticipationState.QUARANTINED:
            return False
        self.quarantine_rounds_remaining -= 1
        if self.quarantine_rounds_remaining <= 0:
            self.participation = ParticipationState.ACTIVE
            self.quarantine_rounds_remaining = 0
            return True
        return False


# ---------------------------------------------------------------------------
# RL coalition / fog
# ---------------------------------------------------------------------------


@dataclass
class RLCoalition(Coalition):
    """Coalition that aggregates only active edge UAVs (Algorithm 1)."""

    @property
    def active_edge_uavs(self) -> List[RLEdgeUAV]:
        return [u for u in self.edge_uavs if u.is_active]

    @property
    def active_samples(self) -> int:
        return sum(u.num_samples for u in self.active_edge_uavs)

    def aggregate_active_weights(self) -> Dict[str, torch.Tensor]:
        """FedAvg over u_j in c_k intersect U^A (Eq. 4)."""
        active = self.active_edge_uavs
        if not active:
            raise RuntimeError(f"Coalition {self.coalition_id} has no active edge UAVs.")

        total = self.active_samples
        aggregated: Dict[str, torch.Tensor] = {}
        for uav in active:
            weight = uav.num_samples / total
            for key, tensor in uav.state_dict().items():
                aggregated[key] = aggregated.get(key, torch.zeros_like(tensor)) + weight * tensor
        return aggregated

    def evaluate_accuracy(self, dataset: Dataset, config: HFLConfig) -> float:
        """Acc(M_ck, D_val) used in Eq. 27."""
        if not self.active_edge_uavs:
            return 0.0
        weights = self.aggregate_active_weights()
        probe = copy.deepcopy(self.active_edge_uavs[0].model)
        probe.load_state_dict(weights)
        probe.eval()

        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(config.device)
                labels = labels.to(config.device)
                preds = probe(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / max(total, 1)


@dataclass
class RLFogUAV(FogUAV):
    """Fog UAV that runs PPO state decisions for its coalition (Algorithm 1)."""

    coalition: RLCoalition = field(default_factory=RLCoalition)  # type: ignore[assignment]

    def apply_action(self, uav: RLEdgeUAV, action: UAVAction, tau: float) -> None:
        if action == UAVAction.ALLOW:
            return
        if action == UAVAction.QUARANTINE:
            uav.assign_quarantine(tau)
        elif action == UAVAction.EXCLUDE:
            uav.exclude()


# ---------------------------------------------------------------------------
# PPO agent (Eq. 21, 24–25, 29–33)
# ---------------------------------------------------------------------------


class ActorCritic(nn.Module):
    """Shared actor-critic network for PPO."""

    def __init__(self, obs_dim: int = 3, action_dim: int = 3, hidden_dim: int = 64) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(obs)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value

    def act(self, obs: torch.Tensor) -> Tuple[int, float, float, torch.Tensor]:
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return int(action.item()), float(dist.log_prob(action).item()), float(value.item()), dist.entropy()


@dataclass
class Transition:
    obs: torch.Tensor
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool


@dataclass
class UAVRoundSnapshot:
    """Per-UAV metrics captured each FL round for readiness reporting."""

    uav_id: str
    coalition_id: str
    contamination_score: float
    reputation: float
    flag_count: int
    action: str
    participation: str
    quarantine_remaining: int
    is_malicious: bool


@dataclass
class RoundSnapshot:
    """One FL round summary for visible system output."""

    round_idx: int
    global_accuracy: float
    global_loss: float
    active_uavs: int
    quarantined_uavs: int
    excluded_uavs: int
    uav_snapshots: List[UAVRoundSnapshot]
    ppo_policy_loss: float = 0.0
    last_checkpoint_round: int = 0


class PPOAgent:
    """Proximal Policy Optimization for UAV state management."""

    def __init__(self, rl_config: RLConfig, device: str = "cpu") -> None:
        self.config = rl_config
        self.device = device
        self.network = ActorCritic(hidden_dim=rl_config.hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=rl_config.ppo_lr)
        self.buffer: List[Transition] = []

    def select_action(self, obs: torch.Tensor) -> Tuple[UAVAction, float, float]:
        obs = obs.to(self.device)
        action_idx, log_prob, value, _ = self.network.act(obs)
        return UAVAction(action_idx), log_prob, value

    def store(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def clear_buffer(self) -> None:
        self.buffer.clear()

    def _compute_returns(self) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards = [t.reward for t in self.buffer]
        values = [t.value for t in self.buffer]
        dones = [t.done for t in self.buffer]

        returns: List[float] = []
        g = 0.0
        for step in reversed(range(len(rewards))):
            if dones[step]:
                g = 0.0
            g = rewards[step] + self.config.gamma * g
            returns.insert(0, g)

        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        values_t = torch.tensor(values, dtype=torch.float32, device=self.device)
        advantages = returns_t - values_t
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns_t, advantages

    def update(self) -> Dict[str, float]:
        """Optimize L_PPO = L_CLIP - l1 * L_VF - l2 * entropy  (Eq. 29)."""
        if not self.buffer:
            return {}

        obs = torch.stack([t.obs for t in self.buffer]).to(self.device)
        actions = torch.tensor([t.action for t in self.buffer], device=self.device)
        old_log_probs = torch.tensor([t.log_prob for t in self.buffer], device=self.device)
        returns, advantages = self._compute_returns()

        clip_stats: Dict[str, float] = {}
        for _ in range(self.config.ppo_epochs):
            logits, values = self.network(obs)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, returns)
            loss = (
                policy_loss
                + self.config.value_coef * value_loss
                - self.config.entropy_coef * entropy
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            clip_stats = {
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy.item()),
            }

        self.clear_buffer()
        return clip_stats


# ---------------------------------------------------------------------------
# Reward and energy helpers
# ---------------------------------------------------------------------------


def estimate_retraining_energy(uav: RLEdgeUAV, coalition: RLCoalition, scale: float) -> float:
    """Simplified stand-in for Eq. 13–15 until full energy model is wired."""
    return scale * max(len(coalition.edge_uavs) - 1, 1)


def compute_reward(
    uav: RLEdgeUAV,
    delta_psi: float,
    quarantine_assigned: int,
    rl_config: RLConfig,
) -> float:
    lam = uav.contamination_score
    e_ret = max(uav.retraining_energy, 0.0)
    return (
        uav.reputation
        + delta_psi
        - float(quarantine_assigned)
        - lam * math.log1p(e_ret)
    )


# ---------------------------------------------------------------------------
# HFL + PPO base station (Algorithm 1)
# ---------------------------------------------------------------------------


class HFLRLStation(BaseStation):
    """
    Extends BaseStation with PPO-based UAV state management (Algorithm 1).

    Recovery / checkpoint steps from Sec. 4.3 are not included.
    """

    def __init__(
        self,
        config: Optional[HFLConfig] = None,
        rl_config: Optional[RLConfig] = None,
        contamination_detector: Optional[ContaminationDetector] = None,
    ) -> None:
        super().__init__(config)
        self.rl_config = rl_config or RLConfig()
        self.detector = contamination_detector or zero_contamination_detector
        self.ppo = PPOAgent(self.rl_config, device=self.config.device)
        self.validation_set: Optional[Dataset] = None
        self.round_actions: Dict[str, UAVAction] = {}
        self.round_history: List[RoundSnapshot] = []

    @property
    def rl_edge_uavs(self) -> List[RLEdgeUAV]:
        return [u for u in self.all_edge_uavs if isinstance(u, RLEdgeUAV)]

    @property
    def active_uavs(self) -> List[RLEdgeUAV]:
        return [u for u in self.rl_edge_uavs if u.is_active]

    @property
    def quarantined_uavs(self) -> List[RLEdgeUAV]:
        return [u for u in self.rl_edge_uavs if u.participation == ParticipationState.QUARANTINED]

    @property
    def excluded_uavs(self) -> List[RLEdgeUAV]:
        return [u for u in self.rl_edge_uavs if u.participation == ParticipationState.EXCLUDED]

    @property
    def rl_coalitions(self) -> List[RLCoalition]:
        return [c for c in self.coalitions.values() if isinstance(c, RLCoalition)]

    def _run_contamination_detection(self, reference_weights: Dict[str, torch.Tensor]) -> None:
        """
        Run φ on each coalition via fog UAV (Sec. 3, Eq. lamda_def).

        Supports coalition-level adapters (FLGuardianHFLAdapter) and legacy
        per-UAV callables.
        """
        if hasattr(self.detector, "set_reference_weights"):
            self.detector.set_reference_weights(reference_weights)
        if hasattr(self.detector, "clear_scores"):
            self.detector.clear_scores()

        for coalition in self.rl_coalitions:
            active = coalition.active_edge_uavs
            if hasattr(self.detector, "score_coalition"):
                self.detector.score_coalition(active)
            else:
                for uav in active:
                    uav.contamination_score = self.detector(uav)

    def _update_reputations(self) -> None:
        for coalition in self.rl_coalitions:
            for uav in coalition.active_edge_uavs:
                uav.contamination_score = self.detector(uav)
                uav.retraining_energy = estimate_retraining_energy(
                    uav, coalition, self.rl_config.retraining_energy_scale
                )
                uav.residual_energy = max(uav.residual_energy - 1.0, 0.0)
                uav.update_reputation(self.rl_config.reputation_lr, self.rl_config.penalty_tuning)

    def _score_and_update_reputation(self, uav: RLEdgeUAV, coalition: RLCoalition) -> None:
        """Legacy helper — prefer _run_contamination_detection + _update_reputations."""
        uav.contamination_score = self.detector(uav)
        uav.retraining_energy = estimate_retraining_energy(uav, coalition, self.rl_config.retraining_energy_scale)
        uav.residual_energy = max(uav.residual_energy - 1.0, 0.0)
        uav.update_reputation(self.rl_config.reputation_lr, self.rl_config.penalty_tuning)

    def _select_governance_action(self, uav: RLEdgeUAV, rl_action: UAVAction) -> UAVAction:
        """
        Bootstrap policy from FLGuardian λ while PPO is learning (Sec. 4.2).

        High λ → quarantine/exclude; low λ → allow.  Ambiguous band defers to PPO.
        """
        lam = uav.contamination_score
        if lam >= 0.85 and uav.flag_count >= 1:
            return UAVAction.EXCLUDE
        if lam >= 0.55:
            return UAVAction.QUARANTINE
        if lam <= 0.25:
            return UAVAction.ALLOW
        return rl_action

    def _manage_states(self, round_idx: int) -> Dict[str, Dict[str, float]]:
        """
        PPO state decision per coalition (Algorithm 1, lines 10–24).
        Returns per-UAV metadata needed for reward computation.
        """
        meta: Dict[str, Dict[str, float]] = {}

        for coalition in self.rl_coalitions:
            acc_before = (
                coalition.evaluate_accuracy(self.validation_set, self.config)
                if self.validation_set is not None
                else 0.0
            )

            for uav in list(coalition.active_edge_uavs):
                obs = uav.observation()
                rl_action, log_prob, value = self.ppo.select_action(obs)
                action = self._select_governance_action(uav, rl_action)
                quarantine_assigned = 0

                fog = coalition.fog_uav
                if isinstance(fog, RLFogUAV):
                    if action == UAVAction.QUARANTINE:
                        quarantine_assigned = uav.assign_quarantine(self.rl_config.penalty_tuning)
                    elif action == UAVAction.EXCLUDE:
                        fog.apply_action(uav, action, self.rl_config.penalty_tuning)

                self.round_actions[uav.uav_id] = action

                acc_after = (
                    coalition.evaluate_accuracy(self.validation_set, self.config)
                    if self.validation_set is not None
                    else 0.0
                )
                delta_psi = acc_after - acc_before
                acc_before = acc_after

                reward = compute_reward(uav, delta_psi, quarantine_assigned, self.rl_config)
                meta[uav.uav_id] = {
                    "reward": reward,
                    "delta_psi": delta_psi,
                    "quarantine_assigned": float(quarantine_assigned),
                }

                self.ppo.store(Transition(
                    obs=obs,
                    action=action.value,
                    log_prob=log_prob,
                    value=value,
                    reward=reward,
                    done=(action == UAVAction.EXCLUDE),
                ))

        return meta

    def train_round(self, round_idx: int) -> Dict[str, float]:
        """One HFL round with PPO state management (Algorithm 1)."""
        self._process_quarantine_expiry()
        self.distribute_global_model()
        reference_weights = copy.deepcopy(self.global_model.state_dict())

        local_losses: Dict[str, float] = {}
        for uav in self.active_uavs:
            local_losses[uav.uav_id] = uav.train_local(self.config)
            uav.apply_poison(reference_weights)

        self._run_contamination_detection(reference_weights)
        self._update_reputations()

        self._manage_states(round_idx)

        coalition_weights: Dict[str, Dict[str, torch.Tensor]] = {}
        for coalition in self.rl_coalitions:
            if coalition.active_edge_uavs:
                coalition_weights[coalition.coalition_id] = coalition.aggregate_active_weights()
            elif coalition.edge_uavs:
                # Keep previous coalition weights if everyone is isolated.
                coalition_weights[coalition.coalition_id] = coalition.edge_uavs[0].state_dict()

        for coalition in self.rl_coalitions:
            if coalition.coalition_id in coalition_weights:
                coalition.distribute_weights(coalition_weights[coalition.coalition_id])

        if coalition_weights:
            total_samples = sum(
                c.active_samples for c in self.rl_coalitions if c.coalition_id in coalition_weights
            )
            global_weights: Dict[str, torch.Tensor] = {}
            for coalition in self.rl_coalitions:
                if coalition.coalition_id not in coalition_weights:
                    continue
                coef = coalition.active_samples / max(total_samples, 1)
                for key, tensor in coalition_weights[coalition.coalition_id].items():
                    global_weights[key] = global_weights.get(key, torch.zeros_like(tensor)) + coef * tensor
            if global_weights:
                self.global_model.load_state_dict(global_weights)

        self.distribute_global_model()
        ppo_stats = self.ppo.update() if self.ppo.buffer else {}
        local_losses["__ppo_policy_loss__"] = ppo_stats.get("policy_loss", 0.0)
        return local_losses

    def _process_quarantine_expiry(self) -> List[RLEdgeUAV]:
        """Algorithm 1, line 29: decrement T_j^Q and restore to U^A."""
        rejoined: List[RLEdgeUAV] = []
        for uav in self.quarantined_uavs:
            if uav.tick_quarantine():
                uav.load_state_dict(self.global_model.state_dict())
                rejoined.append(uav)
        return rejoined

    def _capture_round_snapshot(
        self,
        round_idx: int,
        global_loss: float,
        global_accuracy: float,
        ppo_policy_loss: float,
    ) -> RoundSnapshot:
        snapshots: List[UAVRoundSnapshot] = []
        for uav in self.rl_edge_uavs:
            action = self.round_actions.get(uav.uav_id, UAVAction.ALLOW)
            snapshots.append(
                UAVRoundSnapshot(
                    uav_id=uav.uav_id,
                    coalition_id=uav.coalition_id,
                    contamination_score=uav.contamination_score,
                    reputation=uav.reputation,
                    flag_count=uav.flag_count,
                    action=action.name,
                    participation=uav.participation.name,
                    quarantine_remaining=uav.quarantine_rounds_remaining,
                    is_malicious=uav.is_malicious,
                )
            )
        return RoundSnapshot(
            round_idx=round_idx,
            global_accuracy=global_accuracy,
            global_loss=global_loss,
            active_uavs=len(self.active_uavs),
            quarantined_uavs=len(self.quarantined_uavs),
            excluded_uavs=len(self.excluded_uavs),
            uav_snapshots=snapshots,
            ppo_policy_loss=ppo_policy_loss,
        )

    def run(self, test_dataset: Dataset, validation_dataset: Optional[Dataset] = None) -> List[Dict[str, float]]:
        self.validation_set = validation_dataset or test_dataset
        self.round_history.clear()
        self.round_actions.clear()

        for uav in self.rl_edge_uavs:
            uav.reputation = self.rl_config.initial_reputation
            uav.residual_energy = self.rl_config.initial_energy
            uav.model_contribution = self.rl_config.default_contribution
            uav.participation = ParticipationState.ACTIVE
            uav.flag_count = 0
            uav.quarantine_rounds_remaining = 0

        history: List[Dict[str, float]] = []
        for round_idx in range(1, self.config.num_rounds + 1):
            self.round_actions.clear()
            losses = self.train_round(round_idx)
            loss, acc = self._evaluate_global(test_dataset)
            ppo_loss = losses.get("__ppo_policy_loss__", 0.0)
            snapshot = self._capture_round_snapshot(round_idx, loss, acc, ppo_loss)
            self.round_history.append(snapshot)
            history.append({
                "round": round_idx,
                "global_loss": loss,
                "global_accuracy": acc,
                "active_uavs": len(self.active_uavs),
                "quarantined_uavs": len(self.quarantined_uavs),
                "excluded_uavs": len(self.excluded_uavs),
                "avg_local_loss": sum(
                    v for k, v in losses.items() if not k.startswith("__")
                ) / max(len(self.active_uavs), 1),
                "ppo_policy_loss": ppo_loss,
            })
            self._print_round_summary(snapshot)
        return history

    def _print_round_summary(self, snapshot: RoundSnapshot) -> None:
        print(
            f"Round {snapshot.round_idx}/{self.config.num_rounds} | "
            f"acc={snapshot.global_accuracy:.4f} loss={snapshot.global_loss:.4f} | "
            f"active={snapshot.active_uavs} "
            f"quarantined={snapshot.quarantined_uavs} excluded={snapshot.excluded_uavs}"
        )
        for uav in snapshot.uav_snapshots:
            if uav.participation != "ACTIVE" or uav.contamination_score > 0.3 or uav.is_malicious:
                print(
                    f"  {uav.uav_id} [{uav.coalition_id}] "
                    f"λ={uav.contamination_score:.3f} ρ={uav.reputation:.3f} "
                    f"q={uav.flag_count} → {uav.action} ({uav.participation})"
                    + (f" TQ={uav.quarantine_remaining}" if uav.quarantine_remaining else "")
                    + (" [MALICIOUS]" if uav.is_malicious else "")
                )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_hfl_rl_system(
    coalition_specs: Sequence[Tuple[str, Sequence[str]]],
    config: Optional[HFLConfig] = None,
    rl_config: Optional[RLConfig] = None,
    contamination_detector: Optional[ContaminationDetector] = None,
    malicious_uavs: Optional[Sequence[str]] = None,
    poison_scale: float = 50.0,
) -> HFLRLStation:
    """Build an HFL system with RL-enabled edge/coalition/fog UAVs."""
    cfg = config or HFLConfig()
    rl_cfg = rl_config or RLConfig()
    malicious = set(malicious_uavs or [])
    all_edge_ids = [eid for _, members in coalition_specs for eid in members]
    train_set, _ = load_fashion_mnist(cfg.data_dir)
    shards = partition_dataset(train_set, len(all_edge_ids))
    shard_map = dict(zip(all_edge_ids, shards))

    station = HFLRLStation(cfg, rl_cfg, contamination_detector)
    for coalition_id, edge_ids in coalition_specs:
        edge_uavs = [
            RLEdgeUAV(
                uav_id=eid,
                coalition_id=coalition_id,
                dataset=shard_map[eid],
                sensors=[f"{eid}_cam"],
                reputation=rl_cfg.initial_reputation,
                residual_energy=rl_cfg.initial_energy,
                model_contribution=rl_cfg.default_contribution,
                is_malicious=eid in malicious,
                poison_scale=poison_scale,
            )
            for eid in edge_ids
        ]
        coalition = RLCoalition(coalition_id=coalition_id, edge_uavs=edge_uavs)
        fog = RLFogUAV(fog_id=f"fog_{coalition_id}", coalition=coalition)
        station.register_fog_uav(fog)

    return station
