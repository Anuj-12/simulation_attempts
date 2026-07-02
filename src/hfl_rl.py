"""
PPO-based UAV state management for ReCon HFL (Sec. 4.2, Algorithm 1).

Builds on hfl_base.py and implements:
  - Observation o_j = (lambda_j, q_j, rho_j)           (Eq. 20)
  - Action space {Allow, Quarantine, Exclude}        (Eq. 23), a_j ~ pi_theta
  - Reputation update                                 (Eq. 7)
  - Quarantine duration T_j^Q                         (Eq. 28)
  - Reward R_j = rho_j + Delta psi_j + Lambda(a)      (Eq. 26-27)
  - PPO clipped objective + value + entropy losses    (Eq. 21, 29-33)

Contamination detection phi(g_j) is injectable; a zero-score stub is used
by default until an external detector is provided.

Checkpoint urgency + rollback recovery (Sec. 4.3, Algorithm 2) is NOT
implemented here; it is exclusive to hfl_recovery.HFLRecoveryStation, which
extends this station. Per main.py's mode semantics, plain "rl" mode is PPO
state management only.
"""

from __future__ import annotations

import copy
import math
import random
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
    """Hyper-parameters for reputation and PPO (ReCon Sec. 4.1-4.2)."""

    initial_reputation: float = 0.5          # rho_0
    reputation_lr: float = 0.1               # eta
    penalty_tuning: float = 1.0              # tau
    initial_energy: float = 1000.0           # E_j^init (Eq. energy_res)
    default_contribution: float = 1.0        # phi_j fallback before first GTG-Shapley pass
    checkpoint_threshold: float = 5.0        # kappa_th (Eq. checkpoint_trigger)

    # DVFS computation-energy model (Eq. energy_comp / energy_prop / energy_agg)
    flops_per_sample: float = 1e6            # alpha: FLOPs required per data sample
    flops_per_cycle: float = 1e9             # B_j: FLOPs computed per CPU cycle
    zeta: float = 1e-27                      # zeta_j: effective switched capacitance
    cpu_frequency: float = 1e9               # f_j: CPU clock frequency (Hz)
    prop_c1: float = 0.5                     # c1: parasitic-power constant (Eq. energy_prop)
    prop_c2: float = 100.0                   # c2: induced-power constant (Eq. energy_prop)
    velocity: float = 10.0                   # v: constant UAV flight velocity (m/s)

    # GTG-Shapley contribution estimate (Eq. contri / contri_psi)
    shapley_permutations: int = 1            # number of MC permutation samples per round

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
    reputation_at_t: float = 0.5             # rho_j^(t), cached before Eq. 7 update, used in reward Eq. 26
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
        """Eq. 7 (energy terms normalized to keep rho in a stable range)."""
        self.reputation_at_t = self.reputation  # rho_j^(t), used by reward Eq. 26
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
        self.quarantine_rounds_remaining = max(duration, 0)
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
# PPO agent (Eq. 21, 24-25, 29-33)
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

    def act(
        self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[int, float, float, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        if action_mask is not None:
            # Masked actions get -inf logits so Categorical never samples them,
            # and log_prob/entropy stay consistent with the restricted support.
            logits = logits.masked_fill(~action_mask, float("-inf"))
        dist = Categorical(logits=logits)
        action = dist.sample()
        return (
            int(action.item()),
            float(dist.log_prob(action).item()),
            float(value.item()),
            dist.entropy(),
            dist.probs.detach(),
        )


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
        # Training-stability guard (not part of the paper's formulation): the
        # policy network starts randomly initialized, so its action distribution
        # is close to uniform until it has received at least one gradient step.
        # Since Exclude is an absorbing action (excluded UAVs never rejoin),
        # letting an untrained policy pick it risks permanently and arbitrarily
        # removing benign UAVs before the reward signal has taught the policy
        # anything. has_updated tracks whether update() has run at least once;
        # callers can use it to mask Exclude out of the action space until then.
        self.has_updated: bool = False

    def select_action(
        self, obs: torch.Tensor, allow_exclude: bool = True
    ) -> Tuple[UAVAction, float, float, List[float]]:
        obs = obs.to(self.device)
        action_mask = None
        if not allow_exclude:
            action_mask = torch.tensor(
                [a != UAVAction.EXCLUDE.value for a in range(len(UAVAction))],
                dtype=torch.bool,
                device=self.device,
            )
        action_idx, log_prob, value, _, probs = self.network.act(obs, action_mask=action_mask)
        return UAVAction(action_idx), log_prob, value, probs.tolist()

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
        self.has_updated = True
        return clip_stats


# ---------------------------------------------------------------------------
# Reward and energy helpers
# ---------------------------------------------------------------------------


def _print_state_transitions(
    rows: List[Tuple[int, str, str, str, str, Optional[int], float]]
) -> None:
    """Print one line per UAV decision this round, showing the participation
    state transition actually caused by the PPO action (not just the raw
    action/observation metrics)."""
    for round_idx, uav_id, prev_state, new_state, action_name, quarantine_duration, reward in rows:
        transition = f"{prev_state} -> {new_state}" if prev_state != new_state else f"{prev_state} (unchanged)"
        extra = f" T_Q={quarantine_duration}" if quarantine_duration else ""
        print(
            f"Round {round_idx:>3} | {uav_id:<8} | {transition:<28} | "
            f"action={action_name:<10}{extra} | reward={reward:8.3f}"
        )


def compute_computation_energy(num_samples: int, cfg: "RLConfig") -> float:
    """E_j^comp = (G_j / B_j) * zeta_j * f_j^2,  G_j = alpha * n_j  (Eq. energy_comp).

    Sensor datasets partition u_j's data, so sum_h |D_h^j| = n_j.
    """
    g_j = cfg.flops_per_sample * num_samples
    return (g_j / cfg.flops_per_cycle) * cfg.zeta * (cfg.cpu_frequency ** 2)


def compute_propulsion_energy(num_samples: int, cfg: "RLConfig") -> float:
    """E_j^prop = (c1 v^3 + c2/v) * T_j^comp,  T_j^comp = G_j / (B_j f_j)  (Eq. energy_prop)."""
    g_j = cfg.flops_per_sample * num_samples
    t_comp = g_j / (cfg.flops_per_cycle * cfg.cpu_frequency)
    return (cfg.prop_c1 * cfg.velocity ** 3 + cfg.prop_c2 / cfg.velocity) * t_comp


def compute_residual_energy(num_samples: int, cfg: "RLConfig") -> float:
    """E_j^res = E_j^init - E_j^comp - E_j^prop  (Eq. energy_res)."""
    e_comp = compute_computation_energy(num_samples, cfg)
    e_prop = compute_propulsion_energy(num_samples, cfg)
    return cfg.initial_energy - e_comp - e_prop


def compute_aggregation_energy(coalition_size: int, total_params: int, cfg: "RLConfig") -> float:
    """E_j^agg = (G_k^agg / B_k) * zeta_k * f_k^2,  G_k^agg = Omega * (2|c_k| - 1)  (Eq. energy_agg)."""
    g_agg = total_params * max(2 * coalition_size - 1, 1)
    return (g_agg / cfg.flops_per_cycle) * cfg.zeta * (cfg.cpu_frequency ** 2)


def compute_retraining_energy(uav: RLEdgeUAV, coalition: "RLCoalition", cfg: "RLConfig") -> float:
    """E_j^ret = sum_{u_h in c_k \\ {u_j}} E_h^comp + E_j^agg  (Eq. energy_ret)."""
    peers = [u for u in coalition.edge_uavs if u.uav_id != uav.uav_id]
    comp_sum = sum(compute_computation_energy(u.num_samples, cfg) for u in peers)
    total_params = sum(p.numel() for p in uav.model.parameters())
    agg = compute_aggregation_energy(len(coalition.edge_uavs), total_params, cfg)
    return comp_sum + agg


def _aggregate_subset_weights(
    base_weights: Dict[str, torch.Tensor], subset: Sequence[RLEdgeUAV]
) -> Dict[str, torch.Tensor]:
    """M_S = M^(t) + sum_{u in S} (n_u / N_S) * Delta_u^(t+1)  (Eq. contri_psi)."""
    if not subset:
        return {k: v.clone() for k, v in base_weights.items()}
    total = sum(u.num_samples for u in subset)
    result = {k: v.clone() for k, v in base_weights.items()}
    for u in subset:
        weight = u.num_samples / total
        u_state = u.state_dict()
        for key in result:
            result[key] = result[key] + weight * (u_state[key] - base_weights[key])
    return result


def _evaluate_weights_accuracy(
    weights: Dict[str, torch.Tensor],
    probe: nn.Module,
    dataset: Dataset,
    config: HFLConfig,
) -> float:
    """psi(S) = Acc(M_S, D^val)  (Eq. contri_psi)."""
    probe.load_state_dict(weights)
    probe.eval()
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(config.device)
            labels = labels.to(config.device)
            correct += (probe(images).argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def compute_gtg_shapley_contribution(
    uav: RLEdgeUAV,
    coalition: "RLCoalition",
    base_weights: Dict[str, torch.Tensor],
    validation_dataset: Dataset,
    config: HFLConfig,
    num_permutations: int = 1,
) -> float:
    """phi_j = E_{pi ~ Pi_k}[ psi(S_uj^pi U {u_j}) - psi(S_uj^pi) ]  (Eq. contri).

    GTG-Shapley approximation: reuses each peer's already-trained local model
    update (Delta_u^(t+1)) instead of retraining from scratch, and estimates
    the expectation over permutations with `num_permutations` random draws.
    """
    peers = [u for u in coalition.active_edge_uavs if u.uav_id != uav.uav_id]
    if num_permutations <= 0:
        return 0.0
    probe = copy.deepcopy(uav.model)
    samples: List[float] = []
    for _ in range(max(num_permutations, 1)):
        perm = peers[:]
        random.shuffle(perm)
        prefix_len = random.randint(0, len(perm))
        prefix = perm[:prefix_len]
        s_weights = _aggregate_subset_weights(base_weights, prefix)
        s_with_uj_weights = _aggregate_subset_weights(base_weights, prefix + [uav])
        acc_before = _evaluate_weights_accuracy(s_weights, probe, validation_dataset, config)
        acc_after = _evaluate_weights_accuracy(s_with_uj_weights, probe, validation_dataset, config)
        samples.append(acc_after - acc_before)
    return sum(samples) / len(samples)


def compute_reward(
    uav: RLEdgeUAV,
    delta_psi: float,
    action: UAVAction,
    quarantine_duration: int,
) -> float:
    """R_j^(t+1) = rho_j^(t) + Delta psi_j^(t) + Lambda(a)  (Eq. 26-27).

    Lambda(a) is piecewise in the action actually taken:
      Allow      -> 0
      Quarantine -> -T_j^Q(t)
      Exclude    -> -(1 - lambda_j^(t)) * rho_j^(t)

    Unlike a flat penalty, this term must apply *only* when the UAV is
    excluded, and must scale with (1 - lambda_j) and rho_j^(t), not
    lambda_j, per Eq. 27.
    """
    if action == UAVAction.ALLOW:
        penalty = 0.0
    elif action == UAVAction.QUARANTINE:
        penalty = -float(quarantine_duration)
    elif action == UAVAction.EXCLUDE:
        lam = uav.contamination_score
        rho = uav.reputation_at_t
        penalty = -(1.0 - lam) * rho
    else:
        raise ValueError(f"Unrecognized action {action!r}")

    # Eq. 26 uses rho_j^(t) explicitly (not the rho_j^(t+1) already computed
    # this round by _update_reputations via Eq. 7), so the pre-update value
    # cached in reputation_at_t must be used here rather than uav.reputation.
    return uav.reputation_at_t + delta_psi + penalty


# ---------------------------------------------------------------------------
# HFL + PPO base station (Algorithm 1)
# ---------------------------------------------------------------------------


class HFLRLStation(BaseStation):
    """
    Extends BaseStation with PPO-based UAV state management (Algorithm 1)
    and checkpoint-based contamination recovery (Sec. 4.3, Algorithm 2).
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
        # NOTE: checkpoint/rollback (Sec. 4.3, Algorithm 2) is intentionally NOT
        # implemented here. Per main.py's mode semantics, "rl" mode is PPO state
        # management only; checkpoint-based recovery is exclusive to "recovery"
        # mode, implemented in hfl_recovery.HFLRecoveryStation. Duplicating it
        # here previously caused self.checkpoint_store's type to be clobbered
        # (CheckpointStore -> plain dict) and used the wrong threshold
        # (rl_config instead of recovery_config), corrupting HFLRecoveryStation.

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
        Run phi on each coalition via fog UAV (Sec. 3, Eq. lamda_def).

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
                scores = self.detector.score_coalition(active)
                for uav in active:
                    uav.contamination_score = float(scores.get(uav.uav_id, 0.0))
            else:
                for uav in active:
                    uav.contamination_score = self.detector(uav)

    def _update_reputations(self, reference_weights: Dict[str, torch.Tensor]) -> None:
        for coalition in self.rl_coalitions:
            for uav in coalition.active_edge_uavs:
                # lambda_j^(t) already set by _run_contamination_detection (Eq. 6);
                # do not resample it here.
                uav.retraining_energy = compute_retraining_energy(uav, coalition, self.rl_config)
                uav.residual_energy = compute_residual_energy(uav.num_samples, self.rl_config)
                if self.validation_set is not None:
                    uav.model_contribution = compute_gtg_shapley_contribution(
                        uav,
                        coalition,
                        reference_weights,
                        self.validation_set,
                        self.config,
                        self.rl_config.shapley_permutations,
                    )
                uav.update_reputation(self.rl_config.reputation_lr, self.rl_config.penalty_tuning)

    def _score_and_update_reputation(self, uav: RLEdgeUAV, coalition: RLCoalition) -> None:
        """Legacy helper - prefer _run_contamination_detection + _update_reputations."""
        uav.contamination_score = self.detector(uav)
        uav.retraining_energy = compute_retraining_energy(uav, coalition, self.rl_config)
        uav.residual_energy = compute_residual_energy(uav.num_samples, self.rl_config)
        uav.update_reputation(self.rl_config.reputation_lr, self.rl_config.penalty_tuning)

    def _manage_states(self, round_idx: int) -> Dict[str, Dict[str, float]]:
        """
        PPO state decision per coalition (Algorithm 1, lines 10-24).
        Returns per-UAV metadata needed for reward computation.

        Only UAVs currently in ParticipationState.ACTIVE are ever fed through
        the PPO agent. EXCLUDED UAVs are permanently removed from Algorithm 1's
        decision loop (exclude() is a terminal state - nothing ever resets a
        UAV out of EXCLUDED), so they must never reappear here. QUARANTINED
        UAVs are likewise skipped, but unlike EXCLUDED they are eligible to
        re-enter this loop once tick_quarantine() restores them to ACTIVE
        (Algorithm 1, line 29 / _process_quarantine_expiry).
        """
        meta: Dict[str, Dict[str, float]] = {}
        transition_rows: List[Tuple[int, str, str, str, str, Optional[int], float]] = []

        for coalition in self.rl_coalitions:
            acc_before = (
                coalition.evaluate_accuracy(self.validation_set, self.config)
                if self.validation_set is not None
                else 0.0
            )

            for uav in list(coalition.active_edge_uavs):
                # Defensive guard: active_edge_uavs already filters to
                # ParticipationState.ACTIVE, but we assert it explicitly here
                # so that EXCLUDED (terminal) and QUARANTINED (temporarily
                # ineligible) UAVs can never be handed a PPO decision, even if
                # this method is refactored to iterate a different source list
                # in the future.
                if uav.participation != ParticipationState.ACTIVE:
                    continue

                prev_participation = uav.participation
                obs = uav.observation()
                action, log_prob, value, probs = self.ppo.select_action(
                    obs, allow_exclude=self.ppo.has_updated
                )
                quarantine_assigned = 0

                # The action PPO selects must always be the action that is
                # implemented on the UAV's participation state - previously
                # this was gated behind `isinstance(fog, RLFogUAV)`, so a
                # coalition with a plain FogUAV would silently record a
                # QUARANTINE/EXCLUDE action (and reward it accordingly) while
                # never actually changing uav.participation. Applying the
                # action directly to the UAV removes that dependency on the
                # fog's type and guarantees decision == effect.
                if action == UAVAction.QUARANTINE:
                    quarantine_assigned = uav.assign_quarantine(self.rl_config.penalty_tuning)
                elif action == UAVAction.EXCLUDE:
                    uav.exclude()

                self.round_actions[uav.uav_id] = action

                acc_after = (
                    coalition.evaluate_accuracy(self.validation_set, self.config)
                    if self.validation_set is not None
                    else 0.0
                )
                delta_psi = acc_after - acc_before
                acc_before = acc_after

                reward = compute_reward(uav, delta_psi, action, quarantine_assigned)
                meta[uav.uav_id] = {
                    "reward": reward,
                    "delta_psi": delta_psi,
                    "quarantine_assigned": float(quarantine_assigned),
                }

                transition_rows.append((
                    round_idx,
                    uav.uav_id,
                    prev_participation.name,
                    uav.participation.name,
                    action.name,
                    quarantine_assigned or None,
                    reward,
                ))

                self.ppo.store(Transition(
                    obs=obs,
                    action=action.value,
                    log_prob=log_prob,
                    value=value,
                    reward=reward,
                    done=(action == UAVAction.EXCLUDE),
                ))

        if transition_rows:
            _print_state_transitions(transition_rows)

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
        self._update_reputations(reference_weights)

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
                    f"lambda={uav.contamination_score:.3f} rho={uav.reputation:.3f} "
                    f"q={uav.flag_count} -> {uav.action} ({uav.participation})"
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
