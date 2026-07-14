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
    # eta: raised from 0.1 to 0.35. At 0.1, ~50 rounds in, ρ for malicious
    # (λ~1) and benign (λ~0) UAVs is still nearly indistinguishable (~0.50 for
    # everyone), since the reward/penalty terms in Eq. 7 only move ρ a small
    # amount per round. A faster eta separates the two populations sooner,
    # which matters twice over: (1) it makes the Exclude penalty
    # -(1-λ)*ρ actually discriminate earlier instead of being ~-0.5 for
    # everyone regardless of contamination, and (2) it feeds back into
    # T_j^Q = e^(τq)/(1+ρ) (Eq. 28) - a benign UAV whose ρ climbs faster gets
    # shorter quarantines sooner even if flagged again, partially
    # self-correcting the q-accumulation-during-warmup risk (q never decays,
    # so early false-positive quarantines compound; faster ρ growth is a
    # partial mitigation, not a fix for q itself never decreasing).
    reputation_lr: float = 0.35              # eta
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
    # l2: raised from 0.01 to 0.08. At 0.01 the entropy bonus barely
    # counteracts the policy settling into a locally-safe pattern (e.g.
    # quarantining broadly) before the reward signal has had enough rounds to
    # differentiate contamination levels - especially risky during the
    # exclude_warmup_fraction window, where Exclude is unavailable and any
    # early convergence toward over-quarantining has more rounds to
    # compound q (which never decays) before Exclude even becomes an option.
    # A higher coefficient keeps action selection noisier for longer,
    # matching the paper's own stated intent for this term (line 617:
    # "promote exploration... discouraging premature convergence") - see
    # the entropy-sign note elsewhere in this file/the changes doc for the
    # separate issue of the paper's literal equation contradicting that
    # intent. Kept well under ~0.2, past which the entropy term risks
    # dominating the loss and preventing convergence entirely.
    entropy_coef: float = 0.08               # l2
    ppo_epochs: int = 4
    hidden_dim: int = 64

    # Delta_psi scale (Eq. 26: R = rho_t + Delta_psi + Lambda(a) - a straight,
    # unweighted sum in the paper, no coefficient specified for any term).
    # Confirmed deviation from the paper's literal equation, added after
    # observing that Delta_psi's natural magnitude (mean ~0.05, max ~0.26 in
    # practice - a leave-one-out accuracy shift is just inherently small) is
    # ~10x smaller than rho_t's typical magnitude (~0.51) and ~170x smaller
    # than a typical Quarantine/Exclude penalty (~8.25) - meaning Delta_psi
    # essentially never has enough weight to influence which action wins,
    # even when a removal genuinely hurt accuracy. Default 10.0 targets
    # rho_t's scale specifically, not Lambda(a)'s: rho_t and Delta_psi are
    # both intrinsic "was this a good outcome" signals (reputation
    # trajectory and accuracy consequence), whereas Lambda(a) is a
    # deliberately dominant administrative cost meant to override both when
    # an action is expensive - Delta_psi doesn't need to compete with a
    # large Lambda(a), it needs to matter when Lambda(a) is already small
    # (a confidently-high-lambda UAV, where quarantine/exclude is cheap),
    # which is exactly where "but did this actually help?" should still
    # get a vote. This value was computed from one run's observed
    # magnitudes, not derived from first principles - worth re-checking
    # against future runs, not a settled constant.
    delta_psi_scale: float = 10.0

    # Exclusion-unlock warm-up (implementation-level training-stability
    # guard; NOT part of the paper's Algorithm 1 / Eq. 23 specification).
    # EXCLUDE stays masked out of the PPO action space until at least this
    # fraction of the run's total rounds have elapsed (see PPOAgent
    # `exclude_unlocked`). Expressing it as a fraction rather than a fixed
    # update count means the warm-up scales automatically with --rounds:
    # "protect the first 10% of training" holds whether the run is 20
    # rounds or 1000, whereas a fixed count of updates would be a 50%
    # warm-up in a short run and a 1% warm-up in a long one.
    #
    # Default = 0.10. Each PPOAgent.update() call runs ppo_epochs (4)
    # full-batch gradient steps over that round's collected transitions, so
    # a 10% warm-up on the default 100-round horizon gives ~40 gradient
    # steps and exposure to every active UAV's (lambda, q, rho) observation
    # across 10 full rounds. Under the Eq. 28 quarantine-duration formula,
    # T_Q is short for a first flag (q=1) with these defaults (~1-2 rounds),
    # so that warm-up comfortably allows several QUARANTINE -> rejoin
    # cycles to be observed and rewarded before the irreversible EXCLUDE
    # action becomes available, while still leaving 90% of training with
    # the full 3-action space.
    exclude_warmup_fraction: float = 0.10

    # Fallback used only if total_rounds is unavailable to PPOAgent at
    # construction time (e.g. it's built standalone, without an HFLConfig).
    # Not used when exclude_warmup_fraction can be resolved against an
    # actual round count.
    min_ppo_updates_before_exclude: int = 10

    # One-time flag_count (q) reset after the "primer" period ends -
    # implementation-level mitigation for a real risk, NOT in the paper.
    # q only ever increases (assign_quarantine's self.flag_count += 1 is the
    # only place that touches it anywhere in this file); Eq. 7/28 specify no
    # decay or reset for q_j at all. That means a UAV quarantined during
    # early, near-random policy exploration (before the reward signal has
    # had time to differentiate contamination levels) accumulates a
    # permanently higher q from what may have been a false positive - and
    # since T_j^Q = e^(tau*q)/(1+rho) (Eq. 28) is exponential in q, that
    # early flag means an exponentially longer quarantine on any FUTURE
    # trigger, for the rest of the run, long after the false positive is
    # forgotten. Enabling this wipes every UAV's q back to 0 exactly once,
    # at the round the primer period ends, so early-exploration flags don't
    # keep compounding through the rest of training.
    reset_flags_after_primer: bool = True

    # Fraction of total_rounds treated as the "primer" period for this
    # reset. Deliberately a SEPARATE knob from exclude_warmup_fraction
    # (even though both default to 0.10) rather than reusing it - you may
    # want q to reset at a different point than when EXCLUDE unlocks, e.g.
    # a longer/shorter primer for this specific mitigation.
    flag_reset_fraction: float = 0.10

    # At the same primer-end event as the q reset above, also force every
    # currently QUARANTINED UAV back to ACTIVE (participation reset,
    # quarantine_rounds_remaining zeroed, given the current global model
    # weights - same as a normal tick_quarantine()-triggered rejoin).
    # Also NOT in the paper. Rationale: resetting q alone still leaves any
    # UAV mid-quarantine serving out a (possibly inflated, pre-reset) T_j^Q
    # sentence computed under the near-random primer-period policy; this
    # clears that sentence too, so nothing from the primer period's
    # exploration noise carries into the rest of the run. Does NOT affect
    # EXCLUDED UAVs - exclude remains a terminal state, per Algorithm 1.
    release_quarantine_after_primer: bool = True

    # Shadow-exclude during the primer period (also NOT in the paper).
    # When exclude_unlocked is False, EXCLUDE was previously masked out of
    # the action space entirely (see ActorCritic.act's action_mask), so the
    # policy could never sample it and could only ever receive negative
    # reinforcement on its logit (softmax gradients touch every output, and
    # a never-chosen action can only ever be pushed down, never up) - by the
    # time EXCLUDE unlocked, its head had been passively suppressed for the
    # whole primer period with zero chance to learn when excluding would
    # actually have paid off. When this is True, EXCLUDE can be sampled
    # during the primer period like any other action, and its reward is
    # computed exactly as if it had been applied (delta_psi's counterfactual
    # already works this way regardless of whether the removal is real) -
    # but the UAV's real participation state is NOT changed; it stays
    # ACTIVE. Only once exclude_unlocked is True does a sampled EXCLUDE
    # become a real, applied exclusion. Set False to fall back to the old
    # masking behavior (EXCLUDE unsampleable, no shadow reward, during the
    # primer period).
    shadow_exclude_during_primer: bool = True


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
        """o_j = (lambda_j, q_j, rho_j)  (Eq. 20).

        Defensive guard (not part of the paper): sanitizes non-finite
        (NaN/+-Inf) values before they can reach the PPO network. This can
        happen if the simulated system diverges numerically upstream (e.g.
        unbounded global-model weight growth feeding into contamination
        scoring or Eq. 7's arithmetic) - Python's built-in min/max do NOT
        reliably clamp NaN regardless of argument order (e.g.
        max(min(nan, 10.0), 0.0) can still evaluate to nan), so
        update_reputation's own clamp is not a guaranteed backstop. A NaN
        reaching Categorical(logits=...) crashes the run; worse, if it
        reaches PPOAgent.update() it can permanently corrupt the network's
        weights (see that method's own guard). contamination_score defaults
        to 1.0 (maximally suspicious), not 0.0, on a non-finite read:
        failing open toward "assume innocent" is exactly the failure mode
        already diagnosed elsewhere (a coalition collapsed to a single
        active UAV silently reads as fully clean) - failing closed toward
        "assume contaminated" is the safer default here. reputation defaults
        to initial_reputation's value (0.5) as a neutral fallback.
        """
        lam = self.contamination_score
        if not math.isfinite(lam):
            lam = 1.0
        rho = self.reputation
        if not math.isfinite(rho):
            rho = 0.5
        return torch.tensor(
            [
                lam,
                float(self.flag_count) / 5.0,
                rho / 10.0,
            ],
            dtype=torch.float32,
        )

    def update_reputation(self, eta: float, tau: float) -> float:
        """Eq. 7 (energy terms normalized to keep rho in a stable range).

        Defensive guards (not part of the paper): both phi and the final
        clamp on self.reputation previously used Python's built-in max/min,
        which do NOT reliably clamp NaN (e.g. max(nan, 1e-8) checks
        `1e-8 < nan`, which is False, so it returns the unclamped nan) -
        confirmed as a real, not just theoretical, problem by an actual run:
        once the global model overflowed to NaN/Inf from compounding,
        unclipped poisoned aggregation, model_contribution (computed from
        GTG-Shapley accuracy evaluations on that corrupted model) went NaN,
        passed through the old `max(self.model_contribution, 1e-8)`
        unclamped, propagated through reward_term/penalty_term into
        self.reputation, and then passed through the old
        `max(min(self.reputation, 10.0), 0.0)` unclamped too - eventually
        crashing assign_quarantine's int() conversion several calls later.
        Fixing this at the source (every place self.reputation gets WRITTEN)
        is more robust than guarding every downstream read individually.
        """
        self.reputation_at_t = self.reputation  # rho_j^(t), used by reward Eq. 26
        lam = self.contamination_score
        q = self.flag_count
        e_res = self.residual_energy / 1000.0
        e_ret = max(self.retraining_energy, 0.0) / 100.0
        phi = self.model_contribution if math.isfinite(self.model_contribution) else 1e-8
        phi = max(phi, 1e-8)

        reward_term = (1.0 - lam) * (e_res / (1.0 + q)) * phi
        penalty_term = lam * (e_ret / phi) * math.exp(tau * q)
        new_reputation = self.reputation + eta * (reward_term - penalty_term)
        if not math.isfinite(new_reputation):
            new_reputation = self.reputation_at_t  # fall back to last known-good value
        self.reputation = max(min(new_reputation, 10.0), 0.0)
        return self.reputation

    def assign_quarantine(self, tau: float) -> int:
        """T_j^Q = exp(tau * q_j) / (1 + rho_j)  (Eq. 28).

        Defensive guard (not part of the paper): sanitizes a non-finite
        self.reputation before it reaches this formula. Confirmed as a real
        crash site by an actual run: max(self.reputation, 0.0) does NOT
        reliably clamp NaN (max(nan, 0.0) checks `0.0 < nan`, which is
        False, so it returns the unclamped nan), so a NaN reputation
        (propagated here from Eq. 7 once the global model itself overflows
        to NaN/Inf from compounding, unclipped poisoned aggregation - see
        RLConfig/changes-doc notes on poison_scale) reached
        int(math.exp(...) / (1.0 + nan)) and crashed with "cannot convert
        float NaN to integer". Falls back to initial_reputation's default
        (0.5) on a non-finite read, same neutral fallback used in
        observation().
        """
        self.flag_count += 1
        rho = self.reputation if math.isfinite(self.reputation) else 0.5
        duration = int(math.exp(tau * self.flag_count) / (1.0 + max(rho, 0.0)))
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
class RewardBreakdown:
    """Diagnostic decomposition of compute_reward's output, so a decision's
    reward can be inspected term-by-term (rho_j^(t), Delta_psi, Lambda(a))
    instead of only the summed total - added after two rounds of guessing
    which term was driving over-exclusion turned out to need actual
    component-level numbers rather than another blind reward-formula patch.
    """

    total: float
    rho_t: float
    delta_psi: float
    penalty: float  # Lambda(a)


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

    def __init__(
        self,
        rl_config: RLConfig,
        device: str = "cpu",
        total_rounds: Optional[int] = None,
    ) -> None:
        self.config = rl_config
        self.device = device
        self.network = ActorCritic(hidden_dim=rl_config.hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=rl_config.ppo_lr)
        self.buffer: List[Transition] = []
        # Training-stability guard (not part of the paper's formulation): the
        # policy network starts randomly initialized, so its action distribution
        # is close to uniform until it has received several gradient steps.
        # Since Exclude is an absorbing action (excluded UAVs never rejoin),
        # letting an undertrained policy pick it risks permanently and
        # arbitrarily removing benign UAVs before the reward signal has taught
        # the policy anything. update_count tracks how many update() calls
        # have completed; exclude_unlocked (below) gates Exclude out of the
        # action space until enough of them have run, rather than unlocking
        # it after a single update as before.
        self.update_count: int = 0

        # Resolve the warm-up as an absolute update count once, at
        # construction time. When total_rounds is known (the normal case -
        # HFLRLStation passes its HFLConfig.num_rounds), this is
        # round(exclude_warmup_fraction * total_rounds), so the warm-up is a
        # fixed *fraction* of the run regardless of --rounds. If total_rounds
        # isn't available (e.g. PPOAgent constructed standalone), it falls
        # back to the fixed rl_config.min_ppo_updates_before_exclude count.
        if total_rounds is not None:
            self._min_updates_before_exclude = max(
                1, round(rl_config.exclude_warmup_fraction * total_rounds)
            )
        else:
            self._min_updates_before_exclude = rl_config.min_ppo_updates_before_exclude

    @property
    def exclude_unlocked(self) -> bool:
        """True once enough PPOAgent.update() calls have run to allow EXCLUDE.

        Threshold is exclude_warmup_fraction * total_rounds (resolved at
        construction time), or min_ppo_updates_before_exclude as a fallback.
        """
        return self.update_count >= self._min_updates_before_exclude

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
        """Optimize L_PPO = L_CLIP - l1 * L_VF - l2 * entropy  (Eq. 29).

        Defensive guard (not part of the paper): if the observations, returns,
        or advantages collected this round contain any non-finite value (NaN
        or +/-Inf) - which can happen upstream if the simulated system itself
        diverges numerically (e.g. unbounded global-model weight growth from
        repeated poisoned aggregation) - skip this update entirely rather
        than let torch.optim apply a NaN/Inf gradient. Once a network
        parameter becomes NaN, every subsequent forward pass through it is
        NaN forever (NaN self-propagates through all arithmetic) - there is
        no recovering from it later, so the only safe response is to never
        let a bad update reach optimizer.step() in the first place. This
        does not fix whatever caused the non-finite values upstream; it only
        prevents that upstream problem from also permanently destroying the
        policy network.
        """
        if not self.buffer:
            return {}

        obs = torch.stack([t.obs for t in self.buffer]).to(self.device)
        actions = torch.tensor([t.action for t in self.buffer], device=self.device)
        old_log_probs = torch.tensor([t.log_prob for t in self.buffer], device=self.device)
        returns, advantages = self._compute_returns()

        if not (
            torch.isfinite(obs).all()
            and torch.isfinite(old_log_probs).all()
            and torch.isfinite(returns).all()
            and torch.isfinite(advantages).all()
        ):
            print(
                "[PPO WARNING] Non-finite value in this round's transitions "
                "(obs/log_prob/return/advantage) - skipping this PPOAgent.update() "
                "call entirely to avoid corrupting the policy network with a "
                "NaN/Inf gradient. This does not fix the underlying numerical "
                "issue upstream (likely diverged global-model weights)."
            )
            self.clear_buffer()
            return {}

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

            if not torch.isfinite(loss):
                print(
                    f"[PPO WARNING] Non-finite loss ({float(loss)!r}) this epoch - "
                    "skipping optimizer.step() for this epoch only, network weights "
                    "left unchanged."
                )
                continue

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            clip_stats = {
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy.item()),
            }

        self.clear_buffer()
        self.update_count += 1
        return clip_stats


# ---------------------------------------------------------------------------
# Reward and energy helpers
# ---------------------------------------------------------------------------


def _print_state_transitions(
    rows: List[Tuple[int, str, str, str, str, Optional[int], "RewardBreakdown"]]
) -> None:
    """Print one line per UAV decision this round, showing the participation
    state transition actually caused by the PPO action (not just the raw
    action/observation metrics), plus the reward broken into its rho_t /
    Delta_psi / Lambda(a) components rather than only the summed total -
    added so a given decision's reward can be diagnosed term-by-term
    directly from the logs, instead of having to guess which term is
    driving it from the total alone."""
    for round_idx, uav_id, prev_state, new_state, action_name, quarantine_duration, reward in rows:
        transition = f"{prev_state} -> {new_state}" if prev_state != new_state else f"{prev_state} (unchanged)"
        extra = f" T_Q={quarantine_duration}" if quarantine_duration else ""
        print(
            f"Round {round_idx:>3} | {uav_id:<8} | {transition:<28} | "
            f"action={action_name:<18}{extra} | "
            f"reward={reward.total:8.3f} "
            f"(rho_t={reward.rho_t:6.3f} + d_psi={reward.delta_psi:10.6f} + "
            f"penalty={reward.penalty:8.3f})"
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
    tau: float,
) -> RewardBreakdown:
    """R_j^(t+1) = rho_j^(t) + Delta psi_j^(t) + Lambda(a)  (Eq. 26-27).

    Lambda(a) is piecewise in the action actually taken:
      Allow      -> -lambda_j^(t)
      Quarantine -> -(1 - lambda_j^(t)) * T_j^Q(t)
      Exclude    -> -(1 - lambda_j^(t)) * rho_j^(t) * e^(tau * q_j^(t))

    Allow's penalty was changed from 0 (the paper's literal Eq. 27) to -lambda_j^(t):
    directly punishing the policy for allowing a UAV in proportion to how contaminated
    the detector currently believes it is, independent of Delta_psi's noisier signal.

    Both governance actions are discounted by (1 - lambda_j) for internal
    consistency: quarantining/excluding a UAV the detector is confident is
    contaminated (lambda -> 1) should cost the policy little, while doing
    so to a UAV with lambda -> 0 should cost the full penalty. Previously
    only the Exclude branch carried this discount, so Quarantine gave the
    policy zero contamination-linked signal and could not learn to treat a
    lambda=0 UAV differently from a lambda=1 UAV when quarantining.

    The e^(tau*q) factor on Exclude's penalty (added after observing
    persistent over-exclusion in practice) is NOT a new tuned constant -
    it reuses the exact same exponential term already present in Eq. 28
    (T_j^Q = e^(tau*q)/(1+rho)) and in Eq. 7's penalty term
    (lambda*E_ret*e^(tau*q)/phi), using the same tau (RLConfig.penalty_tuning)
    Quarantine's own formula uses. Without it, rho (typically ~0.4-1.0 in
    practice) stays roughly flat while T_j^Q grows exponentially with q
    (e.g. ~1 at q=1 vs ~13 at q=3), making Exclude systematically ~10-25x
    cheaper than Quarantine for the same UAV once q accumulates even a
    little - a scale mismatch baked into the reward function itself, not
    a training or exploration problem, that would bias the policy toward
    Exclude regardless of how well-trained it is. Scaling by the same
    e^(tau*q) term Quarantine already uses closes most of that gap while
    keeping Exclude somewhat cheaper than Quarantine at very high q, which
    is arguably the right shape given Exclude is meant to be the higher-
    confidence, more severe, last-resort action.

    IMPORTANT: uses (flag_count + 1), not the raw flag_count. assign_quarantine
    increments flag_count BEFORE computing T_j^Q (Eq. 28), so Quarantine's own
    cost is always computed from the POST-increment q - comparing that against
    Exclude's PRE-increment q was an apples-to-oranges mismatch that left this
    e^(tau*q) fix completely inert for any UAV that had never been quarantined
    before (q=0 -> e^0=1 -> collapses back to the original flat, too-cheap
    -(1-lambda)*rho with no scaling at all). This matters most exactly at the
    moment EXCLUDE first unlocks: reset_flags_after_primer and
    release_quarantine_after_primer both fire at the same round
    exclude_unlocked does by default, meaning every UAV has q=0 right when
    Exclude becomes real for the first time - precisely where the unfixed
    version provided zero protection. Using q+1 consistently also gives this
    a sensible "escalation" property: a first-time flag (q=0->1) now costs
    MORE to Exclude than to Quarantine, i.e. try the reversible action before
    the irreversible one, rather than either being systematically favored.
    """
    if action == UAVAction.ALLOW:
        penalty = -uav.contamination_score
    elif action == UAVAction.QUARANTINE:
        lam = uav.contamination_score
        penalty = -(1.0 - lam) * float(quarantine_duration)
    elif action == UAVAction.EXCLUDE:
        lam = uav.contamination_score
        rho = uav.reputation_at_t
        penalty = -(1.0 - lam) * rho * math.exp(tau * (uav.flag_count + 1))
    else:
        raise ValueError(f"Unrecognized action {action!r}")

    # Eq. 26 uses rho_j^(t) explicitly (not the rho_j^(t+1) already computed
    # this round by _update_reputations via Eq. 7), so the pre-update value
    # cached in reputation_at_t must be used here rather than uav.reputation.
    total = uav.reputation_at_t + delta_psi + penalty
    return RewardBreakdown(
        total=total, rho_t=uav.reputation_at_t, delta_psi=delta_psi, penalty=penalty
    )


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
        self.ppo = PPOAgent(
            self.rl_config, device=self.config.device, total_rounds=self.config.num_rounds
        )
        self.validation_set: Optional[Dataset] = None
        self.round_actions: Dict[str, UAVAction] = {}
        self.shadow_excluded_this_round: set = set()
        self.round_history: List[RoundSnapshot] = []
        # Resolved once here, same pattern as PPOAgent's exclude-unlock
        # threshold: an absolute round number so the "primer" period scales
        # with --rounds rather than being a fixed count that means something
        # different at different horizon lengths. See RLConfig.
        # flag_reset_fraction/reset_flags_after_primer for rationale.
        self._flag_reset_round = max(
            1, round(self.rl_config.flag_reset_fraction * self.config.num_rounds)
        )
        self._flags_reset_done = False
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

    def _manage_states(
        self, round_idx: int, reference_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, float]]:
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

        Delta_psi_j (this round's accuracy attribution for UAV j) is computed
        as a per-UAV counterfactual against a single fixed baseline, not the
        previous sequential before/after-each-decision scheme:
          - baseline_acc: accuracy of the coalition with every UAV that was
            ACTIVE at the *start* of this round included (i.e. as if nobody's
            action had been applied yet).
          - For ALLOW: delta_psi = 0 by definition - the UAV's presence in
            the aggregation is unchanged from baseline, so there is nothing
            to attribute.
          - For QUARANTINE/EXCLUDE: delta_psi = accuracy(baseline set minus
            this UAV) - baseline_acc, i.e. "what if only this UAV's action
            differed from ALLOW, holding every other UAV at ALLOW."
        This removes the previous scheme's order-dependence (where UAV k's
        delta_psi depended on whichever actions had already been applied to
        UAVs processed earlier in the same round's loop) at the cost of not
        capturing interaction effects between simultaneous decisions within
        the same round - a deliberate reliability-over-speed tradeoff.
        """
        meta: Dict[str, Dict[str, float]] = {}
        transition_rows: List[Tuple[int, str, str, str, str, Optional[int], RewardBreakdown]] = []

        for coalition in self.rl_coalitions:
            # Fixed snapshot of who was active BEFORE any decision this round -
            # both the baseline and every counterfactual are evaluated against
            # this same set, never the live/mutating active_edge_uavs.
            full_active = list(coalition.active_edge_uavs)

            have_validation = self.validation_set is not None and len(full_active) > 0
            probe = copy.deepcopy(full_active[0].model) if have_validation else None
            baseline_acc = (
                _evaluate_weights_accuracy(
                    _aggregate_subset_weights(reference_weights, full_active),
                    probe,
                    self.validation_set,
                    self.config,
                )
                if have_validation
                else 0.0
            )

            for uav in full_active:
                # Defensive guard: full_active already filters to
                # ParticipationState.ACTIVE, but we assert it explicitly here
                # so that EXCLUDED (terminal) and QUARANTINED (temporarily
                # ineligible) UAVs can never be handed a PPO decision, even if
                # this method is refactored to iterate a different source list
                # in the future.
                if uav.participation != ParticipationState.ACTIVE:
                    continue

                prev_participation = uav.participation
                obs = uav.observation()
                if self.rl_config.shadow_exclude_during_primer:
                    # EXCLUDE is always sampleable; whether it's REALLY
                    # applied is decided below by exclude_unlocked. See
                    # RLConfig.shadow_exclude_during_primer for rationale.
                    action, log_prob, value, probs = self.ppo.select_action(obs)
                else:
                    # Old behavior: EXCLUDE is masked out of the action
                    # space entirely until exclude_unlocked.
                    action, log_prob, value, probs = self.ppo.select_action(
                        obs, allow_exclude=self.ppo.exclude_unlocked
                    )
                quarantine_assigned = 0
                shadow_excluded = False

                # The action PPO selects must always be the action that is
                # implemented on the UAV's participation state - previously
                # this was gated behind `isinstance(fog, RLFogUAV)`, so a
                # coalition with a plain FogUAV would silently record a
                # QUARANTINE/EXCLUDE action (and reward it accordingly) while
                # never actually changing uav.participation. Applying the
                # action directly to the UAV removes that dependency on the
                # fog's type and guarantees decision == effect.
                # EXCEPTION: a sampled EXCLUDE while exclude_unlocked is
                # still False and shadow_exclude_during_primer is True is a
                # deliberate exception to "decision == effect" - the reward
                # is computed as if it happened (see compute_reward/delta_psi
                # below, both already action-based rather than state-based),
                # but uav.participation is intentionally left unchanged so
                # the primer period carries zero real exclusion risk.
                if action == UAVAction.QUARANTINE:
                    quarantine_assigned = uav.assign_quarantine(self.rl_config.penalty_tuning)
                elif action == UAVAction.EXCLUDE:
                    if self.ppo.exclude_unlocked:
                        uav.exclude()
                    elif self.rl_config.shadow_exclude_during_primer:
                        shadow_excluded = True
                        self.shadow_excluded_this_round.add(uav.uav_id)
                    # else: unreachable when shadow_exclude_during_primer is
                    # False, since select_action was called with
                    # allow_exclude=False in that branch above, masking
                    # EXCLUDE out of the sampled distribution entirely.

                self.round_actions[uav.uav_id] = action

                if action == UAVAction.ALLOW:
                    delta_psi = 0.0
                elif have_validation:
                    counterfactual_subset = [u for u in full_active if u.uav_id != uav.uav_id]
                    counterfactual_acc = _evaluate_weights_accuracy(
                        _aggregate_subset_weights(reference_weights, counterfactual_subset),
                        probe,
                        self.validation_set,
                        self.config,
                    )
                    delta_psi = (counterfactual_acc - baseline_acc) * self.rl_config.delta_psi_scale
                else:
                    delta_psi = 0.0

                reward = compute_reward(
                    uav, delta_psi, action, quarantine_assigned, self.rl_config.penalty_tuning
                )
                meta[uav.uav_id] = {
                    "reward": reward.total,
                    "delta_psi": delta_psi,
                    "quarantine_assigned": float(quarantine_assigned),
                }

                transition_rows.append((
                    round_idx,
                    uav.uav_id,
                    prev_participation.name,
                    uav.participation.name,
                    action.name + (" (SHADOW)" if shadow_excluded else ""),
                    quarantine_assigned or None,
                    reward,
                ))

                self.ppo.store(Transition(
                    obs=obs,
                    action=action.value,
                    log_prob=log_prob,
                    value=value,
                    reward=reward.total,
                    # Every transition is its own complete episode: this buffer
                    # holds one round's decisions across every active UAV in
                    # every coalition, flattened into a single list and wiped
                    # before next round - it is NOT one UAV's continuing
                    # trajectory across time. Marking only EXCLUDE as done
                    # (previous behavior) caused _compute_returns to chain
                    # rewards backward across *adjacent but unrelated* UAVs'
                    # transitions (e.g. UAV A's ALLOW return would include a
                    # discounted share of UAV B's EXCLUDE penalty, purely
                    # because B happened to be next in the list) - there is no
                    # legitimate "next state" relationship between rows here
                    # to bootstrap across. done=True makes every return equal
                    # to its own reward, matching what this buffer actually
                    # contains. A persistent per-UAV rollout buffer spanning
                    # multiple rounds would be a prerequisite for done to mean
                    # anything more than this.
                    done=True,
                ))

        if transition_rows:
            _print_state_transitions(transition_rows)

        return meta

    def _maybe_reset_flags(self, round_idx: int) -> bool:
        """One-time q (flag_count) reset, and optionally quarantine release,
        once the primer period ends.

        See RLConfig.reset_flags_after_primer/flag_reset_fraction/
        release_quarantine_after_primer for the rationale (mitigating
        early-exploration quarantine flags - and the sentences computed
        from them - compounding via Eq. 28's exponential T_j^Q for the rest
        of the run). Fires at most once per run() call, at round
        self._flag_reset_round. The q reset applies to every RLEdgeUAV
        regardless of current participation state (harmless no-op for
        EXCLUDED UAVs, since they never re-enter the decision loop anyway).
        The quarantine release only ever applies to currently QUARANTINED
        UAVs - EXCLUDED remains terminal, untouched here, per Algorithm 1.
        """
        if not self.rl_config.reset_flags_after_primer or self._flags_reset_done:
            return False
        if round_idx < self._flag_reset_round:
            return False

        for uav in self.rl_edge_uavs:
            uav.flag_count = 0

        released = []
        if self.rl_config.release_quarantine_after_primer:
            for uav in self.quarantined_uavs:
                uav.participation = ParticipationState.ACTIVE
                uav.quarantine_rounds_remaining = 0
                uav.load_state_dict(self.global_model.state_dict())
                released.append(uav.uav_id)

        self._flags_reset_done = True
        print(
            f"[Flag Reset] q reset to 0 for all {len(self.rl_edge_uavs)} UAVs "
            f"at round {round_idx} (primer period ended)"
            + (f"; released from quarantine: {released}" if released else "")
        )
        return True

    def train_round(self, round_idx: int) -> Dict[str, float]:
        """One HFL round with PPO state management (Algorithm 1)."""
        self._maybe_reset_flags(round_idx)
        self._process_quarantine_expiry()
        self.distribute_global_model()
        reference_weights = copy.deepcopy(self.global_model.state_dict())

        local_losses: Dict[str, float] = {}
        for uav in self.active_uavs:
            local_losses[uav.uav_id] = uav.train_local(self.config)
            uav.apply_poison(reference_weights)

        self._run_contamination_detection(reference_weights)
        self._update_reputations(reference_weights)

        self._manage_states(round_idx, reference_weights)

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
            # round_actions is cleared at the top of every round (run()) and
            # only repopulated for UAVs actually processed by _manage_states
            # that round (i.e. ParticipationState.ACTIVE at round start). A
            # UAV still serving out an earlier quarantine sentence is skipped
            # entirely this round - no fresh PPO decision was made for it.
            # Previously this fell back to UAVAction.ALLOW, which mislabeled
            # every such holdover as if the policy had freshly chosen Allow
            # this round (readable in logs as e.g. "-> ALLOW (QUARANTINED)"),
            # when no decision was made at all. "HOLD" makes that distinction
            # explicit instead of silently reusing a real action's name.
            if uav.uav_id in self.round_actions:
                action_label = self.round_actions[uav.uav_id].name
                if uav.uav_id in self.shadow_excluded_this_round:
                    # Sampled EXCLUDE during the primer period, rewarded as
                    # if applied, but participation deliberately left
                    # unchanged - see RLConfig.shadow_exclude_during_primer.
                    # Marked explicitly so this isn't confused with a real
                    # exclude when read alongside participation=ACTIVE.
                    action_label += " (SHADOW)"
            else:
                action_label = "HOLD"
            snapshots.append(
                UAVRoundSnapshot(
                    uav_id=uav.uav_id,
                    coalition_id=uav.coalition_id,
                    contamination_score=uav.contamination_score,
                    reputation=uav.reputation,
                    flag_count=uav.flag_count,
                    action=action_label,
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
        self.shadow_excluded_this_round.clear()
        self._flags_reset_done = False

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
            self.shadow_excluded_this_round.clear()
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
