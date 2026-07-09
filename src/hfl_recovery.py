"""
ReCon Sec. 4.3 – Checkpoint-Based Recovery and Quarantine Management.

Implements Algorithm 2 (Checkpoint-Based Recovery and Model Reconstruction):
  - Checkpoint urgency score κ = (t - t_c) * Σ_{c_k} Σ_{u_j in c_k} λ_j * q_j   (Eq. checkpoint_urgency)
  - Checkpoint creation when κ ≥ κ_th                                               (Eq. checkpoint_trigger)
  - Coalition model rollback to the last safe checkpoint when an EXCLUDE fires
  - Global model reconstruction from rolled-back coalition weights
  - Quarantine expiry: rejoining UAVs receive the current global model

NOTE - confirmed deviation from Algorithm 2 as literally written, default
behavior (RecoveryConfig.checkpoint_after_rollback=True; see that field's
docstring and HFLRecoveryStation.train_round for full detail):
Algorithm 2 (ReCon.tex line 635-659, \\label{alg:checkpoint_recovery})
checkpoints unconditionally FIRST every round, THEN checks for exclusion and
rolls back - same iteration, in that literal order. That has a real design
weakness the paper itself doesn't address: kappa (checkpoint urgency) is
driven by accumulating contamination, so a checkpoint that fires is already
reactive to damage that's been building; if an exclusion ALSO fires the
same round, rolling back to a checkpoint saved moments earlier that same
round barely helps, since it already reflects most of the same accumulated
contamination. This module instead checkpoints AFTER any same-round
rollback by default, so a fresh checkpoint (when warranted) captures the
just-cleaned, post-rollback state. Set checkpoint_after_rollback=False to
restore the paper's literal ordering exactly.

Known remaining gap with the default (True) ordering, not yet addressed:
during a stretch of several CONSECUTIVE rounds that each trigger a rollback
(no "quiet" round in between), no new checkpoint is ever saved - the
default ordering only checkpoints on rounds that did NOT just roll back, so
t_c can stay anchored to an increasingly old round throughout a sustained
attack, before any of THOSE rounds' governance actions get a chance to be
captured as a new trusted baseline. Worth revisiting if sustained multi-
round contamination streaks turn out to be common in practice.

Dependency graph (acyclic):
  hfl_common  ->  hfl_base  ->  hfl_rl  ->  hfl_recovery

This module imports from hfl_rl and hfl_base but is never imported by them,
so no circular dependency is introduced.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from hfl_base import BaseStation, Coalition, FogUAV
from hfl_common import FashionMNISTNet, HFLConfig, load_fashion_mnist, partition_dataset
from hfl_rl import (
    ContaminationDetector,
    HFLRLStation,
    ParticipationState,
    RLCoalition,
    RLConfig,
    RLEdgeUAV,
    RLFogUAV,
    UAVAction,
    build_hfl_rl_system,
    zero_contamination_detector,
)


# ---------------------------------------------------------------------------
# Recovery configuration
# ---------------------------------------------------------------------------


@dataclass
class RecoveryConfig:
    """
    Hyper-parameters for the checkpoint and rollback mechanism (Sec. 4.3).

    Attributes
    ----------
    checkpoint_threshold:
        κ_th — urgency threshold above which a checkpoint is created.
        Lower values create more frequent checkpoints (tighter bound on
        contamination exposure per Theorem 2) at the cost of more storage.
    checkpoint_after_rollback:
        Confirmed deviation from Algorithm 2 (ReCon.tex line 635-659,
        \\label{alg:checkpoint_recovery}) if True (the default). The paper's
        literal pseudocode checkpoints FIRST every round (line 645-647,
        unconditional on kappa), THEN checks for exclusion and rolls back
        (line 649-652) - same iteration, checkpoint always precedes the
        exclusion check. That ordering has a real weakness: kappa is driven
        by accumulating contamination (Eq. checkpoint_urgency), so any
        checkpoint that fires is, by construction, already reactive to
        contamination that's been building - if an exclusion ALSO fires the
        same round, rolling back to a checkpoint saved moments earlier in
        that same round provides little protection, since that checkpoint
        already reflects most of the same accumulated damage. True (default)
        moves the checkpoint decision to the end of train_round, after any
        same-round rollback, so a fresh checkpoint (when kappa still
        warrants one) captures the just-cleaned, post-rollback state instead.
        Set False to restore the paper's literal Algorithm 2 ordering exactly.
    """

    checkpoint_threshold: float = 5.0    # κ_th (design parameter)
    checkpoint_after_rollback: bool = True


# ---------------------------------------------------------------------------
# Checkpoint store
# ---------------------------------------------------------------------------


@dataclass
class CheckpointStore:
    """
    Stores the per-coalition fog model weights at the last trusted round.

    M_ckpt maps coalition_id -> deep copy of fog coalition weights at t_c.
    t_c is the round index at which the last checkpoint was taken.
    """

    t_c: int = 0                                               # last checkpoint round
    coalition_weights: Dict[str, Dict[str, torch.Tensor]] = field(default_factory=dict)

    def save(
        self,
        round_idx: int,
        coalition_weights: Dict[str, Dict[str, torch.Tensor]],
    ) -> None:
        """Overwrite the stored checkpoint with fresh coalition weights."""
        self.t_c = round_idx
        self.coalition_weights = {
            cid: {k: v.clone() for k, v in weights.items()}
            for cid, weights in coalition_weights.items()
        }

    def is_empty(self) -> bool:
        return not self.coalition_weights


# ---------------------------------------------------------------------------
# Checkpoint urgency (Eq. checkpoint_urgency)
# ---------------------------------------------------------------------------


def compute_checkpoint_urgency(
    round_idx: int,
    last_checkpoint_round: int,
    active_uavs: List[RLEdgeUAV],
) -> float:
    """
    κ = (t - t_c) * Σ_{c_k} Σ_{u_j ∈ c_k} λ_j^(t) * q_j^(t)   (Eq. checkpoint_urgency)

    The sum runs over all currently-active edge UAVs since excluded/quarantined
    UAVs do not contribute gradients to the coalition model.

    Parameters
    ----------
    round_idx:
        Current FL round t.
    last_checkpoint_round:
        Round index t_c of the previous checkpoint (0 if none yet).
    active_uavs:
        All RLEdgeUAV objects whose participation == ACTIVE.

    Returns
    -------
    float
        Checkpoint urgency score κ ≥ 0.
    """
    rounds_since_checkpoint = round_idx - last_checkpoint_round
    contamination_mass = sum(
        uav.contamination_score * uav.flag_count
        for uav in active_uavs
    )
    return rounds_since_checkpoint * contamination_mass


# ---------------------------------------------------------------------------
# HFL station with checkpoint + rollback
# ---------------------------------------------------------------------------


class HFLRecoveryStation(HFLRLStation):
    """
    Extends HFLRLStation (PPO state management) with the checkpoint-based
    recovery mechanism described in Sec. 4.3 / Algorithm 2.

    New behaviour per round (inserted into train_round). Ordering is
    controlled by RecoveryConfig.checkpoint_after_rollback (default True);
    see that field's docstring and train_round's own docstring for the full
    rationale and the confirmed deviation from Algorithm 2 as literally
    written (ReCon.tex line 635-659) when the default is used:
      1. Run this round's PPO state decisions and aggregation first.
      2. If any UAV was newly EXCLUDED this round -> roll back all
         coalition models to the last checkpoint (from a prior round),
         reconstruct and broadcast the global model from rolled-back weights.
      3. Otherwise, compute κ; if κ ≥ κ_th -> save a fresh coalition model
         checkpoint (t_c = t) reflecting this round's already-validated state.
      4. Quarantine expiry is handled identically to HFLRLStation but the
         rejoining UAV also receives the (possibly rolled-back) global model.
    """

    def __init__(
        self,
        config: Optional[HFLConfig] = None,
        rl_config: Optional[RLConfig] = None,
        recovery_config: Optional[RecoveryConfig] = None,
        contamination_detector: Optional[ContaminationDetector] = None,
    ) -> None:
        super().__init__(config, rl_config, contamination_detector)
        self.recovery_config = recovery_config or RecoveryConfig()
        self.checkpoint_store = CheckpointStore()

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _current_coalition_weights(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Collect post-aggregation fog weights for every RL coalition."""
        weights: Dict[str, Dict[str, torch.Tensor]] = {}
        for coalition in self.rl_coalitions:
            if coalition.active_edge_uavs:
                weights[coalition.coalition_id] = coalition.aggregate_active_weights()
        return weights

    def _maybe_checkpoint(self, round_idx: int) -> bool:
        """
        Evaluate κ and, if κ ≥ κ_th, persist the current coalition models.

        Returns True when a new checkpoint was created.
        """
        kappa = compute_checkpoint_urgency(
            round_idx,
            self.checkpoint_store.t_c,
            self.active_uavs,
        )
        if kappa >= self.recovery_config.checkpoint_threshold:
            coalition_weights = self._current_coalition_weights()
            if coalition_weights:
                self.checkpoint_store.save(round_idx, coalition_weights)
                print(
                    f"  [Recovery] Checkpoint saved at round {round_idx} "
                    f"(κ={kappa:.3f} ≥ κ_th={self.recovery_config.checkpoint_threshold})"
                )
                return True
        return False

    # ------------------------------------------------------------------
    # Rollback helpers
    # ------------------------------------------------------------------

    def _rollback_and_reconstruct(self, round_idx: int) -> None:
        """
        Algorithm 2, lines 9–11:
          Restore M_{c_k} ← M_{c_k}^{t_c} for all coalitions, then
          reconstruct M^{t_c} = Aggregate({M_{c_k}^{t_c}}) and distribute.

        If no checkpoint exists yet we cannot roll back; a warning is printed
        and training continues from the current global model, which is the
        best available fallback.
        """
        if self.checkpoint_store.is_empty():
            print(
                f"  [Recovery] EXCLUDE detected at round {round_idx} but no checkpoint "
                "exists yet — cannot roll back. Consider lowering checkpoint_threshold."
            )
            return

        t_c = self.checkpoint_store.t_c
        print(
            f"  [Recovery] EXCLUDE detected at round {round_idx}. "
            f"Rolling back all coalitions to checkpoint t_c={t_c}."
        )

        # Restore each coalition's fog model to the checkpointed weights.
        restored_coalition_weights: Dict[str, Dict[str, torch.Tensor]] = {}
        for coalition in self.rl_coalitions:
            cid = coalition.coalition_id
            if cid in self.checkpoint_store.coalition_weights:
                ckpt_weights = {
                    k: v.clone()
                    for k, v in self.checkpoint_store.coalition_weights[cid].items()
                }
                # Push rolled-back weights to all (non-excluded) edge UAVs in the coalition.
                for uav in coalition.edge_uavs:
                    if isinstance(uav, RLEdgeUAV) and uav.participation != ParticipationState.EXCLUDED:
                        uav.load_state_dict(ckpt_weights)
                restored_coalition_weights[cid] = ckpt_weights
            else:
                # Coalition had no checkpoint (e.g., was empty at t_c).
                # Use current aggregation as-is rather than corrupting model.
                if coalition.active_edge_uavs:
                    restored_coalition_weights[cid] = coalition.aggregate_active_weights()

        if not restored_coalition_weights:
            print("  [Recovery] No coalition weights available for reconstruction.")
            return

        # Reconstruct global model from rolled-back coalition weights (Algorithm 2 line 10).
        total_samples = sum(
            coalition.active_samples
            for coalition in self.rl_coalitions
            if coalition.coalition_id in restored_coalition_weights
        )
        global_weights: Dict[str, torch.Tensor] = {}
        for coalition in self.rl_coalitions:
            cid = coalition.coalition_id
            if cid not in restored_coalition_weights:
                continue
            coef = coalition.active_samples / max(total_samples, 1)
            for key, tensor in restored_coalition_weights[cid].items():
                global_weights[key] = (
                    global_weights.get(key, torch.zeros_like(tensor)) + coef * tensor
                )

        if global_weights:
            self.global_model.load_state_dict(global_weights)

        # Distribute the reconstructed global model to all active UAVs (Algorithm 2 line 11).
        self.distribute_global_model()
        print(f"  [Recovery] Global model reconstructed from t_c={t_c} and distributed.")

    # ------------------------------------------------------------------
    # Quarantine expiry (override to add logging)
    # ------------------------------------------------------------------

    def _process_quarantine_expiry(self) -> List[RLEdgeUAV]:
        """
        Algorithm 2, lines 13–15:
          For each u_j in U^Q with T_j^Q = 0: send M^(t) to u_j, move u_j -> U^A.

        The base class already loads the current global model into the rejoining
        UAV and flips its participation state; we only add a log line here.
        """
        rejoined = super()._process_quarantine_expiry()
        for uav in rejoined:
            print(
                f"  [Recovery] UAV {uav.uav_id} quarantine expired — "
                f"rejoined U^A with current global model (flags={uav.flag_count})."
            )
        return rejoined

    # ------------------------------------------------------------------
    # Main round override
    # ------------------------------------------------------------------

    def train_round(self, round_idx: int) -> Dict[str, float]:
        """
        One HFL round with PPO + checkpoint-based recovery.

        Two orderings, controlled by RecoveryConfig.checkpoint_after_rollback:

        checkpoint_after_rollback=False (paper-literal, Algorithm 2 line
        635-659 exactly):
          (a) Compute kappa; if kappa >= kappa_th, checkpoint (t_c = t) -
              unconditionally, every round, BEFORE this round's decisions.
          (b) Run this round's decisions (local training, state management,
              aggregation).
          (c) If any EXCLUDE action occurred, roll back to the last
              checkpoint (t_c) and reconstruct - which, if (a) just fired
              this same round, is the checkpoint saved moments earlier in
              step (a) of THIS round.

        checkpoint_after_rollback=True (default; confirmed deviation from
        Algorithm 2 - see RecoveryConfig docstring for the full rationale):
          (a) Run this round's decisions first.
          (b) If any EXCLUDE action occurred, roll back to the last
              checkpoint (from a PRIOR round, never one just saved this
              same round under this ordering) and reconstruct.
          (c) Otherwise (a "quiet" round), evaluate kappa and checkpoint if
              warranted, using this round's freshly-computed active UAVs
              and lambda/q values (a side effect of running (a) first:
              _run_contamination_detection has already updated every active
              UAV's contamination_score by the time this checkpoint decision
              runs, whereas under the paper-literal ordering it uses
              whatever contamination_score values were left over from the
              PREVIOUS round's detection pass).
        """
        if not self.recovery_config.checkpoint_after_rollback:
            # Paper-literal Algorithm 2 ordering: checkpoint unconditionally
            # first, every round, before this round's decisions even run.
            self._maybe_checkpoint(round_idx)
            previously_excluded = {u.uav_id for u in self.excluded_uavs}
            losses = super().train_round(round_idx)
            newly_excluded = {u.uav_id for u in self.excluded_uavs} - previously_excluded
            if newly_excluded:
                print(
                    f"  [Recovery] UAV(s) {newly_excluded} permanently excluded. "
                    "Initiating checkpoint rollback."
                )
                self._rollback_and_reconstruct(round_idx)
            return losses

        # Snapshot which UAVs were excluded before this round's decisions so
        # we can detect new exclusions after the round completes.
        previously_excluded = {u.uav_id for u in self.excluded_uavs}

        # (a) Run the standard RL training round (local train -> state mgmt -> aggregate -> PPO update).
        losses = super().train_round(round_idx)

        # (b) Check for new exclusions and trigger rollback if any occurred.
        newly_excluded = {u.uav_id for u in self.excluded_uavs} - previously_excluded
        if newly_excluded:
            print(
                f"  [Recovery] UAV(s) {newly_excluded} permanently excluded. "
                "Initiating checkpoint rollback."
            )
            self._rollback_and_reconstruct(round_idx)
        else:
            # (c) Only checkpoint on rounds that didn't just roll back - a
            # round immediately following (or containing) a rollback
            # reverted to an already-checkpointed state, so re-checkpointing
            # it here would just re-save something we already have, and
            # kappa's inputs (lambda/q of whoever triggered the exclusion)
            # would likely still read as elevated immediately afterward.
            self._maybe_checkpoint(round_idx)

        return losses

    # ------------------------------------------------------------------
    # Run override for cleaner reporting
    # ------------------------------------------------------------------

    def run(
        self,
        test_dataset: Dataset,
        validation_dataset: Optional[Dataset] = None,
    ) -> List[Dict[str, float]]:
        """
        Execute all FL rounds with recovery, extending the history entries
        with checkpoint and exclusion metadata.
        """
        self.validation_set = validation_dataset or test_dataset

        for uav in self.rl_edge_uavs:
            uav.reputation = self.rl_config.initial_reputation
            uav.residual_energy = self.rl_config.initial_energy
            uav.model_contribution = self.rl_config.default_contribution
            uav.participation = ParticipationState.ACTIVE
            uav.flag_count = 0
            uav.quarantine_rounds_remaining = 0

        # Reset recovery state.
        self.checkpoint_store = CheckpointStore()
        self.round_history.clear()
        self.round_actions.clear()

        # Save initial trusted checkpoint (t_c = 0) before any training round.
        initial_weights = {
            cid: {k: v.clone() for k, v in coalition.aggregate_active_weights().items()}
            for cid, coalition in (
                (c.coalition_id, c) for c in self.rl_coalitions if c.active_edge_uavs
            )
        }
        if initial_weights:
            self.checkpoint_store.save(0, initial_weights)
            print(f"  [Recovery] Initial checkpoint saved at t_c=0")

        history: List[Dict[str, float]] = []

        for round_idx in range(1, self.config.num_rounds + 1):
            self.round_actions.clear()
            losses = self.train_round(round_idx)
            loss, acc = self._evaluate_global(test_dataset)

            entry: Dict[str, float] = {
                "round": round_idx,
                "global_loss": loss,
                "global_accuracy": acc,
                "active_uavs": len(self.active_uavs),
                "quarantined_uavs": len(self.quarantined_uavs),
                "excluded_uavs": len(self.excluded_uavs),
                "avg_local_loss": sum(
                    v for k, v in losses.items() if not k.startswith("__")
                ) / max(len(self.active_uavs), 1),
                "ppo_policy_loss": losses.get("__ppo_policy_loss__", 0.0),
                "last_checkpoint_round": float(self.checkpoint_store.t_c),
            }
            history.append(entry)

            ppo_loss = losses.get("__ppo_policy_loss__", 0.0)
            snapshot = self._capture_round_snapshot(round_idx, loss, acc, ppo_loss)
            snapshot.last_checkpoint_round = self.checkpoint_store.t_c
            self.round_history.append(snapshot)
            self._print_round_summary(snapshot)
            if self.checkpoint_store.t_c:
                print(f"  [Recovery] last checkpoint at round t_c={self.checkpoint_store.t_c}")

        return history


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_hfl_recovery_system(
    coalition_specs,
    config: Optional[HFLConfig] = None,
    rl_config: Optional[RLConfig] = None,
    recovery_config: Optional[RecoveryConfig] = None,
    contamination_detector: Optional[ContaminationDetector] = None,
    malicious_uavs: Optional[Sequence[str]] = None,
    poison_scale: float = 50.0,
) -> HFLRecoveryStation:
    """
    Build a full ReCon system: hierarchical FL + PPO state management +
    checkpoint-based recovery.

    Parameters
    ----------
    coalition_specs:
        Sequence of (coalition_id, [edge_uav_id, ...]).
    config:
        HFL training hyper-parameters.
    rl_config:
        PPO and reputation hyper-parameters.
    recovery_config:
        Checkpoint urgency threshold κ_th.
    contamination_detector:
        Callable phi: RLEdgeUAV -> [0, 1].  Supply your own detector module
        here; defaults to the zero stub if omitted.
    """
    cfg = config or HFLConfig()
    rl_cfg = rl_config or RLConfig()
    rec_cfg = recovery_config or RecoveryConfig()
    detector = contamination_detector or zero_contamination_detector
    malicious = set(malicious_uavs or [])

    all_edge_ids = [eid for _, members in coalition_specs for eid in members]
    train_set, _ = load_fashion_mnist(cfg.data_dir)
    shards = partition_dataset(train_set, len(all_edge_ids))
    shard_map = dict(zip(all_edge_ids, shards))

    station = HFLRecoveryStation(cfg, rl_cfg, rec_cfg, detector)

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
