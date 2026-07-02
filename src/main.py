"""
Entry point for the ReCon HFL simulation.

Dependency graph (acyclic):
  hfl_common  ->  hfl_base  ->  hfl_rl  ->  hfl_recovery
                      ^            ^               ^
                      |            |               |
                 flguardian_det ----+---------------+
                      |
                      +----------- main -----------+

ReCon pipeline (ReCon.tex):
  Edge UAVs train locally -> Fog runs FLGuardian (φ) -> reputation update
  -> PPO {Allow, Quarantine, Exclude} -> hierarchical aggregation
  -> checkpoint rollback on Exclude (recovery mode)
"""

from __future__ import annotations

import argparse
from typing import List, Sequence, Tuple

from flguardian_det import build_flguardian_hfl_adapter
from hfl_base import build_hfl_system
from hfl_common import HFLConfig, load_fashion_mnist
from hfl_recovery import RecoveryConfig, build_hfl_recovery_system
from hfl_rl import (
    HFLRLStation,
    RLConfig,
    RoundSnapshot,
    build_hfl_rl_system,
    zero_contamination_detector,
)

CoalitionSpec = Tuple[str, Sequence[str]]

def _build_default_coalitions(num_coalitions: int = 5, uavs_per_coalition: int = 10) -> List[CoalitionSpec]:
    """c1..cN, each holding a contiguous block of u1..u(N*uavs_per_coalition).

    e.g. c1 = u1-u10, c2 = u11-u20, ... c5 = u41-u50.
    """
    coalitions: List[CoalitionSpec] = []
    next_uav = 1
    for c_idx in range(1, num_coalitions + 1):
        members = [f"u{next_uav + i}" for i in range(uavs_per_coalition)]
        next_uav += uavs_per_coalition
        coalitions.append((f"c{c_idx}", members))
    return coalitions


DEFAULT_COALITIONS: List[CoalitionSpec] = _build_default_coalitions()

# ~20% (10/50) malicious UAVs, 2 per coalition so every coalition (c1..c5)
# contains at least one attacker.
DEFAULT_MALICIOUS: List[str] = [
    "u3", "u8",     # c1 (u1-u10)
    "u13", "u18",   # c2 (u11-u20)
    "u23", "u28",   # c3 (u21-u30)
    "u33", "u38",   # c4 (u31-u40)
    "u43", "u48",   # c5 (u41-u50)
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ReCon: reputation-aware HFL with FLGuardian contamination detection"
    )
    parser.add_argument(
        "--mode",
        choices=["base", "rl", "recovery"],
        default="recovery",
        help=(
            "base: hierarchical FL only; "
            "rl: FL + PPO state management; "
            "recovery: FL + PPO + checkpoint rollback (default)"
        ),
    )
    parser.add_argument(
        "--detector",
        choices=["flguardian", "none"],
        default="flguardian",
        help="Contamination detector φ (default: flguardian)",
    )
    parser.add_argument(
        "--malicious",
        nargs="*",
        default=DEFAULT_MALICIOUS,
        help="Edge UAV ids simulated as model-poisoning attackers "
        "(default: 10 UAVs, ~20%%, 2 per coalition)",
    )
    parser.add_argument(
        "--poison-scale",
        type=float,
        default=50.0,
        help="Gradient scale for malicious UAV updates (default: 50)",
    )
    parser.add_argument(
        "--checkpoint-threshold",
        type=float,
        default=2.0,
        help="κ_th: urgency score for checkpoint creation (default: 2.0)",
    )
    parser.add_argument("--rounds", type=int, default=100, help="Number of FL rounds")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Local SGD learning rate")
    parser.add_argument("--data-dir", type=str, default="./data", help="FashionMNIST cache dir")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="(unused - kept for CLI compatibility; results now print to terminal only)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (cpu or cuda)",
    )
    return parser.parse_args()


def make_hfl_config(args: argparse.Namespace) -> HFLConfig:
    return HFLConfig(
        data_dir=args.data_dir,
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )


def make_contamination_detector(args: argparse.Namespace):
    if args.detector == "none":
        return zero_contamination_detector
    return build_flguardian_hfl_adapter(device=args.device)


def run_base_mode(config: HFLConfig, coalitions: List[CoalitionSpec]):
    print("Running base HFL (no RL state management)")
    station = build_hfl_system(coalitions, config=config)
    _, test_set = load_fashion_mnist(config.data_dir)
    station.run(test_set)
    return station


def run_rl_mode(
    config: HFLConfig,
    coalitions: List[CoalitionSpec],
    detector,
    malicious_uavs: Sequence[str],
    poison_scale: float,
    rl_config: RLConfig | None = None,
) -> HFLRLStation:
    print("Running HFL with PPO-based UAV state management + FLGuardian φ")
    station = build_hfl_rl_system(
        coalitions,
        config=config,
        rl_config=rl_config or RLConfig(),
        contamination_detector=detector,
        malicious_uavs=malicious_uavs,
        poison_scale=poison_scale,
    )
    _, test_set = load_fashion_mnist(config.data_dir)
    station.run(test_set)
    return station


def run_recovery_mode(
    config: HFLConfig,
    coalitions: List[CoalitionSpec],
    detector,
    malicious_uavs: Sequence[str],
    poison_scale: float,
    rl_config: RLConfig | None = None,
    recovery_config: RecoveryConfig | None = None,
):
    print("Running full ReCon: FLGuardian φ + PPO governance + checkpoint recovery")
    station = build_hfl_recovery_system(
        coalitions,
        config=config,
        rl_config=rl_config or RLConfig(),
        recovery_config=recovery_config,
        contamination_detector=detector,
        malicious_uavs=malicious_uavs,
        poison_scale=poison_scale,
    )
    _, test_set = load_fashion_mnist(config.data_dir)
    station.run(test_set)
    return station


def _count_checkpoints_created(station, history: List[RoundSnapshot]) -> object:
    """Exact count of checkpoints created during the run.

    hfl_recovery.HFLRecoveryStation.run() saves an initial checkpoint at
    t_c=0 before round 1, then _maybe_checkpoint() bumps
    checkpoint_store.t_c to the current round whenever kappa >= kappa_th.
    Each RoundSnapshot.last_checkpoint_round mirrors checkpoint_store.t_c
    for that round, so every time that value changes from the previous
    round, exactly one new checkpoint was created that round.

    Returns "N/A" if this station has no checkpoint_store (base/rl mode).
    """
    if not hasattr(station, "checkpoint_store") or not history:
        return "N/A"
    tc_sequence = [0] + [s.last_checkpoint_round for s in history]
    created = 1  # the initial checkpoint saved at t_c=0 before round 1
    for prev_tc, tc in zip(tc_sequence, tc_sequence[1:]):
        if tc != prev_tc:
            created += 1
    return created


def _count_rollbacks(station, history: List[RoundSnapshot]) -> object:
    """Exact count of checkpoint rollbacks during the run.

    HFLRecoveryStation.train_round() calls _rollback_and_reconstruct()
    exactly once per round in which any UAV was newly EXCLUDED that round.
    EXCLUDED is a terminal state, so excluded_uavs only ever increases;
    counting rounds where it grew versus the previous round gives the
    exact number of rollbacks triggered.

    Returns "N/A" if this station has no checkpoint_store (base/rl mode).
    """
    if not hasattr(station, "checkpoint_store") or not history:
        return "N/A"
    excluded_sequence = [0] + [s.excluded_uavs for s in history]
    return sum(
        1 for prev_n, n in zip(excluded_sequence, excluded_sequence[1:]) if n > prev_n
    )


def print_simulation_summary(
    station,
    mode: str,
    coalitions: List[CoalitionSpec],
    malicious_uavs: Sequence[str],
    num_rounds: int,
) -> None:
    """Print a concise terminal-only execution summary (no file output)."""
    history: List[RoundSnapshot] = getattr(station, "round_history", []) or []
    total_coalitions = len(coalitions)
    total_uavs = sum(len(members) for _, members in coalitions)
    malicious_set = set(malicious_uavs)
    total_malicious = len(malicious_set)

    print("\n====================================")
    print("Simulation Summary")
    print("====================================")
    print(f"Total rounds        : {num_rounds}")
    print(f"Total coalitions     : {total_coalitions}")
    print(f"Total UAVs           : {total_uavs}")
    print(f"Total malicious UAVs : {total_malicious}")

    if not history:
        print("\nNo round history was recorded for this run (mode="
              f"{mode!r}); RL/reputation/checkpoint stats are unavailable.")
        print("====================================")
        return

    final = history[-1]
    final_uavs = final.uav_snapshots

    accuracies = [s.global_accuracy for s in history]
    avg_accuracy = sum(accuracies) / len(accuracies)
    max_accuracy = max(accuracies)

    avg_reputation = (
        sum(u.reputation for u in final_uavs) / len(final_uavs) if final_uavs else 0.0
    )
    avg_contamination = (
        sum(u.contamination_score for u in final_uavs) / len(final_uavs) if final_uavs else 0.0
    )

    checkpoints_created = _count_checkpoints_created(station, history)
    rollbacks = _count_rollbacks(station, history)

    mal_quarantined = sum(
        1 for u in final_uavs if u.uav_id in malicious_set and u.participation == "QUARANTINED"
    )
    mal_excluded = sum(
        1 for u in final_uavs if u.uav_id in malicious_set and u.participation == "EXCLUDED"
    )
    mal_active = sum(
        1 for u in final_uavs if u.uav_id in malicious_set and u.participation == "ACTIVE"
    )

    ben_quarantined = sum(
        1 for u in final_uavs if u.uav_id not in malicious_set and u.participation == "QUARANTINED"
    )
    ben_excluded = sum(
        1 for u in final_uavs if u.uav_id not in malicious_set and u.participation == "EXCLUDED"
    )

    print("\nFinal counts:")
    print(f"  Active      : {final.active_uavs}")
    print(f"  Quarantined : {final.quarantined_uavs}")
    print(f"  Excluded    : {final.excluded_uavs}")

    print(f"\nAverage global accuracy    : {avg_accuracy:.4f}")
    print(f"Highest global accuracy    : {max_accuracy:.4f}")
    print(f"Average reputation         : {avg_reputation:.4f}")
    print(f"Average contamination score: {avg_contamination:.4f}")

    print(f"\nNumber of checkpoint rollbacks : {rollbacks}")
    print(f"Number of checkpoints created  : {checkpoints_created}")

    print("\nMalicious UAV statistics:")
    print(f"  Quarantined  : {mal_quarantined}")
    print(f"  Excluded     : {mal_excluded}")
    print(f"  Still active : {mal_active}")

    print("\nBenign UAV statistics:")
    print(f"  Quarantined : {ben_quarantined}")
    print(f"  Excluded    : {ben_excluded}")
    print("====================================")


def main() -> None:
    args = parse_args()
    config = make_hfl_config(args)
    detector = make_contamination_detector(args)
    malicious = args.malicious if args.mode != "base" else []

    print("=" * 72)
    print("ReCon — Reputation-Aware Contamination Governance for UAV-HFL")
    print("=" * 72)
    print(f"Coalitions : {DEFAULT_COALITIONS}")
    print(f"Detector   : {args.detector}")
    print(f"Malicious  : {malicious or 'none'}")
    print(f"Rounds     : {args.rounds}")
    print("=" * 72)

    station = None
    if args.mode == "base":
        station = run_base_mode(config, DEFAULT_COALITIONS)
    elif args.mode == "rl":
        station = run_rl_mode(
            config, DEFAULT_COALITIONS, detector, malicious, args.poison_scale
        )
    else:
        recovery_config = RecoveryConfig(checkpoint_threshold=args.checkpoint_threshold)
        station = run_recovery_mode(
            config,
            DEFAULT_COALITIONS,
            detector,
            malicious,
            args.poison_scale,
            recovery_config=recovery_config,
        )

    if station is not None:
        print_simulation_summary(
            station,
            args.mode,
            DEFAULT_COALITIONS,
            malicious,
            args.rounds,
        )


if __name__ == "__main__":
    main()
