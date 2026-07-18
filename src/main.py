"""
Entry point for the ReCon HFL simulation.

Dependency graph (acyclic):
  hfl_common  ->  hfl_base  ->  hfl_rl  ->  hfl_recovery
                      ^            ^               ^
                      |            |               |
                 flguardian_det ----+---------------+
                      |             |
                   fltrust ---------+
                      |
                      +----------- main -----------+

ReCon pipeline (ReCon.tex):
  Edge UAVs train locally -> Fog runs contamination detection (phi -
  FLGuardian or FLTrust, both black-box per ReCon.tex line 180) ->
  reputation update -> PPO {Allow, Quarantine, Exclude} -> hierarchical
  aggregation -> checkpoint rollback on Exclude (recovery mode)
"""

from __future__ import annotations

import argparse
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# Windows' console defaults to a legacy codepage (cp1252) that can't encode
# the Greek-letter symbols (phi, lambda, rho, kappa, Delta, etc.) printed
# throughout this codebase's logs - this crashes specifically when
# redirecting stdout to a file (e.g. `python main.py > logs.txt`), since the
# file handle inherits that same encoding. Forcing UTF-8 here fixes it
# regardless of the user's terminal/locale settings, rather than relying on
# them to set PYTHONIOENCODING or run `chcp 65001` every session.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

from flguardian_det import build_flguardian_hfl_adapter
from fltrust import build_fltrust_hfl_adapter
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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ReCon: reputation-aware HFL with pluggable contamination detection "
        "(FLGuardian or FLTrust, per ReCon.tex's phi black-box interface)"
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
        choices=["flguardian", "fltrust", "none"],
        default="flguardian",
        help="Contamination detector φ (default: flguardian). Both flguardian and "
        "fltrust are drop-in per ReCon.tex line 180 ('φ is essentially a black box').",
    )
    parser.add_argument(
        "--fltrust-root-size",
        type=int,
        default=100,
        help="|D0|: size of FLTrust's server-held root dataset (default: 100, "
        "the paper's default across all evaluated datasets, Table I). Only used "
        "when --detector fltrust.",
    )
    parser.add_argument(
        "--fltrust-no-magnitude-signal",
        action="store_true",
        help="Use FLTrust's literal Eq. 2 trust score only (1 - ReLU(cosine)) with "
        "no magnitude-penalty term. Default (off) adds a magnitude-deviation signal "
        "since ReCon's poisoning attack (apply_poison) is a pure magnitude scaling "
        "of an otherwise-honest gradient, which a direction-only cosine trust score "
        "is blind to by construction - see fltrust.py's FLTrustDetector docstring. "
        "Only used when --detector fltrust.",
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
    parser.add_argument(
        "--paper-literal-checkpoint-order",
        action="store_true",
        help="Use Algorithm 2's literal ordering (checkpoint unconditionally every "
        "round, before checking for exclusion/rollback), instead of the default "
        "improved ordering (checkpoint after any same-round rollback). Only "
        "affects --mode recovery. See hfl_recovery.py's module docstring for why "
        "the default deviates from Algorithm 2 as written.",
    )
    parser.add_argument(
        "--no-flag-reset",
        action="store_true",
        help="Disable the one-time q (flag_count) reset that otherwise fires once, "
        "after the primer period, wiping every UAV's accumulated quarantine flags. "
        "Off by default means falling back to the paper's literal never-decreasing "
        "q_j (Eq. 7/28 have no reset term).",
    )
    parser.add_argument(
        "--primer-fraction",
        type=float,
        default=None,
        help="Convenience flag: sets BOTH --exclude-warmup-fraction and "
        "--flag-reset-fraction to this value at once. These are independent "
        "RLConfig fields for a reason (you may want EXCLUDE to unlock at a "
        "different point than when q resets) - passing either flag individually "
        "overrides this for that specific field. Without this or the individual "
        "flags, each defaults to 0.10 independently.",
    )
    parser.add_argument(
        "--delta-psi-scale",
        type=float,
        default=None,
        help="Multiplier applied to Delta_psi (the accuracy-consequence term in "
        "Eq. 26) before it's summed into the reward. The paper's Eq. 26 is an "
        "unweighted sum with no coefficient specified for any term; default 10.0 "
        "here was computed from one run's observed magnitudes (Delta_psi's "
        "natural scale is ~10x smaller than rho_t's and ~170x smaller than a "
        "typical Quarantine/Exclude penalty, so it rarely had enough weight to "
        "influence which action wins) - re-verify against your own runs rather "
        "than treating 10.0 as settled.",
    )
    parser.add_argument(
        "--exclude-warmup-fraction",
        type=float,
        default=None,
        help="Fraction of --rounds during which EXCLUDE stays shadow-only (or fully "
        "masked out if --no-shadow-exclude) rather than being really applied "
        "(default: 0.10, i.e. 10%% of the run, unless --primer-fraction is set). "
        "THIS is the flag that actually controls when EXCLUDE goes live for real - "
        "--flag-reset-fraction only controls the separate q-reset/quarantine-release "
        "event and does NOT extend real-exclusion protection on its own.",
    )
    parser.add_argument(
        "--flag-reset-fraction",
        type=float,
        default=None,
        help="Fraction of --rounds treated as the primer period before the one-time "
        "q reset fires (default: 0.10, i.e. 10%% of the run, unless --primer-fraction "
        "is set). Independent of --rounds itself - e.g. 0.10 means round 10 on a "
        "100-round run but round 100 on a 1000-round run. Ignored if --no-flag-reset "
        "is set. NOTE: does not affect when EXCLUDE goes live for real - see "
        "--exclude-warmup-fraction for that.",
    )
    parser.add_argument(
        "--no-quarantine-release",
        action="store_true",
        help="Disable releasing every currently-QUARANTINED UAV back to ACTIVE at "
        "the same primer-end event as the q reset (default: released). Does not "
        "affect EXCLUDED UAVs, which remain terminal. Ignored if --no-flag-reset "
        "is set, since this fires at the same event.",
    )
    parser.add_argument(
        "--no-shadow-exclude",
        action="store_true",
        help="Disable shadow-exclude during the primer period (default: enabled). "
        "By default, EXCLUDE can be sampled and rewarded during the primer period "
        "like any other action, but is not actually applied to the UAV's "
        "participation state until exclude_unlocked - this lets the policy learn "
        "when excluding would pay off without any real exclusion risk while it's "
        "still undertrained. Disabling falls back to masking EXCLUDE out of the "
        "action space entirely during the primer period (the old behavior), which "
        "gives EXCLUDE's logit no chance at positive reinforcement before it "
        "unlocks.",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Global random seed (torch + Python random). Controls model "
        "initialization, PPO action sampling, DataLoader shuffling, and "
        "GTG-Shapley's permutation sampling. Does NOT affect the data "
        "partition (hfl_common.partition_dataset) or FLTrust's root dataset "
        "sample (fltrust.sample_root_dataset) - both already use their own "
        "fixed seeds, independent of this one, so varying --seed compares "
        "training/policy randomness on an identical data split rather than "
        "conflating that with a different split each time. Default (unset) "
        "is fully unseeded, same as before this flag existed.",
    )
    return parser.parse_args(argv)


def make_hfl_config(args: argparse.Namespace) -> HFLConfig:
    return HFLConfig(
        data_dir=args.data_dir,
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )


def make_rl_config(args: argparse.Namespace) -> RLConfig:
    """Overrides the flag-reset/quarantine-release/shadow-exclude/exclude-warmup
    fields from CLI args; everything else (reputation_lr, entropy_coef, etc.)
    still uses RLConfig's dataclass defaults, unchanged by this function.

    Precedence for the two primer-length fields: an explicitly-set individual
    flag (--exclude-warmup-fraction / --flag-reset-fraction) always wins;
    otherwise --primer-fraction sets both; otherwise each falls back to
    RLConfig's own default (0.10) independently. These two fields are
    deliberately independent - --primer-fraction is a convenience for the
    common case of wanting them equal, not a merge of the fields themselves.
    """
    default_kwargs = {}
    exclude_warmup = (
        args.exclude_warmup_fraction
        if args.exclude_warmup_fraction is not None
        else args.primer_fraction
    )
    if exclude_warmup is not None:
        default_kwargs["exclude_warmup_fraction"] = exclude_warmup

    flag_reset = args.flag_reset_fraction if args.flag_reset_fraction is not None else args.primer_fraction
    if flag_reset is not None:
        default_kwargs["flag_reset_fraction"] = flag_reset

    if args.delta_psi_scale is not None:
        default_kwargs["delta_psi_scale"] = args.delta_psi_scale

    return RLConfig(
        reset_flags_after_primer=not args.no_flag_reset,
        release_quarantine_after_primer=not args.no_quarantine_release,
        shadow_exclude_during_primer=not args.no_shadow_exclude,
        **default_kwargs,
    )


def make_contamination_detector(args: argparse.Namespace, config: HFLConfig):
    if args.detector == "none":
        return zero_contamination_detector
    if args.detector == "fltrust":
        # Passing `config` (not just device) so the server's root-dataset
        # training uses the same batch_size/lr/local_epochs as the edge UAVs
        # (Algorithm 2 uses the same b, beta, Rl for clients and server) -
        # see build_fltrust_hfl_adapter's docstring in fltrust.py.
        return build_fltrust_hfl_adapter(
            config=config,
            root_size=args.fltrust_root_size,
            include_magnitude_signal=not args.fltrust_no_magnitude_signal,
        )
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
    warm_start_state_dict: Optional[dict] = None,
) -> HFLRLStation:
    print("Running HFL with PPO-based UAV state management + contamination detection (φ)")
    station = build_hfl_rl_system(
        coalitions,
        config=config,
        rl_config=rl_config or RLConfig(),
        contamination_detector=detector,
        malicious_uavs=malicious_uavs,
        poison_scale=poison_scale,
    )
    if warm_start_state_dict is not None:
        station.ppo.network.load_state_dict(warm_start_state_dict)
        print("[Warm-start] Loaded policy network weights from a previous run.")
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
    warm_start_state_dict: Optional[dict] = None,
):
    print("Running full ReCon: contamination detection (φ) + PPO governance + checkpoint recovery")
    station = build_hfl_recovery_system(
        coalitions,
        config=config,
        rl_config=rl_config or RLConfig(),
        recovery_config=recovery_config,
        contamination_detector=detector,
        malicious_uavs=malicious_uavs,
        poison_scale=poison_scale,
    )
    if warm_start_state_dict is not None:
        station.ppo.network.load_state_dict(warm_start_state_dict)
        print("[Warm-start] Loaded policy network weights from a previous run.")
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


def compute_simulation_summary(
    station,
    mode: str,
    coalitions: List[CoalitionSpec],
    malicious_uavs: Sequence[str],
    num_rounds: int,
) -> Dict:
    """Compute the same statistics print_simulation_summary prints, as a
    plain dict instead of printed text - so a multi-seed runner (run_seeds.py)
    can collect and aggregate these across runs without parsing printed
    output."""
    history: List[RoundSnapshot] = getattr(station, "round_history", []) or []
    total_coalitions = len(coalitions)
    total_uavs = sum(len(members) for _, members in coalitions)
    malicious_set = set(malicious_uavs)
    total_malicious = len(malicious_set)

    base = {
        "mode": mode,
        "total_rounds": num_rounds,
        "total_coalitions": total_coalitions,
        "total_uavs": total_uavs,
        "total_malicious": total_malicious,
        "has_history": bool(history),
    }
    if not history:
        return base

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

    base.update({
        "final_active": final.active_uavs,
        "final_quarantined": final.quarantined_uavs,
        "final_excluded": final.excluded_uavs,
        "avg_accuracy": avg_accuracy,
        "max_accuracy": max_accuracy,
        "avg_reputation": avg_reputation,
        "avg_contamination": avg_contamination,
        "checkpoints_created": checkpoints_created,
        "rollbacks": rollbacks,
        "malicious_quarantined": mal_quarantined,
        "malicious_excluded": mal_excluded,
        "malicious_active": mal_active,
        "benign_quarantined": ben_quarantined,
        "benign_excluded": ben_excluded,
    })
    return base


def print_simulation_summary(
    station,
    mode: str,
    coalitions: List[CoalitionSpec],
    malicious_uavs: Sequence[str],
    num_rounds: int,
) -> Dict:
    """Print a concise terminal-only execution summary (no file output).
    Returns the same dict compute_simulation_summary produces, in case the
    caller wants it (main() ignores the return value; run_seeds.py uses
    compute_simulation_summary directly instead, to run silently)."""
    summary = compute_simulation_summary(station, mode, coalitions, malicious_uavs, num_rounds)

    print("\n====================================")
    print("Simulation Summary")
    print("====================================")
    print(f"Total rounds        : {summary['total_rounds']}")
    print(f"Total coalitions     : {summary['total_coalitions']}")
    print(f"Total UAVs           : {summary['total_uavs']}")
    print(f"Total malicious UAVs : {summary['total_malicious']}")

    if not summary["has_history"]:
        print("\nNo round history was recorded for this run (mode="
              f"{mode!r}); RL/reputation/checkpoint stats are unavailable.")
        print("====================================")
        return summary

    print("\nFinal counts:")
    print(f"  Active      : {summary['final_active']}")
    print(f"  Quarantined : {summary['final_quarantined']}")
    print(f"  Excluded    : {summary['final_excluded']}")

    print(f"\nAverage global accuracy    : {summary['avg_accuracy']:.4f}")
    print(f"Highest global accuracy    : {summary['max_accuracy']:.4f}")
    print(f"Average reputation         : {summary['avg_reputation']:.4f}")
    print(f"Average contamination score: {summary['avg_contamination']:.4f}")

    print(f"\nNumber of checkpoint rollbacks : {summary['rollbacks']}")
    print(f"Number of checkpoints created  : {summary['checkpoints_created']}")

    print("\nMalicious UAV statistics:")
    print(f"  Quarantined  : {summary['malicious_quarantined']}")
    print(f"  Excluded     : {summary['malicious_excluded']}")
    print(f"  Still active : {summary['malicious_active']}")

    print("\nBenign UAV statistics:")
    print(f"  Quarantined : {summary['benign_quarantined']}")
    print(f"  Excluded    : {summary['benign_excluded']}")
    print("====================================")
    return summary


def apply_seed(seed: Optional[int]) -> None:
    """Seed torch + Python's random module. See --seed's help text for what
    this does and does not cover (notably: not partition_dataset's or
    sample_root_dataset's own fixed internal seeds)."""
    if seed is None:
        return
    torch.manual_seed(seed)
    random.seed(seed)


def run_simulation(
    args: argparse.Namespace,
    verbose: bool = True,
    warm_start_state_dict: Optional[dict] = None,
) -> Dict:
    """Core simulation runner, extracted from main() so it's directly
    reusable from run_seeds.py without CLI/subprocess plumbing. Returns the
    same summary dict compute_simulation_summary produces (empty dict if
    the run produced no station, e.g. an unrecognized mode - shouldn't
    happen given argparse's choices= constraint, but kept defensive).

    warm_start_state_dict: if given (and mode is "rl" or "recovery"), loaded
    into the PPO policy network BEFORE training starts, instead of the
    network's normal random initialization. Everything else (UAV models,
    reputation, participation state) still starts fresh regardless - only
    the policy network itself carries forward. Ignored for mode="base"
    (no PPO there). The returned dict always includes the just-trained
    network's weights under "_ppo_state_dict" (None for mode="base"), so a
    caller (run_seeds.py) can pass this run's result straight into the next
    call's warm_start_state_dict to chain training across seeds.

    verbose=True (default, used by main()) prints the run banner and full
    summary, exactly as before this function existed. verbose=False (used
    by run_seeds.py for multi-seed runs) suppresses both, but NOT the
    per-round training/governance logs emitted from within hfl_rl.py/
    hfl_recovery.py themselves - those aren't gated by this flag, so a
    multi-seed run will still be verbose per-round; only the banner/summary
    printing here is suppressed.
    """
    apply_seed(args.seed)
    config = make_hfl_config(args)
    rl_config = make_rl_config(args)
    detector = make_contamination_detector(args, config)
    malicious = args.malicious if args.mode != "base" else []

    if verbose:
        print("=" * 72)
        print("ReCon — Reputation-Aware Contamination Governance for UAV-HFL")
        print("=" * 72)
        print(f"Coalitions : {DEFAULT_COALITIONS}")
        print(f"Detector   : {args.detector}")
        print(f"Malicious  : {malicious or 'none'}")
        print(f"Rounds     : {args.rounds}")
        print(f"Seed       : {args.seed if args.seed is not None else 'unseeded'}")
        if warm_start_state_dict is not None:
            print("Warm-start : loaded policy weights from a previous run")
        print("=" * 72)

    station = None
    if args.mode == "base":
        station = run_base_mode(config, DEFAULT_COALITIONS)
    elif args.mode == "rl":
        station = run_rl_mode(
            config, DEFAULT_COALITIONS, detector, malicious, args.poison_scale,
            rl_config=rl_config, warm_start_state_dict=warm_start_state_dict,
        )
    else:
        recovery_config = RecoveryConfig(
            checkpoint_threshold=args.checkpoint_threshold,
            checkpoint_after_rollback=not args.paper_literal_checkpoint_order,
        )
        station = run_recovery_mode(
            config,
            DEFAULT_COALITIONS,
            detector,
            malicious,
            args.poison_scale,
            rl_config=rl_config,
            recovery_config=recovery_config,
            warm_start_state_dict=warm_start_state_dict,
        )

    if station is None:
        return {}

    if verbose:
        summary = print_simulation_summary(
            station, args.mode, DEFAULT_COALITIONS, malicious, args.rounds,
        )
    else:
        summary = compute_simulation_summary(
            station, args.mode, DEFAULT_COALITIONS, malicious, args.rounds,
        )

    # Attach the just-trained policy weights so a caller (run_seeds.py) can
    # warm-start the next run from this one. None for mode="base" (no PPO).
    ppo = getattr(station, "ppo", None)
    summary["_ppo_state_dict"] = ppo.network.state_dict() if ppo is not None else None
    return summary


def main() -> None:
    args = parse_args()
    run_simulation(args, verbose=True)


if __name__ == "__main__":
    main()
