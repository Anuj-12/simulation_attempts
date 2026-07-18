"""
Run the ReCon simulation across multiple seeds and report mean/std/min/max
for each summary metric, instead of trusting a single run's outcome.

Motivation: nothing in this codebase was seeded before this script existed
(model init, PPO action sampling, DataLoader shuffling, GTG-Shapley's
permutation sampling were all uncontrolled - see main.py's --seed help text
for exactly what apply_seed() does and doesn't cover). Several rounds of
this debugging session amounted to "is this outcome real or a fluke" -
questions that a single run literally cannot answer. This script answers
them by running the identical configuration N times, varying only --seed.

Usage: pass any of main.py's own flags through unchanged, plus --seeds:
    python run_seeds.py --seeds 1 2 3 4 5 --mode recovery \
        --poison-scale 10 --delta-psi-scale 10 --rounds 100

Any --seed you also pass in the remaining args is ignored - one is injected
per iteration from --seeds instead. All other flags (--mode, --detector,
--poison-scale, --delta-psi-scale, --rounds, etc.) are identical across
every seed in the run, by design - the whole point is to hold configuration
fixed and vary only the random seed.

This intentionally reuses main.py's own parse_args()/run_simulation()
rather than duplicating CLI parsing or simulation logic, so this script
never drifts out of sync with main.py's actual behavior.
"""

from __future__ import annotations

import argparse
import statistics
from typing import Dict, List, Tuple

import main as recon_main

# Metrics from compute_simulation_summary worth aggregating. Excludes
# non-numeric/context fields (mode, total_rounds, total_coalitions,
# total_uavs, total_malicious, has_history) which are identical across
# seeds by construction (same config every run) and not meaningful to
# average.
NUMERIC_METRICS = [
    "final_active",
    "final_quarantined",
    "final_excluded",
    "avg_accuracy",
    "max_accuracy",
    "avg_reputation",
    "avg_contamination",
    "checkpoints_created",
    "rollbacks",
    "malicious_quarantined",
    "malicious_excluded",
    "malicious_active",
    "benign_quarantined",
    "benign_excluded",
]


def parse_seed_runner_args(argv: List[str] = None) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Run the ReCon simulation across multiple seeds and aggregate results.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Seeds to run, one full simulation per seed (default: 1 2 3 4 5).",
    )
    parser.add_argument(
        "--quiet-runs",
        action="store_true",
        help="Suppress each run's own banner/final-summary printing (only the aggregate "
        "table at the end is shown). Does NOT suppress the per-round training/governance "
        "logs from hfl_rl.py/hfl_recovery.py themselves - those print regardless, so a "
        "multi-seed run is still verbose per-round unless you redirect stdout.",
    )
    parser.add_argument(
        "--warm-start",
        action="store_true",
        help="Chain the PPO policy network's trained weights from each seed into the next "
        "one, instead of every seed starting from a fresh random policy (the default). "
        "UAV models, reputation, and participation state still reset fresh each seed "
        "regardless - only the policy network itself carries forward. This changes what "
        "--seeds measures: WITHOUT this flag, each run is an independent trial of the same "
        "configuration (use this to check whether a result is stable/typical or a fluke - "
        "the original purpose of this script). WITH this flag, seeds become sequential "
        "training extensions of one continuously-improving policy (use this to keep "
        "training a policy longer than one run, while still getting fresh random UAV/data "
        "conditions each leg). Ignored for --mode base (no PPO there).",
    )
    # parse_known_args so every other flag (--mode, --poison-scale, etc.)
    # passes through untouched to main.parse_args() for each seed.
    return parser.parse_known_args(argv)


def run_all_seeds(runner_args: argparse.Namespace, remaining_argv: List[str]) -> List[Dict]:
    results: List[Dict] = []
    warm_start_state_dict = None
    for seed in runner_args.seeds:
        argv = list(remaining_argv) + ["--seed", str(seed)]
        args = recon_main.parse_args(argv)
        print(f"\n{'#' * 72}\n# Running seed={seed}"
              f"{' (warm-started from previous seed)' if warm_start_state_dict is not None else ''}"
              f"\n{'#' * 72}")
        summary = recon_main.run_simulation(
            args,
            verbose=not runner_args.quiet_runs,
            warm_start_state_dict=warm_start_state_dict if runner_args.warm_start else None,
        )
        summary["seed"] = seed
        results.append(summary)
        if runner_args.warm_start:
            warm_start_state_dict = summary.get("_ppo_state_dict")
            if warm_start_state_dict is None:
                print(f"[Warm-start] seed={seed} produced no PPO network (mode=base?) - "
                      "next seed will start fresh instead of warm-started.")
    return results


def print_aggregate_table(results: List[Dict]) -> None:
    print("\n" + "=" * 72)
    seeds_run = [r.get("seed") for r in results]
    print(f"Aggregate results across {len(results)} seeds: {seeds_run}")
    print("=" * 72)

    missing_history = [r["seed"] for r in results if not r.get("has_history")]
    if missing_history:
        print(f"WARNING: no round history recorded for seed(s) {missing_history} - "
              "excluded from aggregation below.")
    usable = [r for r in results if r.get("has_history")]
    if not usable:
        print("No usable results to aggregate.")
        return

    header = f"{'metric':<24} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}"
    print(header)
    print("-" * len(header))
    for metric in NUMERIC_METRICS:
        values = [r[metric] for r in usable if metric in r]
        if not values:
            continue
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"{metric:<24} {mean:>10.4f} {std:>10.4f} {min(values):>10.4f} {max(values):>10.4f}")
    print("=" * 72)
    print(
        "A small std relative to the mean means this outcome is stable across random "
        "seeds; a large one means a single run's result (like the ones we've been "
        "debugging from) isn't representative on its own."
    )


def main() -> None:
    runner_args, remaining_argv = parse_seed_runner_args()
    results = run_all_seeds(runner_args, remaining_argv)
    print_aggregate_table(results)


if __name__ == "__main__":
    main()
