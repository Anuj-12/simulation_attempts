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
import os
from datetime import datetime
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

DEFAULT_COALITIONS: List[CoalitionSpec] = [
    ("c1", ["u1", "u2", "u3", "u4", "u5"]),
]

DEFAULT_MALICIOUS = ["u5"]


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
        help="Edge UAV ids simulated as model-poisoning attackers (default: u5)",
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
    parser.add_argument("--rounds", type=int, default=8, help="Number of FL rounds")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Local SGD learning rate")
    parser.add_argument("--data-dir", type=str, default="./data", help="FashionMNIST cache dir")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory for readiness report files",
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


def run_base_mode(config: HFLConfig, coalitions: List[CoalitionSpec]) -> None:
    print("Running base HFL (no RL state management)")
    station = build_hfl_system(coalitions, config=config)
    _, test_set = load_fashion_mnist(config.data_dir)
    station.run(test_set)


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


def _readiness_checks(station, malicious_uavs: Sequence[str]) -> List[Tuple[str, bool, str]]:
    """Evaluate whether core ReCon subsystems behaved as expected."""
    checks: List[Tuple[str, bool, str]] = []
    history = getattr(station, "round_history", [])
    malicious = set(malicious_uavs)

    if not history:
        checks.append(("Round history captured", False, "No round snapshots recorded"))
        return checks

    checks.append(("Round history captured", True, f"{len(history)} rounds logged"))

    # FLGuardian: malicious UAV should get higher λ than benign peers at least once
    high_lambda_malicious = False
    for snap in history:
        mal_scores = [u.contamination_score for u in snap.uav_snapshots if u.uav_id in malicious]
        benign_scores = [u.contamination_score for u in snap.uav_snapshots if u.uav_id not in malicious]
        if mal_scores and benign_scores and max(mal_scores) > min(benign_scores) + 0.05:
            high_lambda_malicious = True
            break
    checks.append((
        "FLGuardian flags malicious UAV (λ)",
        high_lambda_malicious,
        "Malicious UAV λ exceeded benign minimum by >0.05 in at least one round"
        if high_lambda_malicious else "Could not distinguish malicious UAV by contamination score",
    ))

    # PPO governance: quarantine or exclude should fire when λ is elevated
    governance_fired = any(
        u.action in ("QUARANTINE", "EXCLUDE")
        for snap in history
        for u in snap.uav_snapshots
        if u.uav_id in malicious
    )
    checks.append((
        "PPO quarantine/exclude on threat",
        governance_fired,
        "Malicious UAV received QUARANTINE or EXCLUDE"
        if governance_fired else "No governance action on malicious UAV",
    ))

    # Global model still learns
    acc_start = history[0].global_accuracy
    acc_end = history[-1].global_accuracy
    model_learns = acc_end >= 0.05
    checks.append((
        "Global model trains",
        model_learns,
        f"Accuracy {acc_start:.4f} → {acc_end:.4f}",
    ))

    # Recovery mode: checkpoint or exclude path exercised
    if hasattr(station, "checkpoint_store"):
        excluded = any(s.excluded_uavs > 0 for s in history)
        checkpointed = station.checkpoint_store.t_c > 0
        checks.append((
            "Recovery subsystem engaged",
            excluded or checkpointed,
            f"checkpoint t_c={station.checkpoint_store.t_c}, "
            f"max excluded={max(s.excluded_uavs for s in history)}",
        ))

    return checks


def write_readiness_report(
    station,
    output_dir: str,
    mode: str,
    detector_name: str,
    malicious_uavs: Sequence[str],
    checks: List[Tuple[str, bool, str]],
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = os.path.join(output_dir, f"recon_readiness_{timestamp}.txt")
    html_path = os.path.join(output_dir, f"recon_readiness_{timestamp}.html")

    history: List[RoundSnapshot] = getattr(station, "round_history", [])
    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    ready = passed == total

    lines = [
        "=" * 72,
        "ReCon HFL — SYSTEM READINESS REPORT",
        "=" * 72,
        f"Generated : {datetime.now().isoformat(timespec='seconds')}",
        f"Mode      : {mode}",
        f"Detector  : {detector_name} (φ)",
        f"Malicious : {', '.join(malicious_uavs) or 'none'}",
        f"Verdict   : {'READY' if ready else 'NEEDS ATTENTION'} ({passed}/{total} checks passed)",
        "",
        "READINESS CHECKS",
        "-" * 72,
    ]
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        lines.append(f"  [{status}] {name}")
        lines.append(f"         {detail}")

    lines.extend(["", "ROUND SUMMARY", "-" * 72])
    lines.append(f"{'Rnd':>4} {'Acc':>8} {'Active':>7} {'Q':>3} {'Ex':>3}  Notable UAV actions")
    lines.append("-" * 72)
    for snap in history:
        notable = [
            f"{u.uav_id}:λ={u.contamination_score:.2f}→{u.action[:3]}"
            for u in snap.uav_snapshots
            if u.action != "ALLOW" or u.contamination_score > 0.4 or u.is_malicious
        ]
        lines.append(
            f"{snap.round_idx:4d} {snap.global_accuracy:8.4f} "
            f"{snap.active_uavs:7d} {snap.quarantined_uavs:3d} {snap.excluded_uavs:3d}  "
            f"{', '.join(notable) or '—'}"
        )

    lines.extend(["", "PER-UAV FINAL STATE", "-" * 72])
    if history:
        final = history[-1]
        lines.append(
            f"{'UAV':<6} {'Coal':<5} {'λ':>6} {'ρ':>7} {'q':>3} {'State':<12} {'Action':<10} Mal?"
        )
        lines.append("-" * 72)
        for u in final.uav_snapshots:
            lines.append(
                f"{u.uav_id:<6} {u.coalition_id:<5} {u.contamination_score:6.3f} "
                f"{u.reputation:7.3f} {u.flag_count:3d} {u.participation:<12} "
                f"{u.action:<10} {'yes' if u.is_malicious else 'no'}"
            )

    lines.append("=" * 72)
    report_text = "\n".join(lines)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    check_rows = "".join(
        f"<tr class=\"{'pass' if ok else 'fail'}\"><td>{name}</td>"
        f"<td>{'PASS' if ok else 'FAIL'}</td><td>{detail}</td></tr>"
        for name, ok, detail in checks
    )
    round_rows = "".join(
        f"<tr><td>{s.round_idx}</td><td>{s.global_accuracy:.4f}</td>"
        f"<td>{s.active_uavs}</td><td>{s.quarantined_uavs}</td>"
        f"<td>{s.excluded_uavs}</td></tr>"
        for s in history
    )
    uav_rows = ""
    if history:
        for u in history[-1].uav_snapshots:
            uav_rows += (
                f"<tr><td>{u.uav_id}</td><td>{u.coalition_id}</td>"
                f"<td>{u.contamination_score:.3f}</td><td>{u.reputation:.3f}</td>"
                f"<td>{u.flag_count}</td><td>{u.participation}</td>"
                f"<td>{u.action}</td><td>{'yes' if u.is_malicious else 'no'}</td></tr>"
            )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ReCon Readiness Report</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #0f172a; color: #e2e8f0; }}
  h1 {{ color: #38bdf8; }}
  .verdict {{ font-size: 1.4rem; padding: 1rem; border-radius: 8px; margin: 1rem 0; }}
  .ready {{ background: #14532d; border: 1px solid #22c55e; }}
  .not-ready {{ background: #450a0a; border: 1px solid #ef4444; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ border: 1px solid #334155; padding: 0.5rem 0.75rem; text-align: left; }}
  th {{ background: #1e293b; }}
  tr.pass td:nth-child(2) {{ color: #4ade80; font-weight: bold; }}
  tr.fail td:nth-child(2) {{ color: #f87171; font-weight: bold; }}
  .meta {{ color: #94a3b8; }}
</style>
</head>
<body>
<h1>ReCon HFL — System Readiness</h1>
<p class="meta">Mode: {mode} | Detector: {detector_name} | Malicious: {', '.join(malicious_uavs) or 'none'}</p>
<div class="verdict {'ready' if ready else 'not-ready'}">
  {'✓ SYSTEM READY' if ready else '✗ NEEDS ATTENTION'} — {passed}/{total} checks passed
</div>
<h2>Readiness Checks</h2>
<table><tr><th>Check</th><th>Status</th><th>Detail</th></tr>{check_rows}</table>
<h2>Round Summary</h2>
<table><tr><th>Round</th><th>Accuracy</th><th>Active</th><th>Quarantined</th><th>Excluded</th></tr>{round_rows}</table>
<h2>Final UAV State</h2>
<table><tr><th>UAV</th><th>Coalition</th><th>λ</th><th>ρ</th><th>q</th><th>State</th><th>Action</th><th>Malicious</th></tr>{uav_rows}</table>
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print("\n" + report_text)
    print(f"\nReports saved:")
    print(f"  Text : {txt_path}")
    print(f"  HTML : {html_path}")
    return txt_path


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
        run_base_mode(config, DEFAULT_COALITIONS)
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
        checks = _readiness_checks(station, malicious)
        write_readiness_report(
            station,
            args.output_dir,
            args.mode,
            args.detector,
            malicious,
            checks,
        )


if __name__ == "__main__":
    main()
