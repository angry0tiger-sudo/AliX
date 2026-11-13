"""Deterministic synthetic-only fixture and contract runner for the four-arm study."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import random
import struct
from typing import Sequence

from research.ai_ensemble.four_arm_gates import (
    ArmGateEvidence,
    TRIAL_IDS,
    arm_passes,
    dsr_sensitivity_passes,
    fold_stability_passes,
    select_winner,
)

FIXED_SYNTHETIC_SEED = 20261113
DEFAULT_SYNTHETIC_BAR_COUNT = 1200


@dataclass(frozen=True, slots=True)
class SyntheticOHLCVBar:
    """In-memory synthetic bar; values are never serialized into the report."""

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


def build_synthetic_ohlcv(
    seed: int = FIXED_SYNTHETIC_SEED,
    bar_count: int = DEFAULT_SYNTHETIC_BAR_COUNT,
) -> tuple[SyntheticOHLCVBar, ...]:
    """Build a deterministic, in-memory fixture with no market-data dependency."""
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("synthetic_seed_invalid")
    if isinstance(bar_count, bool) or bar_count < 1:
        raise ValueError("synthetic_bar_count_invalid")

    generator = random.Random(seed)
    previous_close = 1000.0
    bars: list[SyntheticOHLCVBar] = []
    for index in range(bar_count):
        open_value = previous_close
        close_value = open_value * (1.0 + generator.uniform(-0.0015, 0.0015))
        high_value = max(open_value, close_value) * (
            1.0 + generator.uniform(0.0, 0.0008)
        )
        low_value = min(open_value, close_value) * (
            1.0 - generator.uniform(0.0, 0.0008)
        )
        volume_value = generator.uniform(1.0, 100.0)
        bars.append(
            SyntheticOHLCVBar(
                timestamp=index * 3600,
                open=open_value,
                high=high_value,
                low=low_value,
                close=close_value,
                volume=volume_value,
            )
        )
        previous_close = close_value
    return tuple(bars)


def _fixture_fingerprint(bars: Sequence[SyntheticOHLCVBar]) -> str:
    digest = sha256()
    for bar in bars:
        digest.update(
            struct.pack(
                ">qddddd",
                bar.timestamp,
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume,
            )
        )
    return digest.hexdigest()


def _sample_evidence(
    arm_id: str,
    dsr: float,
    eligible: bool = True,
) -> ArmGateEvidence:
    net_pf = 1.10 if eligible else 1.09
    return ArmGateEvidence(
        arm_id=arm_id,
        net_profit_factor_50bps=net_pf,
        net_profit_factor_90bps=net_pf,
        completed_trades=100,
        max_drawdown=0.15,
        dsr=dsr,
        dsr_sensitivity=(0.01, 0.02, 0.03),
        fold_passes=(True, True, True),
    )


def _run_gate_contract_checks() -> dict[str, bool]:
    first = _sample_evidence(TRIAL_IDS[0], 0.10)
    second = _sample_evidence(TRIAL_IDS[1], 0.20)
    failing = _sample_evidence(TRIAL_IDS[2], 0.30, eligible=False)
    return {
        "fold_stability_pass_boundary": fold_stability_passes(
            (True, True, True)
        ),
        "fold_stability_fail_boundary": not fold_stability_passes(
            (True, True, False)
        ),
        "dsr_sensitivity_pass_boundary": dsr_sensitivity_passes(
            (0.01, 0.02, 0.03)
        ),
        "dsr_sensitivity_fail_boundary": not dsr_sensitivity_passes(
            (0.01, 0.0, 0.03)
        ),
        "arm_gate_pass_boundary": arm_passes(first),
        "arm_gate_fail_boundary": not arm_passes(failing),
        "winner_selection_highest_dsr": (
            select_winner((first, second)) == TRIAL_IDS[1]
        ),
        "winner_selection_no_passing_arm": select_winner((failing,)) is None,
    }


def run_mock_validation(
    output_dir: Path,
    seed: int = FIXED_SYNTHETIC_SEED,
    bar_count: int = DEFAULT_SYNTHETIC_BAR_COUNT,
) -> dict[str, object]:
    """Run fixture and gate-contract checks; never train, backtest, or read market data."""
    bars = build_synthetic_ohlcv(seed=seed, bar_count=bar_count)
    checks = _run_gate_contract_checks()
    if not all(checks.values()):
        raise RuntimeError("four_arm_mock_contract_check_failed")

    report: dict[str, object] = {
        "schema_version": "alix.four-arm.mock-report.v1",
        "mode": "MOCK_ONLY",
        "synthetic_seed": seed,
        "synthetic_bar_count": len(bars),
        "synthetic_fixture_sha256": _fixture_fingerprint(bars),
        "preregistered_trial_ids_only": list(TRIAL_IDS),
        "trial_evaluation_performed": False,
        "gate_contract_checks": checks,
        "gate_contract_checks_passed": True,
        "real_market_data_accessed": False,
        "holdout_accessed": False,
        "model_weights_loaded": False,
        "training_or_fit_performed": False,
        "backtest_performed": False,
        "economic_evaluation_performed": False,
        "network_requests": 0,
        "private_api_calls": 0,
        "real_order_attempts": 0,
        "output_contains_ohlcv_values": False,
        "execution_authorization_required_for_real_research": True,
        "execution_authorization_present": False,
        "safety_invariants": {
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "ALIX_DRY_RUN": True,
            "ALIX_ALLOW_LIVE_TRADING": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "four-arm-mock-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run mock-only four-arm contracts.")
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args(argv)
    report = run_mock_validation(Path(arguments.output_dir))
    if report["gate_contract_checks_passed"] is not True:
        return 1
    print("FOUR_ARM_MOCK_VALIDATION=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
