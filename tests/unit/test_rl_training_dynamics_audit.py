"""Tests for the read-only RL training dynamics audit."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from scripts.audit_rl_training_dynamics import (
    EXPECTED_SOURCE_COMMIT,
    EXPECTED_SOURCE_RUN_ID,
    _entropy_summary,
    _nearest_snapshot,
    _validate_source,
    parse_training_logs,
)


def _source_report() -> dict[str, Any]:
    return {
        "schema_version": "alix.rl-trading-agent-report.v10",
        "commit_sha": EXPECTED_SOURCE_COMMIT,
        "run_mode": "historical_simulation_only",
        "model": {
            "algorithm": "RecurrentPPO",
            "total_timesteps_trained": 1_000_448,
            "model_sha256": "a" * 64,
        },
        "safety_invariants": {
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "private_api_calls": 0,
            "orders_or_execution_api_used": False,
            "real_order_attempts": 0,
        },
        "production_promotion_allowed": False,
    }


def _source_run() -> dict[str, Any]:
    return {
        "id": EXPECTED_SOURCE_RUN_ID,
        "head_sha": EXPECTED_SOURCE_COMMIT,
        "status": "completed",
        "conclusion": "success",
    }


class TrainingDynamicsAuditTests(unittest.TestCase):
    def test_parse_periodic_logger_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = root / "run.log"
            blocks = []
            for step, entropy_loss, auxiliary_loss in (
                (10_240, -1.44, 1.17),
                (20_480, -0.04, 5.8),
                (30_720, -0.0004, 11.2),
            ):
                blocks.extend(
                    [
                        "| time/ | |",
                        f"| total_timesteps | {step} |",
                        "| train/ | |",
                        f"| entropy_loss | {entropy_loss} |",
                        f"| auxiliary_loss | {auxiliary_loss} |",
                        "| auxiliary_loss_weight | 0 |",
                    ]
                )
            # A short fixture is intentionally repeated to satisfy the 20-snapshot gate.
            for index in range(7):
                step = 40_960 + index * 10_240
                blocks.extend(
                    [
                        f"| total_timesteps | {step} |",
                        f"| entropy_loss | {-0.2 / (index + 1)} |",
                        f"| auxiliary_loss | {2.0 + index} |",
                        "| auxiliary_loss_weight | 0 |",
                    ]
                )
            # Parser's minimum metric-history guard is tested separately from small fixtures.
            for index in range(20):
                step = 850_000 + index * 10_240
                blocks.extend(
                    [
                        f"| total_timesteps | {step} |",
                        f"| entropy_loss | {-0.1 / (index + 1)} |",
                        f"| auxiliary_loss | {2.0 + index} |",
                    ]
                )
            log.write_text("\n".join(blocks) + "\n", encoding="utf-8")
            parsed = parse_training_logs(root)
            self.assertGreaterEqual(len(parsed), 20)
            self.assertEqual(parsed[0]["timesteps"], 10_240)
            self.assertIn("auxiliary_loss", parsed[0])
            self.assertIn("entropy_loss", parsed[-1])

    def test_entropy_summary_marks_low_entropy_without_claiming_zero(self) -> None:
        rows = [
            {"timesteps": 10_240, "entropy_loss": -1.44, "auxiliary_loss": 1.17},
            {"timesteps": 471_040, "entropy_loss": -0.0442, "auxiliary_loss": 5.8},
            {"timesteps": 870_400, "entropy_loss": -0.000407, "auxiliary_loss": 11.2},
        ]
        result = _entropy_summary(rows)
        self.assertEqual(result["first_snapshot_at_or_below_0_05_nats"], 471_040)
        self.assertEqual(result["minimum_observed_entropy_timestep"], 870_400)
        self.assertFalse(result["exact_zero_observed"])

    def test_nearest_snapshot_reports_gap_instead_of_faking_exact_milestone(self) -> None:
        rows = [
            {"timesteps": 92_160, "entropy_loss": -0.5, "auxiliary_loss": 3.0},
            {"timesteps": 102_400, "entropy_loss": -0.4, "auxiliary_loss": 3.2},
        ]
        result = _nearest_snapshot(rows, 100_000)
        self.assertEqual(result["nearest_logged_timestep"], 102_400)
        self.assertEqual(result["absolute_timestep_gap"], 2_400)

    def test_source_identity_and_safety_are_required(self) -> None:
        _validate_source(
            _source_report(),
            _source_run(),
            EXPECTED_SOURCE_RUN_ID,
            EXPECTED_SOURCE_COMMIT,
        )
        unsafe = _source_report()
        unsafe["safety_invariants"]["order_permission"] = True
        with self.assertRaisesRegex(ValueError, "source_safety_invariants_invalid"):
            _validate_source(
                unsafe,
                _source_run(),
                EXPECTED_SOURCE_RUN_ID,
                EXPECTED_SOURCE_COMMIT,
            )

    def test_source_commit_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "source_run_metadata_commit_mismatch"):
            _validate_source(
                _source_report(),
                {**_source_run(), "head_sha": "0" * 40},
                EXPECTED_SOURCE_RUN_ID,
                EXPECTED_SOURCE_COMMIT,
            )


if __name__ == "__main__":
    unittest.main()
