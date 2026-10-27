from __future__ import annotations

import unittest
from typing import Any

from scripts.diagnose_rl_checkpoint_policy import _checkpoint_crossover_summary


class CheckpointPolicyDiagnosticTests(unittest.TestCase):
    def test_reports_checkpoint_window_without_inventing_exact_timestep(self) -> None:
        rows: list[dict[str, Any]] = [
            {
                "checkpoint_timesteps": 100_000,
                "mean_probability_by_side": {"HOLD": 0.25, "BUY": 0.45, "SELL": 0.30},
                "hold_deterministic_action_fraction": 0.80,
            },
            {
                "checkpoint_timesteps": 200_000,
                "mean_probability_by_side": {"HOLD": 0.55, "BUY": 0.25, "SELL": 0.20},
                "hold_deterministic_action_fraction": 0.92,
            },
            {
                "checkpoint_timesteps": 300_000,
                "mean_probability_by_side": {"HOLD": 0.80, "BUY": 0.12, "SELL": 0.08},
                "hold_deterministic_action_fraction": 0.995,
            },
            {
                "checkpoint_timesteps": 400_000,
                "mean_probability_by_side": {"HOLD": 0.84, "BUY": 0.10, "SELL": 0.06},
                "hold_deterministic_action_fraction": 0.996,
            },
            {
                "checkpoint_timesteps": 500_000,
                "mean_probability_by_side": {"HOLD": 0.85, "BUY": 0.09, "SELL": 0.06},
                "hold_deterministic_action_fraction": 0.997,
            },
        ]
        result = _checkpoint_crossover_summary(rows)
        self.assertEqual(
            result[
                "first_checkpoint_mean_hold_probability_exceeds_mean_buy_probability"
            ],
            200_000,
        )
        self.assertEqual(
            result["first_checkpoint_deterministic_hold_share_at_least_99pct"],
            300_000,
        )
        self.assertEqual(
            result[
                "first_persistent_checkpoint_with_deterministic_hold_share_at_least_99pct"
            ],
            300_000,
        )
        self.assertEqual(
            result["persistent_hold_bracket_previous_checkpoint_timestep"],
            200_000,
        )
        self.assertEqual(result["resolution_steps"], 100_000)
        self.assertFalse(result["exact_training_timestep_identifiable"])

    def test_no_persistent_hold_collapse_is_reported_as_none(self) -> None:
        rows: list[dict[str, Any]] = [
            {
                "checkpoint_timesteps": 100_000,
                "mean_probability_by_side": {"HOLD": 0.40, "BUY": 0.30, "SELL": 0.30},
                "hold_deterministic_action_fraction": 0.80,
            },
            {
                "checkpoint_timesteps": 200_000,
                "mean_probability_by_side": {"HOLD": 0.50, "BUY": 0.25, "SELL": 0.25},
                "hold_deterministic_action_fraction": 0.90,
            },
        ]
        result = _checkpoint_crossover_summary(rows)
        self.assertIsNone(
            result[
                "first_persistent_checkpoint_with_deterministic_hold_share_at_least_99pct"
            ]
        )
        self.assertFalse(result["exact_training_timestep_identifiable"])


if __name__ == "__main__":
    unittest.main()
