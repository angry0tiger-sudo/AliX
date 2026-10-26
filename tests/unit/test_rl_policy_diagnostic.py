"""Unit tests for RL action-diagnostic helpers."""

from __future__ import annotations

import math
from collections import Counter

from scripts.diagnose_rl_trading_agent import (
    _classify_policy_modes,
    _count_distribution,
    _entropy,
)


def test_entropy_is_zero_for_a_certain_action() -> None:
    assert _entropy([1.0, 0.0, 0.0]) == 0.0


def test_entropy_is_log_three_for_uniform_direction_probabilities() -> None:
    assert math.isclose(_entropy([1 / 3, 1 / 3, 1 / 3]), math.log(3), rel_tol=1e-12)


def test_count_distribution_uses_requested_action_denominator() -> None:
    result = _count_distribution(
        ["BUY", "SELL", "HOLD"],
        Counter({"HOLD": 8, "BUY": 2}),
        denominator=10,
    )
    assert result["BUY"] == {"count": 2, "fraction": 0.2}
    assert result["SELL"] == {"count": 0, "fraction": 0.0}
    assert result["HOLD"] == {"count": 8, "fraction": 0.8}


def test_count_distribution_marks_empty_denominator_as_undefined() -> None:
    result = _count_distribution(["25%", "50%"], Counter(), denominator=0)
    assert result["25%"] == {"count": 0, "fraction": None}
    assert result["50%"] == {"count": 0, "fraction": None}


def test_mode_classifier_detects_stochastic_only_trades() -> None:
    result = _classify_policy_modes(
        {"BUY": 0, "SELL": 0, "HOLD": 100},
        {"BUY": 6, "SELL": 4, "HOLD": 90},
    )
    assert result["code"] == "STOCHASTIC_TRADES_ONLY"


def test_mode_classifier_does_not_infer_trade_mass_from_action_counts() -> None:
    result = _classify_policy_modes(
        {"BUY": 0, "SELL": 0, "HOLD": 100},
        {"BUY": 0, "SELL": 0, "HOLD": 100},
    )
    assert result["code"] == "NO_BUY_SELL_SAMPLED"
