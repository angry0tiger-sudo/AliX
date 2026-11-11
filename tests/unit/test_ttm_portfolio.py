"""Tests for TTM risk-only sizing over a fixed out-of-sample trade stream."""

from __future__ import annotations

import numpy as np
import pytest

from research.ai_ensemble.ttm_portfolio import (
    attach_asof_risk_multipliers,
    evaluate_sized_portfolio,
)


def test_asof_multiplier_never_uses_a_future_forecast() -> None:
    trades = [
        {"entry_market_index": 11, "net_return": 0.02},
        {"entry_market_index": 20, "net_return": -0.03},
        {"entry_market_index": 21, "net_return": 0.01},
    ]
    attached = attach_asof_risk_multipliers(
        trades,
        np.asarray([10, 20], dtype=np.int64),
        np.asarray([0.8, 0.4], dtype=np.float64),
    )
    assert [trade["risk_multiplier"] for trade in attached] == [0.8, 0.4, 0.4]


def test_risk_managed_portfolio_changes_exposure_not_trade_signals() -> None:
    trades = [
        {"net_return": 0.10, "risk_multiplier": 0.8},
        {"net_return": -0.10, "risk_multiplier": 0.2},
        {"net_return": 0.04, "risk_multiplier": 0.6},
        {"net_return": -0.05, "risk_multiplier": 0.25},
    ]
    fixed = evaluate_sized_portfolio(trades, risk_managed=False)
    managed = evaluate_sized_portfolio(trades, risk_managed=True)
    assert fixed["trade_count"] == managed["trade_count"] == 4
    assert fixed["win_rate"] == managed["win_rate"] == 0.5
    assert managed["net_profit_factor"] > fixed["net_profit_factor"]
    assert managed["max_drawdown"] < fixed["max_drawdown"]
    assert managed["average_position_fraction"] <= fixed["average_position_fraction"]
    assert managed["average_position_fraction"] <= 0.10


def test_position_sizing_rejects_multiplier_above_one() -> None:
    trades = [{"net_return": 0.01, "risk_multiplier": 1.01}]
    with pytest.raises(ValueError, match="multiplier_invalid"):
        evaluate_sized_portfolio(trades, risk_managed=True)


def test_empty_trade_stream_has_no_economic_metrics() -> None:
    metrics = evaluate_sized_portfolio([], risk_managed=True)
    assert metrics["trade_count"] == 0
    assert metrics["win_rate"] is None
    assert metrics["net_profit_factor"] is None
    assert metrics["max_drawdown"] is None
