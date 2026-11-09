from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.train_risk_allocation_ppo import _evaluate_validation_segment


def validation_segment(
    signals: tuple[bool, ...],
) -> pd.DataFrame:
    count = len(signals)
    close = np.full(count, 100.0, dtype=np.float64)
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01T00:00:00Z", periods=count, freq="1h"),
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(count, 10.0, dtype=np.float64),
            "atr_fraction": np.full(count, 0.01, dtype=np.float64),
            "primary_buy_signal": np.asarray(signals, dtype=np.bool_),
        }
    )


def test_fixed_position_baseline_uses_full_exposure_only_on_primary_buy() -> None:
    metrics = _evaluate_validation_segment(
        None,
        validation_segment((True, True, True, True, True)),
        "1h",
        17,
    )

    assert metrics["max_target_exposure_fraction"] == 1.0
    assert metrics["mean_target_exposure_fraction"] == 1.0
    assert metrics["time_in_market_fraction"] == 1.0
    assert metrics["target_exposure_distribution"]["1.00"] == 1.0
    assert metrics["total_trading_cost_quote"] > 0.0
    assert metrics["total_traded_quote"] > 0.0
    assert metrics["turnover_fraction_of_initial_equity"] > 0.0
    assert metrics["position_adjustments"] > 0
    assert metrics["equity_observations"] == 4


def test_fixed_position_baseline_stays_flat_without_primary_buy_signal() -> None:
    metrics = _evaluate_validation_segment(
        None,
        validation_segment((False, False, False, False, False)),
        "1h",
        17,
    )

    assert metrics["max_target_exposure_fraction"] == 0.0
    assert metrics["mean_target_exposure_fraction"] == 0.0
    assert metrics["time_in_market_fraction"] == 0.0
    assert metrics["target_exposure_distribution"]["0.00"] == 1.0
    assert metrics["net_return_fraction"] == 0.0
    assert metrics["total_trading_cost_quote"] == 0.0
    assert metrics["total_traded_quote"] == 0.0
    assert metrics["position_adjustments"] == 0
    assert metrics["annualized_sharpe_ratio"] is None
    assert metrics["annualized_sortino_ratio_mar_zero"] is None
