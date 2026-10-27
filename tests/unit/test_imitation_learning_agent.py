from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_imitation_learning_agent import (
    action_distribution,
    build_feature_matrix,
    make_sequences,
    mask_spot_actions,
    _target_positions,
    simulate_economics,
)
from services.rl_research.institutional_teacher import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    generate_teacher_labels,
)


def _market_frame(rows: int = 1_200) -> pd.DataFrame:
    timestamp = pd.date_range("2024-01-01", periods=rows, freq="min", tz="UTC")
    trend = np.linspace(100.0, 130.0, rows)
    wave = np.sin(np.arange(rows) / 11.0) * 0.4
    close = trend + wave
    open_price = np.r_[close[0], close[:-1]]
    high = np.maximum(open_price, close) + 0.3
    low = np.minimum(open_price, close) - 0.3
    volume = 100.0 + np.arange(rows, dtype=np.float64) % 17
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_feature_matrix_and_sequences_are_finite_and_aligned() -> None:
    frame = _market_frame()
    teacher = generate_teacher_labels(
        frame,
        higher_timeframe="4h+1D",
        ema_span=3,
        swing_window=5,
        volume_window=10,
    )
    features, names = build_feature_matrix(frame, teacher)
    actions = teacher["teacher_action"].to_numpy(dtype=np.int64)
    sequences, labels, target_indices = make_sequences(features, actions, window=32)
    assert features.shape == (len(frame), len(names))
    assert sequences.shape == (len(frame) - 31, 32, len(names))
    assert labels.shape == target_indices.shape == (len(frame) - 31,)
    assert target_indices[0] == 31
    assert target_indices[-1] == len(frame) - 1
    assert np.isfinite(features).all()
    assert np.isfinite(sequences).all()
    assert set(np.unique(labels)).issubset({ACTION_HOLD, ACTION_BUY, ACTION_SELL})


def test_action_distribution_includes_zero_count_classes() -> None:
    distribution = action_distribution(np.asarray([0, 0, 1, 2, 0], dtype=np.int64))
    assert distribution["HOLD"]["count"] == 3
    assert distribution["BUY"]["count"] == 1
    assert distribution["SELL"]["count"] == 1
    assert distribution["HOLD"]["fraction"] == 0.6


def test_spot_sell_without_asset_is_masked_and_short_is_impossible() -> None:
    actions = np.asarray(
        [ACTION_SELL, ACTION_BUY, ACTION_SELL, ACTION_SELL, ACTION_HOLD, ACTION_BUY],
        dtype=np.int64,
    )
    effective = mask_spot_actions(actions)
    np.testing.assert_array_equal(
        effective,
        np.asarray(
            [ACTION_HOLD, ACTION_BUY, ACTION_SELL, ACTION_HOLD, ACTION_HOLD, ACTION_BUY],
            dtype=np.int64,
        ),
    )
    positions = _target_positions(actions)
    assert set(np.unique(positions)).issubset({0, 1})
    np.testing.assert_array_equal(positions, np.asarray([0, 1, 0, 0, 0, 1]))


def test_economic_simulation_is_spot_only_and_cost_sensitive() -> None:
    close = np.asarray([100.0, 101.0, 110.0, 99.0], dtype=np.float64)
    no_position = np.full(4, ACTION_HOLD, dtype=np.int64)
    hold_result = simulate_economics(
        no_position, close, fee_bps_per_side=0.0, slippage_bps_per_side=0.0
    )
    assert hold_result["net_return_fraction"] == 0.0
    assert hold_result["max_drawdown_fraction"] == 0.0
    assert hold_result["closed_trade_count"] == 0
    assert hold_result["win_rate_fraction"] is None

    actions = np.asarray(
        [ACTION_BUY, ACTION_HOLD, ACTION_SELL, ACTION_HOLD], dtype=np.int64
    )
    result = simulate_economics(
        actions, close, fee_bps_per_side=0.0, slippage_bps_per_side=0.0
    )
    result_with_costs = simulate_economics(actions, close)
    assert result["position_change_count"] == 2
    assert result["closed_trade_count"] == 1
    assert result["winning_trades"] == 1
    assert result["win_rate_fraction"] == 1.0
    assert result["net_return_fraction"] > 0.0
    assert result_with_costs["net_return_fraction"] < result["net_return_fraction"]
    assert "positive_period_fraction" not in result


def test_win_rate_and_profit_factor_use_closed_trades_only() -> None:
    actions = np.asarray(
        [ACTION_BUY, ACTION_SELL, ACTION_BUY, ACTION_SELL, ACTION_BUY, ACTION_SELL],
        dtype=np.int64,
    )
    close = np.asarray([100.0, 110.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64)

    result = simulate_economics(
        actions, close, fee_bps_per_side=0.0, slippage_bps_per_side=0.0
    )

    assert result["closed_trade_count"] == 3
    assert result["winning_trades"] == 1
    assert result["losing_trades"] == 1
    assert result["breakeven_trades"] == 1
    assert np.isclose(result["win_rate_fraction"], 1.0 / 3.0)
    assert np.isclose(
        result["profit_factor"], result["gross_profit"] / result["gross_loss"]
    )
    assert "positive_period_fraction" not in result
) -> None:
    close = np.asarray([100.0, 101.0, 100.0, 99.0], dtype=np.float64)
    no_position = np.full(4, ACTION_HOLD, dtype=np.int64)
    hold_result = simulate_economics(
        no_position, close, fee_bps_per_side=0.0, slippage_bps_per_side=0.0
    )
    assert hold_result["net_return_fraction"] == 0.0
    assert hold_result["max_drawdown_fraction"] == 0.0
    actions = np.asarray(
        [ACTION_BUY, ACTION_HOLD, ACTION_SELL, ACTION_HOLD], dtype=np.int64
    )
    result = simulate_economics(
        actions, close, fee_bps_per_side=0.0, slippage_bps_per_side=0.0
    )
    assert result["position_change_count"] == 2
    assert result["evaluated_periods"] == 3
    assert result["net_return_fraction"] > 0.0
