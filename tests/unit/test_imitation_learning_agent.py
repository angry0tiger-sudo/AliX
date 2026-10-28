from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.run_imitation_learning_agent import (
    action_distribution,
    build_feature_matrix,
    make_sequences,
    mask_spot_actions,
    _target_positions,
    _compute_class_weights,
    _teacher_backtest_gate,
    _cost_adjusted_risk_diagnostics,
    _select_teacher_mode,
    _write_teacher_datasets,
    _weighted_cross_entropy_loss,
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
            [
                ACTION_HOLD,
                ACTION_BUY,
                ACTION_SELL,
                ACTION_HOLD,
                ACTION_HOLD,
                ACTION_BUY,
            ],
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
    assert result_with_costs["fee_bps_per_side"] == 25.0
    assert result_with_costs["slippage_bps_per_side"] == 20.0
    assert "positive_period_fraction" not in result


def test_win_rate_and_profit_factor_use_closed_trades_only() -> None:
    actions = np.asarray(
        [ACTION_BUY, ACTION_SELL, ACTION_BUY, ACTION_SELL, ACTION_BUY, ACTION_SELL],
        dtype=np.int64,
    )
    close = np.asarray([64.0, 128.0, 128.0, 64.0, 64.0, 64.0], dtype=np.float64)

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


def test_class_weights_upweight_only_minority_actions_from_train_labels() -> None:
    train_actions = np.asarray(
        [ACTION_HOLD] * 1_000 + [ACTION_BUY] * 4 + [ACTION_SELL] * 5,
        dtype=np.int64,
    )
    weights = _compute_class_weights(train_actions)

    assert weights[ACTION_HOLD] == 1.0
    assert weights[ACTION_BUY] == 25.0
    assert weights[ACTION_SELL] == 25.0


def test_class_weighting_changes_per_sample_loss_for_minority_actions() -> None:
    class FakeFunctional:
        @staticmethod
        def cross_entropy(
            logits: object, targets: np.ndarray, reduction: str
        ) -> np.ndarray:
            assert reduction == "none"
            return np.asarray([2.0, 2.0], dtype=np.float64)

    class FakeNN:
        functional = FakeFunctional()

    class FakeTorch:
        nn = FakeNN()

    targets = np.asarray([ACTION_HOLD, ACTION_BUY], dtype=np.int64)
    weights = np.asarray([1.0, 25.0, 25.0], dtype=np.float64)
    loss = _weighted_cross_entropy_loss(FakeTorch, object(), targets, weights)

    assert loss == 26.0


def test_class_weights_fail_closed_when_train_split_lacks_a_class() -> None:
    train_actions = np.asarray([ACTION_HOLD, ACTION_BUY, ACTION_HOLD], dtype=np.int64)
    try:
        _compute_class_weights(train_actions)
    except ValueError as exc:
        assert str(exc) == "teacher_train_split_missing_action_class"
    else:
        raise AssertionError("missing training class must fail closed")


def test_cost_adjusted_tp_sl_and_break_even_win_rate_are_reported() -> None:
    diagnostics = _cost_adjusted_risk_diagnostics()
    target_return = diagnostics["estimated_net_return_at_tp_without_gaps"]
    stop_return = diagnostics["estimated_net_return_at_sl_without_gaps"]
    required_win_rate = diagnostics["break_even_win_rate_fraction_without_gaps"]

    assert np.isclose(target_return, 0.0158, atol=0.0001)
    assert np.isclose(stop_return, -0.0238, atol=0.0001)
    assert 0.60 < required_win_rate < 0.61


def test_mode_selection_uses_validation_pf_among_passing_modes() -> None:
    results: dict[str, dict[str, Any]] = {
        "institutional": {
            "gate": {"passed": True},
            "validation": {"economics": {"profit_factor": 1.7}},
        },
        "trend_pullback": {
            "gate": {"passed": True},
            "validation": {"economics": {"profit_factor": 1.8}},
        },
    }
    assert _select_teacher_mode(results) == "trend_pullback"

    results["trend_pullback"]["gate"]["passed"] = False
    results["trend_pullback"]["validation"]["economics"]["profit_factor"] = 9.0
    assert _select_teacher_mode(results) == "institutional"

    results["institutional"]["gate"]["passed"] = False
    assert _select_teacher_mode(results) is None


def test_both_teacher_modes_are_generated_and_saved_independently(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path
    candidates = _write_teacher_datasets(
        _market_frame(),
        {"provider": "Binance Spot public klines API", "csv_sha256": "test-hash"},
        output_dir,
        "test-commit",
    )
    assert set(candidates) == {"institutional", "trend_pullback"}
    for mode, (_, manifest) in candidates.items():
        assert manifest["teacher_entry_mode"] == mode
        assert manifest["selected_for_training"] is False
        assert (output_dir / f"teacher-dataset-{mode}.csv").exists()
        assert (output_dir / f"teacher-dataset-{mode}-manifest.json").exists()


def test_teacher_economic_gate_requires_profitability_and_signal_density() -> None:
    passing = {"profit_factor": 1.6}
    failing = {"profit_factor": 1.49}
    sufficient_signals = {
        "BUY": {"count": 400},
        "HOLD": {"count": 16_720},
        "SELL": {"count": 400},
    }
    sparse_signals = {
        "BUY": {"count": 20},
        "HOLD": {"count": 17_480},
        "SELL": {"count": 20},
    }

    assert (
        _teacher_backtest_gate(passing, passing, sufficient_signals)["passed"] is True
    )
    assert (
        _teacher_backtest_gate(passing, failing, sufficient_signals)["passed"] is False
    )
    assert (
        _teacher_backtest_gate(passing, passing, sparse_signals)[
            "lstm_training_allowed"
        ]
        is False
    )
    assert (
        _teacher_backtest_gate({"profit_factor": None}, passing, sufficient_signals)[
            "lstm_training_allowed"
        ]
        is False
    )


def test_economic_simulation_uses_protective_exit_fill_price() -> None:
    actions = np.asarray([ACTION_BUY, ACTION_HOLD, ACTION_SELL], dtype=np.int64)
    close = np.asarray([100.0, 101.0, 98.0], dtype=np.float64)
    protective_exit = np.asarray([np.nan, np.nan, 100.9], dtype=np.float64)

    result_at_close = simulate_economics(
        actions, close, fee_bps_per_side=0.0, slippage_bps_per_side=0.0
    )
    result_at_stop = simulate_economics(
        actions,
        close,
        exit_prices=protective_exit,
        fee_bps_per_side=0.0,
        slippage_bps_per_side=0.0,
    )

    assert result_at_close["net_return_fraction"] < 0
    assert result_at_stop["net_return_fraction"] > 0
