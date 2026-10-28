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
    _execution_cost_counterfactual_diagnosis,
    _select_teacher_mode,
    _teacher_entry_logic,
    _teacher_risk_parameters,
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


def test_execution_cost_counterfactual_distinguishes_cost_and_entry_failures() -> None:
    profitable_before_costs = _execution_cost_counterfactual_diagnosis(
        {"profit_factor": 0.8},
        {"profit_factor": 0.9},
        {"profit_factor": 1.2, "net_return_fraction": 0.08},
        {"profit_factor": 1.1, "net_return_fraction": 0.03},
    )
    assert profitable_before_costs["validation_pf_flipped_below_one_by_direct_costs"]
    assert profitable_before_costs["holdout_pf_flipped_below_one_by_direct_costs"]

    unprofitable_before_costs = _execution_cost_counterfactual_diagnosis(
        {"profit_factor": 0.8},
        {"profit_factor": 0.9},
        {"profit_factor": 0.95, "net_return_fraction": -0.02},
        {"profit_factor": 0.97, "net_return_fraction": -0.01},
    )
    assert not unprofitable_before_costs[
        "validation_pf_flipped_below_one_by_direct_costs"
    ]
    assert not unprofitable_before_costs["holdout_pf_flipped_below_one_by_direct_costs"]
    assert (
        "unprofitable even before direct execution costs"
        in unprofitable_before_costs["interpretation"]
    )


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
        "mean_reversion": {
            "gate": {"passed": True},
            "validation": {"economics": {"profit_factor": 1.9}},
        },
        "breakout": {
            "gate": {"passed": True},
            "validation": {"economics": {"profit_factor": 2.0}},
        },
    }
    assert _select_teacher_mode(results) == "breakout"

    results["breakout"]["gate"]["passed"] = False
    results["breakout"]["validation"]["economics"]["profit_factor"] = 9.0
    assert _select_teacher_mode(results) == "mean_reversion"

    results["mean_reversion"]["gate"]["passed"] = False
    results["mean_reversion"]["validation"]["economics"]["profit_factor"] = 9.0
    assert _select_teacher_mode(results) == "trend_pullback"

    results["trend_pullback"]["gate"]["passed"] = False
    results["trend_pullback"]["validation"]["economics"]["profit_factor"] = 9.0
    assert _select_teacher_mode(results) == "institutional"

    results["institutional"]["gate"]["passed"] = False
    assert _select_teacher_mode(results) is None


def test_all_teacher_modes_are_generated_and_saved_independently(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path
    candidates = _write_teacher_datasets(
        _market_frame(),
        {"provider": "Binance Spot public klines API", "csv_sha256": "test-hash"},
        output_dir,
        "test-commit",
    )
    assert set(candidates) == {
        "institutional",
        "trend_pullback",
        "mean_reversion",
        "breakout",
    }
    for mode, (_, manifest) in candidates.items():
        assert manifest["teacher_entry_mode"] == mode
        assert manifest["selected_for_training"] is False
        assert (output_dir / f"teacher-dataset-{mode}.csv").exists()
        assert (output_dir / f"teacher-dataset-{mode}-manifest.json").exists()


def test_breakout_entry_logic_reports_donchian_and_4h_trend_filter() -> None:
    logic = _teacher_entry_logic("breakout")
    assert logic["donchian_lookback_candles"] == 10
    assert "prior 10 candles" in logic["buy_condition"]
    assert "completed 4h EMA bias is bullish" in logic["buy_condition"]
    assert "regardless of the trend filter" in logic["sell_condition"]
    assert logic["signals_use_completed_candles_only"] is True


def test_mean_reversion_entry_logic_reports_looser_thresholds() -> None:
    logic = _teacher_entry_logic("mean_reversion")
    assert logic["bollinger_period_candles"] == 20
    assert logic["bollinger_standard_deviations"] == 1.5
    assert logic["rsi_period_candles"] == 14
    assert logic["rsi_buy_below"] == 35.0
    assert logic["rsi_sell_above"] == 65.0
    assert "RSI-14 < 35" in logic["buy_condition"]
    assert "RSI-14 > 65" in logic["sell_condition"]


def test_mean_reversion_risk_profile_uses_requested_levels_and_costs() -> None:
    risk = _teacher_risk_parameters("mean_reversion")
    assert risk["take_profit_fraction"] == 0.015
    assert risk["stop_loss_fraction"] == 0.01
    assert risk["trailing_trigger_fraction"] == 0.01
    assert risk["trailing_distance_fraction"] == 0.01
    assert risk["fee_bps_per_side"] == 25.0
    assert risk["slippage_bps_per_side"] == 20.0

    diagnostics = _cost_adjusted_risk_diagnostics(
        risk["take_profit_fraction"], risk["stop_loss_fraction"]
    )
    assert diagnostics["take_profit_fraction"] == 0.015
    assert diagnostics["stop_loss_fraction"] == 0.01
    assert diagnostics["estimated_net_return_at_tp_without_gaps"] > 0.0
    assert diagnostics["estimated_net_return_at_sl_without_gaps"] < 0.0
    assert 0.0 < diagnostics["break_even_win_rate_fraction_without_gaps"] < 1.0


def test_breakout_risk_profile_uses_existing_strict_economic_assumptions() -> None:
    risk = _teacher_risk_parameters("breakout")
    assert risk["take_profit_fraction"] == 0.025
    assert risk["stop_loss_fraction"] == 0.015
    assert risk["trailing_trigger_fraction"] == 0.01
    assert risk["fee_bps_per_side"] == 25.0
    assert risk["slippage_bps_per_side"] == 20.0


def test_teacher_economic_gate_requires_profitability_and_signal_density() -> None:
    passing = {"profit_factor": 1.6, "max_drawdown_fraction": 0.10}
    failing = {"profit_factor": 1.49, "max_drawdown_fraction": 0.10}
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

    accepted = _teacher_backtest_gate(passing, passing, sufficient_signals)
    assert accepted["passed"] is True
    assert accepted["maximum_allowed_drawdown_fraction"] == 0.15
    assert accepted["validation_profit_factor_passed"] is True
    assert accepted["holdout_profit_factor_passed"] is True
    assert accepted["validation_drawdown_passed"] is True
    assert accepted["holdout_drawdown_passed"] is True
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


def test_teacher_economic_gate_fails_closed_on_excessive_or_missing_drawdown() -> None:
    sufficient_signals = {
        "BUY": {"count": 400},
        "HOLD": {"count": 16_720},
        "SELL": {"count": 400},
    }
    passing = {"profit_factor": 1.6, "max_drawdown_fraction": 0.10}
    excessive_drawdown = {
        "profit_factor": 1.6,
        "max_drawdown_fraction": 0.150001,
    }
    result = _teacher_backtest_gate(excessive_drawdown, passing, sufficient_signals)
    assert result["validation_profit_factor_passed"] is True
    assert result["validation_drawdown_passed"] is False
    assert result["validation_passed"] is False
    assert result["passed"] is False
    assert result["lstm_training_allowed"] is False

    missing_drawdown = _teacher_backtest_gate(
        {"profit_factor": 1.6}, passing, sufficient_signals
    )
    assert missing_drawdown["validation_drawdown_passed"] is False
    assert missing_drawdown["lstm_training_allowed"] is False


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
