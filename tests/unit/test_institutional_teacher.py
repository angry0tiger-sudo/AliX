from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from services.rl_research.institutional_teacher import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    DEFAULT_MEAN_REVERSION_BOLLINGER_PERIOD,
    DEFAULT_MEAN_REVERSION_BOLLINGER_STDDEV,
    DEFAULT_MEAN_REVERSION_RSI_BUY_BELOW,
    DEFAULT_MEAN_REVERSION_RSI_PERIOD,
    DEFAULT_MEAN_REVERSION_RSI_SELL_ABOVE,
    DEFAULT_MIN_ATR_COST_MULTIPLE,
    DEFAULT_STOP_LOSS_FRACTION,
    DEFAULT_TAKE_PROFIT_FRACTION,
    _average_true_range_fraction,
    _generate_teacher_actions,
    _smc_features,
    _teacher_setups,
    _trend_pullback_setups,
    _mean_reversion_setups,
    _breakout_setups,
    generate_teacher_labels,
)


def _bars(rows: int = 2_000) -> pd.DataFrame:
    timestamp = pd.date_range("2024-01-01", periods=rows, freq="min", tz="UTC")
    trend = np.linspace(100.0, 150.0, rows)
    wave = np.sin(np.arange(rows) / 7.0) * 0.8
    close = trend + wave
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    volume = np.full(rows, 100.0)
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_teacher_output_is_causal_schema_and_valid_actions() -> None:
    data = _bars()
    result = generate_teacher_labels(
        data, higher_timeframe="4h", ema_span=3, swing_window=5, volume_window=10
    )
    assert len(result) == len(data)
    assert result["teacher_action"].isin([ACTION_HOLD, ACTION_BUY, ACTION_SELL]).all()
    assert result["higher_timeframe_bias"].isin([-1, 0, 1]).all()
    assert result["teacher_action_name"].isin(["HOLD", "BUY", "SELL"]).all()
    assert result["teacher_action"].dtype == np.int8


def test_teacher_does_not_change_past_labels_when_future_data_changes() -> None:
    data = _bars()
    original = generate_teacher_labels(
        data, ema_span=3, swing_window=5, volume_window=10
    )
    changed = data.copy()
    changed.loc[1_500:, ["open", "high", "low", "close"]] *= 1.25
    rerun = generate_teacher_labels(
        changed, ema_span=3, swing_window=5, volume_window=10
    )
    pd.testing.assert_series_equal(
        original.loc[:1_499, "teacher_action"],
        rerun.loc[:1_499, "teacher_action"],
        check_names=False,
    )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda df: df.drop(columns=["volume"]), "missing_ohlcv_columns"),
        (lambda df: df.assign(volume=-1.0), "volume_must_be_non_negative"),
        (lambda df: df.assign(close=np.nan), "ohlcv_contains_non_finite_values"),
    ],
)
def test_teacher_rejects_invalid_market_data(
    mutator: Callable[[pd.DataFrame], pd.DataFrame], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        generate_teacher_labels(
            mutator(_bars(200)), ema_span=3, swing_window=5, volume_window=10
        )


def test_teacher_configuration_is_bounded() -> None:
    with pytest.raises(ValueError, match=r"higher_timeframe_must_be_4h_1D_or_4h\+1D"):
        generate_teacher_labels(
            _bars(200),
            higher_timeframe="1m",
            ema_span=3,
            swing_window=5,
            volume_window=10,
        )


def test_multi_timeframe_teacher_requires_bias_confluence() -> None:
    data = _bars(2_000)
    result = generate_teacher_labels(
        data,
        higher_timeframe="4h+1D",
        ema_span=3,
        swing_window=5,
        volume_window=10,
    )
    assert result["higher_timeframe_bias_4h"].isin([-1, 0, 1]).all()
    assert result["higher_timeframe_bias_1d"].isin([-1, 0, 1]).all()
    assert result["higher_timeframe_bias"].isin([-1, 0, 1]).all()
    active_bias = result["higher_timeframe_bias"] != 0
    assert (
        result.loc[active_bias, "higher_timeframe_bias_4h"]
        == result.loc[active_bias, "higher_timeframe_bias_1d"]
    ).all()


def test_higher_timeframe_bias_excludes_boundary_candle() -> None:
    data = _bars(4_000)
    original = generate_teacher_labels(
        data, higher_timeframe="4h", ema_span=2, swing_window=5, volume_window=10
    )
    changed = data.copy()
    for column in ("open", "high", "low", "close"):
        values = changed[column].to_numpy(dtype=np.float64)
        values[1_440] *= 0.05
        changed[column] = values
    rerun = generate_teacher_labels(
        changed, higher_timeframe="4h", ema_span=2, swing_window=5, volume_window=10
    )
    assert (
        original.loc[1_440, "higher_timeframe_bias"]
        == rerun.loc[1_440, "higher_timeframe_bias"]
    )


def test_teacher_position_persists_across_hold_actions() -> None:
    result = generate_teacher_labels(
        _bars(2_000),
        higher_timeframe="4h+1D",
        ema_span=3,
        swing_window=5,
        volume_window=10,
    )
    expected = np.zeros(len(result), dtype=np.int8)
    position = 0
    for index, action in enumerate(result["teacher_action"].to_numpy()):
        if action == ACTION_BUY:
            assert position == 0
            position = 1
        elif action == ACTION_SELL:
            assert position == 1
            position = 0
        expected[index] = position
    np.testing.assert_array_equal(result["teacher_position"].to_numpy(), expected)
    assert set(np.unique(expected)).issubset({0, 1})


def test_order_block_touch_uses_candle_range_not_close_only() -> None:
    data = pd.DataFrame(
        {
            "open": [100.0, 102.0, 103.0],
            "high": [101.0, 102.5, 104.5],
            "low": [99.5, 99.5, 102.5],
            "close": [100.5, 100.0, 104.0],
        }
    )
    _, _, _, bullish_touch, _ = _smc_features(data, swing_window=2)
    assert bullish_touch.iloc[2]
    assert float(data["close"].iloc[2]) > float(data["high"].iloc[1])


def test_order_block_touch_is_valid_without_bos_fvg_or_volume() -> None:
    bias = pd.Series([1, -1], dtype=np.int8)
    no_structure = pd.Series([False, False])
    bullish_touch = pd.Series([True, False])
    bearish_touch = pd.Series([False, True])
    no_volume_confirmation = pd.Series([False, False])

    buy_setup, sell_setup = _teacher_setups(
        bias,
        no_structure,
        no_structure,
        bullish_touch,
        bearish_touch,
        no_volume_confirmation,
        no_volume_confirmation,
    )

    assert buy_setup.tolist() == [True, False]
    assert sell_setup.tolist() == [False, True]


def test_atr_fraction_is_causal_and_uses_true_range() -> None:
    data = _bars(80)
    atr = _average_true_range_fraction(data, period=14)
    changed = data.copy()
    changed.loc[60:, ["open", "high", "low", "close"]] *= 2.0
    rerun = _average_true_range_fraction(changed, period=14)
    pd.testing.assert_series_equal(atr.loc[:59], rerun.loc[:59], check_names=False)
    assert atr.iloc[:13].isna().all()
    assert np.isfinite(atr.iloc[13:]).all()


def test_low_atr_blocks_new_teacher_entries() -> None:
    data = _bars(8).reset_index(drop=True)
    buy_setup = pd.Series([True] * len(data))
    sell_setup = pd.Series([False] * len(data))
    low_atr = pd.Series([0.001] * len(data))

    actions, positions, _, _ = _generate_teacher_actions(
        data,
        buy_setup,
        sell_setup,
        low_atr,
        cooldown_candles=2,
        min_atr_cost_multiple=DEFAULT_MIN_ATR_COST_MULTIPLE,
        take_profit_fraction=DEFAULT_TAKE_PROFIT_FRACTION,
        stop_loss_fraction=DEFAULT_STOP_LOSS_FRACTION,
        trailing_trigger_fraction=0.01,
        trailing_distance_fraction=0.005,
        fee_bps_per_side=25.0,
        slippage_bps_per_side=20.0,
    )

    assert ACTION_BUY not in actions
    assert positions.tolist() == [0] * len(data)


def test_trailing_stop_protects_cost_adjusted_breakeven() -> None:
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC"),
            "open": [100.0, 100.0, 101.0, 101.0],
            "high": [100.0, 101.2, 101.3, 101.2],
            "low": [100.0, 99.9, 100.8, 100.5],
            "close": [100.0, 101.0, 101.1, 100.8],
            "volume": [100.0] * 4,
        }
    )
    buy_setup = pd.Series([True, False, False, False])
    sell_setup = pd.Series([False] * 4)
    atr_fraction = pd.Series([0.02] * 4)

    actions, positions, exit_prices, exit_reasons = _generate_teacher_actions(
        data,
        buy_setup,
        sell_setup,
        atr_fraction,
        cooldown_candles=2,
        min_atr_cost_multiple=DEFAULT_MIN_ATR_COST_MULTIPLE,
        take_profit_fraction=DEFAULT_TAKE_PROFIT_FRACTION,
        stop_loss_fraction=DEFAULT_STOP_LOSS_FRACTION,
        trailing_trigger_fraction=0.01,
        trailing_distance_fraction=0.005,
        fee_bps_per_side=25.0,
        slippage_bps_per_side=20.0,
    )

    assert actions.tolist() == [ACTION_BUY, ACTION_HOLD, ACTION_SELL, ACTION_HOLD]
    assert positions.tolist() == [1, 1, 0, 0]
    assert exit_reasons[2] == "trailing_stop"
    assert exit_prices[2] > 100.0


def test_teacher_defaults_match_daily_opportunity_profile() -> None:
    assert DEFAULT_TAKE_PROFIT_FRACTION == 0.025
    assert DEFAULT_STOP_LOSS_FRACTION == 0.015
    assert DEFAULT_MIN_ATR_COST_MULTIPLE == 1.0
    assert (DEFAULT_MIN_ATR_COST_MULTIPLE * (25.0 + 20.0) / 10_000.0) == 0.0045


def test_trend_pullback_fallback_generates_directional_setups() -> None:
    count = 120
    close_up = pd.Series(100.0 + np.arange(count, dtype=np.float64) * 0.1)
    bullish = pd.DataFrame(
        {
            "close": close_up,
            "low": close_up - 0.8,
            "high": close_up + 0.2,
        }
    )
    buy_setup, sell_setup = _trend_pullback_setups(
        bullish, pd.Series([1] * count, dtype=np.int8)
    )
    assert buy_setup.any()
    assert not sell_setup.any()

    close_down = close_up.iloc[::-1].reset_index(drop=True)
    bearish = pd.DataFrame(
        {
            "close": close_down,
            "low": close_down - 0.2,
            "high": close_down + 0.8,
        }
    )
    buy_setup, sell_setup = _trend_pullback_setups(
        bearish, pd.Series([-1] * count, dtype=np.int8)
    )
    assert sell_setup.any()
    assert not buy_setup.any()


def test_breakout_setups_use_only_prior_donchian_channel() -> None:
    flat_close = np.full(20, 100.0, dtype=np.float64)
    for final_close, expected_buy, expected_sell in (
        (101.0, True, False),
        (99.0, False, True),
    ):
        close = np.concatenate((flat_close, np.asarray([final_close])))
        open_ = np.r_[close[0], close[:-1]]
        frame = pd.DataFrame(
            {
                "close": close,
                "high": np.maximum(open_, close) + 0.2,
                "low": np.minimum(open_, close) - 0.2,
            }
        )
        buy_setup, sell_setup = _breakout_setups(frame, lookback_period=20)
        assert not bool(buy_setup.iloc[:20].any())
        assert not bool(sell_setup.iloc[:20].any())
        assert bool(buy_setup.iloc[-1]) is expected_buy
        assert bool(sell_setup.iloc[-1]) is expected_sell


def test_mean_reversion_defaults_use_looser_bands_and_rsi_thresholds() -> None:
    assert DEFAULT_MEAN_REVERSION_BOLLINGER_PERIOD == 20
    assert DEFAULT_MEAN_REVERSION_BOLLINGER_STDDEV == 1.5
    assert DEFAULT_MEAN_REVERSION_RSI_PERIOD == 14
    assert DEFAULT_MEAN_REVERSION_RSI_BUY_BELOW == 35.0
    assert DEFAULT_MEAN_REVERSION_RSI_SELL_ABOVE == 65.0


def test_mean_reversion_setups_require_bollinger_touch_and_rsi_extreme() -> None:
    flat_close = np.full(39, 100.0, dtype=np.float64)

    oversold_close = np.concatenate((flat_close, np.asarray([80.0])))
    oversold_open = np.r_[oversold_close[0], oversold_close[:-1]]
    oversold = pd.DataFrame(
        {
            "close": oversold_close,
            "high": np.maximum(oversold_open, oversold_close) + 0.1,
            "low": np.minimum(oversold_open, oversold_close) - 0.1,
        }
    )
    buy_setup, sell_setup = _mean_reversion_setups(oversold)
    assert bool(buy_setup.iloc[-1])
    assert not bool(sell_setup.iloc[-1])

    overbought_close = np.concatenate((flat_close, np.asarray([120.0])))
    overbought_open = np.r_[overbought_close[0], overbought_close[:-1]]
    overbought = pd.DataFrame(
        {
            "close": overbought_close,
            "high": np.maximum(overbought_open, overbought_close) + 0.1,
            "low": np.minimum(overbought_open, overbought_close) - 0.1,
        }
    )
    buy_setup, sell_setup = _mean_reversion_setups(overbought)
    assert not bool(buy_setup.iloc[-1])
    assert bool(sell_setup.iloc[-1])


def test_teacher_rejects_unknown_entry_mode() -> None:
    with pytest.raises(
        ValueError,
        match="teacher_entry_mode_must_be_institutional_trend_pullback_mean_reversion_or_breakout",
    ):
        generate_teacher_labels(_bars(200), entry_mode="unrestricted")
