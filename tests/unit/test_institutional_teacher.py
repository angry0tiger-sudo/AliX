from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from services.rl_research.institutional_teacher import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    _smc_features,
    _teacher_setups,
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
