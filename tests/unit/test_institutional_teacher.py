from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services.rl_research.institutional_teacher import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
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
    return pd.DataFrame({
        "timestamp": timestamp,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


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
    kwargs = {"ema_span": 3, "swing_window": 5, "volume_window": 10}
    original = generate_teacher_labels(data, **kwargs)
    changed = data.copy()
    changed.loc[1_500:, ["open", "high", "low", "close"]] *= 1.25
    rerun = generate_teacher_labels(changed, **kwargs)
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
def test_teacher_rejects_invalid_market_data(mutator, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        generate_teacher_labels(mutator(_bars(200)), ema_span=3, swing_window=5, volume_window=10)


def test_teacher_configuration_is_bounded() -> None:
    with pytest.raises(ValueError, match="higher_timeframe_must_be_4h_or_1D"):
        generate_teacher_labels(_bars(200), higher_timeframe="1m", ema_span=3, swing_window=5, volume_window=10)
