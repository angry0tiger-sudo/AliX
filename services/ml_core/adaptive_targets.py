"""ATR-adaptive Triple-Barrier labels for offline research only.

The canonical target geometry is absolute-price ATR(14): TP = 2 * ATR and
SL = 1 * ATR. Labels inspect only the next 24 candles after each entry close.
When TP and SL are both touched on one candle, stop-loss wins. Incomplete
ATR history or an incomplete 24-candle future horizon is left unlabeled.
This module does not create features, train models, or place orders.
"""

from __future__ import annotations

from math import isfinite
from typing import Final

import numpy as np
import pandas as pd
from pandas import DataFrame

ATR_PERIOD: Final[int] = 14
TAKE_PROFIT_MULTIPLIER: Final[float] = 2.0
STOP_LOSS_MULTIPLIER: Final[float] = 1.0
LABEL_HORIZON_CANDLES: Final[int] = 24

_OHLCV_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
_PRICE_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close", "volume")


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name}_must_be_positive_integer")


def _validated_ohlcv(candles: DataFrame) -> DataFrame:
    missing = sorted(set(_OHLCV_COLUMNS) - set(candles.columns))
    if missing:
        raise ValueError(f"ohlcv_missing_columns:{','.join(missing)}")
    if candles.empty:
        raise ValueError("ohlcv_empty")

    result = candles.loc[:, list(_OHLCV_COLUMNS)].copy().reset_index(drop=True)
    dates = pd.to_datetime(result["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("timestamp_invalid")
    if dates.dt.tz is None:
        raise ValueError("timezone_required")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("timestamps_must_be_unique_and_increasing")
    deltas = dates.diff().iloc[1:].dt.total_seconds().to_numpy(dtype=float)
    if len(deltas) and (
        not np.isfinite(deltas).all()
        or np.any(deltas <= 0)
        or not np.all(deltas == deltas[0])
    ):
        raise ValueError("timestamp_interval_inconsistent")
    result["date"] = dates.dt.tz_convert("UTC")

    for name in _PRICE_COLUMNS:
        values = pd.to_numeric(result[name], errors="coerce")
        numeric = values.to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"ohlcv_non_finite:{name}")
        result[name] = values

    opened = result["open"].to_numpy(dtype=float)
    high = result["high"].to_numpy(dtype=float)
    low = result["low"].to_numpy(dtype=float)
    close = result["close"].to_numpy(dtype=float)
    volume = result["volume"].to_numpy(dtype=float)
    if (
        np.any(opened <= 0)
        or np.any(high <= 0)
        or np.any(low <= 0)
        or np.any(close <= 0)
    ):
        raise ValueError("ohlcv_prices_must_be_positive")
    if np.any(volume < 0):
        raise ValueError("ohlcv_volume_negative")
    if (
        np.any(high < low)
        or np.any(high < np.maximum(opened, close))
        or np.any(low > np.minimum(opened, close))
    ):
        raise ValueError("ohlcv_geometry_invalid")
    return result


def calculate_atr(candles: DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """Calculate Wilder-style true range with a simple rolling ATR mean.

    True range is max(high-low, abs(high-prev_close), abs(low-prev_close)).
    The first `period - 1` ATR values are NaN because a full window is required.
    Values are absolute price units, not returns or percentages.
    """
    _positive_integer(period, "atr_period")
    normalized = _validated_ohlcv(candles)
    high = normalized["high"]
    low = normalized["low"]
    close = normalized["close"]
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=True)
    return (
        true_range.rolling(window=period, min_periods=period)
        .mean()
        .rename(f"atr_{period}")
    )


def _validate_barrier_configuration(
    atr_period: int,
    horizon: int,
    take_profit_multiplier: float,
    stop_loss_multiplier: float,
) -> None:
    _positive_integer(atr_period, "atr_period")
    _positive_integer(horizon, "horizon")
    if (
        not isfinite(float(take_profit_multiplier))
        or not isfinite(float(stop_loss_multiplier))
        or take_profit_multiplier <= 0
        or stop_loss_multiplier <= 0
    ):
        raise ValueError("barrier_multipliers_must_be_finite_and_positive")


def apply_adaptive_triple_barrier(
    candles: DataFrame,
    *,
    atr_values: pd.Series | None = None,
    atr_period: int = ATR_PERIOD,
    horizon: int = LABEL_HORIZON_CANDLES,
    take_profit_multiplier: float = TAKE_PROFIT_MULTIPLIER,
    stop_loss_multiplier: float = STOP_LOSS_MULTIPLIER,
) -> DataFrame:
    """Create BUY/SELL/HOLD outcome labels without combining them with features.

    At index t, entry is close[t] and ATR[t] uses only OHLCV through t.
    Future OHLCV at t+1 ... t+horizon is inspected only to determine the label.
    A bar touching both barriers resolves to SELL because stop-loss has priority.
    Rows without enough ATR history or a complete future horizon have label=None.
    """
    _validate_barrier_configuration(
        atr_period, horizon, take_profit_multiplier, stop_loss_multiplier
    )
    normalized = _validated_ohlcv(candles)
    computed_atr = atr_values is None
    atr_series = calculate_atr(normalized, atr_period) if computed_atr else atr_values
    if atr_series is None or len(atr_series) != len(normalized):
        raise ValueError("atr_length_mismatch")
    atr_array = pd.to_numeric(atr_series, errors="coerce").to_numpy(dtype=float)

    records: list[dict[str, int | float | str | None | pd.Timestamp]] = []
    row_count = len(normalized)
    for event_index in range(row_count):
        entry_price = float(normalized.iloc[event_index]["close"])
        event_time = pd.Timestamp(normalized.iloc[event_index]["date"])
        current_atr = float(atr_array[event_index])

        if computed_atr and event_index < atr_period - 1:
            records.append(
                {
                    "event_index": event_index,
                    "event_timestamp": event_time,
                    "entry_price": entry_price,
                    "atr": None,
                    "take_profit_price": None,
                    "stop_loss_price": None,
                    "label": None,
                    "exit_reason": "insufficient_atr_history",
                    "label_end_index": None,
                }
            )
            continue
        if not isfinite(current_atr):
            raise ValueError("atr_must_be_finite")
        if current_atr <= 0:
            raise ValueError("atr_must_be_positive")

        take_profit_price = entry_price + take_profit_multiplier * current_atr
        stop_loss_price = entry_price - stop_loss_multiplier * current_atr
        if not isfinite(take_profit_price) or not isfinite(stop_loss_price):
            raise ValueError("atr_barrier_price_not_finite")
        if stop_loss_price <= 0:
            raise ValueError("atr_stop_price_not_positive")

        last_horizon_index = event_index + horizon
        if last_horizon_index >= row_count:
            records.append(
                {
                    "event_index": event_index,
                    "event_timestamp": event_time,
                    "entry_price": entry_price,
                    "atr": current_atr,
                    "take_profit_price": take_profit_price,
                    "stop_loss_price": stop_loss_price,
                    "label": None,
                    "exit_reason": "incomplete_horizon",
                    "label_end_index": None,
                }
            )
            continue

        label = "HOLD"
        exit_reason = "horizon_expiry"
        label_end_index = last_horizon_index
        for candle_index in range(event_index + 1, last_horizon_index + 1):
            candle = normalized.iloc[candle_index]
            take_hit = float(candle["high"]) >= take_profit_price
            stop_hit = float(candle["low"]) <= stop_loss_price
            if stop_hit:
                label = "SELL"
                exit_reason = "stop_loss_collision" if take_hit else "stop_loss"
                label_end_index = candle_index
                break
            if take_hit:
                label = "BUY"
                exit_reason = "take_profit"
                label_end_index = candle_index
                break

        records.append(
            {
                "event_index": event_index,
                "event_timestamp": event_time,
                "entry_price": entry_price,
                "atr": current_atr,
                "take_profit_price": take_profit_price,
                "stop_loss_price": stop_loss_price,
                "label": label,
                "exit_reason": exit_reason,
                "label_end_index": label_end_index,
            }
        )

    return DataFrame.from_records(
        records,
        columns=(
            "event_index",
            "event_timestamp",
            "entry_price",
            "atr",
            "take_profit_price",
            "stop_loss_price",
            "label",
            "exit_reason",
            "label_end_index",
        ),
    )
