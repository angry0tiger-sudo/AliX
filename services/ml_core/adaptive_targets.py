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
K_UPPER_DEFAULT: Final[float] = 2.0
K_LOWER_DEFAULT: Final[float] = 1.0
LABEL_HORIZON_CANDLES: Final[int] = 24
EXIT_REASONS: Final[tuple[str, ...]] = (
    "TP_HIT",
    "SL_HIT",
    "COLLISION",
    "HORIZON_EXPIRY",
)
UNLABELED_REASONS: Final[tuple[str, ...]] = (
    "insufficient_atr_history",
    "incomplete_horizon",
)

_OHLCV_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
_PRICE_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close", "volume")
_RESULT_COLUMNS: Final[tuple[str, ...]] = (
    "event_index",
    "event_timestamp",
    "entry_price",
    "atr",
    "take_profit_price",
    "stop_loss_price",
    "label",
    "exit_reason",
    "unlabeled_reason",
    "label_end_idx",
)
TargetRecord = dict[str, int | float | str | None | pd.Timestamp]


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name}_must_be_positive_integer")


def _validate_timestamps(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce")
    if dates.isna().any():
        raise ValueError("timestamp_invalid")
    if dates.dt.tz is None:
        raise ValueError("timezone_required")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("timestamps_must_be_unique_and_increasing")
    deltas = dates.diff().iloc[1:].dt.total_seconds().to_numpy(dtype=float)
    if len(deltas) and not _interval_deltas_are_consistent(deltas):
        raise ValueError("timestamp_interval_inconsistent")
    return dates.dt.tz_convert("UTC")


def _interval_deltas_are_consistent(deltas: np.ndarray) -> bool:
    return (
        bool(np.isfinite(deltas).all())
        and not bool(np.any(deltas <= 0))
        and bool(np.all(deltas == deltas[0]))
    )


def _coerce_numeric_column(result: DataFrame, name: str) -> pd.Series:
    values = pd.to_numeric(result[name], errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"ohlcv_non_finite:{name}")
    return values


def _validate_price_geometry(result: DataFrame) -> None:
    opened = result["open"].to_numpy(dtype=float)
    high = result["high"].to_numpy(dtype=float)
    low = result["low"].to_numpy(dtype=float)
    close = result["close"].to_numpy(dtype=float)
    volume = result["volume"].to_numpy(dtype=float)
    prices = (opened, high, low, close)
    if any(bool(np.any(values <= 0)) for values in prices):
        raise ValueError("ohlcv_prices_must_be_positive")
    if np.any(volume < 0):
        raise ValueError("ohlcv_volume_negative")
    if _has_invalid_ohlc_geometry(opened, high, low, close):
        raise ValueError("ohlcv_geometry_invalid")


def _has_invalid_ohlc_geometry(
    opened: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> bool:
    return bool(
        np.any(high < low)
        or np.any(high < np.maximum(opened, close))
        or np.any(low > np.minimum(opened, close))
    )


def _validated_ohlcv(candles: DataFrame) -> DataFrame:
    missing = sorted(set(_OHLCV_COLUMNS) - set(candles.columns))
    if missing:
        raise ValueError(f"ohlcv_missing_columns:{','.join(missing)}")
    if candles.empty:
        raise ValueError("ohlcv_empty")

    result = candles.loc[:, list(_OHLCV_COLUMNS)].copy().reset_index(drop=True)
    result["date"] = _validate_timestamps(result["date"])
    for name in _PRICE_COLUMNS:
        result[name] = _coerce_numeric_column(result, name)
    _validate_price_geometry(result)
    return result


def calculate_atr(candles: DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """Calculate a full-window simple ATR mean from absolute true ranges.

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
    k_upper: float,
    k_lower: float,
) -> None:
    _positive_integer(atr_period, "atr_period")
    _positive_integer(horizon, "horizon")
    if (
        not isfinite(float(k_upper))
        or not isfinite(float(k_lower))
        or k_upper <= 0
        or k_lower <= 0
    ):
        raise ValueError("barrier_multipliers_must_be_finite_and_positive")


def _record(
    event_index: int,
    event_time: pd.Timestamp,
    entry_price: float,
    atr: float | None,
    take_profit_price: float | None,
    stop_loss_price: float | None,
    label: str | None,
    exit_reason: str | None,
    label_end_idx: int | None,
    unlabeled_reason: str | None = None,
) -> TargetRecord:
    return {
        "event_index": event_index,
        "event_timestamp": event_time,
        "entry_price": entry_price,
        "atr": atr,
        "take_profit_price": take_profit_price,
        "stop_loss_price": stop_loss_price,
        "label": label,
        "exit_reason": exit_reason,
        "unlabeled_reason": unlabeled_reason,
        "label_end_idx": label_end_idx,
    }


def _unlabeled_record(
    event_index: int, event_time: pd.Timestamp, entry_price: float, reason: str
) -> TargetRecord:
    return _record(
        event_index,
        event_time,
        entry_price,
        None,
        None,
        None,
        None,
        None,
        None,
        reason,
    )


def _validated_atr_value(raw_atr: float) -> float:
    if not isfinite(raw_atr):
        raise ValueError("atr_must_be_finite")
    if raw_atr <= 0:
        raise ValueError("atr_must_be_positive")
    return raw_atr


def _barrier_prices(
    entry_price: float,
    atr: float,
    k_upper: float,
    k_lower: float,
) -> tuple[float, float]:
    take_profit_price = entry_price + k_upper * atr
    stop_loss_price = entry_price - k_lower * atr
    if not isfinite(take_profit_price) or not isfinite(stop_loss_price):
        raise ValueError("atr_barrier_price_not_finite")
    if stop_loss_price <= 0:
        raise ValueError("atr_stop_price_not_positive")
    return take_profit_price, stop_loss_price


def _resolve_event_horizon(
    normalized: DataFrame,
    event_index: int,
    entry_time: pd.Timestamp,
    entry_price: float,
    atr: float,
    take_profit_price: float,
    stop_loss_price: float,
    horizon: int,
) -> TargetRecord:
    last_horizon_index = event_index + horizon
    if last_horizon_index >= len(normalized):
        return _record(
            event_index,
            entry_time,
            entry_price,
            atr,
            take_profit_price,
            stop_loss_price,
            None,
            None,
            None,
            "incomplete_horizon",
        )

    label, exit_reason, label_end_idx = _first_barrier_outcome(
        normalized, event_index, last_horizon_index, take_profit_price, stop_loss_price
    )
    return _record(
        event_index,
        entry_time,
        entry_price,
        atr,
        take_profit_price,
        stop_loss_price,
        label,
        exit_reason,
        label_end_idx,
    )


def _first_barrier_outcome(
    normalized: DataFrame,
    event_index: int,
    last_horizon_index: int,
    take_profit_price: float,
    stop_loss_price: float,
) -> tuple[str, str, int]:
    for candle_index in range(event_index + 1, last_horizon_index + 1):
        candle = normalized.iloc[candle_index]
        take_hit = float(candle["high"]) >= take_profit_price
        stop_hit = float(candle["low"]) <= stop_loss_price
        if stop_hit:
            reason = "COLLISION" if take_hit else "SL_HIT"
            return "SELL", reason, candle_index
        if take_hit:
            return "BUY", "TP_HIT", candle_index
    return "HOLD", "HORIZON_EXPIRY", last_horizon_index


def _label_one_event(
    normalized: DataFrame,
    event_index: int,
    atr_values: np.ndarray,
    *,
    computed_atr: bool,
    atr_period: int,
    horizon: int,
    k_upper: float,
    k_lower: float,
) -> TargetRecord:
    entry_row = normalized.iloc[event_index]
    entry_price = float(entry_row["close"])
    event_time = pd.Timestamp(entry_row["date"])
    if computed_atr and event_index < atr_period - 1:
        return _unlabeled_record(
            event_index, event_time, entry_price, "insufficient_atr_history"
        )

    atr = _validated_atr_value(float(atr_values[event_index]))
    take_profit_price, stop_loss_price = _barrier_prices(
        entry_price, atr, k_upper, k_lower
    )
    return _resolve_event_horizon(
        normalized,
        event_index,
        event_time,
        entry_price,
        atr,
        take_profit_price,
        stop_loss_price,
        horizon,
    )


def apply_adaptive_triple_barrier(
    candles: DataFrame,
    *,
    atr_values: pd.Series | None = None,
    atr_period: int = ATR_PERIOD,
    horizon: int = LABEL_HORIZON_CANDLES,
    k_upper: float = K_UPPER_DEFAULT,
    k_lower: float = K_LOWER_DEFAULT,
) -> DataFrame:
    """Create BUY/SELL/HOLD labels while keeping future outcomes out of features.

    At index t, entry is close[t] and ATR[t] uses only OHLCV through t.
    Future OHLCV at t+1 ... t+horizon is inspected only to determine the label.
    Stop-loss wins when both barriers are hit on the same candle. Tail rows
    without a complete future horizon are intentionally left unlabeled.
    """
    _validate_barrier_configuration(atr_period, horizon, k_upper, k_lower)
    normalized = _validated_ohlcv(candles)
    computed_atr = atr_values is None
    atr_series = calculate_atr(normalized, atr_period) if computed_atr else atr_values
    if atr_series is None or len(atr_series) != len(normalized):
        raise ValueError("atr_length_mismatch")
    atr_array = pd.to_numeric(atr_series, errors="coerce").to_numpy(dtype=float)

    records = [
        _label_one_event(
            normalized,
            event_index,
            atr_array,
            computed_atr=computed_atr,
            atr_period=atr_period,
            horizon=horizon,
            k_upper=k_upper,
            k_lower=k_lower,
        )
        for event_index in range(len(normalized))
    ]
    return DataFrame.from_records(records, columns=list(_RESULT_COLUMNS))
