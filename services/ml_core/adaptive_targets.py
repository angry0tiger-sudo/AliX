"""ATR-adaptive Triple-Barrier labels for offline research only.

The default target geometry is absolute-price ATR(14): TP = 2 * ATR and
SL = 1 * ATR over the next 24 candles. Horizon and both barrier multipliers
are explicit per-run inputs. When TP and SL are both touched on one candle,
stop-loss wins. Incomplete ATR history or future horizon is left unlabeled.
This module does not create features, train models, or place orders.
"""

from __future__ import annotations

from math import isfinite
from typing import Final

import numpy as np
import pandas as pd
from pandas import DataFrame

ATR_PERIOD: Final[int] = 14
DOCUMENTED_BINANCE_SPOT_GAP_START_UTC: Final[str] = "2023-03-24T12:40:00Z"
DOCUMENTED_BINANCE_SPOT_GAP_END_UTC: Final[str] = "2023-03-24T13:59:00Z"
TAKE_PROFIT_MULTIPLIER_DEFAULT: Final[float] = 2.0
STOP_LOSS_MULTIPLIER_DEFAULT: Final[float] = 1.0
LABEL_HORIZON_CANDLES: Final[int] = 24
# Backward-compatible constant aliases.
K_UPPER_DEFAULT: Final[float] = TAKE_PROFIT_MULTIPLIER_DEFAULT
K_LOWER_DEFAULT: Final[float] = STOP_LOSS_MULTIPLIER_DEFAULT
EXIT_REASONS: Final[tuple[str, ...]] = (
    "TP_HIT",
    "SL_HIT",
    "COLLISION",
    "HORIZON_EXPIRY",
)
UNLABELED_REASONS: Final[tuple[str, ...]] = (
    "insufficient_atr_history",
    "incomplete_horizon",
    "common_24_candle_tail",
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


def _parse_documented_gap_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    """Parse a timezone-aware documented-gap endpoint as UTC."""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("timestamp_documented_gap_manifest_invalid") from None
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("timestamp_documented_gap_manifest_invalid")
    return timestamp.tz_convert("UTC")


def _validate_documented_gap_range(
    gap: tuple[str | pd.Timestamp, str | pd.Timestamp]
    | list[str | pd.Timestamp],
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Validate one exact, inclusive 80-minute Binance Spot gap declaration."""
    if not isinstance(gap, (tuple, list)) or len(gap) != 2:
        raise ValueError("timestamp_documented_gap_manifest_invalid")
    start = _parse_documented_gap_timestamp(gap[0])
    end = _parse_documented_gap_timestamp(gap[1])
    if start != pd.Timestamp(DOCUMENTED_BINANCE_SPOT_GAP_START_UTC):
        raise ValueError("timestamp_documented_gap_manifest_invalid")
    if end != pd.Timestamp(DOCUMENTED_BINANCE_SPOT_GAP_END_UTC):
        raise ValueError("timestamp_documented_gap_manifest_invalid")
    if end - start != pd.Timedelta(minutes=79):
        raise ValueError("timestamp_documented_gap_manifest_invalid")
    return start, end


def _validated_documented_gap_ranges(
    documented_gap_ranges: tuple[tuple[str | pd.Timestamp, str | pd.Timestamp], ...]
    | list[tuple[str | pd.Timestamp, str | pd.Timestamp]],
) -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
    """Accept only the exact, single documented Binance Spot gap."""
    if not isinstance(documented_gap_ranges, (tuple, list)):
        raise ValueError("timestamp_documented_gap_manifest_invalid")
    if len(documented_gap_ranges) > 1:
        raise ValueError("timestamp_documented_gap_manifest_invalid")
    return tuple(_validate_documented_gap_range(gap) for gap in documented_gap_ranges)


def _validate_inferred_timestamp_cadence(
    deltas: pd.Series,
    documented_gaps: tuple[tuple[pd.Timestamp, pd.Timestamp], ...],
) -> None:
    """Preserve the legacy fixed-cadence contract when no interval is supplied."""
    if documented_gaps:
        raise ValueError("timestamp_expected_interval_required")
    seconds = deltas.dt.total_seconds().to_numpy(dtype=float)
    if not bool(np.isfinite(seconds).all()):
        raise ValueError("timestamp_interval_inconsistent")
    if bool(np.any(seconds <= 0)):
        raise ValueError("timestamp_interval_inconsistent")
    if len(seconds) and not bool(np.all(seconds == seconds[0])):
        raise ValueError("timestamp_interval_inconsistent")


def _validated_expected_interval(value: pd.Timedelta) -> pd.Timedelta:
    """Require a positive expected interval aligned to whole minutes."""
    try:
        interval = pd.Timedelta(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("timestamp_expected_interval_invalid") from None
    if pd.isna(interval) or interval <= pd.Timedelta(0):
        raise ValueError("timestamp_expected_interval_invalid")
    minutes = interval / pd.Timedelta(minutes=1)
    if not float(minutes).is_integer():
        raise ValueError("timestamp_expected_interval_invalid")
    return interval


def _allowed_gap_transition_indices(
    dates: pd.Series,
    interval: pd.Timedelta,
    documented_gaps: tuple[tuple[pd.Timestamp, pd.Timestamp], ...],
) -> tuple[int, ...]:
    """Map the source manifest gap to its epoch-aligned resampled-bar boundary."""
    frequency = f"{int(interval / pd.Timedelta(minutes=1))}min"
    allowed_indices: list[int] = []
    for gap_start, gap_end in documented_gaps:
        first_bucket = gap_start.floor(frequency)
        last_bucket = gap_end.floor(frequency)
        expected_previous = first_bucket - interval
        expected_next = last_bucket + interval
        matches = [
            index
            for index in range(1, len(dates))
            if pd.Timestamp(dates.iloc[index - 1]) == expected_previous
            and pd.Timestamp(dates.iloc[index]) == expected_next
        ]
        if len(matches) != 1:
            raise ValueError("timestamp_documented_gap_boundary_mismatch")
        allowed_indices.append(matches[0])
    if len(set(allowed_indices)) != len(allowed_indices):
        raise ValueError("timestamp_documented_gap_manifest_invalid")
    return tuple(allowed_indices)


def _observed_gap_transition_indices(
    deltas: pd.Series, expected_interval: pd.Timedelta
) -> tuple[int, ...]:
    """Return every observed cadence break as the index of its post-gap candle."""
    return tuple(
        index
        for index, delta in enumerate(deltas, start=1)
        if delta != expected_interval
    )


def _validate_gap_transition_indices(
    observed_indices: tuple[int, ...], allowed_indices: tuple[int, ...]
) -> None:
    """Require the observed discontinuities to match the manifest exactly."""
    if set(observed_indices) != set(allowed_indices):
        raise ValueError("timestamp_interval_inconsistent")


def _gap_transition_indices(
    dates: pd.Series,
    *,
    expected_interval: pd.Timedelta | None,
    documented_gap_ranges: tuple[tuple[str | pd.Timestamp, str | pd.Timestamp], ...]
    | list[tuple[str | pd.Timestamp, str | pd.Timestamp]],
) -> tuple[int, ...]:
    """Validate regular cadence plus only manifest-authorized segment transitions."""
    documented_gaps = _validated_documented_gap_ranges(documented_gap_ranges)
    deltas = dates.diff().iloc[1:]
    if expected_interval is None:
        _validate_inferred_timestamp_cadence(deltas, documented_gaps)
        return ()
    interval = _validated_expected_interval(expected_interval)
    allowed_indices = _allowed_gap_transition_indices(
        dates, interval, documented_gaps
    )
    observed_indices = _observed_gap_transition_indices(deltas, interval)
    _validate_gap_transition_indices(observed_indices, allowed_indices)
    return tuple(sorted(allowed_indices))


def _validate_timestamps(
    values: pd.Series,
    *,
    expected_interval: pd.Timedelta | None = None,
    documented_gap_ranges: tuple[tuple[str | pd.Timestamp, str | pd.Timestamp], ...]
    | list[tuple[str | pd.Timestamp, str | pd.Timestamp]] = (),
) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce")
    if dates.isna().any():
        raise ValueError("timestamp_invalid")
    if dates.dt.tz is None:
        raise ValueError("timezone_required")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("timestamps_must_be_unique_and_increasing")
    normalized = dates.dt.tz_convert("UTC")
    _gap_transition_indices(
        normalized,
        expected_interval=expected_interval,
        documented_gap_ranges=documented_gap_ranges,
    )
    return normalized


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


def _validated_ohlcv(
    candles: DataFrame,
    *,
    expected_interval: pd.Timedelta | None = None,
    documented_gap_ranges: tuple[tuple[str | pd.Timestamp, str | pd.Timestamp], ...]
    | list[tuple[str | pd.Timestamp, str | pd.Timestamp]] = (),
) -> DataFrame:
    missing = sorted(set(_OHLCV_COLUMNS) - set(candles.columns))
    if missing:
        raise ValueError(f"ohlcv_missing_columns:{','.join(missing)}")
    if candles.empty:
        raise ValueError("ohlcv_empty")

    result = candles.loc[:, list(_OHLCV_COLUMNS)].copy().reset_index(drop=True)
    result["date"] = _validate_timestamps(
        result["date"],
        expected_interval=expected_interval,
        documented_gap_ranges=documented_gap_ranges,
    )
    for name in _PRICE_COLUMNS:
        result[name] = _coerce_numeric_column(result, name)
    _validate_price_geometry(result)
    return result


def calculate_atr(
    candles: DataFrame,
    period: int = ATR_PERIOD,
    *,
    expected_interval: pd.Timedelta | None = None,
    documented_gap_ranges: tuple[tuple[str | pd.Timestamp, str | pd.Timestamp], ...]
    | list[tuple[str | pd.Timestamp, str | pd.Timestamp]] = (),
) -> pd.Series:
    """Calculate causal ATR independently inside each manifest-validated segment.

    At a documented source gap, the first bar's true range uses high-low only,
    and the ATR rolling window restarts. It remains NaN until `period` complete
    post-gap bars are available. No candle is synthesized or interpolated.
    """
    _positive_integer(period, "atr_period")
    normalized = _validated_ohlcv(
        candles,
        expected_interval=expected_interval,
        documented_gap_ranges=documented_gap_ranges,
    )
    gap_indices = _gap_transition_indices(
        normalized["date"],
        expected_interval=expected_interval,
        documented_gap_ranges=documented_gap_ranges,
    )
    high = normalized["high"]
    low = normalized["low"]
    close = normalized["close"]
    previous_close = close.shift(1)
    for gap_index in gap_indices:
        previous_close.iloc[gap_index] = np.nan
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=True)

    atr = pd.Series(np.nan, index=normalized.index, dtype=float)
    segment_starts = (0, *gap_indices)
    segment_ends = (*gap_indices, len(normalized))
    for segment_start, segment_end in zip(segment_starts, segment_ends):
        segment_true_range = true_range.iloc[segment_start:segment_end]
        atr.iloc[segment_start:segment_end] = segment_true_range.rolling(
            window=period, min_periods=period
        ).mean()
    return atr.rename(f"atr_{period}")


def _validate_positive_barrier_multiplier(value: float, reason: str) -> None:
    if not isfinite(float(value)):
        raise ValueError(reason)
    if value <= 0:
        raise ValueError(reason)


def _validate_entry_reference(entry_reference: str) -> None:
    if entry_reference not in {"SIGNAL_CANDLE_CLOSE", "NEXT_BAR_OPEN"}:
        raise ValueError("entry_reference_unsupported")


def _validate_entry_slippage(entry_slippage_bps: float) -> None:
    if isinstance(entry_slippage_bps, bool):
        raise ValueError("entry_slippage_bps_must_be_finite_and_nonnegative")
    if not isfinite(float(entry_slippage_bps)):
        raise ValueError("entry_slippage_bps_must_be_finite_and_nonnegative")
    if entry_slippage_bps < 0:
        raise ValueError("entry_slippage_bps_must_be_finite_and_nonnegative")


def _validate_entry_reference_slippage_compatibility(
    entry_reference: str, entry_slippage_bps: float
) -> None:
    if entry_reference == "SIGNAL_CANDLE_CLOSE" and entry_slippage_bps != 0:
        raise ValueError("entry_slippage_requires_next_bar_open_reference")


def _validate_barrier_configuration(
    atr_period: int,
    horizon_candles: int,
    take_profit_multiplier: float,
    stop_loss_multiplier: float,
    entry_reference: str,
    entry_slippage_bps: float,
) -> None:
    _positive_integer(atr_period, "atr_period")
    _positive_integer(horizon_candles, "horizon_candles")
    _validate_positive_barrier_multiplier(
        take_profit_multiplier,
        "take_profit_multiplier_must_be_finite_and_positive",
    )
    _validate_positive_barrier_multiplier(
        stop_loss_multiplier,
        "stop_loss_multiplier_must_be_finite_and_positive",
    )
    _validate_entry_reference(entry_reference)
    _validate_entry_slippage(entry_slippage_bps)
    _validate_entry_reference_slippage_compatibility(
        entry_reference, entry_slippage_bps
    )


def _record(
    event_index: int,
    event_time: pd.Timestamp,
    entry_price: float | None,
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
    event_index: int,
    event_time: pd.Timestamp,
    entry_price: float | None,
    reason: str,
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
    take_profit_multiplier: float,
    stop_loss_multiplier: float,
) -> tuple[float, float]:
    take_profit_price = entry_price + take_profit_multiplier * atr
    stop_loss_price = entry_price - stop_loss_multiplier * atr
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
    horizon_candles: int,
) -> TargetRecord:
    last_horizon_index = event_index + horizon_candles
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
    horizon_candles: int,
    take_profit_multiplier: float,
    stop_loss_multiplier: float,
    entry_reference: str,
    entry_slippage_bps: float,
) -> TargetRecord:
    signal_row = normalized.iloc[event_index]
    event_time = pd.Timestamp(signal_row["date"])
    entry_price: float | None
    if entry_reference == "NEXT_BAR_OPEN":
        if event_index + 1 >= len(normalized):
            return _unlabeled_record(
                event_index, event_time, None, "incomplete_horizon"
            )
        next_bar_open = float(normalized.iloc[event_index + 1]["open"])
        entry_price = next_bar_open * (1.0 + entry_slippage_bps / 10_000.0)
    else:
        entry_price = float(signal_row["close"])
    if not isfinite(entry_price) or entry_price <= 0:
        raise ValueError("label_entry_price_invalid")

    if computed_atr and event_index < atr_period - 1:
        return _unlabeled_record(
            event_index, event_time, entry_price, "insufficient_atr_history"
        )

    # ATR is read at signal index t; no t+1 OHLC enters the ATR calculation.
    atr = _validated_atr_value(float(atr_values[event_index]))
    take_profit_price, stop_loss_price = _barrier_prices(
        entry_price, atr, take_profit_multiplier, stop_loss_multiplier
    )
    return _resolve_event_horizon(
        normalized,
        event_index,
        event_time,
        entry_price,
        atr,
        take_profit_price,
        stop_loss_price,
        horizon_candles,
    )


def apply_adaptive_triple_barrier(
    candles: DataFrame,
    *,
    atr_values: pd.Series | None = None,
    atr_period: int = ATR_PERIOD,
    horizon_candles: int = LABEL_HORIZON_CANDLES,
    take_profit_multiplier: float = TAKE_PROFIT_MULTIPLIER_DEFAULT,
    stop_loss_multiplier: float = STOP_LOSS_MULTIPLIER_DEFAULT,
    entry_reference: str = "SIGNAL_CANDLE_CLOSE",
    entry_slippage_bps: float = 0.0,
) -> DataFrame:
    """Create BUY/SELL/HOLD labels with an explicit causal entry reference.

    ATR[t] uses OHLCV only through signal candle t. The default preserves the
    historical Close[t] contract. V4-Q selects NEXT_BAR_OPEN and applies the
    same adverse long-entry slippage as the economic simulator. Future bars are
    inspected only to determine the label; incomplete horizons are unlabeled.
    """
    _validate_barrier_configuration(
        atr_period,
        horizon_candles,
        take_profit_multiplier,
        stop_loss_multiplier,
        entry_reference,
        entry_slippage_bps,
    )
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
            horizon_candles=horizon_candles,
            take_profit_multiplier=take_profit_multiplier,
            stop_loss_multiplier=stop_loss_multiplier,
            entry_reference=entry_reference,
            entry_slippage_bps=entry_slippage_bps,
        )
        for event_index in range(len(normalized))
    ]
    return DataFrame.from_records(records, columns=list(_RESULT_COLUMNS))
