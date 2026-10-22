"""Point-in-time-safe Triple-Barrier outcome labels for offline research.

Future candles are inspected only to create outcomes/labels. This module never
adds those future-derived values to the primary or meta-model feature matrix.
A same-candle take-profit/stop-loss collision resolves conservatively to stop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Final

import numpy as np
import pandas as pd
from pandas import DataFrame

TRIPLE_BARRIER_SCHEMA_VERSION: Final[str] = "alix.meta_labeling.triple-barrier.v1"
_META_EVENT_COLUMNS: Final[tuple[str, ...]] = (
    "event_index",
    "side",
    "target_volatility",
)
_CANDLE_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
_RESULT_COLUMNS: Final[tuple[str, ...]] = (
    "event_index",
    "event_timestamp",
    "entry_price",
    "entry_timestamp",
    "side",
    "target_volatility",
    "barrier_label",
    "exit_reason",
    "exit_price",
    "exit_timestamp",
    "label_end_index",
    "gross_return",
    "fee_return",
    "slippage_return",
    "net_return",
    "meta_label",
)


@dataclass(frozen=True)
class TripleBarrierConfig:
    """Explicit barrier, holding-period and round-trip cost assumptions."""

    profit_taking_multiplier: float = 1.0
    stop_loss_multiplier: float = 1.0
    max_holding_bars: int = 12
    fee_bps_per_side: float = 10.0
    slippage_bps_per_side: float = 5.0
    timeframe_seconds: int = 3600

    def __post_init__(self) -> None:
        numeric_values = (
            self.profit_taking_multiplier,
            self.stop_loss_multiplier,
            self.fee_bps_per_side,
            self.slippage_bps_per_side,
        )
        if any(not math.isfinite(float(value)) for value in numeric_values):
            raise ValueError("barrier configuration values must be finite")
        if self.profit_taking_multiplier <= 0 or self.stop_loss_multiplier <= 0:
            raise ValueError("barrier multipliers must be positive")
        if self.fee_bps_per_side < 0 or self.slippage_bps_per_side < 0:
            raise ValueError("fee and slippage assumptions cannot be negative")
        if (
            isinstance(self.max_holding_bars, bool)
            or not isinstance(self.max_holding_bars, int)
            or self.max_holding_bars < 1
        ):
            raise ValueError("max_holding_bars must be a positive integer")
        if (
            isinstance(self.timeframe_seconds, bool)
            or not isinstance(self.timeframe_seconds, int)
            or self.timeframe_seconds < 1
        ):
            raise ValueError("timeframe_seconds must be a positive integer")

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-compatible, explicit configuration manifest."""
        return asdict(self)


def _validate_candles(candles: DataFrame, timeframe_seconds: int) -> DataFrame:
    missing = sorted(set(_CANDLE_COLUMNS) - set(candles.columns))
    if missing:
        raise ValueError(f"ohlcv_missing_columns:{','.join(missing)}")
    if candles.empty:
        raise ValueError("ohlcv_empty")

    result = candles.loc[:, list(_CANDLE_COLUMNS)].copy().reset_index(drop=True)
    dates = pd.to_datetime(result["date"], errors="raise")
    if getattr(dates.dt, "tz", None) is None:
        raise ValueError("timezone_required")
    result["date"] = dates
    if dates.isna().any():
        raise ValueError("timestamp_invalid")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("timestamps_must_be_unique_and_increasing")
    deltas = dates.diff().iloc[1:].dt.total_seconds().to_numpy(dtype=float)
    if len(deltas) and not np.all(deltas == float(timeframe_seconds)):
        raise ValueError("gap_detected_or_timeframe_mismatch")

    for name in ("open", "high", "low", "close", "volume"):
        values = pd.to_numeric(result[name], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"ohlcv_non_finite:{name}")
        result[name] = values

    opened = result["open"].to_numpy(dtype=float)
    high = result["high"].to_numpy(dtype=float)
    low = result["low"].to_numpy(dtype=float)
    close = result["close"].to_numpy(dtype=float)
    volume = result["volume"].to_numpy(dtype=float)
    if np.any(opened <= 0) or np.any(close <= 0):
        raise ValueError("ohlcv_prices_must_be_positive")
    if np.any(high < low) or np.any(high < np.maximum(opened, close)):
        raise ValueError("ohlcv_high_invalid")
    if np.any(low > np.minimum(opened, close)):
        raise ValueError("ohlcv_low_invalid")
    if np.any(volume < 0):
        raise ValueError("ohlcv_volume_negative")
    return result


def _validate_event(
    row: pd.Series, row_count: int, config: TripleBarrierConfig
) -> tuple[int, int, float]:
    raw_index = row["event_index"]
    if isinstance(raw_index, (bool, np.bool_)) or not isinstance(
        raw_index, (int, float, np.integer, np.floating)
    ):
        raise ValueError("event_index_must_be_integer")
    numeric_index = float(raw_index)
    if not math.isfinite(numeric_index) or not numeric_index.is_integer():
        raise ValueError("event_index_must_be_integer")
    event_index = int(numeric_index)
    if event_index < 0 or event_index + config.max_holding_bars >= row_count:
        raise ValueError("event_horizon_out_of_bounds")

    raw_side = row["side"]
    if isinstance(raw_side, (bool, np.bool_)) or raw_side not in (-1, 1):
        raise ValueError("side_must_be_minus_one_or_plus_one")
    side = int(raw_side)
    volatility = float(row["target_volatility"])
    if not math.isfinite(volatility) or volatility <= 0:
        raise ValueError("target_volatility_must_be_finite_and_positive")
    return event_index, side, volatility


def _barrier_touch(
    candle: pd.Series,
    side: int,
    take_price: float,
    stop_price: float,
) -> str | None:
    high = float(candle["high"])
    low = float(candle["low"])
    if side == 1:
        take_hit = high >= take_price
        stop_hit = low <= stop_price
    else:
        take_hit = low <= take_price
        stop_hit = high >= stop_price
    if stop_hit:
        return "stop_loss"
    if take_hit:
        return "take_profit"
    return None


def _label_one_event(
    candles: DataFrame,
    row: pd.Series,
    config: TripleBarrierConfig,
) -> dict[str, int | float | str | pd.Timestamp]:
    event_index, side, volatility = _validate_event(row, len(candles), config)
    entry_price = float(candles.iloc[event_index]["close"])
    entry_time = pd.Timestamp(candles.iloc[event_index]["date"])
    take_distance = config.profit_taking_multiplier * volatility
    stop_distance = config.stop_loss_multiplier * volatility
    if side == 1:
        take_price = entry_price * math.exp(take_distance)
        stop_price = entry_price * math.exp(-stop_distance)
    else:
        take_price = entry_price * math.exp(-take_distance)
        stop_price = entry_price * math.exp(stop_distance)

    exit_index = event_index + config.max_holding_bars
    exit_price = float(candles.iloc[exit_index]["close"])
    exit_reason = "vertical"
    for candle_index in range(event_index + 1, exit_index + 1):
        touch = _barrier_touch(candles.iloc[candle_index], side, take_price, stop_price)
        if touch is None:
            continue
        exit_index = candle_index
        exit_reason = touch
        exit_price = stop_price if touch == "stop_loss" else take_price
        break

    barrier_label = {
        "take_profit": 1,
        "stop_loss": -1,
        "vertical": 0,
    }[exit_reason]
    gross_return = side * (exit_price / entry_price - 1.0)
    fee_return = 2.0 * config.fee_bps_per_side / 10_000.0
    slippage_return = 2.0 * config.slippage_bps_per_side / 10_000.0
    net_return = gross_return - fee_return - slippage_return
    return {
        "event_index": event_index,
        "event_timestamp": entry_time,
        "entry_price": entry_price,
        "entry_timestamp": entry_time,
        "side": side,
        "target_volatility": volatility,
        "barrier_label": barrier_label,
        "exit_reason": exit_reason,
        "exit_price": exit_price,
        "exit_timestamp": pd.Timestamp(candles.iloc[exit_index]["date"]),
        "label_end_index": exit_index,
        "gross_return": gross_return,
        "fee_return": fee_return,
        "slippage_return": slippage_return,
        "net_return": net_return,
        "meta_label": int(net_return > 0.0),
    }


def apply_triple_barrier(
    candles: DataFrame,
    events: DataFrame,
    config: TripleBarrierConfig | None = None,
) -> DataFrame:
    """Label events by first PT/SL touch or vertical timeout, for research only.

    Events start at the close of event_index and are evaluated from the next
    candle. Barrier width is a positive historical volatility target in log-return
    units. Labels are outcomes, not features or trading instructions.
    """
    selected = config or TripleBarrierConfig()
    if not set(_META_EVENT_COLUMNS).issubset(events.columns):
        missing = sorted(set(_META_EVENT_COLUMNS) - set(events.columns))
        raise ValueError(f"events_missing_columns:{','.join(missing)}")
    if events.empty:
        return DataFrame(columns=list(_RESULT_COLUMNS))
    normalized = _validate_candles(candles, selected.timeframe_seconds)
    results: list[dict[str, int | float | str | pd.Timestamp]] = []
    seen: set[int] = set()
    for _, row in events.loc[:, list(_META_EVENT_COLUMNS)].iterrows():
        event_index, _, _ = _validate_event(row, len(normalized), selected)
        if event_index in seen:
            raise ValueError("duplicate_event_index")
        seen.add(event_index)
        results.append(_label_one_event(normalized, row, selected))
    return DataFrame.from_records(results, columns=list(_RESULT_COLUMNS))
