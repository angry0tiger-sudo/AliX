"""Small deterministic Data Quality Gate; independent of exchange APIs."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class QualityResult:
    passed: bool
    reasons: tuple[str, ...]


REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "volume")
PRICE_COLUMNS = ("open", "high", "low", "close", "volume")


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _missing_column_reasons(index: int, row: Mapping[str, object]) -> list[str]:
    missing = [column for column in REQUIRED_COLUMNS if column not in row]
    if not missing:
        return []
    return [f"row_{index}_missing_columns:{','.join(missing)}"]


def _price_values(
    index: int, row: Mapping[str, object]
) -> tuple[dict[str, float], list[str]]:
    values: dict[str, float] = {}
    reasons: list[str] = []
    for column in PRICE_COLUMNS:
        value = row[column]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            reasons.append(f"row_{index}_invalid_{column}")
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            reasons.append(f"row_{index}_non_finite_{column}")
            continue
        values[column] = numeric
    return values, reasons


def _price_relationship_reasons(
    index: int, values: Mapping[str, float]
) -> list[str]:
    if len(values) != len(PRICE_COLUMNS):
        return []
    reasons: list[str] = []
    if values["low"] > min(values["open"], values["close"]) or values[
        "high"
    ] < max(values["open"], values["close"]):
        reasons.append(f"row_{index}_invalid_ohlc_range")
    if values["volume"] < 0:
        reasons.append(f"row_{index}_negative_volume")
    return reasons


def _timestamp_reasons(
    index: int, row: Mapping[str, object], previous: float | None
) -> tuple[float | None, list[str]]:
    timestamp = _finite_number(row.get("date"))
    if timestamp is None:
        return None, [f"row_{index}_invalid_date"]
    if previous is not None and timestamp <= previous:
        return timestamp, [f"row_{index}_timestamp_not_strictly_increasing"]
    return timestamp, []


def _is_stale(
    rows: list[Mapping[str, object]], max_age_seconds: int, now_epoch: float
) -> bool:
    timestamp = _finite_number(rows[-1].get("date"))
    return timestamp is None or now_epoch - timestamp > max_age_seconds


def validate_candles(
    rows: Iterable[Mapping[str, object]],
    *,
    max_age_seconds: int | None = None,
    now_epoch: float | None = None,
) -> QualityResult:
    data = list(rows)
    if not data:
        return QualityResult(False, ("empty_dataset",))

    reasons: list[str] = []
    previous_timestamp: float | None = None
    for index, row in enumerate(data):
        missing = _missing_column_reasons(index, row)
        if missing:
            reasons.extend(missing)
            continue
        values, price_reasons = _price_values(index, row)
        reasons.extend(price_reasons)
        reasons.extend(_price_relationship_reasons(index, values))
        timestamp, timestamp_reasons = _timestamp_reasons(
            index, row, previous_timestamp
        )
        reasons.extend(timestamp_reasons)
        if timestamp is not None:
            previous_timestamp = timestamp

    if max_age_seconds is not None and now_epoch is not None:
        if _is_stale(data, max_age_seconds, now_epoch):
            reasons.append("stale_latest_candle")
    return QualityResult(not reasons, tuple(reasons))
