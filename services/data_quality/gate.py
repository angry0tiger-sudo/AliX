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


def validate_candles(
    rows: Iterable[Mapping[str, object]],
    *,
    max_age_seconds: int | None = None,
    now_epoch: float | None = None,
) -> QualityResult:
    data = list(rows)
    reasons: list[str] = []
    if not data:
        return QualityResult(False, ("empty_dataset",))

    previous_timestamp: float | None = None
    for index, row in enumerate(data):
        missing = [column for column in REQUIRED_COLUMNS if column not in row]
        if missing:
            reasons.append(f"row_{index}_missing_columns:{','.join(missing)}")
            continue

        numbers: dict[str, float] = {}
        for column in PRICE_COLUMNS:
            value = row[column]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                reasons.append(f"row_{index}_invalid_{column}")
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                reasons.append(f"row_{index}_non_finite_{column}")
                continue
            numbers[column] = numeric

        if len(numbers) == len(PRICE_COLUMNS):
            if numbers["low"] > min(numbers["open"], numbers["close"]) or numbers[
                "high"
            ] < max(numbers["open"], numbers["close"]):
                reasons.append(f"row_{index}_invalid_ohlc_range")
            if numbers["volume"] < 0:
                reasons.append(f"row_{index}_negative_volume")

        timestamp = _finite_number(row.get("date"))
        if timestamp is None:
            reasons.append(f"row_{index}_invalid_date")
        else:
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                reasons.append(
                    f"row_{index}_timestamp_not_strictly_increasing"
                )
            previous_timestamp = timestamp

    if max_age_seconds is not None and now_epoch is not None:
        last_timestamp = _finite_number(data[-1].get("date"))
        if (
            last_timestamp is None
            or now_epoch - last_timestamp > max_age_seconds
        ):
            reasons.append("stale_latest_candle")

    return QualityResult(not reasons, tuple(reasons))
