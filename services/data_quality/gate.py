"""Small deterministic Data Quality Gate; independent of exchange APIs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class QualityResult:
    passed: bool
    reasons: tuple[str, ...]


REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "volume")


def validate_candles(
    rows: Iterable[dict],
    *,
    max_age_seconds: int | None = None,
    now_epoch: float | None = None,
) -> QualityResult:
    data = list(rows)
    reasons: list[str] = []
    if not data:
        return QualityResult(False, ("empty_dataset",))
    for index, row in enumerate(data):
        missing = [column for column in REQUIRED_COLUMNS if column not in row]
        if missing:
            reasons.append(f"row_{index}_missing_columns:{','.join(missing)}")
            continue
        for column in ("open", "high", "low", "close", "volume"):
            value = row[column]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                reasons.append(f"row_{index}_invalid_{column}")
            elif value != value or value in (float("inf"), float("-inf")):
                reasons.append(f"row_{index}_non_finite_{column}")
        numeric = all(
            isinstance(row.get(column), (int, float))
            and not isinstance(row.get(column), bool)
            for column in ("open", "high", "low", "close", "volume")
        )
        if numeric:
            if row["low"] > min(row["open"], row["close"]) or row["high"] < max(
                row["open"], row["close"]
            ):
                reasons.append(f"row_{index}_invalid_ohlc_range")
            if row["volume"] < 0:
                reasons.append(f"row_{index}_negative_volume")
        if index and row.get("date") <= data[index - 1].get("date"):
            reasons.append(f"row_{index}_timestamp_not_strictly_increasing")
    if max_age_seconds is not None and now_epoch is not None:
        last = data[-1].get("date")
        if not isinstance(last, (int, float)) or now_epoch - last > max_age_seconds:
            reasons.append("stale_latest_candle")
    return QualityResult(not reasons, tuple(reasons))
