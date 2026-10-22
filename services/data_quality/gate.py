"""Fail-closed, deterministic data-quality contracts; no provider or model I/O."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal, cast

OHLCV_SCHEMA_VERSION = "ohlcv.v1"
FEATURE_SCHEMA_VERSION = "feature-vector.v1"
EXTERNAL_RECORD_SCHEMA_VERSION = "external-record.v1"
MODEL_OUTPUT_SCHEMA_VERSION = "model-output.v1"

REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "volume")
PRICE_COLUMNS = ("open", "high", "low", "close", "volume")


class QualityStatus(StrEnum):
    VALID = "VALID"
    DEGRADED_ALLOWED = "DEGRADED_ALLOWED"
    REJECTED = "REJECTED"


class QualityReasonCode(StrEnum):
    EMPTY_DATASET = "empty_dataset"
    UNSUPPORTED_TIMEFRAME = "unsupported_timeframe"
    MISSING_COLUMNS = "missing_columns"
    INVALID_TIMESTAMP = "invalid_timestamp"
    TIMEZONE_REQUIRED = "timezone_required"
    DUPLICATE_TIMESTAMP = "duplicate_timestamp"
    TIMESTAMP_NOT_INCREASING = "timestamp_not_strictly_increasing"
    DATA_GAP = "gap_detected"
    STALE_DATA = "stale_latest_candle"
    FUTURE_TIMESTAMP = "future_timestamp"
    INVALID_NUMERIC_VALUE = "invalid_numeric_value"
    NON_FINITE_VALUE = "non_finite_value"
    INVALID_PRICE_RANGE = "invalid_price_range"
    OHLC_INVARIANT_VIOLATION = "invalid_ohlc_range"
    NEGATIVE_VOLUME = "negative_volume"
    SCHEMA_VERSION_MISMATCH = "schema_version_mismatch"
    MISSING_FEATURES = "missing_features"
    UNEXPECTED_FEATURES = "unexpected_features"
    MISSING_TIMESTAMP_METADATA = "missing_timestamp_metadata"
    FUTURE_DATA_LEAKAGE = "future_data_leakage"
    MISSING_PROVENANCE = "missing_provenance"
    SOURCE_NOT_APPROVED = "source_not_approved"
    RATE_LIMIT_BLOCKED = "rate_limit_blocked"
    RATE_LIMIT_LIMITED = "rate_limit_limited"
    INVALID_RATE_LIMIT_STATUS = "invalid_rate_limit_status"
    INVALID_TIMESTAMP_ORDER = "invalid_timestamp_order"
    INVALID_MODEL_OUTPUT = "invalid_model_output"


@dataclass(frozen=True, slots=True)
class QualityResult:
    passed: bool
    reasons: tuple[str, ...]
    status: QualityStatus | None = None
    reason_codes: tuple[QualityReasonCode, ...] = ()
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        status = self.status
        if status is None:
            status = (
                QualityStatus.VALID
                if self.passed and not self.reasons
                else QualityStatus.DEGRADED_ALLOWED
                if self.passed
                else QualityStatus.REJECTED
            )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "passed", status is not QualityStatus.REJECTED)
        if not self.reason_codes and self.reasons:
            codes = tuple(
                dict.fromkeys(_reason_code(reason) for reason in self.reasons)
            )
            object.__setattr__(self, "reason_codes", codes)


_REASON_PREFIXES: tuple[tuple[str, QualityReasonCode], ...] = (
    ("empty_dataset", QualityReasonCode.EMPTY_DATASET),
    ("unsupported_timeframe", QualityReasonCode.UNSUPPORTED_TIMEFRAME),
    ("missing_columns", QualityReasonCode.MISSING_COLUMNS),
    ("missing_features", QualityReasonCode.MISSING_FEATURES),
    ("unexpected_features", QualityReasonCode.UNEXPECTED_FEATURES),
    ("timestamp_timezone_missing", QualityReasonCode.TIMEZONE_REQUIRED),
    ("timezone_required", QualityReasonCode.TIMEZONE_REQUIRED),
    ("invalid_price_range", QualityReasonCode.INVALID_PRICE_RANGE),
    ("invalid_ohlc_range", QualityReasonCode.OHLC_INVARIANT_VIOLATION),
    ("invalid_rate_limit_status", QualityReasonCode.INVALID_RATE_LIMIT_STATUS),
    ("invalid_timestamp_order", QualityReasonCode.INVALID_TIMESTAMP_ORDER),
    ("invalid_model_output", QualityReasonCode.INVALID_MODEL_OUTPUT),
    ("invalid_available_at", QualityReasonCode.INVALID_TIMESTAMP),
    ("invalid_decision_timestamp", QualityReasonCode.INVALID_TIMESTAMP),
    ("invalid_event_timestamp", QualityReasonCode.INVALID_TIMESTAMP),
    ("invalid_ingested_at", QualityReasonCode.INVALID_TIMESTAMP),
    ("invalid_produced_at", QualityReasonCode.INVALID_TIMESTAMP),
    ("invalid_date", QualityReasonCode.INVALID_TIMESTAMP),
    ("invalid_timestamp", QualityReasonCode.INVALID_TIMESTAMP),
    ("duplicate_timestamp", QualityReasonCode.DUPLICATE_TIMESTAMP),
    (
        "timestamp_not_strictly_increasing",
        QualityReasonCode.TIMESTAMP_NOT_INCREASING,
    ),
    ("gap_detected", QualityReasonCode.DATA_GAP),
    ("stale_latest_candle", QualityReasonCode.STALE_DATA),
    ("future_timestamp", QualityReasonCode.FUTURE_TIMESTAMP),
    ("non_finite_", QualityReasonCode.NON_FINITE_VALUE),
    ("feature_non_finite", QualityReasonCode.NON_FINITE_VALUE),
    ("negative_volume", QualityReasonCode.NEGATIVE_VOLUME),
    ("invalid_", QualityReasonCode.INVALID_NUMERIC_VALUE),
    ("feature_non_numeric", QualityReasonCode.INVALID_NUMERIC_VALUE),
    ("schema_version_mismatch", QualityReasonCode.SCHEMA_VERSION_MISMATCH),
    ("missing_timestamp_metadata", QualityReasonCode.MISSING_TIMESTAMP_METADATA),
    ("future_data_leakage", QualityReasonCode.FUTURE_DATA_LEAKAGE),
    ("missing_provenance", QualityReasonCode.MISSING_PROVENANCE),
    ("source_not_approved", QualityReasonCode.SOURCE_NOT_APPROVED),
    ("rate_limit_blocked", QualityReasonCode.RATE_LIMIT_BLOCKED),
    ("rate_limit_limited", QualityReasonCode.RATE_LIMIT_LIMITED),
)


def _reason_code(reason: str) -> QualityReasonCode:
    normalized = re.sub(r"^row_\d+_", "", reason.split(":", maxsplit=1)[0])
    for prefix, code in _REASON_PREFIXES:
        if normalized == prefix or normalized.startswith(prefix):
            return code
    return QualityReasonCode.INVALID_NUMERIC_VALUE


def _result(
    reasons: Iterable[str],
    *,
    fingerprint: str | None = None,
    degraded_only_codes: frozenset[QualityReasonCode] = frozenset(),
) -> QualityResult:
    rendered = tuple(reasons)
    codes = tuple(dict.fromkeys(_reason_code(reason) for reason in rendered))
    if not rendered:
        status = QualityStatus.VALID
    elif degraded_only_codes and set(codes).issubset(degraded_only_codes):
        status = QualityStatus.DEGRADED_ALLOWED
    else:
        status = QualityStatus.REJECTED
    return QualityResult(
        status is not QualityStatus.REJECTED,
        rendered,
        status,
        codes,
        fingerprint,
    )


def _fingerprint(value: object) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _parse_timestamp(value: object) -> tuple[float | None, str | None]:
    numeric = _finite_number(value)
    if numeric is not None:
        return numeric, None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None, "invalid_timestamp"
    else:
        return None, "invalid_timestamp"
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None, "timestamp_timezone_missing"
    try:
        return parsed.timestamp(), None
    except (OverflowError, OSError, ValueError):
        return None, "invalid_timestamp"


def _candle_fingerprint(rows: list[Mapping[str, object]]) -> str:
    snapshot = [{key: row.get(key) for key in REQUIRED_COLUMNS} for row in rows]
    return _fingerprint({"schema_version": OHLCV_SCHEMA_VERSION, "rows": snapshot})


def _validate_candle_policy(
    max_age_seconds: int | None,
    expected_interval_seconds: int | None,
    gap_tolerance_seconds: int,
    gap_policy: str,
    max_future_skew_seconds: int,
) -> None:
    if max_age_seconds is not None and max_age_seconds < 0:
        raise ValueError("max_age_seconds cannot be negative")
    if expected_interval_seconds is not None and expected_interval_seconds <= 0:
        raise ValueError("expected_interval_seconds must be positive")
    if gap_tolerance_seconds < 0 or max_future_skew_seconds < 0:
        raise ValueError("timestamp tolerances cannot be negative")
    if gap_policy not in {"reject", "degraded_allowed"}:
        raise ValueError("gap_policy must be reject or degraded_allowed")


def _missing_column_reasons(index: int, row: Mapping[str, object]) -> list[str]:
    missing = [column for column in REQUIRED_COLUMNS if column not in row]
    if not missing:
        return []
    return [f"row_{index}_missing_columns:{','.join(missing)}"]


def _numeric_cell_reason(index: int, column: str, value: object) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return f"row_{index}_invalid_{column}"
    if not math.isfinite(float(value)):
        return f"row_{index}_non_finite_{column}"
    return None


def _price_values(
    index: int, row: Mapping[str, object]
) -> tuple[dict[str, float], list[str]]:
    values: dict[str, float] = {}
    reasons: list[str] = []
    for column in PRICE_COLUMNS:
        value = row[column]
        issue = _numeric_cell_reason(index, column, value)
        if issue is not None:
            reasons.append(issue)
            continue
        values[column] = float(cast(int | float | Decimal, value))
    return values, reasons


def _price_relationship_reasons(index: int, values: Mapping[str, float]) -> list[str]:
    reasons: list[str] = []
    if "volume" in values and values["volume"] < 0:
        reasons.append(f"row_{index}_negative_volume")
    price_keys = ("open", "high", "low", "close")
    if not all(key in values for key in price_keys):
        return reasons
    opened, high, low, close = (
        values["open"],
        values["high"],
        values["low"],
        values["close"],
    )
    if min(opened, high, low, close) <= 0 or high < low:
        reasons.append(f"row_{index}_invalid_price_range")
    if low > min(opened, close) or high < max(opened, close):
        reasons.append(f"row_{index}_invalid_ohlc_range")
    return reasons


def _timestamp_reasons(
    index: int, row: Mapping[str, object], previous: float | None
) -> tuple[float | None, list[str]]:
    timestamp, error = _parse_timestamp(row.get("date"))
    if error == "timestamp_timezone_missing":
        return None, [f"row_{index}_timestamp_timezone_missing"]
    if timestamp is None:
        return None, [f"row_{index}_invalid_date"]
    if previous is None:
        return timestamp, []
    if timestamp == previous:
        return timestamp, [f"row_{index}_duplicate_timestamp"]
    if timestamp < previous:
        return timestamp, [f"row_{index}_timestamp_not_strictly_increasing"]
    return timestamp, []


def _future_timestamp_reason(
    index: int, timestamp: float, now_epoch: float, allowed_skew: int
) -> str | None:
    if timestamp <= now_epoch + allowed_skew:
        return None
    return f"row_{index}_future_timestamp"


def _gap_reason(
    index: int,
    timestamp: float,
    previous: float | None,
    expected_interval_seconds: int | None,
    tolerance_seconds: int,
) -> str | None:
    if previous is None or expected_interval_seconds is None:
        return None
    elapsed = timestamp - previous
    if elapsed <= expected_interval_seconds + tolerance_seconds:
        return None
    return f"row_{index}_gap_detected:delta_seconds={elapsed:g}"


def _latest_candle_reasons(
    rows: list[Mapping[str, object]],
    now_epoch: float,
    max_age_seconds: int | None,
) -> list[str]:
    timestamp, _ = _parse_timestamp(rows[-1].get("date"))
    if timestamp is None:
        return []
    if max_age_seconds is not None and now_epoch - timestamp > max_age_seconds:
        return ["stale_latest_candle"]
    return []


def _validate_candle_row(
    index: int,
    row: Mapping[str, object],
    *,
    previous_timestamp: float | None,
    reference_time: float,
    expected_interval_seconds: int | None,
    gap_tolerance_seconds: int,
    max_future_skew_seconds: int,
) -> tuple[list[str], float | None]:
    reasons = _missing_column_reasons(index, row)
    if not reasons:
        values, price_reasons = _price_values(index, row)
        reasons.extend(price_reasons)
        reasons.extend(_price_relationship_reasons(index, values))

    timestamp, timestamp_reasons = _timestamp_reasons(index, row, previous_timestamp)
    reasons.extend(timestamp_reasons)
    if timestamp is None:
        return reasons, previous_timestamp

    future_reason = _future_timestamp_reason(
        index, timestamp, reference_time, max_future_skew_seconds
    )
    if future_reason:
        reasons.append(future_reason)
    gap = _gap_reason(
        index,
        timestamp,
        previous_timestamp,
        expected_interval_seconds,
        gap_tolerance_seconds,
    )
    if gap:
        reasons.append(gap)
    next_previous = (
        timestamp
        if previous_timestamp is None or timestamp > previous_timestamp
        else previous_timestamp
    )
    return reasons, next_previous


def validate_candles(
    rows: Iterable[Mapping[str, object]],
    *,
    max_age_seconds: int | None = None,
    now_epoch: float | None = None,
    expected_interval_seconds: int | None = None,
    gap_tolerance_seconds: int = 0,
    gap_policy: Literal["reject", "degraded_allowed"] = "reject",
    max_future_skew_seconds: int = 0,
) -> QualityResult:
    """Validate OHLCV shape, numeric invariants, chronology, gaps and freshness.

    Numeric timestamps are epoch seconds. String/datetime timestamps must carry
    a timezone. Gaps are never repaired or interpolated by this gate.
    """
    _validate_candle_policy(
        max_age_seconds,
        expected_interval_seconds,
        gap_tolerance_seconds,
        gap_policy,
        max_future_skew_seconds,
    )
    data = list(rows)
    if not data:
        return _result(("empty_dataset",), fingerprint=_fingerprint([]))
    reference_time = time.time() if now_epoch is None else _finite_number(now_epoch)
    if reference_time is None:
        raise ValueError("now_epoch must be a finite epoch timestamp")

    reasons: list[str] = []
    previous_timestamp: float | None = None
    for index, row in enumerate(data):
        row_reasons, previous_timestamp = _validate_candle_row(
            index,
            row,
            previous_timestamp=previous_timestamp,
            reference_time=reference_time,
            expected_interval_seconds=expected_interval_seconds,
            gap_tolerance_seconds=gap_tolerance_seconds,
            max_future_skew_seconds=max_future_skew_seconds,
        )
        reasons.extend(row_reasons)

    reasons.extend(_latest_candle_reasons(data, reference_time, max_age_seconds))
    degraded_codes = (
        frozenset({QualityReasonCode.DATA_GAP})
        if gap_policy == "degraded_allowed"
        else frozenset()
    )
    return _result(
        reasons,
        fingerprint=_candle_fingerprint(data),
        degraded_only_codes=degraded_codes,
    )


def _feature_numeric_issue(name: str, value: object) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return f"feature_non_numeric:{name}"
    if not math.isfinite(float(value)):
        return f"feature_non_finite:{name}"
    return None


def _feature_schema_reasons(
    values: Mapping[str, object],
    required_features: Iterable[str],
    schema_version: str,
    expected_schema_version: str,
) -> list[str]:
    reasons: list[str] = []
    required = tuple(sorted(set(required_features)))
    if schema_version != expected_schema_version:
        reasons.append("schema_version_mismatch:feature_vector")
    missing = sorted(set(required) - set(values))
    unexpected = sorted(set(values) - set(required))
    if missing:
        reasons.append(f"missing_features:{','.join(missing)}")
    if unexpected:
        reasons.append(f"unexpected_features:{','.join(unexpected)}")
    for name in required:
        if name in values:
            issue = _feature_numeric_issue(name, values[name])
            if issue:
                reasons.append(issue)
    return reasons


def _feature_timestamp_reasons(
    available_at: object | None, decision_timestamp: object | None
) -> list[str]:
    reasons: list[str] = []
    available_epoch, available_error = _parse_timestamp(available_at)
    decision_epoch, decision_error = _parse_timestamp(decision_timestamp)
    if available_at is None or decision_timestamp is None:
        reasons.append("missing_timestamp_metadata:feature_vector")
    elif available_error or decision_error:
        reasons.append("invalid_available_at:feature_vector")
    elif available_epoch is not None and decision_epoch is not None:
        if available_epoch > decision_epoch:
            reasons.append("future_data_leakage:feature_vector")
    return reasons


def validate_feature_vector(
    values: Mapping[str, object],
    *,
    required_features: Iterable[str],
    available_at: object | None,
    decision_timestamp: object | None,
    schema_version: str = FEATURE_SCHEMA_VERSION,
    expected_schema_version: str = FEATURE_SCHEMA_VERSION,
) -> QualityResult:
    """Validate a versioned numeric feature vector and prevent future-data use."""
    reasons = _feature_schema_reasons(
        values, required_features, schema_version, expected_schema_version
    )
    reasons.extend(_feature_timestamp_reasons(available_at, decision_timestamp))
    safe_values = {name: str(values[name]) for name in sorted(values)}
    fingerprint = _fingerprint(
        {"schema_version": schema_version, "features": safe_values}
    )
    return _result(reasons, fingerprint=fingerprint)


def _required_text_fields(
    record: Mapping[str, object], names: tuple[str, ...]
) -> list[str]:
    missing = [
        name
        for name in names
        if not isinstance(record.get(name), str) or not str(record.get(name)).strip()
    ]
    if not missing:
        return []
    return [f"missing_provenance:{','.join(missing)}"]


def _parse_external_timestamps(
    record: Mapping[str, object],
) -> tuple[dict[str, float], list[str]]:
    timestamps: dict[str, float] = {}
    reasons: list[str] = []
    for field in ("event_timestamp", "ingested_at", "available_at"):
        parsed, error = _parse_timestamp(record.get(field))
        if error:
            reasons.append(f"invalid_{field}")
        elif parsed is not None:
            timestamps[field] = parsed
    return timestamps, reasons


def _external_as_of_reasons(
    timestamps: Mapping[str, float], as_of_timestamp: object
) -> list[str]:
    reasons: list[str] = []
    as_of, as_of_error = _parse_timestamp(as_of_timestamp)
    if as_of_error or as_of is None:
        reasons.append("invalid_decision_timestamp")
    ingested = timestamps.get("ingested_at")
    available = timestamps.get("available_at")
    if ingested is not None and available is not None and ingested > available:
        reasons.append("invalid_timestamp_order:ingested_after_available")
    if as_of is not None and available is not None and available > as_of:
        reasons.append("future_data_leakage:external_record")
    return reasons


def _external_timestamp_reasons(
    record: Mapping[str, object], as_of_timestamp: object
) -> list[str]:
    timestamps, reasons = _parse_external_timestamps(record)
    reasons.extend(_external_as_of_reasons(timestamps, as_of_timestamp))
    return reasons


def _external_policy_reasons(record: Mapping[str, object]) -> tuple[list[str], bool]:
    reasons: list[str] = []
    degraded = False
    license_status = str(record.get("license_status") or "").strip().lower()
    if license_status != "approved":
        reasons.append("source_not_approved")
    rate_status = str(record.get("rate_limit_status") or "").strip().lower()
    if rate_status == "blocked":
        reasons.append("rate_limit_blocked")
    elif rate_status == "limited":
        reasons.append("rate_limit_limited")
        degraded = True
    elif rate_status != "ok":
        reasons.append("invalid_rate_limit_status")
    return reasons, degraded


def validate_external_record(
    record: Mapping[str, object],
    *,
    as_of_timestamp: object,
    expected_schema_version: str = EXTERNAL_RECORD_SCHEMA_VERSION,
) -> QualityResult:
    """Validate external-data provenance, licensing, rate-limit state and as-of time.

    The event timestamp can describe a future scheduled event. Availability must
    precede the decision timestamp, so future-known-to-the-pipeline data cannot
    leak into an earlier feature/inference window.
    """
    reasons = _required_text_fields(
        record,
        (
            "schema_version",
            "provider_id",
            "source_id",
            "unit",
            "quote_currency",
            "revision",
            "license_status",
            "rate_limit_status",
        ),
    )
    if record.get("schema_version") != expected_schema_version:
        reasons.append("schema_version_mismatch:external_record")
    reasons.extend(_external_timestamp_reasons(record, as_of_timestamp))
    policy_reasons, rate_limited = _external_policy_reasons(record)
    reasons.extend(policy_reasons)
    safe_metadata = {
        key: str(record.get(key, ""))
        for key in (
            "schema_version",
            "provider_id",
            "source_id",
            "event_timestamp",
            "ingested_at",
            "available_at",
            "unit",
            "quote_currency",
            "revision",
            "license_status",
            "rate_limit_status",
        )
    }
    degraded_codes = (
        frozenset({QualityReasonCode.RATE_LIMIT_LIMITED})
        if rate_limited
        else frozenset()
    )
    return _result(
        reasons,
        fingerprint=_fingerprint(safe_metadata),
        degraded_only_codes=degraded_codes,
    )


def _model_schema_reasons(
    output: Mapping[str, object], expected_schema_version: str
) -> list[str]:
    reasons: list[str] = []
    if output.get("schema_version") != expected_schema_version:
        reasons.append("schema_version_mismatch:model_output")
    reasons.extend(_required_text_fields(output, ("model_revision", "proposal")))
    required = ("model_revision", "proposal", "confidence", "uncertainty")
    missing = [key for key in required if key not in output]
    if missing:
        reasons.append(f"invalid_model_output:missing_{','.join(missing)}")
    return reasons


def _model_value_reasons(output: Mapping[str, object]) -> list[str]:
    reasons: list[str] = []
    proposal = str(output.get("proposal") or "").strip().upper()
    if proposal not in {"ENTRY", "HOLD", "EXIT"}:
        reasons.append("invalid_model_output:proposal")
    for name in ("confidence", "uncertainty"):
        value = _finite_number(output.get(name))
        if value is None or not 0 <= value <= 1:
            reasons.append(f"invalid_model_output:{name}")
    return reasons


def _model_timestamp_reasons(
    produced_at_value: object, decision_timestamp: object
) -> list[str]:
    reasons: list[str] = []
    produced_at, produced_error = _parse_timestamp(produced_at_value)
    decision_at, decision_error = _parse_timestamp(decision_timestamp)
    if produced_error or decision_error or produced_at is None or decision_at is None:
        reasons.append("invalid_produced_at:model_output")
    elif produced_at > decision_at:
        reasons.append("future_data_leakage:model_output")
    return reasons


def validate_model_output(
    output: Mapping[str, object],
    *,
    decision_timestamp: object,
    expected_schema_version: str = MODEL_OUTPUT_SCHEMA_VERSION,
) -> QualityResult:
    """Validate a model proposal only; this function does not approve risk or orders."""
    reasons = _model_schema_reasons(output, expected_schema_version)
    reasons.extend(_model_value_reasons(output))
    reasons.extend(
        _model_timestamp_reasons(output.get("produced_at"), decision_timestamp)
    )
    fingerprint_fields = (
        "schema_version",
        "produced_at",
        "model_revision",
        "proposal",
        "confidence",
        "uncertainty",
    )
    safe_output = {key: str(output.get(key, "")) for key in fingerprint_fields}
    return _result(reasons, fingerprint=_fingerprint(safe_output))
