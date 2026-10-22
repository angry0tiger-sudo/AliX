from __future__ import annotations

import pytest

from services.data_quality.gate import (
    EXTERNAL_RECORD_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    MODEL_OUTPUT_SCHEMA_VERSION,
    QualityReasonCode,
    QualityStatus,
    validate_candles,
    validate_external_record,
    validate_feature_vector,
    validate_model_output,
)


def candle(ts: int | str = 1_000) -> dict[str, object]:
    return {
        "date": ts,
        "open": 10.0,
        "high": 12.0,
        "low": 9.0,
        "close": 11.0,
        "volume": 5.0,
    }


def external_record(**changes: object) -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": EXTERNAL_RECORD_SCHEMA_VERSION,
        "provider_id": "fixture-provider",
        "source_id": "fixture-feed",
        "event_timestamp": "2026-10-22T08:00:00Z",
        "ingested_at": "2026-10-22T06:00:00Z",
        "available_at": "2026-10-22T06:01:00Z",
        "unit": "count",
        "quote_currency": "USDT",
        "revision": "fixture-r1",
        "license_status": "approved",
        "rate_limit_status": "ok",
    }
    record.update(changes)
    return record


def test_candles_return_versioned_fingerprint_and_valid_status() -> None:
    result = validate_candles([candle(1_000), candle(1_060)])

    assert result.passed
    assert result.status is QualityStatus.VALID
    assert result.reason_codes == ()
    assert len(result.fingerprint or "") == 64


def test_candle_validation_distinguishes_duplicate_and_out_of_order_timestamps() -> None:
    duplicate = validate_candles([candle(1_000), candle(1_000)])
    out_of_order = validate_candles([candle(1_060), candle(1_000)])

    assert QualityReasonCode.DUPLICATE_TIMESTAMP in duplicate.reason_codes
    assert (
        QualityReasonCode.TIMESTAMP_NOT_INCREASING in out_of_order.reason_codes
    )


def test_candle_gap_is_rejected_by_default_and_can_be_explicitly_degraded() -> None:
    rows = [candle(1_000), candle(1_180)]
    rejected = validate_candles(
        rows,
        now_epoch=1_200,
        expected_interval_seconds=60,
    )
    degraded = validate_candles(
        rows,
        now_epoch=1_200,
        expected_interval_seconds=60,
        gap_policy="degraded_allowed",
    )

    assert rejected.status is QualityStatus.REJECTED
    assert QualityReasonCode.DATA_GAP in rejected.reason_codes
    assert degraded.status is QualityStatus.DEGRADED_ALLOWED
    assert degraded.passed


def test_candle_validation_rejects_naive_iso_timestamp_and_future_candle() -> None:
    naive = validate_candles([candle("2026-10-22T06:00:00")])
    future = validate_candles([candle(2_000)], now_epoch=1_000)

    assert QualityReasonCode.TIMEZONE_REQUIRED in naive.reason_codes
    assert QualityReasonCode.FUTURE_TIMESTAMP in future.reason_codes


def test_candle_validation_keeps_existing_stale_and_ohlc_reason_strings() -> None:
    invalid = candle(1_000)
    invalid["high"] = 8.0
    invalid["volume"] = -1.0
    result = validate_candles(
        [invalid],
        max_age_seconds=60,
        now_epoch=2_000,
    )

    assert not result.passed
    assert any("invalid_ohlc_range" in reason for reason in result.reasons)
    assert any("negative_volume" in reason for reason in result.reasons)
    assert "stale_latest_candle" in result.reasons


def test_feature_vector_rejects_schema_mismatch_nan_and_missing_features() -> None:
    result = validate_feature_vector(
        {"rsi": float("nan"), "momentum": 0.2},
        required_features=("rsi", "volume_delta", "momentum"),
        available_at=1_100,
        decision_timestamp=1_200,
        schema_version="feature-vector.v0",
    )

    assert not result.passed
    assert QualityReasonCode.SCHEMA_VERSION_MISMATCH in result.reason_codes
    assert QualityReasonCode.NON_FINITE_VALUE in result.reason_codes
    assert QualityReasonCode.MISSING_FEATURES in result.reason_codes


def test_feature_vector_rejects_future_available_data_leakage() -> None:
    result = validate_feature_vector(
        {"rsi": 55.0},
        required_features=("rsi",),
        available_at=1_201,
        decision_timestamp=1_200,
    )

    assert result.status is QualityStatus.REJECTED
    assert QualityReasonCode.FUTURE_DATA_LEAKAGE in result.reason_codes


def test_feature_vector_accepts_valid_as_of_values() -> None:
    result = validate_feature_vector(
        {"rsi": 55.0, "momentum": -0.02},
        required_features=("rsi", "momentum"),
        available_at="2026-10-22T06:00:00Z",
        decision_timestamp="2026-10-22T06:01:00+00:00",
        schema_version=FEATURE_SCHEMA_VERSION,
    )

    assert result.status is QualityStatus.VALID


def test_external_record_rejects_unapproved_source_and_future_availability() -> None:
    unapproved = validate_external_record(
        external_record(license_status="pending"),
        as_of_timestamp="2026-10-22T06:02:00Z",
    )
    future = validate_external_record(
        external_record(available_at="2026-10-22T06:03:00Z"),
        as_of_timestamp="2026-10-22T06:02:00Z",
    )

    assert QualityReasonCode.SOURCE_NOT_APPROVED in unapproved.reason_codes
    assert QualityReasonCode.FUTURE_DATA_LEAKAGE in future.reason_codes
    assert not unapproved.passed and not future.passed


def test_external_rate_limit_status_is_reason_coded() -> None:
    limited = validate_external_record(
        external_record(rate_limit_status="limited"),
        as_of_timestamp="2026-10-22T06:02:00Z",
    )
    blocked = validate_external_record(
        external_record(rate_limit_status="blocked"),
        as_of_timestamp="2026-10-22T06:02:00Z",
    )

    assert limited.status is QualityStatus.DEGRADED_ALLOWED
    assert QualityReasonCode.RATE_LIMIT_LIMITED in limited.reason_codes
    assert blocked.status is QualityStatus.REJECTED
    assert QualityReasonCode.RATE_LIMIT_BLOCKED in blocked.reason_codes


def test_external_record_rejects_ingestion_after_availability() -> None:
    result = validate_external_record(
        external_record(
            ingested_at="2026-10-22T06:02:00Z",
            available_at="2026-10-22T06:01:00Z",
        ),
        as_of_timestamp="2026-10-22T06:03:00Z",
    )

    assert QualityReasonCode.INVALID_TIMESTAMP_ORDER in result.reason_codes


def test_model_output_is_validated_as_proposal_not_risk_approval() -> None:
    output = {
        "schema_version": MODEL_OUTPUT_SCHEMA_VERSION,
        "produced_at": "2026-10-22T06:00:00Z",
        "model_revision": "fixture-model-r1",
        "proposal": "HOLD",
        "confidence": 0.7,
        "uncertainty": 0.3,
    }
    result = validate_model_output(
        output,
        decision_timestamp="2026-10-22T06:01:00Z",
    )

    assert result.status is QualityStatus.VALID


def test_model_output_rejects_future_timestamp_and_invalid_confidence() -> None:
    output = {
        "schema_version": MODEL_OUTPUT_SCHEMA_VERSION,
        "produced_at": "2026-10-22T06:02:00Z",
        "model_revision": "fixture-model-r1",
        "proposal": "ENTRY",
        "confidence": 1.5,
        "uncertainty": 0.3,
    }
    result = validate_model_output(
        output,
        decision_timestamp="2026-10-22T06:01:00Z",
    )

    assert not result.passed
    assert QualityReasonCode.FUTURE_DATA_LEAKAGE in result.reason_codes
    assert QualityReasonCode.INVALID_MODEL_OUTPUT in result.reason_codes


def test_invalid_quality_policy_is_not_silently_accepted() -> None:
    with pytest.raises(ValueError, match="expected_interval_seconds"):
        validate_candles([candle()], expected_interval_seconds=0)
    with pytest.raises(ValueError, match="gap_policy"):
        validate_candles([candle()], gap_policy="interpolate")  # type: ignore[arg-type]
