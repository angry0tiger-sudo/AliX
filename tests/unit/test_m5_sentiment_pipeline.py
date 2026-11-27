from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from scripts.align_m5_sentiment import align_records, parse_aware_timestamp
from scripts.fetch_m5_sentiment_data import (
    AVAILABILITY_POLICY,
    fetch_and_store,
    normalize_records,
    store_snapshot,
)

FETCHED_AT = datetime(2026, 11, 27, 12, 0, tzinfo=timezone.utc)
DAY_ONE = int(datetime(2026, 11, 24, tzinfo=timezone.utc).timestamp())
DAY_TWO = int(datetime(2026, 11, 25, tzinfo=timezone.utc).timestamp())


def provider_payload() -> dict[str, Any]:
    return {
        "name": "Fear and Greed Index",
        "data": [
            {
                "value": "70",
                "value_classification": "Greed",
                "timestamp": str(DAY_TWO),
            },
            {
                "value": "35",
                "value_classification": "Fear",
                "timestamp": str(DAY_ONE),
            },
        ],
    }


def source_record(
    *,
    observation: str,
    available_at: str,
    value: int,
    classification: str,
) -> dict[str, object]:
    return {
        "provider_id": "alternative_me_fear_greed",
        "target_market": "Nobitex Spot/USDT BTCUSDT",
        "market_scope": "global_btc_sentiment_not_nobitex_pair_specific",
        "timestamp": 1,
        "observation_time_utc": observation,
        "value": value,
        "value_classification": classification,
        "fetched_at": "2026-11-27T12:00:00Z",
        "available_at": available_at,
        "available_at_basis": AVAILABILITY_POLICY,
    }


def test_normalize_records_sorts_and_uses_conservative_availability_proxy() -> None:
    records = normalize_records(provider_payload(), FETCHED_AT)

    assert [record["value"] for record in records] == [35, 70]
    assert records[0]["available_at"] == "2026-11-25T00:00:00Z"
    assert records[1]["available_at"] == "2026-11-26T00:00:00Z"
    assert records[0]["available_at_basis"] == AVAILABILITY_POLICY
    assert records[0]["fetched_at"] == "2026-11-27T12:00:00Z"


def test_normalize_records_rejects_duplicate_timestamps() -> None:
    payload = provider_payload()
    payload["data"].append(payload["data"][0])

    with pytest.raises(ValueError, match="duplicate_timestamp"):
        normalize_records(payload, FETCHED_AT)


def test_normalize_records_rejects_out_of_range_values() -> None:
    payload = provider_payload()
    payload["data"][0]["value"] = "101"

    with pytest.raises(ValueError, match="value_out_of_range"):
        normalize_records(payload, FETCHED_AT)


def test_normalize_records_rejects_future_provider_timestamp() -> None:
    payload = provider_payload()
    payload["data"][0]["timestamp"] = str(
        int(datetime(2026, 11, 28, tzinfo=timezone.utc).timestamp())
    )

    with pytest.raises(ValueError, match="timestamp_in_future"):
        normalize_records(payload, FETCHED_AT)


def test_store_snapshot_writes_raw_normalized_and_manifest_with_hashes(
    tmp_path: Path,
) -> None:
    raw = json.dumps(provider_payload(), separators=(",", ":")).encode("utf-8")
    manifest = store_snapshot(raw, provider_payload(), tmp_path, FETCHED_AT)

    raw_path = tmp_path / str(manifest["raw_file"])
    normalized_path = tmp_path / str(manifest["normalized_file"])
    manifest_path = next(tmp_path.glob("*.manifest.json"))

    assert raw_path.read_bytes() == raw
    assert hashlib.sha256(raw).hexdigest() == manifest["raw_sha256"]
    assert (
        hashlib.sha256(normalized_path.read_bytes()).hexdigest()
        == manifest["normalized_sha256"]
    )
    assert manifest["record_count"] == 2
    assert manifest["start_observation_time_utc"] == "2026-11-24T00:00:00Z"
    assert manifest["end_observation_time_utc"] == "2026-11-25T00:00:00Z"
    assert manifest["provider_supplies_historical_available_at"] is False
    assert manifest_path.is_file()


def test_store_snapshot_never_overwrites_existing_snapshot(tmp_path: Path) -> None:
    raw = json.dumps(provider_payload(), separators=(",", ":")).encode("utf-8")
    store_snapshot(raw, provider_payload(), tmp_path, FETCHED_AT)

    with pytest.raises(FileExistsError, match="already_exists"):
        store_snapshot(raw, provider_payload(), tmp_path, FETCHED_AT)


def test_fetch_and_store_uses_public_endpoint_and_limit_zero(tmp_path: Path) -> None:
    requested_urls: list[str] = []
    body = json.dumps(provider_payload()).encode("utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "application/json"},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        manifest = fetch_and_store(tmp_path, client=client, fetched_at=FETCHED_AT)

    assert len(requested_urls) == 1
    assert requested_urls[0] == "https://api.alternative.me/fng/?limit=0"
    assert manifest["record_count"] == 2


def test_alignment_does_not_use_value_available_at_noon_for_ten_am_decision() -> None:
    records = [
        source_record(
            observation="2026-11-25T00:00:00Z",
            available_at="2026-11-26T00:00:00Z",
            value=35,
            classification="Fear",
        ),
        source_record(
            observation="2026-11-26T00:00:00Z",
            available_at="2026-11-27T12:00:00Z",
            value=70,
            classification="Greed",
        ),
    ]

    aligned = align_records(
        records,
        ["2026-11-27T10:00:00Z", "2026-11-27T12:00:00Z"],
        timeframe="1h",
    )

    assert aligned[0]["feature_value"] == 35
    assert aligned[0]["source_available_at"] == "2026-11-26T00:00:00Z"
    assert aligned[1]["feature_value"] == 70
    assert aligned[1]["source_available_at"] == "2026-11-27T12:00:00Z"
    assert parse_aware_timestamp(
        aligned[0]["source_available_at"], "source_available_at"
    ) <= parse_aware_timestamp(aligned[0]["decision_time"], "decision_time")
    assert parse_aware_timestamp(
        aligned[1]["source_available_at"], "source_available_at"
    ) <= parse_aware_timestamp(aligned[1]["decision_time"], "decision_time")


@pytest.mark.parametrize("timeframe", ["1h", "4h"])
def test_alignment_forward_fills_latest_available_value(timeframe: str) -> None:
    records = [
        source_record(
            observation="2026-11-25T00:00:00Z",
            available_at="2026-11-26T00:00:00Z",
            value=45,
            classification="Neutral",
        )
    ]

    aligned = align_records(
        records,
        ["2026-11-26T01:00:00Z", "2026-11-26T04:00:00Z"],
        timeframe=timeframe,
    )

    assert [item["feature_value"] for item in aligned] == [45, 45]
    assert all(item["status"] == "asof_match_or_forward_fill" for item in aligned)


def test_alignment_does_not_backfill_before_first_available_value() -> None:
    records = [
        source_record(
            observation="2026-11-26T00:00:00Z",
            available_at="2026-11-27T12:00:00Z",
            value=70,
            classification="Greed",
        )
    ]

    aligned = align_records(
        records,
        ["2026-11-27T10:00:00Z"],
        timeframe="4h",
    )

    assert aligned[0]["feature_value"] is None
    assert aligned[0]["status"] == "no_prior_available_record"


def test_alignment_rejects_naive_or_non_increasing_decision_times() -> None:
    records = [
        source_record(
            observation="2026-11-25T00:00:00Z",
            available_at="2026-11-26T00:00:00Z",
            value=45,
            classification="Neutral",
        )
    ]

    with pytest.raises(ValueError, match="timezone"):
        align_records(records, ["2026-11-27T10:00:00"], timeframe="1h")

    with pytest.raises(ValueError, match="strictly_increasing"):
        align_records(
            records,
            ["2026-11-27T10:00:00Z", "2026-11-27T10:00:00Z"],
            timeframe="1h",
        )


def test_alignment_rejects_unsupported_timeframe() -> None:
    with pytest.raises(ValueError, match="timeframe_not_allowed"):
        align_records([], [], timeframe="15m")
