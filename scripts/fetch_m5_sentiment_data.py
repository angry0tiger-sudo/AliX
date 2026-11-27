"""Fetch and preserve immutable Alternative.me Fear & Greed snapshots.

Alternative.me publishes a daily observation timestamp but does not expose a
per-record historical publication timestamp. available_at is therefore a
conservative proxy: the provider timestamp plus 24 hours, as required by
ADR-061. It is explicitly not represented as a provider-measured release time.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Mapping, cast

import httpx

API_URL = "https://api.alternative.me/fng/"
PROVIDER_ID = "alternative_me_fear_greed"
TARGET_MARKET = "Nobitex Spot/USDT BTCUSDT"
SCHEMA_VERSION = "alix.m5.alternative-me-fng.v1"
AVAILABILITY_POLICY = "observation_timestamp_plus_24h_conservative_proxy"
MAX_RESPONSE_BYTES = 10 * 1024 * 1024
DEFAULT_OUTPUT_DIR = Path("data/sentiment/alternative_me/fng")
REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
REQUEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "AliX-M5.1-Research/1.0",
}


def _utc_datetime(value: datetime, field_name: str) -> datetime:
    """Require an aware timestamp and normalize it to UTC."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name}_must_be_timezone_aware")
    return value.astimezone(timezone.utc)


def _iso_utc(value: datetime) -> str:
    """Render an aware timestamp as an ISO-8601 UTC value."""
    return _utc_datetime(value, "timestamp").isoformat().replace("+00:00", "Z")


def _parse_integer(value: object, field_name: str) -> int:
    """Parse provider numeric strings without accepting booleans or floats."""
    if isinstance(value, bool):
        raise ValueError(f"provider_{field_name}_invalid")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.isdecimal():
        parsed = int(value)
    else:
        raise ValueError(f"provider_{field_name}_invalid")
    return parsed


def normalize_records(
    payload: Mapping[str, object], fetched_at: datetime
) -> list[dict[str, object]]:
    """Validate and normalize the provider response into ascending UTC records."""
    fetched_at_utc = _utc_datetime(fetched_at, "fetched_at")
    if payload.get("name") != "Fear and Greed Index":
        raise ValueError("alternative_me_provider_name_mismatch")

    raw_data = payload.get("data")
    if not isinstance(raw_data, list) or not raw_data:
        raise ValueError("alternative_me_data_missing_or_empty")

    records: list[dict[str, object]] = []
    seen_timestamps: set[int] = set()
    for item in raw_data:
        if not isinstance(item, dict):
            raise ValueError("alternative_me_record_not_object")

        timestamp = _parse_integer(item.get("timestamp"), "timestamp")
        if timestamp <= 0:
            raise ValueError("alternative_me_timestamp_out_of_range")
        if timestamp in seen_timestamps:
            raise ValueError("alternative_me_duplicate_timestamp")
        seen_timestamps.add(timestamp)

        try:
            observation_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            raise ValueError("alternative_me_timestamp_unrepresentable") from None

        if observation_time > fetched_at_utc + timedelta(minutes=5):
            raise ValueError("alternative_me_source_timestamp_in_future")

        value = _parse_integer(item.get("value"), "value")
        if not 0 <= value <= 100:
            raise ValueError("alternative_me_value_out_of_range")

        classification = item.get("value_classification")
        if not isinstance(classification, str) or not classification.strip():
            raise ValueError("alternative_me_classification_missing")

        available_at = observation_time + timedelta(days=1)
        records.append(
            {
                "provider_id": PROVIDER_ID,
                "target_market": TARGET_MARKET,
                "market_scope": "global_btc_sentiment_not_nobitex_pair_specific",
                "timestamp": timestamp,
                "observation_time_utc": _iso_utc(observation_time),
                "value": value,
                "value_classification": classification.strip(),
                "fetched_at": _iso_utc(fetched_at_utc),
                "available_at": _iso_utc(available_at),
                "available_at_basis": AVAILABILITY_POLICY,
            }
        )

    records.sort(key=lambda record: cast(int, record["timestamp"]))
    return records


def _json_bytes(payload: object) -> bytes:
    """Serialize JSON deterministically for repeatable content hashes."""
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    return (rendered + "\n").encode("utf-8")


def _write_new_file(path: Path, content: bytes) -> None:
    """Create a file exclusively; never replace an existing snapshot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as file_handle:
        file_handle.write(content)
        file_handle.flush()


def store_snapshot(
    raw_content: bytes,
    payload: Mapping[str, object],
    output_dir: Path,
    fetched_at: datetime,
) -> dict[str, object]:
    """Store raw, normalized, and manifest files as an append-only snapshot."""
    if not raw_content:
        raise ValueError("alternative_me_empty_response_body")
    if len(raw_content) > MAX_RESPONSE_BYTES:
        raise ValueError("alternative_me_response_too_large")

    fetched_at_utc = _utc_datetime(fetched_at, "fetched_at")
    records = normalize_records(payload, fetched_at_utc)
    raw_sha256 = hashlib.sha256(raw_content).hexdigest()
    normalized_payload = {
        "schema_version": SCHEMA_VERSION,
        "provider_id": PROVIDER_ID,
        "target_market": TARGET_MARKET,
        "records": records,
    }
    normalized_content = _json_bytes(normalized_payload)
    normalized_sha256 = hashlib.sha256(normalized_content).hexdigest()

    snapshot_stem = (
        "alternative_me_fng_"
        + fetched_at_utc.strftime("%Y%m%dT%H%M%S%fZ")
        + "_"
        + raw_sha256[:12]
    )
    raw_name = f"{snapshot_stem}.raw.json"
    normalized_name = f"{snapshot_stem}.normalized.json"
    manifest_name = f"{snapshot_stem}.manifest.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / raw_name
    normalized_path = output_dir / normalized_name
    manifest_path = output_dir / manifest_name
    target_paths = (raw_path, normalized_path, manifest_path)
    if any(path.exists() for path in target_paths):
        raise FileExistsError("immutable_sentiment_snapshot_already_exists")

    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "provider_id": PROVIDER_ID,
        "api_url": API_URL + "?limit=0",
        "target_market": TARGET_MARKET,
        "market_scope": "global_btc_sentiment_not_nobitex_pair_specific",
        "fetched_at": _iso_utc(fetched_at_utc),
        "record_count": len(records),
        "start_observation_time_utc": records[0]["observation_time_utc"],
        "end_observation_time_utc": records[-1]["observation_time_utc"],
        "availability_policy": AVAILABILITY_POLICY,
        "provider_supplies_historical_available_at": False,
        "availability_notice": (
            "available_at is a conservative timestamp+24h proxy, not a "
            "provider-measured historical publication timestamp"
        ),
        "raw_file": raw_name,
        "raw_sha256": raw_sha256,
        "normalized_file": normalized_name,
        "normalized_sha256": normalized_sha256,
    }
    manifest_content = _json_bytes(manifest)

    _write_new_file(raw_path, raw_content)
    _write_new_file(normalized_path, normalized_content)
    # The manifest is written last and acts as the complete-snapshot marker.
    _write_new_file(manifest_path, manifest_content)
    return manifest


def fetch_and_store(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    client: httpx.Client | None = None,
    fetched_at: datetime | None = None,
) -> dict[str, object]:
    """Fetch all public observations and store one immutable snapshot."""
    if client is None:
        with httpx.Client(timeout=REQUEST_TIMEOUT, headers=REQUEST_HEADERS) as owned:
            response = owned.get(API_URL, params={"limit": 0})
    else:
        response = client.get(API_URL, params={"limit": 0})

    response.raise_for_status()
    if len(response.content) > MAX_RESPONSE_BYTES:
        raise ValueError("alternative_me_response_too_large")

    payload_object = response.json()
    if not isinstance(payload_object, dict):
        raise ValueError("alternative_me_payload_not_object")
    payload = cast(Mapping[str, object], payload_object)
    retrieval_time = fetched_at or datetime.now(timezone.utc)
    _utc_datetime(retrieval_time, "fetched_at")
    return store_snapshot(response.content, payload, output_dir, retrieval_time)


def main() -> int:
    """Fetch and persist one snapshot from the public endpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for append-only raw, normalized, and manifest snapshots.",
    )
    arguments = parser.parse_args()

    manifest = fetch_and_store(arguments.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
