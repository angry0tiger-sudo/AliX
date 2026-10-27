from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.fetch_nobitex_public_ohlcv import (
    _validate_historical_rows,
    CandleRecord,
    INTERVAL_SECONDS,
    MAX_CANDLES_PER_REQUEST,
    fetch_historical_candles,
    ALLOWED_DAYS,
    parse_udf_payload,
    write_artifacts,
)
from scripts.run_phase10_public_price_observation import observe_public_price

UTC = timezone.utc


def test_two_year_historical_window_is_explicitly_allowed() -> None:
    assert 730 in ALLOWED_DAYS


def make_payload(start: int, count: int = 2) -> dict[str, Any]:
    # Nobitex candle timestamps can be offset from UTC hour boundaries. Only the
    # spacing between bars is invariant; UTC epoch modulo 3600 is not required.
    stamps = [start + i * 3600 for i in range(count)]
    return {
        "s": "ok",
        "t": stamps,
        "o": [100.0] * count,
        "h": [102.0] * count,
        "l": [99.0] * count,
        "c": [101.0] * count,
        "v": [5.0] * count,
    }


def test_parser_handles_offset_hour_boundaries_and_rejects_bad_arrays() -> None:
    rows = parse_udf_payload(
        make_payload(1800001800), requested_from=1800000000, requested_to=1800005400
    )
    assert len(rows) == 2
    assert str(rows[0]["date"]).endswith("Z")
    payload = make_payload(1800000000)
    payload["c"] = [100.0]
    with pytest.raises(ValueError, match="array_length"):
        parse_udf_payload(payload, requested_from=1800000000, requested_to=1800003600)


def test_parser_rejects_ohlc_invalid_and_future_outside_requested_window() -> None:
    payload = make_payload(1800000000)
    payload["h"] = [98.0, 98.0]
    with pytest.raises(ValueError, match="ohlc_invariant"):
        parse_udf_payload(payload, requested_from=1800000000, requested_to=1800003600)
    with pytest.raises(ValueError, match="out_of_bounds"):
        parse_udf_payload(
            make_payload(1800000000), requested_from=1800003600, requested_to=1800007200
        )


def _drop_timestamp(payload: dict[str, Any], timestamp: int) -> dict[str, Any]:
    if timestamp in payload["t"]:
        index = payload["t"].index(timestamp)
        for key in ("t", "o", "h", "l", "c", "v"):
            payload[key] = payload[key][:index] + payload[key][index + 1 :]
    return payload


def test_historical_fetch_is_bounded_rejects_gaps_and_uses_fake_transport() -> None:
    # Nobitex BTCUSDT hourly candles observed on the provider's :30 grid.
    now = datetime(2026, 10, 23, 4, 18, tzinfo=UTC)

    def getter(params: dict[str, str]) -> dict[str, Any]:
        start, end = int(params["from"]), int(params["to"])
        offset = INTERVAL_SECONDS // 2
        first = (
            (start - offset + INTERVAL_SECONDS - 1) // INTERVAL_SECONDS
        ) * INTERVAL_SECONDS + offset
        stamps = list(range(first, end, INTERVAL_SECONDS))
        return {
            "s": "ok",
            "t": stamps,
            "o": [100.0] * len(stamps),
            "h": [102.0] * len(stamps),
            "l": [99.0] * len(stamps),
            "c": [101.0] * len(stamps),
            "v": [5.0] * len(stamps),
        }

    empty_pages = [0]

    def transient_empty(params: dict[str, str]) -> dict[str, Any]:
        if int(params["to"]) < int(now.timestamp()) and empty_pages[0] == 0:
            empty_pages[0] += 1
            return {"s": "no_data"}
        return getter(params)

    rows = fetch_historical_candles(
        days=30, now=now, request_json=transient_empty, sleep_fn=lambda _: None
    )
    assert len(rows) == 720
    assert empty_pages[0] == 1
    assert rows[0]["timestamp"] % INTERVAL_SECONDS == INTERVAL_SECONDS // 2
    assert rows[-1]["timestamp"] % INTERVAL_SECONDS == INTERVAL_SECONDS // 2
    assert rows[-1]["timestamp"] + INTERVAL_SECONDS <= int(now.timestamp())
    expected_end = int(datetime(2026, 10, 23, 2, 30, tzinfo=UTC).timestamp())
    assert rows[-1]["timestamp"] == expected_end
    for incomplete_rows in (rows[:-1], rows[1:]):
        with pytest.raises(ValueError, match="coverage_mismatch"):
            _validate_historical_rows(incomplete_rows, 30)

    with pytest.raises(ValueError, match="historical_days"):
        fetch_historical_candles(
            days=10, now=now, request_json=getter, sleep_fn=lambda _: None
        )

    gap_timestamp = expected_end - 100 * INTERVAL_SECONDS

    def gapped(params: dict[str, str]) -> dict[str, Any]:
        return _drop_timestamp(getter(params), gap_timestamp)

    with pytest.raises(ValueError, match="gap_detected"):
        fetch_historical_candles(
            days=30, now=now, request_json=gapped, sleep_fn=lambda _: None
        )

def test_historical_fetch_recovers_missing_hour_from_exact_public_read() -> None:
    now = datetime(2026, 10, 23, 4, 18, tzinfo=UTC)
    expected_end = int(datetime(2026, 10, 23, 2, 30, tzinfo=UTC).timestamp())
    gap_timestamp = expected_end - 100 * INTERVAL_SECONDS
    removed = [False]

    def getter(params: dict[str, str]) -> dict[str, Any]:
        start, end = int(params["from"]), int(params["to"])
        offset = INTERVAL_SECONDS // 2
        first = (
            (start - offset + INTERVAL_SECONDS - 1) // INTERVAL_SECONDS
        ) * INTERVAL_SECONDS + offset
        stamps = list(range(first, end, INTERVAL_SECONDS))
        return {
            "s": "ok",
            "t": stamps,
            "o": [100.0] * len(stamps),
            "h": [102.0] * len(stamps),
            "l": [99.0] * len(stamps),
            "c": [101.0] * len(stamps),
            "v": [5.0] * len(stamps),
        }

    def one_transient_gap(params: dict[str, str]) -> dict[str, Any]:
        response = getter(params)
        span = int(params["to"]) - int(params["from"])
        if (
            not removed[0]
            and span < MAX_CANDLES_PER_REQUEST * INTERVAL_SECONDS
            and span > 2 * INTERVAL_SECONDS
            and gap_timestamp in response["t"]
        ):
            removed[0] = True
            return _drop_timestamp(response, gap_timestamp)
        return response

    rows = fetch_historical_candles(
        days=30,
        now=now,
        request_json=one_transient_gap,
        sleep_fn=lambda _: None,
    )
    assert len(rows) == 30 * 24
    assert any(row["timestamp"] == gap_timestamp for row in rows)
    assert removed[0] is True


def test_manifest_contains_data_hash_and_closed_authorization_state(
    tmp_path: Path,
) -> None:
    rows: list[CandleRecord] = [
        {
            "timestamp": 1792629000,
            "date": "2026-10-22T00:30:00Z",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 3.0,
        },
        {
            "timestamp": 1792632600,
            "date": "2026-10-22T01:30:00Z",
            "open": 100.5,
            "high": 101.5,
            "low": 100.0,
            "close": 101.0,
            "volume": 4.0,
        },
    ]
    manifest = write_artifacts(
        rows,
        output_dir=tmp_path,
        days=30,
        retrieved_at=datetime.now(UTC),
        commit_sha="a" * 40,
    )
    assert manifest["csv_sha256"]
    assert manifest["credentials_used"] is False
    assert manifest["order_permission"] is False
    assert manifest["Champion"] is None
    assert manifest["trading_authorization"] == "NO-GO"
    assert (tmp_path / "nobitex-btcusdt-1h.csv").is_file()


def test_price_observer_always_abstains_and_has_a_bounded_duration() -> None:
    elapsed = [0.0]

    def sleep_and_advance(seconds: float) -> None:
        elapsed[0] += seconds

    records = observe_public_price(
        duration_seconds=300,
        fetch_price=lambda: 100.0,
        monotonic=lambda: elapsed[0],
        clock=lambda: datetime(2026, 10, 23, 4, 0, tzinfo=UTC),
        sleep_fn=sleep_and_advance,
        interval_seconds=60,
    )
    assert records
    assert all(row["decision"] == "ABSTAIN" for row in records)
    assert all(row["orders_submitted"] == 0 for row in records)
    with pytest.raises(ValueError, match="observation_duration"):
        observe_public_price(duration_seconds=3601)
