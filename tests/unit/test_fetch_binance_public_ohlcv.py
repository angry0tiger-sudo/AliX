from __future__ import annotations
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
import pytest
from scripts import fetch_binance_public_ohlcv as fetcher


def klines(start: int, count: int) -> list[list[Any]]:
    return [
        [
            start + i * 60_000,
            "100",
            "101",
            "99",
            "100.5",
            "0" if i == 2 else "3.5",
            start + i * 60_000 + 59_999,
            "0",
            0,
            "0",
            "0",
            "0",
        ]
        for i in range(count)
    ]


def test_paginated_fetch_preserves_zero_volume_and_checks_continuity() -> None:
    now = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)
    end = int(now.timestamp() * 1000) - 60_000
    start = end - 1439 * 60_000
    calls: list[dict[str, str]] = []
    sleeps: list[float] = []

    def provider(params: dict[str, str]) -> list[Any]:
        calls.append(params.copy())
        cursor = int(params["startTime"])
        count = min(1000, (end - cursor) // 60_000 + 1)
        return klines(cursor, count)

    rows = fetcher.fetch_historical_candles(
        days=1, now=now, request_json=provider, sleep_fn=sleeps.append
    )
    assert len(rows) == 1440
    assert rows[0]["timestamp"] * 1000 == start
    assert rows[-1]["timestamp"] * 1000 == end
    assert rows[2]["volume"] == 0.0
    assert len(calls) == 2
    assert calls[1]["startTime"] == str(start + 1000 * 60_000)
    assert all(c["interval"] == "1m" and c["limit"] == "1000" for c in calls)
    assert sleeps == [fetcher.PAGE_DELAY_SECONDS]


def test_two_year_window_is_supported_with_bounded_page_pacing() -> None:
    assert 730 in fetcher.ALLOWED_DAYS
    assert fetcher.PAGE_SIZE == 1000
    assert fetcher.PAGE_DELAY_SECONDS >= 0.25


def test_incomplete_page_retries_then_fails_closed() -> None:
    now = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)
    calls = 0

    def incomplete(params: dict[str, str]) -> list[Any]:
        nonlocal calls
        calls += 1
        return klines(int(params["startTime"]), 2)

    with pytest.raises(ValueError, match="binance_kline_page_count_mismatch"):
        fetcher.fetch_historical_candles(
            days=1, now=now, request_json=incomplete, sleep_fn=lambda _seconds: None
        )
    assert calls == 3


def test_manifest_hash_safety_fields_and_gap_rejection(tmp_path: Path) -> None:
    now = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)
    end = int(now.timestamp() * 1000) - 60_000
    start = end - 1439 * 60_000
    rows = [
        fetcher._parse_kline(row, start + i * 60_000)
        for i, row in enumerate(klines(start, 1440))
    ]
    manifest = fetcher.write_artifacts(
        rows,
        output_dir=tmp_path,
        days=1,
        retrieved_at=now,
        commit_sha="abc123",
        symbol="ETHUSDT",
    )
    csv_path = tmp_path / "binance-ethusdt-1m.csv"
    saved = json.loads((tmp_path / "binance-ethusdt-1m.manifest.json").read_text())
    assert manifest["csv_sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert saved["commit_sha"] == "abc123" and saved["missing_minute_intervals"] == 0
    assert saved["private_api_calls"] == 0 and saved["real_order_attempts"] == 0
    assert saved["Champion"] is None and saved["order_permission"] is False
    assert saved["trading_authorization"] == "NO-GO"
    with pytest.raises(ValueError, match="binance_historical_row_count_mismatch"):
        fetcher.write_artifacts(
            rows[:-1],
            output_dir=tmp_path,
            days=1,
            retrieved_at=now,
            commit_sha="abc123",
            symbol="ETHUSDT",
        )


def test_request_uses_official_market_data_host_and_surfaces_http_451(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hosts: list[str] = []

    class FakeResponse:
        status = 451

        def read(self) -> bytes:
            return (
                b'{"code":-1002,"msg":"Service unavailable from a restricted location"}'
            )

        def getheader(self, name: str) -> str | None:
            return None

    class FakeConnection:
        def __init__(self, host: str, **_kwargs: Any) -> None:
            hosts.append(host)

        def request(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def getresponse(self) -> FakeResponse:
            return FakeResponse()

        def close(self) -> None:
            return None

    monkeypatch.setattr(fetcher, "HTTPSConnection", FakeConnection)

    with pytest.raises(
        RuntimeError, match="binance_public_api_restricted_location"
    ) as error:
        fetcher.request_public_json(
            {
                "symbol": "BTCUSDT",
                "interval": "1m",
                "startTime": "1000",
                "endTime": "61000",
                "limit": "1000",
            }
        )

    assert hosts == ["data-api.binance.vision"]
    assert "status=451" in str(error.value)
    assert "restricted location" in str(error.value)


def test_fetch_historical_candles_uses_exact_inclusive_utc_window() -> None:
    start = datetime(2024, 10, 30, 12, 5, tzinfo=timezone.utc)
    end = start + timedelta(days=1) - timedelta(minutes=1)
    calls: list[dict[str, str]] = []

    def provider(params: dict[str, str]) -> list[Any]:
        calls.append(params.copy())
        cursor = int(params["startTime"])
        end_ms = int(params["endTime"])
        count = min(1000, (end_ms - cursor) // fetcher.INTERVAL_MS + 1)
        return klines(cursor, count)

    rows = fetcher.fetch_historical_candles(
        days=1,
        now=datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc),
        start_utc=start,
        end_utc=end,
        request_json=provider,
        sleep_fn=lambda _seconds: None,
    )

    assert len(rows) == 1440
    assert rows[0]["date"] == "2024-10-30T12:05:00Z"
    assert rows[-1]["date"] == "2024-10-31T12:04:00Z"
    assert calls[0]["startTime"] == str(int(start.timestamp() * 1000))
    assert calls[-1]["endTime"] == str(int(end.timestamp() * 1000))


def test_fetch_historical_candles_rejects_partial_or_misaligned_exact_windows() -> None:
    now = datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc)
    start = datetime(2024, 10, 30, 12, 5, tzinfo=timezone.utc)
    end = start + timedelta(days=1) - timedelta(minutes=1)

    with pytest.raises(ValueError, match="historical_window_requires_both_boundaries"):
        fetcher.fetch_historical_candles(
            days=1, now=now, start_utc=start, request_json=lambda _params: []
        )

    with pytest.raises(
        ValueError, match="historical_window_does_not_match_requested_days"
    ):
        fetcher.fetch_historical_candles(
            days=7,
            now=now,
            start_utc=start,
            end_utc=end,
            request_json=lambda _params: [],
        )

    with pytest.raises(ValueError, match="historical_start_utc_must_align_to_minute"):
        fetcher.fetch_historical_candles(
            days=1,
            now=now,
            start_utc=start.replace(second=1),
            end_utc=end,
            request_json=lambda _params: [],
        )


def test_five_year_window_is_allowed_with_disclosed_gap_policy() -> None:
    assert 1825 in fetcher.ALLOWED_DAYS
    assert fetcher.GAP_POLICY_ALLOW_EXCHANGE_GAPS != fetcher.GAP_POLICY
    assert fetcher.KNOWN_EXCHANGE_GAP_DATE_UTC == "2023-03-24"
    assert fetcher.KNOWN_EXCHANGE_GAP_MISSING_MINUTES == 120


def _sparse_klines(stamps: list[int]) -> list[list[Any]]:
    return [
        [
            stamp,
            "100",
            "101",
            "99",
            "100.5",
            "3.5",
            stamp + 59_999,
            "0",
            0,
            "0",
            "0",
            "0",
        ]
        for stamp in stamps
    ]


def _provider_with_gap(
    start: int, end: int, missing_start: int, missing_end: int
) -> Any:
    real_stamps = [
        stamp
        for stamp in range(start, end + fetcher.INTERVAL_MS, fetcher.INTERVAL_MS)
        if not missing_start <= stamp <= missing_end
    ]

    def provider(params: dict[str, str]) -> list[Any]:
        cursor = int(params["startTime"])
        upper = int(params["endTime"])
        return _sparse_klines(
            [stamp for stamp in real_stamps if cursor <= stamp <= upper][
                : fetcher.PAGE_SIZE
            ]
        )

    return provider


def _known_outage_window() -> tuple[datetime, int, int, int, int]:
    now = datetime(2023, 3, 25, 0, 0, tzinfo=timezone.utc)
    end = int(now.timestamp() * 1000) - fetcher.INTERVAL_MS
    start = end - 1439 * fetcher.INTERVAL_MS
    missing_start = start + 720 * fetcher.INTERVAL_MS
    missing_end = missing_start + 119 * fetcher.INTERVAL_MS
    return now, start, end, missing_start, missing_end


def test_gap_tolerant_fetch_accepts_only_the_documented_2023_outage() -> None:
    now, start, end, missing_start, missing_end = _known_outage_window()
    calls: list[dict[str, str]] = []
    provider = _provider_with_gap(start, end, missing_start, missing_end)

    def tracking_provider(params: dict[str, str]) -> list[Any]:
        calls.append(params.copy())
        return provider(params)

    rows = fetcher.fetch_historical_candles(
        days=1,
        now=now,
        request_json=tracking_provider,
        sleep_fn=lambda _seconds: None,
        allow_exchange_gaps=True,
    )
    stamps = [int(row["timestamp"]) for row in rows]
    assert len(rows) == 1440 - 120
    assert stamps[0] * 1000 == start
    assert stamps[-1] * 1000 == end
    assert all((stamp * 1000) % fetcher.INTERVAL_MS == 0 for stamp in stamps)
    assert fetcher._gap_ranges_from_stamps(stamps) == [
        {
            "start_utc": "2023-03-24T12:00:00Z",
            "end_utc": "2023-03-24T13:59:00Z",
            "missing_minute_intervals": 120,
        }
    ]
    cursors = [int(item["startTime"]) for item in calls]
    assert all(current > previous for previous, current in zip(cursors, cursors[1:]))


def test_gap_tolerant_fetch_rejects_undocumented_gap_date_or_duration() -> None:
    now = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)
    end = int(now.timestamp() * 1000) - fetcher.INTERVAL_MS
    start = end - 1439 * fetcher.INTERVAL_MS
    missing_start = start + 720 * fetcher.INTERVAL_MS
    missing_end = missing_start + 119 * fetcher.INTERVAL_MS
    with pytest.raises(ValueError, match="binance_exchange_gap_out_of_policy"):
        fetcher.fetch_historical_candles(
            days=1,
            now=now,
            request_json=_provider_with_gap(start, end, missing_start, missing_end),
            sleep_fn=lambda _seconds: None,
            allow_exchange_gaps=True,
        )

    now, start, end, missing_start, _ = _known_outage_window()
    with pytest.raises(ValueError, match="binance_exchange_gap_out_of_policy"):
        fetcher.fetch_historical_candles(
            days=1,
            now=now,
            request_json=_provider_with_gap(
                start, end, missing_start, missing_start + 118 * fetcher.INTERVAL_MS
            ),
            sleep_fn=lambda _seconds: None,
            allow_exchange_gaps=True,
        )


def test_gap_tolerant_fetch_rejects_empty_pages_and_truncated_end() -> None:
    now, start, end, missing_start, missing_end = _known_outage_window()
    with pytest.raises(ValueError, match="binance_kline_empty_page_under_gap_policy"):
        fetcher.fetch_historical_candles(
            days=1,
            now=now,
            request_json=lambda _params: [],
            sleep_fn=lambda _seconds: None,
            allow_exchange_gaps=True,
        )

    truncated_provider = _provider_with_gap(
        start, end - fetcher.INTERVAL_MS, missing_start, missing_end
    )
    with pytest.raises(ValueError, match="binance_historical_window_boundary_mismatch"):
        fetcher.fetch_historical_candles(
            days=1,
            now=now,
            request_json=truncated_provider,
            sleep_fn=lambda _seconds: None,
            allow_exchange_gaps=True,
        )


def test_strict_mode_still_fails_closed_on_exchange_gap() -> None:
    now, start, end, missing_start, missing_end = _known_outage_window()
    with pytest.raises(ValueError, match="binance_kline_timestamp_mismatch"):
        fetcher.fetch_historical_candles(
            days=1,
            now=now,
            request_json=_provider_with_gap(start, end, missing_start, missing_end),
            sleep_fn=lambda _seconds: None,
        )


def test_gap_manifest_discloses_real_missing_minutes_without_fill(
    tmp_path: Path,
) -> None:
    now, start, end, missing_start, missing_end = _known_outage_window()
    real_stamps = [
        stamp
        for stamp in range(start, end + fetcher.INTERVAL_MS, fetcher.INTERVAL_MS)
        if not missing_start <= stamp <= missing_end
    ]
    rows = [
        fetcher._parse_gap_tolerant_kline(row) for row in _sparse_klines(real_stamps)
    ]
    manifest = fetcher.write_artifacts(
        rows,
        output_dir=tmp_path,
        days=1,
        retrieved_at=now,
        commit_sha="abc123",
        symbol="ETHUSDT",
        allow_exchange_gaps=True,
    )
    saved = json.loads(
        (tmp_path / "binance-ethusdt-1m.manifest.json").read_text(encoding="utf-8")
    )
    assert saved["row_count"] == 1320
    assert saved["missing_minute_intervals"] == 120
    assert saved["missing_gap_ranges"] == fetcher._gap_ranges_from_stamps(real_stamps)
    assert saved["gap_policy"] == fetcher.GAP_POLICY_ALLOW_EXCHANGE_GAPS
    assert saved["private_api_calls"] == 0 and saved["real_order_attempts"] == 0
    assert saved["Champion"] is None and saved["order_permission"] is False
    assert saved["trading_authorization"] == "NO-GO"
    assert (
        manifest["csv_sha256"]
        == hashlib.sha256(
            (tmp_path / "binance-ethusdt-1m.csv").read_bytes()
        ).hexdigest()
    )
