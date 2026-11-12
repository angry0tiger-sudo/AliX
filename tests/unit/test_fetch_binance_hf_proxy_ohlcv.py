from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import fetch_nobitex_hf_ohlcv as fetcher


def _kline(timestamp: int, interval_seconds: int) -> list[Any]:
    return [
        timestamp * 1000,
        "100.0",
        "101.0",
        "99.0",
        "100.5",
        "2.0",
        (timestamp + interval_seconds) * 1000 - 1,
        "200.0",
        10,
        "1.0",
        "100.0",
        "0",
    ]


def _rows(start: int, count: int, interval_seconds: int) -> list[fetcher.ProxyCandle]:
    return [
        fetcher._parse_kline(
            _kline(start + index * interval_seconds, interval_seconds),
            interval="15m" if interval_seconds == 900 else "30m",
            requested_start=start,
            requested_end=start + (count - 1) * interval_seconds,
        )
        for index in range(count)
    ]


@pytest.mark.parametrize(("interval", "seconds"), [("15m", 900), ("30m", 1800)])
def test_proxy_interval_policy_and_exact_full_history_count(
    interval: str, seconds: int
) -> None:
    assert fetcher.interval_seconds(interval) == seconds
    expected = (
        (fetcher.PROXY_END_TIMESTAMP - fetcher.PROXY_START_TIMESTAMP) // seconds
    ) + 1
    assert expected == (127201 if interval == "15m" else 63601)


def test_proxy_rejects_unsupported_and_unaligned_ranges() -> None:
    start = (1_700_000_000 // 900) * 900
    with pytest.raises(ValueError, match="interval_out_of_policy"):
        fetcher.fetch_proxy_intraday(
            "1h", start_timestamp=start, end_timestamp=start + 3600
        )
    with pytest.raises(ValueError, match="not_interval_aligned"):
        fetcher.fetch_proxy_intraday(
            "15m",
            start_timestamp=start + 1,
            end_timestamp=start + 10 * 900,
        )


@pytest.mark.parametrize(("interval", "seconds"), [("15m", 900), ("30m", 1800)])
def test_fetch_proxy_intraday_uses_public_klines_and_paginates(
    interval: str, seconds: int
) -> None:
    start = (1_700_000_000 // seconds) * seconds
    count = 1002
    end = start + (count - 1) * seconds
    requests: list[dict[str, str]] = []
    sleeps: list[float] = []

    provider_page_limit = 995

    def provider(params: dict[str, str]) -> list[list[Any]]:
        requests.append(params.copy())
        cursor = int(params["startTime"]) // 1000
        limit = int(params["limit"])
        returned_limit = min(limit, provider_page_limit)
        page_end = min(end, cursor + (returned_limit - 1) * seconds)
        return [
            _kline(timestamp, seconds)
            for timestamp in range(cursor, page_end + 1, seconds)
        ]

    rows, request_count = fetcher.fetch_proxy_intraday(
        interval,
        start_timestamp=start,
        end_timestamp=end,
        request_json=provider,
        sleep_fn=sleeps.append,
    )

    assert len(rows) == count
    assert request_count == 2
    assert rows[0]["timestamp"] == start
    assert rows[-1]["timestamp"] == end
    first_page_request_end = (
        start + (fetcher.BINANCE_MAX_CANDLES_PER_REQUEST - 1) * seconds
    )
    second_page_start = start + provider_page_limit * seconds
    assert requests[0] == {
        "symbol": "BTCUSDT",
        "interval": interval,
        "startTime": str(start * 1000),
        "endTime": str(first_page_request_end * 1000),
        "limit": str(fetcher.BINANCE_MAX_CANDLES_PER_REQUEST),
    }
    assert requests[1] == {
        "symbol": "BTCUSDT",
        "interval": interval,
        "startTime": str(second_page_start * 1000),
        "endTime": str(end * 1000),
        "limit": str(fetcher.BINANCE_MAX_CANDLES_PER_REQUEST),
    }
    assert len(sleeps) == 1
    assert sleeps == [fetcher.BINANCE_PAGE_SLEEP_SECONDS]


def test_fetch_proxy_rejects_incomplete_page_instead_of_accepting_a_gap() -> None:
    seconds = 900
    start = (1_700_000_000 // seconds) * seconds
    end = start + 1000 * seconds

    def provider(params: dict[str, str]) -> list[list[Any]]:
        cursor = int(params["startTime"]) // 1000
        return [
            _kline(cursor + index * seconds, seconds)
            for index in range(1000)
            if index != 20
        ]

    with pytest.raises(ValueError, match="page_gap_detected"):
        fetcher.fetch_proxy_intraday(
            "15m",
            start_timestamp=start,
            end_timestamp=end,
            request_json=provider,
            sleep_fn=lambda _delay: None,
        )


def test_fetch_proxy_rejects_short_page_when_continuation_is_empty() -> None:
    seconds = 900
    start = (1_700_000_000 // seconds) * seconds
    end = start + 1001 * seconds
    requests: list[dict[str, str]] = []

    def provider(params: dict[str, str]) -> list[list[Any]]:
        requests.append(params.copy())
        if len(requests) == 1:
            return [_kline(start + index * seconds, seconds) for index in range(995)]
        return []

    with pytest.raises(ValueError, match="empty_page"):
        fetcher.fetch_proxy_intraday(
            "15m",
            start_timestamp=start,
            end_timestamp=end,
            request_json=provider,
            sleep_fn=lambda _delay: None,
        )

    assert len(requests) == 2
    assert int(requests[1]["startTime"]) // 1000 == start + 995 * seconds


def test_fetch_proxy_rejects_empty_api_page() -> None:
    seconds = 900
    start = (1_700_000_000 // seconds) * seconds
    with pytest.raises(ValueError, match="empty_page"):
        fetcher.fetch_proxy_intraday(
            "15m",
            start_timestamp=start,
            end_timestamp=start + 2 * seconds,
            request_json=lambda _params: [],
            sleep_fn=lambda _delay: None,
        )


def test_parse_kline_rejects_invalid_timestamp_or_ohlc() -> None:
    seconds = 900
    start = (1_700_000_000 // seconds) * seconds
    with pytest.raises(ValueError, match="open_time_invalid"):
        fetcher._parse_kline(
            ["invalid", "100", "101", "99", "100", "1"],
            interval="15m",
            requested_start=start,
            requested_end=start,
        )
    invalid_ohlc = _kline(start, seconds)
    invalid_ohlc[2] = "98"
    with pytest.raises(ValueError, match="ohlc_invariant_failed"):
        fetcher._parse_kline(
            invalid_ohlc,
            interval="15m",
            requested_start=start,
            requested_end=start,
        )
    non_finite = _kline(start, seconds)
    non_finite[5] = "NaN"
    with pytest.raises(ValueError, match="not_finite"):
        fetcher._parse_kline(
            non_finite,
            interval="15m",
            requested_start=start,
            requested_end=start,
        )


def test_public_client_honors_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    statuses = [429, 200]
    delays: list[float] = []
    requests: list[str] = []

    class Response:
        def __init__(self, status: int) -> None:
            self.status = status

        def getheader(self, name: str) -> str | None:
            return "2" if name == "Retry-After" and self.status == 429 else None

        def read(self) -> bytes:
            return b"[]"

    class Connection:
        def __init__(self, host: str, **_kwargs: Any) -> None:
            assert host == fetcher.BINANCE_HOST

        def request(self, method: str, path: str, **_kwargs: Any) -> None:
            requests.append(f"{method} {path}")

        def getresponse(self) -> Response:
            return Response(statuses.pop(0))

        def close(self) -> None:
            return None

    monkeypatch.setattr(fetcher, "HTTPSConnection", Connection)
    assert (
        fetcher.request_binance_public_json(
            {"symbol": "BTCUSDT", "interval": "15m"},
            sleep_fn=delays.append,
        )
        == []
    )
    assert delays == [2.0]
    assert len(requests) == 2
    assert all("api-key" not in item.lower() for item in requests)


def test_public_client_rejects_non_retryable_http_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        status = 400

        def getheader(self, _name: str) -> None:
            return None

        def read(self) -> bytes:
            return b"{}"

    class Connection:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def request(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def getresponse(self) -> Response:
            return Response()

        def close(self) -> None:
            return None

    monkeypatch.setattr(fetcher, "HTTPSConnection", Connection)
    with pytest.raises(RuntimeError, match="http_status_invalid:400"):
        fetcher.request_binance_public_json({"symbol": "BTCUSDT"})


def test_retry_after_above_safe_limit_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="exceeds_safe_wait"):
        fetcher._retry_delay_seconds(0, "301")


def test_proxy_writer_generates_sha256_manifest_and_research_labels(
    tmp_path: Path,
) -> None:
    seconds = 900
    start = (1_700_000_000 // seconds) * seconds
    end = start + 2 * seconds
    manifest = fetcher.write_proxy_artifacts(
        _rows(start, 3, seconds),
        interval="15m",
        output_dir=tmp_path,
        start_timestamp=start,
        end_timestamp=end,
        retrieved_at=datetime.now(timezone.utc),
        commit_sha="a" * 40,
        public_api_requests=1,
    )
    csv_path = tmp_path / "PROXY_BINANCE_15M.csv"
    manifest_path = tmp_path / "PROXY_BINANCE_15M.manifest.json"
    assert csv_path.exists()
    assert manifest_path.exists()
    assert json.loads(manifest_path.read_text()) == manifest
    assert manifest["dataset_id"] == "PROXY_BINANCE_15M"
    assert manifest["dataset_role"] == "RESEARCH_PROXY_ONLY"
    assert manifest["source_policy"] == "BINANCE_PROXY_RESEARCH_ONLY"
    assert manifest["execution_venue"] == "NOBITEX"
    assert manifest["public_api_requests"] == 1
    assert manifest["credentials_used"] is False
    assert manifest["private_api_calls"] == 0
    assert manifest["trading_authorization"] == "NO-GO"
    assert len(manifest["csv_sha256"]) == 64


def test_proxy_writer_rejects_missing_rows_and_invalid_request_count(
    tmp_path: Path,
) -> None:
    seconds = 900
    start = (1_700_000_000 // seconds) * seconds
    end = start + 2 * seconds
    with pytest.raises(ValueError, match="exact_range_coverage_mismatch"):
        fetcher.write_proxy_artifacts(
            _rows(start, 2, seconds),
            interval="15m",
            output_dir=tmp_path,
            start_timestamp=start,
            end_timestamp=end,
            retrieved_at=datetime.now(timezone.utc),
            commit_sha="a" * 40,
            public_api_requests=1,
        )
    with pytest.raises(ValueError, match="request_count_invalid"):
        fetcher.write_proxy_artifacts(
            _rows(start, 3, seconds),
            interval="15m",
            output_dir=tmp_path,
            start_timestamp=start,
            end_timestamp=end,
            retrieved_at=datetime.now(timezone.utc),
            commit_sha="a" * 40,
            public_api_requests=0,
        )


def test_fetch_proxy_cli_writes_both_requested_artifact_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    seconds = 900
    start = (1_700_000_000 // seconds) * seconds
    rows = _rows(start, 3, seconds)
    monkeypatch.setattr(fetcher, "PROXY_START_TIMESTAMP", start)
    monkeypatch.setattr(fetcher, "PROXY_END_TIMESTAMP", start + 2 * seconds)
    monkeypatch.setattr(fetcher, "fetch_proxy_intraday", lambda _interval: (rows, 2))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fetcher",
            "--interval",
            "15m",
            "--output-dir",
            str(tmp_path),
            "--commit-sha",
            "b" * 40,
        ],
    )
    assert fetcher.main() == 0
    assert (tmp_path / "PROXY_BINANCE_15M.csv").exists()
    assert (tmp_path / "PROXY_BINANCE_15M.manifest.json").exists()
