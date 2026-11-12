from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts import fetch_nobitex_public_ohlcv as fetcher
from scripts.fetch_nobitex_public_ohlcv import CandleRecord


def payload(stamps: list[int]) -> dict[str, Any]:
    count = len(stamps)
    return {
        "s": "ok",
        "t": stamps,
        "o": [100.0] * count,
        "h": [101.0] * count,
        "l": [99.0] * count,
        "c": [100.5] * count,
        "v": [1.0] * count,
    }


def test_exact_range_uses_inclusive_unpadded_request() -> None:
    start = 1_700_000_000
    end = start + 5 * 3600
    calls: list[dict[str, str]] = []

    def provider(params: dict[str, str]) -> dict[str, Any]:
        calls.append(params.copy())
        return payload(list(range(int(params["from"]), int(params["to"]) + 1, 3600)))

    rows = fetcher.fetch_candles_between(
        start_timestamp=start,
        end_timestamp=end,
        interval="1h",
        request_json=provider,
        sleep_fn=lambda _: None,
    )
    assert len(rows) == 6
    assert rows[0]["timestamp"] == start
    assert rows[-1]["timestamp"] == end
    assert calls == [
        {
            "symbol": "BTCUSDT",
            "resolution": "60",
            "from": str(start),
            "to": str(end + 1),
        }
    ]


def test_unresolved_nobitex_gap_fails_closed_after_same_request_retries() -> None:
    start = 1_700_000_000
    end = start + 5 * 3600
    calls: list[dict[str, str]] = []

    def provider(params: dict[str, str]) -> dict[str, Any]:
        calls.append(params.copy())
        return payload([start, start + 3600, start + 3 * 3600, end])

    with pytest.raises(ValueError, match="nobitex_historical_gap_detected"):
        fetcher.fetch_candles_between(
            start_timestamp=start,
            end_timestamp=end,
            interval="1h",
            request_json=provider,
            sleep_fn=lambda _: None,
        )
    assert len(calls) == 3
    assert all(
        int(call["from"]) == start and int(call["to"]) == end + 1 for call in calls
    )


def test_fixed_prehack_window_has_31801_inclusive_candles() -> None:
    assert fetcher.PREHACK_START_TIMESTAMP == 1_635_712_200
    assert fetcher.PREHACK_END_TIMESTAMP == 1_750_192_200
    assert fetcher.PREHACK_REQUESTED_DAYS == 1325
    assert fetcher.PREHACK_EXPECTED_CANDLES == 31801


def test_manifest_is_nobitex_only_and_safety_closed(tmp_path: Path) -> None:
    start = 1_700_000_000
    rows: list[CandleRecord] = [
        {
            "timestamp": start + index * 3600,
            "date": datetime.fromtimestamp(start + index * 3600, timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1.0,
        }
        for index in range(24)
    ]
    manifest = fetcher.write_artifacts(
        rows,
        output_dir=tmp_path,
        days=1,
        retrieved_at=datetime.now(timezone.utc),
        commit_sha="test-commit",
        interval="1h",
    )
    assert manifest["source_policy"] == "NOBITEX_ONLY_NO_BACKFILL"
    assert manifest["data_sources"] == ["Nobitex public UDF history API"]
    assert manifest["expected_candles"] == 24
    assert "source_overrides" not in manifest
    assert manifest["Champion"] is None
    assert manifest["trading_authorization"] == "NO-GO"


def test_public_http_client_honors_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    statuses = [429, 200]
    delays: list[float] = []

    class Response:
        def __init__(self, status: int) -> None:
            self.status = status

        def getheader(self, name: str) -> str | None:
            return "3" if name == "Retry-After" and self.status == 429 else None

        def read(self) -> bytes:
            return b'{"s":"ok"}'

    class Connection:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def request(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def getresponse(self) -> Response:
            return Response(statuses.pop(0))

        def close(self) -> None:
            return None

    monkeypatch.setattr(fetcher, "HTTPSConnection", Connection)
    monkeypatch.setattr(time, "sleep", delays.append)
    assert fetcher.request_public_json({"symbol": "BTCUSDT"}) == {"s": "ok"}
    assert delays == [3.0]


@pytest.mark.parametrize(
    ("interval", "seconds", "resolution"),
    [("15m", 900, "15"), ("30m", 1800, "30")],
)
def test_native_intraday_resolution_is_requested_without_resampling(
    interval: str, seconds: int, resolution: str
) -> None:
    start = 1_700_000_000 // seconds * seconds
    end = start + 2 * seconds
    requests: list[dict[str, str]] = []

    def provider(params: dict[str, str]) -> dict[str, Any]:
        requests.append(params.copy())
        return payload(list(range(start, end + 1, seconds)))

    rows = fetcher.fetch_candles_between(
        start_timestamp=start,
        end_timestamp=end,
        interval=interval,
        request_json=provider,
        sleep_fn=lambda _: None,
    )
    assert len(rows) == 3
    assert requests == [
        {
            "symbol": "BTCUSDT",
            "resolution": resolution,
            "from": str(start),
            "to": str(end + 1),
        }
    ]
