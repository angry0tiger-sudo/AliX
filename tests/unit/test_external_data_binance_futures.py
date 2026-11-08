from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import BytesIO
import json
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit

import pytest

from external_data.binance_futures import (
    ALLOWED_SYMBOLS,
    BINANCE_FUTURES_BASE_URL,
    FUNDING_PAGE_LIMIT,
    OPEN_INTEREST_HISTORY_LIMIT_DAYS,
    BinanceFuturesSnapshot,
    BinanceUsdMFuturesConnector,
    FundingRatePoint,
    OpenInterestPoint,
    UrllibPublicJsonTransport,
    _content_hash,
    _deduplicate_funding,
    _deduplicate_open_interest,
    _epoch_milliseconds,
    _exchange_timestamp,
    _get_payload,
    _numeric,
    _parse_funding_row,
    _parse_open_interest_row,
    _request_url,
    _response_payload,
    snapshot_manifest,
)
from external_data.contracts import ConnectorResponse, ProviderError, RateLimitStatus
import external_data.binance_futures as provider_module

UTC = timezone.utc
START = datetime(2020, 1, 1, tzinfo=UTC)
END = datetime(2021, 1, 1, tzinfo=UTC)
RETRIEVED = datetime(2026, 11, 8, tzinfo=UTC)


def _funding_row(timestamp: datetime, *, rate: str = "0.0001") -> dict[str, object]:
    return {
        "symbol": "BTCUSDT",
        "fundingTime": _epoch_milliseconds(timestamp),
        "fundingRate": rate,
        "markPrice": "50000.0",
    }


def _oi_row(timestamp: datetime, *, interest: str = "100.0") -> dict[str, object]:
    return {
        "symbol": "BTCUSDT",
        "sumOpenInterest": interest,
        "sumOpenInterestValue": "5000000.0",
        "timestamp": _epoch_milliseconds(timestamp),
    }


def _response(payload: object, status: int = 200) -> ConnectorResponse:
    return ConnectorResponse(
        status_code=status,
        body=json.dumps(payload),
        received_at=RETRIEVED,
        rate_limit_status=RateLimitStatus.OK,
    )


class FakeTransport:
    def __init__(self, responses: list[ConnectorResponse | Exception]) -> None:
        self.responses = list(responses)
        self.urls: list[str] = []

    def get(self, url: str) -> ConnectorResponse:
        self.urls.append(url)
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def test_funding_paginates_and_oi_window_is_capped_to_latest_month() -> None:
    page_one = [
        _funding_row(START + timedelta(hours=8 * index), rate=str(index / 100_000))
        for index in range(FUNDING_PAGE_LIMIT)
    ]
    page_two_time = START + timedelta(hours=8 * FUNDING_PAGE_LIMIT)
    oi_start = END - timedelta(days=OPEN_INTEREST_HISTORY_LIMIT_DAYS)
    oi_rows = [_oi_row(oi_start), _oi_row(oi_start + timedelta(hours=4), interest="101.0")]
    transport = FakeTransport(
        [_response(page_one), _response([_funding_row(page_two_time)]), _response(oi_rows)]
    )
    delays: list[float] = []
    connector = BinanceUsdMFuturesConnector(
        transport,
        sleeper=delays.append,
        inter_page_delay_seconds=0.0,
    )

    snapshot = connector.fetch("BTCUSDT", start=START, end=END, oi_period="4h")

    assert snapshot.symbol == "BTCUSDT"
    assert len(snapshot.funding_rates) == FUNDING_PAGE_LIMIT + 1
    assert len(snapshot.open_interest) == 2
    assert snapshot.oi_requested_start == oi_start
    assert snapshot.funding_rates[0].available_at == START + timedelta(minutes=5)
    assert snapshot.open_interest[0].available_at == oi_start + timedelta(hours=4, minutes=5)
    first_query = parse_qs(urlsplit(transport.urls[0]).query)
    second_query = parse_qs(urlsplit(transport.urls[1]).query)
    oi_query = parse_qs(urlsplit(transport.urls[2]).query)
    assert first_query["limit"] == ["1000"]
    assert int(second_query["startTime"][0]) == _epoch_milliseconds(page_two_time)
    assert int(oi_query["startTime"][0]) == _epoch_milliseconds(oi_start)
    assert oi_query["period"] == ["4h"]
    assert delays == [0.0]


def test_fetch_rejects_unapproved_symbol_period_and_invalid_window_before_io() -> None:
    transport = FakeTransport([])
    connector = BinanceUsdMFuturesConnector(transport, sleeper=lambda _: None)

    with pytest.raises(ProviderError, match="symbol_not_allowed"):
        connector.fetch("DOGEUSDT", start=START, end=END)
    with pytest.raises(ProviderError, match="period_not_allowed"):
        connector.fetch("BTCUSDT", start=START, end=END, oi_period="5m")
    with pytest.raises(ProviderError, match="time_window_invalid"):
        connector.fetch("BTCUSDT", start=END, end=START)
    with pytest.raises(ProviderError, match="timezone_required"):
        connector.fetch("BTCUSDT", start="2020-01-01T00:00:00", end=END)
    assert transport.urls == []


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (True, "timestamp_invalid"),
        ("123", "timestamp_invalid"),
        (float("nan"), "timestamp_invalid"),
        (float("inf"), "timestamp_invalid"),
        (-1, "timestamp_invalid"),
        (1e30, "timestamp_invalid"),
    ],
)
def test_exchange_timestamps_reject_invalid_values(value: object, reason: str) -> None:
    with pytest.raises(ProviderError, match=reason):
        _exchange_timestamp(value, "test")


@pytest.mark.parametrize(
    ("value", "positive"),
    [
        (True, False),
        (None, False),
        ("not-a-number", False),
        ("nan", False),
        ("inf", False),
        ("0", True),
        ("-1", True),
    ],
)
def test_numeric_parser_fails_closed(value: object, positive: bool) -> None:
    with pytest.raises(ProviderError, match="numeric_invalid"):
        _numeric(value, "test", positive=positive)


def test_funding_parser_keeps_negative_and_null_mark_price() -> None:
    row = _funding_row(START, rate="-0.001")
    row["markPrice"] = ""
    parsed = _parse_funding_row(row, "BTCUSDT")
    assert parsed.funding_rate == pytest.approx(-0.001)
    assert parsed.mark_price is None
    assert parsed.available_at == START + timedelta(minutes=5)
    assert len(parsed.content_hash) == 64


def test_funding_parser_rejects_wrong_symbol_and_invalid_mark_price() -> None:
    row = _funding_row(START)
    row["symbol"] = "ETHUSDT"
    with pytest.raises(ProviderError, match="symbol_or_schema"):
        _parse_funding_row(row, "BTCUSDT")
    row = _funding_row(START)
    row["markPrice"] = "0"
    with pytest.raises(ProviderError, match="numeric_invalid:markPrice"):
        _parse_funding_row(row, "BTCUSDT")


def test_open_interest_parser_validates_positive_values_and_availability() -> None:
    parsed = _parse_open_interest_row(_oi_row(START), "BTCUSDT", "4h")
    assert parsed.sum_open_interest == 100.0
    assert parsed.available_at == START + timedelta(hours=4, minutes=5)
    row = _oi_row(START, interest="0")
    with pytest.raises(ProviderError, match="numeric_invalid:sumOpenInterest"):
        _parse_open_interest_row(row, "BTCUSDT", "4h")


def test_deduplication_accepts_identical_replays_and_rejects_conflicts() -> None:
    row = _funding_row(START)
    one = _parse_funding_row(row, "BTCUSDT")
    assert _deduplicate_funding([one, one]) == (one,)
    other = _parse_funding_row({**row, "fundingRate": "0.0002"}, "BTCUSDT")
    with pytest.raises(ProviderError, match="funding_timestamp_conflict"):
        _deduplicate_funding([one, other])

    oi_row = _oi_row(START)
    oi_one = _parse_open_interest_row(oi_row, "BTCUSDT", "4h")
    assert _deduplicate_open_interest([oi_one, oi_one]) == (oi_one,)
    oi_other = _parse_open_interest_row(
        {**oi_row, "sumOpenInterest": "120.0"}, "BTCUSDT", "4h"
    )
    with pytest.raises(ProviderError, match="oi_timestamp_conflict"):
        _deduplicate_open_interest([oi_one, oi_other])


def test_request_builder_restricts_endpoint_paths() -> None:
    url = _request_url("/fapi/v1/fundingRate", {"symbol": "BTCUSDT", "limit": 1000})
    assert url.startswith(BINANCE_FUTURES_BASE_URL)
    with pytest.raises(ProviderError, match="endpoint_not_allowed"):
        _request_url("/fapi/v1/order", {"symbol": "BTCUSDT"})


def test_payload_reader_rejects_invalid_responses_and_json() -> None:
    with pytest.raises(ProviderError, match="transport_response_invalid"):
        _response_payload(object())  # type: ignore[arg-type]
    with pytest.raises(ProviderError, match="rate_limit_blocked"):
        _response_payload(
            ConnectorResponse(200, "{}", RETRIEVED, RateLimitStatus.BLOCKED)
        )
    with pytest.raises(ProviderError, match="http_status:401"):
        _response_payload(ConnectorResponse(401, "{}", RETRIEVED))
    with pytest.raises(ProviderError, match="response_too_large"):
        _response_payload(
            ConnectorResponse(200, "x" * (provider_module.MAX_RESPONSE_BYTES + 1), RETRIEVED)
        )
    with pytest.raises(ProviderError, match="json_invalid"):
        _response_payload(ConnectorResponse(200, "{", RETRIEVED))
    assert _response_payload(_response({"ok": True})) == {"ok": True}


def test_get_payload_retries_rate_limits_and_timeouts() -> None:
    transport = FakeTransport(
        [
            _response({"error": "limited"}, status=429),
            TimeoutError("fixture timeout"),
            _response([{"ok": True}]),
        ]
    )
    delays: list[float] = []
    payload = _get_payload(
        transport,
        f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/fundingRate?symbol=BTCUSDT",
        delays.append,
    )
    assert payload == [{"ok": True}]
    assert delays == [1.0, 1.0]


def test_get_payload_fails_closed_for_ban_server_error_transport_and_host() -> None:
    url = f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/fundingRate"
    with pytest.raises(ProviderError, match="ip_banned"):
        _get_payload(FakeTransport([_response({}, status=418)]), url, lambda _: None)

    server_errors = FakeTransport(
        [_response({}, status=500), _response({}, status=502), _response({}, status=503)]
    )
    with pytest.raises(ProviderError, match="http_status:503"):
        _get_payload(server_errors, url, lambda _: None)

    with pytest.raises(ProviderError, match="transport_error"):
        _get_payload(FakeTransport([RuntimeError("no network")]), url, lambda _: None)

    with pytest.raises(ProviderError, match="host_not_allowed"):
        _get_payload(FakeTransport([]), "https://example.com/fapi/v1/fundingRate", lambda _: None)

    with pytest.raises(ProviderError, match="public_timeout"):
        _get_payload(
            FakeTransport([TimeoutError(), TimeoutError(), TimeoutError()]),
            url,
            lambda _: None,
        )


def test_funding_and_oi_fetchers_reject_bad_page_shapes_and_windows() -> None:
    connector = BinanceUsdMFuturesConnector(
        FakeTransport([_response({"not": "a list"})]),
        sleeper=lambda _: None,
    )
    with pytest.raises(ProviderError, match="funding_page_schema_invalid"):
        connector.fetch("BTCUSDT", start=START, end=END)

    outside = _funding_row(END + timedelta(minutes=1))
    connector = BinanceUsdMFuturesConnector(
        FakeTransport([_response([outside])]),
        sleeper=lambda _: None,
    )
    with pytest.raises(ProviderError, match="funding_timestamp_outside_window"):
        connector.fetch("BTCUSDT", start=START, end=END)

    oi_start = END - timedelta(days=30)
    connector = BinanceUsdMFuturesConnector(
        FakeTransport([_response([]), _response([_oi_row(START)])]),
        sleeper=lambda _: None,
    )
    with pytest.raises(ProviderError, match="oi_timestamp_outside_window"):
        connector.fetch("BTCUSDT", start=START, end=END)


def test_connector_allows_empty_pages_but_preserves_empty_history() -> None:
    connector = BinanceUsdMFuturesConnector(
        FakeTransport([_response([]), _response([])]),
        sleeper=lambda _: None,
    )
    snapshot = connector.fetch("BTCUSDT", start=START, end=END)
    assert snapshot.funding_rates == ()
    assert snapshot.open_interest == ()


def _full_snapshot(days: int = 365) -> BinanceFuturesSnapshot:
    end = START + timedelta(days=days)
    funding = tuple(
        FundingRatePoint(
            "BTCUSDT",
            START + timedelta(hours=8 * index),
            0.0001,
            50000.0,
            START + timedelta(hours=8 * index, minutes=5),
            _content_hash({"funding": index}),
        )
        for index in range(days * 3 + 1)
    )
    oi = tuple(
        OpenInterestPoint(
            "BTCUSDT",
            "4h",
            START + timedelta(hours=4 * index),
            100.0 + index,
            5_000_000.0,
            START + timedelta(hours=4 * index, minutes=5),
            _content_hash({"oi": index}),
        )
        for index in range(days * 6 + 1)
    )
    return BinanceFuturesSnapshot(
        "BTCUSDT",
        START,
        end,
        START,
        funding,
        oi,
        RETRIEVED,
        "4h",
    )


def test_snapshot_manifest_accepts_full_coverage_and_keeps_safety_closed() -> None:
    manifest = snapshot_manifest(
        _full_snapshot(),
        commit_sha="a" * 40,
        requested_days=365,
    )
    assert manifest["funding"]["full_requested_history_available"] is True
    assert manifest["open_interest"]["full_requested_history_available"] is True
    assert manifest["private_api_calls"] == 0
    assert manifest["credentials_used"] is False
    assert manifest["Champion"] is None
    assert manifest["order_permission"] is False
    assert manifest["trading_authorization"] == "NO-GO"


def test_snapshot_manifest_blocks_latest_month_oi_and_records_missing_periods() -> None:
    full = _full_snapshot(days=30)
    bounded = BinanceFuturesSnapshot(
        full.symbol,
        START,
        START + timedelta(days=1825),
        START + timedelta(days=1795),
        full.funding_rates,
        full.open_interest,
        RETRIEVED,
        "4h",
    )
    manifest = snapshot_manifest(
        bounded,
        commit_sha="b" * 40,
        requested_days=1825,
    )
    assert manifest["open_interest"]["full_requested_history_available"] is False
    assert manifest["research_status"] == "BLOCKED_INSUFFICIENT_OI_HISTORY"
    assert manifest["open_interest"]["missing_periods_are_not_filled"] is True
    with pytest.raises(ProviderError, match="commit_sha_invalid"):
        snapshot_manifest(bounded, commit_sha="bad", requested_days=1825)
    with pytest.raises(ProviderError, match="requested_days_invalid"):
        snapshot_manifest(bounded, commit_sha="b" * 40, requested_days=0)


class _FakeUrllibResponse:
    status = 200

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "_FakeUrllibResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, maximum: int) -> bytes:
        return self._body[:maximum]


def test_urllib_transport_sends_only_public_get_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: object, *, timeout: float) -> _FakeUrllibResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeUrllibResponse(b'{"status":"ok"}')

    monkeypatch.setattr(provider_module, "urlopen", fake_urlopen)
    transport = UrllibPublicJsonTransport(timeout_seconds=9.0)
    response = transport.get(f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/fundingRate")
    request = captured["request"]
    assert response.status_code == 200
    assert response.body == '{"status":"ok"}'
    assert getattr(request, "method") == "GET"
    assert request.get_header("Authorization") is None
    assert request.get_header("X-mbx-apikey") is None
    assert captured["timeout"] == 9.0


def test_urllib_transport_handles_http_errors_and_rejects_network_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def limited_urlopen(request: object, *, timeout: float) -> object:
        raise HTTPError(
            url="https://fapi.binance.com/fapi/v1/fundingRate",
            code=429,
            msg="rate limited",
            hdrs=None,
            fp=BytesIO(b'{"error":"limited"}'),
        )

    monkeypatch.setattr(provider_module, "urlopen", limited_urlopen)
    response = UrllibPublicJsonTransport().get(
        f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/fundingRate"
    )
    assert response.status_code == 429
    assert response.rate_limit_status == RateLimitStatus.LIMITED

    def failed_urlopen(request: object, *, timeout: float) -> object:
        raise URLError("offline")

    monkeypatch.setattr(provider_module, "urlopen", failed_urlopen)
    with pytest.raises(TimeoutError):
        UrllibPublicJsonTransport().get(
            f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/fundingRate"
        )


def test_urllib_transport_rejects_wrong_host_and_invalid_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = UrllibPublicJsonTransport()
    with pytest.raises(ProviderError, match="host_not_allowed"):
        transport.get("https://example.com/data")

    with pytest.raises(ProviderError, match="encoding_invalid"):
        transport._response_body(b"\xff")

    with pytest.raises(ProviderError, match="response_too_large"):
        transport._response_body(b"x" * (provider_module.MAX_RESPONSE_BYTES + 1))

    with pytest.raises(ProviderError, match="timeout_configuration_invalid"):
        UrllibPublicJsonTransport(timeout_seconds=0)
