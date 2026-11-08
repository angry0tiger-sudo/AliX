from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from urllib.parse import parse_qs, urlsplit

import pytest

from external_data.binance_futures import (
    BINANCE_FUTURES_BASE_URL,
    FUNDING_PAGE_LIMIT,
    BinanceFundingSnapshot,
    BinancePublicHttpsTransport,
    BinanceUsdMFuturesConnector,
    FundingRatePoint,
    _content_hash,
    _deduplicate_funding,
    _epoch_milliseconds,
    _exchange_timestamp,
    _get_payload,
    _numeric,
    _parse_funding_row,
    _request_url,
    _response_payload,
    snapshot_manifest,
)
from external_data.contracts import ConnectorResponse, ProviderError, RateLimitStatus
import external_data.binance_futures as provider_module

UTC = timezone.utc
START = datetime(2020, 1, 1, tzinfo=UTC)
END = START + timedelta(days=365) - timedelta(minutes=1)
RETRIEVED = datetime(2026, 11, 8, tzinfo=UTC)


def _funding_row(timestamp: datetime, *, rate: str = "0.0001") -> dict[str, object]:
    return {
        "symbol": "BTCUSDT",
        "fundingTime": _epoch_milliseconds(timestamp),
        "fundingRate": rate,
        "markPrice": "50000.0",
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


def _full_snapshot(days: int = 365) -> BinanceFundingSnapshot:
    end = START + timedelta(days=days) - timedelta(minutes=1)
    count = days * 3 + 1
    rates = tuple(
        FundingRatePoint(
            "BTCUSDT",
            START + timedelta(hours=8 * index),
            0.0001 + index / 100_000_000,
            50000.0,
            START + timedelta(hours=8 * index, minutes=5),
            _content_hash({"funding": index}),
        )
        for index in range(count)
    )
    return BinanceFundingSnapshot("BTCUSDT", START, end, rates, RETRIEVED)


def test_funding_paginates_and_resumes_after_last_observed_timestamp() -> None:
    page_one = [
        _funding_row(START + timedelta(hours=8 * index), rate=str(index / 100_000))
        for index in range(FUNDING_PAGE_LIMIT)
    ]
    page_two_time = START + timedelta(hours=8 * FUNDING_PAGE_LIMIT)
    transport = FakeTransport(
        [_response(page_one), _response([_funding_row(page_two_time)])]
    )
    delays: list[float] = []
    connector = BinanceUsdMFuturesConnector(
        transport, sleeper=delays.append, inter_page_delay_seconds=0.0
    )

    snapshot = connector.fetch(
        "BTCUSDT",
        start=START,
        end=START + timedelta(days=400),
    )

    assert snapshot.symbol == "BTCUSDT"
    assert len(snapshot.funding_rates) == FUNDING_PAGE_LIMIT + 1
    assert snapshot.funding_rates[0].available_at == START + timedelta(minutes=5)
    first_query = parse_qs(urlsplit(transport.urls[0]).query)
    second_query = parse_qs(urlsplit(transport.urls[1]).query)
    assert first_query["limit"] == ["1000"]
    last_page_time = START + timedelta(hours=8 * (FUNDING_PAGE_LIMIT - 1))
    assert int(second_query["startTime"][0]) == _epoch_milliseconds(last_page_time) + 1
    assert delays == [0.0]
    assert all(urlsplit(url).path == "/fapi/v1/fundingRate" for url in transport.urls)


def test_fetch_rejects_unapproved_symbols_and_invalid_windows_before_io() -> None:
    transport = FakeTransport([])
    connector = BinanceUsdMFuturesConnector(transport, sleeper=lambda _: None)

    with pytest.raises(ProviderError, match="symbol_not_allowed"):
        connector.fetch("DOGEUSDT", start=START, end=END)
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


def test_funding_parser_preserves_negative_rates_and_null_mark_price() -> None:
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


def test_deduplication_accepts_identical_replays_and_rejects_conflicts() -> None:
    row = _funding_row(START)
    one = _parse_funding_row(row, "BTCUSDT")
    assert _deduplicate_funding([one, one]) == (one,)
    changed = _parse_funding_row({**row, "fundingRate": "0.0002"}, "BTCUSDT")
    with pytest.raises(ProviderError, match="funding_timestamp_conflict"):
        _deduplicate_funding([one, changed])


def test_request_builder_allows_only_the_public_funding_endpoint() -> None:
    url = _request_url("/fapi/v1/fundingRate", {"symbol": "BTCUSDT", "limit": 1000})
    assert url.startswith(BINANCE_FUTURES_BASE_URL)
    with pytest.raises(ProviderError, match="endpoint_not_allowed"):
        _request_url("/fapi/v1/order", {"symbol": "BTCUSDT"})
    with pytest.raises(ProviderError, match="endpoint_not_allowed"):
        _request_url("/futures/data/openInterestHist", {"symbol": "BTCUSDT"})


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
            ConnectorResponse(
                200, "x" * (provider_module.MAX_RESPONSE_BYTES + 1), RETRIEVED
            )
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
    assert delays == [1.0, 2.0]


def test_get_payload_fails_closed_for_ban_server_error_transport_and_host() -> None:
    url = f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/fundingRate"
    with pytest.raises(ProviderError, match="ip_banned"):
        _get_payload(FakeTransport([_response({}, status=418)]), url, lambda _: None)

    server_errors = FakeTransport(
        [
            _response({}, status=500),
            _response({}, status=502),
            _response({}, status=503),
        ]
    )
    with pytest.raises(ProviderError, match="http_status:503"):
        _get_payload(server_errors, url, lambda _: None)

    with pytest.raises(ProviderError, match="transport_error"):
        _get_payload(FakeTransport([RuntimeError("offline")]), url, lambda _: None)

    with pytest.raises(ProviderError, match="host_or_path_not_allowed"):
        _get_payload(
            FakeTransport([]), "https://example.com/fapi/v1/fundingRate", lambda _: None
        )
    with pytest.raises(ProviderError, match="host_or_path_not_allowed"):
        _get_payload(
            FakeTransport([]), "https://fapi.binance.com/fapi/v1/order", lambda _: None
        )

    with pytest.raises(ProviderError, match="public_timeout"):
        _get_payload(
            FakeTransport([TimeoutError(), TimeoutError(), TimeoutError()]),
            url,
            lambda _: None,
        )


def test_funding_fetcher_rejects_bad_page_shapes_and_out_of_window_rows() -> None:
    connector = BinanceUsdMFuturesConnector(
        FakeTransport([_response({"not": "a list"})]),
        sleeper=lambda _: None,
    )
    with pytest.raises(ProviderError, match="funding_page_schema_invalid"):
        connector.fetch("BTCUSDT", start=START, end=END)

    outside = _funding_row(END + timedelta(minutes=1))
    connector = BinanceUsdMFuturesConnector(
        FakeTransport([_response([outside])]), sleeper=lambda _: None
    )
    with pytest.raises(ProviderError, match="funding_timestamp_outside_window"):
        connector.fetch("BTCUSDT", start=START, end=END)

    unsorted = [
        _funding_row(START + timedelta(hours=8)),
        _funding_row(START),
    ]
    connector = BinanceUsdMFuturesConnector(
        FakeTransport([_response(unsorted)]), sleeper=lambda _: None
    )
    with pytest.raises(ProviderError, match="funding_page_order_invalid"):
        connector.fetch("BTCUSDT", start=START, end=END)


def test_connector_preserves_empty_funding_history() -> None:
    connector = BinanceUsdMFuturesConnector(
        FakeTransport([_response([])]), sleeper=lambda _: None
    )
    snapshot = connector.fetch("BTCUSDT", start=START, end=END)
    assert snapshot.funding_rates == ()


def test_snapshot_manifest_accepts_full_funding_coverage_and_closed_safety() -> None:
    manifest = snapshot_manifest(
        _full_snapshot(),
        commit_sha="a" * 40,
        requested_days=365,
    )
    funding = manifest["funding"]
    assert isinstance(funding, dict)
    assert funding["full_requested_history_available"] is True
    assert funding["coverage_status"] == "FULL_REQUESTED_HISTORY"
    assert funding["coverage_fraction_estimate"] >= 0.90
    assert manifest["endpoint"] == "/fapi/v1/fundingRate"
    assert "open_interest" not in manifest
    assert manifest["private_api_calls"] == 0
    assert manifest["credentials_used"] is False
    assert manifest["Champion"] is None
    assert manifest["order_permission"] is False
    assert manifest["trading_authorization"] == "NO-GO"


def test_snapshot_manifest_blocks_sparse_or_incomplete_funding_history() -> None:
    complete = _full_snapshot(days=30)
    sparse = BinanceFundingSnapshot(
        complete.symbol,
        complete.requested_start,
        complete.requested_end,
        complete.funding_rates[::4],
        RETRIEVED,
    )
    manifest = snapshot_manifest(
        sparse,
        commit_sha="b" * 40,
        requested_days=1825,
    )
    funding = manifest["funding"]
    assert isinstance(funding, dict)
    assert funding["full_requested_history_available"] is False
    assert manifest["research_status"] == "BLOCKED_INSUFFICIENT_FUNDING_HISTORY"
    with pytest.raises(ProviderError, match="commit_sha_invalid"):
        snapshot_manifest(sparse, commit_sha="bad", requested_days=1825)
    with pytest.raises(ProviderError, match="requested_days_invalid"):
        snapshot_manifest(sparse, commit_sha="b" * 40, requested_days=0)


class _FakeHttpsResponse:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    def read(self, maximum: int) -> bytes:
        return self._body[:maximum]


class _FakeHttpsConnection:
    def __init__(
        self,
        response: _FakeHttpsResponse | None = None,
        *,
        failure: Exception | None = None,
    ) -> None:
        self.response = response
        self.failure = failure
        self.requests: list[tuple[str, str, dict[str, str]]] = []
        self.closed = False

    def request(self, method: str, target: str, *, headers: dict[str, str]) -> None:
        self.requests.append((method, target, headers))
        if self.failure is not None:
            raise self.failure

    def getresponse(self) -> _FakeHttpsResponse:
        if self.response is None:
            raise AssertionError("fixture response missing")
        return self.response

    def close(self) -> None:
        self.closed = True


def test_https_transport_sends_only_public_get_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeHttpsConnection(_FakeHttpsResponse(200, b'{"status":"ok"}'))
    captured: dict[str, object] = {}

    def fake_connection(
        host: str, *, port: int, timeout: float
    ) -> _FakeHttpsConnection:
        captured.update({"host": host, "port": port, "timeout": timeout})
        return fake

    monkeypatch.setattr(provider_module, "HTTPSConnection", fake_connection)
    response = BinancePublicHttpsTransport(timeout_seconds=9.0).get(
        f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/fundingRate"
    )
    method, target, headers = fake.requests[0]
    assert response.status_code == 200
    assert response.body == '{"status":"ok"}'
    assert method == "GET"
    assert target == "/fapi/v1/fundingRate"
    assert headers["Accept"] == "application/json"
    assert "Authorization" not in headers
    assert "X-mbx-apikey" not in headers
    assert captured == {"host": "fapi.binance.com", "port": 443, "timeout": 9.0}
    assert fake.closed is True


def test_https_transport_maps_rate_limits_and_network_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limited = _FakeHttpsConnection(_FakeHttpsResponse(429, b'{"error":"limited"}'))
    monkeypatch.setattr(
        provider_module,
        "HTTPSConnection",
        lambda host, *, port, timeout: limited,
    )
    response = BinancePublicHttpsTransport().get(
        f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/fundingRate"
    )
    assert response.status_code == 429
    assert response.rate_limit_status == RateLimitStatus.LIMITED

    failed = _FakeHttpsConnection(failure=OSError("offline"))
    monkeypatch.setattr(
        provider_module,
        "HTTPSConnection",
        lambda host, *, port, timeout: failed,
    )
    with pytest.raises(TimeoutError):
        BinancePublicHttpsTransport().get(
            f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/fundingRate"
        )
    assert failed.closed is True


def test_https_transport_rejects_wrong_host_path_bytes_and_timeout() -> None:
    transport = BinancePublicHttpsTransport()
    with pytest.raises(ProviderError, match="host_or_path_not_allowed"):
        transport.get("https://example.com/data")
    with pytest.raises(ProviderError, match="host_or_path_not_allowed"):
        transport.get("https://fapi.binance.com/fapi/v1/order")
    with pytest.raises(ProviderError, match="host_or_path_not_allowed"):
        transport.get("http://fapi.binance.com/fapi/v1/fundingRate")
    with pytest.raises(ProviderError, match="encoding_invalid"):
        transport._response_body(b"\xff")
    with pytest.raises(ProviderError, match="response_too_large"):
        transport._response_body(b"x" * (provider_module.MAX_RESPONSE_BYTES + 1))
    with pytest.raises(ProviderError, match="timeout_configuration_invalid"):
        BinancePublicHttpsTransport(timeout_seconds=0)
