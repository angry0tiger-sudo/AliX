"""Public Binance USD-M Funding Rate history with bounded provenance.

Only the allow-listed unauthenticated Funding Rate GET endpoint is implemented.
No Open Interest, account, private, or order endpoint exists in this adapter.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http.client import HTTPSConnection, HTTPException
import hashlib
import json
import math
import time
from typing import Protocol
from urllib.parse import urlencode, urlsplit

from external_data.contracts import (
    ConnectorResponse,
    ProviderError,
    RateLimitStatus,
    parse_timestamp,
)

BINANCE_FUTURES_BASE_URL = "https://fapi.binance.com"
FUNDING_RATE_PATH = "/fapi/v1/fundingRate"
ALLOWED_SYMBOLS = ("BTCUSDT", "ETHUSDT")
FUNDING_PAGE_LIMIT = 1000
PUBLICATION_LAG = timedelta(minutes=5)
MAX_RESPONSE_BYTES = 2_000_000
MAX_RETRY_ATTEMPTS = 3
MAX_FUNDING_PAGES = 1000
FUNDING_MAX_EVENT_GAP = timedelta(days=2)
COVERAGE_BOUNDARY_TOLERANCE = timedelta(days=1)
MANIFEST_SCHEMA_VERSION = "alix.binance.usdm-funding.v1"


class JsonTransport(Protocol):
    """Injected transport contract for deterministic tests and bounded HTTPS."""

    def get(self, url: str) -> ConnectorResponse:
        """Perform one unauthenticated public GET."""


@dataclass(frozen=True, slots=True)
class FundingRatePoint:
    """One provider settlement with a conservative point-in-time availability."""

    symbol: str
    funding_time: datetime
    funding_rate: float
    mark_price: float | None
    available_at: datetime
    content_hash: str


@dataclass(frozen=True, slots=True)
class BinanceFundingSnapshot:
    """Funding observations and the exact requested UTC window."""

    symbol: str
    requested_start: datetime
    requested_end: datetime
    funding_rates: tuple[FundingRatePoint, ...]
    retrieved_at: datetime


def _utc(value: datetime | str, field_name: str) -> datetime:
    return parse_timestamp(value, field_name)


def _validate_request(symbol: str, start: datetime, end: datetime) -> None:
    if symbol not in ALLOWED_SYMBOLS:
        raise ProviderError("binance_futures_symbol_not_allowed")
    if start >= end:
        raise ProviderError("binance_futures_time_window_invalid")


def _epoch_milliseconds(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _exchange_timestamp(value: object, field_name: str) -> datetime:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProviderError(f"binance_timestamp_invalid:{field_name}")
    number = float(value)
    if not math.isfinite(number) or number <= 0 or not number.is_integer():
        raise ProviderError(f"binance_timestamp_invalid:{field_name}")
    try:
        return datetime.fromtimestamp(number / 1000.0, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        raise ProviderError(f"binance_timestamp_invalid:{field_name}") from None


def _numeric(value: object, field_name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ProviderError(f"binance_numeric_invalid:{field_name}")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ProviderError(f"binance_numeric_invalid:{field_name}") from None
    if not math.isfinite(number) or (positive and number <= 0):
        raise ProviderError(f"binance_numeric_invalid:{field_name}")
    return number


def _content_hash(row: Mapping[str, object]) -> str:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_funding_row(row: object, symbol: str) -> FundingRatePoint:
    if not isinstance(row, Mapping) or row.get("symbol") != symbol:
        raise ProviderError("binance_funding_record_symbol_or_schema_invalid")
    funding_time = _exchange_timestamp(row.get("fundingTime"), "fundingTime")
    rate = _numeric(row.get("fundingRate"), "fundingRate")
    raw_mark = row.get("markPrice")
    mark = None if raw_mark in (None, "") else _numeric(
        raw_mark, "markPrice", positive=True
    )
    return FundingRatePoint(
        symbol=symbol,
        funding_time=funding_time,
        funding_rate=rate,
        mark_price=mark,
        available_at=funding_time + PUBLICATION_LAG,
        content_hash=_content_hash(row),
    )


def _request_url(path: str, parameters: Mapping[str, object]) -> str:
    if path != FUNDING_RATE_PATH:
        raise ProviderError("binance_public_endpoint_not_allowed")
    return f"{BINANCE_FUTURES_BASE_URL}{path}?{urlencode(parameters)}"


def _retryable_status(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


def _response_payload(response: ConnectorResponse) -> object:
    if not isinstance(response, ConnectorResponse):
        raise ProviderError("binance_transport_response_invalid")
    if response.rate_limit_status == RateLimitStatus.BLOCKED:
        raise ProviderError("binance_rate_limit_blocked")
    if response.status_code != 200:
        raise ProviderError(f"binance_public_http_status:{response.status_code}")
    if len(response.body.encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise ProviderError("binance_public_response_too_large")
    try:
        return json.loads(response.body)
    except json.JSONDecodeError:
        raise ProviderError("binance_public_json_invalid") from None


def _validate_public_transport_url(url: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "fapi.binance.com"
        or parsed.path != FUNDING_RATE_PATH
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or parsed.fragment
    ):
        raise ProviderError("binance_public_host_or_path_not_allowed")


def _transport_attempt(transport: JsonTransport, url: str) -> ConnectorResponse:
    try:
        response = transport.get(url)
    except TimeoutError:
        raise ProviderError("binance_public_timeout") from None
    except Exception:
        raise ProviderError("binance_public_transport_error") from None
    if not isinstance(response, ConnectorResponse):
        raise ProviderError("binance_transport_response_invalid")
    if response.status_code == 418:
        raise ProviderError("binance_public_ip_banned")
    return response


def _retry_delay(
    response: ConnectorResponse,
    attempt: int,
    sleeper: Callable[[float], None],
) -> None:
    if not _retryable_status(response.status_code):
        raise ProviderError("binance_retry_delay_not_applicable")
    if attempt + 1 == MAX_RETRY_ATTEMPTS:
        raise ProviderError(f"binance_public_http_status:{response.status_code}")
    sleeper(float(2**attempt))


def _get_with_retry(
    transport: JsonTransport,
    url: str,
    sleeper: Callable[[float], None],
) -> ConnectorResponse:
    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:
            response = _transport_attempt(transport, url)
        except ProviderError as error:
            if (
                str(error) != "binance_public_timeout"
                or attempt + 1 == MAX_RETRY_ATTEMPTS
            ):
                raise
            sleeper(float(2**attempt))
            continue
        if _retryable_status(response.status_code):
            _retry_delay(response, attempt, sleeper)
            continue
        return response
    raise ProviderError("binance_public_retry_exhausted")


def _get_payload(
    transport: JsonTransport,
    url: str,
    sleeper: Callable[[float], None],
) -> object:
    _validate_public_transport_url(url)
    return _response_payload(_get_with_retry(transport, url, sleeper))


def _deduplicate_funding(
    records: Sequence[FundingRatePoint],
) -> tuple[FundingRatePoint, ...]:
    by_time: dict[datetime, FundingRatePoint] = {}
    for item in records:
        previous = by_time.get(item.funding_time)
        if previous is not None and previous.content_hash != item.content_hash:
            raise ProviderError("binance_funding_timestamp_conflict")
        by_time[item.funding_time] = item
    return tuple(by_time[key] for key in sorted(by_time))


class BinanceUsdMFuturesConnector:
    """Read-only public Funding Rate adapter with bounded pagination."""

    def __init__(
        self,
        transport: JsonTransport,
        *,
        sleeper: Callable[[float], None] = time.sleep,
        inter_page_delay_seconds: float = 0.25,
    ) -> None:
        if not math.isfinite(inter_page_delay_seconds) or inter_page_delay_seconds < 0:
            raise ProviderError("binance_inter_page_delay_invalid")
        self._transport = transport
        self._sleeper = sleeper
        self._inter_page_delay = inter_page_delay_seconds

    def _funding_page(
        self,
        symbol: str,
        cursor: int,
        end_ms: int,
    ) -> list[FundingRatePoint]:
        url = _request_url(
            FUNDING_RATE_PATH,
            {
                "symbol": symbol,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": FUNDING_PAGE_LIMIT,
            },
        )
        payload = _get_payload(self._transport, url, self._sleeper)
        if not isinstance(payload, list) or len(payload) > FUNDING_PAGE_LIMIT:
            raise ProviderError("binance_funding_page_schema_invalid")
        return [_parse_funding_row(row, symbol) for row in payload]

    @staticmethod
    def _validate_funding_page(
        page: Sequence[FundingRatePoint],
        start: datetime,
        end: datetime,
    ) -> None:
        if any(not start <= record.funding_time <= end for record in page):
            raise ProviderError("binance_funding_timestamp_outside_window")
        if any(
            later.funding_time <= earlier.funding_time
            for earlier, later in zip(page, page[1:])
        ):
            raise ProviderError("binance_funding_page_order_invalid")

    def _next_funding_cursor(
        self,
        page: Sequence[FundingRatePoint],
        cursor: int,
    ) -> int | None:
        if len(page) < FUNDING_PAGE_LIMIT:
            return None
        last_ms = max(_epoch_milliseconds(item.funding_time) for item in page)
        if last_ms < cursor or last_ms >= 2**63 - 1:
            raise ProviderError("binance_funding_pagination_stalled")
        self._sleeper(self._inter_page_delay)
        return last_ms + 1

    def _fetch_funding(
        self, symbol: str, start: datetime, end: datetime
    ) -> tuple[FundingRatePoint, ...]:
        cursor = _epoch_milliseconds(start)
        end_ms = _epoch_milliseconds(end)
        collected: list[FundingRatePoint] = []
        for _ in range(MAX_FUNDING_PAGES):
            page = self._funding_page(symbol, cursor, end_ms)
            if not page:
                break
            self._validate_funding_page(page, start, end)
            collected.extend(page)
            next_cursor = self._next_funding_cursor(page, cursor)
            if next_cursor is None:
                break
            cursor = next_cursor
        else:
            raise ProviderError("binance_funding_page_limit_exceeded")
        return _deduplicate_funding(collected)

    def fetch(
        self,
        symbol: str,
        *,
        start: datetime | str,
        end: datetime | str,
    ) -> BinanceFundingSnapshot:
        """Fetch all available Funding Rate settlements in the requested window."""
        start_utc = _utc(start, "start")
        end_utc = _utc(end, "end")
        _validate_request(symbol, start_utc, end_utc)
        funding = self._fetch_funding(symbol, start_utc, end_utc)
        return BinanceFundingSnapshot(
            symbol=symbol,
            requested_start=start_utc,
            requested_end=end_utc,
            funding_rates=funding,
            retrieved_at=datetime.now(timezone.utc),
        )


class BinancePublicHttpsTransport:
    """Verified-TLS HTTPS GET transport with a fixed public endpoint allow-list."""

    def __init__(self, *, timeout_seconds: float = 20.0) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ProviderError("binance_timeout_configuration_invalid")
        self._timeout = timeout_seconds

    @staticmethod
    def _response_body(payload: bytes) -> str:
        if len(payload) > MAX_RESPONSE_BYTES:
            raise ProviderError("binance_public_response_too_large")
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError:
            raise ProviderError("binance_public_response_encoding_invalid") from None

    @staticmethod
    def _validated_request_target(url: str) -> tuple[str, str]:
        _validate_public_transport_url(url)
        parsed = urlsplit(url)
        target = parsed.path
        if parsed.query:
            target = f"{target}?{parsed.query}"
        return "fapi.binance.com", target

    def get(self, url: str) -> ConnectorResponse:
        host, target = self._validated_request_target(url)
        received_at = datetime.now(timezone.utc)
        connection = HTTPSConnection(host, port=443, timeout=self._timeout)
        try:
            connection.request(
                "GET",
                target,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "AliX-AI6-Funding-Research/1.0",
                },
            )
            response = connection.getresponse()
            body = self._response_body(response.read(MAX_RESPONSE_BYTES + 1))
            status = int(response.status)
            rate_status = (
                RateLimitStatus.LIMITED
                if status == 429
                else RateLimitStatus.BLOCKED
                if status in (418, 403)
                else RateLimitStatus.OK
            )
            return ConnectorResponse(
                status_code=status,
                body=body,
                received_at=received_at,
                rate_limit_status=rate_status,
            )
        except ProviderError:
            raise
        except (OSError, TimeoutError, HTTPException):
            raise TimeoutError("binance_public_request_timeout_or_network_error") from None
        finally:
            connection.close()


def _funding_coverage_metrics(
    snapshot: BinanceFundingSnapshot,
) -> tuple[float, float, int, bool]:
    records = snapshot.funding_rates
    if not records:
        return 0.0, 0.0, 0, False
    first = records[0].funding_time
    last = records[-1].funding_time
    span = max(0.0, (last - first).total_seconds() / 86400.0)
    max_gap_hours = max(
        (
            (later.funding_time - earlier.funding_time).total_seconds() / 3600.0
            for earlier, later in zip(records, records[1:])
        ),
        default=0.0,
    )
    covers_boundaries = (
        first <= snapshot.requested_start + COVERAGE_BOUNDARY_TOLERANCE
        and last >= snapshot.requested_end - COVERAGE_BOUNDARY_TOLERANCE
    )
    complete = covers_boundaries and max_gap_hours <= FUNDING_MAX_EVENT_GAP.total_seconds() / 3600
    expected_events = max(1, int(round((snapshot.requested_end - snapshot.requested_start).total_seconds() / (8 * 3600))) + 1)
    coverage_fraction = min(1.0, len(records) / expected_events)
    missing_estimate = max(0, expected_events - len(records))
    return span, max_gap_hours, missing_estimate, complete


def _validate_snapshot_manifest_request(commit_sha: str, requested_days: int) -> None:
    if len(commit_sha) != 40 or any(
        character not in "0123456789abcdef" for character in commit_sha
    ):
        raise ProviderError("binance_commit_sha_invalid")
    if isinstance(requested_days, bool) or not isinstance(requested_days, int) or requested_days <= 0:
        raise ProviderError("binance_requested_days_invalid")


def snapshot_manifest(
    snapshot: BinanceFundingSnapshot,
    *,
    commit_sha: str,
    requested_days: int,
) -> dict[str, object]:
    """Describe exact Funding coverage and closed safety state without fabricating gaps."""
    _validate_snapshot_manifest_request(commit_sha, requested_days)
    span, max_gap_hours, missing_estimate, complete = _funding_coverage_metrics(snapshot)
    records = snapshot.funding_rates
    first = records[0].funding_time.isoformat() if records else None
    last = records[-1].funding_time.isoformat() if records else None
    expected_events = max(
        1,
        int(round((snapshot.requested_end - snapshot.requested_start).total_seconds() / (8 * 3600))) + 1,
    )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "commit_sha": commit_sha,
        "provider": "Binance USD-M public Funding Rate History",
        "host": "fapi.binance.com",
        "endpoint": FUNDING_RATE_PATH,
        "symbol": snapshot.symbol,
        "requested_days": requested_days,
        "requested_start_utc": snapshot.requested_start.isoformat(),
        "requested_end_utc": snapshot.requested_end.isoformat(),
        "retrieved_at": snapshot.retrieved_at.isoformat(),
        "public_api_only": True,
        "credentials_used": False,
        "api_key_header_sent": False,
        "private_api_calls": 0,
        "exchange_order_calls": 0,
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
        "funding": {
            "row_count": len(records),
            "first_funding_time": first,
            "last_funding_time": last,
            "observed_span_days": span,
            "max_event_gap_hours": max_gap_hours,
            "expected_settlements_estimate": expected_events,
            "missing_settlements_estimate": missing_estimate,
            "coverage_fraction_estimate": (
                min(1.0, len(records) / expected_events) if records else 0.0
            ),
            "coverage_status": "FULL_REQUESTED_HISTORY" if complete else "INSUFFICIENT_HISTORY",
            "full_requested_history_available": complete,
            "available_at_policy": "fundingTime + 5 minutes (conservative research proxy)",
        },
        "research_status": (
            "READY_FOR_FUNDING_FEATURE_JOIN"
            if complete
            else "BLOCKED_INSUFFICIENT_FUNDING_HISTORY"
        ),
    }
