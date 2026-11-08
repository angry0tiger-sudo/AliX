"""Public Binance USD-M funding and open-interest history with bounded provenance.

Only unauthenticated public market-data GET endpoints are implemented. The
Binance Open Interest History endpoint is capped to the latest 30 days; the
connector records that limitation and never fabricates older observations.
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
OPEN_INTEREST_HISTORY_PATH = "/futures/data/openInterestHist"
ALLOWED_SYMBOLS = ("BTCUSDT", "ETHUSDT")
ALLOWED_OI_PERIODS = ("4h",)
FUNDING_PAGE_LIMIT = 1000
OPEN_INTEREST_PAGE_LIMIT = 500
OPEN_INTEREST_HISTORY_LIMIT_DAYS = 30
PUBLICATION_LAG = timedelta(minutes=5)
MAX_RESPONSE_BYTES = 2_000_000
MAX_RETRY_ATTEMPTS = 3
MAX_FUNDING_PAGES = 1000


class JsonTransport(Protocol):
    """Small transport contract; tests inject fixtures and Actions inject urllib."""

    def get(self, url: str) -> ConnectorResponse:
        """Perform one unauthenticated GET and return the response envelope."""


@dataclass(frozen=True, slots=True)
class FundingRatePoint:
    """One Binance funding settlement observation with conservative availability."""

    symbol: str
    funding_time: datetime
    funding_rate: float
    mark_price: float | None
    available_at: datetime
    content_hash: str


@dataclass(frozen=True, slots=True)
class OpenInterestPoint:
    """One period-aggregated open-interest observation."""

    symbol: str
    period: str
    timestamp: datetime
    sum_open_interest: float
    sum_open_interest_value: float | None
    available_at: datetime
    content_hash: str


@dataclass(frozen=True, slots=True)
class BinanceFuturesSnapshot:
    """Fetched public series and the actual historical window returned."""

    symbol: str
    requested_start: datetime
    requested_end: datetime
    oi_requested_start: datetime
    funding_rates: tuple[FundingRatePoint, ...]
    open_interest: tuple[OpenInterestPoint, ...]
    retrieved_at: datetime
    oi_period: str


def _utc(value: datetime | str, field_name: str) -> datetime:
    return parse_timestamp(value, field_name)


def _validate_request(symbol: str, start: datetime, end: datetime, period: str) -> None:
    if symbol not in ALLOWED_SYMBOLS:
        raise ProviderError("binance_futures_symbol_not_allowed")
    if period not in ALLOWED_OI_PERIODS:
        raise ProviderError("binance_oi_period_not_allowed")
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
    mark = (
        None
        if raw_mark in (None, "")
        else _numeric(raw_mark, "markPrice", positive=True)
    )
    return FundingRatePoint(
        symbol=symbol,
        funding_time=funding_time,
        funding_rate=rate,
        mark_price=mark,
        available_at=funding_time + PUBLICATION_LAG,
        content_hash=_content_hash(row),
    )


def _parse_open_interest_row(
    row: object, symbol: str, period: str
) -> OpenInterestPoint:
    if not isinstance(row, Mapping) or row.get("symbol") != symbol:
        raise ProviderError("binance_oi_record_symbol_or_schema_invalid")
    timestamp = _exchange_timestamp(row.get("timestamp"), "timestamp")
    interest = _numeric(row.get("sumOpenInterest"), "sumOpenInterest", positive=True)
    raw_value = row.get("sumOpenInterestValue")
    value = (
        None
        if raw_value in (None, "")
        else _numeric(raw_value, "sumOpenInterestValue", positive=True)
    )
    period_duration = timedelta(hours=4)
    return OpenInterestPoint(
        symbol=symbol,
        period=period,
        timestamp=timestamp,
        sum_open_interest=interest,
        sum_open_interest_value=value,
        available_at=timestamp + period_duration + PUBLICATION_LAG,
        content_hash=_content_hash(row),
    )


def _request_url(path: str, parameters: Mapping[str, object]) -> str:
    if path not in (FUNDING_RATE_PATH, OPEN_INTEREST_HISTORY_PATH):
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
    if parsed.scheme != "https" or parsed.hostname != "fapi.binance.com":
        raise ProviderError("binance_public_host_not_allowed")


def _transport_attempt(
    transport: JsonTransport,
    url: str,
) -> ConnectorResponse:
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
    response = _get_with_retry(transport, url, sleeper)
    return _response_payload(response)



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


def _deduplicate_open_interest(
    records: Sequence[OpenInterestPoint],
) -> tuple[OpenInterestPoint, ...]:
    by_time: dict[datetime, OpenInterestPoint] = {}
    for item in records:
        previous = by_time.get(item.timestamp)
        if previous is not None and previous.content_hash != item.content_hash:
            raise ProviderError("binance_oi_timestamp_conflict")
        by_time[item.timestamp] = item
    return tuple(by_time[key] for key in sorted(by_time))


class BinanceUsdMFuturesConnector:
    """Read-only public Funding/OI adapter with explicit history-window limits."""

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

    def _next_funding_cursor(
        self,
        page: Sequence[FundingRatePoint],
        cursor: int,
    ) -> int | None:
        if len(page) < FUNDING_PAGE_LIMIT:
            return None
        last_ms = max(_epoch_milliseconds(item.funding_time) for item in page)
        if last_ms < cursor or last_ms == 2**63 - 1:
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


    def _fetch_open_interest(
        self, symbol: str, start: datetime, end: datetime, period: str
    ) -> tuple[datetime, tuple[OpenInterestPoint, ...]]:
        effective_start = max(
            start,
            end - timedelta(days=OPEN_INTEREST_HISTORY_LIMIT_DAYS),
        )
        url = _request_url(
            OPEN_INTEREST_HISTORY_PATH,
            {
                "symbol": symbol,
                "period": period,
                "startTime": _epoch_milliseconds(effective_start),
                "endTime": _epoch_milliseconds(end),
                "limit": OPEN_INTEREST_PAGE_LIMIT,
            },
        )
        payload = _get_payload(self._transport, url, self._sleeper)
        if not isinstance(payload, list) or len(payload) > OPEN_INTEREST_PAGE_LIMIT:
            raise ProviderError("binance_oi_page_schema_invalid")
        page = [_parse_open_interest_row(row, symbol, period) for row in payload]
        if any(not effective_start <= record.timestamp <= end for record in page):
            raise ProviderError("binance_oi_timestamp_outside_window")
        records = _deduplicate_open_interest(page)
        return effective_start, records

    def fetch(
        self,
        symbol: str,
        *,
        start: datetime | str,
        end: datetime | str,
        oi_period: str = "4h",
    ) -> BinanceFuturesSnapshot:
        """Fetch five-year-capable funding and latest-month-only OI history."""
        start_utc = _utc(start, "start")
        end_utc = _utc(end, "end")
        _validate_request(symbol, start_utc, end_utc, oi_period)
        funding = self._fetch_funding(symbol, start_utc, end_utc)
        oi_start, open_interest = self._fetch_open_interest(
            symbol, start_utc, end_utc, oi_period
        )
        return BinanceFuturesSnapshot(
            symbol=symbol,
            requested_start=start_utc,
            requested_end=end_utc,
            oi_requested_start=oi_start,
            funding_rates=funding,
            open_interest=open_interest,
            retrieved_at=datetime.now(timezone.utc),
            oi_period=oi_period,
        )


class BinancePublicHttpsTransport:
    """Bounded HTTPS GET transport with a fixed host/path allow-list.

    HTTPSConnection uses Python's default TLS context, including certificate
    verification. The client sends no credentials and cannot follow redirects.
    """

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
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "fapi.binance.com"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in (None, 443)
            or parsed.fragment
            or parsed.path not in (FUNDING_RATE_PATH, OPEN_INTEREST_HISTORY_PATH)
        ):
            raise ProviderError("binance_public_host_or_path_not_allowed")
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
                    "User-Agent": "AliX-AI6-Research/1.0",
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
            raise TimeoutError(
                "binance_public_request_timeout_or_network_error"
            ) from None
        finally:
            connection.close()


def _covers_requested_window(
    first: datetime | None,
    last: datetime | None,
    start: datetime,
    end: datetime,
) -> bool:
    if first is None or last is None:
        return False
    tolerance = timedelta(days=1)
    return first <= start + tolerance and last >= end - tolerance


def _funding_has_large_gaps(records: Sequence[FundingRatePoint]) -> bool:
    return any(
        later.funding_time - earlier.funding_time > timedelta(days=1)
        for earlier, later in zip(records, records[1:])
    )


def _oi_coverage_metrics(
    snapshot: BinanceFuturesSnapshot,
) -> tuple[float, int, bool]:
    if not snapshot.open_interest:
        return 0.0, 0, False
    period = timedelta(hours=4)
    duration = snapshot.requested_end - snapshot.oi_requested_start
    expected = max(1, int(duration.total_seconds() // period.total_seconds()) + 1)
    actual = len(snapshot.open_interest)
    fraction = min(1.0, actual / expected)
    first = snapshot.open_interest[0].timestamp
    last = snapshot.open_interest[-1].timestamp
    complete_span = _covers_requested_window(
        first, last, snapshot.oi_requested_start, snapshot.requested_end
    )
    expected_interval = timedelta(hours=4)
    all_intervals_observed = all(
        later.timestamp - earlier.timestamp == expected_interval
        for earlier, later in zip(snapshot.open_interest, snapshot.open_interest[1:])
    )
    return (
        fraction,
        max(0, expected - actual),
        complete_span and actual == expected and all_intervals_observed,
    )


def _validate_snapshot_manifest_request(
    commit_sha: str,
    requested_days: int,
) -> None:
    if len(commit_sha) != 40 or any(
        character not in "0123456789abcdef" for character in commit_sha
    ):
        raise ProviderError("binance_commit_sha_invalid")
    if (
        isinstance(requested_days, bool)
        or not isinstance(requested_days, int)
        or requested_days < 1
    ):
        raise ProviderError("binance_requested_days_invalid")


def _funding_manifest_section(
    snapshot: BinanceFuturesSnapshot,
) -> dict[str, object]:
    records = snapshot.funding_rates
    first = records[0].funding_time if records else None
    last = records[-1].funding_time if records else None
    span_days = (
        (last - first).total_seconds() / 86400.0
        if first is not None and last is not None
        else 0.0
    )
    complete = _covers_requested_window(
        first, last, snapshot.requested_start, snapshot.requested_end
    ) and not _funding_has_large_gaps(records)
    return {
        "endpoint": FUNDING_RATE_PATH,
        "page_limit": FUNDING_PAGE_LIMIT,
        "row_count": len(records),
        "first_timestamp_utc": first.isoformat() if first is not None else None,
        "last_timestamp_utc": last.isoformat() if last is not None else None,
        "observed_span_days": round(span_days, 6),
        "availability_policy": "fundingTime_plus_5_minutes_conservative_proxy",
        "history_limit_days_documented": None,
        "full_requested_history_available": complete,
        "coverage_status": (
            "FULL_REQUESTED_HISTORY" if complete else "PARTIAL_FUNDING_HISTORY"
        ),
    }


def _open_interest_manifest_section(
    snapshot: BinanceFuturesSnapshot,
    requested_days: int,
) -> dict[str, object]:
    records = snapshot.open_interest
    first = records[0].timestamp if records else None
    last = records[-1].timestamp if records else None
    span_days = (
        (last - first).total_seconds() / 86400.0
        if first is not None and last is not None
        else 0.0
    )
    fraction, missing_count, full_span = _oi_coverage_metrics(snapshot)
    full_history = (
        full_span
        and span_days >= requested_days - 1.0
        and snapshot.oi_requested_start <= snapshot.requested_start + timedelta(days=1)
    )
    return {
        "endpoint": OPEN_INTEREST_HISTORY_PATH,
        "period": snapshot.oi_period,
        "page_limit": OPEN_INTEREST_PAGE_LIMIT,
        "documented_history_limit_days": OPEN_INTEREST_HISTORY_LIMIT_DAYS,
        "requested_start_utc": snapshot.oi_requested_start.isoformat(),
        "row_count": len(records),
        "first_timestamp_utc": first.isoformat() if first is not None else None,
        "last_timestamp_utc": last.isoformat() if last is not None else None,
        "observed_span_days": round(span_days, 6),
        "coverage_fraction": round(fraction, 6),
        "expected_period_count": len(records) + missing_count,
        "missing_period_count": missing_count,
        "availability_policy": "timestamp_plus_period_plus_5_minutes_conservative_proxy",
        "full_requested_history_available": full_history,
        "coverage_status": (
            "FULL_REQUESTED_HISTORY"
            if full_history
            else "INSUFFICIENT_HISTORICAL_WINDOW"
        ),
        "missing_periods_are_not_filled": True,
    }


def _snapshot_manifest_limitations() -> list[str]:
    return [
        "Binance Open Interest History exposes only the latest 30 days; it cannot supply a five-year OI series.",
        "The availability timestamps are conservative research proxies, not a source-certified historical publication-time feed.",
        "No missing OI observations are backfilled, interpolated, or synthesized.",
        "Internal research metadata does not grant redistribution or commercial-use rights.",
    ]


def snapshot_manifest(
    snapshot: BinanceFuturesSnapshot,
    *,
    commit_sha: str,
    requested_days: int,
) -> dict[str, object]:
    """Build a canonical manifest without implying complete OI history."""
    _validate_snapshot_manifest_request(commit_sha, requested_days)
    funding = _funding_manifest_section(snapshot)
    oi = _open_interest_manifest_section(snapshot, requested_days)
    full_oi_history = oi["full_requested_history_available"] is True
    return {
        "schema_version": "alix.binance.usdm-external-market.v1",
        "commit_sha": commit_sha,
        "symbol": snapshot.symbol,
        "provider": "Binance USD-M Futures public market-data API",
        "base_url": BINANCE_FUTURES_BASE_URL,
        "public_api_only": True,
        "credentials_used": False,
        "api_key_header_sent": False,
        "private_api_calls": 0,
        "exchange_order_calls": 0,
        "requested_days": requested_days,
        "requested_start_utc": snapshot.requested_start.isoformat(),
        "requested_end_utc": snapshot.requested_end.isoformat(),
        "retrieved_at": snapshot.retrieved_at.isoformat(),
        "funding": funding,
        "open_interest": oi,
        "data_quality": {
            "timestamps_unique_and_ordered": True,
            "numeric_values_finite": True,
            "missing_oi_periods_interpolated": False,
            "synthetic_rows_created": 0,
        },
        "limitations": _snapshot_manifest_limitations(),
        "research_status": (
            "READY_FOR_FULL_WINDOW_FEATURE_JOIN"
            if full_oi_history
            else "BLOCKED_INSUFFICIENT_OI_HISTORY"
        ),
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
        "real_order_attempts": 0,
    }
