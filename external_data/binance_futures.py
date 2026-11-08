"""Public Binance USD-M funding and open-interest history with bounded provenance.

Only unauthenticated public market-data GET endpoints are implemented. The
Binance Open Interest History endpoint is capped to the latest 30 days; the
connector records that limitation and never fabricates older observations.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import time
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from external_data.contracts import ConnectorResponse, ProviderError, RateLimitStatus, parse_timestamp

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
    mark = None if raw_mark in (None, "") else _numeric(raw_mark, "markPrice", positive=True)
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
    value = None if raw_value in (None, "") else _numeric(
        raw_value, "sumOpenInterestValue", positive=True
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


def _get_payload(
    transport: JsonTransport,
    url: str,
    sleeper: Callable[[float], None],
) -> object:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "fapi.binance.com":
        raise ProviderError("binance_public_host_not_allowed")
    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:
            response = transport.get(url)
        except TimeoutError:
            if attempt + 1 == MAX_RETRY_ATTEMPTS:
                raise ProviderError("binance_public_timeout") from None
            sleeper(float(2**attempt))
            continue
        except Exception:
            raise ProviderError("binance_public_transport_error") from None
        if not isinstance(response, ConnectorResponse):
            raise ProviderError("binance_transport_response_invalid")
        if response.status_code == 418:
            raise ProviderError("binance_public_ip_banned")
        if _retryable_status(response.status_code):
            if attempt + 1 == MAX_RETRY_ATTEMPTS:
                raise ProviderError(f"binance_public_http_status:{response.status_code}")
            sleeper(float(2**attempt))
            continue
        return _response_payload(response)
    raise ProviderError("binance_public_retry_exhausted")


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

    def _fetch_funding(
        self, symbol: str, start: datetime, end: datetime
    ) -> tuple[FundingRatePoint, ...]:
        cursor = _epoch_milliseconds(start)
        end_ms = _epoch_milliseconds(end)
        collected: list[FundingRatePoint] = []
        for _ in range(MAX_FUNDING_PAGES):
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
            page = [_parse_funding_row(row, symbol) for row in payload]
            if not page:
                break
            for record in page:
                if not start <= record.funding_time <= end:
                    raise ProviderError("binance_funding_timestamp_outside_window")
            collected.extend(page)
            last_ms = max(_epoch_milliseconds(item.funding_time) for item in page)
            if len(page) < FUNDING_PAGE_LIMIT:
                break
            if last_ms < cursor or last_ms == 2**63 - 1:
                raise ProviderError("binance_funding_pagination_stalled")
            cursor = last_ms + 1
            self._sleeper(self._inter_page_delay)
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


class UrllibPublicJsonTransport:
    """Bounded urllib transport. It never reads API keys or sends auth headers."""

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

    def get(self, url: str) -> ConnectorResponse:
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname != "fapi.binance.com":
            raise ProviderError("binance_public_host_not_allowed")
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "AliX-AI6-Research/1.0",
            },
            method="GET",
        )
        received_at = datetime.now(timezone.utc)
        try:
            with urlopen(request, timeout=self._timeout) as response:
                body = self._response_body(response.read(MAX_RESPONSE_BYTES + 1))
                return ConnectorResponse(
                    status_code=int(response.status),
                    body=body,
                    received_at=received_at,
                )
        except HTTPError as error:
            return ConnectorResponse(
                status_code=int(error.code),
                body=self._response_body(error.read(MAX_RESPONSE_BYTES + 1)),
                received_at=received_at,
                rate_limit_status=(
                    RateLimitStatus.LIMITED
                    if error.code == 429
                    else RateLimitStatus.BLOCKED
                    if error.code in (418, 403)
                    else RateLimitStatus.OK
                ),
            )
        except (URLError, TimeoutError, OSError):
            raise TimeoutError("binance_public_request_timeout_or_network_error") from None


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
    return fraction, max(0, expected - actual), complete_span and fraction >= 0.98


def snapshot_manifest(
    snapshot: BinanceFuturesSnapshot,
    *,
    commit_sha: str,
    requested_days: int,
) -> dict[str, object]:
    """Build a canonical manifest without implying complete OI history."""
    if len(commit_sha) != 40 or any(character not in "0123456789abcdef" for character in commit_sha):
        raise ProviderError("binance_commit_sha_invalid")
    if isinstance(requested_days, bool) or not isinstance(requested_days, int) or requested_days < 1:
        raise ProviderError("binance_requested_days_invalid")
    funding = snapshot.funding_rates
    oi = snapshot.open_interest
    funding_days = (
        (funding[-1].funding_time - funding[0].funding_time).total_seconds() / 86400.0
        if len(funding) > 1
        else 0.0
    )
    oi_days = (
        (oi[-1].timestamp - oi[0].timestamp).total_seconds() / 86400.0
        if len(oi) > 1
        else 0.0
    )
    funding_start_time = funding[0].funding_time if funding else None
    funding_end_time = funding[-1].funding_time if funding else None
    oi_start_time = oi[0].timestamp if oi else None
    oi_end_time = oi[-1].timestamp if oi else None
    funding_full_history = (
        _covers_requested_window(
            funding_start_time,
            funding_end_time,
            snapshot.requested_start,
            snapshot.requested_end,
        )
        and not _funding_has_large_gaps(funding)
    )
    oi_fraction, oi_missing_periods, oi_full_span = _oi_coverage_metrics(snapshot)
    oi_full_history = (
        oi_full_span
        and oi_days >= requested_days - 1.0
        and snapshot.oi_requested_start <= snapshot.requested_start + timedelta(days=1)
    )
    funding_start = funding_start_time.isoformat() if funding_start_time else None
    funding_end = funding_end_time.isoformat() if funding_end_time else None
    oi_start = oi_start_time.isoformat() if oi_start_time else None
    oi_end = oi_end_time.isoformat() if oi_end_time else None
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
        "funding": {
            "endpoint": FUNDING_RATE_PATH,
            "page_limit": FUNDING_PAGE_LIMIT,
            "row_count": len(funding),
            "first_timestamp_utc": funding_start,
            "last_timestamp_utc": funding_end,
            "observed_span_days": round(funding_days, 6),
            "availability_policy": "fundingTime_plus_5_minutes_conservative_proxy",
            "history_limit_days_documented": None,
            "full_requested_history_available": funding_full_history,
            "coverage_status": (
                "FULL_REQUESTED_HISTORY" if funding_full_history else "PARTIAL_FUNDING_HISTORY"
            ),
        },
        "open_interest": {
            "endpoint": OPEN_INTEREST_HISTORY_PATH,
            "period": snapshot.oi_period,
            "page_limit": OPEN_INTEREST_PAGE_LIMIT,
            "documented_history_limit_days": OPEN_INTEREST_HISTORY_LIMIT_DAYS,
            "requested_start_utc": snapshot.oi_requested_start.isoformat(),
            "row_count": len(oi),
            "first_timestamp_utc": oi_start,
            "last_timestamp_utc": oi_end,
            "observed_span_days": round(oi_days, 6),
            "coverage_fraction": round(oi_fraction, 6),
            "expected_period_count": len(oi) + oi_missing_periods,
            "missing_period_count": oi_missing_periods,
            "availability_policy": "timestamp_plus_period_plus_5_minutes_conservative_proxy",
            "full_requested_history_available": oi_full_history,
            "coverage_status": (
                "FULL_REQUESTED_HISTORY" if oi_full_history else "INSUFFICIENT_HISTORICAL_WINDOW"
            ),
            "missing_periods_are_not_filled": True,
        },
        "data_quality": {
            "timestamps_unique_and_ordered": True,
            "numeric_values_finite": True,
            "missing_oi_periods_interpolated": False,
            "synthetic_rows_created": 0,
        },
        "limitations": [
            "Binance Open Interest History exposes only the latest 30 days; it cannot supply a five-year OI series.",
            "The availability timestamps are conservative research proxies, not a source-certified historical publication-time feed.",
            "No missing OI observations are backfilled, interpolated, or synthesized.",
            "Internal research metadata does not grant redistribution or commercial-use rights.",
        ],
        "research_status": (
            "READY_FOR_FULL_WINDOW_FEATURE_JOIN"
            if oi_full_history
            else "BLOCKED_INSUFFICIENT_OI_HISTORY"
        ),
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
        "private_api_calls": 0,
        "real_order_attempts": 0,
    }
