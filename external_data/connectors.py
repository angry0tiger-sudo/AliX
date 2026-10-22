"""Injected-transport provider adapters. No network client is configured here."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from email.utils import parsedate_to_datetime
import hashlib
import json
import math
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit
import xml.etree.ElementTree as ET

from external_data.contracts import (
    ConnectorResponse,
    MarketObservation,
    NewsEvent,
    ProviderError,
    ProviderPolicyError,
    ProviderRegistration,
    RateLimitStatus,
    digest_content,
    normalize_news_text,
    parse_timestamp,
    validate_market_observation,
    validate_news_event,
)

MAX_RESPONSE_BYTES = 1_000_000


class Transport(Protocol):
    """Injected transport interface; callers explicitly supply any I/O adapter."""

    def get(self, url: str) -> ConnectorResponse:
        """Return a provider response."""


def _request(
    registration: ProviderRegistration, transport: Transport, expected_kind: str
) -> ConnectorResponse:
    if registration.data_kind != expected_kind:
        raise ProviderPolicyError("provider_kind_mismatch")
    registration.assert_ingestion_allowed()
    try:
        response = transport.get(registration.endpoint_url)
    except TimeoutError:
        raise ProviderError("provider_timeout") from None
    except Exception:
        raise ProviderError("provider_transport_error") from None
    if not isinstance(response, ConnectorResponse):
        raise ProviderError("provider_response_contract_invalid")
    if not 200 <= response.status_code < 300:
        raise ProviderError(f"provider_http_status_{response.status_code}")
    if response.rate_limit_status not in RateLimitStatus.VALUES:
        raise ProviderError("provider_rate_limit_status_invalid")
    if response.rate_limit_status != RateLimitStatus.OK:
        raise ProviderError(f"provider_rate_limit_{response.rate_limit_status}")
    if len(response.body.encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise ProviderError("provider_response_too_large")
    parse_timestamp(response.received_at, "received_at")
    return response


def _local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1].casefold()


def _child_text(element: ET.Element, *names: str) -> str:
    accepted = set(names)
    for child in list(element):
        if _local_name(child.tag) in accepted:
            return normalize_news_text(" ".join(child.itertext()))
    return ""


def _source_timestamp(value: str) -> datetime:
    if not value.strip():
        raise ProviderError("rss_published_at_missing")
    try:
        return parse_timestamp(value, "published_at")
    except ProviderError:
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            raise ProviderError("rss_published_at_invalid") from None
        try:
            return parse_timestamp(parsed, "published_at")
        except ProviderError:
            raise ProviderError("rss_published_at_invalid") from None


def _safe_source_id(value: str, fallback: str) -> str:
    if not value.strip():
        return fallback
    parts = urlsplit(value.strip())
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        return fallback
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


class RssNewsConnector:
    """RSS/Atom fixture reader that requires an explicitly injected transport."""

    def __init__(
        self,
        registration: ProviderRegistration,
        transport: Transport,
        asset_resolver: Callable[[str], tuple[str, ...]] | None = None,
    ) -> None:
        if registration.data_kind != "news":
            raise ProviderPolicyError("provider_kind_mismatch")
        self.registration = registration
        self.transport = transport
        self.asset_resolver = asset_resolver

    def _parse_entry(self, entry: ET.Element, received_at: datetime) -> NewsEvent:
        title = _child_text(entry, "title")
        body = _child_text(entry, "description", "summary", "content", "encoded")
        text = normalize_news_text(f"{title}\n{body}")
        if not text:
            raise ProviderError("rss_text_missing")
        published_at = _source_timestamp(
            _child_text(entry, "pubdate", "published", "updated", "date")
        )
        link = _child_text(entry, "link")
        guid = _child_text(entry, "guid", "id")
        digest = digest_content(text)
        event_id = guid or link or digest
        source_id = _safe_source_id(link, self.registration.endpoint_url)
        try:
            mapped = self.asset_resolver(text) if self.asset_resolver else ()
        except Exception:
            raise ProviderError("news_asset_mapping_failed") from None
        assets = tuple(sorted({item.strip().upper() for item in mapped if item.strip()}))
        result = NewsEvent(
            event_id=event_id,
            provider_id=self.registration.provider_id,
            source_id=source_id,
            published_at=published_at,
            ingested_at=received_at,
            available_at=received_at,
            revision=f"content:{digest[:16]}",
            asset_mapping=assets,
            license_status="approved",
            content_hash=digest,
            text=text,
            title=title,
        )
        validate_news_event(result)
        return result

    def ingest(self) -> tuple[NewsEvent, ...]:
        response = _request(self.registration, self.transport, "news")
        lowered = response.body.casefold()
        if "<!doctype" in lowered or "<!entity" in lowered:
            raise ProviderError("rss_unsafe_xml_declaration")
        try:
            root = ET.fromstring(response.body)
        except ET.ParseError:
            raise ProviderError("rss_xml_invalid") from None
        entries = [
            element
            for element in root.iter()
            if _local_name(element.tag) in {"item", "entry"}
        ]
        if not entries:
            raise ProviderError("rss_entries_missing")
        received = parse_timestamp(response.received_at, "received_at")
        records = tuple(self._parse_entry(entry, received) for entry in entries)
        identifiers = {(item.event_id, item.revision) for item in records}
        if len(identifiers) != len(records):
            raise ProviderError("rss_duplicate_event_revision")
        return records


class MarketAnalyticsConnector:
    """Normalized JSON metric adapter with injected transport and no credentials."""

    def __init__(
        self,
        registration: ProviderRegistration,
        transport: Transport,
        *,
        unit: str,
        quote_currency: str,
    ) -> None:
        if registration.data_kind != "market_analytics":
            raise ProviderPolicyError("provider_kind_mismatch")
        if not unit.strip() or not quote_currency.strip():
            raise ProviderError("market_unit_and_quote_required")
        self.registration = registration
        self.transport = transport
        self.unit = unit
        self.quote_currency = quote_currency.upper()

    def _parse_row(self, row: Mapping[str, object], received: datetime) -> MarketObservation:
        event_id, market, revision = row.get("id"), row.get("market"), row.get("revision")
        if not all(isinstance(item, str) and item.strip() for item in (event_id, market, revision)):
            raise ProviderError("market_record_required_field_missing")
        value = row.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProviderError("market_value_non_finite_or_invalid")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ProviderError("market_value_non_finite_or_invalid")
        timestamp = row.get("timestamp")
        if not isinstance(timestamp, (str, datetime)):
            raise ProviderError("market_event_timestamp_missing")
        event_time = parse_timestamp(timestamp, "event_timestamp")
        raw = json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        observation = MarketObservation(
            event_id=str(event_id),
            provider_id=self.registration.provider_id,
            source_id=self.registration.endpoint_url,
            market=str(market).strip().upper(),
            event_timestamp=event_time,
            ingested_at=received,
            available_at=received,
            revision=str(revision),
            value=numeric,
            unit=self.unit,
            quote_currency=self.quote_currency,
            license_status="approved",
            rate_limit_status=RateLimitStatus.OK,
            content_hash=digest,
        )
        validate_market_observation(observation)
        return observation

    def ingest(self) -> tuple[MarketObservation, ...]:
        response = _request(self.registration, self.transport, "market_analytics")
        try:
            payload: object = json.loads(response.body)
        except json.JSONDecodeError:
            raise ProviderError("market_json_invalid") from None
        rows: object = (
            payload.get("data", payload.get("records"))
            if isinstance(payload, Mapping)
            else payload
        )
        if not isinstance(rows, list) or not rows:
            raise ProviderError("market_records_missing")
        received = parse_timestamp(response.received_at, "received_at")
        observations: list[MarketObservation] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ProviderError("market_record_invalid")
            observations.append(self._parse_row(row, received))
        return tuple(observations)
