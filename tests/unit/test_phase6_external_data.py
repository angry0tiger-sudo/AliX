"""Offline-only contracts for providers, stores and temporal joins."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pytest

from external_data.connectors import MarketAnalyticsConnector, RssNewsConnector
from external_data.contracts import (
    ConnectorResponse,
    MarketObservation,
    NewsEvent,
    NewsFeature,
    ProviderError,
    ProviderPolicyError,
    ProviderRegistration,
    digest_content,
    parse_timestamp,
    validate_market_observation,
)
from external_data.point_in_time import join_news_features_as_of
from external_data.stores import (
    InMemoryExternalFeatureStore,
    InMemoryNewsStore,
    InMemoryNlpFeatureStore,
    StoreWriteResult,
)


def ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def registration(
    provider_id: str = "fixture-rss",
    data_kind: str = "news",
    *,
    active: bool = True,
    license_status: str = "approved",
    endpoint_url: str = "https://rss.fixture.invalid/feed.xml",
) -> ProviderRegistration:
    return ProviderRegistration(
        provider_id=provider_id,
        data_kind=data_kind,
        endpoint_url=endpoint_url,
        license_status=license_status,
        rate_limit_policy="synthetic fixture only",
        retention_policy="no real content",
        timestamp_semantics="synthetic publication and receive times",
        owner="phase-6-tests",
        active=active,
        is_test_fixture=True,
    )


class FixtureTransport:
    def __init__(self, result: ConnectorResponse | None = None) -> None:
        self.result = result
        self.calls: list[str] = []
        self.failure: Exception | None = None

    def get(self, url: str) -> ConnectorResponse:
        self.calls.append(url)
        if self.failure is not None:
            raise self.failure
        if self.result is None:
            raise RuntimeError("fixture missing")
        return self.result


def response(
    body: str,
    *,
    status: int = 200,
    rate_limit_status: str = "ok",
    received_at: str = "2026-10-22T10:00:00Z",
) -> ConnectorResponse:
    return ConnectorResponse(status, body, ts(received_at), rate_limit_status)


def rss_item(
    guid: str = "story-1",
    title: str = "BTC market update",
    published: str = "Thu, 22 Oct 2026 09:00:00 GMT",
    body: str = "A neutral synthetic test item.",
) -> str:
    return (
        "<item><guid>" + guid + "</guid><title>" + title + "</title>"
        "<pubDate>"
        + published
        + "</pubDate><description>"
        + body
        + "</description></item>"
    )


def rss_body(item: str | None = None) -> str:
    return "<rss><channel>" + (item or rss_item()) + "</channel></rss>"


def news_event(
    *,
    event_id: str = "event-1",
    revision: str = "content:abc",
    available_at: str | None = "2026-10-22T09:01:00Z",
    text: str = "synthetic headline",
    license_status: str = "approved",
    assets: tuple[str, ...] = ("BTC/USDT",),
) -> NewsEvent:
    return NewsEvent(
        event_id=event_id,
        provider_id="fixture-news",
        source_id="https://news.fixture.invalid/story",
        published_at=ts("2026-10-22T09:00:00Z"),
        ingested_at=ts("2026-10-22T09:01:00Z"),
        available_at=ts(available_at) if available_at else None,
        revision=revision,
        asset_mapping=assets,
        license_status=license_status,
        content_hash=digest_content(text),
        text=text,
    )


def feature(
    *,
    event_id: str = "event-1",
    revision: str = "r1",
    available_at: str = "2026-10-22T09:03:00Z",
    text: str = "synthetic feature",
    assets: tuple[str, ...] = ("BTC/USDT",),
    license_status: str = "approved",
) -> NewsFeature:
    instant = ts(available_at)
    return NewsFeature(
        event_id=event_id,
        provider_id="fixture-news",
        source_id="https://news.fixture.invalid/story",
        published_at=ts("2026-10-22T09:00:00Z"),
        ingested_at=ts("2026-10-22T09:01:00Z"),
        available_at=instant,
        processed_at=instant,
        revision=revision,
        asset_mapping=assets,
        license_status=license_status,
        content_hash=digest_content(text),
        sentiment_label="neutral",
        sentiment_score=0.0,
        confidence=0.5,
        event_type="unknown",
        model_id="fixture/sentiment-test",
        model_revision="a" * 40,
    )


def market_observation(**changes: object) -> MarketObservation:
    values: dict[str, object] = {
        "event_id": "metric-1",
        "provider_id": "fixture-market",
        "source_id": "https://market.fixture.invalid/metrics",
        "market": "BTC/USDT",
        "event_timestamp": ts("2026-10-22T09:45:00Z"),
        "ingested_at": ts("2026-10-22T09:46:00Z"),
        "available_at": ts("2026-10-22T09:46:00Z"),
        "revision": "v1",
        "value": 123.5,
        "unit": "index",
        "quote_currency": "USDT",
        "license_status": "approved",
        "rate_limit_status": "ok",
        "content_hash": digest_content("metric"),
    }
    values.update(changes)
    return MarketObservation(**values)  # type: ignore[arg-type]


def test_registry_fails_before_transport_for_inactive_or_unapproved_source() -> None:
    transport = FixtureTransport(response(rss_body()))
    with pytest.raises(ProviderPolicyError, match="provider_inactive"):
        RssNewsConnector(registration(active=False), transport).ingest()
    with pytest.raises(ProviderPolicyError, match="license_not_approved"):
        RssNewsConnector(registration(license_status="conditional"), transport).ingest()
    with pytest.raises(ProviderPolicyError, match="credentials_forbidden"):
        registration(endpoint_url="https://user:pass@rss.fixture.invalid/feed")
    with pytest.raises(ProviderPolicyError, match="credentials"):
        registration(endpoint_url="https://rss.fixture.invalid/feed?api_key=fake")
    with pytest.raises(ProviderPolicyError, match="invalid_domain"):
        ProviderRegistration(
            "fixture-live",
            "news",
            "https://example.com/feed",
            "approved",
            "test",
            "none",
            "observed",
            "tests",
            True,
            True,
        )
    with pytest.raises(ProviderPolicyError, match="tradingview"):
        ProviderRegistration(
            "tradingview",
            "market_analytics",
            "https://tradingview.fixture.invalid/api",
            "approved",
            "test",
            "none",
            "observed",
            "tests",
            True,
            True,
        )
    assert transport.calls == []


def test_rss_ingest_parses_timestamps_mapping_and_normalized_content() -> None:
    body = rss_body(
        rss_item(body="<![CDATA[<p>Neutral &amp; synthetic text</p>]]>")
        + rss_item(
            "story-2",
            "Unknown story",
            "Thu, 22 Oct 2026 09:30:00 GMT",
            "No mapped asset",
        )
    )
    transport = FixtureTransport(response(body))
    records = RssNewsConnector(
        registration(),
        transport,
        asset_resolver=lambda text: ("btc/usdt",) if "BTC" in text else (),
    ).ingest()
    assert len(records) == 2
    assert records[0].published_at == ts("2026-10-22T09:00:00Z")
    assert records[0].ingested_at == records[0].available_at
    assert records[0].asset_mapping == ("BTC/USDT",)
    assert records[1].asset_mapping == ()
    assert len(records[0].content_hash) == 64
    assert transport.calls == ["https://rss.fixture.invalid/feed.xml"]


def test_rss_ingest_fails_closed_for_malformed_or_unsafe_responses() -> None:
    transport = FixtureTransport(response("<!DOCTYPE rss><rss/>"))
    connector = RssNewsConnector(registration(), transport)
    with pytest.raises(ProviderError, match="unsafe_xml"):
        connector.ingest()
    transport.result = response("<rss><channel><item></channel></rss>")
    with pytest.raises(ProviderError, match="xml_invalid"):
        connector.ingest()
    transport.result = response(rss_body(rss_item(published="not a date")))
    with pytest.raises(ProviderError, match="published_at_invalid"):
        connector.ingest()
    transport.result = response("<rss><channel/></rss>")
    with pytest.raises(ProviderError, match="entries_missing"):
        connector.ingest()
    transport.result = response(rss_body(rss_item(published="")))
    with pytest.raises(ProviderError, match="published_at_missing"):
        connector.ingest()
    transport.result = response(rss_body(), status=429)
    with pytest.raises(ProviderError, match="http_status_429"):
        connector.ingest()
    transport.result = response(rss_body(), rate_limit_status="limited")
    with pytest.raises(ProviderError, match="rate_limit_limited"):
        connector.ingest()
    transport.result = response(rss_body() + "x" * 1_000_001)
    with pytest.raises(ProviderError, match="response_too_large"):
        connector.ingest()
    transport.result = response(rss_body())
    transport.failure = TimeoutError("secret detail")
    with pytest.raises(ProviderError, match="provider_timeout") as caught:
        connector.ingest()
    assert "secret detail" not in str(caught.value)
    transport.failure = RuntimeError("secret detail")
    with pytest.raises(ProviderError, match="transport_error") as caught:
        connector.ingest()
    assert "secret detail" not in str(caught.value)


def test_rss_ingest_rejects_duplicate_event_revisions_and_future_publication() -> None:
    transport = FixtureTransport(response(rss_body(rss_item() + rss_item())))
    with pytest.raises(ProviderError, match="duplicate_event_revision"):
        RssNewsConnector(registration(), transport).ingest()
    transport.result = response(
        rss_body(rss_item(published="Fri, 23 Oct 2026 09:00:00 GMT"))
    )
    with pytest.raises(ProviderError, match="published_after_ingest"):
        RssNewsConnector(registration(), transport).ingest()


def test_market_connector_uses_shared_data_quality_contract() -> None:
    body = (
        '{"data":[{"id":"metric-1","market":"btc/usdt",'
        '"timestamp":"2026-10-22T09:45:00Z","revision":"v1","value":123.5}]}'
    )
    transport = FixtureTransport(response(body))
    connector = MarketAnalyticsConnector(
        registration(
            "fixture-market",
            "market_analytics",
            endpoint_url="https://market.fixture.invalid/metrics",
        ),
        transport,
        unit="index",
        quote_currency="usdt",
    )
    rows = connector.ingest()
    assert len(rows) == 1
    assert rows[0].market == "BTC/USDT"
    assert rows[0].quote_currency == "USDT"
    validate_market_observation(rows[0])
    store = InMemoryExternalFeatureStore()
    assert store.add(rows[0]) is StoreWriteResult.STORED
    assert store.add(rows[0]) is StoreWriteResult.DUPLICATE
    assert len(store.records()) == 1


def test_market_connector_rejects_bad_payload_schema_values_and_limits() -> None:
    transport = FixtureTransport(response("not json"))
    reg = registration(
        "fixture-market",
        "market_analytics",
        endpoint_url="https://market.fixture.invalid/metrics",
    )
    connector = MarketAnalyticsConnector(
        reg, transport, unit="index", quote_currency="USDT"
    )
    with pytest.raises(ProviderError, match="json_invalid"):
        connector.ingest()
    transport.result = response('{"data":[{"id":"x"}]}')
    with pytest.raises(ProviderError, match="required_field_missing"):
        connector.ingest()
    transport.result = response(
        '{"data":[{"id":"x","market":"BTC/USDT","timestamp":"2026-10-22T09:45:00Z",'
        '"revision":"v1","value":NaN}]}'
    )
    with pytest.raises(ProviderError, match="non_finite"):
        connector.ingest()
    transport.result = response(
        '{"data":[{"id":"x","market":"BTC/USDT","timestamp":"2026-10-22T09:45:00Z",'
        '"revision":"v1","value":1}]}',
        rate_limit_status="blocked",
    )
    with pytest.raises(ProviderError, match="rate_limit_blocked"):
        connector.ingest()
    with pytest.raises(ProviderError, match="unit_and_quote"):
        MarketAnalyticsConnector(reg, transport, unit="", quote_currency="USDT")
    with pytest.raises(ProviderPolicyError, match="kind_mismatch"):
        MarketAnalyticsConnector(
            registration(), transport, unit="index", quote_currency="USDT"
        )


def test_market_record_rejects_invalid_provenance_and_values() -> None:
    with pytest.raises(ProviderPolicyError, match="source_not_approved"):
        validate_market_observation(market_observation(license_status="conditional"))
    with pytest.raises(ProviderPolicyError, match="rate_limit_policy"):
        validate_market_observation(market_observation(rate_limit_status="limited"))
    with pytest.raises(ProviderError, match="ingest_after_available"):
        validate_market_observation(
            market_observation(available_at=ts("2026-10-22T09:45:00Z"))
        )
    with pytest.raises(ProviderError, match="non_finite"):
        validate_market_observation(market_observation(value=float("inf")))
    with pytest.raises(ProviderError, match="hash_invalid"):
        validate_market_observation(market_observation(content_hash="not-a-hash"))


def test_news_store_deduplicates_normalized_content_and_rejects_revision_conflicts() -> (
    None
):
    store = InMemoryNewsStore()
    first = news_event(event_id="one", text="BTC   market update")
    alias = news_event(event_id="two", text="btc market update")
    assert store.add(first) is StoreWriteResult.STORED
    assert store.add(first) is StoreWriteResult.DUPLICATE
    assert store.add(alias) is StoreWriteResult.DUPLICATE
    assert len(store.records()) == 1
    conflicting = replace(
        first, content_hash=digest_content("different"), text="different"
    )
    with pytest.raises(ProviderError, match="revision_conflict"):
        store.add(conflicting)
    with pytest.raises(ProviderPolicyError, match="source_not_approved"):
        store.add(news_event(license_status="conditional"))
    with pytest.raises(ProviderError, match="missing_timestamp"):
        store.add(news_event(available_at=None))


def test_numeric_and_nlp_stores_are_separate_and_detect_conflicts() -> None:
    numeric = InMemoryExternalFeatureStore()
    metric = market_observation()
    assert numeric.add(metric) is StoreWriteResult.STORED
    assert numeric.add(metric) is StoreWriteResult.DUPLICATE
    with pytest.raises(ProviderError, match="revision_conflict"):
        numeric.add(replace(metric, value=999.0, content_hash=digest_content("new")))

    derived = InMemoryNlpFeatureStore()
    sentiment = feature()
    assert derived.add(sentiment) is StoreWriteResult.STORED
    assert derived.add(sentiment) is StoreWriteResult.DUPLICATE
    with pytest.raises(ProviderError, match="revision_conflict"):
        derived.add(replace(sentiment, sentiment_label="positive", sentiment_score=0.8))
    with pytest.raises(ProviderPolicyError, match="source_not_approved"):
        derived.add(feature(license_status="conditional"))


def test_point_in_time_join_selects_only_revision_available_at_decision() -> None:
    old = feature(
        event_id="same", revision="r1", available_at="2026-10-22T09:03:00Z", text="old"
    )
    new = feature(
        event_id="same", revision="r2", available_at="2026-10-22T09:08:00Z", text="new"
    )
    late = feature(event_id="late", available_at="2026-10-22T10:30:00Z")
    stale = feature(event_id="stale", available_at="2026-10-22T08:00:00Z")
    unmapped = feature(event_id="unmapped", assets=())
    records = [old, new, late, stale, unmapped]
    early = join_news_features_as_of(
        records,
        asset="btc/usdt",
        decision_timestamp="2026-10-22T09:05:00Z",
        window_start="2026-10-22T08:30:00Z",
    )
    assert [item.event_id for item in early.records] == ["same"]
    assert early.records[0].revision == "r1"
    later = join_news_features_as_of(
        records,
        asset="BTC/USDT",
        decision_timestamp="2026-10-22T09:10:00Z",
        window_start="2026-10-22T08:30:00Z",
    )
    assert [item.event_id for item in later.records] == ["same"]
    assert later.records[0].revision == "r2"
    assert early.fingerprint != later.fingerprint
    assert (
        join_news_features_as_of(
            records,
            asset="ETH/USDT",
            decision_timestamp="2026-10-22T09:05:00Z",
            window_start="2026-10-22T08:30:00Z",
        ).records
        == ()
    )
    with pytest.raises(ProviderError, match="window_start_after_decision"):
        join_news_features_as_of(
            records,
            asset="BTC/USDT",
            decision_timestamp="2026-10-22T09:05:00Z",
            window_start="2026-10-22T09:06:00Z",
        )
    with pytest.raises(ProviderError, match="asset_required"):
        join_news_features_as_of(
            [],
            asset=" ",
            decision_timestamp="2026-10-22T09:05:00Z",
            window_start="2026-10-22T08:30:00Z",
        )


def test_point_in_time_join_is_invariant_to_future_additions_and_deduplicates() -> None:
    known = feature(event_id="known", available_at="2026-10-22T09:03:00Z")
    alias = replace(feature(event_id="zz-alias"), content_hash=known.content_hash)
    future = feature(event_id="future", available_at="2026-10-22T09:06:00Z")
    kwargs = {
        "asset": "BTC/USDT",
        "decision_timestamp": "2026-10-22T09:05:00Z",
        "window_start": "2026-10-22T08:30:00Z",
    }
    baseline = join_news_features_as_of([known], **kwargs)
    augmented = join_news_features_as_of([known, future, alias], **kwargs)
    assert baseline.records == augmented.records
    assert baseline.fingerprint == augmented.fingerprint
    assert len(augmented.records) == 1


def test_point_in_time_join_fails_closed_on_bad_timestamp_rights_and_conflicts() -> (
    None
):
    with pytest.raises(ProviderError, match="missing_timestamp"):
        join_news_features_as_of(
            [replace(feature(), available_at=None)],  # type: ignore[arg-type]
            asset="BTC/USDT",
            decision_timestamp="2026-10-22T09:05:00Z",
            window_start="2026-10-22T08:30:00Z",
        )
    with pytest.raises(ProviderError, match="timezone_required"):
        join_news_features_as_of(
            [replace(feature(), available_at=datetime(2026, 10, 22, 9, 3))],
            asset="BTC/USDT",
            decision_timestamp="2026-10-22T09:05:00Z",
            window_start="2026-10-22T08:30:00Z",
        )
    with pytest.raises(ProviderPolicyError, match="source_not_approved"):
        join_news_features_as_of(
            [feature(license_status="conditional")],
            asset="BTC/USDT",
            decision_timestamp="2026-10-22T09:05:00Z",
            window_start="2026-10-22T08:30:00Z",
        )
    first = feature(
        event_id="conflict", available_at="2026-10-22T09:03:00Z", text="one"
    )
    second = replace(first, content_hash=digest_content("two"), revision="r2")
    with pytest.raises(ProviderError, match="revision_timestamp_conflict"):
        join_news_features_as_of(
            [first, second],
            asset="BTC/USDT",
            decision_timestamp="2026-10-22T09:05:00Z",
            window_start="2026-10-22T08:30:00Z",
        )


def test_timestamp_parser_requires_timezone_and_rejects_malformed_values() -> None:
    assert parse_timestamp("2026-10-22T10:00:00Z", "test") == ts("2026-10-22T10:00:00Z")
    with pytest.raises(ProviderError, match="timezone_required"):
        parse_timestamp("2026-10-22T10:00:00", "test")
    with pytest.raises(ProviderError, match="invalid_timestamp"):
        parse_timestamp("not-a-timestamp", "test")
