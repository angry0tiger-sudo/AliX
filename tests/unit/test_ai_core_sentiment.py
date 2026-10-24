from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from external_data.contracts import (
    NewsEvent,
    NewsFeature,
    ProviderError,
    ProviderPolicyError,
    ProviderRegistration,
)
from services.ai_core.features import SENTIMENT_COLUMNS
from services.ai_core.sentiment import (
    aggregate_point_in_time_sentiment,
    parse_persian_rss,
)


def candles_4h(rows: int = 120) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    close = 1000.0 * np.exp(np.cumsum(0.0005 + 0.006 * np.sin(index / 7.0)))
    opened = np.r_[close[0] * 0.999, close[:-1]]
    high = np.maximum(opened, close) * 1.002
    low = np.minimum(opened, close) * 0.998
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01T00:30:00Z", periods=rows, freq="4h"),
            "open": opened,
            "high": high,
            "low": low,
            "close": close,
            "volume": 100.0 + (index % 17) * 7.0,
        }
    )


def provider(**changes: object) -> ProviderRegistration:
    values: dict[str, object] = {
        "provider_id": "persian-test-rss",
        "data_kind": "news",
        "endpoint_url": "https://rss.example.invalid/feed.xml",
        "license_status": "approved",
        "rate_limit_policy": "fixture only; one document per test",
        "retention_policy": "fixture text is not retained beyond the unit test",
        "timestamp_semantics": "published_at is source publication time; ingestion sets availability",
        "owner": "ai-core-test",
        "active": True,
        "is_test_fixture": True,
        "algorithmic_non_display_approved": False,
    }
    values.update(changes)
    return ProviderRegistration(**values)  # type: ignore[arg-type]


def valid_rss() -> bytes:
    return """<?xml version="1.0" encoding="UTF-8"?>
    <rss><channel>
      <item><title>قیمت بیت‌کوین</title><description>افزایش بازار</description>
        <pubDate>Mon, 20 Oct 2025 10:00:00 +0000</pubDate></item>
      <item><title> قیمت بیت‌کوین </title><description> افزایش بازار </description>
        <pubDate>Mon, 20 Oct 2025 10:00:00 +0000</pubDate></item>
    </channel></rss>""".encode("utf-8")


def news_feature(
    *,
    event_id: str = "event-1",
    published_at: datetime | None = None,
    ingested_at: datetime | None = None,
    available_at: datetime | None = None,
    processed_at: datetime | None = None,
    score: float = 0.7,
    confidence: float = 0.9,
    calibration_status: str = "calibrated",
    license_status: str = "approved",
) -> NewsFeature:
    published = published_at or datetime(2024, 1, 1, 2, tzinfo=timezone.utc)
    ingested = ingested_at or published + timedelta(minutes=10)
    available = available_at or ingested + timedelta(minutes=20)
    processed = processed_at or available - timedelta(minutes=2)
    return NewsFeature(
        event_id=event_id,
        provider_id="persian-test-rss",
        source_id="persian-test-rss",
        published_at=published,
        ingested_at=ingested,
        available_at=available,
        processed_at=processed,
        revision="model-rev-1",
        asset_mapping=("BTC/USDT",),
        license_status=license_status,
        content_hash=("a" if event_id != "event-2" else "b") * 64,
        sentiment_label="positive" if score >= 0 else "negative",
        sentiment_score=score,
        confidence=confidence,
        event_type="market_commentary",
        model_id="owner-review-pending-model",
        model_revision="sha256-pinned-test-model",
        calibration_status=calibration_status,
    )


def test_persian_rss_parser_normalizes_deduplicates_and_sets_availability() -> None:
    ingested = datetime(2025, 10, 20, 10, 5, tzinfo=timezone.utc)
    events = parse_persian_rss(valid_rss(), provider=provider(), ingested_at=ingested)
    assert len(events) == 1
    event = events[0]
    assert event.text == "قیمت بیت‌کوین افزایش بازار"
    assert event.published_at < event.ingested_at
    assert event.available_at == ingested
    assert len(event.content_hash) == 64
    assert event.asset_mapping == ("BTC/USDT",)


def test_persian_rss_parser_rejects_unapproved_source_and_wrong_provider_kind() -> None:
    ingested = datetime(2025, 10, 20, 10, 5, tzinfo=timezone.utc)
    with pytest.raises(ProviderPolicyError, match="provider_inactive"):
        parse_persian_rss(
            valid_rss(),
            provider=provider(active=False),
            ingested_at=ingested,
        )
    with pytest.raises(ProviderError, match="provider_kind_invalid"):
        parse_persian_rss(
            valid_rss(),
            provider=provider(data_kind="market_analytics"),
            ingested_at=ingested,
        )


@pytest.mark.parametrize(
    ("body", "reason"),
    (
        (b"<rss>", "xml_invalid"),
        (b"<rss><channel /></rss>", "items_missing"),
        (
            b"<rss><channel><item><title></title><description></description>"
            b"<pubDate>Mon, 20 Oct 2025 10:00:00 +0000</pubDate></item></channel></rss>",
            "item_text_missing",
        ),
        (
            b"<rss><channel><item><title>news</title><description>x</description>"
            b"<pubDate>Tue, 21 Oct 2025 10:00:00 +0000</pubDate></item></channel></rss>",
            "published_after_ingestion",
        ),
    ),
)
def test_persian_rss_parser_rejects_invalid_or_future_items(
    body: bytes, reason: str
) -> None:
    with pytest.raises(ProviderError, match=reason):
        parse_persian_rss(
            body,
            provider=provider(),
            ingested_at=datetime(2025, 10, 20, 10, 5, tzinfo=timezone.utc),
        )


def test_sentiment_aggregation_is_point_in_time_and_requires_calibration() -> None:
    candles = candles_4h(3)
    decision_time = pd.Timestamp(candles["date"].iloc[0]).to_pydatetime() + timedelta(
        hours=4
    )
    recent_available = decision_time - timedelta(hours=1)
    old_available = decision_time - timedelta(hours=48)
    recent = news_feature(
        event_id="event-1",
        published_at=recent_available - timedelta(minutes=30),
        ingested_at=recent_available - timedelta(minutes=10),
        available_at=recent_available,
        processed_at=recent_available - timedelta(minutes=2),
        score=0.8,
    )
    old = news_feature(
        event_id="event-2",
        published_at=old_available - timedelta(minutes=30),
        ingested_at=old_available - timedelta(minutes=10),
        available_at=old_available,
        processed_at=old_available - timedelta(minutes=2),
        score=-0.4,
    )
    uncalibrated = news_feature(
        event_id="event-3",
        available_at=recent_available,
        ingested_at=recent_available - timedelta(minutes=20),
        published_at=recent_available - timedelta(minutes=30),
        processed_at=recent_available - timedelta(minutes=2),
        score=-0.9,
        calibration_status="uncalibrated",
        confidence=0.99,
    )
    future = news_feature(
        event_id="event-4",
        published_at=decision_time + timedelta(minutes=10),
        ingested_at=decision_time + timedelta(minutes=15),
        available_at=decision_time + timedelta(minutes=20),
        processed_at=decision_time + timedelta(minutes=18),
        score=-1.0,
    )
    output = aggregate_point_in_time_sentiment(
        candles, (recent, old, uncalibrated, future)
    )
    row = output.iloc[0]
    assert row["sentiment_mean_24h"] == pytest.approx(0.8)
    assert row["sentiment_mean_72h"] == pytest.approx(0.2)
    assert row["sentiment_event_count_24h"] == 1.0
    assert row["sentiment_available"] == np.bool_(True)
    assert tuple(name for name in SENTIMENT_COLUMNS if name not in output.columns) == ()


def test_sentiment_aggregation_does_not_impute_missing_news_and_rejects_bad_timezones() -> (
    None
):
    candles = candles_4h(2)
    output = aggregate_point_in_time_sentiment(candles, ())
    assert output["sentiment_mean_24h"].isna().all()
    assert not output["sentiment_available"].any()
    naive = candles.copy()
    naive["date"] = naive["date"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone_aware"):
        aggregate_point_in_time_sentiment(naive, ())


def test_sentiment_aggregation_rejects_unapproved_feature_license() -> None:
    candles = candles_4h(2)
    unapproved = news_feature(license_status="conditional")
    with pytest.raises(ProviderPolicyError, match="source_not_approved"):
        aggregate_point_in_time_sentiment(candles, (unapproved,))


def test_news_event_schema_fixture_keeps_owner_approved_provenance_fields() -> None:
    published = datetime(2025, 10, 20, 10, tzinfo=timezone.utc)
    event = NewsEvent(
        event_id="event-id",
        provider_id="persian-test-rss",
        source_id="persian-test-rss",
        published_at=published,
        ingested_at=published + timedelta(minutes=2),
        available_at=published + timedelta(minutes=3),
        revision="revision-1",
        asset_mapping=("BTC/USDT",),
        license_status="approved",
        content_hash="d" * 64,
        text="خبر آزمایشی",
        title="خبر",
    )
    assert event.schema_version
    assert event.available_at is not None
    assert event.available_at >= event.published_at
