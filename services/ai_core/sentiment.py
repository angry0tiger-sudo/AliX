"""Persian RSS parsing and strictly point-in-time sentiment feature aggregation."""

from __future__ import annotations

from datetime import datetime
from email.utils import parsedate_to_datetime
import hashlib
from collections.abc import Sequence

import pandas as pd
from defusedxml import ElementTree  # type: ignore[import-untyped]
from pandas import DataFrame

from external_data.contracts import (
    NewsEvent,
    NewsFeature,
    ProviderError,
    ProviderRegistration,
    digest_content,
    normalize_news_text,
    parse_timestamp,
    validate_news_feature,
)
from services.ai_core.features import SENTIMENT_COLUMNS

SENTIMENT_MINIMUM_CONFIDENCE = 0.65
SENTIMENT_WINDOW_24H_SECONDS = 24 * 60 * 60
SENTIMENT_WINDOW_72H_SECONDS = 72 * 60 * 60


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].casefold()


def _child_text(element: ElementTree.Element, *names: str) -> str:
    expected = {name.casefold() for name in names}
    for child in element.iter():
        if child is element:
            continue
        text = child.text
        if _local_name(str(child.tag)) in expected and isinstance(text, str):
            return text.strip()
    return ""


def _published_timestamp(value: str) -> datetime:
    if not value.strip():
        raise ProviderError("persian_rss_published_timestamp_missing")
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            raise ProviderError("persian_rss_published_timestamp_invalid") from None
    return parse_timestamp(parsed, "published_at")


def parse_persian_rss(
    xml_body: bytes,
    *,
    provider: ProviderRegistration,
    ingested_at: datetime,
    asset_mapping: tuple[str, ...] = ("BTC/USDT",),
) -> tuple[NewsEvent, ...]:
    """Parse an already-fetched RSS/Atom document after source approval; no network I/O."""
    provider.assert_ingestion_allowed()
    if provider.data_kind != "news":
        raise ProviderError("persian_rss_provider_kind_invalid")
    ingestion_time = parse_timestamp(ingested_at, "ingested_at")
    try:
        root = ElementTree.fromstring(xml_body)
    except (ElementTree.ParseError, ValueError):
        raise ProviderError("persian_rss_xml_invalid") from None
    items = [
        element
        for element in root.iter()
        if _local_name(element.tag) in {"item", "entry"}
    ]
    if not items:
        raise ProviderError("persian_rss_items_missing")
    events: list[NewsEvent] = []
    seen: set[str] = set()
    for item in items:
        title = _child_text(item, "title")
        description = _child_text(item, "description", "summary", "content")
        rendered = normalize_news_text(f"{title} {description}")
        if not rendered:
            raise ProviderError("persian_rss_item_text_missing")
        published = _published_timestamp(
            _child_text(item, "pubdate", "published", "updated")
        )
        if published > ingestion_time:
            raise ProviderError("persian_rss_published_after_ingestion")
        content_hash = digest_content(rendered)
        if content_hash in seen:
            continue
        seen.add(content_hash)
        identity = hashlib.sha256(
            f"{provider.provider_id}|{published.isoformat()}|{content_hash}".encode(
                "utf-8"
            )
        ).hexdigest()
        events.append(
            NewsEvent(
                event_id=identity,
                provider_id=provider.provider_id,
                source_id=provider.provider_id,
                published_at=published,
                ingested_at=ingestion_time,
                available_at=ingestion_time,
                revision=content_hash,
                asset_mapping=asset_mapping,
                license_status=provider.license_status,
                content_hash=content_hash,
                text=rendered,
                title=normalize_news_text(title),
            )
        )
    if not events:
        raise ProviderError("persian_rss_no_unique_items")
    return tuple(events)


def _eligible_sentiment(feature: NewsFeature) -> bool:
    validate_news_feature(feature)
    return (
        feature.license_status == "approved"
        and feature.calibration_status == "calibrated"
        and feature.confidence >= SENTIMENT_MINIMUM_CONFIDENCE
    )


def _window_summary(
    features: Sequence[NewsFeature], decision_time: datetime, window_seconds: int
) -> tuple[float | None, float | None, float | None, int]:
    eligible = [
        feature
        for feature in features
        if 0 <= (decision_time - feature.available_at).total_seconds() <= window_seconds
        and _eligible_sentiment(feature)
    ]
    if not eligible:
        return None, None, None, 0
    scores = [float(item.sentiment_score) for item in eligible]
    confidences = [float(item.confidence) for item in eligible]
    weighted_denominator = sum(confidences)
    weighted_mean = (
        sum(
            score * confidence
            for score, confidence in zip(scores, confidences, strict=True)
        )
        / weighted_denominator
    )
    return (
        sum(scores) / len(scores),
        weighted_mean,
        sum(confidences) / len(confidences),
        len(eligible),
    )


def aggregate_point_in_time_sentiment(
    candles_4h: DataFrame, news_features: Sequence[NewsFeature]
) -> DataFrame:
    """Join only calibrated, approved news features available by each 4H bar close."""
    if "date" not in candles_4h.columns or candles_4h.empty:
        raise ValueError("sentiment_4h_candle_timestamps_missing")
    dates = pd.to_datetime(candles_4h["date"], errors="raise")
    if dates.dt.tz is None:
        raise ValueError("sentiment_candle_timestamps_must_be_timezone_aware")
    dates = dates.dt.tz_convert("UTC")
    rows: list[dict[str, object]] = []
    sorted_features = tuple(
        sorted(
            news_features,
            key=lambda item: (item.available_at, item.event_id),
        )
    )
    for feature in sorted_features:
        parse_timestamp(feature.available_at, "sentiment_available_at")
        parse_timestamp(feature.published_at, "sentiment_published_at")
        if feature.available_at < feature.published_at:
            raise ProviderError("sentiment_available_before_publication")
        validate_news_feature(feature)
    for timestamp in dates:
        decision_time = (
            timestamp.to_pydatetime() + pd.Timedelta(hours=4).to_pytimedelta()
        )
        mean_24, weighted_24, confidence_24, count_24 = _window_summary(
            sorted_features, decision_time, SENTIMENT_WINDOW_24H_SECONDS
        )
        mean_72, _, _, count_72 = _window_summary(
            sorted_features, decision_time, SENTIMENT_WINDOW_72H_SECONDS
        )
        rows.append(
            {
                "date": timestamp,
                "sentiment_mean_24h": mean_24,
                "sentiment_mean_72h": mean_72,
                "sentiment_weighted_24h": weighted_24,
                "sentiment_event_count_24h": float(count_24),
                "sentiment_confidence_mean_24h": confidence_24,
                "sentiment_available": count_72 > 0,
            }
        )
    return DataFrame(rows, columns=["date", *SENTIMENT_COLUMNS])


__all__ = [
    "SENTIMENT_MINIMUM_CONFIDENCE",
    "aggregate_point_in_time_sentiment",
    "parse_persian_rss",
]
