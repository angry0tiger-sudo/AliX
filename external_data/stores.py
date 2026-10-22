"""Separated in-memory stores for deterministic contract tests and research."""

from __future__ import annotations

from enum import StrEnum

from external_data.contracts import (
    MarketObservation,
    NewsEvent,
    NewsFeature,
    ProviderError,
    validate_market_observation,
    validate_news_event,
    validate_news_feature,
)


class StoreWriteResult(StrEnum):
    STORED = "stored"
    DUPLICATE = "duplicate"


class InMemoryNewsStore:
    """Store raw text separately and deduplicate normalized content per provider."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str, str], NewsEvent] = {}
        self._content_keys: set[tuple[str, str]] = set()

    def add(self, event: NewsEvent) -> StoreWriteResult:
        validate_news_event(event)
        content_key = (event.provider_id, event.content_hash)
        record_key = (event.provider_id, event.event_id, event.revision)
        previous = self._records.get(record_key)
        if previous is not None:
            if previous.content_hash != event.content_hash:
                raise ProviderError("news_event_revision_conflict")
            return StoreWriteResult.DUPLICATE
        if content_key in self._content_keys:
            return StoreWriteResult.DUPLICATE
        self._records[record_key] = event
        self._content_keys.add(content_key)
        return StoreWriteResult.STORED

    def records(self) -> tuple[NewsEvent, ...]:
        return tuple(
            sorted(
                self._records.values(),
                key=lambda item: (
                    item.available_at.isoformat() if item.available_at else "",
                    item.provider_id,
                    item.event_id,
                    item.revision,
                ),
            )
        )


class InMemoryExternalFeatureStore:
    """Store numeric provider observations without accepting raw text."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str, str], MarketObservation] = {}

    def add(self, observation: MarketObservation) -> StoreWriteResult:
        validate_market_observation(observation)
        key = (observation.provider_id, observation.event_id, observation.revision)
        previous = self._records.get(key)
        if previous is not None:
            if previous.content_hash != observation.content_hash:
                raise ProviderError("market_event_revision_conflict")
            return StoreWriteResult.DUPLICATE
        self._records[key] = observation
        return StoreWriteResult.STORED

    def records(self) -> tuple[MarketObservation, ...]:
        return tuple(
            sorted(
                self._records.values(),
                key=lambda item: (
                    item.available_at.isoformat() if item.available_at else "",
                    item.provider_id,
                    item.event_id,
                    item.revision,
                ),
            )
        )


class InMemoryNlpFeatureStore:
    """Store derived NLP features separately from raw text and numeric analytics."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str, str, str], NewsFeature] = {}

    def add(self, feature: NewsFeature) -> StoreWriteResult:
        validate_news_feature(feature)
        key = (
            feature.provider_id,
            feature.event_id,
            feature.revision,
            feature.model_revision,
        )
        previous = self._records.get(key)
        if previous is not None:
            matches = (
                previous.content_hash == feature.content_hash
                and previous.sentiment_label == feature.sentiment_label
                and previous.sentiment_score == feature.sentiment_score
                and previous.confidence == feature.confidence
            )
            if not matches:
                raise ProviderError("nlp_feature_revision_conflict")
            return StoreWriteResult.DUPLICATE
        self._records[key] = feature
        return StoreWriteResult.STORED

    def records(self) -> tuple[NewsFeature, ...]:
        return tuple(
            sorted(
                self._records.values(),
                key=lambda item: (
                    item.available_at.isoformat(),
                    item.provider_id,
                    item.event_id,
                    item.revision,
                    item.model_revision,
                ),
            )
        )
