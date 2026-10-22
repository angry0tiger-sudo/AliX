"""Revision-aware, available-at-only join for NLP features."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json

from external_data.contracts import (
    NewsFeature,
    ProviderError,
    parse_timestamp,
    validate_news_feature,
)


@dataclass(frozen=True, slots=True)
class PointInTimeJoinResult:
    """Selected eligible features and a deterministic fingerprint."""

    records: tuple[NewsFeature, ...]
    decision_timestamp: datetime
    asset: str
    fingerprint: str


def _eligible_records(
    records: Sequence[NewsFeature],
    asset: str,
    start: datetime,
    decision: datetime,
) -> list[NewsFeature]:
    eligible: list[NewsFeature] = []
    for feature in records:
        validate_news_feature(feature)
        available = parse_timestamp(feature.available_at, "available_at")
        in_window = start <= available <= decision
        matches_asset = asset in {item.upper() for item in feature.asset_mapping}
        if in_window and matches_asset:
            eligible.append(feature)
    return eligible


def _latest_revision_per_event(
    eligible: Sequence[NewsFeature],
) -> list[NewsFeature]:
    groups: dict[tuple[str, str], list[NewsFeature]] = {}
    for feature in eligible:
        groups.setdefault((feature.provider_id, feature.event_id), []).append(feature)

    selected: list[NewsFeature] = []
    for versions in groups.values():
        latest_time = max(item.available_at for item in versions)
        latest = [item for item in versions if item.available_at == latest_time]
        identities = {(item.revision, item.content_hash) for item in latest}
        if len(identities) > 1:
            raise ProviderError("news_revision_timestamp_conflict")
        selected.append(latest[0])
    return selected


def _deduplicate_content(records: Sequence[NewsFeature]) -> tuple[NewsFeature, ...]:
    ordered = sorted(
        records,
        key=lambda item: (
            item.available_at,
            item.provider_id,
            item.event_id,
            item.revision,
        ),
    )
    unique: dict[tuple[str, str], NewsFeature] = {}
    for feature in ordered:
        unique.setdefault((feature.provider_id, feature.content_hash), feature)
    return tuple(unique.values())


def _join_fingerprint(
    records: Sequence[NewsFeature], asset: str, start: datetime, decision: datetime
) -> str:
    manifest = {
        "schema_version": "alix.point-in-time-join.v1",
        "asset": asset,
        "decision_timestamp": decision.isoformat(),
        "window_start": start.isoformat(),
        "features": [
            {
                "provider_id": item.provider_id,
                "event_id": item.event_id,
                "revision": item.revision,
                "available_at": item.available_at.isoformat(),
                "content_hash": item.content_hash,
                "model_revision": item.model_revision,
            }
            for item in records
        ],
    }
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def join_news_features_as_of(
    records: Sequence[NewsFeature],
    *,
    asset: str,
    decision_timestamp: datetime | str,
    window_start: datetime | str,
) -> PointInTimeJoinResult:
    """Join only allowed features available within an explicit time window."""
    decision = parse_timestamp(decision_timestamp, "decision_timestamp")
    start = parse_timestamp(window_start, "window_start")
    normalized_asset = asset.strip().upper()
    if not normalized_asset:
        raise ProviderError("asset_required")
    if start > decision:
        raise ProviderError("window_start_after_decision")
    eligible = _eligible_records(records, normalized_asset, start, decision)
    latest = _latest_revision_per_event(eligible)
    selected = _deduplicate_content(latest)
    fingerprint = _join_fingerprint(selected, normalized_asset, start, decision)
    return PointInTimeJoinResult(selected, decision, normalized_asset, fingerprint)
