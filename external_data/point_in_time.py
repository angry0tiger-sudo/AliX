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

    eligible: list[NewsFeature] = []
    for feature in records:
        validate_news_feature(feature)
        available = parse_timestamp(feature.available_at, "available_at")
        if available < start or available > decision:
            continue
        if normalized_asset in {item.upper() for item in feature.asset_mapping}:
            eligible.append(feature)

    groups: dict[tuple[str, str], list[NewsFeature]] = {}
    for feature in eligible:
        groups.setdefault((feature.provider_id, feature.event_id), []).append(feature)

    selected: list[NewsFeature] = []
    for versions in groups.values():
        latest_time = max(item.available_at for item in versions)
        latest = [item for item in versions if item.available_at == latest_time]
        distinct_versions = {(item.revision, item.content_hash) for item in latest}
        if len(distinct_versions) != 1:
            raise ProviderError("news_revision_timestamp_conflict")
        selected.append(latest[0])

    unique: dict[tuple[str, str], NewsFeature] = {}
    for feature in sorted(
        selected,
        key=lambda item: (
            item.available_at, item.provider_id, item.event_id, item.revision,
        ),
    ):
        unique.setdefault((feature.provider_id, feature.content_hash), feature)
    ordered = tuple(unique.values())
    manifest = {
        "schema_version": "alix.point-in-time-join.v1",
        "asset": normalized_asset,
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
            for item in ordered
        ],
    }
    fingerprint = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return PointInTimeJoinResult(ordered, decision, normalized_asset, fingerprint)
