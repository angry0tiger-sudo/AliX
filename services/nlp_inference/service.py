"""Local NLP inference with immutable model and data-license contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import PurePosixPath
import re
from typing import Protocol

from external_data.contracts import (
    NewsEvent,
    NewsFeature,
    ProviderError,
    parse_timestamp,
    validate_news_event,
    validate_news_feature,
)

MODEL_MANIFEST_SCHEMA_VERSION = "alix.nlp.model-manifest.v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
IMMUTABLE_REVISION_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
CANONICAL_LABELS = frozenset({"positive", "negative", "neutral"})


class NlpServiceError(ProviderError):
    """A stable, sanitized NLP failure."""


def _validate_manifest_required_fields(manifest: ModelManifest) -> None:
    fields = (
        manifest.model_id,
        manifest.revision,
        manifest.artifact_dir,
        manifest.weights_file,
        manifest.tokenizer_file,
        manifest.weights_sha256,
        manifest.tokenizer_sha256,
        manifest.model_license_status,
        manifest.dataset_license_status,
        manifest.attribution,
    )
    if any(not isinstance(item, str) or not item.strip() for item in fields):
        raise NlpServiceError("nlp_model_manifest_incomplete")
    if manifest.schema_version != MODEL_MANIFEST_SCHEMA_VERSION:
        raise NlpServiceError("nlp_model_manifest_schema_mismatch")


def _validate_manifest_hashes(manifest: ModelManifest) -> None:
    if not IMMUTABLE_REVISION_PATTERN.fullmatch(manifest.revision):
        raise NlpServiceError("nlp_model_revision_not_immutable")
    if not SHA256_PATTERN.fullmatch(manifest.weights_sha256):
        raise NlpServiceError("nlp_weights_hash_invalid")
    if not SHA256_PATTERN.fullmatch(manifest.tokenizer_sha256):
        raise NlpServiceError("nlp_tokenizer_hash_invalid")


def _validate_manifest_paths(manifest: ModelManifest) -> None:
    for name in (manifest.weights_file, manifest.tokenizer_file):
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise NlpServiceError("nlp_artifact_path_invalid")


def _validate_manifest_labels(manifest: ModelManifest) -> None:
    labels = dict(manifest.label_map)
    if not labels or len(labels) != len(manifest.label_map):
        raise NlpServiceError("nlp_label_map_invalid")
    if any(value not in CANONICAL_LABELS for value in labels.values()):
        raise NlpServiceError("nlp_label_map_invalid")


def _validate_manifest_rights(manifest: ModelManifest) -> None:
    if manifest.test_fixture:
        valid_fixture = (
            manifest.model_id.startswith("fixture/")
            and manifest.model_license_status == "test_only"
            and manifest.dataset_license_status == "test_only"
            and not manifest.commercial_use_approved
        )
        if not valid_fixture:
            raise NlpServiceError("nlp_test_fixture_manifest_invalid")
        return
    approved_rights = (
        manifest.model_license_status == "approved"
        and manifest.dataset_license_status == "approved"
        and manifest.commercial_use_approved
    )
    if not approved_rights:
        raise NlpServiceError("nlp_model_or_dataset_license_not_approved")


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """Reviewed model artifact, immutable revision and license manifest."""

    model_id: str
    revision: str
    artifact_dir: str
    weights_file: str
    tokenizer_file: str
    weights_sha256: str
    tokenizer_sha256: str
    model_license_status: str
    dataset_license_status: str
    commercial_use_approved: bool
    label_map: tuple[tuple[str, str], ...]
    attribution: str
    test_fixture: bool = False
    schema_version: str = MODEL_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_manifest_required_fields(self)
        _validate_manifest_hashes(self)
        _validate_manifest_paths(self)
        _validate_manifest_labels(self)
        _validate_manifest_rights(self)


class TextClassifier(Protocol):
    """Injected classifier; live/network transport is outside this service."""

    def predict(self, text: str) -> Mapping[str, object]:
        """Return a raw label and its confidence-like score."""


def _classify(classifier: TextClassifier, text: str) -> Mapping[str, object]:
    try:
        raw = classifier.predict(text)
    except TimeoutError:
        raise NlpServiceError("nlp_timeout") from None
    except Exception:
        raise NlpServiceError("nlp_unavailable") from None
    if not isinstance(raw, Mapping):
        raise NlpServiceError("nlp_output_schema_invalid")
    return raw


def _validate_classifier_output(
    raw: Mapping[str, object], manifest: ModelManifest
) -> tuple[str, float]:
    raw_label = raw.get("label")
    raw_score = raw.get("score")
    if not isinstance(raw_label, str):
        raise NlpServiceError("nlp_output_label_invalid")
    label = dict(manifest.label_map).get(raw_label)
    if label is None:
        raise NlpServiceError("nlp_output_label_unmapped")
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        raise NlpServiceError("nlp_output_confidence_invalid")
    confidence = float(raw_score)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise NlpServiceError("nlp_output_confidence_invalid")
    return label, confidence


def _signed_sentiment_score(label: str, confidence: float) -> float:
    if label == "positive":
        return confidence
    if label == "negative":
        return -confidence
    return 0.0


def _build_news_feature(
    event: NewsEvent,
    manifest: ModelManifest,
    label: str,
    confidence: float,
    processed: datetime,
) -> NewsFeature:
    feature = NewsFeature(
        event_id=event.event_id,
        provider_id=event.provider_id,
        source_id=event.source_id,
        published_at=parse_timestamp(event.published_at, "published_at"),
        ingested_at=parse_timestamp(event.ingested_at, "ingested_at"),
        available_at=processed,
        processed_at=processed,
        revision=event.revision,
        asset_mapping=event.asset_mapping,
        license_status=event.license_status,
        content_hash=event.content_hash,
        sentiment_label=label,
        sentiment_score=_signed_sentiment_score(label, confidence),
        confidence=confidence,
        event_type="unknown",
        model_id=manifest.model_id,
        model_revision=manifest.revision,
        calibration_status="uncalibrated",
    )
    validate_news_feature(feature)
    return feature


@dataclass(frozen=True, slots=True)
class NlpService:
    """Validate input and produce auditable features without trading authority."""

    manifest: ModelManifest
    classifier: TextClassifier
    clock: Callable[[], datetime]
    allow_test_fixture: bool = False

    def __post_init__(self) -> None:
        if self.manifest.test_fixture and not self.allow_test_fixture:
            raise NlpServiceError("nlp_test_fixture_not_allowed_for_runtime")

    def infer(self, event: NewsEvent) -> NewsFeature:
        validate_news_event(event)
        if not event.text.strip():
            raise NlpServiceError("nlp_input_text_missing")
        raw = _classify(self.classifier, event.text)
        label, confidence = _validate_classifier_output(raw, self.manifest)
        processed = parse_timestamp(self.clock(), "processed_at")
        available = parse_timestamp(event.available_at, "available_at")
        if processed < available:
            raise NlpServiceError("nlp_clock_before_source_available")
        return _build_news_feature(event, self.manifest, label, confidence, processed)
