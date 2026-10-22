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
        fields = (
            self.model_id, self.revision, self.artifact_dir, self.weights_file,
            self.tokenizer_file, self.weights_sha256, self.tokenizer_sha256,
            self.model_license_status, self.dataset_license_status, self.attribution,
        )
        if any(not isinstance(item, str) or not item.strip() for item in fields):
            raise NlpServiceError("nlp_model_manifest_incomplete")
        if self.schema_version != MODEL_MANIFEST_SCHEMA_VERSION:
            raise NlpServiceError("nlp_model_manifest_schema_mismatch")
        if not IMMUTABLE_REVISION_PATTERN.fullmatch(self.revision):
            raise NlpServiceError("nlp_model_revision_not_immutable")
        if not SHA256_PATTERN.fullmatch(self.weights_sha256):
            raise NlpServiceError("nlp_weights_hash_invalid")
        if not SHA256_PATTERN.fullmatch(self.tokenizer_sha256):
            raise NlpServiceError("nlp_tokenizer_hash_invalid")
        for name in (self.weights_file, self.tokenizer_file):
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise NlpServiceError("nlp_artifact_path_invalid")
        labels = dict(self.label_map)
        if not labels or len(labels) != len(self.label_map):
            raise NlpServiceError("nlp_label_map_invalid")
        if any(value not in CANONICAL_LABELS for value in labels.values()):
            raise NlpServiceError("nlp_label_map_invalid")
        if self.test_fixture:
            valid_fixture = (
                self.model_id.startswith("fixture/")
                and self.model_license_status == "test_only"
                and self.dataset_license_status == "test_only"
                and not self.commercial_use_approved
            )
            if not valid_fixture:
                raise NlpServiceError("nlp_test_fixture_manifest_invalid")
        elif (
            self.model_license_status != "approved"
            or self.dataset_license_status != "approved"
            or not self.commercial_use_approved
        ):
            raise NlpServiceError("nlp_model_or_dataset_license_not_approved")


class TextClassifier(Protocol):
    """Injected classifier; live/network transport is outside this service."""

    def predict(self, text: str) -> Mapping[str, object]:
        """Return a raw label and its confidence-like score."""


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
        try:
            raw = self.classifier.predict(event.text)
        except TimeoutError:
            raise NlpServiceError("nlp_timeout") from None
        except Exception:
            raise NlpServiceError("nlp_unavailable") from None
        if not isinstance(raw, Mapping):
            raise NlpServiceError("nlp_output_schema_invalid")
        raw_label, raw_score = raw.get("label"), raw.get("score")
        if not isinstance(raw_label, str):
            raise NlpServiceError("nlp_output_label_invalid")
        label = dict(self.manifest.label_map).get(raw_label)
        if label is None:
            raise NlpServiceError("nlp_output_label_unmapped")
        if (
            isinstance(raw_score, bool)
            or not isinstance(raw_score, (int, float))
            or not math.isfinite(float(raw_score))
            or not 0.0 <= float(raw_score) <= 1.0
        ):
            raise NlpServiceError("nlp_output_confidence_invalid")
        processed = parse_timestamp(self.clock(), "processed_at")
        available = parse_timestamp(event.available_at, "available_at")
        if processed < available:
            raise NlpServiceError("nlp_clock_before_source_available")
        confidence = float(raw_score)
        score = confidence if label == "positive" else -confidence if label == "negative" else 0.0
        result = NewsFeature(
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
            sentiment_score=score,
            confidence=confidence,
            event_type="unknown",
            model_id=self.manifest.model_id,
            model_revision=self.manifest.revision,
            calibration_status="uncalibrated",
        )
        validate_news_feature(result)
        return result
