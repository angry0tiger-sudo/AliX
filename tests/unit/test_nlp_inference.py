"""Offline tests for the manifest-gated local Hugging Face inference boundary."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import hashlib
from pathlib import Path

import pytest

from external_data.contracts import NewsEvent, ProviderError, digest_content
from services.nlp_inference.huggingface_local import load_local_huggingface_classifier
from services.nlp_inference.service import ModelManifest, NlpService, NlpServiceError


def ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class FakeClassifier:
    def __init__(self, value: dict[str, object] | None = None) -> None:
        self.value = value or {"label": "POS", "score": 0.8}
        self.failure: Exception | None = None

    def predict(self, text: str) -> dict[str, object]:
        if self.failure is not None:
            raise self.failure
        return self.value


def manifest(
    *,
    model_license: str = "test_only",
    dataset_license: str = "test_only",
    commercial_approved: bool = False,
    fixture: bool = True,
    artifact_dir: str = "/fixture/model",
    weights_file: str = "model.safetensors",
    tokenizer_file: str = "tokenizer.json",
    weights_hash: str = "1" * 64,
    tokenizer_hash: str = "2" * 64,
    revision: str = "a" * 40,
) -> ModelManifest:
    return ModelManifest(
        model_id="fixture/sentiment-test" if fixture else "local/model",
        revision=revision,
        artifact_dir=artifact_dir,
        weights_file=weights_file,
        tokenizer_file=tokenizer_file,
        weights_sha256=weights_hash,
        tokenizer_sha256=tokenizer_hash,
        model_license_status=model_license,
        dataset_license_status=dataset_license,
        commercial_use_approved=commercial_approved,
        label_map=(("POS", "positive"), ("NEG", "negative"), ("NEU", "neutral")),
        attribution="synthetic-only test fixture",
        test_fixture=fixture,
    )


def event(text: str = "synthetic text", license_status: str = "approved") -> NewsEvent:
    return NewsEvent(
        event_id="event-1",
        provider_id="fixture-news",
        source_id="https://news.fixture.invalid/story",
        published_at=ts("2026-10-22T09:00:00Z"),
        ingested_at=ts("2026-10-22T09:01:00Z"),
        available_at=ts("2026-10-22T09:01:00Z"),
        revision="content:" + digest_content(text)[:16],
        asset_mapping=("BTC/USDT",),
        license_status=license_status,
        content_hash=digest_content(text),
        text=text,
        title="synthetic headline",
    )


def service(
    *,
    classifier: FakeClassifier | None = None,
    clock: datetime | None = None,
    model_manifest: ModelManifest | None = None,
) -> NlpService:
    return NlpService(
        manifest=model_manifest or manifest(),
        classifier=classifier or FakeClassifier(),
        clock=lambda: clock or ts("2026-10-22T09:03:00Z"),
        allow_test_fixture=True,
    )


def test_nlp_emits_versioned_features_after_local_inference() -> None:
    result = service().infer(event())
    assert result.sentiment_label == "positive"
    assert result.sentiment_score == pytest.approx(0.8)
    assert result.confidence == pytest.approx(0.8)
    assert result.processed_at == ts("2026-10-22T09:03:00Z")
    assert result.available_at == result.processed_at
    assert result.model_revision == "a" * 40
    assert result.calibration_status == "uncalibrated"
    assert result.event_type == "unknown"
    assert result.schema_version == "alix.nlp.sentiment.v1"
    negative = service(classifier=FakeClassifier({"label": "NEG", "score": 0.7})).infer(
        event()
    )
    neutral = service(classifier=FakeClassifier({"label": "NEU", "score": 0.0})).infer(
        event()
    )
    assert negative.sentiment_score == pytest.approx(-0.7)
    assert neutral.sentiment_score == 0.0


def test_manifest_rejects_unapproved_license_unpinned_revision_and_paths() -> None:
    with pytest.raises(NlpServiceError, match="license_not_approved"):
        manifest(
            model_license="conditional",
            dataset_license="approved",
            commercial_approved=True,
            fixture=False,
        )
    with pytest.raises(NlpServiceError, match="license_not_approved"):
        manifest(
            model_license="approved",
            dataset_license="conditional",
            commercial_approved=True,
            fixture=False,
        )
    with pytest.raises(NlpServiceError, match="revision_not_immutable"):
        manifest(revision="main")
    with pytest.raises(NlpServiceError, match="hash_invalid"):
        manifest(weights_hash="bad")
    with pytest.raises(NlpServiceError, match="path_invalid"):
        manifest(weights_file="../outside.safetensors")
    with pytest.raises(NlpServiceError, match="fixture_manifest_invalid"):
        manifest(model_license="approved")
    with pytest.raises(NlpServiceError, match="label_map_invalid"):
        replace(manifest(), label_map=(("POS", "unknown"),))


def test_nlp_fails_closed_for_invalid_input_output_and_failure() -> None:
    with pytest.raises(NlpServiceError, match="test_fixture_not_allowed"):
        NlpService(manifest(), FakeClassifier(), lambda: ts("2026-10-22T09:03:00Z"))
    with pytest.raises(ProviderError, match="source_not_approved"):
        service().infer(event(license_status="conditional"))
    with pytest.raises(NlpServiceError, match="input_text_missing"):
        service().infer(event("  "))
    timeout = FakeClassifier()
    timeout.failure = TimeoutError("private exception detail")
    with pytest.raises(NlpServiceError, match="nlp_timeout") as caught:
        service(classifier=timeout).infer(event())
    assert "private exception detail" not in str(caught.value)
    failure = FakeClassifier()
    failure.failure = RuntimeError("secret detail")
    with pytest.raises(NlpServiceError, match="nlp_unavailable") as caught:
        service(classifier=failure).infer(event())
    assert "secret detail" not in str(caught.value)
    with pytest.raises(NlpServiceError, match="label_unmapped"):
        service(classifier=FakeClassifier({"label": "LABEL_77", "score": 0.9})).infer(
            event()
        )
    for score in (-0.1, 1.1, float("nan"), float("inf"), True):
        with pytest.raises(NlpServiceError, match="confidence_invalid"):
            service(classifier=FakeClassifier({"label": "POS", "score": score})).infer(
                event()
            )
    with pytest.raises(NlpServiceError, match="clock_before_source"):
        service(clock=ts("2026-10-22T09:00:00Z")).infer(event())


def test_nlp_rejects_bad_shape_missing_timestamp_and_future_publication() -> None:
    class BadClassifier:
        def predict(self, text: str) -> str:
            return "positive"

    with pytest.raises(NlpServiceError, match="output_schema_invalid"):
        NlpService(
            manifest(), BadClassifier(), lambda: ts("2026-10-22T09:03:00Z"), True
        ).infer(event())  # type: ignore[arg-type]
    with pytest.raises(ProviderError, match="published_after_ingest"):
        service().infer(replace(event(), published_at=ts("2026-10-22T09:05:00Z")))
    with pytest.raises(ProviderError, match="missing_timestamp"):
        service().infer(replace(event(), available_at=None))


def test_local_model_loader_verifies_hashes_and_never_uses_remote_id(
    tmp_path: Path,
) -> None:
    weights, tokenizer = b"synthetic weight bytes", b"synthetic tokenizer bytes"
    (tmp_path / "model.safetensors").write_bytes(weights)
    (tmp_path / "tokenizer.json").write_bytes(tokenizer)
    approved = manifest(
        model_license="approved",
        dataset_license="approved",
        commercial_approved=True,
        fixture=False,
        artifact_dir=str(tmp_path),
        weights_hash=hashlib.sha256(weights).hexdigest(),
        tokenizer_hash=hashlib.sha256(tokenizer).hexdigest(),
    )
    seen: list[Path] = []

    def fake_factory(path: Path) -> object:
        seen.append(path)
        return lambda text, **kwargs: [{"label": "POS", "score": 0.75}]

    loaded = load_local_huggingface_classifier(approved, pipeline_factory=fake_factory)
    assert seen == [tmp_path.resolve()]
    assert loaded.predict("fixture") == {"label": "POS", "score": 0.75}
    (tmp_path / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(NlpServiceError, match="hash_mismatch"):
        load_local_huggingface_classifier(approved, pipeline_factory=fake_factory)


def test_local_model_loader_rejects_fixture_manifest_missing_files_path_escape_and_bad_pipeline(
    tmp_path: Path,
) -> None:
    with pytest.raises(NlpServiceError, match="test_fixture_cannot_load"):
        load_local_huggingface_classifier(
            manifest(), pipeline_factory=lambda path: None
        )
    approved = manifest(
        model_license="approved",
        dataset_license="approved",
        commercial_approved=True,
        fixture=False,
        artifact_dir=str(tmp_path),
    )
    with pytest.raises(NlpServiceError, match="artifact_file_missing"):
        load_local_huggingface_classifier(approved, pipeline_factory=lambda path: None)
    with pytest.raises(NlpServiceError, match="artifact_directory_missing"):
        load_local_huggingface_classifier(
            replace(approved, artifact_dir=str(tmp_path / "missing")),
            pipeline_factory=lambda path: None,
        )
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside")
    (tmp_path / "tokenizer.json").write_bytes(b"tokenizer")
    (tmp_path / "link.safetensors").symlink_to(outside)
    escaped = replace(
        approved,
        weights_file="link.safetensors",
        weights_sha256=hashlib.sha256(b"outside").hexdigest(),
        tokenizer_sha256=hashlib.sha256(b"tokenizer").hexdigest(),
    )
    with pytest.raises(NlpServiceError, match="path_escape"):
        load_local_huggingface_classifier(escaped, pipeline_factory=lambda path: None)
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    proper = replace(
        approved,
        weights_sha256=hashlib.sha256(b"weights").hexdigest(),
        tokenizer_sha256=hashlib.sha256(b"tokenizer").hexdigest(),
    )
    with pytest.raises(NlpServiceError, match="pipeline_invalid"):
        load_local_huggingface_classifier(
            proper, pipeline_factory=lambda path: object()
        )
