"""Regression tests for the shadow validator's HMM artifact contract."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import run_ai_shadow_validation as shadow_validation

MODEL_SHA256 = "a" * 64
DATASET_SHA256 = "b" * 64
QUALITY_FINGERPRINT = "c" * 64


def _hmm_artifact(*, nested_owner_approved: bool = False) -> dict[str, Any]:
    return {
        "status": "TRAINED_UNAPPROVED",
        "model_sha256": MODEL_SHA256,
        "model": {"owner_approved": nested_owner_approved},
        "dataset": {
            "csv_sha256": DATASET_SHA256,
            "data_quality_fingerprint": QUALITY_FINGERPRINT,
        },
        "economic_acceptance": "NO-GO",
        "order_permission": False,
        "trading_authorization": "NO-GO",
    }


def _run_hmm_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hmm: dict[str, Any],
) -> list[dict[str, Any]]:
    artifact_path = tmp_path / "hmm-regime-model.json"
    artifact_path.write_text(json.dumps(hmm), encoding="utf-8")
    loaded_documents: list[dict[str, Any]] = []

    def load_model_document(document: str) -> SimpleNamespace:
        decoded: Any = json.loads(document)
        assert isinstance(decoded, dict)
        model = decoded.get("model")
        assert isinstance(model, dict)
        assert model.get("owner_approved") is False
        loaded_documents.append(decoded)
        return SimpleNamespace(owner_approved=False)

    monkeypatch.setattr(
        shadow_validation, "load_hmm_model_document", load_model_document
    )
    shadow_validation._validate_hmm_training_artifact(
        hmm,
        {
            "data_quality_fingerprint": QUALITY_FINGERPRINT,
            "hmm": {"model_sha256": MODEL_SHA256},
        },
        dataset_hash=DATASET_SHA256,
        quality_fingerprint=QUALITY_FINGERPRINT,
        artifact_path=artifact_path,
    )
    return loaded_documents


def test_hmm_shadow_gate_reads_nested_unapproved_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hmm = _hmm_artifact()

    loaded = _run_hmm_validation(tmp_path, monkeypatch, hmm)

    assert len(loaded) == 1
    assert "owner_approved" not in hmm
    assert loaded[0]["model"]["owner_approved"] is False


def test_hmm_shadow_gate_rejects_nested_approved_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hmm = _hmm_artifact(nested_owner_approved=True)
    artifact_path = tmp_path / "hmm-regime-model.json"
    artifact_path.write_text(json.dumps(hmm), encoding="utf-8")

    with pytest.raises(
        ValueError, match="shadow_validation_hmm_source_must_remain_unapproved"
    ):
        shadow_validation._validate_hmm_training_artifact(
            hmm,
            {
                "data_quality_fingerprint": QUALITY_FINGERPRINT,
                "hmm": {"model_sha256": MODEL_SHA256},
            },
            dataset_hash=DATASET_SHA256,
            quality_fingerprint=QUALITY_FINGERPRINT,
            artifact_path=artifact_path,
        )
