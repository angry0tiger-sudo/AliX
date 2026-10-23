"""Strict JSON loader for versioned HMM training artifacts."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from services.strategy_selection.regime import (
    GaussianRegimeEmission,
    RegimeHMMModel,
)

MODEL_SCHEMA_VERSION = "alix.hmm.regime-model.v1"


def _canonical_hash(value: Mapping[str, Any]) -> str:
    content = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _decimal(value: object, field: str) -> Decimal:
    if not isinstance(value, str):
        raise ValueError(f"hmm_artifact_{field}_must_be_string")
    try:
        result = Decimal(value)
    except Exception as error:
        raise ValueError(f"hmm_artifact_{field}_invalid") from error
    if not result.is_finite():
        raise ValueError(f"hmm_artifact_{field}_non_finite")
    return result


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"hmm_artifact_{field}_must_be_object")
    return value


def _string_list(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"hmm_artifact_{field}_invalid")
    return tuple(value)


def _decimal_list(value: object, field: str) -> tuple[Decimal, ...]:
    if not isinstance(value, list):
        raise ValueError(f"hmm_artifact_{field}_invalid")
    return tuple(_decimal(item, field) for item in value)


def _model_from_payload(model: Mapping[str, Any], model_hash: str) -> RegimeHMMModel:
    emissions_payload = model.get("emissions")
    if not isinstance(emissions_payload, list):
        raise ValueError("hmm_artifact_emissions_invalid")
    emissions: list[GaussianRegimeEmission] = []
    for item in emissions_payload:
        emission = _mapping(item, "emission")
        emissions.append(
            GaussianRegimeEmission(
                regime=str(emission.get("regime", "")),
                means=_decimal_list(emission.get("means"), "means"),
                standard_deviations=_decimal_list(
                    emission.get("standard_deviations"), "standard_deviations"
                ),
            )
        )
    transition_payload = model.get("transition_matrix")
    if not isinstance(transition_payload, list):
        raise ValueError("hmm_artifact_transition_matrix_invalid")
    transition = tuple(
        _decimal_list(row, "transition_row") for row in transition_payload
    )
    initial = _decimal_list(model.get("initial_probabilities"), "initial_probabilities")
    feature_names = _string_list(model.get("feature_names"), "feature_names")
    regimes = _string_list(model.get("regimes"), "regimes")
    trained_at_text = model.get("trained_at")
    if not isinstance(trained_at_text, str):
        raise ValueError("hmm_artifact_trained_at_invalid")
    try:
        trained_at = datetime.fromisoformat(trained_at_text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("hmm_artifact_trained_at_invalid") from error
    integer_fields = ("max_model_age_seconds", "max_observation_age_seconds")
    values: dict[str, object] = {
        "revision": model.get("revision"),
        "model_sha256": model_hash,
        "trained_at": trained_at,
        "owner_approved": model.get("owner_approved"),
        "feature_schema_version": model.get("feature_schema_version"),
        "feature_names": feature_names,
        "regimes": regimes,
        "initial_probabilities": initial,
        "transition_matrix": transition,
        "emissions": tuple(emissions),
        "minimum_confidence": _decimal(model.get("minimum_confidence"), "minimum_confidence"),
        "minimum_posterior_margin": _decimal(
            model.get("minimum_posterior_margin"), "minimum_posterior_margin"
        ),
    }
    for field in integer_fields:
        value = model.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"hmm_artifact_{field}_invalid")
        values[field] = value
    if not isinstance(values["revision"], str) or not isinstance(
        values["feature_schema_version"], str
    ):
        raise ValueError("hmm_artifact_identity_invalid")
    if not isinstance(values["owner_approved"], bool):
        raise ValueError("hmm_artifact_owner_approval_invalid")
    return RegimeHMMModel(**values)  # type: ignore[arg-type]


def load_hmm_model_document(document: str) -> RegimeHMMModel:
    """Parse and verify an HMM artifact; invalid or mutated weights are rejected."""
    try:
        decoded = json.loads(document)
    except json.JSONDecodeError as error:
        raise ValueError("hmm_artifact_json_invalid") from error
    payload = _mapping(decoded, "root")
    if payload.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError("hmm_artifact_schema_version_mismatch")
    if payload.get("status") != "TRAINED_UNAPPROVED":
        raise ValueError("hmm_artifact_status_not_loadable")
    if payload.get("economic_acceptance") != "NO-GO":
        raise ValueError("hmm_artifact_economic_gate_must_remain_no_go")
    if payload.get("trading_authorization") != "NO-GO":
        raise ValueError("hmm_artifact_trading_authorization_must_remain_no_go")
    model = _mapping(payload.get("model"), "model")
    model_hash = payload.get("model_sha256")
    if not isinstance(model_hash, str) or len(model_hash) != 64:
        raise ValueError("hmm_artifact_model_hash_invalid")
    if _canonical_hash(model) != model_hash:
        raise ValueError("hmm_artifact_model_hash_mismatch")
    dataset = _mapping(payload.get("dataset"), "dataset")
    if dataset.get("provider") != "Nobitex public UDF history API":
        raise ValueError("hmm_artifact_dataset_provider_invalid")
    if dataset.get("row_count") != 8760 or dataset.get("timeframe") != "1h":
        raise ValueError("hmm_artifact_dataset_scope_invalid")
    digest = dataset.get("csv_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("hmm_artifact_dataset_hash_invalid")
    validation = _mapping(payload.get("validation"), "validation")
    if validation.get("passed") is not True:
        raise ValueError("hmm_artifact_validation_not_passed")
    return _model_from_payload(model, model_hash)


def load_hmm_model_file(path: str | Path) -> RegimeHMMModel:
    """Load an immutable, auditable model artifact from a configured path."""
    model_path = Path(path)
    try:
        document = model_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError("hmm_artifact_unavailable") from error
    return load_hmm_model_document(document)
