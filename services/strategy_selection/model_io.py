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
APPROVAL_SCHEMA_VERSION = "alix.model-technical-approval.v1"


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
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise ValueError(f"hmm_artifact_{field}_invalid")
    return tuple(value)


def _decimal_list(value: object, field: str) -> tuple[Decimal, ...]:
    if not isinstance(value, list):
        raise ValueError(f"hmm_artifact_{field}_invalid")
    return tuple(_decimal(item, field) for item in value)


def _parse_emissions(value: object) -> tuple[GaussianRegimeEmission, ...]:
    if not isinstance(value, list):
        raise ValueError("hmm_artifact_emissions_invalid")
    result: list[GaussianRegimeEmission] = []
    for item in value:
        emission = _mapping(item, "emission")
        result.append(
            GaussianRegimeEmission(
                regime=str(emission.get("regime", "")),
                means=_decimal_list(emission.get("means"), "means"),
                standard_deviations=_decimal_list(
                    emission.get("standard_deviations"), "standard_deviations"
                ),
            )
        )
    return tuple(result)


def _parse_transition_matrix(value: object) -> tuple[tuple[Decimal, ...], ...]:
    if not isinstance(value, list):
        raise ValueError("hmm_artifact_transition_matrix_invalid")
    return tuple(_decimal_list(row, "transition_row") for row in value)


def _parse_model_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("hmm_artifact_trained_at_invalid")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("hmm_artifact_trained_at_invalid") from error


def _validate_model_identity(values: Mapping[str, object]) -> None:
    if not isinstance(values["revision"], str) or not isinstance(
        values["feature_schema_version"], str
    ):
        raise ValueError("hmm_artifact_identity_invalid")
    if not isinstance(values["owner_approved"], bool):
        raise ValueError("hmm_artifact_owner_approval_invalid")


def _add_model_age_limits(model: Mapping[str, Any], values: dict[str, object]) -> None:
    for field in ("max_model_age_seconds", "max_observation_age_seconds"):
        value = model.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"hmm_artifact_{field}_invalid")
        values[field] = value


def _model_from_payload(model: Mapping[str, Any], model_hash: str) -> RegimeHMMModel:
    values: dict[str, object] = {
        "revision": model.get("revision"),
        "model_sha256": model_hash,
        "trained_at": _parse_model_timestamp(model.get("trained_at")),
        "owner_approved": model.get("owner_approved"),
        "feature_schema_version": model.get("feature_schema_version"),
        "feature_names": _string_list(model.get("feature_names"), "feature_names"),
        "regimes": _string_list(model.get("regimes"), "regimes"),
        "initial_probabilities": _decimal_list(
            model.get("initial_probabilities"), "initial_probabilities"
        ),
        "transition_matrix": _parse_transition_matrix(model.get("transition_matrix")),
        "emissions": _parse_emissions(model.get("emissions")),
        "minimum_confidence": _decimal(
            model.get("minimum_confidence"), "minimum_confidence"
        ),
        "minimum_posterior_margin": _decimal(
            model.get("minimum_posterior_margin"), "minimum_posterior_margin"
        ),
    }
    _add_model_age_limits(model, values)
    _validate_model_identity(values)
    return RegimeHMMModel(**values)  # type: ignore[arg-type]


def _decode_approval(document: str) -> Mapping[str, Any]:
    try:
        decoded = json.loads(document)
    except json.JSONDecodeError as error:
        raise ValueError("model_approval_json_invalid") from error
    return _mapping(decoded, "approval")


def _validate_approval_identity(
    approval: Mapping[str, Any],
    *,
    model_type: str,
    model_hash: str,
    dataset_hash: str,
) -> None:
    if approval.get("schema_version") != APPROVAL_SCHEMA_VERSION:
        raise ValueError("model_approval_schema_version_mismatch")
    if approval.get("model_type") != model_type:
        raise ValueError("model_approval_type_mismatch")
    if approval.get("status") != "OWNER_APPROVED_TECHNICAL_ONLY":
        raise ValueError("model_approval_not_approved")
    if approval.get("model_sha256") != model_hash:
        raise ValueError("model_approval_hash_mismatch")
    if approval.get("dataset_sha256") != dataset_hash:
        raise ValueError("model_approval_dataset_hash_mismatch")
    if approval.get("owner") != "angry0tiger-sudo":
        raise ValueError("model_approval_owner_mismatch")
    approval_id = approval.get("approval_id")
    if not isinstance(approval_id, str) or not approval_id.strip():
        raise ValueError("model_approval_id_missing")


def _validate_approval_timestamp(approval: Mapping[str, Any]) -> None:
    approved_at = approval.get("approved_at")
    if not isinstance(approved_at, str):
        raise ValueError("model_approval_timestamp_invalid")
    try:
        timestamp = datetime.fromisoformat(approved_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("model_approval_timestamp_invalid") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("model_approval_timestamp_must_be_aware")


def _validate_approval_release_flags(approval: Mapping[str, Any]) -> None:
    if (
        approval.get("economic_acceptance") != "NO-GO"
        or approval.get("order_permission") is not False
        or approval.get("trading_authorization") != "NO-GO"
    ):
        raise ValueError("model_approval_release_flags_invalid")


def _validate_technical_approval(
    document: str, *, model_type: str, model_hash: str, dataset_hash: str
) -> None:
    """Require a separate review-controlled approval record for technical inference."""
    approval = _decode_approval(document)
    _validate_approval_identity(
        approval,
        model_type=model_type,
        model_hash=model_hash,
        dataset_hash=dataset_hash,
    )
    _validate_approval_timestamp(approval)
    _validate_approval_release_flags(approval)


def _validate_artifact_identity(payload: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    if payload.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError("hmm_artifact_schema_version_mismatch")
    status = payload.get("status")
    if status not in {"TRAINED_UNAPPROVED", "OWNER_APPROVED_TECHNICAL_ONLY"}:
        raise ValueError("hmm_artifact_status_not_loadable")
    if payload.get("economic_acceptance") != "NO-GO":
        raise ValueError("hmm_artifact_economic_gate_must_remain_no_go")
    if payload.get("trading_authorization") != "NO-GO":
        raise ValueError("hmm_artifact_trading_authorization_must_remain_no_go")
    return str(status), _mapping(payload.get("model"), "model")


def _validate_artifact_model_hash(
    payload: Mapping[str, Any], model: Mapping[str, Any], status: str
) -> str:
    model_hash = payload.get("model_sha256")
    if not isinstance(model_hash, str) or len(model_hash) != 64:
        raise ValueError("hmm_artifact_model_hash_invalid")
    if _canonical_hash(model) != model_hash:
        raise ValueError("hmm_artifact_model_hash_mismatch")
    if model.get("owner_approved") is not (status == "OWNER_APPROVED_TECHNICAL_ONLY"):
        raise ValueError("hmm_artifact_approval_status_mismatch")
    return model_hash


def _validate_dataset_provenance(payload: Mapping[str, Any]) -> str:
    dataset = _mapping(payload.get("dataset"), "dataset")
    if dataset.get("provider") != "Nobitex public UDF history API":
        raise ValueError("hmm_artifact_dataset_provider_invalid")
    if dataset.get("row_count") != 8760 or dataset.get("timeframe") != "1h":
        raise ValueError("hmm_artifact_dataset_scope_invalid")
    digest = dataset.get("csv_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("hmm_artifact_dataset_hash_invalid")
    return digest


def _validate_artifact_validation(payload: Mapping[str, Any]) -> None:
    validation = _mapping(payload.get("validation"), "validation")
    if validation.get("passed") is not True:
        raise ValueError("hmm_artifact_validation_not_passed")


def _validate_unapproved_document(document: str) -> None:
    approval = _decode_approval(document)
    if (
        approval.get("schema_version") != APPROVAL_SCHEMA_VERSION
        or approval.get("model_type") != "gaussian_hmm"
        or approval.get("status") != "NOT_APPROVED"
        or approval.get("economic_acceptance") != "NO-GO"
        or approval.get("order_permission") is not False
        or approval.get("trading_authorization") != "NO-GO"
    ):
        raise ValueError("hmm_artifact_unapproved_but_approval_supplied")


def _validate_artifact_approval(
    status: str,
    approval_document: str | None,
    model_hash: str,
    dataset_hash: str,
) -> None:
    if status == "OWNER_APPROVED_TECHNICAL_ONLY":
        if approval_document is None:
            raise ValueError("hmm_artifact_separate_approval_required")
        _validate_technical_approval(
            approval_document,
            model_type="gaussian_hmm",
            model_hash=model_hash,
            dataset_hash=dataset_hash,
        )
    elif approval_document is not None:
        _validate_unapproved_document(approval_document)


def load_hmm_model_document(
    document: str, *, approval_document: str | None = None
) -> RegimeHMMModel:
    """Parse and verify an HMM artifact; invalid or mutated weights are rejected."""
    try:
        decoded = json.loads(document)
    except json.JSONDecodeError as error:
        raise ValueError("hmm_artifact_json_invalid") from error
    payload = _mapping(decoded, "root")
    status, model = _validate_artifact_identity(payload)
    model_hash = _validate_artifact_model_hash(payload, model, status)
    dataset_hash = _validate_dataset_provenance(payload)
    _validate_artifact_validation(payload)
    _validate_artifact_approval(status, approval_document, model_hash, dataset_hash)
    return _model_from_payload(model, model_hash)


def load_hmm_model_file(
    path: str | Path, *, approval_path: str | Path | None = None
) -> RegimeHMMModel:
    """Load an immutable, auditable model artifact from a configured path."""
    model_path = Path(path)
    try:
        document = model_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError("hmm_artifact_unavailable") from error
    approval_document: str | None = None
    if approval_path is not None:
        try:
            approval_document = Path(approval_path).read_text(encoding="utf-8")
        except OSError as error:
            raise ValueError("hmm_artifact_approval_unavailable") from error
    return load_hmm_model_document(document, approval_document=approval_document)
