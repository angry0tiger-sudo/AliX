"""Owner authorization contract for any future non-mock four-arm research path."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, cast

PREHACK_DATASET_SHA256 = (
    "0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e"
)
EXECUTION_AUTHORIZATION_ENV = "ALIX_FOUR_ARM_EXECUTION_AUTHORIZED"
LOCK_SCHEMA_VERSION = "alix.four-arm.execution-lock.v1"


class ExecutionAuthorizationError(RuntimeError):
    """Raised when an execution lock is absent, incomplete, or unauthorized."""


def _is_hex_digest(value: object, length: int) -> bool:
    if not isinstance(value, str) or len(value) != length:
        return False
    return all(character in "0123456789abcdefABCDEF" for character in value)


def _is_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _model_lock_is_complete(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        _is_non_empty_string(value.get("repository"))
        and _is_hex_digest(value.get("revision"), 40)
        and _is_hex_digest(value.get("weights_sha256"), 64)
    )


def _contracts_are_complete(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    required_digests = (
        "kronos_adapter_sha256",
        "chronos_adapter_sha256",
        "combined_rule_sha256",
        "fold_plan_sha256",
        "triple_barrier_sha256",
        "cost_fill_simulator_sha256",
        "dsr_method_sha256",
        "license_and_corpus_disclosure_sha256",
        "runtime_lock_sha256",
    )
    return all(_is_hex_digest(value.get(name), 64) for name in required_digests)


def _authorization_metadata_is_valid(payload: dict[str, object]) -> bool:
    expected_fields = (
        ("schema_version", LOCK_SCHEMA_VERSION),
        (
            "authorization_status",
            "OWNER_AUTHORIZED_PREHACK_RESEARCH_ONLY",
        ),
        ("execution_scope", "PREHACK_RESEARCH_ONLY"),
        ("owner_authorization_id", "OWNER-2026-11-14-FOURARM"),
    )
    return all(
        payload.get(field) == expected
        for field, expected in expected_fields
    ) and _is_hex_digest(payload.get("owner_authorization_commit_sha"), 40)


def _dataset_contract_is_valid(payload: dict[str, object]) -> bool:
    return payload.get("dataset_sha256") == PREHACK_DATASET_SHA256


def _deterministic_seed_is_valid(payload: dict[str, object]) -> bool:
    seed = payload.get("deterministic_seed")
    return type(seed) is int and seed == 42


def _model_set_is_complete(payload: dict[str, object]) -> bool:
    models = payload.get("models")
    if not isinstance(models, dict):
        return False
    kronos = models.get("kronos_small")
    return (
        _model_lock_is_complete(kronos)
        and isinstance(kronos, dict)
        and _model_lock_is_complete(kronos.get("tokenizer"))
        and _model_lock_is_complete(models.get("chronos_2"))
    )


def _lock_is_complete(payload: dict[str, object]) -> bool:
    return all(
        (
            _authorization_metadata_is_valid(payload),
            _dataset_contract_is_valid(payload),
            _deterministic_seed_is_valid(payload),
            _model_set_is_complete(payload),
            _contracts_are_complete(payload.get("contracts")),
        )
    )


def load_owner_authorized_execution_lock(
    lock_path: Path,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Validate owner flags before reading the lock file or any research dataset."""
    environment = os.environ if environ is None else environ
    if environment.get(EXECUTION_AUTHORIZATION_ENV) != "true":
        raise ExecutionAuthorizationError("execution_authorization_missing")
    if environment.get("ALIX_DRY_RUN") != "true":
        raise ExecutionAuthorizationError("dry_run_invariant_not_satisfied")
    if environment.get("ALIX_ALLOW_LIVE_TRADING") != "false":
        raise ExecutionAuthorizationError("live_trading_must_remain_disabled")

    try:
        raw_payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ExecutionAuthorizationError("execution_lock_unreadable") from None

    if not isinstance(raw_payload, dict):
        raise ExecutionAuthorizationError("execution_lock_schema_invalid")
    payload = cast(dict[str, object], raw_payload)
    if not _lock_is_complete(payload):
        raise ExecutionAuthorizationError("execution_lock_incomplete_or_unauthorized")
    return payload
