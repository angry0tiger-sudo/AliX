"""Strict, fail-closed parser for the versioned owner initial-stake policy."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import cast

from risk.contracts import InitialStakePolicy

POLICY_SCHEMA_VERSION = "alix.initial-stake-policy.v1"
_ROOT_KEYS = {
    "schema_version",
    "revision",
    "directive_id",
    "owner_approved",
    "target_usdt",
    "use_all_free_balance_below_target",
}


@dataclass(frozen=True, slots=True)
class InitialStakePolicyLoadResult:
    policy: InitialStakePolicy | None
    reason_codes: tuple[str, ...]


def load_initial_stake_policy_document(document: str) -> InitialStakePolicyLoadResult:
    """Malformed settings yield no policy; there is no approved fallback."""
    try:
        policy = _parse_policy_document(document)
    except (ValueError, TypeError, KeyError, InvalidOperation, json.JSONDecodeError):
        return InitialStakePolicyLoadResult(
            policy=None,
            reason_codes=("initial_stake_policy_document_invalid",),
        )
    return InitialStakePolicyLoadResult(policy=policy, reason_codes=())


def _parse_policy_document(document: str) -> InitialStakePolicy:
    raw = _decode_policy_document(document)
    return InitialStakePolicy(
        revision=_required_text(raw, "revision"),
        directive_id=_required_text(raw, "directive_id"),
        owner_approved=_required_bool(raw, "owner_approved"),
        target_usdt=_required_decimal(raw, "target_usdt"),
        use_all_free_balance_below_target=_required_bool(
            raw, "use_all_free_balance_below_target"
        ),
    )


def _decode_policy_document(document: str) -> Mapping[str, object]:
    raw: object = json.loads(document)
    if not isinstance(raw, dict) or set(raw) != _ROOT_KEYS:
        raise ValueError("initial_stake_policy_schema_invalid")
    values = cast(Mapping[str, object], raw)
    if values["schema_version"] != POLICY_SCHEMA_VERSION:
        raise ValueError("initial_stake_policy_schema_version_unsupported")
    return values


def _required_text(values: Mapping[str, object], key: str) -> str:
    value = values[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"initial_stake_policy_{key}_invalid")
    return value


def _required_bool(values: Mapping[str, object], key: str) -> bool:
    value = values[key]
    if not isinstance(value, bool):
        raise ValueError(f"initial_stake_policy_{key}_invalid")
    return value


def _required_decimal(values: Mapping[str, object], key: str) -> Decimal:
    value = values[key]
    if not isinstance(value, str):
        raise ValueError(f"initial_stake_policy_{key}_must_be_string")
    try:
        return Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"initial_stake_policy_{key}_invalid") from error
