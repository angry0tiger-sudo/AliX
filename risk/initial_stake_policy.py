"""Strict, fail-closed parser for the versioned owner initial-stake policy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

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
        raw = json.loads(document)
        if not isinstance(raw, dict) or set(raw) != _ROOT_KEYS:
            raise ValueError("initial_stake_policy_schema_invalid")
        if raw["schema_version"] != POLICY_SCHEMA_VERSION:
            raise ValueError("initial_stake_policy_schema_version_unsupported")
        if not isinstance(raw["revision"], str) or not raw["revision"].strip():
            raise ValueError("initial_stake_policy_revision_missing")
        if not isinstance(raw["directive_id"], str) or not raw["directive_id"].strip():
            raise ValueError("initial_stake_policy_directive_missing")
        if not isinstance(raw["owner_approved"], bool):
            raise ValueError("initial_stake_policy_approval_invalid")
        if not isinstance(raw["use_all_free_balance_below_target"], bool):
            raise ValueError("initial_stake_policy_balance_rule_invalid")
        if not isinstance(raw["target_usdt"], str):
            raise ValueError("initial_stake_policy_target_must_be_string")
        try:
            target = Decimal(raw["target_usdt"])
        except InvalidOperation as error:
            raise ValueError("initial_stake_policy_target_invalid") from error
        policy = InitialStakePolicy(
            revision=raw["revision"],
            directive_id=raw["directive_id"],
            owner_approved=raw["owner_approved"],
            target_usdt=target,
            use_all_free_balance_below_target=raw["use_all_free_balance_below_target"],
        )
    except (
        ValueError,
        TypeError,
        KeyError,
        InvalidOperation,
        json.JSONDecodeError,
    ):
        return InitialStakePolicyLoadResult(
            policy=None,
            reason_codes=("initial_stake_policy_document_invalid",),
        )
    return InitialStakePolicyLoadResult(policy=policy, reason_codes=())
