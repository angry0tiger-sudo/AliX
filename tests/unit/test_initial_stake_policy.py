"""Contract tests for the persisted initial-stake policy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from risk.initial_stake_policy import (
    POLICY_SCHEMA_VERSION,
    load_initial_stake_policy_document,
)


def test_versioned_policy_file_loads_explicit_owner_approval() -> None:
    document = Path("config/risk/initial-stake-policy.json").read_text(encoding="utf-8")
    result = load_initial_stake_policy_document(document)

    assert result.reason_codes == ()
    assert result.policy is not None
    assert result.policy.owner_approved
    assert result.policy.revision == "initial-stake-10-usdt-v1"
    assert result.policy.directive_id == "ALIX-OD-2026-10-23-INITIAL-STAKE-10-USDT"
    assert str(result.policy.target_usdt) == "10"
    assert result.policy.use_all_free_balance_below_target
    assert len(result.policy.fingerprint) == 64


@pytest.mark.parametrize(
    "document",
    (
        "{not-json",
        "[]",
        json.dumps(
            {
                "schema_version": "unsupported",
                "revision": "r1",
                "directive_id": "d1",
                "owner_approved": True,
                "target_usdt": "10",
                "use_all_free_balance_below_target": True,
            }
        ),
        json.dumps(
            {
                "schema_version": POLICY_SCHEMA_VERSION,
                "revision": "r1",
                "directive_id": "d1",
                "owner_approved": "true",
                "target_usdt": "10",
                "use_all_free_balance_below_target": True,
            }
        ),
        json.dumps(
            {
                "schema_version": POLICY_SCHEMA_VERSION,
                "revision": "r1",
                "directive_id": "d1",
                "owner_approved": True,
                "target_usdt": 10,
                "use_all_free_balance_below_target": True,
            }
        ),
        json.dumps(
            {
                "schema_version": POLICY_SCHEMA_VERSION,
                "revision": "r1",
                "directive_id": "d1",
                "owner_approved": True,
                "target_usdt": "0",
                "use_all_free_balance_below_target": True,
            }
        ),
    ),
)
def test_invalid_policy_documents_fail_closed_without_fallback(
    document: str,
) -> None:
    result = load_initial_stake_policy_document(document)
    assert result.policy is None
    assert result.reason_codes == ("initial_stake_policy_document_invalid",)
