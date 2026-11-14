"""Tests for the deterministic mock-only four-arm infrastructure."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Callable

import pytest

from research.ai_ensemble.four_arm_execution_lock import (
    EXECUTION_AUTHORIZATION_ENV,
    LOCK_SCHEMA_VERSION,
    PREHACK_DATASET_SHA256,
    ExecutionAuthorizationError,
    load_owner_authorized_execution_lock,
)
from research.ai_ensemble.four_arm_gates import (
    ArmGateEvidence,
    TRIAL_IDS,
    arm_passes,
    dsr_sensitivity_passes,
    fold_stability_passes,
    select_winner,
)
from research.ai_ensemble.four_arm_mock import (
    build_synthetic_ohlcv,
    run_mock_validation,
)


def _passing_evidence(
    arm_id: str = TRIAL_IDS[0],
    dsr: float = 0.1,
) -> ArmGateEvidence:
    return ArmGateEvidence(
        arm_id=arm_id,
        net_profit_factor_50bps=1.10,
        net_profit_factor_90bps=1.10,
        completed_trades=100,
        max_drawdown=0.15,
        dsr=dsr,
        dsr_sensitivity=(0.001, 0.02),
        fold_passes=(True, True, True),
    )


def _valid_execution_lock() -> dict[str, object]:
    digest = "a" * 64
    model_lock: dict[str, object] = {
        "repository": "owner/immutable-model",
        "revision": "b" * 40,
        "weights_sha256": digest,
    }
    tokenizer_lock: dict[str, object] = {
        "repository": "owner/immutable-tokenizer",
        "revision": "d" * 40,
        "weights_sha256": digest,
    }
    return {
        "schema_version": LOCK_SCHEMA_VERSION,
        "authorization_status": "OWNER_AUTHORIZED_PREHACK_RESEARCH_ONLY",
        "execution_scope": "PREHACK_RESEARCH_ONLY",
        "owner_authorization_id": "OWNER-2026-11-14-FOURARM",
        "owner_authorization_commit_sha": "c" * 40,
        "dataset_sha256": PREHACK_DATASET_SHA256,
        "deterministic_seed": 42,
        "models": {
            "kronos_small": {**model_lock, "tokenizer": tokenizer_lock},
            "chronos_2": model_lock,
        },
        "contracts": {
            "kronos_adapter_sha256": digest,
            "chronos_adapter_sha256": digest,
            "combined_rule_sha256": digest,
            "fold_plan_sha256": digest,
            "triple_barrier_sha256": digest,
            "cost_fill_simulator_sha256": digest,
            "dsr_method_sha256": digest,
            "license_and_corpus_disclosure_sha256": digest,
            "runtime_lock_sha256": digest,
        },
    }


def test_synthetic_ohlcv_fixture_is_deterministic_and_well_formed() -> None:
    first = build_synthetic_ohlcv(seed=71, bar_count=64)
    second = build_synthetic_ohlcv(seed=71, bar_count=64)

    assert first == second
    assert len(first) == 64
    assert first[0].timestamp == 0
    assert first[-1].timestamp == 63 * 3600
    assert all(
        bar.low <= min(bar.open, bar.close)
        and bar.high >= max(bar.open, bar.close)
        and bar.volume > 0
        for bar in first
    )


def test_synthetic_ohlcv_changes_when_seed_changes() -> None:
    assert build_synthetic_ohlcv(seed=71, bar_count=16) != build_synthetic_ohlcv(
        seed=72, bar_count=16
    )


@pytest.mark.parametrize(
    ("folds", "expected"),
    [
        ((True, True, True), True),
        ((True, True, False), False),
        ((True, False, False), False),
        ((), False),
    ],
)
def test_fold_stability_has_pass_and_fail_boundaries(
    folds: tuple[bool, ...],
    expected: bool,
) -> None:
    assert fold_stability_passes(folds) is expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ((0.001, 0.02, 0.03), True),
        ((0.001, 0.0, 0.03), False),
        ((0.001, -0.01, 0.03), False),
        ((), False),
    ],
)
def test_dsr_sensitivity_has_pass_and_fail_boundaries(
    values: tuple[float, ...],
    expected: bool,
) -> None:
    assert dsr_sensitivity_passes(values) is expected


def test_arm_gate_accepts_exact_preregistered_numeric_boundaries() -> None:
    assert arm_passes(_passing_evidence())


@pytest.mark.parametrize("arm_id", TRIAL_IDS)
def test_every_preregistered_arm_uses_the_same_gate_contract(
    arm_id: str,
) -> None:
    assert arm_passes(_passing_evidence(arm_id))


@pytest.mark.parametrize(
    "mutator",
    [
        lambda item: replace(item, net_profit_factor_50bps=1.0999),
        lambda item: replace(item, net_profit_factor_90bps=1.0999),
        lambda item: replace(item, completed_trades=99),
        lambda item: replace(item, max_drawdown=0.150001),
        lambda item: replace(item, dsr=0.0),
        lambda item: replace(item, dsr_sensitivity=(0.01, 0.0)),
        lambda item: replace(item, fold_passes=(True, True, False)),
        lambda item: replace(item, completed_trades=True),
    ],
)
def test_arm_gate_rejects_each_failed_boundary(
    mutator: Callable[[ArmGateEvidence], ArmGateEvidence],
) -> None:
    assert not arm_passes(mutator(_passing_evidence()))


def test_winner_selection_uses_highest_dsr_among_passing_arms() -> None:
    first = _passing_evidence(TRIAL_IDS[0], dsr=0.10)
    second = _passing_evidence(TRIAL_IDS[1], dsr=0.25)
    failing = replace(
        _passing_evidence(TRIAL_IDS[2], dsr=0.90),
        net_profit_factor_90bps=0.99,
    )

    assert select_winner((first, second, failing)) == TRIAL_IDS[1]


def test_winner_selection_fails_closed_without_a_passing_arm() -> None:
    failing = replace(_passing_evidence(), dsr=0.0)
    assert select_winner((failing,)) is None


def test_winner_selection_rejects_duplicate_trial_ids() -> None:
    evidence = _passing_evidence()
    assert select_winner((evidence, evidence)) is None


def test_execution_lock_checks_owner_flag_before_reading_file(
    tmp_path: Path,
) -> None:
    missing_lock = tmp_path / "must-not-be-read.json"
    with pytest.raises(
        ExecutionAuthorizationError,
        match="execution_authorization_missing",
    ):
        load_owner_authorized_execution_lock(missing_lock, {})


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        (
            {
                EXECUTION_AUTHORIZATION_ENV: "true",
                "ALIX_DRY_RUN": "false",
                "ALIX_ALLOW_LIVE_TRADING": "false",
            },
            "dry_run_invariant_not_satisfied",
        ),
        (
            {
                EXECUTION_AUTHORIZATION_ENV: "true",
                "ALIX_DRY_RUN": "true",
                "ALIX_ALLOW_LIVE_TRADING": "true",
            },
            "live_trading_must_remain_disabled",
        ),
    ],
)
def test_execution_lock_rejects_unsafe_environment_before_file_read(
    tmp_path: Path,
    environment: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(ExecutionAuthorizationError, match=message):
        load_owner_authorized_execution_lock(
            tmp_path / "must-not-be-read.json",
            environment,
        )


def test_execution_lock_accepts_complete_owner_authorized_contract(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "execution-lock.json"
    lock = _valid_execution_lock()
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    environment = {
        EXECUTION_AUTHORIZATION_ENV: "true",
        "ALIX_DRY_RUN": "true",
        "ALIX_ALLOW_LIVE_TRADING": "false",
    }

    assert load_owner_authorized_execution_lock(lock_path, environment) == lock


def test_execution_lock_rejects_default_unapproved_template(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "execution-lock.json"
    lock = _valid_execution_lock()
    lock["authorization_status"] = "NOT_AUTHORIZED"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    environment = {
        EXECUTION_AUTHORIZATION_ENV: "true",
        "ALIX_DRY_RUN": "true",
        "ALIX_ALLOW_LIVE_TRADING": "false",
    }

    with pytest.raises(
        ExecutionAuthorizationError,
        match="execution_lock_incomplete_or_unauthorized",
    ):
        load_owner_authorized_execution_lock(lock_path, environment)


def test_mock_report_contains_only_synthetic_metadata(
    tmp_path: Path,
) -> None:
    report = run_mock_validation(tmp_path, seed=91, bar_count=32)
    stored = json.loads(
        (tmp_path / "four-arm-mock-report.json").read_text(encoding="utf-8")
    )

    assert report == stored
    assert report["mode"] == "MOCK_ONLY"
    assert report["gate_contract_checks_passed"] is True
    assert report["economic_evaluation_performed"] is False
    assert report["real_market_data_accessed"] is False
    assert report["holdout_accessed"] is False
    assert report["model_weights_loaded"] is False
    assert report["training_or_fit_performed"] is False
    assert report["backtest_performed"] is False
    assert report["network_requests"] == 0
    assert report["private_api_calls"] == 0
    assert report["real_order_attempts"] == 0
    assert report["output_contains_ohlcv_values"] is False
    assert "open" not in stored
    assert "high" not in stored
    assert "low" not in stored
    assert "close" not in stored
    assert "volume" not in stored


def test_committed_execution_lock_records_a_bounded_owner_state() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    lock_path = (
        repository_root / "config" / "experiment" / "four_arm_execution_lock.json"
    )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))

    assert lock["schema_version"] == LOCK_SCHEMA_VERSION
    assert lock["execution_scope"] == "PREHACK_RESEARCH_ONLY"
    assert lock["owner_authorization_id"] == "OWNER-2026-11-14-FOURARM"
    assert lock["dataset_sha256"] == PREHACK_DATASET_SHA256
    assert lock["deterministic_seed"] == 42
    assert isinstance(lock["models"]["kronos_small"]["tokenizer"], dict)

    status = lock["authorization_status"]
    assert status in {
        "NOT_AUTHORIZED",
        "OWNER_AUTHORIZED_PREHACK_RESEARCH_ONLY",
    }
    if status == "NOT_AUTHORIZED":
        assert lock["owner_authorization_commit_sha"] is None
    else:
        assert len(lock["owner_authorization_commit_sha"]) == 40
        assert all(
            isinstance(value, str) and len(value) == 64
            for value in lock["contracts"].values()
        )




def test_four_arm_workflow_is_dispatch_only_and_mock_only() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    workflow_path = (
        repository_root / ".github" / "workflows" / "four-arm-experiment.yml"
    )
    workflow = workflow_path.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "\n  push:" not in workflow
    assert "\n  pull_request:" not in workflow
    assert 'ALIX_FOUR_ARM_EXECUTION_AUTHORIZED: "false"' in workflow
    assert 'ALIX_DRY_RUN: "true"' in workflow
    assert 'ALIX_ALLOW_LIVE_TRADING: "false"' in workflow
    assert "python -m research.ai_ensemble.four_arm_mock" in workflow
    assert "fetch_nobitex" not in workflow
    assert "four_arm_execution_lock.json" not in workflow
    assert "--mode research" not in workflow
