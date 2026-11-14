"""Tests for the real four-arm runner's pure and synthetic-only contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.ai_ensemble.four_arm_execution_lock import ExecutionAuthorizationError
from scripts.run_four_arm_experiment import (
    build_walk_forward_folds,
    run_real_experiment,
    run_synthetic_mode,
)


def test_walk_forward_folds_are_three_expanding_chronological_windows() -> None:
    folds = build_walk_forward_folds(31_801)

    assert len(folds) == 3
    assert [fold.prefix_end for fold in folds] == [22_260, 27_030, 31_801]
    assert [fold.fold_id for fold in folds] == [1, 2, 3]
    assert all(fold.train_start == 0 for fold in folds)
    assert all(
        fold.train_end
        < fold.validation_start
        < fold.validation_end
        < fold.test_start
        < fold.test_end
        for fold in folds
    )
    assert folds[0].test_end < folds[1].test_end < folds[2].test_end


@pytest.mark.parametrize("row_count", [0, 1, 999, True])
def test_walk_forward_rejects_invalid_row_counts(row_count: int) -> None:
    with pytest.raises(ValueError, match="walk_forward_row_count_invalid"):
        build_walk_forward_folds(row_count)


def test_synthetic_mode_writes_four_mock_manifests_without_economic_results(
    tmp_path: Path,
) -> None:
    report = run_synthetic_mode(tmp_path)

    assert report["mode"] == "MOCK_ONLY"
    assert report["arm_count"] == 4
    assert report["economic_evaluation_performed"] is False
    for arm_id in ("A", "B", "C", "D"):
        path = tmp_path / f"arm-{arm_id}.json"
        assert path.is_file()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["mode"] == "MOCK_ONLY"
        assert manifest["economic_evaluation_performed"] is False
        assert manifest["metrics"] is None
        assert manifest["dsr"] is None
        assert manifest["winner"] is False
    assert (tmp_path / "four-arm-run-report.json").is_file()


def test_synthetic_mode_does_not_create_real_economic_metrics(tmp_path: Path) -> None:
    report = run_synthetic_mode(tmp_path)
    assert "dataset_sha256" not in report
    assert "netPF_50bps" not in json.dumps(report)
    assert report["result"] == "MOCK_VALIDATION_ONLY_NOT_AN_ECONOMIC_RESULT"


def test_contract_mapping_is_fixed_to_repository_relative_files() -> None:
    from scripts.run_four_arm_experiment import CONTRACT_PATHS

    assert set(CONTRACT_PATHS) == {
        "kronos_adapter_sha256",
        "chronos_adapter_sha256",
        "combined_rule_sha256",
        "fold_plan_sha256",
        "triple_barrier_sha256",
        "cost_fill_simulator_sha256",
        "dsr_method_sha256",
        "license_and_corpus_disclosure_sha256",
        "runtime_lock_sha256",
    }



def test_real_runner_checks_authorization_before_dataset_or_model_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALIX_FOUR_ARM_EXECUTION_AUTHORIZED", "false")
    monkeypatch.setenv("ALIX_DRY_RUN", "true")
    monkeypatch.setenv("ALIX_ALLOW_LIVE_TRADING", "false")

    with pytest.raises(ExecutionAuthorizationError, match="execution_authorization_missing"):
        run_real_experiment(
            csv_path=tmp_path / "must-not-be-read.csv",
            data_manifest_path=tmp_path / "must-not-be-read.json",
            lock_path=tmp_path / "must-not-be-read-lock.json",
            model_pin_path=tmp_path / "must-not-be-read-pins.json",
            kronos_code_dir=tmp_path / "must-not-be-loaded",
            output_dir=tmp_path / "output",
            repository_root=tmp_path,
            validate_lock_only=False,
        )
