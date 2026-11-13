from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_four_arm_experiment as experiment


CONFIG = Path("config/experiment/four_arm_kronos_chronos.json")
PREREGISTRATION = Path("docs/research/preregistration/FOUR_ARM_KRONOS_CHRONOS.md")


def test_mock_run_writes_four_synthetic_manifests(tmp_path: Path) -> None:
    config = experiment.load_contract(CONFIG, PREREGISTRATION)
    summary = experiment.run_mock(config, tmp_path, "test-sha")

    assert summary["mode"] == "MOCK_ONLY"
    assert summary["winner"] is None
    assert summary["economic_verdict"] == "NOT_EVALUATED"
    assert summary["economic_evaluation_performed"] is False
    assert summary["safety"]["Champion"] is None
    assert summary["safety"]["order_permission"] is False
    assert summary["safety"]["private_api_calls"] == 0
    assert summary["safety"]["real_order_attempts"] == 0

    for arm_id in ("A", "B", "C", "D"):
        path = tmp_path / f"arm-{arm_id}.manifest.json"
        assert path.is_file()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["arm_id"] == arm_id
        assert manifest["trial_id"] == f"NOBITEX-KC26-{arm_id}"
        assert manifest["dataset"]["kind"] == "synthetic"
        assert manifest["dataset"]["market_data_read"] is False
        assert manifest["trades"] == 120
        assert len(manifest["fold_results"]) == 3
        assert manifest["DSR"] is None
        assert manifest["winner"] is False
        assert manifest["gates_evaluated"] is False
        assert manifest["economic_evaluation_performed"] is False
    assert (tmp_path / "run-log.jsonl").is_file()
    assert (tmp_path / "run-summary.json").is_file()


def test_mock_metrics_are_deterministic_and_separate_costs() -> None:
    gross = experiment._fixture_returns(0, 0)
    metrics = experiment._metrics(gross)
    assert metrics == experiment._metrics(gross)
    assert metrics["netPF_50bps"] is not None
    assert metrics["netPF_90bps"] is not None
    assert metrics["netPF_50bps"] != metrics["netPF_90bps"]


def test_config_rejects_wrong_prehack_sha(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["pre_hack_sha"] = "f" * 64
    bad_config = tmp_path / "config.json"
    bad_config.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(experiment.ExperimentError, match="pre_hack_sha_mismatch"):
        experiment.load_contract(bad_config, PREREGISTRATION)


def test_research_preflight_fails_closed_without_lock(tmp_path: Path) -> None:
    with pytest.raises(experiment.ExperimentError, match="execution_lock_missing"):
        experiment.validate_execution_lock(tmp_path / "missing-lock.json")


def test_research_mode_checks_lock_before_opening_dataset(tmp_path: Path) -> None:
    config = experiment.load_contract(CONFIG, PREREGISTRATION)
    with pytest.raises(experiment.ExperimentError, match="execution_lock_missing"):
        experiment.run_research(
            config=config,
            execution_lock=tmp_path / "missing-lock.json",
            ohlcv_path=tmp_path / "must-not-be-opened.csv",
            manifest_path=tmp_path / "must-not-be-opened.json",
            output_dir=tmp_path / "unused",
        )
    assert not (tmp_path / "unused").exists()
