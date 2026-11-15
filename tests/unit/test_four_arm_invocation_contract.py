"""Subprocess contracts for the exact four-arm module invocation in Actions."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scripts import run_four_arm_experiment as runner


def _authorized_lock_payload(root: Path) -> dict[str, Any]:
    pins: dict[str, Any] = json.loads(
        (root / "config/four_arm/model_pins.json").read_text(encoding="utf-8")
    )
    models = pins["models"]
    contracts = {
        key: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for key, path in runner.CONTRACT_PATHS.items()
    }
    kronos = models["kronos_small"]
    tokenizer = kronos["tokenizer"]
    chronos = models["chronos_2"]
    return {
        "schema_version": "alix.four-arm.execution-lock.v1",
        "authorization_status": "OWNER_AUTHORIZED_PREHACK_RESEARCH_ONLY",
        "execution_scope": "PREHACK_RESEARCH_ONLY",
        "owner_authorization_id": "OWNER-2026-11-14-FOURARM",
        "owner_authorization_commit_sha": "a" * 40,
        "dataset_sha256": runner.PREHACK_DATASET_SHA256,
        "deterministic_seed": 42,
        "models": {
            "kronos_small": {
                "repository": kronos["repository"],
                "revision": kronos["revision"],
                "weights_sha256": kronos["weights_sha256"],
                "tokenizer": {
                    "repository": tokenizer["repository"],
                    "revision": tokenizer["revision"],
                    "weights_sha256": tokenizer["weights_sha256"],
                },
            },
            "chronos_2": {
                "repository": chronos["repository"],
                "revision": chronos["revision"],
                "weights_sha256": chronos["weights_sha256"],
            },
        },
        "contracts": contracts,
    }


def _minimal_environment(root: Path, extra_path: Path | None = None) -> dict[str, str]:
    paths = [str(root)]
    if extra_path is not None:
        paths.append(str(extra_path))
    environment = {
        "PYTHONPATH": os.pathsep.join(paths),
        "ALIX_FOUR_ARM_EXECUTION_AUTHORIZED": "true",
        "ALIX_DRY_RUN": "true",
        "ALIX_ALLOW_LIVE_TRADING": "false",
    }
    if "SYSTEMROOT" in os.environ:
        environment["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    return environment


def test_workflow_uses_one_module_entrypoint_and_real_smoke_before_trials() -> None:
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/four-arm-real-experiment.yml").read_text(
        encoding="utf-8"
    )
    assert "python scripts/run_four_arm_experiment.py" not in workflow
    assert workflow.count("python -m scripts.run_four_arm_experiment") == 3
    assert "python -S -m scripts.run_four_arm_experiment" in workflow
    assert "--dependency-import-preflight" in workflow
    assert "--smoke" in workflow
    assert "PYTHONPATH: ${{ github.workspace }}" in workflow
    assert workflow.index("--smoke") < workflow.index(
        "Run all twelve preregistered trials"
    )
    assert "timeout-minutes: 20" in workflow


def test_lock_only_workflow_command_reaches_the_lock_gate_in_a_minimal_env(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    lock_path = tmp_path / "authorized-lock.json"
    lock_path.write_text(
        json.dumps(_authorized_lock_payload(root)),
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-S",
        "-m",
        "scripts.run_four_arm_experiment",
        "--mode",
        "real",
        "--validate-lock-only",
        "--lock",
        str(lock_path),
        "--model-pins",
        str(root / "config/four_arm/model_pins.json"),
        "--output-dir",
        str(tmp_path / "lock-output"),
        "--repository-root",
        str(root),
    ]
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=_minimal_environment(root),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FOUR_ARM_EXECUTION_LOCK=PASS" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr


def test_dependency_preflight_imports_runner_and_all_pinned_modules_in_minimal_env(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    lock_path = tmp_path / "authorized-lock.json"
    lock_path.write_text(
        json.dumps(_authorized_lock_payload(root)),
        encoding="utf-8",
    )
    fake_modules = tmp_path / "fake-modules"
    fake_modules.mkdir()
    for module_name in (
        "numpy",
        "pandas",
        "torch",
        "transformers",
        "safetensors",
        "huggingface_hub",
        "einops",
        "accelerate",
        "tqdm",
    ):
        (fake_modules / f"{module_name}.py").write_text("", encoding="utf-8")
    (fake_modules / "chronos").mkdir()
    (fake_modules / "chronos" / "__init__.py").write_text(
        "class Chronos2Pipeline: pass\n",
        encoding="utf-8",
    )
    kronos_source = tmp_path / "kronos-source"
    (kronos_source / "model").mkdir(parents=True)
    (kronos_source / "model" / "__init__.py").write_text(
        "class Kronos: pass\n"
        "class KronosPredictor: pass\n"
        "class KronosTokenizer: pass\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-m",
        "scripts.run_four_arm_experiment",
        "--mode",
        "real",
        "--dependency-import-preflight",
        "--lock",
        str(lock_path),
        "--model-pins",
        str(root / "config/four_arm/model_pins.json"),
        "--kronos-code-dir",
        str(kronos_source),
        "--output-dir",
        str(tmp_path / "import-output"),
        "--repository-root",
        str(root),
    ]
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=_minimal_environment(root, fake_modules),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FOUR_ARM_DEPENDENCY_IMPORT_PREFLIGHT=PASS" in result.stdout
    assert "FOUR_ARM_RUNNER_MODULE_IMPORT=PASS" in result.stdout
    report = json.loads(
        (tmp_path / "import-output" / "dependency-import-preflight.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["runner_module_imported"] is True
    assert report["runner_module_name"] == "scripts.run_four_arm_experiment"
    assert report["module_count"] == 11
    assert report["data_accessed"] is False
    assert report["model_weights_loaded"] is False


def test_real_smoke_report_is_explicitly_non_registry_and_non_economic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[2]
    lock_path = tmp_path / "authorized-lock.json"
    lock_path.write_text(
        json.dumps(_authorized_lock_payload(root)),
        encoding="utf-8",
    )
    monkeypatch.setenv("ALIX_FOUR_ARM_EXECUTION_AUTHORIZED", "true")
    monkeypatch.setenv("ALIX_DRY_RUN", "true")
    monkeypatch.setenv("ALIX_ALLOW_LIVE_TRADING", "false")

    row_count = runner.PREHACK_ROW_COUNT
    close = np.linspace(100.0, 120.0, row_count, dtype=np.float64)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2022-01-01T00:30:00Z", periods=row_count, freq="h"
            ),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.ones(row_count, dtype=np.float64),
        }
    )
    monkeypatch.setattr(
        runner,
        "_load_verified_prehack_data",
        lambda *_args: (frame, runner.PREHACK_DATASET_SHA256),
    )
    monkeypatch.setattr(
        runner,
        "_load_cpu_models",
        lambda *_args: (object(), object()),
    )
    cache = runner.ForecastCache(
        anchors=(511, 535, 559),
        forecasts=tuple(runner.Forecast(0.02, 0.03) for _ in range(3)),
        identity_sha256="a" * 64,
        cache_sha256="b" * 64,
        entries_json="[]",
    )
    monkeypatch.setattr(
        runner,
        "_build_forecast_cache",
        lambda *_args: cache,
    )
    report = runner.run_real_smoke(
        csv_path=tmp_path / runner.PREHACK_FILENAME,
        data_manifest_path=tmp_path / "manifest.json",
        lock_path=lock_path,
        model_pin_path=root / "config/four_arm/model_pins.json",
        kronos_code_dir=tmp_path / "kronos-source",
        output_dir=tmp_path / "smoke-output",
        repository_root=root,
        model_artifacts_dir=tmp_path / "model-artifacts",
    )
    assert report["mode"] == "REAL_PREHACK_SMOKE"
    assert report["result"] == "SMOKE_PASS"
    assert report["smoke_rows"] == 600
    assert report["arm_id"] == "D"
    assert report["threshold_bps"] == 180
    assert report["fold_count"] == 1
    assert report["trial_registry_eligible"] is False
    assert report["economic_evaluation_performed"] is False
    assert report["trial_count"] == 0
    assert "trial_id" not in report
    assert report["forecast_cache"]["anchors"] == 3
    assert sorted(path.name for path in (tmp_path / "smoke-output").iterdir()) == [
        "smoke-report.json"
    ]
