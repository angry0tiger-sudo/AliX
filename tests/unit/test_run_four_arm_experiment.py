"""ADR-038 real-entrypoint, trial inventory, cache and authorization contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from research.four_arm_core.lock_validator import ExecutionAuthorizationError
from scripts import run_four_arm_experiment as four_arm


def _authorized_lock_payload(repository_root: Path) -> dict[str, object]:
    pins = json.loads(
        (repository_root / "config/four_arm/model_pins.json").read_text(
            encoding="utf-8"
        )
    )
    models = pins["models"]
    contracts = {
        key: hashlib.sha256((repository_root / path).read_bytes()).hexdigest()
        for key, path in four_arm.CONTRACT_PATHS.items()
    }
    return {
        "schema_version": "alix.four-arm.execution-lock.v1",
        "authorization_status": "OWNER_AUTHORIZED_PREHACK_RESEARCH_ONLY",
        "execution_scope": "PREHACK_RESEARCH_ONLY",
        "owner_authorization_id": "OWNER-2026-11-14-FOURARM",
        "owner_authorization_commit_sha": "a" * 40,
        "dataset_sha256": four_arm.PREHACK_DATASET_SHA256,
        "deterministic_seed": 42,
        "models": {
            "kronos_small": {
                "repository": models["kronos_small"]["repository"],
                "revision": models["kronos_small"]["revision"],
                "weights_sha256": models["kronos_small"]["weights_sha256"],
                "tokenizer": {
                    "repository": models["kronos_small"]["tokenizer"]["repository"],
                    "revision": models["kronos_small"]["tokenizer"]["revision"],
                    "weights_sha256": models["kronos_small"]["tokenizer"][
                        "weights_sha256"
                    ],
                },
            },
            "chronos_2": {
                "repository": models["chronos_2"]["repository"],
                "revision": models["chronos_2"]["revision"],
                "weights_sha256": models["chronos_2"]["weights_sha256"],
            },
        },
        "contracts": contracts,
    }


def test_confirmatory_prefix_and_folds_exclude_contaminated_tail() -> None:
    assert four_arm.PREHACK_ROW_COUNT == 31_801
    assert four_arm.CONFIRMATORY_ROW_COUNT == 27_825
    assert four_arm.EXCLUDED_TAIL_ROWS == 3_976
    folds = four_arm.build_walk_forward_folds(four_arm.CONFIRMATORY_ROW_COUNT)
    assert len(folds) == 3
    assert [fold.prefix_end for fold in folds] == [19_477, 23_651, 27_825]
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


def test_exactly_twelve_forecast_driven_trial_ids_are_registered() -> None:
    ids = [
        trial_id
        for threshold_map in four_arm.TRIAL_ID_MATRIX.values()
        for trial_id in threshold_map.values()
    ]
    assert len(ids) == 12
    assert len(set(ids)) == 12
    assert set(four_arm.THRESHOLDS_BPS) == {120, 180, 240}
    assert set(four_arm.TRIAL_ID_MATRIX[120]) == {"A", "B", "C", "D"}
    assert four_arm.TRIAL_ID_MATRIX[120]["A"] == "NOBITEX-KC26-FD-120-A"
    assert four_arm.TRIAL_ID_MATRIX[180]["D"] == "NOBITEX-KC26-FD-180-D"
    assert four_arm.TRIAL_ID_MATRIX[240]["B"] == "NOBITEX-KC26-FD-240-B"


def test_model_pins_require_nested_tokenizer_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    pins = json.loads(
        (root / "config/four_arm/model_pins.json").read_text(encoding="utf-8")
    )
    assert isinstance(pins["models"]["kronos_small"]["tokenizer"], dict)
    assert pins["models"]["kronos_small"]["tokenizer"]["repository"] == (
        "NeoQuasar/Kronos-Tokenizer-base"
    )


def test_forecast_cache_computes_once_and_is_reused_for_all_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def fake_forecast(
        frame: pd.DataFrame,
        anchor: int,
        kronos_predictor: object,
        chronos_pipeline: object,
    ) -> four_arm.Forecast:
        calls.append(anchor)
        return four_arm.Forecast(0.02, 0.03)

    monkeypatch.setattr(four_arm, "_forecast_at", fake_forecast)
    model_pins = {
        "models": {
            "kronos_small": {
                "repository": "NeoQuasar/Kronos-small",
                "revision": "1" * 40,
                "weights_sha256": "2" * 64,
                "tokenizer": {
                    "repository": "NeoQuasar/Kronos-Tokenizer-base",
                    "revision": "3" * 40,
                    "weights_sha256": "4" * 64,
                },
            },
            "chronos_2": {
                "repository": "autogluon/chronos-2-small",
                "revision": "5" * 40,
                "weights_sha256": "6" * 64,
            },
        }
    }
    lock = {"contracts": {key: "7" * 64 for key in four_arm.CONTRACT_PATHS}}
    frame = pd.DataFrame({"timestamp": range(640)})
    cache = four_arm._build_forecast_cache(
        frame, (object(), object()), model_pins, lock, "8" * 64
    )
    expected_anchors = [511, 535, 559, 583, 607]
    assert calls == expected_anchors
    assert cache.anchors == tuple(expected_anchors)
    assert len(cache.forecasts) == len(expected_anchors)

    first = four_arm._signals_from_forecast_cache(cache, 640, "B", 120)
    second = four_arm._signals_from_forecast_cache(cache, 640, "C", 180)
    third = four_arm._signals_from_forecast_cache(cache, 640, "D", 240)
    assert first[511:535].all()
    assert second[511:535].all()
    assert third[511:535].all()
    assert calls == expected_anchors


def test_forecast_arms_do_not_intersect_donchian_baseline() -> None:
    source = Path("scripts/run_four_arm_experiment.py").read_text(encoding="utf-8")
    assert "_build_model_filter_signals" not in source
    assert "baseline & filters" not in source
    assert "baseline &" not in source
    assert "_signals_from_forecast_cache(" in source


def test_synthetic_mode_remains_non_economic(tmp_path: Path) -> None:
    report = four_arm.run_synthetic_mode(tmp_path)
    assert report["mode"] == "MOCK_ONLY"
    assert report["arm_count"] == 4
    assert report["economic_evaluation_performed"] is False
    assert report["result"] == "MOCK_VALIDATION_ONLY_NOT_AN_ECONOMIC_RESULT"
    for arm_id in ("A", "B", "C", "D"):
        manifest = json.loads(
            (tmp_path / f"arm-{arm_id}.json").read_text(encoding="utf-8")
        )
        assert manifest["metrics"] is None
        assert manifest["dsr"] is None
        assert manifest["winner"] is False


def test_real_entrypoint_rejects_incomplete_lock_before_dataset_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[2]
    lock_path = tmp_path / "lock.json"
    lock_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("ALIX_FOUR_ARM_EXECUTION_AUTHORIZED", "true")
    monkeypatch.setenv("ALIX_DRY_RUN", "true")
    monkeypatch.setenv("ALIX_ALLOW_LIVE_TRADING", "false")
    with pytest.raises(ExecutionAuthorizationError):
        four_arm.run_real_experiment(
            csv_path=tmp_path / "must-not-be-read.csv",
            data_manifest_path=tmp_path / "must-not-be-read.json",
            lock_path=lock_path,
            model_pin_path=root / "config/four_arm/model_pins.json",
            kronos_code_dir=tmp_path / "must-not-be-loaded",
            output_dir=tmp_path / "output",
            repository_root=root,
        )


def test_real_entrypoint_accepts_complete_lock_and_rejects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[2]
    payload = _authorized_lock_payload(root)
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("ALIX_FOUR_ARM_EXECUTION_AUTHORIZED", "true")
    monkeypatch.setenv("ALIX_DRY_RUN", "true")
    monkeypatch.setenv("ALIX_ALLOW_LIVE_TRADING", "false")

    result = four_arm.run_real_experiment(
        csv_path=tmp_path / "must-not-be-read.csv",
        data_manifest_path=tmp_path / "must-not-be-read.json",
        lock_path=lock_path,
        model_pin_path=root / "config/four_arm/model_pins.json",
        kronos_code_dir=tmp_path / "must-not-be-loaded",
        output_dir=tmp_path / "output",
        repository_root=root,
        validate_lock_only=True,
    )
    assert result is None

    tampered = json.loads(json.dumps(payload))
    tampered["contracts"]["chronos_adapter_sha256"] = "0" * 64
    lock_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(
        four_arm.FourArmExecutionError, match="execution_contract_hash_mismatch"
    ):
        four_arm.run_real_experiment(
            csv_path=tmp_path / "must-not-be-read.csv",
            data_manifest_path=tmp_path / "must-not-be-read.json",
            lock_path=lock_path,
            model_pin_path=root / "config/four_arm/model_pins.json",
            kronos_code_dir=tmp_path / "must-not-be-loaded",
            output_dir=tmp_path / "output",
            repository_root=root,
            validate_lock_only=True,
        )
