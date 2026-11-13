from __future__ import annotations

from pathlib import Path

import pytest

from scripts.open_posthack_holdout import (
    HoldoutLocked,
    authorized_dataset_path,
    require_release_authorization,
)


def test_holdout_opener_is_locked_by_default(tmp_path: Path) -> None:
    dataset = tmp_path / "mock-dataset.csv"
    dataset.write_text("mock-only")
    with pytest.raises(HoldoutLocked, match="HOLDOUT_LOCKED"):
        authorized_dataset_path(dataset)


def test_blank_owner_release_authorization_remains_locked() -> None:
    with pytest.raises(HoldoutLocked):
        require_release_authorization("   ")


def test_explicit_release_reference_is_required_before_path_access(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "mock-dataset.csv"
    dataset.write_text("mock-only")
    assert (
        authorized_dataset_path(
            dataset, owner_release_authorization="OWNER-DIRECTIVE-REFERENCE"
        )
        == dataset
    )


def test_training_evaluation_and_strategy_modules_do_not_reference_holdout() -> None:
    root = Path(__file__).resolve().parents[2]
    roots = (
        root / "user_data" / "strategies",
        root / "user_data" / "freqaimodels",
        root / "research" / "ai_ensemble",
        root / "research" / "deap",
    )
    forbidden = (
        "HOLDOUT_COMMITMENT_V2.md",
        "posthack_window_spec.json",
        "nobitex-btcusdt-posthack-1h",
    )
    violations: list[str] = []
    for scan_root in roots:
        for path in scan_root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in source:
                    violations.append(f"{path.relative_to(root)}:{token}")
    script_patterns = (
        "train_*.py",
        "run_*backtest*.py",
        "run_*walkforward*.py",
        "run_meta_labeling_walkforward.py",
    )
    scripts_dir = root / "scripts"
    for pattern in script_patterns:
        for path in scripts_dir.glob(pattern):
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in source:
                    violations.append(f"{path.relative_to(root)}:{token}")
    assert violations == []
