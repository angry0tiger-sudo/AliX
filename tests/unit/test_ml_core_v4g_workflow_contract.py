from __future__ import annotations

from pathlib import Path


def test_v4g_workflow_is_push_gated_two_profile_five_fold_and_fail_closed() -> None:
    workflow_path = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "workflows"
        / "ml-core-v4g-trend-calibration.yml"
    )
    workflow = workflow_path.read_text(encoding="utf-8")

    assert "  push:\n" in workflow
    assert "      - research/ml-core-v4g-trend-calibration\n" in workflow
    assert "Require standard CI success for this exact V4-G research commit" in workflow
    assert "--feature-ablation" in workflow
    assert "--walk-forward-folds 5" in workflow
    assert '"V4G_Flat_12_Candle:12:2.0:1.0"' in workflow
    assert "Baseline+Trend" in workflow
    assert "80" in workflow
    assert "economic_evaluation_performed" in workflow
    assert "Champion" in workflow
    assert "trading_authorization" in workflow
    assert "NO-GO" in workflow
    assert "private_api_calls" in workflow
    assert "real_order_attempts" in workflow


def test_v4g_workflow_declares_exactly_five_test_windows_and_eighty_runs() -> None:
    workflow_path = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "workflows"
        / "ml-core-v4g-trend-calibration.yml"
    )
    workflow = workflow_path.read_text(encoding="utf-8")

    assert "--walk-forward-folds 5" in workflow
    assert 'assert report["experiment_count"] == 80' in workflow
    assert 'assert len(grouped) == 40' in workflow
    assert 'assert report["walk_forward_summary"]["fold_count"] == 5' in workflow
    assert 'assert len(report["walk_forward_summary"]["folds"]) == 5' in workflow
