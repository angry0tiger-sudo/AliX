from __future__ import annotations

from pathlib import Path


def test_v4q_economic_workflow_is_push_triggered_ci_gated_and_fail_closed() -> None:
    workflow_path = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "workflows"
        / "ml-core-v4q-economic.yml"
    )
    workflow = workflow_path.read_text(encoding="utf-8")

    assert "name: AliX V4-Q Economic Evaluation" in workflow
    assert "  push:" in workflow
    assert "      - research/ml-core-v4p-feature-target-rebuild" in workflow
    assert "Require standard CI success for this exact V4-Q commit" in workflow
    assert "scripts.run_ml_core_v4q_feature_target_rebuild" in workflow
    assert "alix-v4q-economic-evaluation" in workflow
    assert "Feature Importance Gain" in workflow
    assert "Comparison CSV (exact persisted rows)" in workflow
    assert (
        'report["model"]["hyperparameter_selection_objective"] == "BUY_F1"' in workflow
    )
    assert "NEXT_BAR_OPEN_WITH_LONG_ENTRY_SLIPPAGE" in workflow
    assert '"Champion": None' in workflow
    assert "order_permission" in workflow
    assert "NO-GO" in workflow
    assert "private_api_calls" in workflow
    assert "real_order_attempts" in workflow
    assert "cancel-in-progress: false" in workflow
