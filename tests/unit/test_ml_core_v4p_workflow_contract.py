from __future__ import annotations

from pathlib import Path


def test_v4p_economic_workflow_is_manual_ci_gated_and_fail_closed() -> None:
    workflow_path = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "workflows"
        / "ml-core-v4p-economic.yml"
    )
    workflow = workflow_path.read_text(encoding="utf-8")

    assert "  workflow_dispatch:" in workflow
    assert "research/ml-core-v4p-feature-target-rebuild" in workflow
    assert "Require standard CI success for this exact V4-P commit" in workflow
    assert "--days 730" in workflow
    assert "--random-search-iterations 9" in workflow
    assert "alix-v4p-economic-evaluation" in workflow
    assert "alix-v4p-economic-evaluation-report.json" in workflow
    assert "alix-v4p-economic-comparison.csv" in workflow
    assert "alix-v4p-trades.csv" in workflow
    assert "take_profit_multiplier" in workflow
    assert "stop_loss_multiplier" in workflow
    assert "sequence_model_used" in workflow
    assert "Champion" in workflow
    assert "order_permission" in workflow
    assert "trading_authorization" in workflow
    assert "NO-GO" in workflow
    assert "private_api_calls" in workflow
    assert "real_order_attempts" in workflow
    assert "cancel-in-progress: false" in workflow
