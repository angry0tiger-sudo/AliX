from __future__ import annotations

from pathlib import Path


def test_v4f_workflow_is_push_triggered_gated_and_fail_closed() -> None:
    workflow_path = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "workflows"
        / "ml-core-v4f-feature-ablation.yml"
    )
    workflow = workflow_path.read_text(encoding="utf-8")

    assert "  push:\n" in workflow
    assert "      - research/ml-core-v4f-feature-engineering\n" in workflow
    assert "Require standard CI success for this exact V4-F research commit" in workflow
    assert "--feature-ablation" in workflow
    assert '"V4F_Flat_12_Candle:12:2.0:1.0"' in workflow
    assert "Champion=null" in workflow
    assert "trading_authorization=NO-GO" in workflow
    assert 'assert safety["private_api_calls"] == 0' in workflow
    assert 'assert safety["real_order_attempts"] == 0' in workflow
    assert 'assert report["economic_evaluation_performed"] is False' in workflow
