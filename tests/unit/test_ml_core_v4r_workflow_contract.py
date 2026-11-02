from __future__ import annotations

from pathlib import Path


def test_v4r_economic_workflow_is_ci_gated_branch_scoped_and_fail_closed() -> None:
    workflow_path = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "workflows"
        / "ml-core-v4r-economic.yml"
    )
    workflow = workflow_path.read_text(encoding="utf-8")
    assert "name: AliX V4-R Economic Architecture Evaluation" in workflow
    assert "research/ml-core-v4r-economic-architecture" in workflow
    assert "Require standard CI success for this exact V4-R commit" in workflow
    assert "scripts.run_ml_core_v4r_economic_architecture" in workflow
    assert "alix-v4r-economic-architecture" in workflow
    assert "TP=2.0 ATR" in workflow
    assert "SL=1.0 ATR" in workflow
    assert "Horizon=24" in workflow
    assert "NET_EXPECTANCY_PROXY" in workflow
    assert "custom_eval_metric" in workflow
    assert "market_timeframes=1h,4h" in workflow
    assert "scripts/fetch_binance_public_ohlcv.py" in workflow
    assert "Champion=null" in workflow
    assert "order_permission=false" in workflow
    assert "trading_authorization=NO-GO" in workflow
    assert "private_api_calls=0" in workflow
    assert "real_order_attempts=0" in workflow
    assert "cancel-in-progress: false" in workflow
