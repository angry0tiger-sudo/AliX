from __future__ import annotations

import numpy as np
import pytest

from scripts.train_risk_allocation_ppo import (
    MINIMUM_OOS_TRADES,
    _json_default,
    _promotion_decision,
    _validate_commit_sha,
)


def metrics(
    *,
    pf: float | None,
    drawdown: float,
    trades: int = MINIMUM_OOS_TRADES,
    z_score: float | None = 1.0,
) -> dict[str, float | int | None]:
    return {
        "net_profit_factor": pf,
        "max_drawdown": drawdown,
        "trade_count": trades,
        "statistical_z_score": z_score,
    }


def test_promotion_requires_validation_and_holdout_to_pass_all_gates() -> None:
    baseline = metrics(pf=0.9, drawdown=0.30)
    candidate = metrics(pf=1.2, drawdown=0.10)
    decision, checks = _promotion_decision(baseline, candidate, baseline, candidate)
    assert decision == "GO_FOR_OWNER_REVIEW"
    assert all(checks.values())


def test_promotion_fails_when_holdout_pf_or_drawdown_gate_fails() -> None:
    baseline = metrics(pf=0.9, drawdown=0.30)
    validation = metrics(pf=1.2, drawdown=0.10)
    weak_holdout = metrics(pf=1.05, drawdown=0.16)
    decision, checks = _promotion_decision(baseline, validation, baseline, weak_holdout)
    assert decision == "NO-GO"
    assert checks["validation_net_pf_at_least_1_1"] is True
    assert checks["holdout_net_pf_at_least_1_1"] is False
    assert checks["holdout_drawdown_below_15_percent"] is False


def test_promotion_fails_on_insufficient_trades_or_non_positive_z_score() -> None:
    baseline = metrics(pf=0.9, drawdown=0.30)
    insufficient = metrics(pf=1.3, drawdown=0.10, trades=MINIMUM_OOS_TRADES - 1)
    weak_z = metrics(pf=1.3, drawdown=0.10, z_score=-0.1)
    decision, checks = _promotion_decision(baseline, insufficient, baseline, weak_z)
    assert decision == "NO-GO"
    assert checks["validation_minimum_trade_support"] is False
    assert checks["holdout_positive_statistical_z_score"] is False


def test_promotion_fails_when_baseline_or_candidate_pf_is_undefined() -> None:
    undefined = metrics(pf=None, drawdown=0.30)
    candidate = metrics(pf=1.3, drawdown=0.10)
    decision, checks = _promotion_decision(undefined, candidate, undefined, candidate)
    assert decision == "NO-GO"
    assert checks["validation_net_pf_improved"] is False
    assert checks["holdout_net_pf_improved"] is False


def test_promotion_requires_strictly_more_than_fifty_completed_trades() -> None:
    assert MINIMUM_OOS_TRADES == 51
    baseline = metrics(pf=0.9, drawdown=0.30, trades=51)
    candidate_at_floor = metrics(pf=1.2, drawdown=0.10, trades=50)
    decision, checks = _promotion_decision(
        baseline, candidate_at_floor, baseline, baseline
    )
    assert decision == "NO-GO"
    assert checks["validation_minimum_trade_support"] is False


def test_training_commit_sha_must_be_full_lowercase_hex() -> None:
    _validate_commit_sha("e6c86b08da98ad28e479b45827fb8de6cd3cc8dd")
    with pytest.raises(ValueError, match="rl_commit_sha_invalid"):
        _validate_commit_sha("e6c86b0")
    with pytest.raises(ValueError, match="rl_commit_sha_invalid"):
        _validate_commit_sha("E6C86B08DA98AD28E479B45827FB8DE6CD3CC8DD")


def test_json_default_normalizes_numpy_scalar_and_array_values() -> None:
    assert _json_default(np.int64(31_801)) == 31_801
    assert _json_default(np.float64(0.25)) == 0.25
    assert _json_default(np.bool_(True)) is True
    assert _json_default(np.asarray([1, 2], dtype=np.int64)) == [1, 2]


def test_json_default_rejects_unsupported_objects() -> None:
    with pytest.raises(TypeError, match="unsupported_json_value:object"):
        _json_default(object())
