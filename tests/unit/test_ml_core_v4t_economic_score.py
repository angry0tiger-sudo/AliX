from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from sklearn.base import BaseEstimator  # type: ignore[import-untyped]

from services.ml_core.hyperparameter_tuning import tune_lightgbm_classifier

import numpy as np
import pytest
from pandas import DataFrame, Series

from scripts import run_ml_core_v4t_economic_score as v4t
from services.ml_core.hyperparameter_tuning import (
    Custom_Economic_Score,
    TUNING_OBJECTIVES,
)


class FixedPredictions:
    def __init__(self, predictions: list[str]) -> None:
        self._predictions = predictions

    def predict(self, features: DataFrame) -> np.ndarray:
        assert len(features) == len(self._predictions)
        return np.asarray(self._predictions, dtype=object)


def _probability_record(
    row_index: int,
    buy: float,
    sell: float,
    hold: float,
    label: str = "BUY",
) -> dict[str, Any]:
    return {
        "row_index": row_index,
        "timestamp": f"2026-01-01T{row_index % 24:02d}:00:00+00:00",
        "actual_label": label,
        "probabilities": {"BUY": buy, "SELL": sell, "HOLD": hold},
    }


def test_custom_economic_score_is_registered_as_tuning_objective() -> None:
    assert "CUSTOM_ECONOMIC_SCORE" in TUNING_OBJECTIVES


def test_custom_economic_score_rewards_positive_net_expectancy() -> None:
    features = DataFrame({"x": [0, 1, 2]})
    actual = Series(["BUY", "SELL", "HOLD"])
    gross = Series([0.02, 0.03, 0.01])
    score = Custom_Economic_Score(
        FixedPredictions(["BUY", "BUY", "SELL"]),
        features,
        actual,
        gross_returns_by_index=gross,
        estimated_cost_bps=10.0,
    )
    assert score > 0.0


def test_custom_economic_score_penalizes_drawdown_and_low_trade_count() -> None:
    features = DataFrame({"x": list(range(30))})
    actual = Series(["BUY"] * 30)
    gross = Series([0.01] * 30)
    score = Custom_Economic_Score(
        FixedPredictions(["BUY"] * 30),
        features,
        actual,
        gross_returns_by_index=gross,
        estimated_cost_bps=0.0,
    )
    lower_count_score = Custom_Economic_Score(
        FixedPredictions(["BUY"] * 10 + ["HOLD"] * 20),
        features,
        actual,
        gross_returns_by_index=gross,
        estimated_cost_bps=0.0,
    )
    assert lower_count_score < score


def test_custom_economic_score_rejects_missing_outcomes() -> None:
    features = DataFrame({"x": [0, 1]})
    with pytest.raises(
        ValueError, match="tuning_economic_outcomes_missing_or_non_finite"
    ):
        Custom_Economic_Score(
            FixedPredictions(["BUY", "HOLD"]),
            features,
            Series(["BUY", "HOLD"]),
            gross_returns_by_index=Series([0.01, np.nan]),
        )


def test_custom_objective_requires_economic_outcomes() -> None:
    features = DataFrame({"x": [0, 1, 2]})
    from services.ml_core.hyperparameter_tuning import tune_lightgbm_classifier

    with pytest.raises(ValueError, match="tuning_economic_objective_requires_outcomes"):
        tune_lightgbm_classifier(
            object(),
            features,
            Series(["BUY", "SELL", "HOLD"]),
            [0, 1, 2],
            [0, 1, 2],
            horizon_candles=1,
            class_weight_policy="weight_none",
            minimum_hold_recall=0.2,
            tuning_objective="CUSTOM_ECONOMIC_SCORE",
        )


def test_v4t_dynamic_threshold_selects_top_buy_confidence_only() -> None:
    records = [
        _probability_record(10, 0.90, 0.05, 0.05),
        _probability_record(11, 0.80, 0.10, 0.10),
        _probability_record(12, 0.70, 0.20, 0.10),
        _probability_record(13, 0.60, 0.30, 0.10),
        _probability_record(14, 0.50, 0.40, 0.10),
        _probability_record(15, 0.40, 0.50, 0.10, "SELL"),
        _probability_record(16, 0.30, 0.60, 0.10, "SELL"),
        _probability_record(17, 0.20, 0.70, 0.10, "SELL"),
        _probability_record(18, 0.10, 0.80, 0.10, "SELL"),
        _probability_record(19, 0.05, 0.85, 0.10, "SELL"),
    ]
    result = v4t.select_dynamic_buy_confidence_threshold(
        records, calibration_fraction=0.5, top_buy_share=0.30
    )
    assert result["status"] == "SELECTED"
    assert result["calibration_row_count"] == 5
    assert result["calibration_raw_buy_count"] == 5
    assert result["target_selected_buy_count"] == 2
    assert result["selected_threshold"] == pytest.approx(0.80)
    assert result["selected_buy_count_in_calibration"] == 2
    assert result["forward_start_row_index"] == 15


def test_v4t_dynamic_threshold_reports_ties_without_hiding_extra_coverage() -> None:
    records = [
        _probability_record(0, 0.8, 0.1, 0.1),
        _probability_record(1, 0.8, 0.1, 0.1),
        _probability_record(2, 0.8, 0.1, 0.1),
        _probability_record(3, 0.7, 0.2, 0.1),
        _probability_record(4, 0.6, 0.3, 0.1),
        _probability_record(5, 0.5, 0.4, 0.1),
        _probability_record(6, 0.4, 0.5, 0.1),
        _probability_record(7, 0.3, 0.6, 0.1),
        _probability_record(8, 0.2, 0.7, 0.1),
        _probability_record(9, 0.1, 0.8, 0.1),
    ]
    result = v4t.select_dynamic_buy_confidence_threshold(
        records, calibration_fraction=0.5, top_buy_share=0.30
    )
    assert result["target_selected_buy_count"] == 2
    assert result["threshold_tie_count"] == 3
    assert result["selected_buy_count_in_calibration"] == 3
    assert result["achieved_buy_share"] == pytest.approx(0.60)


def test_v4t_threshold_fails_closed_when_calibration_has_no_raw_buy() -> None:
    records = [
        _probability_record(0, 0.1, 0.1, 0.8, "HOLD"),
        _probability_record(1, 0.2, 0.1, 0.7, "HOLD"),
        _probability_record(2, 0.3, 0.1, 0.6, "HOLD"),
        _probability_record(3, 0.1, 0.8, 0.1, "SELL"),
        _probability_record(4, 0.2, 0.7, 0.1, "SELL"),
    ]
    result = v4t.select_dynamic_buy_confidence_threshold(
        records, calibration_fraction=0.6
    )
    assert result["status"] == "NO_BUY_SIGNALS_IN_CALIBRATION"
    assert result["selected_threshold"] == 1.0


def test_v4t_probability_rows_preserve_split_timestamp_label_and_probabilities() -> (
    None
):
    records = [
        _probability_record(10, 0.7, 0.2, 0.1),
        _probability_record(11, 0.6, 0.3, 0.1),
        _probability_record(12, 0.5, 0.4, 0.1),
        _probability_record(13, 0.4, 0.5, 0.1, "SELL"),
    ]
    rows = v4t._row_level_probability_rows(
        "candidate",
        v4t.V4T_ARMS[1],
        "BTC/USDT",
        "1h",
        "Baseline",
        "weight_none",
        records,
        2,
    )
    assert len(rows) == 4
    assert rows[0]["split_role"] == "CALIBRATION"
    assert rows[-1]["split_role"] == "FORWARD_VALIDATION"
    assert rows[0]["timestamp"] == records[0]["timestamp"]
    assert rows[0]["actual_label"] == "BUY"
    assert rows[0]["prob_buy"] == pytest.approx(0.7)
    assert rows[0]["prob_sell"] == pytest.approx(0.2)
    assert rows[0]["prob_hold"] == pytest.approx(0.1)


def test_v4t_target_geometry_removes_two_to_one_ratio() -> None:
    assert len(v4t.V4T_ARMS) == 2
    assert {
        (
            arm["take_profit_multiplier"],
            arm["stop_loss_multiplier"],
            arm["horizon_candles"],
        )
        for arm in v4t.V4T_ARMS
    } == {(1.0, 1.0, 24), (1.5, 1.0, 24)}
    assert all(arm["take_profit_multiplier"] != 2.0 for arm in v4t.V4T_ARMS)


def test_v4t_soft_trade_target_does_not_create_a_hard_gate() -> None:
    metrics = {
        "trade_count": 8,
        "profit_factor_net_of_costs": 1.8,
        "profit_factor_infinite_no_losses": False,
        "max_drawdown_fraction": 0.08,
        "trade_count_gate_passed": False,
        "minimum_trade_count_for_acceptance": 30,
        "economic_candidate_gate_passed": False,
    }
    result = v4t._apply_soft_sample_diagnostics(metrics)
    assert "trade_count_gate_passed" not in result
    assert "minimum_trade_count_for_acceptance" not in result
    assert result["economic_candidate_gate_passed"] is True
    assert result["economic_data_status"] == "BELOW_SOFT_SAMPLE_TARGET"
    assert result["trade_count_soft_penalty_factor"] == pytest.approx(22 / 30)


def test_v4t_safety_contract_stays_closed() -> None:
    safety = {
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
        "private_api_calls": 0,
        "real_order_attempts": 0,
        "orders_or_execution_api_used": False,
        "short_positions_supported": False,
        "sequence_model_used": False,
    }
    assert safety["Champion"] is None
    assert safety["order_permission"] is False
    assert safety["trading_authorization"] == "NO-GO"
    assert safety["private_api_calls"] == 0
    assert safety["real_order_attempts"] == 0
    assert safety["short_positions_supported"] is False


class TuningClassifier(BaseEstimator):  # type: ignore[misc]
    def __init__(
        self,
        num_leaves: int = 15,
        learning_rate: float = 0.05,
        min_child_samples: int = 20,
    ) -> None:
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.min_child_samples = min_child_samples

    def fit(self, features: DataFrame, labels: Series) -> TuningClassifier:
        assert len(features) == len(labels)
        self.is_fitted_ = True
        return self

    def predict(self, features: DataFrame) -> np.ndarray:
        codes = features["signal"].to_numpy(dtype=int) % 3
        return cast(np.ndarray, np.asarray(["BUY", "SELL", "HOLD"], dtype=object)[codes])


def test_custom_objective_executes_inner_cv_and_reports_its_selection_score() -> None:
    training_indices = list(range(10, 610))
    features = DataFrame(
        {"signal": [index % 3 for index in training_indices]},
        index=training_indices,
    )
    labels = Series(
        [["BUY", "SELL", "HOLD"][index % 3] for index in training_indices],
        index=training_indices,
        dtype=object,
    )
    label_end_indices = [min(index + 12, 699) for index in range(700)]
    outcomes = Series(0.01, index=features.index, dtype=float)
    model, diagnostics = tune_lightgbm_classifier(
        TuningClassifier(),
        features,
        labels,
        training_indices,
        label_end_indices,
        horizon_candles=12,
        class_weight_policy="weight_none",
        minimum_hold_recall=0.2,
        n_iter=1,
        n_splits=3,
        tuning_objective="CUSTOM_ECONOMIC_SCORE",
        economic_outcomes=outcomes,
        estimated_cost_bps=90.0,
    )
    assert model.is_fitted_ is True
    assert diagnostics["selection_objective"] == "CUSTOM_ECONOMIC_SCORE"
    assert diagnostics["economic_scorer"] == "Custom_Economic_Score"
    assert diagnostics["selected_mean_cv_custom_economic_score"] is not None
    assert (
        diagnostics["custom_economic_score_contract"]["minimum_trade_count_soft_target"]
        == 30
    )
    assert diagnostics["validation_accessed_during_search"] is False


def test_custom_score_returns_soft_penalty_when_no_buy_is_predicted() -> None:
    features = DataFrame({"x": [0, 1]})
    score = Custom_Economic_Score(
        FixedPredictions(["SELL", "HOLD"]),
        features,
        Series(["BUY", "HOLD"]),
        gross_returns_by_index=Series([0.01, -0.01]),
    )
    assert score == pytest.approx(-50.0)


def test_custom_score_rejects_invalid_net_event_return() -> None:
    features = DataFrame({"x": [0]})
    with pytest.raises(ValueError, match="tuning_custom_score_net_returns_invalid"):
        Custom_Economic_Score(
            FixedPredictions(["BUY"]),
            features,
            Series(["BUY"]),
            gross_returns_by_index=Series([-1.0]),
            estimated_cost_bps=0.0,
        )


def test_custom_score_rejects_invalid_soft_penalty_configuration() -> None:
    features = DataFrame({"x": [0]})
    with pytest.raises(ValueError, match="tuning_custom_score_penalty_invalid"):
        Custom_Economic_Score(
            FixedPredictions(["BUY"]),
            features,
            Series(["BUY"]),
            gross_returns_by_index=Series([0.01]),
            trade_count_penalty_bps=-1.0,
        )


def test_v4t_workflow_is_branch_scoped_and_waits_for_exact_sha_ci() -> None:
    workflow_text = Path(".github/workflows/ml-core-v4t-economic-score.yml").read_text(
        encoding="utf-8"
    )
    assert "research/ml-core-v4t-economic-score" in workflow_text
    assert "Require standard CI success for this exact V4-T commit" in workflow_text
    assert "TARGET_SHA" in workflow_text
    assert "Custom_Economic_Score" in workflow_text
    assert "alix-v4t-row-level-probabilities.csv" in workflow_text
    assert "trading_authorization=NO-GO" in workflow_text
