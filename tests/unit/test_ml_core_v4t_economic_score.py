from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from sklearn.base import BaseEstimator  # type: ignore[import-untyped]

from services.ml_core.hyperparameter_tuning import tune_lightgbm_classifier

import numpy as np
import pytest
from pandas import DataFrame, Series

from scripts import run_ml_core_v4h_tuning_geometry as v4h
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
        return cast(
            np.ndarray, np.asarray(["BUY", "SELL", "HOLD"], dtype=object)[codes]
        )


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


def _v4t_inner_cv_preflight_fixture() -> tuple[DataFrame, DataFrame, dict[str, DataFrame]]:
    row_count = 48
    bars = DataFrame({"close": np.full(row_count, 100.0)})
    labels = DataFrame(
        {
            "label": ["BUY", "SELL", "HOLD"] * 16,
            "label_end_idx": [
                min(index + 24, row_count - 1) for index in range(row_count)
            ],
        }
    )
    features = {
        "Baseline": DataFrame({"base": range(row_count)}),
        "Baseline+Trend": DataFrame(
            {"base": range(row_count), "trend": range(row_count)}
        ),
    }
    return bars, labels, features


def test_v4t_inner_cv_class_support_failure_is_recorded_as_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars, labels, features = _v4t_inner_cv_preflight_fixture()
    train_indices = list(range(32))
    validation_indices = list(range(32, 48))
    observed: dict[str, Any] = {}

    def fake_build_splits(*args: Any, **kwargs: Any) -> tuple[list[Any], list[Any]]:
        observed["horizon_candles"] = kwargs["horizon_candles"]
        observed["n_splits"] = kwargs["n_splits"]
        observed["minimum_class_samples"] = kwargs["minimum_class_samples"]
        raise ValueError("tuning_inner_cv_class_support_insufficient")

    def fake_skipped_result(
        pair: str,
        timeframe: str,
        prepared_bars: DataFrame,
        prepared_labels: DataFrame,
        fit_indices: list[int],
        forward_indices: list[int],
        split_diagnostics: dict[str, Any],
        reason: str,
        sample_gate: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "status": reason,
            "sample_gate": sample_gate,
            "model_variants": None,
        }

    monkeypatch.setattr(
        v4h,
        "purged_temporal_indices",
        lambda *args, **kwargs: (
            train_indices,
            validation_indices,
            {"method": "TEST_PURGED_CHRONOLOGICAL_SPLIT"},
        ),
    )
    monkeypatch.setattr(
        v4h,
        "sample_gate_diagnostics",
        lambda *args: {
            "status": "PASS",
            "train_class_counts": {"BUY": 100, "SELL": 100, "HOLD": 100},
        },
    )
    monkeypatch.setattr(v4h, "label_audit_diagnostics", lambda _: {"status": "TEST"})
    monkeypatch.setattr(v4h, "_skipped_result", fake_skipped_result)
    monkeypatch.setattr(v4h, "build_purged_time_series_splits", fake_build_splits)
    def fail_if_fit_started(*args: Any, **kwargs: Any) -> dict[str, Any]:
        pytest.fail("model fitting must not start after failed preflight")

    monkeypatch.setattr(v4h, "_fit_v4h_variants", fail_if_fit_started)

    experiment = v4h.run_v4h_experiment(
        "BTC/USDT",
        "1h",
        bars,
        labels,
        features,
        scenario=v4t.V4T_ARMS[1],
        feature_profile="Baseline",
        calibration_fraction=0.20,
        minimum_hold_precision=0.40,
        minimum_hold_recall=0.20,
        random_search_iterations=1,
        tuning_objective="CUSTOM_ECONOMIC_SCORE",
        economic_outcomes=Series(np.zeros(len(bars))),
        estimated_cost_bps=90.0,
        split_horizon_candles=24,
    )

    assert experiment["status"] == v4h.V4H_INNER_CV_CLASS_SUPPORT_SKIP_STATUS
    assert experiment["sample_gate"]["outer_sample_gate_status"] == "PASS"
    assert experiment["sample_gate"]["status"] == v4h.V4H_INNER_CV_CLASS_SUPPORT_SKIP_STATUS
    assert (
        experiment["sample_gate"]["inner_cv_preflight"]["reason_code"]
        == "tuning_inner_cv_class_support_insufficient"
    )
    assert observed == {
        "horizon_candles": 24,
        "n_splits": v4h.DEFAULT_INNER_CV_SPLITS,
        "minimum_class_samples": v4h.DEFAULT_MINIMUM_CLASS_SAMPLES,
    }
    status, _ = v4t._validate_v4t_experiment_status(experiment)
    candidate = v4t._skipped_candidate_record(
        v4t.V4T_ARMS[1],
        "BTC/USDT",
        "1h",
        "Baseline",
        "weight_none",
        experiment,
    )
    assert status == v4h.V4H_INNER_CV_CLASS_SUPPORT_SKIP_STATUS
    assert candidate["status"] == v4h.V4H_INNER_CV_CLASS_SUPPORT_SKIP_STATUS
    assert candidate["evaluation_performed"] is False
    assert candidate["metrics"]["economic_candidate_gate_passed"] is False



def test_v4t_inner_cv_preflight_propagates_other_purge_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars, labels, _ = _v4t_inner_cv_preflight_fixture()

    def fail_for_unrelated_reason(
        *args: Any, **kwargs: Any
    ) -> tuple[list[Any], list[Any]]:
        raise ValueError("tuning_inner_cv_label_overlap")

    monkeypatch.setattr(
        v4h,
        "build_purged_time_series_splits",
        fail_for_unrelated_reason,
    )
    with pytest.raises(ValueError, match="tuning_inner_cv_label_overlap"):
        v4h._inner_cv_preflight_diagnostics(
            list(range(32)),
            labels.iloc[:32]["label"],
            labels,
            24,
        )


def test_v4t_inner_cv_preflight_passes_only_after_all_default_folds_validate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, labels, _ = _v4t_inner_cv_preflight_fixture()
    observed: dict[str, Any] = {}
    fold_diagnostics = [{"fold": index} for index in range(v4h.DEFAULT_INNER_CV_SPLITS)]

    def successful_build(
        training_indices: list[int],
        train_labels: Series,
        label_end_indices: list[int],
        **kwargs: Any,
    ) -> tuple[list[Any], list[dict[str, int]]]:
        observed["training_rows"] = len(training_indices)
        observed["label_rows"] = len(train_labels)
        observed["label_end_rows"] = len(label_end_indices)
        observed["horizon_candles"] = kwargs["horizon_candles"]
        observed["n_splits"] = kwargs["n_splits"]
        observed["minimum_class_samples"] = kwargs["minimum_class_samples"]
        return [([], []) for _ in fold_diagnostics], fold_diagnostics

    monkeypatch.setattr(v4h, "build_purged_time_series_splits", successful_build)
    result = v4h._inner_cv_preflight_diagnostics(
        list(range(32)),
        labels.iloc[:32]["label"],
        labels,
        24,
    )

    assert result == {
        "status": "PASS",
        "reason_code": None,
        "minimum_class_samples_per_class": v4h.DEFAULT_MINIMUM_CLASS_SAMPLES,
        "expected_fold_count": v4h.DEFAULT_INNER_CV_SPLITS,
        "validated_fold_count": v4h.DEFAULT_INNER_CV_SPLITS,
        "complete_fold_validation": True,
    }
    assert observed == {
        "training_rows": 32,
        "label_rows": 32,
        "label_end_rows": 48,
        "horizon_candles": 24,
        "n_splits": v4h.DEFAULT_INNER_CV_SPLITS,
        "minimum_class_samples": v4h.DEFAULT_MINIMUM_CLASS_SAMPLES,
    }

