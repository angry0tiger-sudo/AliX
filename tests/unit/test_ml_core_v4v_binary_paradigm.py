from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator  # type: ignore[import-untyped]

from scripts import run_ml_core_v4v_binary_paradigm as v4v
from services.ml_core.extended_features import (
    feature_profile_columns,
    feature_profile_groups,
)
from services.ml_core.binary_calibration import select_dynamic_buy_threshold
from services.ml_core.binary_targets import (
    BINARY_BUY_LABELS,
    apply_binary_buy_targets,
)
from services.ml_core.hyperparameter_tuning import (
    Binary_BUY_Economic_Score,
    build_purged_time_series_splits,
    tune_lightgbm_binary_classifier,
)


class FixedBinaryProbabilities:
    classes_ = np.asarray(["NO_BUY", "BUY"], dtype=object)

    def __init__(self, buy_probabilities: list[float]) -> None:
        self.buy_probabilities = np.asarray(buy_probabilities, dtype=float)

    def predict_proba(self, features: DataFrame) -> np.ndarray:
        assert len(features) == len(self.buy_probabilities)
        return np.column_stack((1.0 - self.buy_probabilities, self.buy_probabilities))


class BinaryTuningClassifier(BaseEstimator):  # type: ignore[misc]
    def __init__(
        self,
        num_leaves: int = 15,
        learning_rate: float = 0.05,
        min_child_samples: int = 20,
    ) -> None:
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.min_child_samples = min_child_samples

    def fit(self, features: DataFrame, labels: Series) -> BinaryTuningClassifier:
        if len(features) != len(labels):
            raise ValueError("test_binary_fit_alignment_invalid")
        self.classes_ = np.asarray(["NO_BUY", "BUY"], dtype=object)
        self.is_fitted_ = True
        return self

    def predict_proba(self, features: DataFrame) -> np.ndarray:
        signal = features["signal"].to_numpy(dtype=float)
        buy = np.where(signal >= 0.5, 0.80, 0.20)
        return np.column_stack((1.0 - buy, buy))

    def predict(self, features: DataFrame) -> np.ndarray:
        probabilities = self.predict_proba(features)[:, 1]
        return np.where(probabilities >= 0.5, "BUY", "NO_BUY")


def _source_labels() -> DataFrame:
    return DataFrame(
        {
            "event_index": list(range(6)),
            "label_end_idx": [1, 2, 3, 4, 5, 5],
            "label": ["BUY", "SELL", "HOLD", None, "BUY", "SELL"],
            "exit_reason": [
                "TP_HIT",
                "SL_HIT",
                "HORIZON_EXPIRY",
                None,
                "TP_HIT",
                "COLLISION",
            ],
        }
    )


def test_binary_target_merges_sell_and_hold_but_preserves_unlabeled_rows() -> None:
    source = _source_labels()
    result = apply_binary_buy_targets(source)
    assert result["label"].iloc[0] == "BUY"
    assert result["label"].iloc[1] == "NO_BUY"
    assert result["label"].iloc[2] == "NO_BUY"
    assert pd.isna(result["label"].iloc[3])
    assert result["label"].iloc[4] == "BUY"
    assert result["label"].iloc[5] == "NO_BUY"
    assert result["source_label"].tolist()[:3] == ["BUY", "SELL", "HOLD"]
    assert source["label"].tolist() == ["BUY", "SELL", "HOLD", None, "BUY", "SELL"]
    assert set(result["label"].dropna().unique()) == set(BINARY_BUY_LABELS)


def test_binary_target_does_not_claim_the_new_classes_are_balanced() -> None:
    source = DataFrame(
        {
            "event_index": list(range(100)),
            "label_end_idx": [min(index + 1, 99) for index in range(100)],
            "label": ["BUY"] * 21 + ["SELL"] * 70 + ["HOLD"] * 9,
            "exit_reason": ["TP_HIT"] * 21 + ["SL_HIT"] * 70 + ["HORIZON_EXPIRY"] * 9,
        }
    )
    result = apply_binary_buy_targets(source)
    counts = result["label"].value_counts().to_dict()
    assert counts == {"NO_BUY": 79, "BUY": 21}
    assert counts["NO_BUY"] != counts["BUY"]


def test_binary_target_rejects_unknown_source_labels_fail_closed() -> None:
    source = _source_labels()
    source.loc[0, "label"] = "SHORT"
    with pytest.raises(ValueError, match="v4v_source_label_values_invalid"):
        apply_binary_buy_targets(source)


def test_binary_target_requires_barrier_provenance_columns() -> None:
    with pytest.raises(ValueError, match="v4v_label_columns_missing"):
        apply_binary_buy_targets(DataFrame({"label": ["BUY"]}))


def test_binary_purged_outer_split_retains_chronology_and_label_purge() -> None:
    row_count = 400
    labels = DataFrame(
        {
            "label": ["BUY", "NO_BUY"] * (row_count // 2),
            "label_end_idx": [
                min(index + 12, row_count - 1) for index in range(row_count)
            ],
        }
    )
    features = DataFrame(
        {
            "signal": np.arange(row_count, dtype=float),
            "trend": np.arange(row_count, dtype=float) / row_count,
        }
    )
    train, validation, diagnostics = v4v._binary_outer_split(
        features, labels, horizon_candles=24
    )
    assert train
    assert validation
    assert train[-1] < validation[0]
    assert diagnostics["maximum_training_label_end_index"] < validation[0]
    assert diagnostics["inner_tuning_uses_outer_train_only"] is True


def test_binary_outer_sample_gate_checks_both_classes_without_rebalancing() -> None:
    train = Series(["BUY"] * 30 + ["NO_BUY"] * 70)
    validation = Series(["BUY"] * 40 + ["NO_BUY"] * 60)
    result = v4v._binary_sample_gate(train, validation)
    assert result["status"] == "PASS"
    assert result["train_class_counts"] == {"BUY": 30, "NO_BUY": 70}
    assert result["validation_class_counts"] == {"BUY": 40, "NO_BUY": 60}


def test_binary_inner_cv_requires_support_in_each_fit_and_validation_fold() -> None:
    training_indices = list(range(10, 610))
    labels = Series(
        ["BUY" if index % 2 else "NO_BUY" for index in training_indices],
        index=training_indices,
        dtype=object,
    )
    label_end_indices = [min(index + 12, 699) for index in range(700)]
    splits, diagnostics = build_purged_time_series_splits(
        training_indices,
        labels,
        label_end_indices,
        horizon_candles=24,
        n_splits=3,
        minimum_class_samples=20,
        class_labels=BINARY_BUY_LABELS,
        minimum_validation_class_samples=20,
    )
    assert len(splits) == 3
    for (fit_positions, validation_positions), fold in zip(
        splits, diagnostics, strict=True
    ):
        assert fold["fit_buy_count"] >= 20
        assert fold["fit_no_buy_count"] >= 20
        assert fold["validation_buy_count"] >= 20
        assert fold["validation_no_buy_count"] >= 20
        assert fit_positions[-1] < validation_positions[0]


def test_binary_economic_score_penalizes_missed_positive_buy_opportunities() -> None:
    features = DataFrame({"x": [0, 1, 2]})
    estimator = FixedBinaryProbabilities([0.10, 0.40, 0.64])
    score = Binary_BUY_Economic_Score(
        estimator,
        features,
        Series(["BUY", "NO_BUY", "BUY"]),
        gross_returns_by_index=Series([0.10, -0.02, 0.05]),
        estimated_cost_bps=0.0,
        buy_probability_threshold=0.95,
        missed_opportunity_penalty_weight=0.5,
    )
    assert score == pytest.approx(-750.0)


def test_binary_economic_score_penalizes_false_buy_even_when_no_buy_event_return_is_positive() -> (
    None
):
    features = DataFrame({"x": [0, 1]})
    estimator = FixedBinaryProbabilities([0.90, 0.80])
    score = Binary_BUY_Economic_Score(
        estimator,
        features,
        Series(["NO_BUY", "NO_BUY"]),
        gross_returns_by_index=Series([0.02, 0.01]),
        estimated_cost_bps=0.0,
        buy_probability_threshold=0.65,
    )
    assert score < 0.0


def test_binary_economic_score_rewards_profitable_true_buy_predictions() -> None:
    features = DataFrame({"x": [0, 1]})
    estimator = FixedBinaryProbabilities([0.90, 0.80])
    score = Binary_BUY_Economic_Score(
        estimator,
        features,
        Series(["BUY", "BUY"]),
        gross_returns_by_index=Series([0.02, 0.015]),
        estimated_cost_bps=0.0,
        buy_probability_threshold=0.65,
    )
    assert score > 0.0


def test_binary_economic_score_rejects_missing_outcomes() -> None:
    with pytest.raises(
        ValueError,
        match="tuning_binary_economic_outcomes_missing_or_non_finite",
    ):
        Binary_BUY_Economic_Score(
            FixedBinaryProbabilities([0.90, 0.80]),
            DataFrame({"x": [0, 1]}),
            Series(["BUY", "NO_BUY"]),
            gross_returns_by_index=Series([0.01, np.nan]),
        )


def test_binary_tuner_runs_purged_search_and_keeps_outer_validation_unused() -> None:
    training_indices = list(range(10, 610))
    features = DataFrame(
        {"signal": [index % 2 for index in training_indices]},
        index=training_indices,
    )
    labels = Series(
        ["BUY" if index % 2 else "NO_BUY" for index in training_indices],
        index=training_indices,
        dtype=object,
    )
    label_end_indices = [min(index + 12, 699) for index in range(700)]
    outcomes = Series(0.02, index=features.index, dtype=float)
    model, diagnostics = tune_lightgbm_binary_classifier(
        BinaryTuningClassifier(),
        features,
        labels,
        training_indices,
        label_end_indices,
        horizon_candles=24,
        class_weight_policy="weight_none",
        n_iter=1,
        n_splits=3,
        economic_outcomes=outcomes,
        estimated_cost_bps=90.0,
        target_buy_fraction=0.10,
        missed_opportunity_penalty_weight=0.50,
    )
    assert model.is_fitted_ is True
    assert set(model.classes_) == set(BINARY_BUY_LABELS)
    assert diagnostics["selection_objective"] == "BINARY_BUY_ECONOMIC_SCORE"
    assert diagnostics["economic_scorer"] == "Binary_BUY_Economic_Score"
    assert diagnostics["threshold_selection"] == "DYNAMIC_TOP_PERCENTILE_PER_INNER_CV_FOLD"
    assert diagnostics["target_buy_fraction"] == pytest.approx(0.10)
    assert diagnostics["validation_accessed_during_search"] is False
    assert diagnostics["minimum_fit_class_samples"] == 20
    assert diagnostics["minimum_validation_class_samples"] == 20
    assert (
        diagnostics["custom_economic_score_contract"]["missed_opportunity_penalty_weight"]
        == pytest.approx(0.50)
    )
    assert (
        diagnostics["custom_economic_score_contract"]["minimum_trade_count_penalty"]
        is False
    )


def test_binary_metrics_use_selected_dynamic_threshold_and_continuous_roc_auc() -> None:
    actual = Series(["BUY", "BUY", "NO_BUY", "NO_BUY"])
    result = v4v._binary_classification_metrics(
        actual, np.asarray([0.75, 0.64, 0.66, 0.20], dtype=float), 0.65
    )
    assert result["threshold"] == pytest.approx(0.65)
    assert result["buy_prediction_count"] == 2
    assert result["roc_auc"] == pytest.approx(0.50)
    assert result["class_labels"] == ["BUY", "NO_BUY"]
    assert result["confusion_matrix"]["rows_are"] == "actual"


def test_binary_execution_signals_map_no_buy_to_no_entry_and_never_sell() -> None:
    records = [
        {
            "row_index": 100,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "buy_probability_threshold": 0.70,
            "prob_buy": 0.70,
        },
        {
            "row_index": 101,
            "timestamp": "2026-01-01T01:00:00+00:00",
            "buy_probability_threshold": 0.70,
            "prob_buy": 0.699,
        },
    ]
    signals = v4v._execution_signals(records, "1h")
    assert signals[0]["signal"] == "BUY"
    assert signals[1]["signal"] == "NO_ENTRY"
    assert signals[0]["confidence_threshold"] == pytest.approx(0.70)
    assert all(item["signal"] != "SELL" for item in signals)
    assert signals[0]["execution_timestamp"] == pd.Timestamp(
        "2026-01-01T01:00:00+00:00"
    )


def test_combined_feature_profile_contains_volatility_and_volume_groups() -> None:
    assert feature_profile_groups("Combined") == (
        "Baseline",
        "Trend",
        "Volatility",
        "Volume",
    )
    columns = feature_profile_columns("Combined")
    assert "%-volatility_bollinger_bandwidth_20" in columns
    assert "%-volume_relative_20" in columns


def test_v4v_step_summary_contains_economic_metrics_and_closed_safety() -> None:
    summary = v4v._render_step_summary(
        {
            "commit_sha": "test-sha",
            "arms": [
                {
                    "name": "PRIMARY_ASYMMETRIC_1_5R",
                    "aggregate": {
                        "candidate_count": 8,
                        "trained_candidate_count": 4,
                        "trade_count": 123,
                        "net_profit_factor": 1.23,
                        "net_profit_factor_infinite_no_losses": False,
                        "win_rate": 0.45,
                        "mean_candidate_max_drawdown_percent": 12.0,
                    },
                }
            ],
        }
    )
    assert "PRIMARY_ASYMMETRIC_1_5R" in summary
    assert "| 123 | 1.23 | 0.45 | 12.0 |" in summary
    assert "trading_authorization=NO-GO" in summary
    assert "portfolio overlap is not modeled" in summary


def test_v4v_economic_outcomes_reconstruct_barrier_exit_returns() -> None:
    bars = DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "close": [101.0, 102.0, 103.0, 104.0],
        }
    )
    labels = DataFrame(
        {
            "event_index": [0, 1, 2],
            "label_end_idx": [1, 2, 3],
            "source_label": ["BUY", "SELL", "HOLD"],
            "take_profit_price": [102.0, 101.0, 105.0],
            "stop_loss_price": [98.0, 99.0, 95.0],
            "exit_reason": ["TP_HIT", "SL_HIT", "HORIZON_EXPIRY"],
        }
    )
    outcomes = v4v._economic_outcomes_by_index(bars, labels)
    assert outcomes.iloc[0] == pytest.approx(0.02)
    assert outcomes.iloc[1] == pytest.approx(-0.01)
    assert outcomes.iloc[2] == pytest.approx(0.04)
    assert pd.isna(outcomes.iloc[3])


def test_v4v_trade_metrics_reconcile_net_gross_costs_and_profit_factor() -> None:
    trades = [
        {
            "net_pnl_usd": 4.0,
            "gross_market_pnl_usd": 4.5,
            "fees_usd": 0.25,
            "slippage_cost_usd": 0.25,
        },
        {
            "net_pnl_usd": -1.0,
            "gross_market_pnl_usd": -0.5,
            "fees_usd": 0.25,
            "slippage_cost_usd": 0.25,
        },
    ]
    metrics = v4v._trade_metrics(trades)
    assert metrics["trade_count"] == 2
    assert metrics["winning_trade_count"] == 1
    assert metrics["losing_trade_count"] == 1
    assert metrics["win_rate"] == pytest.approx(0.5)
    assert metrics["net_profit_factor"] == pytest.approx(4.0)
    assert metrics["gross_profit_factor"] == pytest.approx(9.0)
    assert metrics["net_pnl_usd"] == pytest.approx(3.0)
    assert metrics["gross_market_pnl_usd"] == pytest.approx(4.0)
    assert metrics["fees_usd"] == pytest.approx(0.5)
    assert metrics["slippage_cost_usd"] == pytest.approx(0.5)
    assert metrics["net_profit_factor_infinite_no_losses"] is False


def test_v4v_arm_aggregate_keeps_skips_out_of_economic_totals() -> None:
    trade = {
        "net_pnl_usd": 4.0,
        "gross_market_pnl_usd": 4.5,
        "fees_usd": 0.25,
        "slippage_cost_usd": 0.25,
    }
    candidates: list[dict[str, Any]] = [
        {
            "arm": "PRIMARY_ASYMMETRIC_1_5R",
            "status": "TRAINED",
            "trades": [trade],
            "metrics": {
                "max_drawdown_percent": 7.5,
                "economic_candidate_gate_passed": True,
            },
        },
        {
            "arm": "PRIMARY_ASYMMETRIC_1_5R",
            "status": "SKIPPED_INSUFFICIENT_SAMPLES",
            "trades": [],
            "metrics": {},
        },
    ]
    aggregate = v4v._arm_aggregate(candidates, "PRIMARY_ASYMMETRIC_1_5R")
    assert aggregate["candidate_count"] == 2
    assert aggregate["trained_candidate_count"] == 1
    assert aggregate["skipped_candidate_count"] == 1
    assert aggregate["trade_count"] == 1
    assert aggregate["net_pnl_usd"] == pytest.approx(4.0)
    assert aggregate["mean_candidate_max_drawdown_percent"] == pytest.approx(7.5)
    assert aggregate["economic_pass_candidate_count"] == 1


def test_v4v_geometry_contract_is_paired_control_and_primary() -> None:
    assert {
        (
            float(arm["take_profit_multiplier"]),
            float(arm["stop_loss_multiplier"]),
            int(arm["horizon_candles"]),
        )
        for arm in v4v.V4V_ARMS
    } == {(1.0, 1.0, 24), (1.5, 1.0, 24)}
    assert all(arm["take_profit_multiplier"] != 2.0 for arm in v4v.V4V_ARMS)


def test_v4v_safety_contract_stays_closed() -> None:
    safety = {
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
        "private_api_calls": 0,
        "real_order_attempts": 0,
        "short_positions_supported": False,
        "sequence_model_used": False,
    }
    assert safety["Champion"] is None
    assert safety["order_permission"] is False
    assert safety["trading_authorization"] == "NO-GO"
    assert safety["private_api_calls"] == 0
    assert safety["real_order_attempts"] == 0
    assert safety["short_positions_supported"] is False
    assert safety["sequence_model_used"] is False


def test_v4v_workflow_is_branch_scoped_and_waits_for_exact_sha_ci() -> None:
    workflow_text = Path(".github/workflows/ml-core-v4v-binary-paradigm.yml").read_text(
        encoding="utf-8"
    )
    assert "research/ml-core-v4v-binary-paradigm" in workflow_text
    assert "Require standard CI success for this exact V4-V commit" in workflow_text
    assert "TARGET_SHA" in workflow_text
    assert "Binary_BUY_Economic_Score" in workflow_text
    assert "alix-v4v-row-level-probabilities.csv" in workflow_text
    assert "trading_authorization=NO-GO" in workflow_text



def test_dynamic_probability_calibration_selects_top_fraction_and_reports_ties() -> None:
    result = select_dynamic_buy_threshold(
        np.asarray([0.10, 0.20, 0.30, 0.80, 0.80, 0.90], dtype=float),
        target_buy_fraction=0.34,
    )
    assert result["threshold"] == pytest.approx(0.80)
    assert result["target_selected_rows"] == 3
    assert result["selected_calibration_rows"] == 3
    assert result["selected_calibration_fraction"] == pytest.approx(0.50)
    assert result["tie_expansion_rows"] == 0


def test_dynamic_probability_calibration_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="binary_calibration_probabilities_invalid"):
        select_dynamic_buy_threshold(np.asarray([], dtype=float))
    with pytest.raises(ValueError, match="binary_calibration_probabilities_invalid"):
        select_dynamic_buy_threshold(np.asarray([0.2, np.nan], dtype=float))
    with pytest.raises(ValueError, match="binary_calibration_target_fraction_invalid"):
        select_dynamic_buy_threshold(np.asarray([0.2, 0.3]), target_buy_fraction=0.0)


def test_dynamic_probability_calibration_reports_tie_expansion() -> None:
    result = select_dynamic_buy_threshold(
        np.asarray([0.20, 0.80, 0.80, 0.80], dtype=float),
        target_buy_fraction=0.25,
    )
    assert result["target_selected_rows"] == 1
    assert result["selected_calibration_rows"] == 3
    assert result["tie_expansion_rows"] == 2


def test_binary_calibration_split_is_purged_and_chronological() -> None:
    outer_train = list(range(100, 1100))
    label_ends = [min(index + 12, 1199) for index in range(1200)]
    fit, calibration, diagnostics = v4v._binary_calibration_split(
        outer_train, label_ends, horizon_candles=24
    )
    assert fit
    assert calibration
    assert fit[-1] < calibration[0]
    assert diagnostics["maximum_fit_label_end_index"] < calibration[0]
    assert diagnostics["fit_label_windows_end_before_calibration"] is True
