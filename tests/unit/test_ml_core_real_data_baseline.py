from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series

from services.ml_core.adaptive_targets import (
    STAGE1_TARGET_COLUMN,
    STAGE2_TARGET_COLUMN,
)

from scripts.run_ml_core_real_data_baseline import (
    _build_diagnostic_report,
    _new_classifier,
    _new_binary_classifier,
    binary_probability_matrix,
    binary_classification_metrics,
    combine_hierarchical_probabilities,
    hierarchical_sample_gate_diagnostics,
    _default_barrier_scenarios,
    _validate_barrier_scenarios,
    baseline_validation_diagnostics,
    classification_metrics,
    exit_reason_distribution,
    label_audit_diagnostics,
    label_distribution,
    purged_temporal_indices,
    resample_ohlcv,
    run_experiment,
    sample_gate_diagnostics,
    validation_classification_diagnostics,
    hold_class_metrics,
    hold_threshold_predictions,
    optimize_hold_threshold,
    ordered_probability_matrix,
    predictions_from_probabilities,
    split_validation_for_calibration,
    threshold_optimization_diagnostics,
    probability_quality_metrics,
    probability_diagnostics_by_temporal_segment,
    weighting_policy_specs,
    _serialize_validation_probabilities,
)


def minute_frame(rows: int = 65) -> DataFrame:
    dates = pd.date_range("2026-01-01T00:00:00Z", periods=rows, freq="min")
    opened = 100.0 + np.arange(rows, dtype=float) * 0.01
    close = opened + 0.002
    return DataFrame(
        {
            "date": dates,
            "open": opened,
            "high": close + 0.01,
            "low": opened - 0.01,
            "close": close,
            "volume": np.ones(rows, dtype=float),
        }
    )


def test_classifier_factory_supports_unweighted_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

    class FakeClassifier:
        def __init__(self, **kwargs: object) -> None:
            captured.append(kwargs)

    monkeypatch.setattr(
        "scripts.run_ml_core_real_data_baseline.importlib.import_module",
        lambda _: SimpleNamespace(LGBMClassifier=FakeClassifier),
    )

    _new_classifier()
    _new_classifier(class_weight=None)

    assert captured[0]["class_weight"] == "balanced"
    assert captured[1]["class_weight"] is None
    assert captured[0]["random_state"] == captured[1]["random_state"] == 42


def test_resample_uses_only_complete_candles() -> None:
    bars = resample_ohlcv(minute_frame(), "15m")

    assert len(bars) == 4
    assert bars["date"].iloc[0] == pd.Timestamp("2026-01-01T00:00:00Z")
    assert bars["date"].iloc[-1] == pd.Timestamp("2026-01-01T00:45:00Z")
    assert (bars["volume"] == 15.0).all()


def test_resample_rejects_minute_gaps() -> None:
    candles = minute_frame().drop(index=17).reset_index(drop=True)

    with pytest.raises(ValueError, match="minute_data_gap_or_interval_mismatch"):
        resample_ohlcv(candles, "15m")


def test_resample_rejects_disallowed_entry_timeframe() -> None:
    with pytest.raises(ValueError, match="entry_timeframe_out_of_policy"):
        resample_ohlcv(minute_frame(), "1d")


def test_label_distribution_reports_fixed_label_contract() -> None:
    distribution = label_distribution(
        Series(["BUY", "BUY", "SELL", "HOLD", "unlabeled"], dtype=object)
    )

    assert distribution["total"] == 4
    assert distribution["classes"]["BUY"] == {"count": 2, "percent": 50.0}
    assert distribution["classes"]["SELL"] == {"count": 1, "percent": 25.0}
    assert distribution["classes"]["HOLD"] == {"count": 1, "percent": 25.0}


def test_classification_metrics_include_each_class_and_macro_scores() -> None:
    metrics = classification_metrics(
        Series(["BUY", "SELL", "HOLD"]),
        Series(["BUY", "BUY", "HOLD"]),
    )

    assert set(metrics["per_class"]) == {"BUY", "SELL", "HOLD"}
    assert metrics["per_class"]["BUY"]["support"] == 1
    assert 0.0 <= metrics["macro"]["f1"] <= 1.0
    assert 0.0 <= metrics["macro"]["precision"] <= 1.0
    assert 0.0 <= metrics["macro"]["recall"] <= 1.0


def test_sample_gate_requires_minimum_support_for_every_class() -> None:
    train = Series(["BUY"] * 20 + ["SELL"] * 20 + ["HOLD"] * 20)
    validation = Series(["BUY"] * 30 + ["SELL"] * 30 + ["HOLD"] * 30)

    assert sample_gate_diagnostics(train, validation)["status"] == "PASS"

    low_train = Series(["BUY"] * 19 + ["SELL"] * 20 + ["HOLD"] * 20)
    train_result = sample_gate_diagnostics(low_train, validation)
    assert train_result["status"] == "SKIPPED_INSUFFICIENT_SAMPLES"
    assert train_result["insufficient_train_classes"] == ["BUY"]

    low_validation = Series(["BUY"] * 30 + ["SELL"] * 30 + ["HOLD"] * 29)
    validation_result = sample_gate_diagnostics(train, low_validation)
    assert validation_result["status"] == "SKIPPED_INSUFFICIENT_SAMPLES"
    assert validation_result["insufficient_validation_classes"] == ["HOLD"]

    missing_class = Series(["BUY"] * 20 + ["SELL"] * 20)
    diversity_result = sample_gate_diagnostics(missing_class, validation)
    assert diversity_result["status"] == "SKIPPED_INSUFFICIENT_CLASS_DIVERSITY"
    assert diversity_result["missing_train_classes"] == ["HOLD"]


def test_validation_diagnostics_report_fixed_confusion_matrix_and_predictions() -> None:
    actual = Series(["BUY", "SELL", "HOLD", "HOLD"])
    predicted = Series(["BUY", "BUY", "SELL", "HOLD"])

    diagnostics = validation_classification_diagnostics(actual, predicted)

    assert diagnostics["confusion_matrix"]["labels"] == ["BUY", "SELL", "HOLD"]
    assert diagnostics["confusion_matrix"]["matrix"] == [
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 1],
    ]
    assert diagnostics["prediction_distribution"]["total"] == 4
    assert diagnostics["prediction_distribution"]["classes"]["BUY"] == {
        "count": 2,
        "percent": 50.0,
    }


def test_exit_reason_distribution_is_fixed_and_rejects_unknown_reasons() -> None:
    distribution = exit_reason_distribution(
        Series(["TP_HIT", "SL_HIT", "COLLISION", "HORIZON_EXPIRY", None])
    )

    assert distribution["total"] == 4
    assert distribution["reasons"]["TP_HIT"] == {"count": 1, "percent": 25.0}
    assert distribution["reasons"]["SL_HIT"]["count"] == 1
    assert distribution["reasons"]["COLLISION"]["count"] == 1
    assert distribution["reasons"]["HORIZON_EXPIRY"]["count"] == 1
    with pytest.raises(ValueError, match="exit_reason_out_of_policy"):
        exit_reason_distribution(Series(["take_profit"]))


def test_label_audit_separates_exit_and_unlabeled_reasons() -> None:
    labels = DataFrame(
        {
            "label": ["BUY", "SELL", "HOLD", None],
            "exit_reason": ["TP_HIT", "SL_HIT", "HORIZON_EXPIRY", None],
            "unlabeled_reason": [None, None, None, "incomplete_horizon"],
        }
    )

    audit = label_audit_diagnostics(labels)

    assert audit["scope"] == "ALL_TARGET_BARS_BEFORE_SPLIT"
    assert audit["labeled_events"] == 3
    assert audit["unlabeled_events"] == 1
    assert audit["exit_reason_distribution"]["total"] == 3
    assert (
        audit["unlabeled_reason_distribution"]["reasons"]["incomplete_horizon"]["count"]
        == 1
    )


def test_baseline_validation_diagnostics_use_fixed_class_order() -> None:
    actual = Series(["BUY", "SELL", "HOLD", "SELL"])
    predicted = Series(["SELL"] * len(actual))

    baseline = baseline_validation_diagnostics(actual, predicted)

    assert baseline["classification_metrics"]["per_class"]["SELL"]["support"] == 2
    assert baseline["prediction_distribution"]["classes"]["SELL"]["count"] == 4
    assert baseline["confusion_matrix"]["matrix"] == [
        [0, 1, 0],
        [0, 2, 0],
        [0, 1, 0],
    ]


def test_temporal_split_keeps_training_before_validation_and_embargoes_horizon() -> (
    None
):
    row_count = 120
    labels: list[tuple[str | None, int | None]] = [
        ("BUY", index + 24) for index in range(row_count - 24)
    ]
    labels.extend((None, None) for _ in range(24))
    label_frame = DataFrame(
        {
            "label": [row[0] for row in labels],
            "label_end_idx": [row[1] for row in labels],
        }
    )
    feature_frame = DataFrame({"feature": np.arange(row_count, dtype=float)})

    train, validation, diagnostics = purged_temporal_indices(feature_frame, label_frame)

    assert train
    assert validation
    assert max(train) < diagnostics["validation_start_index"]
    assert min(validation) >= diagnostics["validation_start_index"]
    assert max(validation) <= diagnostics["validation_end_index"]
    assert set(train).isdisjoint(validation)
    assert diagnostics["embargo_size_bars"] == 24
    assert diagnostics["label_horizon_candles"] == 24
    assert diagnostics["purged_train_rows"] == 0
    assert diagnostics["eligible_pre_validation_embargo_rows"] == 24


def test_short_real_data_window_is_skipped_without_training() -> None:
    result = run_experiment("BTC/USDT", "4h", minute_frame(rows=24 * 60))

    assert result["status"] == "SKIPPED_INSUFFICIENT_SAMPLES"
    assert result["bar_count"] == 6
    assert result["train_rows"] == 0
    assert result["validation_rows"] == 0
    assert result["classification_metrics"] is None
    assert result["split"]["status"] == "NOT_SPLIT_INSUFFICIENT_BARS"
    assert result["label_distribution"]["available_unpartitioned"]["total"] == 0


def test_weighting_policy_specs_expose_only_none_and_balanced() -> None:
    policies = weighting_policy_specs()

    assert set(policies) == {"weight_none", "weight_balanced"}
    assert policies["weight_none"] == {"class_weight": None}
    assert policies["weight_balanced"] == {"class_weight": "balanced"}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("take_profit_multiplier", None),
        ("take_profit_multiplier", True),
        ("take_profit_multiplier", 0.0),
        ("take_profit_multiplier", float("nan")),
        ("stop_loss_multiplier", None),
        ("stop_loss_multiplier", True),
        ("stop_loss_multiplier", 0.0),
        ("stop_loss_multiplier", float("inf")),
    ],
)
def test_barrier_scenario_validation_rejects_invalid_multipliers(
    field: str, value: Any
) -> None:
    scenario: dict[str, Any] = {
        "name": "Baseline",
        "horizon_candles": 24,
        "take_profit_multiplier": 2.0,
        "stop_loss_multiplier": 1.0,
    }
    scenario[field] = value

    with pytest.raises(ValueError, match=f"{field}_must_be_finite_and_positive"):
        _validate_barrier_scenarios([scenario])


def test_probability_matrix_is_reordered_to_fixed_label_contract() -> None:
    class FakeModel:
        classes_ = np.asarray(["BUY", "HOLD", "SELL"])

        def predict_proba(self, _: DataFrame) -> np.ndarray:
            return np.asarray([[0.2, 0.7, 0.1]])

    probabilities = ordered_probability_matrix(
        FakeModel(), DataFrame({"feature": [1.0]})
    )

    assert probabilities.tolist() == [[0.2, 0.1, 0.7]]


def test_predictions_from_probabilities_uses_fixed_label_order() -> None:
    probabilities = np.asarray([[0.2, 0.1, 0.7], [0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])

    predicted = predictions_from_probabilities(probabilities)

    assert predicted.tolist() == ["HOLD", "SELL", "BUY"]


def test_hold_threshold_filters_only_low_confidence_argmax_hold_predictions() -> None:
    probabilities = np.asarray(
        [
            [0.20, 0.20, 0.60],
            [0.10, 0.75, 0.15],
            [0.45, 0.35, 0.20],
        ]
    )

    predicted = hold_threshold_predictions(probabilities, 0.70)

    assert predicted.tolist() == ["BUY", "SELL", "BUY"]


def test_hold_threshold_optimizer_meets_precision_and_recall_constraints() -> None:
    actual = Series(
        ["HOLD", "HOLD", "HOLD", "HOLD", "BUY", "SELL", "BUY", "SELL", "BUY", "SELL"]
    )
    probabilities = np.asarray(
        [
            [0.02, 0.03, 0.95],
            [0.10, 0.15, 0.75],
            [0.20, 0.30, 0.50],
            [0.60, 0.20, 0.20],
            [0.05, 0.07, 0.88],
            [0.15, 0.15, 0.70],
            [0.70, 0.20, 0.10],
            [0.20, 0.70, 0.10],
            [0.80, 0.10, 0.10],
            [0.10, 0.80, 0.10],
        ]
    )

    result = optimize_hold_threshold(
        actual, probabilities, minimum_precision=0.40, minimum_recall=0.20
    )

    assert result["status"] == "SELECTED"
    assert result["selected_threshold"] is not None
    selected = result["selected_calibration_metrics"]
    assert selected["precision"] >= 0.40
    assert selected["recall"] >= 0.20
    assert result["feasible_threshold_count"] > 0


def test_hold_threshold_optimizer_fails_closed_when_constraints_are_infeasible() -> (
    None
):
    actual = Series(["HOLD", "HOLD", "BUY", "SELL"])
    probabilities = np.asarray(
        [
            [0.60, 0.30, 0.10],
            [0.60, 0.30, 0.10],
            [0.10, 0.10, 0.80],
            [0.10, 0.10, 0.80],
        ]
    )

    result = optimize_hold_threshold(
        actual, probabilities, minimum_precision=0.90, minimum_recall=0.90
    )

    assert result["status"] == "NO_FEASIBLE_THRESHOLD"
    assert result["selected_threshold"] is None
    assert result["selected_calibration_metrics"] is None


def test_calibration_split_uses_earlier_rows_for_forward_time_evaluation() -> None:
    indices = list(range(100, 110))

    calibration, forward = split_validation_for_calibration(indices, 0.20)

    assert calibration == [100, 101]
    assert forward == [102, 103, 104, 105, 106, 107, 108, 109]
    assert max(calibration) < min(forward)


@pytest.mark.parametrize("invalid", [0.0, 0.5, 1.0, float("nan"), True])
def test_calibration_split_rejects_invalid_fractions(invalid: float) -> None:
    with pytest.raises(
        ValueError, match="calibration_fraction_must_be_between_zero_and_half"
    ):
        split_validation_for_calibration([1, 2, 3, 4], invalid)


def test_threshold_diagnostics_evaluate_later_validation_rows_without_overlap() -> None:
    actual_values = ["HOLD" if index % 2 == 0 else "BUY" for index in range(300)]
    actual = Series(actual_values, index=range(1000, 1300))
    probabilities = np.asarray(
        [
            [0.05, 0.05, 0.90] if label == "HOLD" else [0.80, 0.10, 0.10]
            for label in actual_values
        ]
    )

    result = threshold_optimization_diagnostics(
        actual, probabilities, calibration_fraction=0.20
    )

    assert result["status"] == "SELECTED"
    assert result["calibration"]["row_count"] == 60
    assert result["calibration"]["start_row_index"] == 1000
    assert result["calibration"]["end_row_index"] == 1059
    assert result["post_calibration_validation"]["row_count"] == 240
    assert result["post_calibration_validation"]["calibration_overlap"] is False
    assert (
        result["full_validation_including_calibration"]["calibration_overlap"] is True
    )
    selected = result["calibration"]["selected_calibration_metrics"]["hold_metrics"]
    assert selected["precision"] >= 0.40
    assert selected["recall"] >= 0.20


def test_threshold_diagnostics_skip_small_calibration_support() -> None:
    actual = Series(["HOLD", "BUY", "SELL", "HOLD", "BUY"] * 20)
    probabilities = np.tile(np.asarray([[0.2, 0.2, 0.6]]), (len(actual), 1))

    result = threshold_optimization_diagnostics(
        actual, probabilities, calibration_fraction=0.20
    )

    assert result["status"] == "SKIPPED_INSUFFICIENT_CALIBRATION_SAMPLES"
    assert result["selected_threshold"] is None
    assert result["post_calibration_validation"] is None


def test_probability_serialization_preserves_row_timestamp_and_class_order() -> None:
    bars = DataFrame(
        {"date": pd.date_range("2026-01-01T00:00:00Z", periods=2, freq="15min")}
    )
    actual = Series(["BUY", "HOLD"])
    probabilities = np.asarray([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]])

    records = _serialize_validation_probabilities(bars, [0, 1], actual, probabilities)

    assert records[0]["row_index"] == 0
    assert records[0]["actual_label"] == "BUY"
    assert records[1]["timestamp"] == "2026-01-01T00:15:00+00:00"
    assert records[1]["probabilities"] == {"BUY": 0.1, "SELL": 0.2, "HOLD": 0.7}


def test_hold_class_metrics_count_false_positives_and_false_negatives() -> None:
    actual = Series(["HOLD", "BUY", "SELL", "HOLD"])
    predicted = Series(["HOLD", "HOLD", "SELL", "BUY"])

    result = hold_class_metrics(actual, predicted)

    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5


def test_diagnostic_report_unweighted_policy_uses_top_level_class_weight_field() -> (
    None
):
    report = _build_diagnostic_report(
        days=730,
        commit_sha="a" * 40,
        provenance=[
            {
                "fetch_page_size_candles": 1000,
                "inter_page_delay_seconds": 0.25,
            }
        ],
        experiments=[],
        barrier_scenarios=[
            {
                "name": "Baseline",
                "horizon_candles": 24,
                "take_profit_multiplier": 2.0,
                "stop_loss_multiplier": 1.0,
            },
            {
                "name": "Short_Horizon",
                "horizon_candles": 12,
                "take_profit_multiplier": 2.0,
                "stop_loss_multiplier": 1.0,
            },
        ],
        calibration_fraction=0.20,
        minimum_hold_precision=0.40,
        minimum_hold_recall=0.20,
        policy_specs=weighting_policy_specs(),
    )

    unweighted = report["baseline_comparisons"]["lightgbm_unweighted"]
    assert unweighted["class_weight"] is None
    assert "model_parameters" not in unweighted


def test_probability_quality_metrics_compute_brier_ece_auc_and_average_precision() -> (
    None
):
    actual = Series(["HOLD", "HOLD", "BUY", "SELL"])
    probabilities = np.asarray(
        [
            [0.05, 0.05, 0.90],
            [0.10, 0.10, 0.80],
            [0.80, 0.10, 0.10],
            [0.10, 0.80, 0.10],
        ]
    )
    result = probability_quality_metrics(actual, probabilities)

    assert result["sample_count"] == 4
    assert result["support"] == {"hold_positive": 2, "non_hold_negative": 2}
    assert result["brier_score"]["multiclass"] == pytest.approx(0.04875, abs=1e-8)
    assert result["brier_score"]["hold_binary"] == pytest.approx(0.0175, abs=1e-8)
    assert result["ece"]["hold_binary"] is not None
    assert result["ece"]["multiclass_macro"] is not None
    assert result["hold_auc"] == 1.0
    assert result["hold_pr_auc"] == 1.0


@pytest.mark.parametrize(
    "calibration_bins",
    [True, 0, 1, 1.5],
)
def test_probability_quality_metrics_reject_invalid_calibration_bins(
    calibration_bins: Any,
) -> None:
    actual = Series(["HOLD", "BUY"])
    probabilities = np.asarray([[0.1, 0.1, 0.8], [0.8, 0.1, 0.1]])

    with pytest.raises(
        ValueError, match="calibration_bins_must_be_integer_at_least_two"
    ):
        probability_quality_metrics(
            actual, probabilities, calibration_bins=calibration_bins
        )


def test_probability_quality_metrics_reject_invalid_probability_shape() -> None:
    with pytest.raises(ValueError, match="classifier_probability_shape_invalid"):
        probability_quality_metrics(
            Series(["HOLD", "BUY"]), np.asarray([[0.1, 0.1, 0.8]])
        )


def test_probability_quality_metrics_reject_unknown_actual_label() -> None:
    with pytest.raises(ValueError, match="probability_metrics_labels_invalid"):
        probability_quality_metrics(
            Series(["HOLD", "UNKNOWN"]), np.asarray([[0.1, 0.1, 0.8], [0.8, 0.1, 0.1]])
        )


def test_barrier_scenario_validation_rejects_empty_and_duplicate_names() -> None:
    with pytest.raises(ValueError, match="barrier_scenarios_empty"):
        _validate_barrier_scenarios([])

    scenario = {
        "name": "Baseline",
        "horizon_candles": 24,
        "take_profit_multiplier": 2.0,
        "stop_loss_multiplier": 1.0,
    }
    with pytest.raises(ValueError, match="barrier_scenario_name_missing_or_duplicate"):
        _validate_barrier_scenarios([scenario, scenario.copy()])


@pytest.mark.parametrize("horizon", [True, 0, -1, 1.5])
def test_barrier_scenario_validation_rejects_invalid_horizon(horizon: Any) -> None:
    scenario = {
        "name": "Baseline",
        "horizon_candles": horizon,
        "take_profit_multiplier": 2.0,
        "stop_loss_multiplier": 1.0,
    }
    with pytest.raises(ValueError, match="horizon_candles_must_be_positive_integer"):
        _validate_barrier_scenarios([scenario])


def test_probability_quality_metrics_return_null_auc_without_both_hold_classes() -> (
    None
):
    actual = Series(["BUY", "SELL", "BUY"])
    probabilities = np.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])
    result = probability_quality_metrics(actual, probabilities)

    assert result["hold_auc"] is None
    assert result["hold_pr_auc"] is None
    assert result["brier_score"]["hold_binary"] is not None


def test_probability_metrics_use_disjoint_calibration_and_forward_rows() -> None:
    actual = Series(["HOLD", "BUY", "HOLD", "SELL", "BUY"] * 20)
    probabilities = np.tile(np.asarray([[0.2, 0.2, 0.6]]), (len(actual), 1))
    result = probability_diagnostics_by_temporal_segment(
        actual, probabilities, calibration_fraction=0.20
    )

    assert result["calibration"]["sample_count"] == 20
    assert result["forward_validation"]["sample_count"] == 80
    assert result["calibration_overlap"] is False



def test_binary_classifier_factory_uses_deterministic_binary_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

    class FakeClassifier:
        def __init__(self, **kwargs: object) -> None:
            captured.append(kwargs)

    monkeypatch.setattr(
        "scripts.run_ml_core_real_data_baseline.importlib.import_module",
        lambda _: SimpleNamespace(LGBMClassifier=FakeClassifier),
    )
    _new_binary_classifier()
    _new_binary_classifier(class_weight=None)
    assert captured[0]["objective"] == "binary"
    assert captured[0]["class_weight"] == "balanced"
    assert captured[1]["class_weight"] is None
    assert captured[0]["random_state"] == 42


def test_binary_probability_matrix_reorders_classes_to_zero_one() -> None:
    class FakeModel:
        classes_ = np.asarray([1, 0])

        @staticmethod
        def predict_proba(_: DataFrame) -> np.ndarray:
            return np.asarray([[0.7, 0.3], [0.2, 0.8]])

    probabilities = binary_probability_matrix(
        FakeModel(), DataFrame({"feature": [1.0, 2.0]})
    )
    assert probabilities.tolist() == pytest.approx([[0.3, 0.7], [0.8, 0.2]])


def test_hierarchical_probability_composition_is_normalized() -> None:
    stage1 = np.asarray([[0.0, 0.8], [0.0, 0.3]])
    stage2 = np.asarray([[0.25, 0.75], [0.6, 0.4]])
    result = combine_hierarchical_probabilities(stage1, stage2)
    assert result.tolist() == pytest.approx([[0.60, 0.20, 0.20], [0.12, 0.18, 0.70]])
    assert result.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])


@pytest.mark.parametrize(
    "stage1,stage2",
    [
        (np.asarray([[0.5, 0.5]]), np.asarray([[1.0, 0.0], [0.0, 1.0]])),
        (np.asarray([[0.5, 0.5, 0.0]]), np.asarray([[1.0, 0.0]])),
    ],
)
def test_hierarchical_probability_composition_rejects_invalid_shapes(
    stage1: np.ndarray, stage2: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="hierarchical_probability_shape_invalid"):
        combine_hierarchical_probabilities(stage1, stage2)


def test_binary_metrics_report_precision_recall_f1_for_each_stage() -> None:
    actual = Series([0, 0, 1, 1])
    predicted = Series([0, 1, 1, 1])
    result = binary_classification_metrics(
        actual, predicted, negative_label="NO_HIT", positive_label="HIT"
    )
    assert result["per_class"]["NO_HIT"]["support"] == 2
    assert result["macro"]["precision"] == pytest.approx(0.833333, abs=1e-6)
    assert result["macro"]["recall"] == pytest.approx(0.75, abs=1e-6)
    assert result["macro"]["f1"] == pytest.approx(0.733333, abs=1e-6)


def test_hierarchical_sample_gate_requires_both_stage_two_classes() -> None:
    stage1_train = Series([0] * 30 + [1] * 30)
    stage1_validation = Series([0] * 35 + [1] * 35)
    stage2_train = Series([0] * 19 + [1] * 25)
    stage2_validation = Series([0] * 31 + [1] * 31)
    failed = hierarchical_sample_gate_diagnostics(
        stage1_train, stage1_validation, stage2_train, stage2_validation
    )
    assert failed["status"] == "SKIPPED_INSUFFICIENT_SAMPLES"
    assert failed["stage2_direction_classifier"]["insufficient_train_classes"] == ["SELL"]

    passed = hierarchical_sample_gate_diagnostics(
        stage1_train, stage1_validation,
        Series([0] * 20 + [1] * 25),
        stage2_validation,
    )
    assert passed["status"] == "PASS"


def test_default_scenario_is_exactly_one_12_candle_hierarchical_target() -> None:
    scenarios = _default_barrier_scenarios()
    assert len(scenarios) == 1
    assert scenarios[0] == {
        "name": "Hierarchical_12_Candle",
        "horizon_candles": 12,
        "take_profit_multiplier": 2.0,
        "stop_loss_multiplier": 1.0,
    }


def test_run_experiment_fits_both_binary_models_and_reports_all_stage_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = minute_frame(15 * 15)
    bars = resample_ohlcv(frame, "15m")
    # Create repeated, deterministic first hits in the 12-candle label horizon.
    for index in range(1, len(bars), 4):
        if index < len(bars):
            bars.loc[index, "high"] = bars.loc[index, "close"] + 3.0
    monkeypatch.setattr(
        "scripts.run_ml_core_real_data_baseline.resample_ohlcv",
        lambda _candles, _timeframe: bars,
    )
    monkeypatch.setattr(
        "scripts.run_ml_core_real_data_baseline.build_causal_feature_matrix",
        lambda data: DataFrame(
            {
                "feature_a": np.linspace(0.0, 1.0, len(data)),
                "feature_b": np.linspace(1.0, 0.0, len(data)),
            }
        ),
    )
    monkeypatch.setattr(
        "scripts.run_ml_core_real_data_baseline.purged_temporal_indices",
        lambda features, labels, *, horizon_candles: (
            list(range(0, 100)),
            list(range(120, 190)),
            {"status": "PASS", "embargo_size_bars": horizon_candles},
        ),
    )
    fits: list[int] = []

    class FakeBinaryClassifier:
        def __init__(self, class_weight: str | None) -> None:
            self.class_weight = class_weight

        def fit(self, x: DataFrame, y: Series) -> None:
            fits.append(len(x))
            assert len(x) == len(y)
            self.classes_ = np.asarray([0, 1])

        def predict_proba(self, x: DataFrame) -> np.ndarray:
            hit = 0.45 if self.class_weight is None else 0.5
            return np.tile(np.asarray([[1.0 - hit, hit]]), (len(x), 1))

    monkeypatch.setattr(
        "scripts.run_ml_core_real_data_baseline._new_binary_classifier",
        lambda class_weight: FakeBinaryClassifier(class_weight),
    )
    # Ensure both hit and no-hit samples are abundant in both splits.
    monkeypatch.setattr(
        "scripts.run_ml_core_real_data_baseline.apply_adaptive_hierarchical_targets",
        lambda data, **kwargs: DataFrame(
            {
                "label": ["BUY" if i % 4 == 0 else "SELL" if i % 4 == 1 else "HOLD" for i in range(len(data))],
                "exit_reason": ["TP_HIT" if i % 4 == 0 else "SL_HIT" if i % 4 == 1 else "HORIZON_EXPIRY" for i in range(len(data))],
                "unlabeled_reason": [None] * len(data),
                "label_end_idx": [min(i + 12, len(data) - 1) for i in range(len(data))],
                STAGE1_TARGET_COLUMN: [1.0 if i % 4 in (0, 1) else 0.0 for i in range(len(data))],
                STAGE2_TARGET_COLUMN: [1.0 if i % 4 == 0 else 0.0 if i % 4 == 1 else np.nan for i in range(len(data))],
            }
        ),
    )
    # Fake classifier training is intentionally test-only; no real estimator is constructed.
    result = run_experiment(
        "BTC/USDT",
        "15m",
        frame,
        scenario_name="Hierarchical_12_Candle",
        horizon_candles=12,
    )
    assert result["status"] == "TRAINED"
    assert len(fits) == 4  # two stages times two existing weighting policies
    assert result["hierarchical_metrics"]["stage1_hit_classifier"]["validation"]
    assert result["hierarchical_metrics"]["stage2_direction_classifier"]["validation"]
    assert result["hierarchical_metrics"]["combined_three_class"]["validation"]
    for variant in result["model_variants"].values():
        for row in variant["validation_probabilities"]:
            assert sum(row["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
