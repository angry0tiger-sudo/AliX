from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series

from scripts.run_ml_core_real_data_baseline import (
    _build_diagnostic_report,
    _new_classifier,
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


def test_temporal_split_keeps_training_before_validation_and_purges() -> None:
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
    assert diagnostics["embargo_size_bars"] >= 14
    assert diagnostics["purged_train_rows"] > 0
    assert diagnostics["eligible_pre_validation_embargo_rows"] > 0


def test_short_real_data_window_is_skipped_without_training() -> None:
    result = run_experiment("BTC/USDT", "4h", minute_frame(rows=24 * 60))

    assert result["status"] == "SKIPPED_INSUFFICIENT_SAMPLES"
    assert result["bar_count"] == 6
    assert result["train_rows"] == 0
    assert result["validation_rows"] == 0
    assert result["classification_metrics"] is None
    assert result["split"]["status"] == "NOT_SPLIT_INSUFFICIENT_BARS"
    assert result["label_distribution"]["available_unpartitioned"]["total"] == 0


def test_weighting_policy_specs_apply_only_the_configured_hold_multiplier() -> None:
    policies = weighting_policy_specs(3.0)

    assert policies["weight_none"]["class_weight"] is None
    assert policies["weight_balanced"]["class_weight"] == "balanced"
    assert policies["weight_intermediate"]["class_weight"] == {
        "BUY": 1.0,
        "SELL": 1.0,
        "HOLD": 3.0,
    }
    assert policies["weight_intermediate"]["hold_weight_multiplier"] == 3.0


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("nan"), float("inf"), True])
def test_weighting_policy_rejects_invalid_hold_multipliers(invalid: float) -> None:
    with pytest.raises(
        ValueError, match="hold_weight_multiplier_must_be_finite_and_positive"
    ):
        weighting_policy_specs(invalid)


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
        hold_weight_multiplier=3.0,
        calibration_fraction=0.20,
        minimum_hold_precision=0.40,
        minimum_hold_recall=0.20,
        policy_specs=weighting_policy_specs(3.0),
    )

    unweighted = report["baseline_comparisons"]["lightgbm_unweighted"]
    assert unweighted["class_weight"] is None
    assert "model_parameters" not in unweighted
