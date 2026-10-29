from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series

from scripts.run_ml_core_real_data_baseline import (
    classification_metrics,
    label_distribution,
    purged_temporal_indices,
    resample_ohlcv,
    baseline_validation_diagnostics,
    exit_reason_distribution,
    label_audit_diagnostics,
    run_experiment,
    sample_gate_diagnostics,
    validation_classification_diagnostics,
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
    assert audit["unlabeled_reason_distribution"]["reasons"]["incomplete_horizon"][
        "count"
    ] == 1


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
