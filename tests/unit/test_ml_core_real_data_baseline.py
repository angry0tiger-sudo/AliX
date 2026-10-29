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


def test_temporal_split_keeps_training_before_validation_and_purges() -> None:
    row_count = 120
    labels = [("BUY", index + 24) for index in range(row_count - 24)]
    labels.extend((None, None) for _ in range(24))
    label_frame = DataFrame(
        {
            "label": [row[0] for row in labels],
            "label_end_idx": [row[1] for row in labels],
        }
    )
    feature_frame = DataFrame({"feature": np.arange(row_count, dtype=float)})

    train, validation, diagnostics = purged_temporal_indices(
        feature_frame, label_frame
    )

    assert train
    assert validation
    assert max(train) < diagnostics["validation_start_index"]
    assert min(validation) >= diagnostics["validation_start_index"]
    assert max(validation) <= diagnostics["validation_end_index"]
    assert set(train).isdisjoint(validation)
    assert diagnostics["embargo_size_bars"] >= 14
    assert diagnostics["purged_train_rows"] > 0
    assert diagnostics["eligible_pre_validation_embargo_rows"] > 0
