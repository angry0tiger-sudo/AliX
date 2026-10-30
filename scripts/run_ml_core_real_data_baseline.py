"""Public-market diagnostic classification with ATR labels; no economic evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn import metrics  # type: ignore[import-untyped]

from services.ml_core.adaptive_targets import (
    EXIT_REASONS,
    UNLABELED_REASONS,
    apply_adaptive_triple_barrier,
)
from services.ml_core.causal_features import build_causal_feature_matrix
from services.ml_core.purged_embargo import create_purged_embargo_split

SYMBOLS = ("BTCUSDT", "ETHUSDT")
TIMEFRAME_MINUTES = {"15m": 15, "30m": 30, "1h": 60, "4h": 240}
LABELS = ("BUY", "SELL", "HOLD")
OHLCV_COLUMNS = ("date", "open", "high", "low", "close", "volume")
FEATURE_WINDOW = 14
VALIDATION_START_FRACTION = 0.70
VALIDATION_END_FRACTION = 0.85
OVERFIT_F1_GAP_THRESHOLD = 0.15
MIN_TRAIN_SAMPLES_PER_CLASS = 20
MIN_VALIDATION_SAMPLES_PER_CLASS = 30
MIN_CALIBRATION_HOLD_SAMPLES = 30
MIN_CALIBRATION_NON_HOLD_SAMPLES = 30
DEFAULT_HOLD_WEIGHT_MULTIPLIER = 3.0
DEFAULT_CALIBRATION_FRACTION = 0.20
DEFAULT_MINIMUM_HOLD_PRECISION = 0.40
DEFAULT_MINIMUM_HOLD_RECALL = 0.20
HOLD_THRESHOLD_GRID_STEPS = 201
REPORT_SCHEMA = "alix.ml-core.real-data-classification-diagnostic.v4"


def _validated_minute_timestamps(values: Series) -> Series:
    dates = pd.to_datetime(values, errors="coerce", utc=True)
    if dates.isna().any():
        raise ValueError("timestamp_invalid")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("timestamps_must_be_unique_and_increasing")
    deltas = dates.diff().iloc[1:].dt.total_seconds().to_numpy(dtype=float)
    if len(deltas) and not bool(np.all(deltas == 60.0)):
        raise ValueError("minute_data_gap_or_interval_mismatch")
    return dates


def _validate_minute_numeric_values(result: DataFrame) -> None:
    for name in ("open", "high", "low", "close", "volume"):
        result[name] = pd.to_numeric(result[name], errors="coerce")
    values = result.loc[:, ["open", "high", "low", "close", "volume"]].to_numpy(
        dtype=float
    )
    if not bool(np.isfinite(values).all()):
        raise ValueError("ohlcv_value_not_finite")
    _validate_minute_price_geometry(result)


def _validate_minute_price_geometry(result: DataFrame) -> None:
    opened = result["open"].to_numpy(dtype=float)
    high = result["high"].to_numpy(dtype=float)
    low = result["low"].to_numpy(dtype=float)
    close = result["close"].to_numpy(dtype=float)
    volume = result["volume"].to_numpy(dtype=float)
    if bool(np.any(np.column_stack((opened, high, low, close)) <= 0)):
        raise ValueError("ohlcv_price_must_be_positive")
    if bool(np.any(volume < 0)):
        raise ValueError("ohlcv_volume_negative")
    invalid_geometry = (
        np.any(high < low)
        or np.any(high < np.maximum(opened, close))
        or np.any(low > np.minimum(opened, close))
    )
    if bool(invalid_geometry):
        raise ValueError("ohlcv_geometry_invalid")


def _validate_minute_candles(candles: DataFrame) -> DataFrame:
    missing = sorted(set(OHLCV_COLUMNS) - set(candles.columns))
    if missing:
        raise ValueError(f"ohlcv_missing_columns:{','.join(missing)}")
    if candles.empty:
        raise ValueError("ohlcv_empty")
    result = candles.loc[:, list(OHLCV_COLUMNS)].copy().reset_index(drop=True)
    result["date"] = _validated_minute_timestamps(result["date"])
    _validate_minute_numeric_values(result)
    return result


def resample_ohlcv(candles: DataFrame, timeframe: str) -> DataFrame:
    """Build target bars only from complete, gap-free one-minute public candles."""
    if timeframe not in TIMEFRAME_MINUTES:
        raise ValueError("entry_timeframe_out_of_policy")
    source = _validate_minute_candles(candles)
    minutes = TIMEFRAME_MINUTES[timeframe]
    indexed = source.set_index("date")
    rule = f"{minutes}min"
    grouped = indexed.resample(rule, origin="epoch", closed="left", label="left")
    bars = grouped.agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    counts = grouped["close"].count()
    complete = counts == minutes
    result = bars.loc[complete].dropna().reset_index()
    if result.empty:
        raise ValueError(f"no_complete_resampled_bars:{timeframe}")
    return result.loc[:, list(OHLCV_COLUMNS)]


def label_distribution(values: Series) -> dict[str, Any]:
    """Return counts and percentages for the fixed BUY/SELL/HOLD label contract."""
    observed = values.loc[values.isin(LABELS)]
    total = int(len(observed))
    counts = observed.value_counts()
    classes: dict[str, dict[str, int | float]] = {}
    for label in LABELS:
        count = int(counts.get(label, 0))
        classes[label] = {
            "count": count,
            "percent": round(100.0 * count / total, 4) if total else 0.0,
        }
    return {"total": total, "classes": classes}


def exit_reason_distribution(values: Series) -> dict[str, Any]:
    """Count only the explicitly allowed reasons for labeled target events."""
    observed = values.dropna()
    unknown = sorted(set(observed.astype(str).tolist()) - set(EXIT_REASONS))
    if unknown:
        raise ValueError("exit_reason_out_of_policy")
    total = int(len(observed))
    counts = observed.value_counts()
    reasons = {
        reason: {
            "count": int(counts.get(reason, 0)),
            "percent": round(100.0 * int(counts.get(reason, 0)) / total, 4)
            if total
            else 0.0,
        }
        for reason in EXIT_REASONS
    }
    return {"total": total, "reasons": reasons}


def unlabeled_reason_distribution(values: Series) -> dict[str, Any]:
    """Keep incomplete target-window diagnostics separate from exit reasons."""
    observed = values.dropna()
    unknown = sorted(set(observed.astype(str).tolist()) - set(UNLABELED_REASONS))
    if unknown:
        raise ValueError("unlabeled_reason_out_of_policy")
    total = int(len(observed))
    counts = observed.value_counts()
    reasons = {
        reason: {
            "count": int(counts.get(reason, 0)),
            "percent": round(100.0 * int(counts.get(reason, 0)) / total, 4)
            if total
            else 0.0,
        }
        for reason in UNLABELED_REASONS
    }
    return {"total": total, "reasons": reasons}


def label_audit_diagnostics(labels: DataFrame) -> dict[str, Any]:
    """Summarize target outcomes before splitting into train and validation."""
    exit_distribution = exit_reason_distribution(labels["exit_reason"])
    unlabeled_distribution = unlabeled_reason_distribution(labels["unlabeled_reason"])
    labeled_events = int(labels["label"].isin(LABELS).sum())
    unlabeled_events = int(labels["label"].isna().sum())
    if (
        exit_distribution["total"] != labeled_events
        or unlabeled_distribution["total"] != unlabeled_events
    ):
        raise ValueError("label_audit_count_mismatch")
    return {
        "scope": "ALL_TARGET_BARS_BEFORE_SPLIT",
        "labeled_events": labeled_events,
        "unlabeled_events": unlabeled_events,
        "exit_reason_distribution": exit_distribution,
        "unlabeled_reason_distribution": unlabeled_distribution,
    }


def classification_metrics(actual: Series, predicted: Series) -> dict[str, Any]:
    """Calculate class-wise and macro classification metrics only."""
    report = cast(
        dict[str, Any],
        metrics.classification_report(
            actual.tolist(),
            predicted.tolist(),
            labels=list(LABELS),
            target_names=list(LABELS),
            output_dict=True,
            zero_division=0,
        ),
    )
    macro = report["macro avg"]
    per_class: dict[str, dict[str, int | float]] = {}
    for label in LABELS:
        row = report[label]
        per_class[label] = {
            "precision": round(float(row["precision"]), 6),
            "recall": round(float(row["recall"]), 6),
            "f1": round(float(row["f1-score"]), 6),
            "support": int(row["support"]),
        }
    return {
        "macro": {
            "precision": round(float(macro["precision"]), 6),
            "recall": round(float(macro["recall"]), 6),
            "f1": round(float(macro["f1-score"]), 6),
        },
        "accuracy": round(float(report["accuracy"]), 6),
        "per_class": per_class,
    }


def _class_counts(values: Series) -> dict[str, int]:
    """Count each label in the fixed classification contract."""
    return {label: int((values == label).sum()) for label in LABELS}


def _classes_below_threshold(counts: dict[str, int], minimum: int) -> list[str]:
    """Return required classes with less than the requested sample count."""
    return [label for label in LABELS if counts[label] < minimum]


def sample_gate_diagnostics(
    train_labels: Series, validation_labels: Series
) -> dict[str, Any]:
    """Reject experiments without enough examples of every required class."""
    train_counts = _class_counts(train_labels)
    validation_counts = _class_counts(validation_labels)
    missing_train_classes = _classes_below_threshold(train_counts, 1)
    insufficient_train_classes = _classes_below_threshold(
        train_counts, MIN_TRAIN_SAMPLES_PER_CLASS
    )
    insufficient_validation_classes = _classes_below_threshold(
        validation_counts, MIN_VALIDATION_SAMPLES_PER_CLASS
    )

    if missing_train_classes:
        status = "SKIPPED_INSUFFICIENT_CLASS_DIVERSITY"
    elif insufficient_train_classes or insufficient_validation_classes:
        status = "SKIPPED_INSUFFICIENT_SAMPLES"
    else:
        status = "PASS"

    return {
        "status": status,
        "minimum_train_samples_per_class": MIN_TRAIN_SAMPLES_PER_CLASS,
        "minimum_validation_samples_per_class": MIN_VALIDATION_SAMPLES_PER_CLASS,
        "train_class_counts": train_counts,
        "validation_class_counts": validation_counts,
        "missing_train_classes": missing_train_classes,
        "insufficient_train_classes": insufficient_train_classes,
        "insufficient_validation_classes": insufficient_validation_classes,
    }


def validation_classification_diagnostics(
    actual: Series, predicted: Series
) -> dict[str, Any]:
    """Return a fixed-order validation confusion matrix and prediction counts."""
    matrix = metrics.confusion_matrix(
        actual.tolist(), predicted.tolist(), labels=list(LABELS)
    )
    return {
        "confusion_matrix": {
            "labels": list(LABELS),
            "rows_are": "actual",
            "columns_are": "predicted",
            "matrix": [[int(value) for value in row] for row in matrix.tolist()],
        },
        "prediction_distribution": label_distribution(predicted),
    }


def baseline_validation_diagnostics(
    actual: Series, predicted: Series
) -> dict[str, Any]:
    """Return comparable classification-only metrics for a simple baseline."""
    return {
        "classification_metrics": classification_metrics(actual, predicted),
        **validation_classification_diagnostics(actual, predicted),
    }


def weighting_policy_specs(hold_weight_multiplier: float) -> dict[str, dict[str, Any]]:
    """Return unweighted, fully balanced, and controlled HOLD-weight policies."""
    if (
        isinstance(hold_weight_multiplier, bool)
        or not isinstance(hold_weight_multiplier, (int, float))
        or not np.isfinite(float(hold_weight_multiplier))
        or hold_weight_multiplier <= 0
    ):
        raise ValueError("hold_weight_multiplier_must_be_finite_and_positive")
    multiplier = float(hold_weight_multiplier)
    return {
        "weight_none": {"class_weight": None, "hold_weight_multiplier": None},
        "weight_balanced": {"class_weight": "balanced", "hold_weight_multiplier": None},
        "weight_intermediate": {
            "class_weight": {"BUY": 1.0, "SELL": 1.0, "HOLD": multiplier},
            "hold_weight_multiplier": multiplier,
        },
    }


def split_validation_for_calibration(
    validation_indices: list[int], calibration_fraction: float
) -> tuple[list[int], list[int]]:
    """Use early Validation rows for threshold calibration and later rows for testing."""
    if (
        isinstance(calibration_fraction, bool)
        or not isinstance(calibration_fraction, (int, float))
        or not np.isfinite(float(calibration_fraction))
        or not 0.0 < calibration_fraction < 0.5
    ):
        raise ValueError("calibration_fraction_must_be_between_zero_and_half")
    if len(validation_indices) < 2:
        raise ValueError("validation_rows_insufficient_for_calibration_split")
    count = max(1, int(np.floor(len(validation_indices) * calibration_fraction)))
    if count >= len(validation_indices):
        raise ValueError("calibration_split_leaves_no_forward_validation_rows")
    return validation_indices[:count], validation_indices[count:]


def ordered_probability_matrix(model: Any, features: DataFrame) -> np.ndarray:
    """Return predict_proba columns in the fixed BUY, SELL, HOLD order."""
    raw = np.asarray(model.predict_proba(features), dtype=float)
    classes = [str(value) for value in model.classes_]
    if raw.ndim != 2 or raw.shape != (len(features), len(classes)):
        raise ValueError("classifier_probability_shape_invalid")
    if len(classes) != len(LABELS) or set(classes) != set(LABELS):
        raise ValueError("classifier_probability_classes_invalid")
    ordered = raw[:, [classes.index(label) for label in LABELS]]
    if (
        not bool(np.isfinite(ordered).all())
        or bool(np.any(ordered < 0.0))
        or bool(np.any(ordered > 1.0))
        or not bool(np.allclose(ordered.sum(axis=1), 1.0, atol=1e-6))
    ):
        raise ValueError("classifier_probabilities_invalid")
    return ordered


def predictions_from_probabilities(
    probabilities: np.ndarray, index: pd.Index | None = None
) -> Series:
    """Use fixed-order argmax as the unthresholded three-class decision."""
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(LABELS):
        raise ValueError("classifier_probability_shape_invalid")
    return Series(
        [LABELS[int(value)] for value in np.argmax(matrix, axis=1)],
        index=index,
        dtype=object,
    )


def hold_class_metrics(actual: Series, predicted: Series) -> dict[str, int | float]:
    """Return binary HOLD precision, recall, F1 and confusion counts."""
    if len(actual) != len(predicted):
        raise ValueError("hold_metric_series_length_mismatch")
    actual_hold = actual.to_numpy(dtype=object) == "HOLD"
    predicted_hold = predicted.to_numpy(dtype=object) == "HOLD"
    tp = int(np.count_nonzero(actual_hold & predicted_hold))
    fp = int(np.count_nonzero(~actual_hold & predicted_hold))
    fn = int(np.count_nonzero(actual_hold & ~predicted_hold))
    support = int(np.count_nonzero(actual_hold))
    predicted_support = int(np.count_nonzero(predicted_hold))
    precision = tp / predicted_support if predicted_support else 0.0
    recall = tp / support if support else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "support": support,
        "predicted_support": predicted_support,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def hold_threshold_predictions(
    probabilities: np.ndarray, threshold: float, index: pd.Index | None = None
) -> Series:
    """Retain HOLD only when it wins argmax and clears the probability threshold."""
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not np.isfinite(float(threshold))
        or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError("hold_threshold_must_be_between_zero_and_one")
    matrix = np.asarray(probabilities, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != len(LABELS)
        or not bool(np.isfinite(matrix).all())
    ):
        raise ValueError("classifier_probability_shape_invalid")
    non_hold = np.where(matrix[:, 0] >= matrix[:, 1], 0, 1)
    predicted = np.asarray([LABELS[int(value)] for value in non_hold], dtype=object)
    is_top = matrix[:, 2] > np.max(matrix[:, :2], axis=1)
    predicted[(matrix[:, 2] >= float(threshold)) & is_top] = "HOLD"
    return Series(predicted.tolist(), index=index, dtype=object)


def optimize_hold_threshold(
    actual: Series,
    probabilities: np.ndarray,
    *,
    minimum_precision: float = DEFAULT_MINIMUM_HOLD_PRECISION,
    minimum_recall: float = DEFAULT_MINIMUM_HOLD_RECALL,
) -> dict[str, Any]:
    """Maximize HOLD F1 while satisfying the configured precision and recall floors."""
    for value, name in (
        (minimum_precision, "minimum_precision"),
        (minimum_recall, "minimum_recall"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError(f"{name}_must_be_between_zero_and_one")
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2 or matrix.shape != (len(actual), len(LABELS)):
        raise ValueError("classifier_probability_shape_invalid")
    if len(actual) == 0:
        raise ValueError("threshold_calibration_set_empty")
    feasible: list[dict[str, Any]] = []
    for threshold in np.linspace(0.0, 1.0, HOLD_THRESHOLD_GRID_STEPS):
        prediction = hold_threshold_predictions(matrix, float(threshold), actual.index)
        hold_metrics = hold_class_metrics(actual, prediction)
        if hold_metrics["precision"] >= float(minimum_precision) and hold_metrics[
            "recall"
        ] >= float(minimum_recall):
            feasible.append({"threshold": round(float(threshold), 6), **hold_metrics})
    grid = {
        "minimum": 0.0,
        "maximum": 1.0,
        "steps": HOLD_THRESHOLD_GRID_STEPS,
        "increment": round(1.0 / (HOLD_THRESHOLD_GRID_STEPS - 1), 6),
    }
    if not feasible:
        return {
            "status": "NO_FEASIBLE_THRESHOLD",
            "selected_threshold": None,
            "feasible_threshold_count": 0,
            "search_grid": grid,
            "minimum_hold_precision": float(minimum_precision),
            "minimum_hold_recall": float(minimum_recall),
            "selected_calibration_metrics": None,
        }
    selected = max(
        feasible,
        key=lambda item: (item["f1"], item["precision"], item["threshold"]),
    )
    return {
        "status": "SELECTED",
        "selected_threshold": float(selected["threshold"]),
        "feasible_threshold_count": len(feasible),
        "search_grid": grid,
        "minimum_hold_precision": float(minimum_precision),
        "minimum_hold_recall": float(minimum_recall),
        "selected_calibration_metrics": selected,
        "selection_objective": "MAXIMIZE_HOLD_F1_THEN_PRECISION_THEN_THRESHOLD",
    }


def threshold_optimization_diagnostics(
    actual: Series,
    probabilities: np.ndarray,
    *,
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION,
    minimum_precision: float = DEFAULT_MINIMUM_HOLD_PRECISION,
    minimum_recall: float = DEFAULT_MINIMUM_HOLD_RECALL,
) -> dict[str, Any]:
    """Calibrate on early Validation rows, then score later rows without overlap."""
    if (
        isinstance(calibration_fraction, bool)
        or not isinstance(calibration_fraction, (int, float))
        or not np.isfinite(float(calibration_fraction))
        or not 0.0 < calibration_fraction < 0.5
    ):
        raise ValueError("calibration_fraction_must_be_between_zero_and_half")
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2 or matrix.shape != (len(actual), len(LABELS)):
        raise ValueError("classifier_probability_shape_invalid")
    count = max(1, int(np.floor(len(actual) * calibration_fraction)))
    if count >= len(actual):
        raise ValueError("calibration_split_leaves_no_forward_validation_rows")
    calibration_y = actual.iloc[:count]
    forward_y = actual.iloc[count:]
    calibration_p = matrix[:count]
    forward_p = matrix[count:]
    counts = _class_counts(calibration_y)
    metadata = {
        "row_count": int(len(calibration_y)),
        "start_row_index": int(calibration_y.index[0]),
        "end_row_index": int(calibration_y.index[-1]),
        "temporal_position": "FIRST_CHRONOLOGICAL_SEGMENT_OF_VALIDATION",
        "label_distribution": label_distribution(calibration_y),
        "minimum_hold_support": MIN_CALIBRATION_HOLD_SAMPLES,
        "minimum_non_hold_support": MIN_CALIBRATION_NON_HOLD_SAMPLES,
    }
    constraints = {
        "minimum_hold_precision": float(minimum_precision),
        "minimum_hold_recall": float(minimum_recall),
    }
    if (
        counts["HOLD"] < MIN_CALIBRATION_HOLD_SAMPLES
        or len(calibration_y) - counts["HOLD"] < MIN_CALIBRATION_NON_HOLD_SAMPLES
    ):
        return {
            "status": "SKIPPED_INSUFFICIENT_CALIBRATION_SAMPLES",
            "selected_threshold": None,
            "constraints": constraints,
            "calibration": metadata,
            "post_calibration_validation": None,
            "full_validation_including_calibration": None,
        }
    selection = optimize_hold_threshold(
        calibration_y,
        calibration_p,
        minimum_precision=minimum_precision,
        minimum_recall=minimum_recall,
    )
    result: dict[str, Any] = {
        "status": selection["status"],
        "selected_threshold": selection["selected_threshold"],
        "constraints": constraints,
        "calibration": {**metadata, "threshold_selection": selection},
        "post_calibration_validation": None,
        "full_validation_including_calibration": None,
    }
    if selection["status"] != "SELECTED":
        return result
    threshold = float(selection["selected_threshold"])
    calibration_prediction = hold_threshold_predictions(
        calibration_p, threshold, calibration_y.index
    )
    forward_prediction = hold_threshold_predictions(
        forward_p, threshold, forward_y.index
    )
    full_prediction = hold_threshold_predictions(matrix, threshold, actual.index)
    result["calibration"]["selected_calibration_metrics"] = {
        "classification_metrics": classification_metrics(
            calibration_y, calibration_prediction
        ),
        "hold_metrics": hold_class_metrics(calibration_y, calibration_prediction),
    }
    result["post_calibration_validation"] = {
        "row_count": int(len(forward_y)),
        "start_row_index": int(forward_y.index[0]),
        "calibration_overlap": False,
        "classification_metrics": classification_metrics(forward_y, forward_prediction),
        "hold_metrics": hold_class_metrics(forward_y, forward_prediction),
        **validation_classification_diagnostics(forward_y, forward_prediction),
    }
    result["full_validation_including_calibration"] = {
        "row_count": int(len(actual)),
        "calibration_overlap": True,
        "evaluation_scope": "DESCRIPTIVE_ONLY_INCLUDES_CALIBRATION_ROWS",
        "classification_metrics": classification_metrics(actual, full_prediction),
        "hold_metrics": hold_class_metrics(actual, full_prediction),
        **validation_classification_diagnostics(actual, full_prediction),
    }
    return result


def _serialize_validation_probabilities(
    bars: DataFrame,
    validation_indices: list[int],
    actual: Series,
    probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    """Persist rounded per-row probabilities with source timestamp and row index."""
    records: list[dict[str, Any]] = []
    for row_index, actual_label, row_probs in zip(
        validation_indices, actual.tolist(), probabilities, strict=True
    ):
        records.append(
            {
                "row_index": int(row_index),
                "timestamp": pd.Timestamp(bars["date"].iloc[row_index]).isoformat(),
                "actual_label": str(actual_label),
                "probabilities": {
                    label: round(float(row_probs[position]), 8)
                    for position, label in enumerate(LABELS)
                },
            }
        )
    return records


def _validation_window(row_count: int) -> tuple[int, int]:
    validation_start = int(row_count * VALIDATION_START_FRACTION)
    validation_end = int(row_count * VALIDATION_END_FRACTION) - 1
    if validation_end <= validation_start:
        raise ValueError("validation_window_invalid")
    return validation_start, validation_end


def _label_end_indices(labels: DataFrame) -> list[int]:
    return [
        index if pd.isna(value) else int(value)
        for index, value in enumerate(labels["label_end_idx"].tolist())
    ]


def _eligible_sample_indices(features: DataFrame, labels: DataFrame) -> set[int]:
    eligible_mask = (
        features.notna().all(axis=1)
        & labels["label"].isin(LABELS)
        & labels["label_end_idx"].notna()
    )
    return {int(index) for index in np.flatnonzero(eligible_mask.to_numpy())}


def _eligible_train_indices(
    train_candidates: tuple[int, ...],
    eligible_indices: set[int],
    train_limit: int,
) -> list[int]:
    return [
        index
        for index in train_candidates
        if index in eligible_indices and index < train_limit
    ]


def _eligible_validation_indices(
    validation_candidates: tuple[int, ...], eligible_indices: set[int]
) -> list[int]:
    return [index for index in validation_candidates if index in eligible_indices]


def _split_diagnostics(
    split: Any,
    eligible_indices: set[int],
    row_count: int,
    validation_start: int,
    validation_end: int,
    train_limit: int,
) -> dict[str, int]:
    purged_train_rows = sum(
        index in eligible_indices and index < train_limit
        for index in split.purged_indices
    )
    eligible_gap_rows = sum(
        index in eligible_indices
        for index in range(max(0, train_limit), validation_start)
    )
    return {
        "validation_start_index": validation_start,
        "validation_end_index": validation_end,
        "purged_train_rows": int(purged_train_rows),
        "embargo_size_bars": int(split.embargo_size),
        "eligible_pre_validation_embargo_rows": int(eligible_gap_rows),
        "post_validation_embargo_bars": len(split.embargo_indices),
        "future_rows_excluded_for_chronology": row_count - validation_end - 1,
    }


def purged_temporal_indices(
    features: DataFrame, labels: DataFrame
) -> tuple[list[int], list[int], dict[str, int]]:
    """Create a chronological split, actual-label-window purge, and conservative gap."""
    row_count = len(labels)
    if len(features) != row_count or row_count < 100:
        raise ValueError("insufficient_or_misaligned_rows_for_temporal_split")

    validation_start, validation_end = _validation_window(row_count)
    ends = _label_end_indices(labels)
    eligible_indices = _eligible_sample_indices(features, labels)
    split = create_purged_embargo_split(
        ends,
        validation_start=validation_start,
        validation_end=validation_end,
        longest_feature_window=FEATURE_WINDOW,
    )
    train_limit = validation_start - split.embargo_size
    train_indices = _eligible_train_indices(
        split.train_indices, eligible_indices, train_limit
    )
    validation_indices = _eligible_validation_indices(
        split.validation_indices, eligible_indices
    )
    diagnostics = _split_diagnostics(
        split,
        eligible_indices,
        row_count,
        validation_start,
        validation_end,
        train_limit,
    )
    return train_indices, validation_indices, diagnostics


def _new_classifier(
    class_weight: str | dict[str, float] | None = "balanced",
) -> Any:
    """Load the pinned deterministic tree classifier for one weighting policy."""
    lightgbm = importlib.import_module("lightgbm")
    classifier = getattr(lightgbm, "LGBMClassifier")
    return classifier(
        objective="multiclass",
        n_estimators=80,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=4,
        class_weight=class_weight,
        random_state=42,
        n_jobs=1,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )


def _skipped_result(
    pair: str,
    timeframe: str,
    bars: DataFrame,
    labels: DataFrame,
    train_indices: list[int],
    validation_indices: list[int],
    split_diagnostics: dict[str, int],
    reason: str,
    sample_gate: dict[str, Any],
) -> dict[str, Any]:
    train_labels = labels.iloc[train_indices]["label"]
    validation_labels = labels.iloc[validation_indices]["label"]
    return {
        "pair": pair,
        "timeframe": timeframe,
        "status": reason,
        "bar_count": int(len(bars)),
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(validation_indices)),
        "label_distribution": {
            "train": label_distribution(train_labels),
            "validation": label_distribution(validation_labels),
        },
        "classification_metrics": None,
        "confusion_matrix": None,
        "prediction_distribution": None,
        "baseline_comparisons": None,
        "model_variants": None,
        "threshold_optimization": None,
        "label_audit": label_audit_diagnostics(labels),
        "sample_gate": sample_gate,
        "overfitting_assessment": {
            "status": "NOT_ASSESSED",
            "macro_f1_gap_train_minus_validation": None,
        },
        "split": split_diagnostics,
    }


def _insufficient_bar_result(
    pair: str, timeframe: str, bars: DataFrame, labels: DataFrame
) -> dict[str, Any]:
    """Report short real-data windows without pretending a train/validation split exists."""
    empty_distribution = label_distribution(Series(dtype=object))
    return {
        "pair": pair,
        "timeframe": timeframe,
        "status": "SKIPPED_INSUFFICIENT_SAMPLES",
        "bar_count": int(len(bars)),
        "train_rows": 0,
        "validation_rows": 0,
        "label_distribution": {
            "train": empty_distribution,
            "validation": empty_distribution,
            "available_unpartitioned": label_distribution(labels["label"]),
        },
        "classification_metrics": None,
        "confusion_matrix": None,
        "prediction_distribution": None,
        "baseline_comparisons": None,
        "model_variants": None,
        "threshold_optimization": None,
        "label_audit": label_audit_diagnostics(labels),
        "sample_gate": {
            "status": "NOT_EVALUATED",
            "reason": "INSUFFICIENT_BARS",
            "minimum_train_samples_per_class": MIN_TRAIN_SAMPLES_PER_CLASS,
            "minimum_validation_samples_per_class": MIN_VALIDATION_SAMPLES_PER_CLASS,
            "train_class_counts": None,
            "validation_class_counts": None,
        },
        "overfitting_assessment": {
            "status": "NOT_ASSESSED",
            "macro_f1_gap_train_minus_validation": None,
        },
        "split": {
            "status": "NOT_SPLIT_INSUFFICIENT_BARS",
            "minimum_required_bars": 100,
        },
    }


def run_experiment(
    pair: str,
    timeframe: str,
    minute_candles: DataFrame,
    *,
    hold_weight_multiplier: float = DEFAULT_HOLD_WEIGHT_MULTIPLIER,
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION,
    minimum_hold_precision: float = DEFAULT_MINIMUM_HOLD_PRECISION,
    minimum_hold_recall: float = DEFAULT_MINIMUM_HOLD_RECALL,
) -> dict[str, Any]:
    """Fit the three weighting policies and threshold-calibrate each from time-ordered data."""
    policy_specs = weighting_policy_specs(hold_weight_multiplier)
    bars = resample_ohlcv(minute_candles, timeframe)
    labels = apply_adaptive_triple_barrier(bars)
    if len(bars) < 100:
        return _insufficient_bar_result(pair, timeframe, bars, labels)
    features = build_causal_feature_matrix(bars)
    train_indices, validation_indices, split_info = purged_temporal_indices(
        features, labels
    )
    train_y = labels.iloc[train_indices]["label"]
    validation_y = labels.iloc[validation_indices]["label"]
    sample_gate = sample_gate_diagnostics(train_y, validation_y)

    base_result: dict[str, Any] = {
        "pair": pair,
        "timeframe": timeframe,
        "bar_count": int(len(bars)),
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(validation_indices)),
        "label_distribution": {
            "train": label_distribution(train_y),
            "validation": label_distribution(validation_y),
        },
        "label_audit": label_audit_diagnostics(labels),
        "sample_gate": sample_gate,
        "split": split_info,
    }
    if sample_gate["status"] != "PASS":
        return _skipped_result(
            pair,
            timeframe,
            bars,
            labels,
            train_indices,
            validation_indices,
            split_info,
            str(sample_gate["status"]),
            sample_gate,
        )

    x_train = features.iloc[train_indices]
    x_validation = features.iloc[validation_indices]
    variants: dict[str, Any] = {}
    for name, policy in policy_specs.items():
        model = _new_classifier(class_weight=policy["class_weight"])
        model.fit(x_train, train_y)
        train_probabilities = ordered_probability_matrix(model, x_train)
        validation_probabilities = ordered_probability_matrix(model, x_validation)
        train_prediction = predictions_from_probabilities(
            train_probabilities, train_y.index
        )
        validation_prediction = predictions_from_probabilities(
            validation_probabilities, validation_y.index
        )
        threshold_diagnostics = threshold_optimization_diagnostics(
            validation_y,
            validation_probabilities,
            calibration_fraction=calibration_fraction,
            minimum_precision=minimum_hold_precision,
            minimum_recall=minimum_hold_recall,
        )
        variants[name] = {
            "model_parameters": {
                "class_weight": policy["class_weight"],
                "hold_weight_multiplier": policy["hold_weight_multiplier"],
            },
            "classification_metrics": {
                "train_in_sample": classification_metrics(train_y, train_prediction),
                "validation": classification_metrics(
                    validation_y, validation_prediction
                ),
            },
            **validation_classification_diagnostics(
                validation_y, validation_prediction
            ),
            "hold_metrics": hold_class_metrics(validation_y, validation_prediction),
            "threshold_optimization": threshold_diagnostics,
            "validation_probabilities": _serialize_validation_probabilities(
                bars, validation_indices, validation_y, validation_probabilities
            ),
        }

    balanced = variants["weight_balanced"]
    unweighted = variants["weight_none"]
    always_sell_prediction = Series("SELL", index=validation_y.index, dtype=object)
    baseline_comparisons = {
        "always_sell": {
            "baseline_type": "constant_prediction",
            "predicted_label": "SELL",
            **baseline_validation_diagnostics(validation_y, always_sell_prediction),
        },
        "lightgbm_unweighted": {
            "baseline_type": "lightgbm",
            "model_variant": "weight_none",
            "model_parameters": {"class_weight": None},
            "classification_metrics": unweighted["classification_metrics"][
                "validation"
            ],
            "confusion_matrix": unweighted["confusion_matrix"],
            "prediction_distribution": unweighted["prediction_distribution"],
        },
    }
    train_f1 = balanced["classification_metrics"]["train_in_sample"]["macro"]["f1"]
    validation_f1 = balanced["classification_metrics"]["validation"]["macro"]["f1"]
    f1_gap = round(train_f1 - validation_f1, 6)
    status = (
        "POSSIBLE_OVERFITTING"
        if f1_gap >= OVERFIT_F1_GAP_THRESHOLD
        else "NO_LARGE_MACRO_F1_GAP_OBSERVED"
    )
    return {
        **base_result,
        "status": "TRAINED",
        "classification_metrics": balanced["classification_metrics"],
        "confusion_matrix": balanced["confusion_matrix"],
        "prediction_distribution": balanced["prediction_distribution"],
        "hold_metrics": balanced["hold_metrics"],
        "model_variants": variants,
        "threshold_optimization": balanced["threshold_optimization"],
        "baseline_comparisons": baseline_comparisons,
        "overfitting_assessment": {
            "status": status,
            "macro_f1_gap_train_minus_validation": f1_gap,
            "heuristic_threshold": OVERFIT_F1_GAP_THRESHOLD,
            "caveat": "Single chronological split; diagnostic heuristic, not a statistical proof.",
        },
    }


def _load_manifested_data(
    input_dir: Path, symbol: str, days: int, commit_sha: str
) -> tuple[DataFrame, dict[str, Any]]:
    stem = f"binance-{symbol.lower()}-1m"
    csv_path = input_dir / f"{stem}.csv"
    manifest_path = input_dir / f"{stem}.manifest.json"
    if not csv_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"public_data_artifact_missing:{symbol}")

    manifest_bytes = manifest_path.read_bytes()
    manifest = cast(dict[str, Any], json.loads(manifest_bytes.decode("utf-8")))
    csv_digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    expected_rows = days * 24 * 60
    checks = (
        manifest.get("symbol") == symbol,
        manifest.get("interval") == "1m",
        manifest.get("requested_days") == days,
        manifest.get("row_count") == expected_rows,
        manifest.get("commit_sha") == commit_sha,
        manifest.get("csv_sha256") == csv_digest,
        manifest.get("provider") == "Binance Spot public klines API",
        manifest.get("public_api_only") is True,
        manifest.get("credentials_used") is False,
        manifest.get("private_api_calls") == 0,
        manifest.get("real_order_attempts") == 0,
        manifest.get("gap_policy")
        == "reject; no interpolation, resampling, or synthetic fill",
        manifest.get("missing_minute_intervals") == 0,
        manifest.get("fetch_page_size_candles") == 1000,
        manifest.get("inter_page_delay_seconds") == 0.25,
    )
    if not all(checks):
        raise ValueError(f"public_data_manifest_validation_failed:{symbol}")

    candles = _validate_minute_candles(pd.read_csv(csv_path))
    if len(candles) != expected_rows:
        raise ValueError(f"public_data_row_count_mismatch:{symbol}")
    provenance = {
        "symbol": symbol,
        "pair": f"{symbol[:-4]}/USDT",
        "provider": manifest["provider"],
        "interval": "1m",
        "row_count": int(len(candles)),
        "requested_days": int(manifest["requested_days"]),
        "missing_minute_intervals": int(manifest["missing_minute_intervals"]),
        "fetch_page_size_candles": int(manifest["fetch_page_size_candles"]),
        "inter_page_delay_seconds": float(manifest["inter_page_delay_seconds"]),
        "start_utc": candles["date"].iloc[0].isoformat(),
        "end_utc": candles["date"].iloc[-1].isoformat(),
        "csv_sha256": csv_digest,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "public_api_only": True,
        "credentials_used": False,
    }
    return candles, provenance


def _distribution_cell(distribution: dict[str, Any], label: str) -> str:
    row = distribution["classes"][label]
    return f"{row['percent']:.2f}% ({row['count']})"


def _render_label_distribution_section(
    experiments: list[dict[str, Any]],
) -> list[str]:
    """Render actual target-label distributions for every experiment."""
    lines = [
        "### Label distribution",
        "",
        "| Pair | TF | Train n | BUY % (n) | SELL % (n) | HOLD % (n) | Validation n | BUY % (n) | SELL % (n) | HOLD % (n) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in experiments:
        distribution = item["label_distribution"]
        train = distribution["train"]
        validation = distribution["validation"]
        lines.append(
            f"| {item['pair']} | {item['timeframe']} | {item['train_rows']} "
            f"| {_distribution_cell(train, 'BUY')} | {_distribution_cell(train, 'SELL')} "
            f"| {_distribution_cell(train, 'HOLD')} | {item['validation_rows']} "
            f"| {_distribution_cell(validation, 'BUY')} "
            f"| {_distribution_cell(validation, 'SELL')} "
            f"| {_distribution_cell(validation, 'HOLD')} |"
        )
    return lines


def _render_macro_metrics_section(
    experiments: list[dict[str, Any]],
) -> list[str]:
    """Render macro metrics and the diagnostic Train/Validation F1 gap."""
    lines = [
        "### Macro classification metrics and overfitting screen",
        "",
        "| Pair | TF | Train Precision | Train Recall | Train F1 | Validation Precision | Validation Recall | Validation F1 | F1 gap | Assessment |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in experiments:
        classification = item["classification_metrics"]
        if classification is None:
            lines.append(
                f"| {item['pair']} | {item['timeframe']} | N/A | N/A | N/A "
                f"| N/A | N/A | N/A | N/A | {item['status']} |"
            )
            continue
        train = classification["train_in_sample"]["macro"]
        validation = classification["validation"]["macro"]
        gap = item["overfitting_assessment"]["macro_f1_gap_train_minus_validation"]
        lines.append(
            f"| {item['pair']} | {item['timeframe']} | {train['precision']:.4f} "
            f"| {train['recall']:.4f} | {train['f1']:.4f} "
            f"| {validation['precision']:.4f} | {validation['recall']:.4f} "
            f"| {validation['f1']:.4f} | {gap:.4f} "
            f"| {item['overfitting_assessment']['status']} |"
        )
    return lines


def _render_classwise_metrics_section(
    experiments: list[dict[str, Any]],
) -> list[str]:
    """Render class-wise Validation metrics with per-class support."""
    lines = [
        "### Validation class-wise metrics",
        "",
        "| Pair | TF | Class | Precision | Recall | F1 | Support |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for item in experiments:
        classification = item["classification_metrics"]
        if classification is None:
            continue
        for label in LABELS:
            row = classification["validation"]["per_class"][label]
            lines.append(
                f"| {item['pair']} | {item['timeframe']} | {label} "
                f"| {row['precision']:.4f} | {row['recall']:.4f} "
                f"| {row['f1']:.4f} | {row['support']} |"
            )
    return lines


def _render_weighting_metrics_section(
    experiments: list[dict[str, Any]],
) -> list[str]:
    """Render raw argmax metrics for each weighting policy."""
    lines = [
        "### Controlled weighting comparison (raw argmax)",
        "",
        "| Pair | TF | Weighting policy | Macro-F1 | HOLD precision | HOLD recall | HOLD F1 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for item in experiments:
        variants = item.get("model_variants")
        for name in ("weight_none", "weight_balanced", "weight_intermediate"):
            if variants is None:
                lines.append(
                    f"| {item['pair']} | {item['timeframe']} | {name} "
                    f"| N/A | N/A | N/A | N/A |"
                )
                continue
            variant = variants[name]
            macro_f1 = variant["classification_metrics"]["validation"]["macro"]["f1"]
            hold = variant["hold_metrics"]
            lines.append(
                f"| {item['pair']} | {item['timeframe']} | {name} "
                f"| {macro_f1:.4f} | {hold['precision']:.4f} "
                f"| {hold['recall']:.4f} | {hold['f1']:.4f} |"
            )
    return lines


def _render_threshold_optimization_section(
    experiments: list[dict[str, Any]],
) -> list[str]:
    """Render threshold selection and non-overlapping forward metrics."""
    lines = [
        "### HOLD threshold calibration",
        "",
        "| Pair | TF | Weighting policy | Status | Threshold | Calibration P/R | Forward Validation P/R | Full Validation P/R (overlap) |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for item in experiments:
        variants = item.get("model_variants")
        for name in ("weight_none", "weight_balanced", "weight_intermediate"):
            if variants is None:
                lines.append(
                    f"| {item['pair']} | {item['timeframe']} | {name} "
                    f"| {item['status']} | N/A | N/A | N/A | N/A |"
                )
                continue
            threshold = variants[name]["threshold_optimization"]
            selection = threshold.get("calibration", {}).get("threshold_selection")
            selected = (
                selection.get("selected_calibration_metrics") if selection else None
            )
            calibration_pr = (
                f"{selected['precision']:.3f}/{selected['recall']:.3f}"
                if selected
                else "N/A"
            )
            forward = threshold.get("post_calibration_validation")
            forward_pr = (
                f"{forward['hold_metrics']['precision']:.3f}/"
                f"{forward['hold_metrics']['recall']:.3f}"
                if forward
                else "N/A"
            )
            full = threshold.get("full_validation_including_calibration")
            full_pr = (
                f"{full['hold_metrics']['precision']:.3f}/"
                f"{full['hold_metrics']['recall']:.3f}"
                if full
                else "N/A"
            )
            value = threshold.get("selected_threshold")
            value_text = f"{value:.3f}" if value is not None else "N/A"
            lines.append(
                f"| {item['pair']} | {item['timeframe']} | {name} "
                f"| {threshold['status']} | {value_text} "
                f"| {calibration_pr} | {forward_pr} | {full_pr} |"
            )
    return lines


def _render_prediction_distribution_section(
    experiments: list[dict[str, Any]],
) -> list[str]:
    """Render the model's predicted class distribution on Validation."""
    lines = [
        "### Validation prediction distribution",
        "",
        "| Pair | TF | Validation n | BUY % (n) | SELL % (n) | HOLD % (n) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in experiments:
        predictions = item.get("prediction_distribution")
        if predictions is None:
            lines.append(
                f"| {item['pair']} | {item['timeframe']} | N/A | N/A | N/A | N/A |"
            )
            continue
        lines.append(
            f"| {item['pair']} | {item['timeframe']} | {predictions['total']} "
            f"| {_distribution_cell(predictions, 'BUY')} "
            f"| {_distribution_cell(predictions, 'SELL')} "
            f"| {_distribution_cell(predictions, 'HOLD')} |"
        )
    return lines


def _render_confusion_matrix_section(
    experiments: list[dict[str, Any]],
) -> list[str]:
    """Render fixed-order Validation confusion matrices."""
    lines = [
        "### Validation confusion matrices",
        "",
        "| Pair | TF | Actual class (rows) | Predicted BUY | Predicted SELL | Predicted HOLD |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for item in experiments:
        matrix = item.get("confusion_matrix")
        for index, label in enumerate(LABELS):
            values = (
                ("N/A", "N/A", "N/A")
                if matrix is None
                else tuple(matrix["matrix"][index])
            )
            lines.append(
                f"| {item['pair']} | {item['timeframe']} | {label} "
                f"| {values[0]} | {values[1]} | {values[2]} |"
            )
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    """Render classification diagnostics for the GitHub Actions summary."""
    experiments = report["experiments"]
    lines = [
        "## AliX real-data diagnostic baseline",
        "",
        f"Commit SHA: {report['commit_sha']}",
        "",
        "Scope: classification diagnostics only. No economic metrics are calculated.",
        "",
    ]
    lines.extend(_render_label_distribution_section(experiments))
    lines.extend(["", *_render_macro_metrics_section(experiments)])
    lines.extend(["", *_render_classwise_metrics_section(experiments)])
    lines.extend(["", *_render_weighting_metrics_section(experiments)])
    lines.extend(["", *_render_threshold_optimization_section(experiments)])
    lines.extend(["", *_render_prediction_distribution_section(experiments)])
    lines.extend(["", *_render_confusion_matrix_section(experiments)])
    lines.extend(
        [
            "",
            "The overfitting screen uses the in-sample versus validation macro-F1 gap "
            f"of {OVERFIT_F1_GAP_THRESHOLD:.2f} as a heuristic. It is not proof.",
            "",
            "Safety: Champion=null, order_permission=false, "
            "trading_authorization=NO-GO, real_order_attempts=0, private_api_calls=0.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_diagnostic_parameters(
    days: int,
    hold_weight_multiplier: float,
    calibration_fraction: float,
    minimum_hold_precision: float,
    minimum_hold_recall: float,
) -> dict[str, dict[str, Any]]:
    """Validate the configuration and return the canonical weighting policies."""
    if days not in (1, 7, 30, 90, 730):
        raise ValueError("diagnostic_days_out_of_policy")
    policies = weighting_policy_specs(hold_weight_multiplier)
    _validate_calibration_fraction(calibration_fraction)
    _validate_threshold_constraints(minimum_hold_precision, minimum_hold_recall)
    return policies


def _load_diagnostic_datasets(
    input_dir: Path, days: int, commit_sha: str
) -> tuple[dict[str, DataFrame], list[dict[str, Any]]]:
    """Load and validate manifested public datasets for each approved symbol."""
    datasets: dict[str, DataFrame] = {}
    provenance: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        candles, manifest = _load_manifested_data(
            input_dir, symbol, days, commit_sha
        )
        datasets[symbol] = candles
        provenance.append(manifest)
    return datasets, provenance


def _run_diagnostic_experiments(
    datasets: dict[str, DataFrame],
    *,
    hold_weight_multiplier: float,
    calibration_fraction: float,
    minimum_hold_precision: float,
    minimum_hold_recall: float,
) -> list[dict[str, Any]]:
    """Run the approved pair/timeframe grid with the supplied v4 policies."""
    experiments: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        pair = f"{symbol[:-4]}/USDT"
        for timeframe in TIMEFRAME_MINUTES:
            experiments.append(
                run_experiment(
                    pair,
                    timeframe,
                    datasets[symbol],
                    hold_weight_multiplier=hold_weight_multiplier,
                    calibration_fraction=calibration_fraction,
                    minimum_hold_precision=minimum_hold_precision,
                    minimum_hold_recall=minimum_hold_recall,
                )
            )
    return experiments


def _build_diagnostic_report(
    *,
    days: int,
    commit_sha: str,
    provenance: list[dict[str, Any]],
    experiments: list[dict[str, Any]],
    hold_weight_multiplier: float,
    calibration_fraction: float,
    minimum_hold_precision: float,
    minimum_hold_recall: float,
    policy_specs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Construct the v4 diagnostic report without performing economic evaluation."""
    trained_count = sum(item["status"] == "TRAINED" for item in experiments)
    return {
        "schema_version": REPORT_SCHEMA,
        "commit_sha": commit_sha,
        "diagnostic_scope": "CLASSIFICATION_ONLY",
        "training_scope": "DIAGNOSTIC_CLASSIFICATION_ONLY",
        "economic_evaluation_performed": False,
        "economic_metrics_reported": False,
        "requested_days": days,
        "data_window_selection": {
            "configured_days": int(
                os.environ.get("ALIX_DIAGNOSTIC_CONFIGURED_DAYS", str(days))
            ),
            "selected_days": days,
            "fallback_reason": os.environ.get(
                "ALIX_DIAGNOSTIC_FALLBACK_REASON", "NONE"
            ),
            "policy": (
                "Fetch the configured window in individually validated pages of "
                "at most 1,000 one-minute candles with a 0.25-second delay between "
                "pages; retry incomplete pages and transient HTTP failures with "
                "bounded retries; never shorten the requested window or synthesize "
                "missing candles."
            ),
            "fetch_page_size_candles": provenance[0]["fetch_page_size_candles"],
            "inter_page_delay_seconds": provenance[0]["inter_page_delay_seconds"],
        },
        "sample_gate_policy": {
            "minimum_train_samples_per_class": MIN_TRAIN_SAMPLES_PER_CLASS,
            "minimum_validation_samples_per_class": MIN_VALIDATION_SAMPLES_PER_CLASS,
            "required_classes": list(LABELS),
        },
        "label_audit_policy": {
            "exit_reason_values": list(EXIT_REASONS),
            "unlabeled_reason_values": list(UNLABELED_REASONS),
            "distribution_scope": (
                "All target bars before train/validation splitting; only labeled "
                "BUY/SELL/HOLD events contribute to exit-reason counts."
            ),
        },
        "provider": "Binance Spot public klines API",
        "input_datasets": provenance,
        "allowed_timeframes": list(TIMEFRAME_MINUTES),
        "labels": list(LABELS),
        "model": {
            "family": "LightGBM LGBMClassifier",
            "n_estimators": 80,
            "learning_rate": 0.05,
            "max_depth": 4,
            "num_leaves": 15,
            "random_state": 42,
            "class_weight": "balanced",
            "primary_variant": "weight_balanced",
        },
        "weighting_policy": {
            "hold_weight_multiplier": float(hold_weight_multiplier),
            "policies": {
                name: {
                    "class_weight": spec["class_weight"],
                    "hold_weight_multiplier": spec["hold_weight_multiplier"],
                }
                for name, spec in policy_specs.items()
            },
        },
        "calibration_policy": {
            "fraction_of_validation": float(calibration_fraction),
            "temporal_position": "FIRST_CHRONOLOGICAL_SEGMENT_OF_VALIDATION",
            "reason": (
                "Calibration precedes forward threshold evaluation. Using the final "
                "Validation segment to tune a threshold and then scoring all of "
                "Validation would leak calibration observations into evaluation."
            ),
            "minimum_hold_support": MIN_CALIBRATION_HOLD_SAMPLES,
            "minimum_non_hold_support": MIN_CALIBRATION_NON_HOLD_SAMPLES,
            "minimum_hold_precision": float(minimum_hold_precision),
            "minimum_hold_recall": float(minimum_hold_recall),
            "threshold_grid_steps": HOLD_THRESHOLD_GRID_STEPS,
            "primary_evaluation_scope": "POST_CALIBRATION_REMAINING_VALIDATION",
            "full_validation_scope": "DESCRIPTIVE_ONLY_INCLUDES_CALIBRATION_ROWS",
        },
        "baseline_comparisons": {
            "always_sell": {
                "type": "constant_prediction",
                "predicted_label": "SELL",
                "metrics_scope": "VALIDATION_ONLY",
            },
            "lightgbm_unweighted": {
                "family": "LightGBM LGBMClassifier",
                "n_estimators": 80,
                "learning_rate": 0.05,
                "max_depth": 4,
                "num_leaves": 15,
                "random_state": 42,
                "class_weight": None,
                "metrics_scope": "VALIDATION_ONLY",
            },
        },
        "split_policy": {
            "validation_start_fraction": VALIDATION_START_FRACTION,
            "validation_end_fraction": VALIDATION_END_FRACTION,
            "chronological_training_only": True,
            "label_window_purge": "create_purged_embargo_split",
            "pre_validation_embargo": True,
            "post_validation_embargo": True,
            "label_horizon_candles": 24,
            "atr_period": 14,
            "take_profit_atr_multiple": 2.0,
            "stop_loss_atr_multiple": 1.0,
        },
        "trained_models": trained_count,
        "experiment_count": len(experiments),
        "experiments": experiments,
        "safety": {
            "public_api_only": True,
            "credentials_used": False,
            "private_api_calls": 0,
            "real_order_attempts": 0,
            "orders_or_execution_api_used": False,
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
        },
    }


def _write_diagnostic_report(report: dict[str, Any], output_path: Path) -> None:
    """Persist JSON and its matching GitHub Step Summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as summary:
            summary.write(render_markdown(report))


def _print_diagnostic_completion(
    report: dict[str, Any], output_path: Path, commit_sha: str
) -> None:
    """Emit a compact machine-readable completion record."""
    print(
        json.dumps(
            {
                "report_path": str(output_path),
                "commit_sha": commit_sha,
                "experiment_count": report["experiment_count"],
                "trained_models": report["trained_models"],
                "economic_evaluation_performed": False,
                "Champion": None,
                "order_permission": False,
                "trading_authorization": "NO-GO",
            },
            sort_keys=True,
        )
    )


def run_diagnostic(
    input_dir: Path,
    output_path: Path,
    *,
    days: int,
    commit_sha: str,
    hold_weight_multiplier: float = DEFAULT_HOLD_WEIGHT_MULTIPLIER,
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION,
    minimum_hold_precision: float = DEFAULT_MINIMUM_HOLD_PRECISION,
    minimum_hold_recall: float = DEFAULT_MINIMUM_HOLD_RECALL,
) -> dict[str, Any]:
    """Execute the classification-only real-data diagnostic and persist v4 output."""
    policy_specs = _validate_diagnostic_parameters(
        days,
        hold_weight_multiplier,
        calibration_fraction,
        minimum_hold_precision,
        minimum_hold_recall,
    )
    datasets, provenance = _load_diagnostic_datasets(
        input_dir, days, commit_sha
    )
    experiments = _run_diagnostic_experiments(
        datasets,
        hold_weight_multiplier=hold_weight_multiplier,
        calibration_fraction=calibration_fraction,
        minimum_hold_precision=minimum_hold_precision,
        minimum_hold_recall=minimum_hold_recall,
    )
    report = _build_diagnostic_report(
        days=days,
        commit_sha=commit_sha,
        provenance=provenance,
        experiments=experiments,
        hold_weight_multiplier=hold_weight_multiplier,
        calibration_fraction=calibration_fraction,
        minimum_hold_precision=minimum_hold_precision,
        minimum_hold_recall=minimum_hold_recall,
        policy_specs=policy_specs,
    )
    _write_diagnostic_report(report, output_path)
    _print_diagnostic_completion(report, output_path, commit_sha)
    return report



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--days", type=int, choices=(1, 7, 30, 90, 730), default=730)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument(
        "--hold-weight-multiplier",
        type=float,
        default=DEFAULT_HOLD_WEIGHT_MULTIPLIER,
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=DEFAULT_CALIBRATION_FRACTION,
    )
    parser.add_argument(
        "--minimum-hold-precision",
        type=float,
        default=DEFAULT_MINIMUM_HOLD_PRECISION,
    )
    parser.add_argument(
        "--minimum-hold-recall",
        type=float,
        default=DEFAULT_MINIMUM_HOLD_RECALL,
    )
    args = parser.parse_args()
    report = run_diagnostic(
        args.input_dir,
        args.output,
        days=args.days,
        commit_sha=args.commit_sha,
        hold_weight_multiplier=args.hold_weight_multiplier,
        calibration_fraction=args.calibration_fraction,
        minimum_hold_precision=args.minimum_hold_precision,
        minimum_hold_recall=args.minimum_hold_recall,
    )
    if report["experiment_count"] != 8:
        raise RuntimeError("diagnostic_experiment_count_mismatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
