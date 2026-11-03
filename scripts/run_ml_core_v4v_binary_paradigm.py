"""V4-V binary BUY/NO_BUY LightGBM training and Spot economic evaluation."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from scripts.run_ml_core_real_data_baseline import (
    MIN_TRAIN_SAMPLES_PER_CLASS,
    MIN_VALIDATION_SAMPLES_PER_CLASS,
    SYMBOLS,
    TIMEFRAME_MINUTES,
    _load_diagnostic_datasets,
    resample_ohlcv,
    weighting_policy_specs,
)
from services.ml_core.adaptive_targets import apply_adaptive_triple_barrier
from services.ml_core.binary_targets import (
    BINARY_BUY_LABELS,
    BINARY_TARGET_SCHEMA_VERSION,
    apply_binary_buy_targets,
)
from services.ml_core.economic_backtest import (
    ExecutionConfig,
    TradeManagementConfig,
    simulate_signals,
)
from services.ml_core.extended_features import (
    EXTENDED_FEATURE_SCHEMA_VERSION,
    build_feature_profiles,
    feature_profile_manifest,
)
from services.ml_core.hyperparameter_tuning import (
    BINARY_BUY_OBJECTIVE,
    DEFAULT_INNER_CV_SPLITS,
    DEFAULT_MINIMUM_CLASS_SAMPLES,
    DEFAULT_RANDOM_SEARCH_ITERATIONS,
    build_purged_time_series_splits,
    tune_lightgbm_binary_classifier,
)
from services.ml_core.purged_embargo import create_purged_embargo_split

V4V_REPORT_SCHEMA = "alix.ml-core.v4v-binary-paradigm.v1"
V4V_TIMEFRAMES = ("1h", "4h")
V4V_FEATURE_PROFILES = ("Combined",)
V4V_ARMS: tuple[dict[str, Any], ...] = (
    {
        "name": "CONTROL_SYMMETRIC_1R",
        "geometry": "SYMMETRIC_1R",
        "take_profit_multiplier": 1.0,
        "stop_loss_multiplier": 1.0,
        "horizon_candles": 24,
    },
    {
        "name": "PRIMARY_ASYMMETRIC_1_5R",
        "geometry": "ASYMMETRIC_1_5R",
        "take_profit_multiplier": 1.5,
        "stop_loss_multiplier": 1.0,
        "horizon_candles": 24,
    },
)
V4V_ATR_PERIOD = 14
V4V_ENTRY_SLIPPAGE_BPS = 20.0
V4V_ESTIMATED_ROUND_TRIP_COST_BPS = 90.0
V4V_FEE_BPS_PER_SIDE = 25.0
V4V_SLIPPAGE_BPS_PER_SIDE = 20.0
V4V_BUY_PROBABILITY_THRESHOLD = 0.65
V4V_MINIMUM_INNER_FIT_CLASS_SAMPLES = DEFAULT_MINIMUM_CLASS_SAMPLES
V4V_MINIMUM_INNER_VALIDATION_CLASS_SAMPLES = DEFAULT_MINIMUM_CLASS_SAMPLES
V4V_MAXIMUM_DRAWDOWN = 0.15
V4V_RANDOM_SEARCH_ITERATIONS = DEFAULT_RANDOM_SEARCH_ITERATIONS
V4V_OUTER_VALIDATION_START_FRACTION = 0.70
V4V_OUTER_VALIDATION_END_FRACTION = 0.85
V4V_EXPECTED_EXPERIMENTS = (
    len(SYMBOLS) * len(V4V_TIMEFRAMES) * len(V4V_ARMS) * len(V4V_FEATURE_PROFILES)
)
V4V_EXPECTED_CANDIDATES = V4V_EXPECTED_EXPERIMENTS * len(weighting_policy_specs())
V4V_SKIP_STATUSES = (
    "SKIPPED_INSUFFICIENT_CLASS_DIVERSITY",
    "SKIPPED_INSUFFICIENT_SAMPLES",
    "SKIPPED_INSUFFICIENT_INNER_CV_CLASS_SUPPORT",
)
V4V_PROBABILITY_COLUMNS = (
    "candidate_id",
    "arm",
    "pair",
    "timeframe",
    "feature_profile",
    "weighting_policy",
    "row_index",
    "timestamp",
    "split_role",
    "actual_label",
    "source_label",
    "predicted_label",
    "executable_signal",
    "prob_buy",
    "prob_no_buy",
)


def _execution_config() -> ExecutionConfig:
    return ExecutionConfig(
        initial_equity=10_000.0,
        fee_bps=V4V_FEE_BPS_PER_SIDE,
        slippage_bps=V4V_SLIPPAGE_BPS_PER_SIDE,
        position_fraction=1.0,
        maximum_drawdown=V4V_MAXIMUM_DRAWDOWN,
    )


def _candidate_identifier(
    arm: str,
    pair: str,
    timeframe: str,
    feature_profile: str,
    weighting_policy: str,
) -> str:
    return "|".join((arm, pair, timeframe, feature_profile, weighting_policy))


def _binary_label_distribution(values: Series) -> dict[str, Any]:
    observed = values.loc[values.isin(BINARY_BUY_LABELS)]
    total = int(len(observed))
    counts = observed.value_counts()
    return {
        "total": total,
        "classes": {
            label: {
                "count": int(counts.get(label, 0)),
                "percent": round(100.0 * int(counts.get(label, 0)) / total, 6)
                if total
                else 0.0,
            }
            for label in BINARY_BUY_LABELS
        },
    }


def _source_label_distribution(values: Series) -> dict[str, Any]:
    labels = ("BUY", "SELL", "HOLD")
    observed = values.loc[values.isin(labels)]
    total = int(len(observed))
    counts = observed.value_counts()
    return {
        "total": total,
        "classes": {
            label: {
                "count": int(counts.get(label, 0)),
                "percent": round(100.0 * int(counts.get(label, 0)) / total, 6)
                if total
                else 0.0,
            }
            for label in labels
        },
    }


def _outer_validation_bounds(row_count: int) -> tuple[int, int]:
    if row_count < 100:
        raise ValueError("v4v_outer_split_rows_invalid")
    validation_start = int(row_count * V4V_OUTER_VALIDATION_START_FRACTION)
    validation_end = int(row_count * V4V_OUTER_VALIDATION_END_FRACTION) - 1
    if validation_end <= validation_start:
        raise ValueError("v4v_outer_validation_window_invalid")
    return validation_start, validation_end


def _outer_label_end_indices(labels: DataFrame) -> list[int]:
    return [
        index if pd.isna(value) else int(value)
        for index, value in enumerate(labels["label_end_idx"].tolist())
    ]


def _binary_eligible_indices(features: DataFrame, labels: DataFrame) -> set[int]:
    eligible_mask = (
        features.notna().all(axis=1)
        & labels["label"].isin(BINARY_BUY_LABELS)
        & labels["label_end_idx"].notna()
    )
    return {int(index) for index in np.flatnonzero(eligible_mask.to_numpy(dtype=bool))}


def _retained_outer_split_indices(
    split: Any, eligible_indices: set[int], validation_start: int
) -> tuple[list[int], list[int], int]:
    train_limit = validation_start - int(split.embargo_size)
    train_indices = [
        int(index)
        for index in split.train_indices
        if index in eligible_indices and index < train_limit
    ]
    validation_indices = [
        int(index) for index in split.validation_indices if index in eligible_indices
    ]
    return train_indices, validation_indices, train_limit


def _validate_outer_split_chronology(
    train_indices: list[int],
    validation_indices: list[int],
    label_ends: list[int],
) -> int:
    indices_are_valid = (
        bool(train_indices)
        and bool(validation_indices)
        and train_indices == sorted(set(train_indices))
        and validation_indices == sorted(set(validation_indices))
        and train_indices[-1] < validation_indices[0]
    )
    if not indices_are_valid:
        raise ValueError("v4v_outer_split_chronology_invalid")
    maximum_train_label_end = max(label_ends[index] for index in train_indices)
    if maximum_train_label_end >= validation_indices[0]:
        raise ValueError("v4v_outer_split_label_overlap")
    return maximum_train_label_end


def _outer_split_diagnostics(
    *,
    validation_start: int,
    validation_end: int,
    train_limit: int,
    train_indices: list[int],
    validation_indices: list[int],
    maximum_train_label_end: int,
    split: Any,
    eligible_count: int,
) -> dict[str, Any]:
    return {
        "method": "PURGED_CHRONOLOGICAL_OUTER_HOLDOUT",
        "validation_start_index": validation_start,
        "validation_end_index": validation_end,
        "train_limit_index": train_limit,
        "train_start_index": train_indices[0],
        "train_end_index": train_indices[-1],
        "validation_first_index": validation_indices[0],
        "validation_last_index": validation_indices[-1],
        "maximum_training_label_end_index": maximum_train_label_end,
        "purged_train_rows": len(split.purged_indices),
        "pre_validation_embargo_candles": split.embargo_size,
        "post_validation_embargo_start_index": split.embargo_start,
        "post_validation_embargo_end_index": split.embargo_end,
        "eligible_sample_count": eligible_count,
        "inner_tuning_uses_outer_train_only": True,
    }


def _binary_outer_split(
    features: DataFrame,
    labels: DataFrame,
    *,
    horizon_candles: int,
) -> tuple[list[int], list[int], dict[str, Any]]:
    """Apply the established chronological purged split to binary labels."""
    row_count = len(labels)
    if len(features) != row_count:
        raise ValueError("v4v_outer_split_rows_invalid")
    validation_start, validation_end = _outer_validation_bounds(row_count)
    label_ends = _outer_label_end_indices(labels)
    split = create_purged_embargo_split(
        label_ends,
        validation_start=validation_start,
        validation_end=validation_end,
        label_horizon_candles=horizon_candles,
    )
    eligible_indices = _binary_eligible_indices(features, labels)
    train_indices, validation_indices, train_limit = _retained_outer_split_indices(
        split, eligible_indices, validation_start
    )
    maximum_train_label_end = _validate_outer_split_chronology(
        train_indices, validation_indices, label_ends
    )
    diagnostics = _outer_split_diagnostics(
        validation_start=validation_start,
        validation_end=validation_end,
        train_limit=train_limit,
        train_indices=train_indices,
        validation_indices=validation_indices,
        maximum_train_label_end=maximum_train_label_end,
        split=split,
        eligible_count=len(eligible_indices),
    )
    return train_indices, validation_indices, diagnostics


def _binary_class_counts(labels: Series) -> dict[str, int]:
    return {label: int((labels == label).sum()) for label in BINARY_BUY_LABELS}


def _classes_below_minimum(counts: dict[str, int], minimum_samples: int) -> list[str]:
    return [label for label, count in counts.items() if count < minimum_samples]


def _binary_sample_gate_status(
    missing_classes: list[str],
    insufficient_train: list[str],
    insufficient_validation: list[str],
) -> str:
    if missing_classes:
        return "SKIPPED_INSUFFICIENT_CLASS_DIVERSITY"
    if insufficient_train or insufficient_validation:
        return "SKIPPED_INSUFFICIENT_SAMPLES"
    return "PASS"


def _binary_sample_gate(
    train_labels: Series, validation_labels: Series
) -> dict[str, Any]:
    train_counts = _binary_class_counts(train_labels)
    validation_counts = _binary_class_counts(validation_labels)
    missing = [label for label, count in train_counts.items() if count == 0]
    insufficient_train = _classes_below_minimum(
        train_counts, MIN_TRAIN_SAMPLES_PER_CLASS
    )
    insufficient_validation = _classes_below_minimum(
        validation_counts, MIN_VALIDATION_SAMPLES_PER_CLASS
    )
    return {
        "status": _binary_sample_gate_status(
            missing, insufficient_train, insufficient_validation
        ),
        "class_labels": list(BINARY_BUY_LABELS),
        "minimum_train_samples_per_class": MIN_TRAIN_SAMPLES_PER_CLASS,
        "minimum_validation_samples_per_class": MIN_VALIDATION_SAMPLES_PER_CLASS,
        "train_class_counts": train_counts,
        "validation_class_counts": validation_counts,
        "missing_train_classes": missing,
        "insufficient_train_classes": insufficient_train,
        "insufficient_validation_classes": insufficient_validation,
    }


def _binary_inner_cv_preflight(
    train_indices: list[int],
    train_labels: Series,
    label_end_indices: list[int],
    horizon_candles: int,
) -> dict[str, Any]:
    try:
        _, diagnostics = build_purged_time_series_splits(
            train_indices,
            train_labels,
            label_end_indices,
            horizon_candles=horizon_candles,
            n_splits=DEFAULT_INNER_CV_SPLITS,
            minimum_class_samples=V4V_MINIMUM_INNER_FIT_CLASS_SAMPLES,
            class_labels=BINARY_BUY_LABELS,
            minimum_validation_class_samples=V4V_MINIMUM_INNER_VALIDATION_CLASS_SAMPLES,
        )
    except ValueError as error:
        reason = str(error)
        if reason not in {
            "tuning_inner_cv_class_support_insufficient",
            "tuning_inner_cv_validation_class_support_insufficient",
        }:
            raise
        return {
            "status": "SKIPPED_INSUFFICIENT_INNER_CV_CLASS_SUPPORT",
            "reason_code": reason,
            "expected_fold_count": DEFAULT_INNER_CV_SPLITS,
            "minimum_fit_class_samples": V4V_MINIMUM_INNER_FIT_CLASS_SAMPLES,
            "minimum_validation_class_samples": V4V_MINIMUM_INNER_VALIDATION_CLASS_SAMPLES,
            "complete_fold_validation": False,
            "folds": [],
        }
    return {
        "status": "PASS",
        "reason_code": None,
        "expected_fold_count": DEFAULT_INNER_CV_SPLITS,
        "minimum_fit_class_samples": V4V_MINIMUM_INNER_FIT_CLASS_SAMPLES,
        "minimum_validation_class_samples": V4V_MINIMUM_INNER_VALIDATION_CLASS_SAMPLES,
        "complete_fold_validation": len(diagnostics) == DEFAULT_INNER_CV_SPLITS,
        "folds": diagnostics,
    }


def _validate_economic_label_columns(labels: DataFrame) -> None:
    required = {
        "event_index",
        "label_end_idx",
        "source_label",
        "take_profit_price",
        "stop_loss_price",
        "exit_reason",
    }
    if not required.issubset(labels.columns):
        raise ValueError("v4v_economic_label_columns_missing")


def _economic_exit_reference(bars: DataFrame, row: Any, label_end: int) -> float:
    reason = str(getattr(row, "exit_reason"))
    if reason == "TP_HIT":
        return float(getattr(row, "take_profit_price"))
    if reason in {"SL_HIT", "COLLISION"}:
        return float(getattr(row, "stop_loss_price"))
    if reason == "HORIZON_EXPIRY":
        return float(bars.iloc[label_end]["close"])
    raise ValueError("v4v_economic_exit_reason_invalid")


def _supported_source_label(value: Any) -> bool:
    return not pd.isna(value) and str(value) in {"BUY", "SELL", "HOLD"}


def _economic_event_index(row: Any, bar_count: int) -> int | None:
    event_index = int(getattr(row, "event_index"))
    if event_index < 0 or event_index + 1 >= bar_count:
        return None
    return event_index


def _economic_label_end_index(row: Any, event_index: int, bar_count: int) -> int | None:
    raw_end = getattr(row, "label_end_idx")
    if pd.isna(raw_end):
        return None
    label_end = int(raw_end)
    if label_end <= event_index or label_end >= bar_count:
        raise ValueError("v4v_economic_label_end_index_invalid")
    return label_end


def _validate_economic_price_references(
    entry_reference: float, exit_reference: float
) -> None:
    prices_are_invalid = (
        not np.isfinite(entry_reference)
        or not np.isfinite(exit_reference)
        or entry_reference <= 0.0
        or exit_reference <= 0.0
    )
    if prices_are_invalid:
        raise ValueError("v4v_economic_price_reference_invalid")


def _economic_outcome_for_label(bars: DataFrame, row: Any) -> tuple[int, float] | None:
    source_label = getattr(row, "source_label")
    if not _supported_source_label(source_label):
        return None
    event_index = _economic_event_index(row, len(bars))
    if event_index is None:
        return None
    label_end = _economic_label_end_index(row, event_index, len(bars))
    if label_end is None:
        return None
    entry_reference = float(bars.iloc[event_index + 1]["open"])
    exit_reference = _economic_exit_reference(bars, row, label_end)
    _validate_economic_price_references(entry_reference, exit_reference)
    return event_index, exit_reference / entry_reference - 1.0


def _economic_outcomes_by_index(bars: DataFrame, labels: DataFrame) -> Series:
    """Build next-open gross long returns from the original barrier outcomes."""
    _validate_economic_label_columns(labels)
    values = np.full(len(bars), np.nan, dtype=float)
    for row in labels.itertuples(index=False):
        outcome = _economic_outcome_for_label(bars, row)
        if outcome is not None:
            event_index, gross_return = outcome
            values[event_index] = gross_return
    return Series(values, index=bars.index, dtype=float)


def _new_binary_classifier(class_weight: str | None) -> Any:
    lightgbm = importlib.import_module("lightgbm")
    classifier = getattr(lightgbm, "LGBMClassifier")
    return classifier(
        objective="binary",
        n_estimators=80,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        max_depth=6,
        class_weight=class_weight,
        random_state=42,
        n_jobs=1,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )


def _buy_probabilities(model: Any, features: DataFrame) -> np.ndarray:
    classes = [str(value) for value in getattr(model, "classes_", [])]
    if len(classes) != 2 or set(classes) != set(BINARY_BUY_LABELS):
        raise ValueError("v4v_model_probability_classes_invalid")
    raw = np.asarray(model.predict_proba(features), dtype=float)
    if raw.shape != (len(features), 2):
        raise ValueError("v4v_model_probability_shape_invalid")
    buy = raw[:, classes.index("BUY")]
    if (
        not bool(np.isfinite(raw).all())
        or bool(np.any(raw < 0.0))
        or bool(np.any(raw > 1.0))
        or not bool(np.allclose(raw.sum(axis=1), 1.0, atol=1e-6))
    ):
        raise ValueError("v4v_model_probabilities_invalid")
    return buy


def _binary_classification_metrics(
    actual: Series,
    buy_probabilities: np.ndarray,
) -> dict[str, Any]:
    if (
        len(actual) != len(buy_probabilities)
        or not bool(actual.isin(BINARY_BUY_LABELS).all())
        or int(actual.nunique()) != 2
    ):
        raise ValueError("v4v_binary_metric_support_invalid")
    probabilities = np.asarray(buy_probabilities, dtype=float)
    if (
        probabilities.ndim != 1
        or not bool(np.isfinite(probabilities).all())
        or bool(np.any(probabilities < 0.0))
        or bool(np.any(probabilities > 1.0))
    ):
        raise ValueError("v4v_binary_metric_probabilities_invalid")
    predicted_values = np.where(
        probabilities >= V4V_BUY_PROBABILITY_THRESHOLD, "BUY", "NO_BUY"
    )
    actual_values = actual.to_numpy(dtype=object)
    targets = (actual_values == "BUY").astype(int)
    predicted = (predicted_values == "BUY").astype(int)
    matrix = confusion_matrix(
        actual_values,
        predicted_values,
        labels=list(BINARY_BUY_LABELS),
    )
    return {
        "sample_count": int(len(actual)),
        "threshold": V4V_BUY_PROBABILITY_THRESHOLD,
        "class_labels": list(BINARY_BUY_LABELS),
        "buy_support": int(np.count_nonzero(targets)),
        "no_buy_support": int(len(targets) - np.count_nonzero(targets)),
        "buy_prediction_count": int(np.count_nonzero(predicted)),
        "no_buy_prediction_count": int(len(predicted) - np.count_nonzero(predicted)),
        "buy_precision": round(
            float(precision_score(targets, predicted, zero_division=0)), 8
        ),
        "buy_recall": round(
            float(recall_score(targets, predicted, zero_division=0)), 8
        ),
        "buy_f1": round(float(f1_score(targets, predicted, zero_division=0)), 8),
        "roc_auc": round(float(roc_auc_score(targets, probabilities)), 8),
        "average_precision": round(
            float(average_precision_score(targets, probabilities)), 8
        ),
        "brier_score": round(
            float(np.mean((probabilities - targets.astype(float)) ** 2)), 8
        ),
        "confusion_matrix": {
            "labels": list(BINARY_BUY_LABELS),
            "rows_are": "actual",
            "columns_are": "predicted",
            "matrix": [[int(value) for value in row] for row in matrix.tolist()],
        },
    }


def _feature_importance_gain(model: Any, features: DataFrame) -> list[dict[str, Any]]:
    booster = getattr(model, "booster_", None)
    if booster is None:
        raise ValueError("v4v_feature_importance_booster_missing")
    feature_names = [str(name) for name in booster.feature_name()]
    expected_names = [str(name) for name in features.columns]
    if feature_names != expected_names:
        raise ValueError("v4v_feature_importance_names_mismatch")
    gains = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
    if (
        gains.ndim != 1
        or len(gains) != len(feature_names)
        or not bool(np.isfinite(gains).all())
        or bool(np.any(gains < 0.0))
    ):
        raise ValueError("v4v_feature_importance_gain_invalid")
    return sorted(
        [
            {"feature": name, "gain": round(float(gain), 8)}
            for name, gain in zip(feature_names, gains, strict=True)
        ],
        key=lambda item: (-float(item["gain"]), str(item["feature"])),
    )


def _candidate_probability_records(
    bars: DataFrame,
    validation_indices: list[int],
    actual_binary: Series,
    source_labels: Series,
    buy_probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    if not (
        len(validation_indices)
        == len(actual_binary)
        == len(source_labels)
        == len(buy_probabilities)
    ):
        raise ValueError("v4v_probability_record_alignment_invalid")
    records: list[dict[str, Any]] = []
    for row_index, actual_label, source_label, buy_probability in zip(
        validation_indices,
        actual_binary.tolist(),
        source_labels.tolist(),
        buy_probabilities,
        strict=True,
    ):
        probability = float(buy_probability)
        records.append(
            {
                "row_index": int(row_index),
                "timestamp": pd.Timestamp(bars["date"].iloc[row_index]).isoformat(),
                "split_role": "FORWARD_VALIDATION",
                "actual_label": str(actual_label),
                "source_label": str(source_label),
                "predicted_label": (
                    "BUY" if probability >= V4V_BUY_PROBABILITY_THRESHOLD else "NO_BUY"
                ),
                "executable_signal": (
                    "BUY"
                    if probability >= V4V_BUY_PROBABILITY_THRESHOLD
                    else "NO_ENTRY"
                ),
                "prob_buy": round(probability, 8),
                "prob_no_buy": round(1.0 - probability, 8),
            }
        )
    return records


def _execution_timestamp(record: dict[str, Any]) -> pd.Timestamp:
    timestamp = pd.Timestamp(record["timestamp"])
    if timestamp.tzinfo is None:
        raise ValueError("v4v_probability_timestamp_timezone_required")
    return timestamp.tz_convert("UTC")


def _validate_execution_probability(value: Any) -> float:
    probability = float(value)
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("v4v_buy_probability_invalid")
    return probability


def _validate_execution_order(
    row_index: int,
    timestamp: pd.Timestamp,
    previous_index: int,
    previous_timestamp: pd.Timestamp | None,
) -> None:
    index_out_of_order = row_index <= previous_index
    timestamp_out_of_order = (
        previous_timestamp is not None and timestamp <= previous_timestamp
    )
    if index_out_of_order or timestamp_out_of_order:
        raise ValueError("v4v_probability_order_invalid")


def _execution_signal_record(
    record: dict[str, Any],
    delta: timedelta,
) -> dict[str, Any]:
    row_index = int(record["row_index"])
    timestamp = _execution_timestamp(record)
    buy_probability = _validate_execution_probability(record["prob_buy"])
    is_buy = buy_probability >= V4V_BUY_PROBABILITY_THRESHOLD
    return {
        "row_index": row_index,
        "signal_timestamp": timestamp,
        "execution_timestamp": timestamp + delta,
        "signal": "BUY" if is_buy else "NO_ENTRY",
        "raw_signal": "BUY" if buy_probability >= 0.50 else "NO_BUY",
        "buy_probability": buy_probability,
        "confidence_threshold": V4V_BUY_PROBABILITY_THRESHOLD,
        "confidence_passed": is_buy,
    }


def _execution_signals(
    probability_records: list[dict[str, Any]],
    timeframe: str,
) -> list[dict[str, Any]]:
    if timeframe not in TIMEFRAME_MINUTES:
        raise ValueError("v4v_execution_timeframe_unsupported")
    delta = timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
    signals: list[dict[str, Any]] = []
    previous_index = -1
    previous_timestamp: pd.Timestamp | None = None
    for record in probability_records:
        signal = _execution_signal_record(record, delta)
        _validate_execution_order(
            int(signal["row_index"]),
            signal["signal_timestamp"],
            previous_index,
            previous_timestamp,
        )
        signals.append(signal)
        previous_index = int(signal["row_index"])
        previous_timestamp = signal["signal_timestamp"]
    if not signals:
        raise ValueError("v4v_forward_signals_empty")
    return signals


def _profit_factor(values: list[float]) -> float | None:
    gross_profit = sum(value for value in values if value > 0.0)
    gross_loss = abs(sum(value for value in values if value < 0.0))
    if gross_loss == 0.0:
        return None
    return round(gross_profit / gross_loss, 8)


def _trade_values(trades: list[dict[str, Any]], field_name: str) -> list[float]:
    return [float(item[field_name]) for item in trades]


def _profit_factor_has_no_losses(values: list[float]) -> bool:
    has_profit = sum(value for value in values if value > 0.0) > 0.0
    has_losses = any(value < 0.0 for value in values)
    return bool(values) and has_profit and not has_losses


def _trade_side_metrics(
    trades: list[dict[str, Any]], pnl_field: str, prefix: str
) -> dict[str, Any]:
    values = _trade_values(trades, pnl_field)
    return {
        f"{prefix}_profit_factor": _profit_factor(values),
        f"{prefix}_profit_factor_infinite_no_losses": (
            _profit_factor_has_no_losses(values)
        ),
        f"{prefix}_pnl_usd": round(sum(values), 8),
    }


def _trade_count_metrics(
    trades: list[dict[str, Any]], net_values: list[float]
) -> dict[str, Any]:
    wins = sum(value > 0.0 for value in net_values)
    losses = sum(value < 0.0 for value in net_values)
    return {
        "trade_count": len(trades),
        "winning_trade_count": wins,
        "losing_trade_count": losses,
        "win_rate": round(wins / len(trades), 8) if trades else None,
    }


def _trade_cost_metrics(trades: list[dict[str, Any]]) -> dict[str, float]:
    fees = sum(float(item["fees_usd"]) for item in trades)
    slippage = sum(float(item["slippage_cost_usd"]) for item in trades)
    return {
        "fees_usd": round(fees, 8),
        "slippage_cost_usd": round(slippage, 8),
    }


def _trade_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    net_values = _trade_values(trades, "net_pnl_usd")
    net_metrics = _trade_side_metrics(trades, "net_pnl_usd", "net")
    gross_metrics = _trade_side_metrics(trades, "gross_market_pnl_usd", "gross")
    gross_metrics["gross_market_pnl_usd"] = gross_metrics.pop("gross_pnl_usd")
    return {
        **_trade_count_metrics(trades, net_values),
        **net_metrics,
        **gross_metrics,
        **_trade_cost_metrics(trades),
    }


def _evaluate_trained_candidate(
    *,
    arm: dict[str, Any],
    pair: str,
    timeframe: str,
    bars: DataFrame,
    feature_profile: str,
    weighting_policy: str,
    experiment: dict[str, Any],
    model: Any,
    tuning_diagnostics: dict[str, Any],
    train_features: DataFrame,
    validation_features: DataFrame,
    validation_labels: Series,
    validation_source_labels: Series,
    validation_indices: list[int],
    feature_importance: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidate_id = _candidate_identifier(
        str(arm["name"]), pair, timeframe, feature_profile, weighting_policy
    )
    buy_probabilities = _buy_probabilities(model, validation_features)
    classification = _binary_classification_metrics(
        validation_labels, buy_probabilities
    )
    probability_records = _candidate_probability_records(
        bars,
        validation_indices,
        validation_labels,
        validation_source_labels,
        buy_probabilities,
    )
    signals = _execution_signals(probability_records, timeframe)
    management = TradeManagementConfig(
        atr_period=V4V_ATR_PERIOD,
        take_profit_atr_multiplier=float(arm["take_profit_multiplier"]),
        stop_loss_atr_multiplier=float(arm["stop_loss_multiplier"]),
        horizon_candles=int(arm["horizon_candles"]),
        confidence_threshold=V4V_BUY_PROBABILITY_THRESHOLD,
        trailing_stop_enabled=False,
    )
    replay = simulate_signals(
        bars,
        signals,
        _execution_config(),
        trade_management_config=management,
    )
    raw_buy_count = sum(
        float(record["prob_buy"]) >= 0.50 for record in probability_records
    )
    accepted_buy_count = sum(
        float(record["prob_buy"]) >= V4V_BUY_PROBABILITY_THRESHOLD
        for record in probability_records
    )
    replay["confidence_filter"] = {
        "threshold": V4V_BUY_PROBABILITY_THRESHOLD,
        "threshold_selection": "FIXED_OWNER_DIRECTIVE",
        "validation_rows": len(probability_records),
        "raw_buy_count_at_0_50": raw_buy_count,
        "accepted_buy_count_at_0_65": accepted_buy_count,
        "confidence_rejected_buy_count": raw_buy_count - accepted_buy_count,
        "predicted_no_buy_count": len(probability_records) - raw_buy_count,
        "missed_opportunity_penalty": False,
        "no_buy_closes_existing_position": False,
    }
    trades = list(replay["trades"])
    metrics = dict(replay["metrics"])
    metrics.update(_trade_metrics(trades))
    profit_factor_infinite = bool(
        metrics.get("net_profit_factor_infinite_no_losses", False)
    )
    profit_factor = metrics.get("profit_factor_net_of_costs")
    profit_factor_passed = profit_factor_infinite or (
        profit_factor is not None and float(cast(float, profit_factor)) > 1.5
    )
    metrics["economic_candidate_gate_passed"] = bool(
        profit_factor_passed
        and float(metrics["max_drawdown_fraction"]) < V4V_MAXIMUM_DRAWDOWN
    )
    candidate = {
        "candidate_id": candidate_id,
        "status": "TRAINED",
        "evaluation_performed": True,
        "arm": str(arm["name"]),
        "geometry": str(arm["geometry"]),
        "pair": pair,
        "timeframe": timeframe,
        "feature_profile": feature_profile,
        "weighting_policy": weighting_policy,
        "class_labels": list(BINARY_BUY_LABELS),
        "train_rows": int(experiment["train_rows"]),
        "validation_rows": int(experiment["validation_rows"]),
        "tuning_objective": BINARY_BUY_OBJECTIVE,
        "tuning_diagnostics": dict(tuning_diagnostics),
        "selected_parameters": dict(tuning_diagnostics["selected_parameters"]),
        "model_class_order": [str(value) for value in model.classes_],
        "buy_probability_threshold": V4V_BUY_PROBABILITY_THRESHOLD,
        "validation_classification_metrics": classification,
        "metrics": metrics,
        "confidence_filter": dict(replay["confidence_filter"]),
        "raw_binary_prediction_counts": {
            "BUY_AT_0_50": raw_buy_count,
            "NO_BUY_BELOW_0_50": len(probability_records) - raw_buy_count,
        },
        "executable_signal_counts": {
            "BUY_AT_0_65": accepted_buy_count,
            "NO_ENTRY_BELOW_0_65": len(probability_records) - accepted_buy_count,
        },
        "exit_reason_counts": dict(replay["exit_reason_counts"]),
        "total_fees_usd": float(replay["total_fees_usd"]),
        "total_slippage_cost_usd": float(replay["total_slippage_cost_usd"]),
        "trade_management": dict(replay["trade_management"]),
        "trades": trades,
        "equity_curve": list(replay["equity_curve"]),
        "feature_importance_gain": feature_importance,
        "probability_records": probability_records,
        "execution_signals": signals,
        "model_family": "LightGBM LGBMClassifier objective=binary",
        "sequence_model_used": False,
    }
    probability_rows = _probability_artifact_rows(
        candidate_id,
        arm,
        pair,
        timeframe,
        feature_profile,
        weighting_policy,
        probability_records,
    )
    return candidate, probability_rows


def _probability_artifact_rows(
    candidate_id: str,
    arm: dict[str, Any],
    pair: str,
    timeframe: str,
    feature_profile: str,
    weighting_policy: str,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "candidate_id": candidate_id,
                "arm": str(arm["name"]),
                "pair": pair,
                "timeframe": timeframe,
                "feature_profile": feature_profile,
                "weighting_policy": weighting_policy,
                "row_index": int(record["row_index"]),
                "timestamp": str(record["timestamp"]),
                "split_role": str(record["split_role"]),
                "actual_label": str(record["actual_label"]),
                "source_label": str(record["source_label"]),
                "predicted_label": str(record["predicted_label"]),
                "executable_signal": str(record["executable_signal"]),
                "prob_buy": float(record["prob_buy"]),
                "prob_no_buy": float(record["prob_no_buy"]),
            }
        )
    return rows


def _skipped_candidate(
    arm: dict[str, Any],
    pair: str,
    timeframe: str,
    feature_profile: str,
    weighting_policy: str,
    experiment: dict[str, Any],
) -> dict[str, Any]:
    status = str(experiment.get("status"))
    if status not in V4V_SKIP_STATUSES:
        raise ValueError("v4v_skip_status_invalid")
    return {
        "candidate_id": _candidate_identifier(
            str(arm["name"]), pair, timeframe, feature_profile, weighting_policy
        ),
        "status": status,
        "evaluation_performed": False,
        "arm": str(arm["name"]),
        "geometry": str(arm["geometry"]),
        "pair": pair,
        "timeframe": timeframe,
        "feature_profile": feature_profile,
        "weighting_policy": weighting_policy,
        "class_labels": list(BINARY_BUY_LABELS),
        "train_rows": int(experiment["train_rows"]),
        "validation_rows": int(experiment["validation_rows"]),
        "sample_gate": dict(experiment["sample_gate"]),
        "tuning_diagnostics": {},
        "metrics": {
            "trade_count": 0,
            "win_rate": None,
            "profit_factor_net_of_costs": None,
            "max_drawdown_percent": None,
            "economic_candidate_gate_passed": False,
        },
        "trades": [],
        "equity_curve": [],
        "feature_importance_gain": [],
        "probability_records": [],
        "execution_signals": [],
        "model_family": "LightGBM LGBMClassifier objective=binary",
        "sequence_model_used": False,
    }


def _run_binary_experiment(
    *,
    arm: dict[str, Any],
    pair: str,
    timeframe: str,
    bars: DataFrame,
    feature_matrices: dict[str, DataFrame],
    feature_profile: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    raw_labels = apply_adaptive_triple_barrier(
        bars,
        atr_period=V4V_ATR_PERIOD,
        horizon_candles=int(arm["horizon_candles"]),
        take_profit_multiplier=float(arm["take_profit_multiplier"]),
        stop_loss_multiplier=float(arm["stop_loss_multiplier"]),
        entry_reference="NEXT_BAR_OPEN",
        entry_slippage_bps=V4V_ENTRY_SLIPPAGE_BPS,
    )
    economic_outcomes = _economic_outcomes_by_index(
        bars, raw_labels.assign(source_label=raw_labels["label"])
    )
    labels = apply_binary_buy_targets(raw_labels)
    features = feature_matrices[feature_profile]
    train_indices, validation_indices, split_diagnostics = _binary_outer_split(
        features,
        labels,
        horizon_candles=int(arm["horizon_candles"]),
    )
    train_labels = labels.iloc[train_indices]["label"]
    validation_labels = labels.iloc[validation_indices]["label"]
    sample_gate = _binary_sample_gate(train_labels, validation_labels)
    experiment: dict[str, Any] = {
        "arm": str(arm["name"]),
        "geometry": str(arm["geometry"]),
        "pair": pair,
        "timeframe": timeframe,
        "feature_profile": feature_profile,
        "status": str(sample_gate["status"]),
        "bar_count": int(len(bars)),
        "train_rows": len(train_indices),
        "validation_rows": len(validation_indices),
        "sample_gate": sample_gate,
        "split": split_diagnostics,
        "binary_label_distribution": {
            "train": _binary_label_distribution(train_labels),
            "validation": _binary_label_distribution(validation_labels),
        },
        "source_label_distribution": {
            "train": _source_label_distribution(
                labels.iloc[train_indices]["source_label"]
            ),
            "validation": _source_label_distribution(
                labels.iloc[validation_indices]["source_label"]
            ),
        },
        "label_contract": {
            "schema_version": BINARY_TARGET_SCHEMA_VERSION,
            "class_mapping": {
                "BUY": "BUY",
                "SELL": "NO_BUY",
                "HOLD": "NO_BUY",
            },
            "unlabeled_rows_remain_unlabeled": True,
        },
        "model_variants": {},
    }
    candidates: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    if sample_gate["status"] != "PASS":
        experiment["skip_reason"] = str(sample_gate["status"])
        for weighting_policy_name in weighting_policy_specs():
            candidates.append(
                _skipped_candidate(
                    arm,
                    pair,
                    timeframe,
                    feature_profile,
                    weighting_policy_name,
                    experiment,
                )
            )
        return experiment, candidates, probability_rows

    label_end_indices = [
        index if pd.isna(value) else int(value)
        for index, value in enumerate(labels["label_end_idx"].tolist())
    ]
    inner_preflight = _binary_inner_cv_preflight(
        train_indices,
        train_labels,
        label_end_indices,
        int(arm["horizon_candles"]),
    )
    experiment["split"]["inner_cv_preflight"] = inner_preflight
    if inner_preflight["status"] != "PASS":
        experiment["status"] = str(inner_preflight["status"])
        experiment["skip_reason"] = str(inner_preflight["reason_code"])
        experiment["sample_gate"]["status"] = str(inner_preflight["status"])
        for weighting_policy_name in weighting_policy_specs():
            candidates.append(
                _skipped_candidate(
                    arm,
                    pair,
                    timeframe,
                    feature_profile,
                    weighting_policy_name,
                    experiment,
                )
            )
        return experiment, candidates, probability_rows

    x_train = features.iloc[train_indices]
    x_validation = features.iloc[validation_indices]
    validation_source_labels = labels.iloc[validation_indices]["source_label"]
    for policy_name, policy_config in weighting_policy_specs().items():
        model, tuning_diagnostics = tune_lightgbm_binary_classifier(
            _new_binary_classifier(policy_config["class_weight"]),
            x_train,
            train_labels,
            train_indices,
            label_end_indices,
            horizon_candles=int(arm["horizon_candles"]),
            class_weight_policy=policy_name,
            n_iter=V4V_RANDOM_SEARCH_ITERATIONS,
            n_splits=DEFAULT_INNER_CV_SPLITS,
            minimum_class_samples=V4V_MINIMUM_INNER_FIT_CLASS_SAMPLES,
            minimum_validation_class_samples=V4V_MINIMUM_INNER_VALIDATION_CLASS_SAMPLES,
            economic_outcomes=economic_outcomes,
            estimated_cost_bps=V4V_ESTIMATED_ROUND_TRIP_COST_BPS,
            buy_probability_threshold=V4V_BUY_PROBABILITY_THRESHOLD,
        )
        feature_importance = _feature_importance_gain(model, x_train)
        candidate, rows = _evaluate_trained_candidate(
            arm=arm,
            pair=pair,
            timeframe=timeframe,
            bars=bars,
            feature_profile=feature_profile,
            weighting_policy=policy_name,
            experiment=experiment,
            model=model,
            tuning_diagnostics=tuning_diagnostics,
            train_features=x_train,
            validation_features=x_validation,
            validation_labels=validation_labels,
            validation_source_labels=validation_source_labels,
            validation_indices=validation_indices,
            feature_importance=feature_importance,
        )
        candidates.append(candidate)
        probability_rows.extend(rows)
        experiment["model_variants"][policy_name] = {
            "candidate_id": candidate["candidate_id"],
            "status": candidate["status"],
            "selected_parameters": dict(candidate["selected_parameters"]),
            "tuning_objective": candidate["tuning_objective"],
            "selected_mean_cv_binary_buy_economic_score": tuning_diagnostics[
                "selected_mean_cv_binary_buy_economic_score"
            ],
            "validation_classification_metrics": dict(
                candidate["validation_classification_metrics"]
            ),
            "trade_metrics": _trade_metrics(list(candidate["trades"])),
        }
    experiment["status"] = "TRAINED"
    return experiment, candidates, probability_rows


def _run_timeframe_experiments(
    symbol: str,
    timeframe: str,
    source_bars: DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pair = f"{symbol[:-4]}/USDT"
    bars = resample_ohlcv(source_bars, timeframe)
    if len(bars) < 100:
        raise ValueError(f"v4v_insufficient_resampled_bars:{symbol}:{timeframe}")
    feature_matrices = build_feature_profiles(bars, profile_names=V4V_FEATURE_PROFILES)
    experiments: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    for arm in V4V_ARMS:
        for feature_profile in V4V_FEATURE_PROFILES:
            experiment, variant_candidates, rows = _run_binary_experiment(
                arm=arm,
                pair=pair,
                timeframe=timeframe,
                bars=bars,
                feature_matrices=feature_matrices,
                feature_profile=feature_profile,
            )
            experiments.append(experiment)
            candidates.extend(variant_candidates)
            probability_rows.extend(rows)
    return experiments, candidates, probability_rows


def _validate_v4v_experiment_batch(
    experiments: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    probability_rows: list[dict[str, Any]],
) -> None:
    if len(experiments) != V4V_EXPECTED_EXPERIMENTS:
        raise RuntimeError("v4v_experiment_count_mismatch")
    if len(candidates) != V4V_EXPECTED_CANDIDATES:
        raise RuntimeError("v4v_candidate_count_mismatch")
    trained = [item for item in candidates if item["status"] == "TRAINED"]
    if not trained:
        raise RuntimeError("v4v_no_trained_candidates")
    expected_probability_rows = sum(int(item["validation_rows"]) for item in trained)
    if len(probability_rows) != expected_probability_rows:
        raise RuntimeError("v4v_probability_row_count_mismatch")


def run_v4v_experiments(
    datasets: dict[str, DataFrame],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if set(datasets) != set(SYMBOLS):
        raise ValueError("v4v_dataset_symbol_contract_mismatch")
    experiments: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        for timeframe in V4V_TIMEFRAMES:
            batch = _run_timeframe_experiments(symbol, timeframe, datasets[symbol])
            experiments.extend(batch[0])
            candidates.extend(batch[1])
            probability_rows.extend(batch[2])
    _validate_v4v_experiment_batch(experiments, candidates, probability_rows)
    return experiments, candidates, probability_rows


def _candidate_comparison_rows(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        metrics = candidate["metrics"]
        classification = candidate.get("validation_classification_metrics", {})
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "arm": candidate["arm"],
                "geometry": candidate["geometry"],
                "pair": candidate["pair"],
                "timeframe": candidate["timeframe"],
                "feature_profile": candidate["feature_profile"],
                "weighting_policy": candidate["weighting_policy"],
                "status": candidate["status"],
                "trade_count": metrics.get("trade_count", 0),
                "win_rate": metrics.get("win_rate"),
                "net_profit_factor": metrics.get("profit_factor_net_of_costs"),
                "gross_profit_factor": metrics.get("gross_profit_factor"),
                "net_pnl_usd": metrics.get("net_pnl_usd"),
                "max_drawdown_percent": metrics.get("max_drawdown_percent"),
                "buy_precision": classification.get("buy_precision"),
                "buy_recall": classification.get("buy_recall"),
                "roc_auc": classification.get("roc_auc"),
                "buy_probability_threshold": candidate.get("buy_probability_threshold"),
            }
        )
    return rows


def _arm_candidate_sets(
    candidates: list[dict[str, Any]], arm_name: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = [item for item in candidates if item["arm"] == arm_name]
    trained = [item for item in selected if item["status"] == "TRAINED"]
    return selected, trained


def _arm_trade_list(trained: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [trade for candidate in trained for trade in candidate["trades"]]


def _arm_candidate_drawdowns(trained: list[dict[str, Any]]) -> list[float]:
    return [
        float(candidate["metrics"]["max_drawdown_percent"])
        for candidate in trained
        if candidate["metrics"].get("max_drawdown_percent") is not None
    ]


def _arm_economic_pass_count(trained: list[dict[str, Any]]) -> int:
    return sum(
        bool(candidate["metrics"].get("economic_candidate_gate_passed"))
        for candidate in trained
    )


def _arm_aggregate(candidates: list[dict[str, Any]], arm_name: str) -> dict[str, Any]:
    selected, trained = _arm_candidate_sets(candidates, arm_name)
    trades = _arm_trade_list(trained)
    drawdowns = _arm_candidate_drawdowns(trained)
    return {
        "candidate_count": len(selected),
        "trained_candidate_count": len(trained),
        "skipped_candidate_count": len(selected) - len(trained),
        **_trade_metrics(trades),
        "mean_candidate_max_drawdown_percent": (
            round(float(np.mean(drawdowns)), 8) if drawdowns else None
        ),
        "economic_pass_candidate_count": _arm_economic_pass_count(trained),
    }


def _trade_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for index, trade in enumerate(candidate.get("trades", [])):
            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "arm": candidate["arm"],
                    "pair": candidate["pair"],
                    "timeframe": candidate["timeframe"],
                    "feature_profile": candidate["feature_profile"],
                    "weighting_policy": candidate["weighting_policy"],
                    "trade_index": index,
                    "entry_timestamp": trade["entry_timestamp"],
                    "exit_timestamp": trade["exit_timestamp"],
                    "entry_reference_price": trade["entry_reference_price"],
                    "entry_fill_price": trade["entry_fill_price"],
                    "exit_reference_price": trade["exit_reference_price"],
                    "exit_fill_price": trade["exit_fill_price"],
                    "gross_market_pnl_usd": trade["gross_market_pnl_usd"],
                    "fees_usd": trade["fees_usd"],
                    "slippage_cost_usd": trade["slippage_cost_usd"],
                    "net_pnl_usd": trade["net_pnl_usd"],
                    "exit_reason": trade["exit_reason"],
                    "direction": trade["direction"],
                }
            )
    return rows


def _feature_importance_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for item in candidate.get("feature_importance_gain", []):
            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "arm": candidate["arm"],
                    "pair": candidate["pair"],
                    "timeframe": candidate["timeframe"],
                    "feature_profile": candidate["feature_profile"],
                    "weighting_policy": candidate["weighting_policy"],
                    "feature": str(item["feature"]),
                    "gain": float(item["gain"]),
                }
            )
    return rows


def _report_candidate_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    excluded = {
        "trades",
        "equity_curve",
        "feature_importance_gain",
        "probability_records",
        "execution_signals",
    }
    return [
        {key: value for key, value in candidate.items() if key not in excluded}
        for candidate in candidates
    ]


def _build_report(
    *,
    commit_sha: str,
    provenance: list[dict[str, Any]],
    experiments: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    probability_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    feature_manifest = feature_profile_manifest(V4V_FEATURE_PROFILES)
    feature_columns = cast(list[str], feature_manifest[0]["columns"])
    comparison_rows = _candidate_comparison_rows(candidates)
    trade_rows = _trade_rows(candidates)
    feature_rows = _feature_importance_rows(candidates)
    trained = [item for item in candidates if item["status"] == "TRAINED"]
    return {
        "schema_version": V4V_REPORT_SCHEMA,
        "commit_sha": commit_sha,
        "requested_days": 730,
        "provider": "Binance Spot public klines API",
        "input_datasets": provenance,
        "model_training_performed": bool(trained),
        "economic_evaluation_performed": bool(trained),
        "economic_metrics_reported": bool(trained),
        "symbols": list(SYMBOLS),
        "market_timeframes": list(V4V_TIMEFRAMES),
        "feature_profiles": list(V4V_FEATURE_PROFILES),
        "feature_manifest": feature_manifest,
        "model": {
            "family": "LightGBM LGBMClassifier objective=binary",
            "class_labels": list(BINARY_BUY_LABELS),
            "positive_class": "BUY",
            "negative_class": "NO_BUY",
            "sequence_model_used": False,
            "new_models_trained": len(trained),
        },
        "experiment_count": len(experiments),
        "candidate_count": len(candidates),
        "trained_candidate_count": len(trained),
        "skipped_candidate_count": len(candidates) - len(trained),
        "trained_experiment_count": sum(
            item["status"] == "TRAINED" for item in experiments
        ),
        "experimental_design": "PAIRED_1R_VS_1_5R_BINARY_BUY_NO_BUY",
        "arms": [
            {
                "name": str(arm["name"]),
                "geometry": str(arm["geometry"]),
                "take_profit_multiplier": float(arm["take_profit_multiplier"]),
                "stop_loss_multiplier": float(arm["stop_loss_multiplier"]),
                "horizon_candles": int(arm["horizon_candles"]),
                "aggregate": _arm_aggregate(candidates, str(arm["name"])),
            }
            for arm in V4V_ARMS
        ],
        "target_contract": {
            "schema_version": BINARY_TARGET_SCHEMA_VERSION,
            "source_label_schema": "BUY_SELL_HOLD_TRIPLE_BARRIER",
            "class_mapping": {
                "BUY": "BUY",
                "SELL": "NO_BUY",
                "HOLD": "NO_BUY",
            },
            "unlabeled_rows_remain_unlabeled": True,
            "source_label_retained_for_audit": True,
            "atr_period": V4V_ATR_PERIOD,
            "horizon_candles": 24,
            "label_entry_reference": "NEXT_BAR_OPEN",
            "entry_slippage_bps": V4V_ENTRY_SLIPPAGE_BPS,
            "same_candle_collision_policy": "STOP_LOSS_WINS",
            "common_outer_and_inner_purge_horizon_candles": 24,
        },
        "feature_contract": {
            "schema_version": EXTENDED_FEATURE_SCHEMA_VERSION,
            "selected_profile": "Combined",
            "groups": ["Baseline", "Trend", "Volatility", "Volume"],
            "feature_count": len(feature_columns),
            "point_in_time": True,
            "volatility_features_included": True,
            "volume_features_included": True,
            "microstructure_data_used": False,
        },
        "objective_contract": {
            "primary_objective": BINARY_BUY_OBJECTIVE,
            "scorer": "Binary_BUY_Economic_Score",
            "scorer_scope": "PURGED_INNER_CV_FOLDS_WITHIN_OUTER_TRAIN",
            "scored_rows": "PREDICTED_BUY_ONLY",
            "buy_probability_threshold": V4V_BUY_PROBABILITY_THRESHOLD,
            "estimated_round_trip_cost_bps": V4V_ESTIMATED_ROUND_TRIP_COST_BPS,
            "missed_opportunity_penalty_bps": 0.0,
            "no_buy_predictions_are_penalized": False,
            "minimum_trade_count_penalty": False,
            "drawdown_penalty_weight": 0.5,
            "positive_net_expectancy_reward_multiplier": 2.0,
            "portfolio_overlap_modeled": False,
            "economic_acceptance_authority": "FORWARD_VALIDATION_COSTED_SPOT_EXECUTION_REPLAY",
        },
        "confidence_threshold_contract": {
            "method": "FIXED_OWNER_DIRECTIVE_THRESHOLD",
            "buy_probability_threshold": V4V_BUY_PROBABILITY_THRESHOLD,
            "comparison": "GREATER_THAN_OR_EQUAL",
            "selected_using_validation_labels": False,
            "calibration_window_used": False,
            "no_buy_execution_mapping": "NO_ENTRY",
        },
        "inner_cv_contract": {
            "method": "PURGED_EXPANDING_TIME_SERIES",
            "fold_count": DEFAULT_INNER_CV_SPLITS,
            "minimum_fit_samples_per_class": V4V_MINIMUM_INNER_FIT_CLASS_SAMPLES,
            "minimum_validation_samples_per_class": V4V_MINIMUM_INNER_VALIDATION_CLASS_SAMPLES,
            "outer_train_only": True,
            "both_classes_required_in_every_fold": True,
        },
        "execution_cost_contract": {
            "fee_bps_per_side": V4V_FEE_BPS_PER_SIDE,
            "slippage_bps_per_side": V4V_SLIPPAGE_BPS_PER_SIDE,
            "round_trip_nominal_cost_bps": 2.0
            * (V4V_FEE_BPS_PER_SIDE + V4V_SLIPPAGE_BPS_PER_SIDE),
            "position_fraction": 1.0,
            "leverage": 1.0,
            "pyramiding": False,
            "short_positions_supported": False,
            "no_buy_closes_existing_long": False,
        },
        "economic_acceptance_contract": {
            "minimum_net_profit_factor": 1.5,
            "maximum_drawdown_fraction": V4V_MAXIMUM_DRAWDOWN,
            "minimum_trade_count_is_hard_gate": False,
        },
        "row_level_probability_artifact": {
            "file": "alix-v4v-row-level-probabilities.csv",
            "row_count": len(probability_rows),
            "columns": list(V4V_PROBABILITY_COLUMNS),
            "contains_calibration_rows": False,
            "contains_forward_validation_rows": True,
        },
        "comparison_rows": comparison_rows,
        "trade_rows": trade_rows,
        "feature_importance_rows": feature_rows,
        "experiments": experiments,
        "candidates": _report_candidate_rows(candidates),
        "promotion_status": "BLOCKED_PENDING_INDEPENDENT_OOS_AND_OWNER_REVIEW",
        "safety": {
            "public_api_only": True,
            "credentials_used": False,
            "private_api_calls": 0,
            "real_order_attempts": 0,
            "orders_or_execution_api_used": False,
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "short_positions_supported": False,
            "sequence_model_used": False,
        },
    }


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: tuple[str, ...] | list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fieldnames), extrasaction="raise"
        )
        writer.writeheader()
        writer.writerows(rows)


def _render_step_summary(report: dict[str, Any]) -> str:
    lines = [
        "## AliX V4-V Binary BUY / NO_BUY Evaluation",
        "",
        f"Commit SHA: {report['commit_sha']}",
        "",
        "Target: BUY vs NO_BUY; source SELL and HOLD map to NO_BUY.",
        "Features: Combined Baseline + Trend + Volatility + Volume.",
        "BUY threshold: fixed 0.65; NO_BUY maps to NO_ENTRY and never opens shorts.",
        "Objective: Binary_BUY_Economic_Score on predicted BUY rows only; missed opportunities are not penalized.",
        "",
        "| Arm | Candidates | Trained | Trades | Net PF | Win Rate | Mean Candidate Max DD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in report["arms"]:
        aggregate = arm["aggregate"]
        net_pf = (
            "INF"
            if aggregate.get("net_profit_factor_infinite_no_losses", False)
            else aggregate["net_profit_factor"]
        )
        lines.append(
            f"| {arm['name']} | {aggregate['candidate_count']} "
            f"| {aggregate['trained_candidate_count']} | {aggregate['trade_count']} "
            f"| {net_pf} | {aggregate['win_rate']} "
            f"| {aggregate['mean_candidate_max_drawdown_percent']} |"
        )
    lines.extend(
        [
            "",
            "Binary classification metrics (BUY precision/recall and ROC-AUC) are stored per candidate in the JSON report.",
            "Economic acceptance: net PF > 1.5 and max drawdown < 15%; trade count is diagnostic, not a hard gate.",
            "",
            "Safety: Champion=null, order_permission=false, trading_authorization=NO-GO, private_api_calls=0, real_order_attempts=0.",
            "Spot-only: short positions unsupported. Sequence models: none.",
            "Candidate rows are aggregated descriptively; portfolio overlap is not modeled.",
            "Promotion: BLOCKED_PENDING_INDEPENDENT_OOS_AND_OWNER_REVIEW.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_outputs(
    report: dict[str, Any],
    report_path: Path,
    comparison_path: Path,
    trades_path: Path,
    probabilities_path: Path,
    feature_importance_path: Path,
    probability_rows: list[dict[str, Any]],
) -> None:
    if len(report["row_level_probability_artifact"]["columns"]) != len(
        V4V_PROBABILITY_COLUMNS
    ):
        raise ValueError("v4v_probability_artifact_schema_invalid")
    if len(report["row_level_probability_artifact"]["columns"]) != len(
        set(report["row_level_probability_artifact"]["columns"])
    ):
        raise ValueError("v4v_probability_artifact_columns_duplicate")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        comparison_path,
        report["comparison_rows"],
        (
            "candidate_id",
            "arm",
            "geometry",
            "pair",
            "timeframe",
            "feature_profile",
            "weighting_policy",
            "status",
            "trade_count",
            "win_rate",
            "net_profit_factor",
            "gross_profit_factor",
            "net_pnl_usd",
            "max_drawdown_percent",
            "buy_precision",
            "buy_recall",
            "roc_auc",
            "buy_probability_threshold",
        ),
    )
    _write_csv(
        trades_path,
        report["trade_rows"],
        (
            "candidate_id",
            "arm",
            "pair",
            "timeframe",
            "feature_profile",
            "weighting_policy",
            "trade_index",
            "entry_timestamp",
            "exit_timestamp",
            "entry_reference_price",
            "entry_fill_price",
            "exit_reference_price",
            "exit_fill_price",
            "gross_market_pnl_usd",
            "fees_usd",
            "slippage_cost_usd",
            "net_pnl_usd",
            "exit_reason",
            "direction",
        ),
    )
    _write_csv(probabilities_path, probability_rows, V4V_PROBABILITY_COLUMNS)
    _write_csv(
        feature_importance_path,
        report["feature_importance_rows"],
        (
            "candidate_id",
            "arm",
            "pair",
            "timeframe",
            "feature_profile",
            "weighting_policy",
            "feature",
            "gain",
        ),
    )


def run_v4v(
    input_dir: Path,
    report_path: Path,
    comparison_path: Path,
    trades_path: Path,
    probabilities_path: Path,
    feature_importance_path: Path,
    *,
    days: int,
    commit_sha: str,
) -> dict[str, Any]:
    if days != 730:
        raise ValueError("v4v_dataset_window_must_be_730_days")
    datasets, provenance = _load_diagnostic_datasets(input_dir, days, commit_sha)
    experiments, candidates, probability_rows = run_v4v_experiments(datasets)
    report = _build_report(
        commit_sha=commit_sha,
        provenance=provenance,
        experiments=experiments,
        candidates=candidates,
        probability_rows=probability_rows,
    )
    _write_outputs(
        report,
        report_path,
        comparison_path,
        trades_path,
        probabilities_path,
        feature_importance_path,
        probability_rows,
    )
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as handle:
            handle.write(_render_step_summary(report))
    print(
        json.dumps(
            {
                "commit_sha": commit_sha,
                "experiment_count": report["experiment_count"],
                "candidate_count": report["candidate_count"],
                "trained_candidate_count": report["trained_candidate_count"],
                "probability_row_count": report["row_level_probability_artifact"][
                    "row_count"
                ],
                "promotion_status": report["promotion_status"],
                "Champion": None,
                "order_permission": False,
                "trading_authorization": "NO-GO",
                "private_api_calls": 0,
                "real_order_attempts": 0,
            },
            sort_keys=True,
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-comparison-csv", type=Path, required=True)
    parser.add_argument("--output-trades-csv", type=Path, required=True)
    parser.add_argument("--output-probabilities-csv", type=Path, required=True)
    parser.add_argument("--output-feature-importance-csv", type=Path, required=True)
    parser.add_argument("--days", type=int, choices=(730,), default=730)
    parser.add_argument("--commit-sha", required=True)
    args = parser.parse_args()
    report = run_v4v(
        args.input_dir,
        args.output_report,
        args.output_comparison_csv,
        args.output_trades_csv,
        args.output_probabilities_csv,
        args.output_feature_importance_csv,
        days=args.days,
        commit_sha=args.commit_sha,
    )
    return 0 if report["economic_evaluation_performed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
