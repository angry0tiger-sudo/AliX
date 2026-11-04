"""V4-X causal SMA primary signals and OOF-only LightGBM meta filtering."""

from __future__ import annotations

import importlib
import math
from typing import Any, Final, cast

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from services.ml_core.binary_calibration import select_dynamic_buy_threshold

V4X_SMA_FAST_PERIOD: Final[int] = 20
V4X_SMA_SLOW_PERIOD: Final[int] = 50
V4X_OOF_FOLD_COUNT: Final[int] = 5
V4X_META_CALIBRATION_FRACTION: Final[float] = 0.15
V4X_TARGET_ACCEPT_FRACTION: Final[float] = 0.50
V4X_MINIMUM_FIT_ROWS: Final[int] = 40
V4X_MINIMUM_CLASS_ROWS: Final[int] = 8
V4X_META_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "%-primary_sma_spread_pct",
    "%-primary_sma_fast_slope_5",
    "%-primary_close_vs_sma_slow_pct",
)


def _finite_feature_rows(features: DataFrame) -> np.ndarray:
    numeric = features.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return cast(np.ndarray, np.isfinite(numeric).all(axis=1))


def _validate_sma_periods(fast_period: int, slow_period: int) -> None:
    if (
        isinstance(fast_period, bool)
        or isinstance(slow_period, bool)
        or not isinstance(fast_period, int)
        or not isinstance(slow_period, int)
        or fast_period < 2
        or slow_period <= fast_period
    ):
        raise ValueError("v4x_sma_period_configuration_invalid")


def _validate_sma_feature_inputs(bars: DataFrame, candle_features: DataFrame) -> None:
    if len(bars) != len(candle_features) or bars.empty:
        raise ValueError("v4x_sma_input_alignment_invalid")
    if not candle_features.columns.is_unique:
        raise ValueError("v4x_feature_columns_must_be_unique")
    if any(str(name).startswith("&-") for name in candle_features.columns):
        raise ValueError("v4x_target_column_in_primary_features")
    if "close" not in bars.columns:
        raise ValueError("v4x_sma_close_column_missing")


def _validated_sma_close(bars: DataFrame) -> Series:
    close = pd.to_numeric(bars["close"], errors="coerce")
    if not np.isfinite(close.to_numpy(dtype=float)).all() or bool((close <= 0).any()):
        raise ValueError("v4x_sma_close_values_invalid")
    return close


def build_sma_primary_features(
    bars: DataFrame,
    candle_features: DataFrame,
    *,
    fast_period: int = V4X_SMA_FAST_PERIOD,
    slow_period: int = V4X_SMA_SLOW_PERIOD,
) -> tuple[DataFrame, Series]:
    """Create causal SMA-crossover BUY events and point-in-time meta features."""
    _validate_sma_periods(fast_period, slow_period)
    _validate_sma_feature_inputs(bars, candle_features)
    close = _validated_sma_close(bars)
    fast = close.rolling(fast_period, min_periods=fast_period).mean()
    slow = close.rolling(slow_period, min_periods=slow_period).mean()
    crossover = (fast > slow) & (fast.shift(1) <= slow.shift(1))

    result = candle_features.reset_index(drop=True).copy()
    result["%-primary_sma_spread_pct"] = fast / slow - 1.0
    result["%-primary_sma_fast_slope_5"] = fast.pct_change(periods=5, fill_method=None)
    result["%-primary_close_vs_sma_slow_pct"] = close / slow - 1.0
    finite = _finite_feature_rows(result)
    signals = crossover.fillna(False).to_numpy(dtype=bool) & finite
    return result, Series(signals, name="primary_buy_signal", dtype=bool)


def _validate_oof_window(
    row_count: int,
    start_index: int,
    stop_index: int,
) -> None:
    invalid_type = (
        isinstance(start_index, bool)
        or isinstance(stop_index, bool)
        or not isinstance(start_index, int)
        or not isinstance(stop_index, int)
    )
    invalid_bounds = (
        start_index < 0 or stop_index > row_count or stop_index <= start_index
    )
    if invalid_type or invalid_bounds:
        raise ValueError("v4x_oof_window_invalid")


def _validate_oof_fold_count(
    start_index: int,
    stop_index: int,
    fold_count: int,
) -> None:
    invalid_type = isinstance(fold_count, bool) or not isinstance(fold_count, int)
    invalid_count = fold_count < 2 or stop_index - start_index < fold_count
    if invalid_type or invalid_count:
        raise ValueError("v4x_oof_fold_configuration_invalid")


def _validate_oof_configuration(
    row_count: int,
    start_index: int,
    stop_index: int,
    fold_count: int,
) -> None:
    _validate_oof_window(row_count, start_index, stop_index)
    _validate_oof_fold_count(start_index, stop_index, fold_count)


def _oof_fold_records(
    signal_values: np.ndarray,
    fold_id: int,
    fold_start: int,
    fold_stop: int,
) -> list[dict[str, int | str | bool]]:
    event_indices = np.flatnonzero(signal_values[fold_start:fold_stop]) + fold_start
    return [
        {
            "event_index": int(event_index),
            "fold_id": fold_id,
            "fold_start_index": fold_start,
            "fold_stop_index": fold_stop,
            "primary_signal": "BUY",
            "primary_signal_generation": "CAUSAL_FIXED_SMA_CROSSOVER",
            "primary_oof": True,
            "primary_history_end_index": int(event_index),
        }
        for event_index in event_indices.tolist()
    ]


def _oof_primary_event_frame(
    records: list[dict[str, int | str | bool]],
) -> DataFrame:
    if not records:
        raise ValueError("v4x_oof_primary_signals_empty")
    return (
        DataFrame.from_records(records)
        .sort_values("event_index")
        .reset_index(drop=True)
    )


def _validate_oof_primary_event_frame(result: DataFrame) -> None:
    if result["event_index"].duplicated().any():
        raise ValueError("v4x_oof_primary_events_duplicate")
    if not result["primary_oof"].eq(True).all():
        raise ValueError("v4x_primary_oof_provenance_invalid")
    if (
        result["primary_history_end_index"].to_numpy(dtype=int)
        > result["event_index"].to_numpy(dtype=int)
    ).any():
        raise ValueError("v4x_primary_signal_future_history")


def generate_oof_primary_events(
    primary_buy_signal: Series,
    *,
    start_index: int,
    stop_index: int,
    fold_count: int = V4X_OOF_FOLD_COUNT,
) -> DataFrame:
    """Record fixed-rule BUY signals in disjoint chronological OOF test folds.

    The primary is a fixed, non-fitted SMA rule. Each fold emits signals only
    for its own test interval; inputs are candle features available at that
    event close and earlier. No target or future outcome is consulted.
    """
    row_count = len(primary_buy_signal)
    _validate_oof_configuration(row_count, start_index, stop_index, fold_count)
    boundaries = np.linspace(start_index, stop_index, fold_count + 1, dtype=int)
    signal_values = primary_buy_signal.to_numpy(dtype=bool)
    records = [
        record
        for fold_id in range(fold_count)
        for record in _oof_fold_records(
            signal_values,
            fold_id,
            int(boundaries[fold_id]),
            int(boundaries[fold_id + 1]),
        )
    ]
    result = _oof_primary_event_frame(records)
    _validate_oof_primary_event_frame(result)
    return result


def _validate_primary_events(primary_events: DataFrame) -> None:
    required = {"event_index", "primary_signal", "primary_oof", "fold_id"}
    if not required.issubset(primary_events.columns):
        raise ValueError("v4x_primary_event_schema_invalid")
    if not primary_events["primary_signal"].eq("BUY").all():
        raise ValueError("v4x_meta_training_requires_primary_buy_only")
    if not primary_events["primary_oof"].eq(True).all():
        raise ValueError("v4x_meta_training_requires_oof_primary_signals")
    if primary_events["event_index"].duplicated().any():
        raise ValueError("v4x_primary_events_duplicate")


def _validate_barrier_labels(barrier_labels: DataFrame) -> None:
    required = {
        "event_index",
        "label_end_idx",
        "entry_price",
        "take_profit_price",
        "stop_loss_price",
        "exit_reason",
    }
    if not required.issubset(barrier_labels.columns):
        raise ValueError("v4x_barrier_label_schema_invalid")
    if barrier_labels["event_index"].duplicated().any():
        raise ValueError("v4x_barrier_events_duplicate")


def _validate_cost_configuration(
    fee_bps_per_side: float,
    slippage_bps_per_side: float,
    entry_slippage_bps: float,
) -> tuple[float, float, float]:
    values = (fee_bps_per_side, slippage_bps_per_side, entry_slippage_bps)
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in values):
        raise ValueError("v4x_cost_configuration_invalid")
    return (
        float(fee_bps_per_side) / 10_000.0,
        float(slippage_bps_per_side) / 10_000.0,
        float(entry_slippage_bps) / 10_000.0,
    )


def _meta_exit_reference(
    bars: DataFrame,
    row: dict[str, Any],
    label_end: int,
) -> float:
    reason = str(row["exit_reason"])
    if reason == "TP_HIT":
        exit_reference = float(row["take_profit_price"])
    elif reason in {"SL_HIT", "COLLISION"}:
        exit_reference = float(row["stop_loss_price"])
    elif reason == "HORIZON_EXPIRY":
        exit_reference = float(bars.iloc[label_end]["close"])
    else:
        raise ValueError("v4x_meta_exit_reason_invalid")
    if not math.isfinite(exit_reference) or exit_reference <= 0:
        raise ValueError("v4x_meta_exit_price_invalid")
    return exit_reference


def _costed_meta_label_record(
    bars: DataFrame,
    row: dict[str, Any],
    fee_rate: float,
    slippage_rate: float,
    entry_rate: float,
) -> dict[str, int | float | str] | None:
    raw_label_end = row["label_end_idx"]
    raw_entry_price = row["entry_price"]
    if pd.isna(raw_label_end) or pd.isna(raw_entry_price):
        return None
    event_index = int(row["event_index"])
    label_end = int(raw_label_end)
    if event_index < 0 or label_end <= event_index or label_end >= len(bars):
        raise ValueError("v4x_meta_label_horizon_invalid")
    entry_fill = float(raw_entry_price)
    if not math.isfinite(entry_fill) or entry_fill <= 0 or entry_rate >= 1:
        raise ValueError("v4x_meta_entry_price_invalid")
    entry_reference = entry_fill / (1.0 + entry_rate)
    exit_reference = _meta_exit_reference(bars, row, label_end)
    exit_fill = exit_reference * (1.0 - slippage_rate)
    gross_return = (exit_reference - entry_reference) / entry_reference
    entry_fee = entry_fill * fee_rate / entry_reference
    exit_fee = exit_fill * fee_rate / entry_reference
    entry_slippage = (entry_fill - entry_reference) / entry_reference
    exit_slippage = (exit_reference - exit_fill) / entry_reference
    net_return = gross_return - entry_fee - exit_fee - entry_slippage - exit_slippage
    if not math.isfinite(net_return):
        raise ValueError("v4x_meta_net_return_invalid")
    return {
        "event_index": event_index,
        "fold_id": int(row["fold_id"]),
        "label_end_idx": label_end,
        "meta_label": int(net_return > 0.0),
        "net_return": float(net_return),
        "exit_reason": str(row["exit_reason"]),
        "primary_signal": str(row["primary_signal"]),
        "primary_oof": bool(row["primary_oof"]),
    }


def _costed_meta_label_frame(
    records: list[dict[str, int | float | str]],
) -> DataFrame:
    if not records:
        raise ValueError("v4x_costed_meta_labels_empty")
    result = (
        DataFrame.from_records(records)
        .sort_values("event_index")
        .reset_index(drop=True)
    )
    if result["event_index"].duplicated().any():
        raise ValueError("v4x_costed_meta_labels_duplicate")
    return result


def build_costed_meta_labels(
    bars: DataFrame,
    primary_events: DataFrame,
    barrier_labels: DataFrame,
    *,
    fee_bps_per_side: float,
    slippage_bps_per_side: float,
    entry_slippage_bps: float,
) -> DataFrame:
    """Create binary net-positive labels for primary BUY events only.

    Entry price in the adaptive Triple-Barrier output already includes entry
    slippage. This function reconstructs the reference entry and applies both
    sides' fee and slippage using the same accounting convention as the Spot
    replay. Future prices are used only for this target, never as features.
    """
    _validate_primary_events(primary_events)
    _validate_barrier_labels(barrier_labels)
    fee_rate, slippage_rate, entry_rate = _validate_cost_configuration(
        fee_bps_per_side, slippage_bps_per_side, entry_slippage_bps
    )
    selected = barrier_labels.loc[
        :,
        [
            "event_index",
            "label_end_idx",
            "entry_price",
            "take_profit_price",
            "stop_loss_price",
            "exit_reason",
        ],
    ].copy()
    merged = primary_events.merge(
        selected, on="event_index", how="inner", validate="one_to_one"
    ).sort_values("event_index")
    records = [
        record
        for row in merged.to_dict(orient="records")
        if (
            record := _costed_meta_label_record(
                bars, cast(dict[str, Any], row), fee_rate, slippage_rate, entry_rate
            )
        )
        is not None
    ]
    return _costed_meta_label_frame(records)


def _validate_meta_training_events(
    events: DataFrame,
    outer_train_end_index: int,
    calibration_fraction: float,
) -> float:
    required = {
        "event_index",
        "fold_id",
        "label_end_idx",
        "meta_label",
        "primary_signal",
        "primary_oof",
    }
    if not required.issubset(events.columns):
        raise ValueError("v4x_meta_training_schema_invalid")
    fraction = float(calibration_fraction)
    if not math.isfinite(fraction) or not 0 < fraction < 0.5:
        raise ValueError("v4x_meta_calibration_fraction_invalid")
    if (
        events["event_index"].duplicated().any()
        or not events["event_index"].is_monotonic_increasing
    ):
        raise ValueError("v4x_meta_training_event_order_invalid")
    if not events["primary_signal"].eq("BUY").all():
        raise ValueError("v4x_meta_training_requires_primary_buy_only")
    if not events["primary_oof"].eq(True).all():
        raise ValueError("v4x_meta_training_requires_oof_primary_signals")
    indices = events["event_index"].to_numpy(dtype=int)
    label_ends = events["label_end_idx"].to_numpy(dtype=int)
    if (indices >= outer_train_end_index).any() or (
        label_ends >= outer_train_end_index
    ).any():
        raise ValueError("v4x_meta_training_outer_validation_contamination")
    return fraction


def _build_meta_fit_calibration_frames(
    events: DataFrame,
    calibration_fraction: float,
) -> tuple[DataFrame, DataFrame, int]:
    calibration_count = max(2, int(math.ceil(len(events) * calibration_fraction)))
    if len(events) - calibration_count < 1 or calibration_count >= len(events):
        raise ValueError("v4x_insufficient_meta_calibration_rows")
    calibration = events.iloc[-calibration_count:].copy()
    calibration_start = int(calibration["event_index"].iloc[0])
    fit_candidates = events.iloc[:-calibration_count].copy()
    fit = fit_candidates.loc[fit_candidates["label_end_idx"] < calibration_start].copy()
    return fit, calibration, calibration_start


def _validate_meta_fit_support(
    fit: DataFrame,
    calibration: DataFrame,
    calibration_start: int,
    minimum_fit_rows: int,
    minimum_class_rows: int,
) -> None:
    if len(fit) < minimum_fit_rows or len(calibration) < 2:
        raise ValueError("v4x_insufficient_meta_training_rows")
    counts = fit["meta_label"].value_counts().to_dict()
    if any(int(counts.get(label, 0)) < minimum_class_rows for label in (0, 1)):
        raise ValueError("v4x_insufficient_meta_class_support")
    if int(fit["label_end_idx"].max()) >= calibration_start:
        raise ValueError("v4x_meta_calibration_purge_failed")


def split_meta_fit_calibration(
    labelled_oof_events: DataFrame,
    *,
    outer_train_end_index: int,
    calibration_fraction: float = V4X_META_CALIBRATION_FRACTION,
    minimum_fit_rows: int = V4X_MINIMUM_FIT_ROWS,
    minimum_class_rows: int = V4X_MINIMUM_CLASS_ROWS,
) -> tuple[DataFrame, DataFrame]:
    """Split matured OOF BUY events into purged fit and label-blind calibration."""
    fraction = _validate_meta_training_events(
        labelled_oof_events,
        outer_train_end_index,
        calibration_fraction,
    )
    events = labelled_oof_events.copy().reset_index(drop=True)
    fit, calibration, calibration_start = _build_meta_fit_calibration_frames(
        events, fraction
    )
    _validate_meta_fit_support(
        fit, calibration, calibration_start, minimum_fit_rows, minimum_class_rows
    )
    return fit.reset_index(drop=True), calibration.reset_index(drop=True)


def _validate_meta_matrix(features: DataFrame) -> np.ndarray:
    if features.empty or not features.columns.is_unique:
        raise ValueError("v4x_meta_feature_matrix_invalid")
    values = features.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("v4x_meta_features_non_finite")
    return cast(np.ndarray, values)


def _positive_class_probability(model: Any, features: DataFrame) -> np.ndarray:
    values = _validate_meta_matrix(features)
    classes = np.asarray(getattr(model, "classes_", []))
    if classes.tolist() != [0, 1]:
        raise ValueError("v4x_meta_model_class_order_invalid")
    probabilities = np.asarray(model.predict_proba(values), dtype=float)
    if (
        probabilities.shape != (len(features), 2)
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
    ):
        raise ValueError("v4x_meta_probabilities_invalid")
    return probabilities[:, 1]


def fit_lightgbm_meta_filter(
    fit_features: DataFrame,
    fit_labels: Series,
    calibration_features: DataFrame,
    *,
    target_accept_fraction: float = V4X_TARGET_ACCEPT_FRACTION,
) -> tuple[Any, dict[str, Any]]:
    """Fit LightGBM on earlier OOF BUY rows; calibrate cutoff without labels."""
    if tuple(fit_features.columns) != tuple(calibration_features.columns):
        raise ValueError("v4x_meta_calibration_feature_schema_mismatch")
    if len(fit_features) != len(fit_labels) or len(fit_features) < V4X_MINIMUM_FIT_ROWS:
        raise ValueError("v4x_insufficient_meta_training_rows")
    labels = pd.to_numeric(fit_labels, errors="coerce").to_numpy(dtype=float)
    if (
        not np.isfinite(labels).all()
        or not np.isin(labels, [0.0, 1.0]).all()
        or set(np.unique(labels).tolist()) != {0.0, 1.0}
    ):
        raise ValueError("v4x_meta_training_requires_both_binary_classes")
    counts = {int(label): int(np.sum(labels == label)) for label in (0.0, 1.0)}
    if min(counts.values()) < V4X_MINIMUM_CLASS_ROWS:
        raise ValueError("v4x_insufficient_meta_class_support")
    _validate_meta_matrix(fit_features)
    _validate_meta_matrix(calibration_features)

    lightgbm = importlib.import_module("lightgbm")
    classifier_type = getattr(lightgbm, "LGBMClassifier")
    model = classifier_type(
        objective="binary",
        n_estimators=120,
        learning_rate=0.03,
        num_leaves=7,
        max_depth=4,
        min_child_samples=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )
    model.fit(fit_features.to_numpy(dtype=float), labels.astype(int))
    calibration_probabilities = _positive_class_probability(model, calibration_features)
    threshold = select_dynamic_buy_threshold(
        calibration_probabilities,
        target_buy_fraction=target_accept_fraction,
    )
    diagnostics: dict[str, Any] = {
        "model_family": "LightGBM LGBMClassifier objective=binary",
        "positive_class": 1,
        "negative_class": 0,
        "fit_rows": int(len(fit_features)),
        "fit_class_counts": {"meta_failure": counts[0], "meta_success": counts[1]},
        "calibration_rows": int(len(calibration_features)),
        "calibration_labels_used": False,
        "calibration_scope": "PURGED_OUTER_TRAIN_OOF_BUY_EVENTS_ONLY",
        "target_accept_fraction": float(target_accept_fraction),
        "selected_threshold": float(threshold["threshold"]),
        "threshold_diagnostics": threshold,
        "selected_using_validation_labels": False,
        "selected_using_validation_probabilities": False,
        "sequence_model_used": False,
    }
    return model, diagnostics


def predict_meta_probabilities(model: Any, features: DataFrame) -> np.ndarray:
    """Predict calibrated success scores for primary BUY opportunities only."""
    return _positive_class_probability(model, features)
