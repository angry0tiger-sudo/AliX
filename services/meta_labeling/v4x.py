"""V4-X causal SMA primary signals and OOF-only LightGBM meta filtering."""

from __future__ import annotations

import importlib
import math
from typing import Any, Final

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
    return np.isfinite(numeric).all(axis=1)


def build_sma_primary_features(
    bars: DataFrame,
    candle_features: DataFrame,
    *,
    fast_period: int = V4X_SMA_FAST_PERIOD,
    slow_period: int = V4X_SMA_SLOW_PERIOD,
) -> tuple[DataFrame, Series]:
    """Create causal SMA-crossover BUY events and point-in-time meta features."""
    if (
        isinstance(fast_period, bool)
        or isinstance(slow_period, bool)
        or not isinstance(fast_period, int)
        or not isinstance(slow_period, int)
        or fast_period < 2
        or slow_period <= fast_period
    ):
        raise ValueError("v4x_sma_period_configuration_invalid")
    if len(bars) != len(candle_features) or bars.empty:
        raise ValueError("v4x_sma_input_alignment_invalid")
    if not candle_features.columns.is_unique:
        raise ValueError("v4x_feature_columns_must_be_unique")
    if any(str(name).startswith("&-") for name in candle_features.columns):
        raise ValueError("v4x_target_column_in_primary_features")
    if not {"close"}.issubset(bars.columns):
        raise ValueError("v4x_sma_close_column_missing")

    close = pd.to_numeric(bars["close"], errors="coerce")
    if not np.isfinite(close.to_numpy(dtype=float)).all() or bool((close <= 0).any()):
        raise ValueError("v4x_sma_close_values_invalid")
    fast = close.rolling(fast_period, min_periods=fast_period).mean()
    slow = close.rolling(slow_period, min_periods=slow_period).mean()
    previous_fast = fast.shift(1)
    previous_slow = slow.shift(1)
    crossover = (fast > slow) & (previous_fast <= previous_slow)

    result = candle_features.reset_index(drop=True).copy()
    result["%-primary_sma_spread_pct"] = fast / slow - 1.0
    result["%-primary_sma_fast_slope_5"] = fast.pct_change(
        periods=5, fill_method=None
    )
    result["%-primary_close_vs_sma_slow_pct"] = close / slow - 1.0
    finite = _finite_feature_rows(result)
    signals = (crossover.fillna(False).to_numpy(dtype=bool) & finite)
    return result, Series(signals, name="primary_buy_signal", dtype=bool)


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
    if (
        isinstance(start_index, bool)
        or isinstance(stop_index, bool)
        or not isinstance(start_index, int)
        or not isinstance(stop_index, int)
        or start_index < 0
        or stop_index > row_count
        or stop_index <= start_index
    ):
        raise ValueError("v4x_oof_window_invalid")
    if (
        isinstance(fold_count, bool)
        or not isinstance(fold_count, int)
        or fold_count < 2
        or stop_index - start_index < fold_count
    ):
        raise ValueError("v4x_oof_fold_configuration_invalid")

    boundaries = np.linspace(start_index, stop_index, fold_count + 1, dtype=int)
    signal_values = primary_buy_signal.to_numpy(dtype=bool)
    records: list[dict[str, int | str | bool]] = []
    for fold_id in range(fold_count):
        fold_start = int(boundaries[fold_id])
        fold_stop = int(boundaries[fold_id + 1])
        indices = np.flatnonzero(signal_values[fold_start:fold_stop]) + fold_start
        for event_index in indices.tolist():
            records.append(
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
            )
    if not records:
        raise ValueError("v4x_oof_primary_signals_empty")
    result = DataFrame.from_records(records).sort_values("event_index").reset_index(
        drop=True
    )
    if result["event_index"].duplicated().any():
        raise ValueError("v4x_oof_primary_events_duplicate")
    if not result["primary_oof"].eq(True).all():
        raise ValueError("v4x_primary_oof_provenance_invalid")
    if (
        result["primary_history_end_index"].to_numpy(dtype=int)
        > result["event_index"].to_numpy(dtype=int)
    ).any():
        raise ValueError("v4x_primary_signal_future_history")
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
    required_events = {"event_index", "primary_signal", "primary_oof", "fold_id"}
    required_labels = {
        "event_index",
        "label_end_idx",
        "entry_price",
        "take_profit_price",
        "stop_loss_price",
        "exit_reason",
    }
    if not required_events.issubset(primary_events.columns):
        raise ValueError("v4x_primary_event_schema_invalid")
    if not required_labels.issubset(barrier_labels.columns):
        raise ValueError("v4x_barrier_label_schema_invalid")
    cost_values = (fee_bps_per_side, slippage_bps_per_side, entry_slippage_bps)
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in cost_values):
        raise ValueError("v4x_cost_configuration_invalid")
    if not primary_events["primary_signal"].eq("BUY").all():
        raise ValueError("v4x_meta_training_requires_primary_buy_only")
    if not primary_events["primary_oof"].eq(True).all():
        raise ValueError("v4x_meta_training_requires_oof_primary_signals")
    if primary_events["event_index"].duplicated().any():
        raise ValueError("v4x_primary_events_duplicate")
    if barrier_labels["event_index"].duplicated().any():
        raise ValueError("v4x_barrier_events_duplicate")

    selected = barrier_labels.loc[:, list(required_labels)].copy()
    merged = primary_events.merge(
        selected, on="event_index", how="inner", validate="one_to_one"
    ).sort_values("event_index")
    rows: list[dict[str, int | float | str]] = []
    fee_rate = float(fee_bps_per_side) / 10_000.0
    slippage_rate = float(slippage_bps_per_side) / 10_000.0
    entry_rate = float(entry_slippage_bps) / 10_000.0
    for row in merged.itertuples(index=False):
        if pd.isna(row.label_end_idx) or pd.isna(row.entry_price):
            continue
        event_index = int(row.event_index)
        label_end = int(row.label_end_idx)
        if event_index < 0 or label_end <= event_index or label_end >= len(bars):
            raise ValueError("v4x_meta_label_horizon_invalid")
        entry_fill = float(row.entry_price)
        if not math.isfinite(entry_fill) or entry_fill <= 0 or entry_rate >= 1:
            raise ValueError("v4x_meta_entry_price_invalid")
        entry_reference = entry_fill / (1.0 + entry_rate)
        reason = str(row.exit_reason)
        if reason == "TP_HIT":
            exit_reference = float(row.take_profit_price)
        elif reason in {"SL_HIT", "COLLISION"}:
            exit_reference = float(row.stop_loss_price)
        elif reason == "HORIZON_EXPIRY":
            exit_reference = float(bars.iloc[label_end]["close"])
        else:
            raise ValueError("v4x_meta_exit_reason_invalid")
        if not math.isfinite(exit_reference) or exit_reference <= 0:
            raise ValueError("v4x_meta_exit_price_invalid")
        exit_fill = exit_reference * (1.0 - slippage_rate)
        gross_return = (exit_reference - entry_reference) / entry_reference
        entry_fee = entry_fill * fee_rate / entry_reference
        exit_fee = exit_fill * fee_rate / entry_reference
        entry_slippage = (entry_fill - entry_reference) / entry_reference
        exit_slippage = (exit_reference - exit_fill) / entry_reference
        net_return = (
            gross_return
            - entry_fee
            - exit_fee
            - entry_slippage
            - exit_slippage
        )
        if not math.isfinite(net_return):
            raise ValueError("v4x_meta_net_return_invalid")
        rows.append(
            {
                "event_index": event_index,
                "fold_id": int(row.fold_id),
                "label_end_idx": label_end,
                "meta_label": int(net_return > 0.0),
                "net_return": float(net_return),
                "exit_reason": reason,
                "primary_signal": str(row.primary_signal),
                "primary_oof": bool(row.primary_oof),
            }
        )
    if not rows:
        raise ValueError("v4x_costed_meta_labels_empty")
    result = DataFrame.from_records(rows).sort_values("event_index").reset_index(
        drop=True
    )
    if result["event_index"].duplicated().any():
        raise ValueError("v4x_costed_meta_labels_duplicate")
    return result


def split_meta_fit_calibration(
    labelled_oof_events: DataFrame,
    *,
    outer_train_end_index: int,
    calibration_fraction: float = V4X_META_CALIBRATION_FRACTION,
    minimum_fit_rows: int = V4X_MINIMUM_FIT_ROWS,
    minimum_class_rows: int = V4X_MINIMUM_CLASS_ROWS,
) -> tuple[DataFrame, DataFrame]:
    """Split matured OOF BUY events into purged fit and label-blind calibration."""
    required = {
        "event_index",
        "fold_id",
        "label_end_idx",
        "meta_label",
        "primary_signal",
        "primary_oof",
    }
    if not required.issubset(labelled_oof_events.columns):
        raise ValueError("v4x_meta_training_schema_invalid")
    fraction = float(calibration_fraction)
    if not math.isfinite(fraction) or not 0 < fraction < 0.5:
        raise ValueError("v4x_meta_calibration_fraction_invalid")
    if (
        labelled_oof_events["event_index"].duplicated().any()
        or not labelled_oof_events["event_index"].is_monotonic_increasing
    ):
        raise ValueError("v4x_meta_training_event_order_invalid")
    events = labelled_oof_events.copy().reset_index(drop=True)
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
    calibration_count = max(2, int(math.ceil(len(events) * fraction)))
    if len(events) - calibration_count < 1 or calibration_count >= len(events):
        raise ValueError("v4x_insufficient_meta_calibration_rows")
    calibration = events.iloc[-calibration_count:].copy()
    calibration_start = int(calibration["event_index"].iloc[0])
    fit_candidates = events.iloc[:-calibration_count].copy()
    fit = fit_candidates.loc[fit_candidates["label_end_idx"] < calibration_start].copy()
    if len(fit) < minimum_fit_rows or len(calibration) < 2:
        raise ValueError("v4x_insufficient_meta_training_rows")
    counts = fit["meta_label"].value_counts().to_dict()
    if any(int(counts.get(label, 0)) < minimum_class_rows for label in (0, 1)):
        raise ValueError("v4x_insufficient_meta_class_support")
    if int(fit["label_end_idx"].max()) >= calibration_start:
        raise ValueError("v4x_meta_calibration_purge_failed")
    return fit.reset_index(drop=True), calibration.reset_index(drop=True)


def _validate_meta_matrix(features: DataFrame) -> np.ndarray:
    if features.empty or not features.columns.is_unique:
        raise ValueError("v4x_meta_feature_matrix_invalid")
    values = features.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("v4x_meta_features_non_finite")
    return values


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
    calibration_probabilities = _positive_class_probability(
        model, calibration_features
    )
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
