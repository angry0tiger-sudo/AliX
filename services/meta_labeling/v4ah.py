"""V4-AH research-only meta-model, regime, calibration and risk primitives.

All estimators are fit on explicitly supplied chronological training slices.
This module has no exchange, account, order or production-risk dependencies.
"""

from __future__ import annotations

import math
from typing import Any, Final, cast

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

V4AH_TOP_K_FEATURES: Final[int] = 30
V4AH_REGIME_COUNT: Final[int] = 3
V4AH_FIT_FRACTION: Final[float] = 0.70
V4AH_TEMPERATURE_FRACTION: Final[float] = 0.15
V4AH_THRESHOLD_FRACTION: Final[float] = 0.15
V4AH_MINIMUM_FIT_ROWS: Final[int] = 40
V4AH_MINIMUM_CLASS_ROWS: Final[int] = 8
V4AH_MINIMUM_CALIBRATION_ROWS: Final[int] = 30
V4AH_MINIMUM_THRESHOLD_ROWS: Final[int] = 20
V4AH_MINIMUM_TRAILING_POLICY_ROWS: Final[int] = 30
V4AH_TEMPERATURE_BOUNDS: Final[tuple[float, float]] = (0.05, 20.0)
V4AH_TRAILING_STEP_OPTIONS: Final[tuple[float, ...]] = (1.0, 1.5, 2.0)
V4AH_FRACTIONAL_KELLY: Final[float] = 0.25
V4AH_MAXIMUM_KELLY_FRACTION: Final[float] = 0.10

REGIME_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "%-atr_proxy_14",
    "%-volatility_realized_12",
    "%-volatility_realized_24",
    "%-trend_adx_14",
    "%-trend_ema_spread_12_26",
)


def _numeric_matrix(frame: DataFrame, error_code: str) -> np.ndarray:
    if frame.empty or not frame.columns.is_unique:
        raise ValueError(error_code)
    values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(error_code)
    return cast(np.ndarray, values)


def _validate_binary_labels(labels: Any, expected_rows: int) -> np.ndarray:
    values = np.asarray(labels, dtype=float)
    if (
        values.ndim != 1
        or len(values) != expected_rows
        or not np.isfinite(values).all()
        or not np.isin(values, (0.0, 1.0)).all()
        or set(np.unique(values).tolist()) != {0.0, 1.0}
    ):
        raise ValueError("v4ah_binary_labels_invalid")
    counts = np.bincount(values.astype(int), minlength=2)
    if int(counts.min()) < V4AH_MINIMUM_CLASS_ROWS:
        raise ValueError("v4ah_binary_class_support_insufficient")
    return values.astype(int)



def _validate_regime_request(
    base_features: DataFrame, fit_end: int, cluster_count: int
) -> None:
    if (
        isinstance(fit_end, bool)
        or not isinstance(fit_end, int)
        or fit_end < 100
        or fit_end > len(base_features)
    ):
        raise ValueError("v4ah_regime_training_window_invalid")
    if (
        isinstance(cluster_count, bool)
        or not isinstance(cluster_count, int)
        or not 2 <= cluster_count <= 6
    ):
        raise ValueError("v4ah_regime_cluster_count_invalid")
    if not base_features.columns.is_unique or not set(REGIME_INPUT_COLUMNS).issubset(
        base_features.columns
    ):
        raise ValueError("v4ah_regime_input_schema_invalid")


def _regime_training_matrix(
    base_features: DataFrame, fit_end: int, cluster_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source = base_features.loc[:, list(REGIME_INPUT_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    matrix = source.to_numpy(dtype=float)
    valid_mask = np.isfinite(matrix).all(axis=1)
    train_mask = valid_mask & (np.arange(len(source)) < fit_end)
    train_values = matrix[train_mask]
    if len(train_values) < max(100, cluster_count * 20):
        raise ValueError("v4ah_regime_training_support_insufficient")
    return matrix, valid_mask, train_mask, train_values


def _fit_regime_cluster_model(
    train_values: np.ndarray, cluster_count: int
) -> tuple[Any, Any]:
    from sklearn.cluster import KMeans  # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_values)
    model = KMeans(
        n_clusters=cluster_count,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=42,
        algorithm="lloyd",
    )
    model.fit(scaled_train)
    return scaler, model


def _predict_regime_assignments(
    matrix: np.ndarray, valid_mask: np.ndarray, scaler: Any, model: Any
) -> tuple[np.ndarray, np.ndarray]:
    states = np.full(len(matrix), -1, dtype=int)
    distances = np.zeros(len(matrix), dtype=float)
    transformed = scaler.transform(matrix[valid_mask])
    states[valid_mask] = model.predict(transformed)
    distances[valid_mask] = np.min(model.transform(transformed), axis=1)
    return states, distances


def _assemble_regime_features(
    base_features: DataFrame,
    states: np.ndarray,
    distances: np.ndarray,
    valid_mask: np.ndarray,
    train_mask: np.ndarray,
    cluster_count: int,
    fit_end: int,
    model: Any,
) -> tuple[DataFrame, dict[str, Any]]:
    result = base_features.reset_index(drop=True).copy()
    for cluster_id in range(cluster_count):
        result[f"%-regime_cluster_{cluster_id}"] = (states == cluster_id).astype(float)
    result["%-regime_center_distance"] = distances
    result["%-regime_state_valid"] = (states >= 0).astype(float)
    if not result.columns.is_unique:
        raise ValueError("v4ah_regime_feature_collision")
    counts = np.bincount(states[train_mask], minlength=cluster_count)
    diagnostics = {
        "algorithm": "KMeans",
        "cluster_count": cluster_count,
        "fit_rows": int(train_mask.sum()),
        "fit_end_index_exclusive": fit_end,
        "fit_scope": "EXPLICIT_PRE_REGISTERED_TRAIN_WARMUP_ONLY",
        "feature_columns": list(REGIME_INPUT_COLUMNS),
        "train_cluster_counts": {
            str(index): int(value) for index, value in enumerate(counts.tolist())
        },
        "cluster_centers_scaled": model.cluster_centers_.round(8).tolist(),
        "hard_regime_filter_enabled": False,
        "invalid_warmup_rows": int((~valid_mask).sum()),
    }
    return result, diagnostics


def build_regime_features(
    base_features: DataFrame,
    *,
    fit_end: int,
    cluster_count: int = V4AH_REGIME_COUNT,
) -> tuple[DataFrame, dict[str, Any], Any, Any]:
    """Fit K-Means only on the explicitly bounded historical warm-up window.

    Cluster labels are model features, not hard entry filters. Rows before
    sufficient causal feature history receive an explicit invalid-state marker.
    """
    _validate_regime_request(base_features, fit_end, cluster_count)
    matrix, valid_mask, train_mask, train_values = _regime_training_matrix(
        base_features, fit_end, cluster_count
    )
    scaler, model = _fit_regime_cluster_model(train_values, cluster_count)
    states, distances = _predict_regime_assignments(
        matrix, valid_mask, scaler, model
    )
    result, diagnostics = _assemble_regime_features(
        base_features,
        states,
        distances,
        valid_mask,
        train_mask,
        cluster_count,
        fit_end,
        model,
    )
    return result, diagnostics, scaler, model


def _validate_split_fractions(fit_fraction: float, temperature_fraction: float) -> None:
    if not 0.5 <= fit_fraction <= 0.85:
        raise ValueError("v4ah_fit_fraction_invalid")
    if not 0.05 <= temperature_fraction <= 0.25:
        raise ValueError("v4ah_temperature_fraction_invalid")
    if fit_fraction + temperature_fraction >= 0.95:
        raise ValueError("v4ah_training_split_fractions_invalid")


def _validated_training_events(
    labelled_events: DataFrame, outer_train_end: int
) -> DataFrame:
    required = {"event_index", "label_end_idx", "meta_label", "net_return"}
    if not required.issubset(labelled_events.columns) or len(labelled_events) < 100:
        raise ValueError("v4ah_three_way_training_schema_or_support_invalid")
    events = labelled_events.sort_values("event_index").reset_index(drop=True).copy()
    if events["event_index"].duplicated().any():
        raise ValueError("v4ah_training_events_duplicate")
    event_indices = pd.to_numeric(events["event_index"], errors="coerce").to_numpy(
        dtype=float
    )
    label_ends = pd.to_numeric(events["label_end_idx"], errors="coerce").to_numpy(
        dtype=float
    )
    valid = (
        np.isfinite(event_indices).all()
        and np.isfinite(label_ends).all()
        and np.equal(event_indices, np.floor(event_indices)).all()
        and np.equal(label_ends, np.floor(label_ends)).all()
        and not np.any(event_indices < 0)
        and not np.any(label_ends <= event_indices)
        and not np.any(event_indices >= outer_train_end)
        and not np.any(label_ends >= outer_train_end)
    )
    if not valid:
        raise ValueError("v4ah_training_label_crosses_outer_train")
    return events


def _purged_training_slices(
    events: DataFrame, fit_fraction: float, temperature_fraction: float
) -> tuple[DataFrame, DataFrame, DataFrame, int, int]:
    fit_boundary = int(math.floor(len(events) * fit_fraction))
    temperature_boundary = int(
        math.floor(len(events) * (fit_fraction + temperature_fraction))
    )
    if fit_boundary <= 0 or temperature_boundary <= fit_boundary:
        raise ValueError("v4ah_training_boundaries_invalid")
    temperature_start = int(events["event_index"].iloc[fit_boundary])
    threshold_start = int(events["event_index"].iloc[temperature_boundary])
    fit_candidates = events.iloc[:fit_boundary]
    temperature_candidates = events.iloc[fit_boundary:temperature_boundary]
    threshold = events.iloc[temperature_boundary:].copy()
    fit = fit_candidates.loc[fit_candidates["label_end_idx"] < temperature_start].copy()
    temperature = temperature_candidates.loc[
        temperature_candidates["label_end_idx"] < threshold_start
    ].copy()
    return fit, temperature, threshold, temperature_start, threshold_start


def _validate_purged_training_slices(
    fit: DataFrame,
    temperature: DataFrame,
    threshold: DataFrame,
    temperature_start: int,
    threshold_start: int,
) -> None:
    if len(fit) < V4AH_MINIMUM_FIT_ROWS:
        raise ValueError("v4ah_fit_rows_insufficient_after_purge")
    if len(temperature) < V4AH_MINIMUM_CALIBRATION_ROWS:
        raise ValueError("v4ah_temperature_rows_insufficient_after_purge")
    if len(threshold) < V4AH_MINIMUM_THRESHOLD_ROWS:
        raise ValueError("v4ah_threshold_rows_insufficient")
    _validate_binary_labels(fit["meta_label"], len(fit))
    _validate_binary_labels(temperature["meta_label"], len(temperature))
    if int(fit["label_end_idx"].max()) >= temperature_start:
        raise ValueError("v4ah_fit_temperature_purge_failed")
    if int(temperature["label_end_idx"].max()) >= threshold_start:
        raise ValueError("v4ah_temperature_threshold_purge_failed")


def _training_split_diagnostics(
    fit: DataFrame,
    temperature: DataFrame,
    threshold: DataFrame,
    outer_train_end: int,
) -> dict[str, Any]:
    return {
        "fit_rows": int(len(fit)),
        "temperature_rows": int(len(temperature)),
        "threshold_rows": int(len(threshold)),
        "fit_event_start": int(fit["event_index"].min()),
        "fit_event_end": int(fit["event_index"].max()),
        "temperature_event_start": int(temperature["event_index"].min()),
        "temperature_event_end": int(temperature["event_index"].max()),
        "threshold_event_start": int(threshold["event_index"].min()),
        "threshold_event_end": int(threshold["event_index"].max()),
        "fit_labels_end_before_temperature": bool(
            int(fit["label_end_idx"].max()) < int(temperature["event_index"].min())
        ),
        "temperature_labels_end_before_threshold": bool(
            int(temperature["label_end_idx"].max())
            < int(threshold["event_index"].min())
        ),
        "all_slices_inside_outer_train": bool(
            int(threshold["label_end_idx"].max()) < outer_train_end
        ),
        "split_policy": "CHRONOLOGICAL_70_15_15_WITH_LABEL_END_PURGE",
    }


def split_fit_temperature_threshold(
    labelled_events: DataFrame,
    *,
    outer_train_end: int,
    fit_fraction: float = V4AH_FIT_FRACTION,
    temperature_fraction: float = V4AH_TEMPERATURE_FRACTION,
) -> tuple[DataFrame, DataFrame, DataFrame, dict[str, Any]]:
    """Chronologically split OOF labels into purged Fit/Temp/Threshold slices."""
    _validate_split_fractions(fit_fraction, temperature_fraction)
    events = _validated_training_events(labelled_events, outer_train_end)
    fit, temperature, threshold, temperature_start, threshold_start = (
        _purged_training_slices(events, fit_fraction, temperature_fraction)
    )
    _validate_purged_training_slices(
        fit, temperature, threshold, temperature_start, threshold_start
    )
    diagnostics = _training_split_diagnostics(
        fit, temperature, threshold, outer_train_end
    )
    return (
        fit.reset_index(drop=True),
        temperature.reset_index(drop=True),
        threshold.reset_index(drop=True),
        diagnostics,
    )


def _new_v4ah_lightgbm_model() -> Any:
    import lightgbm as lgb  # type: ignore[import-not-found]

    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=160,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=-1,
        min_child_samples=10,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        deterministic=True,
        force_col_wise=True,
        n_jobs=1,
        verbosity=-1,
    )


def _tree_shap_matrix(model: Any, fit_features: DataFrame, expected_shape: tuple[int, ...]) -> np.ndarray:
    import shap  # type: ignore[import-not-found]

    explainer = shap.TreeExplainer(model)
    raw_values = explainer.shap_values(fit_features)
    if isinstance(raw_values, list):
        raw_values = raw_values[-1]
    shap_values = np.asarray(raw_values, dtype=float)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]
    if (
        shap_values.ndim != 2
        or shap_values.shape != expected_shape
        or not np.isfinite(shap_values).all()
    ):
        raise ValueError("v4ah_shap_output_shape_invalid")
    return shap_values


def _shap_feature_ranking(
    fit_features: DataFrame, shap_values: np.ndarray, top_k: int
) -> tuple[list[str], list[dict[str, Any]]]:
    importance = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(-importance, kind="stable")
    selected_count = min(top_k, len(fit_features.columns))
    selected_indices = order[:selected_count]
    selected_features = [str(fit_features.columns[index]) for index in selected_indices]
    if not selected_features or len(set(selected_features)) != len(selected_features):
        raise ValueError("v4ah_shap_selected_feature_schema_invalid")
    rankings = [
        {
            "rank": rank,
            "feature": str(fit_features.columns[index]),
            "mean_abs_shap": float(importance[index]),
        }
        for rank, index in enumerate(order.tolist(), start=1)
    ]
    return selected_features, rankings


def fit_shap_selected_lightgbm(
    fit_features: DataFrame,
    fit_labels: Series,
    *,
    top_k: int = V4AH_TOP_K_FEATURES,
) -> tuple[Any, list[str], list[dict[str, Any]], dict[str, Any]]:
    """Rank with TreeSHAP on Fit only, then refit on the top-k columns."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
        raise ValueError("v4ah_shap_top_k_invalid")
    values = _numeric_matrix(fit_features, "v4ah_fit_feature_matrix_invalid")
    labels = _validate_binary_labels(fit_labels, len(values))
    if len(values) < V4AH_MINIMUM_FIT_ROWS:
        raise ValueError("v4ah_fit_rows_insufficient")
    initial = _new_v4ah_lightgbm_model()
    initial.fit(fit_features, labels)
    shap_values = _tree_shap_matrix(initial, fit_features, values.shape)
    selected_features, rankings = _shap_feature_ranking(
        fit_features, shap_values, top_k
    )
    final_model = _new_v4ah_lightgbm_model()
    final_model.fit(fit_features.loc[:, selected_features], labels)
    diagnostics = {
        "method": "TREE_SHAP_MEAN_ABSOLUTE_ATTRIBUTION_ON_FIT_SLICE",
        "top_k_requested": top_k,
        "top_k_selected": len(selected_features),
        "input_feature_count": int(len(fit_features.columns)),
        "selected_features": selected_features,
        "importance_fit_rows": int(len(fit_features)),
        "importance_fit_scope": "PURGED_OOF_FIT_SLICE_ONLY",
        "refit_model": "LightGBM LGBMClassifier objective=binary",
    }
    return final_model, selected_features, rankings, diagnostics

def _positive_probability(model: Any, features: DataFrame) -> np.ndarray:
    values = _numeric_matrix(features, "v4ah_probability_features_invalid")
    classes = np.asarray(getattr(model, "classes_", []))
    if classes.tolist() != [0, 1]:
        raise ValueError("v4ah_model_class_order_invalid")
    probabilities = np.asarray(model.predict_proba(values), dtype=float)
    if (
        probabilities.shape != (len(features), 2)
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
    ):
        raise ValueError("v4ah_raw_probabilities_invalid")
    return probabilities[:, 1]


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    return cast(np.ndarray, np.log(clipped / (1.0 - clipped)))


def apply_temperature_scaling(probabilities: Any, temperature: float) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
        or not math.isfinite(float(temperature))
        or not V4AH_TEMPERATURE_BOUNDS[0] <= temperature <= V4AH_TEMPERATURE_BOUNDS[1]
    ):
        raise ValueError("v4ah_temperature_inputs_invalid")
    logits = _logit(values) / float(temperature)
    result = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    if not np.isfinite(result).all():
        raise ValueError("v4ah_calibrated_probabilities_non_finite")
    return result


def fit_temperature_scaler(
    calibration_probabilities: Any,
    calibration_labels: Any,
) -> tuple[float, dict[str, Any]]:
    """Fit a scalar temperature with L-BFGS-B on the dedicated calibration slice."""
    raw = np.asarray(calibration_probabilities, dtype=float)
    labels = np.asarray(calibration_labels, dtype=float)
    if (
        raw.ndim != 1
        or labels.shape != raw.shape
        or len(raw) < V4AH_MINIMUM_CALIBRATION_ROWS
        or not np.isfinite(raw).all()
        or np.any(raw < 0.0)
        or np.any(raw > 1.0)
    ):
        raise ValueError("v4ah_temperature_calibration_inputs_invalid")
    labels = _validate_binary_labels(labels, len(raw))
    from scipy.optimize import minimize  # type: ignore[import-untyped]
    from scipy.special import expit  # type: ignore[import-untyped]

    logits = _logit(raw)

    def objective(log_temperature: np.ndarray) -> float:
        temperature = float(np.exp(log_temperature[0]))
        calibrated = expit(np.clip(logits / temperature, -40.0, 40.0))
        epsilon = 1e-12
        loss = -np.mean(
            labels * np.log(np.clip(calibrated, epsilon, 1.0 - epsilon))
            + (1 - labels) * np.log(np.clip(1.0 - calibrated, epsilon, 1.0 - epsilon))
        )
        return float(loss)

    initial_loss = objective(np.asarray([0.0]))
    result = minimize(
        objective,
        x0=np.asarray([0.0]),
        method="L-BFGS-B",
        bounds=[
            (
                math.log(V4AH_TEMPERATURE_BOUNDS[0]),
                math.log(V4AH_TEMPERATURE_BOUNDS[1]),
            )
        ],
        options={"maxiter": 200, "ftol": 1e-12},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError("v4ah_temperature_optimizer_failed")
    temperature = float(np.exp(result.x[0]))
    if not V4AH_TEMPERATURE_BOUNDS[0] <= temperature <= V4AH_TEMPERATURE_BOUNDS[1]:
        raise ValueError("v4ah_temperature_out_of_bounds")
    diagnostics = {
        "method": "TEMPERATURE_SCALING_L_BFGS_B",
        "temperature": temperature,
        "optimizer_success": bool(result.success),
        "optimizer_iterations": int(result.nit),
        "calibration_rows": int(len(raw)),
        "calibration_labels_used": True,
        "initial_nll": initial_loss,
        "calibrated_nll": float(result.fun),
        "fit_scope": "DEDICATED_PURGED_TEMPERATURE_SLICE_ONLY",
    }
    return temperature, diagnostics



def _validate_calibration_diagnostics_inputs(
    values: np.ndarray, outcomes: np.ndarray, bin_count: int
) -> None:
    valid = (
        values.ndim == 1
        and outcomes.shape == values.shape
        and len(values) > 0
        and np.isfinite(values).all()
        and np.isfinite(outcomes).all()
        and not np.any(values < 0.0)
        and not np.any(values > 1.0)
        and np.isin(outcomes, (0.0, 1.0)).all()
        and bin_count >= 2
    )
    if not valid:
        raise ValueError("v4ah_calibration_diagnostics_inputs_invalid")


def _equal_frequency_calibration_curve(
    values: np.ndarray, outcomes: np.ndarray, bin_count: int
) -> tuple[list[dict[str, Any]], float]:
    order = np.argsort(values, kind="stable")
    groups = np.array_split(order, min(bin_count, len(values)))
    curve: list[dict[str, Any]] = []
    weighted_error = 0.0
    for group in groups:
        if len(group):
            mean_probability = float(np.mean(values[group]))
            observed_rate = float(np.mean(outcomes[group]))
            weighted_error += len(group) / len(values) * abs(
                mean_probability - observed_rate
            )
            curve.append(
                {
                    "count": int(len(group)),
                    "mean_probability": mean_probability,
                    "observed_positive_rate": observed_rate,
                }
            )
    return curve, float(weighted_error)


def calibration_diagnostics(
    probabilities: Any,
    labels: Any,
    *,
    bin_count: int = 10,
) -> dict[str, Any]:
    values = np.asarray(probabilities, dtype=float)
    outcomes = np.asarray(labels, dtype=float)
    _validate_calibration_diagnostics_inputs(values, outcomes, bin_count)
    curve, weighted_error = _equal_frequency_calibration_curve(
        values, outcomes, bin_count
    )
    return {
        "binning": "EQUAL_FREQUENCY",
        "bin_count": len(curve),
        "curve": curve,
        "brier_score": float(np.mean(np.square(values - outcomes))),
        "expected_calibration_error": weighted_error,
        "sample_count": int(len(values)),
        "scope": "EXPLICIT_TRAINING_DIAGNOSTIC_NOT_FINAL_HOLDOUT",
    }


def _validate_threshold_inputs(
    values: np.ndarray, returns: np.ndarray
) -> None:
    valid = (
        values.ndim == 1
        and returns.shape == values.shape
        and len(values) >= V4AH_MINIMUM_THRESHOLD_ROWS
        and np.isfinite(values).all()
        and np.isfinite(returns).all()
        and not np.any(values < 0.0)
        and not np.any(values > 1.0)
        and not np.any(returns <= -1.0)
    )
    if not valid:
        raise ValueError("v4ah_threshold_inputs_invalid")


def _threshold_candidate_is_better(
    candidate: tuple[float, int, float],
    best: tuple[float, int, float] | None,
) -> bool:
    if best is None or candidate[0] > best[0]:
        return True
    return math.isclose(candidate[0], best[0], rel_tol=0.0, abs_tol=1e-12) and (
        candidate[1] > best[1]
    )


def _best_net_expectancy_threshold(
    values: np.ndarray, returns: np.ndarray, minimum_selected: int
) -> tuple[float, int, float] | None:
    best: tuple[float, int, float] | None = None
    thresholds = sorted(set(float(value) for value in values), reverse=True)
    for threshold in thresholds:
        selected = values >= threshold
        selected_count = int(selected.sum())
        if selected_count >= minimum_selected:
            candidate = (float(np.mean(returns[selected])), selected_count, threshold)
            if _threshold_candidate_is_better(candidate, best):
                best = candidate
    return best


def _no_positive_threshold_diagnostics(
    row_count: int, minimum_selected: int
) -> dict[str, Any]:
    return {
        "method": "MAX_NET_EXPECTANCY_WITH_MINIMUM_SUPPORT",
        "status": "NO_POSITIVE_EXPECTANCY_THRESHOLD",
        "threshold": None,
        "threshold_rows": row_count,
        "minimum_selected_rows": minimum_selected,
        "selected_rows": 0,
        "labels_used": True,
    }


def select_net_expectancy_threshold(
    probabilities: Any,
    net_returns: Any,
) -> tuple[float | None, dict[str, Any]]:
    """Select a positive-expectancy cutoff on an independent labelled slice."""
    values = np.asarray(probabilities, dtype=float)
    returns = np.asarray(net_returns, dtype=float)
    _validate_threshold_inputs(values, returns)
    minimum_selected = max(12, int(math.ceil(len(values) * 0.30)))
    best = _best_net_expectancy_threshold(values, returns, minimum_selected)
    if best is None or best[0] <= 0.0:
        return None, _no_positive_threshold_diagnostics(len(values), minimum_selected)
    expectancy, selected_count, threshold = best
    return threshold, {
        "method": "MAX_NET_EXPECTANCY_WITH_MINIMUM_SUPPORT",
        "status": "SELECTED_POSITIVE_EXPECTANCY_THRESHOLD",
        "threshold": float(threshold),
        "threshold_rows": int(len(values)),
        "minimum_selected_rows": minimum_selected,
        "selected_rows": selected_count,
        "selected_fraction": selected_count / len(values),
        "selected_net_expectancy": expectancy,
        "labels_used": True,
        "selection_scope": "THIRD_PURGED_TRAINING_SLICE_ONLY",
    }

def fit_adaptive_trailing_policy(
    event_features: DataFrame,
    candidate_net_returns: Any,
    candidate_label_end_indices: Any,
    *,
    calibration_start_index: int,
) -> tuple[Any, dict[str, Any]]:
    """Fit a step policy from isolated counterfactuals that mature before calibration.

    Rows are excluded unless every candidate stop has exited before the
    temperature-calibration boundary. This prevents an alternative stop path
    from leaking calibration-period prices into policy training.
    """
    values = _numeric_matrix(event_features, "v4ah_trailing_feature_matrix_invalid")
    returns = np.asarray(candidate_net_returns, dtype=float)
    ends = np.asarray(candidate_label_end_indices, dtype=float)
    expected_shape = (len(values), len(V4AH_TRAILING_STEP_OPTIONS))
    if (
        returns.shape != expected_shape
        or ends.shape != expected_shape
        or not np.isfinite(returns).all()
        or not np.isfinite(ends).all()
        or not np.equal(ends, np.floor(ends)).all()
    ):
        raise ValueError("v4ah_trailing_counterfactual_schema_invalid")
    usable = np.all(ends < calibration_start_index, axis=1)
    if int(usable.sum()) < V4AH_MINIMUM_TRAILING_POLICY_ROWS:
        raise ValueError("v4ah_trailing_policy_training_support_insufficient")

    from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]

    selected_returns = returns[usable]
    targets = np.asarray(V4AH_TRAILING_STEP_OPTIONS, dtype=float)[
        np.argmax(selected_returns, axis=1)
    ]
    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=3,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=1,
    )
    model.fit(values[usable], targets)
    classes, counts = np.unique(targets, return_counts=True)
    max_end = int(np.max(ends[usable]))
    if max_end >= calibration_start_index:
        raise ValueError("v4ah_trailing_policy_calibration_boundary_leakage")
    diagnostics = {
        "model_family": "RandomForestClassifier",
        "step_options_atr": list(V4AH_TRAILING_STEP_OPTIONS),
        "training_rows": int(usable.sum()),
        "excluded_boundary_rows": int((~usable).sum()),
        "maximum_counterfactual_exit_index": max_end,
        "calibration_start_index": calibration_start_index,
        "all_counterfactual_exits_before_calibration": True,
        "fit_scope": "PURGED_META_FIT_ROWS_ONLY",
        "target_step_counts": {
            str(float(label)): int(count)
            for label, count in zip(classes.tolist(), counts.tolist(), strict=True)
        },
    }
    return model, diagnostics


def predict_trailing_steps(model: Any, features: DataFrame) -> np.ndarray:
    values = _numeric_matrix(features, "v4ah_trailing_prediction_features_invalid")
    predictions = np.asarray(model.predict(values), dtype=float)
    if (
        predictions.shape != (len(values),)
        or not np.isfinite(predictions).all()
        or not set(predictions.tolist()).issubset(set(V4AH_TRAILING_STEP_OPTIONS))
    ):
        raise ValueError("v4ah_trailing_predictions_invalid")
    return predictions


def estimate_training_payoff_ratio(net_returns: Any) -> tuple[float, dict[str, Any]]:
    values = np.asarray(net_returns, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or len(values) < 10:
        raise ValueError("v4ah_kelly_training_returns_invalid")
    winners = values[values > 0.0]
    losers = values[values < 0.0]
    if len(winners) < V4AH_MINIMUM_CLASS_ROWS or len(losers) < V4AH_MINIMUM_CLASS_ROWS:
        raise ValueError("v4ah_kelly_payoff_support_insufficient")
    average_win = float(np.mean(winners))
    average_loss = abs(float(np.mean(losers)))
    if average_win <= 0.0 or average_loss <= 0.0:
        raise ValueError("v4ah_kelly_payoff_invalid")
    ratio = average_win / average_loss
    return ratio, {
        "average_net_win_return": average_win,
        "average_net_loss_return": average_loss,
        "payoff_ratio": ratio,
        "training_rows": int(len(values)),
        "fit_scope": "PURGED_META_FIT_ROWS_ONLY",
    }


def fractional_kelly_fraction(
    probability: float,
    payoff_ratio: float,
    *,
    fractional_multiplier: float = V4AH_FRACTIONAL_KELLY,
    maximum_fraction: float = V4AH_MAXIMUM_KELLY_FRACTION,
) -> float:
    """Return capped fractional-Kelly sizing for offline comparison only."""
    values = (probability, payoff_ratio, fractional_multiplier, maximum_fraction)
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("v4ah_kelly_inputs_non_finite")
    if not 0.0 <= probability <= 1.0 or payoff_ratio <= 0.0:
        raise ValueError("v4ah_kelly_inputs_out_of_range")
    if not 0.0 < fractional_multiplier <= 1.0:
        raise ValueError("v4ah_kelly_fractional_multiplier_invalid")
    if not 0.0 < maximum_fraction <= 0.10:
        raise ValueError("v4ah_kelly_maximum_fraction_invalid")
    full_kelly = probability - (1.0 - probability) / payoff_ratio
    return min(maximum_fraction, max(0.0, fractional_multiplier * full_kelly))



def _validate_sized_trade_inputs(
    trades: list[dict[str, Any]], stake_fractions: list[float], initial_equity: float
) -> None:
    if len(trades) != len(stake_fractions) or not math.isfinite(initial_equity):
        raise ValueError("v4ah_sized_trade_alignment_invalid")
    if initial_equity <= 0.0:
        raise ValueError("v4ah_initial_equity_invalid")


def _apply_sized_trade(
    trade: dict[str, Any],
    raw_fraction: float,
    equity: float,
    peak: float,
) -> tuple[dict[str, Any], float, float, float, float, int]:
    fraction = float(raw_fraction)
    raw_return = float(trade["net_return_fraction"])
    valid = (
        math.isfinite(fraction)
        and 0.0 <= fraction <= 1.0
        and math.isfinite(raw_return)
        and raw_return > -1.0
    )
    if not valid:
        raise ValueError("v4ah_sized_trade_values_invalid")
    pnl = equity * fraction * raw_return
    next_equity = equity + pnl
    next_peak = max(peak, next_equity)
    drawdown = (next_peak - next_equity) / next_peak if next_peak > 0.0 else 1.0
    enriched = {
        **trade,
        "position_fraction_of_equity": fraction,
        "net_pnl_usd": pnl,
        "sized_net_return_fraction": fraction * raw_return,
        "equity_after_trade_usd": next_equity,
    }
    return enriched, next_equity, next_peak, drawdown, pnl, int(raw_return > 0.0)


def _sized_trade_metrics(
    pnl_values: list[float],
    unscaled_returns: list[float],
    stake_fractions: list[float],
    wins: int,
    maximum_drawdown: float,
    equity: float,
    initial_equity: float,
) -> dict[str, Any]:
    profit = sum(value for value in pnl_values if value > 0.0)
    loss = abs(sum(value for value in pnl_values if value < 0.0))
    count = len(pnl_values)
    return {
        "trade_count": count,
        "net_profit_factor": round(profit / loss, 8) if loss else None,
        "net_profit_factor_infinite_no_losses": bool(profit > 0.0 and loss == 0.0),
        "win_rate": round(wins / count, 8) if count else 0.0,
        "max_drawdown_percent": round(maximum_drawdown * 100.0, 8),
        "net_pnl_total_usd": round(sum(pnl_values), 8),
        "ending_equity_usd": round(equity, 8),
        "mean_position_fraction": (
            round(float(np.mean(stake_fractions)), 8) if stake_fractions else 0.0
        ),
        "max_position_fraction": max(stake_fractions, default=0.0),
        "mean_unscaled_net_return_fraction": (
            round(float(np.mean(unscaled_returns)), 8) if unscaled_returns else 0.0
        ),
        "mean_unscaled_net_expectancy_usd_at_full_equity": (
            round(float(np.mean(unscaled_returns)) * initial_equity, 8)
            if unscaled_returns
            else 0.0
        ),
    }


def summarize_sized_trades(
    trades: list[dict[str, Any]],
    stake_fractions: list[float],
    *,
    initial_equity: float = 10_000.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Rebuild an equity curve for non-overlapping variable-sized trade returns."""
    _validate_sized_trade_inputs(trades, stake_fractions, initial_equity)
    equity = float(initial_equity)
    peak = equity
    maximum_drawdown = 0.0
    enriched: list[dict[str, Any]] = []
    pnl_values: list[float] = []
    unscaled_returns: list[float] = []
    wins = 0
    for trade, raw_fraction in zip(trades, stake_fractions, strict=True):
        row, equity, peak, drawdown, pnl, won = _apply_sized_trade(
            trade, raw_fraction, equity, peak
        )
        maximum_drawdown = max(maximum_drawdown, drawdown)
        pnl_values.append(pnl)
        unscaled_returns.append(float(trade["net_return_fraction"]))
        wins += won
        enriched.append(row)
    metrics = _sized_trade_metrics(
        pnl_values,
        unscaled_returns,
        stake_fractions,
        wins,
        maximum_drawdown,
        equity,
        initial_equity,
    )
    return metrics, enriched

