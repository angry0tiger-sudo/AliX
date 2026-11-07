"""Causal CUSUM and Gaussian-HMM regime features for V4-AM research."""

from __future__ import annotations

import math
from typing import Any, Final, cast

import numpy as np
import pandas as pd
from pandas import DataFrame

CUSUM_WINDOW: Final[int] = 24
CUSUM_THRESHOLD_SIGMA: Final[float] = 2.0
CUSUM_DRIFT_ALLOWANCE_SIGMA: Final[float] = 0.5
HMM_STATE_COUNT: Final[int] = 3
HMM_MINIMUM_TRAIN_ROWS: Final[int] = 100
HMM_MAXIMUM_ITERATIONS: Final[int] = 50
HMM_CONVERGENCE_TOLERANCE: Final[float] = 1e-5
HMM_VARIANCE_FLOOR: Final[float] = 1e-3

LOG_RETURN_FEATURE: Final[str] = "%-v4am_log_return"
REALIZED_VOLATILITY_FEATURE: Final[str] = "%-v4am_realized_volatility_24"
CUSUM_DRIFT_FEATURE: Final[str] = "%-cusum_drift_magnitude"
CUSUM_STANDARDIZED_RETURN_FEATURE: Final[str] = "%-cusum_standardized_return"
CUSUM_POSITIVE_EVENT_FEATURE: Final[str] = "%-cusum_positive_event"
CUSUM_NEGATIVE_EVENT_FEATURE: Final[str] = "%-cusum_negative_event"
HMM_TRENDING_UP_FEATURE: Final[str] = "%-hmm_trending_up_probability"
HMM_TRENDING_DOWN_FEATURE: Final[str] = "%-hmm_trending_down_probability"
HMM_RANGING_FEATURE: Final[str] = "%-hmm_ranging_probability"
HMM_STATE_VALID_FEATURE: Final[str] = "%-hmm_state_valid"


def _validate_window_end(apply_end: int, row_count: int) -> None:
    if (
        isinstance(apply_end, bool)
        or not isinstance(apply_end, int)
        or not 2 <= apply_end <= row_count
    ):
        raise ValueError("v4am_apply_window_invalid")


def _validate_cusum_inputs(
    bars: DataFrame, base_features: DataFrame, apply_end: int
) -> None:
    _validate_window_end(apply_end, len(bars))
    if len(bars) != len(base_features) or bars.empty:
        raise ValueError("v4am_feature_alignment_invalid")
    if not bars.columns.is_unique or not base_features.columns.is_unique:
        raise ValueError("v4am_feature_schema_invalid")
    if "close" not in bars.columns:
        raise ValueError("v4am_close_column_missing")
    generated = {
        LOG_RETURN_FEATURE,
        REALIZED_VOLATILITY_FEATURE,
        CUSUM_DRIFT_FEATURE,
        CUSUM_STANDARDIZED_RETURN_FEATURE,
        CUSUM_POSITIVE_EVENT_FEATURE,
        CUSUM_NEGATIVE_EVENT_FEATURE,
    }
    if generated.intersection(base_features.columns):
        raise ValueError("v4am_cusum_feature_collision")
    close = pd.to_numeric(bars["close"].iloc[:apply_end], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(close).all() or np.any(close <= 0.0):
        raise ValueError("v4am_close_values_invalid")


def _causal_cusum_events(
    standardized_returns: np.ndarray,
    *,
    threshold_sigma: float,
    drift_allowance_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive_events = np.zeros(len(standardized_returns), dtype=float)
    negative_events = np.zeros(len(standardized_returns), dtype=float)
    drift_magnitude = np.full(len(standardized_returns), np.nan, dtype=float)
    positive_sum = 0.0
    negative_sum = 0.0
    for index, standardized_return in enumerate(standardized_returns):
        if not np.isfinite(standardized_return):
            positive_sum = 0.0
            negative_sum = 0.0
            continue
        positive_sum = max(
            0.0, positive_sum + float(standardized_return) - drift_allowance_sigma
        )
        negative_sum = min(
            0.0, negative_sum + float(standardized_return) + drift_allowance_sigma
        )
        magnitude = max(positive_sum, abs(negative_sum))
        if positive_sum >= threshold_sigma:
            positive_events[index] = 1.0
            magnitude = positive_sum
            positive_sum = 0.0
            negative_sum = 0.0
        elif negative_sum <= -threshold_sigma:
            negative_events[index] = 1.0
            magnitude = abs(negative_sum)
            positive_sum = 0.0
            negative_sum = 0.0
        drift_magnitude[index] = magnitude
    return positive_events, negative_events, drift_magnitude


def build_cusum_features(
    bars: DataFrame,
    base_features: DataFrame,
    *,
    apply_end: int,
) -> tuple[DataFrame, dict[str, Any]]:
    """Build two-sided CUSUM events from closed-bar log returns only."""
    _validate_cusum_inputs(bars, base_features, apply_end)
    close = pd.to_numeric(bars["close"], errors="coerce").reset_index(drop=True)
    log_returns = np.log(close.iloc[:apply_end]).diff()
    realized_volatility = log_returns.rolling(
        CUSUM_WINDOW, min_periods=CUSUM_WINDOW
    ).std(ddof=0)
    standardized_returns = (
        log_returns / realized_volatility.where(realized_volatility > 0.0)
    ).to_numpy(dtype=float)
    positive, negative, drift = _causal_cusum_events(
        standardized_returns,
        threshold_sigma=CUSUM_THRESHOLD_SIGMA,
        drift_allowance_sigma=CUSUM_DRIFT_ALLOWANCE_SIGMA,
    )
    row_count = len(bars)
    arrays = {
        LOG_RETURN_FEATURE: np.full(row_count, np.nan, dtype=float),
        REALIZED_VOLATILITY_FEATURE: np.full(row_count, np.nan, dtype=float),
        CUSUM_STANDARDIZED_RETURN_FEATURE: np.full(row_count, np.nan, dtype=float),
        CUSUM_DRIFT_FEATURE: np.full(row_count, np.nan, dtype=float),
        CUSUM_POSITIVE_EVENT_FEATURE: np.full(row_count, np.nan, dtype=float),
        CUSUM_NEGATIVE_EVENT_FEATURE: np.full(row_count, np.nan, dtype=float),
    }
    arrays[LOG_RETURN_FEATURE][:apply_end] = log_returns.to_numpy(dtype=float)
    arrays[REALIZED_VOLATILITY_FEATURE][:apply_end] = realized_volatility.to_numpy(
        dtype=float
    )
    arrays[CUSUM_STANDARDIZED_RETURN_FEATURE][:apply_end] = standardized_returns
    arrays[CUSUM_DRIFT_FEATURE][:apply_end] = drift
    arrays[CUSUM_POSITIVE_EVENT_FEATURE][:apply_end] = positive
    arrays[CUSUM_NEGATIVE_EVENT_FEATURE][:apply_end] = negative
    transformed = base_features.reset_index(drop=True).copy()
    for name, values in arrays.items():
        transformed[name] = values
    diagnostics = {
        "algorithm": "TWO_SIDED_CUSUM_STANDARDIZED_LOG_RETURN",
        "rolling_window_bars": CUSUM_WINDOW,
        "threshold_sigma": CUSUM_THRESHOLD_SIGMA,
        "drift_allowance_sigma": CUSUM_DRIFT_ALLOWANCE_SIGMA,
        "realized_volatility_ddof": 0,
        "standardization": "CURRENT_CLOSED_BAR_LOG_RETURN_OVER_TRAILING_24_BAR_STD",
        "positive_event_count": int(positive.sum()),
        "negative_event_count": int(negative.sum()),
        "total_event_count": int(positive.sum() + negative.sum()),
        "valid_standardized_return_rows": int(np.isfinite(standardized_returns).sum()),
        "apply_end_index_exclusive": apply_end,
        "final_holdout_features_unavailable": bool(
            all(np.isnan(values[apply_end:]).all() for values in arrays.values())
        ),
        "lookahead_used": False,
    }
    return transformed, diagnostics


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    if not np.isfinite(maximum).all():
        raise ValueError("v4am_hmm_log_probability_invalid")
    result = maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True))
    return cast(np.ndarray, np.squeeze(result, axis=axis))


def _emission_log_probabilities(
    observations: np.ndarray, means: np.ndarray, variances: np.ndarray
) -> np.ndarray:
    if (
        observations.ndim != 2
        or means.ndim != 2
        or variances.shape != means.shape
        or observations.shape[1] != means.shape[1]
    ):
        raise ValueError("v4am_hmm_emission_shape_invalid")
    squared = (observations[:, np.newaxis, :] - means[np.newaxis, :, :]) ** 2
    log_density = -0.5 * (
        np.log(2.0 * math.pi * variances)[np.newaxis, :, :]
        + squared / variances[np.newaxis, :, :]
    )
    result = log_density.sum(axis=2)
    if not np.isfinite(result).all():
        raise ValueError("v4am_hmm_emission_non_finite")
    return cast(np.ndarray, result)


def _forward_backward(
    emission_log_probabilities: np.ndarray,
    start_probabilities: np.ndarray,
    transition_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    row_count, state_count = emission_log_probabilities.shape
    if (
        row_count < 2
        or start_probabilities.shape != (state_count,)
        or transition_matrix.shape != (state_count, state_count)
    ):
        raise ValueError("v4am_hmm_forward_backward_shape_invalid")
    log_start = np.log(np.clip(start_probabilities, 1e-300, 1.0))
    log_transition = np.log(np.clip(transition_matrix, 1e-300, 1.0))
    forward = np.empty((row_count, state_count), dtype=float)
    backward = np.zeros((row_count, state_count), dtype=float)
    forward[0] = log_start + emission_log_probabilities[0]
    for index in range(1, row_count):
        forward[index] = (
            _logsumexp(forward[index - 1][:, np.newaxis] + log_transition, axis=0)
            + emission_log_probabilities[index]
        )
    log_likelihood = float(_logsumexp(forward[-1], axis=0))
    for index in range(row_count - 2, -1, -1):
        backward[index] = _logsumexp(
            log_transition
            + emission_log_probabilities[index + 1][np.newaxis, :]
            + backward[index + 1][np.newaxis, :],
            axis=1,
        )
    gamma = np.exp(forward + backward - log_likelihood)
    gamma /= gamma.sum(axis=1, keepdims=True)
    log_xi = (
        forward[:-1, :, np.newaxis]
        + log_transition[np.newaxis, :, :]
        + emission_log_probabilities[1:, np.newaxis, :]
        + backward[1:, np.newaxis, :]
        - log_likelihood
    )
    xi = np.exp(log_xi).sum(axis=0)
    if not (
        np.isfinite(gamma).all()
        and np.isfinite(xi).all()
        and math.isfinite(log_likelihood)
    ):
        raise ValueError("v4am_hmm_estimation_non_finite")
    return gamma, xi, log_likelihood


def _validate_hmm_training_values(train_values: np.ndarray) -> None:
    valid = (
        train_values.ndim == 2
        and train_values.shape[1] == 2
        and len(train_values) >= HMM_MINIMUM_TRAIN_ROWS
        and np.isfinite(train_values).all()
    )
    if not valid:
        raise ValueError("v4am_hmm_training_support_insufficient")


def _initialize_gaussian_hmm(
    train_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(train_values[:, 0], kind="stable")
    chunks = np.array_split(order, HMM_STATE_COUNT)
    means = np.vstack([train_values[chunk].mean(axis=0) for chunk in chunks])
    global_variance = np.maximum(train_values.var(axis=0), HMM_VARIANCE_FLOOR)
    variances = np.tile(global_variance, (HMM_STATE_COUNT, 1))
    start_probabilities = np.full(
        HMM_STATE_COUNT, 1.0 / HMM_STATE_COUNT, dtype=float
    )
    transition_matrix = np.full(
        (HMM_STATE_COUNT, HMM_STATE_COUNT),
        0.10 / (HMM_STATE_COUNT - 1),
        dtype=float,
    )
    np.fill_diagonal(transition_matrix, 0.90)
    return start_probabilities, transition_matrix, means, variances


def _update_hmm_emission_parameters(
    train_values: np.ndarray, gamma: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    state_weights = gamma.sum(axis=0)
    if np.any(state_weights <= 1e-8):
        raise ValueError("v4am_hmm_state_support_collapsed")
    means = (gamma.T @ train_values) / state_weights[:, np.newaxis]
    variances = np.empty_like(means)
    for state in range(HMM_STATE_COUNT):
        deviations = train_values - means[state]
        variances[state] = (
            gamma[:, state, np.newaxis] * deviations**2
        ).sum(axis=0) / state_weights[state]
    return means, np.maximum(variances, HMM_VARIANCE_FLOOR)


def _gaussian_hmm_em_step(
    train_values: np.ndarray,
    start_probabilities: np.ndarray,
    transition_matrix: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    emission = _emission_log_probabilities(train_values, means, variances)
    gamma, xi, likelihood = _forward_backward(
        emission, start_probabilities, transition_matrix
    )
    next_start = gamma[0] + 1e-3
    next_start /= next_start.sum()
    next_transition = xi + 1e-3
    next_transition /= next_transition.sum(axis=1, keepdims=True)
    next_means, next_variances = _update_hmm_emission_parameters(
        train_values, gamma
    )
    return next_start, next_transition, next_means, next_variances, likelihood


def _hmm_likelihood_converged(previous: float, current: float) -> bool:
    return math.isfinite(previous) and abs(current - previous) <= HMM_CONVERGENCE_TOLERANCE


def _fit_gaussian_hmm(
    train_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    _validate_hmm_training_values(train_values)
    start_probabilities, transition_matrix, means, variances = (
        _initialize_gaussian_hmm(train_values)
    )
    previous_likelihood = -math.inf
    likelihood = -math.inf
    converged = False
    iteration_count = 0
    for iteration in range(1, HMM_MAXIMUM_ITERATIONS + 1):
        (
            start_probabilities,
            transition_matrix,
            means,
            variances,
            likelihood,
        ) = _gaussian_hmm_em_step(
            train_values,
            start_probabilities,
            transition_matrix,
            means,
            variances,
        )
        iteration_count = iteration
        if _hmm_likelihood_converged(previous_likelihood, likelihood):
            converged = True
            break
        previous_likelihood = likelihood
    diagnostics = {
        "iterations": iteration_count,
        "converged": converged,
        "log_likelihood": float(likelihood),
        "state_count": HMM_STATE_COUNT,
    }
    return start_probabilities, transition_matrix, means, variances, diagnostics




def _causal_forward_filter(
    observations: np.ndarray,
    valid_mask: np.ndarray,
    apply_end: int,
    start_probabilities: np.ndarray,
    transition_matrix: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    state_count = len(start_probabilities)
    posterior = np.full((len(observations), state_count), np.nan, dtype=float)
    log_start = np.log(np.clip(start_probabilities, 1e-300, 1.0))
    log_transition = np.log(np.clip(transition_matrix, 1e-300, 1.0))
    previous_log_alpha: np.ndarray | None = None
    for index in range(apply_end):
        if not bool(valid_mask[index]):
            previous_log_alpha = None
            continue
        emission = _emission_log_probabilities(
            observations[index : index + 1], means, variances
        )[0]
        if previous_log_alpha is None:
            current = log_start + emission
        else:
            current = (
                _logsumexp(previous_log_alpha[:, np.newaxis] + log_transition, axis=0)
                + emission
            )
        normalizer = float(_logsumexp(current, axis=0))
        previous_log_alpha = current - normalizer
        posterior[index] = np.exp(previous_log_alpha)
    return posterior


def _validate_hmm_inputs(
    features: DataFrame, train_end: int, apply_end: int
) -> tuple[np.ndarray, np.ndarray]:
    _validate_window_end(apply_end, len(features))
    if (
        isinstance(train_end, bool)
        or not isinstance(train_end, int)
        or not HMM_MINIMUM_TRAIN_ROWS <= train_end < apply_end
    ):
        raise ValueError("v4am_hmm_training_window_invalid")
    if not features.columns.is_unique:
        raise ValueError("v4am_feature_schema_invalid")
    if {LOG_RETURN_FEATURE, REALIZED_VOLATILITY_FEATURE}.difference(features.columns):
        raise ValueError("v4am_hmm_input_features_missing")
    matrix = (
        features.loc[:, [LOG_RETURN_FEATURE, REALIZED_VOLATILITY_FEATURE]]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    valid_mask = np.isfinite(matrix).all(axis=1)
    train_mask = valid_mask & (np.arange(len(features)) < train_end)
    if int(train_mask.sum()) < HMM_MINIMUM_TRAIN_ROWS:
        raise ValueError("v4am_hmm_training_support_insufficient")
    generated = {
        HMM_TRENDING_UP_FEATURE,
        HMM_TRENDING_DOWN_FEATURE,
        HMM_RANGING_FEATURE,
        HMM_STATE_VALID_FEATURE,
    }
    if generated.intersection(features.columns):
        raise ValueError("v4am_hmm_feature_collision")
    return matrix, train_mask


def build_hmm_regime_features(
    base_features: DataFrame,
    *,
    train_end: int,
    apply_end: int,
) -> tuple[DataFrame, dict[str, Any]]:
    """Fit a 3-state diagonal Gaussian HMM on outer train, then forward-filter."""
    matrix, train_mask = _validate_hmm_inputs(base_features, train_end, apply_end)
    train_values = matrix[train_mask]
    scaler_mean = train_values.mean(axis=0)
    scaler_scale = train_values.std(axis=0)
    scaler_scale = np.where(scaler_scale > 1e-12, scaler_scale, 1.0)
    scaled = (matrix - scaler_mean) / scaler_scale
    scaled[~np.isfinite(matrix).all(axis=1)] = 0.0
    (
        start_probabilities,
        transition_matrix,
        means,
        variances,
        fit_diagnostics,
    ) = _fit_gaussian_hmm(scaled[train_mask])
    ordered_states = np.argsort(means[:, 0], kind="stable")
    down_state = int(ordered_states[0])
    ranging_state = int(ordered_states[1])
    up_state = int(ordered_states[2])
    posterior_raw = _causal_forward_filter(
        scaled,
        np.isfinite(matrix).all(axis=1),
        apply_end,
        start_probabilities,
        transition_matrix,
        means,
        variances,
    )
    row_count = len(base_features)
    output_arrays = {
        HMM_TRENDING_UP_FEATURE: np.full(row_count, np.nan, dtype=float),
        HMM_TRENDING_DOWN_FEATURE: np.full(row_count, np.nan, dtype=float),
        HMM_RANGING_FEATURE: np.full(row_count, np.nan, dtype=float),
        HMM_STATE_VALID_FEATURE: np.full(row_count, np.nan, dtype=float),
    }
    output_arrays[HMM_TRENDING_UP_FEATURE][:apply_end] = posterior_raw[
        :apply_end, up_state
    ]
    output_arrays[HMM_TRENDING_DOWN_FEATURE][:apply_end] = posterior_raw[
        :apply_end, down_state
    ]
    output_arrays[HMM_RANGING_FEATURE][:apply_end] = posterior_raw[
        :apply_end, ranging_state
    ]
    output_arrays[HMM_STATE_VALID_FEATURE][:apply_end] = (
        np.isfinite(posterior_raw[:apply_end]).all(axis=1).astype(float)
    )
    transformed = base_features.reset_index(drop=True).copy()
    for name, values in output_arrays.items():
        if name in transformed.columns:
            raise ValueError("v4am_hmm_feature_collision")
        transformed[name] = values
    valid_train = train_mask & np.isfinite(posterior_raw).all(axis=1)
    train_states = np.full(len(base_features), -1, dtype=int)
    train_states[valid_train] = np.argmax(posterior_raw[valid_train], axis=1)
    semantic_counts = {
        "trending_down": int(np.sum(train_states[valid_train] == down_state)),
        "ranging": int(np.sum(train_states[valid_train] == ranging_state)),
        "trending_up": int(np.sum(train_states[valid_train] == up_state)),
    }
    diagnostics = {
        "algorithm": "GAUSSIAN_HMM_DIAGONAL",
        "state_count": HMM_STATE_COUNT,
        "fit_rows": int(train_mask.sum()),
        "fit_end_index_exclusive": train_end,
        "apply_end_index_exclusive": apply_end,
        "fit_scope": "OUTER_TRAIN_ONLY",
        "estimation_method": "BAUM_WELCH_TRAINING_SLICE_ONLY",
        "posterior_method": "CAUSAL_FORWARD_FILTER",
        "smoothing_used": False,
        "lookahead_used": False,
        "mapping_method": "ORDER_BY_MEAN_STANDARDIZED_LOG_RETURN",
        "state_mapping": {
            "trending_down": down_state,
            "ranging": ranging_state,
            "trending_up": up_state,
        },
        "train_posterior_dominant_counts": semantic_counts,
        "scaler_mean_train_only": scaler_mean.tolist(),
        "scaler_scale_train_only": scaler_scale.tolist(),
        "state_means_original_scale": (
            means * scaler_scale[np.newaxis, :] + scaler_mean[np.newaxis, :]
        ).tolist(),
        "transition_matrix": transition_matrix.tolist(),
        "start_probabilities": start_probabilities.tolist(),
        "invalid_rows_before_apply_end": int(
            (~np.isfinite(posterior_raw[:apply_end]).all(axis=1)).sum()
        ),
        "final_holdout_features_unavailable": bool(
            all(np.isnan(values[apply_end:]).all() for values in output_arrays.values())
        ),
        **fit_diagnostics,
    }
    return transformed, diagnostics
