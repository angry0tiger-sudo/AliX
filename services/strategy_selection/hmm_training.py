"""Deterministic Gaussian-HMM training helpers for audited historical datasets."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Final

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.cluster import KMeans  # type: ignore[import-untyped]

FEATURE_SCHEMA_VERSION: Final[str] = "market-regime-features.v1"
FEATURE_NAMES: Final[tuple[str, ...]] = (
    "directional_efficiency",
    "realized_volatility",
)
REGIME_NAMES: Final[tuple[str, ...]] = ("trend", "range", "high_volatility")
_EPSILON = 1e-6
_VARIANCE_FLOOR = 1e-12


@dataclass(frozen=True, slots=True)
class HMMFit:
    """Validated numeric parameters and diagnostics from one fit."""

    initial_probabilities: tuple[float, ...]
    transition_matrix: tuple[tuple[float, ...], ...]
    means: tuple[tuple[float, ...], ...]
    standard_deviations: tuple[tuple[float, ...], ...]
    mean_log_likelihood: float
    iterations: int
    state_occupancy: tuple[float, ...]


def calculate_regime_features(candles: DataFrame) -> DataFrame:
    """Calculate point-in-time efficiency and volatility from OHLCV candles."""
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(candles.columns):
        raise ValueError("hmm_training_ohlcv_columns_missing")
    close = pd.to_numeric(candles["close"], errors="coerce")
    returns = close.pct_change(fill_method=None)
    absolute_moves = close.diff().abs()
    path_length = absolute_moves.rolling(12, min_periods=12).sum()
    net_move = (close - close.shift(12)).abs()
    efficiency = net_move.div(path_length.where(path_length > 0))
    efficiency = efficiency.mask(path_length.eq(0), 0.0)
    volatility = returns.rolling(12, min_periods=12).std(ddof=0)
    result = DataFrame(
        {
            FEATURE_NAMES[0]: efficiency.replace([np.inf, -np.inf], np.nan),
            FEATURE_NAMES[1]: volatility.replace([np.inf, -np.inf], np.nan),
        },
        index=candles.index,
    )
    return result


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    shifted = np.exp(values - maximum)
    answer = maximum + np.log(np.sum(shifted, axis=axis, keepdims=True))
    return np.asarray(np.squeeze(answer, axis=axis), dtype=float)


def _validate_features(features: DataFrame) -> np.ndarray:
    if tuple(features.columns) != FEATURE_NAMES:
        raise ValueError("hmm_training_feature_schema_mismatch")
    values = features.to_numpy(dtype=float)
    if values.ndim != 2 or values.shape[0] < 100 or values.shape[1] != 2:
        raise ValueError("hmm_training_sample_count_invalid")
    if not np.isfinite(values).all():
        raise ValueError("hmm_training_features_non_finite")
    if np.any(values[:, 0] < 0) or np.any(values[:, 0] > 1):
        raise ValueError("hmm_training_efficiency_out_of_range")
    if np.any(values[:, 1] < 0):
        raise ValueError("hmm_training_volatility_out_of_range")
    return values


def _initial_parameters(
    values: np.ndarray, random_seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scaled_mean = values.mean(axis=0)
    scaled_std = values.std(axis=0)
    safe_std = np.where(scaled_std > 1e-12, scaled_std, 1.0)
    standardized = (values - scaled_mean) / safe_std
    labels = KMeans(
        n_clusters=3,
        n_init=10,
        random_state=random_seed,
        algorithm="lloyd",
    ).fit_predict(standardized)
    weights = np.bincount(labels, minlength=3).astype(float)
    if np.any(weights < 2):
        raise ValueError("hmm_training_initial_state_underpopulated")
    means = np.vstack([values[labels == state].mean(axis=0) for state in range(3)])
    variances = np.vstack([values[labels == state].var(axis=0) for state in range(3)])
    global_variance = np.maximum(values.var(axis=0), _VARIANCE_FLOOR)
    variances = np.maximum(variances, global_variance * 1e-4)
    transitions = np.ones((3, 3), dtype=float)
    for prior, current in zip(labels[:-1], labels[1:], strict=True):
        transitions[prior, current] += 1.0
    transitions /= transitions.sum(axis=1, keepdims=True)
    return np.full(3, 1.0 / 3.0), transitions, means, variances


def _emission_log_probabilities(
    values: np.ndarray, means: np.ndarray, variances: np.ndarray
) -> np.ndarray:
    delta = values[:, None, :] - means[None, :, :]
    return -0.5 * np.sum(
        np.log(2.0 * math.pi * variances)[None, :, :]
        + (delta * delta) / variances[None, :, :],
        axis=2,
    )


def _forward_backward(
    values: np.ndarray,
    initial: np.ndarray,
    transitions: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    log_initial = np.log(np.maximum(initial, _EPSILON))
    log_transitions = np.log(np.maximum(transitions, _EPSILON))
    emissions = _emission_log_probabilities(values, means, variances)
    row_count, state_count = emissions.shape
    alpha = np.empty((row_count, state_count), dtype=float)
    beta = np.zeros((row_count, state_count), dtype=float)
    alpha[0] = log_initial + emissions[0]
    for index in range(1, row_count):
        alpha[index] = emissions[index] + _logsumexp(
            alpha[index - 1][:, None] + log_transitions, axis=0
        )
    log_likelihood = float(_logsumexp(alpha[-1], axis=0))
    for index in range(row_count - 2, -1, -1):
        beta[index] = _logsumexp(
            log_transitions + emissions[index + 1][None, :] + beta[index + 1][None, :],
            axis=1,
        )
    log_gamma = alpha + beta - log_likelihood
    gamma = np.exp(log_gamma - _logsumexp(log_gamma, axis=1)[:, None])
    xi_counts = np.full((state_count, state_count), _EPSILON, dtype=float)
    for index in range(row_count - 1):
        log_xi = (
            alpha[index][:, None]
            + log_transitions
            + emissions[index + 1][None, :]
            + beta[index + 1][None, :]
            - log_likelihood
        )
        xi_counts += np.exp(log_xi - _logsumexp(log_xi.ravel(), axis=0))
    if not math.isfinite(log_likelihood) or not np.isfinite(gamma).all():
        raise ValueError("hmm_training_likelihood_non_finite")
    return gamma, log_likelihood, xi_counts


def _validate_fit_configuration(max_iter: int, tolerance: float) -> None:
    if (
        isinstance(max_iter, bool)
        or not isinstance(max_iter, int)
        or not 2 <= max_iter <= 500
    ):
        raise ValueError("hmm_training_iteration_limit_invalid")
    if not math.isfinite(tolerance) or not 1e-8 <= tolerance <= 0.1:
        raise ValueError("hmm_training_tolerance_invalid")


def _update_hmm_parameters(
    values: np.ndarray,
    gamma: np.ndarray,
    transition_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state_mass = np.maximum(gamma.sum(axis=0), _EPSILON)
    initial = np.maximum(gamma[0], _EPSILON)
    initial /= initial.sum()
    transitions = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    means = (gamma.T @ values) / state_mass[:, None]
    delta = values[:, None, :] - means[None, :, :]
    variances = (gamma[:, :, None] * delta * delta).sum(axis=0) / state_mass[:, None]
    variances = np.maximum(variances, _VARIANCE_FLOOR)
    return initial, transitions, means, variances


def _run_baum_welch(
    values: np.ndarray,
    *,
    max_iter: int,
    tolerance: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
    initial, transitions, means, variances = _initial_parameters(values, random_seed)
    prior_likelihood = -math.inf
    iterations = 0
    for iteration in range(1, max_iter + 1):
        gamma, likelihood, transition_counts = _forward_backward(
            values, initial, transitions, means, variances
        )
        initial, transitions, means, variances = _update_hmm_parameters(
            values, gamma, transition_counts
        )
        iterations = iteration
        if math.isfinite(prior_likelihood) and abs(likelihood - prior_likelihood) <= (
            tolerance * max(1.0, abs(prior_likelihood))
        ):
            break
        prior_likelihood = likelihood
    gamma, likelihood, _ = _forward_backward(
        values, initial, transitions, means, variances
    )
    return (
        initial,
        transitions,
        means,
        variances,
        gamma,
        likelihood,
        iterations,
    )


def _semantic_state_order(means: np.ndarray) -> list[int]:
    high_volatility = int(np.argmax(means[:, 1]))
    remaining = [index for index in range(3) if index != high_volatility]
    trend = max(remaining, key=lambda index: means[index, 0])
    range_state = next(index for index in remaining if index != trend)
    return [trend, range_state, high_volatility]


def _fit_numeric(
    values: np.ndarray, *, max_iter: int, tolerance: float, random_seed: int
) -> HMMFit:
    _validate_fit_configuration(max_iter, tolerance)
    (
        initial,
        transitions,
        means,
        variances,
        gamma,
        likelihood,
        iterations,
    ) = _run_baum_welch(
        values,
        max_iter=max_iter,
        tolerance=tolerance,
        random_seed=random_seed,
    )
    order = _semantic_state_order(means)
    occupancy = gamma.mean(axis=0)
    return HMMFit(
        initial_probabilities=tuple(float(initial[index]) for index in order),
        transition_matrix=tuple(
            tuple(float(transitions[i, j]) for j in order) for i in order
        ),
        means=tuple(tuple(float(value) for value in means[index]) for index in order),
        standard_deviations=tuple(
            tuple(
                float(math.sqrt(value))
                for value in np.maximum(variances[index], _VARIANCE_FLOOR)
            )
            for index in order
        ),
        mean_log_likelihood=float(likelihood / len(values)),
        iterations=iterations,
        state_occupancy=tuple(float(occupancy[index]) for index in order),
    )


def fit_gaussian_hmm(
    features: DataFrame,
    *,
    max_iter: int = 80,
    tolerance: float = 1e-4,
    random_seed: int = 42,
) -> HMMFit:
    """Fit a three-state diagonal-Gaussian HMM using Baum-Welch/EM."""
    values = _validate_features(features)
    return _fit_numeric(
        values,
        max_iter=max_iter,
        tolerance=tolerance,
        random_seed=random_seed,
    )


def validate_hmm_holdout(
    train_features: DataFrame, validation_features: DataFrame
) -> dict[str, object]:
    """Evaluate log likelihood and state coverage on a chronological holdout."""
    train = _validate_features(train_features)
    validation = _validate_features(validation_features)
    fitted = _fit_numeric(
        train,
        max_iter=80,
        tolerance=1e-4,
        random_seed=42,
    )
    gamma, log_likelihood, _ = _forward_backward(
        validation,
        np.asarray(fitted.initial_probabilities, dtype=float),
        np.asarray(fitted.transition_matrix, dtype=float),
        np.asarray(fitted.means, dtype=float),
        np.square(np.asarray(fitted.standard_deviations, dtype=float)),
    )
    occupancy = gamma.mean(axis=0)
    passed = (
        math.isfinite(log_likelihood)
        and math.isfinite(log_likelihood / len(validation))
        and len(validation) >= 100
        and np.isfinite(occupancy).all()
        and all(value > 0 for value in occupancy)
    )
    return {
        "status": "TECHNICAL_VALIDATION_PASSED"
        if passed
        else "TECHNICAL_VALIDATION_FAILED",
        "passed": bool(passed),
        "validation_rows": int(len(validation)),
        "mean_log_likelihood": float(log_likelihood / len(validation)),
        "state_occupancy": {
            name: float(occupancy[index]) for index, name in enumerate(REGIME_NAMES)
        },
        "economic_acceptance": "NOT_EVALUATED",
    }


def fit_parameters_for_artifact(
    features: DataFrame,
    *,
    max_iter: int = 80,
    tolerance: float = 1e-4,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Convert fitted parameters into JSON-native values in the runtime schema."""
    fit = fit_gaussian_hmm(
        features,
        max_iter=max_iter,
        tolerance=tolerance,
        random_seed=random_seed,
    )
    return {
        "initial_probabilities": [
            format(value, ".17g") for value in fit.initial_probabilities
        ],
        "transition_matrix": [
            [format(value, ".17g") for value in row] for row in fit.transition_matrix
        ],
        "emissions": [
            {
                "regime": regime,
                "means": [format(value, ".17g") for value in fit.means[index]],
                "standard_deviations": [
                    format(value, ".17g") for value in fit.standard_deviations[index]
                ],
            }
            for index, regime in enumerate(REGIME_NAMES)
        ],
        "fit_diagnostics": {
            "mean_log_likelihood": fit.mean_log_likelihood,
            "iterations": fit.iterations,
            "state_occupancy": {
                name: fit.state_occupancy[index]
                for index, name in enumerate(REGIME_NAMES)
            },
        },
    }
