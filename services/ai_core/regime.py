"""Causal forward filtering for research-fitted Gaussian HMM parameters."""

from __future__ import annotations

import math

import numpy as np
from pandas import DataFrame

from services.strategy_selection.hmm_training import (
    FEATURE_NAMES,
    HMMFit,
    REGIME_NAMES,
)


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + math.log(float(np.exp(values - maximum).sum()))


def _log_emission(
    observation: np.ndarray,
    means: np.ndarray,
    standard_deviations: np.ndarray,
) -> np.ndarray:
    variances = np.square(standard_deviations)
    delta = observation[None, :] - means
    log_density = -0.5 * np.sum(
        np.log(2.0 * math.pi * variances) + (delta * delta / variances),
        axis=1,
    )
    return np.asarray(log_density, dtype=float)


def infer_hmm_regimes(
    features: DataFrame, fitted: HMMFit
) -> tuple[tuple[str, ...], tuple[tuple[float, ...], ...]]:
    """Forward-filter unseen rows in order; this routine does not approve a model."""
    if tuple(features.columns) != FEATURE_NAMES:
        raise ValueError("ai_core_hmm_feature_schema_mismatch")
    values = features.to_numpy(dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("ai_core_hmm_observations_invalid")
    if (
        np.any(values[:, 0] < 0)
        or np.any(values[:, 0] > 1)
        or np.any(values[:, 1] < 0)
    ):
        raise ValueError("ai_core_hmm_observations_invalid")
    initial = np.asarray(fitted.initial_probabilities, dtype=float)
    transitions = np.asarray(fitted.transition_matrix, dtype=float)
    means = np.asarray(fitted.means, dtype=float)
    standard_deviations = np.asarray(fitted.standard_deviations, dtype=float)
    if (
        initial.shape != (3,)
        or transitions.shape != (3, 3)
        or means.shape != (3, len(FEATURE_NAMES))
        or standard_deviations.shape != means.shape
    ):
        raise ValueError("ai_core_hmm_parameter_shape_invalid")
    if (
        not np.isfinite(initial).all()
        or not np.isfinite(transitions).all()
        or not np.isfinite(means).all()
        or not np.isfinite(standard_deviations).all()
        or (initial <= 0).any()
        or (transitions <= 0).any()
        or (standard_deviations <= 0).any()
    ):
        raise ValueError("ai_core_hmm_parameters_invalid")
    if abs(float(initial.sum()) - 1.0) > 1e-6 or not np.allclose(
        transitions.sum(axis=1), 1.0, atol=1e-6
    ):
        raise ValueError("ai_core_hmm_probability_contract_invalid")

    log_transition = np.log(transitions)
    prior = np.log(initial)
    regimes: list[str] = []
    posteriors: list[tuple[float, ...]] = []
    for observation in values:
        candidate = prior + _log_emission(observation, means, standard_deviations)
        normalizer = _logsumexp(candidate)
        posterior = np.exp(candidate - normalizer)
        selected = int(np.argmax(posterior))
        regimes.append(REGIME_NAMES[selected])
        posteriors.append(tuple(float(value) for value in posterior))
        prior = np.asarray(
            [
                _logsumexp(np.log(posterior) + log_transition[:, state])
                for state in range(3)
            ],
            dtype=float,
        )
    return tuple(regimes), tuple(posteriors)


__all__ = ["infer_hmm_regimes"]
