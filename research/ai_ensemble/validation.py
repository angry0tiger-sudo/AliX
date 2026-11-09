"""Purged expanding-window validation and Deflated Sharpe diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from numbers import Integral
from statistics import NormalDist
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class PurgedWalkForwardFold:
    fold_index: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    purged_training_rows: int
    embargo_bars: int


def create_purged_walk_forward_folds(
    eligible_indices: Sequence[int],
    label_end_indices: Sequence[int],
    *,
    fold_count: int = 3,
    first_validation_fraction: float = 0.60,
    embargo_bars: int = 24,
    minimum_train_samples: int = 120,
) -> tuple[PurgedWalkForwardFold, ...]:
    """Create strictly expanding past-only folds with label-overlap purge.

    A training sample is admitted only when its forward label ends before the
    validation start and it is outside the configured pre-validation embargo.
    Validation blocks are disjoint and chronological.
    """
    if isinstance(fold_count, bool) or not isinstance(fold_count, Integral) or fold_count < 2:
        raise ValueError("walk_forward_fold_count_invalid")
    if isinstance(embargo_bars, bool) or not isinstance(embargo_bars, Integral) or embargo_bars < 1:
        raise ValueError("walk_forward_embargo_invalid")
    if (
        isinstance(minimum_train_samples, bool)
        or not isinstance(minimum_train_samples, Integral)
        or minimum_train_samples < 2
    ):
        raise ValueError("walk_forward_minimum_train_samples_invalid")
    if not isfinite(first_validation_fraction) or not 0.40 <= first_validation_fraction < 0.90:
        raise ValueError("walk_forward_first_validation_fraction_invalid")
    if not eligible_indices:
        raise ValueError("walk_forward_eligible_indices_empty")
    if len(label_end_indices) <= max(eligible_indices):
        raise ValueError("walk_forward_label_end_indices_short")
    selected = tuple(int(index) for index in eligible_indices)
    if any(isinstance(index, bool) for index in eligible_indices):
        raise ValueError("walk_forward_index_invalid")
    if selected != tuple(sorted(set(selected))) or selected[0] < 0:
        raise ValueError("walk_forward_indices_must_be_unique_and_increasing")
    ends: list[int] = []
    for index, raw_end in enumerate(label_end_indices):
        if isinstance(raw_end, bool) or not isinstance(raw_end, Integral):
            raise ValueError("walk_forward_label_end_index_invalid")
        end = int(raw_end)
        if end < -1 or end >= len(label_end_indices):
            raise ValueError("walk_forward_label_end_index_out_of_bounds")
        if end != -1 and end < index:
            raise ValueError("walk_forward_label_end_before_event")
        ends.append(end)
    first_validation_position = max(
        int(len(selected) * first_validation_fraction),
        int(minimum_train_samples) + int(embargo_bars),
    )
    if len(selected) - first_validation_position < fold_count:
        raise ValueError("walk_forward_insufficient_validation_rows")
    boundaries = np.linspace(
        first_validation_position, len(selected), int(fold_count) + 1, dtype=int
    )
    folds: list[PurgedWalkForwardFold] = []
    for fold_index in range(int(fold_count)):
        left = int(boundaries[fold_index])
        right = int(boundaries[fold_index + 1])
        validation = selected[left:right]
        if not validation:
            raise ValueError("walk_forward_empty_validation_fold")
        validation_start = validation[0]
        candidates = selected[:left]
        train = tuple(
            index
            for index in candidates
            if index < validation_start - int(embargo_bars)
            and ends[index] >= 0
            and ends[index] < validation_start
        )
        if len(train) < int(minimum_train_samples):
            raise ValueError("walk_forward_insufficient_purged_training_rows")
        if max(ends[index] for index in train) >= validation_start:
            raise ValueError("walk_forward_training_label_overlaps_validation")
        folds.append(
            PurgedWalkForwardFold(
                fold_index=fold_index + 1,
                train_indices=train,
                validation_indices=validation,
                purged_training_rows=len(candidates) - len(train),
                embargo_bars=int(embargo_bars),
            )
        )
    return tuple(folds)


def deflated_sharpe_ratio(
    trade_returns: Sequence[float],
    *,
    tested_strategies: int = 1,
) -> float | None:
    """Estimate the probability Sharpe exceeds the expected multiple-test maximum.

    Inputs are non-overlapping trade returns, not overlapping hourly bar returns.
    The approximation uses finite-sample skewness and kurtosis adjustment.
    """
    if isinstance(tested_strategies, bool) or not isinstance(tested_strategies, Integral) or tested_strategies < 1:
        raise ValueError("dsr_tested_strategies_invalid")
    values = np.asarray(trade_returns, dtype=float)
    if values.ndim != 1 or not bool(np.isfinite(values).all()):
        raise ValueError("dsr_returns_must_be_finite_1d")
    if len(values) < 2:
        return None
    deviation = float(values.std(ddof=1))
    mean = float(values.mean())
    if deviation <= 0.0:
        return 1.0 if mean > 0 else 0.0 if mean < 0 else 0.5
    sharpe = mean / deviation
    standardized = (values - mean) / deviation
    skewness = float(np.mean(standardized**3))
    kurtosis = float(np.mean(standardized**4))
    variance = (
        1.0 - skewness * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe * sharpe
    ) / (len(values) - 1)
    standard_error = sqrt(max(variance, 1e-12))
    benchmark = 0.0
    if int(tested_strategies) > 1:
        gamma = 0.5772156649015329
        distribution = NormalDist()
        trials = int(tested_strategies)
        first_quantile = distribution.inv_cdf(1.0 - 1.0 / trials)
        second_quantile = distribution.inv_cdf(1.0 - 1.0 / (trials * np.e))
        benchmark = standard_error * (
            (1.0 - gamma) * first_quantile + gamma * second_quantile
        )
    return float(NormalDist().cdf((sharpe - benchmark) / standard_error))


def non_overlapping_trade_returns(
    validation_indices: Sequence[int],
    probabilities: Sequence[float],
    net_returns: Sequence[float],
    label_end_indices: Sequence[int],
    *,
    probability_threshold: float = 0.50,
) -> tuple[float, ...]:
    """Select long-only entries sequentially and skip until each label window ends."""
    if not isfinite(probability_threshold) or not 0.0 <= probability_threshold <= 1.0:
        raise ValueError("trade_probability_threshold_invalid")
    if len(validation_indices) != len(probabilities):
        raise ValueError("trade_probability_count_mismatch")
    selected: list[float] = []
    blocked_until = -1
    for index, raw_probability in zip(validation_indices, probabilities):
        probability = float(raw_probability)
        if not isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("trade_probability_invalid")
        if index <= blocked_until or probability < probability_threshold:
            continue
        end_index = int(label_end_indices[index])
        result = float(net_returns[index])
        if end_index < index or not isfinite(result):
            raise ValueError("trade_label_window_or_return_invalid")
        selected.append(result)
        blocked_until = end_index
    return tuple(selected)
