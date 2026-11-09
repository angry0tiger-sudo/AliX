"""Purged expanding-window validation and mean-return Z-score diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
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
    if isinstance(fold_count, bool) or fold_count < 2:
        raise ValueError("walk_forward_fold_count_invalid")
    if isinstance(embargo_bars, bool) or embargo_bars < 1:
        raise ValueError("walk_forward_embargo_invalid")
    if isinstance(minimum_train_samples, bool) or minimum_train_samples < 2:
        raise ValueError("walk_forward_minimum_train_samples_invalid")
    if (
        not isfinite(first_validation_fraction)
        or not 0.40 <= first_validation_fraction < 0.90
    ):
        raise ValueError("walk_forward_first_validation_fraction_invalid")
    if len(eligible_indices) == 0:
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
        if isinstance(raw_end, bool):
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


def mean_return_z_score(
    trade_returns: Sequence[float],
) -> float | None:
    """One-sample normal-approximation Z-score for mean net trade return.

    This is not the Deflated Sharpe Ratio. It is undefined when fewer than two
    observations are available or the observed sample variance is zero.
    """
    values = np.asarray(trade_returns, dtype=float)
    if values.ndim != 1 or not bool(np.isfinite(values).all()):
        raise ValueError("z_score_returns_must_be_finite_1d")
    if len(values) < 2:
        return None
    deviation = float(values.std(ddof=1))
    if deviation <= 0.0:
        return None
    standard_error = deviation / sqrt(len(values))
    return float(values.mean() / standard_error)


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
