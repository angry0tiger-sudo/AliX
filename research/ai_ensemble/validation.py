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


def _validate_fold_count(fold_count: int) -> None:
    if isinstance(fold_count, bool) or fold_count < 2:
        raise ValueError("walk_forward_fold_count_invalid")


def _validate_embargo_bars(embargo_bars: int) -> None:
    if isinstance(embargo_bars, bool) or embargo_bars < 1:
        raise ValueError("walk_forward_embargo_invalid")


def _validate_minimum_train_samples(minimum_train_samples: int) -> None:
    if isinstance(minimum_train_samples, bool) or minimum_train_samples < 2:
        raise ValueError("walk_forward_minimum_train_samples_invalid")


def _validate_first_validation_fraction(fraction: float) -> None:
    if not isfinite(fraction) or not 0.40 <= fraction < 0.90:
        raise ValueError("walk_forward_first_validation_fraction_invalid")


def _validate_fold_configuration(
    fold_count: int,
    first_validation_fraction: float,
    embargo_bars: int,
    minimum_train_samples: int,
) -> None:
    _validate_fold_count(fold_count)
    _validate_embargo_bars(embargo_bars)
    _validate_minimum_train_samples(minimum_train_samples)
    _validate_first_validation_fraction(first_validation_fraction)


def _validate_eligible_index_source(
    eligible_indices: Sequence[int], label_end_indices: Sequence[int]
) -> None:
    if len(eligible_indices) == 0:
        raise ValueError("walk_forward_eligible_indices_empty")
    if len(label_end_indices) <= max(eligible_indices):
        raise ValueError("walk_forward_label_end_indices_short")


def _normalize_eligible_indices(
    eligible_indices: Sequence[int],
) -> tuple[int, ...]:
    selected = tuple(int(index) for index in eligible_indices)
    if any(isinstance(index, bool) for index in eligible_indices):
        raise ValueError("walk_forward_index_invalid")
    if selected != tuple(sorted(set(selected))) or selected[0] < 0:
        raise ValueError("walk_forward_indices_must_be_unique_and_increasing")
    return selected


def _validate_label_end_index(index: int, raw_end: int, row_count: int) -> int:
    if isinstance(raw_end, bool):
        raise ValueError("walk_forward_label_end_index_invalid")
    end = int(raw_end)
    if end < -1 or end >= row_count:
        raise ValueError("walk_forward_label_end_index_out_of_bounds")
    if end != -1 and end < index:
        raise ValueError("walk_forward_label_end_before_event")
    return end


def _normalize_label_end_indices(
    label_end_indices: Sequence[int],
) -> tuple[int, ...]:
    row_count = len(label_end_indices)
    return tuple(
        _validate_label_end_index(index, raw_end, row_count)
        for index, raw_end in enumerate(label_end_indices)
    )


def _first_validation_position(
    selected_count: int,
    first_validation_fraction: float,
    minimum_train_samples: int,
    embargo_bars: int,
) -> int:
    return max(
        int(selected_count * first_validation_fraction),
        int(minimum_train_samples) + int(embargo_bars),
    )


def _validation_boundaries(
    selected_count: int, first_position: int, fold_count: int
) -> np.ndarray:
    if selected_count - first_position < fold_count:
        raise ValueError("walk_forward_insufficient_validation_rows")
    return np.linspace(first_position, selected_count, int(fold_count) + 1, dtype=int)


def _select_purged_training_indices(
    candidates: Sequence[int],
    label_end_indices: Sequence[int],
    validation_start: int,
    embargo_bars: int,
) -> tuple[int, ...]:
    embargo_cutoff = validation_start - embargo_bars
    return tuple(
        index
        for index in candidates
        if index < embargo_cutoff
        and label_end_indices[index] >= 0
        and label_end_indices[index] < validation_start
    )


def _validate_purged_training_indices(
    train_indices: Sequence[int],
    label_end_indices: Sequence[int],
    validation_start: int,
    minimum_train_samples: int,
) -> None:
    if len(train_indices) < minimum_train_samples:
        raise ValueError("walk_forward_insufficient_purged_training_rows")
    if max(label_end_indices[index] for index in train_indices) >= validation_start:
        raise ValueError("walk_forward_training_label_overlaps_validation")


def _create_purged_walk_forward_fold(
    selected: tuple[int, ...],
    label_end_indices: tuple[int, ...],
    *,
    fold_index: int,
    left: int,
    right: int,
    embargo_bars: int,
    minimum_train_samples: int,
) -> PurgedWalkForwardFold:
    validation = selected[left:right]
    if not validation:
        raise ValueError("walk_forward_empty_validation_fold")
    validation_start = validation[0]
    candidates = selected[:left]
    train = _select_purged_training_indices(
        candidates, label_end_indices, validation_start, embargo_bars
    )
    _validate_purged_training_indices(
        train, label_end_indices, validation_start, minimum_train_samples
    )
    return PurgedWalkForwardFold(
        fold_index=fold_index,
        train_indices=train,
        validation_indices=validation,
        purged_training_rows=len(candidates) - len(train),
        embargo_bars=embargo_bars,
    )


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
    _validate_fold_configuration(
        fold_count,
        first_validation_fraction,
        embargo_bars,
        minimum_train_samples,
    )
    _validate_eligible_index_source(eligible_indices, label_end_indices)
    selected = _normalize_eligible_indices(eligible_indices)
    normalized_label_ends = _normalize_label_end_indices(label_end_indices)
    first_position = _first_validation_position(
        len(selected),
        first_validation_fraction,
        minimum_train_samples,
        embargo_bars,
    )
    boundaries = _validation_boundaries(len(selected), first_position, fold_count)
    normalized_embargo_bars = int(embargo_bars)
    normalized_minimum_train_samples = int(minimum_train_samples)
    folds: list[PurgedWalkForwardFold] = []
    for fold_index in range(int(fold_count)):
        folds.append(
            _create_purged_walk_forward_fold(
                selected,
                normalized_label_ends,
                fold_index=fold_index + 1,
                left=int(boundaries[fold_index]),
                right=int(boundaries[fold_index + 1]),
                embargo_bars=normalized_embargo_bars,
                minimum_train_samples=normalized_minimum_train_samples,
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


def _validate_probability_threshold(probability_threshold: float) -> None:
    if not isfinite(probability_threshold) or not 0.0 <= probability_threshold <= 1.0:
        raise ValueError("trade_probability_threshold_invalid")


def _validated_trade_probability(raw_probability: float) -> float:
    probability = float(raw_probability)
    if not isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("trade_probability_invalid")
    return probability


def _should_skip_trade_entry(
    index: int,
    blocked_until: int,
    probability: float,
    probability_threshold: float,
) -> bool:
    return index <= blocked_until or probability < probability_threshold


def _validated_trade_result(
    index: int,
    net_returns: Sequence[float],
    label_end_indices: Sequence[int],
) -> tuple[int, float]:
    end_index = int(label_end_indices[index])
    result = float(net_returns[index])
    if end_index < index or not isfinite(result):
        raise ValueError("trade_label_window_or_return_invalid")
    return end_index, result


def non_overlapping_trade_returns(
    validation_indices: Sequence[int],
    probabilities: Sequence[float],
    net_returns: Sequence[float],
    label_end_indices: Sequence[int],
    *,
    probability_threshold: float = 0.50,
) -> tuple[float, ...]:
    """Select long-only entries sequentially and skip until each label window ends."""
    _validate_probability_threshold(probability_threshold)
    if len(validation_indices) != len(probabilities):
        raise ValueError("trade_probability_count_mismatch")
    selected: list[float] = []
    blocked_until = -1
    for index, raw_probability in zip(validation_indices, probabilities):
        probability = _validated_trade_probability(raw_probability)
        if _should_skip_trade_entry(
            index, blocked_until, probability, probability_threshold
        ):
            continue
        end_index, result = _validated_trade_result(
            index, net_returns, label_end_indices
        )
        selected.append(result)
        blocked_until = end_index
    return tuple(selected)
