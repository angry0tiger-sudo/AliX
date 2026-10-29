"""Purged validation split mechanics with an explicit post-validation embargo."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence


@dataclass(frozen=True, slots=True)
class PurgedEmbargoSplit:
    """Sample indices and exact temporal windows for a purged CV fold."""

    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    purged_indices: tuple[int, ...]
    embargo_indices: tuple[int, ...]
    purge_start: int
    embargo_start: int
    embargo_end: int
    embargo_size: int


def _as_integer(value: object, reason: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(reason)
    return int(value)


def _positive_integer(value: object, reason: str) -> int:
    integer = _as_integer(value, reason)
    if integer < 1:
        raise ValueError(reason)
    return integer


def _validated_window(
    row_count: int, validation_start: int, validation_end: int
) -> tuple[int, int]:
    start = _as_integer(validation_start, "validation_start_must_be_integer")
    end = _as_integer(validation_end, "validation_end_must_be_integer")
    if start < 0 or end < start or end >= row_count:
        raise ValueError("validation_window_out_of_bounds")
    return start, end


def _validated_label_ends(label_end_indices: Sequence[object]) -> list[int]:
    row_count = len(label_end_indices)
    label_ends: list[int] = []
    for index, raw_end in enumerate(label_end_indices):
        label_end = _as_integer(raw_end, "label_end_index_must_be_integer")
        if label_end < index or label_end >= row_count:
            raise ValueError("label_end_index_out_of_bounds")
        label_ends.append(label_end)
    return label_ends


def _windows_overlap(
    left_start: int, left_end: int, right_start: int, right_end: int
) -> bool:
    return left_start <= right_end and right_start <= left_end


def _training_partition(
    label_ends: list[int],
    *,
    validation_indices: tuple[int, ...],
    embargo_indices: tuple[int, ...],
    test_windows: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    validation_set = set(validation_indices)
    embargo_set = set(embargo_indices)
    train_indices: list[int] = []
    purged_indices: list[int] = []
    for index, label_end in enumerate(label_ends):
        if index in validation_set or index in embargo_set:
            continue
        forward_start = index + 1
        if forward_start <= label_end and any(
            _windows_overlap(forward_start, label_end, test_start, test_end)
            for test_start, test_end in test_windows
        ):
            purged_indices.append(index)
            continue
        train_indices.append(index)
    return tuple(train_indices), tuple(purged_indices)


def create_purged_embargo_split(
    label_end_indices: Sequence[object],
    *,
    validation_start: int,
    validation_end: int,
    longest_feature_window: int,
) -> PurgedEmbargoSplit:
    """Split samples by actual label-window overlap and a post-test embargo.

    Each sample's forward label window is [index + 1, label_end_indices[index]].
    Any candidate training window overlapping a test sample's forward window is
    purged. Test and embargo rows are excluded from the training indices.
    """
    if not label_end_indices:
        raise ValueError("label_end_indices_empty")
    feature_window = _positive_integer(
        longest_feature_window, "longest_feature_window_must_be_positive_integer"
    )
    embargo = max(feature_window, int(0.01 * len(label_end_indices)))
    start, end = _validated_window(
        len(label_end_indices), validation_start, validation_end
    )
    label_ends = _validated_label_ends(label_end_indices)
    validation_indices = tuple(range(start, end + 1))
    test_windows = tuple(
        (index + 1, label_ends[index])
        for index in validation_indices
        if index + 1 <= label_ends[index]
    )
    purge_start = min((window[0] for window in test_windows), default=start + 1)
    embargo_start = end + 1
    embargo_end = min(len(label_ends) - 1, end + embargo)
    embargo_indices = tuple(range(embargo_start, embargo_end + 1))
    train_indices, purged_indices = _training_partition(
        label_ends,
        validation_indices=validation_indices,
        embargo_indices=embargo_indices,
        test_windows=test_windows,
    )
    return PurgedEmbargoSplit(
        train_indices=train_indices,
        validation_indices=validation_indices,
        purged_indices=purged_indices,
        embargo_indices=embargo_indices,
        purge_start=purge_start,
        embargo_start=embargo_start,
        embargo_end=embargo_end,
        embargo_size=embargo,
    )
