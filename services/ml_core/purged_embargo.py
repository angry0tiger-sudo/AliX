"""Purged validation split mechanics with an explicit post-validation embargo."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Final, Sequence

PURGE_HORIZON_CANDLES: Final[int] = 24
MINIMUM_EMBARGO_CANDLES: Final[int] = 1


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


def _training_partition(
    label_ends: list[int],
    *,
    start: int,
    purge_start: int,
    validation_indices: tuple[int, ...],
    embargo_indices: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    validation_set = set(validation_indices)
    embargo_set = set(embargo_indices)
    train_indices: list[int] = []
    purged_indices: list[int] = []
    for index, label_end in enumerate(label_ends):
        if index in validation_set or index in embargo_set:
            continue
        if index < start and (index >= purge_start or label_end >= purge_start):
            purged_indices.append(index)
            continue
        train_indices.append(index)
    return tuple(train_indices), tuple(purged_indices)


def create_purged_embargo_split(
    label_end_indices: Sequence[object],
    *,
    validation_start: int,
    validation_end: int,
    label_horizon: int = PURGE_HORIZON_CANDLES,
    embargo_candles: int = MINIMUM_EMBARGO_CANDLES,
) -> PurgedEmbargoSplit:
    """Create a time-index fold with the full label horizon purged.

    A pre-validation training event is retained only if its label ends strictly
    before the purge zone. The embargo reserves at least one candle directly
    after validation; neither a training event nor its label interval may consume
    data from that embargo candle.
    """
    if not label_end_indices:
        raise ValueError("label_end_indices_empty")
    horizon = _positive_integer(
        label_horizon, "label_horizon_must_be_positive_integer"
    )
    embargo = _positive_integer(
        embargo_candles, "embargo_candles_must_be_at_least_one"
    )
    start, end = _validated_window(
        len(label_end_indices), validation_start, validation_end
    )
    label_ends = _validated_label_ends(label_end_indices)
    purge_start = max(0, start - horizon)
    validation_indices = tuple(range(start, end + 1))
    embargo_start = end + 1
    embargo_end = min(len(label_ends) - 1, end + embargo)
    embargo_indices = tuple(range(embargo_start, embargo_end + 1))
    train_indices, purged_indices = _training_partition(
        label_ends,
        start=start,
        purge_start=purge_start,
        validation_indices=validation_indices,
        embargo_indices=embargo_indices,
    )
    return PurgedEmbargoSplit(
        train_indices=train_indices,
        validation_indices=validation_indices,
        purged_indices=purged_indices,
        embargo_indices=embargo_indices,
        purge_start=purge_start,
        embargo_start=embargo_start,
        embargo_end=embargo_end,
    )
