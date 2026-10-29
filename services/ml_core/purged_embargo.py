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


def _positive_integer(value: int, reason: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(reason)


def create_purged_embargo_split(
    label_end_indices: Sequence[int],
    *,
    validation_start: int,
    validation_end: int,
    label_horizon: int = PURGE_HORIZON_CANDLES,
    embargo_candles: int = MINIMUM_EMBARGO_CANDLES,
) -> PurgedEmbargoSplit:
    """Create one time-index fold without labels overlapping purge/validation.

    The purge zone is the `label_horizon` candles immediately before validation.
    A pre-validation training event is retained only if its label ends strictly
    before that purge zone. The embargo reserves at least one candle directly
    after validation; neither a training event nor its label interval may consume
    data from that embargo candle.
    """
    if not label_end_indices:
        raise ValueError("label_end_indices_empty")
    _positive_integer(label_horizon, "label_horizon_must_be_positive_integer")
    _positive_integer(embargo_candles, "embargo_candles_must_be_at_least_one")

    row_count = len(label_end_indices)
    for name, value in (
        ("validation_start", validation_start),
        ("validation_end", validation_end),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name}_must_be_integer")
    start = int(validation_start)
    end = int(validation_end)
    if start < 0 or end < start or end >= row_count:
        raise ValueError("validation_window_out_of_bounds")

    label_ends: list[int] = []
    for index, raw_end in enumerate(label_end_indices):
        if isinstance(raw_end, bool) or not isinstance(raw_end, Integral):
            raise ValueError("label_end_index_must_be_integer")
        label_end = int(raw_end)
        if label_end < index or label_end >= row_count:
            raise ValueError("label_end_index_out_of_bounds")
        label_ends.append(label_end)

    purge_start = max(0, start - int(label_horizon))
    embargo_start = end + 1
    embargo_end = min(row_count - 1, end + int(embargo_candles))
    validation_indices = tuple(range(start, end + 1))
    embargo_indices = tuple(range(embargo_start, embargo_end + 1))

    train_indices: list[int] = []
    purged_indices: list[int] = []
    validation_set = set(validation_indices)
    embargo_set = set(embargo_indices)
    for index, label_end in enumerate(label_ends):
        if index in validation_set or index in embargo_set:
            continue
        if index < start and (index >= purge_start or label_end >= purge_start):
            purged_indices.append(index)
            continue
        if index < start and label_end >= start:
            purged_indices.append(index)
            continue
        # Any post-validation sample must start after embargo; its label window
        # therefore cannot consume an embargo candle.
        train_indices.append(index)

    return PurgedEmbargoSplit(
        train_indices=tuple(train_indices),
        validation_indices=validation_indices,
        purged_indices=tuple(purged_indices),
        embargo_indices=embargo_indices,
        purge_start=purge_start,
        embargo_start=embargo_start,
        embargo_end=embargo_end,
    )
