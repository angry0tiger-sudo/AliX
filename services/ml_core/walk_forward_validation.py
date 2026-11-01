"""Expanding-window walk-forward splits with disjoint calibration and test windows."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite
from numbers import Integral
from typing import Any

import pandas as pd
from pandas import DataFrame

from services.ml_core.purged_embargo import create_purged_embargo_split

LABELS = ("BUY", "SELL", "HOLD")
DEFAULT_WALK_FORWARD_FOLDS = 3
MINIMUM_WALK_FORWARD_FOLDS = 3
MAXIMUM_WALK_FORWARD_FOLDS = 10
FIRST_TEST_START_FRACTION = 0.60
FINAL_TEST_END_FRACTION = 0.90
MINIMUM_CALIBRATION_HOLD_SAMPLES = 30
MINIMUM_CALIBRATION_NON_HOLD_SAMPLES = 30


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """Immutable row indices and diagnostics for one temporal fold."""

    fold_index: int
    train_indices: tuple[int, ...]
    calibration_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    diagnostics: dict[str, Any]


def _positive_integer(value: object, reason: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise ValueError(reason)
    return int(value)


def _validated_label_ends(labels: DataFrame, row_count: int) -> list[int]:
    if "label_end_idx" not in labels.columns:
        raise ValueError("walk_forward_label_end_indices_missing")
    ends: list[int] = []
    for index, raw_value in enumerate(labels["label_end_idx"].tolist()):
        if pd.isna(raw_value):
            ends.append(index)
            continue
        if isinstance(raw_value, bool):
            raise ValueError("walk_forward_label_end_index_invalid")
        try:
            numeric = float(raw_value)
        except (TypeError, ValueError):
            raise ValueError("walk_forward_label_end_index_invalid") from None
        if not isfinite(numeric) or not numeric.is_integer():
            raise ValueError("walk_forward_label_end_index_invalid")
        end_index = int(numeric)
        if end_index < index or end_index >= row_count:
            raise ValueError("walk_forward_label_end_index_out_of_bounds")
        ends.append(end_index)
    return ends


def _maximum_or_none(values: tuple[int, ...], ends: list[int]) -> int | None:
    if not values:
        return None
    return max(ends[index] for index in values)


def _maximum_index_or_none(values: tuple[int, ...]) -> int | None:
    return max(values) if values else None


def _minimum_or_none(values: tuple[int, ...]) -> int | None:
    return min(values) if values else None


def _calibration_support(
    labels: DataFrame, indices: tuple[int, ...]
) -> tuple[int, int]:
    values = labels.iloc[list(indices)]["label"]
    hold_count = int((values == "HOLD").sum())
    return hold_count, len(indices) - hold_count


def _calibration_status(hold_count: int, non_hold_count: int) -> str:
    support_is_sufficient = (
        hold_count >= MINIMUM_CALIBRATION_HOLD_SAMPLES
        and non_hold_count >= MINIMUM_CALIBRATION_NON_HOLD_SAMPLES
    )
    return (
        "PASS" if support_is_sufficient else "SKIPPED_INSUFFICIENT_CALIBRATION_SAMPLES"
    )


def _indices_are_disjoint(
    train: tuple[int, ...],
    calibration: tuple[int, ...],
    test: tuple[int, ...],
) -> bool:
    return (
        not set(train).intersection(calibration)
        and not set(train).intersection(test)
        and not set(calibration).intersection(test)
    )


def _build_fold_indices(
    ends: list[int],
    eligible_indices: set[int],
    *,
    calibration_start: int,
    test_start: int,
    test_end: int,
    horizon: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], Any, int]:
    split = create_purged_embargo_split(
        ends,
        validation_start=calibration_start,
        validation_end=test_end,
        label_horizon_candles=horizon,
    )
    train_limit = calibration_start - split.embargo_size
    train = tuple(
        index
        for index in split.train_indices
        if index in eligible_indices
        and index < train_limit
        and ends[index] < calibration_start
    )
    calibration = tuple(
        index
        for index in range(calibration_start, test_start)
        if index in eligible_indices and ends[index] < test_start
    )
    test = tuple(
        index for index in range(test_start, test_end + 1) if index in eligible_indices
    )
    return train, calibration, test, split, train_limit


def _build_fold_diagnostics(
    features: DataFrame,
    labels: DataFrame,
    ends: list[int],
    eligible_indices: set[int],
    *,
    fold_index: int,
    fold_count: int,
    train: tuple[int, ...],
    calibration: tuple[int, ...],
    test: tuple[int, ...],
    split: Any,
    train_limit: int,
    calibration_start: int,
    test_start: int,
    test_end: int,
) -> dict[str, Any]:
    hold_count, non_hold_count = _calibration_support(labels, calibration)
    train_label_end = _maximum_or_none(train, ends)
    calibration_label_end = _maximum_or_none(calibration, ends)
    return {
        "fold_index": fold_index,
        "fold_count": fold_count,
        "train_start_index": _minimum_or_none(train),
        "train_end_index": _maximum_index_or_none(train),
        "training_label_end_max_index": train_label_end,
        "calibration_start_index": calibration_start,
        "calibration_end_index": test_start - 1,
        "calibration_last_sample_index": _maximum_index_or_none(calibration),
        "calibration_label_end_max_index": calibration_label_end,
        "test_start_index": test_start,
        "test_end_index": test_end,
        "train_rows": len(train),
        "calibration_rows": len(calibration),
        "test_rows": len(test),
        "calibration_hold_samples": hold_count,
        "calibration_non_hold_samples": non_hold_count,
        "calibration_support_status": _calibration_status(hold_count, non_hold_count),
        "minimum_calibration_hold_samples": MINIMUM_CALIBRATION_HOLD_SAMPLES,
        "minimum_calibration_non_hold_samples": MINIMUM_CALIBRATION_NON_HOLD_SAMPLES,
        "pre_calibration_embargo_bars": split.embargo_size,
        "post_test_embargo_bars": len(split.embargo_indices),
        "purged_train_candidate_rows": sum(
            index in eligible_indices and index < train_limit
            for index in split.purged_indices
        ),
        "training_label_end_before_calibration": (
            train_label_end is not None and train_label_end < calibration_start
        ),
        "calibration_labels_end_before_test": (
            calibration_label_end is not None and calibration_label_end < test_start
        ),
        "train_calibration_test_indices_disjoint": _indices_are_disjoint(
            train, calibration, test
        ),
        "common_feature_rows": int(features.notna().all(axis=1).sum()),
    }


def _fold_test_bounds(
    row_count: int, fold_index: int, fold_count: int
) -> tuple[int, int]:
    width = (FINAL_TEST_END_FRACTION - FIRST_TEST_START_FRACTION) / fold_count
    start_fraction = FIRST_TEST_START_FRACTION + (fold_index - 1) * width
    end_fraction = FIRST_TEST_START_FRACTION + fold_index * width
    start = floor(row_count * start_fraction)
    end_exclusive = floor(row_count * end_fraction)
    if end_exclusive <= start:
        raise ValueError("walk_forward_test_window_empty")
    return start, end_exclusive - 1


def _build_one_fold(
    *,
    features: DataFrame,
    labels: DataFrame,
    ends: list[int],
    eligible_indices: set[int],
    fold_index: int,
    fold_count: int,
    horizon: int,
    calibration_fraction: float,
) -> WalkForwardFold:
    test_start, test_end = _fold_test_bounds(len(labels), fold_index, fold_count)
    calibration_start = floor(test_start * (1.0 - calibration_fraction))
    if not 0 < calibration_start < test_start:
        raise ValueError("walk_forward_calibration_window_invalid")
    train, calibration, test, split, train_limit = _build_fold_indices(
        ends,
        eligible_indices,
        calibration_start=calibration_start,
        test_start=test_start,
        test_end=test_end,
        horizon=horizon,
    )
    diagnostics = _build_fold_diagnostics(
        features,
        labels,
        ends,
        eligible_indices,
        fold_index=fold_index,
        fold_count=fold_count,
        train=train,
        calibration=calibration,
        test=test,
        split=split,
        train_limit=train_limit,
        calibration_start=calibration_start,
        test_start=test_start,
        test_end=test_end,
    )
    return WalkForwardFold(
        fold_index=fold_index,
        train_indices=train,
        calibration_indices=calibration,
        test_indices=test,
        diagnostics=diagnostics,
    )


def _validated_fold_count(value: object) -> int:
    folds = _positive_integer(value, "walk_forward_fold_count_invalid")
    if not MINIMUM_WALK_FORWARD_FOLDS <= folds <= MAXIMUM_WALK_FORWARD_FOLDS:
        raise ValueError("walk_forward_fold_count_out_of_policy")
    return folds


def _validated_calibration_fraction(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or not 0.0 < float(value) < 0.5
    ):
        raise ValueError("walk_forward_calibration_fraction_invalid")
    return float(value)


def _validate_walk_forward_frames(
    common_features: DataFrame, labels: DataFrame
) -> None:
    if len(common_features) != len(labels):
        raise ValueError("walk_forward_features_labels_length_mismatch")
    if len(labels) < 100:
        raise ValueError("walk_forward_rows_insufficient")
    if "label" not in labels.columns:
        raise ValueError("walk_forward_labels_missing")


def _eligible_sample_indices(common_features: DataFrame, labels: DataFrame) -> set[int]:
    eligible_mask = (
        common_features.notna().all(axis=1)
        & labels["label"].isin(LABELS)
        & labels["label_end_idx"].notna()
    )
    return {
        int(index)
        for index, eligible in enumerate(eligible_mask.to_numpy(dtype=bool))
        if eligible
    }


def create_walk_forward_folds(
    common_features: DataFrame,
    labels: DataFrame,
    *,
    fold_count: int = DEFAULT_WALK_FORWARD_FOLDS,
    label_horizon_candles: int = 12,
    calibration_fraction: float = 0.20,
) -> tuple[WalkForwardFold, ...]:
    """Create expanding train windows and disjoint tail-calibration / forward-test windows.

    Test windows partition the 60%-90% chronological interval into equal,
    non-overlapping segments according to fold_count. Calibration is taken from
    the tail of the history preceding each test window and excluded from model
    fitting. Training is purged against the combined calibration/test span and
    embargoed by the complete target-label horizon before calibration.
    """
    folds = _validated_fold_count(fold_count)
    horizon = _positive_integer(
        label_horizon_candles, "walk_forward_label_horizon_invalid"
    )
    fraction = _validated_calibration_fraction(calibration_fraction)
    _validate_walk_forward_frames(common_features, labels)
    ends = _validated_label_ends(labels, len(labels))
    eligible_indices = _eligible_sample_indices(common_features, labels)
    return tuple(
        _build_one_fold(
            features=common_features,
            labels=labels,
            ends=ends,
            eligible_indices=eligible_indices,
            fold_index=fold_index,
            fold_count=folds,
            horizon=horizon,
            calibration_fraction=fraction,
        )
        for fold_index in range(1, folds + 1)
    )
