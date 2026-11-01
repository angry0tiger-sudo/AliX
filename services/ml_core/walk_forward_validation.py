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


def _calibration_support(
    labels: DataFrame, indices: tuple[int, ...]
) -> tuple[int, int]:
    values = labels.iloc[list(indices)]["label"]
    hold_count = int((values == "HOLD").sum())
    return hold_count, len(indices) - hold_count


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
    row_count = len(labels)
    test_start, test_end = _fold_test_bounds(row_count, fold_index, fold_count)
    calibration_start = floor(test_start * (1.0 - calibration_fraction))
    if calibration_start <= 0 or calibration_start >= test_start:
        raise ValueError("walk_forward_calibration_window_invalid")

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

    hold_count, non_hold_count = _calibration_support(labels, calibration)
    train_label_end = _maximum_or_none(train, ends)
    calibration_label_end = _maximum_or_none(calibration, ends)
    index_sets_disjoint = (
        not set(train).intersection(calibration)
        and not set(train).intersection(test)
        and not set(calibration).intersection(test)
    )
    diagnostics: dict[str, Any] = {
        "fold_index": fold_index,
        "fold_count": fold_count,
        "train_start_index": min(train) if train else None,
        "train_end_index": max(train) if train else None,
        "training_label_end_max_index": train_label_end,
        "calibration_start_index": calibration_start,
        "calibration_end_index": test_start - 1,
        "calibration_last_sample_index": max(calibration) if calibration else None,
        "calibration_label_end_max_index": calibration_label_end,
        "test_start_index": test_start,
        "test_end_index": test_end,
        "train_rows": len(train),
        "calibration_rows": len(calibration),
        "test_rows": len(test),
        "calibration_hold_samples": hold_count,
        "calibration_non_hold_samples": non_hold_count,
        "calibration_support_status": (
            "PASS"
            if hold_count >= MINIMUM_CALIBRATION_HOLD_SAMPLES
            and non_hold_count >= MINIMUM_CALIBRATION_NON_HOLD_SAMPLES
            else "SKIPPED_INSUFFICIENT_CALIBRATION_SAMPLES"
        ),
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
        "train_calibration_test_indices_disjoint": index_sets_disjoint,
        "common_feature_rows": int(features.notna().all(axis=1).sum()),
    }
    return WalkForwardFold(
        fold_index=fold_index,
        train_indices=train,
        calibration_indices=calibration,
        test_indices=test,
        diagnostics=diagnostics,
    )


def create_walk_forward_folds(
    common_features: DataFrame,
    labels: DataFrame,
    *,
    fold_count: int = DEFAULT_WALK_FORWARD_FOLDS,
    label_horizon_candles: int = 12,
    calibration_fraction: float = 0.20,
) -> tuple[WalkForwardFold, ...]:
    """Create expanding train windows and disjoint tail-calibration / forward-test windows.

    The three default test windows cover 60%-70%, 70%-80%, and 80%-90% of
    the chronological rows. Calibration is taken from the tail of the history
    preceding each test window and is excluded from model fitting. Training is
    purged against the combined calibration/test span and embargoed by the
    complete target-label horizon before calibration.
    """
    folds = _positive_integer(fold_count, "walk_forward_fold_count_invalid")
    if folds < MINIMUM_WALK_FORWARD_FOLDS or folds > MAXIMUM_WALK_FORWARD_FOLDS:
        raise ValueError("walk_forward_fold_count_out_of_policy")
    horizon = _positive_integer(
        label_horizon_candles, "walk_forward_label_horizon_invalid"
    )
    if (
        isinstance(calibration_fraction, bool)
        or not isinstance(calibration_fraction, (int, float))
        or not isfinite(float(calibration_fraction))
        or not 0.0 < float(calibration_fraction) < 0.5
    ):
        raise ValueError("walk_forward_calibration_fraction_invalid")
    if len(common_features) != len(labels):
        raise ValueError("walk_forward_features_labels_length_mismatch")
    if len(labels) < 100:
        raise ValueError("walk_forward_rows_insufficient")
    if "label" not in labels.columns:
        raise ValueError("walk_forward_labels_missing")

    row_count = len(labels)
    ends = _validated_label_ends(labels, row_count)
    eligible_mask = (
        common_features.notna().all(axis=1)
        & labels["label"].isin(LABELS)
        & labels["label_end_idx"].notna()
    )
    eligible_indices = {
        int(index)
        for index, eligible in enumerate(eligible_mask.to_numpy(dtype=bool))
        if eligible
    }
    return tuple(
        _build_one_fold(
            features=common_features,
            labels=labels,
            ends=ends,
            eligible_indices=eligible_indices,
            fold_index=fold_index,
            fold_count=folds,
            horizon=horizon,
            calibration_fraction=float(calibration_fraction),
        )
        for fold_index in range(1, folds + 1)
    )
