from __future__ import annotations

import numpy as np
import pytest
from pandas import DataFrame

from services.ml_core.walk_forward_validation import create_walk_forward_folds


def _frames(rows: int = 1200) -> tuple[DataFrame, DataFrame]:
    features = DataFrame({"feature": np.arange(rows, dtype=float)})
    labels = DataFrame(
        {
            "label": ["HOLD", "BUY", "SELL"] * (rows // 3)
            + ["HOLD", "BUY", "SELL"][: rows % 3],
            "label_end_idx": [min(index + 12, rows - 1) for index in range(rows)],
        }
    )
    return features, labels


def test_three_walk_forward_folds_are_expanding_and_temporally_disjoint() -> None:
    features, labels = _frames()

    folds = create_walk_forward_folds(
        features, labels, fold_count=3, label_horizon_candles=12
    )

    assert len(folds) == 3
    prior_train_size = 0
    previous_test_end = -1
    for fold in folds:
        assert fold.fold_index in {1, 2, 3}
        assert len(fold.train_indices) > prior_train_size
        prior_train_size = len(fold.train_indices)
        assert min(fold.test_indices) > previous_test_end
        previous_test_end = max(fold.test_indices)
        assert max(fold.train_indices) < min(fold.calibration_indices)
        assert max(fold.calibration_indices) < min(fold.test_indices)
        diagnostics = fold.diagnostics
        assert diagnostics["training_label_end_before_calibration"] is True
        assert diagnostics["calibration_labels_end_before_test"] is True
        assert diagnostics["train_calibration_test_indices_disjoint"] is True
        assert diagnostics["calibration_hold_samples"] >= 30
        assert diagnostics["calibration_non_hold_samples"] >= 30
        assert diagnostics["calibration_support_status"] == "PASS"
        assert diagnostics["pre_calibration_embargo_bars"] == 12


def test_walk_forward_calibration_support_fails_closed_below_thirty_hold_rows() -> None:
    features, labels = _frames()
    labels.loc[576:707, "label"] = "BUY"

    folds = create_walk_forward_folds(features, labels, fold_count=3)

    assert folds[0].diagnostics["calibration_hold_samples"] < 30
    assert (
        folds[0].diagnostics["calibration_support_status"]
        == "SKIPPED_INSUFFICIENT_CALIBRATION_SAMPLES"
    )


@pytest.mark.parametrize("fold_count", [True, 0, 2, 11, 3.5])
def test_walk_forward_rejects_fold_counts_outside_policy(fold_count: object) -> None:
    features, labels = _frames()

    with pytest.raises(ValueError, match="walk_forward_fold_count"):
        create_walk_forward_folds(features, labels, fold_count=fold_count)  # type: ignore[arg-type]


@pytest.mark.parametrize("fraction", [True, 0.0, 0.5, 1.0, float("nan")])
def test_walk_forward_rejects_invalid_calibration_fraction(fraction: object) -> None:
    features, labels = _frames()

    with pytest.raises(ValueError, match="walk_forward_calibration_fraction_invalid"):
        create_walk_forward_folds(
            features,
            labels,
            calibration_fraction=fraction,  # type: ignore[arg-type]
        )


def test_walk_forward_rejects_misaligned_features_and_labels() -> None:
    features, labels = _frames()

    with pytest.raises(
        ValueError, match="walk_forward_features_labels_length_mismatch"
    ):
        create_walk_forward_folds(features.iloc[:-1], labels)


def test_walk_forward_rejects_missing_label_end_column() -> None:
    features, labels = _frames()

    with pytest.raises(ValueError, match="walk_forward_label_end_indices_missing"):
        create_walk_forward_folds(features, labels.drop(columns=["label_end_idx"]))


def test_walk_forward_rejects_invalid_label_end_index() -> None:
    features, labels = _frames()
    labels.loc[10, "label_end_idx"] = 9

    with pytest.raises(ValueError, match="walk_forward_label_end_index_out_of_bounds"):
        create_walk_forward_folds(features, labels)


def test_walk_forward_rejects_noninteger_label_end_index() -> None:
    features, labels = _frames()
    labels.loc[10, "label_end_idx"] = 10.5

    with pytest.raises(ValueError, match="walk_forward_label_end_index_invalid"):
        create_walk_forward_folds(features, labels)


def test_walk_forward_rejects_short_dataset_and_missing_labels() -> None:
    features, labels = _frames(90)

    with pytest.raises(ValueError, match="walk_forward_rows_insufficient"):
        create_walk_forward_folds(features, labels)

    features, labels = _frames()
    with pytest.raises(ValueError, match="walk_forward_labels_missing"):
        create_walk_forward_folds(features, labels.drop(columns=["label"]))

def test_walk_forward_support_skips_when_feature_mask_removes_calibration_rows() -> (
    None
):
    features, labels = _frames()
    features.loc[576:719, "feature"] = np.nan

    folds = create_walk_forward_folds(features, labels, fold_count=3)

    assert folds[0].calibration_indices == ()
    assert folds[0].diagnostics["calibration_rows"] == 0
    assert folds[0].diagnostics["calibration_label_end_max_index"] is None
    assert (
        folds[0].diagnostics["calibration_support_status"]
        == "SKIPPED_INSUFFICIENT_CALIBRATION_SAMPLES"
    )
    assert folds[0].diagnostics["calibration_labels_end_before_test"] is False


@pytest.mark.parametrize("horizon", [True, 0, -1, 1.5])
def test_walk_forward_rejects_invalid_label_horizon(horizon: object) -> None:
    features, labels = _frames()

    with pytest.raises(ValueError, match="walk_forward_label_horizon_invalid"):
        create_walk_forward_folds(
            features,
            labels,
            label_horizon_candles=horizon,  # type: ignore[arg-type]
        )


def test_walk_forward_handles_unlabeled_rows_without_using_them() -> None:
    features, labels = _frames()
    labels.loc[600, "label"] = None
    labels.loc[601, "label_end_idx"] = np.nan

    folds = create_walk_forward_folds(features, labels, fold_count=3)

    assert 600 not in folds[0].calibration_indices
    assert 601 not in folds[0].calibration_indices
