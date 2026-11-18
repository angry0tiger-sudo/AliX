"""Evaluate one frozen LogisticRegression baseline on Regime T labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from scripts.label_regime_t_triple_barrier import (
    EXPECTED_DATA_SHA256,
    EXPECTED_G3_SHA256,
    EXPECTED_PREREG_SHA256,
    PREREG_PATH,
    build_price_features,
    load_verified_prehack_frame,
)

ROOT = Path(__file__).resolve().parents[1]
G3_PATH = ROOT / "config/regime_r/hgbq_cqr_preregistration.json"
FEATURE_COLUMNS = (
    "log_return_1",
    "log_return_4",
    "log_return_24",
    "realized_volatility_6",
    "realized_volatility_24",
    "range_fraction_1",
    "atr_fraction_14",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
)
LABELS = (0, 1)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fold_boundaries(
    anchor: pd.Timestamp, item: dict[str, Any]
) -> tuple[pd.Timestamp, ...]:
    offsets = (
        item["training_month_offsets"],
        item["calibration_month_offsets"],
        item["embargo_month_offsets"],
        item["test_month_offsets"],
    )
    return (
        anchor + pd.DateOffset(months=int(offsets[0][0])),
        anchor + pd.DateOffset(months=int(offsets[0][1])),
        anchor + pd.DateOffset(months=int(offsets[1][0])),
        anchor + pd.DateOffset(months=int(offsets[1][1])),
        anchor + pd.DateOffset(months=int(offsets[2][0])),
        anchor + pd.DateOffset(months=int(offsets[2][1])),
        anchor + pd.DateOffset(months=int(offsets[3][0])),
        anchor + pd.DateOffset(months=int(offsets[3][1])),
    )


def _validate_walk_forward_contract(protocol: dict[str, Any]) -> None:
    expected = {
        "fold_count": 3,
        "training_window_months": 6,
        "calibration_window_months": 1,
        "embargo_window_months": 1,
        "test_window_months": 1,
        "maximum_target_horizon_hours": 24,
    }
    actual = {key: protocol.get(key) for key in expected}
    if actual != expected or protocol.get("purge_policy", {}).get("enabled") is not True:
        raise RuntimeError("regime_t_g3_walk_forward_contract_mismatch")


def _validate_fold_indices(indices: list[int], row_count: int) -> None:
    if not all(0 <= index < row_count for index in indices[:-1]):
        raise RuntimeError("regime_t_fold_boundary_outside_dataset")
    if indices[-1] > row_count:
        raise RuntimeError("regime_t_fold_test_end_outside_dataset")
    interval_pairs = ((0, 1), (2, 3), (4, 5), (6, 7))
    if not all(indices[start] < indices[finish] for start, finish in interval_pairs):
        raise RuntimeError("regime_t_fold_interval_empty")
    if not (
        indices[1] == indices[2]
        and indices[3] == indices[4]
        and indices[5] == indices[6]
    ):
        raise RuntimeError("regime_t_fold_intervals_not_contiguous")


def _build_fold(
    item: dict[str, Any],
    anchor: pd.Timestamp,
    timestamp_ns: np.ndarray,
    row_count: int,
) -> dict[str, Any]:
    boundaries = _fold_boundaries(anchor, item)
    indexes = [
        int(np.searchsorted(timestamp_ns, boundary.value, side="left"))
        for boundary in boundaries
    ]
    _validate_fold_indices(indexes, row_count)
    return {
        "fold_id": str(item["fold_id"]),
        "train_start": indexes[0],
        "train_end": indexes[1],
        "test_start": indexes[6],
        "test_end": indexes[7],
        "train_start_utc": boundaries[0].isoformat(),
        "train_end_utc": boundaries[1].isoformat(),
        "calibration_start_utc": boundaries[2].isoformat(),
        "calibration_end_utc": boundaries[3].isoformat(),
        "embargo_start_utc": boundaries[4].isoformat(),
        "embargo_end_utc": boundaries[5].isoformat(),
        "test_start_utc": boundaries[6].isoformat(),
        "test_end_utc": boundaries[7].isoformat(),
    }


def build_folds(
    frame: pd.DataFrame, manifest: dict[str, Any], g3: dict[str, Any]
) -> list[dict[str, Any]]:
    """Recreate the exact three registered UTC folds without importing Regime R code."""
    protocol = g3["walk_forward_protocol"]
    _validate_walk_forward_contract(protocol)
    anchor = pd.Timestamp(manifest["start_utc"])
    if anchor.tzinfo is None:
        raise RuntimeError("regime_t_fold_anchor_must_be_timezone_aware")
    timestamp_ns = pd.DatetimeIndex(frame["timestamp"]).astype("int64").to_numpy()
    fold_items = protocol["folds"]
    folds = [
        _build_fold(item, anchor, timestamp_ns, len(frame)) for item in fold_items
    ]
    if len(folds) != 3:
        raise RuntimeError("regime_t_expected_three_walk_forward_folds")
    return folds


def build_model() -> Pipeline:
    """Construct the single parameter-locked baseline; no tuning is performed."""
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs",
                    C=1.0,
                    penalty="l2",
                    class_weight=None,
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    matrix = confusion_matrix(y_true, y_pred, labels=list(LABELS)).astype(int)
    return cast(list[list[int]], matrix.tolist())


def _build_fold_masks(
    index: np.ndarray,
    fold: dict[str, Any],
    finite_features: np.ndarray,
    valid_labels: np.ndarray,
    barrier_end: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    full_horizon_ends = index + 24
    common = finite_features & valid_labels
    train_mask = (
        (index >= fold["train_start"])
        & (index < fold["train_end"])
        & (full_horizon_ends < fold["train_end"])
        & common
        & (barrier_end < fold["train_end"])
    )
    test_mask = (
        (index >= fold["test_start"])
        & (index < fold["test_end"])
        & (full_horizon_ends < fold["test_end"])
        & common
        & (barrier_end < fold["test_end"])
    )
    return train_mask, test_mask


def _evaluate_one_fold(
    fold: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    barrier_end: np.ndarray,
    finite_features: np.ndarray,
    valid_labels: np.ndarray,
    index: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    train_mask, test_mask = _build_fold_masks(
        index, fold, finite_features, valid_labels, barrier_end
    )
    train_index = index[train_mask]
    test_index = index[test_mask]
    y_train = y[train_index].astype(np.int64)
    y_test = y[test_index].astype(np.int64)
    if train_index.size < 100 or test_index.size < 20:
        raise RuntimeError(
            f"regime_t_fold_support_insufficient:{fold['fold_id']}:"
            f"train={train_index.size}:test={test_index.size}"
        )
    if np.unique(y_train).size != 2:
        raise RuntimeError(
            f"regime_t_training_fold_missing_binary_class:{fold['fold_id']}"
        )

    model = build_model()
    model.fit(x[train_index], y_train)
    y_pred = model.predict(x[test_index]).astype(np.int64)
    majority_label = int(np.bincount(y_train, minlength=2).argmax())
    majority_pred = np.full(y_test.shape, majority_label, dtype=np.int64)
    report = {
        "fold_id": fold["fold_id"],
        "boundaries": {
            key: value for key, value in fold.items() if key.endswith("_utc")
        },
        "train_rows": int(train_index.size),
        "test_rows": int(test_index.size),
        "train_class_counts": {
            "lower_first_0": int(np.count_nonzero(y_train == 0)),
            "upper_first_1": int(np.count_nonzero(y_train == 1)),
        },
        "test_class_counts": {
            "lower_first_0": int(np.count_nonzero(y_test == 0)),
            "upper_first_1": int(np.count_nonzero(y_test == 1)),
        },
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "uniform_random_expected_accuracy": 0.5,
        "training_majority_label": majority_label,
        "training_majority_baseline_accuracy": float(
            accuracy_score(y_test, majority_pred)
        ),
        "confusion_matrix_labels_0_1": _confusion(y_test, y_pred),
    }
    prediction = pd.DataFrame(
        {
            "fold_id": fold["fold_id"],
            "decision_index": test_index,
            "true_label": y_test,
            "predicted_label": y_pred,
            "prediction_probability_upper_first": model.predict_proba(
                x[test_index]
            )[:, 1],
            "training_majority_label": majority_label,
        }
    )
    return report, prediction


def evaluate_walk_forward(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    manifest: dict[str, Any],
    g3: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    folds = build_folds(frame, manifest, g3)
    x = features.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    y = labels["label"].to_numpy(dtype=np.float64, na_value=np.nan)
    barrier_end = labels["barrier_end_index"].to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    if len(x) != len(y) or len(y) != len(frame):
        raise RuntimeError("regime_t_feature_label_length_mismatch")
    finite_features = np.isfinite(x).all(axis=1)
    valid_labels = np.isin(y, [0.0, 1.0])
    index = np.arange(len(frame), dtype=np.int64)
    evaluated = [
        _evaluate_one_fold(
            fold, x, y, barrier_end, finite_features, valid_labels, index
        )
        for fold in folds
    ]
    fold_reports = [item[0] for item in evaluated]
    prediction_frame = pd.concat([item[1] for item in evaluated], ignore_index=True)
    pooled_true = prediction_frame["true_label"].to_numpy(dtype=np.int64)
    pooled_pred = prediction_frame["predicted_label"].to_numpy(dtype=np.int64)
    pooled_majority = prediction_frame["training_majority_label"].to_numpy(
        dtype=np.int64
    )
    pooled_accuracy = float(accuracy_score(pooled_true, pooled_pred))
    pooled_majority_accuracy = float(accuracy_score(pooled_true, pooled_majority))
    label_counts = labels["outcome"].value_counts().to_dict()
    result = {
        "schema_version": "alix.regime-t.triple-barrier-logistic-evaluation.v1",
        "status": "COMPLETED",
        "dataset": {
            "path": "data/nobitex/prehack/nobitex-btcusdt-1h.csv",
            "sha256": EXPECTED_DATA_SHA256,
            "row_count": int(len(frame)),
            "scope": "PRE_HACK_ONLY",
            "confirmatory_holdout": False,
        },
        "model": {
            "name": "StandardScaler + LogisticRegression",
            "parameters": {
                "solver": "lbfgs",
                "C": 1.0,
                "penalty": "l2",
                "class_weight": None,
                "max_iter": 2000,
                "random_state": 42,
                "threshold": 0.5,
            },
            "fit_count": len(folds),
            "feature_columns": list(FEATURE_COLUMNS),
        },
        "label_outcome_counts": {
            "upper_first": int(label_counts.get("UPPER_FIRST", 0)),
            "lower_first": int(label_counts.get("LOWER_FIRST", 0)),
            "timeout": int(label_counts.get("TIMEOUT", 0)),
            "ambiguous": int(label_counts.get("AMBIGUOUS", 0)),
            "unavailable": int(label_counts.get("UNAVAILABLE", 0)),
        },
        "walk_forward": {
            "fold_count": len(folds),
            "purge_hours": 24,
            "calibration_and_embargo_reserved_unused": True,
            "folds": fold_reports,
        },
        "metrics": {
            "pooled_out_of_fold_rows": int(len(prediction_frame)),
            "pooled_accuracy": pooled_accuracy,
            "pooled_accuracy_pct": pooled_accuracy * 100.0,
            "pooled_balanced_accuracy": float(
                balanced_accuracy_score(pooled_true, pooled_pred)
            ),
            "uniform_random_expected_accuracy": 0.5,
            "pooled_training_majority_baseline_accuracy": pooled_majority_accuracy,
            "pooled_confusion_matrix_labels_0_1": _confusion(
                pooled_true, pooled_pred
            ),
            "confusion_matrix_axis_semantics": (
                "rows=true [0,1], columns=predicted [0,1]"
            ),
            "owner_screening_threshold_pct": 55.0,
            "owner_threshold_passed": pooled_accuracy > 0.55,
            "lift_over_random_guess_pp": (pooled_accuracy - 0.5) * 100.0,
            "lift_over_majority_baseline_pp": (
                pooled_accuracy - pooled_majority_accuracy
            )
            * 100.0,
        },
        "safety": {
            "real_data_scope": "PRE_HACK_ONLY",
            "holdout_bytes_read": False,
            "posthack_data_read": False,
            "private_api_calls": 0,
            "real_order_attempts": 0,
            "paper_orders": 0,
            "exchange_network_access": False,
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
        },
    }
    return result, prediction_frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--predictions-out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("ALIX_REGIME_T_EXECUTION_AUTHORIZED") != "true":
        raise PermissionError("regime_t_requires_owner_execution_authorization")
    if (
        os.environ.get("ALIX_DRY_RUN") != "true"
        or os.environ.get("ALIX_ALLOW_LIVE_TRADING") != "false"
    ):
        raise PermissionError("regime_t_requires_dry_run_and_live_trading_disabled")

    frame, manifest = load_verified_prehack_frame()
    if (
        _sha(PREREG_PATH) != EXPECTED_PREREG_SHA256
        or _sha(G3_PATH) != EXPECTED_G3_SHA256
    ):
        raise RuntimeError("regime_t_preregistration_or_g3_hash_mismatch")
    labels_frame = pd.read_csv(args.labels)
    label_meta_path = args.labels.with_name("labeling-summary.json")
    label_meta = json.loads(label_meta_path.read_text(encoding="utf-8"))
    if (
        label_meta.get("dataset_sha256") != EXPECTED_DATA_SHA256
        or label_meta.get("preregistration_sha256") != EXPECTED_PREREG_SHA256
        or label_meta.get("safety", {}).get("holdout_bytes_read") is not False
    ):
        raise RuntimeError("regime_t_label_artifact_commitment_mismatch")
    features = build_price_features(frame)
    if tuple(features.columns) != FEATURE_COLUMNS:
        raise RuntimeError("regime_t_feature_schema_mismatch")
    g3 = json.loads(G3_PATH.read_text(encoding="utf-8"))
    report, predictions = evaluate_walk_forward(
        frame, features, labels_frame, manifest, g3
    )
    report["preregistration_sha256"] = EXPECTED_PREREG_SHA256
    report["frozen_g3_sha256"] = EXPECTED_G3_SHA256
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    predictions.to_csv(args.predictions_out, index=False)
    print(json.dumps(report["metrics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
