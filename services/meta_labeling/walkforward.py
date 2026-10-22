"""Purged OOF primary-to-meta walk-forward evaluation for offline research."""

from __future__ import annotations

from hashlib import sha256
import math
from typing import Any, Final

import numpy as np
from pandas import DataFrame

from services.meta_labeling.meta_model import (
    META_MODEL_SCHEMA_VERSION,
    MetaLabelModel,
)

_REQUIRED_OOF_COLUMNS: Final[tuple[str, ...]] = (
    "event_index",
    "fold_id",
    "primary_prediction",
    "primary_confidence",
    "primary_train_max_index",
    "primary_train_max_label_end_index",
    "primary_train_count",
)
_REQUIRED_LABEL_COLUMNS: Final[tuple[str, ...]] = (
    "event_index",
    "label_end_index",
    "meta_label",
    "net_return",
    "barrier_label",
    "exit_reason",
)
_LABEL_COLUMNS: Final[tuple[str, ...]] = (
    "event_index",
    "label_end_index",
    "meta_label",
    "net_return",
    "barrier_label",
    "exit_reason",
)
_PROVENANCE_COLUMNS: Final[tuple[str, ...]] = (
    "event_index",
    "fold_id",
    "primary_train_max_index",
    "primary_train_max_label_end_index",
)
SampleReport = dict[str, int | float | bool]


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    brier = float(np.mean((probabilities - labels) ** 2))
    log_loss = float(
        -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
    )
    return {"brier_score": brier, "log_loss": log_loss}


def _validate_columns(frame: DataFrame, required: tuple[str, ...], error: str) -> None:
    if not set(required).issubset(frame.columns):
        raise ValueError(error)


def _validate_unique_events(frame: DataFrame, error: str) -> None:
    if frame["event_index"].duplicated().any():
        raise ValueError(error)


def _validate_inputs(
    oof_predictions: DataFrame,
    barrier_labels: DataFrame,
    minimum_probability: float,
) -> float:
    _validate_columns(
        oof_predictions, _REQUIRED_OOF_COLUMNS, "oof_prediction_schema_missing"
    )
    _validate_columns(
        barrier_labels, _REQUIRED_LABEL_COLUMNS, "triple_barrier_schema_missing"
    )
    cutoff = float(minimum_probability)
    if not math.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError("research_probability_cutoff_invalid")
    _validate_unique_events(oof_predictions, "oof_events_must_be_unique")
    _validate_unique_events(barrier_labels, "barrier_events_must_be_unique")
    return cutoff


def _merge_oof_labels(
    oof_predictions: DataFrame,
    barrier_labels: DataFrame,
) -> DataFrame:
    merged = oof_predictions.merge(
        barrier_labels.loc[:, list(_LABEL_COLUMNS)],
        on="event_index",
        how="inner",
        validate="one_to_one",
    ).sort_values("event_index")
    if merged.empty:
        raise ValueError("no_oof_predictions_matched_barrier_labels")
    return merged


def _validate_primary_oof_provenance(merged: DataFrame) -> None:
    event_indices = merged["event_index"].to_numpy(dtype=int)
    primary_train_max = merged["primary_train_max_index"].to_numpy(dtype=int)
    primary_end_max = merged["primary_train_max_label_end_index"].to_numpy(dtype=int)
    if np.any(primary_train_max >= event_indices):
        raise ValueError("primary_oof_training_index_leaked")
    if np.any(primary_end_max >= event_indices):
        raise ValueError("primary_oof_label_horizon_leaked")


def _build_meta_features(
    candle_features: DataFrame, merged: DataFrame
) -> tuple[DataFrame, list[str]]:
    event_indices = merged["event_index"].to_numpy(dtype=int)
    base_features = candle_features.reset_index(drop=True)
    invalid = (event_indices < 0) | (event_indices >= len(base_features))
    if np.any(invalid):
        raise ValueError("oof_event_index_out_of_feature_bounds")
    result = base_features.iloc[event_indices].copy()
    result.index = event_indices
    result["primary_prediction"] = merged["primary_prediction"].to_numpy(dtype=float)
    result["primary_confidence"] = merged["primary_confidence"].to_numpy(dtype=float)
    return result, list(result.columns)


def _prior_mature_labels(
    merged: DataFrame, fold_id: int, test_start: int
) -> DataFrame:
    eligible = (
        (merged["fold_id"] < fold_id)
        & (merged["event_index"] < test_start)
        & (merged["label_end_index"] < test_start)
    )
    return merged.loc[eligible].copy()


def _insufficient_meta_training(train_rows: DataFrame) -> bool:
    return len(train_rows) < 2 or train_rows["meta_label"].nunique() < 2


def _skipped_fold_report(
    fold_id: int, train_rows: DataFrame, test_rows: DataFrame, test_start: int
) -> dict[str, Any]:
    return {
        "fold_id": fold_id,
        "status": "SKIPPED_INSUFFICIENT_PRIOR_OOF_LABELS",
        "train_count": int(len(train_rows)),
        "test_count": int(len(test_rows)),
        "test_start_index": test_start,
    }


def _fit_and_reload_meta_model(
    train_rows: DataFrame,
    test_rows: DataFrame,
    meta_features: DataFrame,
    feature_names: list[str],
    test_start: int,
) -> tuple[np.ndarray, float, str]:
    train_indices = train_rows["event_index"].to_numpy(dtype=int)
    test_indices = test_rows["event_index"].to_numpy(dtype=int)
    train_features = meta_features.loc[train_indices, feature_names].copy()
    test_features = meta_features.loc[test_indices, feature_names].copy()
    train_provenance = train_rows.loc[:, list(_PROVENANCE_COLUMNS)].reset_index(
        drop=True
    )
    train_features.index = train_indices
    test_features.index = test_indices
    model = MetaLabelModel()
    model.fit(
        train_features,
        train_rows["meta_label"].to_numpy(dtype=int),
        train_provenance,
        train_rows["label_end_index"].to_numpy(dtype=int),
        test_start,
    )
    probabilities = model.predict_probability(test_features)
    restored = MetaLabelModel.from_json(model.to_json())
    restored_probabilities = restored.predict_probability(test_features)
    reload_delta = (
        float(np.max(np.abs(probabilities - restored_probabilities)))
        if len(probabilities)
        else 0.0
    )
    if reload_delta > 1e-12:
        raise ValueError("meta_model_json_reload_roundtrip_failed")
    model_hash = sha256(model.to_json().encode("utf-8")).hexdigest()
    return probabilities, reload_delta, model_hash


def _fold_metrics_report(
    fold_id: int,
    train_rows: DataFrame,
    test_rows: DataFrame,
    test_start: int,
    probabilities: np.ndarray,
    cutoff: float,
    reload_delta: float,
    model_hash: str,
) -> dict[str, Any]:
    labels = test_rows["meta_label"].to_numpy(dtype=float)
    scores = _metrics(labels, probabilities)
    baseline_probability = float(train_rows["meta_label"].mean())
    baseline_scores = _metrics(
        labels, np.full(len(labels), baseline_probability, dtype=float)
    )
    selected = probabilities >= cutoff
    test_net = test_rows["net_return"].to_numpy(dtype=float)
    selected_returns = test_net[selected]
    filtered_mean = float(np.mean(selected_returns)) if len(selected_returns) else None
    return {
        "fold_id": fold_id,
        "status": "EVALUATED",
        "train_count": int(len(train_rows)),
        "test_count": int(len(test_rows)),
        "test_start_index": test_start,
        "test_end_index": int(test_rows["event_index"].max()),
        "train_max_label_end_index": int(train_rows["label_end_index"].max()),
        "brier_score": scores["brier_score"],
        "log_loss": scores["log_loss"],
        "baseline_brier_score": baseline_scores["brier_score"],
        "baseline_log_loss": baseline_scores["log_loss"],
        "research_cutoff": cutoff,
        "research_filter_pass_count": int(np.sum(selected)),
        "research_filter_pass_fraction": float(np.mean(selected)),
        "all_proposal_mean_net_return": float(np.mean(test_net)),
        "filtered_proposal_mean_net_return": filtered_mean,
        "meta_model_reload_roundtrip": "passed",
        "meta_model_reload_max_abs_diff": reload_delta,
        "meta_model_sha256": model_hash,
    }


def _fold_sample_reports(
    fold_id: int,
    test_rows: DataFrame,
    probabilities: np.ndarray,
    cutoff: float,
) -> list[SampleReport]:
    event_indices = test_rows["event_index"].to_numpy(dtype=int)
    labels = test_rows["meta_label"].to_numpy(dtype=int)
    net_returns = test_rows["net_return"].to_numpy(dtype=float)
    selected = probabilities >= cutoff
    return [
        {
            "event_index": int(event_index),
            "fold_id": fold_id,
            "meta_probability": float(probability),
            "meta_label": int(label),
            "net_return": float(net_return),
            "research_filter_pass": bool(accepted),
        }
        for event_index, probability, label, net_return, accepted in zip(
            event_indices,
            probabilities,
            labels,
            net_returns,
            selected,
            strict=True,
        )
    ]


def _evaluate_one_fold(
    fold_id: int,
    merged: DataFrame,
    meta_features: DataFrame,
    feature_names: list[str],
    cutoff: float,
) -> tuple[dict[str, Any], list[SampleReport]]:
    test_rows = merged.loc[merged["fold_id"] == fold_id].copy()
    test_start = int(test_rows["event_index"].min())
    train_rows = _prior_mature_labels(merged, fold_id, test_start)
    if _insufficient_meta_training(train_rows):
        return _skipped_fold_report(fold_id, train_rows, test_rows, test_start), []
    probabilities, reload_delta, model_hash = _fit_and_reload_meta_model(
        train_rows, test_rows, meta_features, feature_names, test_start
    )
    report = _fold_metrics_report(
        fold_id,
        train_rows,
        test_rows,
        test_start,
        probabilities,
        cutoff,
        reload_delta,
        model_hash,
    )
    samples = _fold_sample_reports(fold_id, test_rows, probabilities, cutoff)
    return report, samples


def _finalize_report(
    fold_ids: list[int],
    fold_reports: list[dict[str, Any]],
    samples: list[SampleReport],
    cutoff: float,
    feature_names: list[str],
) -> dict[str, Any]:
    evaluated = [row for row in fold_reports if row["status"] == "EVALUATED"]
    if not evaluated:
        raise ValueError("no_meta_oos_folds_evaluated")
    return {
        "schema_version": META_MODEL_SCHEMA_VERSION,
        "training_signal_requirement": "OUT_OF_FOLD_ONLY",
        "label_maturity_rule": "label_end_index < next_test_start",
        "economic_gates": {"status": "NOT_EVALUATED"},
        "research_probability_cutoff": cutoff,
        "research_cutoff_is_owner_approved": False,
        "fold_count": len(fold_ids),
        "evaluated_fold_count": len(evaluated),
        "skipped_fold_count": len(fold_reports) - len(evaluated),
        "sample_count": len(samples),
        "folds": fold_reports,
        "oos_samples": samples,
        "metrics_are_synthetic_mechanics_only": True,
        "champion": None,
        "promotion_status": "BLOCKED_NO_ECONOMIC_GATE",
        "trading_authorization": "NO_GO",
        "model_output_is_risk_approval": False,
        "feature_names": feature_names,
    }


def evaluate_meta_oof(
    candle_features: DataFrame,
    oof_predictions: DataFrame,
    barrier_labels: DataFrame,
    minimum_probability: float,
) -> dict[str, Any]:
    """Evaluate the meta model using only prior OOF predictions and mature labels.

    The probability cutoff is a synthetic-mechanics diagnostic only. It is not
    economic policy, Risk Gate approval or order permission.
    """
    cutoff = _validate_inputs(
        oof_predictions, barrier_labels, minimum_probability
    )
    merged = _merge_oof_labels(oof_predictions, barrier_labels)
    _validate_primary_oof_provenance(merged)
    meta_features, feature_names = _build_meta_features(candle_features, merged)
    fold_ids = sorted(int(value) for value in merged["fold_id"].unique())
    fold_reports: list[dict[str, Any]] = []
    sample_reports: list[SampleReport] = []
    for fold_id in fold_ids:
        report, samples = _evaluate_one_fold(
            fold_id, merged, meta_features, feature_names, cutoff
        )
        fold_reports.append(report)
        sample_reports.extend(samples)
    return _finalize_report(
        fold_ids, fold_reports, sample_reports, cutoff, feature_names
    )
