"""Purged OOF primary-to-meta walk-forward evaluation for offline research."""

from __future__ import annotations

from hashlib import sha256
import math
from typing import Any

import numpy as np
from pandas import DataFrame

from services.meta_labeling.meta_model import (
    META_MODEL_SCHEMA_VERSION,
    MetaLabelModel,
)


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    brier = float(np.mean((probabilities - labels) ** 2))
    log_loss = float(
        -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
    )
    return {"brier_score": brier, "log_loss": log_loss}


def evaluate_meta_oof(
    candle_features: DataFrame,
    oof_predictions: DataFrame,
    barrier_labels: DataFrame,
    minimum_probability: float,
) -> dict[str, Any]:
    """Evaluate the meta model on later OOF folds using only mature earlier labels.

    The configured probability cutoff is a synthetic-mechanics diagnostic only.
    It is not an owner-approved economic threshold, Risk Gate approval or order
    permission. Each evaluated fold has a separately fitted/reloaded JSON model.
    """
    required_oof = {
        "event_index",
        "fold_id",
        "primary_prediction",
        "primary_confidence",
        "primary_train_max_index",
        "primary_train_max_label_end_index",
        "primary_train_count",
    }
    required_labels = {
        "event_index",
        "label_end_index",
        "meta_label",
        "net_return",
        "barrier_label",
        "exit_reason",
    }
    if not required_oof.issubset(oof_predictions.columns):
        raise ValueError("oof_prediction_schema_missing")
    if not required_labels.issubset(barrier_labels.columns):
        raise ValueError("triple_barrier_schema_missing")
    cutoff = float(minimum_probability)
    if not math.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError("research_probability_cutoff_invalid")
    if oof_predictions["event_index"].duplicated().any():
        raise ValueError("oof_events_must_be_unique")
    if barrier_labels["event_index"].duplicated().any():
        raise ValueError("barrier_events_must_be_unique")

    merged = oof_predictions.merge(
        barrier_labels.loc[
            :,
            [
                "event_index",
                "label_end_index",
                "meta_label",
                "net_return",
                "barrier_label",
                "exit_reason",
            ],
        ],
        on="event_index",
        how="inner",
        validate="one_to_one",
    ).sort_values("event_index")
    if merged.empty:
        raise ValueError("no_oof_predictions_matched_barrier_labels")

    event_indices = merged["event_index"].to_numpy(dtype=int)
    if np.any(merged["primary_train_max_index"].to_numpy(dtype=int) >= event_indices):
        raise ValueError("primary_oof_training_index_leaked")
    if np.any(
        merged["primary_train_max_label_end_index"].to_numpy(dtype=int) >= event_indices
    ):
        raise ValueError("primary_oof_label_horizon_leaked")

    base_features = candle_features.reset_index(drop=True)
    invalid_indices = event_indices[
        (event_indices < 0) | (event_indices >= len(base_features))
    ]
    if len(invalid_indices):
        raise ValueError("oof_event_index_out_of_feature_bounds")
    meta_features = base_features.iloc[event_indices].copy()
    meta_features.index = event_indices
    meta_features["primary_prediction"] = merged["primary_prediction"].to_numpy(
        dtype=float
    )
    meta_features["primary_confidence"] = merged["primary_confidence"].to_numpy(
        dtype=float
    )
    feature_names = list(meta_features.columns)

    fold_ids = sorted(int(value) for value in merged["fold_id"].unique())
    fold_reports: list[dict[str, Any]] = []
    sample_reports: list[dict[str, int | float | bool]] = []
    for fold_id in fold_ids:
        fold_rows = merged.loc[merged["fold_id"] == fold_id].copy()
        if fold_rows.empty:
            continue
        test_start = int(fold_rows["event_index"].min())
        train_mask = (
            (merged["fold_id"] < fold_id)
            & (merged["event_index"] < test_start)
            & (merged["label_end_index"] < test_start)
        )
        train_rows = merged.loc[train_mask].copy()
        test_rows = fold_rows
        if len(train_rows) < 2 or train_rows["meta_label"].nunique() < 2:
            fold_reports.append(
                {
                    "fold_id": fold_id,
                    "status": "SKIPPED_INSUFFICIENT_PRIOR_OOF_LABELS",
                    "train_count": int(len(train_rows)),
                    "test_count": int(len(test_rows)),
                    "test_start_index": test_start,
                }
            )
            continue

        train_event_indices = train_rows["event_index"].to_numpy(dtype=int)
        test_event_indices = test_rows["event_index"].to_numpy(dtype=int)
        train_features = meta_features.loc[train_event_indices, feature_names].copy()
        test_features = meta_features.loc[test_event_indices, feature_names].copy()
        train_provenance = train_rows.loc[
            :,
            [
                "event_index",
                "fold_id",
                "primary_train_max_index",
                "primary_train_max_label_end_index",
            ],
        ].reset_index(drop=True)
        train_features.index = train_event_indices
        test_features.index = test_event_indices
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

        labels = test_rows["meta_label"].to_numpy(dtype=float)
        metrics = _metrics(labels, probabilities)
        baseline_probability = float(train_rows["meta_label"].mean())
        baseline_metrics = _metrics(
            labels, np.full(len(labels), baseline_probability, dtype=float)
        )
        selected = probabilities >= cutoff
        test_net = test_rows["net_return"].to_numpy(dtype=float)
        selected_returns = test_net[selected]
        model_json = model.to_json()
        model_hash = sha256(model_json.encode("utf-8")).hexdigest()
        fold_reports.append(
            {
                "fold_id": fold_id,
                "status": "EVALUATED",
                "train_count": int(len(train_rows)),
                "test_count": int(len(test_rows)),
                "test_start_index": test_start,
                "test_end_index": int(test_rows["event_index"].max()),
                "train_max_label_end_index": int(train_rows["label_end_index"].max()),
                "brier_score": metrics["brier_score"],
                "log_loss": metrics["log_loss"],
                "baseline_brier_score": baseline_metrics["brier_score"],
                "baseline_log_loss": baseline_metrics["log_loss"],
                "research_cutoff": cutoff,
                "research_filter_pass_count": int(np.sum(selected)),
                "research_filter_pass_fraction": float(np.mean(selected)),
                "all_proposal_mean_net_return": float(np.mean(test_net)),
                "filtered_proposal_mean_net_return": (
                    float(np.mean(selected_returns)) if len(selected_returns) else None
                ),
                "meta_model_reload_roundtrip": "passed",
                "meta_model_reload_max_abs_diff": reload_delta,
                "meta_model_sha256": model_hash,
            }
        )
        for event_index, probability, label, net_return, accepted in zip(
            test_event_indices,
            probabilities,
            labels,
            test_net,
            selected,
            strict=True,
        ):
            sample_reports.append(
                {
                    "event_index": int(event_index),
                    "fold_id": fold_id,
                    "meta_probability": float(probability),
                    "meta_label": int(label),
                    "net_return": float(net_return),
                    "research_filter_pass": bool(accepted),
                }
            )

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
        "sample_count": len(sample_reports),
        "folds": fold_reports,
        "oos_samples": sample_reports,
        "metrics_are_synthetic_mechanics_only": True,
        "champion": None,
        "promotion_status": "BLOCKED_NO_ECONOMIC_GATE",
        "trading_authorization": "NO_GO",
        "model_output_is_risk_approval": False,
        "feature_names": feature_names,
    }
