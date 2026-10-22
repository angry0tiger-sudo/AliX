"""OOF-only Meta-Label classifier with JSON-native, fail-closed inference."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
import math
from typing import Any, Final

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

META_MODEL_SCHEMA_VERSION: Final[str] = "alix.meta_labeling.logistic.v1"
OOF_PROVENANCE_COLUMNS: Final[tuple[str, ...]] = (
    "event_index",
    "fold_id",
    "primary_train_max_index",
    "primary_train_max_label_end_index",
)


@dataclass(frozen=True)
class OOFFold:
    """One expanding, chronological primary-model fold using positional indices."""

    train_start: int
    train_stop: int
    test_start: int
    test_stop: int


def _finite_matrix(features: DataFrame) -> np.ndarray:
    if features.empty or len(features.columns) == 0:
        raise ValueError("meta_features_empty")
    if not features.columns.is_unique:
        raise ValueError("meta_feature_columns_must_be_unique")
    values: np.ndarray = (
        features.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    )
    if not np.isfinite(values).all():
        raise ValueError("meta_features_non_finite")
    return values


def _validate_folds(folds: Sequence[OOFFold], row_count: int) -> None:
    if not folds:
        raise ValueError("oof_folds_required")
    previous_test_stop = -1
    for fold in folds:
        if (
            fold.train_start != 0
            or fold.train_stop <= fold.train_start
            or fold.test_start < fold.train_stop
            or fold.test_start >= fold.test_stop
            or fold.test_stop > row_count
        ):
            raise ValueError("invalid_expanding_oof_fold")
        if fold.test_start < previous_test_stop:
            raise ValueError("oof_test_folds_overlap_or_are_out_of_order")
        previous_test_stop = fold.test_stop


def generate_oof_primary_predictions(
    features: DataFrame,
    primary_target: pd.Series,
    label_end_indices: Sequence[int] | np.ndarray,
    folds: Sequence[OOFFold],
    estimator_factory: Callable[[], Any],
) -> DataFrame:
    """Generate primary signals only on held-out chronological fold rows.

    A training sample is retained only if its primary label had ended strictly
    before the test fold begins. The output carries provenance for the meta model
    to independently validate; in-sample predictions are never emitted.
    """
    row_count = len(features)
    if row_count != len(primary_target) or row_count != len(label_end_indices):
        raise ValueError("primary_feature_target_end_length_mismatch")
    finite_features = features.reset_index(drop=True)
    targets = pd.to_numeric(primary_target, errors="coerce").to_numpy(dtype=float)
    label_ends = np.asarray(label_end_indices, dtype=int)
    if label_ends.shape != (row_count,):
        raise ValueError("primary_label_end_index_shape_mismatch")
    if np.any(label_ends < np.arange(row_count)):
        raise ValueError("primary_label_end_index_precedes_event")
    _validate_folds(folds, row_count)
    records: list[dict[str, int | float]] = []
    prior_test_stop = -1
    for fold_id, fold in enumerate(folds):
        if fold.test_start < prior_test_stop:
            raise ValueError("oof_test_folds_overlap_or_are_out_of_order")
        train_indices = np.arange(fold.train_start, fold.train_stop, dtype=int)
        train_ends = label_ends[train_indices]
        train_x = (
            finite_features.iloc[train_indices]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )
        train_y = targets[train_indices]
        eligible = (
            (train_ends < fold.test_start)
            & (train_indices < fold.test_start)
            & np.isfinite(train_y)
            & np.isfinite(train_x).all(axis=1)
        )
        fit_indices = train_indices[eligible]
        if len(fit_indices) < 2:
            raise ValueError("oof_fold_has_insufficient_purged_training_rows")
        fit_y = targets[fit_indices]
        if len(np.unique(fit_y)) < 2:
            raise ValueError("oof_primary_training_target_has_one_class")

        estimator = estimator_factory()
        estimator.fit(train_x[eligible], fit_y)
        test_indices = np.arange(fold.test_start, fold.test_stop, dtype=int)
        test_x = (
            finite_features.iloc[test_indices]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )
        valid_test = np.isfinite(test_x).all(axis=1)
        if not valid_test.all():
            test_indices = test_indices[valid_test]
            test_x = test_x[valid_test]
        if len(test_indices) == 0:
            raise ValueError("oof_fold_has_no_finite_test_features")
        predictions = np.asarray(estimator.predict(test_x), dtype=float).reshape(-1)
        if len(predictions) != len(test_indices) or not np.isfinite(predictions).all():
            raise ValueError("oof_primary_prediction_invalid")

        confidence = np.ones(len(test_indices), dtype=float)
        predict_proba = getattr(estimator, "predict_proba", None)
        classes = np.asarray(getattr(estimator, "classes_", []))
        if callable(predict_proba):
            probabilities = np.asarray(predict_proba(test_x), dtype=float)
            if (
                probabilities.ndim != 2
                or probabilities.shape[0] != len(test_indices)
                or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0)
                or np.any(probabilities > 1)
            ):
                raise ValueError("oof_primary_probability_invalid")
            confidence = np.max(probabilities, axis=1)
            if classes.size and len(classes) != probabilities.shape[1]:
                raise ValueError("oof_primary_class_schema_mismatch")

        max_train_index = int(np.max(fit_indices))
        max_label_end = int(np.max(label_ends[fit_indices]))
        if max_label_end >= fold.test_start:
            raise ValueError("oof_purge_contract_failed")
        for event_index, prediction, score in zip(
            test_indices, predictions, confidence, strict=True
        ):
            records.append(
                {
                    "event_index": int(event_index),
                    "fold_id": int(fold_id),
                    "primary_prediction": float(prediction),
                    "primary_confidence": float(score),
                    "primary_train_max_index": max_train_index,
                    "primary_train_max_label_end_index": max_label_end,
                    "primary_train_count": int(len(fit_indices)),
                }
            )
        prior_test_stop = fold.test_stop
    result = DataFrame.from_records(records)
    if result.empty or result["event_index"].duplicated().any():
        raise ValueError("oof_predictions_empty_or_duplicate")
    return result


class MetaLabelModel:
    """Binary meta model; its output filters research candidates only."""

    def __init__(self, regularization_c: float = 1.0, max_iter: int = 1000) -> None:
        if not math.isfinite(regularization_c) or regularization_c <= 0:
            raise ValueError("regularization_c_must_be_positive")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter_must_be_positive_integer")
        self.regularization_c = float(regularization_c)
        self.max_iter = max_iter
        self.feature_names_: tuple[str, ...] = ()
        self.means_: np.ndarray | None = None
        self.scales_: np.ndarray | None = None
        self.coefficients_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(
        self,
        features: DataFrame,
        labels: Sequence[int] | np.ndarray,
        provenance: DataFrame,
        meta_label_end_indices: Sequence[int] | np.ndarray,
        evaluation_cutoff_index: int,
    ) -> MetaLabelModel:
        """Fit only on mature labels backed by purged OOF primary predictions."""
        values: np.ndarray = _finite_matrix(features)
        label_values = np.asarray(labels)
        if (
            label_values.shape != (len(features),)
            or not np.isin(label_values, [0, 1]).all()
        ):
            raise ValueError("meta_labels_must_be_binary_and_aligned")
        if len(np.unique(label_values)) != 2:
            raise ValueError("meta_training_requires_both_classes")
        if len(provenance) != len(features) or not set(OOF_PROVENANCE_COLUMNS).issubset(
            provenance.columns
        ):
            raise ValueError("oof_provenance_missing_or_misaligned")
        event_indices = provenance["event_index"].to_numpy(dtype=int)
        if not np.array_equal(event_indices, features.index.to_numpy(dtype=int)):
            raise ValueError("oof_event_indices_do_not_match_meta_features")
        fold_ids = provenance["fold_id"].to_numpy(dtype=int)
        primary_train_max = provenance["primary_train_max_index"].to_numpy(dtype=int)
        primary_end_max = provenance["primary_train_max_label_end_index"].to_numpy(
            dtype=int
        )
        meta_ends = np.asarray(meta_label_end_indices, dtype=int)
        if meta_ends.shape != (len(features),):
            raise ValueError("meta_label_end_indices_misaligned")
        if (
            np.any(fold_ids < 0)
            or np.any(primary_train_max >= event_indices)
            or np.any(primary_end_max >= event_indices)
            or np.any(meta_ends >= int(evaluation_cutoff_index))
            or np.any(event_indices >= int(evaluation_cutoff_index))
        ):
            raise ValueError("meta_training_oof_or_purge_contract_failed")

        scaler = StandardScaler()
        scaled = scaler.fit_transform(values)
        classifier = LogisticRegression(
            C=self.regularization_c,
            max_iter=self.max_iter,
            random_state=42,
            solver="lbfgs",
        )
        classifier.fit(scaled, label_values.astype(int))
        if list(np.asarray(classifier.classes_).tolist()) != [0, 1]:
            raise ValueError("meta_classifier_class_order_invalid")
        self.feature_names_ = tuple(str(column) for column in features.columns)
        self.means_ = np.asarray(scaler.mean_, dtype=float)
        self.scales_ = np.asarray(scaler.scale_, dtype=float)
        self.scales_[self.scales_ == 0.0] = 1.0
        self.coefficients_ = np.asarray(classifier.coef_[0], dtype=float)
        self.intercept_ = float(classifier.intercept_[0])
        return self

    @property
    def fitted(self) -> bool:
        return (
            bool(self.feature_names_)
            and self.means_ is not None
            and self.scales_ is not None
            and self.coefficients_ is not None
            and self.intercept_ is not None
        )

    def predict_probability(self, features: DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("meta_model_not_fitted")
        if tuple(str(column) for column in features.columns) != self.feature_names_:
            raise ValueError("meta_feature_schema_mismatch")
        values = _finite_matrix(features)
        assert self.means_ is not None
        assert self.scales_ is not None
        assert self.coefficients_ is not None
        assert self.intercept_ is not None
        logits: np.ndarray = (
            ((values - self.means_) / self.scales_) @ self.coefficients_
        )
        logits = logits + self.intercept_
        probability: np.ndarray = 1.0 / (1.0 + np.exp(-np.clip(logits, -700.0, 700.0)))
        if not np.isfinite(probability).all():
            raise ValueError("meta_probability_non_finite")
        return probability

    def filter_research_candidates(
        self, features: DataFrame, minimum_probability: float
    ) -> DataFrame:
        """Return scores/filter flags; never orders, signals or risk authorization."""
        result = DataFrame(index=features.index)
        try:
            threshold = float(minimum_probability)
            if not math.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError("research_threshold_invalid")
            probabilities = self.predict_probability(features)
            result["meta_probability"] = probabilities
            result["research_filter_pass"] = probabilities >= threshold
            result["reason_code"] = np.where(
                result["research_filter_pass"],
                "score_above_research_cutoff",
                "score_below_research_cutoff",
            )
        except (ValueError, TypeError):
            result["meta_probability"] = np.nan
            result["research_filter_pass"] = False
            result["reason_code"] = "meta_model_or_input_invalid_fail_closed"
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize with a transparent JSON schema; no pickle/joblib is used."""
        if not self.fitted:
            raise ValueError("cannot_serialize_unfitted_meta_model")
        assert self.means_ is not None
        assert self.scales_ is not None
        assert self.coefficients_ is not None
        assert self.intercept_ is not None
        return {
            "schema_version": META_MODEL_SCHEMA_VERSION,
            "model_type": "standardized_logistic_regression",
            "training_signal_requirement": "out_of_fold_only",
            "feature_names": list(self.feature_names_),
            "means": self.means_.tolist(),
            "scales": self.scales_.tolist(),
            "coefficients": self.coefficients_.tolist(),
            "intercept": self.intercept_,
            "regularization_c": self.regularization_c,
            "max_iter": self.max_iter,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MetaLabelModel:
        if payload.get("schema_version") != META_MODEL_SCHEMA_VERSION:
            raise ValueError("meta_model_schema_version_mismatch")
        if payload.get("training_signal_requirement") != "out_of_fold_only":
            raise ValueError("meta_model_oof_contract_missing")
        feature_names = payload.get("feature_names")
        if (
            not isinstance(feature_names, list)
            or not feature_names
            or not all(isinstance(name, str) and name for name in feature_names)
        ):
            raise ValueError("meta_model_feature_schema_invalid")
        model = cls(
            regularization_c=float(payload["regularization_c"]),
            max_iter=int(payload["max_iter"]),
        )
        means = np.asarray(payload["means"], dtype=float)
        scales = np.asarray(payload["scales"], dtype=float)
        coefficients = np.asarray(payload["coefficients"], dtype=float)
        intercept = float(payload["intercept"])
        expected_shape = (len(feature_names),)
        if (
            any(
                array.shape != expected_shape or not np.isfinite(array).all()
                for array in (means, scales, coefficients)
            )
            or np.any(scales <= 0)
            or not math.isfinite(intercept)
        ):
            raise ValueError("meta_model_serialized_parameters_invalid")
        model.feature_names_ = tuple(feature_names)
        model.means_ = means
        model.scales_ = scales
        model.coefficients_ = coefficients
        model.intercept_ = intercept
        return model

    @classmethod
    def from_json(cls, content: str) -> MetaLabelModel:
        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise ValueError("meta_model_json_root_must_be_object")
        return cls.from_dict(payload)
