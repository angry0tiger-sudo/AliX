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
    values: np.ndarray = features.apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(values).all():
        raise ValueError("meta_features_non_finite")
    return values


def _validate_folds(folds: Sequence[OOFFold], row_count: int) -> None:
    if not folds:
        raise ValueError("oof_folds_required")
    previous_test_stop = -1
    for fold in folds:
        invalid_bounds = (
            fold.train_start != 0
            or fold.train_stop <= fold.train_start
            or fold.test_start < fold.train_stop
            or fold.test_start >= fold.test_stop
            or fold.test_stop > row_count
        )
        if invalid_bounds:
            raise ValueError("invalid_expanding_oof_fold")
        if fold.test_start < previous_test_stop:
            raise ValueError("oof_test_folds_overlap_or_are_out_of_order")
        previous_test_stop = fold.test_stop


def _prepare_primary_arrays(
    features: DataFrame,
    primary_target: pd.Series,
    label_end_indices: Sequence[int] | np.ndarray,
    folds: Sequence[OOFFold],
) -> tuple[DataFrame, np.ndarray, np.ndarray]:
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
    return finite_features, targets, label_ends


def _primary_training_data(
    finite_features: DataFrame,
    targets: np.ndarray,
    label_ends: np.ndarray,
    fold: OOFFold,
) -> tuple[np.ndarray, DataFrame, np.ndarray]:
    train_indices = np.arange(fold.train_start, fold.train_stop, dtype=int)
    train_x = (
        finite_features.iloc[train_indices]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    train_y = targets[train_indices]
    eligible = (
        (label_ends[train_indices] < fold.test_start)
        & (train_indices < fold.test_start)
        & np.isfinite(train_y)
        & np.isfinite(train_x).all(axis=1)
    )
    fit_indices = train_indices[eligible]
    fit_features = finite_features.iloc[fit_indices]
    fit_labels = targets[fit_indices]
    if len(fit_indices) < 2:
        raise ValueError("oof_fold_has_insufficient_purged_training_rows")
    if len(np.unique(fit_labels)) < 2:
        raise ValueError("oof_primary_training_target_has_one_class")
    return fit_indices, fit_features, fit_labels


def _fit_primary_estimator(
    estimator_factory: Callable[[], Any],
    fit_features: DataFrame,
    fit_labels: np.ndarray,
) -> Any:
    estimator = estimator_factory()
    estimator.fit(fit_features, fit_labels)
    return estimator


def _test_feature_window(
    finite_features: DataFrame,
    fold: OOFFold,
) -> tuple[np.ndarray, np.ndarray]:
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
    return test_indices, test_x


def _validate_primary_probabilities(
    probabilities: np.ndarray,
    row_count: int,
) -> None:
    if probabilities.ndim != 2 or probabilities.shape[0] != row_count:
        raise ValueError("oof_primary_probability_invalid")
    if not np.isfinite(probabilities).all():
        raise ValueError("oof_primary_probability_invalid")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("oof_primary_probability_invalid")


def _primary_confidence(estimator: Any, test_x: np.ndarray) -> np.ndarray:
    confidence = np.ones(len(test_x), dtype=float)
    predict_proba = getattr(estimator, "predict_proba", None)
    if not callable(predict_proba):
        return confidence
    probabilities = np.asarray(predict_proba(test_x), dtype=float)
    _validate_primary_probabilities(probabilities, len(test_x))
    classes = np.asarray(getattr(estimator, "classes_", []))
    if classes.size and len(classes) != probabilities.shape[1]:
        raise ValueError("oof_primary_class_schema_mismatch")
    return np.max(probabilities, axis=1)


def _validate_primary_predictions(
    estimator: Any,
    test_x: np.ndarray,
    test_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.asarray(estimator.predict(test_x), dtype=float).reshape(-1)
    if len(predictions) != test_count or not np.isfinite(predictions).all():
        raise ValueError("oof_primary_prediction_invalid")
    confidence = _primary_confidence(estimator, test_x)
    return predictions, confidence


def _fold_prediction_records(
    fold_id: int,
    fold: OOFFold,
    finite_features: DataFrame,
    targets: np.ndarray,
    label_ends: np.ndarray,
    estimator_factory: Callable[[], Any],
) -> list[dict[str, int | float]]:
    fit_indices, fit_features, fit_labels = _primary_training_data(
        finite_features, targets, label_ends, fold
    )
    max_label_end = int(np.max(label_ends[fit_indices]))
    if max_label_end >= fold.test_start:
        raise ValueError("oof_purge_contract_failed")
    estimator = _fit_primary_estimator(estimator_factory, fit_features, fit_labels)
    test_indices, test_x = _test_feature_window(finite_features, fold)
    predictions, confidence = _validate_primary_predictions(
        estimator, test_x, len(test_indices)
    )
    max_train_index = int(np.max(fit_indices))
    return [
        {
            "event_index": int(event_index),
            "fold_id": int(fold_id),
            "primary_prediction": float(prediction),
            "primary_confidence": float(score),
            "primary_train_max_index": max_train_index,
            "primary_train_max_label_end_index": max_label_end,
            "primary_train_count": int(len(fit_indices)),
        }
        for event_index, prediction, score in zip(
            test_indices, predictions, confidence, strict=True
        )
    ]


def generate_oof_primary_predictions(
    features: DataFrame,
    primary_target: pd.Series,
    label_end_indices: Sequence[int] | np.ndarray,
    folds: Sequence[OOFFold],
    estimator_factory: Callable[[], Any],
) -> DataFrame:
    """Generate primary signals only on held-out chronological fold rows.

    A training sample is retained only if its primary label ended strictly before
    the test fold begins. Each output row carries provenance for independent
    Meta-Label checks; in-sample predictions are never emitted.
    """
    finite_features, targets, label_ends = _prepare_primary_arrays(
        features, primary_target, label_end_indices, folds
    )
    records: list[dict[str, int | float]] = []
    for fold_id, fold in enumerate(folds):
        records.extend(
            _fold_prediction_records(
                fold_id,
                fold,
                finite_features,
                targets,
                label_ends,
                estimator_factory,
            )
        )
    result = DataFrame.from_records(records)
    if result.empty or result["event_index"].duplicated().any():
        raise ValueError("oof_predictions_empty_or_duplicate")
    return result


def _validate_meta_labels(
    labels: Sequence[int] | np.ndarray, row_count: int
) -> np.ndarray:
    label_values = np.asarray(labels)
    if label_values.shape != (row_count,) or not np.isin(label_values, [0, 1]).all():
        raise ValueError("meta_labels_must_be_binary_and_aligned")
    if len(np.unique(label_values)) != 2:
        raise ValueError("meta_training_requires_both_classes")
    return label_values


def _validate_provenance_schema(
    features: DataFrame,
    provenance: DataFrame,
) -> np.ndarray:
    if len(provenance) != len(features):
        raise ValueError("oof_provenance_missing_or_misaligned")
    if not set(OOF_PROVENANCE_COLUMNS).issubset(provenance.columns):
        raise ValueError("oof_provenance_missing_or_misaligned")
    event_indices = provenance["event_index"].to_numpy(dtype=int)
    if not np.array_equal(event_indices, features.index.to_numpy(dtype=int)):
        raise ValueError("oof_event_indices_do_not_match_meta_features")
    return event_indices


def _validate_primary_provenance_indices(
    event_indices: np.ndarray,
    primary_train_max: np.ndarray,
    primary_end_max: np.ndarray,
    fold_ids: np.ndarray,
) -> None:
    if np.any(fold_ids < 0):
        raise ValueError("meta_training_oof_or_purge_contract_failed")
    if np.any(primary_train_max >= event_indices):
        raise ValueError("meta_training_oof_or_purge_contract_failed")
    if np.any(primary_end_max >= event_indices):
        raise ValueError("meta_training_oof_or_purge_contract_failed")


def _validate_meta_label_maturity(
    event_indices: np.ndarray,
    meta_ends: np.ndarray,
    evaluation_cutoff_index: int,
) -> None:
    if meta_ends.shape != event_indices.shape:
        raise ValueError("meta_label_end_indices_misaligned")
    if np.any(meta_ends >= int(evaluation_cutoff_index)):
        raise ValueError("meta_training_oof_or_purge_contract_failed")
    if np.any(event_indices >= int(evaluation_cutoff_index)):
        raise ValueError("meta_training_oof_or_purge_contract_failed")


def _validate_meta_training_provenance(
    features: DataFrame,
    provenance: DataFrame,
    meta_label_end_indices: Sequence[int] | np.ndarray,
    evaluation_cutoff_index: int,
) -> None:
    event_indices = _validate_provenance_schema(features, provenance)
    fold_ids = provenance["fold_id"].to_numpy(dtype=int)
    primary_train_max = provenance["primary_train_max_index"].to_numpy(dtype=int)
    primary_end_max = provenance["primary_train_max_label_end_index"].to_numpy(
        dtype=int
    )
    meta_ends = np.asarray(meta_label_end_indices, dtype=int)
    _validate_primary_provenance_indices(
        event_indices, primary_train_max, primary_end_max, fold_ids
    )
    _validate_meta_label_maturity(event_indices, meta_ends, evaluation_cutoff_index)


def _validate_serialized_feature_names(payload: dict[str, Any]) -> list[str]:
    feature_names = payload.get("feature_names")
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("meta_model_feature_schema_invalid")
    if not all(isinstance(name, str) and name for name in feature_names):
        raise ValueError("meta_model_feature_schema_invalid")
    return feature_names


def _validate_serialized_parameters(
    payload: dict[str, Any], feature_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    means = np.asarray(payload["means"], dtype=float)
    scales = np.asarray(payload["scales"], dtype=float)
    coefficients = np.asarray(payload["coefficients"], dtype=float)
    intercept = float(payload["intercept"])
    expected_shape = (feature_count,)
    arrays_valid = all(
        array.shape == expected_shape and np.isfinite(array).all()
        for array in (means, scales, coefficients)
    )
    if not arrays_valid:
        raise ValueError("meta_model_serialized_parameters_invalid")
    if np.any(scales <= 0):
        raise ValueError("meta_model_serialized_parameters_invalid")
    if not math.isfinite(intercept):
        raise ValueError("meta_model_serialized_parameters_invalid")
    return means, scales, coefficients, intercept


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
        label_values = _validate_meta_labels(labels, len(features))
        _validate_meta_training_provenance(
            features,
            provenance,
            meta_label_end_indices,
            evaluation_cutoff_index,
        )
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
            (values - self.means_) / self.scales_
        ) @ self.coefficients_
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
        feature_names = _validate_serialized_feature_names(payload)
        model = cls(
            regularization_c=float(payload["regularization_c"]),
            max_iter=int(payload["max_iter"]),
        )
        means, scales, coefficients, intercept = _validate_serialized_parameters(
            payload, len(feature_names)
        )
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
