"""LightGBM/XGBoost stacking with a filtered HMM and logistic meta learner."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from research.ai_ensemble.features import REGIME_OBSERVATION_COLUMNS
from research.ai_ensemble.hmm import CausalGaussianHMM
from research.ai_ensemble.validation import (
    create_purged_walk_forward_folds,
    mean_return_z_score,
    non_overlapping_trade_returns,
)


@dataclass(frozen=True, slots=True)
class StackingConfig:
    fold_count: int = 3
    embargo_bars: int = 24
    minimum_train_samples: int = 120
    n_estimators: int = 160
    random_state: int = 17
    meta_regularization_c: float = 1.0
    probability_threshold: float = 0.50

    def validate(self) -> None:
        if (
            self.fold_count < 2
            or self.embargo_bars < 1
            or self.minimum_train_samples < 2
        ):
            raise ValueError("stacking_split_configuration_invalid")
        if self.n_estimators < 10 or self.random_state < 0:
            raise ValueError("stacking_estimator_configuration_invalid")
        if (
            not np.isfinite(self.meta_regularization_c)
            or self.meta_regularization_c <= 0
        ):
            raise ValueError("stacking_meta_regularization_invalid")
        if not 0.0 <= self.probability_threshold <= 1.0:
            raise ValueError("stacking_probability_threshold_invalid")


def _binary_labels(values: Sequence[int], row_count: int) -> np.ndarray:
    labels = np.asarray(values, dtype=int)
    if labels.shape != (row_count,) or not bool(np.isin(labels, [0, 1]).all()):
        raise ValueError("stacking_labels_must_be_binary_and_row_aligned")
    return labels


def _model_probability(model: Any, features: DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    classes = np.asarray(model.classes_)
    locations = np.flatnonzero(classes == 1)
    if probabilities.ndim != 2 or len(locations) != 1:
        raise ValueError("base_model_positive_class_probability_missing")
    result = probabilities[:, int(locations[0])]
    if len(result) != len(features) or not bool(np.isfinite(result).all()):
        raise ValueError("base_model_probability_output_invalid")
    return result


def _new_base_models(config: StackingConfig) -> dict[str, Any]:
    try:
        lightgbm = importlib.import_module("lightgbm")
        xgboost = importlib.import_module("xgboost")
    except ImportError as exc:
        raise RuntimeError("lightgbm_and_xgboost_are_required_for_training") from exc
    lightgbm_model = lightgbm.LGBMClassifier(
        n_estimators=config.n_estimators,
        learning_rate=0.04,
        num_leaves=15,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=config.random_state,
        n_jobs=1,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )
    xgboost_model = xgboost.XGBClassifier(
        n_estimators=config.n_estimators,
        learning_rate=0.04,
        max_depth=4,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=config.random_state,
        n_jobs=1,
        verbosity=0,
    )
    return {"lightgbm": lightgbm_model, "xgboost": xgboost_model}


def _fit_base_models(
    features: DataFrame, labels: np.ndarray, config: StackingConfig
) -> dict[str, Any]:
    if len(np.unique(labels)) != 2:
        raise ValueError("base_model_training_requires_both_target_classes")
    models = _new_base_models(config)
    for model in models.values():
        model.fit(features, labels)
    return models


def _meta_features(
    models: dict[str, Any],
    features: DataFrame,
    regime_probabilities: np.ndarray,
) -> np.ndarray:
    base_probabilities = [
        _model_probability(models[name], features) for name in ("lightgbm", "xgboost")
    ]
    if regime_probabilities.shape != (len(features), 3):
        raise ValueError("hmm_regime_probability_shape_invalid")
    result = np.column_stack([*base_probabilities, regime_probabilities])
    if not bool(np.isfinite(result).all()):
        raise ValueError("stacking_meta_features_not_finite")
    return result


class StackingEnsemble:
    """Two heterogeneous tree learners plus a three-state causal HMM and logistic stack."""

    def __init__(self, config: StackingConfig | None = None) -> None:
        self.config = config or StackingConfig()
        self.config.validate()
        self.feature_columns_: tuple[str, ...] | None = None
        self.base_models_: dict[str, Any] = {}
        self.regime_model_: CausalGaussianHMM | None = None
        self.meta_model_: LogisticRegression | None = None

    @staticmethod
    def _validated_features(features: DataFrame) -> DataFrame:
        if features.empty or features.columns.duplicated().any():
            raise ValueError("stacking_features_empty_or_duplicate_columns")
        result = features.copy()
        if not bool(np.isfinite(result.to_numpy(dtype=float)).all()):
            raise ValueError("stacking_features_must_be_finite")
        missing = sorted(set(REGIME_OBSERVATION_COLUMNS) - set(result.columns))
        if missing:
            raise ValueError(f"stacking_regime_features_missing:{','.join(missing)}")
        return result

    def fit(
        self,
        features: DataFrame,
        labels: Sequence[int],
        label_end_indices: Sequence[int],
        training_indices: Sequence[int],
    ) -> StackingEnsemble:
        data = self._validated_features(features)
        target = _binary_labels(labels, len(data))
        if len(label_end_indices) != len(data):
            raise ValueError("stacking_label_end_indices_row_mismatch")
        if len(training_indices) == 0:
            raise ValueError("stacking_training_indices_empty")
        selected = tuple(int(index) for index in training_indices)
        if (
            selected != tuple(sorted(set(selected)))
            or selected[0] < 0
            or selected[-1] >= len(data)
        ):
            raise ValueError("stacking_training_indices_invalid")
        self.feature_columns_ = tuple(data.columns)
        inner_folds = create_purged_walk_forward_folds(
            selected,
            label_end_indices,
            fold_count=self.config.fold_count,
            first_validation_fraction=0.60,
            embargo_bars=self.config.embargo_bars,
            minimum_train_samples=self.config.minimum_train_samples,
        )
        meta_rows: list[np.ndarray] = []
        meta_targets: list[int] = []
        for fold in inner_folds:
            train_idx = list(fold.train_indices)
            validation_idx = list(fold.validation_indices)
            train_x = data.iloc[train_idx]
            validation_x = data.iloc[validation_idx]
            train_y = target[train_idx]
            base_models = _fit_base_models(train_x, train_y, self.config)
            regime_model = CausalGaussianHMM().fit(
                train_x.loc[:, list(REGIME_OBSERVATION_COLUMNS)].to_numpy(dtype=float)
            )
            regime_validation = regime_model.filter(
                validation_x.loc[:, list(REGIME_OBSERVATION_COLUMNS)].to_numpy(
                    dtype=float
                ),
                initial_state_probabilities=regime_model.next_prior_probabilities(),
            )
            meta_rows.append(
                _meta_features(base_models, validation_x, regime_validation)
            )
            meta_targets.extend(target[validation_idx].tolist())
        oof_features = np.vstack(meta_rows)
        oof_labels = np.asarray(meta_targets, dtype=int)
        if len(np.unique(oof_labels)) != 2:
            raise ValueError("meta_learner_oof_requires_both_target_classes")
        self.meta_model_ = LogisticRegression(
            C=self.config.meta_regularization_c,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            random_state=self.config.random_state,
        )
        self.meta_model_.fit(oof_features, oof_labels)
        full_x = data.iloc[list(selected)]
        full_y = target[list(selected)]
        self.base_models_ = _fit_base_models(full_x, full_y, self.config)
        self.regime_model_ = CausalGaussianHMM().fit(
            full_x.loc[:, list(REGIME_OBSERVATION_COLUMNS)].to_numpy(dtype=float)
        )
        return self

    def predict_proba(self, features: DataFrame) -> np.ndarray:
        if (
            self.feature_columns_ is None
            or self.meta_model_ is None
            or self.regime_model_ is None
            or set(self.base_models_) != {"lightgbm", "xgboost"}
        ):
            raise RuntimeError("stacking_ensemble_not_fitted")
        data = self._validated_features(features)
        if tuple(data.columns) != self.feature_columns_:
            raise ValueError("stacking_prediction_feature_contract_mismatch")
        regime = self.regime_model_.filter(
            data.loc[:, list(REGIME_OBSERVATION_COLUMNS)].to_numpy(dtype=float),
            initial_state_probabilities=self.regime_model_.next_prior_probabilities(),
        )
        meta = _meta_features(self.base_models_, data, regime)
        raw = self.meta_model_.predict_proba(meta)
        classes = np.asarray(self.meta_model_.classes_)
        positive = np.flatnonzero(classes == 1)
        if len(positive) != 1:
            raise ValueError("meta_learner_positive_class_probability_missing")
        result = np.asarray(raw[:, int(positive[0])], dtype=float)
        if len(result) != len(data) or not bool(np.isfinite(result).all()):
            raise ValueError("meta_learner_probability_output_invalid")
        return result


def _fold_trade_metrics(returns: Sequence[float]) -> dict[str, object]:
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or not bool(np.isfinite(values).all()):
        raise ValueError("trade_metric_returns_must_be_finite_1d")
    if bool((values <= -1.0).any()):
        raise ValueError("trade_metric_return_would_destroy_equity")
    if len(values) == 0:
        return {
            "trade_count": 0,
            "win_rate": None,
            "mean_net_return": None,
            "compounded_net_return": 0.0,
            "profit_factor": None,
            "max_drawdown": None,
            "trade_sharpe": None,
            "statistical_z_score": None,
        }
    positive = float(values[values > 0].sum())
    negative = float(-values[values < 0].sum())
    profit_factor = positive / negative if negative > 0 else None
    equity = np.cumprod(1.0 + values)
    peaks = np.maximum.accumulate(np.concatenate(([1.0], equity)))[1:]
    drawdowns = 1.0 - equity / peaks
    max_drawdown = float(max(0.0, np.max(drawdowns)))
    compounded = float(equity[-1] - 1.0)
    deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    sharpe = float(values.mean() / deviation) if deviation > 0 else None
    return {
        "trade_count": len(values),
        "win_rate": float((values > 0).mean()),
        "mean_net_return": float(values.mean()),
        "compounded_net_return": compounded,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "trade_sharpe": sharpe,
        "statistical_z_score": mean_return_z_score(values.tolist()),
    }


def evaluate_walk_forward(
    features: DataFrame,
    labels: Sequence[int],
    label_end_indices: Sequence[int],
    net_returns: Sequence[float],
    eligible_indices: Sequence[int],
    *,
    config: StackingConfig | None = None,
) -> dict[str, object]:
    """Nested purged walk-forward evaluation; each outer fold owns its stack."""
    active_config = config or StackingConfig()
    active_config.validate()
    if len(net_returns) != len(features):
        raise ValueError("walk_forward_net_returns_row_mismatch")
    folds = create_purged_walk_forward_folds(
        eligible_indices,
        label_end_indices,
        fold_count=active_config.fold_count,
        first_validation_fraction=0.60,
        embargo_bars=active_config.embargo_bars,
        minimum_train_samples=active_config.minimum_train_samples,
    )
    target = _binary_labels(labels, len(features))
    net = np.asarray(net_returns, dtype=float)
    if not bool(np.isfinite(net).all()):
        raise ValueError("walk_forward_net_returns_must_be_finite")
    all_trade_returns: list[float] = []
    fold_reports: list[dict[str, object]] = []
    oos_rows = 0
    for fold in folds:
        model = StackingEnsemble(active_config).fit(
            features,
            target.tolist(),
            label_end_indices,
            fold.train_indices,
        )
        validation_indices = list(fold.validation_indices)
        probabilities = model.predict_proba(features.iloc[validation_indices])
        realized = non_overlapping_trade_returns(
            validation_indices,
            probabilities.tolist(),
            net.tolist(),
            label_end_indices,
            probability_threshold=active_config.probability_threshold,
        )
        all_trade_returns.extend(realized)
        oos_rows += len(validation_indices)
        fold_report: dict[str, object] = {
            "fold_index": fold.fold_index,
            "training_rows": len(fold.train_indices),
            "validation_rows": len(validation_indices),
            "purged_training_rows": fold.purged_training_rows,
            "embargo_bars": fold.embargo_bars,
        }
        fold_report.update(_fold_trade_metrics(realized))
        fold_reports.append(fold_report)
    aggregate = _fold_trade_metrics(all_trade_returns)
    return {
        "fold_count": len(fold_reports),
        "out_of_sample_rows": oos_rows,
        "folds": fold_reports,
        "aggregate": aggregate,
        "validation_method": "NESTED_PURGED_EXPANDING_WALK_FORWARD",
        "overlapping_trade_policy": "ONE_OPEN_POSITION; SKIP_UNTIL_LABEL_END",
        "probability_threshold": active_config.probability_threshold,
    }
