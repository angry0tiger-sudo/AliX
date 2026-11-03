"""V4-H classification-only LightGBM tuning and Triple-Barrier geometry experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pandas import DataFrame, Series

from scripts.run_ml_core_real_data_baseline import (
    DEFAULT_CALIBRATION_FRACTION,
    DEFAULT_MINIMUM_HOLD_PRECISION,
    DEFAULT_MINIMUM_HOLD_RECALL,
    LABELS,
    SYMBOLS,
    TIMEFRAME_MINUTES,
    _build_trained_experiment_result,
    _feature_profile_diagnostics,
    _insufficient_bar_result,
    _load_diagnostic_datasets,
    _skipped_result,
    classification_metrics,
    hold_class_metrics,
    label_audit_diagnostics,
    label_distribution,
    ordered_probability_matrix,
    predictions_from_probabilities,
    probability_diagnostics_by_temporal_segment,
    purged_temporal_indices,
    resample_ohlcv,
    sample_gate_diagnostics,
    threshold_optimization_diagnostics,
    validation_classification_diagnostics,
    weighting_policy_specs,
    _serialize_validation_probabilities,
)
from services.ml_core.adaptive_targets import apply_adaptive_triple_barrier
from services.ml_core.extended_features import (
    EXTENDED_FEATURE_SCHEMA_VERSION,
    build_feature_profiles,
    feature_profile_manifest,
)
from services.ml_core.hyperparameter_tuning import (
    DEFAULT_INNER_CV_SPLITS,
    DEFAULT_MINIMUM_CLASS_SAMPLES,
    DEFAULT_RANDOM_SEARCH_ITERATIONS,
    HYPERPARAMETER_DISTRIBUTIONS,
    build_purged_time_series_splits,
    tune_lightgbm_classifier,
)

V4H_REPORT_SCHEMA = "alix.ml-core.v4h-tuning-geometry.v1"
V4H_FEATURE_PROFILES = ("Baseline", "Baseline+Trend")
V4H_SCENARIO_CONTRACT = (
    ("V4H_Current", 12, 2.0, 1.0),
    ("V4H_Symmetric", 12, 1.5, 1.5),
    ("V4H_Tighter_SL", 12, 2.0, 1.5),
)
V4H_OUTER_VALIDATION_START_FRACTION = 0.70
V4H_OUTER_VALIDATION_END_FRACTION = 0.85
V4H_EXPECTED_EXPERIMENTS = 48
V4H_EXPECTED_MODELS = 96
V4H_INNER_CV_CLASS_SUPPORT_SKIP_STATUS = (
    "SKIPPED_INSUFFICIENT_INNER_CV_CLASS_SUPPORT"
)


def _v4h_scenarios() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "horizon_candles": horizon,
            "take_profit_multiplier": take_profit,
            "stop_loss_multiplier": stop_loss,
        }
        for name, horizon, take_profit, stop_loss in V4H_SCENARIO_CONTRACT
    ]


def _new_v4h_classifier(class_weight: str | None) -> Any:
    """Create a deterministic three-class LightGBM candidate with sufficient depth."""
    lightgbm = importlib.import_module("lightgbm")
    classifier = getattr(lightgbm, "LGBMClassifier")
    return classifier(
        objective="multiclass",
        n_estimators=80,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        max_depth=6,
        class_weight=class_weight,
        random_state=42,
        n_jobs=1,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )


def _v4h_sample_signature(
    train_indices: list[int], validation_indices: list[int]
) -> str:
    payload = json.dumps(
        {
            "train_indices": train_indices,
            "validation_indices": validation_indices,
        },
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _feature_importance_gain(model: Any, features: DataFrame) -> list[dict[str, Any]]:
    """Extract named LightGBM gain importance from the fitted outer-Train model."""
    booster = getattr(model, "booster_", None)
    if booster is None:
        raise ValueError("v4h_feature_importance_booster_missing")
    feature_names = [str(name) for name in booster.feature_name()]
    expected_names = [str(name) for name in features.columns]
    if feature_names != expected_names:
        raise ValueError("v4h_feature_importance_feature_names_mismatch")
    gains = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
    if (
        gains.ndim != 1
        or len(gains) != len(feature_names)
        or not bool(np.isfinite(gains).all())
        or bool(np.any(gains < 0))
    ):
        raise ValueError("v4h_feature_importance_gain_invalid")
    rows = [
        {"feature": name, "gain": round(float(gain), 8)}
        for name, gain in zip(feature_names, gains, strict=True)
    ]
    return sorted(rows, key=lambda item: (-float(item["gain"]), str(item["feature"])))


def _fit_v4h_variants(
    features: DataFrame,
    bars: DataFrame,
    labels: DataFrame,
    train_indices: list[int],
    validation_indices: list[int],
    train_y: Series,
    validation_y: Series,
    *,
    feature_profile: str,
    feature_count: int,
    horizon_candles: int,
    calibration_fraction: float,
    minimum_hold_precision: float,
    minimum_hold_recall: float,
    random_search_iterations: int,
    tuning_objective: str,
    economic_outcomes: Series | None = None,
    estimated_cost_bps: float = 90.0,
) -> dict[str, Any]:
    """Tune on outer Train, refit on all Train, and evaluate on untouched Validation."""
    variants: dict[str, Any] = {}
    x_train = features.iloc[train_indices]
    x_validation = features.iloc[validation_indices]
    label_end_indices = labels["label_end_idx"].tolist()

    for policy_name, policy in weighting_policy_specs().items():
        estimator = _new_v4h_classifier(policy["class_weight"])
        model, tuning = tune_lightgbm_classifier(
            estimator,
            x_train,
            train_y,
            train_indices,
            label_end_indices,
            horizon_candles=horizon_candles,
            class_weight_policy=policy_name,
            minimum_hold_recall=minimum_hold_recall,
            n_iter=random_search_iterations,
            n_splits=DEFAULT_INNER_CV_SPLITS,
            tuning_objective=tuning_objective,
            economic_outcomes=economic_outcomes,
            estimated_cost_bps=estimated_cost_bps,
        )
        train_probabilities = ordered_probability_matrix(model, x_train)
        validation_probabilities = ordered_probability_matrix(model, x_validation)
        train_prediction = predictions_from_probabilities(
            train_probabilities, train_y.index
        )
        validation_prediction = predictions_from_probabilities(
            validation_probabilities, validation_y.index
        )
        threshold_diagnostics = threshold_optimization_diagnostics(
            validation_y,
            validation_probabilities,
            calibration_fraction=calibration_fraction,
            minimum_precision=minimum_hold_precision,
            minimum_recall=minimum_hold_recall,
        )
        selected_parameters = dict(tuning["selected_parameters"])
        variants[policy_name] = {
            "model_parameters": {
                "class_weight": policy["class_weight"],
                "feature_profile": feature_profile,
                "feature_count": feature_count,
                "max_depth": 6,
                **selected_parameters,
            },
            "classification_metrics": {
                "train_in_sample": classification_metrics(train_y, train_prediction),
                "validation": classification_metrics(
                    validation_y, validation_prediction
                ),
            },
            **validation_classification_diagnostics(
                validation_y, validation_prediction
            ),
            "hold_metrics": hold_class_metrics(validation_y, validation_prediction),
            "feature_importance_gain": _feature_importance_gain(model, x_train),
            "probability_diagnostics": probability_diagnostics_by_temporal_segment(
                validation_y,
                validation_probabilities,
                calibration_fraction=calibration_fraction,
            ),
            "threshold_optimization": threshold_diagnostics,
            "validation_probabilities": _serialize_validation_probabilities(
                bars, validation_indices, validation_y, validation_probabilities
            ),
            "hyperparameter_tuning": tuning,
        }
    return variants


def _inner_cv_preflight_diagnostics(
    train_indices: list[int],
    train_labels: Series,
    labels: DataFrame,
    horizon_candles: int,
) -> dict[str, Any]:
    """Preflight the exact purged CV contract before fitting economic candidates."""
    try:
        _, fold_diagnostics = build_purged_time_series_splits(
            train_indices,
            train_labels,
            labels["label_end_idx"].tolist(),
            horizon_candles=horizon_candles,
            n_splits=DEFAULT_INNER_CV_SPLITS,
            minimum_class_samples=DEFAULT_MINIMUM_CLASS_SAMPLES,
        )
    except ValueError as error:
        if str(error) != "tuning_inner_cv_class_support_insufficient":
            raise
        return {
            "status": V4H_INNER_CV_CLASS_SUPPORT_SKIP_STATUS,
            "reason_code": str(error),
            "minimum_class_samples_per_class": DEFAULT_MINIMUM_CLASS_SAMPLES,
            "expected_fold_count": DEFAULT_INNER_CV_SPLITS,
            "complete_fold_validation": False,
        }
    return {
        "status": "PASS",
        "reason_code": None,
        "minimum_class_samples_per_class": DEFAULT_MINIMUM_CLASS_SAMPLES,
        "expected_fold_count": DEFAULT_INNER_CV_SPLITS,
        "validated_fold_count": len(fold_diagnostics),
        "complete_fold_validation": True,
    }


def _skipped_inner_cv_class_support_experiment(
    base_result: dict[str, Any],
    pair: str,
    timeframe: str,
    bars: DataFrame,
    labels: DataFrame,
    train_indices: list[int],
    validation_indices: list[int],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    """Persist an explicit non-trained experiment when purged CV support is inadequate."""
    status = V4H_INNER_CV_CLASS_SUPPORT_SKIP_STATUS
    sample_gate = {
        **base_result["sample_gate"],
        "outer_sample_gate_status": "PASS",
        "status": status,
        "inner_cv_preflight": preflight,
    }
    split = {
        **base_result["split"],
        "inner_cv_preflight": preflight,
    }
    base_result["sample_gate"] = sample_gate
    base_result["split"] = split
    skipped = _skipped_result(
        pair,
        timeframe,
        bars,
        labels,
        train_indices,
        validation_indices,
        split,
        status,
        sample_gate,
    )
    return {**skipped, **base_result, "status": status}


def run_v4h_experiment(
    pair: str,
    timeframe: str,
    bars: DataFrame,
    labels: DataFrame,
    feature_matrices: dict[str, DataFrame],
    *,
    scenario: dict[str, Any],
    feature_profile: str,
    calibration_fraction: float,
    minimum_hold_precision: float,
    minimum_hold_recall: float,
    random_search_iterations: int,
    tuning_objective: str = "HOLD_PRECISION",
    economic_outcomes: Series | None = None,
    estimated_cost_bps: float = 90.0,
    split_horizon_candles: int | None = None,
) -> dict[str, Any]:
    """Run one profile/geometry experiment using a purged outer split."""
    profile_metadata = _feature_profile_diagnostics(feature_profile)
    common_features = feature_matrices["Baseline+Trend"]
    features = feature_matrices[feature_profile]
    split_horizon = (
        int(split_horizon_candles)
        if split_horizon_candles is not None
        else int(scenario["horizon_candles"])
    )
    train_indices, validation_indices, split_info = purged_temporal_indices(
        common_features,
        labels,
        horizon_candles=split_horizon,
    )
    train_y = labels.iloc[train_indices]["label"]
    validation_y = labels.iloc[validation_indices]["label"]
    sample_gate = sample_gate_diagnostics(train_y, validation_y)
    base_result: dict[str, Any] = {
        "pair": pair,
        "timeframe": timeframe,
        "scenario": scenario["name"],
        "tuning_objective": tuning_objective,
        "triple_barrier": {
            "horizon_candles": int(scenario["horizon_candles"]),
            "take_profit_multiplier": float(scenario["take_profit_multiplier"]),
            "stop_loss_multiplier": float(scenario["stop_loss_multiplier"]),
        },
        "bar_count": int(len(bars)),
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(validation_indices)),
        "sample_signature_sha256": _v4h_sample_signature(
            train_indices, validation_indices
        ),
        **profile_metadata,
        "label_distribution": {
            "train": label_distribution(train_y),
            "validation": label_distribution(validation_y),
        },
        "label_audit": label_audit_diagnostics(labels),
        "sample_gate": sample_gate,
        "split": {
            **split_info,
            "method": "PURGED_CHRONOLOGICAL_OUTER_HOLDOUT",
            "inner_tuning_uses_outer_train_only": True,
        },
    }
    if sample_gate["status"] != "PASS":
        skipped = _skipped_result(
            pair,
            timeframe,
            bars,
            labels,
            train_indices,
            validation_indices,
            split_info,
            str(sample_gate["status"]),
            sample_gate,
        )
        return {**skipped, **base_result}

    if tuning_objective == "CUSTOM_ECONOMIC_SCORE":
        inner_cv_preflight = _inner_cv_preflight_diagnostics(
            train_indices, train_y, labels, split_horizon
        )
        base_result["split"]["inner_cv_preflight"] = inner_cv_preflight
        if inner_cv_preflight["status"] == V4H_INNER_CV_CLASS_SUPPORT_SKIP_STATUS:
            return _skipped_inner_cv_class_support_experiment(
                base_result,
                pair,
                timeframe,
                bars,
                labels,
                train_indices,
                validation_indices,
                inner_cv_preflight,
            )

    variants = _fit_v4h_variants(
        features,
        bars,
        labels,
        train_indices,
        validation_indices,
        train_y,
        validation_y,
        feature_profile=feature_profile,
        feature_count=int(profile_metadata["feature_count"]),
        horizon_candles=split_horizon,
        calibration_fraction=calibration_fraction,
        minimum_hold_precision=minimum_hold_precision,
        minimum_hold_recall=minimum_hold_recall,
        random_search_iterations=random_search_iterations,
        tuning_objective=tuning_objective,
        economic_outcomes=economic_outcomes,
        estimated_cost_bps=estimated_cost_bps,
    )
    result = _build_trained_experiment_result(
        base_result, train_y, validation_y, variants
    )
    result["overfitting_assessment"]["caveat"] = (
        "In-sample versus outer chronological Validation macro-F1; "
        "diagnostic heuristic, not proof."
    )
    return result


def _validate_v4h_random_search_iterations(random_search_iterations: int) -> None:
    valid = (
        not isinstance(random_search_iterations, bool)
        and isinstance(random_search_iterations, int)
        and 1 <= random_search_iterations <= 27
    )
    if not valid:
        raise ValueError("v4h_random_search_iterations_invalid")


def _insufficient_context_experiments(
    pair: str,
    timeframe: str,
    bars: DataFrame,
    scenarios: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    for scenario in scenarios:
        labels = apply_adaptive_triple_barrier(
            bars,
            horizon_candles=int(scenario["horizon_candles"]),
            take_profit_multiplier=float(scenario["take_profit_multiplier"]),
            stop_loss_multiplier=float(scenario["stop_loss_multiplier"]),
        )
        for profile in V4H_FEATURE_PROFILES:
            result = _insufficient_bar_result(pair, timeframe, bars, labels)
            result.update(
                {
                    "scenario": scenario["name"],
                    "triple_barrier": {
                        "horizon_candles": scenario["horizon_candles"],
                        "take_profit_multiplier": scenario["take_profit_multiplier"],
                        "stop_loss_multiplier": scenario["stop_loss_multiplier"],
                    },
                    **_feature_profile_diagnostics(profile),
                }
            )
            experiments.append(result)
    return experiments


def _run_v4h_scenario_experiments(
    pair: str,
    timeframe: str,
    bars: DataFrame,
    feature_matrices: dict[str, DataFrame],
    scenario: dict[str, Any],
    *,
    calibration_fraction: float,
    minimum_hold_precision: float,
    minimum_hold_recall: float,
    random_search_iterations: int,
) -> list[dict[str, Any]]:
    labels = apply_adaptive_triple_barrier(
        bars,
        horizon_candles=int(scenario["horizon_candles"]),
        take_profit_multiplier=float(scenario["take_profit_multiplier"]),
        stop_loss_multiplier=float(scenario["stop_loss_multiplier"]),
    )
    return [
        run_v4h_experiment(
            pair,
            timeframe,
            bars,
            labels,
            feature_matrices,
            scenario=scenario,
            feature_profile=profile,
            calibration_fraction=calibration_fraction,
            minimum_hold_precision=minimum_hold_precision,
            minimum_hold_recall=minimum_hold_recall,
            random_search_iterations=random_search_iterations,
        )
        for profile in V4H_FEATURE_PROFILES
    ]


def _run_v4h_pair_timeframe(
    symbol: str,
    timeframe: str,
    datasets: dict[str, DataFrame],
    scenarios: list[dict[str, Any]],
    *,
    calibration_fraction: float,
    minimum_hold_precision: float,
    minimum_hold_recall: float,
    random_search_iterations: int,
) -> list[dict[str, Any]]:
    pair = f"{symbol[:-4]}/USDT"
    bars = resample_ohlcv(datasets[symbol], timeframe)
    if len(bars) < 100:
        return _insufficient_context_experiments(pair, timeframe, bars, scenarios)
    feature_matrices = build_feature_profiles(bars, profile_names=V4H_FEATURE_PROFILES)
    experiments: list[dict[str, Any]] = []
    for scenario in scenarios:
        experiments.extend(
            _run_v4h_scenario_experiments(
                pair,
                timeframe,
                bars,
                feature_matrices,
                scenario,
                calibration_fraction=calibration_fraction,
                minimum_hold_precision=minimum_hold_precision,
                minimum_hold_recall=minimum_hold_recall,
                random_search_iterations=random_search_iterations,
            )
        )
    return experiments


def run_v4h_experiments(
    datasets: dict[str, DataFrame],
    *,
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION,
    minimum_hold_precision: float = DEFAULT_MINIMUM_HOLD_PRECISION,
    minimum_hold_recall: float = DEFAULT_MINIMUM_HOLD_RECALL,
    random_search_iterations: int = DEFAULT_RANDOM_SEARCH_ITERATIONS,
) -> list[dict[str, Any]]:
    """Run 2 profiles x 3 geometries x 8 pair/timeframe contexts."""
    _validate_v4h_random_search_iterations(random_search_iterations)
    scenarios = _v4h_scenarios()
    experiments: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAME_MINUTES:
            experiments.extend(
                _run_v4h_pair_timeframe(
                    symbol,
                    timeframe,
                    datasets,
                    scenarios,
                    calibration_fraction=calibration_fraction,
                    minimum_hold_precision=minimum_hold_precision,
                    minimum_hold_recall=minimum_hold_recall,
                    random_search_iterations=random_search_iterations,
                )
            )
    if len(experiments) != V4H_EXPECTED_EXPERIMENTS:
        raise RuntimeError("v4h_experiment_count_mismatch")
    return experiments


def _mean(values: list[float]) -> float | None:
    return round(float(np.mean(values)), 6) if values else None


def _forward_hold_metrics(
    experiment: dict[str, Any], policy: str
) -> tuple[float, float] | None:
    variant = (experiment.get("model_variants") or {}).get(policy)
    if not isinstance(variant, dict):
        return None
    forward = variant.get("threshold_optimization", {}).get(
        "post_calibration_validation"
    )
    if not isinstance(forward, dict):
        return None
    hold = forward.get("hold_metrics")
    if not isinstance(hold, dict):
        return None
    precision = hold.get("precision")
    recall = hold.get("recall")
    if precision is None or recall is None:
        return None
    return float(precision), float(recall)


def _variant_metric_record(
    experiment: dict[str, Any], policy: str
) -> dict[str, Any] | None:
    variant = (experiment.get("model_variants") or {}).get(policy)
    if not isinstance(variant, dict):
        return None
    forward = _forward_hold_metrics(experiment, policy)
    macro = (
        variant.get("classification_metrics", {}).get("validation", {}).get("macro", {})
    )
    tuning = variant.get("hyperparameter_tuning") or {}
    return {
        "scenario": experiment["scenario"],
        "pair": experiment["pair"],
        "timeframe": experiment["timeframe"],
        "feature_profile": experiment["feature_profile"],
        "weighting_policy": policy,
        "hold_precision": forward[0] if forward is not None else None,
        "hold_recall": forward[1] if forward is not None else None,
        "macro_f1": macro.get("f1") if isinstance(macro, dict) else None,
        "selected_parameters": tuning.get("selected_parameters"),
        "tuning_selection_status": tuning.get("selection_status"),
    }


def _aggregate_label_distribution(
    experiments: list[dict[str, Any]], scenario_name: str
) -> dict[str, Any]:
    counts = {label: 0 for label in LABELS}
    # One Baseline experiment per pair/timeframe avoids double-counting profiles.
    for experiment in experiments:
        if experiment.get("scenario") != scenario_name:
            continue
        if experiment.get("feature_profile") != "Baseline":
            continue
        distribution = experiment.get("label_distribution", {}).get("validation", {})
        classes = distribution.get("classes", {})
        for label in LABELS:
            counts[label] += int(classes.get(label, {}).get("count", 0))
    total = sum(counts.values())
    return {
        "scope": "POOLED_OUTER_VALIDATION_ROWS_FROM_EIGHT_PAIR_TIMEFRAME_CONTEXTS",
        "total": total,
        "classes": {
            label: {
                "count": counts[label],
                "percent": round(100.0 * counts[label] / total, 4) if total else 0.0,
            }
            for label in LABELS
        },
    }


def _append_experiment_candidates(
    grouped: dict[str, list[dict[str, Any]]],
    experiment: dict[str, Any],
) -> None:
    variants = experiment.get("model_variants") or {}
    for policy in weighting_policy_specs():
        variant = variants.get(policy)
        if not isinstance(variant, dict):
            continue
        tuning = variant.get("hyperparameter_tuning") or {}
        _append_candidate_results(grouped, tuning.get("candidate_results", []))


def _append_candidate_results(
    grouped: dict[str, list[dict[str, Any]]],
    candidates: list[dict[str, Any]],
) -> None:
    for candidate in candidates:
        key = json.dumps(candidate["parameters"], sort_keys=True, separators=(",", ":"))
        grouped[key].append(candidate)


def _candidate_summary_row(
    key: str, candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "parameters": json.loads(key),
        "model_context_count": len(candidates),
        "mean_inner_cv_hold_precision": _mean(
            [float(item["mean_cv_hold_precision"]) for item in candidates]
        ),
        "mean_inner_cv_hold_recall": _mean(
            [float(item["mean_cv_hold_recall"]) for item in candidates]
        ),
        "mean_inner_cv_macro_f1": _mean(
            [float(item["mean_cv_macro_f1"]) for item in candidates]
        ),
    }


def _candidate_precision_rank(item: dict[str, Any]) -> tuple[float, float]:
    return (
        item["mean_inner_cv_hold_precision"] or -1.0,
        item["mean_inner_cv_macro_f1"] or -1.0,
    )


def _candidate_macro_rank(item: dict[str, Any]) -> tuple[float, float]:
    return (
        item["mean_inner_cv_macro_f1"] or -1.0,
        item["mean_inner_cv_hold_precision"] or -1.0,
    )


def _aggregate_parameter_candidates(
    experiments: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for experiment in experiments:
        _append_experiment_candidates(grouped, experiment)
    rows = [
        _candidate_summary_row(key, candidates) for key, candidates in grouped.items()
    ]
    by_precision = sorted(rows, key=_candidate_precision_rank, reverse=True)
    by_macro = sorted(rows, key=_candidate_macro_rank, reverse=True)
    return {
        "candidate_combinations_evaluated": len(rows),
        "aggregation_scope": "MEAN_INNER_CV_SCORE_ACROSS_ALL_TRAINED_MODEL_CONTEXTS",
        "ranking_primary_metric": "HOLD_PRECISION",
        "ranking_tiebreaker": "MACRO_F1",
        "candidates_ranked_by_hold_precision": by_precision,
        "candidates_ranked_by_macro_f1": by_macro,
        "best_by_mean_inner_cv_hold_precision": by_precision[0]
        if by_precision
        else None,
        "best_by_mean_inner_cv_macro_f1": by_macro[0] if by_macro else None,
    }


def _append_selected_parameter_record(
    grouped: dict[str, list[dict[str, Any]]],
    record: dict[str, Any],
) -> None:
    parameters = record.get("selected_parameters")
    if isinstance(parameters, dict):
        key = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
        grouped[key].append(record)


def _selected_parameter_summary_row(
    key: str, records: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "parameters": json.loads(key),
        "selected_model_count": len(records),
        "mean_forward_hold_precision": _mean(
            [
                float(item["hold_precision"])
                for item in records
                if item["hold_precision"] is not None
            ]
        ),
        "mean_forward_hold_recall": _mean(
            [
                float(item["hold_recall"])
                for item in records
                if item["hold_recall"] is not None
            ]
        ),
        "mean_outer_validation_macro_f1": _mean(
            [
                float(item["macro_f1"])
                for item in records
                if item["macro_f1"] is not None
            ]
        ),
    }


def _selected_parameter_rank(item: dict[str, Any]) -> tuple[float, float]:
    return (
        item["mean_forward_hold_precision"] or -1.0,
        item["mean_outer_validation_macro_f1"] or -1.0,
    )


def _aggregate_selected_parameters(
    metric_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in metric_records:
        _append_selected_parameter_record(grouped, record)
    rows = [
        _selected_parameter_summary_row(key, records)
        for key, records in grouped.items()
    ]
    return sorted(rows, key=_selected_parameter_rank, reverse=True)


def _geometry_records(
    metric_records: list[dict[str, Any]], scenario_name: str
) -> list[dict[str, Any]]:
    return [item for item in metric_records if item["scenario"] == scenario_name]


def _geometry_metric_values(
    records: list[dict[str, Any]], metric_name: str
) -> list[float]:
    return [
        float(item[metric_name]) for item in records if item[metric_name] is not None
    ]


def _geometry_acceptance_count(records: list[dict[str, Any]]) -> int:
    return sum(
        item["hold_precision"] is not None
        and item["hold_recall"] is not None
        and float(item["hold_precision"]) >= DEFAULT_MINIMUM_HOLD_PRECISION
        and float(item["hold_recall"]) >= DEFAULT_MINIMUM_HOLD_RECALL
        for item in records
    )


def _best_geometry_record(
    records: list[dict[str, Any]], metric_name: str
) -> dict[str, Any] | None:
    available = [item for item in records if item[metric_name] is not None]
    if not available:
        return None
    return max(available, key=lambda item: float(item[metric_name]))


def _geometry_experiment_count(
    experiments: list[dict[str, Any]], scenario_name: str
) -> int:
    return sum(item.get("scenario") == scenario_name for item in experiments)


def _geometry_summary_row(
    experiments: list[dict[str, Any]],
    metric_records: list[dict[str, Any]],
    scenario: tuple[str, int, float, float],
) -> dict[str, Any]:
    scenario_name, horizon, take_profit, stop_loss = scenario
    records = _geometry_records(metric_records, scenario_name)
    precisions = _geometry_metric_values(records, "hold_precision")
    recalls = _geometry_metric_values(records, "hold_recall")
    macro_scores = _geometry_metric_values(records, "macro_f1")
    return {
        "scenario": scenario_name,
        "horizon_candles": horizon,
        "take_profit_multiplier": take_profit,
        "stop_loss_multiplier": stop_loss,
        "experiment_count": _geometry_experiment_count(experiments, scenario_name),
        "trained_model_count": len(records),
        "label_distribution": _aggregate_label_distribution(experiments, scenario_name),
        "mean_forward_hold_precision": _mean(precisions),
        "mean_forward_hold_recall": _mean(recalls),
        "mean_outer_validation_macro_f1": _mean(macro_scores),
        "strict_acceptance_count": _geometry_acceptance_count(records),
        "best_individual_forward_hold_precision_result": _best_geometry_record(
            records, "hold_precision"
        ),
        "best_individual_outer_validation_macro_f1_result": _best_geometry_record(
            records, "macro_f1"
        ),
    }


def _build_geometry_summary(
    experiments: list[dict[str, Any]],
    metric_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        _geometry_summary_row(experiments, metric_records, scenario)
        for scenario in V4H_SCENARIO_CONTRACT
    ]


def _append_experiment_metric_records(
    records: list[dict[str, Any]], experiment: dict[str, Any]
) -> None:
    for policy in weighting_policy_specs():
        record = _variant_metric_record(experiment, policy)
        if record is not None:
            records.append(record)


def _collect_metric_records(
    experiments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for experiment in experiments:
        _append_experiment_metric_records(records, experiment)
    return records


def _training_counts(experiments: list[dict[str, Any]]) -> tuple[int, int]:
    trained_experiments = 0
    trained_models = 0
    for experiment in experiments:
        trained_experiments += int(experiment.get("status") == "TRAINED")
        trained_models += len(experiment.get("model_variants") or {})
    return trained_experiments, trained_models


def _candidate_fit_total(experiments: list[dict[str, Any]]) -> int:
    fit_count = 0
    for experiment in experiments:
        variants = experiment.get("model_variants") or {}
        for variant in variants.values():
            tuning = variant.get("hyperparameter_tuning", {})
            fit_count += int(tuning.get("candidate_fit_count", 0))
    return fit_count


def _build_report_feature_engineering() -> dict[str, Any]:
    return {
        "enabled": True,
        "schema_version": EXTENDED_FEATURE_SCHEMA_VERSION,
        "selected_profiles": list(V4H_FEATURE_PROFILES),
        "profile_manifest": feature_profile_manifest(V4H_FEATURE_PROFILES),
        "common_sample_policy": "FINITE_ROW_INTERSECTION_OF_BASELINE_PLUS_TREND",
        "point_in_time": True,
    }


def _build_report_model() -> dict[str, Any]:
    return {
        "family": "LightGBM LGBMClassifier",
        "objective": "multiclass",
        "n_estimators": 80,
        "max_depth": 6,
        "random_state": 42,
        "deterministic": True,
        "class_weight_policies": list(weighting_policy_specs()),
        "binary_only_parameters_excluded": ["scale_pos_weight", "is_unbalance"],
    }


def _build_report_hyperparameter_tuning(
    experiments: list[dict[str, Any]],
    metric_records: list[dict[str, Any]],
    random_search_iterations: int,
    trained_models: int,
) -> dict[str, Any]:
    return {
        "method": "RANDOM_SEARCH",
        "parameter_distributions": {
            name: list(values) for name, values in HYPERPARAMETER_DISTRIBUTIONS.items()
        },
        "search_space_size": 27,
        "candidate_count_per_model": random_search_iterations,
        "inner_cv_method": "PURGED_EXPANDING_TIME_SERIES",
        "inner_cv_fold_count": DEFAULT_INNER_CV_SPLITS,
        "inner_cv_scope": "OUTER_TRAIN_ONLY",
        "validation_accessed_during_search": False,
        "final_refit_scope": "FULL_OUTER_TRAIN_ONLY",
        "selected_candidate_scores_are_inner_cv_only": True,
        "model_count": trained_models,
        "candidate_fit_count": _candidate_fit_total(experiments),
        "final_refit_count": trained_models,
        "candidate_summary": _aggregate_parameter_candidates(experiments),
        "selected_parameter_summary": _aggregate_selected_parameters(metric_records),
    }


def _build_report_validation_design(calibration_fraction: float) -> dict[str, Any]:
    return {
        "method": "SINGLE_PURGED_CHRONOLOGICAL_OUTER_HOLDOUT",
        "validation_start_fraction": V4H_OUTER_VALIDATION_START_FRACTION,
        "validation_end_fraction": V4H_OUTER_VALIDATION_END_FRACTION,
        "calibration_fraction_of_outer_validation": calibration_fraction,
        "calibration_is_excluded_from_model_fit": True,
        "inner_search_uses_train_only": True,
        "pre_validation_embargo": "EXACT_LABEL_HORIZON_CANDLES",
        "label_window_purge": "ACTUAL_LABEL_END_INDEX_OVERLAP",
        "forward_hold_metric_scope": "POST_CALIBRATION_VALIDATION",
        "macro_f1_scope": "FULL_OUTER_VALIDATION_DESCRIPTIVE",
    }


def _build_report_safety() -> dict[str, Any]:
    return {
        "public_api_only": True,
        "credentials_used": False,
        "private_api_calls": 0,
        "real_order_attempts": 0,
        "orders_or_execution_api_used": False,
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
    }


def build_v4h_report(
    *,
    commit_sha: str,
    days: int,
    provenance: list[dict[str, Any]],
    experiments: list[dict[str, Any]],
    random_search_iterations: int,
    calibration_fraction: float,
) -> dict[str, Any]:
    metric_records = _collect_metric_records(experiments)
    candidate_summary = _aggregate_parameter_candidates(experiments)
    geometry_summary = _build_geometry_summary(experiments, metric_records)
    trained_experiments, trained_models = _training_counts(experiments)
    candidate_fit_count = _candidate_fit_total(experiments)
    acceptance_count = sum(item["strict_acceptance_count"] for item in geometry_summary)
    tuning_report = _build_report_hyperparameter_tuning(
        experiments,
        metric_records,
        random_search_iterations,
        trained_models,
    )
    tuning_report["candidate_fit_count"] = candidate_fit_count
    tuning_report["candidate_summary"] = candidate_summary
    return {
        "schema_version": V4H_REPORT_SCHEMA,
        "commit_sha": commit_sha,
        "diagnostic_scope": "CLASSIFICATION_ONLY",
        "training_scope": "DIAGNOSTIC_CLASSIFICATION_ONLY",
        "economic_evaluation_performed": False,
        "economic_metrics_reported": False,
        "requested_days": days,
        "provider": "Binance Spot public klines API",
        "input_datasets": provenance,
        "allowed_timeframes": list(TIMEFRAME_MINUTES),
        "symbols": list(SYMBOLS),
        "labels": list(LABELS),
        "triple_barrier_scenarios": _v4h_scenarios(),
        "feature_engineering": _build_report_feature_engineering(),
        "model": _build_report_model(),
        "hyperparameter_tuning": tuning_report,
        "validation_design": _build_report_validation_design(calibration_fraction),
        "acceptance_policy": {
            "minimum_hold_precision": DEFAULT_MINIMUM_HOLD_PRECISION,
            "minimum_hold_recall": DEFAULT_MINIMUM_HOLD_RECALL,
            "strict_acceptance_count": acceptance_count,
            "status": "MET" if acceptance_count else "NOT_MET",
        },
        "experiment_count": len(experiments),
        "trained_experiments": trained_experiments,
        "trained_models": trained_models,
        "expected_experiments": V4H_EXPECTED_EXPERIMENTS,
        "expected_models": V4H_EXPECTED_MODELS,
        "experiments": experiments,
        "geometry_summary": geometry_summary,
        "safety": _build_report_safety(),
    }


def render_v4h_markdown(report: dict[str, Any]) -> str:
    lines = [
        "## AliX V4-H Tuning & Target Geometry",
        "",
        f"Commit SHA: {report['commit_sha']}",
        "",
        "Scope: classification only. No economic evaluation was performed.",
        "",
        "| Geometry | Validation BUY | Validation SELL | Validation HOLD | Mean forward HOLD Precision | Mean forward HOLD Recall | Mean outer Validation Macro-F1 | Strict passes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["geometry_summary"]:
        distribution = item["label_distribution"]["classes"]
        lines.append(
            f"| {item['scenario']} | {distribution['BUY']['percent']:.4f}% "
            f"({distribution['BUY']['count']}) | "
            f"{distribution['SELL']['percent']:.4f}% "
            f"({distribution['SELL']['count']}) | "
            f"{distribution['HOLD']['percent']:.4f}% "
            f"({distribution['HOLD']['count']}) | "
            f"{item['mean_forward_hold_precision']} | "
            f"{item['mean_forward_hold_recall']} | "
            f"{item['mean_outer_validation_macro_f1']} | "
            f"{item['strict_acceptance_count']} |"
        )
    best_precision = report["hyperparameter_tuning"]["candidate_summary"][
        "best_by_mean_inner_cv_hold_precision"
    ]
    best_macro = report["hyperparameter_tuning"]["candidate_summary"][
        "best_by_mean_inner_cv_macro_f1"
    ]
    lines.extend(
        [
            "",
            "### Best inner-CV hyperparameter combinations",
            "",
            f"- HOLD Precision: {best_precision}",
            f"- Macro-F1: {best_macro}",
            "",
            f"Experiments: {report['experiment_count']}; trained models: {report['trained_models']}.",
            f"Strict HOLD acceptance: {report['acceptance_policy']['status']}.",
            "",
            "Safety: Champion=null, order_permission=false, trading_authorization=NO-GO, "
            "private_api_calls=0, real_order_attempts=0.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_v4h_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as summary:
            summary.write(render_v4h_markdown(report))


def run_v4h_diagnostic(
    input_dir: Path,
    output_path: Path,
    *,
    days: int,
    commit_sha: str,
    random_search_iterations: int = DEFAULT_RANDOM_SEARCH_ITERATIONS,
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION,
) -> dict[str, Any]:
    if days not in (1, 7, 30, 90, 730):
        raise ValueError("diagnostic_days_out_of_policy")
    datasets, provenance = _load_diagnostic_datasets(input_dir, days, commit_sha)
    experiments = run_v4h_experiments(
        datasets,
        calibration_fraction=calibration_fraction,
        random_search_iterations=random_search_iterations,
    )
    report = build_v4h_report(
        commit_sha=commit_sha,
        days=days,
        provenance=provenance,
        experiments=experiments,
        random_search_iterations=random_search_iterations,
        calibration_fraction=calibration_fraction,
    )
    if report["experiment_count"] != V4H_EXPECTED_EXPERIMENTS:
        raise RuntimeError("v4h_report_experiment_count_mismatch")
    if report["trained_models"] != V4H_EXPECTED_MODELS:
        raise RuntimeError("v4h_report_model_count_mismatch")
    write_v4h_report(report, output_path)
    print(
        json.dumps(
            {
                "report_path": str(output_path),
                "commit_sha": commit_sha,
                "experiment_count": report["experiment_count"],
                "trained_experiments": report["trained_experiments"],
                "trained_models": report["trained_models"],
                "candidate_fit_count": report["hyperparameter_tuning"][
                    "candidate_fit_count"
                ],
                "economic_evaluation_performed": False,
                "Champion": None,
                "order_permission": False,
                "trading_authorization": "NO-GO",
            },
            sort_keys=True,
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--days", type=int, choices=(1, 7, 30, 90, 730), default=730)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument(
        "--random-search-iterations",
        type=int,
        default=DEFAULT_RANDOM_SEARCH_ITERATIONS,
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=DEFAULT_CALIBRATION_FRACTION,
    )
    args = parser.parse_args()
    run_v4h_diagnostic(
        args.input_dir,
        args.output,
        days=args.days,
        commit_sha=args.commit_sha,
        random_search_iterations=args.random_search_iterations,
        calibration_fraction=args.calibration_fraction,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
