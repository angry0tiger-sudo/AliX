"""Offline V4-C label-transition audit; this module never trains a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from pandas import DataFrame

from services.ml_core.adaptive_targets import (
    EXIT_REASONS,
    UNLABELED_REASONS,
    apply_adaptive_triple_barrier,
)
from services.ml_core.causal_features import build_causal_feature_matrix

REPORT_SCHEMA = "alix.ml-core.real-data-classification-diagnostic.v5"
LEGACY_REPORT_SCHEMA = "alix.ml-core.real-data-classification-diagnostic.v4"
AUDIT_SCHEMA = "alix.ml-core.label-transition-audit.v1"
SOURCE_COMMIT = "ac428dd94383173570fe3cf8af510e5b0941336e"
LEGACY_SOURCE_COMMIT = "5f874e6f56436a9b79bbc8adbfec1cf4fd31e4b9"
SOURCE_RUN_NUMBER = 1433
SOURCE_RUN_ID = 41052324389
SOURCE_ARTIFACT_ID = 12978226236
SYMBOLS = ("BTCUSDT", "ETHUSDT")
PAIR_NAMES = {"BTCUSDT": "BTC/USDT", "ETHUSDT": "ETH/USDT"}
TIMEFRAMES = {"15m": 15, "30m": 30, "1h": 60, "4h": 240}
SCENARIOS = {
    "Baseline": (24, 2.0, 1.0),
    "Short_Horizon": (12, 2.0, 1.0),
}
LABELS = ("BUY", "SELL", "HOLD")
MINUTE_COLUMNS = ("date", "open", "high", "low", "close", "volume")
VALIDATION_START_FRACTION = 0.70
VALIDATION_END_FRACTION = 0.85
WEIGHTING_POLICIES = ("weight_none", "weight_balanced")
SAFE_VALUES = {
    "Champion": None,
    "credentials_used": False,
    "order_permission": False,
    "orders_or_execution_api_used": False,
    "private_api_calls": 0,
    "public_api_only": True,
    "real_order_attempts": 0,
    "trading_authorization": "NO-GO",
}


def _mapping(value: Any, error: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(error)
    return cast(dict[str, Any], value)


def _integer(value: Any, error: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(error)
    return cast(int, value)


def _number(value: Any, error: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(error)
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(error)
    return result


def _validate_report_identity(report: dict[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError("v4c_report_schema_mismatch")
    if report.get("commit_sha") != SOURCE_COMMIT:
        raise ValueError("v4c_report_commit_mismatch")
    if report.get("experiment_count") != 16:
        raise ValueError("v4c_experiment_count_mismatch")


def _validate_report_window(report: dict[str, Any]) -> None:
    window = _mapping(
        report.get("data_window_selection"), "data_window_metadata_missing"
    )
    selected_days = _integer(window.get("selected_days"), "selected_days_invalid")
    if selected_days < 1 or selected_days != report.get("requested_days"):
        raise ValueError("v4c_data_window_mismatch")


def _validate_report_safety(report: dict[str, Any]) -> None:
    safety = _mapping(report.get("safety"), "safety_metadata_missing")
    if any(safety.get(key) != value for key, value in SAFE_VALUES.items()):
        raise ValueError("v4c_safety_invariant_violation")


def _parse_scenario(raw: Any) -> tuple[str, tuple[int, float, float]]:
    scenario = _mapping(raw, "v4c_scenario_metadata_invalid")
    name = str(scenario.get("name"))
    if name not in SCENARIOS:
        raise ValueError("v4c_scenario_out_of_policy")
    geometry = (
        _integer(scenario.get("horizon_candles"), "v4c_horizon_invalid"),
        _number(
            scenario.get("take_profit_multiplier"),
            "v4c_take_profit_multiplier_invalid",
        ),
        _number(
            scenario.get("stop_loss_multiplier"),
            "v4c_stop_loss_multiplier_invalid",
        ),
    )
    return name, geometry


def _validate_report_scenarios(report: dict[str, Any]) -> None:
    rows = report.get("triple_barrier_scenarios")
    if not isinstance(rows, list):
        raise ValueError("v4c_scenario_metadata_missing")
    observed = dict(_parse_scenario(row) for row in rows)
    if observed != SCENARIOS:
        raise ValueError("v4c_scenario_geometry_mismatch")


def _validate_legacy_report_identity(report: dict[str, Any]) -> None:
    if report.get("schema_version") != LEGACY_REPORT_SCHEMA:
        raise ValueError("legacy_report_schema_mismatch")
    if report.get("commit_sha") != LEGACY_SOURCE_COMMIT:
        raise ValueError("legacy_report_commit_mismatch")
    if report.get("diagnostic_scope") != "CLASSIFICATION_ONLY":
        raise ValueError("legacy_report_scope_mismatch")
    if report.get("training_scope") != "DIAGNOSTIC_CLASSIFICATION_ONLY":
        raise ValueError("legacy_report_training_scope_mismatch")
    if report.get("experiment_count") != 8:
        raise ValueError("legacy_report_experiment_count_mismatch")
    if report.get("economic_evaluation_performed") is not False:
        raise ValueError("legacy_report_economic_scope_violation")
    if report.get("economic_metrics_reported") is not False:
        raise ValueError("legacy_report_economic_metrics_violation")


def _validate_legacy_target_policy(report: dict[str, Any]) -> None:
    policy = _mapping(report.get("split_policy"), "legacy_split_policy_missing")
    expected = {
        "atr_period": 14,
        "chronological_training_only": True,
        "label_horizon_candles": 24,
        "label_window_purge": "create_purged_embargo_split",
        "post_validation_embargo": True,
        "pre_validation_embargo": True,
        "stop_loss_atr_multiple": 1.0,
        "take_profit_atr_multiple": 2.0,
        "validation_end_fraction": VALIDATION_END_FRACTION,
        "validation_start_fraction": VALIDATION_START_FRACTION,
    }
    if policy != expected:
        raise ValueError("legacy_report_target_policy_mismatch")
    if report.get("labels") != list(LABELS):
        raise ValueError("legacy_report_label_contract_mismatch")


def _legacy_experiment_key(item: dict[str, Any]) -> tuple[str, str, str]:
    pair = str(item.get("pair"))
    symbol = next(
        (candidate for candidate, pair_name in PAIR_NAMES.items() if pair_name == pair),
        "",
    )
    timeframe = str(item.get("timeframe"))
    if symbol not in SYMBOLS or timeframe not in TIMEFRAMES:
        raise ValueError("legacy_experiment_key_out_of_policy")
    return symbol, timeframe, "Baseline"


def _validate_legacy_experiment(item: dict[str, Any]) -> None:
    status = item.get("status")
    if status == "TRAINED":
        variants = _mapping(item.get("model_variants"), "legacy_trained_models_missing")
        expected_policies = {
            "weight_none",
            "weight_balanced",
            "weight_intermediate",
        }
        if set(variants) != expected_policies:
            raise ValueError("legacy_weighting_policy_matrix_invalid")
        for variant in variants.values():
            model = _mapping(variant, "legacy_model_variant_invalid")
            rows = model.get("validation_probabilities")
            if not isinstance(rows, list):
                raise ValueError("legacy_validation_probabilities_missing")
    elif status == "SKIPPED_INSUFFICIENT_SAMPLES":
        if item.get("model_variants") is not None:
            raise ValueError("legacy_skipped_experiment_has_models")
    else:
        raise ValueError("legacy_experiment_status_invalid")

    _integer(item.get("bar_count"), "legacy_bar_count_invalid")
    _integer(item.get("train_rows"), "legacy_train_rows_invalid")
    _integer(item.get("validation_rows"), "legacy_validation_rows_invalid")
    split = _mapping(item.get("split"), "legacy_split_metadata_missing")
    _integer(split.get("validation_start_index"), "legacy_validation_start_invalid")
    _integer(split.get("validation_end_index"), "legacy_validation_end_invalid")


def _index_legacy_report(
    report: dict[str, Any],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    raw_experiments = report.get("experiments")
    if not isinstance(raw_experiments, list) or len(raw_experiments) != 8:
        raise ValueError("legacy_experiment_matrix_invalid")
    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for raw in raw_experiments:
        item = _mapping(raw, "legacy_experiment_invalid")
        key = _legacy_experiment_key(item)
        if key in indexed:
            raise ValueError("legacy_duplicate_experiment")
        _validate_legacy_experiment(item)
        indexed[key] = item
    expected = {
        (symbol, timeframe, "Baseline")
        for symbol in SYMBOLS
        for timeframe in TIMEFRAMES
    }
    if set(indexed) != expected:
        raise ValueError("legacy_experiment_matrix_incomplete")
    return indexed


def _experiment_key(item: dict[str, Any]) -> tuple[str, str, str]:
    pair = str(item.get("pair"))
    symbol = next(
        (candidate for candidate, pair_name in PAIR_NAMES.items() if pair_name == pair),
        "",
    )
    timeframe = str(item.get("timeframe"))
    scenario_name = str(item.get("scenario"))
    if (
        symbol not in SYMBOLS
        or timeframe not in TIMEFRAMES
        or scenario_name not in SCENARIOS
    ):
        raise ValueError("v4c_experiment_key_out_of_policy")
    return symbol, timeframe, scenario_name


def _validate_experiment_barrier(item: dict[str, Any], scenario_name: str) -> None:
    expected = SCENARIOS[scenario_name]
    barrier = _mapping(item.get("triple_barrier"), "v4c_barrier_metadata_missing")
    actual = (
        _integer(barrier.get("horizon_candles"), "v4c_horizon_invalid"),
        _number(
            barrier.get("take_profit_multiplier"),
            "v4c_take_profit_multiplier_invalid",
        ),
        _number(
            barrier.get("stop_loss_multiplier"),
            "v4c_stop_loss_multiplier_invalid",
        ),
    )
    if actual != expected:
        raise ValueError("v4c_experiment_barrier_mismatch")


def _validate_experiment_status(item: dict[str, Any]) -> tuple[int, int]:
    status = item.get("status")
    if status == "TRAINED":
        variants = _mapping(item.get("model_variants"), "v4c_trained_models_missing")
        if set(variants) != set(WEIGHTING_POLICIES):
            raise ValueError("v4c_weighting_policy_matrix_invalid")
        return 1, len(variants)
    skipped_statuses = {
        "SKIPPED_INSUFFICIENT_SAMPLES",
        "SKIPPED_INSUFFICIENT_CLASS_DIVERSITY",
    }
    if status in skipped_statuses:
        if item.get("model_variants") is not None:
            raise ValueError("v4c_skipped_experiment_has_models")
        return 0, 0
    raise ValueError("v4c_experiment_status_invalid")


def _expected_experiment_keys() -> set[tuple[str, str, str]]:
    return {
        (symbol, timeframe, scenario)
        for symbol in SYMBOLS
        for timeframe in TIMEFRAMES
        for scenario in SCENARIOS
    }


def _validate_trained_counts(
    report: dict[str, Any], trained_experiments: int, trained_models: int
) -> None:
    if report.get("trained_experiments") != trained_experiments:
        raise ValueError("v4c_trained_experiment_count_mismatch")
    if report.get("trained_models") != trained_models:
        raise ValueError("v4c_trained_model_count_mismatch")


def _index_report_experiments(
    report: dict[str, Any],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    raw_experiments = report.get("experiments")
    if not isinstance(raw_experiments, list) or len(raw_experiments) != 16:
        raise ValueError("v4c_experiment_matrix_invalid")

    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    trained_experiments = 0
    trained_models = 0
    for raw in raw_experiments:
        item = _mapping(raw, "v4c_experiment_invalid")
        key = _experiment_key(item)
        if key in indexed:
            raise ValueError("v4c_duplicate_experiment")
        _validate_experiment_barrier(item, key[2])
        trained, models = _validate_experiment_status(item)
        trained_experiments += trained
        trained_models += models
        indexed[key] = item

    if set(indexed) != _expected_experiment_keys():
        raise ValueError("v4c_experiment_matrix_incomplete")
    _validate_trained_counts(report, trained_experiments, trained_models)
    return indexed


def validate_v4c_report(
    report: dict[str, Any],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Validate either the V4-C v5 report or the pinned V4 v4 source artifact."""
    schema = report.get("schema_version")
    _validate_report_window(report)
    _validate_report_safety(report)
    if schema == REPORT_SCHEMA:
        _validate_report_identity(report)
        _validate_report_scenarios(report)
        return _index_report_experiments(report)
    if schema == LEGACY_REPORT_SCHEMA:
        _validate_legacy_report_identity(report)
        _validate_legacy_target_policy(report)
        return _index_legacy_report(report)
    raise ValueError("v4c_report_schema_mismatch")


def _validate_minute_timestamps(result: DataFrame) -> None:
    dates = pd.to_datetime(result["date"], errors="coerce", utc=True)
    if dates.isna().any():
        raise ValueError("timestamp_invalid")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("timestamps_must_be_unique_and_increasing")
    deltas = dates.diff().iloc[1:].dt.total_seconds().to_numpy(dtype=float)
    if len(deltas) and not bool(np.all(deltas == 60.0)):
        raise ValueError("minute_data_gap_or_interval_mismatch")
    result["date"] = dates


def _validate_minute_geometry(result: DataFrame) -> None:
    opened = result["open"].to_numpy(dtype=float)
    high = result["high"].to_numpy(dtype=float)
    low = result["low"].to_numpy(dtype=float)
    close = result["close"].to_numpy(dtype=float)
    volume = result["volume"].to_numpy(dtype=float)
    if bool(np.any(np.column_stack((opened, high, low, close)) <= 0)):
        raise ValueError("ohlcv_price_must_be_positive")
    if bool(np.any(volume < 0)):
        raise ValueError("ohlcv_volume_negative")
    invalid_geometry = (
        np.any(high < low)
        or np.any(high < np.maximum(opened, close))
        or np.any(low > np.minimum(opened, close))
    )
    if bool(invalid_geometry):
        raise ValueError("ohlcv_geometry_invalid")


def _validate_minute_numeric_values(result: DataFrame) -> None:
    for name in ("open", "high", "low", "close", "volume"):
        result[name] = pd.to_numeric(result[name], errors="coerce")
    numeric = result.loc[:, ["open", "high", "low", "close", "volume"]].to_numpy(
        dtype=float
    )
    if not bool(np.isfinite(numeric).all()):
        raise ValueError("ohlcv_value_not_finite")
    _validate_minute_geometry(result)


def validate_minute_candles(candles: DataFrame, expected_rows: int) -> DataFrame:
    """Validate the original gap-free one-minute source before any resampling."""
    missing = sorted(set(MINUTE_COLUMNS) - set(candles.columns))
    if missing:
        raise ValueError(f"ohlcv_missing_columns:{','.join(missing)}")
    if candles.empty:
        raise ValueError("ohlcv_empty")
    result = candles.loc[:, list(MINUTE_COLUMNS)].copy().reset_index(drop=True)
    _validate_minute_timestamps(result)
    _validate_minute_numeric_values(result)
    if len(result) != expected_rows:
        raise ValueError("minute_candle_count_mismatch")
    return result


def resample_minute_candles(candles: DataFrame, timeframe: str) -> DataFrame:
    """Mirror V4-C's epoch-aligned, complete-bar resampling without model imports."""
    if timeframe not in TIMEFRAMES:
        raise ValueError("entry_timeframe_out_of_policy")
    minutes = TIMEFRAMES[timeframe]
    indexed = candles.set_index("date")
    rule = f"{minutes}min"
    grouped = indexed.resample(rule, origin="epoch", closed="left", label="left")
    bars = grouped.agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    counts = grouped["close"].count()
    result = bars.loc[counts == minutes].dropna().reset_index()
    if result.empty:
        raise ValueError(f"no_complete_resampled_bars:{timeframe}")
    return result.loc[:, list(MINUTE_COLUMNS)]


def _label_audit_counts(labels: DataFrame) -> dict[str, Any]:
    labeled = labels["label"].isin(LABELS)
    exits = labels["exit_reason"].dropna()
    unlabeled = labels["unlabeled_reason"].dropna()
    return {
        "labeled_events": int(labeled.sum()),
        "unlabeled_events": int(labels["label"].isna().sum()),
        "exit_reasons": {
            reason: int((exits == reason).sum()) for reason in EXIT_REASONS
        },
        "unlabeled_reasons": {
            reason: int((unlabeled == reason).sum()) for reason in UNLABELED_REASONS
        },
    }


def _validate_label_audit(labels: DataFrame, experiment: dict[str, Any]) -> None:
    actual = _label_audit_counts(labels)
    reported = _mapping(experiment.get("label_audit"), "v4c_label_audit_missing")
    exits = _mapping(
        _mapping(
            reported.get("exit_reason_distribution"), "v4c_exit_audit_missing"
        ).get("reasons"),
        "v4c_exit_reason_counts_missing",
    )
    unlabeled = _mapping(
        _mapping(
            reported.get("unlabeled_reason_distribution"),
            "v4c_unlabeled_audit_missing",
        ).get("reasons"),
        "v4c_unlabeled_reason_counts_missing",
    )
    if actual["labeled_events"] != reported.get("labeled_events"):
        raise ValueError("v4c_labeled_event_count_mismatch")
    if actual["unlabeled_events"] != reported.get("unlabeled_events"):
        raise ValueError("v4c_unlabeled_event_count_mismatch")
    for reason in EXIT_REASONS:
        if actual["exit_reasons"][reason] != exits.get(reason, {}).get("count"):
            raise ValueError("v4c_exit_reason_count_mismatch")
    for reason in UNLABELED_REASONS:
        if actual["unlabeled_reasons"][reason] != unlabeled.get(reason, {}).get(
            "count"
        ):
            raise ValueError("v4c_unlabeled_reason_count_mismatch")


def _eligible_indices(
    bars: DataFrame, labels: DataFrame, start_index: int, end_index: int
) -> list[int]:
    features = build_causal_feature_matrix(bars)
    if len(features) != len(labels):
        raise ValueError("feature_label_row_count_mismatch")
    eligible = (
        features.notna().all(axis=1)
        & labels["label"].isin(LABELS)
        & labels["label_end_idx"].notna()
    )
    return [
        index
        for index in range(start_index, end_index + 1)
        if bool(eligible.iloc[index])
    ]


def _validate_label_distribution(
    labels: DataFrame, indices: list[int], experiment: dict[str, Any]
) -> None:
    observed = labels.iloc[indices]["label"]
    distribution = _mapping(
        _mapping(
            experiment.get("label_distribution"), "v4c_label_distribution_missing"
        ).get("validation"),
        "v4c_validation_label_distribution_missing",
    )
    classes = _mapping(
        distribution.get("classes"), "v4c_validation_label_counts_missing"
    )
    if distribution.get("total") != len(indices):
        raise ValueError("v4c_validation_row_count_mismatch")
    for label in LABELS:
        expected = _integer(
            _mapping(classes.get(label), "v4c_validation_class_missing").get("count"),
            "v4c_validation_class_count_invalid",
        )
        actual = int((observed == label).sum())
        if actual != expected:
            raise ValueError("v4c_validation_label_count_mismatch")


def _validate_serialized_validation(
    experiment: dict[str, Any],
    labels: DataFrame,
    bars: DataFrame,
    indices: list[int],
    weighting_policies: tuple[str, ...] = WEIGHTING_POLICIES,
) -> None:
    variants = _mapping(experiment.get("model_variants"), "v4c_trained_models_missing")
    expected_indices = indices
    if set(variants) != set(weighting_policies):
        raise ValueError("validation_weighting_policy_matrix_invalid")
    for policy in weighting_policies:
        variant = _mapping(variants.get(policy), "v4c_model_variant_missing")
        raw_rows = variant.get("validation_probabilities")
        if not isinstance(raw_rows, list) or len(raw_rows) != len(expected_indices):
            raise ValueError("v4c_validation_probability_rows_mismatch")
        for raw, expected_index in zip(raw_rows, expected_indices, strict=True):
            row = _mapping(raw, "v4c_validation_probability_row_invalid")
            row_index = _integer(
                row.get("row_index"), "v4c_validation_row_index_invalid"
            )
            if row_index != expected_index:
                raise ValueError("v4c_validation_row_index_mismatch")
            if row.get("actual_label") != labels.iloc[row_index]["label"]:
                raise ValueError("v4c_validation_actual_label_mismatch")
            raw_timestamp = row.get("timestamp")
            if not isinstance(raw_timestamp, str):
                raise ValueError("v4c_validation_timestamp_invalid")
            timestamp = pd.Timestamp(raw_timestamp)
            if timestamp != pd.Timestamp(bars["date"].iloc[row_index]):
                raise ValueError("v4c_validation_timestamp_mismatch")


def _transition_event(
    index: int, bars: DataFrame, baseline_labels: DataFrame, short_labels: DataFrame
) -> dict[str, Any] | None:
    baseline = baseline_labels.iloc[index]
    short = short_labels.iloc[index]
    direction = str(baseline["label"])
    if direction not in {"BUY", "SELL"} or str(short["label"]) != "HOLD":
        return None
    offset = int(baseline["label_end_idx"]) - index
    if not 13 <= offset <= 24:
        raise ValueError("transition_hit_offset_out_of_late_window")
    if short["exit_reason"] != "HORIZON_EXPIRY":
        raise ValueError("short_horizon_hold_reason_mismatch")
    if int(short["label_end_idx"]) - index != 12:
        raise ValueError("short_horizon_expiry_offset_mismatch")
    reason = str(baseline["exit_reason"])
    if direction == "BUY" and reason != "TP_HIT":
        raise ValueError("baseline_buy_exit_reason_mismatch")
    if direction == "SELL" and reason not in {"SL_HIT", "COLLISION"}:
        raise ValueError("baseline_sell_exit_reason_mismatch")
    return _serialize_transition_event(index, offset, direction, reason, bars)


def _serialize_transition_event(
    index: int,
    offset: int,
    direction: str,
    reason: str,
    bars: DataFrame,
) -> dict[str, Any]:
    bucket = "13-18" if offset <= 18 else "19-24"
    hit_index = int(index + offset)
    return {
        "row_index": int(index),
        "event_timestamp": pd.Timestamp(bars["date"].iloc[index]).isoformat(),
        "baseline_label": direction,
        "short_horizon_label": "HOLD",
        "short_horizon_exit_reason": "HORIZON_EXPIRY",
        "late_hit_candle_offset": offset,
        "late_hit_bucket": bucket,
        "first_touch_direction": direction,
        "first_touch_reason": reason,
        "first_touch_row_index": hit_index,
        "first_touch_timestamp": pd.Timestamp(bars["date"].iloc[hit_index]).isoformat(),
    }


def collect_transition_events(
    bars: DataFrame,
    baseline_labels: DataFrame,
    short_labels: DataFrame,
    validation_indices: list[int],
) -> list[dict[str, Any]]:
    """Return only 24-candle directional hits that become 12-candle HOLD expiry."""
    events: list[dict[str, Any]] = []
    for index in validation_indices:
        event = _transition_event(index, bars, baseline_labels, short_labels)
        if event is not None:
            events.append(event)
    return events


def _transition_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = {"13-18": 0, "19-24": 0}
    directions = {"BUY": 0, "SELL": 0}
    reasons = {reason: 0 for reason in ("TP_HIT", "SL_HIT", "COLLISION")}
    for event in events:
        buckets[str(event["late_hit_bucket"])] += 1
        directions[str(event["first_touch_direction"])] += 1
        reasons[str(event["first_touch_reason"])] += 1
    return {
        "total": len(events),
        "late_hit_candle_buckets": buckets,
        "first_touch_direction_counts": directions,
        "first_touch_reason_counts": reasons,
    }


def _source_data_metadata(
    candles: DataFrame, source_hash: str | None
) -> dict[str, Any]:
    return {
        "minute_rows": len(candles),
        "first_timestamp": pd.Timestamp(candles["date"].iloc[0]).isoformat(),
        "last_timestamp": pd.Timestamp(candles["date"].iloc[-1]).isoformat(),
        "sha256": source_hash,
    }


def _scenario_validation_data(
    bars: DataFrame,
    experiment: dict[str, Any],
    scenario_name: str,
    start: int,
    end: int,
    *,
    validate_reference: bool = True,
    weighting_policies: tuple[str, ...] = WEIGHTING_POLICIES,
) -> tuple[DataFrame, list[int]]:
    split = _mapping(experiment.get("split"), "v4c_split_metadata_missing")
    if split.get("validation_start_index") != start:
        raise ValueError("v4c_validation_start_index_mismatch")
    if split.get("validation_end_index") != end:
        raise ValueError("v4c_validation_end_index_mismatch")
    horizon, take_profit, stop_loss = SCENARIOS[scenario_name]
    labels = apply_adaptive_triple_barrier(
        bars,
        horizon_candles=horizon,
        take_profit_multiplier=take_profit,
        stop_loss_multiplier=stop_loss,
    )
    if validate_reference:
        _validate_label_audit(labels, experiment)
    indices = _eligible_indices(bars, labels, start, end)
    if validate_reference:
        if experiment.get("validation_rows") != len(indices):
            raise ValueError("v4c_validation_eligible_row_count_mismatch")
        _validate_label_distribution(labels, indices, experiment)
        if experiment.get("status") == "TRAINED":
            _validate_serialized_validation(
                experiment, labels, bars, indices, weighting_policies
            )
    return labels, indices


def _pair_experiments(
    indexed: dict[tuple[str, str, str], dict[str, Any]],
    symbol: str,
    timeframe: str,
) -> dict[str, dict[str, Any]]:
    return {scenario: indexed[(symbol, timeframe, scenario)] for scenario in SCENARIOS}


def _audit_timeframe(
    symbol: str,
    timeframe: str,
    candles: DataFrame,
    indexed: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    bars = resample_minute_candles(candles, timeframe)
    baseline_key = (symbol, timeframe, "Baseline")
    short_key = (symbol, timeframe, "Short_Horizon")
    legacy_reference = short_key not in indexed
    if legacy_reference:
        baseline_experiment = indexed[baseline_key]
        pair_results = {
            "Baseline": baseline_experiment,
            "Short_Horizon": baseline_experiment,
        }
    else:
        pair_results = _pair_experiments(indexed, symbol, timeframe)
    if any(item.get("bar_count") != len(bars) for item in pair_results.values()):
        raise ValueError("v4c_resampled_bar_count_mismatch")
    start = int(len(bars) * VALIDATION_START_FRACTION)
    end = int(len(bars) * VALIDATION_END_FRACTION) - 1
    if end <= start:
        raise ValueError("v4c_validation_window_invalid")

    labels_by_scenario: dict[str, DataFrame] = {}
    indices_by_scenario: dict[str, list[int]] = {}
    for scenario_name in SCENARIOS:
        labels, indices = _scenario_validation_data(
            bars,
            pair_results[scenario_name],
            scenario_name,
            start,
            end,
            validate_reference=not legacy_reference or scenario_name == "Baseline",
            weighting_policies=(
                ("weight_none", "weight_balanced", "weight_intermediate")
                if legacy_reference
                else WEIGHTING_POLICIES
            ),
        )
        labels_by_scenario[scenario_name] = labels
        indices_by_scenario[scenario_name] = indices

    validation_indices = sorted(
        set(indices_by_scenario["Baseline"]) & set(indices_by_scenario["Short_Horizon"])
    )
    events = collect_transition_events(
        bars,
        labels_by_scenario["Baseline"],
        labels_by_scenario["Short_Horizon"],
        validation_indices,
    )
    return {
        "symbol": symbol,
        "pair": PAIR_NAMES[symbol],
        "timeframe": timeframe,
        "validation_window": {
            "start_row_index": start,
            "end_row_index": end,
            "shared_eligible_rows": len(validation_indices),
        },
        "baseline": {
            "horizon_candles": SCENARIOS["Baseline"][0],
            "status": pair_results["Baseline"]["status"],
        },
        "short_horizon": {
            "horizon_candles": SCENARIOS["Short_Horizon"][0],
            "status": (
                "AUDIT_ONLY_NO_MODEL"
                if legacy_reference
                else pair_results["Short_Horizon"]["status"]
            ),
        },
        "transition_summary": _transition_summary(events),
        "events": events,
    }


def _find_source_dataset(
    report: dict[str, Any], symbol: str
) -> dict[str, Any]:
    datasets = report.get("input_datasets")
    if not isinstance(datasets, list) or len(datasets) != len(SYMBOLS):
        raise ValueError("source_dataset_metadata_missing")
    for item in datasets:
        if isinstance(item, dict) and item.get("symbol") == symbol:
            return _mapping(item, "source_dataset_metadata_invalid")
    raise ValueError("source_dataset_symbol_missing")


def _validate_source_dataset_identity(
    dataset: dict[str, Any],
    symbol: str,
    selected_days: int,
    source_hash: str,
) -> None:
    expected_rows = selected_days * 1440
    matches = (
        dataset.get("pair") == PAIR_NAMES[symbol]
        and dataset.get("provider") == "Binance Spot public klines API"
        and dataset.get("interval") == "1m"
        and dataset.get("requested_days") == selected_days
        and dataset.get("row_count") == expected_rows
        and dataset.get("missing_minute_intervals") == 0
        and dataset.get("public_api_only") is True
        and dataset.get("credentials_used") is False
        and dataset.get("csv_sha256") == source_hash
    )
    if not matches:
        raise ValueError("source_dataset_metadata_mismatch")


def _validate_source_dataset_window(
    dataset: dict[str, Any], candles: DataFrame
) -> None:
    raw_start = dataset.get("start_utc")
    raw_end = dataset.get("end_utc")
    if not isinstance(raw_start, str) or not isinstance(raw_end, str):
        raise ValueError("source_dataset_timestamp_invalid")
    start = pd.Timestamp(raw_start)
    end = pd.Timestamp(raw_end)
    actual_start = pd.Timestamp(candles["date"].iloc[0])
    actual_end = pd.Timestamp(candles["date"].iloc[-1])
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("source_dataset_timestamp_timezone_missing")
    if start.tz_convert("UTC") != actual_start or end.tz_convert("UTC") != actual_end:
        raise ValueError("source_dataset_window_mismatch")


def _validate_source_dataset_metadata(
    report: dict[str, Any],
    symbol: str,
    candles: DataFrame,
    source_hash: str | None,
) -> None:
    if source_hash is None:
        return
    selected_days = _integer(report.get("requested_days"), "requested_days_invalid")
    dataset = _find_source_dataset(report, symbol)
    _validate_source_dataset_identity(dataset, symbol, selected_days, source_hash)
    _validate_source_dataset_window(dataset, candles)


def _audit_symbol(
    symbol: str,
    raw_candles: DataFrame,
    expected_minute_rows: int,
    source_hash: str | None,
    indexed: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candles = validate_minute_candles(raw_candles, expected_minute_rows)
    source_metadata = _source_data_metadata(candles, source_hash)
    results = [
        _audit_timeframe(symbol, timeframe, candles, indexed)
        for timeframe in TIMEFRAMES
    ]
    return source_metadata, results


def build_transition_audit(
    report: dict[str, Any],
    minute_frames: dict[str, DataFrame],
    source_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Rebuild both V4-C label sets from source OHLCV and audit the validation window."""
    indexed = validate_v4c_report(report)
    window = _mapping(
        report.get("data_window_selection"), "data_window_metadata_missing"
    )
    selected_days = _integer(window.get("selected_days"), "selected_days_invalid")
    expected_minute_rows = selected_days * 24 * 60
    results: list[dict[str, Any]] = []
    source_data: dict[str, Any] = {}
    hashes = source_hashes or {}
    for symbol in SYMBOLS:
        if symbol not in minute_frames:
            raise ValueError(f"minute_data_missing:{symbol}")
        _validate_source_dataset_metadata(
            report, symbol, minute_frames[symbol], hashes.get(symbol)
        )
        source_data[symbol], symbol_results = _audit_symbol(
            symbol,
            minute_frames[symbol],
            expected_minute_rows,
            hashes.get(symbol),
            indexed,
        )
        results.extend(symbol_results)

    all_events = [event for experiment in results for event in experiment["events"]]
    return {
        "schema_version": AUDIT_SCHEMA,
        "source_artifact": {
            "schema_version": report["schema_version"],
            "commit_sha": report["commit_sha"],
            "requested_days": report["requested_days"],
            "source_run_number": SOURCE_RUN_NUMBER,
            "source_run_id": SOURCE_RUN_ID,
            "source_artifact_id": SOURCE_ARTIFACT_ID,
        },
        "source_data": source_data,
        "audit_scope": (
            "LEGACY_V4_BASELINE_24_VS_REBUILT_SHORT_HORIZON_12"
            if report["schema_version"] == LEGACY_REPORT_SCHEMA
            else "V4_C_CHRONOLOGICAL_VALIDATION_WINDOW"
        ),
        "transition_definition": (
            "Baseline label BUY/SELL with first TP/SL touch at candles 13-24, "
            "while Short_Horizon label is HOLD/HORIZON_EXPIRY at candle 12."
        ),
        "experiments": results,
        "aggregate_transition_summary": _transition_summary(all_events),
        "safety": dict(SAFE_VALUES),
        "model_training_performed": False,
        "production_target_modified": False,
    }


def audit_report_files(
    report_path: Path,
    btc_candles_path: Path,
    eth_candles_path: Path,
) -> dict[str, Any]:
    """Load report and matching raw CSV inputs, then build the auditable JSON result."""
    report_bytes = report_path.read_bytes()
    try:
        report = _mapping(json.loads(report_bytes), "v4c_report_json_invalid")
    except json.JSONDecodeError as exc:
        raise ValueError("v4c_report_json_invalid") from exc
    frames: dict[str, DataFrame] = {}
    hashes: dict[str, str] = {}
    for symbol, path in (
        ("BTCUSDT", btc_candles_path),
        ("ETHUSDT", eth_candles_path),
    ):
        file_bytes = path.read_bytes()
        hashes[symbol] = hashlib.sha256(file_bytes).hexdigest()
        frames[symbol] = pd.read_csv(path)
    result = build_transition_audit(report, frames, hashes)
    result["source_artifact"]["sha256"] = hashlib.sha256(report_bytes).hexdigest()
    result["source_artifact"]["filename"] = report_path.name
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit V4-C 24-to-12 candle label transitions using original OHLCV."
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--btc-1m-csv", type=Path, required=True)
    parser.add_argument("--eth-1m-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = audit_report_files(
            args.report,
            args.btc_1m_csv,
            args.eth_1m_csv,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "experiments": len(result["experiments"]),
                "transitions": result["aggregate_transition_summary"]["total"],
                "model_training_performed": False,
                "Champion": result["safety"]["Champion"],
                "trading_authorization": result["safety"]["trading_authorization"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
