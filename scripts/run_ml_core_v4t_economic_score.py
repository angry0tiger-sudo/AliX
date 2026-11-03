"""V4-T economic evaluation: calibrated BUY confidence, 1.5R geometry and row logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from scripts.run_ml_core_real_data_baseline import (
    DEFAULT_CALIBRATION_FRACTION,
    DEFAULT_MINIMUM_HOLD_PRECISION,
    DEFAULT_MINIMUM_HOLD_RECALL,
    LABELS,
    SYMBOLS,
    _load_diagnostic_datasets,
    resample_ohlcv,
    weighting_policy_specs,
)
from scripts.run_ml_core_v4h_tuning_geometry import (
    V4H_FEATURE_PROFILES,
    run_v4h_experiment,
)
from services.ml_core.adaptive_targets import apply_adaptive_triple_barrier
from services.ml_core.economic_backtest import (
    ExecutionConfig,
    TradeManagementConfig,
    evaluate_candidate,
    probability_to_signal,
)
from services.ml_core.extended_features import build_feature_profiles

V4T_REPORT_SCHEMA = "alix.ml-core.v4t-economic-score.v1"
V4T_TIMEFRAMES = ("1h", "4h")
V4T_ARMS: tuple[dict[str, Any], ...] = (
    {
        "name": "CONTROL_SYMMETRIC_1R",
        "geometry": "SYMMETRIC_1R",
        "take_profit_multiplier": 1.0,
        "stop_loss_multiplier": 1.0,
        "horizon_candles": 24,
    },
    {
        "name": "PRIMARY_1_5R",
        "geometry": "ASYMMETRIC_1_5R",
        "take_profit_multiplier": 1.5,
        "stop_loss_multiplier": 1.0,
        "horizon_candles": 24,
    },
)
V4T_ATR_PERIOD = 14
V4T_ENTRY_SLIPPAGE_BPS = 20.0
V4T_ESTIMATED_ROUND_TRIP_COST_BPS = 90.0
V4T_FEE_BPS_PER_SIDE = 25.0
V4T_SLIPPAGE_BPS_PER_SIDE = 20.0
V4T_HOLD_THRESHOLD = 0.60
V4T_TOP_BUY_SHARE = 0.30
V4T_CALIBRATION_FRACTION = DEFAULT_CALIBRATION_FRACTION
V4T_SOFT_TRADE_TARGET = 30
V4T_MAXIMUM_DRAWDOWN = 0.15
V4T_RANDOM_SEARCH_ITERATIONS = 9
V4T_EXPECTED_EXPERIMENTS = (
    len(SYMBOLS) * len(V4T_TIMEFRAMES) * len(V4T_ARMS) * len(V4H_FEATURE_PROFILES)
)
V4T_EXPECTED_CANDIDATES = V4T_EXPECTED_EXPERIMENTS * len(weighting_policy_specs())
V4T_SAMPLE_GATE_SKIP_STATUSES = (
    "SKIPPED_INSUFFICIENT_CLASS_DIVERSITY",
    "SKIPPED_INSUFFICIENT_SAMPLES",
)
V4T_PROBABILITY_COLUMNS = (
    "candidate_id",
    "arm",
    "pair",
    "timeframe",
    "feature_profile",
    "weighting_policy",
    "row_index",
    "timestamp",
    "split_role",
    "actual_label",
    "prob_buy",
    "prob_sell",
    "prob_hold",
)


def _execution_config() -> ExecutionConfig:
    return ExecutionConfig(
        initial_equity=10_000.0,
        fee_bps=V4T_FEE_BPS_PER_SIDE,
        slippage_bps=V4T_SLIPPAGE_BPS_PER_SIDE,
        position_fraction=1.0,
        maximum_drawdown=V4T_MAXIMUM_DRAWDOWN,
    )


def _validated_v4t_label_end(
    raw_end: object, event_index: int, row_count: int
) -> int | None:
    if raw_end is None:
        return None
    try:
        numeric = float(cast(float | int | str, raw_end))
    except (TypeError, ValueError):
        raise ValueError("v4t_economic_label_end_index_invalid") from None
    if not np.isfinite(numeric):
        return None
    if not numeric.is_integer():
        raise ValueError("v4t_economic_label_end_index_invalid")
    label_end = int(numeric)
    if label_end <= event_index or label_end >= row_count:
        raise ValueError("v4t_economic_label_end_index_invalid")
    return label_end


def _v4t_exit_reference(bars: DataFrame, row: Any, label_end: int) -> float:
    reason = getattr(row, "exit_reason")
    if reason == "TP_HIT":
        exit_reference = float(getattr(row, "take_profit_price"))
    elif reason in ("SL_HIT", "COLLISION"):
        exit_reference = float(getattr(row, "stop_loss_price"))
    elif reason == "HORIZON_EXPIRY":
        exit_reference = float(bars.iloc[label_end]["close"])
    else:
        raise ValueError("v4t_economic_exit_reason_invalid")
    if not np.isfinite(exit_reference) or exit_reference <= 0.0:
        raise ValueError("v4t_economic_exit_reference_invalid")
    return exit_reference


def _v4t_event_gross_return(bars: DataFrame, row: Any) -> tuple[int, float] | None:
    event_index = int(getattr(row, "event_index"))
    if event_index < 0 or event_index >= len(bars):
        raise ValueError("v4t_economic_event_index_invalid")
    if getattr(row, "label") not in LABELS:
        return None
    label_end = _validated_v4t_label_end(
        getattr(row, "label_end_idx"), event_index, len(bars)
    )
    if label_end is None or event_index + 1 >= len(bars):
        return None
    entry_reference = float(bars.iloc[event_index + 1]["open"])
    if not np.isfinite(entry_reference) or entry_reference <= 0.0:
        raise ValueError("v4t_economic_entry_reference_invalid")
    exit_reference = _v4t_exit_reference(bars, row, label_end)
    return event_index, exit_reference / entry_reference - 1.0


def _economic_outcomes_by_index(bars: DataFrame, labels: DataFrame) -> Series:
    """Build causal gross event returns using next-open entry and target exits."""
    required = {
        "event_index",
        "label_end_idx",
        "label",
        "take_profit_price",
        "stop_loss_price",
        "exit_reason",
    }
    if not required.issubset(labels.columns):
        raise ValueError("v4t_economic_label_columns_missing")
    values = np.full(len(bars), np.nan, dtype=float)
    for row in labels.itertuples(index=False):
        event_return = _v4t_event_gross_return(bars, row)
        if event_return is not None:
            event_index, gross_return = event_return
            values[event_index] = gross_return
    return Series(values, index=bars.index, dtype=float)


def _probability_values(record: dict[str, Any]) -> dict[str, float]:
    raw = record.get("probabilities")
    if not isinstance(raw, dict) or set(raw) != set(LABELS):
        raise ValueError("v4t_probability_record_schema_invalid")
    probabilities = {label: float(raw[label]) for label in LABELS}
    values = np.asarray(list(probabilities.values()), dtype=float)
    if (
        not bool(np.isfinite(values).all())
        or bool(np.any(values < 0.0))
        or bool(np.any(values > 1.0))
        or not math.isclose(float(values.sum()), 1.0, rel_tol=0.0, abs_tol=1e-5)
    ):
        raise ValueError("v4t_probability_record_values_invalid")
    return probabilities


def _validate_dynamic_threshold_inputs(
    validation_probability_records: list[dict[str, Any]],
    calibration_fraction: float,
    hold_threshold: float,
    top_buy_share: float,
) -> int:
    if (
        isinstance(calibration_fraction, bool)
        or not np.isfinite(float(calibration_fraction))
        or not 0.0 < float(calibration_fraction) < 1.0
    ):
        raise ValueError("v4t_calibration_fraction_invalid")
    if (
        isinstance(top_buy_share, bool)
        or not np.isfinite(float(top_buy_share))
        or not 0.0 < float(top_buy_share) <= 1.0
    ):
        raise ValueError("v4t_top_buy_share_invalid")
    if not 0.0 < float(hold_threshold) <= 1.0:
        raise ValueError("v4t_hold_threshold_invalid")
    if len(validation_probability_records) < 2:
        raise ValueError("v4t_validation_probability_rows_insufficient")
    calibration_count = max(
        1,
        int(
            math.floor(
                len(validation_probability_records) * float(calibration_fraction)
            )
        ),
    )
    if calibration_count >= len(validation_probability_records):
        raise ValueError("v4t_calibration_leaves_no_forward_rows")
    return calibration_count


def _calibration_raw_buy_confidences(
    records: list[dict[str, Any]],
    calibration_count: int,
    hold_threshold: float,
) -> list[float]:
    confidences: list[float] = []
    for record in records[:calibration_count]:
        probabilities = _probability_values(record)
        if probability_to_signal(probabilities, float(hold_threshold)) == "BUY":
            confidences.append(probabilities["BUY"])
    return confidences


def _no_buy_threshold_result(
    records: list[dict[str, Any]],
    calibration_count: int,
    top_buy_share: float,
) -> dict[str, Any]:
    return {
        "status": "NO_BUY_SIGNALS_IN_CALIBRATION",
        "selected_threshold": 1.0,
        "calibration_row_count": calibration_count,
        "calibration_raw_buy_count": 0,
        "target_buy_share": float(top_buy_share),
        "target_selected_buy_count": 0,
        "selected_buy_count_in_calibration": 0,
        "achieved_buy_share": 0.0,
        "threshold_tie_count": 0,
        "calibration_start_row_index": int(records[0]["row_index"]),
        "forward_start_row_index": int(records[calibration_count]["row_index"]),
    }


def _selected_buy_threshold_result(
    records: list[dict[str, Any]],
    calibration_count: int,
    raw_buy_confidences: list[float],
    top_buy_share: float,
) -> dict[str, Any]:
    target_count = max(1, int(math.ceil(len(raw_buy_confidences) * top_buy_share)))
    ordered = sorted(raw_buy_confidences, reverse=True)
    threshold = max(float(ordered[target_count - 1]), float(np.finfo(float).eps))
    selected_count = sum(value >= threshold for value in raw_buy_confidences)
    tie_count = sum(value == threshold for value in raw_buy_confidences)
    return {
        "status": "SELECTED",
        "selected_threshold": threshold,
        "calibration_row_count": calibration_count,
        "calibration_raw_buy_count": len(raw_buy_confidences),
        "target_buy_share": float(top_buy_share),
        "target_selected_buy_count": target_count,
        "selected_buy_count_in_calibration": int(selected_count),
        "achieved_buy_share": round(selected_count / len(raw_buy_confidences), 8),
        "threshold_tie_count": int(tie_count),
        "calibration_start_row_index": int(records[0]["row_index"]),
        "forward_start_row_index": int(records[calibration_count]["row_index"]),
    }


def select_dynamic_buy_confidence_threshold(
    validation_probability_records: list[dict[str, Any]],
    *,
    calibration_fraction: float = V4T_CALIBRATION_FRACTION,
    hold_threshold: float = V4T_HOLD_THRESHOLD,
    top_buy_share: float = V4T_TOP_BUY_SHARE,
) -> dict[str, Any]:
    """Select BUY confidence on early Calibration rows; replay later rows only."""
    calibration_count = _validate_dynamic_threshold_inputs(
        validation_probability_records,
        calibration_fraction,
        hold_threshold,
        top_buy_share,
    )
    raw_buy_confidences = _calibration_raw_buy_confidences(
        validation_probability_records, calibration_count, hold_threshold
    )
    if not raw_buy_confidences:
        return _no_buy_threshold_result(
            validation_probability_records, calibration_count, top_buy_share
        )
    return _selected_buy_threshold_result(
        validation_probability_records,
        calibration_count,
        raw_buy_confidences,
        top_buy_share,
    )


def _candidate_identifier(
    arm_name: str, pair: str, timeframe: str, profile: str, policy: str
) -> str:
    return "|".join((arm_name, pair, timeframe, profile, policy))


def _sample_gate_skipped_metrics() -> dict[str, Any]:
    return {
        "initial_equity_usd": 10_000.0,
        "final_equity_usd": None,
        "net_return_fraction": None,
        "net_return_percent": None,
        "trade_count": 0,
        "win_rate": None,
        "profit_factor_net_of_costs": None,
        "profit_factor_infinite_no_losses": False,
        "max_drawdown_fraction": None,
        "max_drawdown_percent": None,
        "economic_data_status": "INSUFFICIENT_TRAINING_SAMPLES",
        "minimum_trade_count_soft_target": V4T_SOFT_TRADE_TARGET,
        "minimum_trade_count_soft_target_met": False,
        "trade_count_soft_penalty_factor": 1.0,
        "profit_factor_gate_passed": False,
        "drawdown_gate_passed": False,
        "economic_candidate_gate_passed": False,
    }


def _apply_soft_sample_diagnostics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Remove the legacy hard trade-count gate; retain sample size as a soft diagnostic."""
    result = dict(metrics)
    trade_count = int(result["trade_count"])
    drawdown = float(result["max_drawdown_fraction"])
    profit_factor = result.get("profit_factor_net_of_costs")
    infinite_pf = bool(result.get("profit_factor_infinite_no_losses", False))
    trade_count_penalty = max(
        0.0,
        float(V4T_SOFT_TRADE_TARGET - trade_count) / float(V4T_SOFT_TRADE_TARGET),
    )
    pf_pass = infinite_pf or (profit_factor is not None and float(profit_factor) > 1.5)
    drawdown_pass = drawdown < V4T_MAXIMUM_DRAWDOWN
    result.pop("trade_count_gate_passed", None)
    result.pop("minimum_trade_count_for_acceptance", None)
    result["economic_data_status"] = (
        "BELOW_SOFT_SAMPLE_TARGET"
        if trade_count < V4T_SOFT_TRADE_TARGET
        else "SAMPLE_TARGET_MET"
    )
    result["minimum_trade_count_soft_target"] = V4T_SOFT_TRADE_TARGET
    result["minimum_trade_count_soft_target_met"] = trade_count >= V4T_SOFT_TRADE_TARGET
    result["trade_count_soft_shortfall"] = max(0, V4T_SOFT_TRADE_TARGET - trade_count)
    result["trade_count_soft_penalty_factor"] = round(trade_count_penalty, 8)
    result["profit_factor_gate_passed"] = bool(pf_pass)
    result["drawdown_gate_passed"] = bool(drawdown_pass)
    result["economic_candidate_gate_passed"] = bool(pf_pass and drawdown_pass)
    return result


def _skipped_candidate_record(
    arm: dict[str, Any],
    pair: str,
    timeframe: str,
    profile: str,
    policy: str,
    experiment: dict[str, Any],
) -> dict[str, Any]:
    status = str(experiment.get("status"))
    sample_gate = experiment.get("sample_gate")
    if status not in V4T_SAMPLE_GATE_SKIP_STATUSES:
        raise ValueError("v4t_sample_gate_skip_status_invalid")
    if not isinstance(sample_gate, dict) or sample_gate.get("status") != status:
        raise ValueError("v4t_sample_gate_skip_diagnostics_invalid")
    return {
        "candidate_id": _candidate_identifier(
            str(arm["name"]), pair, timeframe, profile, policy
        ),
        "status": status,
        "evaluation_performed": False,
        "arm": str(arm["name"]),
        "geometry": str(arm["geometry"]),
        "objective": "CUSTOM_ECONOMIC_SCORE",
        "pair": pair,
        "timeframe": timeframe,
        "feature_profile": profile,
        "weighting_policy": policy,
        "sample_gate": dict(sample_gate),
        "train_rows": int(experiment["train_rows"]),
        "validation_rows": int(experiment["validation_rows"]),
        "metrics": _sample_gate_skipped_metrics(),
        "confidence_threshold_selection": None,
        "raw_signal_counts": {},
        "confidence_filter": {},
        "exit_reason_counts": {},
        "total_fees_usd": 0.0,
        "total_slippage_cost_usd": 0.0,
        "trades": [],
        "feature_importance_gain": [],
        "tuning_diagnostics": {},
        "probability_records": [],
    }


def _row_level_probability_rows(
    candidate_id: str,
    arm: dict[str, Any],
    pair: str,
    timeframe: str,
    profile: str,
    policy: str,
    records: list[dict[str, Any]],
    calibration_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if calibration_count <= 0 or calibration_count >= len(records):
        raise ValueError("v4t_probability_split_invalid")
    for index, record in enumerate(records):
        probabilities = _probability_values(record)
        rows.append(
            {
                "candidate_id": candidate_id,
                "arm": str(arm["name"]),
                "pair": pair,
                "timeframe": timeframe,
                "feature_profile": profile,
                "weighting_policy": policy,
                "row_index": int(record["row_index"]),
                "timestamp": str(record["timestamp"]),
                "split_role": (
                    "CALIBRATION" if index < calibration_count else "FORWARD_VALIDATION"
                ),
                "actual_label": str(record["actual_label"]),
                "prob_buy": probabilities["BUY"],
                "prob_sell": probabilities["SELL"],
                "prob_hold": probabilities["HOLD"],
            }
        )
    return rows


def _evaluate_trained_variant(
    arm: dict[str, Any],
    pair: str,
    timeframe: str,
    bars: DataFrame,
    profile: str,
    policy: str,
    experiment: dict[str, Any],
    variant: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidate_id = _candidate_identifier(
        str(arm["name"]), pair, timeframe, profile, policy
    )
    all_probability_records = variant.get("validation_probabilities")
    if not isinstance(all_probability_records, list) or len(
        all_probability_records
    ) != int(experiment["validation_rows"]):
        raise ValueError("v4t_validation_probability_count_mismatch")
    threshold_selection = select_dynamic_buy_confidence_threshold(
        all_probability_records
    )
    calibration_count = int(threshold_selection["calibration_row_count"])
    forward_records = all_probability_records[calibration_count:]
    if len(forward_records) != int(experiment["validation_rows"]) - calibration_count:
        raise ValueError("v4t_forward_probability_row_count_mismatch")
    if not forward_records:
        raise ValueError("v4t_forward_probability_rows_empty")
    for record in forward_records:
        _probability_values(record)
    hold_metrics = dict(variant["hold_metrics"])
    candidate_payload = {
        "candidate_id": candidate_id,
        "scenario": str(arm["name"]),
        "pair": pair,
        "timeframe": timeframe,
        "feature_profile": profile,
        "weighting_policy": policy,
        "selected_parameters": dict(
            variant["hyperparameter_tuning"]["selected_parameters"]
        ),
        "hold_precision": float(hold_metrics["precision"]),
        "hold_recall": float(hold_metrics["recall"]),
        "selected_hold_threshold": V4T_HOLD_THRESHOLD,
        "forward_start_row_index": int(forward_records[0]["row_index"]),
        "forward_row_count": len(forward_records),
        "probability_records": forward_records,
    }
    management = TradeManagementConfig(
        atr_period=V4T_ATR_PERIOD,
        take_profit_atr_multiplier=float(arm["take_profit_multiplier"]),
        stop_loss_atr_multiplier=float(arm["stop_loss_multiplier"]),
        horizon_candles=int(arm["horizon_candles"]),
        confidence_threshold=float(threshold_selection["selected_threshold"]),
        trailing_stop_enabled=False,
    )
    replay = evaluate_candidate(
        candidate_payload, bars, _execution_config(), management
    )
    metrics = _apply_soft_sample_diagnostics(dict(replay["metrics"]))
    candidate = {
        "candidate_id": candidate_id,
        "status": "TRAINED",
        "evaluation_performed": True,
        "arm": str(arm["name"]),
        "geometry": str(arm["geometry"]),
        "objective": "CUSTOM_ECONOMIC_SCORE",
        "pair": pair,
        "timeframe": timeframe,
        "feature_profile": profile,
        "weighting_policy": policy,
        "sample_gate": dict(experiment["sample_gate"]),
        "sample_signature_sha256": str(experiment["sample_signature_sha256"]),
        "train_rows": int(experiment["train_rows"]),
        "validation_rows": int(experiment["validation_rows"]),
        "forward_validation_rows": len(forward_records),
        "tuning_objective": "CUSTOM_ECONOMIC_SCORE",
        "selected_parameters": dict(
            variant["hyperparameter_tuning"]["selected_parameters"]
        ),
        "tuning_diagnostics": dict(variant["hyperparameter_tuning"]),
        "validation_classification_metrics": dict(variant["classification_metrics"]),
        "hold_metrics_argmax_full_validation": hold_metrics,
        "confidence_threshold_selection": threshold_selection,
        "selected_hold_threshold": V4T_HOLD_THRESHOLD,
        "selected_buy_confidence_threshold": float(
            threshold_selection["selected_threshold"]
        ),
        "raw_signal_counts": dict(replay["raw_signal_counts"]),
        "confidence_filter": dict(replay["confidence_filter"]),
        "metrics": metrics,
        "exit_reason_counts": dict(replay["exit_reason_counts"]),
        "total_fees_usd": float(replay["total_fees_usd"]),
        "total_slippage_cost_usd": float(replay["total_slippage_cost_usd"]),
        "trades": list(replay["trades"]),
        "feature_importance_gain": list(variant["feature_importance_gain"]),
        "probability_records": forward_records,
    }
    rows = _row_level_probability_rows(
        candidate_id,
        arm,
        pair,
        timeframe,
        profile,
        policy,
        all_probability_records,
        calibration_count,
    )
    return candidate, rows


def _run_context_profile(
    arm: dict[str, Any],
    pair: str,
    timeframe: str,
    bars: DataFrame,
    labels: DataFrame,
    economic_outcomes: Series,
    feature_matrices: dict[str, DataFrame],
    profile: str,
    random_search_iterations: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    experiment = run_v4h_experiment(
        pair,
        timeframe,
        bars,
        labels,
        feature_matrices,
        scenario=arm,
        feature_profile=profile,
        calibration_fraction=V4T_CALIBRATION_FRACTION,
        minimum_hold_precision=DEFAULT_MINIMUM_HOLD_PRECISION,
        minimum_hold_recall=DEFAULT_MINIMUM_HOLD_RECALL,
        random_search_iterations=random_search_iterations,
        tuning_objective="CUSTOM_ECONOMIC_SCORE",
        economic_outcomes=economic_outcomes,
        estimated_cost_bps=V4T_ESTIMATED_ROUND_TRIP_COST_BPS,
        split_horizon_candles=24,
    )
    status, sample_gate = _validate_v4t_experiment_status(experiment)
    experiment_record = _v4t_experiment_report_record(
        arm, pair, timeframe, profile, experiment, status, sample_gate
    )
    if status in V4T_SAMPLE_GATE_SKIP_STATUSES:
        skipped = _v4t_skipped_profile_candidates(
            arm, pair, timeframe, profile, experiment
        )
        return experiment_record, skipped, []
    candidates, probabilities = _v4t_trained_profile_candidates(
        arm, pair, timeframe, bars, profile, experiment, experiment_record
    )
    return experiment_record, candidates, probabilities


def _validate_v4t_experiment_status(
    experiment: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    status = str(experiment.get("status"))
    sample_gate = experiment.get("sample_gate")
    if not isinstance(sample_gate, dict):
        raise ValueError("v4t_sample_gate_diagnostics_missing")
    if status in V4T_SAMPLE_GATE_SKIP_STATUSES:
        if sample_gate.get("status") != status:
            raise ValueError("v4t_sample_gate_skip_status_mismatch")
        return status, sample_gate
    if status != "TRAINED" or sample_gate.get("status") != "PASS":
        raise ValueError(f"v4t_experiment_status_invalid:{status}")
    return status, sample_gate


def _v4t_experiment_report_record(
    arm: dict[str, Any],
    pair: str,
    timeframe: str,
    profile: str,
    experiment: dict[str, Any],
    status: str,
    sample_gate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "arm": str(arm["name"]),
        "geometry": str(arm["geometry"]),
        "pair": pair,
        "timeframe": timeframe,
        "feature_profile": profile,
        "status": status,
        "sample_gate": dict(sample_gate),
        "train_rows": int(experiment["train_rows"]),
        "validation_rows": int(experiment["validation_rows"]),
        "sample_signature_sha256": str(experiment["sample_signature_sha256"]),
        "split": dict(experiment["split"]),
        "label_distribution": dict(experiment["label_distribution"]),
        "label_audit": dict(experiment["label_audit"]),
        "model_variants": {},
    }


def _v4t_skipped_profile_candidates(
    arm: dict[str, Any],
    pair: str,
    timeframe: str,
    profile: str,
    experiment: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        _skipped_candidate_record(arm, pair, timeframe, profile, policy, experiment)
        for policy in weighting_policy_specs()
    ]


def _v4t_trained_profile_candidates(
    arm: dict[str, Any],
    pair: str,
    timeframe: str,
    bars: DataFrame,
    profile: str,
    experiment: dict[str, Any],
    experiment_record: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    variants = experiment.get("model_variants")
    expected_policies = set(weighting_policy_specs())
    if not isinstance(variants, dict) or set(variants) != expected_policies:
        raise ValueError("v4t_model_variant_contract_mismatch")
    candidates: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    for policy, variant in variants.items():
        candidate, rows = _evaluate_trained_variant(
            arm, pair, timeframe, bars, profile, policy, experiment, variant
        )
        candidates.append(candidate)
        probability_rows.extend(rows)
        experiment_record["model_variants"][policy] = {
            "candidate_id": candidate["candidate_id"],
            "selected_parameters": dict(candidate["selected_parameters"]),
            "tuning_objective": candidate["tuning_objective"],
            "confidence_threshold_selection": dict(
                candidate["confidence_threshold_selection"]
            ),
            "metrics": dict(candidate["metrics"]),
        }
    return candidates, probability_rows


def _run_context_arm(
    arm: dict[str, Any],
    pair: str,
    timeframe: str,
    bars: DataFrame,
    feature_matrices: dict[str, DataFrame],
    random_search_iterations: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    labels = apply_adaptive_triple_barrier(
        bars,
        atr_period=V4T_ATR_PERIOD,
        horizon_candles=int(arm["horizon_candles"]),
        take_profit_multiplier=float(arm["take_profit_multiplier"]),
        stop_loss_multiplier=float(arm["stop_loss_multiplier"]),
        entry_reference="NEXT_BAR_OPEN",
        entry_slippage_bps=V4T_ENTRY_SLIPPAGE_BPS,
    )
    economic_outcomes = _economic_outcomes_by_index(bars, labels)
    experiments: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    for profile in V4H_FEATURE_PROFILES:
        experiment, profile_candidates, profile_probabilities = _run_context_profile(
            arm,
            pair,
            timeframe,
            bars,
            labels,
            economic_outcomes,
            feature_matrices,
            profile,
            random_search_iterations,
        )
        experiments.append(experiment)
        candidates.extend(profile_candidates)
        probability_rows.extend(profile_probabilities)
    return experiments, candidates, probability_rows


def _run_context(
    symbol: str,
    timeframe: str,
    minute_candles: DataFrame,
    *,
    random_search_iterations: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pair = f"{symbol[:-4]}/USDT"
    bars = resample_ohlcv(minute_candles, timeframe)
    if len(bars) < 100:
        raise ValueError(f"v4t_insufficient_resampled_bars:{symbol}:{timeframe}")
    feature_matrices = build_feature_profiles(bars, profile_names=V4H_FEATURE_PROFILES)
    experiments: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    for arm in V4T_ARMS:
        arm_experiments, arm_candidates, arm_probabilities = _run_context_arm(
            arm,
            pair,
            timeframe,
            bars,
            feature_matrices,
            random_search_iterations,
        )
        experiments.extend(arm_experiments)
        candidates.extend(arm_candidates)
        probability_rows.extend(arm_probabilities)
    return experiments, candidates, probability_rows


def _validate_v4t_run_inputs(
    datasets: dict[str, DataFrame], random_search_iterations: int
) -> None:
    if (
        isinstance(random_search_iterations, bool)
        or not isinstance(random_search_iterations, int)
        or not 1 <= random_search_iterations <= 27
    ):
        raise ValueError("v4t_random_search_iterations_invalid")
    if set(datasets) != set(SYMBOLS):
        raise ValueError("v4t_dataset_symbol_contract_mismatch")


def _collect_v4t_contexts(
    datasets: dict[str, DataFrame],
    random_search_iterations: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    experiments: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        for timeframe in V4T_TIMEFRAMES:
            context_experiments, context_candidates, context_probabilities = (
                _run_context(
                    symbol,
                    timeframe,
                    datasets[symbol],
                    random_search_iterations=random_search_iterations,
                )
            )
            experiments.extend(context_experiments)
            candidates.extend(context_candidates)
            probability_rows.extend(context_probabilities)
    return experiments, candidates, probability_rows


def _validate_v4t_run_outputs(
    experiments: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    probability_rows: list[dict[str, Any]],
) -> None:
    if len(experiments) != V4T_EXPECTED_EXPERIMENTS:
        raise RuntimeError("v4t_experiment_count_mismatch")
    if len(candidates) != V4T_EXPECTED_CANDIDATES:
        raise RuntimeError("v4t_candidate_count_mismatch")
    if not any(item.get("status") == "TRAINED" for item in candidates):
        raise RuntimeError("v4t_no_trained_candidates")
    expected_probability_rows = sum(
        int(item["validation_rows"])
        for item in candidates
        if item.get("status") == "TRAINED"
    )
    if len(probability_rows) != expected_probability_rows:
        raise RuntimeError("v4t_row_level_probability_count_mismatch")


def run_v4t_experiments(
    datasets: dict[str, DataFrame],
    *,
    random_search_iterations: int = V4T_RANDOM_SEARCH_ITERATIONS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    _validate_v4t_run_inputs(datasets, random_search_iterations)
    experiments, candidates, probability_rows = _collect_v4t_contexts(
        datasets, random_search_iterations
    )
    _validate_v4t_run_outputs(experiments, candidates, probability_rows)
    return experiments, candidates, probability_rows


def _profit_factor(pnls: list[float]) -> float | None:
    gross_profit = sum(value for value in pnls if value > 0.0)
    gross_loss = abs(sum(value for value in pnls if value < 0.0))
    return round(gross_profit / gross_loss, 8) if gross_loss > 0.0 else None


def _candidate_comparison_rows(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        metrics = candidate["metrics"]
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "arm": candidate["arm"],
                "geometry": candidate["geometry"],
                "pair": candidate["pair"],
                "timeframe": candidate["timeframe"],
                "feature_profile": candidate["feature_profile"],
                "weighting_policy": candidate["weighting_policy"],
                "status": candidate["status"],
                "trade_count": metrics["trade_count"],
                "economic_data_status": metrics["economic_data_status"],
                "net_return_percent": metrics["net_return_percent"],
                "profit_factor_net_of_costs": metrics["profit_factor_net_of_costs"],
                "profit_factor_infinite_no_losses": metrics[
                    "profit_factor_infinite_no_losses"
                ],
                "max_drawdown_percent": metrics["max_drawdown_percent"],
                "minimum_trade_count_soft_target_met": metrics[
                    "minimum_trade_count_soft_target_met"
                ],
                "trade_count_soft_penalty_factor": metrics[
                    "trade_count_soft_penalty_factor"
                ],
                "profit_factor_gate_passed": metrics["profit_factor_gate_passed"],
                "drawdown_gate_passed": metrics["drawdown_gate_passed"],
                "economic_candidate_gate_passed": metrics[
                    "economic_candidate_gate_passed"
                ],
                "selected_buy_confidence_threshold": candidate.get(
                    "selected_buy_confidence_threshold"
                ),
                "selected_hold_threshold": candidate.get("selected_hold_threshold"),
            }
        )
    return rows


def _arm_trade_metrics(
    selected: list[dict[str, Any]],
    trained: list[dict[str, Any]],
    trades: list[dict[str, Any]],
) -> dict[str, Any]:
    net_pnls = [float(item["net_pnl_usd"]) for item in trades]
    gross_pnls = [float(item["gross_market_pnl_usd"]) for item in trades]
    drawdowns = [
        float(item["metrics"]["max_drawdown_percent"])
        for item in trained
        if item["metrics"]["max_drawdown_percent"] is not None
    ]
    return {
        "candidate_count": len(selected),
        "trained_candidate_count": len(trained),
        "sample_gate_skipped_candidate_count": len(selected) - len(trained),
        "trade_count": len(trades),
        "winning_trade_count": sum(value > 0.0 for value in net_pnls),
        "losing_trade_count": sum(value < 0.0 for value in net_pnls),
        "combined_net_profit_factor": _profit_factor(net_pnls),
        "combined_gross_profit_factor": _profit_factor(gross_pnls),
        "total_net_pnl_usd": round(sum(net_pnls), 8),
        "total_fees_usd": round(
            sum(float(item["total_fees_usd"]) for item in trained), 8
        ),
        "total_slippage_cost_usd": round(
            sum(float(item["total_slippage_cost_usd"]) for item in trained), 8
        ),
        "mean_candidate_max_drawdown_percent": (
            round(float(np.mean(drawdowns)), 8) if drawdowns else None
        ),
    }


def _arm_risk_metrics(trained: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "strict_economic_pass_candidate_count": sum(
            bool(item["metrics"]["economic_candidate_gate_passed"]) for item in trained
        ),
        "below_soft_sample_target_candidate_count": sum(
            not bool(item["metrics"]["minimum_trade_count_soft_target_met"])
            for item in trained
        ),
    }


def _arm_aggregate(candidates: list[dict[str, Any]], arm_name: str) -> dict[str, Any]:
    selected = [item for item in candidates if item["arm"] == arm_name]
    trained = [item for item in selected if item["status"] == "TRAINED"]
    trades = [trade for item in trained for trade in item["trades"]]
    return {
        **_arm_trade_metrics(selected, trained, trades),
        **_arm_risk_metrics(trained),
    }


def _trade_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for index, trade in enumerate(candidate["trades"]):
            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "arm": candidate["arm"],
                    "pair": candidate["pair"],
                    "timeframe": candidate["timeframe"],
                    "feature_profile": candidate["feature_profile"],
                    "weighting_policy": candidate["weighting_policy"],
                    "trade_index": index,
                    "entry_timestamp": trade["entry_timestamp"],
                    "exit_timestamp": trade["exit_timestamp"],
                    "entry_reference_price": trade["entry_reference_price"],
                    "entry_fill_price": trade["entry_fill_price"],
                    "exit_reference_price": trade["exit_reference_price"],
                    "exit_fill_price": trade["exit_fill_price"],
                    "gross_market_pnl_usd": trade["gross_market_pnl_usd"],
                    "fees_usd": trade["fees_usd"],
                    "slippage_cost_usd": trade["slippage_cost_usd"],
                    "net_pnl_usd": trade["net_pnl_usd"],
                    "exit_reason": trade["exit_reason"],
                }
            )
    return rows


def _feature_importance_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for item in candidate["feature_importance_gain"]:
            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "arm": candidate["arm"],
                    "pair": candidate["pair"],
                    "timeframe": candidate["timeframe"],
                    "feature_profile": candidate["feature_profile"],
                    "weighting_policy": candidate["weighting_policy"],
                    "feature": str(item["feature"]),
                    "gain": float(item["gain"]),
                }
            )
    return rows


def _report_candidate_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    excluded = {
        "trades",
        "probability_records",
        "feature_importance_gain",
        "tuning_diagnostics",
    }
    return [
        {key: value for key, value in item.items() if key not in excluded}
        for item in candidates
    ]


def _render_step_summary(report: dict[str, Any]) -> str:
    lines = [
        "## AliX V4-T Economic Evaluation",
        "",
        f"Commit SHA: {report['commit_sha']}",
        "",
        "Primary target: TP=1.5 ATR / SL=1.0 ATR / 24 candles.",
        "Control target: TP=1.0 ATR / SL=1.0 ATR / 24 candles.",
        "Objective: Custom_Economic_Score on purged inner-CV folds within outer Train.",
        "",
        "| Arm | Geometry | Candidates | Trained | Trades | Net PF | Mean Max DD | Net PnL USD | Strict metric passes | Below soft sample target |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in report["arms"]:
        aggregate = arm["aggregate"]
        lines.append(
            f"| {arm['name']} | {arm['geometry']} | {aggregate['candidate_count']} "
            f"| {aggregate['trained_candidate_count']} | {aggregate['trade_count']} "
            f"| {aggregate['combined_net_profit_factor']} "
            f"| {aggregate['mean_candidate_max_drawdown_percent']} "
            f"| {aggregate['total_net_pnl_usd']:.2f} "
            f"| {aggregate['strict_economic_pass_candidate_count']} "
            f"| {aggregate['below_soft_sample_target_candidate_count']} |"
        )
    lines.extend(
        [
            "",
            "Economic acceptance metrics: net PF > 1.5 and Max Drawdown < 15%.",
            "The 30-trade count is a soft sample target, not a hard pass/fail gate.",
            "BUY confidence threshold is calibrated on the early Calibration slice and replayed only on later Forward Validation rows.",
            "Row-level probabilities are persisted as an independent CSV artifact with timestamp, actual label and all three class probabilities.",
            "",
            "Safety: Champion=null, order_permission=false, trading_authorization=NO-GO, private_api_calls=0, real_order_attempts=0.",
            "Spot-only: short positions unsupported. Sequence models: none.",
            "Promotion: BLOCKED_PENDING_INDEPENDENT_OOS_AND_OWNER_REVIEW.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_csv(
    path: Path, rows: list[dict[str, Any]], fieldnames: tuple[str, ...] | list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fieldnames), extrasaction="raise"
        )
        writer.writeheader()
        writer.writerows(rows)


def _build_report(
    *,
    commit_sha: str,
    provenance: list[dict[str, Any]],
    experiments: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    probability_rows: list[dict[str, Any]],
    random_search_iterations: int,
) -> dict[str, Any]:
    comparison_rows = _candidate_comparison_rows(candidates)
    feature_rows = _feature_importance_rows(candidates)
    trade_rows = _trade_rows(candidates)
    trained = [item for item in candidates if item["status"] == "TRAINED"]
    return {
        "schema_version": V4T_REPORT_SCHEMA,
        "commit_sha": commit_sha,
        "requested_days": 730,
        "provider": "Binance Spot public klines API",
        "input_datasets": provenance,
        "model_training_performed": bool(trained),
        "economic_evaluation_performed": bool(trained),
        "economic_metrics_reported": bool(trained),
        "symbols": list(SYMBOLS),
        "market_timeframes": list(V4T_TIMEFRAMES),
        "feature_profiles": list(V4H_FEATURE_PROFILES),
        "model": {
            "family": "LightGBM LGBMClassifier",
            "sequence_model_used": False,
            "new_models_trained": len(trained),
        },
        "random_search_iterations": int(random_search_iterations),
        "candidate_count": len(candidates),
        "trained_candidate_count": len(trained),
        "sample_gate_skipped_candidate_count": len(candidates) - len(trained),
        "experiment_count": len(experiments),
        "trained_experiment_count": sum(
            item["status"] == "TRAINED" for item in experiments
        ),
        "sample_gate_skipped_experiment_count": sum(
            item["status"] in V4T_SAMPLE_GATE_SKIP_STATUSES for item in experiments
        ),
        "experimental_design": "PAIRED_1R_VS_1_5R_GEOMETRY",
        "arms": [
            {
                "name": str(arm["name"]),
                "geometry": str(arm["geometry"]),
                "take_profit_multiplier": float(arm["take_profit_multiplier"]),
                "stop_loss_multiplier": float(arm["stop_loss_multiplier"]),
                "horizon_candles": int(arm["horizon_candles"]),
                "aggregate": _arm_aggregate(candidates, str(arm["name"])),
            }
            for arm in V4T_ARMS
        ],
        "target_contract": {
            "atr_period": V4T_ATR_PERIOD,
            "primary_geometry": "TP=1.5 ATR; SL=1.0 ATR",
            "primary_horizon_candles": 24,
            "control_geometry": "TP=1.0 ATR; SL=1.0 ATR",
            "control_horizon_candles": 24,
            "label_entry_reference": "NEXT_BAR_OPEN",
            "entry_slippage_bps": V4T_ENTRY_SLIPPAGE_BPS,
            "atr_reference": "SIGNAL_CANDLE_CLOSE_ONLY",
            "same_candle_collision_policy": "STOP_LOSS_WINS",
            "common_outer_and_inner_purge_horizon_candles": 24,
            "common_incomplete_tail_policy": "LABELER_HORIZON_24_FOR_BOTH_ARMS",
        },
        "objective_contract": {
            "primary_objective": "CUSTOM_ECONOMIC_SCORE",
            "scorer": "Custom_Economic_Score",
            "scorer_scope": "PURGED_INNER_CV_FOLDS_WITHIN_OUTER_TRAIN",
            "estimated_round_trip_cost_bps": V4T_ESTIMATED_ROUND_TRIP_COST_BPS,
            "positive_net_expectancy_reward_multiplier": 2.0,
            "negative_net_expectancy_multiplier": 1.0,
            "drawdown_penalty_weight": 0.5,
            "soft_minimum_trade_count_target": V4T_SOFT_TRADE_TARGET,
            "soft_trade_count_penalty_max_bps": 50.0,
            "minimum_trade_count_is_hard_gate": False,
            "portfolio_overlap_modeled": False,
            "economic_acceptance_authority": "FORWARD_VALIDATION_COSTED_SPOT_EXECUTION_REPLAY",
        },
        "confidence_threshold_contract": {
            "method": "TOP_30_PERCENT_RAW_BUY_CONFIDENCE",
            "selection_scope": "EARLY_CALIBRATION_SLICE_OF_OUTER_VALIDATION",
            "application_scope": "LATER_FORWARD_VALIDATION_ONLY",
            "calibration_fraction": V4T_CALIBRATION_FRACTION,
            "target_buy_share": V4T_TOP_BUY_SHARE,
            "hold_probability_threshold": V4T_HOLD_THRESHOLD,
            "threshold_tie_policy": "INCLUDE_EQUAL_THRESHOLD_AND_REPORT_ACHIEVED_SHARE",
            "actual_label_used_for_threshold_selection": False,
        },
        "execution_cost_contract": {
            "fee_bps_per_side": V4T_FEE_BPS_PER_SIDE,
            "slippage_bps_per_side": V4T_SLIPPAGE_BPS_PER_SIDE,
            "round_trip_nominal_cost_bps": V4T_FEE_BPS_PER_SIDE * 2.0
            + V4T_SLIPPAGE_BPS_PER_SIDE * 2.0,
            "position_fraction": 1.0,
            "leverage": 1.0,
            "pyramiding": False,
            "short_positions_supported": False,
        },
        "row_level_probability_artifact": {
            "file": "alix-v4t-row-level-probabilities.csv",
            "row_count": len(probability_rows),
            "columns": list(V4T_PROBABILITY_COLUMNS),
            "contains_calibration_and_forward_validation": True,
        },
        "comparison_rows": comparison_rows,
        "trade_rows": trade_rows,
        "feature_importance_rows": feature_rows,
        "experiments": experiments,
        "candidates": _report_candidate_rows(candidates),
        "promotion_status": "BLOCKED_PENDING_INDEPENDENT_OOS_AND_OWNER_REVIEW",
        "safety": {
            "public_api_only": True,
            "credentials_used": False,
            "private_api_calls": 0,
            "real_order_attempts": 0,
            "orders_or_execution_api_used": False,
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "short_positions_supported": False,
            "sequence_model_used": False,
        },
    }


def _write_outputs(
    report: dict[str, Any],
    report_path: Path,
    comparison_path: Path,
    trades_path: Path,
    probabilities_path: Path,
    feature_importance_path: Path,
    probability_rows: list[dict[str, Any]],
) -> None:
    if len(probability_rows) != int(
        report["row_level_probability_artifact"]["row_count"]
    ):
        raise ValueError("v4t_probability_artifact_row_count_mismatch")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        comparison_path,
        report["comparison_rows"],
        list(report["comparison_rows"][0]),
    )
    _write_csv(
        trades_path,
        report["trade_rows"],
        (
            "candidate_id",
            "arm",
            "pair",
            "timeframe",
            "feature_profile",
            "weighting_policy",
            "trade_index",
            "entry_timestamp",
            "exit_timestamp",
            "entry_reference_price",
            "entry_fill_price",
            "exit_reference_price",
            "exit_fill_price",
            "gross_market_pnl_usd",
            "fees_usd",
            "slippage_cost_usd",
            "net_pnl_usd",
            "exit_reason",
        ),
    )
    _write_csv(probabilities_path, probability_rows, V4T_PROBABILITY_COLUMNS)
    _write_csv(
        feature_importance_path,
        report["feature_importance_rows"],
        (
            "candidate_id",
            "arm",
            "pair",
            "timeframe",
            "feature_profile",
            "weighting_policy",
            "feature",
            "gain",
        ),
    )
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as summary:
            summary.write(_render_step_summary(report))


def run_v4t(
    input_dir: Path,
    report_path: Path,
    comparison_path: Path,
    trades_path: Path,
    probabilities_path: Path,
    feature_importance_path: Path,
    *,
    days: int,
    commit_sha: str,
    random_search_iterations: int = V4T_RANDOM_SEARCH_ITERATIONS,
) -> dict[str, Any]:
    if days != 730:
        raise ValueError("v4t_dataset_window_must_be_730_days")
    datasets, provenance = _load_diagnostic_datasets(input_dir, days, commit_sha)
    experiments, candidates, probability_rows = run_v4t_experiments(
        datasets, random_search_iterations=random_search_iterations
    )
    report = _build_report(
        commit_sha=commit_sha,
        provenance=provenance,
        experiments=experiments,
        candidates=candidates,
        probability_rows=probability_rows,
        random_search_iterations=random_search_iterations,
    )
    _write_outputs(
        report,
        report_path,
        comparison_path,
        trades_path,
        probabilities_path,
        feature_importance_path,
        probability_rows,
    )
    print(
        json.dumps(
            {
                "commit_sha": commit_sha,
                "experiment_count": report["experiment_count"],
                "candidate_count": report["candidate_count"],
                "trained_candidate_count": report["trained_candidate_count"],
                "row_level_probability_count": report["row_level_probability_artifact"][
                    "row_count"
                ],
                "promotion_status": report["promotion_status"],
                "Champion": None,
                "order_permission": False,
                "trading_authorization": "NO-GO",
                "private_api_calls": 0,
                "real_order_attempts": 0,
            },
            sort_keys=True,
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-comparison-csv", type=Path, required=True)
    parser.add_argument("--output-trades-csv", type=Path, required=True)
    parser.add_argument("--output-probabilities-csv", type=Path, required=True)
    parser.add_argument("--output-feature-importance-csv", type=Path, required=True)
    parser.add_argument("--days", type=int, choices=(730,), default=730)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument(
        "--random-search-iterations",
        type=int,
        default=V4T_RANDOM_SEARCH_ITERATIONS,
    )
    args = parser.parse_args()
    report = run_v4t(
        args.input_dir,
        args.output_report,
        args.output_comparison_csv,
        args.output_trades_csv,
        args.output_probabilities_csv,
        args.output_feature_importance_csv,
        days=args.days,
        commit_sha=args.commit_sha,
        random_search_iterations=args.random_search_iterations,
    )
    if report["experiment_count"] != V4T_EXPECTED_EXPERIMENTS:
        raise RuntimeError("v4t_experiment_count_mismatch")
    if report["candidate_count"] != V4T_EXPECTED_CANDIDATES:
        raise RuntimeError("v4t_candidate_count_mismatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
