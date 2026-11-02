"""V4-R paired economic architecture experiment.

A 2x2 paired design isolates Triple-Barrier geometry and model-selection
objective on one frozen 1h/4h data snapshot. No orders or private APIs are used.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

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
from scripts.run_ml_core_v4h_tuning_geometry import run_v4h_experiment
from services.ml_core.adaptive_targets import apply_adaptive_triple_barrier
from services.ml_core.economic_backtest import (
    ExecutionConfig,
    TradeManagementConfig,
    evaluate_candidate,
)
from services.ml_core.extended_features import (
    build_feature_profiles,
)

V4R_REPORT_SCHEMA = "alix.ml-core.v4r-economic-architecture.v1"
V4R_TIMEFRAMES = ("1h", "4h")
V4R_FEATURE_PROFILES = ("Baseline", "Baseline+Trend")
V4R_ENTRY_SLIPPAGE_BPS = 20.0
V4R_ESTIMATED_ROUND_TRIP_COST_BPS = 90.0
V4R_FEE_BPS_PER_SIDE = 25.0
V4R_SLIPPAGE_BPS_PER_SIDE = 20.0
V4R_CONFIDENCE_THRESHOLD = 0.60
V4R_HOLD_THRESHOLD = 0.60
V4R_ATR_PERIOD = 14
V4R_COMMON_PURGE_HORIZON = 24
V4R_MINIMUM_TRADES = 30
V4R_MAXIMUM_DRAWDOWN = 0.15
V4R_RANDOM_SEARCH_ITERATIONS = 9
V4R_EXPECTED_CONTEXTS = 4
V4R_EXPECTED_ARMS = 4
V4R_EXPECTED_EXPERIMENTS = 32
V4R_EXPECTED_CANDIDATES = 64
V4R_SAMPLE_GATE_SKIP_STATUSES = (
    "SKIPPED_INSUFFICIENT_CLASS_DIVERSITY",
    "SKIPPED_INSUFFICIENT_SAMPLES",
)

V4R_ARMS: tuple[dict[str, Any], ...] = (
    {
        "name": "CONTROL_SYMMETRIC_BUY_F1",
        "take_profit_multiplier": 1.0,
        "stop_loss_multiplier": 1.0,
        "horizon_candles": 12,
        "tuning_objective": "BUY_F1",
        "geometry": "SYMMETRIC_1R",
        "objective": "BUY_F1",
    },
    {
        "name": "GEOMETRY_2R_BUY_F1",
        "take_profit_multiplier": 2.0,
        "stop_loss_multiplier": 1.0,
        "horizon_candles": 24,
        "tuning_objective": "BUY_F1",
        "geometry": "ASYMMETRIC_2R",
        "objective": "BUY_F1",
    },
    {
        "name": "SYMMETRIC_NET_EXPECTANCY",
        "take_profit_multiplier": 1.0,
        "stop_loss_multiplier": 1.0,
        "horizon_candles": 12,
        "tuning_objective": "NET_EXPECTANCY_PROXY",
        "geometry": "SYMMETRIC_1R",
        "objective": "NET_EXPECTANCY_PROXY",
    },
    {
        "name": "V4R_2R_NET_EXPECTANCY",
        "take_profit_multiplier": 2.0,
        "stop_loss_multiplier": 1.0,
        "horizon_candles": 24,
        "tuning_objective": "NET_EXPECTANCY_PROXY",
        "geometry": "ASYMMETRIC_2R",
        "objective": "NET_EXPECTANCY_PROXY",
    },
)


def _execution_config() -> ExecutionConfig:
    return ExecutionConfig(
        initial_equity=10_000.0,
        fee_bps=V4R_FEE_BPS_PER_SIDE,
        slippage_bps=V4R_SLIPPAGE_BPS_PER_SIDE,
        position_fraction=1.0,
        maximum_drawdown=V4R_MAXIMUM_DRAWDOWN,
    )


def _management_config(arm: dict[str, Any]) -> TradeManagementConfig:
    return TradeManagementConfig(
        atr_period=V4R_ATR_PERIOD,
        take_profit_atr_multiplier=float(arm["take_profit_multiplier"]),
        stop_loss_atr_multiplier=float(arm["stop_loss_multiplier"]),
        horizon_candles=int(arm["horizon_candles"]),
        confidence_threshold=V4R_CONFIDENCE_THRESHOLD,
        trailing_stop_enabled=False,
    )


def _mask_to_common_horizon(labels: DataFrame, row_count: int) -> DataFrame:
    """Make the eligible sample identical for 12- and 24-candle target arms."""
    result = labels.copy(deep=True)
    first_incomplete = max(0, row_count - V4R_COMMON_PURGE_HORIZON)
    tail_mask = result["event_index"].astype(int) >= first_incomplete
    result.loc[tail_mask, "label"] = None
    result.loc[tail_mask, "exit_reason"] = None
    result.loc[tail_mask, "label_end_idx"] = None
    result.loc[tail_mask, "unlabeled_reason"] = "common_24_candle_tail"
    return result


def _economic_label_arrays(
    labels: DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    event_indices = pd.to_numeric(labels["event_index"], errors="raise").to_numpy(
        dtype=np.int64
    )
    label_end_indices = pd.to_numeric(
        labels["label_end_idx"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    take_profit_prices = pd.to_numeric(
        labels["take_profit_price"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    stop_loss_prices = pd.to_numeric(
        labels["stop_loss_price"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    target_labels = labels["label"].to_numpy(dtype=object)
    exit_reasons = labels["exit_reason"].to_numpy(dtype=object)
    return (
        event_indices,
        label_end_indices,
        take_profit_prices,
        stop_loss_prices,
        target_labels,
        exit_reasons,
    )


def _validated_proxy_label_end_index(
    event_index: int, raw_label_end_value: float, row_count: int
) -> int | None:
    if not np.isfinite(raw_label_end_value):
        return None
    if not raw_label_end_value.is_integer():
        raise ValueError("v4r_economic_proxy_label_end_index_invalid")
    label_end_index = int(raw_label_end_value)
    if event_index < 0 or event_index >= row_count:
        raise ValueError("v4r_economic_proxy_event_index_invalid")
    if label_end_index <= event_index or label_end_index >= row_count:
        raise ValueError("v4r_economic_proxy_label_end_index_invalid")
    return label_end_index


def _proxy_entry_reference(bars: DataFrame, event_index: int) -> float:
    entry_reference = float(bars.iloc[event_index + 1]["open"])
    if entry_reference <= 0 or not np.isfinite(entry_reference):
        raise ValueError("v4r_economic_proxy_entry_price_invalid")
    return entry_reference


def _proxy_exit_reference(
    bars: DataFrame,
    position: int,
    label_end_index: int,
    exit_reason: object,
    take_profit_prices: np.ndarray,
    stop_loss_prices: np.ndarray,
) -> float:
    if exit_reason == "TP_HIT":
        exit_reference = float(take_profit_prices[position])
    elif exit_reason in ("SL_HIT", "COLLISION"):
        exit_reference = float(stop_loss_prices[position])
    elif exit_reason == "HORIZON_EXPIRY":
        exit_reference = float(bars.iloc[label_end_index]["close"])
    else:
        raise ValueError("v4r_economic_proxy_exit_reason_invalid")
    if exit_reference <= 0 or not np.isfinite(exit_reference):
        raise ValueError("v4r_economic_proxy_exit_price_invalid")
    return exit_reference


def _proxy_event_gross_return(
    bars: DataFrame,
    event_index: int,
    position: int,
    label_end_index: int,
    exit_reason: object,
    take_profit_prices: np.ndarray,
    stop_loss_prices: np.ndarray,
) -> float | None:
    if event_index + 1 >= len(bars):
        return None
    entry_reference = _proxy_entry_reference(bars, event_index)
    exit_reference = _proxy_exit_reference(
        bars,
        position,
        label_end_index,
        exit_reason,
        take_profit_prices,
        stop_loss_prices,
    )
    return exit_reference / entry_reference - 1.0


def _economic_outcomes_by_index(bars: DataFrame, labels: DataFrame) -> Series:
    """Estimate event gross returns using next-open reference and label barriers.

    Entry slippage is not embedded in gross returns; the scorer deducts the full
    90 bps round-trip fee/slippage estimate. Intrabar outcomes follow the target
    label's stop-first collision rule. Final portfolio acceptance uses replay.
    """
    (
        event_indices,
        label_end_indices,
        take_profit_prices,
        stop_loss_prices,
        target_labels,
        exit_reasons,
    ) = _economic_label_arrays(labels)
    values = np.full(len(bars), np.nan, dtype=float)
    for position, raw_event_index in enumerate(event_indices):
        event_index = int(raw_event_index)
        if target_labels[position] not in LABELS:
            continue
        label_end_index = _validated_proxy_label_end_index(
            event_index, float(label_end_indices[position]), len(bars)
        )
        if label_end_index is None:
            continue
        gross_return = _proxy_event_gross_return(
            bars,
            event_index,
            position,
            label_end_index,
            exit_reasons[position],
            take_profit_prices,
            stop_loss_prices,
        )
        if gross_return is not None:
            values[event_index] = gross_return
    return Series(values, index=bars.index, dtype=float)


def _candidate_identifier(
    arm_name: str,
    pair: str,
    timeframe: str,
    profile: str,
    policy: str,
) -> str:
    return "|".join((arm_name, pair, timeframe, profile, policy))


def _evaluate_candidate(
    *,
    arm: dict[str, Any],
    pair: str,
    timeframe: str,
    bars: DataFrame,
    experiment: dict[str, Any],
    profile: str,
    policy: str,
    variant: dict[str, Any],
) -> dict[str, Any]:
    probabilities = variant.get("validation_probabilities")
    if not isinstance(probabilities, list) or not probabilities:
        raise ValueError("v4r_validation_probabilities_missing")
    if len(probabilities) != int(experiment["validation_rows"]):
        raise ValueError("v4r_validation_probability_count_mismatch")
    candidate_id = _candidate_identifier(
        str(arm["name"]), pair, timeframe, profile, policy
    )
    candidate = {
        "candidate_id": candidate_id,
        "scenario": str(arm["name"]),
        "pair": pair,
        "timeframe": timeframe,
        "feature_profile": profile,
        "weighting_policy": policy,
        "selected_parameters": dict(
            variant["hyperparameter_tuning"]["selected_parameters"]
        ),
        "hold_precision": float(variant["hold_metrics"]["precision"]),
        "hold_recall": float(variant["hold_metrics"]["recall"]),
        "selected_hold_threshold": V4R_HOLD_THRESHOLD,
        "forward_row_count": len(probabilities),
        "probability_records": probabilities,
    }
    replay = evaluate_candidate(
        candidate,
        bars,
        _execution_config(),
        _management_config(arm),
    )
    metrics = dict(replay["metrics"])
    return {
        "candidate_id": candidate_id,
        "status": "TRAINED",
        "evaluation_performed": True,
        "sample_gate": dict(experiment["sample_gate"]),
        "arm": str(arm["name"]),
        "geometry": str(arm["geometry"]),
        "objective": str(arm["objective"]),
        "pair": pair,
        "timeframe": timeframe,
        "feature_profile": profile,
        "weighting_policy": policy,
        "sample_signature_sha256": str(experiment["sample_signature_sha256"]),
        "train_rows": int(experiment["train_rows"]),
        "validation_rows": int(experiment["validation_rows"]),
        "tuning_objective": str(arm["tuning_objective"]),
        "selected_parameters": dict(
            variant["hyperparameter_tuning"]["selected_parameters"]
        ),
        "tuning_diagnostics": dict(variant["hyperparameter_tuning"]),
        "feature_importance_gain": list(variant["feature_importance_gain"]),
        "metrics": metrics,
        "exit_reason_counts": dict(replay["exit_reason_counts"]),
        "raw_signal_counts": dict(replay["raw_signal_counts"]),
        "confidence_filter": dict(replay["confidence_filter"]),
        "total_fees_usd": float(replay["total_fees_usd"]),
        "total_slippage_cost_usd": float(replay["total_slippage_cost_usd"]),
        "trades": list(replay["trades"]),
    }


def _sample_gate_skipped_metrics() -> dict[str, Any]:
    """Return non-performance metrics for a candidate that never trained."""
    return {
        "initial_equity_usd": 10_000.0,
        "final_equity_usd": None,
        "net_return_fraction": None,
        "net_return_percent": None,
        "trade_count": 0,
        "win_rate": None,
        "profit_factor_net_of_costs": None,
        "profit_factor_infinite_no_losses": False,
        "gross_profit_usd": None,
        "gross_loss_usd": None,
        "average_win_usd": None,
        "average_loss_usd": None,
        "average_net_pnl_per_trade_usd": None,
        "max_drawdown_fraction": None,
        "max_drawdown_percent": None,
        "trade_count_gate_passed": False,
        "profit_factor_gate_passed": False,
        "drawdown_gate_passed": False,
        "economic_candidate_gate_passed": False,
        "economic_data_status": "INSUFFICIENT_DATA",
        "maximum_drawdown_limit_fraction": V4R_MAXIMUM_DRAWDOWN,
    }


def _sample_gate_skipped_candidate_record(
    *,
    arm: dict[str, Any],
    pair: str,
    timeframe: str,
    profile: str,
    policy: str,
    experiment: dict[str, Any],
) -> dict[str, Any]:
    """Persist a skipped model slot without fabricating economic performance."""
    status = str(experiment.get("status"))
    sample_gate = experiment.get("sample_gate")
    if status not in V4R_SAMPLE_GATE_SKIP_STATUSES:
        raise ValueError(f"v4r_sample_gate_skip_status_invalid:{status}")
    if not isinstance(sample_gate, dict) or sample_gate.get("status") != status:
        raise ValueError("v4r_sample_gate_skip_diagnostics_mismatch")
    signature = experiment.get("sample_signature_sha256")
    if not isinstance(signature, str) or not signature:
        raise ValueError("v4r_sample_gate_skip_signature_missing")
    return {
        "candidate_id": _candidate_identifier(
            str(arm["name"]), pair, timeframe, profile, policy
        ),
        "status": status,
        "evaluation_performed": False,
        "sample_gate": dict(sample_gate),
        "arm": str(arm["name"]),
        "geometry": str(arm["geometry"]),
        "objective": str(arm["objective"]),
        "pair": pair,
        "timeframe": timeframe,
        "feature_profile": profile,
        "weighting_policy": policy,
        "sample_signature_sha256": signature,
        "train_rows": int(experiment["train_rows"]),
        "validation_rows": int(experiment["validation_rows"]),
        "tuning_objective": str(arm["tuning_objective"]),
        "selected_parameters": None,
        "tuning_diagnostics": {
            "status": status,
            "sample_gate": dict(sample_gate),
        },
        "feature_importance_gain": [],
        "metrics": _sample_gate_skipped_metrics(),
        "exit_reason_counts": {},
        "raw_signal_counts": {},
        "confidence_filter": {},
        "total_fees_usd": 0.0,
        "total_slippage_cost_usd": 0.0,
        "trades": [],
    }


def _validated_experiment_sample_gate(
    experiment: dict[str, Any],
    *,
    pair: str,
    timeframe: str,
    profile: str,
    arm_name: str,
) -> tuple[str, dict[str, Any]]:
    """Accept only trained experiments or explicitly permitted sample skips."""
    status = str(experiment.get("status"))
    sample_gate = experiment.get("sample_gate")
    if not isinstance(sample_gate, dict):
        raise ValueError("v4r_sample_gate_diagnostics_missing")
    if status == "TRAINED":
        if sample_gate.get("status") != "PASS":
            raise ValueError("v4r_trained_experiment_sample_gate_mismatch")
    elif status in V4R_SAMPLE_GATE_SKIP_STATUSES:
        if sample_gate.get("status") != status:
            raise ValueError("v4r_skipped_experiment_sample_gate_mismatch")
        if experiment.get("model_variants") not in (None, {}):
            raise ValueError("v4r_skipped_experiment_has_model_variants")
    else:
        raise ValueError(
            "v4r_experiment_status_out_of_policy:"
            f"{pair}:{timeframe}:{profile}:{arm_name}:{status}"
        )
    return status, sample_gate


def _run_pair_timeframe(
    symbol: str,
    timeframe: str,
    minute_candles: DataFrame,
    *,
    calibration_fraction: float,
    random_search_iterations: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pair = f"{symbol[:-4]}/USDT"
    bars = resample_ohlcv(minute_candles, timeframe)
    if len(bars) < 100:
        raise ValueError(f"v4r_insufficient_resampled_bars:{symbol}:{timeframe}")
    feature_matrices = build_feature_profiles(bars, profile_names=V4R_FEATURE_PROFILES)
    experiments: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for arm in V4R_ARMS:
        labels = apply_adaptive_triple_barrier(
            bars,
            atr_period=V4R_ATR_PERIOD,
            horizon_candles=int(arm["horizon_candles"]),
            take_profit_multiplier=float(arm["take_profit_multiplier"]),
            stop_loss_multiplier=float(arm["stop_loss_multiplier"]),
            entry_reference="NEXT_BAR_OPEN",
            entry_slippage_bps=V4R_ENTRY_SLIPPAGE_BPS,
        )
        labels = _mask_to_common_horizon(labels, len(bars))
        economic_outcomes = _economic_outcomes_by_index(bars, labels)
        for profile in V4R_FEATURE_PROFILES:
            experiment = run_v4h_experiment(
                pair,
                timeframe,
                bars,
                labels,
                feature_matrices,
                scenario=arm,
                feature_profile=profile,
                calibration_fraction=calibration_fraction,
                minimum_hold_precision=DEFAULT_MINIMUM_HOLD_PRECISION,
                minimum_hold_recall=DEFAULT_MINIMUM_HOLD_RECALL,
                random_search_iterations=random_search_iterations,
                tuning_objective=str(arm["tuning_objective"]),
                economic_outcomes=(
                    economic_outcomes
                    if arm["tuning_objective"] == "NET_EXPECTANCY_PROXY"
                    else None
                ),
                estimated_cost_bps=V4R_ESTIMATED_ROUND_TRIP_COST_BPS,
                split_horizon_candles=V4R_COMMON_PURGE_HORIZON,
            )
            status, sample_gate = _validated_experiment_sample_gate(
                experiment,
                pair=pair,
                timeframe=timeframe,
                profile=profile,
                arm_name=str(arm["name"]),
            )
            experiment_record = {
                "arm": str(arm["name"]),
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
            experiments.append(experiment_record)
            if status in V4R_SAMPLE_GATE_SKIP_STATUSES:
                for policy in weighting_policy_specs():
                    records.append(
                        _sample_gate_skipped_candidate_record(
                            arm=arm,
                            pair=pair,
                            timeframe=timeframe,
                            profile=profile,
                            policy=policy,
                            experiment=experiment,
                        )
                    )
                continue
            model_variants = experiment.get("model_variants")
            if not isinstance(model_variants, dict) or set(model_variants) != set(
                weighting_policy_specs()
            ):
                raise ValueError("v4r_trained_model_variant_contract_mismatch")
            for policy, variant in (experiment.get("model_variants") or {}).items():
                record = _evaluate_candidate(
                    arm=arm,
                    pair=pair,
                    timeframe=timeframe,
                    bars=bars,
                    experiment=experiment,
                    profile=profile,
                    policy=policy,
                    variant=variant,
                )
                records.append(record)
                experiment_variant = {
                    key: value
                    for key, value in variant.items()
                    if key != "validation_probabilities"
                }
                experiment_variant["economic_evaluation"] = {
                    "candidate_id": record["candidate_id"],
                    "metrics": record["metrics"],
                    "exit_reason_counts": record["exit_reason_counts"],
                }
                experiments[-1]["model_variants"][policy] = experiment_variant
    return experiments, records, []


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 8) if values else None


def _profit_factor(pnls: list[float]) -> float | None:
    wins = sum(value for value in pnls if value > 0.0)
    losses = abs(sum(value for value in pnls if value < 0.0))
    return round(wins / losses, 8) if losses > 0 else None


def _arm_candidate_count_summary(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_metrics = [record["metrics"] for record in records]
    eligible_count = sum(
        metric["economic_data_status"] == "ELIGIBLE" for metric in candidate_metrics
    )
    return {
        "candidate_count": len(records),
        "trained_candidate_count": sum(
            record.get("status", "TRAINED") == "TRAINED" for record in records
        ),
        "sample_gate_skipped_candidate_count": sum(
            record.get("status") in V4R_SAMPLE_GATE_SKIP_STATUSES for record in records
        ),
        "active_candidate_count": sum(
            int(metric["trade_count"]) > 0 for metric in candidate_metrics
        ),
        "insufficient_data_candidate_count": len(records) - eligible_count,
        "minimum_trade_count_for_acceptance": V4R_MINIMUM_TRADES,
    }


def _arm_trade_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    trades = [trade for record in records for trade in record["trades"]]
    net_pnls = [float(trade["net_pnl_usd"]) for trade in trades]
    gross_pnls = [float(trade["gross_market_pnl_usd"]) for trade in trades]
    total_fees = sum(float(record["total_fees_usd"]) for record in records)
    total_slippage = sum(float(record["total_slippage_cost_usd"]) for record in records)
    return {
        "trade_count": len(trades),
        "winning_trade_count": sum(value > 0 for value in net_pnls),
        "losing_trade_count": sum(value < 0 for value in net_pnls),
        "win_rate": round(
            sum(value > 0 for value in net_pnls) / max(1, len(net_pnls)),
            8,
        ),
        "combined_net_profit_factor": _profit_factor(net_pnls),
        "combined_gross_profit_factor": _profit_factor(gross_pnls),
        "total_gross_market_pnl_usd": round(sum(gross_pnls), 8),
        "total_fees_usd": round(total_fees, 8),
        "total_slippage_cost_usd": round(total_slippage, 8),
        "total_net_pnl_usd": round(sum(net_pnls), 8),
    }


def _arm_candidate_risk_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_metrics = [record["metrics"] for record in records]
    defined_profit_factors = [
        float(metric["profit_factor_net_of_costs"])
        for metric in candidate_metrics
        if metric["profit_factor_net_of_costs"] is not None
    ]
    drawdowns = [
        float(metric["max_drawdown_percent"])
        for metric in candidate_metrics
        if metric["max_drawdown_percent"] is not None
    ]
    return {
        "mean_candidate_profit_factor": _mean(defined_profit_factors),
        "mean_candidate_max_drawdown_percent": _mean(drawdowns),
        "pf_above_1_5_candidate_count": sum(
            bool(metric["profit_factor_gate_passed"]) for metric in candidate_metrics
        ),
        "strict_economic_pass_candidate_count": sum(
            bool(metric["economic_candidate_gate_passed"])
            for metric in candidate_metrics
        ),
    }


def _arm_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        **_arm_candidate_count_summary(records),
        **_arm_trade_summary(records),
        **_arm_candidate_risk_summary(records),
    }


def _paired_comparison(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(
        dict
    )
    for record in records:
        key = (
            str(record["pair"]),
            str(record["timeframe"]),
            str(record["feature_profile"]),
            str(record["weighting_policy"]),
        )
        arm = str(record["arm"])
        if arm in groups[key]:
            raise ValueError(f"v4r_duplicate_paired_candidate:{key}:{arm}")
        groups[key][arm] = record
    expected_arms = {str(arm["name"]) for arm in V4R_ARMS}
    geometry_at_buy: list[float] = []
    geometry_at_net: list[float] = []
    objective_at_symmetric: list[float] = []
    objective_at_asymmetric: list[float] = []
    interaction: list[float] = []
    signatures_checked = 0
    skipped_groups = 0
    for key, arm_records in groups.items():
        if set(arm_records) != expected_arms:
            raise ValueError(f"v4r_paired_arm_missing:{key}")
        signatures = {
            str(record["sample_signature_sha256"]) for record in arm_records.values()
        }
        if len(signatures) != 1:
            raise ValueError(f"v4r_paired_sample_signature_mismatch:{key}")
        signatures_checked += 1
        statuses = {
            str(record.get("status", "TRAINED")) for record in arm_records.values()
        }
        if not statuses.issubset({"TRAINED", *V4R_SAMPLE_GATE_SKIP_STATUSES}):
            raise ValueError(f"v4r_paired_candidate_status_invalid:{key}")
        if statuses != {"TRAINED"}:
            skipped_groups += 1
            continue
        metric = {
            name: float(value["metrics"]["net_return_percent"])
            for name, value in arm_records.items()
        }
        geometry_at_buy.append(
            metric["GEOMETRY_2R_BUY_F1"] - metric["CONTROL_SYMMETRIC_BUY_F1"]
        )
        geometry_at_net.append(
            metric["V4R_2R_NET_EXPECTANCY"] - metric["SYMMETRIC_NET_EXPECTANCY"]
        )
        objective_at_symmetric.append(
            metric["SYMMETRIC_NET_EXPECTANCY"] - metric["CONTROL_SYMMETRIC_BUY_F1"]
        )
        objective_at_asymmetric.append(
            metric["V4R_2R_NET_EXPECTANCY"] - metric["GEOMETRY_2R_BUY_F1"]
        )
        interaction.append(
            (metric["V4R_2R_NET_EXPECTANCY"] - metric["GEOMETRY_2R_BUY_F1"])
            - (metric["SYMMETRIC_NET_EXPECTANCY"] - metric["CONTROL_SYMMETRIC_BUY_F1"])
        )
    expected_groups = V4R_EXPECTED_CANDIDATES // len(V4R_ARMS)
    if len(groups) != expected_groups:
        raise RuntimeError("v4r_paired_group_count_mismatch")
    return {
        "design": "2x2_PAIRED_FACTORIAL",
        "candidate_group_count": len(groups),
        "paired_group_count": len(geometry_at_buy),
        "sample_gate_skipped_group_count": skipped_groups,
        "sample_signature_mismatch_count": 0,
        "common_outer_purge_horizon_candles": V4R_COMMON_PURGE_HORIZON,
        "geometry_effect_at_buy_f1_mean_net_return_delta_pp": _mean(geometry_at_buy),
        "geometry_effect_at_net_expectancy_mean_net_return_delta_pp": _mean(
            geometry_at_net
        ),
        "objective_effect_at_symmetric_geometry_mean_net_return_delta_pp": _mean(
            objective_at_symmetric
        ),
        "objective_effect_at_asymmetric_geometry_mean_net_return_delta_pp": _mean(
            objective_at_asymmetric
        ),
        "geometry_objective_interaction_mean_net_return_delta_pp": _mean(interaction),
        "sample_signature_groups_checked": signatures_checked,
        "interpretation": (
            "Paired effects use only groups where all four models passed their "
            "training sample gate; skipped groups are excluded, not imputed."
        ),
    }


def _comparison_csv_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        metrics = record["metrics"]
        rows.append(
            {
                "candidate_id": record["candidate_id"],
                "candidate_status": record.get("status", "TRAINED"),
                "evaluation_performed": record.get("evaluation_performed", True),
                "arm": record["arm"],
                "geometry": record["geometry"],
                "objective": record["objective"],
                "pair": record["pair"],
                "timeframe": record["timeframe"],
                "feature_profile": record["feature_profile"],
                "weighting_policy": record["weighting_policy"],
                "sample_signature_sha256": record["sample_signature_sha256"],
                "trade_count": metrics["trade_count"],
                "economic_data_status": metrics["economic_data_status"],
                "trade_count_gate_passed": metrics["trade_count_gate_passed"],
                "profit_factor_net_of_costs": metrics["profit_factor_net_of_costs"],
                "profit_factor_infinite_no_losses": metrics[
                    "profit_factor_infinite_no_losses"
                ],
                "profit_factor_gate_passed": metrics["profit_factor_gate_passed"],
                "max_drawdown_percent": metrics["max_drawdown_percent"],
                "net_return_percent": metrics["net_return_percent"],
                "gross_profit_usd": metrics["gross_profit_usd"],
                "gross_loss_usd": metrics["gross_loss_usd"],
                "total_fees_usd": record["total_fees_usd"],
                "total_slippage_cost_usd": record["total_slippage_cost_usd"],
                "net_pnl_usd": (
                    round(
                        sum(float(trade["net_pnl_usd"]) for trade in record["trades"]),
                        8,
                    )
                    if record.get("evaluation_performed", True)
                    else None
                ),
                "economic_candidate_gate_passed": metrics[
                    "economic_candidate_gate_passed"
                ],
            }
        )
    return rows


def _trade_csv_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for index, trade in enumerate(record["trades"], start=1):
            rows.append(
                {
                    "candidate_id": record["candidate_id"],
                    "arm": record["arm"],
                    "pair": record["pair"],
                    "timeframe": record["timeframe"],
                    "feature_profile": record["feature_profile"],
                    "weighting_policy": record["weighting_policy"],
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


def _feature_importance_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for item in record["feature_importance_gain"]:
            rows.append(
                {
                    "candidate_id": record["candidate_id"],
                    "arm": record["arm"],
                    "pair": record["pair"],
                    "timeframe": record["timeframe"],
                    "feature_profile": record["feature_profile"],
                    "weighting_policy": record["weighting_policy"],
                    "feature": str(item["feature"]),
                    "gain": float(item["gain"]),
                }
            )
    return rows


def _render_step_summary(report: dict[str, Any]) -> str:
    lines = [
        "## AliX V4-R Economic Architecture Evaluation",
        "",
        f"Commit SHA: {report['commit_sha']}",
        "",
        "Paired factorial experiment: TP/SL geometry x tuning objective.",
        "Raw 1m datasets are loaded once; 1h and 4h resamples and feature matrices are shared across all four arms.",
        "",
        "| Arm | Geometry | Objective | Slots | Trained | Sample skips | Trades | Net PF | Gross PF | Net PnL USD | Strict passes | Insufficient data |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in report["arms"]:
        aggregate = arm["aggregate"]
        net_pf = aggregate["combined_net_profit_factor"]
        gross_pf = aggregate["combined_gross_profit_factor"]
        lines.append(
            f"| {arm['name']} | {arm['geometry']} | {arm['objective']} "
            f"| {aggregate['candidate_count']} "
            f"| {aggregate['trained_candidate_count']} "
            f"| {aggregate['sample_gate_skipped_candidate_count']} "
            f"| {aggregate['trade_count']} "
            f"| {net_pf if net_pf is not None else 'N/A'} "
            f"| {gross_pf if gross_pf is not None else 'N/A'} "
            f"| {aggregate['total_net_pnl_usd']:.2f} "
            f"| {aggregate['strict_economic_pass_candidate_count']} "
            f"| {aggregate['insufficient_data_candidate_count']} |"
        )
    paired = report["paired_comparison"]
    lines.extend(
        [
            "",
            "### Paired effect estimates",
            "",
            f"Candidate groups with matching sample signatures: "
            f"{paired['candidate_group_count']}.",
            f"Complete paired groups included in effects: {paired['paired_group_count']}.",
            f"Groups excluded by sample gates: {paired['sample_gate_skipped_group_count']}.",
            f"Geometry effect under BUY_F1: {paired['geometry_effect_at_buy_f1_mean_net_return_delta_pp']}.",
            f"Geometry effect under Net Expectancy: {paired['geometry_effect_at_net_expectancy_mean_net_return_delta_pp']}.",
            f"Objective effect under symmetric geometry: {paired['objective_effect_at_symmetric_geometry_mean_net_return_delta_pp']}.",
            f"Objective effect under 2R geometry: {paired['objective_effect_at_asymmetric_geometry_mean_net_return_delta_pp']}.",
            "",
            "### Comparison CSV (exact persisted rows)",
            "",
            "| Arm | Pair | TF | Profile | Weight | Trades | Data status | PF | Max DD | Net Return | Economic Gate |",
            "|---|---|---|---|---|---:|---|---:|---:|---:|---|",
        ]
    )
    for item in report["comparison_rows"]:
        drawdown = item["max_drawdown_percent"]
        net_return = item["net_return_percent"]
        drawdown_text = f"{drawdown:.2f}%" if drawdown is not None else "N/A"
        return_text = f"{net_return:.2f}%" if net_return is not None else "N/A"
        lines.append(
            f"| {item['arm']} | {item['pair']} | {item['timeframe']} "
            f"| {item['feature_profile']} | {item['weighting_policy']} "
            f"| {item['trade_count']} | {item['economic_data_status']} "
            f"| {item['profit_factor_net_of_costs']} "
            f"| {drawdown_text} "
            f"| {return_text} "
            f"| {'PASS' if item['economic_candidate_gate_passed'] else 'NO-GO'} |"
        )
    lines.extend(
        [
            "",
            "### Feature Importance Gain",
            "",
            "| Arm | Pair | TF | Profile | Weight | Feature | Gain |",
            "|---|---|---|---|---|---|---:|",
        ]
    )
    for item in report["feature_importance_rows"]:
        lines.append(
            f"| {item['arm']} | {item['pair']} | {item['timeframe']} "
            f"| {item['feature_profile']} | {item['weighting_policy']} "
            f"| {item['feature']} | {item['gain']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Economics proxy: aggregate BUY gross profit minus gross loss and 90 bps round-trip costs per BUY. It does not model portfolio overlap.",
            "Economic acceptance is based on the separate execution replay, net PF > 1.5, Max DD < 15%, and at least 30 closed trades per candidate.",
            "",
            "Safety: Champion=null, order_permission=false, trading_authorization=NO-GO, private_api_calls=0, real_order_attempts=0.",
            "Sequence models: none. Data fetching: unchanged public Binance Spot 1m fetcher; one market snapshot per symbol per run.",
            "",
            "Promotion: NO-GO pending independent OOS evidence and explicit owner review.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_v4r_experiments(
    datasets: dict[str, DataFrame],
    *,
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION,
    random_search_iterations: int = V4R_RANDOM_SEARCH_ITERATIONS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if random_search_iterations < 1 or random_search_iterations > 27:
        raise ValueError("v4r_random_search_iterations_invalid")
    if set(datasets) != set(SYMBOLS):
        raise ValueError("v4r_dataset_symbol_contract_mismatch")
    experiments: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        for timeframe in V4R_TIMEFRAMES:
            pair_experiments, pair_records, _ = _run_pair_timeframe(
                symbol,
                timeframe,
                datasets[symbol],
                calibration_fraction=calibration_fraction,
                random_search_iterations=random_search_iterations,
            )
            experiments.extend(pair_experiments)
            records.extend(pair_records)
    if len(experiments) != V4R_EXPECTED_EXPERIMENTS:
        raise RuntimeError("v4r_experiment_count_mismatch")
    if len(records) != V4R_EXPECTED_CANDIDATES:
        raise RuntimeError("v4r_candidate_count_mismatch")
    return experiments, records


def _build_report(
    *,
    commit_sha: str,
    provenance: list[dict[str, Any]],
    experiments: list[dict[str, Any]],
    records: list[dict[str, Any]],
    random_search_iterations: int = V4R_RANDOM_SEARCH_ITERATIONS,
) -> dict[str, Any]:
    trained_candidate_count = sum(
        record.get("status", "TRAINED") == "TRAINED" for record in records
    )
    skipped_candidate_count = sum(
        record.get("status") in V4R_SAMPLE_GATE_SKIP_STATUSES for record in records
    )
    trained_experiment_count = sum(
        experiment.get("status") == "TRAINED" for experiment in experiments
    )
    skipped_experiment_count = sum(
        experiment.get("status") in V4R_SAMPLE_GATE_SKIP_STATUSES
        for experiment in experiments
    )
    if trained_candidate_count + skipped_candidate_count != len(records):
        raise RuntimeError("v4r_candidate_status_count_mismatch")
    if trained_experiment_count + skipped_experiment_count != len(experiments):
        raise RuntimeError("v4r_experiment_status_count_mismatch")
    if len(records) != V4R_EXPECTED_CANDIDATES or trained_candidate_count == 0:
        raise RuntimeError("v4r_candidate_slot_or_training_count_mismatch")
    arms: list[dict[str, Any]] = []
    for arm in V4R_ARMS:
        arm_records = [record for record in records if record["arm"] == arm["name"]]
        if len(arm_records) != V4R_EXPECTED_CANDIDATES // V4R_EXPECTED_ARMS:
            raise RuntimeError(f"v4r_arm_candidate_count_mismatch:{arm['name']}")
        arms.append(
            {
                "name": arm["name"],
                "geometry": arm["geometry"],
                "objective": arm["objective"],
                "take_profit_multiplier": arm["take_profit_multiplier"],
                "stop_loss_multiplier": arm["stop_loss_multiplier"],
                "horizon_candles": arm["horizon_candles"],
                "aggregate": _arm_aggregate(arm_records),
            }
        )
    comparison_rows = _comparison_csv_rows(records)
    feature_rows = _feature_importance_rows(records)
    trade_rows = _trade_csv_rows(records)
    return {
        "schema_version": V4R_REPORT_SCHEMA,
        "commit_sha": commit_sha,
        "requested_days": 730,
        "provider": "Binance Spot public klines API",
        "input_datasets": provenance,
        "model_training_performed": trained_candidate_count > 0,
        "economic_evaluation_performed": trained_candidate_count > 0,
        "economic_metrics_reported": trained_candidate_count > 0,
        "market_timeframes": list(V4R_TIMEFRAMES),
        "symbols": list(SYMBOLS),
        "feature_profiles": list(V4R_FEATURE_PROFILES),
        "random_search_iterations": int(random_search_iterations),
        "candidate_fit_count": (
            trained_candidate_count * int(random_search_iterations) * 3
        ),
        "model_count": trained_candidate_count,
        "candidate_count": len(records),
        "sample_gate_skipped_candidate_count": skipped_candidate_count,
        "experiment_count": len(experiments),
        "trained_experiment_count": trained_experiment_count,
        "sample_gate_skipped_experiment_count": skipped_experiment_count,
        "experimental_design": "2x2_PAIRED_FACTORIAL",
        "arms": arms,
        "target_contract": {
            "atr_period": V4R_ATR_PERIOD,
            "primary_geometry": "TP=2.0 ATR; SL=1.0 ATR",
            "primary_horizon_candles": 24,
            "control_geometry": "TP=1.0 ATR; SL=1.0 ATR",
            "control_horizon_candles": 12,
            "label_entry_reference": "NEXT_BAR_OPEN",
            "entry_slippage_bps": V4R_ENTRY_SLIPPAGE_BPS,
            "atr_reference": "SIGNAL_CANDLE_CLOSE_ONLY",
            "same_candle_collision_policy": "STOP_LOSS_WINS",
            "common_outer_and_inner_purge_horizon_candles": V4R_COMMON_PURGE_HORIZON,
            "common_incomplete_tail_policy": "LAST_24_CANDLES_UNLABELED_IN_ALL_ARMS",
        },
        "objective_contract": {
            "primary_objective": "NET_EXPECTANCY_PROXY",
            "scorer": "custom_eval_metric",
            "cost_estimate_round_trip_bps": V4R_ESTIMATED_ROUND_TRIP_COST_BPS,
            "scorer_scope": "PURGED_INNER_CV_FOLDS_WITHIN_OUTER_TRAIN",
            "portfolio_overlap_modeled": False,
            "economic_acceptance_authority": "COSTED_OUTER_VALIDATION_EXECUTION_REPLAY",
        },
        "execution_cost_contract": {
            "fee_bps_per_side": V4R_FEE_BPS_PER_SIDE,
            "slippage_bps_per_side": V4R_SLIPPAGE_BPS_PER_SIDE,
            "round_trip_estimated_cost_bps": V4R_ESTIMATED_ROUND_TRIP_COST_BPS,
            "position_fraction": 1.0,
            "pyramiding": False,
        },
        "minimum_trade_count_for_economic_acceptance": V4R_MINIMUM_TRADES,
        "strict_economic_gate": {
            "profit_factor_net_of_costs": ">1.5",
            "maximum_drawdown_percent": "<15",
            "minimum_trade_count": V4R_MINIMUM_TRADES,
            "infinite_pf_below_minimum_sample": "INSUFFICIENT_DATA_AND_NOT_A_PASS",
        },
        "paired_comparison": _paired_comparison(records),
        "experiments": experiments,
        "candidates": [
            {
                key: value
                for key, value in record.items()
                if key
                not in {"trades", "feature_importance_gain", "tuning_diagnostics"}
            }
            for record in records
        ],
        "comparison_rows": comparison_rows,
        "feature_importance_rows": feature_rows,
        "trade_rows": trade_rows,
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
        },
    }


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str] | None = None,
) -> None:
    columns = fieldnames or (list(rows[0]) if rows else None)
    if not columns:
        raise ValueError("v4r_csv_fieldnames_missing")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_outputs(
    report: dict[str, Any],
    report_path: Path,
    comparison_path: Path,
    trades_path: Path,
    feature_importance_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(comparison_path, report["comparison_rows"])
    _write_csv(
        trades_path,
        report["trade_rows"],
        fieldnames=[
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
        ],
    )
    _write_csv(feature_importance_path, report["feature_importance_rows"])
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as summary:
            summary.write(_render_step_summary(report))


def run_v4r(
    input_dir: Path,
    report_path: Path,
    comparison_path: Path,
    trades_path: Path,
    feature_importance_path: Path,
    *,
    days: int,
    commit_sha: str,
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION,
    random_search_iterations: int = V4R_RANDOM_SEARCH_ITERATIONS,
) -> dict[str, Any]:
    if days != 730:
        raise ValueError("v4r_dataset_window_must_be_730_days")
    datasets, provenance = _load_diagnostic_datasets(input_dir, days, commit_sha)
    experiments, records = run_v4r_experiments(
        datasets,
        calibration_fraction=calibration_fraction,
        random_search_iterations=random_search_iterations,
    )
    report = _build_report(
        commit_sha=commit_sha,
        provenance=provenance,
        experiments=experiments,
        records=records,
        random_search_iterations=random_search_iterations,
    )
    _write_outputs(
        report,
        report_path,
        comparison_path,
        trades_path,
        feature_importance_path,
    )
    print(
        json.dumps(
            {
                "report_path": str(report_path),
                "commit_sha": commit_sha,
                "experiment_count": report["experiment_count"],
                "model_count": report["model_count"],
                "candidate_fit_count": report["candidate_fit_count"],
                "paired_group_count": report["paired_comparison"]["paired_group_count"],
                "promotion_status": report["promotion_status"],
                "Champion": None,
                "order_permission": False,
                "trading_authorization": "NO-GO",
                "private_api_calls": 0,
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
    parser.add_argument("--output-feature-importance-csv", type=Path, required=True)
    parser.add_argument("--days", type=int, choices=(730,), default=730)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument(
        "--random-search-iterations",
        type=int,
        default=V4R_RANDOM_SEARCH_ITERATIONS,
    )
    args = parser.parse_args()
    report = run_v4r(
        args.input_dir,
        args.output_report,
        args.output_comparison_csv,
        args.output_trades_csv,
        args.output_feature_importance_csv,
        days=args.days,
        commit_sha=args.commit_sha,
        random_search_iterations=args.random_search_iterations,
    )
    if report["experiment_count"] != V4R_EXPECTED_EXPERIMENTS:
        raise RuntimeError("v4r_experiment_count_mismatch")
    if report["candidate_count"] != V4R_EXPECTED_CANDIDATES:
        raise RuntimeError("v4r_candidate_count_mismatch")
    if (
        report["model_count"] + report["sample_gate_skipped_candidate_count"]
        != (report["candidate_count"])
    ):
        raise RuntimeError("v4r_model_and_sample_gate_count_mismatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
