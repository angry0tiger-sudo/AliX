"""V4-P LightGBM retraining and costed 1-ATR Triple-Barrier evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from pandas import DataFrame

from scripts.run_ml_core_real_data_baseline import (
    DEFAULT_CALIBRATION_FRACTION,
    DEFAULT_MINIMUM_HOLD_PRECISION,
    DEFAULT_MINIMUM_HOLD_RECALL,
    LABELS,
    SYMBOLS,
    TIMEFRAME_MINUTES,
    _load_diagnostic_datasets,
    resample_ohlcv,
)
from scripts.run_ml_core_v4h_tuning_geometry import (
    V4H_FEATURE_PROFILES,
    run_v4h_experiment,
)
from services.freqai_baseline.features import FEATURE_SCHEMA_VERSION
from services.ml_core.adaptive_targets import apply_adaptive_triple_barrier
from services.ml_core.economic_backtest import (
    ExecutionConfig,
    TradeManagementConfig,
    evaluate_candidate,
)
from services.ml_core.extended_features import (
    EXTENDED_FEATURE_SCHEMA_VERSION,
    build_feature_profiles,
    feature_profile_manifest,
)

V4P_REPORT_SCHEMA = "alix.ml-core.v4p-economic-evaluation.v1"
V4P_SCENARIO = {
    "name": "V4P_1ATR_SYMMETRIC",
    "horizon_candles": 12,
    "take_profit_multiplier": 1.0,
    "stop_loss_multiplier": 1.0,
}
V4P_FEATURES = (
    "%-rsi_14",
    "%-macd_histogram",
    "%-volume_zscore_20",
    "%-atr_percentile_252",
)
V4P_CONFIDENCE_THRESHOLD = 0.60
V4P_HOLD_PROBABILITY_THRESHOLD = 0.60
V4P_ATR_PERIOD = 14
V4P_EXPECTED_EXPERIMENTS = 16
V4P_EXPECTED_MODELS = 32
V4P_RANDOM_SEARCH_ITERATIONS = 9
V4P_INITIAL_EQUITY_USD = 10_000.0
V4P_FEE_BPS_PER_SIDE = 25.0
V4P_SLIPPAGE_BPS_PER_SIDE = 20.0
V4P_POSITION_FRACTION = 1.0
V4P_MAXIMUM_DRAWDOWN = 0.15


def _execution_config() -> ExecutionConfig:
    """Return the fixed costed Spot Long-only one-times execution assumptions."""
    return ExecutionConfig(
        initial_equity=V4P_INITIAL_EQUITY_USD,
        fee_bps=V4P_FEE_BPS_PER_SIDE,
        slippage_bps=V4P_SLIPPAGE_BPS_PER_SIDE,
        position_fraction=V4P_POSITION_FRACTION,
        maximum_drawdown=V4P_MAXIMUM_DRAWDOWN,
    )


def _trade_management_config() -> TradeManagementConfig:
    """Return the symmetric one-ATR barriers and fixed 0.60 entry confidence."""
    return TradeManagementConfig(
        atr_period=V4P_ATR_PERIOD,
        take_profit_atr_multiplier=1.0,
        stop_loss_atr_multiplier=1.0,
        horizon_candles=12,
        confidence_threshold=V4P_CONFIDENCE_THRESHOLD,
        trailing_stop_enabled=False,
    )


def _candidate_id(
    pair: str, timeframe: str, feature_profile: str, weighting_policy: str
) -> str:
    """Build a stable identifier for one independently trained model."""
    return "|".join(
        ("V4P_1ATR_SYMMETRIC", pair, timeframe, feature_profile, weighting_policy)
    )


def _trade_csv_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build an auditable row per executed Spot trade."""
    rows: list[dict[str, Any]] = []
    for item in records:
        for trade_index, trade in enumerate(item["trades"], start=1):
            rows.append(
                {
                    "candidate_id": item["candidate_id"],
                    "pair": item["pair"],
                    "timeframe": item["timeframe"],
                    "feature_profile": item["feature_profile"],
                    "weighting_policy": item["weighting_policy"],
                    "trade_index": trade_index,
                    "entry_timestamp": trade.get("entry_timestamp"),
                    "exit_timestamp": trade.get("exit_timestamp"),
                    "entry_fill_price": trade.get("entry_fill_price"),
                    "atr_value": trade.get("atr_value"),
                    "bars_held": trade.get("bars_held"),
                    "net_pnl_usd": trade.get("net_pnl_usd"),
                    "mfe_atr": trade.get("mfe_atr"),
                    "mae_atr": trade.get("mae_atr"),
                    "exit_reason": trade.get("exit_reason"),
                }
            )
    return rows


def _evaluate_model_candidate(
    pair: str,
    timeframe: str,
    bars: DataFrame,
    profile: str,
    policy: str,
    experiment: dict[str, Any],
    variant: dict[str, Any],
) -> dict[str, Any]:
    """Replay new model probabilities with fixed V4-P thresholds and barriers."""
    probability_records = variant.get("validation_probabilities")
    if not isinstance(probability_records, list) or not probability_records:
        raise ValueError("v4p_validation_probabilities_missing")
    if len(probability_records) != int(experiment["validation_rows"]):
        raise ValueError("v4p_validation_probability_count_mismatch")
    hold_metrics = variant["hold_metrics"]
    candidate = {
        "candidate_id": _candidate_id(pair, timeframe, profile, policy),
        "scenario": V4P_SCENARIO["name"],
        "pair": pair,
        "timeframe": timeframe,
        "feature_profile": profile,
        "weighting_policy": policy,
        "selected_parameters": dict(
            variant["hyperparameter_tuning"]["selected_parameters"]
        ),
        "hold_precision": float(hold_metrics["precision"]),
        "hold_recall": float(hold_metrics["recall"]),
        "selected_hold_threshold": V4P_HOLD_PROBABILITY_THRESHOLD,
        "forward_row_count": len(probability_records),
        "probability_records": probability_records,
    }
    replay = evaluate_candidate(
        candidate,
        bars,
        _execution_config(),
        _trade_management_config(),
    )
    return {
        "candidate_id": candidate["candidate_id"],
        "pair": pair,
        "timeframe": timeframe,
        "feature_profile": profile,
        "weighting_policy": policy,
        "feature_count": int(experiment["feature_count"]),
        "train_rows": int(experiment["train_rows"]),
        "validation_rows": int(experiment["validation_rows"]),
        "confidence_threshold": V4P_CONFIDENCE_THRESHOLD,
        "hold_probability_threshold": V4P_HOLD_PROBABILITY_THRESHOLD,
        "trade_count": int(replay["metrics"]["trade_count"]),
        "metrics": replay["metrics"],
        "exit_reason_counts": replay["exit_reason_counts"],
        "raw_signal_counts": replay["raw_signal_counts"],
        "confidence_filter": replay["confidence_filter"],
        "total_fees_usd": float(replay["total_fees_usd"]),
        "total_slippage_cost_usd": float(replay["total_slippage_cost_usd"]),
        "trades": replay["trades"],
        "trade_management": replay["trade_management"],
    }


def _run_pair_timeframe(
    symbol: str,
    timeframe: str,
    minute_candles: DataFrame,
    *,
    calibration_fraction: float,
    random_search_iterations: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Train both approved feature profiles for one market context."""
    pair = f"{symbol[:-4]}/USDT"
    bars = resample_ohlcv(minute_candles, timeframe)
    if len(bars) < 100:
        raise ValueError(f"v4p_insufficient_resampled_bars:{symbol}:{timeframe}")
    matrices = build_feature_profiles(bars, profile_names=V4H_FEATURE_PROFILES)
    labels = apply_adaptive_triple_barrier(
        bars,
        horizon_candles=12,
        take_profit_multiplier=1.0,
        stop_loss_multiplier=1.0,
    )
    experiments: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for profile in V4H_FEATURE_PROFILES:
        experiment = run_v4h_experiment(
            pair,
            timeframe,
            bars,
            labels,
            matrices,
            scenario=V4P_SCENARIO,
            feature_profile=profile,
            calibration_fraction=calibration_fraction,
            minimum_hold_precision=DEFAULT_MINIMUM_HOLD_PRECISION,
            minimum_hold_recall=DEFAULT_MINIMUM_HOLD_RECALL,
            random_search_iterations=random_search_iterations,
        )
        if experiment.get("status") != "TRAINED":
            raise ValueError(
                f"v4p_model_training_skipped:{pair}:{timeframe}:{profile}:"
                + str(experiment.get("status"))
            )
        variants = experiment.get("model_variants") or {}
        if set(variants) != {"weight_none", "weight_balanced"}:
            raise ValueError("v4p_model_variant_contract_mismatch")
        for policy, variant in variants.items():
            record = _evaluate_model_candidate(
                pair, timeframe, bars, profile, policy, experiment, variant
            )
            records.append(record)
            variant["economic_evaluation"] = {
                key: value for key, value in record.items() if key != "trades"
            }
            variant.pop("validation_probabilities", None)
        experiments.append(experiment)
    return experiments, records


def run_v4p_experiments(
    datasets: dict[str, DataFrame],
    *,
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION,
    random_search_iterations: int = V4P_RANDOM_SEARCH_ITERATIONS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Train 2 profiles x 2 weightings x 8 pair/timeframe contexts from scratch."""
    if random_search_iterations < 1 or random_search_iterations > 27:
        raise ValueError("v4p_random_search_iterations_invalid")
    if set(datasets) != set(SYMBOLS):
        raise ValueError("v4p_dataset_symbol_contract_mismatch")
    experiments: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAME_MINUTES:
            context_experiments, context_records = _run_pair_timeframe(
                symbol,
                timeframe,
                datasets[symbol],
                calibration_fraction=calibration_fraction,
                random_search_iterations=random_search_iterations,
            )
            experiments.extend(context_experiments)
            records.extend(context_records)
    if len(experiments) != V4P_EXPECTED_EXPERIMENTS:
        raise RuntimeError("v4p_experiment_count_mismatch")
    if len(records) != V4P_EXPECTED_MODELS:
        raise RuntimeError("v4p_economic_candidate_count_mismatch")
    return experiments, records


def _mean(values: list[float]) -> float | None:
    """Return a rounded mean for a non-empty numeric sample."""
    return round(sum(values) / len(values), 8) if values else None


def _aggregate_economics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate independent candidate replays without merging their equity curves."""
    trades = [trade for item in records for trade in item["trades"]]
    pnls = [float(item["net_pnl_usd"]) for item in trades]
    wins = [value for value in pnls if value > 0.0]
    losses = [value for value in pnls if value < 0.0]
    gross_loss = abs(sum(losses))
    pooled_pf = sum(wins) / gross_loss if gross_loss > 0.0 else None
    defined_pf: list[float] = []
    active_dd: list[float] = []
    all_dd: list[float] = []
    pf_one = 0
    pf_one_five = 0
    preliminary = 0
    strict = 0
    exits: Counter[str] = Counter()
    for item in records:
        metrics = item["metrics"]
        trade_count = int(metrics["trade_count"])
        drawdown = float(metrics["max_drawdown_percent"])
        all_dd.append(drawdown)
        if trade_count:
            active_dd.append(drawdown)
        pf = metrics["profit_factor_net_of_costs"]
        infinite = bool(metrics["profit_factor_infinite_no_losses"])
        if pf is not None:
            defined_pf.append(float(pf))
        above_one = infinite or (pf is not None and float(pf) > 1.0)
        above_one_five = infinite or (pf is not None and float(pf) > 1.5)
        pf_one += int(above_one)
        pf_one_five += int(above_one_five)
        preliminary += int(above_one and drawdown < 15.0)
        strict += int(above_one_five and drawdown < 15.0)
        exits.update(
            {
                str(reason): int(count)
                for reason, count in item["exit_reason_counts"].items()
            }
        )
    return {
        "candidate_model_count": len(records),
        "active_candidate_model_count": len(active_dd),
        "total_trade_count": len(trades),
        "winning_trade_count": len(wins),
        "losing_trade_count": len(losses),
        "flat_trade_count": len(trades) - len(wins) - len(losses),
        "win_rate": round(len(wins) / (len(wins) + len(losses)), 8)
        if wins or losses
        else 0.0,
        "combined_ledger_profit_factor": round(pooled_pf, 8)
        if pooled_pf is not None
        else None,
        "mean_defined_candidate_profit_factor": _mean(defined_pf),
        "defined_candidate_profit_factor_count": len(defined_pf),
        "pf_above_1_0_candidate_count": pf_one,
        "pf_above_1_5_candidate_count": pf_one_five,
        "mean_max_drawdown_percent_active_candidates": _mean(active_dd),
        "mean_max_drawdown_percent_all_candidates": _mean(all_dd),
        "worst_candidate_max_drawdown_percent": round(max(all_dd), 8)
        if all_dd
        else None,
        "preliminary_pf_above_1_and_dd_below_15_candidate_count": preliminary,
        "strict_pf_above_1_5_and_dd_below_15_candidate_count": strict,
        "exit_reason_counts": dict(sorted(exits.items())),
        "take_profit_exit_count": int(exits.get("TP_HIT", 0)),
        "stop_loss_exit_count": int(exits.get("SL_HIT", 0)),
        "collision_exit_count": int(exits.get("COLLISION", 0)),
        "total_fees_usd": round(
            sum(float(item["total_fees_usd"]) for item in records), 8
        ),
        "total_slippage_cost_usd": round(
            sum(float(item["total_slippage_cost_usd"]) for item in records), 8
        ),
        "aggregation_caveat": (
            "Combined-ledger PF pools net trade outcomes from separate model replays; "
            "drawdowns are per-candidate metrics, not one investable portfolio."
        ),
    }


def _pooled_label_distribution(
    experiments: list[dict[str, Any]], split_name: str
) -> dict[str, Any]:
    """Pool one Baseline label distribution per market context."""
    counts = {label: 0 for label in LABELS}
    for experiment in experiments:
        if experiment["feature_profile"] != "Baseline":
            continue
        classes = experiment["label_distribution"][split_name]["classes"]
        for label in LABELS:
            counts[label] += int(classes[label]["count"])
    total = sum(counts.values())
    return {
        "total": total,
        "classes": {
            label: {
                "count": counts[label],
                "percent": round(100.0 * counts[label] / total, 4) if total else 0.0,
            }
            for label in LABELS
        },
    }


def _candidate_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove trade payloads from the JSON candidate summary."""
    return [
        {key: value for key, value in record.items() if key != "trades"}
        for record in records
    ]


def _build_report(
    *,
    commit_sha: str,
    provenance: list[dict[str, Any]],
    experiments: list[dict[str, Any]],
    economic_records: list[dict[str, Any]],
    random_search_iterations: int,
) -> dict[str, Any]:
    """Build target, feature, training, economic, and safety evidence."""
    trained_experiments = sum(item.get("status") == "TRAINED" for item in experiments)
    trained_models = sum(len(item.get("model_variants") or {}) for item in experiments)
    fit_count = sum(
        int(variant["hyperparameter_tuning"]["candidate_fit_count"])
        for experiment in experiments
        for variant in (experiment.get("model_variants") or {}).values()
    )
    aggregate = _aggregate_economics(economic_records)
    return {
        "schema_version": V4P_REPORT_SCHEMA,
        "commit_sha": commit_sha,
        "diagnostic_scope": "NEW_LIGHTGBM_TRAINING_AND_COSTED_OUTER_VALIDATION",
        "training_scope": "PURGED_CHRONOLOGICAL_OUTER_TRAIN_WITH_INNER_CV",
        "model_training_performed": True,
        "economic_evaluation_performed": True,
        "economic_metrics_reported": True,
        "requested_days": 730,
        "provider": "Binance Spot public klines API",
        "input_datasets": provenance,
        "symbols": list(SYMBOLS),
        "timeframes": list(TIMEFRAME_MINUTES),
        "labels": list(LABELS),
        "target_contract": {
            "atr_period": V4P_ATR_PERIOD,
            "horizon_candles": 12,
            "take_profit_multiplier": 1.0,
            "stop_loss_multiplier": 1.0,
            "label_entry_reference": "SIGNAL_CANDLE_CLOSE",
            "economic_barrier_reference": "ACTUAL_NEXT_BAR_ENTRY_FILL",
            "same_candle_collision_policy": "STOP_LOSS_WINS",
            "incomplete_horizon_policy": "UNLABELED",
        },
        "feature_engineering": {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "extended_feature_schema_version": EXTENDED_FEATURE_SCHEMA_VERSION,
            "profiles": feature_profile_manifest(V4H_FEATURE_PROFILES),
            "new_features": list(V4P_FEATURES),
            "point_in_time": True,
            "centered_windows": False,
            "backfill_used": False,
            "future_rows_used": False,
            "target_columns_in_feature_matrix": False,
        },
        "model": {
            "family": "LightGBM LGBMClassifier",
            "objective": "multiclass",
            "new_models_trained": trained_models,
            "n_estimators": 80,
            "max_depth": 6,
            "random_state": 42,
            "deterministic": True,
            "class_weight_policies": ["weight_none", "weight_balanced"],
            "sequence_model_used": False,
            "excluded_model_families": [
                "LSTM",
                "RNN",
                "Transformer",
                "sequence-based",
            ],
        },
        "hyperparameter_tuning": {
            "method": "RANDOM_SEARCH",
            "candidate_count_per_model": random_search_iterations,
            "search_space_size": 27,
            "inner_cv_method": "PURGED_EXPANDING_TIME_SERIES",
            "inner_cv_fold_count": 3,
            "inner_cv_scope": "OUTER_TRAIN_ONLY",
            "validation_accessed_during_search": False,
            "final_refit_scope": "FULL_OUTER_TRAIN_ONLY",
            "model_count": trained_models,
            "candidate_fit_count": fit_count,
            "final_refit_count": trained_models,
        },
        "validation_design": {
            "method": "PURGED_CHRONOLOGICAL_OUTER_HOLDOUT",
            "outer_validation_start_fraction": 0.70,
            "outer_validation_end_fraction": 0.85,
            "label_window_purge": "ACTUAL_LABEL_END_INDEX_OVERLAP",
            "pre_validation_embargo_candles": 12,
            "classification_hold_calibration_fraction": DEFAULT_CALIBRATION_FRACTION,
            "economic_replay_scope": "FULL_OUTER_VALIDATION_ONLY",
            "economic_replay_uses_calibrated_hold_threshold": False,
            "economic_replay_selection_threshold": V4P_HOLD_PROBABILITY_THRESHOLD,
            "economic_replay_entry_confidence_threshold": V4P_CONFIDENCE_THRESHOLD,
        },
        "label_distribution": {
            "train": _pooled_label_distribution(experiments, "train"),
            "outer_validation": _pooled_label_distribution(experiments, "validation"),
            "pooling_policy": "One Baseline experiment per pair/timeframe.",
        },
        "economic_evaluation": {
            "scope": "COSTED_SPOT_LONG_ONLY_OUTER_VALIDATION",
            "candidate_count": len(economic_records),
            "expected_candidate_count": V4P_EXPECTED_MODELS,
            "all_candidates_evaluated": len(economic_records) == V4P_EXPECTED_MODELS,
            "confidence_threshold": V4P_CONFIDENCE_THRESHOLD,
            "hold_probability_threshold": V4P_HOLD_PROBABILITY_THRESHOLD,
            "target_geometry": {
                "atr_period": V4P_ATR_PERIOD,
                "take_profit_atr_multiplier": 1.0,
                "stop_loss_atr_multiplier": 1.0,
                "horizon_candles": 12,
            },
            "execution_assumptions": {
                "initial_equity_usd_per_candidate": V4P_INITIAL_EQUITY_USD,
                "fee_bps_per_side": V4P_FEE_BPS_PER_SIDE,
                "slippage_bps_per_side": V4P_SLIPPAGE_BPS_PER_SIDE,
                "round_trip_nominal_cost_bps": 2.0
                * (V4P_FEE_BPS_PER_SIDE + V4P_SLIPPAGE_BPS_PER_SIDE),
                "position_fraction": V4P_POSITION_FRACTION,
                "leverage": 1.0,
                "pyramiding": False,
                "short_positions_supported": False,
                "one_concurrent_position_per_candidate": True,
                "entry_timing": "NEXT_BAR_OPEN_AFTER_SIGNAL_BAR_CLOSE",
                "exit_timing": "ATR_BARRIER_OR_SELL_OR_12_CANDLE_VERTICAL_BARRIER",
            },
            "aggregate": aggregate,
            "candidates": _candidate_summaries(economic_records),
        },
        "experiment_count": len(experiments),
        "trained_experiments": trained_experiments,
        "trained_models": trained_models,
        "expected_experiments": V4P_EXPECTED_EXPERIMENTS,
        "expected_models": V4P_EXPECTED_MODELS,
        "experiments": experiments,
        "acceptance_policy": {
            "preliminary_profit_factor": ">1.0",
            "strict_profit_factor": ">1.5",
            "maximum_drawdown_percent": "<15.0",
            "preliminary_pass_count": aggregate[
                "preliminary_pf_above_1_and_dd_below_15_candidate_count"
            ],
            "strict_pass_count": aggregate[
                "strict_pf_above_1_5_and_dd_below_15_candidate_count"
            ],
        },
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


def _comparison_csv_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten model-level PF, drawdown, and exit statistics for CSV."""
    rows: list[dict[str, Any]] = []
    for item in records:
        metrics = item["metrics"]
        exits = item["exit_reason_counts"]
        rows.append(
            {
                "candidate_id": item["candidate_id"],
                "pair": item["pair"],
                "timeframe": item["timeframe"],
                "feature_profile": item["feature_profile"],
                "weighting_policy": item["weighting_policy"],
                "feature_count": item["feature_count"],
                "train_rows": item["train_rows"],
                "validation_rows": item["validation_rows"],
                "confidence_threshold": item["confidence_threshold"],
                "hold_probability_threshold": item["hold_probability_threshold"],
                "take_profit_atr_multiplier": 1.0,
                "stop_loss_atr_multiplier": 1.0,
                "horizon_candles": 12,
                "trade_count": metrics["trade_count"],
                "profit_factor": metrics["profit_factor_net_of_costs"],
                "profit_factor_infinite_no_losses": metrics[
                    "profit_factor_infinite_no_losses"
                ],
                "max_drawdown_percent": metrics["max_drawdown_percent"],
                "win_rate": metrics["win_rate"],
                "net_return_percent": metrics["net_return_percent"],
                "tp_exit_count": exits.get("TP_HIT", 0),
                "sl_exit_count": exits.get("SL_HIT", 0),
                "collision_exit_count": exits.get("COLLISION", 0),
                "sell_exit_count": exits.get("SELL", 0),
                "horizon_exit_count": exits.get("HORIZON_EXPIRY", 0),
                "total_fees_usd": item["total_fees_usd"],
                "total_slippage_cost_usd": item["total_slippage_cost_usd"],
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write a deterministic UTF-8 CSV, including a header for empty ledgers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_outputs(
    report: dict[str, Any],
    report_path: Path,
    comparison_path: Path,
    trades_path: Path,
    economic_records: list[dict[str, Any]],
) -> None:
    """Persist the JSON report and two auditable CSV files."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    comparison_fields = [
        "candidate_id",
        "pair",
        "timeframe",
        "feature_profile",
        "weighting_policy",
        "feature_count",
        "train_rows",
        "validation_rows",
        "confidence_threshold",
        "hold_probability_threshold",
        "take_profit_atr_multiplier",
        "stop_loss_atr_multiplier",
        "horizon_candles",
        "trade_count",
        "profit_factor",
        "profit_factor_infinite_no_losses",
        "max_drawdown_percent",
        "win_rate",
        "net_return_percent",
        "tp_exit_count",
        "sl_exit_count",
        "collision_exit_count",
        "sell_exit_count",
        "horizon_exit_count",
        "total_fees_usd",
        "total_slippage_cost_usd",
    ]
    trade_fields = [
        "candidate_id",
        "pair",
        "timeframe",
        "feature_profile",
        "weighting_policy",
        "trade_index",
        "entry_timestamp",
        "exit_timestamp",
        "entry_fill_price",
        "atr_value",
        "bars_held",
        "net_pnl_usd",
        "mfe_atr",
        "mae_atr",
        "exit_reason",
    ]
    _write_csv(
        comparison_path, _comparison_csv_rows(economic_records), comparison_fields
    )
    _write_csv(trades_path, _trade_csv_rows(economic_records), trade_fields)


def _render_markdown(report: dict[str, Any]) -> str:
    """Render compact economics, label balance, and exit-reason evidence."""
    economic = report["economic_evaluation"]["aggregate"]
    labels = report["label_distribution"]["outer_validation"]["classes"]
    lines = [
        "## AliX V4-P Economic Evaluation",
        "",
        f"Commit SHA: {report['commit_sha']}",
        "Target: TP=1.0 ATR, SL=1.0 ATR, Horizon=12 candles.",
        "Model: LightGBM only; LSTM and sequence-based models excluded.",
        "Features: RSI14, MACD Histogram, Volume Z-Score(20), ATR Percentile(252).",
        "Confidence floor: 0.60; HOLD probability threshold: 0.60.",
        "",
        "| Outer Validation label | Count | Percent |",
        "|---|---:|---:|",
    ]
    for label in LABELS:
        lines.append(
            f"| {label} | {labels[label]['count']} | {labels[label]['percent']:.4f}% |"
        )
    lines.extend(
        [
            "",
            "| Economic metric | Result |",
            "|---|---:|",
            f"| Trained models | {report['trained_models']} |",
            f"| Executed trades | {economic['total_trade_count']} |",
            f"| Combined-ledger PF | {economic['combined_ledger_profit_factor']} |",
            f"| Mean defined model PF | {economic['mean_defined_candidate_profit_factor']} |",
            f"| Active mean Max DD | {economic['mean_max_drawdown_percent_active_candidates']}% |",
            f"| Worst candidate Max DD | {economic['worst_candidate_max_drawdown_percent']}% |",
            f"| Models with PF > 1.0 | {economic['pf_above_1_0_candidate_count']} |",
            f"| Models with PF > 1.5 | {economic['pf_above_1_5_candidate_count']} |",
            f"| Preliminary passes (PF > 1, DD < 15%) | {economic['preliminary_pf_above_1_and_dd_below_15_candidate_count']} |",
            f"| Strict passes (PF > 1.5, DD < 15%) | {economic['strict_pf_above_1_5_and_dd_below_15_candidate_count']} |",
            "",
            "| Exit reason | Count |",
            "|---|---:|",
        ]
    )
    for reason, count in economic["exit_reason_counts"].items():
        lines.append(f"| {reason} | {count} |")
    lines.extend(
        [
            "",
            f"Fees: {chr(36)}{economic['total_fees_usd']}; slippage: {chr(36)}{economic['total_slippage_cost_usd']}.",
            "Combined-ledger PF pools separate model replays and is descriptive, not one investable portfolio.",
            "",
            "Safety: Champion=null, order_permission=false, trading_authorization=NO-GO, "
            "private_api_calls=0, real_order_attempts=0.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_v4p_diagnostic(
    input_dir: Path,
    output_report: Path,
    output_comparison_csv: Path,
    output_trades_csv: Path,
    *,
    days: int,
    commit_sha: str,
    random_search_iterations: int = V4P_RANDOM_SEARCH_ITERATIONS,
) -> dict[str, Any]:
    """Train from validated 730-day public data and create costed evaluation artifacts."""
    if days != 730:
        raise ValueError("v4p_requires_exactly_730_days")
    datasets, provenance = _load_diagnostic_datasets(input_dir, days, commit_sha)
    experiments, economic_records = run_v4p_experiments(
        datasets, random_search_iterations=random_search_iterations
    )
    trained_models = sum(len(item.get("model_variants") or {}) for item in experiments)
    if trained_models != V4P_EXPECTED_MODELS:
        raise RuntimeError("v4p_expected_all_lightgbm_models_to_train")
    if len(economic_records) != V4P_EXPECTED_MODELS:
        raise RuntimeError("v4p_expected_all_models_to_be_economically_evaluated")
    report = _build_report(
        commit_sha=commit_sha,
        provenance=provenance,
        experiments=experiments,
        economic_records=economic_records,
        random_search_iterations=random_search_iterations,
    )
    _write_outputs(
        report,
        output_report,
        output_comparison_csv,
        output_trades_csv,
        economic_records,
    )
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as summary:
            summary.write(_render_markdown(report))
    print(
        json.dumps(
            {
                "commit_sha": commit_sha,
                "report_path": str(output_report),
                "comparison_csv": str(output_comparison_csv),
                "trade_ledger_csv": str(output_trades_csv),
                "experiments": report["experiment_count"],
                "trained_models": report["trained_models"],
                "candidate_fit_count": report["hyperparameter_tuning"][
                    "candidate_fit_count"
                ],
                "economic_aggregate": report["economic_evaluation"]["aggregate"],
                "Champion": None,
                "order_permission": False,
                "trading_authorization": "NO-GO",
            },
            sort_keys=True,
        )
    )
    return report


def main() -> int:
    """Run the V4-P training and economic evaluation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-comparison-csv", type=Path, required=True)
    parser.add_argument("--output-trades-csv", type=Path, required=True)
    parser.add_argument("--days", type=int, choices=(730,), default=730)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument(
        "--random-search-iterations",
        type=int,
        default=V4P_RANDOM_SEARCH_ITERATIONS,
    )
    args = parser.parse_args()
    run_v4p_diagnostic(
        args.input_dir,
        args.output_report,
        args.output_comparison_csv,
        args.output_trades_csv,
        days=args.days,
        commit_sha=args.commit_sha,
        random_search_iterations=args.random_search_iterations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
