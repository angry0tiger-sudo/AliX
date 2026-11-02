"""Run V4-O MAE/MFE and dynamic time-exit replay on persisted V4-H signals."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from pandas import DataFrame

from services.ml_core.economic_backtest import (
    V4H_SOURCE_RUN_ID,
    V4H_SOURCE_SHA,
    V4O_TIME_EXIT_CANDLES,
    V4O_TIME_EXIT_MFE_THRESHOLD_ATR,
    ExecutionConfig,
    TradeManagementConfig,
    evaluate_candidate_time_exit_sweep,
    extract_qualifying_candidates,
    load_validated_minute_data,
    resample_ohlcv,
    validate_execution_config,
    validate_source_report,
)

SOURCE_SHA256 = "3f3c08c583fabb6c91fcdee81c036db6c92a988cdf2537306813af97e89de6a0"
SCHEMA = "alix.ml-core.v4o-mae-mfe-time-exit.v1"
HORIZON = 12
ATR_PERIOD = 14
CONFIDENCE = 0.60
STOP_LOSS_ATR = 2.0
TAKE_PROFIT_ATR = 3.0
MAX_DRAWDOWN = 0.15
V4M_RUN_ID = 41345541356
V4M_BEST_LEDGER_PF = 0.04894419
V4M_SAME_GEOMETRY_LEDGER_PF = 0.03802472
EXIT_REASONS = (
    "SL_HIT", "TP_HIT", "HORIZON_EXPIRY", "SELL", "TIME_EXIT",
    "COLLISION", "END_OF_FORWARD_WINDOW",
)
STOP_EXIT_TYPES = ("INITIAL_STOP", "BREAKEVEN_STOP", "TRAILING_STOP")
COMPARISON_CSV_FIELDS = (
    "variant", "time_exit_candles", "candidate_id", "pair", "timeframe",
    "feature_profile", "weighting_policy", "confidence_threshold",
    "stop_loss_atr_multiplier", "take_profit_atr_multiplier", "trade_count",
    "profit_factor", "max_drawdown_percent", "win_rate", "mean_mfe_atr",
    "mean_mae_atr", "mean_losing_trade_mfe_atr", "mean_losing_trade_mae_atr",
    "losing_trades_with_positive_mfe_count", "time_exit_count",
    "initial_stop_exit_count", "tp_exit_count", "horizon_exit_count",
    "sell_exit_count", "collision_exit_count", "total_fees_usd",
    "total_slippage_cost_usd", "economic_candidate_gate_passed",
)
TRADE_CSV_FIELDS = (
    "variant", "time_exit_candles", "candidate_id", "trade_index",
    "entry_timestamp", "exit_timestamp", "direction", "bars_held",
    "entry_fill_price", "atr_value", "mfe_atr", "mae_atr", "net_pnl_usd",
    "exit_reason", "exit_trigger_type", "time_exit_evaluated",
    "time_exit_pending", "time_exit_triggered",
)


def _load_json(path: Path) -> dict[str, Any]:
    """Load one required JSON object."""
    if not path.is_file():
        raise ValueError(f"required_json_file_missing:{path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("economic_json_root_must_be_object")
    return value


def _market_bars(
    candidates: list[dict[str, Any]],
    source: dict[str, Any],
    input_dir: Path,
    commit_sha: str,
) -> tuple[dict[tuple[str, str], DataFrame], list[dict[str, Any]]]:
    """Load exact V4-H candles and rebuild only the candidate timeframes."""
    datasets = source.get("input_datasets")
    if not isinstance(datasets, list):
        raise ValueError("v4h_source_dataset_manifest_invalid")
    metadata = {str(item["symbol"]): item for item in datasets}
    if set(metadata) != {"BTCUSDT", "ETHUSDT"}:
        raise ValueError("v4h_source_dataset_manifest_invalid")
    symbols = {"BTC/USDT": "BTCUSDT", "ETH/USDT": "ETHUSDT"}
    minute: dict[str, DataFrame] = {}
    bars: dict[tuple[str, str], DataFrame] = {}
    provenance: list[dict[str, Any]] = []
    for candidate in candidates:
        symbol = symbols.get(str(candidate["pair"]))
        if symbol is None:
            raise ValueError("economic_pair_out_of_policy")
        if symbol not in minute:
            minute[symbol] = load_validated_minute_data(
                input_dir, metadata[symbol], current_commit_sha=commit_sha
            )
            manifest = _load_json(
                input_dir / f"binance-{symbol.lower()}-1m.manifest.json"
            )
            provenance.append(
                {
                    "symbol": symbol,
                    "row_count": manifest["row_count"],
                    "start_utc": manifest["start_utc"],
                    "end_utc": manifest["end_utc"],
                    "csv_sha256": manifest["csv_sha256"],
                    "source_v4h_csv_sha256": metadata[symbol]["csv_sha256"],
                    "byte_identical_to_v4h": True,
                    "public_api_only": True,
                    "credentials_used": False,
                }
            )
        key = (symbol, str(candidate["timeframe"]))
        if key not in bars:
            bars[key] = resample_ohlcv(minute[symbol], key[1])
    return bars, provenance


def _execution() -> ExecutionConfig:
    """Keep the established one-times Spot and per-side cost assumptions."""
    config = ExecutionConfig(
        initial_equity=10_000.0,
        fee_bps=25.0,
        slippage_bps=20.0,
        position_fraction=1.0,
        maximum_drawdown=MAX_DRAWDOWN,
    )
    validate_execution_config(config)
    return config


def _management() -> TradeManagementConfig:
    """Freeze confidence and TP/SL geometry while disabling trailing stops."""
    return TradeManagementConfig(
        atr_period=ATR_PERIOD,
        take_profit_atr_multiplier=TAKE_PROFIT_ATR,
        stop_loss_atr_multiplier=STOP_LOSS_ATR,
        horizon_candles=HORIZON,
        confidence_threshold=CONFIDENCE,
        trailing_stop_enabled=False,
        time_exit_mfe_threshold_atr=V4O_TIME_EXIT_MFE_THRESHOLD_ATR,
    )


def _validate_trade(trade: dict[str, Any], time_exit_candles: int | None) -> None:
    """Fail closed on invalid trade geometry, excursion, or exit causality."""
    if trade["direction"] != "LONG" or trade["entry_confidence"] < CONFIDENCE:
        raise ValueError("v4o_trade_direction_or_confidence_invalid")
    if not 1 <= int(trade["bars_held"]) <= HORIZON:
        raise ValueError("v4o_trade_horizon_invalid")
    if trade["time_exit_candles"] != time_exit_candles:
        raise ValueError("v4o_trade_time_exit_horizon_mismatch")
    reference = float(trade["barrier_reference_price"])
    atr = float(trade["atr_value"])
    if abs(float(trade["initial_stop_loss_price"]) - (reference - STOP_LOSS_ATR * atr)) > 1e-7:
        raise ValueError("v4o_trade_stop_geometry_invalid")
    if abs(float(trade["take_profit_price"]) - (reference + TAKE_PROFIT_ATR * atr)) > 1e-7:
        raise ValueError("v4o_trade_target_geometry_invalid")
    if abs(float(trade["stop_loss_price"]) - float(trade["initial_stop_loss_price"])) > 1e-7:
        raise ValueError("v4o_trailing_stop_must_be_disabled")
    for field in ("mfe_atr", "mae_atr"):
        value = float(trade[field])
        if value < 0:
            raise ValueError("v4o_trade_excursion_negative")
    if trade["exit_reason"] == "TIME_EXIT":
        if (
            time_exit_candles is None
            or trade["time_exit_pending"] is not True
            or trade["time_exit_evaluated"] is not True
            or trade["time_exit_triggered"] is not True
            or float(trade["mfe_atr"]) >= V4O_TIME_EXIT_MFE_THRESHOLD_ATR
            or trade["exit_trigger_type"] != "TIME_EXIT_OPEN"
        ):
            raise ValueError("v4o_time_exit_causality_invalid")
    elif trade["time_exit_triggered"] is not False:
        raise ValueError("v4o_time_exit_reason_mismatch")


def _candidate_result(
    candidate: dict[str, Any],
    replay: dict[str, Any],
    time_exit_candles: int | None,
) -> dict[str, Any]:
    """Attach the immutable candidate identity to a validated replay."""
    trades = replay["trades"]
    for trade in trades:
        _validate_trade(trade, time_exit_candles)
    return {
        "candidate_id": candidate["candidate_id"],
        "pair": candidate["pair"],
        "timeframe": candidate["timeframe"],
        "feature_profile": candidate["feature_profile"],
        "weighting_policy": candidate["weighting_policy"],
        "source_forward_hold_precision": candidate["hold_precision"],
        "source_forward_hold_recall": candidate["hold_recall"],
        "confidence_threshold": CONFIDENCE,
        "stop_loss_atr_multiplier": STOP_LOSS_ATR,
        "take_profit_atr_multiplier": TAKE_PROFIT_ATR,
        "time_exit_candles": time_exit_candles,
        "metrics": replay["metrics"],
        "trades": trades,
        "exit_reason_counts": replay["exit_reason_counts"],
        "stop_exit_type_counts": replay["stop_exit_type_counts"],
        "time_exit_count": replay["time_exit_count"],
        "total_fees_usd": replay["total_fees_usd"],
        "total_slippage_cost_usd": replay["total_slippage_cost_usd"],
        "trade_direction_counts": {"LONG": len(trades), "SHORT": 0},
        "trade_management": replay["trade_management"],
    }


def _trade_outcomes(
    trades: list[dict[str, Any]],
) -> tuple[list[float], list[float]]:
    """Separate positive and negative net-PnL outcomes."""
    pnl = [float(item["net_pnl_usd"]) for item in trades]
    return [value for value in pnl if value > 0], [value for value in pnl if value < 0]


def _optional_mean(values: list[float]) -> float | None:
    """Return a rounded arithmetic mean for a non-empty sample."""
    return round(sum(values) / len(values), 8) if values else None


def _ledger_profit_factor(
    wins: list[float], losses: list[float]
) -> float | None:
    """Compute descriptive pooled PF across independent candidate replays."""
    gross_loss = abs(sum(losses))
    return round(sum(wins) / gross_loss, 8) if gross_loss > 0 else None


def _sum_exit_reasons(candidates: list[dict[str, Any]]) -> dict[str, int]:
    """Aggregate known exit reasons and preserve zero-count categories."""
    counts = {reason: 0 for reason in EXIT_REASONS}
    for candidate in candidates:
        for reason, count in candidate["exit_reason_counts"].items():
            if reason not in counts:
                raise ValueError("v4o_unexpected_exit_reason")
            counts[reason] += int(count)
    return counts


def _sum_stop_exit_types(candidates: list[dict[str, Any]]) -> dict[str, int]:
    """Aggregate stop categories with a stable report shape."""
    counts = {reason: 0 for reason in STOP_EXIT_TYPES}
    for candidate in candidates:
        for reason, count in candidate["stop_exit_type_counts"].items():
            if reason not in counts:
                raise ValueError("v4o_unexpected_stop_exit_type")
            counts[reason] += int(count)
    return counts


def _excursion_summary(losing: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize MFE and MAE among negative-net-PnL trades."""
    mfe = [float(item["mfe_atr"]) for item in losing]
    mae = [float(item["mae_atr"]) for item in losing]
    return {
        "losing_trade_count": len(losing),
        "mean_mfe_atr_losing_trades": _optional_mean(mfe),
        "mean_mae_atr_losing_trades": _optional_mean(mae),
        "losing_trades_with_positive_mfe_count": sum(value > 0 for value in mfe),
        "losing_trades_with_mfe_at_least_0_5_atr_count": sum(value >= 0.5 for value in mfe),
        "losing_trades_with_zero_mfe_count": sum(value == 0 for value in mfe),
    }


def _best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the active candidate with the highest defined net PF."""
    best = max(
        (
            item for item in candidates
            if item["metrics"]["profit_factor_net_of_costs"] is not None
        ),
        key=lambda item: float(item["metrics"]["profit_factor_net_of_costs"]),
        default=None,
    )
    if best is None:
        return None
    return {
        "candidate_id": best["candidate_id"],
        "profit_factor": best["metrics"]["profit_factor_net_of_costs"],
        "trade_count": best["metrics"]["trade_count"],
        "max_drawdown_percent": best["metrics"]["max_drawdown_percent"],
    }


def _variant_aggregate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one experiment without combining candidate equity curves."""
    if len(candidates) != 13:
        raise ValueError("v4o_candidate_count_invalid")
    trades = [trade for candidate in candidates for trade in candidate["trades"]]
    wins, losses = _trade_outcomes(trades)
    defined_pf = [
        float(item["metrics"]["profit_factor_net_of_costs"])
        for item in candidates
        if item["metrics"]["profit_factor_net_of_costs"] is not None
    ]
    active_dd = [
        float(item["metrics"]["max_drawdown_percent"])
        for item in candidates
        if int(item["metrics"]["trade_count"]) > 0
    ]
    all_dd = [float(item["metrics"]["max_drawdown_percent"]) for item in candidates]
    losing = [trade for trade in trades if float(trade["net_pnl_usd"]) < 0]
    return {
        "candidate_count": len(candidates),
        "active_candidate_count": len(active_dd),
        "total_trade_count": len(trades),
        "winning_trade_count": len(wins),
        "losing_trade_count": len(losses),
        "win_rate": round(len(wins) / (len(wins) + len(losses)), 8) if wins or losses else 0.0,
        "combined_ledger_profit_factor": _ledger_profit_factor(wins, losses),
        "mean_defined_candidate_profit_factor": _optional_mean(defined_pf),
        "defined_candidate_profit_factor_count": len(defined_pf),
        "mean_max_drawdown_percent_active_candidates": _optional_mean(active_dd),
        "mean_max_drawdown_percent_all_candidates": _optional_mean(all_dd),
        "worst_candidate_max_drawdown_percent": round(max(all_dd), 8),
        "pf_above_1_5_candidate_count": sum(value > 1.5 for value in defined_pf),
        "drawdown_below_15_candidate_count": sum(value < 15.0 for value in all_dd),
        "drawdown_below_15_active_candidate_count": sum(value < 15.0 for value in active_dd),
        "economic_gate_passed_candidate_count": sum(
            bool(item["metrics"]["economic_candidate_gate_passed"])
            for item in candidates
        ),
        "exit_reason_counts": _sum_exit_reasons(candidates),
        "stop_exit_type_counts": _sum_stop_exit_types(candidates),
        "time_exit_count": sum(item["time_exit_count"] for item in candidates),
        "mean_mfe_atr_all_trades": _optional_mean(
            [float(item["mfe_atr"]) for item in trades]
        ),
        "mean_mae_atr_all_trades": _optional_mean(
            [float(item["mae_atr"]) for item in trades]
        ),
        **_excursion_summary(losing),
        "total_fees_usd": round(sum(float(item["total_fees_usd"]) for item in candidates), 8),
        "total_slippage_cost_usd": round(sum(float(item["total_slippage_cost_usd"]) for item in candidates), 8),
        "best_defined_profit_factor_candidate": _best_candidate(candidates),
    }


def _variant_name(time_exit_candles: int | None) -> str:
    """Build a stable label for the selected time-exit policy."""
    return "BASELINE" if time_exit_candles is None else f"X={time_exit_candles}"


def _variant_report(
    time_exit_candles: int | None,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build an auditable summary for one time-exit setting."""
    aggregate = _variant_aggregate(candidates)
    return {
        "variant": _variant_name(time_exit_candles),
        "time_exit_candles": time_exit_candles,
        "time_exit_mfe_threshold_atr": V4O_TIME_EXIT_MFE_THRESHOLD_ATR,
        "aggregate": aggregate,
        "candidates": candidates,
        "economic_acceptance": {
            "profit_factor_operator": ">",
            "profit_factor_minimum": 1.5,
            "maximum_drawdown_operator": "<",
            "maximum_drawdown_limit_percent": 15.0,
            "candidate_count": 13,
            "passed_candidate_count": aggregate["economic_gate_passed_candidate_count"],
            "status": "MET" if aggregate["economic_gate_passed_candidate_count"] else "NOT_MET",
        },
    }


def _report(
    commit_sha: str,
    source_digest: str,
    variants: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the immutable-source V4-O report and safety envelope."""
    return {
        "schema_version": SCHEMA,
        "commit_sha": commit_sha,
        "source_v4h_commit_sha": V4H_SOURCE_SHA,
        "source_v4h_run_id": V4H_SOURCE_RUN_ID,
        "source_v4h_report_sha256": source_digest,
        "v4m_reference_run_id": V4M_RUN_ID,
        "scope": "PERSISTED_V4H_FORWARD_PROBABILITY_REPLAY_SPOT_ONLY_DYNAMIC_TIME_EXIT",
        "economic_evaluation_performed": True,
        "model_training_performed": False,
        "model_inference_performed": False,
        "persisted_probability_replay_only": True,
        "candidate_count": 13,
        "variant_count": len(variants),
        "confidence_threshold": CONFIDENCE,
        "input_datasets": provenance,
        "fixed_trade_management_config": {
            "confidence_threshold": CONFIDENCE,
            "atr_period": ATR_PERIOD,
            "horizon_candles": HORIZON,
            "stop_loss_atr_multiplier": STOP_LOSS_ATR,
            "take_profit_atr_multiplier": TAKE_PROFIT_ATR,
            "trailing_stop_enabled": False,
            "time_exit_mfe_threshold_atr": V4O_TIME_EXIT_MFE_THRESHOLD_ATR,
            "time_exit_evaluation_policy": "ONCE_AFTER_X_COMPLETED_CANDLES",
            "time_exit_execution_policy": "NEXT_CANDLE_OPEN",
        },
        "time_exit_variants": variants,
        "excursion_metric_policy": {
            "mfe_atr": "(MAX_ACTIVE_BAR_HIGH - ENTRY_FILL_PRICE) / ENTRY_ATR",
            "mae_atr": "(ENTRY_FILL_PRICE - MIN_ACTIVE_BAR_LOW) / ENTRY_ATR",
            "entry_price_reference": "ACTUAL_NEXT_BAR_ENTRY_FILL",
            "atr_reference": "SIGNAL_BAR_CLOSE_ONLY",
            "bar_scope": "FULL_OHLC_RANGE_OF_EACH_ACTIVE_CANDLE",
            "intrabar_ordering_caveat": (
                "OHLC bars do not reveal the order of high and low within a candle; "
                "excursions are bar-range estimates, not tick-level path measurements."
            ),
        },
        "execution_assumptions": {
            "initial_equity_usd": 10000.0,
            "fee_bps_per_side": 25.0,
            "slippage_bps_per_side": 20.0,
            "leverage": 1.0,
            "pyramiding": False,
            "short_positions_supported": False,
        },
        "v4m_reference": {
            "run_id": V4M_RUN_ID,
            "best_combined_ledger_profit_factor": V4M_BEST_LEDGER_PF,
            "best_combined_ledger_geometry": {"sl_atr": 2.5, "tp_atr": 2.5},
            "same_geometry_combined_ledger_profit_factor": V4M_SAME_GEOMETRY_LEDGER_PF,
            "same_geometry": {"sl_atr": 2.0, "tp_atr": 3.0},
        },
        "economic_acceptance": {
            "profit_factor_operator": ">",
            "profit_factor_minimum": 1.5,
            "maximum_drawdown_operator": "<",
            "maximum_drawdown_limit_percent": 15.0,
            "candidate_count_per_variant": 13,
            "passed_candidate_count_per_variant": {
                item["variant"]: item["economic_acceptance"]["passed_candidate_count"]
                for item in variants
            },
            "status": "MET" if any(
                item["economic_acceptance"]["passed_candidate_count"] > 0
                for item in variants
            ) else "NOT_MET",
        },
        "validation_bias_caveat": (
            "V4-O reuses the V4-H Forward-Validation interval for exploratory exit "
            "management research; independent OOS and owner review remain mandatory."
        ),
        "promotion_status": "BLOCKED_PENDING_INDEPENDENT_OOS_AND_OWNER_REVIEW",
        "safety": {
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "private_api_calls": 0,
            "real_order_attempts": 0,
            "orders_or_execution_api_used": False,
        },
    }


def _comparison_csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Create one comparison row per candidate and policy."""
    rows = []
    for variant in report["time_exit_variants"]:
        for candidate in variant["candidates"]:
            metrics = candidate["metrics"]
            exits = candidate["exit_reason_counts"]
            rows.append(
                {
                    "variant": variant["variant"],
                    "time_exit_candles": variant["time_exit_candles"],
                    "candidate_id": candidate["candidate_id"],
                    "pair": candidate["pair"],
                    "timeframe": candidate["timeframe"],
                    "feature_profile": candidate["feature_profile"],
                    "weighting_policy": candidate["weighting_policy"],
                    "confidence_threshold": CONFIDENCE,
                    "stop_loss_atr_multiplier": STOP_LOSS_ATR,
                    "take_profit_atr_multiplier": TAKE_PROFIT_ATR,
                    "trade_count": metrics["trade_count"],
                    "profit_factor": metrics["profit_factor_net_of_costs"],
                    "max_drawdown_percent": metrics["max_drawdown_percent"],
                    "win_rate": metrics["win_rate"],
                    "mean_mfe_atr": metrics["mean_mfe_atr"],
                    "mean_mae_atr": metrics["mean_mae_atr"],
                    "mean_losing_trade_mfe_atr": metrics["mean_losing_trade_mfe_atr"],
                    "mean_losing_trade_mae_atr": metrics["mean_losing_trade_mae_atr"],
                    "losing_trades_with_positive_mfe_count": metrics[
                        "losing_trades_with_positive_mfe_count"
                    ],
                    "time_exit_count": candidate["time_exit_count"],
                    "initial_stop_exit_count": candidate["stop_exit_type_counts"][
                        "INITIAL_STOP"
                    ],
                    "tp_exit_count": exits.get("TP_HIT", 0),
                    "horizon_exit_count": exits.get("HORIZON_EXPIRY", 0),
                    "sell_exit_count": exits.get("SELL", 0),
                    "collision_exit_count": exits.get("COLLISION", 0),
                    "total_fees_usd": candidate["total_fees_usd"],
                    "total_slippage_cost_usd": candidate["total_slippage_cost_usd"],
                    "economic_candidate_gate_passed": metrics[
                        "economic_candidate_gate_passed"
                    ],
                }
            )
    return rows


def _trade_csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Create the detailed excursion ledger with one row per executed trade."""
    rows = []
    for variant in report["time_exit_variants"]:
        for candidate in variant["candidates"]:
            for trade_index, trade in enumerate(candidate["trades"], start=1):
                rows.append(
                    {
                        "variant": variant["variant"],
                        "time_exit_candles": variant["time_exit_candles"],
                        "candidate_id": candidate["candidate_id"],
                        "trade_index": trade_index,
                        "entry_timestamp": trade["entry_timestamp"],
                        "exit_timestamp": trade["exit_timestamp"],
                        "direction": trade["direction"],
                        "bars_held": trade["bars_held"],
                        "entry_fill_price": trade["entry_fill_price"],
                        "atr_value": trade["atr_value"],
                        "mfe_atr": trade["mfe_atr"],
                        "mae_atr": trade["mae_atr"],
                        "net_pnl_usd": trade["net_pnl_usd"],
                        "exit_reason": trade["exit_reason"],
                        "exit_trigger_type": trade["exit_trigger_type"],
                        "time_exit_evaluated": trade["time_exit_evaluated"],
                        "time_exit_pending": trade["time_exit_pending"],
                        "time_exit_triggered": trade["time_exit_triggered"],
                    }
                )
    return rows


def render_summary(report: dict[str, Any]) -> str:
    """Render variant economics and losing-trade excursion summaries."""
    lines = [
        "## AliX V4-O MAE/MFE and Dynamic Time Exit",
        f"Commit: {report['commit_sha']}",
        "Replay only: persisted V4-H probabilities; no training or inference.",
        "Confidence=0.60; SL=2 ATR; TP=3 ATR; ATR=14; horizon=12.",
        "Time exit: one check after X completed candles; MFE<0.5 ATR; next candle open.",
        "",
        "| Variant | Trades | Ledger PF | Mean model PF | Active mean DD | Losing MFE | Losing MAE | Positive-MFE losses | Time exits | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in report["time_exit_variants"]:
        aggregate = variant["aggregate"]
        lines.append(
            f"| {variant['variant']} | {aggregate['total_trade_count']} "
            f"| {aggregate['combined_ledger_profit_factor']} "
            f"| {aggregate['mean_defined_candidate_profit_factor']} "
            f"| {aggregate['mean_max_drawdown_percent_active_candidates']}% "
            f"| {aggregate['mean_mfe_atr_losing_trades']} "
            f"| {aggregate['mean_mae_atr_losing_trades']} "
            f"| {aggregate['losing_trades_with_positive_mfe_count']} "
            f"| {aggregate['time_exit_count']} "
            f"| {aggregate['economic_gate_passed_candidate_count']}/13 |"
        )
    lines.extend(
        [
            "",
            f"V4-M best grid PF={V4M_BEST_LEDGER_PF}; same SL=2/TP=3 PF={V4M_SAME_GEOMETRY_LEDGER_PF}.",
            "Combined-ledger PF is descriptive across independent candidate replays, not one portfolio.",
            "Active mean DD excludes candidates with no trades; all-candidate DD is reported separately.",
            "MFE/MAE use full OHLC ranges while positions are active; intrabar event order is unavailable.",
            "Reused Forward-Validation interval: independent OOS remains required.",
            "",
            "Safety: Champion=null, order_permission=false, trading_authorization=NO-GO, private_api_calls=0, real_order_attempts=0.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_v4o_mae_mfe_time_exit(
    source_report_path: Path,
    input_dir: Path,
    output_path: Path,
    output_csv_path: Path,
    output_trades_csv_path: Path,
    *,
    commit_sha: str,
) -> dict[str, Any]:
    """Verify source provenance and replay all four fixed time-exit variants."""
    source_digest = hashlib.sha256(source_report_path.read_bytes()).hexdigest()
    if source_digest != SOURCE_SHA256:
        raise ValueError("v4h_source_report_digest_mismatch")
    source = _load_json(source_report_path)
    validate_source_report(source)
    candidates = extract_qualifying_candidates(source)
    if len(candidates) != 13:
        raise ValueError("v4o_expected_13_v4h_candidates")
    bars, provenance = _market_bars(candidates, source, input_dir, commit_sha)
    execution, management = _execution(), _management()
    grouped: dict[int | None, list[dict[str, Any]]] = {
        value: [] for value in V4O_TIME_EXIT_CANDLES
    }
    symbols = {"BTC/USDT": "BTCUSDT", "ETH/USDT": "ETHUSDT"}
    for candidate in candidates:
        key = (symbols[candidate["pair"]], candidate["timeframe"])
        sweeps = evaluate_candidate_time_exit_sweep(
            candidate, bars[key], execution, management,
            time_exit_candles=V4O_TIME_EXIT_CANDLES,
        )
        for sweep in sweeps:
            horizon = sweep["time_exit_candles"]
            grouped[horizon].append(
                _candidate_result(candidate, sweep["candidate_result"], horizon)
            )
    variants = [
        _variant_report(value, grouped[value]) for value in V4O_TIME_EXIT_CANDLES
    ]
    report = _report(commit_sha, source_digest, variants, provenance)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for path, rows, fields in (
        (output_csv_path, _comparison_csv_rows(report), COMPARISON_CSV_FIELDS),
        (output_trades_csv_path, _trade_csv_rows(report), TRADE_CSV_FIELDS),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as handle:
            handle.write(render_summary(report))
    print(
        json.dumps(
            {
                "commit_sha": commit_sha,
                "candidate_count": 13,
                "variant_count": len(variants),
                "variant_aggregates": {
                    item["variant"]: item["aggregate"] for item in variants
                },
                "Champion": None,
                "trading_authorization": "NO-GO",
                "model_training_performed": False,
                "model_inference_performed": False,
            },
            sort_keys=True,
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay persisted V4-H probabilities with V4-O excursions."
    )
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-trades-csv", type=Path, required=True)
    parser.add_argument("--commit-sha", required=True)
    args = parser.parse_args()
    run_v4o_mae_mfe_time_exit(
        args.source_report,
        args.input_dir,
        args.output,
        args.output_csv,
        args.output_trades_csv,
        commit_sha=args.commit_sha,
    )


if __name__ == "__main__":
    main()
