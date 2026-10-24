"""Run adaptive multi-timeframe research without exchange access or orders."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from pandas import DataFrame

from scripts.run_autonomous_meta_strategy import load_verified_history
from services.ai_core.adaptive_autonomous import (
    AdaptiveConfig,
    AdaptiveParameterState,
    analyze_post_trade_failure,
    build_multitimeframe_contexts,
    initial_parameter_state,
    load_adaptive_config,
    plan_dynamic_stop,
    select_adaptive_timeframe,
    size_adaptive_position,
    summarize_adaptive_metrics,
    update_intelligent_trailing_stop,
)

REPORT_NAME = "adaptive-autonomous-ai-report.json"
TRADES_NAME = "adaptive-autonomous-ai-trades.csv"
DECISIONS_NAME = "adaptive-autonomous-ai-decisions.csv"
LEARNING_NAME = "adaptive-autonomous-ai-learning-ledger.csv"
FIXED_TIMEFRAMES = ("1h", "4h", "1d")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_cell(value: object) -> object:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, allow_nan=False)
    return value


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = DataFrame(rows) if rows else DataFrame()
    for column in frame.columns:
        frame[column] = frame[column].map(_json_cell)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _number(row: dict[str, Any], name: str) -> float | None:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _timeframe_context(row: dict[str, Any], timeframe: str) -> dict[str, float] | None:
    names = (
        "close",
        "ema_fast",
        "ema_slow",
        "atr",
        "trend_strength",
        "volatility_score",
        "macd_histogram",
        "support",
    )
    context: dict[str, float] = {}
    for name in names:
        value = _number(row, f"{timeframe}_{name}")
        if value is None:
            return None
        context[name] = value
    if (
        context["close"] <= 0
        or context["atr"] <= 0
        or context["support"] <= 0
        or context["support"] >= context["close"]
    ):
        return None
    return context


def _market_state(context: dict[str, float] | None) -> str:
    if context is None:
        return "unknown"
    if context["trend_strength"] < 0.35:
        return "range"
    if context["trend_strength"] >= 0.65:
        return "trend"
    return "mixed"


def _close_trade(
    position: dict[str, Any],
    candles: DataFrame,
    contexts: DataFrame,
    exit_index: int,
    exit_price: float,
    exit_reason: str,
    equity: float,
    state: AdaptiveParameterState,
    config: AdaptiveConfig,
) -> tuple[dict[str, Any], float, AdaptiveParameterState, dict[str, Any]]:
    entry_price = float(position["entry_price"])
    quantity = float(position["quantity"])
    notional = entry_price * quantity
    gross_pnl = quantity * (exit_price - entry_price)
    cost_rate = (
        2.0 * (config.fee_bps_per_side + config.slippage_bps_per_side) / 10_000.0
    )
    costs = notional * cost_rate
    net_pnl = gross_pnl - costs
    equity_after = max(0.0, equity + net_pnl)
    exit_row = contexts.iloc[exit_index].to_dict()
    exit_context = _timeframe_context(exit_row, str(position["timeframe"]))
    trade = {
        "trade_id": int(position["trade_id"]),
        "policy": str(position["policy"]),
        "timeframe": str(position["timeframe"]),
        "entry_time": str(position["entry_time"]),
        "exit_time": str(candles["date"].iloc[exit_index]),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "entry_atr": float(position["entry_atr"]),
        "initial_stop_loss": float(position["initial_stop_loss"]),
        "final_trailing_stop": float(position["current_stop"]),
        "initial_stop_distance_atr": float(position["initial_stop_distance_atr"]),
        "quantity": quantity,
        "notional_quote": notional,
        "risk_budget_quote": float(position["risk_budget_quote"]),
        "risk_fraction": float(position["risk_fraction"]),
        "entry_confidence": float(position["entry_confidence"]),
        "entry_trend_strength": float(position["entry_trend_strength"]),
        "entry_volatility_score": float(position["entry_volatility_score"]),
        "max_favorable_excursion_atr": float(position["max_favorable_excursion_atr"]),
        "max_adverse_excursion_atr": float(position["max_adverse_excursion_atr"]),
        "entry_regime": str(position["entry_regime"]),
        "exit_regime": _market_state(exit_context),
        "gross_pnl_quote": gross_pnl,
        "estimated_cost_quote": costs,
        "net_pnl_quote": net_pnl,
        "net_return_bps": net_pnl / notional * 10_000.0 if notional > 0 else 0.0,
        "won_net_of_costs": net_pnl > 0,
        "exit_reason": exit_reason,
        "equity_before_quote": equity,
        "equity_after_quote": equity_after,
        "parameter_revision_at_entry": int(position["parameter_revision_at_entry"]),
        "order_permission": False,
        "real_order_attempted": False,
    }
    analysis, next_state = analyze_post_trade_failure(trade, state, config)
    learning_event = {
        "trade_id": trade["trade_id"],
        "timestamp": trade["exit_time"],
        "outcome": "WIN" if net_pnl > 0 else "LOSS",
        **analysis,
    }
    return trade, equity_after, next_state, learning_event


def _decision_row(
    policy: str,
    timestamp: object,
    timeframe: str | None,
    confidence: float,
    action: str,
    reason: str,
    entry_signal: bool,
    stop_loss: float | None = None,
) -> dict[str, Any]:
    return {
        "policy": policy,
        "timestamp": str(timestamp),
        "selected_timeframe": timeframe,
        "selection_confidence": confidence,
        "action": action,
        "reason": reason,
        "entry_signal": entry_signal,
        "planned_or_current_stop": stop_loss,
        "order_permission": False,
        "real_order_attempted": False,
    }


def _record_close(
    position: dict[str, Any],
    candles: DataFrame,
    contexts: DataFrame,
    cursor: int,
    exit_price: float,
    reason: str,
    equity: float,
    state: AdaptiveParameterState,
    config: AdaptiveConfig,
    trades: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    learning_ledger: list[dict[str, Any]],
) -> tuple[float, AdaptiveParameterState]:
    trade, next_equity, next_state, event = _close_trade(
        position, candles, contexts, cursor, exit_price, reason, equity, state, config
    )
    trades.append(trade)
    learning_ledger.append(event)
    decisions.append(
        _decision_row(
            str(position["policy"]),
            candles["date"].iloc[cursor],
            str(position["timeframe"]),
            float(position["entry_confidence"]),
            "CLOSE",
            reason,
            False,
            float(position["current_stop"]),
        )
    )
    return next_equity, next_state


def _manage_open_position(
    position: dict[str, Any],
    cursor: int,
    candles: DataFrame,
    contexts: DataFrame,
    equity: float,
    state: AdaptiveParameterState,
    config: AdaptiveConfig,
    trades: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    learning_ledger: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, float, AdaptiveParameterState]:
    bar = candles.iloc[cursor]
    low, opening = float(bar["low"]), float(bar["open"])
    prior_stop = float(position["current_stop"])
    policy, timeframe = str(position["policy"]), str(position["timeframe"])
    confidence = float(position["entry_confidence"])
    timestamp = bar["date"]
    if low <= prior_stop:
        position["max_adverse_excursion_atr"] = max(
            float(position["max_adverse_excursion_atr"]),
            max(
                0.0,
                (float(position["entry_price"]) - low) / float(position["entry_atr"]),
            ),
        )
        equity, state = _record_close(
            position,
            candles,
            contexts,
            cursor,
            min(opening, prior_stop),
            "trailing_stop",
            equity,
            state,
            config,
            trades,
            decisions,
            learning_ledger,
        )
        return None, equity, state

    high, close = float(bar["high"]), float(bar["close"])
    position["max_adverse_excursion_atr"] = max(
        float(position["max_adverse_excursion_atr"]),
        max(0.0, (float(position["entry_price"]) - low) / float(position["entry_atr"])),
    )
    position["max_favorable_excursion_atr"] = max(
        float(position["max_favorable_excursion_atr"]),
        max(
            0.0, (high - float(position["entry_price"])) / float(position["entry_atr"])
        ),
    )
    context = _timeframe_context(contexts.iloc[cursor].to_dict(), timeframe)
    if context is None:
        decisions.append(
            _decision_row(
                policy,
                timestamp,
                timeframe,
                confidence,
                "HOLD",
                "timeframe_context_unavailable_keep_prior_stop",
                False,
                prior_stop,
            )
        )
        return position, equity, state
    if close < context["ema_fast"] and context["macd_histogram"] < 0:
        equity, state = _record_close(
            position,
            candles,
            contexts,
            cursor,
            close,
            "trend_reversal",
            equity,
            state,
            config,
            trades,
            decisions,
            learning_ledger,
        )
        return None, equity, state

    high_watermark = max(float(position["high_watermark"]), high)
    updated = update_intelligent_trailing_stop(
        current_stop=prior_stop,
        high_watermark=high_watermark,
        current_close=close,
        atr=context["atr"],
        support_level=context["support"],
        trend_strength=context["trend_strength"],
        macd_histogram=context["macd_histogram"],
        state=state,
        config=config,
    )
    mode = "KEEP_PRIOR_STOP"
    if updated is not None:
        position["current_stop"], mode = updated
        position["high_watermark"] = high_watermark
        position["last_trailing_mode"] = mode
    decisions.append(
        _decision_row(
            policy,
            timestamp,
            timeframe,
            confidence,
            "MANAGE",
            f"trailing_policy:{mode}",
            False,
            float(position["current_stop"]),
        )
    )
    return position, equity, state


def _entry_signal(context: dict[str, float], config: AdaptiveConfig) -> bool:
    return (
        context["trend_strength"] >= config.minimum_entry_trend_strength
        and context["close"] > context["ema_fast"]
        and context["ema_fast"] > context["ema_slow"]
        and context["macd_histogram"] > 0
    )


def _try_open_position(
    candles: DataFrame,
    context_row: dict[str, Any],
    cursor: int,
    policy: str,
    fixed_timeframe: str | None,
    equity: float,
    state: AdaptiveParameterState,
    config: AdaptiveConfig,
    trade_count: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    timestamp = candles["date"].iloc[cursor]
    selector = select_adaptive_timeframe(context_row, state)
    timeframe = fixed_timeframe or selector.timeframe
    confidence = selector.confidence
    if equity <= 0 or cursor + 2 >= len(candles):
        return None, _decision_row(
            policy,
            timestamp,
            timeframe,
            confidence,
            "NO_ENTRY",
            "equity_depleted_or_insufficient_future_bars",
            False,
        )
    if timeframe is None:
        return None, _decision_row(
            policy, timestamp, None, confidence, "NO_ENTRY", selector.reason, False
        )
    context = _timeframe_context(context_row, timeframe)
    if context is None:
        return None, _decision_row(
            policy,
            timestamp,
            timeframe,
            confidence,
            "NO_ENTRY",
            "selected_timeframe_context_unavailable",
            False,
        )
    if confidence < state.entry_confidence_floor:
        return None, _decision_row(
            policy,
            timestamp,
            timeframe,
            confidence,
            "NO_ENTRY",
            "selection_confidence_below_adaptive_floor",
            False,
        )
    signal = _entry_signal(context, config)
    if not signal:
        return None, _decision_row(
            policy,
            timestamp,
            timeframe,
            confidence,
            "NO_ENTRY",
            "trend_and_macd_entry_conditions_not_met",
            False,
        )
    entry_index = cursor + 1
    entry_price = float(candles["open"].iloc[entry_index])
    stop_plan = plan_dynamic_stop(
        entry_price=entry_price,
        atr=context["atr"],
        support_level=context["support"],
        state=state,
        config=config,
    )
    if stop_plan is None:
        return None, _decision_row(
            policy,
            timestamp,
            timeframe,
            confidence,
            "NO_ENTRY",
            "dynamic_stop_invalid_or_exceeds_risk_envelope",
            True,
        )
    sizing = size_adaptive_position(
        equity=equity,
        entry_price=entry_price,
        stop_loss=stop_plan.stop_loss,
        confidence=confidence,
        trend_strength=context["trend_strength"],
        volatility_score=context["volatility_score"],
        config=config,
    )
    if sizing is None:
        return None, _decision_row(
            policy,
            timestamp,
            timeframe,
            confidence,
            "NO_ENTRY",
            "adaptive_position_sizing_invalid",
            True,
            stop_plan.stop_loss,
        )
    position: dict[str, Any] = {
        "trade_id": trade_count + 1,
        "policy": policy,
        "timeframe": timeframe,
        "entry_time": str(candles["date"].iloc[entry_index]),
        "entry_price": entry_price,
        "entry_atr": context["atr"],
        "initial_stop_loss": stop_plan.stop_loss,
        "current_stop": stop_plan.stop_loss,
        "initial_stop_distance_atr": stop_plan.stop_distance / context["atr"],
        "quantity": sizing.quantity,
        "risk_budget_quote": sizing.risk_budget,
        "risk_fraction": sizing.risk_fraction,
        "entry_confidence": confidence,
        "entry_trend_strength": context["trend_strength"],
        "entry_volatility_score": context["volatility_score"],
        "entry_regime": _market_state(context),
        "high_watermark": entry_price,
        "max_favorable_excursion_atr": 0.0,
        "max_adverse_excursion_atr": 0.0,
        "parameter_revision_at_entry": state.revision,
        "last_trailing_mode": "INITIAL_STOP",
    }
    return position, _decision_row(
        policy,
        timestamp,
        timeframe,
        confidence,
        "OPEN",
        "adaptive_entry_accepted_next_hour_open",
        True,
        stop_plan.stop_loss,
    )


def _simulate_policy(
    candles: DataFrame,
    contexts: DataFrame,
    config: AdaptiveConfig,
    *,
    policy: str,
    fixed_timeframe: str | None = None,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    state = initial_parameter_state(config)
    equity = config.initial_equity
    trades: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    learning_ledger: list[dict[str, Any]] = []
    position: dict[str, Any] | None = None
    for cursor in range(len(candles)):
        if position is not None:
            position, equity, state = _manage_open_position(
                position,
                cursor,
                candles,
                contexts,
                equity,
                state,
                config,
                trades,
                decisions,
                learning_ledger,
            )
            continue
        position, decision = _try_open_position(
            candles,
            contexts.iloc[cursor].to_dict(),
            cursor,
            policy,
            fixed_timeframe,
            equity,
            state,
            config,
            len(trades),
        )
        decisions.append(decision)
    if position is not None:
        final_index = len(candles) - 1
        equity, state = _record_close(
            position,
            candles,
            contexts,
            final_index,
            float(candles["close"].iloc[final_index]),
            "end_of_data",
            equity,
            state,
            config,
            trades,
            decisions,
            learning_ledger,
        )
    summary = summarize_adaptive_metrics(trades, initial_equity=config.initial_equity)
    summary["final_parameter_state"] = asdict(state)
    summary["parameter_revision_count"] = state.revision
    summary["learning_events"] = len(learning_ledger)
    return trades, decisions, learning_ledger, summary


def _comparison(
    adaptive: dict[str, Any], baselines: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for timeframe, baseline in baselines.items():
        left = adaptive["expectancy_bps_per_trade"]
        right = baseline["expectancy_bps_per_trade"]
        result[timeframe] = {
            "status": "COMPARABLE_EXPLORATORY_METRICS"
            if left is not None and right is not None
            else "INSUFFICIENT_TRADES_FOR_COMPARISON",
            "expectancy_delta_bps": float(left) - float(right)
            if left is not None and right is not None
            else None,
            "trade_count_delta": int(adaptive["trade_count"])
            - int(baseline["trade_count"]),
        }
    return result


def run_research(input_dir: Path, output_dir: Path, commit_sha: str) -> dict[str, Any]:
    if len(commit_sha) != 40 or any(
        char not in "0123456789abcdef" for char in commit_sha
    ):
        raise ValueError("adaptive_commit_sha_invalid")
    project_root = Path(__file__).resolve().parents[1]
    module_path = project_root / "services" / "ai_core" / "adaptive_autonomous.py"
    config_path = project_root / "configs" / "research" / "adaptive-autonomous-ai.json"
    config = load_adaptive_config(config_path, module_path)
    config_document = json.loads(config_path.read_text(encoding="utf-8"))
    candles, source = load_verified_history(input_dir)
    if source.get("final_holdout_used") is not False:
        raise ValueError("adaptive_final_holdout_must_remain_reserved")
    contexts = build_multitimeframe_contexts(candles)
    adaptive_trades, adaptive_decisions, learning, adaptive_metrics = _simulate_policy(
        candles, contexts, config, policy="adaptive_selector"
    )
    baseline_metrics: dict[str, dict[str, Any]] = {}
    all_trades = list(adaptive_trades)
    all_decisions = list(adaptive_decisions)
    all_learning = list(learning)
    for timeframe in FIXED_TIMEFRAMES:
        trades, decisions, ledger, metrics = _simulate_policy(
            candles,
            contexts,
            config,
            policy=f"fixed_{timeframe}",
            fixed_timeframe=timeframe,
        )
        baseline_metrics[timeframe] = metrics
        all_trades.extend(trades)
        all_decisions.extend(decisions)
        all_learning.extend(ledger)
    report: dict[str, Any] = {
        "schema_version": "alix.adaptive-autonomous-ai-report.v1",
        "created_by_commit_sha": commit_sha,
        "run_status": "COMPLETED_RESEARCH_ONLY",
        "source": source,
        "policy": {
            "timeframes": list(FIXED_TIMEFRAMES),
            "timeframe_selection": "context_score_from_closed_1h_4h_1d_bars",
            "confidence_semantics": "bounded score margin; not a calibrated probability",
            "stop_policy": "ATR_plus_prior_20_bar_support; reject stops beyond configured envelope",
            "exit_policy": "no_take_profit; ATR_and_support_trailing_stop_plus_confirmed_reversal",
            "sizing_policy": "risk_fraction_scales_with_confidence_and_trend_and_reduces_with_volatility",
            "learning_policy": "bounded_after_close_updates_only; no in-trade or future-outcome adaptation",
            "training_claim": "interpretable adaptive research policy; not yet a trained ML model",
            "risk_envelope": {
                "minimum_risk_fraction": config.minimum_risk_fraction,
                "maximum_risk_fraction": config.maximum_risk_fraction,
                "maximum_position_fraction": config.maximum_position_fraction,
                "maximum_stop_distance_fraction": config.maximum_stop_distance_fraction,
            },
            "config_revision": config_document["revision"],
            "config_code_sha256": config_document["code_sha256"],
        },
        "economic_evaluation": {
            "status": "SEQUENTIAL_HISTORICAL_BAR_SIMULATION",
            "economic_acceptance": "NO-GO",
            "fee_bps_per_side": config.fee_bps_per_side,
            "slippage_bps_per_side": config.slippage_bps_per_side,
            "spread_observed": False,
            "orderbook_liquidity_observed": False,
            "adaptive_selector_metrics": adaptive_metrics,
            "fixed_timeframe_baselines": baseline_metrics,
            "comparison_vs_fixed_timeframes": _comparison(
                adaptive_metrics, baseline_metrics
            ),
            "final_holdout_used": False,
            "reason": "Exploratory sequential adaptation is not independent economic validation or live execution evidence.",
        },
        "learning": {
            "event_count": len(all_learning),
            "adaptive_policy_event_count": len(learning),
            "parameter_revision_count": adaptive_metrics["parameter_revision_count"],
            "final_parameter_state": adaptive_metrics["final_parameter_state"],
            "update_boundary": "only after trade close; changes apply to later entries",
            "ledger": LEARNING_NAME,
        },
        "artifact_paths": {
            "report": REPORT_NAME,
            "trades": TRADES_NAME,
            "decisions": DECISIONS_NAME,
            "learning": LEARNING_NAME,
        },
        "safety_invariants": {
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "economic_acceptance": "NO-GO",
            "model_promoted": False,
            "production_registry_modified": False,
            "orders_or_execution_api_used": False,
            "private_api_calls": 0,
            "real_order_attempts": 0,
            "final_holdout_used": False,
        },
    }
    safety = report["safety_invariants"]
    if (
        safety["Champion"] is not None
        or safety["order_permission"] is not False
        or safety["trading_authorization"] != "NO-GO"
        or safety["economic_acceptance"] != "NO-GO"
        or safety["real_order_attempts"] != 0
        or safety["final_holdout_used"] is not False
    ):
        raise ValueError("adaptive_safety_contract_violated")
    _write_json(output_dir / REPORT_NAME, report)
    _write_rows(output_dir / TRADES_NAME, all_trades)
    _write_rows(output_dir / DECISIONS_NAME, all_decisions)
    _write_rows(output_dir / LEARNING_NAME, all_learning)
    print(
        json.dumps(
            {
                "commit_sha": commit_sha,
                "run_status": report["run_status"],
                "adaptive_metrics": adaptive_metrics,
                "fixed_timeframe_baselines": baseline_metrics,
                "economic_acceptance": "NO-GO",
                "Champion": None,
                "order_permission": False,
                "trading_authorization": "NO-GO",
                "final_holdout_used": False,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit-sha", required=True)
    args = parser.parse_args()
    run_research(args.input_dir, args.output_dir, args.commit_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
