"""Train and economically evaluate V4-Y Bollinger / EMA200 Primary Meta-Labeling."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame

from scripts.run_ml_core_real_data_baseline import (
    SYMBOLS,
    _load_diagnostic_datasets,
    resample_ohlcv,
)
from services.ml_core.adaptive_targets import apply_adaptive_triple_barrier
from services.ml_core.economic_backtest import (
    ExecutionConfig,
    TradeManagementConfig,
    simulate_signals,
)
from services.ml_core.extended_features import build_feature_profiles
from services.meta_labeling.v4y import (
    V4Y_BOLLINGER_PERIOD,
    V4Y_BOLLINGER_STDDEV,
    V4Y_EMA_TREND_PERIOD,
    V4Y_META_CALIBRATION_FRACTION,
    V4Y_META_FEATURE_COLUMNS,
    V4Y_MINIMUM_CLASS_ROWS,
    V4Y_MINIMUM_FIT_ROWS,
    V4Y_MINIMUM_PRIMARY_TRAIN_SIGNALS,
    V4Y_OOF_FOLD_COUNT,
    V4Y_PRIMARY_SIGNAL_GENERATION,
    V4Y_TARGET_ACCEPT_FRACTION,
    build_costed_meta_labels,
    build_v4y_primary_features,
    evaluate_primary_train_sample_gate,
    fit_lightgbm_meta_filter,
    generate_v4y_oof_primary_events,
    predict_meta_probabilities,
    split_meta_fit_calibration,
)

V4Y_REPORT_SCHEMA = "alix.ml-core.v4y-primary-signal.v1"
V4Y_TIMEFRAMES = ("15m", "30m", "1h")
V4Y_FEATURE_PROFILE = "Combined"
V4Y_OUTER_TRAIN_END_FRACTION = 0.70
V4Y_FORWARD_VALIDATION_END_FRACTION = 0.85
V4Y_ATR_PERIOD = 14
V4Y_TAKE_PROFIT_ATR_MULTIPLIER = 1.5
V4Y_STOP_LOSS_ATR_MULTIPLIER = 1.0
V4Y_HORIZON_CANDLES_BY_TIMEFRAME = {"15m": 96, "30m": 48, "1h": 24}
V4Y_HORIZON_HOURS = 24
V4Y_ENTRY_SLIPPAGE_BPS = 20.0
V4Y_FEE_BPS_PER_SIDE = 25.0
V4Y_SLIPPAGE_BPS_PER_SIDE = 20.0
V4Y_MAXIMUM_DRAWDOWN = 0.15
V4Y_INITIAL_EQUITY_USD = 10_000.0
V4Y_ECONOMIC_MINIMUM_TRADES = 30
V4Y_ECONOMIC_MINIMUM_NET_PF = 1.5


def _execution_config() -> ExecutionConfig:
    return ExecutionConfig(
        initial_equity=V4Y_INITIAL_EQUITY_USD,
        fee_bps=V4Y_FEE_BPS_PER_SIDE,
        slippage_bps=V4Y_SLIPPAGE_BPS_PER_SIDE,
        position_fraction=1.0,
        maximum_drawdown=V4Y_MAXIMUM_DRAWDOWN,
    )


def _horizon_candles(timeframe: str) -> int:
    try:
        return V4Y_HORIZON_CANDLES_BY_TIMEFRAME[timeframe]
    except KeyError:
        raise ValueError("v4y_timeframe_out_of_policy") from None


def _outer_boundaries(
    row_count: int,
    horizon_candles: int,
) -> tuple[int, int]:
    if row_count < 300:
        raise ValueError("v4y_resampled_dataset_too_short")
    train_end = int(row_count * V4Y_OUTER_TRAIN_END_FRACTION)
    validation_end = int(row_count * V4Y_FORWARD_VALIDATION_END_FRACTION)
    if (
        train_end <= V4Y_EMA_TREND_PERIOD + V4Y_OOF_FOLD_COUNT
        or validation_end - train_end <= horizon_candles * 2
        or validation_end >= row_count
    ):
        raise ValueError("v4y_outer_window_configuration_invalid")
    return train_end, validation_end


def _feature_rows_for_events(
    features: DataFrame,
    events: DataFrame,
) -> DataFrame:
    indices = events["event_index"].to_numpy(dtype=int)
    if (indices < 0).any() or (indices >= len(features)).any():
        raise ValueError("v4y_event_feature_index_out_of_bounds")
    selected = features.iloc[indices].reset_index(drop=True)
    if selected.isna().any(axis=None):
        raise ValueError("v4y_primary_event_features_non_finite")
    return pd.concat([events.reset_index(drop=True), selected], axis=1)


def _training_oof_events(
    bars: DataFrame,
    features: DataFrame,
    primary_buy_signal: pd.Series,
    train_end: int,
    horizon_candles: int,
) -> tuple[DataFrame, dict[str, Any]]:
    oof_events = generate_v4y_oof_primary_events(
        primary_buy_signal,
        start_index=V4Y_EMA_TREND_PERIOD,
        stop_index=train_end,
        fold_count=V4Y_OOF_FOLD_COUNT,
    )
    raw_labels = apply_adaptive_triple_barrier(
        bars,
        atr_period=V4Y_ATR_PERIOD,
        horizon_candles=horizon_candles,
        take_profit_multiplier=V4Y_TAKE_PROFIT_ATR_MULTIPLIER,
        stop_loss_multiplier=V4Y_STOP_LOSS_ATR_MULTIPLIER,
        entry_reference="NEXT_BAR_OPEN",
        entry_slippage_bps=V4Y_ENTRY_SLIPPAGE_BPS,
    )
    costed_labels = build_costed_meta_labels(
        bars,
        oof_events,
        raw_labels,
        fee_bps_per_side=V4Y_FEE_BPS_PER_SIDE,
        slippage_bps_per_side=V4Y_SLIPPAGE_BPS_PER_SIDE,
        entry_slippage_bps=V4Y_ENTRY_SLIPPAGE_BPS,
    )
    event_features = _feature_rows_for_events(features, oof_events)
    training = costed_labels.merge(
        event_features,
        on=["event_index", "fold_id", "primary_signal", "primary_oof"],
        how="inner",
        validate="one_to_one",
    ).sort_values("event_index")
    training = training.loc[
        (training["event_index"] < train_end)
        & (training["label_end_idx"] < train_end)
    ].reset_index(drop=True)
    diagnostics = {
        "primary_signal_generation": V4Y_PRIMARY_SIGNAL_GENERATION,
        "primary_oof_event_count": int(len(oof_events)),
        "costed_label_count": int(len(costed_labels)),
        "matured_outer_train_event_count": int(len(training)),
        "oof_fold_count": V4Y_OOF_FOLD_COUNT,
        "oof_fold_ids": sorted(
            int(value) for value in training["fold_id"].unique().tolist()
        ),
        "all_primary_signals_are_buy": bool(training["primary_signal"].eq("BUY").all()),
        "all_primary_signals_are_oof": bool(training["primary_oof"].eq(True).all()),
        "maximum_training_label_end_index": (
            int(training["label_end_idx"].max()) if not training.empty else None
        ),
        "outer_train_end_index": train_end,
        "training_labels_are_matured_before_outer_validation": bool(
            not training.empty and int(training["label_end_idx"].max()) < train_end
        ),
    }
    return training, diagnostics


def _validation_primary_event_indices(
    primary_buy_signal: pd.Series,
    train_end: int,
    validation_end: int,
    horizon_candles: int,
) -> list[int]:
    final_entry_index = validation_end - horizon_candles - 1
    return [
        index
        for index in range(train_end, final_entry_index + 1)
        if bool(primary_buy_signal.iloc[index])
    ]


def _validation_probability_by_index(
    features: DataFrame,
    event_indices: list[int],
    model: Any,
) -> dict[int, float]:
    if not event_indices:
        return {}
    event_features = features.iloc[event_indices].reset_index(drop=True)
    probabilities = predict_meta_probabilities(model, event_features)
    return {
        index: float(probability)
        for index, probability in zip(event_indices, probabilities, strict=True)
    }


def _validation_row_outputs(
    bars: DataFrame,
    primary_buy_signal: pd.Series,
    train_end: int,
    validation_end: int,
    probability_by_index: dict[int, float],
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    signals: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    accepted_count = 0
    for row_index in range(train_end, validation_end):
        timestamp = pd.Timestamp(bars["date"].iloc[row_index])
        execution_timestamp = pd.Timestamp(bars["date"].iloc[row_index + 1])
        probability = probability_by_index.get(row_index)
        accepted = probability is not None and probability >= threshold
        signal = "BUY" if accepted else "NO_ENTRY"
        if accepted:
            accepted_count += 1
        signals.append(
            {
                "row_index": row_index,
                "signal_timestamp": timestamp,
                "execution_timestamp": execution_timestamp,
                "signal": signal,
                "raw_signal": "BUY"
                if bool(primary_buy_signal.iloc[row_index])
                else "NO_BUY",
                "buy_probability": float(probability)
                if probability is not None
                else 0.0,
                "confidence_threshold": threshold,
            }
        )
        if probability is not None:
            decisions.append(
                {
                    "row_index": row_index,
                    "timestamp": timestamp.isoformat(),
                    "primary_signal": "BUY",
                    "meta_probability": probability,
                    "selected_threshold": threshold,
                    "meta_filter_decision": "ACCEPT" if accepted else "REJECT",
                    "executable_signal": signal,
                    "split_role": "FORWARD_VALIDATION",
                }
            )
    return signals, decisions, accepted_count


def _validation_decision_signals(
    bars: DataFrame,
    features: DataFrame,
    primary_buy_signal: pd.Series,
    train_end: int,
    validation_end: int,
    model: Any,
    threshold: float,
    horizon_candles: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Score validation BUY events; all other bars remain explicit NO_ENTRY."""
    event_indices = _validation_primary_event_indices(
        primary_buy_signal,
        train_end,
        validation_end,
        horizon_candles,
    )
    probability_by_index = _validation_probability_by_index(
        features, event_indices, model
    )
    signals, decisions, accepted_count = _validation_row_outputs(
        bars,
        primary_buy_signal,
        train_end,
        validation_end,
        probability_by_index,
        threshold,
    )
    if not signals:
        raise ValueError("v4y_validation_signal_window_empty")
    counts = {
        "primary_buy_event_count": len(event_indices),
        "meta_accepted_buy_count": accepted_count,
        "meta_rejected_buy_count": len(event_indices) - accepted_count,
        "explicit_no_entry_bar_count": len(signals) - accepted_count,
    }
    return signals, decisions, counts


def _candidate_metrics(replay: dict[str, Any]) -> dict[str, Any]:
    metrics = replay["metrics"]
    trades = replay["trades"]
    net_pnl = float(metrics["final_equity_usd"]) - float(metrics["initial_equity_usd"])
    return {
        "trade_count": int(metrics["trade_count"]),
        "win_rate": float(metrics["win_rate"]),
        "net_profit_factor": metrics["profit_factor_net_of_costs"],
        "net_profit_factor_infinite_no_losses": bool(
            metrics["profit_factor_infinite_no_losses"]
        ),
        "net_pnl_usd": round(net_pnl, 8),
        "net_return_percent": float(metrics["net_return_percent"]),
        "max_drawdown_fraction": float(metrics["max_drawdown_fraction"]),
        "max_drawdown_percent": float(metrics["max_drawdown_percent"]),
        "economic_data_status": str(metrics["economic_data_status"]),
        "trade_count_gate_passed": bool(metrics["trade_count_gate_passed"]),
        "economic_candidate_gate_passed": bool(
            metrics["economic_candidate_gate_passed"]
        ),
        "winning_trade_count": sum(float(item["net_pnl_usd"]) > 0 for item in trades),
        "losing_trade_count": sum(float(item["net_pnl_usd"]) < 0 for item in trades),
        "fees_usd": round(sum(float(item["fees_usd"]) for item in trades), 8),
        "slippage_cost_usd": round(
            sum(float(item["slippage_cost_usd"]) for item in trades), 8
        ),
    }


def _empty_metrics() -> dict[str, Any]:
    return {
        "trade_count": 0,
        "win_rate": None,
        "net_profit_factor": None,
        "net_profit_factor_infinite_no_losses": False,
        "net_pnl_usd": 0.0,
        "net_return_percent": 0.0,
        "max_drawdown_fraction": None,
        "max_drawdown_percent": None,
        "economic_data_status": "NOT_RUN",
        "trade_count_gate_passed": False,
        "economic_candidate_gate_passed": False,
        "winning_trade_count": 0,
        "losing_trade_count": 0,
        "fees_usd": 0.0,
        "slippage_cost_usd": 0.0,
    }


def _skipped_candidate(
    *,
    candidate_id: str,
    symbol: str,
    timeframe: str,
    status: str,
    reason: str,
    primary_signal_counts: dict[str, int],
    primary_sample_gate: dict[str, int | bool | str],
    training_diagnostics: dict[str, Any],
    horizon_candles: int,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "status": status,
        "skip_reason": reason,
        "pair": f"{symbol[:-4]}/USDT",
        "symbol": symbol,
        "timeframe": timeframe,
        "feature_profile": V4Y_FEATURE_PROFILE,
        "horizon_candles": horizon_candles,
        "horizon_hours": V4Y_HORIZON_HOURS,
        "primary_model": V4Y_PRIMARY_SIGNAL_GENERATION,
        "meta_model": "LightGBM LGBMClassifier objective=binary",
        "primary_signal_counts": primary_signal_counts,
        "primary_sample_gate": primary_sample_gate,
        "training_diagnostics": training_diagnostics,
        "selected_meta_probability_threshold": None,
        "threshold_selection": None,
        "validation_signal_counts": {
            "primary_buy_event_count": 0,
            "meta_accepted_buy_count": 0,
            "meta_rejected_buy_count": 0,
            "explicit_no_entry_bar_count": 0,
        },
        "metrics": _empty_metrics(),
        "trades": [],
        "decisions": [],
        "sequence_model_used": False,
    }


def _run_candidate(
    *,
    symbol: str,
    timeframe: str,
    source_bars: DataFrame,
) -> dict[str, Any]:
    pair = f"{symbol[:-4]}/USDT"
    candidate_id = f"{pair}|{timeframe}|{V4Y_FEATURE_PROFILE}"
    horizon_candles = _horizon_candles(timeframe)
    bars = resample_ohlcv(source_bars, timeframe)
    train_end, validation_end = _outer_boundaries(len(bars), horizon_candles)
    candle_features = build_feature_profiles(
        bars, profile_names=(V4Y_FEATURE_PROFILE,)
    )[V4Y_FEATURE_PROFILE]
    features, primary_buy_signal = build_v4y_primary_features(
        bars, candle_features
    )
    if not set(V4Y_META_FEATURE_COLUMNS).issubset(features.columns):
        raise ValueError("v4y_primary_meta_feature_schema_invalid")
    full_signal_count = int(primary_buy_signal.sum())
    train_gate = evaluate_primary_train_sample_gate(
        primary_buy_signal,
        train_start_index=V4Y_EMA_TREND_PERIOD,
        train_end_index=train_end,
    )
    primary_signal_counts = {
        "total_primary_buy_event_count": full_signal_count,
        "train_primary_buy_event_count": int(
            train_gate["observed_primary_buy_count"]
        ),
        "forward_validation_primary_buy_event_count": int(
            primary_buy_signal.iloc[train_end:validation_end].sum()
        ),
    }
    if not bool(train_gate["passed"]):
        return _skipped_candidate(
            candidate_id=candidate_id,
            symbol=symbol,
            timeframe=timeframe,
            status="SKIPPED_INSUFFICIENT_PRIMARY_SIGNALS",
            reason="v4y_insufficient_primary_signals",
            primary_signal_counts=primary_signal_counts,
            primary_sample_gate=train_gate,
            training_diagnostics={"primary_sample_gate": train_gate},
            horizon_candles=horizon_candles,
        )

    try:
        training, training_diagnostics = _training_oof_events(
            bars,
            features,
            primary_buy_signal,
            train_end,
            horizon_candles,
        )
    except ValueError as error:
        code = str(error)
        if code in {"v4x_oof_primary_signals_empty", "v4x_costed_meta_labels_empty"}:
            return _skipped_candidate(
                candidate_id=candidate_id,
                symbol=symbol,
                timeframe=timeframe,
                status="SKIPPED_INSUFFICIENT_OOF_META_SUPPORT",
                reason=code,
                primary_signal_counts=primary_signal_counts,
                primary_sample_gate=train_gate,
                training_diagnostics={"primary_sample_gate": train_gate},
                horizon_candles=horizon_candles,
            )
        raise
    primary_signal_counts["matured_oof_train_buy_event_count"] = int(len(training))
    training_diagnostics["primary_sample_gate"] = train_gate
    try:
        fit_rows, calibration_rows = split_meta_fit_calibration(
            training, outer_train_end_index=train_end
        )
    except ValueError as error:
        code = str(error)
        if code.startswith("v4x_insufficient_meta_"):
            return _skipped_candidate(
                candidate_id=candidate_id,
                symbol=symbol,
                timeframe=timeframe,
                status="SKIPPED_INSUFFICIENT_OOF_META_SUPPORT",
                reason=code,
                primary_signal_counts=primary_signal_counts,
                primary_sample_gate=train_gate,
                training_diagnostics=training_diagnostics,
                horizon_candles=horizon_candles,
            )
        raise

    feature_columns = list(features.columns)
    model, meta_training = fit_lightgbm_meta_filter(
        fit_rows.loc[:, feature_columns],
        fit_rows["meta_label"],
        calibration_rows.loc[:, feature_columns],
        target_accept_fraction=V4Y_TARGET_ACCEPT_FRACTION,
    )
    threshold = float(meta_training["selected_threshold"])
    signals, decisions, validation_counts = _validation_decision_signals(
        bars,
        features,
        primary_buy_signal,
        train_end,
        validation_end,
        model,
        threshold,
        horizon_candles,
    )
    management = TradeManagementConfig(
        atr_period=V4Y_ATR_PERIOD,
        take_profit_atr_multiplier=V4Y_TAKE_PROFIT_ATR_MULTIPLIER,
        stop_loss_atr_multiplier=V4Y_STOP_LOSS_ATR_MULTIPLIER,
        horizon_candles=horizon_candles,
        confidence_threshold=threshold,
        trailing_stop_enabled=False,
    )
    replay = simulate_signals(
        bars,
        signals,
        _execution_config(),
        trade_management_config=management,
    )
    trades = [{"candidate_id": candidate_id, **trade} for trade in replay["trades"]]
    for row in decisions:
        row["candidate_id"] = candidate_id
        row["pair"] = pair
        row["timeframe"] = timeframe
    training_diagnostics.update(
        {
            "fit_rows": int(len(fit_rows)),
            "fit_start_event_index": int(fit_rows["event_index"].min()),
            "fit_end_event_index": int(fit_rows["event_index"].max()),
            "calibration_rows": int(len(calibration_rows)),
            "calibration_start_event_index": int(
                calibration_rows["event_index"].min()
            ),
            "fit_max_label_end_index": int(fit_rows["label_end_idx"].max()),
            "fit_labels_end_before_calibration": bool(
                int(fit_rows["label_end_idx"].max())
                < int(calibration_rows["event_index"].min())
            ),
            "fit_labels_are_oof_primary_buy_only": bool(
                fit_rows["primary_oof"].eq(True).all()
                and fit_rows["primary_signal"].eq("BUY").all()
            ),
            "validation_labels_used_for_fit": False,
            "validation_labels_used_for_threshold": False,
        }
    )
    return {
        "candidate_id": candidate_id,
        "status": "TRAINED",
        "pair": pair,
        "symbol": symbol,
        "timeframe": timeframe,
        "feature_profile": V4Y_FEATURE_PROFILE,
        "horizon_candles": horizon_candles,
        "horizon_hours": V4Y_HORIZON_HOURS,
        "primary_model": V4Y_PRIMARY_SIGNAL_GENERATION,
        "primary_signal_definition": (
            "Bollinger(20, 2.0) upper-band breakout on closed candle t, "
            "with close[t] > EMA200[t]; signal is evaluated after candle close "
            "and execution is next-bar open"
        ),
        "primary_signal_counts": primary_signal_counts,
        "primary_sample_gate": train_gate,
        "meta_model": "LightGBM LGBMClassifier objective=binary",
        "meta_target": "NET_POSITIVE_AFTER_FEES_AND_SLIPPAGE",
        "meta_training": meta_training,
        "selected_meta_probability_threshold": threshold,
        "threshold_selection": {
            "method": "TOP_PERCENTILE_ORDER_STATISTIC",
            "target_accept_fraction": V4Y_TARGET_ACCEPT_FRACTION,
            "calibration_fraction": V4Y_META_CALIBRATION_FRACTION,
            "calibration_labels_used": False,
            "validation_labels_used": False,
            "validation_probabilities_used": False,
        },
        "training_diagnostics": training_diagnostics,
        "validation_signal_counts": validation_counts,
        "validation_window": {
            "start_index": train_end,
            "end_index": validation_end - 1,
            "validation_end_exclusive": validation_end,
            "no_new_entries_inside_final_horizon_bars": horizon_candles,
            "all_trades_close_before_holdout": True,
        },
        "metrics": _candidate_metrics(replay),
        "trades": trades,
        "decisions": decisions,
        "sequence_model_used": False,
    }


def _render_summary(report: dict[str, Any]) -> str:
    lines = [
        "## AliX V4-Y Bollinger / EMA200 Primary + LightGBM Meta-Labeling Evaluation",
        "",
        f"Commit SHA: {report['commit_sha']}",
        "",
        "Primary: causal Bollinger(20, 2.0) upper-band breakout, gated by close above EMA200.",
        "Signal timing: closed-candle decision; next-bar-open execution.",
        "Meta-Model: LightGBM binary classifier trained only on matured, purged OOF Primary BUY events.",
        "Meta target: positive net outcome after both-side fees and slippage under ATR Triple-Barrier exits.",
        "Threshold: top 50% order statistic from purged Outer-Train OOF probabilities; calibration labels are not used.",
        "Sample gate: at least 50 raw Primary BUY events in the chronological Outer-Train window before any meta labeling or fit.",
        "Signal target: at least 100 Primary BUY events across the 730-day dataset per candidate; reported diagnostically, not forced by tuning.",
        "Horizon: fixed 24 hours across timeframes (96 / 48 / 24 candles for 15m / 30m / 1h).",
        "",
        "| Candidate | Status | Primary BUY 730d | Train BUY | Matured OOF | Fit / Cal | Threshold | Validation BUY | Accepted | Trades | Win Rate | Net PF | Net PnL USD | Max DD |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["candidates"]:
        counts = item.get("primary_signal_counts", {})
        training = item.get("training_diagnostics", {})
        validation = item.get("validation_signal_counts", {})
        metrics = item.get("metrics", {})
        threshold = item.get("selected_meta_probability_threshold")
        threshold_text = "N/A" if threshold is None else f"{float(threshold):.6f}"
        fit_cal = (
            f"{training.get('fit_rows', 0)} / {training.get('calibration_rows', 0)}"
            if item["status"] == "TRAINED"
            else "N/A"
        )
        win_rate = metrics.get("win_rate")
        win_text = "N/A" if win_rate is None else f"{float(win_rate):.2%}"
        pf_value = metrics.get("net_profit_factor")
        pf = (
            "INF"
            if metrics.get("net_profit_factor_infinite_no_losses")
            else "N/A"
            if pf_value is None
            else f"{float(pf_value):.4f}"
        )
        dd = metrics.get("max_drawdown_percent")
        dd_text = "N/A" if dd is None else f"{float(dd):.2f}%"
        lines.append(
            f"| {item['candidate_id']} | {item['status']} "
            f"| {counts.get('total_primary_buy_event_count', 0)} "
            f"| {counts.get('train_primary_buy_event_count', 0)} "
            f"| {counts.get('matured_oof_train_buy_event_count', 0)} "
            f"| {fit_cal} | {threshold_text} "
            f"| {validation.get('primary_buy_event_count', 0)} "
            f"| {validation.get('meta_accepted_buy_count', 0)} "
            f"| {metrics.get('trade_count', 0)} | {win_text} | {pf} "
            f"| {float(metrics.get('net_pnl_usd', 0.0)):.2f} | {dd_text} |"
        )
        if item.get("skip_reason"):
            lines.append(f"\nSkipped: `{item['skip_reason']}`.")
    passed = sum(
        bool(item["metrics"].get("economic_candidate_gate_passed"))
        for item in report["candidates"]
    )
    signal_targets = sum(
        int(
            item.get("primary_signal_counts", {}).get(
                "total_primary_buy_event_count", 0
            )
        )
        >= 100
        for item in report["candidates"]
    )
    sample_gate_passed = sum(
        bool(item.get("primary_sample_gate", {}).get("passed"))
        for item in report["candidates"]
    )
    lines.extend(
        [
            "",
            f"Primary signal target (>=100 across 730 days): {signal_targets}/{len(report['candidates'])} candidates met.",
            f"Train sample gate (>=50 Primary BUY): {sample_gate_passed}/{len(report['candidates'])} candidates passed.",
            f"Economic gates passed: {passed}/{len(report['candidates'])} candidates.",
            "Acceptance policy is unchanged: at least 30 trades, net PF > 1.5, and Max DD < 15%.",
            "Candidate metrics are reported independently; cross-candidate portfolio overlap and allocation are not modeled.",
            "",
            "Safety: Champion=null, order_permission=false, trading_authorization=NO-GO, private_api_calls=0, real_order_attempts=0.",
            "No orders or execution APIs are called. Sequence models: none.",
            "Forward validation only; final holdout is untouched. Research output is not a promotion or trading authorization.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _validate_v4y_run_request(days: int, commit_sha: str) -> None:
    if days != 730:
        raise ValueError("v4y_dataset_window_must_be_730_days")
    if len(commit_sha) != 40 or any(
        char not in "0123456789abcdef" for char in commit_sha
    ):
        raise ValueError("v4y_commit_sha_invalid")


def _run_v4y_candidates(datasets: dict[str, DataFrame]) -> list[dict[str, Any]]:
    return [
        _run_candidate(
            symbol=symbol,
            timeframe=timeframe,
            source_bars=datasets[symbol],
        )
        for symbol in SYMBOLS
        for timeframe in V4Y_TIMEFRAMES
    ]


def _build_v4y_report(
    commit_sha: str,
    days: int,
    provenance: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    trained = [item for item in candidates if item["status"] == "TRAINED"]
    return {
        "schema_version": V4Y_REPORT_SCHEMA,
        "commit_sha": commit_sha,
        "requested_days": days,
        "provider": "Binance Spot public klines API",
        "input_datasets": provenance,
        "dataset_symbols": list(SYMBOLS),
        "timeframes": list(V4Y_TIMEFRAMES),
        "candidate_count": len(candidates),
        "trained_candidate_count": len(trained),
        "economic_evaluation_performed": bool(trained),
        "primary_model": {
            "type": "FIXED_RULE",
            "definition": V4Y_PRIMARY_SIGNAL_GENERATION,
            "bollinger_period": V4Y_BOLLINGER_PERIOD,
            "bollinger_standard_deviations": V4Y_BOLLINGER_STDDEV,
            "trend_filter": f"CLOSE_ABOVE_EMA{V4Y_EMA_TREND_PERIOD}",
            "minimum_train_primary_buy_signals": V4Y_MINIMUM_PRIMARY_TRAIN_SIGNALS,
            "full_window_signal_target": 100,
            "fit_performed": False,
            "uses_current_and_past_closed_candles_only": True,
            "execution": "NEXT_BAR_OPEN",
        },
        "meta_model": {
            "family": "LightGBM LGBMClassifier objective=binary",
            "target": "NET_POSITIVE_AFTER_FEES_AND_SLIPPAGE",
            "training_signal_requirement": "OOF_PRIMARY_BUY_ONLY",
            "training_labels_must_be_matured_before_next_window": True,
            "calibration_labels_used": False,
            "validation_labels_used_for_fit_or_threshold": False,
            "target_accept_fraction": V4Y_TARGET_ACCEPT_FRACTION,
            "calibration_fraction": V4Y_META_CALIBRATION_FRACTION,
            "minimum_fit_rows": V4Y_MINIMUM_FIT_ROWS,
            "minimum_rows_per_class": V4Y_MINIMUM_CLASS_ROWS,
            "sample_gate": {
                "minimum_primary_train_buy_signals": V4Y_MINIMUM_PRIMARY_TRAIN_SIGNALS,
                "skip_status": "SKIPPED_INSUFFICIENT_PRIMARY_SIGNALS",
                "applied_before_meta_labeling_and_model_fit": True,
            },
        },
        "label_contract": {
            "method": "ATR_ADAPTIVE_TRIPLE_BARRIER",
            "take_profit_atr_multiplier": V4Y_TAKE_PROFIT_ATR_MULTIPLIER,
            "stop_loss_atr_multiplier": V4Y_STOP_LOSS_ATR_MULTIPLIER,
            "horizon_hours": V4Y_HORIZON_HOURS,
            "horizon_candles_by_timeframe": V4Y_HORIZON_CANDLES_BY_TIMEFRAME,
            "entry_reference": "NEXT_BAR_OPEN",
            "entry_slippage_bps": V4Y_ENTRY_SLIPPAGE_BPS,
            "meta_label": "1_IF_NET_RETURN_AFTER_BOTH_SIDE_COSTS_GT_0_ELSE_0",
            "feature_target_separation": True,
        },
        "execution_cost_contract": {
            "fee_bps_per_side": V4Y_FEE_BPS_PER_SIDE,
            "slippage_bps_per_side": V4Y_SLIPPAGE_BPS_PER_SIDE,
            "initial_equity_usd": V4Y_INITIAL_EQUITY_USD,
            "position_fraction": 1.0,
            "leverage": 1.0,
            "pyramiding": False,
            "short_positions_supported": False,
        },
        "outer_split_contract": {
            "method": "CHRONOLOGICAL_OUTER_TRAIN_AND_FORWARD_VALIDATION",
            "outer_train_end_fraction": V4Y_OUTER_TRAIN_END_FRACTION,
            "forward_validation_end_fraction": V4Y_FORWARD_VALIDATION_END_FRACTION,
            "final_holdout_start_fraction": V4Y_FORWARD_VALIDATION_END_FRACTION,
            "validation_used_for_training": False,
            "validation_used_for_threshold": False,
            "final_holdout_used": False,
        },
        "economic_acceptance_contract": {
            "minimum_trade_count": V4Y_ECONOMIC_MINIMUM_TRADES,
            "minimum_net_profit_factor": V4Y_ECONOMIC_MINIMUM_NET_PF,
            "maximum_drawdown_fraction": V4Y_MAXIMUM_DRAWDOWN,
            "promotion_is_automatic": False,
        },
        "candidates": candidates,
        "promotion_status": "BLOCKED_PENDING_INDEPENDENT_OOS_AND_OWNER_REVIEW",
        "safety": {
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "private_api_calls": 0,
            "real_order_attempts": 0,
            "orders_or_execution_api_used": False,
            "credentials_used": False,
            "short_positions_supported": False,
            "sequence_model_used": False,
        },
    }


def _write_v4y_outputs(
    report: dict[str, Any],
    report_path: Path,
    trades_path: Path,
    decisions_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    trade_rows = [trade for item in report["candidates"] for trade in item["trades"]]
    decision_rows = [row for item in report["candidates"] for row in item["decisions"]]
    _write_csv(trades_path, trade_rows)
    _write_csv(decisions_path, decision_rows)


def _print_v4y_run_summary(report: dict[str, Any]) -> None:
    print(
        json.dumps(
            {
                "commit_sha": report["commit_sha"],
                "candidate_count": report["candidate_count"],
                "trained_candidate_count": report["trained_candidate_count"],
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


def run_v4y(
    input_dir: Path,
    report_path: Path,
    trades_path: Path,
    decisions_path: Path,
    *,
    days: int,
    commit_sha: str,
) -> dict[str, Any]:
    _validate_v4y_run_request(days, commit_sha)
    datasets, provenance = _load_diagnostic_datasets(input_dir, days, commit_sha)
    candidates = _run_v4y_candidates(datasets)
    report = _build_v4y_report(commit_sha, days, provenance, candidates)
    _write_v4y_outputs(report, report_path, trades_path, decisions_path)
    _print_v4y_run_summary(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-trades-csv", type=Path, required=True)
    parser.add_argument("--output-decisions-csv", type=Path, required=True)
    parser.add_argument("--days", type=int, choices=(730,), default=730)
    parser.add_argument("--commit-sha", required=True)
    args = parser.parse_args()
    report = run_v4y(
        args.input_dir,
        args.output_report,
        args.output_trades_csv,
        args.output_decisions_csv,
        days=args.days,
        commit_sha=args.commit_sha,
    )
    return 0 if report["candidate_count"] == len(SYMBOLS) * len(V4Y_TIMEFRAMES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
