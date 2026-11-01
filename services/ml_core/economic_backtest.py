"""Deterministic, non-training economic replay for persisted V4-H signals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

V4H_SOURCE_SCHEMA = "alix.ml-core.v4h-tuning-geometry.v1"
V4H_SOURCE_SHA = "e34861c4873b7cc2170aaaf6c79ed9b464df0a9"
V4H_SOURCE_RUN_ID = 41293168191
V4H_SOURCE_ARTIFACT_ID = 13080269872
V4H_ACCEPTED_GEOMETRY = "V4H_Tighter_SL"
V4H_MIN_HOLD_PRECISION = 0.40
V4H_MIN_HOLD_RECALL = 0.20
LABELS = ("BUY", "SELL", "HOLD")
TIMEFRAME_MINUTES = {"15m": 15, "30m": 30, "1h": 60, "4h": 240}
OHLCV_COLUMNS = ("date", "open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class ExecutionConfig:
    """Explicit one-times-equity execution assumptions for replay."""

    initial_equity: float = 10_000.0
    fee_bps: float = 25.0
    slippage_bps: float = 20.0
    position_fraction: float = 1.0
    maximum_drawdown: float = 0.15


@dataclass(frozen=True)
class OpenPosition:
    """One non-pyramided long or short position."""

    direction: int
    entry_timestamp: str
    entry_reference_price: float
    entry_fill_price: float
    quantity: float
    equity_at_entry: float
    entry_fee: float
    entry_slippage_cost: float


def validate_execution_config(config: ExecutionConfig) -> None:
    """Reject invalid or leveraged simulation assumptions."""
    values = (
        config.initial_equity,
        config.fee_bps,
        config.slippage_bps,
        config.position_fraction,
        config.maximum_drawdown,
    )
    if not all(np.isfinite(float(value)) for value in values):
        raise ValueError("economic_config_values_must_be_finite")
    if config.initial_equity <= 0:
        raise ValueError("initial_equity_must_be_positive")
    if config.fee_bps < 0 or config.slippage_bps < 0:
        raise ValueError("execution_costs_must_be_non_negative")
    if not 0 < config.position_fraction <= 1:
        raise ValueError("position_fraction_must_be_in_(0,1]")
    if not 0 < config.maximum_drawdown < 1:
        raise ValueError("maximum_drawdown_must_be_in_(0,1)")


def validate_source_report(report: dict[str, Any]) -> None:
    """Require the exact approved V4-H report and closed safety state."""
    safety = report.get("safety")
    valid_safety = (
        isinstance(safety, dict)
        and safety.get("Champion") is None
        and safety.get("order_permission") is False
        and safety.get("trading_authorization") == "NO-GO"
        and safety.get("private_api_calls") == 0
        and safety.get("real_order_attempts") == 0
    )
    if (
        report.get("schema_version") != V4H_SOURCE_SCHEMA
        or report.get("commit_sha") != V4H_SOURCE_SHA
        or report.get("diagnostic_scope") != "CLASSIFICATION_ONLY"
        or report.get("economic_evaluation_performed") is not False
        or report.get("trained_models") != 96
        or report.get("experiment_count") != 48
        or report.get("acceptance_policy", {}).get("status") != "MET"
        or report.get("acceptance_policy", {}).get("strict_acceptance_count") != 13
        or not valid_safety
    ):
        raise ValueError("v4h_source_report_contract_invalid")


def extract_qualifying_candidates(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the 13 strict Forward-Validation passes and their persisted scores."""
    validate_source_report(report)
    candidates: list[dict[str, Any]] = []
    for experiment in report["experiments"]:
        if experiment.get("scenario") != V4H_ACCEPTED_GEOMETRY:
            continue
        variants = experiment.get("model_variants") or {}
        for policy, variant in variants.items():
            threshold_report = variant.get("threshold_optimization") or {}
            forward = threshold_report.get("post_calibration_validation")
            if not isinstance(forward, dict):
                continue
            hold = forward.get("hold_metrics") or {}
            precision = hold.get("precision")
            recall = hold.get("recall")
            if precision is None or recall is None:
                continue
            if float(precision) < V4H_MIN_HOLD_PRECISION or float(recall) < V4H_MIN_HOLD_RECALL:
                continue
            threshold = threshold_report.get("selected_threshold")
            start_index = forward.get("start_row_index")
            selected_parameters = (variant.get("hyperparameter_tuning") or {}).get(
                "selected_parameters"
            )
            raw_records = variant.get("validation_probabilities")
            if (
                threshold is None
                or start_index is None
                or not isinstance(selected_parameters, dict)
                or not isinstance(raw_records, list)
            ):
                raise ValueError("accepted_v4h_candidate_replay_data_missing")
            forward_records = [
                item
                for item in raw_records
                if int(item["row_index"]) >= int(start_index)
            ]
            if len(forward_records) != int(forward.get("row_count", -1)):
                raise ValueError("forward_probability_row_count_mismatch")
            if not forward_records:
                raise ValueError("forward_probability_rows_empty")
            candidate_id = "|".join(
                (
                    str(experiment["scenario"]),
                    str(experiment["pair"]),
                    str(experiment["timeframe"]),
                    str(experiment["feature_profile"]),
                    str(policy),
                )
            )
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "scenario": experiment["scenario"],
                    "pair": experiment["pair"],
                    "timeframe": experiment["timeframe"],
                    "feature_profile": experiment["feature_profile"],
                    "weighting_policy": policy,
                    "selected_parameters": selected_parameters,
                    "hold_precision": float(precision),
                    "hold_recall": float(recall),
                    "selected_hold_threshold": float(threshold),
                    "forward_start_row_index": int(start_index),
                    "forward_row_count": len(forward_records),
                    "probability_records": forward_records,
                }
            )
    expected = int(report["acceptance_policy"]["strict_acceptance_count"])
    if len(candidates) != expected or len({item["candidate_id"] for item in candidates}) != expected:
        raise ValueError("v4h_qualifying_candidate_count_mismatch")
    if any(item["scenario"] != V4H_ACCEPTED_GEOMETRY for item in candidates):
        raise ValueError("v4h_qualifying_candidate_geometry_mismatch")
    return candidates


def validate_market_manifest(
    source_metadata: dict[str, Any],
    current_manifest: dict[str, Any],
    *,
    current_commit_sha: str,
) -> None:
    """Fail closed unless re-fetched candles exactly match the V4-H input hash."""
    expected_fields = (
        current_manifest.get("symbol") == source_metadata.get("symbol"),
        current_manifest.get("interval") == "1m",
        current_manifest.get("requested_days") == source_metadata.get("requested_days"),
        current_manifest.get("row_count") == source_metadata.get("row_count"),
        current_manifest.get("start_utc") == source_metadata.get("start_utc"),
        current_manifest.get("end_utc") == source_metadata.get("end_utc"),
        current_manifest.get("csv_sha256") == source_metadata.get("csv_sha256"),
        current_manifest.get("commit_sha") == current_commit_sha,
        current_manifest.get("public_api_only") is True,
        current_manifest.get("credentials_used") is False,
        current_manifest.get("missing_minute_intervals") == 0,
        current_manifest.get("private_api_calls") == 0,
        current_manifest.get("real_order_attempts") == 0,
    )
    if not all(expected_fields):
        raise ValueError("v4h_market_data_replay_integrity_mismatch")


def load_validated_minute_data(
    input_dir: Path,
    source_metadata: dict[str, Any],
    *,
    current_commit_sha: str,
) -> DataFrame:
    """Load minute candles only after provenance and byte-level digest checks."""
    symbol = str(source_metadata["symbol"])
    stem = f"binance-{symbol.lower()}-1m"
    csv_path = input_dir / f"{stem}.csv"
    manifest_path = input_dir / f"{stem}.manifest.json"
    if not csv_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"economic_market_data_missing:{symbol}")
    manifest_bytes = manifest_path.read_bytes()
    current_manifest = json.loads(manifest_bytes.decode("utf-8"))
    csv_digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    if current_manifest.get("csv_sha256") != csv_digest:
        raise ValueError(f"economic_market_csv_digest_invalid:{symbol}")
    validate_market_manifest(
        source_metadata, current_manifest, current_commit_sha=current_commit_sha
    )
    candles = pd.read_csv(csv_path)
    expected_rows = int(source_metadata["row_count"])
    if len(candles) != expected_rows:
        raise ValueError(f"economic_market_row_count_invalid:{symbol}")
    candles["date"] = pd.to_datetime(candles["date"], utc=True, errors="coerce")
    if candles["date"].isna().any():
        raise ValueError(f"economic_market_timestamp_invalid:{symbol}")
    if candles["date"].duplicated().any() or not candles["date"].is_monotonic_increasing:
        raise ValueError(f"economic_market_timestamp_order_invalid:{symbol}")
    if len(candles) > 1 and not bool(
        (candles["date"].diff().iloc[1:] == pd.Timedelta(minutes=1)).all()
    ):
        raise ValueError(f"economic_market_gap_detected:{symbol}")
    return candles.loc[:, list(OHLCV_COLUMNS)]


def resample_ohlcv(candles: DataFrame, timeframe: str) -> DataFrame:
    """Rebuild complete epoch-aligned OHLCV bars using the V4-H aggregation rule."""
    if timeframe not in TIMEFRAME_MINUTES:
        raise ValueError("economic_timeframe_out_of_policy")
    missing = sorted(set(OHLCV_COLUMNS) - set(candles.columns))
    if missing or candles.empty:
        raise ValueError("economic_ohlcv_input_invalid")
    source = candles.loc[:, list(OHLCV_COLUMNS)].copy().reset_index(drop=True)
    source["date"] = pd.to_datetime(source["date"], utc=True, errors="coerce")
    numeric = source.loc[:, ["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if source["date"].isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("economic_ohlcv_values_invalid")
    if source["date"].duplicated().any() or not source["date"].is_monotonic_increasing:
        raise ValueError("economic_ohlcv_timestamps_invalid")
    opened = numeric["open"].to_numpy(dtype=float)
    high = numeric["high"].to_numpy(dtype=float)
    low = numeric["low"].to_numpy(dtype=float)
    close = numeric["close"].to_numpy(dtype=float)
    volume = numeric["volume"].to_numpy(dtype=float)
    if (
        np.any(opened <= 0)
        or np.any(high <= 0)
        or np.any(low <= 0)
        or np.any(close <= 0)
        or np.any(volume < 0)
        or np.any(high < np.maximum(opened, close))
        or np.any(low > np.minimum(opened, close))
        or np.any(high < low)
    ):
        raise ValueError("economic_ohlcv_geometry_invalid")
    source.loc[:, ["open", "high", "low", "close", "volume"]] = numeric
    minutes = TIMEFRAME_MINUTES[timeframe]
    grouped = source.set_index("date").resample(
        f"{minutes}min", origin="epoch", closed="left", label="left"
    )
    bars = grouped.agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    complete = grouped["close"].count() == minutes
    result = bars.loc[complete].dropna().reset_index()
    if result.empty:
        raise ValueError("economic_resampled_bars_empty")
    return result.loc[:, list(OHLCV_COLUMNS)]


def probability_to_signal(probabilities: dict[str, Any], hold_threshold: float) -> str:
    """Apply the same HOLD-threshold and BUY/SELL tie-break rule as V4-H."""
    if not np.isfinite(float(hold_threshold)) or not 0 <= float(hold_threshold) <= 1:
        raise ValueError("economic_hold_threshold_invalid")
    try:
        buy = float(probabilities["BUY"])
        sell = float(probabilities["SELL"])
        hold = float(probabilities["HOLD"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("economic_probabilities_invalid") from None
    values = np.asarray([buy, sell, hold], dtype=float)
    if (
        not bool(np.isfinite(values).all())
        or bool(np.any(values < 0))
        or bool(np.any(values > 1))
        or abs(float(values.sum()) - 1.0) > 1e-6
    ):
        raise ValueError("economic_probabilities_invalid")
    if hold >= float(hold_threshold) and hold > max(buy, sell):
        return "HOLD"
    return "BUY" if buy >= sell else "SELL"


def build_forward_signals(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    """Transform persisted Forward-Validation probabilities without using labels."""
    rows = candidate["probability_records"]
    threshold = float(candidate["selected_hold_threshold"])
    signals: list[dict[str, Any]] = []
    previous_index = -1
    previous_timestamp: pd.Timestamp | None = None
    delta = timedelta(minutes=TIMEFRAME_MINUTES[candidate["timeframe"]])
    for row in rows:
        row_index = int(row["row_index"])
        timestamp = pd.Timestamp(row["timestamp"])
        if timestamp.tzinfo is None or row_index <= previous_index:
            raise ValueError("economic_forward_signal_order_invalid")
        timestamp = timestamp.tz_convert("UTC")
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError("economic_forward_signal_timestamp_invalid")
        signal = probability_to_signal(row["probabilities"], threshold)
        signals.append(
            {
                "row_index": row_index,
                "signal_timestamp": timestamp,
                "execution_timestamp": timestamp + delta,
                "signal": signal,
            }
        )
        previous_index = row_index
        previous_timestamp = timestamp
    if len(signals) != int(candidate["forward_row_count"]):
        raise ValueError("economic_forward_signal_count_mismatch")
    return signals


def _open_position(
    equity: float,
    direction: int,
    reference_price: float,
    timestamp: pd.Timestamp,
    config: ExecutionConfig,
) -> tuple[OpenPosition, float]:
    """Open one 1x-equity position and immediately mark entry costs."""
    fee_rate = config.fee_bps / 10_000.0
    slippage_rate = config.slippage_bps / 10_000.0
    quantity = equity * config.position_fraction / reference_price
    entry_fill = reference_price * (1.0 + direction * slippage_rate)
    entry_fee = quantity * entry_fill * fee_rate
    entry_slippage = quantity * abs(entry_fill - reference_price)
    position = OpenPosition(
        direction=direction,
        entry_timestamp=timestamp.isoformat(),
        entry_reference_price=reference_price,
        entry_fill_price=entry_fill,
        quantity=quantity,
        equity_at_entry=equity,
        entry_fee=entry_fee,
        entry_slippage_cost=entry_slippage,
    )
    marked_equity = equity - entry_fee - entry_slippage
    if marked_equity <= 0:
        raise ValueError("economic_equity_non_positive_after_entry")
    return position, marked_equity


def _mark_position(position: OpenPosition, mark_price: float) -> float:
    """Mark a position to market after its entry fee and slippage."""
    market_pnl = (
        position.quantity
        * position.direction
        * (mark_price - position.entry_reference_price)
    )
    return position.equity_at_entry + market_pnl - position.entry_fee - position.entry_slippage_cost


def _close_position(
    position: OpenPosition,
    reference_price: float,
    timestamp: pd.Timestamp,
    reason: str,
    config: ExecutionConfig,
) -> tuple[dict[str, Any], float]:
    """Close a position with exit slippage and a second-side fee."""
    fee_rate = config.fee_bps / 10_000.0
    slippage_rate = config.slippage_bps / 10_000.0
    exit_fill = reference_price * (1.0 - position.direction * slippage_rate)
    exit_fee = position.quantity * exit_fill * fee_rate
    exit_slippage = position.quantity * abs(exit_fill - reference_price)
    gross_pnl = (
        position.quantity
        * position.direction
        * (reference_price - position.entry_reference_price)
    )
    fees = position.entry_fee + exit_fee
    slippage = position.entry_slippage_cost + exit_slippage
    net_pnl = gross_pnl - fees - slippage
    equity = position.equity_at_entry + net_pnl
    if equity <= 0:
        raise ValueError("economic_equity_non_positive_after_exit")
    trade = {
        "direction": "LONG" if position.direction == 1 else "SHORT",
        "entry_timestamp": position.entry_timestamp,
        "exit_timestamp": timestamp.isoformat(),
        "entry_reference_price": round(position.entry_reference_price, 10),
        "entry_fill_price": round(position.entry_fill_price, 10),
        "exit_reference_price": round(reference_price, 10),
        "exit_fill_price": round(exit_fill, 10),
        "quantity": round(position.quantity, 10),
        "gross_market_pnl_usd": round(gross_pnl, 8),
        "fees_usd": round(fees, 8),
        "slippage_cost_usd": round(slippage, 8),
        "net_pnl_usd": round(net_pnl, 8),
        "net_return_fraction": round(net_pnl / position.equity_at_entry, 10),
        "exit_reason": reason,
    }
    return trade, equity


def max_drawdown_from_equity(equity_values: list[float]) -> float:
    """Return peak-to-trough drawdown as a positive fraction."""
    if not equity_values or not all(np.isfinite(float(value)) for value in equity_values):
        raise ValueError("economic_equity_curve_invalid")
    peak = float(equity_values[0])
    maximum = 0.0
    for raw in equity_values:
        equity = float(raw)
        if equity <= 0:
            raise ValueError("economic_equity_non_positive")
        peak = max(peak, equity)
        maximum = max(maximum, (peak - equity) / peak)
    return maximum


def _metrics(
    trades: list[dict[str, Any]],
    equity_values: list[float],
    config: ExecutionConfig,
) -> dict[str, Any]:
    """Calculate net-of-cost portfolio and trade statistics."""
    pnls = [float(item["net_pnl_usd"]) for item in trades]
    wins = [value for value in pnls if value > 0]
    losses = [value for value in pnls if value < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else None
    pf_infinite = bool(trades and gross_profit > 0 and gross_loss == 0)
    final_equity = float(equity_values[-1])
    max_drawdown = max_drawdown_from_equity(equity_values)
    trade_count = len(trades)
    pf_pass = (pf is not None and pf > 1.5) or pf_infinite
    drawdown_pass = max_drawdown < config.maximum_drawdown
    return {
        "initial_equity_usd": round(config.initial_equity, 8),
        "final_equity_usd": round(final_equity, 8),
        "net_return_fraction": round(final_equity / config.initial_equity - 1.0, 10),
        "net_return_percent": round((final_equity / config.initial_equity - 1.0) * 100.0, 8),
        "trade_count": trade_count,
        "win_rate": round(len(wins) / trade_count, 8) if trade_count else 0.0,
        "profit_factor_net_of_costs": round(pf, 8) if pf is not None else None,
        "profit_factor_infinite_no_losses": pf_infinite,
        "gross_profit_usd": round(gross_profit, 8),
        "gross_loss_usd": round(gross_loss, 8),
        "average_win_usd": round(float(np.mean(wins)), 8) if wins else None,
        "average_loss_usd": round(float(np.mean(losses)), 8) if losses else None,
        "average_net_pnl_per_trade_usd": round(float(np.mean(pnls)), 8) if pnls else None,
        "max_drawdown_fraction": round(max_drawdown, 10),
        "max_drawdown_percent": round(max_drawdown * 100.0, 8),
        "profit_factor_gate_passed": pf_pass,
        "drawdown_gate_passed": drawdown_pass,
        "economic_candidate_gate_passed": bool(pf_pass and drawdown_pass),
        "maximum_drawdown_limit_fraction": config.maximum_drawdown,
    }


def simulate_signals(
    bars: DataFrame,
    signals: list[dict[str, Any]],
    config: ExecutionConfig,
) -> dict[str, Any]:
    """Replay next-bar-open signals; HOLD closes and never opens a position."""
    validate_execution_config(config)
    if bars.empty or not signals:
        raise ValueError("economic_bars_or_signals_empty")
    required = {"date", "open", "close"}
    if not required.issubset(bars.columns):
        raise ValueError("economic_execution_bars_invalid")
    normalized = bars.copy().reset_index(drop=True)
    normalized["date"] = pd.to_datetime(normalized["date"], utc=True, errors="coerce")
    if normalized["date"].isna().any() or normalized["date"].duplicated().any():
        raise ValueError("economic_execution_timestamps_invalid")
    if not normalized["date"].is_monotonic_increasing:
        raise ValueError("economic_execution_timestamps_invalid")
    for column in ("open", "close"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        values = normalized[column].to_numpy(dtype=float)
        if not bool(np.isfinite(values).all()) or bool(np.any(values <= 0)):
            raise ValueError("economic_execution_prices_invalid")
    index_by_time = {
        timestamp: index for index, timestamp in enumerate(normalized["date"].tolist())
    }
    action_by_index: dict[int, str] = {}
    for item in signals:
        timestamp = pd.Timestamp(item["execution_timestamp"])
        if timestamp.tzinfo is None:
            raise ValueError("economic_execution_signal_timezone_required")
        timestamp = timestamp.tz_convert("UTC")
        index = index_by_time.get(timestamp)
        if index is None:
            raise ValueError("economic_signal_price_timestamp_mismatch")
        signal = item["signal"]
        if signal not in LABELS or index in action_by_index:
            raise ValueError("economic_signal_contract_invalid")
        action_by_index[index] = signal
    first_index = min(action_by_index)
    last_index = max(action_by_index)
    position: OpenPosition | None = None
    cash_equity = float(config.initial_equity)
    peak = cash_equity
    maximum_drawdown = 0.0
    trades: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = [
        {
            "timestamp": normalized["date"].iloc[first_index].isoformat(),
            "equity_usd": round(cash_equity, 8),
            "event": "START",
        }
    ]
    for index in range(first_index, last_index + 1):
        bar = normalized.iloc[index]
        timestamp = pd.Timestamp(bar["date"])
        desired = action_by_index.get(index)
        if desired is not None:
            direction = 1 if desired == "BUY" else -1 if desired == "SELL" else 0
            if position is not None and direction != position.direction:
                trade, cash_equity = _close_position(
                    position,
                    float(bar["open"]),
                    timestamp,
                    "HOLD" if direction == 0 else "SIGNAL_REVERSAL",
                    config,
                )
                trades.append(trade)
                position = None
                equity_curve.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "equity_usd": round(cash_equity, 8),
                        "event": "POSITION_CLOSED",
                    }
                )
            if direction != 0 and position is None:
                position, marked = _open_position(
                    cash_equity,
                    direction,
                    float(bar["open"]),
                    timestamp,
                    config,
                )
                peak = max(peak, marked)
                maximum_drawdown = max(maximum_drawdown, (peak - marked) / peak)
                equity_curve.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "equity_usd": round(marked, 8),
                        "event": "POSITION_OPENED",
                    }
                )
        if position is not None:
            marked_equity = _mark_position(position, float(bar["close"]))
        else:
            marked_equity = cash_equity
        if marked_equity <= 0:
            raise ValueError("economic_equity_non_positive")
        peak = max(peak, marked_equity)
        maximum_drawdown = max(
            maximum_drawdown, (peak - marked_equity) / peak
        )
        equity_curve.append(
            {
                "timestamp": timestamp.isoformat(),
                "equity_usd": round(marked_equity, 8),
                "event": "BAR_CLOSE_MARK",
            }
        )
    if position is not None:
        final_bar = normalized.iloc[last_index]
        final_timestamp = pd.Timestamp(final_bar["date"])
        trade, cash_equity = _close_position(
            position,
            float(final_bar["close"]),
            final_timestamp,
            "END_OF_FORWARD_WINDOW",
            config,
        )
        trades.append(trade)
        position = None
        peak = max(peak, cash_equity)
        maximum_drawdown = max(maximum_drawdown, (peak - cash_equity) / peak)
        equity_curve.append(
            {
                "timestamp": final_timestamp.isoformat(),
                "equity_usd": round(cash_equity, 8),
                "event": "FORCED_FINAL_CLOSE",
            }
        )
    equity_values = [float(item["equity_usd"]) for item in equity_curve]
    if maximum_drawdown > max_drawdown_from_equity(equity_values):
        raise RuntimeError("economic_drawdown_accounting_inconsistent")
    metrics = _metrics(trades, equity_values, config)
    return {
        "metrics": metrics,
        "trades": trades,
        "equity_curve": equity_curve,
        "total_fees_usd": round(sum(item["fees_usd"] for item in trades), 8),
        "total_slippage_cost_usd": round(
            sum(item["slippage_cost_usd"] for item in trades), 8
        ),
    }


def evaluate_candidate(
    candidate: dict[str, Any],
    bars: DataFrame,
    config: ExecutionConfig,
) -> dict[str, Any]:
    """Evaluate one persisted candidate independently against its aligned bars."""
    signals = build_forward_signals(candidate)
    replay = simulate_signals(bars, signals, config)
    return {
        "candidate_id": candidate["candidate_id"],
        "scenario": candidate["scenario"],
        "pair": candidate["pair"],
        "timeframe": candidate["timeframe"],
        "feature_profile": candidate["feature_profile"],
        "weighting_policy": candidate["weighting_policy"],
        "selected_parameters": candidate["selected_parameters"],
        "source_forward_hold_precision": candidate["hold_precision"],
        "source_forward_hold_recall": candidate["hold_recall"],
        "selected_hold_threshold": candidate["selected_hold_threshold"],
        "forward_signal_count": len(signals),
        **replay,
    }


def render_economic_markdown(report: dict[str, Any]) -> str:
    """Render the auditable per-model economic results into GitHub Step Summary."""
    lines = [
        "## AliX V4-I Economic Evaluation",
        "",
        f"Source V4-H SHA: {report['source_v4h_commit_sha']}",
        f"V4-I commit SHA: {report['commit_sha']}",
        "",
        "Replay only: persisted V4-H Forward-Validation probabilities; no model fitting or inference.",
        "",
        "| Pair | TF | Profile | Weighting | HOLD P | HOLD R | Profit Factor | Max DD | Net Return | Trades | Gate |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["candidates"]:
        metrics = item["metrics"]
        pf = (
            "INF"
            if metrics["profit_factor_infinite_no_losses"]
            else f"{metrics['profit_factor_net_of_costs']:.4f}"
            if metrics["profit_factor_net_of_costs"] is not None
            else "N/A"
        )
        lines.append(
            f"| {item['pair']} | {item['timeframe']} | {item['feature_profile']} "
            f"| {item['weighting_policy']} | {item['source_forward_hold_precision']:.4f} "
            f"| {item['source_forward_hold_recall']:.4f} | {pf} "
            f"| {metrics['max_drawdown_percent']:.2f}% "
            f"| {metrics['net_return_percent']:.2f}% | {metrics['trade_count']} "
            f"| {'PASS' if metrics['economic_candidate_gate_passed'] else 'NO-GO'} |"
        )
    lines.extend(
        [
            "",
            f"Candidates passing PF > 1.5 and Max DD < 15%: {report['economic_acceptance']['passed_candidate_count']}/"
            f"{report['economic_acceptance']['candidate_count']}.",
            "",
            "Execution: next-bar open; BUY=long, SELL=short, HOLD=close/flat; 1.0x equity, no pyramiding.",
            "Costs: 25 bps fee and 20 bps slippage per side. No funding/borrow, stop-loss, take-profit, or trailing stop.",
            "",
            "Safety: Champion=null, order_permission=false, trading_authorization=NO-GO, private_api_calls=0, real_order_attempts=0.",
            "This is a replay of the same Forward-Validation interval used to select V4-H candidates, not an independent out-of-sample promotion test.",
        ]
    )
    return "\n".join(lines) + "\n"
