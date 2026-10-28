"""Train a fresh CPU LSTM to imitate causal teacher labels; never execute orders."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from services.rl_research.institutional_teacher import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    DEFAULT_ATR_PERIOD,
    DEFAULT_FEE_BPS_PER_SIDE,
    DEFAULT_MIN_ATR_COST_MULTIPLE,
    DEFAULT_SLIPPAGE_BPS_PER_SIDE,
    DEFAULT_STOP_LOSS_FRACTION,
    DEFAULT_TAKE_PROFIT_FRACTION,
    DEFAULT_TRAILING_DISTANCE_FRACTION,
    DEFAULT_TRAILING_TRIGGER_FRACTION,
    _generate_teacher_actions,
    generate_teacher_labels,
)

FloatArray = NDArray[np.float32]
Float64Array = NDArray[np.float64]
IntArray = NDArray[np.int64]
_ACTION_NAMES = {ACTION_HOLD: "HOLD", ACTION_BUY: "BUY", ACTION_SELL: "SELL"}
_REQUIRED_SOURCE_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
_MIN_TEACHER_TRADE_SIGNAL_COUNT = 800


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest_must_be_json_object")
    return cast(dict[str, Any], payload)


def _validate_market_manifest(manifest: dict[str, Any], commit_sha: str) -> None:
    allowed_providers = {
        "Nobitex public UDF history API",
        "Binance Spot public klines API",
    }
    if manifest.get("provider") not in allowed_providers:
        raise ValueError("market_manifest_provider_not_approved")
    expected_safety: dict[str, Any] = {
        "public_api_only": True,
        "credentials_used": False,
        "exchange_calls": 0,
        "real_order_attempts": 0,
        "order_permission": False,
        "Champion": None,
        "economic_acceptance": "NO-GO",
        "trading_authorization": "NO-GO",
        "commit_sha": commit_sha,
    }
    for key, expected in expected_safety.items():
        if manifest.get(key) != expected:
            raise ValueError(f"market_manifest_safety_or_identity_mismatch:{key}")


def _read_market_csv(dataset_path: Path, manifest: dict[str, Any]) -> pd.DataFrame:
    observed_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if observed_hash != manifest.get("csv_sha256"):
        raise ValueError("market_csv_sha256_mismatch")
    frame = pd.read_csv(dataset_path)
    if "date" in frame.columns and "timestamp" not in frame.columns:
        frame = frame.rename(columns={"date": "timestamp"})
    return frame


def _validate_market_frame(frame: pd.DataFrame, manifest: dict[str, Any]) -> None:
    missing = set(_REQUIRED_SOURCE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"missing_market_columns:{','.join(sorted(missing))}")
    if len(frame) != manifest.get("row_count") or frame.empty:
        raise ValueError("market_csv_row_count_mismatch")
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("market_timestamps_invalid")
    frame["timestamp"] = timestamps


def load_public_market_data(
    dataset_path: Path, manifest_path: Path, commit_sha: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate provenance, hashes and closed safety invariants before use."""
    manifest = _read_json_object(manifest_path)
    _validate_market_manifest(manifest, commit_sha)
    frame = _read_market_csv(dataset_path, manifest)
    _validate_market_frame(frame, manifest)
    return frame.loc[:, list(_REQUIRED_SOURCE_COLUMNS)].copy(), manifest


def build_feature_matrix(
    frame: pd.DataFrame, teacher: pd.DataFrame
) -> tuple[FloatArray, list[str]]:
    """Create point-in-time OHLCV features and causal higher-timeframe context."""
    if len(frame) != len(teacher):
        raise ValueError("feature_teacher_row_count_mismatch")
    open_price = frame["open"].astype(np.float64)
    high = frame["high"].astype(np.float64)
    low = frame["low"].astype(np.float64)
    close = frame["close"].astype(np.float64)
    volume = frame["volume"].astype(np.float64)
    typical = (high + low + close) / 3.0
    rolling_volume = volume.rolling(24, min_periods=1).sum().replace(0.0, np.nan)
    vwap = (typical * volume).rolling(24, min_periods=1).sum() / rolling_volume
    log_volume = pd.Series(np.log1p(volume.clip(lower=0.0)), index=frame.index)
    volume_std = log_volume.rolling(24, min_periods=2).std().replace(0.0, np.nan)
    volume_z = (log_volume - log_volume.rolling(24, min_periods=1).mean()) / volume_std

    features = pd.DataFrame(
        {
            "return_1": close.pct_change(1, fill_method=None),
            "return_3": close.pct_change(3, fill_method=None),
            "return_6": close.pct_change(6, fill_method=None),
            "return_24": close.pct_change(24, fill_method=None),
            "candle_body_fraction": (close - open_price) / open_price,
            "candle_range_fraction": (high - low) / close,
            "upper_wick_fraction": (high - np.maximum(open_price, close)) / close,
            "lower_wick_fraction": (np.minimum(open_price, close) - low) / close,
            "volume_z_24": volume_z,
            "vwap_distance_24": close / vwap - 1.0,
            "sma_distance_6": close / close.rolling(6, min_periods=1).mean() - 1.0,
            "sma_distance_24": close / close.rolling(24, min_periods=1).mean() - 1.0,
            "sma_distance_72": close / close.rolling(72, min_periods=1).mean() - 1.0,
            "sma_distance_168": close / close.rolling(168, min_periods=1).mean() - 1.0,
            "htf_bias_4h": teacher["higher_timeframe_bias_4h"].astype(np.float32),
            "htf_bias_1d": teacher["higher_timeframe_bias_1d"].astype(np.float32),
            "bos_reference_distance": (
                close - teacher["bos_reference_high"].astype(np.float64)
            )
            / close,
        },
        index=frame.index,
    )
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    names = [str(column) for column in features.columns]
    values = features.to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("feature_matrix_contains_non_finite_values")
    return values, names


def make_sequences(
    features: FloatArray, actions: IntArray, window: int
) -> tuple[FloatArray, IntArray, IntArray]:
    """Return sliding windows and the original candle index for every target."""
    if isinstance(window, bool) or window < 2:
        raise ValueError("sequence_window_out_of_range")
    if features.ndim != 2 or actions.ndim != 1 or len(features) != len(actions):
        raise ValueError("sequence_inputs_shape_mismatch")
    if len(features) <= window:
        raise ValueError("not_enough_rows_for_sequence_window")
    target_indices = np.arange(window - 1, len(features), dtype=np.int64)
    sequences = np.empty(
        (len(target_indices), window, features.shape[1]), dtype=np.float32
    )
    for sequence_index, target_index in enumerate(target_indices):
        start = int(target_index) - window + 1
        stop = int(target_index) + 1
        sequences[sequence_index] = features[start:stop]
    return sequences, actions[target_indices].astype(np.int64), target_indices


def action_distribution(actions: IntArray) -> dict[str, dict[str, int | float]]:
    result: dict[str, dict[str, int | float]] = {}
    total = int(len(actions))
    for action, name in _ACTION_NAMES.items():
        count = int(np.count_nonzero(actions == action))
        result[name] = {
            "count": count,
            "fraction": float(count / total) if total else 0.0,
        }
    return result


def classification_metrics(truth: IntArray, predictions: IntArray) -> dict[str, Any]:
    if truth.shape != predictions.shape or truth.ndim != 1:
        raise ValueError("classification_metric_shape_mismatch")
    matrix = np.zeros((3, 3), dtype=np.int64)
    for actual in range(3):
        for predicted in range(3):
            matrix[actual, predicted] = int(
                np.count_nonzero((truth == actual) & (predictions == predicted))
            )
    class_metrics: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for action, name in _ACTION_NAMES.items():
        true_positive = int(matrix[action, action])
        false_positive = int(matrix[:, action].sum()) - true_positive
        false_negative = int(matrix[action, :].sum()) - true_positive
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_values.append(f1)
        class_metrics[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    accuracy = float(np.mean(truth == predictions)) if len(truth) else 0.0
    return {
        "sample_count": int(len(truth)),
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
        "class_metrics": class_metrics,
        "confusion_matrix_true_rows_predicted_columns": matrix.tolist(),
    }


def _validate_economic_shapes(actions: IntArray, close_prices: Float64Array) -> None:
    if (
        actions.ndim != 1
        or close_prices.ndim != 1
        or len(actions) != len(close_prices)
        or len(actions) < 2
    ):
        raise ValueError("economic_simulation_shape_mismatch")


def _validate_economic_prices_and_costs(
    close_prices: Float64Array, fee_bps_per_side: float, slippage_bps_per_side: float
) -> None:
    if not np.isfinite(close_prices).all() or np.any(close_prices <= 0):
        raise ValueError("economic_simulation_input_invalid")
    if (
        not np.isfinite([fee_bps_per_side, slippage_bps_per_side]).all()
        or fee_bps_per_side < 0
        or slippage_bps_per_side < 0
        or fee_bps_per_side >= 10_000
        or slippage_bps_per_side >= 10_000
    ):
        raise ValueError("economic_simulation_input_invalid")


def mask_spot_actions(actions: IntArray) -> IntArray:
    """Mask invalid Spot transitions: BUY enters long, SELL only closes that long."""
    if actions.ndim != 1:
        raise ValueError("economic_simulation_action_shape_invalid")

    masked = np.full(len(actions), ACTION_HOLD, dtype=np.int64)
    has_asset = False
    for index, raw_action in enumerate(actions):
        try:
            action = int(raw_action)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("economic_simulation_action_invalid") from None
        if action != raw_action or action not in _ACTION_NAMES:
            raise ValueError("economic_simulation_action_invalid")

        if action == ACTION_BUY and not has_asset:
            masked[index] = ACTION_BUY
            has_asset = True
        elif action == ACTION_SELL and has_asset:
            masked[index] = ACTION_SELL
            has_asset = False
    return masked


def _target_positions(actions: IntArray) -> NDArray[np.int8]:
    """Build Spot-only long/flat positions; negative positions are impossible."""
    effective_actions = mask_spot_actions(actions)
    positions = np.zeros(len(effective_actions), dtype=np.int8)
    position = 0
    for index, action in enumerate(effective_actions):
        if action == ACTION_BUY:
            position = 1
        elif action == ACTION_SELL:
            position = 0
        positions[index] = position
    return positions


def _economic_scorecard(
    equity_curve: Float64Array,
    closed_trade_pnls: Float64Array,
    position_change_count: int,
    open_position_at_end: bool,
    fee_bps_per_side: float,
    slippage_bps_per_side: float,
) -> dict[str, Any]:
    drawdown = equity_curve / np.maximum.accumulate(equity_curve) - 1.0
    winners = int(np.count_nonzero(closed_trade_pnls > 0.0))
    losers = int(np.count_nonzero(closed_trade_pnls < 0.0))
    closed_trade_count = int(len(closed_trade_pnls))
    gross_profit = float(closed_trade_pnls[closed_trade_pnls > 0.0].sum())
    gross_loss = float(-closed_trade_pnls[closed_trade_pnls < 0.0].sum())
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0.0 else None
    return {
        "net_return_fraction": float(equity_curve[-1] - 1.0),
        "max_drawdown_fraction": float(max(0.0, -float(drawdown.min()))),
        "profit_factor": profit_factor,
        "closed_trade_count": closed_trade_count,
        "winning_trades": winners,
        "losing_trades": losers,
        "breakeven_trades": closed_trade_count - winners - losers,
        "win_rate_fraction": (
            float(winners / closed_trade_count) if closed_trade_count else None
        ),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "position_change_count": int(position_change_count),
        "open_position_at_end": open_position_at_end,
        "fee_bps_per_side": float(fee_bps_per_side),
        "slippage_bps_per_side": float(slippage_bps_per_side),
    }


def _validate_exit_prices(
    exit_prices: Float64Array | None, expected_count: int
) -> None:
    if exit_prices is None:
        return
    if exit_prices.ndim != 1 or len(exit_prices) != expected_count:
        raise ValueError("economic_simulation_exit_price_shape_mismatch")
    if np.isinf(exit_prices).any():
        raise ValueError("economic_simulation_exit_price_invalid")
    finite_prices = exit_prices[np.isfinite(exit_prices)]
    if np.any(finite_prices <= 0):
        raise ValueError("economic_simulation_exit_price_invalid")


def _buy_spot_quantity(
    cash: float, price: float, fee_rate: float, slippage_rate: float
) -> float:
    buy_fill = price * (1.0 + slippage_rate)
    return cash / (buy_fill * (1.0 + fee_rate))


def _sell_spot_proceeds(
    quantity: float,
    close_price: float,
    index: int,
    exit_prices: Float64Array | None,
    fee_rate: float,
    slippage_rate: float,
) -> float:
    execution_price = close_price
    if exit_prices is not None and np.isfinite(exit_prices[index]):
        execution_price = float(exit_prices[index])
    sell_fill = execution_price * (1.0 - slippage_rate)
    return quantity * sell_fill * (1.0 - fee_rate)


def simulate_economics(
    actions: IntArray,
    close_prices: Float64Array,
    *,
    exit_prices: Float64Array | None = None,
    fee_bps_per_side: float = DEFAULT_FEE_BPS_PER_SIDE,
    slippage_bps_per_side: float = DEFAULT_SLIPPAGE_BPS_PER_SIDE,
) -> dict[str, Any]:
    """Simulate fully funded Spot long/flat trades offline; never opens a short."""
    _validate_economic_shapes(actions, close_prices)
    _validate_economic_prices_and_costs(
        close_prices, fee_bps_per_side, slippage_bps_per_side
    )
    _validate_exit_prices(exit_prices, len(close_prices))
    effective_actions = mask_spot_actions(actions)
    fee_rate = fee_bps_per_side / 10_000.0
    slippage_rate = slippage_bps_per_side / 10_000.0

    cash = 1.0
    quantity = 0.0
    entry_equity = 0.0
    has_asset = False
    closed_trade_pnls: list[float] = []
    position_change_count = 0
    equity_values = [1.0]

    for index, (action, raw_price) in enumerate(
        zip(effective_actions, close_prices, strict=True)
    ):
        price = float(raw_price)
        if action == ACTION_BUY and not has_asset:
            entry_equity = cash
            quantity = _buy_spot_quantity(cash, price, fee_rate, slippage_rate)
            cash = 0.0
            has_asset = True
            position_change_count += 1
        elif action == ACTION_SELL and has_asset:
            cash = _sell_spot_proceeds(
                quantity,
                price,
                index,
                exit_prices,
                fee_rate,
                slippage_rate,
            )
            closed_trade_pnls.append(float(cash - entry_equity))
            quantity = 0.0
            has_asset = False
            entry_equity = 0.0
            position_change_count += 1

        marked_equity = quantity * price if has_asset else cash
        equity_values.append(float(marked_equity))

    return _economic_scorecard(
        np.asarray(equity_values, dtype=np.float64),
        np.asarray(closed_trade_pnls, dtype=np.float64),
        position_change_count,
        has_asset,
        fee_bps_per_side,
        slippage_bps_per_side,
    )


def _simulate_teacher_segment(
    frame: pd.DataFrame,
    teacher: pd.DataFrame,
    start_index: int,
    end_index: int,
    *,
    include_execution_costs: bool = True,
) -> dict[str, Any]:
    """Backtest a fresh flat position; optionally omit direct fees and slippage."""
    if start_index < 0 or end_index > len(frame) or end_index - start_index < 2:
        raise ValueError("teacher_backtest_segment_bounds_invalid")
    segment = frame.iloc[start_index:end_index].reset_index(drop=True)
    signals = teacher.iloc[start_index:end_index].reset_index(drop=True)
    entry_mode = (
        str(signals["teacher_entry_mode"].iloc[0])
        if "teacher_entry_mode" in signals.columns and not signals.empty
        else "institutional"
    )
    risk_parameters = _teacher_risk_parameters(entry_mode)
    actions, _, exit_prices, _ = _generate_teacher_actions(
        segment,
        signals["teacher_setup_buy"],
        signals["teacher_setup_sell"],
        signals["teacher_atr_fraction"],
        cooldown_candles=2,
        min_atr_cost_multiple=float(risk_parameters["atr_cost_multiple"]),
        take_profit_fraction=float(risk_parameters["take_profit_fraction"]),
        stop_loss_fraction=float(risk_parameters["stop_loss_fraction"]),
        trailing_trigger_fraction=float(risk_parameters["trailing_trigger_fraction"]),
        trailing_distance_fraction=float(risk_parameters["trailing_distance_fraction"]),
        fee_bps_per_side=float(risk_parameters["fee_bps_per_side"]),
        slippage_bps_per_side=float(risk_parameters["slippage_bps_per_side"]),
    )
    close_prices = segment["close"].to_numpy(dtype=np.float64)
    fee_bps = (
        float(risk_parameters["fee_bps_per_side"]) if include_execution_costs else 0.0
    )
    slippage_bps = (
        float(risk_parameters["slippage_bps_per_side"])
        if include_execution_costs
        else 0.0
    )
    return simulate_economics(
        actions.astype(np.int64),
        close_prices,
        exit_prices=exit_prices,
        fee_bps_per_side=fee_bps,
        slippage_bps_per_side=slippage_bps,
    )


def _teacher_backtest_gate(
    validation_economics: dict[str, Any],
    holdout_economics: dict[str, Any],
    teacher_action_distribution: dict[str, Any],
    minimum_profit_factor: float = 1.5,
    minimum_trade_signal_count: int = _MIN_TEACHER_TRADE_SIGNAL_COUNT,
) -> dict[str, Any]:
    def passes(economics: dict[str, Any]) -> bool:
        profit_factor = economics.get("profit_factor")
        return (
            isinstance(profit_factor, (int, float))
            and np.isfinite(profit_factor)
            and float(profit_factor) >= minimum_profit_factor
        )

    validation_passed = passes(validation_economics)
    holdout_passed = passes(holdout_economics)
    buy_count = int(teacher_action_distribution.get("BUY", {}).get("count", 0))
    sell_count = int(teacher_action_distribution.get("SELL", {}).get("count", 0))
    trade_signal_count = buy_count + sell_count
    signal_density_passed = (
        trade_signal_count >= minimum_trade_signal_count
        and buy_count > 0
        and sell_count > 0
    )
    passed = validation_passed and holdout_passed and signal_density_passed
    return {
        "minimum_profit_factor": minimum_profit_factor,
        "validation_passed": validation_passed,
        "holdout_passed": holdout_passed,
        "minimum_trade_signal_count": minimum_trade_signal_count,
        "teacher_trade_signal_count": trade_signal_count,
        "signal_density_passed": signal_density_passed,
        "passed": passed,
        "lstm_training_allowed": passed,
    }


def _predict_actions(
    torch: Any, lstm: Any, classifier: Any, features: FloatArray
) -> IntArray:
    lstm.eval()
    classifier.eval()
    tensor = torch.tensor(features, dtype=torch.float32, device="cpu")
    with torch.no_grad():
        output, _ = lstm(tensor)
        logits = classifier(output[:, -1, :])
        predictions = logits.argmax(dim=1).cpu().numpy()
    return np.asarray(predictions, dtype=np.int64)


def _validate_training_parameters(epochs: int, window: int) -> None:
    if isinstance(epochs, bool) or not 1 <= epochs <= 100:
        raise ValueError("epochs_out_of_range")
    if isinstance(window, bool) or not 8 <= window <= 256:
        raise ValueError("sequence_window_out_of_range")


def _teacher_entry_logic(entry_mode: str) -> dict[str, Any]:
    if entry_mode == "breakout":
        return {
            "strategy": "donchian_20_breakout",
            "buy_condition": "completed candle close exceeds the maximum high of the prior 20 candles",
            "sell_condition": "completed candle close falls below the minimum low of the prior 20 candles; Spot long is closed, never shorted",
            "donchian_lookback_candles": 20,
            "breakout_confirmation": "close beyond prior-candle channel; current candle excluded from channel",
            "signals_use_completed_candles_only": True,
            "positioning": "Spot long/flat; SELL closes an existing long only",
        }
    if entry_mode == "mean_reversion":
        return {
            "strategy": "bollinger_rsi_mean_reversion",
            "buy_condition": "low touches or crosses lower Bollinger band and RSI-14 < 30",
            "sell_condition": "high touches or crosses upper Bollinger band and RSI-14 > 70; Spot long is closed, never shorted",
            "bollinger_period_candles": 20,
            "bollinger_standard_deviations": 2.0,
            "rsi_period_candles": 14,
            "rsi_buy_below": 30.0,
            "rsi_sell_above": 70.0,
            "signals_use_completed_candles_only": True,
            "positioning": "Spot long/flat; SELL closes an existing long only",
        }
    if entry_mode == "trend_pullback":
        return {
            "strategy": "trend_pullback",
            "buy_condition": "completed-candle 4h bullish bias with EMA pullback and reclaim",
            "sell_condition": "completed-candle 4h bearish bias with EMA pullback and reclaim; Spot long is closed, never shorted",
            "signals_use_completed_candles_only": True,
            "positioning": "Spot long/flat; SELL closes an existing long only",
        }
    return {
        "strategy": "institutional_structure_proxy",
        "buy_condition": "completed-candle higher-timeframe bias with causal structure or order-block touch",
        "sell_condition": "completed-candle bearish setup closes an existing Spot long",
        "signals_use_completed_candles_only": True,
        "positioning": "Spot long/flat; SELL closes an existing long only",
    }


def _teacher_risk_parameters(entry_mode: str = "institutional") -> dict[str, Any]:
    if entry_mode == "mean_reversion":
        take_profit_fraction = 0.02
        stop_loss_fraction = 0.01
        trailing_trigger_fraction = 0.01
        # At +1% high-watermark, this distance leaves the stop at cost-adjusted breakeven.
        trailing_distance_fraction = 0.01
    elif entry_mode in {"institutional", "trend_pullback", "breakout"}:
        take_profit_fraction = DEFAULT_TAKE_PROFIT_FRACTION
        stop_loss_fraction = DEFAULT_STOP_LOSS_FRACTION
        trailing_trigger_fraction = DEFAULT_TRAILING_TRIGGER_FRACTION
        trailing_distance_fraction = DEFAULT_TRAILING_DISTANCE_FRACTION
    else:
        raise ValueError("unknown_teacher_entry_mode")
    return {
        "entry_mode": entry_mode,
        "atr_period_candles": DEFAULT_ATR_PERIOD,
        "atr_cost_multiple": DEFAULT_MIN_ATR_COST_MULTIPLE,
        "minimum_atr_fraction": (
            DEFAULT_MIN_ATR_COST_MULTIPLE
            * (DEFAULT_FEE_BPS_PER_SIDE + DEFAULT_SLIPPAGE_BPS_PER_SIDE)
            / 10_000.0
        ),
        "take_profit_fraction": take_profit_fraction,
        "stop_loss_fraction": stop_loss_fraction,
        "trailing_trigger_fraction": trailing_trigger_fraction,
        "trailing_distance_fraction": trailing_distance_fraction,
        "trailing_breakeven_includes_round_trip_costs": True,
        "stop_target_same_candle_policy": "stop_first_conservative",
        "fee_bps_per_side": DEFAULT_FEE_BPS_PER_SIDE,
        "slippage_bps_per_side": DEFAULT_SLIPPAGE_BPS_PER_SIDE,
    }


def _cost_adjusted_risk_diagnostics(
    take_profit_fraction: float = DEFAULT_TAKE_PROFIT_FRACTION,
    stop_loss_fraction: float = DEFAULT_STOP_LOSS_FRACTION,
) -> dict[str, Any]:
    fee_rate = DEFAULT_FEE_BPS_PER_SIDE / 10_000.0
    slippage_rate = DEFAULT_SLIPPAGE_BPS_PER_SIDE / 10_000.0
    entry_cost_factor = (1.0 + slippage_rate) * (1.0 + fee_rate)
    exit_cost_factor = (1.0 - slippage_rate) * (1.0 - fee_rate)
    target_factor = (1.0 + take_profit_fraction) * exit_cost_factor / entry_cost_factor
    stop_factor = (1.0 - stop_loss_fraction) * exit_cost_factor / entry_cost_factor
    net_target_return = target_factor - 1.0
    net_stop_return = stop_factor - 1.0
    net_stop_loss = abs(min(0.0, net_stop_return))
    net_target_gain = max(0.0, net_target_return)
    break_even_win_rate = (
        net_stop_loss / (net_stop_loss + net_target_gain)
        if net_stop_loss + net_target_gain > 0.0
        else None
    )
    return {
        "fee_bps_per_side": DEFAULT_FEE_BPS_PER_SIDE,
        "slippage_bps_per_side": DEFAULT_SLIPPAGE_BPS_PER_SIDE,
        "take_profit_fraction": float(take_profit_fraction),
        "stop_loss_fraction": float(stop_loss_fraction),
        "estimated_net_return_at_tp_without_gaps": net_target_return,
        "estimated_net_return_at_sl_without_gaps": net_stop_return,
        "break_even_win_rate_fraction_without_gaps": break_even_win_rate,
        "assumptions": "fully funded Spot long/flat; both-side fees and slippage; no gaps",
    }


def _is_numeric_below(value: Any, threshold: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and isinstance(threshold, (int, float))
        and value < threshold
    )


def _mode_economic_diagnosis(
    validation: dict[str, Any],
    holdout: dict[str, Any],
    risk_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    diagnostics = risk_diagnostics or _cost_adjusted_risk_diagnostics()
    required_win_rate = diagnostics["break_even_win_rate_fraction_without_gaps"]
    validation_win_rate = validation.get("win_rate_fraction")
    holdout_win_rate = holdout.get("win_rate_fraction")
    validation_win_rate_below_break_even = _is_numeric_below(
        validation_win_rate, required_win_rate
    )
    holdout_win_rate_below_break_even = _is_numeric_below(
        holdout_win_rate, required_win_rate
    )
    validation_pf_below_one = _is_numeric_below(validation.get("profit_factor"), 1.0)
    holdout_pf_below_one = _is_numeric_below(holdout.get("profit_factor"), 1.0)
    diagnosis = (
        "realized win rate below cost-adjusted break-even"
        if validation_win_rate_below_break_even and holdout_win_rate_below_break_even
        else "review per-partition PF, win rate, drawdown, and trade counts"
    )
    return {
        "cost_adjusted_break_even_win_rate_fraction": required_win_rate,
        "validation_win_rate_fraction": validation_win_rate,
        "holdout_win_rate_fraction": holdout_win_rate,
        "validation_win_rate_below_break_even": validation_win_rate_below_break_even,
        "holdout_win_rate_below_break_even": holdout_win_rate_below_break_even,
        "validation_pf_below_one": validation_pf_below_one,
        "holdout_pf_below_one": holdout_pf_below_one,
        "diagnostic": diagnosis,
    }


def _profit_factor_above_one(economics: dict[str, Any]) -> bool:
    value = economics.get("profit_factor")
    return isinstance(value, (int, float)) and np.isfinite(value) and value > 1.0


def _execution_cost_counterfactual_diagnosis(
    validation_with_costs: dict[str, Any],
    holdout_with_costs: dict[str, Any],
    validation_without_direct_costs: dict[str, Any],
    holdout_without_direct_costs: dict[str, Any],
) -> dict[str, Any]:
    validation_before_costs_profitable = _profit_factor_above_one(
        validation_without_direct_costs
    )
    holdout_before_costs_profitable = _profit_factor_above_one(
        holdout_without_direct_costs
    )
    validation_after_costs_profitable = _profit_factor_above_one(validation_with_costs)
    holdout_after_costs_profitable = _profit_factor_above_one(holdout_with_costs)
    validation_cost_flip = (
        validation_before_costs_profitable and not validation_after_costs_profitable
    )
    holdout_cost_flip = (
        holdout_before_costs_profitable and not holdout_after_costs_profitable
    )
    if validation_cost_flip or holdout_cost_flip:
        interpretation = (
            "direct fees and slippage flip at least one partition below PF 1"
        )
    elif not validation_before_costs_profitable and not holdout_before_costs_profitable:
        interpretation = "trade path is unprofitable even before direct execution costs"
    else:
        interpretation = "direct costs contribute, but do not explain both PF failures"
    return {
        "validation_profit_factor_without_direct_costs": validation_without_direct_costs.get(
            "profit_factor"
        ),
        "holdout_profit_factor_without_direct_costs": holdout_without_direct_costs.get(
            "profit_factor"
        ),
        "validation_net_return_without_direct_costs": validation_without_direct_costs.get(
            "net_return_fraction"
        ),
        "holdout_net_return_without_direct_costs": holdout_without_direct_costs.get(
            "net_return_fraction"
        ),
        "validation_pf_flipped_below_one_by_direct_costs": validation_cost_flip,
        "holdout_pf_flipped_below_one_by_direct_costs": holdout_cost_flip,
        "interpretation": interpretation,
        "counterfactual_policy": (
            "same generated actions, exit prices, and cost-aware stop/trailing decisions; "
            "only direct fee and slippage deductions in economic accounting are removed"
        ),
    }


def _write_teacher_dataset(
    frame: pd.DataFrame,
    source_manifest: dict[str, Any],
    output_dir: Path,
    commit_sha: str,
    entry_mode: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    risk_parameters = _teacher_risk_parameters(entry_mode)
    teacher_config: dict[str, Any] = {
        "higher_timeframe": "4h",
        "ema_span": 6,
        "swing_window": 8,
        "volume_window": 24,
        "cooldown_candles": 2,
        "atr_period": int(risk_parameters["atr_period_candles"]),
        "min_atr_cost_multiple": float(risk_parameters["atr_cost_multiple"]),
        "take_profit_fraction": float(risk_parameters["take_profit_fraction"]),
        "stop_loss_fraction": float(risk_parameters["stop_loss_fraction"]),
        "trailing_trigger_fraction": float(
            risk_parameters["trailing_trigger_fraction"]
        ),
        "trailing_distance_fraction": float(
            risk_parameters["trailing_distance_fraction"]
        ),
        "fee_bps_per_side": float(risk_parameters["fee_bps_per_side"]),
        "slippage_bps_per_side": float(risk_parameters["slippage_bps_per_side"]),
    }
    teacher = generate_teacher_labels(frame, entry_mode=entry_mode, **teacher_config)
    teacher_path = output_dir / f"teacher-dataset-{entry_mode}.csv"
    teacher.to_csv(teacher_path, index=False)
    teacher_manifest: dict[str, Any] = {
        "schema_version": "alix.imitation-teacher-dataset.v1",
        "commit_sha": commit_sha,
        "source_provider": source_manifest["provider"],
        "source_csv_sha256": source_manifest["csv_sha256"],
        "teacher_csv_sha256": hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
        "teacher_csv_file": teacher_path.name,
        "selected_for_training": False,
        "row_count": int(len(teacher)),
        "teacher_timeframe_mode": "4h_completed_candles",
        "teacher_entry_mode": entry_mode,
        "minimum_teacher_trade_signal_count": _MIN_TEACHER_TRADE_SIGNAL_COUNT,
        "teacher_risk_management": risk_parameters,
        "teacher_entry_logic": _teacher_entry_logic(entry_mode),
        "teacher_action_distribution": action_distribution(
            teacher["teacher_action"].to_numpy(dtype=np.int64)
        ),
        "teacher_label_source": "causal OHLCV heuristic; not ground-truth institutional order flow",
        "public_api_only": True,
        "credentials_used": False,
        "exchange_calls": 0,
        "real_order_attempts": 0,
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
    }
    manifest_path = output_dir / f"teacher-dataset-{entry_mode}-manifest.json"
    manifest_path.write_text(
        json.dumps(teacher_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return teacher, teacher_manifest


def _write_teacher_datasets(
    frame: pd.DataFrame,
    source_manifest: dict[str, Any],
    output_dir: Path,
    commit_sha: str,
) -> dict[str, tuple[pd.DataFrame, dict[str, Any]]]:
    modes = ("institutional", "trend_pullback", "mean_reversion", "breakout")
    return {
        mode: _write_teacher_dataset(
            frame, source_manifest, output_dir, commit_sha, mode
        )
        for mode in modes
    }


def _publish_selected_teacher_dataset(
    output_dir: Path,
    entry_mode: str,
    candidate_manifest: dict[str, Any],
) -> dict[str, Any]:
    candidate_path = output_dir / f"teacher-dataset-{entry_mode}.csv"
    selected_path = output_dir / "teacher-dataset.csv"
    selected_path.write_bytes(candidate_path.read_bytes())
    selected_manifest = dict(candidate_manifest)
    selected_manifest.update(
        {
            "teacher_csv_sha256": hashlib.sha256(
                selected_path.read_bytes()
            ).hexdigest(),
            "teacher_csv_file": selected_path.name,
            "selected_for_training": True,
        }
    )
    manifest_path = output_dir / "teacher-dataset-manifest.json"
    manifest_path.write_text(
        json.dumps(selected_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return selected_manifest


def _evaluate_teacher_mode(
    frame: pd.DataFrame,
    teacher: pd.DataFrame,
    teacher_manifest: dict[str, Any],
    train_end: int,
    validation_end: int,
) -> dict[str, Any]:
    row_count = len(frame)
    validation_economics = _simulate_teacher_segment(
        frame, teacher, train_end, validation_end
    )
    holdout_economics = _simulate_teacher_segment(
        frame, teacher, validation_end, row_count
    )
    validation_without_direct_costs = _simulate_teacher_segment(
        frame, teacher, train_end, validation_end, include_execution_costs=False
    )
    holdout_without_direct_costs = _simulate_teacher_segment(
        frame, teacher, validation_end, row_count, include_execution_costs=False
    )
    cost_counterfactual = _execution_cost_counterfactual_diagnosis(
        validation_economics,
        holdout_economics,
        validation_without_direct_costs,
        holdout_without_direct_costs,
    )
    entry_mode = str(teacher_manifest["teacher_entry_mode"])
    risk_parameters = _teacher_risk_parameters(entry_mode)
    risk_diagnostics = _cost_adjusted_risk_diagnostics(
        float(risk_parameters["take_profit_fraction"]),
        float(risk_parameters["stop_loss_fraction"]),
    )
    distribution = teacher_manifest["teacher_action_distribution"]
    return {
        "entry_mode": entry_mode,
        "entry_logic": teacher_manifest.get(
            "teacher_entry_logic", _teacher_entry_logic(entry_mode)
        ),
        "parameters": risk_parameters,
        "cost_adjusted_risk_diagnostics": risk_diagnostics,
        "teacher_csv_sha256": teacher_manifest["teacher_csv_sha256"],
        "action_distribution": distribution,
        "validation": {
            "row_start_inclusive": train_end,
            "row_end_exclusive": validation_end,
            "economics": validation_economics,
        },
        "holdout": {
            "row_start_inclusive": validation_end,
            "row_end_exclusive": row_count,
            "economics": holdout_economics,
        },
        "gate": _teacher_backtest_gate(
            validation_economics, holdout_economics, distribution
        ),
        "economic_diagnosis": _mode_economic_diagnosis(
            validation_economics, holdout_economics, risk_diagnostics
        ),
        "execution_cost_counterfactual": {
            "validation_without_direct_costs": validation_without_direct_costs,
            "holdout_without_direct_costs": holdout_without_direct_costs,
            "diagnosis": cost_counterfactual,
        },
    }


def _select_teacher_mode(mode_results: dict[str, dict[str, Any]]) -> str | None:
    passing_modes = [
        mode for mode, result in mode_results.items() if result["gate"]["passed"]
    ]
    if not passing_modes:
        return None
    return max(
        passing_modes,
        key=lambda mode: float(
            mode_results[mode]["validation"]["economics"]["profit_factor"]
        ),
    )


def _validation_profit_factor(mode_result: dict[str, Any]) -> float:
    value = mode_result["validation"]["economics"].get("profit_factor")
    if not isinstance(value, (int, float)) or not np.isfinite(value):
        return float("-inf")
    return float(value)


def _build_teacher_backtest_report(
    commit_sha: str,
    source_manifest: dict[str, Any],
    mode_results: dict[str, dict[str, Any]],
    selected_mode: str | None,
) -> dict[str, Any]:
    diagnostic_mode = selected_mode or max(
        mode_results, key=lambda mode: _validation_profit_factor(mode_results[mode])
    )
    diagnostic = mode_results[diagnostic_mode]
    passing_modes = [
        mode for mode, result in mode_results.items() if result["gate"]["passed"]
    ]
    if selected_mode is not None:
        gate = dict(mode_results[selected_mode]["gate"])
    else:
        gate = {
            "minimum_profit_factor": 1.5,
            "minimum_trade_signal_count": _MIN_TEACHER_TRADE_SIGNAL_COUNT,
            "teacher_trade_signal_count": int(
                diagnostic["gate"]["teacher_trade_signal_count"]
            ),
            "teacher_trade_signal_counts_by_mode": {
                mode: int(result["gate"]["teacher_trade_signal_count"])
                for mode, result in mode_results.items()
            },
            "validation_passed": any(
                result["gate"]["validation_passed"] for result in mode_results.values()
            ),
            "holdout_passed": any(
                result["gate"]["holdout_passed"] for result in mode_results.values()
            ),
            "signal_density_passed": any(
                result["gate"]["signal_density_passed"]
                for result in mode_results.values()
            ),
            "passed": False,
            "lstm_training_allowed": False,
        }
    gate["passing_modes"] = passing_modes
    gate["mode_gate_results"] = {
        mode: result["gate"] for mode, result in mode_results.items()
    }
    return {
        "schema_version": "alix.teacher-backtest-report.v2",
        "commit_sha": commit_sha,
        "source_csv_sha256": source_manifest["csv_sha256"],
        "strategy": (
            "independently evaluated institutional, trend-pullback, "
            "Bollinger-20/2 plus RSI-14 mean-reversion, and Donchian-20 "
            "breakout Spot teachers"
        ),
        "entry_mode": selected_mode,
        "selected_mode": selected_mode,
        "diagnostic_mode_by_validation_pf": diagnostic_mode,
        "selection_policy": (
            "Evaluate every mode independently. Train only a mode that passes "
            "the unchanged validation PF, holdout PF, and 800-signal gates; "
            "if multiple modes pass, maximize validation PF and keep holdout as a gate."
        ),
        "parameters": diagnostic["parameters"],
        "cost_adjusted_risk_diagnostics": diagnostic["cost_adjusted_risk_diagnostics"],
        "action_distribution": diagnostic["action_distribution"],
        "teacher_csv_sha256": diagnostic["teacher_csv_sha256"],
        "modes": mode_results,
        "validation": diagnostic["validation"],
        "holdout": diagnostic["holdout"],
        "gate": gate,
        "safety": {
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "real_order_attempts": 0,
            "exchange_calls": 0,
            "private_api_calls": 0,
            "execution_api_implemented": False,
        },
    }


def _write_teacher_backtest_report(output_dir: Path, report: dict[str, Any]) -> None:
    report_path = output_dir / "teacher-backtest-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"teacher_backtest": report}, indent=2, sort_keys=True, allow_nan=False
        )
    )


def _validate_split_boundaries(
    row_count: int, window: int, train_end: int, validation_end: int
) -> None:
    if (
        train_end <= window
        or validation_end <= train_end
        or row_count <= validation_end
    ):
        raise ValueError("insufficient_rows_for_chronological_splits")


def _validate_split_masks(
    train_mask: NDArray[np.bool_],
    validation_mask: NDArray[np.bool_],
    holdout_mask: NDArray[np.bool_],
) -> None:
    if not train_mask.any() or not validation_mask.any() or not holdout_mask.any():
        raise ValueError("chronological_sequence_split_empty")


def _compute_class_weights(train_actions: IntArray) -> Float64Array:
    """Set stronger BUY/SELL weights using chronological training labels only."""
    if train_actions.ndim != 1 or len(train_actions) == 0:
        raise ValueError("teacher_train_actions_invalid")
    if not np.isin(train_actions, list(_ACTION_NAMES)).all():
        raise ValueError("teacher_train_actions_invalid")
    class_counts = np.bincount(train_actions, minlength=3).astype(np.float64)
    if np.any(class_counts == 0):
        raise ValueError("teacher_train_split_missing_action_class")

    class_weights = np.ones(3, dtype=np.float64)
    for action in (ACTION_BUY, ACTION_SELL):
        class_weights[action] = min(
            max(class_counts[ACTION_HOLD] / class_counts[action], 1.0),
            25.0,
        )
    return class_weights


def _weighted_cross_entropy_loss(
    torch: Any, logits: Any, targets: Any, class_weights: Any
) -> Any:
    """Apply fixed train-derived weights before averaging the per-sample loss."""
    per_sample_loss = torch.nn.functional.cross_entropy(
        logits, targets, reduction="none"
    )
    sample_weights = class_weights[targets]
    return (per_sample_loss * sample_weights).mean()


def _prepare_training_data(
    frame: pd.DataFrame, teacher: pd.DataFrame, window: int
) -> dict[str, Any]:
    actions = teacher["teacher_action"].to_numpy(dtype=np.int64)
    features, feature_names = build_feature_matrix(frame, teacher)
    row_count = len(frame)
    train_end = int(row_count * 0.70)
    validation_end = int(row_count * 0.85)
    _validate_split_boundaries(row_count, window, train_end, validation_end)

    scaler_mean = features[:train_end].mean(axis=0, dtype=np.float64)
    scaler_scale = features[:train_end].std(axis=0, dtype=np.float64)
    scaler_scale = np.where(scaler_scale > 1e-12, scaler_scale, 1.0)
    normalized = ((features - scaler_mean) / scaler_scale).astype(np.float32)
    sequences, sequence_actions, target_indices = make_sequences(
        normalized, actions, window
    )
    train_mask = target_indices < train_end
    validation_mask = (target_indices >= train_end) & (target_indices < validation_end)
    holdout_mask = target_indices >= validation_end
    _validate_split_masks(train_mask, validation_mask, holdout_mask)
    train_actions = sequence_actions[train_mask]
    class_weights = _compute_class_weights(train_actions)
    return {
        "features": features,
        "feature_names": feature_names,
        "actions": actions,
        "row_count": row_count,
        "train_end": train_end,
        "validation_end": validation_end,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "sequences": sequences,
        "sequence_actions": sequence_actions,
        "target_indices": target_indices,
        "train_mask": train_mask,
        "validation_mask": validation_mask,
        "holdout_mask": holdout_mask,
        "train_actions": train_actions,
        "class_weights": class_weights,
    }


def _create_training_state(torch: Any, prepared: dict[str, Any]) -> dict[str, Any]:
    torch.manual_seed(20261027)
    torch.set_num_threads(1)
    sequences = prepared["sequences"]
    train_mask = prepared["train_mask"]
    validation_mask = prepared["validation_mask"]
    holdout_mask = prepared["holdout_mask"]
    train_features = torch.tensor(sequences[train_mask], dtype=torch.float32)
    train_targets = torch.tensor(prepared["train_actions"], dtype=torch.long)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_features, train_targets),
        batch_size=128,
        shuffle=True,
        drop_last=False,
    )
    validation_features = sequences[validation_mask]
    validation_targets = prepared["sequence_actions"][validation_mask]
    holdout_features = sequences[holdout_mask]
    holdout_targets = prepared["sequence_actions"][holdout_mask]
    feature_count = int(prepared["features"].shape[1])
    hidden_size = 64
    lstm = torch.nn.LSTM(
        input_size=feature_count,
        hidden_size=hidden_size,
        num_layers=2,
        batch_first=True,
        dropout=0.15,
    )
    classifier = torch.nn.Linear(hidden_size, 3)
    optimizer = torch.optim.Adam(
        list(lstm.parameters()) + list(classifier.parameters()), lr=0.001
    )
    loss_weights = torch.tensor(prepared["class_weights"], dtype=torch.float32)
    return {
        "lstm": lstm,
        "classifier": classifier,
        "optimizer": optimizer,
        "loss_weights": loss_weights,
        "train_loader": train_loader,
        "validation_features": validation_features,
        "validation_targets": validation_targets,
        "holdout_features": holdout_features,
        "holdout_targets": holdout_targets,
        "hidden_size": hidden_size,
        "feature_count": feature_count,
    }


def _snapshot_model_state(model: Any) -> dict[str, Any]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _fit_lstm(
    torch: Any,
    state: dict[str, Any],
    epochs: int,
) -> dict[str, Any]:
    lstm = state["lstm"]
    classifier = state["classifier"]
    optimizer = state["optimizer"]
    loss_weights = state["loss_weights"]
    train_loader = state["train_loader"]
    validation_features = state["validation_features"]
    validation_targets = state["validation_targets"]
    best_validation_macro_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    best_lstm_state: dict[str, Any] = {}
    best_classifier_state: dict[str, Any] = {}
    completed_epochs = 0

    for epoch in range(epochs):
        lstm.train()
        classifier.train()
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs, _ = lstm(batch_features)
            logits = classifier(outputs[:, -1, :])
            loss = _weighted_cross_entropy_loss(
                torch, logits, batch_targets, loss_weights
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(lstm.parameters()) + list(classifier.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

        completed_epochs = epoch + 1
        predictions = _predict_actions(torch, lstm, classifier, validation_features)
        metrics = classification_metrics(validation_targets, predictions)
        score = float(metrics["macro_f1"])
        if score > best_validation_macro_f1 + 1e-9:
            best_validation_macro_f1 = score
            best_epoch = completed_epochs
            best_lstm_state = _snapshot_model_state(lstm)
            best_classifier_state = _snapshot_model_state(classifier)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 5:
                break

    if not best_lstm_state or not best_classifier_state:
        raise RuntimeError("no_validation_checkpoint_was_selected")
    lstm.load_state_dict(best_lstm_state)
    classifier.load_state_dict(best_classifier_state)
    state["best_validation_macro_f1"] = best_validation_macro_f1
    state["best_epoch"] = best_epoch
    state["completed_epochs"] = completed_epochs
    return state


def _evaluate_training_state(
    torch: Any,
    frame: pd.DataFrame,
    prepared: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    validation_targets = state["validation_targets"]
    holdout_targets = state["holdout_targets"]
    validation_predictions = _predict_actions(
        torch, state["lstm"], state["classifier"], state["validation_features"]
    )
    holdout_predictions = _predict_actions(
        torch, state["lstm"], state["classifier"], state["holdout_features"]
    )
    validation_effective_actions = mask_spot_actions(validation_predictions)
    holdout_effective_actions = mask_spot_actions(holdout_predictions)
    close_prices = frame["close"].to_numpy(dtype=np.float64)
    target_indices = prepared["target_indices"]
    validation_close = close_prices[target_indices[prepared["validation_mask"]]]
    holdout_close = close_prices[target_indices[prepared["holdout_mask"]]]
    return {
        "validation_predictions": validation_predictions,
        "holdout_predictions": holdout_predictions,
        "validation_classification": classification_metrics(
            validation_targets, validation_predictions
        ),
        "holdout_classification": classification_metrics(
            holdout_targets, holdout_predictions
        ),
        "validation_teacher_economics": simulate_economics(
            validation_targets, validation_close
        ),
        "validation_model_economics": simulate_economics(
            validation_predictions, validation_close
        ),
        "holdout_teacher_economics": simulate_economics(holdout_targets, holdout_close),
        "holdout_model_economics": simulate_economics(
            holdout_predictions, holdout_close
        ),
        "validation_model_actions": action_distribution(validation_effective_actions),
        "validation_model_raw_actions": action_distribution(validation_predictions),
        "holdout_teacher_actions": action_distribution(holdout_targets),
        "holdout_model_actions": action_distribution(holdout_effective_actions),
        "holdout_model_raw_actions": action_distribution(holdout_predictions),
    }


def _save_model_artifacts(
    torch: Any,
    output_dir: Path,
    commit_sha: str,
    window: int,
    prepared: dict[str, Any],
    state: dict[str, Any],
) -> str:
    model_path = output_dir / "lstm-behavioral-cloning-state.pt"
    torch.save(
        {
            "architecture": "two-layer LSTM plus linear 3-class head",
            "lstm_state_dict": state["lstm"].state_dict(),
            "classifier_state_dict": state["classifier"].state_dict(),
            "input_features": prepared["feature_names"],
            "window": window,
            "hidden_size": state["hidden_size"],
            "action_map": _ACTION_NAMES,
            "initialized_from_scratch": True,
            "training_commit_sha": commit_sha,
        },
        model_path,
    )
    scaler_document = {
        "feature_names": prepared["feature_names"],
        "mean": [float(value) for value in prepared["scaler_mean"]],
        "scale": [float(value) for value in prepared["scaler_scale"]],
        "fit_rows": int(prepared["train_end"]),
        "fit_policy": "training_split_only",
    }
    (output_dir / "feature-scaler.json").write_text(
        json.dumps(scaler_document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return model_path.name


def _build_training_report(
    commit_sha: str,
    source_manifest: dict[str, Any],
    teacher_manifest: dict[str, Any],
    teacher: pd.DataFrame,
    prepared: dict[str, Any],
    state: dict[str, Any],
    evaluation: dict[str, Any],
    model_artifact: str,
    teacher_backtest: dict[str, Any],
) -> dict[str, Any]:
    row_count = int(prepared["row_count"])
    train_end = int(prepared["train_end"])
    validation_end = int(prepared["validation_end"])
    holdout_distribution = evaluation["holdout_model_actions"]
    return {
        "schema_version": "alix.imitation-learning-report.v1",
        "commit_sha": commit_sha,
        "dataset": {
            "provider": source_manifest["provider"],
            "source_csv_sha256": source_manifest["csv_sha256"],
            "teacher_csv_sha256": teacher_manifest["teacher_csv_sha256"],
            "row_count": row_count,
            "train_rows": train_end,
            "validation_rows": validation_end - train_end,
            "holdout_rows": row_count - validation_end,
            "train_end_timestamp": str(teacher.iloc[train_end - 1]["timestamp"]),
            "validation_end_timestamp": str(
                teacher.iloc[validation_end - 1]["timestamp"]
            ),
            "holdout_start_timestamp": str(teacher.iloc[validation_end]["timestamp"]),
        },
        "teacher": {
            "kind": "causal_ohlcv_heuristic",
            "higher_timeframe_mode": "4h_completed_candles",
            "entry_mode": teacher_manifest["teacher_entry_mode"],
            "minimum_trade_signal_count": teacher_manifest[
                "minimum_teacher_trade_signal_count"
            ],
            "ground_truth_claim": False,
            "action_distribution": action_distribution(prepared["actions"]),
        },
        "teacher_backtest": teacher_backtest,
        "model": {
            "architecture": "two-layer CPU LSTM plus linear head",
            "initialized_from_scratch": True,
            "window_candles": int(prepared["sequences"].shape[1]),
            "input_feature_count": int(state["feature_count"]),
            "input_features": prepared["feature_names"],
            "requested_epochs": int(state["requested_epochs"]),
            "completed_epochs": int(state["completed_epochs"]),
            "best_validation_epoch": int(state["best_epoch"]),
            "best_validation_macro_f1": float(state["best_validation_macro_f1"]),
            "class_weighting": {
                "fit_partition": "chronological_train_only",
                "loss_reduction": "mean_of_per_sample_weighted_cross_entropy",
                "train_action_distribution": action_distribution(
                    prepared["train_actions"]
                ),
                "weights_by_action": {
                    name: float(prepared["class_weights"][action])
                    for action, name in _ACTION_NAMES.items()
                },
                "applied_to_validation": False,
                "applied_to_holdout": False,
            },
            "state_artifact": model_artifact,
        },
        "validation": {
            "classification": evaluation["validation_classification"],
            "teacher_economics": evaluation["validation_teacher_economics"],
            "model_economics": evaluation["validation_model_economics"],
            "model_action_distribution": evaluation["validation_model_actions"],
            "raw_model_action_distribution": evaluation["validation_model_raw_actions"],
        },
        "holdout": {
            "classification_vs_teacher": evaluation["holdout_classification"],
            "teacher_economics": evaluation["holdout_teacher_economics"],
            "model_economics": evaluation["holdout_model_economics"],
            "teacher_action_distribution": evaluation["holdout_teacher_actions"],
            "model_action_distribution": holdout_distribution,
            "raw_model_action_distribution": evaluation["holdout_model_raw_actions"],
            "hold_collapse": holdout_distribution["HOLD"]["fraction"] >= 0.95,
        },
        "safety": {
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "real_order_attempts": 0,
            "exchange_calls": 0,
            "private_api_calls": 0,
            "execution_api_implemented": False,
        },
        "decision": "RESEARCH_ONLY_NOT_PROMOTED",
    }


def run_training(
    dataset_path: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    commit_sha: str,
    epochs: int = 20,
    window: int = 48,
) -> dict[str, Any]:
    _validate_training_parameters(epochs, window)
    frame, source_manifest = load_public_market_data(
        dataset_path, manifest_path, commit_sha
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher_candidates = _write_teacher_datasets(
        frame, source_manifest, output_dir, commit_sha
    )

    row_count = len(frame)
    train_end = int(row_count * 0.70)
    validation_end = int(row_count * 0.85)
    mode_backtests = {
        mode: _evaluate_teacher_mode(
            frame, teacher, manifest, train_end, validation_end
        )
        for mode, (teacher, manifest) in teacher_candidates.items()
    }
    selected_mode = _select_teacher_mode(mode_backtests)
    teacher_backtest = _build_teacher_backtest_report(
        commit_sha, source_manifest, mode_backtests, selected_mode
    )
    _write_teacher_backtest_report(output_dir, teacher_backtest)
    if selected_mode is None:
        raise RuntimeError(
            "no_teacher_mode_passed_validation_holdout_pf_and_signal_density_gates_lstm_training_blocked"
        )

    teacher, candidate_manifest = teacher_candidates[selected_mode]
    teacher_manifest = _publish_selected_teacher_dataset(
        output_dir, selected_mode, candidate_manifest
    )

    # Only prepare model sequences and instantiate/train the LSTM after the
    # independent validation and untouched holdout both clear the economic gate.
    prepared = _prepare_training_data(frame, teacher, window)
    torch: Any = importlib.import_module("torch")
    state = _create_training_state(torch, prepared)
    state["requested_epochs"] = epochs
    state = _fit_lstm(torch, state, epochs)
    evaluation = _evaluate_training_state(torch, frame, prepared, state)
    model_artifact = _save_model_artifacts(
        torch, output_dir, commit_sha, window, prepared, state
    )
    report = _build_training_report(
        commit_sha,
        source_manifest,
        teacher_manifest,
        teacher,
        prepared,
        state,
        evaluation,
        model_artifact,
        teacher_backtest,
    )
    report_path = output_dir / "imitation-learning-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "commit_sha": commit_sha,
                "teacher_dataset_rows": int(prepared["row_count"]),
                "teacher_action_distribution": teacher_manifest[
                    "teacher_action_distribution"
                ],
                "teacher_backtest_gate": teacher_backtest["gate"],
                "validation": report["validation"],
                "holdout": report["holdout"],
                "safety": report["safety"],
                "decision": report["decision"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a fresh LSTM by offline behavioral cloning."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--window", type=int, default=48)
    args = parser.parse_args()
    run_training(
        args.dataset,
        args.manifest,
        args.output_dir,
        commit_sha=str(args.commit_sha),
        epochs=int(args.epochs),
        window=int(args.window),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
