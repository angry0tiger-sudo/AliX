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
    DEFAULT_BREAKOUT_DONCHIAN_LOOKBACK,
    DEFAULT_FEE_BPS_PER_SIDE,
    DEFAULT_MEAN_REVERSION_BOLLINGER_PERIOD,
    DEFAULT_MEAN_REVERSION_BOLLINGER_STDDEV,
    DEFAULT_MEAN_REVERSION_RSI_BUY_BELOW,
    DEFAULT_MEAN_REVERSION_RSI_PERIOD,
    DEFAULT_MEAN_REVERSION_RSI_SELL_ABOVE,
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
_MAX_TEACHER_DRAWDOWN_FRACTION = 0.15
_MAX_OPTIMIZATION_DRAWDOWN_FRACTION = 0.20
_DEFAULT_OPTUNA_TRIALS = 60
_DEFAULT_OPTUNA_SEED = 20261028
_WFV_FOLDS = 5
_WFV_PURGE_CANDLES = 72
_WFV_EMBARGO_CANDLES = 48


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
    risk_parameters: dict[str, Any] | None = None,
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
    active_risk_parameters = risk_parameters or _teacher_risk_parameters(entry_mode)
    actions, _, exit_prices, _ = _generate_teacher_actions(
        segment,
        signals["teacher_setup_buy"],
        signals["teacher_setup_sell"],
        signals["teacher_atr_fraction"],
        cooldown_candles=2,
        min_atr_cost_multiple=float(active_risk_parameters["atr_cost_multiple"]),
        take_profit_fraction=float(active_risk_parameters["take_profit_fraction"]),
        stop_loss_fraction=float(active_risk_parameters["stop_loss_fraction"]),
        trailing_trigger_fraction=float(
            active_risk_parameters["trailing_trigger_fraction"]
        ),
        trailing_distance_fraction=float(
            active_risk_parameters["trailing_distance_fraction"]
        ),
        fee_bps_per_side=float(active_risk_parameters["fee_bps_per_side"]),
        slippage_bps_per_side=float(active_risk_parameters["slippage_bps_per_side"]),
    )
    close_prices = segment["close"].to_numpy(dtype=np.float64)
    fee_bps = (
        float(active_risk_parameters["fee_bps_per_side"])
        if include_execution_costs
        else 0.0
    )
    slippage_bps = (
        float(active_risk_parameters["slippage_bps_per_side"])
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


def _validate_wfv_configuration(
    row_count: int, n_splits: int, purge_candles: int, embargo_candles: int
) -> None:
    if isinstance(row_count, bool) or row_count < 1_000:
        raise ValueError("purged_walk_forward_configuration_invalid")
    if isinstance(n_splits, bool) or n_splits < 2:
        raise ValueError("purged_walk_forward_configuration_invalid")
    if isinstance(purge_candles, bool) or purge_candles < 1:
        raise ValueError("purged_walk_forward_configuration_invalid")
    if isinstance(embargo_candles, bool) or embargo_candles < 1:
        raise ValueError("purged_walk_forward_configuration_invalid")


def _wfv_time_boundaries(row_count: int) -> tuple[int, int, int, int]:
    initial_train_end = int(row_count * 0.40)
    cv_region_end = int(row_count * 0.65)
    validation_start = int(row_count * 0.70)
    holdout_start = int(row_count * 0.85)
    if not (initial_train_end < cv_region_end < validation_start < holdout_start):
        raise ValueError("purged_walk_forward_boundaries_invalid")
    return initial_train_end, cv_region_end, validation_start, holdout_start


def _wfv_test_width(
    initial_train_end: int, cv_region_end: int, n_splits: int, embargo_candles: int
) -> int:
    available = cv_region_end - initial_train_end - embargo_candles * (n_splits - 1)
    test_width = available // n_splits
    if test_width < 24:
        raise ValueError("purged_walk_forward_test_window_too_short")
    return test_width


def _build_purged_walk_forward_fold(
    fold_number: int,
    test_start: int,
    test_width: int,
    cv_region_end: int,
    purge_candles: int,
    embargo_candles: int,
) -> dict[str, int]:
    train_end = test_start - purge_candles
    test_end = test_start + test_width
    if train_end <= 0 or test_end > cv_region_end:
        raise ValueError("purged_walk_forward_fold_out_of_bounds")
    return {
        "fold": fold_number,
        "train_start_inclusive": 0,
        "train_end_exclusive": train_end,
        "purge_start_inclusive": train_end,
        "purge_end_exclusive": test_start,
        "test_start_inclusive": test_start,
        "test_end_exclusive": test_end,
        "embargo_end_exclusive": min(cv_region_end, test_end + embargo_candles),
    }


def _validate_wfv_fold_boundaries(
    folds: list[dict[str, int]], validation_start: int, holdout_start: int
) -> None:
    last_test_end = max(fold["test_end_exclusive"] for fold in folds)
    if last_test_end > validation_start:
        raise ValueError("purged_walk_forward_overlaps_final_validation")
    if last_test_end >= holdout_start:
        raise ValueError("purged_walk_forward_touches_final_holdout")


def _purged_walk_forward_splits(
    row_count: int,
    *,
    n_splits: int = _WFV_FOLDS,
    purge_candles: int = _WFV_PURGE_CANDLES,
    embargo_candles: int = _WFV_EMBARGO_CANDLES,
) -> list[dict[str, int]]:
    """Build expanding-window folds entirely before validation and final holdout."""
    _validate_wfv_configuration(row_count, n_splits, purge_candles, embargo_candles)
    initial_train_end, cv_region_end, validation_start, holdout_start = (
        _wfv_time_boundaries(row_count)
    )
    test_width = _wfv_test_width(
        initial_train_end, cv_region_end, n_splits, embargo_candles
    )

    folds: list[dict[str, int]] = []
    test_start = initial_train_end
    for fold_number in range(n_splits):
        fold = _build_purged_walk_forward_fold(
            fold_number,
            test_start,
            test_width,
            cv_region_end,
            purge_candles,
            embargo_candles,
        )
        folds.append(fold)
        test_start = fold["test_end_exclusive"] + embargo_candles

    _validate_wfv_fold_boundaries(folds, validation_start, holdout_start)
    return folds


def _suggest_teacher_parameters(trial: Any) -> dict[str, Any]:
    entry_mode = str(
        trial.suggest_categorical(
            "entry_mode", ["mean_reversion", "breakout", "institutional"]
        )
    )
    parameters: dict[str, Any] = {
        "entry_mode": entry_mode,
        "take_profit_fraction": trial.suggest_float(
            "take_profit_fraction", 0.005, 0.05, step=0.0025
        ),
        "stop_loss_fraction": trial.suggest_float(
            "stop_loss_fraction", 0.005, 0.05, step=0.0025
        ),
        "atr_cost_multiple": trial.suggest_float(
            "atr_cost_multiple", 1.0, 3.0, step=0.25
        ),
    }
    if entry_mode == "mean_reversion":
        rsi_buy_below = trial.suggest_int("rsi_buy_below", 20, 40)
        parameters.update(
            {
                "bollinger_period_candles": DEFAULT_MEAN_REVERSION_BOLLINGER_PERIOD,
                "bollinger_stddev": trial.suggest_float(
                    "bollinger_stddev", 1.0, 3.0, step=0.25
                ),
                "rsi_period_candles": DEFAULT_MEAN_REVERSION_RSI_PERIOD,
                "rsi_buy_below": rsi_buy_below,
                "rsi_sell_above": 100 - rsi_buy_below,
            }
        )
    elif entry_mode == "breakout":
        parameters["donchian_lookback_candles"] = trial.suggest_int(
            "donchian_lookback_candles", 10, 50
        )
    return _teacher_parameter_config(entry_mode, parameters)


def _teacher_trial_score(
    fold_results: list[dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    if not fold_results:
        raise ValueError("optuna_trial_has_no_walk_forward_folds")

    profit_factors: list[float] = []
    drawdowns: list[float] = []
    signal_count = 0
    closed_trades = 0
    undertraded_folds = 0
    for fold in fold_results:
        economics = fold["economics"]
        raw_pf = economics.get("profit_factor")
        if isinstance(raw_pf, (int, float)) and np.isfinite(raw_pf):
            profit_factors.append(min(float(raw_pf), 5.0))
        elif (
            int(economics.get("closed_trade_count", 0)) > 0
            and float(economics.get("gross_profit", 0.0)) > 0.0
        ):
            profit_factors.append(5.0)
        else:
            profit_factors.append(0.0)
        drawdown = economics.get("max_drawdown_fraction")
        drawdowns.append(
            float(drawdown)
            if isinstance(drawdown, (int, float)) and np.isfinite(drawdown)
            else 1.0
        )
        signal_count += int(economics.get("position_change_count", 0))
        trades = int(economics.get("closed_trade_count", 0))
        closed_trades += trades
        undertraded_folds += int(trades < 5)

    median_pf = float(np.median(profit_factors))
    pf_dispersion = float(np.std(profit_factors))
    worst_drawdown = max(drawdowns)
    penalty_over_15 = 5.0 * max(0.0, worst_drawdown - _MAX_TEACHER_DRAWDOWN_FRACTION)
    penalty_over_20 = 10.0 * max(
        0.0, worst_drawdown - _MAX_OPTIMIZATION_DRAWDOWN_FRACTION
    )
    low_activity_penalty = 0.05 * undertraded_folds + 0.005 * max(0, 100 - signal_count)
    score = (
        median_pf
        - 0.15 * pf_dispersion
        - penalty_over_15
        - penalty_over_20
        - low_activity_penalty
    )
    diagnostics = {
        "median_capped_profit_factor": median_pf,
        "profit_factor_standard_deviation": pf_dispersion,
        "worst_fold_max_drawdown_fraction": float(worst_drawdown),
        "maximum_drawdown_penalty_threshold_fraction": (
            _MAX_OPTIMIZATION_DRAWDOWN_FRACTION
        ),
        "project_hard_drawdown_gate_fraction": _MAX_TEACHER_DRAWDOWN_FRACTION,
        "drawdown_penalty_above_15_percent": float(penalty_over_15),
        "additional_drawdown_penalty_above_20_percent": float(penalty_over_20),
        "undertraded_fold_count": int(undertraded_folds),
        "out_of_sample_signal_count": int(signal_count),
        "out_of_sample_closed_trade_count": int(closed_trades),
        "objective_value": float(score),
    }
    if not np.isfinite(score):
        raise ValueError("optuna_objective_score_non_finite")
    return float(score), diagnostics


def _validate_optuna_trial_count(trial_count: int) -> None:
    if isinstance(trial_count, bool) or not 10 <= trial_count <= 250:
        raise ValueError("optuna_trial_count_out_of_range")


def _evaluate_optuna_trial(
    trial: Any,
    optimization_frame: pd.DataFrame,
    folds: list[dict[str, int]],
) -> float:
    parameters = _suggest_teacher_parameters(trial)
    entry_mode = str(parameters["entry_mode"])
    risk_parameters = _teacher_risk_parameters(entry_mode, parameters)
    teacher = generate_teacher_labels(
        optimization_frame,
        entry_mode=entry_mode,
        atr_period=int(risk_parameters["atr_period_candles"]),
        min_atr_cost_multiple=float(risk_parameters["atr_cost_multiple"]),
        take_profit_fraction=float(risk_parameters["take_profit_fraction"]),
        stop_loss_fraction=float(risk_parameters["stop_loss_fraction"]),
        trailing_trigger_fraction=float(
            risk_parameters["trailing_trigger_fraction"]
        ),
        trailing_distance_fraction=float(
            risk_parameters["trailing_distance_fraction"]
        ),
        fee_bps_per_side=float(risk_parameters["fee_bps_per_side"]),
        slippage_bps_per_side=float(risk_parameters["slippage_bps_per_side"]),
        bollinger_period=int(parameters["bollinger_period_candles"]),
        bollinger_stddev=float(parameters["bollinger_stddev"]),
        rsi_period=int(parameters["rsi_period_candles"]),
        rsi_buy_below=float(parameters["rsi_buy_below"]),
        rsi_sell_above=float(parameters["rsi_sell_above"]),
        breakout_donchian_lookback=int(parameters["donchian_lookback_candles"]),
    )
    fold_results: list[dict[str, Any]] = []
    for fold in folds:
        economics = _simulate_teacher_segment(
            optimization_frame,
            teacher,
            fold["test_start_inclusive"],
            fold["test_end_exclusive"],
            risk_parameters=risk_parameters,
        )
        fold_results.append({**fold, "economics": economics})
    score, diagnostics = _teacher_trial_score(fold_results)
    trial.set_user_attr("normalized_parameters", parameters)
    trial.set_user_attr("fold_results", fold_results)
    trial.set_user_attr("score_diagnostics", diagnostics)
    return score


def _completed_optuna_trials(study: Any) -> list[Any]:
    completed = [
        trial
        for trial in study.trials
        if trial.value is not None and trial.state.name == "COMPLETE"
    ]
    return sorted(
        completed,
        key=lambda trial: (
            float(trial.value) if trial.value is not None else float("-inf")
        ),
        reverse=True,
    )


def _top_optuna_trials(completed: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "trial_number": int(trial.number),
            "objective_value": (
                float(trial.value) if trial.value is not None else float("-inf")
            ),
            "parameters": dict(trial.user_attrs.get("normalized_parameters", {})),
            "score_diagnostics": dict(trial.user_attrs.get("score_diagnostics", {})),
        }
        for trial in completed[:10]
    ]


def _build_optuna_report(
    study: Any,
    row_count: int,
    trial_count: int,
    seed: int,
    folds: list[dict[str, int]],
) -> dict[str, Any]:
    best = study.best_trial
    if best.value is None:
        raise RuntimeError("optuna_best_trial_has_no_objective_value")
    completed = _completed_optuna_trials(study)
    validation_start = int(row_count * 0.70)
    validation_end = int(row_count * 0.85)
    return {
        "schema_version": "alix.teacher-optimization-report.v1",
        "method": "Optuna TPE with purged walk-forward validation",
        "objective": (
            "maximize median capped out-of-sample fold Profit Factor, "
            "penalize PF dispersion, low trading activity, and drawdowns above "
            "the project 15% hard gate and the requested 20% penalty threshold"
        ),
        "sampler": "TPESampler",
        "seed": int(seed),
        "requested_trials": int(trial_count),
        "completed_trials": int(len(completed)),
        "dataset_row_count": int(row_count),
        "search_space": {
            "entry_mode": ["mean_reversion", "breakout", "institutional"],
            "rsi_buy_below_integer": [20, 40],
            "rsi_sell_above": "100 - rsi_buy_below",
            "bollinger_stddev": [1.0, 3.0],
            "bollinger_stddev_step": 0.25,
            "donchian_lookback_candles_integer": [10, 50],
            "take_profit_fraction": [0.005, 0.05],
            "stop_loss_fraction": [0.005, 0.05],
            "risk_fraction_step": 0.0025,
            "atr_cost_multiple": [1.0, 3.0],
            "atr_cost_multiple_step": 0.25,
        },
        "best_trial_number": int(best.number),
        "best_objective_value": float(best.value),
        "best_parameters": dict(best.user_attrs["normalized_parameters"]),
        "best_score_diagnostics": dict(best.user_attrs["score_diagnostics"]),
        "best_fold_results": list(best.user_attrs["fold_results"]),
        "top_trials": _top_optuna_trials(completed),
        "validation_start_inclusive": validation_start,
        "validation_end_exclusive": validation_end,
        "final_holdout_start_inclusive": validation_end,
        "holdout_never_used_in_optimization": True,
        "validation_never_used_in_optimization": True,
        "optimization_test_blocks_end_before_validation": (
            max(fold["test_end_exclusive"] for fold in folds) <= validation_start
        ),
        "purge_candles": _WFV_PURGE_CANDLES,
        "embargo_candles": _WFV_EMBARGO_CANDLES,
        "folds": [dict(fold) for fold in folds],
    }


def _optimize_teacher_parameters(
    frame: pd.DataFrame,
    *,
    trial_count: int = _DEFAULT_OPTUNA_TRIALS,
    seed: int = _DEFAULT_OPTUNA_SEED,
) -> dict[str, Any]:
    """Optimize only purged walk-forward blocks ending before validation."""
    _validate_optuna_trial_count(trial_count)
    optuna: Any = importlib.import_module("optuna")
    row_count = len(frame)
    folds = _purged_walk_forward_splits(row_count)
    maximum_test_end = max(fold["test_end_exclusive"] for fold in folds)
    optimization_frame = frame.iloc[:maximum_test_end].reset_index(drop=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(
        seed=seed, n_startup_trials=min(20, max(1, trial_count // 3))
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: _evaluate_optuna_trial(trial, optimization_frame, folds),
        n_trials=trial_count,
        n_jobs=1,
        show_progress_bar=False,
    )
    return _build_optuna_report(study, row_count, trial_count, seed, folds)


def _teacher_backtest_gate(
    validation_economics: dict[str, Any],
    holdout_economics: dict[str, Any],
    teacher_action_distribution: dict[str, Any],
    minimum_profit_factor: float = 1.5,
    minimum_trade_signal_count: int = _MIN_TEACHER_TRADE_SIGNAL_COUNT,
) -> dict[str, Any]:
    def profit_factor_passes(economics: dict[str, Any]) -> bool:
        profit_factor = economics.get("profit_factor")
        return (
            isinstance(profit_factor, (int, float))
            and np.isfinite(profit_factor)
            and float(profit_factor) > minimum_profit_factor
        )

    def drawdown_passes(economics: dict[str, Any]) -> bool:
        drawdown = economics.get("max_drawdown_fraction")
        return (
            isinstance(drawdown, (int, float))
            and np.isfinite(drawdown)
            and 0.0 <= float(drawdown) <= _MAX_TEACHER_DRAWDOWN_FRACTION
        )

    validation_profit_factor_passed = profit_factor_passes(validation_economics)
    holdout_profit_factor_passed = profit_factor_passes(holdout_economics)
    validation_drawdown_passed = drawdown_passes(validation_economics)
    holdout_drawdown_passed = drawdown_passes(holdout_economics)
    validation_passed = validation_profit_factor_passed and validation_drawdown_passed
    holdout_passed = holdout_profit_factor_passed and holdout_drawdown_passed
    buy_count = int(teacher_action_distribution.get("BUY", {}).get("count", 0))
    sell_count = int(teacher_action_distribution.get("SELL", {}).get("count", 0))
    trade_signal_count = buy_count + sell_count
    signal_density_passed = (
        trade_signal_count > minimum_trade_signal_count
        and buy_count > 0
        and sell_count > 0
    )
    passed = validation_passed and holdout_passed and signal_density_passed
    return {
        "minimum_profit_factor": minimum_profit_factor,
        "maximum_allowed_drawdown_fraction": _MAX_TEACHER_DRAWDOWN_FRACTION,
        "validation_profit_factor_passed": validation_profit_factor_passed,
        "holdout_profit_factor_passed": holdout_profit_factor_passed,
        "validation_drawdown_passed": validation_drawdown_passed,
        "holdout_drawdown_passed": holdout_drawdown_passed,
        "validation_passed": validation_passed,
        "holdout_passed": holdout_passed,
        "minimum_trade_signal_count": minimum_trade_signal_count,
        "signal_count_comparison": "strictly_greater_than_minimum",
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


def _teacher_parameter_config(
    entry_mode: str, optimized_parameters: dict[str, Any] | None = None
) -> dict[str, Any]:
    configuration: dict[str, Any] = {
        "entry_mode": entry_mode,
        "bollinger_period_candles": DEFAULT_MEAN_REVERSION_BOLLINGER_PERIOD,
        "bollinger_stddev": DEFAULT_MEAN_REVERSION_BOLLINGER_STDDEV,
        "rsi_period_candles": DEFAULT_MEAN_REVERSION_RSI_PERIOD,
        "rsi_buy_below": DEFAULT_MEAN_REVERSION_RSI_BUY_BELOW,
        "rsi_sell_above": DEFAULT_MEAN_REVERSION_RSI_SELL_ABOVE,
        "donchian_lookback_candles": DEFAULT_BREAKOUT_DONCHIAN_LOOKBACK,
        "take_profit_fraction": (
            0.015 if entry_mode == "mean_reversion" else DEFAULT_TAKE_PROFIT_FRACTION
        ),
        "stop_loss_fraction": (
            0.01 if entry_mode == "mean_reversion" else DEFAULT_STOP_LOSS_FRACTION
        ),
        "atr_cost_multiple": DEFAULT_MIN_ATR_COST_MULTIPLE,
    }
    if optimized_parameters and optimized_parameters.get("entry_mode") == entry_mode:
        for key in configuration:
            if key in optimized_parameters:
                configuration[key] = optimized_parameters[key]
    return configuration


def _teacher_entry_logic(
    entry_mode: str, optimized_parameters: dict[str, Any] | None = None
) -> dict[str, Any]:
    parameters = _teacher_parameter_config(entry_mode, optimized_parameters)
    if entry_mode == "breakout":
        lookback = int(parameters["donchian_lookback_candles"])
        return {
            "strategy": f"donchian_{lookback}_breakout_4h_trend_filtered",
            "buy_condition": (
                f"completed candle close exceeds the maximum high of the prior "
                f"{lookback} candles while completed 4h EMA bias is bullish"
            ),
            "sell_condition": (
                f"completed candle close falls below the minimum low of the prior "
                f"{lookback} candles; Spot long is closed, never shorted, "
                "regardless of the trend filter"
            ),
            "donchian_lookback_candles": lookback,
            "trend_filter": (
                "new Spot entries require bullish EMA bias from completed 4h candles; "
                "confirmed downside channel breaks remain active as exits"
            ),
            "breakout_confirmation": (
                "close beyond prior-candle channel; current candle excluded from "
                "channel; new entries require completed 4h bullish bias"
            ),
            "signals_use_completed_candles_only": True,
            "positioning": "Spot long/flat; SELL closes an existing long only",
        }
    if entry_mode == "mean_reversion":
        buy = float(parameters["rsi_buy_below"])
        sell = float(parameters["rsi_sell_above"])
        stddev = float(parameters["bollinger_stddev"])
        bollinger_period = int(parameters["bollinger_period_candles"])
        rsi_period = int(parameters["rsi_period_candles"])
        return {
            "strategy": "bollinger_rsi_mean_reversion",
            "buy_condition": (
                f"low touches or crosses lower Bollinger({bollinger_period}, "
                f"{stddev:g}) band and RSI-{rsi_period} < {buy:g}"
            ),
            "sell_condition": (
                f"high touches or crosses upper Bollinger({bollinger_period}, "
                f"{stddev:g}) band and RSI-{rsi_period} > {sell:g}; "
                "Spot long is closed, never shorted"
            ),
            "bollinger_period_candles": bollinger_period,
            "bollinger_standard_deviations": stddev,
            "rsi_period_candles": rsi_period,
            "rsi_buy_below": buy,
            "rsi_sell_above": sell,
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


def _teacher_risk_parameters(
    entry_mode: str = "institutional",
    optimized_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = _teacher_parameter_config(entry_mode, optimized_parameters)
    if entry_mode == "mean_reversion":
        trailing_trigger_fraction = 0.01
        trailing_distance_fraction = 0.01
    elif entry_mode in {"institutional", "trend_pullback", "breakout"}:
        trailing_trigger_fraction = DEFAULT_TRAILING_TRIGGER_FRACTION
        trailing_distance_fraction = DEFAULT_TRAILING_DISTANCE_FRACTION
    else:
        raise ValueError("unknown_teacher_entry_mode")
    fee_bps_per_side = DEFAULT_FEE_BPS_PER_SIDE
    slippage_bps_per_side = DEFAULT_SLIPPAGE_BPS_PER_SIDE
    atr_cost_multiple = float(config["atr_cost_multiple"])
    return {
        "entry_mode": entry_mode,
        "atr_period_candles": DEFAULT_ATR_PERIOD,
        "atr_cost_multiple": atr_cost_multiple,
        "minimum_atr_fraction": (
            atr_cost_multiple * (fee_bps_per_side + slippage_bps_per_side) / 10_000.0
        ),
        "take_profit_fraction": float(config["take_profit_fraction"]),
        "stop_loss_fraction": float(config["stop_loss_fraction"]),
        "trailing_trigger_fraction": trailing_trigger_fraction,
        "trailing_distance_fraction": trailing_distance_fraction,
        "trailing_breakeven_includes_round_trip_costs": True,
        "stop_target_same_candle_policy": "stop_first_conservative",
        "fee_bps_per_side": fee_bps_per_side,
        "slippage_bps_per_side": slippage_bps_per_side,
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
    optimized_parameters: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    parameter_config = _teacher_parameter_config(entry_mode, optimized_parameters)
    risk_parameters = _teacher_risk_parameters(entry_mode, optimized_parameters)
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
        "bollinger_period": int(parameter_config["bollinger_period_candles"]),
        "bollinger_stddev": float(parameter_config["bollinger_stddev"]),
        "rsi_period": int(parameter_config["rsi_period_candles"]),
        "rsi_buy_below": float(parameter_config["rsi_buy_below"]),
        "rsi_sell_above": float(parameter_config["rsi_sell_above"]),
        "breakout_donchian_lookback": int(
            parameter_config["donchian_lookback_candles"]
        ),
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
        "signal_count_comparison": "strictly_greater_than_minimum",
        "teacher_parameters": parameter_config,
        "teacher_risk_management": risk_parameters,
        "teacher_entry_logic": _teacher_entry_logic(entry_mode, optimized_parameters),
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
    optimized_parameters: dict[str, Any] | None = None,
) -> dict[str, tuple[pd.DataFrame, dict[str, Any]]]:
    modes = ("institutional", "trend_pullback", "mean_reversion", "breakout")
    return {
        mode: _write_teacher_dataset(
            frame, source_manifest, output_dir, commit_sha, mode, optimized_parameters
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
    entry_mode = str(teacher_manifest["teacher_entry_mode"])
    risk_parameters = dict(
        teacher_manifest.get(
            "teacher_risk_management", _teacher_risk_parameters(entry_mode)
        )
    )
    validation_economics = _simulate_teacher_segment(
        frame, teacher, train_end, validation_end, risk_parameters=risk_parameters
    )
    holdout_economics = _simulate_teacher_segment(
        frame, teacher, validation_end, row_count, risk_parameters=risk_parameters
    )
    validation_without_direct_costs = _simulate_teacher_segment(
        frame,
        teacher,
        train_end,
        validation_end,
        include_execution_costs=False,
        risk_parameters=risk_parameters,
    )
    holdout_without_direct_costs = _simulate_teacher_segment(
        frame,
        teacher,
        validation_end,
        row_count,
        include_execution_costs=False,
        risk_parameters=risk_parameters,
    )
    cost_counterfactual = _execution_cost_counterfactual_diagnosis(
        validation_economics,
        holdout_economics,
        validation_without_direct_costs,
        holdout_without_direct_costs,
    )
    risk_diagnostics = _cost_adjusted_risk_diagnostics(
        float(risk_parameters["take_profit_fraction"]),
        float(risk_parameters["stop_loss_fraction"]),
    )
    distribution = teacher_manifest["teacher_action_distribution"]
    teacher_parameters = dict(teacher_manifest.get("teacher_parameters", {}))
    return {
        "entry_mode": entry_mode,
        "entry_logic": teacher_manifest.get(
            "teacher_entry_logic", _teacher_entry_logic(entry_mode)
        ),
        "parameters": {**teacher_parameters, **risk_parameters},
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


def _validation_profit_factor(mode_result: dict[str, Any]) -> float:
    value = mode_result["validation"]["economics"].get("profit_factor")
    if not isinstance(value, (int, float)) or not np.isfinite(value):
        return float("-inf")
    return float(value)


def _any_mode_gate_passed(
    mode_results: dict[str, dict[str, Any]], gate_key: str
) -> bool:
    return any(result["gate"][gate_key] for result in mode_results.values())


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
            "maximum_allowed_drawdown_fraction": _MAX_TEACHER_DRAWDOWN_FRACTION,
            "validation_profit_factor_passed": _any_mode_gate_passed(
                mode_results, "validation_profit_factor_passed"
            ),
            "holdout_profit_factor_passed": _any_mode_gate_passed(
                mode_results, "holdout_profit_factor_passed"
            ),
            "validation_drawdown_passed": _any_mode_gate_passed(
                mode_results, "validation_drawdown_passed"
            ),
            "holdout_drawdown_passed": _any_mode_gate_passed(
                mode_results, "holdout_drawdown_passed"
            ),
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
        "schema_version": "alix.teacher-backtest-report.v4",
        "commit_sha": commit_sha,
        "source_csv_sha256": source_manifest["csv_sha256"],
        "strategy": (
            "Optuna-parameterized institutional proxy, trend-pullback, "
            "Bollinger-RSI mean-reversion, and trend-filtered Donchian breakout "
            "Spot teachers"
        ),
        "entry_mode": selected_mode,
        "selected_mode": selected_mode,
        "diagnostic_mode_by_wfv_objective": diagnostic_mode,
        "selection_policy": (
            "Select exactly one parameterized candidate by the Optuna purged "
            "walk-forward objective using only rows before validation. The independent "
            "validation segment and final holdout are acceptance-only gates and never "
            "participate in candidate selection."
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


def _teacher_gate_failure_reason(gate: dict[str, Any]) -> str:
    required_gates = (
        "validation_profit_factor_passed",
        "holdout_profit_factor_passed",
        "validation_drawdown_passed",
        "holdout_drawdown_passed",
        "signal_density_passed",
    )
    failures = [key for key in required_gates if gate.get(key) is not True]
    return ",".join(failures) if failures else "unknown_acceptance_gate_failure"


def run_training(
    dataset_path: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    commit_sha: str,
    epochs: int = 20,
    window: int = 48,
    optimization_trials: int = _DEFAULT_OPTUNA_TRIALS,
    optimization_seed: int = _DEFAULT_OPTUNA_SEED,
) -> dict[str, Any]:
    _validate_training_parameters(epochs, window)
    frame, source_manifest = load_public_market_data(
        dataset_path, manifest_path, commit_sha
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    row_count = len(frame)
    train_end = int(row_count * 0.70)
    validation_end = int(row_count * 0.85)
    optimization_report = _optimize_teacher_parameters(
        frame, trial_count=optimization_trials, seed=optimization_seed
    )
    optimization_path = output_dir / "teacher-optimization-report.json"
    optimization_path.write_text(
        json.dumps(optimization_report, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    optimized_parameters = dict(optimization_report["best_parameters"])
    teacher_candidates = _write_teacher_datasets(
        frame, source_manifest, output_dir, commit_sha, optimized_parameters
    )
    mode_backtests = {
        mode: _evaluate_teacher_mode(
            frame, teacher, manifest, train_end, validation_end
        )
        for mode, (teacher, manifest) in teacher_candidates.items()
    }
    selected_mode = str(optimized_parameters["entry_mode"])
    teacher_backtest = _build_teacher_backtest_report(
        commit_sha, source_manifest, mode_backtests, selected_mode
    )
    teacher_backtest["optimization"] = optimization_report
    _write_teacher_backtest_report(output_dir, teacher_backtest)
    if teacher_backtest["gate"]["passed"] is not True:
        reason = _teacher_gate_failure_reason(teacher_backtest["gate"])
        raise RuntimeError(
            "optuna_wfv_best_candidate_failed_independent_validation_or_holdout_gates:"
            + reason
            + ":lstm_training_blocked"
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
    parser.add_argument(
        "--optimization-trials", type=int, default=_DEFAULT_OPTUNA_TRIALS
    )
    parser.add_argument("--optimization-seed", type=int, default=_DEFAULT_OPTUNA_SEED)
    args = parser.parse_args()
    run_training(
        args.dataset,
        args.manifest,
        args.output_dir,
        commit_sha=str(args.commit_sha),
        epochs=int(args.epochs),
        window=int(args.window),
        optimization_trials=int(args.optimization_trials),
        optimization_seed=int(args.optimization_seed),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
