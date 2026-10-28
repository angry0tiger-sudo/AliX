"""Causal institutional-style teacher labels for offline imitation research.

All signals use completed candles only. This is a deterministic heuristic teacher,
not a claim that institutional order flow can be recovered from OHLCV alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2
_REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")

DEFAULT_ATR_PERIOD = 14
DEFAULT_MIN_ATR_COST_MULTIPLE = 1.0
DEFAULT_FEE_BPS_PER_SIDE = 25.0
DEFAULT_SLIPPAGE_BPS_PER_SIDE = 20.0
DEFAULT_TAKE_PROFIT_FRACTION = 0.025
DEFAULT_STOP_LOSS_FRACTION = 0.015
DEFAULT_TRAILING_TRIGGER_FRACTION = 0.01
DEFAULT_TRAILING_DISTANCE_FRACTION = 0.005


def _validate_timestamps(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        timestamp = pd.to_datetime(values, unit="ms", utc=True, errors="coerce")
    else:
        timestamp = pd.to_datetime(values, utc=True, errors="coerce")
    if timestamp.isna().any():
        raise ValueError("timestamps_must_be_valid")
    if not timestamp.is_monotonic_increasing or timestamp.duplicated().any():
        raise ValueError("timestamps_must_be_strictly_increasing")
    return timestamp


def _validate_price_bounds(frame: pd.DataFrame) -> None:
    if (frame["high"] < frame[["open", "low", "close"]].max(axis=1)).any():
        raise ValueError("high_below_ohlc_component")
    if (frame["low"] > frame[["open", "high", "close"]].min(axis=1)).any():
        raise ValueError("low_above_ohlc_component")


def _validate_numeric_ohlcv(frame: pd.DataFrame) -> None:
    numeric = frame.loc[:, ["open", "high", "low", "close", "volume"]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(numeric).all():
        raise ValueError("ohlcv_contains_non_finite_values")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("ohlc_prices_must_be_positive")
    if (frame["volume"] < 0).any():
        raise ValueError("volume_must_be_non_negative")
    _validate_price_bounds(frame)


def _validate_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    missing = set(_REQUIRED_COLUMNS).difference(data.columns)
    if missing:
        raise ValueError(f"missing_ohlcv_columns:{','.join(sorted(missing))}")
    frame = data.loc[:, list(_REQUIRED_COLUMNS)].copy()
    _validate_numeric_ohlcv(frame)
    frame["timestamp"] = _validate_timestamps(frame["timestamp"])
    return frame


def _average_true_range_fraction(frame: pd.DataFrame, period: int) -> pd.Series:
    """Return causal ATR divided by close, based only on completed candles."""
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(period, min_periods=period).mean()
    return (atr / frame["close"]).astype(np.float64)


def _higher_timeframe_bias(frame: pd.DataFrame, rule: str, ema_span: int) -> pd.Series:
    indexed = frame.set_index("timestamp")
    close = (
        indexed["close"].resample(rule, label="right", closed="left").last().dropna()
    )
    fast = close.ewm(span=ema_span, adjust=False, min_periods=ema_span).mean()
    slow = close.ewm(span=ema_span * 3, adjust=False, min_periods=ema_span * 3).mean()
    bias = pd.Series(
        np.where(fast > slow, 1, np.where(fast < slow, -1, 0)), index=close.index
    )
    # At each minute, use only the most recently completed higher-timeframe bar.
    return (
        bias.reindex(indexed.index, method="ffill")
        .fillna(0)
        .astype(np.int8)
        .reset_index(drop=True)
    )


def _overlaps_order_block(
    low: pd.Series,
    high: pd.Series,
    zone_low: pd.Series,
    zone_high: pd.Series,
) -> pd.Series:
    return ((low <= zone_high) & (high >= zone_low)).fillna(False)


def _smc_features(
    frame: pd.DataFrame, swing_window: int
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    high = frame["high"]
    low = frame["low"]
    close = frame["close"]
    # Shift one candle so the current candle cannot define its own break level.
    prior_high = high.rolling(swing_window, min_periods=swing_window).max().shift(1)
    prior_low = low.rolling(swing_window, min_periods=swing_window).min().shift(1)
    bos_up = close > prior_high
    bos_down = close < prior_low

    # Three-candle fair-value gaps, confirmed only after the third candle closes.
    fvg_bull = low > high.shift(2)
    fvg_bear = high < low.shift(2)
    # A last-opposing-candle proxy for order blocks, using prior candles only.
    previous_red = close.shift(1) < frame["open"].shift(1)
    previous_green = close.shift(1) > frame["open"].shift(1)
    bull_ob_low = frame["low"].shift(1).where(previous_red).ffill()
    bull_ob_high = frame["high"].shift(1).where(previous_red).ffill()
    bear_ob_low = frame["low"].shift(1).where(previous_green).ffill()
    bear_ob_high = frame["high"].shift(1).where(previous_green).ffill()
    bullish_ob_touch = _overlaps_order_block(low, high, bull_ob_low, bull_ob_high)
    bearish_ob_touch = _overlaps_order_block(low, high, bear_ob_low, bear_ob_high)

    pullback_bull = ((low <= prior_high) & (close >= prior_high)) | (
        fvg_bull & (close <= high.shift(2) + (low - high.shift(2)) * 0.5)
    )
    pullback_bear = ((high >= prior_low) & (close <= prior_low)) | (
        fvg_bear & (close >= low.shift(2) + (high - low.shift(2)) * 0.5)
    )
    bullish_structure = (bos_up | pullback_bull).fillna(False)
    bearish_structure = (bos_down | pullback_bear).fillna(False)
    return (
        bullish_structure,
        bearish_structure,
        prior_high,
        bullish_ob_touch,
        bearish_ob_touch,
    )


def _volume_confirmations(
    frame: pd.DataFrame, window: int
) -> tuple[pd.Series, pd.Series]:
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    # Rolling VWAP avoids using future data or requiring a session-boundary assumption.
    volume_sum = frame["volume"].rolling(window, min_periods=window).sum()
    vwap = (typical * frame["volume"]).rolling(
        window, min_periods=window
    ).sum() / volume_sum.replace(0, np.nan)
    # Rolling modal price-bin approximation to volume profile, using only trailing data.
    rolling_poc = frame["close"].rolling(window, min_periods=window).median()
    buy_confirm = (frame["close"] >= vwap) & (frame["close"] >= rolling_poc)
    sell_confirm = (frame["close"] <= vwap) & (frame["close"] <= rolling_poc)
    return buy_confirm.fillna(False), sell_confirm.fillna(False)


def _teacher_setups(
    bias: pd.Series,
    bullish_structure: pd.Series,
    bearish_structure: pd.Series,
    bullish_ob_touch: pd.Series,
    bearish_ob_touch: pd.Series,
    buy_volume: pd.Series,
    sell_volume: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Allow an in-trend order-block touch without requiring BOS/FVG or volume."""
    buy_setup = (bias > 0) & ((bullish_structure & buy_volume) | bullish_ob_touch)
    sell_setup = (bias < 0) & ((bearish_structure & sell_volume) | bearish_ob_touch)
    return buy_setup.fillna(False), sell_setup.fillna(False)


def _trend_pullback_setups(
    frame: pd.DataFrame, bias: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Simpler causal pullback/reclaim entries aligned with the completed 4h bias."""
    close = frame["close"]
    fast_ema = close.ewm(span=6, adjust=False, min_periods=6).mean()
    slow_ema = close.ewm(span=24, adjust=False, min_periods=24).mean()
    buy_setup = (
        (bias > 0)
        & (close > slow_ema)
        & (frame["low"] <= fast_ema)
        & (close >= fast_ema)
    )
    sell_setup = (
        (bias < 0)
        & (close < slow_ema)
        & (frame["high"] >= fast_ema)
        & (close <= fast_ema)
    )
    return buy_setup.fillna(False), sell_setup.fillna(False)


def _mean_reversion_setups(
    frame: pd.DataFrame,
    *,
    bollinger_period: int = 20,
    bollinger_stddev: float = 2.0,
    rsi_period: int = 14,
) -> tuple[pd.Series, pd.Series]:
    """Use completed-candle Bollinger touches confirmed by Wilder-style RSI."""
    close = frame["close"].astype(np.float64)
    basis = close.rolling(bollinger_period, min_periods=bollinger_period).mean()
    deviation = close.rolling(bollinger_period, min_periods=bollinger_period).std(
        ddof=0
    )
    lower_band = basis - bollinger_stddev * deviation
    upper_band = basis + bollinger_stddev * deviation

    changes = close.diff()
    gains = changes.clip(lower=0.0)
    losses = -changes.clip(upper=0.0)
    average_gain = gains.ewm(
        alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period
    ).mean()
    average_loss = losses.ewm(
        alpha=1.0 / rsi_period, adjust=False, min_periods=rsi_period
    ).mean()
    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))
    rsi = rsi.mask((average_gain == 0.0) & (average_loss > 0.0), 0.0)
    rsi = rsi.mask((average_loss == 0.0) & (average_gain > 0.0), 100.0)
    rsi = rsi.mask((average_gain == 0.0) & (average_loss == 0.0), 50.0)

    buy_setup = (frame["low"] <= lower_band) & (rsi < 30.0)
    sell_setup = (frame["high"] >= upper_band) & (rsi > 70.0)
    return buy_setup.fillna(False), sell_setup.fillna(False)


def _validate_teacher_entry_mode(entry_mode: str) -> None:
    if entry_mode not in {"institutional", "trend_pullback", "mean_reversion"}:
        raise ValueError(
            "teacher_entry_mode_must_be_institutional_trend_pullback_or_mean_reversion"
        )


def _validate_integer_parameter(name: str, value: int, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name}_out_of_range")


def _validate_teacher_configuration(
    higher_timeframe: str,
    ema_span: int,
    swing_window: int,
    volume_window: int,
    cooldown_candles: int,
) -> None:
    for name, value, minimum in (
        ("ema_span", ema_span, 2),
        ("swing_window", swing_window, 2),
        ("volume_window", volume_window, 2),
        ("cooldown_candles", cooldown_candles, 0),
    ):
        _validate_integer_parameter(name, value, minimum)
    if higher_timeframe not in {"4h", "1D", "4h+1D"}:
        raise ValueError("higher_timeframe_must_be_4h_1D_or_4h+1D")


def _validate_positive_parameter(name: str, value: float, upper_bound: float) -> None:
    if not np.isfinite(value):
        raise ValueError("teacher_risk_parameters_must_be_finite")
    if not 0 < value < upper_bound:
        raise ValueError(f"{name}_out_of_range")


def _validate_cost_parameter(name: str, value: float) -> None:
    if not np.isfinite(value):
        raise ValueError("teacher_risk_parameters_must_be_finite")
    if value < 0 or value >= 10_000:
        raise ValueError(f"{name}_out_of_range")


def _validate_risk_configuration(
    atr_period: int,
    min_atr_cost_multiple: float,
    take_profit_fraction: float,
    stop_loss_fraction: float,
    trailing_trigger_fraction: float,
    trailing_distance_fraction: float,
    fee_bps_per_side: float,
    slippage_bps_per_side: float,
) -> None:
    _validate_integer_parameter("atr_period", atr_period, 2)
    _validate_positive_parameter("min_atr_cost_multiple", min_atr_cost_multiple, np.inf)
    for name, value in (
        ("take_profit_fraction", take_profit_fraction),
        ("stop_loss_fraction", stop_loss_fraction),
        ("trailing_trigger_fraction", trailing_trigger_fraction),
        ("trailing_distance_fraction", trailing_distance_fraction),
    ):
        _validate_positive_parameter(name, value, 1.0)
    _validate_cost_parameter("fee_bps_per_side", fee_bps_per_side)
    _validate_cost_parameter("slippage_bps_per_side", slippage_bps_per_side)


def _select_timeframe_biases(
    frame: pd.DataFrame, higher_timeframe: str, ema_span: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    bias_4h = _higher_timeframe_bias(frame, "4h", ema_span)
    bias_1d = _higher_timeframe_bias(frame, "1D", ema_span)
    if higher_timeframe == "4h":
        return bias_4h, bias_4h, bias_1d
    if higher_timeframe == "1D":
        return bias_1d, bias_4h, bias_1d
    confluence = bias_4h.where(bias_4h == bias_1d, 0).astype(np.int8)
    return confluence, bias_4h, bias_1d


@dataclass
class _TeacherPositionState:
    has_position: bool = False
    entry_price: float = 0.0
    entry_index: int = -1
    stop_price: float = 0.0
    target_price: float = 0.0
    high_watermark: float = 0.0
    active_stop_reason: str = "stop_loss"
    last_change: int = -1


def _record_teacher_exit(
    index: int,
    fill_price: float,
    reason: str,
    state: _TeacherPositionState,
    actions: NDArray[np.int8],
    exit_prices: FloatArray,
    exit_reasons: list[str],
) -> None:
    actions[index] = ACTION_SELL
    exit_prices[index] = fill_price
    exit_reasons[index] = reason
    state.has_position = False
    state.last_change = index


def _advance_trailing_stop(
    high_price: float,
    state: _TeacherPositionState,
    trailing_trigger_fraction: float,
    trailing_distance_fraction: float,
    cost_recovery_fraction: float,
) -> None:
    state.high_watermark = max(state.high_watermark, high_price)
    trigger_price = state.entry_price * (1.0 + trailing_trigger_fraction)
    if state.high_watermark < trigger_price:
        return
    cost_adjusted_breakeven = state.entry_price * (1.0 + cost_recovery_fraction)
    trailing_level = state.high_watermark * (1.0 - trailing_distance_fraction)
    candidate_stop = max(cost_adjusted_breakeven, trailing_level)
    if candidate_stop > state.stop_price + 1e-12:
        state.stop_price = candidate_stop
        state.active_stop_reason = "trailing_stop"


def _check_teacher_protective_exit(
    index: int,
    opens: FloatArray,
    highs: FloatArray,
    lows: FloatArray,
    state: _TeacherPositionState,
    actions: NDArray[np.int8],
    exit_prices: FloatArray,
    exit_reasons: list[str],
    trailing_trigger_fraction: float,
    trailing_distance_fraction: float,
    cost_recovery_fraction: float,
) -> bool:
    if not state.has_position or index <= state.entry_index:
        return False
    if lows[index] <= state.stop_price:
        fill = min(opens[index], state.stop_price)
        _record_teacher_exit(
            index,
            fill,
            state.active_stop_reason,
            state,
            actions,
            exit_prices,
            exit_reasons,
        )
        return True
    if highs[index] >= state.target_price:
        fill = max(opens[index], state.target_price)
        _record_teacher_exit(
            index, fill, "take_profit", state, actions, exit_prices, exit_reasons
        )
        return True
    _advance_trailing_stop(
        highs[index],
        state,
        trailing_trigger_fraction,
        trailing_distance_fraction,
        cost_recovery_fraction,
    )
    return False


def _check_teacher_opposing_exit(
    index: int,
    sell_setup: bool,
    close_price: float,
    cooldown_candles: int,
    state: _TeacherPositionState,
    actions: NDArray[np.int8],
    exit_prices: FloatArray,
    exit_reasons: list[str],
) -> bool:
    if not state.has_position or not sell_setup:
        return False
    if index - state.last_change <= cooldown_candles:
        return False
    _record_teacher_exit(
        index, close_price, "opposing_setup", state, actions, exit_prices, exit_reasons
    )
    return True


def _try_open_teacher_position(
    index: int,
    buy_setup: bool,
    atr_value: float,
    close_price: float,
    cooldown_candles: int,
    min_atr_fraction: float,
    take_profit_fraction: float,
    stop_loss_fraction: float,
    state: _TeacherPositionState,
    actions: NDArray[np.int8],
) -> None:
    if state.has_position or not buy_setup:
        return
    if not np.isfinite(atr_value) or atr_value < min_atr_fraction:
        return
    if index - state.last_change <= cooldown_candles:
        return
    state.has_position = True
    state.entry_price = close_price
    state.entry_index = index
    state.stop_price = close_price * (1.0 - stop_loss_fraction)
    state.target_price = close_price * (1.0 + take_profit_fraction)
    state.high_watermark = close_price
    state.active_stop_reason = "stop_loss"
    state.last_change = index
    actions[index] = ACTION_BUY


def _generate_teacher_actions(
    frame: pd.DataFrame,
    buy_setup: pd.Series,
    sell_setup: pd.Series,
    atr_fraction: pd.Series,
    *,
    cooldown_candles: int,
    min_atr_cost_multiple: float,
    take_profit_fraction: float,
    stop_loss_fraction: float,
    trailing_trigger_fraction: float,
    trailing_distance_fraction: float,
    fee_bps_per_side: float,
    slippage_bps_per_side: float,
) -> tuple[NDArray[np.int8], NDArray[np.int8], FloatArray, list[str]]:
    """Generate Spot transitions with ATR gating and causal TP/SL/trailing exits.

    Setups are known at candle close. A new position becomes active after that
    close; protective levels are checked against later candles only. If a candle
    crosses both its existing stop and target, the stop is prioritized
    conservatively because OHLC data cannot reveal intrabar path ordering.
    """
    count = len(frame)
    if not (len(buy_setup) == len(sell_setup) == len(atr_fraction) == count):
        raise ValueError("teacher_action_input_length_mismatch")
    actions = np.full(count, ACTION_HOLD, dtype=np.int8)
    positions = np.zeros(count, dtype=np.int8)
    exit_prices = np.full(count, np.nan, dtype=np.float64)
    exit_reasons = [""] * count
    buys = buy_setup.fillna(False).to_numpy(dtype=bool)
    sells = sell_setup.fillna(False).to_numpy(dtype=bool)
    atr_values = atr_fraction.to_numpy(dtype=np.float64)
    opens = frame["open"].to_numpy(dtype=np.float64)
    highs = frame["high"].to_numpy(dtype=np.float64)
    lows = frame["low"].to_numpy(dtype=np.float64)
    closes = frame["close"].to_numpy(dtype=np.float64)

    fee_rate = fee_bps_per_side / 10_000.0
    slippage_rate = slippage_bps_per_side / 10_000.0
    cost_recovery_fraction = (1.0 + slippage_rate) * (1.0 + fee_rate) / (
        (1.0 - slippage_rate) * (1.0 - fee_rate)
    ) - 1.0
    min_atr_fraction = (
        min_atr_cost_multiple * (fee_bps_per_side + slippage_bps_per_side) / 10_000.0
    )
    state = _TeacherPositionState(last_change=-cooldown_candles - 1)

    for index in range(count):
        did_exit = _check_teacher_protective_exit(
            index,
            opens,
            highs,
            lows,
            state,
            actions,
            exit_prices,
            exit_reasons,
            trailing_trigger_fraction,
            trailing_distance_fraction,
            cost_recovery_fraction,
        )
        if not did_exit:
            did_exit = _check_teacher_opposing_exit(
                index,
                bool(sells[index]),
                closes[index],
                cooldown_candles,
                state,
                actions,
                exit_prices,
                exit_reasons,
            )
        if not did_exit:
            _try_open_teacher_position(
                index,
                bool(buys[index]),
                float(atr_values[index]),
                closes[index],
                cooldown_candles,
                min_atr_fraction,
                take_profit_fraction,
                stop_loss_fraction,
                state,
                actions,
            )
        positions[index] = 1 if state.has_position else 0
    return actions, positions, exit_prices, exit_reasons


def _build_teacher_result(
    frame: pd.DataFrame,
    bias: pd.Series,
    bias_4h: pd.Series,
    bias_1d: pd.Series,
    prior_high: pd.Series,
    bullish_ob_touch: pd.Series,
    bearish_ob_touch: pd.Series,
    actions: NDArray[np.int8],
    positions: NDArray[np.int8],
) -> pd.DataFrame:
    result = frame.copy()
    result["higher_timeframe_bias"] = bias.to_numpy(dtype=np.int8)
    result["higher_timeframe_bias_4h"] = bias_4h.to_numpy(dtype=np.int8)
    result["higher_timeframe_bias_1d"] = bias_1d.to_numpy(dtype=np.int8)
    result["bos_reference_high"] = prior_high.to_numpy(dtype=np.float64)
    result["order_block_touch_bullish"] = bullish_ob_touch.to_numpy(dtype=bool)
    result["order_block_touch_bearish"] = bearish_ob_touch.to_numpy(dtype=bool)
    result["teacher_action"] = actions
    result["teacher_action_name"] = np.select(
        [actions == ACTION_BUY, actions == ACTION_SELL],
        ["BUY", "SELL"],
        default="HOLD",
    )
    result["teacher_position"] = positions
    return result


def generate_teacher_labels(
    data: pd.DataFrame,
    *,
    higher_timeframe: str = "4h",
    entry_mode: str = "institutional",
    ema_span: int = 6,
    swing_window: int = 8,
    volume_window: int = 24,
    cooldown_candles: int = 2,
    atr_period: int = DEFAULT_ATR_PERIOD,
    min_atr_cost_multiple: float = DEFAULT_MIN_ATR_COST_MULTIPLE,
    take_profit_fraction: float = DEFAULT_TAKE_PROFIT_FRACTION,
    stop_loss_fraction: float = DEFAULT_STOP_LOSS_FRACTION,
    trailing_trigger_fraction: float = DEFAULT_TRAILING_TRIGGER_FRACTION,
    trailing_distance_fraction: float = DEFAULT_TRAILING_DISTANCE_FRACTION,
    fee_bps_per_side: float = DEFAULT_FEE_BPS_PER_SIDE,
    slippage_bps_per_side: float = DEFAULT_SLIPPAGE_BPS_PER_SIDE,
) -> pd.DataFrame:
    """Return causal Spot transitions with volatility gating and managed exits.

    BUY/SELL are research labels, not executable orders. Signals use completed
    candles only; TP/SL and trailing checks begin on candles after entry.
    """
    _validate_teacher_configuration(
        higher_timeframe, ema_span, swing_window, volume_window, cooldown_candles
    )
    _validate_teacher_entry_mode(entry_mode)
    _validate_risk_configuration(
        atr_period,
        min_atr_cost_multiple,
        take_profit_fraction,
        stop_loss_fraction,
        trailing_trigger_fraction,
        trailing_distance_fraction,
        fee_bps_per_side,
        slippage_bps_per_side,
    )
    frame = _validate_ohlcv(data)
    bias, bias_4h, bias_1d = _select_timeframe_biases(frame, higher_timeframe, ema_span)
    (
        bullish_structure,
        bearish_structure,
        prior_high,
        bullish_ob_touch,
        bearish_ob_touch,
    ) = _smc_features(frame, swing_window)
    buy_volume, sell_volume = _volume_confirmations(frame, volume_window)
    if entry_mode == "institutional":
        buy_setup, sell_setup = _teacher_setups(
            bias,
            bullish_structure,
            bearish_structure,
            bullish_ob_touch,
            bearish_ob_touch,
            buy_volume,
            sell_volume,
        )
    elif entry_mode == "trend_pullback":
        buy_setup, sell_setup = _trend_pullback_setups(frame, bias)
    else:
        buy_setup, sell_setup = _mean_reversion_setups(frame)
    atr_fraction = _average_true_range_fraction(frame, atr_period)
    actions, positions, exit_prices, exit_reasons = _generate_teacher_actions(
        frame,
        buy_setup,
        sell_setup,
        atr_fraction,
        cooldown_candles=cooldown_candles,
        min_atr_cost_multiple=min_atr_cost_multiple,
        take_profit_fraction=take_profit_fraction,
        stop_loss_fraction=stop_loss_fraction,
        trailing_trigger_fraction=trailing_trigger_fraction,
        trailing_distance_fraction=trailing_distance_fraction,
        fee_bps_per_side=fee_bps_per_side,
        slippage_bps_per_side=slippage_bps_per_side,
    )
    result = _build_teacher_result(
        frame,
        bias,
        bias_4h,
        bias_1d,
        prior_high,
        bullish_ob_touch,
        bearish_ob_touch,
        actions,
        positions,
    )
    result["teacher_atr_fraction"] = atr_fraction.to_numpy(dtype=np.float64)
    result["teacher_setup_buy"] = buy_setup.to_numpy(dtype=bool)
    result["teacher_setup_sell"] = sell_setup.to_numpy(dtype=bool)
    result["teacher_entry_mode"] = entry_mode
    result["teacher_exit_price"] = exit_prices
    result["teacher_exit_reason"] = exit_reasons
    return result
