"""Research-only adaptive timeframe, risk, trailing and post-trade policy.

The policy is deterministic and auditable in this first research version. Its
bounded post-trade updates are chronological feedback, not a trained or
production-approved model.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from pandas import DataFrame

TIMEFRAMES = ("1h", "4h", "1d")
OHLCV_COLUMNS = ("date", "open", "high", "low", "close", "volume")
POLICY_SCHEMA = "alix.adaptive-autonomous-ai-policy.v1"
_CONTEXT_COLUMNS = (
    "close",
    "ema_fast",
    "ema_slow",
    "atr",
    "atr_ratio",
    "trend_strength",
    "volatility_score",
    "macd_histogram",
    "support",
)


@dataclass(frozen=True, slots=True)
class AdaptiveConfig:
    """Versioned limits for the research policy; not exchange execution settings."""

    minimum_selection_confidence: float = 0.55
    minimum_entry_trend_strength: float = 0.35
    minimum_risk_fraction: float = 0.001
    maximum_risk_fraction: float = 0.0075
    maximum_position_fraction: float = 0.95
    maximum_stop_distance_fraction: float = 0.12
    support_buffer_atr: float = 0.15
    strong_trend_threshold: float = 0.65
    strong_trailing_atr_multiple: float = 3.0
    weak_trailing_atr_multiple: float = 1.25
    initial_stop_atr_multiple: float = 2.0
    initial_trailing_atr_multiple: float = 2.5
    minimum_stop_atr_multiple: float = 1.25
    maximum_stop_atr_multiple: float = 4.0
    minimum_trailing_atr_multiple: float = 1.0
    maximum_trailing_atr_multiple: float = 5.0
    minimum_confidence_floor: float = 0.50
    maximum_confidence_floor: float = 0.80
    stop_adaptation_step: float = 0.25
    confidence_adaptation_step: float = 0.02
    fee_bps_per_side: float = 25.0
    slippage_bps_per_side: float = 20.0
    initial_equity: float = 10_000.0

    def __post_init__(self) -> None:
        _validate_config_values(asdict(self))
        _validate_config_selection(self)
        _validate_config_risk(self)
        _validate_config_structure(self)
        _validate_config_stop(self)
        _validate_config_trailing(self)
        _validate_config_confidence(self)
        _validate_config_costs(self)


def _validate_config_values(values: Mapping[str, object]) -> None:
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values.values()
    ):
        raise ValueError("adaptive_config_values_non_finite")


def _validate_config_selection(config: AdaptiveConfig) -> None:
    if not 0 < config.minimum_selection_confidence <= 1:
        raise ValueError("adaptive_config_selection_confidence_invalid")
    if not 0 <= config.minimum_entry_trend_strength <= 1:
        raise ValueError("adaptive_config_trend_strength_invalid")


def _validate_config_risk(config: AdaptiveConfig) -> None:
    if not 0 < config.minimum_risk_fraction <= config.maximum_risk_fraction < 1:
        raise ValueError("adaptive_config_risk_bounds_invalid")
    if not 0 < config.maximum_position_fraction <= 1:
        raise ValueError("adaptive_config_position_cap_invalid")


def _validate_config_structure(config: AdaptiveConfig) -> None:
    if not 0 < config.maximum_stop_distance_fraction < 1:
        raise ValueError("adaptive_config_stop_distance_cap_invalid")
    if config.support_buffer_atr < 0 or not 0 < config.strong_trend_threshold <= 1:
        raise ValueError("adaptive_config_structure_or_trend_invalid")


def _validate_config_stop(config: AdaptiveConfig) -> None:
    if not (
        0
        < config.minimum_stop_atr_multiple
        <= config.initial_stop_atr_multiple
        <= config.maximum_stop_atr_multiple
    ):
        raise ValueError("adaptive_config_stop_multiple_bounds_invalid")


def _validate_config_trailing(config: AdaptiveConfig) -> None:
    if not (
        0
        < config.minimum_trailing_atr_multiple
        <= config.initial_trailing_atr_multiple
        <= config.maximum_trailing_atr_multiple
    ):
        raise ValueError("adaptive_config_trailing_multiple_bounds_invalid")
    if (
        config.strong_trailing_atr_multiple < config.minimum_trailing_atr_multiple
        or config.strong_trailing_atr_multiple > config.maximum_trailing_atr_multiple
        or config.weak_trailing_atr_multiple < config.minimum_trailing_atr_multiple
        or config.weak_trailing_atr_multiple > config.maximum_trailing_atr_multiple
    ):
        raise ValueError("adaptive_config_trailing_policy_bounds_invalid")


def _validate_config_confidence(config: AdaptiveConfig) -> None:
    if not (
        0
        < config.minimum_confidence_floor
        <= config.minimum_selection_confidence
        <= config.maximum_confidence_floor
        <= 1
    ):
        raise ValueError("adaptive_config_confidence_floor_bounds_invalid")


def _validate_config_costs(config: AdaptiveConfig) -> None:
    if (
        min(
            config.stop_adaptation_step,
            config.confidence_adaptation_step,
            config.initial_equity,
        )
        <= 0
        or min(config.fee_bps_per_side, config.slippage_bps_per_side) < 0
    ):
        raise ValueError("adaptive_config_cost_or_step_invalid")


@dataclass(frozen=True, slots=True)
class AdaptiveParameterState:
    """Small, auditable online state updated only after a trade closes."""

    atr_stop_multiple: float = 2.0
    trailing_atr_multiple: float = 2.5
    entry_confidence_floor: float = 0.55
    revision: int = 0
    trades_observed: int = 0
    consecutive_wins: int = 0


@dataclass(frozen=True, slots=True)
class TimeframeDecision:
    timeframe: str | None
    confidence: float
    scores: Mapping[str, float]
    reason: str


@dataclass(frozen=True, slots=True)
class DynamicStopPlan:
    stop_loss: float
    stop_distance: float
    support_level: float
    atr_stop_multiple: float


@dataclass(frozen=True, slots=True)
class AdaptivePositionPlan:
    quantity: float
    notional: float
    risk_budget: float
    risk_fraction: float


def initial_parameter_state(config: AdaptiveConfig) -> AdaptiveParameterState:
    return AdaptiveParameterState(
        atr_stop_multiple=config.initial_stop_atr_multiple,
        trailing_atr_multiple=config.initial_trailing_atr_multiple,
        entry_confidence_floor=config.minimum_selection_confidence,
    )


def _read_adaptive_policy_document(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("adaptive_config_document_unreadable") from exc


def _validate_adaptive_policy_document(
    payload: object,
) -> tuple[str, dict[str, Any]]:
    expected_keys = {"schema_version", "revision", "code_sha256", "parameters"}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("adaptive_config_document_schema_invalid")
    if payload["schema_version"] != POLICY_SCHEMA:
        raise ValueError("adaptive_config_schema_unsupported")
    revision = payload["revision"]
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("adaptive_config_revision_invalid")
    expected_hash = payload["code_sha256"]
    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or any(char not in "0123456789abcdef" for char in expected_hash)
    ):
        raise ValueError("adaptive_config_code_hash_invalid")
    parameters = payload["parameters"]
    if not isinstance(parameters, dict) or set(parameters) != set(
        asdict(AdaptiveConfig())
    ):
        raise ValueError("adaptive_config_parameters_invalid")
    return expected_hash, parameters


def _verify_adaptive_module_hash(module_path: Path, expected_hash: str) -> None:
    try:
        actual_hash = hashlib.sha256(module_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError("adaptive_config_module_unreadable") from exc
    if actual_hash != expected_hash:
        raise ValueError("adaptive_config_code_hash_mismatch")


def _construct_adaptive_config(parameters: dict[str, Any]) -> AdaptiveConfig:
    try:
        return AdaptiveConfig(**parameters)
    except (TypeError, ValueError) as exc:
        raise ValueError("adaptive_config_parameters_invalid") from exc


def load_adaptive_config(path: Path, module_path: Path) -> AdaptiveConfig:
    """Load a strict policy document and verify the exact module SHA-256."""
    payload = _read_adaptive_policy_document(path)
    expected_hash, parameters = _validate_adaptive_policy_document(payload)
    _verify_adaptive_module_hash(module_path, expected_hash)
    return _construct_adaptive_config(parameters)


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _validate_hourly_timestamps(result: DataFrame) -> None:
    dates = pd.to_datetime(result["date"], errors="raise")
    if dates.dt.tz is None:
        raise ValueError("adaptive_candle_timestamps_must_be_timezone_aware")
    result["date"] = dates.dt.tz_convert("UTC")
    if result["date"].isna().any() or result["date"].duplicated().any():
        raise ValueError("adaptive_candle_timestamps_invalid")
    if not result["date"].is_monotonic_increasing:
        raise ValueError("adaptive_candle_timestamps_not_monotonic")
    if not result["date"].diff().dropna().eq(pd.Timedelta(1, unit="h")).all():
        raise ValueError("adaptive_candles_not_contiguous_hourly")


def _validate_hourly_numeric_values(result: DataFrame) -> None:
    for column in ("open", "high", "low", "close", "volume"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    prices = result[["open", "high", "low", "close"]].to_numpy(dtype=float)
    volume = result["volume"].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or not np.isfinite(volume).all():
        raise ValueError("adaptive_candles_non_finite")
    if (prices <= 0).any() or (volume < 0).any():
        raise ValueError("adaptive_candles_out_of_range")
    if (
        (result["high"] < result[["open", "close"]].max(axis=1)).any()
        or (result["low"] > result[["open", "close"]].min(axis=1)).any()
        or (result["high"] < result["low"]).any()
    ):
        raise ValueError("adaptive_candles_ohlc_invariant_invalid")


def _validate_hourly_candles(candles: DataFrame) -> DataFrame:
    missing = sorted(set(OHLCV_COLUMNS) - set(candles.columns))
    if missing:
        raise ValueError("adaptive_candles_columns_missing")
    if candles.empty:
        raise ValueError("adaptive_candles_empty")
    result = candles.loc[:, list(OHLCV_COLUMNS)].copy()
    _validate_hourly_timestamps(result)
    _validate_hourly_numeric_values(result)
    return result.reset_index(drop=True)


def _resample_complete_bars(bars: DataFrame, timeframe: str) -> DataFrame:
    rule, expected_rows = {"4h": ("4h", 4), "1d": ("1D", 24)}[timeframe]
    indexed = bars.set_index("date")
    aggregation = indexed.resample(
        rule, closed="left", label="right", origin="start_day"
    ).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    counts = (
        indexed["close"]
        .resample(rule, closed="left", label="right", origin="start_day")
        .count()
    )
    return aggregation.loc[counts.eq(expected_rows)].reset_index()


def _context_features(bars: DataFrame, timeframe: str) -> DataFrame:
    close = bars["close"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    fast = close.ewm(span=8, adjust=False, min_periods=8).mean()
    slow = close.ewm(span=21, adjust=False, min_periods=21).mean()
    macd = (
        close.ewm(span=12, adjust=False, min_periods=12).mean()
        - close.ewm(span=26, adjust=False, min_periods=26).mean()
    )
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    prior_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prior_close).abs(),
            (low - prior_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(14, min_periods=14).mean()
    atr_ratio = atr / close
    volatility_median = atr_ratio.rolling(50, min_periods=10).median()
    relative_volatility = atr_ratio / volatility_median.where(volatility_median > 0)
    spread_in_atr = (fast - slow).abs() / atr.where(atr > 0)
    strength = spread_in_atr / (1.0 + spread_in_atr)
    support = low.shift(1).rolling(20, min_periods=20).min()
    result = DataFrame(
        {
            "context_time": bars["date"],
            f"{timeframe}_close": close,
            f"{timeframe}_ema_fast": fast,
            f"{timeframe}_ema_slow": slow,
            f"{timeframe}_atr": atr,
            f"{timeframe}_atr_ratio": atr_ratio,
            f"{timeframe}_trend_strength": strength.clip(0.0, 1.0),
            f"{timeframe}_volatility_score": (relative_volatility.clip(0.0, 2.0) / 2.0),
            f"{timeframe}_macd_histogram": macd - macd_signal,
            f"{timeframe}_support": support,
        }
    )
    result["context_time"] = pd.to_datetime(result["context_time"], utc=True)
    if timeframe == "1h":
        result["context_time"] = result["context_time"] + pd.Timedelta(1, unit="h")
    return result


def build_multitimeframe_contexts(candles: DataFrame) -> DataFrame:
    """Align closed 1h/4h/1d contexts to each hourly close, never to future bars."""
    bars = _validate_hourly_candles(candles)
    decision_times = bars["date"] + pd.Timedelta(1, unit="h")
    aligned = DataFrame({"decision_time": decision_times})
    for timeframe in TIMEFRAMES:
        timeframe_bars = (
            bars if timeframe == "1h" else _resample_complete_bars(bars, timeframe)
        )
        context = _context_features(timeframe_bars, timeframe)
        context = context.sort_values("context_time")
        aligned = pd.merge_asof(
            aligned.sort_values("decision_time"),
            context,
            left_on="decision_time",
            right_on="context_time",
            direction="backward",
            allow_exact_matches=True,
        )
        aligned = aligned.drop(columns=["context_time"])
    return aligned.reset_index(drop=True)


def _valid_context(
    row: Mapping[str, object], timeframe: str
) -> dict[str, float] | None:
    context: dict[str, float] = {}
    for name in _CONTEXT_COLUMNS:
        value = _finite_number(row.get(f"{timeframe}_{name}"))
        if value is None:
            return None
        context[name] = value
    if (
        context["close"] <= 0
        or context["atr"] <= 0
        or context["support"] <= 0
        or context["support"] >= context["close"]
        or not 0 <= context["trend_strength"] <= 1
        or not 0 <= context["volatility_score"] <= 1
    ):
        return None
    return context


def select_adaptive_timeframe(
    row: Mapping[str, object],
    state: AdaptiveParameterState,
) -> TimeframeDecision:
    """Score all valid closed timeframe contexts; confidence is an uncalibrated margin."""
    contexts = {
        timeframe: context
        for timeframe in TIMEFRAMES
        if (context := _valid_context(row, timeframe)) is not None
    }
    if not contexts:
        return TimeframeDecision(None, 0.0, {}, "no_valid_timeframe_context")
    scores: dict[str, float] = {}
    for timeframe, context in contexts.items():
        strength = context["trend_strength"]
        volatility = context["volatility_score"]
        if timeframe == "1h":
            score = 0.20 + 0.25 * (1.0 - strength) + 0.55 * volatility
        elif timeframe == "4h":
            score = 0.15 + 0.55 * strength + 0.20 * volatility
        else:
            score = 0.05 + 0.95 * strength - 0.35 * volatility
        scores[timeframe] = score
    ranked = sorted(scores, key=lambda tf: (-scores[tf], TIMEFRAMES.index(tf)))
    selected = ranked[0]
    margin = scores[selected] - scores[ranked[1]] if len(ranked) > 1 else 0.0
    confidence = min(1.0, max(0.0, 0.50 + margin))
    reason = (
        "selected"
        if confidence >= state.entry_confidence_floor
        else "timeframe_confidence_below_floor"
    )
    return TimeframeDecision(selected, confidence, scores, reason)


def _valid_dynamic_stop_inputs(
    entry_price: float,
    atr: float,
    support_level: float,
    stop_multiple: float,
    config: AdaptiveConfig,
) -> bool:
    values = (entry_price, atr, support_level, stop_multiple)
    if not all(math.isfinite(value) for value in values):
        return False
    if (
        entry_price <= 0
        or atr <= 0
        or support_level <= 0
        or support_level >= entry_price
    ):
        return False
    return (
        config.minimum_stop_atr_multiple
        <= stop_multiple
        <= config.maximum_stop_atr_multiple
    )


def _valid_stop_geometry(
    entry_price: float, stop_loss: float, maximum_distance_fraction: float
) -> bool:
    stop_distance = entry_price - stop_loss
    return (
        stop_loss > 0
        and stop_distance > 0
        and stop_distance / entry_price <= maximum_distance_fraction
    )


def plan_dynamic_stop(
    *,
    entry_price: float,
    atr: float,
    support_level: float,
    state: AdaptiveParameterState,
    config: AdaptiveConfig,
) -> DynamicStopPlan | None:
    """Place a long stop below both the ATR envelope and prior observed support."""
    stop_multiple = state.atr_stop_multiple
    if not _valid_dynamic_stop_inputs(
        entry_price, atr, support_level, stop_multiple, config
    ):
        return None
    atr_stop = entry_price - atr * stop_multiple
    structure_stop = support_level - atr * config.support_buffer_atr
    stop_loss = min(atr_stop, structure_stop)
    if not _valid_stop_geometry(
        entry_price, stop_loss, config.maximum_stop_distance_fraction
    ):
        return None
    return DynamicStopPlan(
        stop_loss=stop_loss,
        stop_distance=entry_price - stop_loss,
        support_level=support_level,
        atr_stop_multiple=stop_multiple,
    )


def _valid_position_inputs(
    equity: float,
    entry_price: float,
    stop_loss: float,
    confidence: float,
    trend_strength: float,
    volatility_score: float,
) -> bool:
    values = (
        equity,
        entry_price,
        stop_loss,
        confidence,
        trend_strength,
        volatility_score,
    )
    if not all(math.isfinite(value) for value in values):
        return False
    return (
        equity > 0
        and entry_price > 0
        and 0 < stop_loss < entry_price
        and 0 <= confidence <= 1
        and 0 <= trend_strength <= 1
        and 0 <= volatility_score <= 1
    )


def _adaptive_risk_fraction(
    confidence: float,
    trend_strength: float,
    volatility_score: float,
    config: AdaptiveConfig,
) -> float:
    risk_strength = confidence * trend_strength * (1.0 - 0.5 * volatility_score)
    requested = (
        config.minimum_risk_fraction
        + (config.maximum_risk_fraction - config.minimum_risk_fraction) * risk_strength
    )
    return min(
        config.maximum_risk_fraction,
        max(config.minimum_risk_fraction, requested),
    )


def _risk_limited_quantity(
    equity: float,
    entry_price: float,
    stop_distance: float,
    risk_budget: float,
    config: AdaptiveConfig,
) -> float | None:
    quantity_by_risk = risk_budget / stop_distance
    quantity_by_notional = equity * config.maximum_position_fraction / entry_price
    quantity = min(quantity_by_risk, quantity_by_notional)
    if not math.isfinite(quantity) or quantity <= 0:
        return None
    return quantity


def size_adaptive_position(
    *,
    equity: float,
    entry_price: float,
    stop_loss: float,
    confidence: float,
    trend_strength: float,
    volatility_score: float,
    config: AdaptiveConfig,
) -> AdaptivePositionPlan | None:
    """Scale risk budget with confidence/trend and reduce it in high volatility."""
    if not _valid_position_inputs(
        equity, entry_price, stop_loss, confidence, trend_strength, volatility_score
    ):
        return None
    risk_fraction = _adaptive_risk_fraction(
        confidence, trend_strength, volatility_score, config
    )
    risk_budget = equity * risk_fraction
    stop_distance = entry_price - stop_loss
    quantity = _risk_limited_quantity(
        equity, entry_price, stop_distance, risk_budget, config
    )
    if quantity is None:
        return None
    return AdaptivePositionPlan(
        quantity=quantity,
        notional=quantity * entry_price,
        risk_budget=risk_budget,
        risk_fraction=risk_fraction,
    )


def update_intelligent_trailing_stop(
    *,
    current_stop: float,
    high_watermark: float,
    current_close: float,
    atr: float,
    support_level: float,
    trend_strength: float,
    macd_histogram: float,
    state: AdaptiveParameterState,
    config: AdaptiveConfig,
) -> tuple[float, str] | None:
    """Update after a bar closes; a strong trend gets more room, a weakening one less."""
    values = (
        current_stop,
        high_watermark,
        current_close,
        atr,
        support_level,
        trend_strength,
        macd_histogram,
        state.trailing_atr_multiple,
    )
    if not all(math.isfinite(value) for value in values):
        return None
    if (
        min(current_stop, high_watermark, current_close, atr, support_level) <= 0
        or current_stop >= current_close
        or not 0 <= trend_strength <= 1
        or not config.minimum_trailing_atr_multiple
        <= state.trailing_atr_multiple
        <= config.maximum_trailing_atr_multiple
    ):
        return None
    strong = trend_strength >= config.strong_trend_threshold and macd_histogram > 0
    weak_adjustment = (
        config.weak_trailing_atr_multiple
        + state.trailing_atr_multiple
        - config.initial_trailing_atr_multiple
    )
    atr_multiple = (
        max(state.trailing_atr_multiple, config.strong_trailing_atr_multiple)
        if strong
        else _bounded(
            weak_adjustment,
            config.minimum_trailing_atr_multiple,
            config.maximum_trailing_atr_multiple,
        )
    )
    candidate = max(
        high_watermark - atr * atr_multiple,
        support_level - atr * config.support_buffer_atr,
    )
    epsilon = max(current_close * 1e-12, 1e-12)
    candidate = min(candidate, current_close - epsilon)
    updated = max(current_stop, candidate)
    return updated, "TREND_RUNNER" if strong else "TIGHTEN_TRAIL"


def _bounded(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def _classify_closed_trade(
    pnl: float,
    stop_atr: float,
    favorable_atr: float,
    exit_reason: str,
    entry_regime: str,
    exit_regime: str,
) -> tuple[str, list[str]]:
    if pnl > 0:
        return "profitable_trade", ["net_pnl_positive_after_assumed_costs"]
    if (
        entry_regime != exit_regime
        and entry_regime != "unknown"
        and exit_regime != "unknown"
    ):
        return "market_state_shift", ["entry_and_exit_market_states_differ"]
    if exit_reason == "trailing_stop" and stop_atr < 2.5 and favorable_atr >= 1.0:
        return "stop_too_tight", [
            "initial_stop_distance_below_2_5_atr",
            "favorable_excursion_at_least_1_atr",
        ]
    if exit_reason == "trend_reversal":
        return "signal_reversal", ["closed_after_confirmed_macd_and_price_reversal"]
    return "signal_or_execution_assumption_failure", [
        "loss_not_explained_by_observed_state_shift_or_tight_stop"
    ]


def _adapt_parameter_values(
    classification: str,
    state: AdaptiveParameterState,
    config: AdaptiveConfig,
) -> tuple[float, float, float, int, bool]:
    next_stop = state.atr_stop_multiple
    next_trailing = state.trailing_atr_multiple
    next_floor = state.entry_confidence_floor
    next_wins = state.consecutive_wins
    updated = False
    if classification == "stop_too_tight":
        next_stop = _bounded(
            next_stop + config.stop_adaptation_step,
            config.minimum_stop_atr_multiple,
            config.maximum_stop_atr_multiple,
        )
        next_trailing = _bounded(
            next_trailing + config.stop_adaptation_step,
            config.minimum_trailing_atr_multiple,
            config.maximum_trailing_atr_multiple,
        )
        next_wins = 0
        updated = (
            next_stop != state.atr_stop_multiple
            or next_trailing != state.trailing_atr_multiple
        )
    elif classification == "market_state_shift":
        next_floor = _bounded(
            next_floor + config.confidence_adaptation_step,
            config.minimum_confidence_floor,
            config.maximum_confidence_floor,
        )
        next_wins = 0
        updated = next_floor != state.entry_confidence_floor
    elif classification in {
        "signal_reversal",
        "signal_or_execution_assumption_failure",
    }:
        next_floor = _bounded(
            next_floor + config.confidence_adaptation_step / 2.0,
            config.minimum_confidence_floor,
            config.maximum_confidence_floor,
        )
        next_wins = 0
        updated = next_floor != state.entry_confidence_floor
    elif classification == "profitable_trade":
        next_wins += 1
        if next_wins >= 3:
            next_floor = _bounded(
                next_floor - config.confidence_adaptation_step / 2.0,
                config.minimum_confidence_floor,
                config.maximum_confidence_floor,
            )
            next_trailing = _bounded(
                next_trailing + config.stop_adaptation_step,
                config.minimum_trailing_atr_multiple,
                config.maximum_trailing_atr_multiple,
            )
            next_wins = 0
            updated = (
                next_floor != state.entry_confidence_floor
                or next_trailing != state.trailing_atr_multiple
            )
    return next_stop, next_trailing, next_floor, next_wins, updated


def analyze_post_trade_failure(
    trade: Mapping[str, object],
    state: AdaptiveParameterState,
    config: AdaptiveConfig,
) -> tuple[dict[str, Any], AdaptiveParameterState]:
    """Classify a closed trade and update bounded parameters for later trades only."""
    pnl = _finite_number(trade.get("net_pnl_quote"))
    stop_atr = _finite_number(trade.get("initial_stop_distance_atr"))
    favorable_atr = _finite_number(trade.get("max_favorable_excursion_atr"))
    if pnl is None or stop_atr is None or favorable_atr is None:
        raise ValueError("adaptive_trade_ledger_numeric_fields_invalid")
    exit_reason = trade.get("exit_reason")
    entry_regime = trade.get("entry_regime")
    exit_regime = trade.get("exit_regime")
    if (
        not isinstance(exit_reason, str)
        or not isinstance(entry_regime, str)
        or not isinstance(exit_regime, str)
    ):
        raise ValueError("adaptive_trade_ledger_categorical_fields_invalid")
    classification, evidence = _classify_closed_trade(
        pnl, stop_atr, favorable_atr, exit_reason, entry_regime, exit_regime
    )
    before = asdict(state)
    next_stop, next_trailing, next_floor, next_wins, updated = _adapt_parameter_values(
        classification, state, config
    )
    new_state = replace(
        state,
        atr_stop_multiple=next_stop,
        trailing_atr_multiple=next_trailing,
        entry_confidence_floor=next_floor,
        revision=state.revision + int(updated),
        trades_observed=state.trades_observed + 1,
        consecutive_wins=next_wins,
    )
    after = asdict(new_state)
    return (
        {
            "classification": classification,
            "evidence": evidence,
            "parameter_update_applied": updated,
            "parameter_state_before": before,
            "parameter_state_after": after,
        },
        new_state,
    )


def _validated_trade_metric_values(
    trades: list[dict[str, Any]],
) -> tuple[list[float], list[float]]:
    returns = [float(trade["net_return_bps"]) for trade in trades]
    pnl_values = [float(trade["net_pnl_quote"]) for trade in trades]
    if any(not math.isfinite(value) for value in returns + pnl_values):
        raise ValueError("adaptive_metrics_non_finite_trade")
    return returns, pnl_values


def _equity_drawdown_metrics(
    pnl_values: list[float], initial_equity: float
) -> tuple[float, float]:
    equity = initial_equity
    peak = initial_equity
    max_drawdown = 0.0
    for pnl in pnl_values:
        equity = max(0.0, equity + pnl)
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
    total_return = (equity / initial_equity - 1.0) * 100.0
    return total_return, max_drawdown * 100.0


def _win_rate_percent(pnl_values: list[float]) -> float | None:
    if not pnl_values:
        return None
    return sum(value > 0 for value in pnl_values) / len(pnl_values) * 100.0


def _expectancy_bps(returns: list[float]) -> float | None:
    return sum(returns) / len(returns) if returns else None


def summarize_adaptive_metrics(
    trades: list[dict[str, Any]], *, initial_equity: float
) -> dict[str, Any]:
    """Summarize closed trades; absent samples remain null rather than zero."""
    if not math.isfinite(initial_equity) or initial_equity <= 0:
        raise ValueError("adaptive_metrics_initial_equity_invalid")
    returns, pnl_values = _validated_trade_metric_values(trades)
    total_return, max_drawdown = _equity_drawdown_metrics(pnl_values, initial_equity)
    expectancy = _expectancy_bps(returns)
    return {
        "status": "METRICS_AVAILABLE" if trades else "NO_TRADES",
        "trade_count": len(trades),
        "win_rate_pct": _win_rate_percent(pnl_values),
        "expectancy_bps_per_trade": expectancy,
        "total_net_pnl_quote": sum(pnl_values),
        "total_return_pct": total_return,
        "max_drawdown_pct": max_drawdown,
        "average_net_return_bps": expectancy,
    }


__all__ = [
    "AdaptiveConfig",
    "AdaptiveParameterState",
    "AdaptivePositionPlan",
    "DynamicStopPlan",
    "POLICY_SCHEMA",
    "TIMEFRAMES",
    "TimeframeDecision",
    "analyze_post_trade_failure",
    "build_multitimeframe_contexts",
    "initial_parameter_state",
    "load_adaptive_config",
    "plan_dynamic_stop",
    "select_adaptive_timeframe",
    "size_adaptive_position",
    "summarize_adaptive_metrics",
    "update_intelligent_trailing_stop",
]
