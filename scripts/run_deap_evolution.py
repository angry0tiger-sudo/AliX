"""Bounded, deterministic DEAP evolution on verified public 1m market data.

Research only: closed-bar signals execute no earlier than the next bar open.
The final 20% of history is excluded before feature construction; no order API
or production champion is reachable from this experiment.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, MutableSequence, Sequence

import numpy as np
import pandas as pd

from services.ai_core.self_evolving_genome import (
    ExpressionGene,
    IndicatorGene,
    StrategyGenome,
    TimeframeGene,
    genome_sha256,
)

# Search spans intraday to monthly bars; all fitness calculations remain offline.
TIMEFRAMES = (1, 5, 15, 30, 60, 120, 240, 360, 720, 1440, 10080, 43200)
LOOKBACKS = (5, 8, 13, 21, 34, 55, 89, 144)
FASTS = (3, 5, 8, 13, 21, 34)
SLOWS = (8, 13, 21, 34, 55, 89, 144, 233)
INDICATORS = ("momentum", "ema_gap", "rsi_reversal", "donchian_breakout")
MOM_ENTRY = (0.0005, 0.001, 0.002, 0.003, 0.005, 0.008)
EMA_ENTRY = (0.0001, 0.0005, 0.001, 0.002, 0.003, 0.005)
BREAKOUT_ENTRY = (0.0, 0.0005, 0.001, 0.002, 0.003, 0.005)
DIRECTIONAL_EXIT = (-0.01, -0.005, -0.003, -0.001, 0.0, 0.001)
RSI_ENTRY = (20.0, 25.0, 30.0, 35.0, 40.0, 45.0)
RSI_EXIT = (55.0, 60.0, 65.0, 70.0, 75.0, 80.0)
EXPOSURES = (0.25, 0.50, 0.75, 0.95)
STOPS = (0.005, 0.01, 0.015, 0.02, 0.03, 0.05)
TRAILING_ACTIVATION = (0.0, 0.01, 0.02, 0.03, 0.05)
EXIT_MODES = (
    "indicator_threshold",
    "indicator_or_ema_cross",
    "indicator_or_close_below_ema",
    "indicator_and_ema_cross",
)
LOWER = (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0)
UPPER = (11, 3, 7, 5, 7, 5, 5, 3, 5, 4, 3)
GENOME_SIZE = 11
FEE = 25.0 / 10_000
SLIPPAGE = 20.0 / 10_000
SIDE_COST = FEE + SLIPPAGE
MAX_DRAWDOWN = 0.15
DRAWDOWN_EXCESS_PENALTY_MULTIPLIER = 10.0
REQUIRED_TIMEFRAMES = (15, 60, 240, 1440)
MIN_REQUIRED_TIMEFRAME_POPULATION = 4
OPPORTUNITY_MOVE_THRESHOLD = 0.05
OPPORTUNITY_COST_WEIGHT = 2.0
OPPORTUNITY_EVENT_PENALTY = 0.05
PROFIT_FACTOR_WEIGHT = 0.75
TOTAL_RETURN_WEIGHT = 2.0
REGIME_CAPTURE_WEIGHT = 0.5
MAX_PROFIT_FACTOR_FOR_FITNESS = 5.0
NO_TRADE_PENALTY = 0.15
MANIFEST_SCHEMA = "alix.self-evolving-data-manifest.v1"


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    trade_count: int
    win_rate: float
    net_expectancy: float | None
    net_return: float
    max_drawdown: float
    fee_bps_per_side: float = 25.0
    slippage_bps_per_side: float = 20.0
    profit_factor: float | None = None
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    missed_upside_move_count: int = 0
    missed_upside_return: float = 0.0
    high_volatility_return: float = 0.0
    strong_trend_return: float = 0.0


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """Training-scored candidate using soft sample confidence, not hard count limits."""

    training: EvaluationMetrics
    validation: EvaluationMetrics
    fitness: float
    eligibility_reasons: tuple[str, ...]

    @property
    def sample_eligible(self) -> bool:
        return (
            self.training.trade_count > 0 and self.training.net_expectancy is not None
        )

    @property
    def training_risk_eligible(self) -> bool:
        return self.training.max_drawdown <= MAX_DRAWDOWN

    @property
    def selection_eligible(self) -> bool:
        return self.sample_eligible and self.training_risk_eligible


@dataclass(slots=True)
class FeatureFrame:
    frame: pd.DataFrame
    timestamp: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    momentum: dict[int, np.ndarray] = field(default_factory=dict)
    ema_gap: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    rsi: dict[int, np.ndarray] = field(default_factory=dict)
    upper_gap: dict[int, np.ndarray] = field(default_factory=dict)
    lower_gap: dict[int, np.ndarray] = field(default_factory=dict)
    close_ema_gap: dict[int, np.ndarray] = field(default_factory=dict)
    high_volatility: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=bool)
    )
    strong_uptrend: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))


@dataclass(slots=True)
class SearchContext:
    features: dict[int, FeatureFrame]
    train_cutoff_ms: int
    validation_end_ms: int
    positions_cache: dict[tuple[int, ...], np.ndarray] = field(default_factory=dict)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_manifest(manifest: dict[str, Any], csv_path: Path) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("market_data_manifest_schema_invalid")
    safety = manifest.get("safety_invariants", {})
    if safety.get("Champion", "missing") is not None:
        raise ValueError("champion_invariant_invalid")
    if safety.get("order_permission") is not False:
        raise ValueError("order_permission_invariant_invalid")
    if safety.get("trading_authorization") != "NO-GO":
        raise ValueError("trading_authorization_invariant_invalid")
    if manifest.get("interval") != "1m":
        raise ValueError("market_data_interval_must_be_1m")
    if manifest.get("missing_minute_intervals_between_rows") != 0:
        raise ValueError("market_data_has_missing_minute_intervals")
    if float(manifest.get("coverage_fraction", 0.0)) != 1.0:
        raise ValueError("market_data_coverage_incomplete")
    if _sha256_file(csv_path) != manifest.get("csv_sha256"):
        raise ValueError("market_data_sha256_mismatch")


def _validate_market_timestamps(frame: pd.DataFrame) -> None:
    timestamps = frame["open_time_ms"].to_numpy(dtype=np.int64)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("market_data_timestamps_not_strictly_increasing")
    close_times = frame["close_time_ms"].to_numpy(dtype=np.int64)
    if np.any(close_times < timestamps):
        raise ValueError("market_data_close_before_open")


def _validate_market_numeric_fields(frame: pd.DataFrame) -> None:
    prices = frame[["open", "high", "low", "close"]].to_numpy(dtype=float)
    volume = frame["volume"].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or not np.isfinite(volume).all():
        raise ValueError("market_data_contains_non_finite_values")
    if np.any(prices <= 0) or np.any(volume < 0):
        raise ValueError("market_data_contains_invalid_price_or_volume")


def _validate_market_price_geometry(frame: pd.DataFrame) -> None:
    prices = frame[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if np.any(prices[:, 1] < prices.max(axis=1)):
        raise ValueError("market_data_high_inconsistent")
    if np.any(prices[:, 2] > prices.min(axis=1)):
        raise ValueError("market_data_low_inconsistent")


def _validate_market_frame(frame: pd.DataFrame, manifest: dict[str, Any]) -> None:
    row_count = len(frame)
    if row_count != int(manifest.get("row_count", -1)):
        raise ValueError("market_data_row_count_mismatch")
    if row_count < 50_000:
        raise ValueError("market_data_history_too_short")
    _validate_market_timestamps(frame)
    _validate_market_numeric_fields(frame)
    _validate_market_price_geometry(frame)


def load_verified_market_data(
    csv_path: Path, manifest_path: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Verify immutable source identity and structural safety contracts."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest(manifest, csv_path)
    frame = pd.read_csv(
        csv_path,
        usecols=[
            "open_time_ms",
            "close_time_ms",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ],
    )
    _validate_market_frame(frame, manifest)
    return frame, manifest


def _resample(data: pd.DataFrame, timeframe: int) -> pd.DataFrame:
    indexed = data.copy()
    indexed.index = pd.to_datetime(indexed["open_time_ms"], unit="ms", utc=True)
    fields = ["open", "high", "low", "close", "volume"]
    if timeframe == 1:
        bars = indexed[fields].copy()
    else:
        grouped = indexed.resample(
            f"{timeframe}min", origin="epoch", closed="left", label="left"
        )
        bars = grouped.agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        bars = bars.loc[grouped["close"].count() == timeframe]
    bars = bars.dropna(subset=fields)
    bars["open_time_ms"] = (
        np.asarray(bars.index, dtype="datetime64[ns]").astype(np.int64) // 1_000_000
    )
    bars["close_time_ms"] = bars["open_time_ms"] + timeframe * 60_000 - 1
    return bars


def _calc_rsi(close: pd.Series, window: int) -> np.ndarray:
    delta = close.diff()
    gains = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    losses = -delta.clip(upper=0).rolling(window, min_periods=window).mean()
    rsi = 100.0 - 100.0 / (1.0 + gains / losses.replace(0, np.nan))
    valid = gains.notna() & losses.notna()
    no_loss = valid & losses.eq(0)
    flat = no_loss & gains.eq(0)
    rsi = rsi.where(~no_loss, 100.0).where(~flat, 50.0)
    return rsi.where(valid).to_numpy(dtype=np.float64)


def _build_features(bars: pd.DataFrame) -> FeatureFrame:
    close = bars["close"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    result = FeatureFrame(
        frame=bars,
        # A resampled candle is not observable until its close, not its open label.
        # Splits and decision eligibility must use this availability timestamp.
        timestamp=bars["close_time_ms"].to_numpy(dtype=np.int64),
        open=bars["open"].to_numpy(dtype=np.float64),
        high=bars["high"].to_numpy(dtype=np.float64),
        low=bars["low"].to_numpy(dtype=np.float64),
        close=close.to_numpy(dtype=np.float64),
    )
    timeframe = int(
        (bars["close_time_ms"].iloc[-1] - bars["open_time_ms"].iloc[-1] + 1) // 60_000
    )
    result.high_volatility, result.strong_uptrend = _build_regime_masks(
        close, timeframe
    )
    for window in LOOKBACKS:
        result.momentum[window] = (close / close.shift(window) - 1.0).to_numpy(
            dtype=np.float64
        )
        upper = high.shift(1).rolling(window, min_periods=window).max()
        lower = low.shift(1).rolling(window, min_periods=window).min()
        result.upper_gap[window] = (close / upper - 1.0).to_numpy(dtype=np.float64)
        result.lower_gap[window] = (close / lower - 1.0).to_numpy(dtype=np.float64)
        result.rsi[window] = _calc_rsi(close, window)
    ema = {
        window: close.ewm(span=window, adjust=False, min_periods=window).mean()
        for window in set(FASTS + SLOWS)
    }
    for fast in FASTS:
        for slow in SLOWS:
            if fast < slow:
                result.ema_gap[(fast, slow)] = (ema[fast] / ema[slow] - 1.0).to_numpy(
                    dtype=np.float64
                )
    for slow in SLOWS:
        result.close_ema_gap[slow] = (close / ema[slow] - 1.0).to_numpy(
            dtype=np.float64
        )
    return result


def build_search_context(
    raw: pd.DataFrame, train_cutoff_ms: int, validation_end_ms: int
) -> SearchContext:
    """Precompute causal features only through the validation window."""

    research = raw.loc[raw["open_time_ms"] < validation_end_ms].copy()
    features = {
        timeframe: _build_features(_resample(research, timeframe))
        for timeframe in TIMEFRAMES
    }
    if any(len(item.frame) < 2 for item in features.values()):
        raise ValueError("insufficient_resampled_history")
    return SearchContext(features, train_cutoff_ms, validation_end_ms)


def repair_individual(individual: MutableSequence[int]) -> MutableSequence[int]:
    """Bound genes and canonicalize unused loci so equivalent strategies dedupe."""

    if len(individual) != GENOME_SIZE:
        raise ValueError("genome_length_invalid")
    for index, (lower, upper) in enumerate(zip(LOWER, UPPER)):
        individual[index] = max(lower, min(upper, int(individual[index])))
    if individual[1] == 1:
        individual[2] = 0
        fast_period = FASTS[individual[3]]
        minimum_slow = next(
            (idx for idx, period in enumerate(SLOWS) if period > fast_period),
            len(SLOWS) - 1,
        )
        individual[4] = max(individual[4], minimum_slow)
    else:
        individual[3:5] = [0, 1]
    return individual


def _thresholds(genes: Sequence[int]) -> tuple[float, float]:
    mode = genes[1]
    entry_index, exit_index = genes[5], genes[6]
    if mode == 0:
        entry = MOM_ENTRY[entry_index]
        return entry, min(DIRECTIONAL_EXIT[exit_index], entry - 1e-6)
    if mode == 1:
        entry = EMA_ENTRY[entry_index]
        return entry, min(DIRECTIONAL_EXIT[exit_index], entry - 1e-6)
    if mode == 2:
        return RSI_ENTRY[entry_index], RSI_EXIT[exit_index]
    return BREAKOUT_ENTRY[entry_index], DIRECTIONAL_EXIT[exit_index]


def _signal_masks(
    features: FeatureFrame, genes: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    mode = genes[1]
    window = LOOKBACKS[genes[2]]
    entry_threshold, exit_threshold = _thresholds(genes)
    if mode == 0:
        value = features.momentum[window]
        entry, indicator_exit = value >= entry_threshold, value <= exit_threshold
    elif mode == 1:
        value = features.ema_gap[(FASTS[genes[3]], SLOWS[genes[4]])]
        entry, indicator_exit = value >= entry_threshold, value <= exit_threshold
    elif mode == 2:
        value = features.rsi[window]
        entry, indicator_exit = value <= entry_threshold, value >= exit_threshold
    else:
        entry = features.upper_gap[window] >= entry_threshold
        indicator_exit = features.lower_gap[window] <= exit_threshold
    fast, slow = FASTS[genes[3]], SLOWS[genes[4]]
    ema_cross_exit = features.ema_gap[(fast, slow)] <= 0.0
    close_below_ema_exit = features.close_ema_gap[slow] <= 0.0
    exit_mode = genes[10]
    if exit_mode == 0:
        exit_signal = indicator_exit
    elif exit_mode == 1:
        exit_signal = indicator_exit | ema_cross_exit
    elif exit_mode == 2:
        exit_signal = indicator_exit | close_below_ema_exit
    else:
        exit_signal = indicator_exit & ema_cross_exit
    return entry, exit_signal


def _build_regime_masks(
    close: pd.Series, timeframe: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build causal trailing volatility and positive-trend masks."""

    horizon = max(2, min(1440, int(round(1440 / timeframe))))
    returns = close.pct_change()
    min_periods = max(2, horizon // 2)
    rolling_vol = returns.rolling(horizon, min_periods=min_periods).std()
    baseline_window = max(horizon, min(3 * horizon, 10_000))
    rolling_baseline = rolling_vol.rolling(
        baseline_window, min_periods=max(2, horizon)
    ).mean()
    high_volatility = (rolling_vol > rolling_baseline * 1.25).fillna(False)
    trend_return = close / close.shift(horizon) - 1.0
    strong_uptrend = (trend_return >= OPPORTUNITY_MOVE_THRESHOLD).fillna(False)
    return (
        high_volatility.to_numpy(dtype=bool),
        strong_uptrend.to_numpy(dtype=bool),
    )


def _find_stop(
    close: np.ndarray,
    entry: int,
    end: int,
    stop_fraction: float,
    activation_fraction: float,
) -> int | None:
    start = entry + 1
    if start >= end:
        return None
    segment = close[start:end]
    peaks = np.maximum.accumulate(np.r_[close[entry], segment])[1:]
    activated = peaks >= close[entry] * (1.0 + activation_fraction)
    hits = np.flatnonzero(activated & (segment <= peaks * (1.0 - stop_fraction)))
    return int(start + hits[0]) if len(hits) else None


def build_position_states(
    entry_signal: np.ndarray,
    exit_signal: np.ndarray,
    close: np.ndarray,
    trailing_stop: float,
    trailing_activation: float = 0.0,
) -> np.ndarray:
    """Causal signal exit plus profit-activated trailing stop; orders fill next bar."""

    count = len(close)
    if count == 0 or len(entry_signal) != count or len(exit_signal) != count:
        raise ValueError("signal_length_mismatch")
    if not 0.0 <= trailing_stop < 1.0 or not 0.0 <= trailing_activation < 1.0:
        raise ValueError("trailing_stop_parameters_invalid")
    entry_edges = entry_signal & np.r_[True, ~entry_signal[:-1]]
    entry_indexes = np.flatnonzero(entry_edges)
    exit_indexes = np.flatnonzero(exit_signal)
    states = np.zeros(count, dtype=bool)
    cursor = 0
    while cursor < count:
        entry_offset = int(np.searchsorted(entry_indexes, cursor))
        if entry_offset >= len(entry_indexes):
            break
        entry = int(entry_indexes[entry_offset])
        exit_offset = int(np.searchsorted(exit_indexes, entry + 1))
        natural_exit = (
            int(exit_indexes[exit_offset]) if exit_offset < len(exit_indexes) else count
        )
        stop_index = _find_stop(
            close, entry, natural_exit, trailing_stop, trailing_activation
        )
        end = stop_index if stop_index is not None else natural_exit
        states[entry:end] = True
        if end >= count:
            break
        cursor = end + 1
    return states


def _trade_returns(
    opens: np.ndarray, closes: np.ndarray, open_positions: np.ndarray
) -> np.ndarray:
    previous = np.r_[False, open_positions[:-1]]
    entries = np.flatnonzero(open_positions & ~previous)
    exits = np.flatnonzero(~open_positions & previous)
    results: list[float] = []
    for entry in entries:
        exit_offset = int(np.searchsorted(exits, entry + 1))
        if exit_offset < len(exits):
            exit_price = float(opens[int(exits[exit_offset])])
        elif open_positions[-1]:
            exit_price = float(closes[-1])
        else:
            continue
        buy_cost = float(opens[int(entry)]) * (1.0 + SLIPPAGE) * (1.0 + FEE)
        sell_proceeds = exit_price * (1.0 - SLIPPAGE) * (1.0 - FEE)
        results.append(sell_proceeds / buy_cost - 1.0)
    return np.asarray(results, dtype=np.float64)


def _portfolio_metrics(
    opens: np.ndarray,
    closes: np.ndarray,
    open_positions: np.ndarray,
    exposure: float,
) -> tuple[float, float]:
    held = open_positions.astype(np.float64) * exposure
    previous = np.r_[0.0, held[:-1]]
    gap = np.zeros(len(opens), dtype=np.float64)
    gap[1:] = opens[1:] / closes[:-1] - 1.0
    intrabar = closes / opens - 1.0
    turnover = np.abs(held - previous)
    returns = previous * gap + held * intrabar - turnover * SIDE_COST
    if open_positions[-1]:
        returns[-1] -= exposure * SIDE_COST
    equity = np.cumprod(np.maximum(1.0 + returns, 1e-12))
    peaks = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    drawdown = float(np.max((peaks - equity) / peaks)) if len(equity) else 0.0
    return float(equity[-1] - 1.0), drawdown


def _opportunity_cost_metrics(
    closes: np.ndarray, positions: np.ndarray, timeframe_minutes: int
) -> tuple[int, float]:
    """Score missed upside windows as an evaluation metric, never a trading feature."""

    horizon = max(1, int(round(1440 / timeframe_minutes)))
    missed_count = 0
    missed_return = 0.0
    start = 0
    while start + horizon < len(closes):
        end = start + horizon
        move = float(closes[end] / closes[start] - 1.0)
        if move >= OPPORTUNITY_MOVE_THRESHOLD:
            if not np.any(positions[start + 1 : end + 1]):
                missed_count += 1
                missed_return += move
            start = end
        else:
            start += 1
    return missed_count, missed_return


def metrics_from_positions(
    opens: np.ndarray,
    closes: np.ndarray,
    positions_after_close: np.ndarray,
    exposure: float,
    timeframe_minutes: int = 1,
    high_volatility_mask: np.ndarray | None = None,
    strong_uptrend_mask: np.ndarray | None = None,
) -> EvaluationMetrics:
    """Report PF, net equity, missed upside and regime-capture metrics."""

    if (
        len(opens) == 0
        or len(opens) != len(closes)
        or len(opens) != len(positions_after_close)
    ):
        raise ValueError("metric_input_length_invalid")
    open_positions = np.r_[False, positions_after_close[:-1]]
    trade_returns = _trade_returns(opens, closes, open_positions)
    expectancy = float(np.mean(trade_returns)) if len(trade_returns) else None
    wins = float(np.mean(trade_returns > 0.0)) if len(trade_returns) else 0.0
    gross_profit = float(trade_returns[trade_returns > 0.0].sum())
    gross_loss = float(-trade_returns[trade_returns < 0.0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0.0 else None
    net_return, drawdown = _portfolio_metrics(opens, closes, open_positions, exposure)
    missed_count, missed_return = _opportunity_cost_metrics(
        closes, open_positions, timeframe_minutes
    )
    if high_volatility_mask is None:
        high_volatility_mask = np.zeros(len(closes), dtype=bool)
    if strong_uptrend_mask is None:
        strong_uptrend_mask = np.zeros(len(closes), dtype=bool)
    if len(high_volatility_mask) != len(closes) or len(strong_uptrend_mask) != len(
        closes
    ):
        raise ValueError("regime_mask_length_invalid")
    close_returns = np.r_[0.0, np.diff(closes) / closes[:-1]]
    exposure_path = open_positions.astype(np.float64) * exposure
    high_volatility_return = float(
        np.sum(close_returns * exposure_path * high_volatility_mask)
    )
    strong_trend_return = float(
        np.sum(close_returns * exposure_path * strong_uptrend_mask)
    )
    return EvaluationMetrics(
        trade_count=int(len(trade_returns)),
        win_rate=wins,
        net_expectancy=expectancy,
        net_return=net_return,
        max_drawdown=drawdown,
        profit_factor=profit_factor,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        missed_upside_move_count=missed_count,
        missed_upside_return=missed_return,
        high_volatility_return=high_volatility_return,
        strong_trend_return=strong_trend_return,
    )


def _period_metrics(
    context: SearchContext, genes: Sequence[int], period: str
) -> EvaluationMetrics:
    timeframe = TIMEFRAMES[genes[0]]
    features = context.features[timeframe]
    entry, exit_signal = _signal_masks(features, genes)
    if period == "train":
        mask = features.timestamp < context.train_cutoff_ms
    elif period == "validation":
        mask = (features.timestamp >= context.train_cutoff_ms) & (
            features.timestamp < context.validation_end_ms
        )
    else:
        raise ValueError("evaluation_period_invalid")
    indexes = np.flatnonzero(mask)
    if len(indexes) < 2:
        return EvaluationMetrics(0, 0.0, None, 0.0, 0.0)
    position_key = (timeframe, 0 if period == "train" else 1, *genes[:11])
    cached = context.positions_cache.get(position_key)
    if cached is None:
        cached = build_position_states(
            entry[indexes],
            exit_signal[indexes],
            features.close[indexes],
            STOPS[genes[8]],
            TRAILING_ACTIVATION[genes[9]],
        )
        context.positions_cache[position_key] = cached
    bars = features.frame.iloc[indexes]
    return metrics_from_positions(
        bars["open"].to_numpy(dtype=np.float64),
        bars["close"].to_numpy(dtype=np.float64),
        cached,
        EXPOSURES[genes[7]],
        timeframe_minutes=timeframe,
        high_volatility_mask=features.high_volatility[indexes],
        strong_uptrend_mask=features.strong_uptrend[indexes],
    )


def _eligibility_reasons(
    training: EvaluationMetrics, validation: EvaluationMetrics
) -> tuple[str, ...]:
    reasons: list[str] = []
    if training.trade_count == 0:
        reasons.append("training_has_no_completed_trades")
    if training.max_drawdown > MAX_DRAWDOWN:
        reasons.append("training_drawdown_above_limit")
    if validation.trade_count == 0:
        reasons.append("validation_has_no_completed_trades")
    if validation.profit_factor is None or validation.profit_factor <= 1.5:
        reasons.append("validation_profit_factor_not_above_1_5")
    if validation.net_return <= 0.0:
        reasons.append("validation_total_return_not_positive")
    return tuple(reasons)


def fitness_score(metrics: EvaluationMetrics) -> float:
    """Bounded opportunity-hunting objective with soft sample confidence."""

    trades = metrics.trade_count
    confidence = math.sqrt(trades / (trades + 10.0)) if trades > 0 else 0.0
    if metrics.profit_factor is not None:
        pf_for_score = min(MAX_PROFIT_FACTOR_FOR_FITNESS, metrics.profit_factor)
    elif metrics.gross_profit > 0.0 and metrics.gross_loss == 0.0:
        pf_for_score = MAX_PROFIT_FACTOR_FOR_FITNESS
    else:
        pf_for_score = 0.0
    pf_component = PROFIT_FACTOR_WEIGHT * math.log1p(pf_for_score)
    return_component = TOTAL_RETURN_WEIGHT * metrics.net_return
    regime_bonus = REGIME_CAPTURE_WEIGHT * (
        max(0.0, metrics.high_volatility_return) + max(0.0, metrics.strong_trend_return)
    )
    opportunity_penalty = (
        OPPORTUNITY_COST_WEIGHT * metrics.missed_upside_return
        + OPPORTUNITY_EVENT_PENALTY * metrics.missed_upside_move_count
    )
    inactivity_penalty = NO_TRADE_PENALTY if trades == 0 else 0.0
    score = (
        confidence * (pf_component + return_component)
        + regime_bonus
        - opportunity_penalty
        - inactivity_penalty
    )
    drawdown_excess = max(0.0, metrics.max_drawdown - MAX_DRAWDOWN)
    if drawdown_excess > 0.0:
        score = min(score, 0.0) - DRAWDOWN_EXCESS_PENALTY_MULTIPLIER * drawdown_excess
    return float(score)


def _evaluate_individual(
    individual: MutableSequence[int],
    context: SearchContext,
    cache: dict[tuple[int, ...], CandidateEvaluation],
) -> tuple[float]:
    repair_individual(individual)
    key = tuple(individual)
    if key not in cache:
        training = _period_metrics(context, individual, "train")
        validation = _period_metrics(context, individual, "validation")
        cache[key] = CandidateEvaluation(
            training=training,
            validation=validation,
            fitness=fitness_score(training),
            eligibility_reasons=_eligibility_reasons(training, validation),
        )
    return (cache[key].fitness,)


def _selection_priority(
    individual: Sequence[int], cache: dict[tuple[int, ...], CandidateEvaluation]
) -> tuple[int, int, float]:
    result = cache.get(tuple(individual))
    if result is None:
        return (0, 0, float("-inf"))
    # Eligibility precedes the numeric score: an invalid zero cannot outrank a
    # valid candidate whose expectancy-based fitness happens to be negative.
    return (
        int(result.sample_eligible),
        int(result.training_risk_eligible),
        float(result.fitness),
    )


def _select_by_eligibility(
    individuals: Sequence[Any],
    k: int,
    cache: dict[tuple[int, ...], CandidateEvaluation],
    tournsize: int = 3,
) -> list[Any]:
    if not individuals or k < 0:
        raise ValueError("selection_population_invalid")
    selected: list[Any] = []
    for _ in range(k):
        aspirants = random.sample(individuals, min(tournsize, len(individuals)))
        selected.append(
            max(aspirants, key=lambda item: _selection_priority(item, cache))
        )
    return selected


def _random_individual() -> MutableSequence[int]:
    genes = [random.randint(low, high) for low, high in zip(LOWER, UPPER)]
    return repair_individual(genes)


def _build_toolbox(
    context: SearchContext,
    cache: dict[tuple[int, ...], CandidateEvaluation],
    seed: int,
) -> Any:
    try:
        from deap import base, creator, tools  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("deap_research_dependency_missing") from exc
    random.seed(seed)
    if getattr(creator, "FitnessAlixNetExpectancy", None) is None:
        creator.create("FitnessAlixNetExpectancy", base.Fitness, weights=(1.0,))
    fitness_type = getattr(creator, "FitnessAlixNetExpectancy")
    if getattr(creator, "AlixResearchIndividual", None) is None:
        creator.create("AlixResearchIndividual", list, fitness=fitness_type)
    individual_type = getattr(creator, "AlixResearchIndividual")
    toolbox = base.Toolbox()
    toolbox.register(
        "individual", tools.initIterate, individual_type, _random_individual
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", _evaluate_individual, context=context, cache=cache)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=LOWER, up=UPPER, indpb=0.20)
    toolbox.register("select", _select_by_eligibility, cache=cache, tournsize=3)
    toolbox.register("clone", copy.deepcopy)
    return toolbox


def _evaluate_invalid(population: Sequence[Any], toolbox: Any) -> None:
    invalid = [item for item in population if not item.fitness.valid]
    for item, fitness in zip(invalid, toolbox.map(toolbox.evaluate, invalid)):
        item.fitness.values = fitness


def _timeframe_label(minutes: int) -> str:
    labels = {
        1: "1m",
        5: "5m",
        15: "15m",
        30: "30m",
        60: "1h",
        120: "2h",
        240: "4h",
        360: "6h",
        720: "12h",
        1440: "1d",
        10080: "7d",
        43200: "30d",
    }
    return labels.get(minutes, f"{minutes}m")


def _unique_timeframe_count(population: Sequence[Any], timeframe_gene: int) -> int:
    return len({tuple(item) for item in population if item[0] == timeframe_gene})


def _ensure_timeframe_coverage(
    population: list[Any],
    toolbox: Any,
    cache: dict[tuple[int, ...], CandidateEvaluation],
) -> list[Any]:
    """Maintain a unique-genome quota for 1h, 4h and 1d each generation."""

    for required_minutes in REQUIRED_TIMEFRAMES:
        required_gene = TIMEFRAMES.index(required_minutes)
        while (
            _unique_timeframe_count(population, required_gene)
            < MIN_REQUIRED_TIMEFRAME_POPULATION
        ):
            replaceable = _replaceable_population_members(population)
            if not replaceable:
                raise RuntimeError("required_timeframe_quota_cannot_be_satisfied")
            index, _ = min(
                replaceable, key=lambda pair: _selection_priority(pair[1], cache)
            )
            population_keys = {
                tuple(item)
                for item_index, item in enumerate(population)
                if item_index != index
            }
            replacement = _make_unique_timeframe_candidate(
                toolbox, required_gene, population_keys, cache
            )
            population[index] = replacement
    return population


def _replaceable_population_members(
    population: Sequence[Any],
) -> list[tuple[int, Any]]:
    genome_frequency = Counter(tuple(item) for item in population)
    unique_by_timeframe = {
        minutes: {tuple(item) for item in population if TIMEFRAMES[item[0]] == minutes}
        for minutes in REQUIRED_TIMEFRAMES
    }
    return [
        (index, item)
        for index, item in enumerate(population)
        if (
            TIMEFRAMES[item[0]] not in REQUIRED_TIMEFRAMES
            or genome_frequency[tuple(item)] > 1
            or len(unique_by_timeframe[TIMEFRAMES[item[0]]])
            > MIN_REQUIRED_TIMEFRAME_POPULATION
        )
    ]


def _make_unique_timeframe_candidate(
    toolbox: Any,
    required_gene: int,
    population_keys: set[tuple[int, ...]],
    cache: dict[tuple[int, ...], CandidateEvaluation],
) -> Any:
    for _ in range(10_000):
        candidate = toolbox.individual()
        candidate[0] = required_gene
        repair_individual(candidate)
        key = tuple(candidate)
        if key not in cache and key not in population_keys:
            return candidate
    raise RuntimeError("required_timeframe_unique_genome_generation_failed")


def _generation_record(
    generation: int,
    population: Sequence[Any],
    cache: dict[tuple[int, ...], CandidateEvaluation],
) -> dict[str, int | float]:
    scores = np.asarray(
        [float(item.fitness.values[0]) for item in population], dtype=np.float64
    )
    record: dict[str, int | float] = {
        "generation": generation,
        "population_size": len(population),
        "best_fitness": float(scores.max()),
        "mean_fitness": float(scores.mean()),
        "median_fitness": float(np.median(scores)),
        "unique_genomes_evaluated": len(cache),
    }
    for required_minutes in REQUIRED_TIMEFRAMES:
        label = _timeframe_label(required_minutes)
        required_gene = TIMEFRAMES.index(required_minutes)
        members = [item for item in population if item[0] == required_gene]
        results = [cache.get(tuple(item)) for item in members]
        record[f"population_{label}"] = len(members)
        record[f"sample_eligible_{label}"] = sum(
            1 for item in results if item is not None and item.sample_eligible
        )
        record[f"selection_eligible_{label}"] = sum(
            1 for item in results if item is not None and item.selection_eligible
        )
    return record


def _timeframe_coverage_summary(
    cache: dict[tuple[int, ...], CandidateEvaluation],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for minutes in TIMEFRAMES:
        label = _timeframe_label(minutes)
        entries = [
            (key, value)
            for key, value in cache.items()
            if TIMEFRAMES[key[0]] == minutes
        ]
        sample_eligible = [
            (key, value) for key, value in entries if value.sample_eligible
        ]
        selection_eligible = [
            (key, value) for key, value in entries if value.selection_eligible
        ]
        best = max(selection_eligible, key=lambda item: item[1].fitness, default=None)
        rejection_counts: Counter[str] = Counter()
        for _, value in entries:
            if value.training.trade_count == 0:
                rejection_counts["training_has_no_completed_trades"] += 1
            if value.training.max_drawdown > MAX_DRAWDOWN:
                rejection_counts["training_drawdown_above_limit"] += 1
            if value.validation.trade_count == 0:
                rejection_counts["validation_has_no_completed_trades"] += 1
            if (
                value.validation.profit_factor is None
                or value.validation.profit_factor <= 1.5
            ):
                rejection_counts["validation_profit_factor_not_above_1_5"] += 1
        summary[label] = {
            "timeframe_minutes": minutes,
            "required": minutes in REQUIRED_TIMEFRAMES,
            "unique_genomes_evaluated": len(entries),
            "sample_eligible_genomes": len(sample_eligible),
            "selection_eligible_genomes": len(selection_eligible),
            "best_valid_fitness": best[1].fitness if best else None,
            "best_valid_training_trade_count": best[1].training.trade_count
            if best
            else None,
            "best_valid_validation_trade_count": best[1].validation.trade_count
            if best
            else None,
            "best_valid_training_profit_factor": best[1].training.profit_factor
            if best
            else None,
            "best_valid_validation_profit_factor": best[1].validation.profit_factor
            if best
            else None,
            "best_valid_training_net_return": best[1].training.net_return
            if best
            else None,
            "best_valid_validation_net_return": best[1].validation.net_return
            if best
            else None,
            "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        }
    return summary


def _next_generation(
    population: Sequence[Any], toolbox: Any, crossover: float, mutation: float
) -> list[Any]:
    children = [
        toolbox.clone(item) for item in toolbox.select(population, len(population))
    ]
    for index in range(1, len(children), 2):
        if random.random() < crossover:
            toolbox.mate(children[index - 1], children[index])
            repair_individual(children[index - 1])
            repair_individual(children[index])
            del children[index - 1].fitness.values
            del children[index].fitness.values
    for child in children:
        if random.random() < mutation:
            toolbox.mutate(child)
            repair_individual(child)
            del child.fitness.values
    return children


def _genome_exit_rule(
    mode: str, exit_threshold: float, exit_mode: str
) -> ExpressionGene:
    indicator_operator = "greater_than" if mode == "rsi_reversal" else "less_than"
    indicator_exit = ExpressionGene(
        indicator_operator,
        children=(
            ExpressionGene("feature", feature=f"{mode}_exit"),
            ExpressionGene("constant", constant=exit_threshold),
        ),
    )
    ema_exit = ExpressionGene(
        "less_than",
        children=(
            ExpressionGene("feature", feature="ema_trend_gap"),
            ExpressionGene("constant", constant=0.0),
        ),
    )
    close_ema_exit = ExpressionGene(
        "less_than",
        children=(
            ExpressionGene("feature", feature="close_vs_trend_ema"),
            ExpressionGene("constant", constant=0.0),
        ),
    )
    if exit_mode == "indicator_or_ema_cross":
        return ExpressionGene("or", children=(indicator_exit, ema_exit))
    if exit_mode == "indicator_or_close_below_ema":
        return ExpressionGene("or", children=(indicator_exit, close_ema_exit))
    if exit_mode == "indicator_and_ema_cross":
        return ExpressionGene("and", children=(indicator_exit, ema_exit))
    return indicator_exit


def _strategy_genome(genes: Sequence[int]) -> StrategyGenome:
    timeframe_minutes = TIMEFRAMES[genes[0]]
    timeframe = TimeframeGene(timeframe_minutes, "minute")
    mode = INDICATORS[genes[1]]
    window = LOOKBACKS[genes[2]]
    fast = FASTS[genes[3]]
    slow = SLOWS[genes[4]]
    entry, exit_threshold = _thresholds(genes)
    stop = STOPS[genes[8]]
    activation = TRAILING_ACTIVATION[genes[9]]
    exit_mode = EXIT_MODES[genes[10]]
    parameters = (float(window), float(fast), float(slow), entry, exit_threshold)
    indicators = (
        IndicatorGene("entry_signal", mode, parameters, timeframe),
        IndicatorGene(f"{mode}_exit", f"{mode}_exit", parameters, timeframe),
        IndicatorGene(
            "ema_trend_gap", "ema_trend_gap", (float(fast), float(slow)), timeframe
        ),
        IndicatorGene("close_vs_trend_ema", "close_vs_ema", (float(slow),), timeframe),
        IndicatorGene(
            "trailing_drawdown",
            "activated_close_peak_drawdown",
            (stop, activation),
            timeframe,
        ),
    )
    entry_operator = "less_than" if mode == "rsi_reversal" else "greater_than"
    return StrategyGenome(
        genome_id="research_candidate_001",
        timeframe=timeframe,
        indicators=indicators,
        entry_rule=ExpressionGene(
            entry_operator,
            children=(
                ExpressionGene("feature", feature="entry_signal"),
                ExpressionGene("constant", constant=entry),
            ),
        ),
        exit_rule=_genome_exit_rule(mode, exit_threshold, exit_mode),
        position_size_rule=ExpressionGene("constant", constant=EXPOSURES[genes[7]]),
        stop_rule=ExpressionGene(
            "activated_trailing_stop",
            children=(
                ExpressionGene("feature", feature="trailing_drawdown"),
                ExpressionGene("constant", constant=stop),
                ExpressionGene("constant", constant=activation),
            ),
        ),
    )


def _condition_text(genes: Sequence[int]) -> tuple[str, str]:
    mode = INDICATORS[genes[1]]
    window = LOOKBACKS[genes[2]]
    entry, exit_threshold = _thresholds(genes)
    if mode == "momentum":
        entry_text = f"{window}-bar return >= {entry:.6f}"
    elif mode == "ema_gap":
        fast, slow = FASTS[genes[3]], SLOWS[genes[4]]
        entry_text = f"EMA({fast})/EMA({slow}) - 1 >= {entry:.6f}"
    elif mode == "rsi_reversal":
        entry_text = f"RSI({window}) <= {entry:.1f}"
    else:
        entry_text = f"close/prior_{window}_bar_high - 1 >= {entry:.6f}"
    exit_text = (
        f"{EXIT_MODES[genes[10]]}; indicator threshold {exit_threshold:.6f}; "
        f"trailing stop {STOPS[genes[8]]:.3%} activates after "
        f"{TRAILING_ACTIVATION[genes[9]]:.3%} favorable move"
    )
    return entry_text, exit_text


def _candidate_payload(
    genes: Sequence[int], score: float, train: EvaluationMetrics
) -> dict[str, Any]:
    genome = _strategy_genome(genes)
    entry_text, exit_text = _condition_text(genes)
    return {
        "candidate_status": "RESEARCH_ONLY",
        "research_genome_sha256": genome_sha256(genome),
        "genome": asdict(genome),
        "genes": {
            "timeframe_minutes": TIMEFRAMES[genes[0]],
            "indicator_family": INDICATORS[genes[1]],
            "lookback_bars": LOOKBACKS[genes[2]],
            "ema_fast_period": FASTS[genes[3]] if genes[1] == 1 else None,
            "ema_slow_period": SLOWS[genes[4]] if genes[1] == 1 else None,
            "entry_threshold": _thresholds(genes)[0],
            "exit_threshold": _thresholds(genes)[1],
            "position_exposure": EXPOSURES[genes[7]],
            "trailing_stop_fraction": STOPS[genes[8]],
            "trailing_activation_fraction": TRAILING_ACTIVATION[genes[9]],
            "dynamic_exit_mode": EXIT_MODES[genes[10]],
        },
        "entry_condition": entry_text,
        "exit_condition": exit_text,
        "fitness": score,
        "fitness_objective": "profit_factor + net_total_return + regime_capture - missed_upside_opportunity_cost",
        "training_metrics": asdict(train),
    }


def _economic_assessment(
    training: EvaluationMetrics | None, validation: EvaluationMetrics | None
) -> dict[str, Any]:
    """Evaluate selected candidate; no hard minimum trade counts are enforced."""

    if training is None or validation is None:
        return {
            "status": "NO-GO",
            "reasons": ["no_candidate_with_training_trades_and_drawdown_within_limit"],
            "minimum_completed_trades_enforced": False,
            "profit_factor_target": 1.5,
            "final_holdout_used": False,
            "final_holdout_reserved": True,
            "owner_approval_required": True,
            "production_promotion_eligible": False,
        }
    reasons: list[str] = []
    if training.trade_count == 0:
        reasons.append("training_has_no_completed_trades")
    if validation.trade_count == 0:
        reasons.append("validation_has_no_completed_trades")
    if validation.profit_factor is None or validation.profit_factor <= 1.5:
        reasons.append("validation_profit_factor_not_above_1_5")
    if validation.net_return <= 0.0:
        reasons.append("validation_total_return_not_positive")
    if training.max_drawdown > MAX_DRAWDOWN:
        reasons.append("training_drawdown_above_limit")
    if validation.max_drawdown > MAX_DRAWDOWN:
        reasons.append("validation_drawdown_above_limit")
    return {
        "status": "NO-GO" if reasons else "OWNER_REVIEW_REQUIRED",
        "reasons": reasons,
        "minimum_completed_trades_enforced": False,
        "profit_factor_target": 1.5,
        "training_trade_count": training.trade_count,
        "validation_trade_count": validation.trade_count,
        "training_profit_factor": training.profit_factor,
        "validation_profit_factor": validation.profit_factor,
        "validation_profit_factor_above_1_5": (
            validation.profit_factor is not None and validation.profit_factor > 1.5
        ),
        "training_total_return": training.net_return,
        "validation_total_return": validation.net_return,
        "training_missed_upside_moves": training.missed_upside_move_count,
        "validation_missed_upside_moves": validation.missed_upside_move_count,
        "training_missed_upside_return": training.missed_upside_return,
        "validation_missed_upside_return": validation.missed_upside_return,
        "final_holdout_used": False,
        "final_holdout_reserved": True,
        "owner_approval_required": True,
        "production_promotion_eligible": False,
    }


def _write_csv_outputs(
    output_dir: Path,
    history: list[dict[str, int | float]],
    ranked_candidates: Sequence[tuple[tuple[int, ...], CandidateEvaluation]],
) -> None:
    with (output_dir / "self-evolving-ga-history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    fields = [
        "rank",
        "gene_vector",
        "timeframe",
        "fitness",
        "eligibility_status",
        "eligibility_reasons",
        "training_trade_count",
        "validation_trade_count",
        "training_win_rate",
        "validation_win_rate",
        "training_net_expectancy",
        "validation_net_expectancy",
        "training_net_return",
        "validation_net_return",
        "training_max_drawdown",
        "validation_max_drawdown",
        "training_profit_factor",
        "validation_profit_factor",
        "training_missed_upside_move_count",
        "validation_missed_upside_move_count",
        "training_missed_upside_return",
        "validation_missed_upside_return",
        "training_high_volatility_return",
        "validation_high_volatility_return",
        "training_strong_trend_return",
        "validation_strong_trend_return",
    ]
    with (output_dir / "self-evolving-ga-candidates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rank, (key, result) in enumerate(ranked_candidates[:100], start=1):
            status = (
                "SELECTION_ELIGIBLE"
                if result.selection_eligible
                else "SAMPLE_ELIGIBLE_HIGH_DRAWDOWN"
                if result.sample_eligible
                else "INVALID_SAMPLE"
            )
            writer.writerow(
                {
                    "rank": rank,
                    "gene_vector": json.dumps(list(key), separators=(",", ":")),
                    "timeframe": _timeframe_label(TIMEFRAMES[key[0]]),
                    "fitness": result.fitness,
                    "eligibility_status": status,
                    "eligibility_reasons": ";".join(result.eligibility_reasons),
                    "training_trade_count": result.training.trade_count,
                    "validation_trade_count": result.validation.trade_count,
                    "training_win_rate": result.training.win_rate,
                    "validation_win_rate": result.validation.win_rate,
                    "training_net_expectancy": result.training.net_expectancy,
                    "validation_net_expectancy": result.validation.net_expectancy,
                    "training_net_return": result.training.net_return,
                    "validation_net_return": result.validation.net_return,
                    "training_max_drawdown": result.training.max_drawdown,
                    "validation_max_drawdown": result.validation.max_drawdown,
                    "training_profit_factor": result.training.profit_factor,
                    "validation_profit_factor": result.validation.profit_factor,
                    "training_missed_upside_move_count": result.training.missed_upside_move_count,
                    "validation_missed_upside_move_count": result.validation.missed_upside_move_count,
                    "training_missed_upside_return": result.training.missed_upside_return,
                    "validation_missed_upside_return": result.validation.missed_upside_return,
                    "training_high_volatility_return": result.training.high_volatility_return,
                    "validation_high_volatility_return": result.validation.high_volatility_return,
                    "training_strong_trend_return": result.training.strong_trend_return,
                    "validation_strong_trend_return": result.validation.strong_trend_return,
                }
            )


def run_evolution(
    csv_path: Path,
    manifest_path: Path,
    output_dir: Path,
    population_size: int = 128,
    generations: int = 10,
    seed: int = 20261025,
) -> dict[str, Any]:
    """Run bounded evolution with explicit sample, risk and timeframe gates."""

    if not 100 <= population_size <= 200:
        raise ValueError("population_size_must_be_between_100_and_200")
    if not 10 <= generations <= 20:
        raise ValueError("generations_must_be_between_10_and_20")
    raw, manifest = load_verified_market_data(csv_path, manifest_path)
    train_cutoff_ms = int(raw["open_time_ms"].iloc[int(len(raw) * 0.70)])
    validation_end_ms = int(raw["open_time_ms"].iloc[int(len(raw) * 0.80)])
    context = build_search_context(raw, train_cutoff_ms, validation_end_ms)
    baseline_genes = [
        TIMEFRAMES.index(10080),
        INDICATORS.index("rsi_reversal"),
        LOOKBACKS.index(8),
        0,
        1,
        RSI_ENTRY.index(35.0),
        RSI_EXIT.index(80.0),
        EXPOSURES.index(0.95),
        STOPS.index(0.05),
        TRAILING_ACTIVATION.index(0.0),
        EXIT_MODES.index("indicator_threshold"),
    ]
    baseline_training = _period_metrics(context, baseline_genes, "train")
    baseline_validation = _period_metrics(context, baseline_genes, "validation")
    cache: dict[tuple[int, ...], CandidateEvaluation] = {}
    toolbox = _build_toolbox(context, cache, seed)

    population = toolbox.population(n=population_size)
    _ensure_timeframe_coverage(population, toolbox, cache)
    history: list[dict[str, int | float]] = []
    for generation in range(1, generations + 1):
        if generation > 1:
            population = _next_generation(population, toolbox, 0.65, 0.35)
            _ensure_timeframe_coverage(population, toolbox, cache)
        _evaluate_invalid(population, toolbox)
        history.append(_generation_record(generation, population, cache))

    ranked_candidates = sorted(
        cache.items(),
        key=lambda item: (
            int(item[1].sample_eligible),
            int(item[1].training_risk_eligible),
            float(item[1].fitness),
        ),
        reverse=True,
    )
    selected = [
        (key, result) for key, result in ranked_candidates if result.selection_eligible
    ]
    if selected:
        best_key, best_result = max(selected, key=lambda item: item[1].fitness)
        candidate = _candidate_payload(
            best_key, best_result.fitness, best_result.training
        )
        candidate["validation_metrics"] = asdict(best_result.validation)
        candidate["validation_return_used_for_selection"] = False
        candidate["validation_sample_count_used_for_eligibility"] = False
        economic_assessment = _economic_assessment(
            best_result.training, best_result.validation
        )
    else:
        candidate = None
        economic_assessment = _economic_assessment(None, None)

    timeframe_coverage = _timeframe_coverage_summary(cache)
    coverage_by_generation = {
        _timeframe_label(minutes): {
            "minimum_population_per_generation": min(
                int(item[f"population_{_timeframe_label(minutes)}"]) for item in history
            ),
            "minimum_sample_eligible_population_per_generation": min(
                int(item[f"sample_eligible_{_timeframe_label(minutes)}"])
                for item in history
            ),
            "generations_with_population": sum(
                1
                for item in history
                if int(item[f"population_{_timeframe_label(minutes)}"]) > 0
            ),
        }
        for minutes in REQUIRED_TIMEFRAMES
    }
    rejected_reasons: Counter[str] = Counter(
        reason
        for evaluation in cache.values()
        for reason in evaluation.eligibility_reasons
    )
    report: dict[str, Any] = {
        "schema_version": "alix.self-evolving-ga-report.v2",
        "run_mode": "historical_simulation_only",
        "data_source": manifest.get("source_venue"),
        "symbol": manifest.get("symbol"),
        "source_interval": "1m",
        "data_window": {
            "start_utc": manifest.get("start_utc"),
            "end_utc": manifest.get("end_utc"),
            "row_count": manifest.get("row_count"),
            "missing_minute_intervals_between_rows": manifest.get(
                "missing_minute_intervals_between_rows"
            ),
            "coverage_fraction": manifest.get("coverage_fraction"),
            "source_csv_sha256": manifest.get("csv_sha256"),
            "manifest_created_at_utc": manifest.get("created_at_utc"),
        },
        "source_csv_sha256": manifest.get("csv_sha256"),
        "population_size": population_size,
        "generations_requested": generations,
        "generations_completed": len(history),
        "random_seed": seed,
        "unique_genomes_evaluated": len(cache),
        "operators": {
            "selection": "eligibility-aware tournament_k3",
            "crossover": "two_point",
            "crossover_probability": 0.65,
            "mutation": "uniform_integer",
            "mutation_probability": 0.35,
            "per_gene_mutation_probability": 0.20,
            "required_timeframe_population_per_generation": MIN_REQUIRED_TIMEFRAME_POPULATION,
        },
        "fitness_definition": {
            "profit_factor_component": f"{PROFIT_FACTOR_WEIGHT} * log1p(min(profit_factor, {MAX_PROFIT_FACTOR_FOR_FITNESS}))",
            "total_return_component": f"{TOTAL_RETURN_WEIGHT} * net_total_return",
            "soft_sample_weight": "sqrt(completed_trade_count / (completed_trade_count + 10)); no hard count threshold",
            "opportunity_cost_penalty": f"{OPPORTUNITY_COST_WEIGHT} * missed_upside_return + {OPPORTUNITY_EVENT_PENALTY} * missed_upside_move_count",
            "opportunity_definition": "non-overlapping 24h window rises at least 5%; candidate has no exposure during entire window",
            "regime_capture_bonus": f"{REGIME_CAPTURE_WEIGHT} * positive_high_volatility_return + positive_strong_trend_return",
            "drawdown_penalty": "if training max_drawdown > 0.15: min(raw_score, 0.0) - 10.0 * (max_drawdown - 0.15)",
            "no_trade_penalty": NO_TRADE_PENALTY,
            "training_metrics_only_for_fitness_and_selection": True,
            "validation_used_for_selection": False,
        },
        "eligibility_gates": {
            "minimum_completed_trade_count_enforced": False,
            "soft_sample_confidence": "sqrt(trade_count / (trade_count + 10))",
            "maximum_drawdown_for_candidate_selection": MAX_DRAWDOWN,
            "profit_factor_target_for_validation": 1.5,
            "trailing_activation_gene": True,
            "dynamic_exit_mode_gene": list(EXIT_MODES),
            "missed_upside_threshold": OPPORTUNITY_MOVE_THRESHOLD,
        },
        "execution_assumptions": {
            "fee_bps_per_side": 25,
            "slippage_bps_per_side": 20,
            "execution": "close-confirmed signal; next-bar-open fill",
            "position_model": "spot long/flat",
            "terminal_liquidation": "final close",
            "shorts_or_leverage_supported": False,
        },
        "split": {
            "training_fraction": 0.70,
            "validation_fraction": 0.10,
            "final_holdout_fraction_reserved": 0.20,
            "train_cutoff_utc": pd.to_datetime(
                train_cutoff_ms, unit="ms", utc=True
            ).isoformat(),
            "validation_end_utc": pd.to_datetime(
                validation_end_ms, unit="ms", utc=True
            ).isoformat(),
            "final_holdout_used": False,
            "final_holdout_metrics_emitted": False,
        },
        "top_research_candidate": candidate,
        "candidate_selection_method": (
            "highest training opportunity-hunter fitness among genomes with at least one "
            "completed training trade and training drawdown <= 15%; validation excluded from selection"
        ),
        "economic_assessment": economic_assessment,
        "research_result": (
            "NO_ECONOMICALLY_ACCEPTABLE_STRATEGY_FOUND"
            if economic_assessment["status"] == "NO-GO"
            else "CANDIDATE_REQUIRES_OWNER_REVIEW"
        ),
        "opportunity_hunter_result": {
            "profit_factor_target": 1.5,
            "validation_profit_factor": economic_assessment.get(
                "validation_profit_factor"
            ),
            "validation_profit_factor_above_1_5": economic_assessment.get(
                "validation_profit_factor_above_1_5", False
            ),
            "trade_count_minimums_enforced": False,
            "opportunity_cost_penalty_active": True,
            "dynamic_exit_genes_active": True,
        },
        "same_dataset_baseline_comparison": {
            "baseline_description": "previous weekly RSI-reversal candidate re-evaluated on this run's data and split",
            "baseline_training_metrics": asdict(baseline_training),
            "baseline_validation_metrics": asdict(baseline_validation),
            "candidate_training_trade_count": candidate["training_metrics"][
                "trade_count"
            ]
            if candidate
            else None,
            "candidate_validation_trade_count": candidate["validation_metrics"][
                "trade_count"
            ]
            if candidate
            else None,
            "training_trade_count_increased": (
                candidate is not None
                and candidate["training_metrics"]["trade_count"]
                > baseline_training.trade_count
            ),
            "validation_trade_count_increased": (
                candidate is not None
                and candidate["validation_metrics"]["trade_count"]
                > baseline_validation.trade_count
            ),
            "comparison_is_same_dataset_and_split": True,
            "baseline_participates_in_selection": False,
        },
        "timeframe_coverage": timeframe_coverage,
        "required_timeframe_coverage_by_generation": coverage_by_generation,
        "genomes_rejected_by_reason": dict(sorted(rejected_reasons.items())),
        "generation_history": history,
        "safety_invariants": {
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "private_api_calls": 0,
            "orders_or_execution_api_used": False,
            "real_order_attempts": 0,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "self-evolving-ga-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv_outputs(output_dir, history, ranked_candidates)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("artifacts/public-minute-bars/binance_BTCUSDT_1m.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/public-minute-bars/manifest.json"),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("artifacts/self-evolving-ga")
    )
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    report = run_evolution(
        args.data,
        args.manifest,
        args.out_dir,
        population_size=args.population,
        generations=args.generations,
        seed=args.seed,
    )
    candidate = report["top_research_candidate"]
    print(
        json.dumps(
            {
                "population_size": report["population_size"],
                "generations_completed": report["generations_completed"],
                "unique_genomes_evaluated": report["unique_genomes_evaluated"],
                "candidate_genes": candidate["genes"] if candidate else None,
                "training_metrics": candidate["training_metrics"]
                if candidate
                else None,
                "validation_metrics": candidate["validation_metrics"]
                if candidate
                else None,
                "research_result": report["research_result"],
                "economic_assessment": report["economic_assessment"],
                "required_timeframe_coverage_by_generation": report[
                    "required_timeframe_coverage_by_generation"
                ],
                "genomes_rejected_by_reason": report["genomes_rejected_by_reason"],
                "Champion": report["safety_invariants"]["Champion"],
                "order_permission": report["safety_invariants"]["order_permission"],
                "trading_authorization": report["safety_invariants"][
                    "trading_authorization"
                ],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
