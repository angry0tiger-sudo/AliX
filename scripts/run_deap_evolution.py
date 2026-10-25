"""Bounded, deterministic DEAP evolution on verified public 1m market data.

Research only: closed-bar signals execute no earlier than the next bar open.
The final 20% of history is excluded before feature construction; no order API
or production champion is reachable from this experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
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

TIMEFRAMES = (1, 5, 15, 60, 240, 1440)
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
LOWER = (0, 0, 0, 0, 1, 0, 0, 0, 0)
UPPER = (5, 3, 7, 5, 7, 5, 5, 3, 5)
GENOME_SIZE = 9
FEE = 25.0 / 10_000
SLIPPAGE = 20.0 / 10_000
SIDE_COST = FEE + SLIPPAGE
MANIFEST_SCHEMA = "alix.self-evolving-data-manifest.v1"


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    trade_count: int
    win_rate: float
    net_expectancy: float
    net_return: float
    max_drawdown: float
    fee_bps_per_side: float = 25.0
    slippage_bps_per_side: float = 20.0


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


def _validate_market_frame(frame: pd.DataFrame, manifest: dict[str, Any]) -> None:
    if len(frame) != int(manifest.get("row_count", -1)):
        raise ValueError("market_data_row_count_mismatch")
    if len(frame) < 50_000:
        raise ValueError("market_data_history_too_short")
    timestamps = frame["open_time_ms"].to_numpy(dtype=np.int64)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("market_data_timestamps_not_strictly_increasing")
    close_times = frame["close_time_ms"].to_numpy(dtype=np.int64)
    if np.any(close_times < timestamps):
        raise ValueError("market_data_close_before_open")
    prices = frame[["open", "high", "low", "close"]].to_numpy(dtype=float)
    volume = frame["volume"].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or not np.isfinite(volume).all():
        raise ValueError("market_data_contains_non_finite_values")
    if np.any(prices <= 0) or np.any(volume < 0):
        raise ValueError("market_data_contains_invalid_price_or_volume")
    if np.any(prices[:, 1] < prices.max(axis=1)):
        raise ValueError("market_data_high_inconsistent")
    if np.any(prices[:, 2] > prices.min(axis=1)):
        raise ValueError("market_data_low_inconsistent")


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
    bars["open_time_ms"] = bars.index.view("int64") // 1_000_000
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
        timestamp=bars["open_time_ms"].to_numpy(dtype=np.int64),
        open=bars["open"].to_numpy(dtype=np.float64),
        high=bars["high"].to_numpy(dtype=np.float64),
        low=bars["low"].to_numpy(dtype=np.float64),
        close=close.to_numpy(dtype=np.float64),
    )
    for window in LOOKBACKS:
        result.momentum[window] = (close / close.shift(window) - 1.0).to_numpy(
            dtype=np.float64
        )
        upper = high.shift(1).rolling(window, min_periods=window).max()
        lower = low.shift(1).rolling(window, min_periods=window).min()
        result.upper_gap[window] = (close / upper - 1.0).to_numpy(
            dtype=np.float64
        )
        result.lower_gap[window] = (close / lower - 1.0).to_numpy(
            dtype=np.float64
        )
        result.rsi[window] = _calc_rsi(close, window)
    ema = {
        window: close.ewm(span=window, adjust=False, min_periods=window).mean()
        for window in set(FASTS + SLOWS)
    }
    for fast in FASTS:
        for slow in SLOWS:
            if fast < slow:
                result.ema_gap[(fast, slow)] = (
                    ema[fast] / ema[slow] - 1.0
                ).to_numpy(dtype=np.float64)
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
    if any(len(item.frame) < 20 for item in features.values()):
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
        return value >= entry_threshold, value <= exit_threshold
    if mode == 1:
        value = features.ema_gap[(FASTS[genes[3]], SLOWS[genes[4]])]
        return value >= entry_threshold, value <= exit_threshold
    if mode == 2:
        value = features.rsi[window]
        return value <= entry_threshold, value >= exit_threshold
    return (
        features.upper_gap[window] >= entry_threshold,
        features.lower_gap[window] <= exit_threshold,
    )


def _find_stop(
    close: np.ndarray, entry: int, end: int, stop_fraction: float
) -> int | None:
    start = entry + 1
    if start >= end:
        return None
    segment = close[start:end]
    peaks = np.maximum.accumulate(np.r_[close[entry], segment])[1:]
    hits = np.flatnonzero(segment <= peaks * (1.0 - stop_fraction))
    return int(start + hits[0]) if len(hits) else None


def build_position_states(
    entry_signal: np.ndarray,
    exit_signal: np.ndarray,
    close: np.ndarray,
    trailing_stop: float,
) -> np.ndarray:
    """Stateful, close-confirmed strategy logic; stop executes at next bar open."""

    count = len(close)
    if count == 0 or len(entry_signal) != count or len(exit_signal) != count:
        raise ValueError("signal_length_mismatch")
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
            int(exit_indexes[exit_offset])
            if exit_offset < len(exit_indexes)
            else count
        )
        stop_index = _find_stop(close, entry, natural_exit, trailing_stop)
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


def metrics_from_positions(
    opens: np.ndarray,
    closes: np.ndarray,
    positions_after_close: np.ndarray,
    exposure: float,
) -> EvaluationMetrics:
    """Measure completed-trade expectancy net of both sides' costs."""

    if len(opens) == 0 or len(opens) != len(closes) or len(opens) != len(
        positions_after_close
    ):
        raise ValueError("metric_input_length_invalid")
    open_positions = np.r_[False, positions_after_close[:-1]]
    returns = _trade_returns(opens, closes, open_positions)
    expectancy = float(np.mean(returns)) if len(returns) else -1.0
    wins = float(np.mean(returns > 0.0)) if len(returns) else 0.0
    net_return, drawdown = _portfolio_metrics(opens, closes, open_positions, exposure)
    return EvaluationMetrics(
        int(len(returns)), wins, expectancy, net_return, drawdown
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
        mask = (
            (features.timestamp >= context.train_cutoff_ms)
            & (features.timestamp < context.validation_end_ms)
        )
    else:
        raise ValueError("evaluation_period_invalid")
    indexes = np.flatnonzero(mask)
    if len(indexes) < 2:
        raise ValueError("evaluation_period_too_short")
    position_key = tuple(genes[:7]) + (genes[8],)
    cached = context.positions_cache.get(position_key)
    if cached is None:
        cached = build_position_states(
            entry, exit_signal, features.close, STOPS[genes[8]]
        )
        context.positions_cache[position_key] = cached
    positions = cached[indexes]
    bars = features.frame.iloc[indexes]
    return metrics_from_positions(
        bars["open"].to_numpy(dtype=np.float64),
        bars["close"].to_numpy(dtype=np.float64),
        positions,
        EXPOSURES[genes[7]],
    )


def fitness_score(metrics: EvaluationMetrics) -> float:
    """Net expectancy minus declared churn, drawdown, and low-sample penalties."""

    if metrics.trade_count == 0:
        return -1.0
    drawdown_penalty = 0.25 * max(0.0, metrics.max_drawdown - 0.15)
    churn_penalty = 0.00001 * max(0, metrics.trade_count - 200)
    sample_penalty = 0.002 * max(0, 20 - metrics.trade_count) / 20
    return metrics.net_expectancy - drawdown_penalty - churn_penalty - sample_penalty


def _evaluate_individual(
    individual: MutableSequence[int],
    context: SearchContext,
    cache: dict[tuple[int, ...], tuple[float, EvaluationMetrics]],
) -> tuple[float]:
    repair_individual(individual)
    key = tuple(individual)
    if key not in cache:
        metrics = _period_metrics(context, individual, "train")
        cache[key] = (fitness_score(metrics), metrics)
    return (cache[key][0],)


def _random_individual() -> MutableSequence[int]:
    genes = [random.randint(low, high) for low, high in zip(LOWER, UPPER)]
    return repair_individual(genes)


def _build_toolbox(
    context: SearchContext,
    cache: dict[tuple[int, ...], tuple[float, EvaluationMetrics]],
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
    toolbox.register("individual", tools.initIterate, individual_type, _random_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", _evaluate_individual, context=context, cache=cache)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register(
        "mutate", tools.mutUniformInt, low=LOWER, up=UPPER, indpb=0.20
    )
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("clone", tools.clone)
    return toolbox


def _evaluate_invalid(population: Sequence[Any], toolbox: Any) -> None:
    invalid = [item for item in population if not item.fitness.valid]
    for item, fitness in zip(invalid, toolbox.map(toolbox.evaluate, invalid)):
        item.fitness.values = fitness


def _generation_record(
    generation: int, population: Sequence[Any], unique_count: int
) -> dict[str, int | float]:
    scores = np.asarray(
        [float(item.fitness.values[0]) for item in population], dtype=np.float64
    )
    return {
        "generation": generation,
        "population_size": len(population),
        "best_fitness": float(scores.max()),
        "mean_fitness": float(scores.mean()),
        "median_fitness": float(np.median(scores)),
        "unique_genomes_evaluated": unique_count,
    }


def _next_generation(
    population: Sequence[Any], toolbox: Any, crossover: float, mutation: float
) -> list[Any]:
    children = [toolbox.clone(item) for item in toolbox.select(population, len(population))]
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


def _strategy_genome(genes: Sequence[int]) -> StrategyGenome:
    timeframe_minutes = TIMEFRAMES[genes[0]]
    timeframe = TimeframeGene(timeframe_minutes, "minute")
    mode = INDICATORS[genes[1]]
    window = LOOKBACKS[genes[2]]
    fast = FASTS[genes[3]]
    slow = SLOWS[genes[4]]
    entry, exit_threshold = _thresholds(genes)
    stop = STOPS[genes[8]]
    parameters = (float(window), float(fast), float(slow), entry, exit_threshold)
    indicators = (
        IndicatorGene("entry_signal", mode, parameters, timeframe),
        IndicatorGene(f"{mode}_exit", f"{mode}_exit", parameters, timeframe),
        IndicatorGene("trailing_drawdown", "close_peak_drawdown", (stop,), timeframe),
    )
    entry_operator = "less_than" if mode == "rsi_reversal" else "greater_than"
    exit_operator = "greater_than" if mode == "rsi_reversal" else "less_than"
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
        exit_rule=ExpressionGene(
            exit_operator,
            children=(
                ExpressionGene("feature", feature=f"{mode}_exit"),
                ExpressionGene("constant", constant=exit_threshold),
            ),
        ),
        position_size_rule=ExpressionGene("constant", constant=EXPOSURES[genes[7]]),
        stop_rule=ExpressionGene(
            "less_than",
            children=(
                ExpressionGene("feature", feature="trailing_drawdown"),
                ExpressionGene("constant", constant=-stop),
            ),
        ),
    )


def _condition_text(genes: Sequence[int]) -> tuple[str, str]:
    mode = INDICATORS[genes[1]]
    window = LOOKBACKS[genes[2]]
    entry, exit_threshold = _thresholds(genes)
    if mode == "momentum":
        return (
            f"{window}-bar return >= {entry:.6f}",
            f"{window}-bar return <= {exit_threshold:.6f}",
        )
    if mode == "ema_gap":
        fast, slow = FASTS[genes[3]], SLOWS[genes[4]]
        return (
            f"EMA({fast})/EMA({slow}) - 1 >= {entry:.6f}",
            f"EMA({fast})/EMA({slow}) - 1 <= {exit_threshold:.6f}",
        )
    if mode == "rsi_reversal":
        return (
            f"RSI({window}) <= {entry:.1f}",
            f"RSI({window}) >= {exit_threshold:.1f}",
        )
    return (
        f"close/prior_{window}_bar_high - 1 >= {entry:.6f}",
        f"close/prior_{window}_bar_low - 1 <= {exit_threshold:.6f}",
    )


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
            "close_based_trailing_stop_fraction": STOPS[genes[8]],
        },
        "entry_condition": entry_text,
        "exit_condition": exit_text,
        "fitness": score,
        "training_metrics": asdict(train),
    }


def _write_csv_outputs(
    output_dir: Path,
    history: list[dict[str, int | float]],
    hall: Sequence[Any],
    cache: dict[tuple[int, ...], tuple[float, EvaluationMetrics]],
) -> None:
    with (output_dir / "self-evolving-ga-history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    fields = [
        "rank", "gene_vector", "fitness", "trade_count", "win_rate",
        "net_expectancy", "net_return", "max_drawdown",
    ]
    with (output_dir / "self-evolving-ga-candidates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rank, item in enumerate(hall, start=1):
            repair_individual(item)
            key = tuple(item)
            score, metrics = cache[key]
            writer.writerow({
                "rank": rank,
                "gene_vector": json.dumps(list(key), separators=(",", ":")),
                "fitness": score,
                "trade_count": metrics.trade_count,
                "win_rate": metrics.win_rate,
                "net_expectancy": metrics.net_expectancy,
                "net_return": metrics.net_return,
                "max_drawdown": metrics.max_drawdown,
            })


def run_evolution(
    csv_path: Path,
    manifest_path: Path,
    output_dir: Path,
    population_size: int = 128,
    generations: int = 6,
    seed: int = 20261025,
) -> dict[str, Any]:
    """Run six bounded generations; validation measures the winner but does not select it."""

    if not 100 <= population_size <= 200:
        raise ValueError("population_size_must_be_between_100_and_200")
    if not 5 <= generations <= 10:
        raise ValueError("generations_must_be_between_5_and_10")
    raw, manifest = load_verified_market_data(csv_path, manifest_path)
    train_cutoff_ms = int(raw["open_time_ms"].iloc[int(len(raw) * 0.70)])
    validation_end_ms = int(raw["open_time_ms"].iloc[int(len(raw) * 0.80)])
    context = build_search_context(raw, train_cutoff_ms, validation_end_ms)
    cache: dict[tuple[int, ...], tuple[float, EvaluationMetrics]] = {}
    toolbox = _build_toolbox(context, cache, seed)
    from deap import tools  # type: ignore[import-not-found]

    population = toolbox.population(n=population_size)
    hall = tools.HallOfFame(20)
    history: list[dict[str, int | float]] = []
    for generation in range(1, generations + 1):
        if generation > 1:
            population = _next_generation(population, toolbox, 0.65, 0.35)
        _evaluate_invalid(population, toolbox)
        hall.update(population)
        history.append(_generation_record(generation, population, len(cache)))

    best = repair_individual(hall[0])
    best_key = tuple(best)
    score, train_metrics = cache[best_key]
    validation_metrics = _period_metrics(context, best, "validation")
    candidate = _candidate_payload(best, score, train_metrics)
    report: dict[str, Any] = {
        "schema_version": "alix.self-evolving-ga-report.v1",
        "run_mode": "historical_simulation_only",
        "data_source": manifest.get("source_venue"),
        "symbol": manifest.get("symbol"),
        "source_interval": "1m",
        "source_csv_sha256": manifest.get("csv_sha256"),
        "population_size": population_size,
        "generations_requested": generations,
        "generations_completed": len(history),
        "random_seed": seed,
        "unique_genomes_evaluated": len(cache),
        "operators": {
            "selection": "tournament_k3",
            "crossover": "two_point",
            "crossover_probability": 0.65,
            "mutation": "uniform_integer",
            "mutation_probability": 0.35,
            "per_gene_mutation_probability": 0.20,
        },
        "fitness_definition": {
            "base": "mean completed-trade return after entry/exit fees and slippage",
            "drawdown_penalty": "0.25 * max(0, max_drawdown - 0.15)",
            "churn_penalty": "0.00001 * max(0, trade_count - 200)",
            "small_sample_penalty": "0.002 * max(0, 20 - trade_count) / 20",
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
            "train_cutoff_utc": pd.to_datetime(train_cutoff_ms, unit="ms", utc=True).isoformat(),
            "validation_end_utc": pd.to_datetime(validation_end_ms, unit="ms", utc=True).isoformat(),
            "final_holdout_used": False,
            "final_holdout_metrics_emitted": False,
        },
        "top_research_candidate": {
            **candidate,
            "validation_metrics": asdict(validation_metrics),
            "validation_used_for_selection": False,
        },
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
    _write_csv_outputs(output_dir, history, hall, cache)
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
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/self-evolving-ga"))
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    report = run_evolution(
        args.data, args.manifest, args.out_dir,
        population_size=args.population, generations=args.generations, seed=args.seed,
    )
    candidate = report["top_research_candidate"]
    print(json.dumps({
        "population_size": report["population_size"],
        "generations_completed": report["generations_completed"],
        "unique_genomes_evaluated": report["unique_genomes_evaluated"],
        "candidate_genes": candidate["genes"],
        "training_metrics": candidate["training_metrics"],
        "validation_metrics": candidate["validation_metrics"],
        "Champion": report["safety_invariants"]["Champion"],
        "order_permission": report["safety_invariants"]["order_permission"],
        "trading_authorization": report["safety_invariants"]["trading_authorization"],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
