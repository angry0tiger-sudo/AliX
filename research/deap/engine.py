"""Bounded DEAP GP on immutable pre-hack Nobitex BTC/USDT Spot 1h data.

The evolved tree returns boolean BUY/NO_BUY signals. Decisions use a completed
bar and simulated fills occur no earlier than the next bar open.
"""

from __future__ import annotations

import hashlib
import json
import operator
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from deap import base, creator, gp, tools  # type: ignore[import-untyped]
from numpy.typing import ArrayLike, NDArray

CSV_NAME = "nobitex-btcusdt-1h.csv"
MANIFEST_NAME = "nobitex-btcusdt-1h.manifest.json"
EXPECTED_ROWS = 31_801
EXPECTED_START_SECONDS = 1_635_712_200
EXPECTED_END_SECONDS = 1_750_192_200
EXPECTED_DATASET_SHA256 = (
    "0aa13607d1da68c4720919f233635bb33d34048e5bd67d3ecc2ce9bb7c2b462e"
)
EXPECTED_PRODUCER_SHA = "e6c86b08da98ad28e479b45827fb8de6cd3cc8dd"
EXPECTED_PROVIDER = "Nobitex public UDF history API"
EXPECTED_SOURCE_POLICY = "NOBITEX_ONLY_NO_BACKFILL"
BAR_INTERVAL_MS = 3_600_000
ROUND_TRIP_COST_BPS = 50.0
FEE_BPS_PER_SIDE = 25.0
SLIPPAGE_BPS_PER_SIDE = 0.0
MAX_DRAWDOWN = 0.15
MIN_TRADES_PER_YEAR = 200.0
MIN_ABSOLUTE_TRADES_3_5Y = 150
PURGE_BARS = 48
EMBARGO_BARS = 48
MAX_HOLDING_BARS = 48
FEATURE_NAMES = (
    "RSI_14",
    "MACD_hist",
    "ATR_14",
    "Close",
    "Volume_ZScore",
    "ROC_6",
    "Range_14",
)
GP_CONSTANTS = (-1.0, -0.75, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
MAX_TREE_HEIGHT = 5
MAX_TREE_NODES = 31
TOURNAMENT_SIZE = 5
CROSSOVER_PROBABILITY = 0.70
MUTATION_PROBABILITY = 0.25
FITNESS_CACHE_LIMIT = 50_000
ALLOWED_TIMEFRAMES_MINUTES = (15, 30, 60, 240)
ACTIVE_TIMEFRAME_MINUTES = 60


class NumericVector(np.ndarray[Any, np.dtype[np.float64]]):
    """DEAP marker type for normalized numeric feature vectors."""


class BooleanVector(np.ndarray[Any, np.dtype[np.bool_]]):
    """DEAP marker type for boolean signal vectors."""


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _numeric(values: ArrayLike) -> NumericVector:
    return cast(
        NumericVector,
        np.asarray(values, dtype=np.float64).view(NumericVector),
    )


def _boolean(values: ArrayLike) -> BooleanVector:
    return cast(
        BooleanVector,
        np.asarray(values, dtype=np.bool_).view(BooleanVector),
    )


def add(left: NumericVector, right: NumericVector) -> NumericVector:
    return _numeric(np.clip(np.asarray(left) + np.asarray(right), -10.0, 10.0))


def sub(left: NumericVector, right: NumericVector) -> NumericVector:
    return _numeric(np.clip(np.asarray(left) - np.asarray(right), -10.0, 10.0))


def mul(left: NumericVector, right: NumericVector) -> NumericVector:
    return _numeric(np.clip(np.asarray(left) * np.asarray(right), -10.0, 10.0))


def div(left: NumericVector, right: NumericVector) -> NumericVector:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    shape = np.broadcast_shapes(left_array.shape, right_array.shape)
    output = np.zeros(shape, dtype=np.float64)
    np.divide(
        left_array,
        right_array,
        out=output,
        where=np.abs(right_array) > 1e-6,
    )
    return _numeric(np.clip(output, -10.0, 10.0))


def gt(left: NumericVector, right: NumericVector) -> BooleanVector:
    return _boolean(np.greater(np.asarray(left), np.asarray(right)))


def lt(left: NumericVector, right: NumericVector) -> BooleanVector:
    return _boolean(np.less(np.asarray(left), np.asarray(right)))


def logical_and(
    left: BooleanVector,
    right: BooleanVector,
) -> BooleanVector:
    return _boolean(np.logical_and(np.asarray(left), np.asarray(right)))


def logical_or(
    left: BooleanVector,
    right: BooleanVector,
) -> BooleanVector:
    return _boolean(np.logical_or(np.asarray(left), np.asarray(right)))


def build_primitive_set() -> Any:
    """Create a strongly typed grammar whose root must return a boolean signal."""
    pset = gp.PrimitiveSetTyped(
        "MAIN",
        [NumericVector for _ in FEATURE_NAMES],
        BooleanVector,
    )
    pset.renameArguments(
        **{f"ARG{idx}": name for idx, name in enumerate(FEATURE_NAMES)}
    )
    pset.addPrimitive(add, [NumericVector, NumericVector], NumericVector)
    pset.addPrimitive(sub, [NumericVector, NumericVector], NumericVector)
    pset.addPrimitive(mul, [NumericVector, NumericVector], NumericVector)
    pset.addPrimitive(div, [NumericVector, NumericVector], NumericVector)
    pset.addPrimitive(gt, [NumericVector, NumericVector], BooleanVector)
    pset.addPrimitive(lt, [NumericVector, NumericVector], BooleanVector)
    pset.addPrimitive(
        logical_and,
        [BooleanVector, BooleanVector],
        BooleanVector,
        name="and_",
    )
    pset.addPrimitive(
        logical_or,
        [BooleanVector, BooleanVector],
        BooleanVector,
        name="or_",
    )
    for idx, value in enumerate(GP_CONSTANTS):
        pset.addTerminal(_numeric(np.asarray(value)), NumericVector, name=f"K{idx}")
    pset.addTerminal(_boolean(np.asarray(True)), BooleanVector, name="TRUE")
    pset.addTerminal(_boolean(np.asarray(False)), BooleanVector, name="FALSE")
    return pset


@dataclass(frozen=True, slots=True)
class FeatureSet:
    values: pd.DataFrame
    valid_mask: BoolArray
    first_valid_row: int


@dataclass(frozen=True, slots=True)
class MarketArrays:
    open: FloatArray
    high: FloatArray
    low: FloatArray
    close: FloatArray
    close_time_ms: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class FoldView:
    market: MarketArrays
    features: dict[str, NumericVector]
    has_preceding_context: bool


@dataclass(frozen=True, slots=True)
class SimulationMetrics:
    trade_count: int
    net_profit_factor: float | None
    win_rate: float | None
    max_drawdown: float
    net_return: float
    gross_profit: float
    gross_loss: float
    trades_per_year: float
    duration_years: float
    average_trade_return: float | None
    total_fees_bps_of_initial_equity: float
    max_holding_bars: int
    trade_returns: tuple[float, ...]

    def report(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("trade_returns", None)
        result["round_trip_cost_bps"] = ROUND_TRIP_COST_BPS
        return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_dataset_manifest() -> dict[str, Any]:
    return {
        "source_policy": EXPECTED_SOURCE_POLICY,
        "data_sources": [EXPECTED_PROVIDER],
        "dataset_window": "PRE_HACK_NOBITEX_ONLY",
        "provider": EXPECTED_PROVIDER,
        "symbol": "BTCUSDT",
        "pair": "BTC/USDT",
        "interval": "1h",
        "requested_days": 1325,
        "row_count": EXPECTED_ROWS,
        "expected_candles": EXPECTED_ROWS,
        "start_timestamp": EXPECTED_START_SECONDS,
        "end_timestamp": EXPECTED_END_SECONDS,
        "missing_intervals": 0,
        "csv_sha256": EXPECTED_DATASET_SHA256,
        "commit_sha": EXPECTED_PRODUCER_SHA,
        "public_api_only": True,
        "credentials_used": False,
        "private_api_calls": 0,
        "exchange_calls": 0,
        "real_order_attempts": 0,
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
    }


def _validate_dataset_manifest(
    manifest: dict[str, Any],
    digest: str,
) -> None:
    if digest != EXPECTED_DATASET_SHA256:
        raise ValueError("dataset_sha256_does_not_match_approved_ai6_artifact")
    for field, expected_value in _expected_dataset_manifest().items():
        if manifest.get(field) != expected_value:
            raise ValueError(f"dataset_manifest_contract_mismatch:{field}")
    if manifest.get("source_overrides") not in (None, []):
        raise ValueError("external_data_source_override_forbidden")


def _validate_dataset_timestamps(data: pd.DataFrame) -> Any:
    expected_columns = ["date", "open", "high", "low", "close", "volume"]
    if data.columns.tolist() != expected_columns:
        raise ValueError("dataset_columns_mismatch")
    dates = pd.to_datetime(data["date"], errors="coerce", utc=True)
    if dates.isna().any() or len(data) != EXPECTED_ROWS:
        raise ValueError("dataset_timestamp_or_row_count_invalid")
    actual_seconds = dates.astype("int64").to_numpy(dtype=np.int64) // 1_000_000_000
    expected_seconds = np.arange(
        EXPECTED_START_SECONDS,
        EXPECTED_END_SECONDS + 1,
        3_600,
        dtype=np.int64,
    )
    if not np.array_equal(actual_seconds, expected_seconds):
        raise ValueError("dataset_prehack_hourly_timestamp_grid_mismatch")
    return dates


def _validate_ohlcv_geometry(data: pd.DataFrame) -> None:
    numeric = data.loc[:, ["open", "high", "low", "close", "volume"]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(numeric).all():
        raise ValueError("dataset_contains_non_finite_ohlcv")
    if np.any(numeric[:, :4] <= 0.0) or np.any(numeric[:, 4] < 0.0):
        raise ValueError("dataset_price_or_volume_out_of_range")
    if np.any(numeric[:, 1] < np.max(numeric[:, [0, 2, 3]], axis=1)):
        raise ValueError("dataset_high_geometry_invalid")
    if np.any(numeric[:, 2] > np.min(numeric[:, [0, 1, 3]], axis=1)):
        raise ValueError("dataset_low_geometry_invalid")


def load_verified_dataset(
    input_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    """Load the exact AI-6 Nobitex artifact; provenance drift fails closed."""
    csv_path = input_dir / CSV_NAME
    manifest_path = input_dir / MANIFEST_NAME
    if not csv_path.is_file() or not manifest_path.is_file():
        raise ValueError("prehack_nobitex_dataset_artifact_missing")

    manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_value, dict):
        raise ValueError("dataset_manifest_must_be_object")
    manifest: dict[str, Any] = manifest_value
    digest = sha256_file(csv_path)
    _validate_dataset_manifest(manifest, digest)

    data = pd.read_csv(csv_path)
    dates = _validate_dataset_timestamps(data)
    _validate_ohlcv_geometry(data)
    for column in ("open", "high", "low", "close", "volume"):
        data[column] = data[column].astype(np.float64)
    data["date"] = dates
    return data, manifest, digest


def build_causal_features(data: pd.DataFrame) -> FeatureSet:
    """Build normalized past-only OHLCV features available after each bar close."""
    close = data["close"].astype(np.float64)
    high = data["high"].astype(np.float64)
    low = data["low"].astype(np.float64)
    volume = data["volume"].astype(np.float64)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(
        alpha=1.0 / 14.0,
        adjust=False,
        min_periods=14,
    ).mean()

    delta = close.diff()
    average_gain = (
        delta.clip(lower=0.0)
        .ewm(
            alpha=1.0 / 14.0,
            adjust=False,
            min_periods=14,
        )
        .mean()
    )
    average_loss = (
        (-delta.clip(upper=0.0))
        .ewm(
            alpha=1.0 / 14.0,
            adjust=False,
            min_periods=14,
        )
        .mean()
    )
    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + relative_strength)
    rsi = rsi.mask((average_loss == 0.0) & (average_gain > 0.0), 100.0)
    rsi = rsi.mask((average_loss == 0.0) & (average_gain == 0.0), 50.0)

    ema_fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    macd_histogram = (macd_line - macd_signal) / atr.replace(0.0, np.nan)

    close_reference = close.ewm(
        span=100,
        adjust=False,
        min_periods=100,
    ).mean()
    close_relative = np.log(close / close_reference)
    volume_log = pd.Series(
        np.log1p(volume.to_numpy(dtype=np.float64)),
        index=volume.index,
        dtype=np.float64,
    )
    volume_mean = volume_log.rolling(96, min_periods=48).mean()
    volume_std = volume_log.rolling(96, min_periods=48).std(ddof=0)
    volume_zscore = (volume_log - volume_mean) / volume_std.replace(0.0, np.nan)
    volume_zscore = volume_zscore.mask(volume_std <= 1e-12, 0.0)

    range_fraction = ((high - low) / close).rolling(14, min_periods=14).mean()
    features = pd.DataFrame(
        {
            "RSI_14": ((rsi - 50.0) / 50.0).clip(-1.0, 1.0),
            "MACD_hist": (macd_histogram / 5.0).clip(-1.0, 1.0),
            "ATR_14": ((atr / close) / 0.10).clip(0.0, 1.0),
            "Close": (close_relative / 0.10).clip(-1.0, 1.0),
            "Volume_ZScore": (volume_zscore / 5.0).clip(-1.0, 1.0),
            "ROC_6": ((close / close.shift(6) - 1.0) / 0.10).clip(-1.0, 1.0),
            "Range_14": (range_fraction / 0.10).clip(0.0, 1.0),
        },
        index=data.index,
    )
    valid_mask = np.isfinite(features.to_numpy(dtype=np.float64)).all(axis=1)
    valid_positions = np.flatnonzero(valid_mask)
    if len(valid_positions) < 2_000:
        raise ValueError("causal_feature_rows_insufficient")
    first_valid = int(valid_positions[0])
    if not np.array_equal(valid_positions, np.arange(first_valid, len(data))):
        raise ValueError("causal_feature_valid_rows_not_contiguous")
    return FeatureSet(
        values=features,
        valid_mask=valid_mask.astype(np.bool_),
        first_valid_row=first_valid,
    )


def _market_arrays(data: pd.DataFrame, positions: NDArray[np.int64]) -> MarketArrays:
    selected = data.iloc[positions]
    close_time_ms = (
        selected["date"].astype("int64").to_numpy(dtype=np.int64) // 1_000_000
        + BAR_INTERVAL_MS
        - 1
    )
    return MarketArrays(
        open=selected["open"].to_numpy(dtype=np.float64),
        high=selected["high"].to_numpy(dtype=np.float64),
        low=selected["low"].to_numpy(dtype=np.float64),
        close=selected["close"].to_numpy(dtype=np.float64),
        close_time_ms=close_time_ms,
    )


def _validate_simulation_inputs(
    market: MarketArrays,
    signals: BoolArray,
) -> tuple[int, BoolArray]:
    count = len(market.close)
    if count < 2 or len(signals) != count:
        raise ValueError("simulation_market_or_signal_length_invalid")
    market_fields = (
        market.open,
        market.high,
        market.low,
        market.close,
        market.close_time_ms,
    )
    if any(len(field) != count for field in market_fields):
        raise ValueError("simulation_market_array_length_mismatch")
    prices = np.column_stack((market.open, market.high, market.low, market.close))
    if not np.isfinite(prices).all():
        raise ValueError("simulation_market_non_finite")
    if np.any(np.diff(market.close_time_ms) != BAR_INTERVAL_MS):
        raise ValueError("simulation_market_bars_not_hourly_contiguous")
    if np.asarray(signals).dtype != np.bool_:
        raise ValueError("simulation_signals_must_be_boolean")
    return count, np.asarray(signals, dtype=np.bool_)


def _resolve_episode_exit(
    start: int,
    false_positions: NDArray[np.int64],
    count: int,
) -> tuple[int, bool]:
    false_offset = int(np.searchsorted(false_positions, start, side="right"))
    has_signal_off = false_offset < len(false_positions)
    signal_exit_open = (
        int(false_positions[false_offset]) + 1 if has_signal_off else count + 1
    )
    forced_exit_open = start + MAX_HOLDING_BARS + 1
    exit_open = min(signal_exit_open, forced_exit_open)
    return exit_open, exit_open >= count


def _mark_episode_drawdown(
    market: MarketArrays,
    quantity: float,
    active_start: int,
    active_end: int,
    peak_equity: float,
    maximum_drawdown: float,
) -> tuple[float, float]:
    active_slice = slice(active_start, active_end + 1)
    active_high = quantity * market.high[active_slice]
    active_low = quantity * market.low[active_slice]
    active_close = quantity * market.close[active_slice]
    local_peak = np.maximum.accumulate(np.maximum(peak_equity, active_high))
    local_drawdown = np.maximum(
        (local_peak - active_low) / local_peak,
        (local_peak - active_close) / local_peak,
    )
    return (
        max(peak_equity, float(np.max(active_high))),
        max(maximum_drawdown, float(np.max(local_drawdown))),
    )


def _build_simulation_metrics(
    market: MarketArrays,
    trade_returns: list[float],
    trade_pnls: list[float],
    cash: float,
    peak_equity: float,
    maximum_drawdown: float,
    total_fees_quote: float,
) -> SimulationMetrics:
    gross_profit = float(sum(value for value in trade_pnls if value > 0.0))
    gross_loss = float(-sum(value for value in trade_pnls if value < 0.0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0.0 else None
    win_rate = (
        sum(value > 0.0 for value in trade_returns) / len(trade_returns)
        if trade_returns
        else None
    )
    duration_years = max(
        (int(market.close_time_ms[-1]) - int(market.close_time_ms[0]))
        / 1000.0
        / (365.25 * 24.0 * 60.0 * 60.0),
        1e-9,
    )
    return SimulationMetrics(
        trade_count=len(trade_returns),
        net_profit_factor=profit_factor,
        win_rate=win_rate,
        max_drawdown=float(maximum_drawdown),
        net_return=float(cash - 1.0),
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        trades_per_year=len(trade_returns) / duration_years,
        duration_years=duration_years,
        average_trade_return=(float(np.mean(trade_returns)) if trade_returns else None),
        total_fees_bps_of_initial_equity=total_fees_quote * 10_000.0,
        max_holding_bars=MAX_HOLDING_BARS,
        trade_returns=tuple(float(value) for value in trade_returns),
    )


def simulate_segment(
    market: MarketArrays,
    signals: BoolArray,
    *,
    initial_previous_signal: bool = False,
) -> SimulationMetrics:
    """Evaluate on close, fill next open, charge 25 bps per executed side."""
    count, signal_values = _validate_simulation_inputs(market, signals)
    prior = np.concatenate(
        (np.asarray([initial_previous_signal], dtype=np.bool_), signal_values[:-1])
    )
    starts = np.flatnonzero(signal_values & ~prior)
    false_positions = np.flatnonzero(~signal_values)

    fee_rate = FEE_BPS_PER_SIDE / 10_000.0
    cash = 1.0
    peak_equity = 1.0
    maximum_drawdown = 0.0
    total_fees_quote = 0.0
    trade_returns: list[float] = []
    trade_pnls: list[float] = []
    processed_until = 0

    for start_value in starts:
        start = int(start_value)
        entry_open = start + 1
        if entry_open >= count or entry_open < processed_until:
            continue

        exit_open, terminal_exit = _resolve_episode_exit(
            start,
            false_positions,
            count,
        )
        active_start = entry_open
        active_end = count - 1 if terminal_exit else exit_open - 1
        exit_price = (
            float(market.close[-1]) if terminal_exit else float(market.open[exit_open])
        )
        if active_start > active_end:
            continue

        entry_equity = cash
        entry_price = float(market.open[entry_open])
        entry_notional = entry_equity / (1.0 + fee_rate)
        total_fees_quote += entry_notional * fee_rate
        quantity = entry_notional / entry_price
        peak_equity, maximum_drawdown = _mark_episode_drawdown(
            market,
            quantity,
            active_start,
            active_end,
            peak_equity,
            maximum_drawdown,
        )

        exit_notional = quantity * exit_price
        exit_fee = exit_notional * fee_rate
        total_fees_quote += exit_fee
        cash = exit_notional - exit_fee
        trade_returns.append(cash / entry_equity - 1.0)
        trade_pnls.append(cash - entry_equity)
        if peak_equity > 0.0:
            maximum_drawdown = max(
                maximum_drawdown,
                (peak_equity - cash) / peak_equity,
            )
        processed_until = count if terminal_exit else exit_open + 1
        if terminal_exit:
            break

    return _build_simulation_metrics(
        market,
        trade_returns,
        trade_pnls,
        cash,
        peak_equity,
        maximum_drawdown,
        total_fees_quote,
    )


def _fitness_from_metrics(
    metrics: SimulationMetrics,
    *,
    tree_nodes: int,
    tree_height: int,
) -> float:
    """Apply severe frequency, PF and drawdown penalties plus light parsimony."""
    score = 0.0
    if metrics.trade_count < MIN_ABSOLUTE_TRADES_3_5Y:
        score -= 400.0
    if metrics.trades_per_year < MIN_TRADES_PER_YEAR:
        score -= 200.0 + (MIN_TRADES_PER_YEAR - metrics.trades_per_year) * 1.5
    else:
        score += (
            min(
                metrics.trades_per_year - MIN_TRADES_PER_YEAR,
                500.0,
            )
            * 0.01
        )

    if metrics.net_profit_factor is None:
        score -= 200.0
    elif metrics.net_profit_factor < 1.0:
        score -= 250.0 + (1.0 - metrics.net_profit_factor) * 100.0
    else:
        score += min(metrics.net_profit_factor, 5.0) * 12.0

    if metrics.net_return <= 0.0:
        score -= 100.0 + min(abs(metrics.net_return), 5.0) * 20.0
    else:
        score += min(metrics.net_return, 3.0) * 10.0

    if metrics.max_drawdown > MAX_DRAWDOWN:
        score -= 500.0 + (metrics.max_drawdown - MAX_DRAWDOWN) * 2_000.0
    else:
        score += (MAX_DRAWDOWN - metrics.max_drawdown) * 20.0

    score -= 0.05 * tree_nodes + 0.10 * tree_height
    return float(score)


def _oos_economic_checks(metrics: SimulationMetrics) -> dict[str, bool]:
    return {
        "oos_net_profit_factor_at_least_1_1": (
            metrics.net_profit_factor is not None and metrics.net_profit_factor >= 1.1
        ),
        "oos_max_drawdown_below_15_percent": metrics.max_drawdown < 0.15,
        "oos_trade_frequency_over_200_per_year": (metrics.trades_per_year > 200.0),
        "oos_positive_net_return": metrics.net_return > 0.0,
    }


def _compile_signals(
    individual: Any,
    primitive_set: Any,
    feature_arrays: dict[str, NumericVector],
) -> BoolArray:
    compiled = gp.compile(expr=individual, pset=primitive_set)
    result = compiled(*(feature_arrays[name] for name in FEATURE_NAMES))
    signal = np.asarray(result, dtype=np.bool_)
    length = len(feature_arrays[FEATURE_NAMES[0]])
    if signal.ndim == 0:
        signal = np.full(length, bool(signal), dtype=np.bool_)
    if signal.ndim != 1 or len(signal) != length:
        raise ValueError("gp_rule_output_shape_invalid")
    return signal.astype(np.bool_, copy=False)


def _feature_arrays(
    features: pd.DataFrame,
    positions: NDArray[np.int64],
) -> dict[str, NumericVector]:
    return {
        name: _numeric(features[name].to_numpy(dtype=np.float64)[positions])
        for name in FEATURE_NAMES
    }


def _fold_positions(
    positions: NDArray[np.int64],
) -> list[NDArray[np.int64]]:
    folds = [
        part.astype(np.int64, copy=False)
        for part in np.array_split(positions, 3)
        if len(part) >= 2
    ]
    if len(folds) != 3:
        raise ValueError("training_walk_forward_folds_insufficient")
    return folds


def _validate_evolution_request(
    data: pd.DataFrame,
    commit_sha: str,
    population_size: int,
    generations: int,
) -> tuple[int, int, int]:
    if len(commit_sha) != 40 or any(
        char not in "0123456789abcdef" for char in commit_sha
    ):
        raise ValueError("commit_sha_invalid")
    if population_size != 500 or generations != 30:
        raise ValueError("owner_approved_population_or_generation_contract_mismatch")
    if ACTIVE_TIMEFRAME_MINUTES not in ALLOWED_TIMEFRAMES_MINUTES:
        raise ValueError("active_timeframe_not_in_owner_allowlist")
    if len(data) != EXPECTED_ROWS:
        raise ValueError("dataset_row_count_invalid_for_evolution")
    split_index = int(len(data) * 0.70)
    return split_index, split_index - PURGE_BARS, split_index + EMBARGO_BARS


def _feature_arrays_with_context(
    features: FeatureSet,
    positions: NDArray[np.int64],
) -> tuple[dict[str, NumericVector], bool]:
    has_context = int(positions[0]) > features.first_valid_row
    extended_positions = (
        np.concatenate((positions[:1] - 1, positions)) if has_context else positions
    )
    return _feature_arrays(features.values, extended_positions), has_context


def _split_segment_signal(
    combined_signal: BoolArray,
    has_context: bool,
) -> tuple[BoolArray, bool]:
    if has_context:
        return combined_signal[1:], bool(combined_signal[0])
    return combined_signal, False


def _build_fold_views(
    data: pd.DataFrame,
    features: FeatureSet,
    training_positions: NDArray[np.int64],
) -> list[FoldView]:
    views: list[FoldView] = []
    for positions in _fold_positions(training_positions):
        feature_arrays, has_context = _feature_arrays_with_context(
            features,
            positions,
        )
        views.append(
            FoldView(
                market=_market_arrays(data, positions),
                features=feature_arrays,
                has_preceding_context=has_context,
            )
        )
    return views


def _prepare_evolution_data(
    data: pd.DataFrame,
    features: FeatureSet,
    training_end: int,
    oos_start: int,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    MarketArrays,
    dict[str, NumericVector],
    list[FoldView],
]:
    rows = np.arange(len(data), dtype=np.int64)
    training_positions = rows[features.valid_mask & (rows < training_end)]
    oos_positions = rows[features.valid_mask & (rows >= oos_start)]
    if len(training_positions) < 5_000 or len(oos_positions) < 2_000:
        raise ValueError("purged_training_or_oos_window_insufficient")
    return (
        training_positions,
        oos_positions,
        _market_arrays(data, training_positions),
        _feature_arrays(features.values, training_positions),
        _build_fold_views(data, features, training_positions),
    )


def _create_deap_toolbox(primitive_set: Any) -> Any:
    if not hasattr(creator, "AliXDeapFitnessMax"):
        creator.create("AliXDeapFitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "AliXDeapIndividual"):
        creator.create(
            "AliXDeapIndividual",
            gp.PrimitiveTree,
            fitness=creator.AliXDeapFitnessMax,
        )
    individual_type = getattr(creator, "AliXDeapIndividual")
    toolbox = base.Toolbox()
    toolbox.register(
        "expression",
        gp.genHalfAndHalf,
        pset=primitive_set,
        min_=1,
        max_=3,
    )
    toolbox.register(
        "individual",
        tools.initIterate,
        individual_type,
        toolbox.expression,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register(
        "mutation_expression",
        gp.genFull,
        min_=0,
        max_=2,
        pset=primitive_set,
    )
    toolbox.register(
        "mutate",
        gp.mutUniform,
        expr=toolbox.mutation_expression,
        pset=primitive_set,
    )
    for operation_name in ("mate", "mutate"):
        toolbox.decorate(
            operation_name,
            gp.staticLimit(
                key=operator.attrgetter("height"),
                max_value=MAX_TREE_HEIGHT,
            ),
        )
        toolbox.decorate(
            operation_name,
            gp.staticLimit(key=len, max_value=MAX_TREE_NODES),
        )
    return toolbox


def _evaluate_candidate(
    individual: Any,
    primitive_set: Any,
    train_market: MarketArrays,
    train_features: dict[str, NumericVector],
    fold_views: list[FoldView],
) -> tuple[float, dict[str, Any]]:
    try:
        train_signal = _compile_signals(
            individual,
            primitive_set,
            train_features,
        )
        metrics = simulate_segment(train_market, train_signal)
        fold_reports: list[dict[str, Any]] = []
        positive_folds = 0
        risk_failures = 0
        for fold in fold_views:
            combined = _compile_signals(
                individual,
                primitive_set,
                fold.features,
            )
            fold_signal, previous_signal = _split_segment_signal(
                combined,
                fold.has_preceding_context,
            )
            fold_metric = simulate_segment(
                fold.market,
                fold_signal,
                initial_previous_signal=previous_signal,
            )
            fold_reports.append(fold_metric.report())
            positive_folds += int(fold_metric.net_return > 0.0)
            risk_failures += int(fold_metric.max_drawdown > MAX_DRAWDOWN)
        score = _fitness_from_metrics(
            metrics,
            tree_nodes=len(individual),
            tree_height=individual.height,
        )
        score += positive_folds * 5.0
        if positive_folds < 2:
            score -= (2 - positive_folds) * 75.0
        score -= risk_failures * 100.0
        payload = {
            "training": metrics.report(),
            "folds": fold_reports,
            "positive_training_folds": positive_folds,
            "folds_over_drawdown_limit": risk_failures,
            "tree_nodes": len(individual),
            "tree_height": individual.height,
        }
    except (
        ArithmeticError,
        FloatingPointError,
        TypeError,
        ValueError,
        SyntaxError,
    ):
        score = -10_000.0
        payload = {
            "training": None,
            "folds": [],
            "positive_training_folds": 0,
            "folds_over_drawdown_limit": 3,
            "tree_nodes": len(individual),
            "tree_height": individual.height,
            "evaluation_error": "invalid_or_non_finite_gp_expression",
        }
    return float(score), payload


def _evaluate_cached_candidate(
    individual: Any,
    primitive_set: Any,
    train_market: MarketArrays,
    train_features: dict[str, NumericVector],
    fold_views: list[FoldView],
    cache: dict[str, tuple[float, dict[str, Any]]],
) -> tuple[float, dict[str, Any]]:
    key = str(individual)
    cached = cache.get(key)
    if cached is not None:
        return cached
    evaluated = _evaluate_candidate(
        individual,
        primitive_set,
        train_market,
        train_features,
        fold_views,
    )
    if len(cache) < FITNESS_CACHE_LIMIT:
        cache[key] = evaluated
    return evaluated


def _bind_evaluator(
    primitive_set: Any,
    train_market: MarketArrays,
    train_features: dict[str, NumericVector],
    fold_views: list[FoldView],
    cache: dict[str, tuple[float, dict[str, Any]]],
) -> Any:
    def evaluate(individual: Any) -> tuple[float]:
        score, _ = _evaluate_cached_candidate(
            individual,
            primitive_set,
            train_market,
            train_features,
            fold_views,
            cache,
        )
        return (score,)

    return evaluate


def _evaluate_invalid_individuals(
    population: list[Any],
    toolbox: Any,
) -> None:
    invalid = [individual for individual in population if not individual.fitness.valid]
    for individual, fitness in zip(
        invalid,
        map(toolbox.evaluate, invalid),
        strict=True,
    ):
        individual.fitness.values = fitness


def _generation_record(
    generation: int,
    population: list[Any],
    cache: dict[str, tuple[float, dict[str, Any]]],
) -> dict[str, float | int]:
    fitness_values = [float(item.fitness.values[0]) for item in population]
    return {
        "generation": generation,
        "best_fitness": max(fitness_values),
        "mean_fitness": float(np.mean(fitness_values)),
        "unique_evaluations": len(cache),
    }


def _evolve_population(
    toolbox: Any,
    population_size: int,
    generations: int,
    cache: dict[str, tuple[float, dict[str, Any]]],
) -> tuple[list[Any], Any, list[dict[str, float | int]]]:
    population = toolbox.population(n=population_size)
    hall_of_fame = tools.HallOfFame(5)
    generation_log: list[dict[str, float | int]] = []
    _evaluate_invalid_individuals(population, toolbox)
    hall_of_fame.update(population)
    generation_log.append(_generation_record(0, population, cache))

    for generation in range(1, generations + 1):
        offspring = list(
            map(
                toolbox.clone,
                toolbox.select(population, len(population)),
            )
        )
        for child_a, child_b in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVER_PROBABILITY:
                toolbox.mate(child_a, child_b)
                del child_a.fitness.values
                del child_b.fitness.values
        for mutant in offspring:
            if random.random() < MUTATION_PROBABILITY:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        _evaluate_invalid_individuals(offspring, toolbox)
        population = offspring
        hall_of_fame.update(population)
        generation_log.append(_generation_record(generation, population, cache))
    return population, hall_of_fame, generation_log


def _freeze_best_candidate(
    hall_of_fame: Any,
    cache: dict[str, tuple[float, dict[str, Any]]],
) -> tuple[Any, str, dict[str, Any]]:
    if not hall_of_fame:
        raise RuntimeError("deap_hall_of_fame_empty")
    best = hall_of_fame[0]
    best_rule = str(best)
    cached = cache.get(best_rule)
    if cached is None or cached[1]["training"] is None:
        raise RuntimeError("best_deap_candidate_has_no_valid_training_evaluation")
    return best, best_rule, cached[1]


def _evaluate_frozen_oos(
    data: pd.DataFrame,
    features: FeatureSet,
    oos_positions: NDArray[np.int64],
    best: Any,
    primitive_set: Any,
) -> tuple[MarketArrays, SimulationMetrics]:
    oos_market = _market_arrays(data, oos_positions)
    oos_features, has_context = _feature_arrays_with_context(
        features,
        oos_positions,
    )
    combined_signal = _compile_signals(best, primitive_set, oos_features)
    oos_signal, previous_signal = _split_segment_signal(
        combined_signal,
        has_context,
    )
    return (
        oos_market,
        simulate_segment(
            oos_market,
            oos_signal,
            initial_previous_signal=previous_signal,
        ),
    )


def _build_promotion_checks(
    oos_metrics: SimulationMetrics,
    training_report: dict[str, Any],
    best: Any,
) -> dict[str, bool]:
    checks = _oos_economic_checks(oos_metrics)
    checks.update(
        {
            "training_two_positive_walk_forward_folds": (
                training_report["positive_training_folds"] >= 2
            ),
            "training_tree_size_within_limits": (
                len(best) <= MAX_TREE_NODES and best.height <= MAX_TREE_HEIGHT
            ),
        }
    )
    return checks


def _build_evolution_report(
    data: pd.DataFrame,
    commit_sha: str,
    population_size: int,
    generations: int,
    seed: int,
    split_index: int,
    training_end: int,
    oos_start: int,
    train_market: MarketArrays,
    oos_market: MarketArrays,
    generation_log: list[dict[str, float | int]],
    cache: dict[str, tuple[float, dict[str, Any]]],
    hall_of_fame: Any,
    best: Any,
    best_rule: str,
    training_report: dict[str, Any],
    oos_metrics: SimulationMetrics,
    checks: dict[str, bool],
) -> dict[str, Any]:
    rule_sha = hashlib.sha256(best_rule.encode("utf-8")).hexdigest()
    full_years = (EXPECTED_END_SECONDS - EXPECTED_START_SECONDS) / (
        365.25 * 24.0 * 60.0 * 60.0
    )
    return {
        "schema_version": "alix.ai2.deap-genetic-programming.v1",
        "commit_sha": commit_sha,
        "configuration": {
            "algorithm": "DEAP strongly-typed genetic programming",
            "population_size": population_size,
            "generations": generations,
            "selection": "tournament",
            "tournament_size": TOURNAMENT_SIZE,
            "crossover": "one-point",
            "crossover_probability": CROSSOVER_PROBABILITY,
            "mutation": "uniform subtree mutation",
            "mutation_probability": MUTATION_PROBABILITY,
            "seed": seed,
            "timeframe": "1h",
            "timeframe_minutes": ACTIVE_TIMEFRAME_MINUTES,
            "allowed_timeframes_minutes": list(ALLOWED_TIMEFRAMES_MINUTES),
            "max_tree_height": MAX_TREE_HEIGHT,
            "max_tree_nodes": MAX_TREE_NODES,
            "max_holding_bars": MAX_HOLDING_BARS,
            "genome_output": "boolean BUY / NO_BUY vector",
            "fitness_uses_oos": False,
            "final_oos_evaluated_once_after_freeze": True,
        },
        "source_dataset": {
            "provider": EXPECTED_PROVIDER,
            "symbol": "BTCUSDT",
            "pair": "BTC/USDT",
            "market": "Spot",
            "interval": "1h",
            "row_count": len(data),
            "csv_sha256": EXPECTED_DATASET_SHA256,
            "producer_commit_sha": EXPECTED_PRODUCER_SHA,
            "start_timestamp_seconds": EXPECTED_START_SECONDS,
            "end_timestamp_seconds": EXPECTED_END_SECONDS,
            "full_history_years": full_years,
            "source_policy": EXPECTED_SOURCE_POLICY,
            "external_backfill_used": False,
            "synthetic_or_interpolated_rows": 0,
        },
        "feature_contract": {
            "feature_names": list(FEATURE_NAMES),
            "normalization": {
                "RSI_14": "RSI centered on 50 and scaled to [-1,1]",
                "MACD_hist": "MACD histogram / ATR, clipped and scaled to [-1,1]",
                "ATR_14": "ATR / close / 0.10, clipped to [0,1]",
                "Close": "log(close / causal EMA100) / 0.10, clipped to [-1,1]",
                "Volume_ZScore": "causal rolling log-volume z-score / 5, clipped to [-1,1]",
                "ROC_6": "six-bar close return / 0.10, clipped to [-1,1]",
                "Range_14": "causal 14-bar mean of (high-low)/close, scaled to [0,1]",
            },
            "future_data_used": False,
            "features_available_after_candle_close": True,
        },
        "primitive_contract": {
            "numeric": ["add", "sub", "mul", "div"],
            "boolean": ["gt", "lt", "and_", "or_"],
            "safe_division_epsilon": 1e-6,
            "numeric_outputs_clipped": [-10.0, 10.0],
        },
        "execution_costs": {
            "maker_fee_bps_per_side": FEE_BPS_PER_SIDE,
            "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        },
        "execution_contract": {
            "market": "Nobitex BTC/USDT Spot",
            "position_mode": "long / flat",
            "leverage": False,
            "shorting": False,
            "entry": "signal after candle close; fill next candle open",
            "exit": "signal false at next open or forced exit after 48 held bars",
            "terminal_liquidation": "final close of each evaluation segment",
            "drawdown_marking": "intrabar high-water mark versus candle low",
            "fills_are_simulated_only": True,
        },
        "temporal_split": {
            "method": "chronological 70/30 with 48-bar purge and 48-bar embargo",
            "split_fraction": 0.70,
            "raw_split_row": split_index,
            "training_end_row_exclusive": training_end,
            "oos_start_row": oos_start,
            "purge_bars": PURGE_BARS,
            "embargo_bars": EMBARGO_BARS,
            "training_first_close_time_ms": int(train_market.close_time_ms[0]),
            "training_last_close_time_ms": int(train_market.close_time_ms[-1]),
            "oos_first_close_time_ms": int(oos_market.close_time_ms[0]),
            "oos_last_close_time_ms": int(oos_market.close_time_ms[-1]),
            "oos_used_in_fitness_or_selection": False,
            "oos_used_for_parameter_or_checkpoint_selection": False,
        },
        "fitness_contract": {
            "minimum_trades_per_year": MIN_TRADES_PER_YEAR,
            "absolute_minimum_trades_3_5y": MIN_ABSOLUTE_TRADES_3_5Y,
            "frequency_shortfall_penalty": "severe",
            "profit_factor_below_1_penalty": "heavy",
            "drawdown_over_15_percent_penalty": "massive",
            "tree_complexity_penalty": "small linear penalty",
            "training_walk_forward_folds": 3,
            "training_only_fitness": True,
        },
        "evolution": {
            "generation_log": generation_log,
            "unique_genomes_evaluated": len(cache),
            "hall_of_fame_size": len(hall_of_fame),
            "best_training_fitness": float(best.fitness.values[0]),
        },
        "best_candidate": {
            "rule": best_rule,
            "rule_sha256": rule_sha,
            "tree_nodes": len(best),
            "tree_height": best.height,
            "training": training_report["training"],
            "training_walk_forward_folds": training_report["folds"],
            "positive_training_folds": training_report["positive_training_folds"],
            "folds_over_drawdown_limit": training_report["folds_over_drawdown_limit"],
        },
        "out_of_sample": oos_metrics.report(),
        "promotion_checks": checks,
        "research_result": ("GO_FOR_OWNER_REVIEW" if all(checks.values()) else "NO-GO"),
        "safety": {
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "ALIX_DRY_RUN": True,
            "ALIX_ALLOW_LIVE_TRADING": False,
            "private_api_calls": 0,
            "exchange_calls": 0,
            "real_order_attempts": 0,
            "orders_or_execution_api_used": False,
            "dataset_provider_calls_during_evolution": 0,
        },
    }


def _write_evolution_artifacts(
    report: dict[str, Any],
    output_dir: Path,
    best_rule: str,
    generation_log: list[dict[str, float | int]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "deap-evolution-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    rule_sha = report["best_candidate"]["rule_sha256"]
    (output_dir / "best-rule.txt").write_text(
        f"{best_rule}\nrule_sha256={rule_sha}\n",
        encoding="utf-8",
    )
    (output_dir / "generation-log.json").write_text(
        json.dumps(generation_log, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "step-summary.md").write_text(
        format_step_summary(report),
        encoding="utf-8",
    )


def run_evolution(
    data: pd.DataFrame,
    features: FeatureSet,
    *,
    output_dir: Path,
    commit_sha: str,
    population_size: int = 500,
    generations: int = 30,
    seed: int = 20_261_111,
) -> dict[str, Any]:
    """Evolve on 70%; evaluate the frozen winner once on the final 30%."""
    split_index, training_end, oos_start = _validate_evolution_request(
        data,
        commit_sha,
        population_size,
        generations,
    )
    (
        _training_positions,
        oos_positions,
        train_market,
        train_features,
        fold_views,
    ) = _prepare_evolution_data(
        data,
        features,
        training_end,
        oos_start,
    )

    primitive_set = build_primitive_set()
    toolbox = _create_deap_toolbox(primitive_set)
    cache: dict[str, tuple[float, dict[str, Any]]] = {}
    toolbox.register(
        "evaluate",
        _bind_evaluator(
            primitive_set,
            train_market,
            train_features,
            fold_views,
            cache,
        ),
    )
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    _population, hall_of_fame, generation_log = _evolve_population(
        toolbox,
        population_size,
        generations,
        cache,
    )
    best, best_rule, training_report = _freeze_best_candidate(
        hall_of_fame,
        cache,
    )
    oos_market, oos_metrics = _evaluate_frozen_oos(
        data,
        features,
        oos_positions,
        best,
        primitive_set,
    )
    checks = _build_promotion_checks(oos_metrics, training_report, best)
    report = _build_evolution_report(
        data,
        commit_sha,
        population_size,
        generations,
        seed,
        split_index,
        training_end,
        oos_start,
        train_market,
        oos_market,
        generation_log,
        cache,
        hall_of_fame,
        best,
        best_rule,
        training_report,
        oos_metrics,
        checks,
    )
    _write_evolution_artifacts(
        report,
        output_dir,
        best_rule,
        generation_log,
    )
    return report


def format_step_summary(report: dict[str, Any]) -> str:
    best = report["best_candidate"]
    oos = report["out_of_sample"]

    def metric(value: Any, percent: bool = False) -> str:
        if value is None:
            return "N/A"
        number = float(value)
        return f"{number * 100.0:.2f}%" if percent else f"{number:.4f}"

    lines = [
        "# AI-2 Revisited — DEAP Genetic Programming",
        "",
        f"Commit SHA: {report['commit_sha']}",
        f"Dataset SHA-256: {report['source_dataset']['csv_sha256']}",
        "Dataset: Nobitex BTC/USDT Spot, 1h, 31,801 pre-hack candles",
        "Cost: 50 bps round trip (25 bps fee/side, 0 bps slippage)",
        "Evolution: population 500 / 30 generations / tournament / one-point crossover / uniform mutation",
        "Split: 70% training; final 30% untouched OOS; 48-bar purge and embargo",
        "",
        "## Best evolved rule",
        "",
        best["rule"],
        "",
        f"Rule SHA-256: {best['rule_sha256']}",
        "",
        "## Untouched out-of-sample metrics",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Net Profit Factor | {metric(oos['net_profit_factor'])} |",
        f"| Win Rate | {metric(oos['win_rate'], True)} |",
        f"| Maximum Drawdown | {metric(oos['max_drawdown'], True)} |",
        f"| Net Return | {metric(oos['net_return'], True)} |",
        f"| Completed Trades | {oos['trade_count']} |",
        f"| Trades per Year | {metric(oos['trades_per_year'])} |",
        f"| OOS duration | {oos['duration_years']:.3f} years |",
        "",
        f"## Research Result: {report['research_result']}",
        "",
        "| Promotion check | Result |",
        "|---|---|",
    ]
    lines.extend(
        f"| {name.replace('_', ' ')} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in report["promotion_checks"].items()
    )
    lines.extend(
        [
            "",
            "Safety: Champion=null; order_permission=false; trading_authorization=NO-GO; "
            "private_api_calls=0; real_order_attempts=0.",
            "No paper or real order was submitted.",
            "",
        ]
    )
    return "\n".join(lines)


def run_cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run bounded DEAP GP research on Nobitex 1h history."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--population", type=int, default=500)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20_261_111)
    args = parser.parse_args()

    data, manifest, digest = load_verified_dataset(args.input_dir)
    features = build_causal_features(data)
    report = run_evolution(
        data,
        features,
        output_dir=args.output_dir,
        commit_sha=args.commit_sha,
        population_size=args.population,
        generations=args.generations,
        seed=args.seed,
    )
    report["source_dataset"]["manifest_source_policy"] = manifest["source_policy"]
    report["source_dataset"]["verified_csv_sha256"] = digest
    (args.output_dir / "deap-evolution-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "step-summary.md").write_text(
        format_step_summary(report),
        encoding="utf-8",
    )


if __name__ == "__main__":
    run_cli()
