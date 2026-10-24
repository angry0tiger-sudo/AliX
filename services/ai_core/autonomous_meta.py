"""Research-only autonomous strategy selection with HMM routing and ATR risk plans.

This module only computes research decisions and historical-bar simulations. It has no
exchange client, order interface, champion assignment, or production promotion path.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from services.ai_core.regime import infer_hmm_regimes
from services.strategy_selection.hmm_training import (
    FEATURE_NAMES,
    REGIME_NAMES,
    calculate_regime_features,
    fit_gaussian_hmm,
)

REGISTRY_SCHEMA = "alix.autonomous-meta-strategy-registry.v1"
STRATEGY_IDS = (
    "trend_following",
    "mean_reversion",
    "volatility_breakout",
    "scalping",
)
OHLCV_COLUMNS = ("date", "open", "high", "low", "close", "volume")
DEFAULT_INITIAL_TRAIN_ROWS = 3 * 365 * 24
DEFAULT_FOLD_COUNT = 3
DEFAULT_MINIMUM_CONFIDENCE = 0.55
DEFAULT_FEE_BPS_PER_SIDE = 25.0
DEFAULT_SLIPPAGE_BPS_PER_SIDE = 20.0
DEFAULT_RISK_FRACTION = 0.005
DEFAULT_MAX_POSITION_FRACTION = 0.95
MINIMUM_TEST_ROWS_PER_FOLD = 30
_REGISTRY_ROOT_KEYS = {"schema_version", "revision", "code_sha256", "strategies"}
_REGISTRY_ENTRY_KEYS = {
    "strategy_id",
    "version",
    "lifecycle",
    "owner_approved",
    "allow_short",
    "timeframe",
    "eligible_regimes",
    "required_features",
    "requires_positive_volume",
    "minimum_relative_volume",
    "max_holding_bars",
    "atr_stop_multiple",
    "atr_target_multiple",
}


@dataclass(frozen=True, slots=True)
class StrategySpec:
    strategy_id: str
    version: str
    lifecycle: str
    owner_approved: bool
    allow_short: bool
    timeframe: str
    eligible_regimes: tuple[str, ...]
    required_features: tuple[str, ...]
    requires_positive_volume: bool
    minimum_relative_volume: float
    max_holding_bars: int
    atr_stop_multiple: float
    atr_target_multiple: float

    def __post_init__(self) -> None:
        if self.strategy_id not in STRATEGY_IDS:
            raise ValueError("autonomous_strategy_id_unknown")
        if self.lifecycle != "RESEARCH_ONLY" or self.owner_approved or self.allow_short:
            raise ValueError("autonomous_strategy_must_remain_research_only")
        if self.timeframe != "1h" or not self.eligible_regimes:
            raise ValueError("autonomous_strategy_contract_invalid")
        if not set(self.eligible_regimes).issubset(REGIME_NAMES):
            raise ValueError("autonomous_strategy_regime_invalid")
        if not self.required_features or self.max_holding_bars < 1:
            raise ValueError("autonomous_strategy_features_or_horizon_invalid")
        numeric_values = (
            self.minimum_relative_volume,
            self.atr_stop_multiple,
            self.atr_target_multiple,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("autonomous_strategy_parameters_non_finite")
        if (
            self.minimum_relative_volume < 0
            or self.atr_stop_multiple <= 0
            or self.atr_target_multiple <= 0
        ):
            raise ValueError("autonomous_strategy_parameters_out_of_range")


@dataclass(frozen=True, slots=True)
class StrategyRegistry:
    revision: str
    code_sha256: str
    document_sha256: str
    strategies: tuple[StrategySpec, ...]

    def __post_init__(self) -> None:
        identifiers = tuple(item.strategy_id for item in self.strategies)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("autonomous_registry_duplicate_strategy_ids")
        if set(identifiers) != set(STRATEGY_IDS):
            raise ValueError("autonomous_registry_candidate_set_mismatch")
        if not self.revision.strip():
            raise ValueError("autonomous_registry_revision_missing")


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    initial_train_rows: int = DEFAULT_INITIAL_TRAIN_ROWS
    fold_count: int = DEFAULT_FOLD_COUNT
    minimum_confidence: float = DEFAULT_MINIMUM_CONFIDENCE
    fee_bps_per_side: float = DEFAULT_FEE_BPS_PER_SIDE
    slippage_bps_per_side: float = DEFAULT_SLIPPAGE_BPS_PER_SIDE
    risk_fraction: float = DEFAULT_RISK_FRACTION
    max_position_fraction: float = DEFAULT_MAX_POSITION_FRACTION
    initial_equity: float = 10_000.0
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.initial_train_rows < 100 or self.fold_count < 1:
            raise ValueError("autonomous_walk_forward_config_invalid")
        finite = (
            self.minimum_confidence,
            self.fee_bps_per_side,
            self.slippage_bps_per_side,
            self.risk_fraction,
            self.max_position_fraction,
            self.initial_equity,
        )
        if not all(math.isfinite(value) for value in finite):
            raise ValueError("autonomous_walk_forward_config_non_finite")
        if (
            not 0.5 <= self.minimum_confidence <= 1
            or min(self.fee_bps_per_side, self.slippage_bps_per_side) < 0
            or not 0 < self.risk_fraction <= 0.02
            or not 0 < self.max_position_fraction <= 1
            or self.initial_equity <= 0
        ):
            raise ValueError("autonomous_walk_forward_config_out_of_range")


@dataclass(frozen=True, slots=True)
class StrategySelection:
    regime: str | None
    confidence: float | None
    strategy_id: str | None
    status: str
    reason: str
    order_permission: bool = False
    champion: None = None

    def __post_init__(self) -> None:
        if self.order_permission or self.champion is not None:
            raise ValueError("autonomous_selection_cannot_authorize_or_promote")


@dataclass(frozen=True, slots=True)
class DynamicRiskPlan:
    entry_price: float
    atr: float
    stop_loss: float
    take_profit: float
    quantity: float
    notional: float
    risk_budget: float
    stop_distance: float
    target_distance: float
    confidence: float
    order_permission: bool = False

    def __post_init__(self) -> None:
        if self.order_permission:
            raise ValueError("autonomous_risk_plan_cannot_authorize_orders")
        if not (self.stop_loss < self.entry_price < self.take_profit):
            raise ValueError("autonomous_risk_plan_price_order_invalid")
        if min(self.quantity, self.notional, self.risk_budget) <= 0:
            raise ValueError("autonomous_risk_plan_size_invalid")


def _require_mapping(value: object, reason: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(reason)
    return value


def _registry_text(value: object, reason: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(reason)
    return value


def _registry_float(value: object, reason: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(reason)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(reason)
    return result


def _registry_integer(value: object, reason: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(reason)
    return value


def _parse_registry_entry(raw: object) -> StrategySpec:
    entry = _require_mapping(raw, "autonomous_registry_entry_invalid")
    if set(entry) != _REGISTRY_ENTRY_KEYS:
        raise ValueError("autonomous_registry_entry_schema_invalid")
    regimes = entry["eligible_regimes"]
    required = entry["required_features"]
    if (
        not isinstance(regimes, list)
        or not all(isinstance(item, str) and item for item in regimes)
        or not isinstance(required, list)
        or not all(isinstance(item, str) and item for item in required)
    ):
        raise ValueError("autonomous_registry_lists_invalid")
    if not isinstance(entry["owner_approved"], bool):
        raise ValueError("autonomous_registry_approval_flag_invalid")
    if not isinstance(entry["allow_short"], bool):
        raise ValueError("autonomous_registry_short_flag_invalid")
    if not isinstance(entry["requires_positive_volume"], bool):
        raise ValueError("autonomous_registry_volume_flag_invalid")
    return StrategySpec(
        strategy_id=_registry_text(entry["strategy_id"], "autonomous_registry_id_invalid"),
        version=_registry_text(entry["version"], "autonomous_registry_version_invalid"),
        lifecycle=_registry_text(entry["lifecycle"], "autonomous_registry_lifecycle_invalid"),
        owner_approved=entry["owner_approved"],
        allow_short=entry["allow_short"],
        timeframe=_registry_text(entry["timeframe"], "autonomous_registry_timeframe_invalid"),
        eligible_regimes=tuple(regimes),
        required_features=tuple(required),
        requires_positive_volume=entry["requires_positive_volume"],
        minimum_relative_volume=_registry_float(
            entry["minimum_relative_volume"], "autonomous_registry_relative_volume_invalid"
        ),
        max_holding_bars=_registry_integer(
            entry["max_holding_bars"], "autonomous_registry_holding_horizon_invalid"
        ),
        atr_stop_multiple=_registry_float(
            entry["atr_stop_multiple"], "autonomous_registry_stop_multiple_invalid"
        ),
        atr_target_multiple=_registry_float(
            entry["atr_target_multiple"], "autonomous_registry_target_multiple_invalid"
        ),
    )


def _parse_registry_document(document: str, expected_code_sha256: str) -> StrategyRegistry:
    try:
        raw_object = json.loads(document)
    except json.JSONDecodeError as error:
        raise ValueError("autonomous_registry_json_invalid") from error
    raw = _require_mapping(raw_object, "autonomous_registry_root_invalid")
    if set(raw) != _REGISTRY_ROOT_KEYS or raw.get("schema_version") != REGISTRY_SCHEMA:
        raise ValueError("autonomous_registry_schema_invalid")
    revision = _registry_text(raw.get("revision"), "autonomous_registry_revision_invalid")
    code_hash = _registry_text(raw.get("code_sha256"), "autonomous_registry_code_hash_invalid")
    if code_hash != expected_code_sha256:
        raise ValueError("autonomous_registry_code_hash_mismatch")
    entries = raw.get("strategies")
    if not isinstance(entries, list) or len(entries) != len(STRATEGY_IDS):
        raise ValueError("autonomous_registry_strategy_count_invalid")
    strategies = tuple(_parse_registry_entry(item) for item in entries)
    return StrategyRegistry(
        revision=revision,
        code_sha256=code_hash,
        document_sha256=hashlib.sha256(document.encode("utf-8")).hexdigest(),
        strategies=strategies,
    )


def load_strategy_registry(path: Path) -> StrategyRegistry:
    if not path.is_file():
        raise ValueError("autonomous_registry_file_missing")
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    registry = _parse_registry_document(path.read_text(encoding="utf-8"), code_hash)
    by_id = {item.strategy_id: item for item in registry.strategies}
    expected_regimes = {
        "trend_following": ("trend",),
        "mean_reversion": ("range",),
        "volatility_breakout": ("high_volatility",),
        "scalping": ("range",),
    }
    for strategy_id, regimes in expected_regimes.items():
        if by_id[strategy_id].eligible_regimes != regimes:
            raise ValueError("autonomous_registry_regime_contract_mismatch")
    if not by_id["scalping"].requires_positive_volume:
        raise ValueError("autonomous_scalping_volume_gate_missing")
    return registry


def _finite_number(row: Mapping[str, object], name: str) -> float | None:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _validate_hourly_candles(candles: DataFrame) -> DataFrame:
    missing = sorted(set(OHLCV_COLUMNS) - set(candles.columns))
    if missing:
        raise ValueError(f"autonomous_ohlcv_columns_missing:{missing}")
    if candles.empty or candles.columns.duplicated().any():
        raise ValueError("autonomous_ohlcv_empty_or_duplicate_columns")
    result = candles.loc[:, list(OHLCV_COLUMNS)].copy()
    dates = pd.to_datetime(result["date"], errors="raise")
    if dates.dt.tz is None:
        raise ValueError("autonomous_timestamps_must_be_timezone_aware")
    result["date"] = dates.dt.tz_convert("UTC")
    if result["date"].duplicated().any() or not result["date"].is_monotonic_increasing:
        raise ValueError("autonomous_timestamps_duplicate_or_non_monotonic")
    if not result["date"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("autonomous_hourly_timestamps_not_contiguous")
    for column in ("open", "high", "low", "close", "volume"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    prices = result[["open", "high", "low", "close"]].to_numpy(dtype=float)
    volume = result["volume"].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or not np.isfinite(volume).all():
        raise ValueError("autonomous_ohlcv_non_finite")
    if (prices <= 0).any() or (volume < 0).any():
        raise ValueError("autonomous_ohlcv_out_of_range")
    if (
        (result["high"] < result[["open", "close"]].max(axis=1)).any()
        or (result["low"] > result[["open", "close"]].min(axis=1)).any()
        or (result["high"] < result["low"]).any()
    ):
        raise ValueError("autonomous_ohlc_invariant_failed")
    return result.reset_index(drop=True)


def _relative_strength_index(close: Series, period: int = 14) -> Series:
    delta = close.diff()
    gains = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    losses = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    strength = gains / losses.where(losses > 0)
    rsi = 100.0 - 100.0 / (1.0 + strength)
    rsi = rsi.mask((losses == 0) & (gains > 0), 100.0)
    return rsi.mask((losses == 0) & (gains == 0), 50.0)


def compute_autonomous_features(candles: DataFrame) -> DataFrame:
    """Build causal 1H features; rolling extrema exclude the current bar."""
    bars = _validate_hourly_candles(candles)
    close = bars["close"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    volume = bars["volume"].astype(float)
    output = bars.copy()
    output["ema_fast"] = close.ewm(span=8, adjust=False, min_periods=8).mean()
    output["ema_slow"] = close.ewm(span=21, adjust=False, min_periods=21).mean()
    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    output["macd"] = ema_12 - ema_26
    output["macd_signal"] = output["macd"].ewm(
        span=9, adjust=False, min_periods=9
    ).mean()
    output["rsi_14"] = _relative_strength_index(close)
    middle = close.rolling(20, min_periods=20).mean()
    deviation = close.rolling(20, min_periods=20).std(ddof=0)
    output["bollinger_lower"] = middle - 2.0 * deviation
    output["bollinger_upper"] = middle + 2.0 * deviation
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    output["atr_14"] = true_range.rolling(14, min_periods=14).mean()
    output["atr_ratio"] = output["atr_14"] / close
    output["previous_high_20"] = high.shift(1).rolling(20, min_periods=20).max()
    output["previous_low_20"] = low.shift(1).rolling(20, min_periods=20).min()
    output["volume_mean_20"] = volume.rolling(20, min_periods=20).mean()
    output["relative_volume"] = volume / output["volume_mean_20"].where(
        output["volume_mean_20"] > 0
    )
    output["return_1h"] = close.pct_change(fill_method=None)
    output["momentum_3h"] = close.pct_change(3, fill_method=None)
    return output


def generate_entry_signal(
    strategy_id: str,
    row: Mapping[str, object],
    *,
    source_volume_available: bool,
    minimum_relative_volume: float = 1.5,
) -> bool:
    """Return a long-only research entry flag for one closed candle."""
    if strategy_id not in STRATEGY_IDS:
        raise ValueError("autonomous_strategy_id_unknown")
    close = _finite_number(row, "close")
    if close is None:
        return False
    if strategy_id == "trend_following":
        fast = _finite_number(row, "ema_fast")
        slow = _finite_number(row, "ema_slow")
        macd = _finite_number(row, "macd")
        signal = _finite_number(row, "macd_signal")
        return fast is not None and slow is not None and macd is not None and signal is not None and close > fast > slow and macd > signal
    if strategy_id == "mean_reversion":
        rsi = _finite_number(row, "rsi_14")
        lower = _finite_number(row, "bollinger_lower")
        return rsi is not None and lower is not None and rsi <= 30.0 and close <= lower
    if strategy_id == "volatility_breakout":
        previous_high = _finite_number(row, "previous_high_20")
        return previous_high is not None and close > previous_high
    volume = _finite_number(row, "volume")
    relative_volume = _finite_number(row, "relative_volume")
    fast = _finite_number(row, "ema_fast")
    momentum = _finite_number(row, "momentum_3h")
    return (
        source_volume_available
        and volume is not None
        and volume > 0
        and relative_volume is not None
        and relative_volume >= minimum_relative_volume
        and fast is not None
        and momentum is not None
        and close > fast
        and momentum > 0
    )


def _registry_by_id(registry: StrategyRegistry) -> dict[str, StrategySpec]:
    return {spec.strategy_id: spec for spec in registry.strategies}


def choose_strategy(
    regime: str | None,
    confidence: float | None,
    row: Mapping[str, object],
    registry: StrategyRegistry,
    *,
    source_volume_available: bool,
    minimum_confidence: float = DEFAULT_MINIMUM_CONFIDENCE,
) -> StrategySelection:
    """Select a registry entry from a confident HMM regime, or explicitly abstain."""
    if regime not in REGIME_NAMES:
        return StrategySelection(regime, confidence, None, "NO_TRADE", "regime_missing_or_unknown")
    if confidence is None or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        return StrategySelection(regime, confidence, None, "NO_TRADE", "confidence_invalid")
    if confidence < minimum_confidence:
        return StrategySelection(regime, confidence, None, "NO_TRADE", "hmm_confidence_below_threshold")
    specs = _registry_by_id(registry)
    if set(specs) != set(STRATEGY_IDS):
        return StrategySelection(regime, confidence, None, "NO_TRADE", "registry_candidate_set_invalid")
    selected_id = {
        "trend": "trend_following",
        "high_volatility": "volatility_breakout",
    }.get(regime)
    if regime == "range":
        scalp_spec = specs["scalping"]
        volume = _finite_number(row, "volume")
        relative_volume = _finite_number(row, "relative_volume")
        scalp_is_eligible = (
            source_volume_available
            and volume is not None
            and volume > 0
            and relative_volume is not None
            and relative_volume >= scalp_spec.minimum_relative_volume
        )
        selected_id = "scalping" if scalp_is_eligible else "mean_reversion"
    if selected_id is None or regime not in specs[selected_id].eligible_regimes:
        return StrategySelection(regime, confidence, None, "NO_TRADE", "regime_has_no_valid_candidate")
    if specs[selected_id].requires_positive_volume and not source_volume_available:
        return StrategySelection(regime, confidence, None, "NO_TRADE", "source_volume_not_observed")
    return StrategySelection(regime, confidence, selected_id, "SELECTED_RESEARCH_ONLY", "hmm_regime_candidate_selected")


def make_research_decision(
    regime: str | None,
    confidence: float | None,
    row: Mapping[str, object],
    registry: StrategyRegistry,
    *,
    source_volume_available: bool,
    minimum_confidence: float = DEFAULT_MINIMUM_CONFIDENCE,
) -> StrategySelection:
    selection = choose_strategy(
        regime,
        confidence,
        row,
        registry,
        source_volume_available=source_volume_available,
        minimum_confidence=minimum_confidence,
    )
    if selection.strategy_id is None:
        return selection
    spec = _registry_by_id(registry)[selection.strategy_id]
    has_signal = generate_entry_signal(
        selection.strategy_id,
        row,
        source_volume_available=source_volume_available,
        minimum_relative_volume=spec.minimum_relative_volume,
    )
    if not has_signal:
        return StrategySelection(
            regime,
            confidence,
            selection.strategy_id,
            "NO_SIGNAL",
            "selected_strategy_entry_conditions_not_met",
        )
    return StrategySelection(
        regime,
        confidence,
        selection.strategy_id,
        "SIGNAL_RESEARCH_ONLY",
        "hmm_selected_strategy_entry_signal",
    )


def plan_dynamic_risk(
    *,
    entry_price: float,
    atr: float,
    regime: str,
    confidence: float,
    equity: float,
    spec: StrategySpec,
    config: WalkForwardConfig,
) -> DynamicRiskPlan | None:
    """Size a simulated long position by ATR distance, equity budget and HMM score."""
    values = (entry_price, atr, confidence, equity)
    if not all(math.isfinite(value) for value in values):
        return None
    if entry_price <= 0 or atr <= 0 or equity <= 0:
        return None
    if confidence < config.minimum_confidence or confidence > 1:
        return None
    stop_adjustment, target_adjustment = {
        "trend": (1.0, 1.0),
        "range": (0.9, 0.9),
        "high_volatility": (1.3, 1.2),
    }.get(regime, (1.0, 1.0))
    stop_distance = atr * spec.atr_stop_multiple * stop_adjustment
    target_distance = atr * spec.atr_target_multiple * target_adjustment
    if min(stop_distance, target_distance) <= 0:
        return None
    risk_budget = equity * config.risk_fraction * confidence
    quantity_by_risk = risk_budget / stop_distance
    quantity_by_notional = equity * config.max_position_fraction / entry_price
    quantity = min(quantity_by_risk, quantity_by_notional)
    if not math.isfinite(quantity) or quantity <= 0:
        return None
    return DynamicRiskPlan(
        entry_price=entry_price,
        atr=atr,
        stop_loss=entry_price - stop_distance,
        take_profit=entry_price + target_distance,
        quantity=quantity,
        notional=quantity * entry_price,
        risk_budget=risk_budget,
        stop_distance=stop_distance,
        target_distance=target_distance,
        confidence=confidence,
    )


def make_walk_forward_folds(
    row_count: int, config: WalkForwardConfig
) -> tuple[tuple[int, int, int], ...]:
    """Make expanding chronological train/test folds; the final reserved holdout is absent."""
    remainder = row_count - config.initial_train_rows
    window = remainder // config.fold_count
    if window < MINIMUM_TEST_ROWS_PER_FOLD:
        raise ValueError("autonomous_walk_forward_rows_insufficient")
    folds: list[tuple[int, int, int]] = []
    for fold_index in range(config.fold_count):
        test_start = config.initial_train_rows + fold_index * window
        test_stop = row_count if fold_index == config.fold_count - 1 else test_start + window
        folds.append((test_start, test_start, test_stop))
    return tuple(folds)


def _regime_predictions_for_folds(
    regime_features: DataFrame,
    folds: tuple[tuple[int, int, int], ...],
    config: WalkForwardConfig,
) -> tuple[dict[int, tuple[str, float]], list[dict[str, Any]]]:
    predictions: dict[int, tuple[str, float]] = {}
    fold_reports: list[dict[str, Any]] = []
    for fold_index, (train_stop, test_start, test_stop) in enumerate(folds, start=1):
        training = regime_features.iloc[:train_stop].dropna()
        testing = regime_features.iloc[test_start:test_stop].dropna()
        fitted = fit_gaussian_hmm(training.loc[:, list(FEATURE_NAMES)], random_seed=config.random_seed)
        regimes, posteriors = infer_hmm_regimes(testing.loc[:, list(FEATURE_NAMES)], fitted)
        for index, regime, posterior in zip(testing.index, regimes, posteriors, strict=True):
            predictions[int(index)] = (regime, max(posterior))
        fold_reports.append(
            {
                "fold": fold_index,
                "training_rows": len(training),
                "test_start": test_start,
                "test_stop_exclusive": test_stop,
                "test_rows_classified": len(testing),
                "hmm_mean_log_likelihood": fitted.mean_log_likelihood,
                "hmm_iterations": fitted.iterations,
                "state_occupancy": dict(zip(REGIME_NAMES, fitted.state_occupancy, strict=True)),
            }
        )
    return predictions, fold_reports


def _exit_fill(
    candles: DataFrame,
    *,
    entry_index: int,
    test_stop: int,
    plan: DynamicRiskPlan,
    max_holding_bars: int,
) -> tuple[int, float, str]:
    last_index = min(test_stop - 1, entry_index + max_holding_bars - 1)
    for index in range(entry_index, last_index + 1):
        opening = float(candles["open"].iloc[index])
        high = float(candles["high"].iloc[index])
        low = float(candles["low"].iloc[index])
        if opening <= plan.stop_loss:
            return index, opening, "STOP_GAP"
        if opening >= plan.take_profit:
            return index, plan.take_profit, "TAKE_PROFIT_GAP"
        stop_hit = low <= plan.stop_loss
        target_hit = high >= plan.take_profit
        if stop_hit:
            return index, plan.stop_loss, "STOP_LOSS_CONSERVATIVE_TIE_BREAK"
        if target_hit:
            return index, plan.take_profit, "TAKE_PROFIT"
    return last_index, float(candles["close"].iloc[last_index]), "TIME_EXIT"


def _net_trade_result(
    entry_price: float,
    exit_price: float,
    quantity: float,
    config: WalkForwardConfig,
) -> tuple[float, float]:
    gross_pnl = quantity * (exit_price - entry_price)
    cost_rate = (config.fee_bps_per_side + config.slippage_bps_per_side) / 10_000.0
    costs = quantity * (entry_price + exit_price) * cost_rate
    return gross_pnl - costs, costs



def _run_policy(
    policy_id: str | None,
    candles: DataFrame,
    features: DataFrame,
    folds: tuple[tuple[int, int, int], ...],
    regime_predictions: dict[int, tuple[str, float]],
    registry: StrategyRegistry,
    *,
    source_volume_available: bool,
    config: WalkForwardConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    specs = _registry_by_id(registry)
    initial_equity = config.initial_equity
    equity = initial_equity
    equity_curve = [equity]
    trades: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    switch_count = 0
    last_selected: str | None = None
    selection_counts = {strategy_id: 0 for strategy_id in STRATEGY_IDS}
    for fold_index, (_, test_start, test_stop) in enumerate(folds, start=1):
        cursor = test_start
        while cursor < test_stop - 1:
            timestamp = str(candles["date"].iloc[cursor])
            regime_info = regime_predictions.get(cursor)
            if regime_info is None:
                if policy_id is None:
                    decisions.append(
                        {
                            "fold": fold_index,
                            "timestamp": timestamp,
                            "regime": None,
                            "hmm_confidence": None,
                            "strategy_id": None,
                            "status": "NO_TRADE",
                            "reason": "hmm_regime_unavailable",
                            "entry_signal": False,
                            "order_permission": False,
                        }
                    )
                cursor += 1
                continue
            regime, confidence = regime_info
            row = features.iloc[cursor]
            if policy_id is None:
                selection = choose_strategy(
                    regime,
                    confidence,
                    row,
                    registry,
                    source_volume_available=source_volume_available,
                    minimum_confidence=config.minimum_confidence,
                )
                strategy_id = selection.strategy_id
            else:
                strategy_id = policy_id
                selection = StrategySelection(
                    regime,
                    confidence,
                    strategy_id,
                    "FIXED_RESEARCH_BASELINE",
                    "predeclared_fixed_strategy_baseline",
                )
            if strategy_id is None:
                if policy_id is None:
                    decisions.append(
                        {
                            "fold": fold_index,
                            "timestamp": timestamp,
                            "regime": regime,
                            "hmm_confidence": confidence,
                            "strategy_id": None,
                            "status": selection.status,
                            "reason": selection.reason,
                            "entry_signal": False,
                            "order_permission": False,
                        }
                    )
                cursor += 1
                continue
            if confidence < config.minimum_confidence:
                if policy_id is None:
                    decisions.append(
                        {
                            "fold": fold_index,
                            "timestamp": timestamp,
                            "regime": regime,
                            "hmm_confidence": confidence,
                            "strategy_id": strategy_id,
                            "status": "NO_TRADE",
                            "reason": "hmm_confidence_below_threshold",
                            "entry_signal": False,
                            "order_permission": False,
                        }
                    )
                cursor += 1
                continue
            spec = specs[strategy_id]
            selection_counts[strategy_id] += 1
            if last_selected is not None and strategy_id != last_selected:
                switch_count += 1
            last_selected = strategy_id
            has_signal = generate_entry_signal(
                strategy_id,
                row,
                source_volume_available=source_volume_available,
                minimum_relative_volume=spec.minimum_relative_volume,
            )
            if policy_id is None:
                decisions.append(
                    {
                        "fold": fold_index,
                        "timestamp": timestamp,
                        "regime": regime,
                        "hmm_confidence": confidence,
                        "strategy_id": strategy_id,
                        "status": "ENTRY_SIGNAL" if has_signal else "NO_SIGNAL",
                        "reason": (
                            "hmm_selected_strategy_entry_signal"
                            if has_signal
                            else "selected_strategy_entry_conditions_not_met"
                        ),
                        "entry_signal": has_signal,
                        "order_permission": False,
                    }
                )
            if not has_signal:
                cursor += 1
                continue
            entry_index = cursor + 1
            atr = _finite_number(row, "atr_14")
            if atr is None:
                cursor += 1
                continue
            entry_price = float(candles["open"].iloc[entry_index])
            plan = plan_dynamic_risk(
                entry_price=entry_price,
                atr=atr,
                regime=regime,
                confidence=confidence,
                equity=equity,
                spec=spec,
                config=config,
            )
            if plan is None:
                cursor += 1
                continue
            exit_index, exit_price, exit_reason = _exit_fill(
                candles,
                entry_index=entry_index,
                test_stop=test_stop,
                plan=plan,
                max_holding_bars=spec.max_holding_bars,
            )
            net_pnl, costs = _net_trade_result(
                entry_price, exit_price, plan.quantity, config
            )
            equity_before = equity
            equity = max(0.0, equity + net_pnl)
            net_return_bps = net_pnl / plan.notional * 10_000.0
            trades.append(
                {
                    "policy": "autonomous_meta" if policy_id is None else policy_id,
                    "fold": fold_index,
                    "strategy_id": strategy_id,
                    "regime": regime,
                    "hmm_confidence": confidence,
                    "signal_time": timestamp,
                    "entry_time": str(candles["date"].iloc[entry_index]),
                    "exit_time": str(candles["date"].iloc[exit_index]),
                    "entry_index": entry_index,
                    "exit_index": exit_index,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "atr_at_signal": atr,
                    "stop_loss": plan.stop_loss,
                    "take_profit": plan.take_profit,
                    "quantity": plan.quantity,
                    "notional_quote": plan.notional,
                    "risk_budget_quote": plan.risk_budget,
                    "gross_pnl_quote": plan.quantity * (exit_price - entry_price),
                    "estimated_cost_quote": costs,
                    "net_pnl_quote": net_pnl,
                    "net_return_bps": net_return_bps,
                    "won_net_of_costs": net_pnl > 0,
                    "exit_reason": exit_reason,
                    "equity_before_quote": equity_before,
                    "equity_after_quote": equity,
                    "order_permission": False,
                    "real_order_attempted": False,
                }
            )
            equity_curve.append(equity)
            cursor = exit_index
    metrics = summarize_policy_metrics(
        trades,
        equity_curve,
        initial_equity=initial_equity,
        switch_count=switch_count,
        selection_counts=selection_counts,
    )
    return metrics, trades, decisions

def summarize_policy_metrics(
    trades: list[dict[str, Any]],
    equity_curve: list[float],
    *,
    initial_equity: float,
    switch_count: int = 0,
    selection_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    returns = [float(trade["net_return_bps"]) for trade in trades]
    wins = [value for value in returns if value > 0]
    losses = [value for value in returns if value <= 0]
    win_rate = 100.0 * len(wins) / len(returns) if returns else None
    expectancy = float(np.mean(returns)) if returns else None
    avg_win = float(np.mean(wins)) if wins else None
    avg_loss = float(np.mean(losses)) if losses else None
    peak = initial_equity
    max_drawdown = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - value) / peak * 100.0)
    final_equity = equity_curve[-1] if equity_curve else initial_equity
    counts = {strategy_id: 0 for strategy_id in STRATEGY_IDS}
    for trade in trades:
        counts[str(trade["strategy_id"])] += 1
    return {
        "status": "METRICS_AVAILABLE" if trades else "NO_TRADES",
        "trade_count": len(trades),
        "win_rate_pct": win_rate,
        "expectancy_bps_per_trade": expectancy,
        "average_win_bps": avg_win,
        "average_loss_bps": avg_loss,
        "total_net_pnl_quote": float(sum(float(trade["net_pnl_quote"]) for trade in trades)),
        "total_return_pct": (final_equity / initial_equity - 1.0) * 100.0,
        "max_drawdown_pct": max_drawdown,
        "strategy_trade_counts": counts,
        "selector_strategy_checks": selection_counts or {strategy_id: 0 for strategy_id in STRATEGY_IDS},
        "strategy_switch_count": switch_count,
    }


def _comparisons(
    meta: dict[str, Any], baselines: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    comparisons: dict[str, dict[str, Any]] = {}
    for strategy_id, baseline in baselines.items():
        meta_win = meta["win_rate_pct"]
        base_win = baseline["win_rate_pct"]
        meta_expectancy = meta["expectancy_bps_per_trade"]
        base_expectancy = baseline["expectancy_bps_per_trade"]
        available = all(
            value is not None
            for value in (meta_win, base_win, meta_expectancy, base_expectancy)
        )
        if not available:
            comparisons[strategy_id] = {
                "status": "INSUFFICIENT_TRADES_FOR_COMPARISON",
                "win_rate_delta_percentage_points": None,
                "expectancy_delta_bps": None,
                "improved_win_rate": None,
                "improved_expectancy": None,
                "improved_both": None,
            }
            continue
        win_delta = float(meta_win) - float(base_win)
        expectancy_delta = float(meta_expectancy) - float(base_expectancy)
        comparisons[strategy_id] = {
            "status": "COMPARABLE_EXPLORATORY_OOS_METRICS",
            "win_rate_delta_percentage_points": win_delta,
            "expectancy_delta_bps": expectancy_delta,
            "improved_win_rate": win_delta > 0,
            "improved_expectancy": expectancy_delta > 0,
            "improved_both": win_delta > 0 and expectancy_delta > 0,
        }
    return comparisons



def evaluate_historical_bars(
    candles: DataFrame,
    registry: StrategyRegistry,
    *,
    commit_sha: str,
    source: Mapping[str, Any],
    config: WalkForwardConfig | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate dynamic regime selection against four fixed-policy baselines."""
    selected_config = config or WalkForwardConfig()
    if len(commit_sha) != 40 or any(character not in "0123456789abcdef" for character in commit_sha):
        raise ValueError("autonomous_commit_sha_invalid")
    bars = _validate_hourly_candles(candles)
    features = compute_autonomous_features(bars)
    regime_features = calculate_regime_features(bars)
    folds = make_walk_forward_folds(len(bars), selected_config)
    regime_predictions, fold_reports = _regime_predictions_for_folds(
        regime_features, folds, selected_config
    )
    volume_positive_rows = int((bars["volume"] > 0).sum())
    source_volume_available = volume_positive_rows > 0 and bool(source.get("volume_observed", True))
    meta_metrics, meta_trades, decisions = _run_policy(
        None,
        bars,
        features,
        folds,
        regime_predictions,
        registry,
        source_volume_available=source_volume_available,
        config=selected_config,
    )
    baseline_metrics: dict[str, dict[str, Any]] = {}
    all_trades = list(meta_trades)
    for strategy_id in STRATEGY_IDS:
        metrics, trades, _ = _run_policy(
            strategy_id,
            bars,
            features,
            folds,
            regime_predictions,
            registry,
            source_volume_available=source_volume_available,
            config=selected_config,
        )
        baseline_metrics[strategy_id] = metrics
        all_trades.extend(trades)
    comparison = _comparisons(meta_metrics, baseline_metrics)
    scalping_status = (
        "ENABLED_SUBJECT_TO_PER_BAR_VOLUME_AND_RELATIVE_VOLUME_GATES"
        if source_volume_available
        else "BLOCKED_NO_POSITIVE_SOURCE_VOLUME"
    )
    report = {
        "schema_version": "alix.autonomous-meta-strategy-report.v1",
        "created_by_commit_sha": commit_sha,
        "run_status": "COMPLETED_RESEARCH_ONLY",
        "source": dict(source),
        "walk_forward": {
            "timeframe": "1h",
            "fold_count": len(fold_reports),
            "folds": fold_reports,
            "final_holdout_used": False,
            "training_policy": "EXPANDING_CHRONOLOGICAL_TRAIN_WINDOWS",
            "regime_feature_names": list(FEATURE_NAMES),
        },
        "strategy_registry": {
            "revision": registry.revision,
            "document_sha256": registry.document_sha256,
            "code_sha256": registry.code_sha256,
            "candidate_count": len(registry.strategies),
            "candidate_ids": [spec.strategy_id for spec in registry.strategies],
            "all_candidates_research_only": all(
                spec.lifecycle == "RESEARCH_ONLY" and not spec.owner_approved
                for spec in registry.strategies
            ),
            "production_registry_modified": False,
            "scalping_status": scalping_status,
        },
        "selector": {
            "method": "HMM_REGIME_PLUS_CONFIDENCE_AND_DATA_ELIGIBILITY",
            "minimum_confidence": selected_config.minimum_confidence,
            "confidence_semantics": (
                "maximum forward-filtered HMM posterior; not calibrated as a probability "
                "and used only as a bounded research sizing scalar"
            ),
            "risk_policy": "ATR_DYNAMIC_STOPS_TARGETS_AND_CONFIDENCE_SCALED_RISK_BUDGET",
        },
        "economic_evaluation": {
            "status": "ASSUMPTION_BASED_HISTORICAL_BAR_SIMULATION",
            "economic_acceptance": "NO-GO",
            "fee_bps_per_side": selected_config.fee_bps_per_side,
            "slippage_bps_per_side": selected_config.slippage_bps_per_side,
            "spread_observed": False,
            "orderbook_liquidity_observed": False,
            "win_rate_definition": "net_pnl_quote_positive_after_assumed_fees_and_slippage",
            "expectancy_definition": "mean_net_return_bps_per_completed_trade",
            "meta_strategy_metrics": meta_metrics,
            "fixed_strategy_baselines": baseline_metrics,
            "comparison_vs_each_fixed_strategy": comparison,
            "reason": (
                "This is an exploratory simulation on verified historical OHLCV bars, "
                "not exchange-validated execution evidence. Volume and order-book availability "
                "gates remain binding; no profitability or trading approval is inferred."
            ),
        },
        "safety_invariants": {
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "economic_acceptance": "NO-GO",
            "model_promoted": False,
            "production_model_updated": False,
            "production_strategy_registry_updated": False,
            "orders_or_execution_api_used": False,
            "private_api_calls": 0,
            "real_order_attempts": 0,
        },
    }
    return report, all_trades, decisions

__all__ = [
    "DEFAULT_FOLD_COUNT",
    "DEFAULT_INITIAL_TRAIN_ROWS",
    "DEFAULT_MINIMUM_CONFIDENCE",
    "DynamicRiskPlan",
    "REGISTRY_SCHEMA",
    "STRATEGY_IDS",
    "StrategyRegistry",
    "StrategySelection",
    "StrategySpec",
    "WalkForwardConfig",
    "choose_strategy",
    "compute_autonomous_features",
    "evaluate_historical_bars",
    "generate_entry_signal",
    "load_strategy_registry",
    "make_research_decision",
    "make_walk_forward_folds",
    "plan_dynamic_risk",
    "summarize_policy_metrics",
]
