"""Regime-routed strategy proposals that cannot authorize orders or promotion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping

from services.strategy_selection.contracts import (
    StrategyLifecycle,
)
from services.strategy_selection.regime import (
    RegimeInference,
    RegimeInferenceStatus,
)
from services.strategy_selection.registry import StrategyRegistry

REGIME_STRATEGY_MAP: dict[str, str] = {
    "trend": "trend_following",
    "range": "mean_reversion",
    "volatile": "momentum_breakout",
    "high_volatility": "momentum_breakout",
}
STRATEGY_IDS = ("trend_following", "mean_reversion", "momentum_breakout")
ROUND_TRIP_COST_BPS = 60.0


@dataclass(frozen=True, slots=True)
class ResearchRoute:
    regime: str | None
    strategy_id: str | None
    status: str
    reason: str
    order_permission: bool = False
    champion: None = None

    def __post_init__(self) -> None:
        if self.order_permission or self.champion is not None:
            raise ValueError("ai_core_research_route_cannot_authorize_or_promote")


@dataclass(frozen=True, slots=True)
class StrategySignal:
    strategy_id: str
    signal: int
    reason: str
    order_permission: bool = False

    def __post_init__(self) -> None:
        if self.strategy_id not in STRATEGY_IDS:
            raise ValueError("ai_core_strategy_id_unknown")
        if self.signal not in (0, 1):
            raise ValueError("ai_core_strategy_signal_must_be_long_or_flat")
        if self.order_permission:
            raise ValueError("ai_core_strategy_signal_cannot_authorize_orders")


def _registry_candidate(
    registry: StrategyRegistry, strategy_id: str, regime: str
) -> bool:
    matching = [item for item in registry.strategies if item.strategy_id == strategy_id]
    if len(matching) != 1:
        return False
    definition = matching[0]
    return (
        regime in definition.eligible_regimes
        and definition.lifecycle is StrategyLifecycle.RESEARCH_ONLY
        and not definition.owner_approved
        and not definition.allow_short
        and "4h" in definition.supported_timeframes
    )


def route_research_regime(
    regime: str | None, registry: StrategyRegistry
) -> ResearchRoute:
    """Map a technical research regime to a research-only registry entry."""
    if regime is None or not regime.strip():
        return ResearchRoute(None, None, "NO_TRADE", "regime_missing")
    strategy_id = REGIME_STRATEGY_MAP.get(regime)
    if strategy_id is None:
        return ResearchRoute(regime, None, "NO_TRADE", "regime_not_mapped")
    if not _registry_candidate(registry, strategy_id, regime):
        return ResearchRoute(
            regime, None, "NO_TRADE", "research_registry_candidate_invalid"
        )
    return ResearchRoute(
        regime, strategy_id, "RESEARCH_PROPOSAL", "hmm_regime_route_resolved"
    )


def route_approved_inference(
    inference: RegimeInference | None, registry: StrategyRegistry
) -> ResearchRoute:
    """Use the existing approved HMM contract; all other outcomes abstain."""
    if (
        inference is None
        or inference.status is not RegimeInferenceStatus.CLASSIFIED
        or inference.order_permission
        or inference.selected_regime is None
    ):
        return ResearchRoute(
            None, None, "NO_TRADE", "approved_hmm_inference_unavailable"
        )
    return route_research_regime(inference.selected_regime, registry)


def _number(row: Mapping[str, object], name: str) -> float | None:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _trend_following_signal(values: Mapping[str, float]) -> int:
    return int(values["ema_crossover_24_72"] > 0 and values["return_6"] > 0)


def _mean_reversion_signal(values: Mapping[str, float]) -> int:
    return int(
        values["rsi_14"] <= 30
        and values["bollinger_position_24"] <= 0
        and values["return_3"] < 0
    )


def _momentum_breakout_signal(values: Mapping[str, float]) -> int:
    return int(
        values["breakout_distance_24"] > 0
        and values["volume_zscore_24"] > 0
        and values["return_6"] > 0
    )


SignalRule = tuple[tuple[str, ...], Callable[[Mapping[str, float]], int]]
_SIGNAL_RULES: dict[str, SignalRule] = {
    "trend_following": (
        ("ema_crossover_24_72", "return_6"),
        _trend_following_signal,
    ),
    "mean_reversion": (
        ("rsi_14", "bollinger_position_24", "return_3"),
        _mean_reversion_signal,
    ),
    "momentum_breakout": (
        ("breakout_distance_24", "volume_zscore_24", "return_6"),
        _momentum_breakout_signal,
    ),
}


def _numeric_signal_values(
    row: Mapping[str, object], required: tuple[str, ...]
) -> dict[str, float] | None:
    values: dict[str, float] = {}
    for name in required:
        value = _number(row, name)
        if value is None:
            return None
        values[name] = value
    return values


def generate_candidate_signal(
    strategy_id: str, row: Mapping[str, object]
) -> StrategySignal:
    """Generate a long/flat research signal from one already-known feature row."""
    rule = _SIGNAL_RULES.get(strategy_id)
    if rule is None:
        raise ValueError("ai_core_strategy_id_unknown")
    required, signal_calculator = rule
    values = _numeric_signal_values(row, required)
    if values is None:
        return StrategySignal(strategy_id, 0, "required_feature_missing_or_non_finite")
    signal = signal_calculator(values)
    return StrategySignal(
        strategy_id,
        signal,
        "predeclared_research_rule_signal"
        if signal
        else "predeclared_rule_not_triggered",
    )


def ensemble_cost_filter(
    predicted_return: float, round_trip_cost_bps: float = ROUND_TRIP_COST_BPS
) -> bool:
    if (
        not math.isfinite(predicted_return)
        or not math.isfinite(round_trip_cost_bps)
        or round_trip_cost_bps < 0
    ):
        return False
    return predicted_return > round_trip_cost_bps / 10_000.0


__all__ = [
    "REGIME_STRATEGY_MAP",
    "STRATEGY_IDS",
    "ResearchRoute",
    "StrategySignal",
    "ensemble_cost_filter",
    "generate_candidate_signal",
    "route_approved_inference",
    "route_research_regime",
]
