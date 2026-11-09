"""Research-ensemble to Nobitex paper-plan boundary; no order-write capability."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

import numpy as np
from pandas import DataFrame

from integrations.nobitex.mvp import (
    MAX_BOOK_AGE_SECONDS,
    MAX_PARTICIPATION,
    PRICE_BAND,
    LiveExecutionBlocked,
    LiquidityRejected,
    MakerOrderPlan,
    OrderBookError,
    OrderBookSnapshot,
    plan_maker_slices_for_base_quantity,
)
from research.ai_ensemble.risk import (
    KellySizingConfig,
    KellySizingResult,
    fractional_kelly_size,
)


class ProbabilityModel(Protocol):
    def predict_proba(self, features: DataFrame) -> np.ndarray:
        """Return the net-profitable class probability for each feature row."""
        ...


def _validate_adapter_symbol(symbol: str) -> None:
    if not symbol or not symbol.isalnum() or symbol.upper() != symbol:
        raise ValueError("ensemble_adapter_symbol_invalid")


def _validate_probability_threshold(probability: float) -> None:
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("ensemble_adapter_probability_threshold_invalid")


def _validate_adapter_limits(band: Decimal, max_participation: Decimal) -> None:
    if band != PRICE_BAND or max_participation != MAX_PARTICIPATION:
        raise ValueError("nobitex_liquidity_limits_are_fixed_by_owner_directive")


def _validate_adapter_paper_mode(dry_run: bool, allow_live: bool) -> None:
    if dry_run is not True or allow_live is not False:
        raise LiveExecutionBlocked("AI_ENSEMBLE_ADAPTER_IS_PAPER_ONLY")


@dataclass(frozen=True, slots=True)
class EnsembleAdapterConfig:
    symbol: str = "BTCUSDT"
    minimum_probability: float = 0.50
    round_trip_cost_bps: int = 50
    band: Decimal = PRICE_BAND
    max_participation: Decimal = MAX_PARTICIPATION
    dry_run: bool = True
    allow_live_trading: bool = False
    sizing: KellySizingConfig = KellySizingConfig()

    def validate(self) -> None:
        _validate_adapter_symbol(self.symbol)
        _validate_probability_threshold(self.minimum_probability)
        if self.round_trip_cost_bps != 50:
            raise ValueError("nobitex_round_trip_cost_must_be_50_bps")
        _validate_adapter_limits(self.band, self.max_participation)
        self.sizing.validate()
        _validate_adapter_paper_mode(self.dry_run, self.allow_live_trading)


@dataclass(frozen=True, slots=True)
class EnsemblePaperDecision:
    status: str
    reason: str
    symbol: str
    probability_of_net_profit: float
    sizing: KellySizingResult | None
    maker_plan: MakerOrderPlan | None
    order_permission: bool = False
    private_api_calls: int = 0
    real_order_attempts: int = 0


class NobitexEnsemblePaperAdapter:
    """Convert an AI probability to a bounded paper maker plan, never an order."""

    def __init__(
        self,
        model: ProbabilityModel,
        config: EnsembleAdapterConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or EnsembleAdapterConfig()
        self.config.validate()

    def plan_entry(
        self,
        features: DataFrame,
        *,
        snapshot: OrderBookSnapshot,
        equity: Decimal,
        atr: Decimal,
    ) -> EnsemblePaperDecision:
        if features.empty:
            raise ValueError("ensemble_adapter_features_empty")
        _validate_entry_order_book(snapshot, self.config.symbol)
        probability = _latest_model_probability(self.model, features)
        if probability < self.config.minimum_probability:
            return _paper_decision(
                "NO_SIGNAL",
                "PROBABILITY_BELOW_MINIMUM",
                self.config.symbol,
                probability,
            )
        sizing = fractional_kelly_size(
            probability_of_net_profit=probability,
            equity=equity,
            reference_price=snapshot.mid_price,
            atr=atr,
            config=self.config.sizing,
        )
        if not sizing.approved:
            return _paper_decision(
                "RISK_REJECTED",
                sizing.reason,
                self.config.symbol,
                probability,
                sizing=sizing,
            )
        maker_plan = _paper_maker_plan(
            snapshot,
            sizing,
            self.config,
            probability,
        )
        if isinstance(maker_plan, EnsemblePaperDecision):
            return maker_plan
        if maker_plan.execution_permission:
            raise RuntimeError("maker_plan_must_never_grant_execution_permission")
        return _paper_decision(
            "PAPER_PLAN_ONLY",
            "AI_SIGNAL_RISK_AND_LIQUIDITY_GATES_PASSED",
            self.config.symbol,
            probability,
            sizing=sizing,
            maker_plan=maker_plan,
        )


def _validate_entry_order_book(
    snapshot: OrderBookSnapshot,
    expected_symbol: str,
) -> None:
    if snapshot.symbol != expected_symbol:
        raise OrderBookError("ensemble_adapter_symbol_mismatch")
    timestamps_missing_timezone = (
        snapshot.last_update.tzinfo is None or snapshot.received_at.tzinfo is None
    )
    age_invalid = (
        snapshot.age_seconds < 0 or snapshot.age_seconds > MAX_BOOK_AGE_SECONDS
    )
    if timestamps_missing_timezone or age_invalid:
        raise OrderBookError("ensemble_adapter_order_book_stale_or_invalid")


def _latest_model_probability(
    model: ProbabilityModel,
    features: DataFrame,
) -> float:
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    shape_invalid = probabilities.ndim != 1 or len(probabilities) != len(features)
    values_invalid = not bool(np.isfinite(probabilities).all()) or bool(
        ((probabilities < 0.0) | (probabilities > 1.0)).any()
    )
    if shape_invalid or values_invalid:
        raise ValueError("ensemble_adapter_model_probability_invalid")
    return float(probabilities[-1])


def _paper_decision(
    status: str,
    reason: str,
    symbol: str,
    probability: float,
    *,
    sizing: KellySizingResult | None = None,
    maker_plan: MakerOrderPlan | None = None,
) -> EnsemblePaperDecision:
    return EnsemblePaperDecision(
        status,
        reason,
        symbol,
        probability,
        sizing,
        maker_plan,
    )


def _paper_maker_plan(
    snapshot: OrderBookSnapshot,
    sizing: KellySizingResult,
    config: EnsembleAdapterConfig,
    probability: float,
) -> MakerOrderPlan | EnsemblePaperDecision:
    try:
        return plan_maker_slices_for_base_quantity(
            snapshot,
            side="buy",
            intended_base_quantity=sizing.base_quantity,
            band=config.band,
            max_participation=config.max_participation,
        )
    except (LiquidityRejected, OrderBookError) as exc:
        return _paper_decision(
            "LIQUIDITY_REJECTED",
            str(exc),
            config.symbol,
            probability,
            sizing=sizing,
        )
