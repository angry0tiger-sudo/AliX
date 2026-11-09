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
        if not self.symbol or not self.symbol.isalnum() or self.symbol.upper() != self.symbol:
            raise ValueError("ensemble_adapter_symbol_invalid")
        if not np.isfinite(self.minimum_probability) or not 0.0 <= self.minimum_probability <= 1.0:
            raise ValueError("ensemble_adapter_probability_threshold_invalid")
        if self.round_trip_cost_bps != 50:
            raise ValueError("nobitex_round_trip_cost_must_be_50_bps")
        if self.band != PRICE_BAND or self.max_participation != MAX_PARTICIPATION:
            raise ValueError("nobitex_liquidity_limits_are_fixed_by_owner_directive")
        self.sizing.validate()
        if self.dry_run is not True or self.allow_live_trading is not False:
            raise LiveExecutionBlocked("AI_ENSEMBLE_ADAPTER_IS_PAPER_ONLY")


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
        if snapshot.symbol != self.config.symbol:
            raise OrderBookError("ensemble_adapter_symbol_mismatch")
        if (
            snapshot.last_update.tzinfo is None
            or snapshot.received_at.tzinfo is None
            or snapshot.age_seconds < 0
            or snapshot.age_seconds > MAX_BOOK_AGE_SECONDS
        ):
            raise OrderBookError("ensemble_adapter_order_book_stale_or_invalid")
        probabilities = np.asarray(self.model.predict_proba(features), dtype=float)
        if (
            probabilities.ndim != 1
            or len(probabilities) != len(features)
            or not bool(np.isfinite(probabilities).all())
            or bool(((probabilities < 0.0) | (probabilities > 1.0)).any())
        ):
            raise ValueError("ensemble_adapter_model_probability_invalid")
        probability = float(probabilities[-1])
        if probability < self.config.minimum_probability:
            return EnsemblePaperDecision(
                "NO_SIGNAL",
                "PROBABILITY_BELOW_MINIMUM",
                self.config.symbol,
                probability,
                None,
                None,
            )
        sizing = fractional_kelly_size(
            probability_of_net_profit=probability,
            equity=equity,
            reference_price=snapshot.mid_price,
            atr=atr,
            config=self.config.sizing,
        )
        if not sizing.approved:
            return EnsemblePaperDecision(
                "RISK_REJECTED",
                sizing.reason,
                self.config.symbol,
                probability,
                sizing,
                None,
            )
        try:
            maker_plan = plan_maker_slices_for_base_quantity(
                snapshot,
                side="buy",
                intended_base_quantity=sizing.base_quantity,
                band=self.config.band,
                max_participation=self.config.max_participation,
            )
        except (LiquidityRejected, OrderBookError) as exc:
            return EnsemblePaperDecision(
                "LIQUIDITY_REJECTED",
                str(exc),
                self.config.symbol,
                probability,
                sizing,
                None,
            )
        if maker_plan.execution_permission:
            raise RuntimeError("maker_plan_must_never_grant_execution_permission")
        return EnsemblePaperDecision(
            "PAPER_PLAN_ONLY",
            "AI_SIGNAL_RISK_AND_LIQUIDITY_GATES_PASSED",
            self.config.symbol,
            probability,
            sizing,
            maker_plan,
        )
