"""Pure OOS metric calculator; no data ingestion or economic GO decision."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_CEILING

from risk.contracts import require_aware_timestamp, require_decimal


@dataclass(frozen=True, slots=True)
class OosObservation:
    fold_id: str
    train_cutoff: datetime
    observed_at: datetime
    regime: str
    allocator_net_return: Decimal
    baseline_net_return: Decimal
    allocator_cost: Decimal
    baseline_cost: Decimal
    allocator_turnover: Decimal
    baseline_turnover: Decimal

    def __post_init__(self) -> None:
        if not self.fold_id.strip() or not self.regime.strip():
            raise ValueError("oos_fold_or_regime_missing")
        require_aware_timestamp(self.train_cutoff, "train_cutoff")
        require_aware_timestamp(self.observed_at, "observed_at")
        if self.observed_at <= self.train_cutoff:
            raise ValueError("oos_temporal_leakage")
        for name in ("allocator_net_return", "baseline_net_return"):
            require_decimal(name, getattr(self, name))
            if getattr(self, name) < -1:
                raise ValueError("return_below_total_loss")
        for name in (
            "allocator_cost",
            "baseline_cost",
            "allocator_turnover",
            "baseline_turnover",
        ):
            require_decimal(name, getattr(self, name), minimum=Decimal("0"))


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    observation_count: int
    cumulative_net_return: Decimal
    mean_net_return: Decimal
    max_drawdown: Decimal
    cvar_tail_loss: Decimal
    total_cost: Decimal
    total_turnover: Decimal


@dataclass(frozen=True, slots=True)
class OosComparison:
    fold_count: int
    allocator: PerformanceMetrics
    baseline: PerformanceMetrics
    regime_mean_returns: tuple[tuple[str, Decimal, Decimal], ...]
    regime_return_spread: Decimal
    economic_decision: str


def compare_walk_forward_oos(
    observations: tuple[OosObservation, ...],
    *,
    tail_fraction: Decimal,
) -> OosComparison:
    """Compare paired OOS returns. It reports metrics but never accepts economics."""
    ordered, fold_count = _validated_observations(observations, tail_fraction)
    allocator_metrics = _metrics(
        tuple(item.allocator_net_return for item in ordered),
        tuple(item.allocator_cost for item in ordered),
        tuple(item.allocator_turnover for item in ordered),
        tail_fraction,
    )
    baseline_metrics = _metrics(
        tuple(item.baseline_net_return for item in ordered),
        tuple(item.baseline_cost for item in ordered),
        tuple(item.baseline_turnover for item in ordered),
        tail_fraction,
    )
    regime_returns = _regime_means(ordered)
    allocator_means = tuple(row[1] for row in regime_returns)
    spread = max(allocator_means) - min(allocator_means)
    return OosComparison(
        fold_count=fold_count,
        allocator=allocator_metrics,
        baseline=baseline_metrics,
        regime_mean_returns=regime_returns,
        regime_return_spread=spread,
        economic_decision="NOT_EVALUATED_REQUIRES_OWNER_APPROVED_DATA_AND_THRESHOLDS",
    )


def _validated_observations(
    observations: tuple[OosObservation, ...], tail_fraction: Decimal
) -> tuple[tuple[OosObservation, ...], int]:
    require_decimal(
        "tail_fraction", tail_fraction, minimum=Decimal("0.00000000000000000001")
    )
    if tail_fraction > 1:
        raise ValueError("tail_fraction_out_of_range")
    if not observations:
        raise ValueError("oos_observations_missing")
    ordered = tuple(sorted(observations, key=lambda item: item.observed_at))
    fold_cutoffs: dict[str, datetime] = {}
    for item in ordered:
        prior = fold_cutoffs.get(item.fold_id)
        if prior is not None and item.train_cutoff != prior:
            raise ValueError("fold_train_cutoff_inconsistent")
        fold_cutoffs[item.fold_id] = item.train_cutoff
    return ordered, len(fold_cutoffs)


def _regime_means(
    observations: tuple[OosObservation, ...],
) -> tuple[tuple[str, Decimal, Decimal], ...]:
    by_regime: dict[str, list[OosObservation]] = {}
    for item in observations:
        by_regime.setdefault(item.regime, []).append(item)
    return tuple(
        (
            regime,
            sum(row.allocator_net_return for row in rows) / Decimal(len(rows)),
            sum(row.baseline_net_return for row in rows) / Decimal(len(rows)),
        )
        for regime, rows in sorted(by_regime.items())
    )


def _metrics(
    returns: tuple[Decimal, ...],
    costs: tuple[Decimal, ...],
    turnover: tuple[Decimal, ...],
    tail_fraction: Decimal,
) -> PerformanceMetrics:
    equity = Decimal("1")
    high_water = Decimal("1")
    drawdown = Decimal("0")
    for result in returns:
        equity *= Decimal("1") + result
        high_water = max(high_water, equity)
        if high_water > 0:
            drawdown = max(drawdown, (high_water - equity) / high_water)
    tail_count = max(
        1,
        int(
            (Decimal(len(returns)) * tail_fraction).to_integral_value(
                rounding=ROUND_CEILING
            )
        ),
    )
    losses = sorted((max(Decimal("0"), -value) for value in returns), reverse=True)[
        :tail_count
    ]
    cvar = sum(losses, Decimal("0")) / Decimal(len(losses))
    return PerformanceMetrics(
        observation_count=len(returns),
        cumulative_net_return=equity - Decimal("1"),
        mean_net_return=sum(returns, Decimal("0")) / Decimal(len(returns)),
        max_drawdown=drawdown,
        cvar_tail_loss=cvar,
        total_cost=sum(costs, Decimal("0")),
        total_turnover=sum(turnover, Decimal("0")),
    )
