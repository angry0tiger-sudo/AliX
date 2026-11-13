"""Pure, fail-closed acceptance gates for the preregistered four-arm trial family."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

TRIAL_IDS = (
    "NOBITEX-KC26-A",
    "NOBITEX-KC26-B",
    "NOBITEX-KC26-C",
    "NOBITEX-KC26-D",
)
MINIMUM_NET_PROFIT_FACTOR = 1.10
MINIMUM_COMPLETED_TRADES = 100
MAXIMUM_DRAWDOWN = 0.15
MINIMUM_STABLE_FOLDS = 3


@dataclass(frozen=True, slots=True)
class ArmGateEvidence:
    """Precomputed evidence supplied to the pure gate evaluator."""

    arm_id: str
    net_profit_factor_50bps: float
    net_profit_factor_90bps: float
    completed_trades: int
    max_drawdown: float
    dsr: float
    dsr_sensitivity: tuple[float, ...]
    fold_passes: tuple[bool, ...]


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def fold_stability_passes(
    fold_passes: Sequence[bool],
    minimum_stable_folds: int = MINIMUM_STABLE_FOLDS,
) -> bool:
    """Require the preregistered minimum number of independently passing folds."""
    if (
        isinstance(minimum_stable_folds, bool)
        or not isinstance(minimum_stable_folds, int)
        or minimum_stable_folds < 1
    ):
        return False
    if not fold_passes or any(not isinstance(value, bool) for value in fold_passes):
        return False
    return sum(fold_passes) >= minimum_stable_folds


def dsr_sensitivity_passes(
    sensitivity_values: Sequence[float],
    dsr_min_exclusive: float = 0.0,
) -> bool:
    """Fail closed unless every declared DSR sensitivity remains strictly positive."""
    threshold = _finite_number(dsr_min_exclusive)
    if threshold is None or not sensitivity_values:
        return False
    for value in sensitivity_values:
        candidate = _finite_number(value)
        if candidate is None or candidate <= threshold:
            return False
    return True


def arm_passes(evidence: ArmGateEvidence) -> bool:
    """Apply every preregistered economic and statistical gate to one arm."""
    if evidence.arm_id not in TRIAL_IDS:
        return False

    net_pf_50 = _finite_number(evidence.net_profit_factor_50bps)
    net_pf_90 = _finite_number(evidence.net_profit_factor_90bps)
    drawdown = _finite_number(evidence.max_drawdown)
    dsr = _finite_number(evidence.dsr)

    if net_pf_50 is None or net_pf_50 < MINIMUM_NET_PROFIT_FACTOR:
        return False
    if net_pf_90 is None or net_pf_90 < MINIMUM_NET_PROFIT_FACTOR:
        return False
    if (
        isinstance(evidence.completed_trades, bool)
        or not isinstance(evidence.completed_trades, int)
        or evidence.completed_trades < MINIMUM_COMPLETED_TRADES
    ):
        return False
    if drawdown is None or drawdown < 0.0 or drawdown > MAXIMUM_DRAWDOWN:
        return False
    if dsr is None or dsr <= 0.0:
        return False
    if not dsr_sensitivity_passes(evidence.dsr_sensitivity):
        return False
    return fold_stability_passes(evidence.fold_passes)


def select_winner(evidence: Sequence[ArmGateEvidence]) -> str | None:
    """Select the highest-DSR passing arm; ties resolve by trial ID, not input order."""
    identifiers = tuple(item.arm_id for item in evidence)
    if len(set(identifiers)) != len(identifiers):
        return None

    passing = [item for item in evidence if arm_passes(item)]
    if not passing:
        return None
    selected = sorted(passing, key=lambda item: (-item.dsr, item.arm_id))[0]
    return selected.arm_id
