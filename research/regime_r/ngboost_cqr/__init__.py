"""Regime R NGBoost + CQR research contracts (mock-only scaffolding)."""

from research.regime_r.ngboost_cqr.contracts import (
    COST_SCENARIOS_BPS,
    FEATURE_NAMES,
    HORIZON_BARS,
    N_BUDGET,
    TRIAL_IDS,
    ConformalInterval,
    DistributionForecast,
    FeatureSnapshot,
    TargetWindow,
    make_cqr_interval,
    passes_cost_gate,
    target_gross_return,
    validate_distribution_forecast,
    validate_feature_snapshot,
    validate_target_window,
    validate_trial_id,
)

__all__ = (
    "COST_SCENARIOS_BPS",
    "FEATURE_NAMES",
    "HORIZON_BARS",
    "N_BUDGET",
    "TRIAL_IDS",
    "ConformalInterval",
    "DistributionForecast",
    "FeatureSnapshot",
    "TargetWindow",
    "make_cqr_interval",
    "passes_cost_gate",
    "target_gross_return",
    "validate_distribution_forecast",
    "validate_feature_snapshot",
    "validate_target_window",
    "validate_trial_id",
)
