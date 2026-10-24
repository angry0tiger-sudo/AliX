"""Research-only AI core components; no order or promotion authority."""

from services.ai_core.ensemble import (
    EnsembleConfig,
    EnsemblePrediction,
    fit_weighted_ensemble,
)
from services.ai_core.features import (
    AI_FEATURE_SCHEMA_VERSION,
    AI_FEATURE_COLUMNS,
    aggregate_hourly_to_4h,
    build_ai_features,
)
from services.ai_core.regime import infer_hmm_regimes
from services.ai_core.strategies import (
    REGIME_STRATEGY_MAP,
    ResearchRoute,
    route_research_regime,
    route_approved_inference,
)

__all__ = [
    "AI_FEATURE_COLUMNS",
    "AI_FEATURE_SCHEMA_VERSION",
    "BASE_AI_FEATURE_COLUMNS",
    "CROSS_ASSET_FEATURE_COLUMNS",
    "EnsembleConfig",
    "EnsemblePrediction",
    "REGIME_STRATEGY_MAP",
    "ResearchRoute",
    "aggregate_hourly_to_4h",
    "build_ai_features",
    "fit_weighted_ensemble",
    "infer_hmm_regimes",
    "route_approved_inference",
    "route_research_regime",
]
