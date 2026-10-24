from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path

import pytest

from services.ai_core.strategies import (
    REGIME_STRATEGY_MAP,
    STRATEGY_IDS,
    ResearchRoute,
    StrategySignal,
    ensemble_cost_filter,
    generate_candidate_signal,
    route_approved_inference,
    route_research_regime,
)
from services.strategy_selection.contracts import (
    StrategyLifecycle,
)
from services.strategy_selection.regime import (
    RegimeInference,
    RegimeInferenceStatus,
    HMMRegimeClassifier,
)
from services.strategy_selection.registry import (
    StrategyRegistry,
    load_registry_document,
)


def research_registry() -> StrategyRegistry:
    path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "research"
        / "ai-core-strategy-registry.json"
    )
    loaded = load_registry_document(path.read_text(encoding="utf-8"))
    assert loaded.registry is not None
    assert not loaded.reason_codes
    return loaded.registry


def classified_inference(regime: str = "trend") -> RegimeInference:
    posterior = (
        ("trend", Decimal("0.90") if regime == "trend" else Decimal("0.05")),
        ("range", Decimal("0.90") if regime == "range" else Decimal("0.05")),
        (
            "high_volatility",
            Decimal("0.90") if regime == "high_volatility" else Decimal("0.05"),
        ),
    )
    total = sum(value for _, value in posterior)
    normalized = tuple((name, value / total) for name, value in posterior)
    confidence = max(value for _, value in normalized)
    return RegimeInference(
        status=RegimeInferenceStatus.CLASSIFIED,
        decision_id="test-decision",
        selected_regime=regime,
        posterior=normalized,
        confidence=confidence,
        model_revision="test-hmm-v1",
        model_fingerprint="a" * 64,
        evidence_fingerprint="b" * 64,
        reason_codes=("hmm_regime_classified",),
        order_permission=False,
        fingerprint="c" * 64,
    )


def test_research_registry_contains_three_unapproved_candidates_only() -> None:
    registry = research_registry()
    assert tuple(item.strategy_id for item in registry.strategies) == (
        "mean_reversion",
        "momentum_breakout",
        "trend_following",
    )
    assert all(item.lifecycle is StrategyLifecycle.RESEARCH_ONLY for item in registry.strategies)
    assert all(not item.owner_approved for item in registry.strategies)
    assert all(item.supported_timeframes == ("4h",) for item in registry.strategies)
    assert all(not item.allow_short for item in registry.strategies)


@pytest.mark.parametrize(
    ("regime", "expected"),
    (
        ("trend", "trend_following"),
        ("range", "mean_reversion"),
        ("volatile", "momentum_breakout"),
        ("high_volatility", "momentum_breakout"),
    ),
)
def test_hmm_regime_routes_to_the_declared_research_candidate(
    regime: str, expected: str
) -> None:
    route = route_research_regime(regime, research_registry())
    assert route.status == "RESEARCH_PROPOSAL"
    assert route.strategy_id == expected
    assert route.order_permission is False
    assert route.champion is None


def test_unknown_regime_or_bad_registry_candidate_abstains() -> None:
    registry = research_registry()
    assert route_research_regime(None, registry).status == "NO_TRADE"
    assert route_research_regime("unseen", registry).strategy_id is None
    trend = next(item for item in registry.strategies if item.strategy_id == "trend_following")
    unapproved_contract = replace(trend, lifecycle=StrategyLifecycle.SELECTABLE)
    invalid_registry = StrategyRegistry(
        revision=registry.revision,
        economic_policy_revision=registry.economic_policy_revision,
        strategies=tuple(
            unapproved_contract if item.strategy_id == "trend_following" else item
            for item in registry.strategies
        ),
    )
    rejected = route_research_regime("trend", invalid_registry)
    assert rejected.status == "NO_TRADE"
    assert rejected.strategy_id is None
    assert rejected.order_permission is False


def test_existing_hmm_result_contract_routes_only_classified_inference() -> None:
    registry = research_registry()
    result = route_approved_inference(classified_inference(), registry)
    assert result.strategy_id == "trend_following"
    assert result.order_permission is False
    no_trade = HMMRegimeClassifier().infer(model=None, observation=None)
    rejected = route_approved_inference(no_trade, registry)
    assert rejected.status == "NO_TRADE"
    assert rejected.strategy_id is None
    assert rejected.champion is None


@pytest.mark.parametrize(
    ("strategy_id", "row"),
    (
        (
            "trend_following",
            {"ema_crossover_24_72": 0.01, "return_6": 0.02},
        ),
        (
            "mean_reversion",
            {
                "rsi_14": 20.0,
                "bollinger_position_24": 0.0,
                "return_3": -0.02,
            },
        ),
        (
            "momentum_breakout",
            {
                "breakout_distance_24": 0.01,
                "volume_zscore_24": 1.1,
                "return_6": 0.03,
            },
        ),
    ),
)
def test_each_strategy_generates_a_pure_long_flat_research_signal(
    strategy_id: str, row: dict[str, object]
) -> None:
    signal = generate_candidate_signal(strategy_id, row)
    assert signal.strategy_id == strategy_id
    assert signal.signal == 1
    assert signal.order_permission is False


def test_strategy_rules_abstain_for_missing_or_untriggered_features() -> None:
    missing = generate_candidate_signal("trend_following", {})
    assert missing.signal == 0
    assert "missing" in missing.reason
    flat = generate_candidate_signal(
        "trend_following",
        {"ema_crossover_24_72": -0.01, "return_6": 0.02},
    )
    assert flat.signal == 0
    invalid = generate_candidate_signal(
        "mean_reversion",
        {
            "rsi_14": True,
            "bollinger_position_24": -0.1,
            "return_3": -0.02,
        },
    )
    assert invalid.signal == 0
    with pytest.raises(ValueError, match="strategy_id_unknown"):
        generate_candidate_signal("unregistered", {})


def test_cost_filter_and_immutable_safety_contracts() -> None:
    assert set(STRATEGY_IDS) == {
        "trend_following",
        "mean_reversion",
        "momentum_breakout",
    }
    assert REGIME_STRATEGY_MAP["trend"] == "trend_following"
    assert ensemble_cost_filter(0.01)
    assert not ensemble_cost_filter(0.006)
    assert not ensemble_cost_filter(float("nan"))
    assert not ensemble_cost_filter(0.1, -1.0)
    with pytest.raises(ValueError, match="cannot_authorize_or_promote"):
        ResearchRoute("trend", "trend_following", "RESEARCH_PROPOSAL", "test", True)
    with pytest.raises(ValueError, match="cannot_authorize_orders"):
        StrategySignal("trend_following", 1, "test", True)
    with pytest.raises(ValueError, match="long_or_flat"):
        StrategySignal("trend_following", -1, "short")
