"""Synthetic contract tests for the HMM regime filter."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from services.data_quality.gate import QualityReasonCode, QualityResult, QualityStatus
from services.strategy_selection.regime import (
    GaussianRegimeEmission,
    HMMRegimeClassifier,
    PreviousRegimePosterior,
    RegimeHMMModel,
    RegimeInferenceStatus,
    RegimeObservation,
)

D = Decimal
NOW = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
DATA_HASH = "a" * 64
MODEL_HASH = "b" * 64
PRIOR_HASH = "c" * 64


def make_model(**changes: object) -> RegimeHMMModel:
    values: dict[str, object] = {
        "revision": "hmm-regimes-v1",
        "model_sha256": MODEL_HASH,
        "trained_at": NOW - timedelta(days=2),
        "owner_approved": True,
        "feature_schema_version": "market-regime-features.v1",
        "feature_names": ("directional_efficiency", "realized_volatility"),
        "regimes": ("trend", "range", "high_volatility"),
        "initial_probabilities": (D("0.34"), D("0.33"), D("0.33")),
        "transition_matrix": (
            (D("0.85"), D("0.10"), D("0.05")),
            (D("0.10"), D("0.85"), D("0.05")),
            (D("0.10"), D("0.10"), D("0.80")),
        ),
        "emissions": (
            GaussianRegimeEmission(
                "trend",
                (D("0.80"), D("0.20")),
                (D("0.15"), D("0.10")),
            ),
            GaussianRegimeEmission(
                "range",
                (D("0.20"), D("0.20")),
                (D("0.15"), D("0.10")),
            ),
            GaussianRegimeEmission(
                "high_volatility",
                (D("0.25"), D("0.80")),
                (D("0.20"), D("0.12")),
            ),
        ),
        "max_model_age_seconds": 30 * 24 * 60 * 60,
        "max_observation_age_seconds": 3600,
        "minimum_confidence": D("0.55"),
        "minimum_posterior_margin": D("0.10"),
    }
    values.update(changes)
    return RegimeHMMModel(**values)  # type: ignore[arg-type]


def make_observation(**changes: object) -> RegimeObservation:
    quality = QualityResult(
        passed=True,
        reasons=(),
        status=QualityStatus.VALID,
        reason_codes=(),
        fingerprint=DATA_HASH,
    )
    values: dict[str, object] = {
        "observed_at": NOW - timedelta(seconds=30),
        "evaluated_at": NOW,
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "feature_schema_version": "market-regime-features.v1",
        "feature_values": (
            ("directional_efficiency", D("0.90")),
            ("realized_volatility", D("0.20")),
        ),
        "data_quality": quality,
        "data_fingerprint": DATA_HASH,
    }
    values.update(changes)
    return RegimeObservation(**values)  # type: ignore[arg-type]


def make_previous(**changes: object) -> PreviousRegimePosterior:
    values: dict[str, object] = {
        "model_revision": "hmm-regimes-v1",
        "as_of": NOW - timedelta(minutes=30),
        "probabilities": (D("0.15"), D("0.80"), D("0.05")),
        "fingerprint": PRIOR_HASH,
    }
    values.update(changes)
    return PreviousRegimePosterior(**values)  # type: ignore[arg-type]


def test_hmm_classifies_a_valid_trend_observation_deterministically() -> None:
    classifier = HMMRegimeClassifier()
    first = classifier.infer(model=make_model(), observation=make_observation())
    second = classifier.infer(model=make_model(), observation=make_observation())

    assert first.status is RegimeInferenceStatus.CLASSIFIED
    assert first.selected_regime == "trend"
    assert first.confidence is not None and first.confidence >= D("0.55")
    assert first.model_revision == "hmm-regimes-v1"
    assert first.model_fingerprint is not None
    assert first.evidence_fingerprint is not None
    assert first.order_permission is False
    assert first.reason_codes == ("hmm_regime_classified",)
    assert first.fingerprint == second.fingerprint


def test_hmm_uses_previous_posterior_and_preserves_state_lineage() -> None:
    classifier = HMMRegimeClassifier()
    model = make_model()
    observation = make_observation(
        feature_values=(
            ("directional_efficiency", D("0.22")),
            ("realized_volatility", D("0.21")),
        )
    )
    prior = make_previous()
    first = classifier.infer(
        model=model, observation=observation, previous=prior
    )
    second = classifier.infer(
        model=model, observation=observation, previous=prior
    )

    assert first.status is RegimeInferenceStatus.CLASSIFIED
    assert first.selected_regime == "range"
    assert first.evidence_fingerprint is not None
    assert first.fingerprint == second.fingerprint


@pytest.mark.parametrize(
    ("model", "observation", "previous", "expected"),
    (
        (None, None, None, "regime_model_missing"),
        (None, make_observation(), None, "regime_model_missing"),
        (make_model(), None, None, "regime_observation_missing"),
        (
            make_model(owner_approved=False),
            make_observation(),
            None,
            "regime_model_not_owner_approved",
        ),
        (
            make_model(trained_at=NOW + timedelta(seconds=1)),
            make_observation(),
            None,
            "regime_model_timestamp_in_future",
        ),
        (
            make_model(trained_at=NOW - timedelta(days=60)),
            make_observation(),
            None,
            "regime_model_stale",
        ),
        (
            make_model(),
            make_observation(observed_at=NOW + timedelta(seconds=1)),
            None,
            "regime_observation_timestamp_in_future",
        ),
        (
            make_model(),
            make_observation(observed_at=NOW - timedelta(hours=2)),
            None,
            "regime_observation_stale",
        ),
        (
            make_model(),
            make_observation(feature_schema_version="features.v2"),
            None,
            "regime_feature_schema_mismatch",
        ),
        (
            make_model(),
            make_observation(symbol="BTC/BTC"),
            None,
            "regime_spot_usdt_symbol_required",
        ),
        (
            make_model(),
            make_observation(
                feature_values=(("directional_efficiency", D("0.8")),)
            ),
            None,
            "regime_feature_set_mismatch",
        ),
        (
            make_model(),
            make_observation(
                data_quality=QualityResult(
                    passed=True,
                    reasons=("gap_detected",),
                    status=QualityStatus.DEGRADED_ALLOWED,
                    reason_codes=(QualityReasonCode.DATA_GAP,),
                    fingerprint=DATA_HASH,
                )
            ),
            None,
            "regime_data_quality_not_strictly_valid",
        ),
        (
            make_model(),
            make_observation(data_fingerprint="d" * 64),
            None,
            "regime_data_quality_fingerprint_mismatch",
        ),
        (
            make_model(),
            make_observation(),
            make_previous(model_revision="different-hmm-v2"),
            "previous_regime_model_revision_mismatch",
        ),
        (
            make_model(),
            make_observation(),
            make_previous(as_of=NOW),
            "previous_regime_posterior_not_earlier_than_observation",
        ),
        (
            make_model(),
            make_observation(),
            make_previous(as_of=NOW - timedelta(hours=2)),
            "previous_regime_posterior_stale",
        ),
    ),
)
def test_invalid_or_untrusted_inputs_fail_closed(
    model: RegimeHMMModel | None,
    observation: RegimeObservation | None,
    previous: PreviousRegimePosterior | None,
    expected: str,
) -> None:
    result = HMMRegimeClassifier().infer(
        model=model, observation=observation, previous=previous
    )
    assert result.status is RegimeInferenceStatus.NO_TRADE
    assert expected in result.reason_codes
    assert result.selected_regime is None
    assert result.confidence is None
    assert result.order_permission is False


def test_missing_degraded_and_rejected_quality_do_not_classify() -> None:
    rejected = QualityResult(
        passed=False,
        reasons=("gap_detected",),
        status=QualityStatus.REJECTED,
        reason_codes=(QualityReasonCode.DATA_GAP,),
        fingerprint=DATA_HASH,
    )
    for quality in (None, rejected):
        result = HMMRegimeClassifier().infer(
            model=make_model(),
            observation=make_observation(data_quality=quality),
        )
        assert result.status is RegimeInferenceStatus.NO_TRADE
        assert result.order_permission is False


def test_confidence_and_posterior_margin_are_independent_gates() -> None:
    observation = make_observation()
    low_confidence = HMMRegimeClassifier().infer(
        model=make_model(minimum_confidence=D("1")),
        observation=observation,
    )
    low_margin = HMMRegimeClassifier().infer(
        model=make_model(minimum_posterior_margin=D("1")),
        observation=observation,
    )

    assert low_confidence.status is RegimeInferenceStatus.NO_TRADE
    assert "regime_posterior_confidence_below_minimum" in low_confidence.reason_codes
    assert low_margin.status is RegimeInferenceStatus.NO_TRADE
    assert "regime_posterior_margin_below_minimum" in low_margin.reason_codes


def test_model_contract_rejects_invalid_state_probabilities_and_emissions() -> None:
    with pytest.raises(ValueError, match="regime_initial_probabilities_must_sum_to_one"):
        make_model(initial_probabilities=(D("0.5"), D("0.5"), D("0.5")))
    with pytest.raises(ValueError, match="regime_transition_row_must_sum_to_one"):
        make_model(
            transition_matrix=(
                (D("0.9"), D("0.1"), D("0.1")),
                (D("0.1"), D("0.85"), D("0.05")),
                (D("0.1"), D("0.1"), D("0.8")),
            )
        )
    with pytest.raises(ValueError, match="regime_emission_feature_shape_invalid"):
        GaussianRegimeEmission("trend", (D("0.1"),), (D("0.1"), D("0.2")))


def test_classifier_result_cannot_grant_order_permission() -> None:
    result = HMMRegimeClassifier().infer(
        model=make_model(), observation=make_observation()
    )
    with pytest.raises(ValueError, match="regime_classifier_cannot_grant_order_permission"):
        replace(result, order_permission=True)
