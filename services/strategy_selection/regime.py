"""Versioned hidden-Markov market-regime inference; classification only."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from services.data_quality.gate import QualityResult, QualityStatus
from services.strategy_selection.contracts import stable_fingerprint

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROBABILITY_TOLERANCE = Decimal("0.000001")


class RegimeInferenceStatus(StrEnum):
    CLASSIFIED = "CLASSIFIED"
    NO_TRADE = "NO_TRADE"


@dataclass(frozen=True, slots=True)
class GaussianRegimeEmission:
    """Independent Gaussian feature emissions for one hidden regime."""

    regime: str
    means: tuple[Decimal, ...]
    standard_deviations: tuple[Decimal, ...]

    def __post_init__(self) -> None:
        if not self.regime.strip():
            raise ValueError("regime_emission_name_missing")
        if not self.means or len(self.means) != len(self.standard_deviations):
            raise ValueError("regime_emission_feature_shape_invalid")
        for value in self.means:
            _require_decimal("regime_emission_mean", value)
        for value in self.standard_deviations:
            _require_decimal(
                "regime_emission_standard_deviation",
                value,
                minimum=Decimal("0.00000000000000000001"),
            )


@dataclass(frozen=True, slots=True)
class RegimeHMMModel:
    """Pinned HMM parameters. Training and promotion occur in a separate gate."""

    revision: str
    model_sha256: str
    trained_at: datetime
    owner_approved: bool
    feature_schema_version: str
    feature_names: tuple[str, ...]
    regimes: tuple[str, ...]
    initial_probabilities: tuple[Decimal, ...]
    transition_matrix: tuple[tuple[Decimal, ...], ...]
    emissions: tuple[GaussianRegimeEmission, ...]
    max_model_age_seconds: int
    max_observation_age_seconds: int
    minimum_confidence: Decimal
    minimum_posterior_margin: Decimal

    def __post_init__(self) -> None:
        if not self.revision.strip() or not self.feature_schema_version.strip():
            raise ValueError("regime_model_identity_missing")
        if _SHA256.fullmatch(self.model_sha256) is None:
            raise ValueError("regime_model_hash_invalid")
        _require_aware("regime_model_trained_at", self.trained_at)
        if not isinstance(self.owner_approved, bool):
            raise ValueError("regime_model_approval_invalid")
        _require_unique("regime_feature_names", self.feature_names, nonempty=True)
        _require_unique("regime_labels", self.regimes, nonempty=True)
        if len(self.regimes) < 2:
            raise ValueError("regime_model_requires_multiple_states")
        if len(self.initial_probabilities) != len(self.regimes):
            raise ValueError("regime_initial_probability_shape_invalid")
        _validate_probability_vector(
            "regime_initial_probabilities", self.initial_probabilities
        )
        if len(self.transition_matrix) != len(self.regimes):
            raise ValueError("regime_transition_matrix_shape_invalid")
        for row in self.transition_matrix:
            if len(row) != len(self.regimes):
                raise ValueError("regime_transition_matrix_shape_invalid")
            _validate_probability_vector("regime_transition_row", row)
        if tuple(item.regime for item in self.emissions) != self.regimes:
            raise ValueError("regime_emission_state_order_mismatch")
        for emission in self.emissions:
            if len(emission.means) != len(self.feature_names):
                raise ValueError("regime_emission_feature_shape_invalid")
        if self.max_model_age_seconds <= 0 or self.max_observation_age_seconds <= 0:
            raise ValueError("regime_model_freshness_limits_invalid")
        _require_decimal(
            "regime_minimum_confidence",
            self.minimum_confidence,
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
        _require_decimal(
            "regime_minimum_posterior_margin",
            self.minimum_posterior_margin,
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )

    @property
    def fingerprint(self) -> str:
        return stable_fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class RegimeObservation:
    """Point-in-time feature vector bound to strict data-quality evidence."""

    observed_at: datetime
    evaluated_at: datetime
    symbol: str
    timeframe: str
    feature_schema_version: str
    feature_values: tuple[tuple[str, Decimal], ...]
    data_quality: QualityResult | None
    data_fingerprint: str

    def __post_init__(self) -> None:
        _require_aware("regime_observation_observed_at", self.observed_at)
        _require_aware("regime_observation_evaluated_at", self.evaluated_at)
        for name, value in (
            ("regime_observation_symbol", self.symbol),
            ("regime_observation_timeframe", self.timeframe),
            ("regime_observation_feature_schema", self.feature_schema_version),
        ):
            if not value.strip():
                raise ValueError(f"{name}_missing")
        names = tuple(name for name, _ in self.feature_values)
        _require_unique("regime_observation_features", names, nonempty=False)
        for name, value in self.feature_values:
            if not name.strip():
                raise ValueError("regime_observation_feature_name_missing")
            _require_decimal("regime_observation_feature_value", value)


@dataclass(frozen=True, slots=True)
class PreviousRegimePosterior:
    """Prior state probability vector produced by the same pinned HMM revision."""

    model_revision: str
    as_of: datetime
    probabilities: tuple[Decimal, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        if not self.model_revision.strip():
            raise ValueError("previous_regime_model_revision_missing")
        _require_aware("previous_regime_as_of", self.as_of)
        _validate_probability_vector("previous_regime_probabilities", self.probabilities)
        if _SHA256.fullmatch(self.fingerprint) is None:
            raise ValueError("previous_regime_fingerprint_invalid")


@dataclass(frozen=True, slots=True)
class RegimeInference:
    status: RegimeInferenceStatus
    decision_id: str
    selected_regime: str | None
    posterior: tuple[tuple[str, Decimal], ...]
    confidence: Decimal | None
    model_revision: str | None
    model_fingerprint: str | None
    evidence_fingerprint: str | None
    reason_codes: tuple[str, ...]
    order_permission: bool
    fingerprint: str

    def __post_init__(self) -> None:
        if not self.decision_id.strip():
            raise ValueError("regime_decision_id_missing")
        if self.order_permission is not False:
            raise ValueError("regime_classifier_cannot_grant_order_permission")
        if not self.reason_codes:
            raise ValueError("regime_reason_codes_required")
        if _SHA256.fullmatch(self.fingerprint) is None:
            raise ValueError("regime_inference_fingerprint_invalid")
        if self.status is RegimeInferenceStatus.CLASSIFIED:
            if (
                not self.selected_regime
                or self.confidence is None
                or self.model_fingerprint is None
                or self.evidence_fingerprint is None
            ):
                raise ValueError("classified_regime_evidence_missing")
        elif self.selected_regime is not None or self.confidence is not None:
            raise ValueError("no_trade_regime_must_not_select_state")


class HMMRegimeClassifier:
    """Forward-filter market features; reject unknown or low-confidence regimes."""

    def infer(
        self,
        *,
        model: RegimeHMMModel | None,
        observation: RegimeObservation | None,
        previous: PreviousRegimePosterior | None = None,
    ) -> RegimeInference:
        decision_id = self._decision_id(observation)
        reasons = _input_reasons(model, observation, previous)
        if reasons:
            return _build_no_trade(
                decision_id=decision_id,
                model=model,
                observation=observation,
                reasons=reasons,
            )
        if model is None or observation is None:
            return _build_no_trade(
                decision_id=decision_id,
                model=model,
                observation=observation,
                reasons=("regime_inputs_unavailable",),
            )
        try:
            probabilities = _forward_filter(model, observation, previous)
        except (OverflowError, ValueError, ZeroDivisionError):
            return _build_no_trade(
                decision_id=decision_id,
                model=model,
                observation=observation,
                reasons=("regime_likelihood_numerically_unstable",),
            )
        ranked = sorted(probabilities, reverse=True)
        margin = ranked[0] - ranked[1]
        if ranked[0] < float(model.minimum_confidence):
            return _build_no_trade(
                decision_id=decision_id,
                model=model,
                observation=observation,
                reasons=("regime_posterior_confidence_below_minimum",),
                posterior=_render_posterior(model, probabilities),
            )
        if margin < float(model.minimum_posterior_margin):
            return _build_no_trade(
                decision_id=decision_id,
                model=model,
                observation=observation,
                reasons=("regime_posterior_margin_below_minimum",),
                posterior=_render_posterior(model, probabilities),
            )
        selected_index = probabilities.index(ranked[0])
        posterior = _render_posterior(model, probabilities)
        confidence = _as_decimal(ranked[0])
        evidence_fingerprint = _observation_fingerprint(observation, previous)
        payload = {
            "decision_id": decision_id,
            "status": RegimeInferenceStatus.CLASSIFIED.value,
            "selected_regime": model.regimes[selected_index],
            "posterior": posterior,
            "confidence": confidence,
            "model_revision": model.revision,
            "model_fingerprint": model.fingerprint,
            "evidence_fingerprint": evidence_fingerprint,
            "order_permission": False,
        }
        return RegimeInference(
            status=RegimeInferenceStatus.CLASSIFIED,
            decision_id=decision_id,
            selected_regime=model.regimes[selected_index],
            posterior=posterior,
            confidence=confidence,
            model_revision=model.revision,
            model_fingerprint=model.fingerprint,
            evidence_fingerprint=evidence_fingerprint,
            reason_codes=("hmm_regime_classified",),
            order_permission=False,
            fingerprint=stable_fingerprint(payload),
        )

    @staticmethod
    def _decision_id(observation: RegimeObservation | None) -> str:
        if observation is None:
            return "unknown"
        return stable_fingerprint(
            {
                "observed_at": observation.observed_at,
                "evaluated_at": observation.evaluated_at,
                "symbol": observation.symbol,
                "timeframe": observation.timeframe,
                "data_fingerprint": observation.data_fingerprint,
            }
        )


def _input_reasons(
    model: RegimeHMMModel | None,
    observation: RegimeObservation | None,
    previous: PreviousRegimePosterior | None,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if model is None:
        reasons.append("regime_model_missing")
    if observation is None:
        reasons.append("regime_observation_missing")
    if model is not None and observation is not None:
        reasons.extend(_model_freshness_reasons(model, observation))
        reasons.extend(_observation_quality_reasons(observation))
        reasons.extend(_feature_contract_reasons(model, observation))
    if model is not None and previous is not None and observation is not None:
        reasons.extend(_previous_posterior_reasons(model, observation, previous))
    return tuple(dict.fromkeys(reasons))


def _model_freshness_reasons(
    model: RegimeHMMModel, observation: RegimeObservation
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not model.owner_approved:
        reasons.append("regime_model_not_owner_approved")
    if model.trained_at > observation.evaluated_at:
        reasons.append("regime_model_timestamp_in_future")
    elif (
        observation.evaluated_at - model.trained_at
    ).total_seconds() > model.max_model_age_seconds:
        reasons.append("regime_model_stale")
    return tuple(reasons)


def _observation_quality_reasons(observation: RegimeObservation) -> tuple[str, ...]:
    reasons: list[str] = []
    if observation.observed_at > observation.evaluated_at:
        reasons.append("regime_observation_timestamp_in_future")
    elif observation.data_quality is None:
        reasons.append("regime_data_quality_missing")
    elif (
        not observation.data_quality.passed
        or observation.data_quality.status is not QualityStatus.VALID
    ):
        reasons.append("regime_data_quality_not_strictly_valid")
    elif (
        not observation.data_quality.fingerprint
        or observation.data_quality.fingerprint != observation.data_fingerprint
    ):
        reasons.append("regime_data_quality_fingerprint_mismatch")
    return tuple(reasons)


def _feature_contract_reasons(
    model: RegimeHMMModel, observation: RegimeObservation
) -> tuple[str, ...]:
    reasons: list[str] = []
    if observation.feature_schema_version != model.feature_schema_version:
        reasons.append("regime_feature_schema_mismatch")
    if (
        observation.symbol != observation.symbol.upper()
        or observation.symbol.count("/") != 1
        or not observation.symbol.endswith("/USDT")
    ):
        reasons.append("regime_spot_usdt_symbol_required")
    if observation.observed_at <= observation.evaluated_at and (
        observation.evaluated_at - observation.observed_at
    ).total_seconds() > model.max_observation_age_seconds:
        reasons.append("regime_observation_stale")
    observed_names = tuple(name for name, _ in observation.feature_values)
    if not observed_names:
        reasons.append("regime_features_missing")
    elif set(observed_names) != set(model.feature_names):
        reasons.append("regime_feature_set_mismatch")
    return tuple(reasons)


def _previous_posterior_reasons(
    model: RegimeHMMModel,
    observation: RegimeObservation,
    previous: PreviousRegimePosterior,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if previous.model_revision != model.revision:
        reasons.append("previous_regime_model_revision_mismatch")
    if len(previous.probabilities) != len(model.regimes):
        reasons.append("previous_regime_probability_shape_mismatch")
    if previous.as_of >= observation.observed_at:
        reasons.append("previous_regime_posterior_not_earlier_than_observation")
    elif (
        observation.observed_at - previous.as_of
    ).total_seconds() > model.max_observation_age_seconds:
        reasons.append("previous_regime_posterior_stale")
    return tuple(reasons)


def _forward_filter(
    model: RegimeHMMModel,
    observation: RegimeObservation,
    previous: PreviousRegimePosterior | None,
) -> tuple[float, ...]:
    if previous is None:
        prior = tuple(float(value) for value in model.initial_probabilities)
    else:
        prior = tuple(
            sum(
                float(previous.probabilities[i])
                * float(model.transition_matrix[i][j])
                for i in range(len(model.regimes))
            )
            for j in range(len(model.regimes))
        )
    feature_map = dict(observation.feature_values)
    values = tuple(float(feature_map[name]) for name in model.feature_names)
    log_weights: list[float] = []
    for state_index, emission in enumerate(model.emissions):
        probability = prior[state_index]
        prior_log = math.log(probability) if probability > 0 else -math.inf
        log_weights.append(
            prior_log + _gaussian_log_likelihood(values, emission)
        )
    finite_weights = [value for value in log_weights if math.isfinite(value)]
    if not finite_weights:
        raise ValueError("regime_all_state_likelihoods_invalid")
    maximum = max(finite_weights)
    unnormalized = tuple(
        math.exp(value - maximum) if math.isfinite(value) else 0.0
        for value in log_weights
    )
    normalizer = sum(unnormalized)
    if not math.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("regime_likelihood_normalizer_invalid")
    return tuple(value / normalizer for value in unnormalized)


def _gaussian_log_likelihood(
    values: tuple[float, ...], emission: GaussianRegimeEmission
) -> float:
    terms: list[float] = []
    for value, mean_decimal, std_decimal in zip(
        values, emission.means, emission.standard_deviations, strict=True
    ):
        mean = float(mean_decimal)
        std = float(std_decimal)
        z_score = (value - mean) / std
        if not math.isfinite(z_score):
            raise ValueError("regime_feature_likelihood_invalid")
        terms.append(-math.log(std) - 0.5 * math.log(2 * math.pi) - 0.5 * z_score**2)
    likelihood = sum(terms)
    if not math.isfinite(likelihood):
        raise ValueError("regime_log_likelihood_invalid")
    return likelihood


def _render_posterior(
    model: RegimeHMMModel, probabilities: tuple[float, ...]
) -> tuple[tuple[str, Decimal], ...]:
    return tuple(
        (name, _as_decimal(probability))
        for name, probability in zip(model.regimes, probabilities, strict=True)
    )


def _observation_fingerprint(
    observation: RegimeObservation,
    previous: PreviousRegimePosterior | None,
) -> str:
    return stable_fingerprint(
        {
            "observed_at": observation.observed_at,
            "evaluated_at": observation.evaluated_at,
            "symbol": observation.symbol,
            "timeframe": observation.timeframe,
            "feature_schema_version": observation.feature_schema_version,
            "feature_values": observation.feature_values,
            "data_fingerprint": observation.data_fingerprint,
            "previous_fingerprint": previous.fingerprint if previous else None,
        }
    )


def _build_no_trade(
    *,
    decision_id: str,
    model: RegimeHMMModel | None,
    observation: RegimeObservation | None,
    reasons: tuple[str, ...],
    posterior: tuple[tuple[str, Decimal], ...] = (),
) -> RegimeInference:
    model_fingerprint = model.fingerprint if model else None
    model_revision = model.revision if model else None
    evidence_fingerprint = (
        _observation_fingerprint(observation, None) if observation else None
    )
    payload = {
        "decision_id": decision_id,
        "status": RegimeInferenceStatus.NO_TRADE.value,
        "posterior": posterior,
        "model_revision": model_revision,
        "model_fingerprint": model_fingerprint,
        "evidence_fingerprint": evidence_fingerprint,
        "reason_codes": reasons,
        "order_permission": False,
    }
    return RegimeInference(
        status=RegimeInferenceStatus.NO_TRADE,
        decision_id=decision_id,
        selected_regime=None,
        posterior=posterior,
        confidence=None,
        model_revision=model_revision,
        model_fingerprint=model_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
        reason_codes=reasons,
        order_permission=False,
        fingerprint=stable_fingerprint(payload),
    )


def _require_aware(name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name}_timezone_required")


def _require_unique(name: str, values: tuple[str, ...], *, nonempty: bool) -> None:
    if nonempty and not values:
        raise ValueError(f"{name}_required")
    if any(not value.strip() for value in values):
        raise ValueError(f"{name}_contains_empty_value")
    if len(values) != len(set(values)):
        raise ValueError(f"{name}_contains_duplicates")


def _require_decimal(
    name: str,
    value: Decimal,
    *,
    minimum: Decimal | None = None,
    maximum: Decimal | None = None,
) -> None:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError(f"{name}_must_be_finite_decimal")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name}_below_minimum")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name}_above_maximum")


def _validate_probability_vector(
    name: str, values: tuple[Decimal, ...]
) -> None:
    if not values:
        raise ValueError(f"{name}_empty")
    for value in values:
        _require_decimal(
            name,
            value,
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
    if abs(sum(values, Decimal("0")) - Decimal("1")) > _PROBABILITY_TOLERANCE:
        raise ValueError(f"{name}_must_sum_to_one")

def _as_decimal(value: float) -> Decimal:
    if not math.isfinite(value):
        raise ValueError("regime_probability_not_finite")
    return Decimal(str(value))
