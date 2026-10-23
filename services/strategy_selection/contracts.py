"""Typed contracts for regime-aware strategy selection.

This package deliberately has no exchange client, order API, secret or risk-policy
write dependency. A selected strategy is a proposal candidate, not permission to trade.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from services.data_quality.gate import QualityResult


class StrategyLifecycle(StrEnum):
    RESEARCH_ONLY = "RESEARCH_ONLY"
    SELECTABLE = "SELECTABLE"
    RETIRED = "RETIRED"


class LicenseStatus(StrEnum):
    APPROVED = "APPROVED"
    CONDITIONAL = "CONDITIONAL"
    REJECTED = "REJECTED"


class SelectionStatus(StrEnum):
    SELECTED = "SELECTED"
    NO_TRADE = "NO_TRADE"


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def stable_fingerprint(value: object) -> str:
    rendered = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}_required")


def _require_hash(name: str, value: str, *, git_sha: bool = False) -> None:
    _require_text(name, value)
    pattern = _GIT_SHA if git_sha else _SHA256
    if pattern.fullmatch(value) is None:
        raise ValueError(f"{name}_invalid_hash")


def _require_aware(name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name}_timezone_required")


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


def _require_unique_texts(name: str, values: tuple[str, ...], *, nonempty: bool) -> None:
    if nonempty and not values:
        raise ValueError(f"{name}_required")
    for value in values:
        _require_text(name, value)
    if len(values) != len(set(values)):
        raise ValueError(f"{name}_contains_duplicates")


def _validate_performance(
    *,
    prefix: str,
    trade_count: int,
    win_rate: Decimal,
    net_expectancy: Decimal,
    max_abs_drawdown: Decimal,
) -> None:
    if isinstance(trade_count, bool) or not isinstance(trade_count, int) or trade_count < 0:
        raise ValueError(f"{prefix}_trade_count_invalid")
    _require_decimal(
        f"{prefix}_win_rate",
        win_rate,
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    _require_decimal(f"{prefix}_net_expectancy", net_expectancy)
    _require_decimal(
        f"{prefix}_max_abs_drawdown",
        max_abs_drawdown,
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )


@dataclass(frozen=True, slots=True)
class StrategyDefinition:
    """Immutable registry entry. SELECTABLE never means order permission."""

    strategy_id: str
    version: str
    code_sha256: str
    lifecycle: StrategyLifecycle
    owner_approved: bool
    eligible_regimes: tuple[str, ...]
    supported_timeframes: tuple[str, ...]
    data_schema_version: str
    feature_schema_version: str
    required_features: tuple[str, ...]
    required_test_ids: tuple[str, ...]
    prohibited_conditions: tuple[str, ...]
    risk_profile_id: str
    risk_profile_revision: str
    max_data_age_seconds: int
    max_model_score_age_seconds: int
    max_fee_bps: Decimal
    max_slippage_bps: Decimal
    max_spread_bps: Decimal
    minimum_quote_liquidity_usdt: Decimal
    strategy_license_status: LicenseStatus
    data_license_status: LicenseStatus
    dependency_license_status: LicenseStatus
    market_type: str = "spot"
    quote_currency: str = "USDT"
    allow_short: bool = False

    def __post_init__(self) -> None:
        _require_text("strategy_id", self.strategy_id)
        if re.fullmatch(r"[a-z][a-z0-9_]{1,63}", self.strategy_id) is None:
            raise ValueError("strategy_id_format_invalid")
        if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", self.version) is None:
            raise ValueError("strategy_version_must_be_semver")
        _require_hash("strategy_code", self.code_sha256)
        _require_text("data_schema_version", self.data_schema_version)
        _require_text("feature_schema_version", self.feature_schema_version)
        _require_text("risk_profile_id", self.risk_profile_id)
        _require_text("risk_profile_revision", self.risk_profile_revision)
        _require_unique_texts("eligible_regimes", self.eligible_regimes, nonempty=True)
        _require_unique_texts(
            "supported_timeframes", self.supported_timeframes, nonempty=True
        )
        _require_unique_texts("required_features", self.required_features, nonempty=True)
        _require_unique_texts("required_test_ids", self.required_test_ids, nonempty=True)
        _require_unique_texts(
            "prohibited_conditions", self.prohibited_conditions, nonempty=False
        )
        if self.market_type != "spot":
            raise ValueError("strategy_market_must_be_spot")
        if self.quote_currency != "USDT":
            raise ValueError("strategy_quote_must_be_usdt")
        if self.allow_short:
            raise ValueError("strategy_shorting_forbidden")
        if not isinstance(self.owner_approved, bool):
            raise ValueError("strategy_owner_approval_invalid")
        if self.max_data_age_seconds <= 0 or self.max_model_score_age_seconds <= 0:
            raise ValueError("strategy_freshness_limits_must_be_positive")
        for name in ("max_fee_bps", "max_slippage_bps", "max_spread_bps"):
            _require_decimal(name, getattr(self, name), minimum=Decimal("0"))
        _require_decimal(
            "minimum_quote_liquidity_usdt",
            self.minimum_quote_liquidity_usdt,
            minimum=Decimal("0.00000001"),
        )

    @property
    def fingerprint(self) -> str:
        return stable_fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class EconomicAcceptancePolicy:
    """Pre-registered OOS thresholds supplied by an explicitly approved policy."""

    revision: str
    owner_approved: bool
    minimum_trade_count: int
    minimum_win_rate: Decimal
    maximum_abs_drawdown: Decimal
    minimum_net_expectancy: Decimal
    max_evidence_age_seconds: int

    def __post_init__(self) -> None:
        _require_text("economic_policy_revision", self.revision)
        if not isinstance(self.owner_approved, bool):
            raise ValueError("economic_policy_approval_invalid")
        if (
            isinstance(self.minimum_trade_count, bool)
            or not isinstance(self.minimum_trade_count, int)
            or self.minimum_trade_count < 1
        ):
            raise ValueError("economic_policy_minimum_trade_count_invalid")
        _require_decimal(
            "economic_policy_minimum_win_rate",
            self.minimum_win_rate,
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
        _require_decimal(
            "economic_policy_maximum_abs_drawdown",
            self.maximum_abs_drawdown,
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
        _require_decimal(
            "economic_policy_minimum_net_expectancy",
            self.minimum_net_expectancy,
            minimum=Decimal("0"),
        )
        if self.max_evidence_age_seconds <= 0:
            raise ValueError("economic_policy_evidence_age_limit_invalid")

    @property
    def fingerprint(self) -> str:
        return stable_fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class PerformanceEvidence:
    """A costed performance measurement; drawdown is a positive absolute fraction."""

    trade_count: int
    win_rate: Decimal
    net_expectancy: Decimal
    max_abs_drawdown: Decimal

    def __post_init__(self) -> None:
        _validate_performance(
            prefix="performance",
            trade_count=self.trade_count,
            win_rate=self.win_rate,
            net_expectancy=self.net_expectancy,
            max_abs_drawdown=self.max_abs_drawdown,
        )


@dataclass(frozen=True, slots=True)
class FoldEvidence:
    fold_id: str
    metrics: PerformanceEvidence

    def __post_init__(self) -> None:
        _require_text("fold_id", self.fold_id)


@dataclass(frozen=True, slots=True)
class RegimeEvidence:
    regime: str
    metrics: PerformanceEvidence

    def __post_init__(self) -> None:
        _require_text("evidence_regime", self.regime)


@dataclass(frozen=True, slots=True)
class StrategyEvidence:
    """Immutable lineage from a strategy version to its OOS research evidence."""

    strategy_id: str
    strategy_version: str
    strategy_code_sha256: str
    evaluation_id: str
    evaluation_commit_sha: str
    dataset_sha256: str
    report_sha256: str
    feature_schema_version: str
    economic_policy_revision: str
    evaluated_at: datetime
    passed_test_ids: tuple[str, ...]
    metrics: PerformanceEvidence
    fee_bps: Decimal
    slippage_bps: Decimal
    folds: tuple[FoldEvidence, ...]
    regimes: tuple[RegimeEvidence, ...]

    def __post_init__(self) -> None:
        _require_text("evidence_strategy_id", self.strategy_id)
        _require_text("evidence_strategy_version", self.strategy_version)
        _require_hash("evidence_strategy_code", self.strategy_code_sha256)
        _require_text("evaluation_id", self.evaluation_id)
        _require_hash("evaluation_commit_sha", self.evaluation_commit_sha, git_sha=True)
        _require_hash("evidence_dataset", self.dataset_sha256)
        _require_hash("evidence_report", self.report_sha256)
        _require_text("evidence_feature_schema_version", self.feature_schema_version)
        _require_text("evidence_economic_policy_revision", self.economic_policy_revision)
        _require_aware("evidence_evaluated_at", self.evaluated_at)
        _require_unique_texts("passed_test_ids", self.passed_test_ids, nonempty=False)
        _require_decimal("evidence_fee_bps", self.fee_bps, minimum=Decimal("0"))
        _require_decimal("evidence_slippage_bps", self.slippage_bps, minimum=Decimal("0"))
        fold_ids = tuple(fold.fold_id for fold in self.folds)
        regime_names = tuple(item.regime for item in self.regimes)
        _require_unique_texts("evidence_fold_ids", fold_ids, nonempty=False)
        _require_unique_texts("evidence_regimes", regime_names, nonempty=False)

    @property
    def fingerprint(self) -> str:
        return stable_fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class SelectionContext:
    """Point-in-time market/data facts only; never includes an account or order client."""

    decision_id: str
    evaluated_at: datetime
    market_data_as_of: datetime
    symbol: str
    timeframe: str
    regime: str
    data_schema_version: str
    feature_schema_version: str
    available_features: tuple[str, ...]
    market_conditions: tuple[str, ...]
    data_quality: QualityResult | None
    data_fingerprint: str
    model_revision: str
    estimated_fee_bps: Decimal | None
    estimated_slippage_bps: Decimal | None
    spread_bps: Decimal | None
    quote_liquidity_usdt: Decimal | None

    def __post_init__(self) -> None:
        _require_text("decision_id", self.decision_id)
        _require_aware("selection_evaluated_at", self.evaluated_at)
        _require_aware("market_data_as_of", self.market_data_as_of)
        _require_text("selection_symbol", self.symbol)
        _require_text("selection_timeframe", self.timeframe)
        _require_text("selection_regime", self.regime)
        _require_text("selection_data_schema_version", self.data_schema_version)
        _require_text("selection_feature_schema_version", self.feature_schema_version)
        _require_text("selection_data_fingerprint", self.data_fingerprint)
        _require_text("selection_model_revision", self.model_revision)
        _require_unique_texts(
            "available_features", self.available_features, nonempty=False
        )
        _require_unique_texts(
            "market_conditions", self.market_conditions, nonempty=False
        )
        for name in (
            "estimated_fee_bps",
            "estimated_slippage_bps",
            "spread_bps",
            "quote_liquidity_usdt",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_decimal(name, value, minimum=Decimal("0"))


@dataclass(frozen=True, slots=True)
class ModelStrategyScore:
    strategy_id: str
    strategy_version: str
    model_revision: str
    scored_at: datetime
    score: Decimal

    def __post_init__(self) -> None:
        _require_text("score_strategy_id", self.strategy_id)
        _require_text("score_strategy_version", self.strategy_version)
        _require_text("score_model_revision", self.model_revision)
        _require_aware("strategy_score_scored_at", self.scored_at)
        _require_decimal(
            "strategy_score",
            self.score,
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )


@dataclass(frozen=True, slots=True)
class CandidateRejection:
    strategy_id: str
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text("rejected_strategy_id", self.strategy_id)
        _require_unique_texts("candidate_rejection_reasons", self.reasons, nonempty=True)


@dataclass(frozen=True, slots=True)
class StrategySelectionResult:
    decision_id: str
    evaluated_at: datetime | None
    status: SelectionStatus
    selected_strategy_id: str | None
    selected_strategy_version: str | None
    registry_revision: str | None
    registry_fingerprint: str | None
    economic_policy_revision: str | None
    evidence_fingerprint: str | None
    model_revision: str | None
    model_score: Decimal | None
    reason_codes: tuple[str, ...]
    candidate_rejections: tuple[CandidateRejection, ...]
    order_permission: bool
    fingerprint: str

    def __post_init__(self) -> None:
        _require_text("selection_result_decision_id", self.decision_id)
        if self.evaluated_at is not None:
            _require_aware("selection_result_evaluated_at", self.evaluated_at)
        if self.order_permission is not False:
            raise ValueError("strategy_selector_cannot_grant_order_permission")
        if self.status is SelectionStatus.SELECTED:
            if not self.selected_strategy_id or not self.selected_strategy_version:
                raise ValueError("selected_strategy_identity_missing")
            if self.model_score is None or self.evidence_fingerprint is None:
                raise ValueError("selected_strategy_evidence_missing")
        elif any(
            value is not None
            for value in (
                self.selected_strategy_id,
                self.selected_strategy_version,
                self.model_score,
                self.evidence_fingerprint,
            )
        ):
            raise ValueError("no_trade_result_must_not_name_selected_strategy")
        _require_unique_texts("selection_result_reasons", self.reason_codes, nonempty=True)
        _require_hash("selection_result_fingerprint", self.fingerprint)
