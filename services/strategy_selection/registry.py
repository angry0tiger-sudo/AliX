"""Strict JSON loader and immutable snapshot for the strategy registry."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from services.strategy_selection.contracts import (
    LicenseStatus,
    StrategyDefinition,
    StrategyLifecycle,
    stable_fingerprint,
)

REGISTRY_SCHEMA_VERSION = "alix.strategy-registry.v1"
_ROOT_KEYS = {
    "schema_version",
    "revision",
    "economic_policy_revision",
    "strategies",
}
_ENTRY_KEYS = {
    "strategy_id",
    "version",
    "code_sha256",
    "lifecycle",
    "owner_approved",
    "eligible_regimes",
    "supported_timeframes",
    "data_schema_version",
    "feature_schema_version",
    "required_features",
    "required_test_ids",
    "prohibited_conditions",
    "risk_profile_id",
    "risk_profile_revision",
    "max_data_age_seconds",
    "max_model_score_age_seconds",
    "max_fee_bps",
    "max_slippage_bps",
    "max_spread_bps",
    "minimum_quote_liquidity_usdt",
    "strategy_license_status",
    "data_license_status",
    "dependency_license_status",
    "market_type",
    "quote_currency",
    "allow_short",
}


@dataclass(frozen=True, slots=True)
class StrategyRegistry:
    revision: str
    economic_policy_revision: str
    strategies: tuple[StrategyDefinition, ...]
    schema_version: str = REGISTRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.revision.strip() or not self.economic_policy_revision.strip():
            raise ValueError("strategy_registry_revision_missing")
        if self.schema_version != REGISTRY_SCHEMA_VERSION:
            raise ValueError("strategy_registry_schema_version_unsupported")
        ids = tuple(item.strategy_id for item in self.strategies)
        if len(ids) != len(set(ids)):
            raise ValueError("strategy_registry_ids_duplicated")
        object.__setattr__(
            self,
            "strategies",
            tuple(sorted(self.strategies, key=lambda item: item.strategy_id)),
        )

    @property
    def fingerprint(self) -> str:
        return stable_fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class RegistryLoadResult:
    registry: StrategyRegistry | None
    reason_codes: tuple[str, ...]


def _object(value: object, *, expected_keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("registry_object_schema_invalid")
    return value


def _text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("registry_text_field_invalid")
    return value


def _boolean(value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError("registry_boolean_field_invalid")
    return value


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("registry_integer_field_invalid")
    return value


def _texts(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("registry_text_list_invalid")
    return tuple(_text(item) for item in value)


def _decimal(value: object) -> Decimal:
    if not isinstance(value, str):
        raise ValueError("registry_decimal_must_be_string")
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("registry_decimal_invalid") from error
    if not result.is_finite():
        raise ValueError("registry_decimal_must_be_finite")
    return result


def _definition(raw: object) -> StrategyDefinition:
    item = _object(raw, expected_keys=_ENTRY_KEYS)
    return StrategyDefinition(
        strategy_id=_text(item["strategy_id"]),
        version=_text(item["version"]),
        code_sha256=_text(item["code_sha256"]),
        lifecycle=StrategyLifecycle(_text(item["lifecycle"])),
        owner_approved=_boolean(item["owner_approved"]),
        eligible_regimes=_texts(item["eligible_regimes"]),
        supported_timeframes=_texts(item["supported_timeframes"]),
        data_schema_version=_text(item["data_schema_version"]),
        feature_schema_version=_text(item["feature_schema_version"]),
        required_features=_texts(item["required_features"]),
        required_test_ids=_texts(item["required_test_ids"]),
        prohibited_conditions=_texts(item["prohibited_conditions"]),
        risk_profile_id=_text(item["risk_profile_id"]),
        risk_profile_revision=_text(item["risk_profile_revision"]),
        max_data_age_seconds=_integer(item["max_data_age_seconds"]),
        max_model_score_age_seconds=_integer(item["max_model_score_age_seconds"]),
        max_fee_bps=_decimal(item["max_fee_bps"]),
        max_slippage_bps=_decimal(item["max_slippage_bps"]),
        max_spread_bps=_decimal(item["max_spread_bps"]),
        minimum_quote_liquidity_usdt=_decimal(item["minimum_quote_liquidity_usdt"]),
        strategy_license_status=LicenseStatus(_text(item["strategy_license_status"])),
        data_license_status=LicenseStatus(_text(item["data_license_status"])),
        dependency_license_status=LicenseStatus(
            _text(item["dependency_license_status"])
        ),
        market_type=_text(item["market_type"]),
        quote_currency=_text(item["quote_currency"]),
        allow_short=_boolean(item["allow_short"]),
    )


def load_registry_document(document: str) -> RegistryLoadResult:
    """Load strict versioned JSON. Invalid documents produce no usable registry."""
    try:
        decoded = json.loads(document)
        raw = _object(decoded, expected_keys=_ROOT_KEYS)
        if raw["schema_version"] != REGISTRY_SCHEMA_VERSION:
            raise ValueError("strategy_registry_schema_version_unsupported")
        if not isinstance(raw["strategies"], list):
            raise ValueError("strategy_registry_strategies_must_be_list")
        registry = StrategyRegistry(
            revision=_text(raw["revision"]),
            economic_policy_revision=_text(raw["economic_policy_revision"]),
            schema_version=_text(raw["schema_version"]),
            strategies=tuple(_definition(entry) for entry in raw["strategies"]),
        )
    except (ValueError, TypeError, KeyError, InvalidOperation, json.JSONDecodeError):
        return RegistryLoadResult(
            registry=None,
            reason_codes=("strategy_registry_document_invalid",),
        )
    return RegistryLoadResult(registry=registry, reason_codes=())
