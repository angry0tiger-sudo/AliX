"""Versioned, fail-closed contracts for external data and temporal NLP features."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import html
import math
import re
import unicodedata
from urllib.parse import parse_qsl, urlsplit

from services.data_quality.gate import (
    EXTERNAL_RECORD_SCHEMA_VERSION,
    QualityStatus,
    validate_external_record,
)

NEWS_RECORD_SCHEMA_VERSION = "alix.news.event.v1"
NLP_RESULT_SCHEMA_VERSION = "alix.nlp.sentiment.v1"
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
PROVIDER_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
ALLOWED_SENTIMENTS = frozenset({"positive", "negative", "neutral"})
SENSITIVE_QUERY_KEYS = frozenset(
    {"key", "api_key", "apikey", "token", "secret", "password", "authorization"}
)


class ProviderError(ValueError):
    """A provider response or record violated its contract."""


class ProviderPolicyError(ProviderError):
    """A provider is inactive, unapproved, or otherwise disallowed."""


class RateLimitStatus:
    OK = "ok"
    LIMITED = "limited"
    BLOCKED = "blocked"
    VALUES = frozenset({OK, LIMITED, BLOCKED})


def parse_timestamp(value: datetime | str | None, field_name: str) -> datetime:
    """Parse an explicit timezone-aware timestamp and return UTC."""
    if value is None:
        raise ProviderError(f"missing_timestamp:{field_name}")
    if isinstance(value, str):
        rendered = value.strip()
        if not rendered:
            raise ProviderError(f"missing_timestamp:{field_name}")
        if rendered.endswith("Z"):
            rendered = f"{rendered[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(rendered)
        except ValueError:
            raise ProviderError(f"invalid_timestamp:{field_name}") from None
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise ProviderError(f"invalid_timestamp:{field_name}")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ProviderError(f"timestamp_timezone_required:{field_name}")
    return parsed.astimezone(timezone.utc)


def normalize_news_text(value: str) -> str:
    """Normalize markup, Unicode and whitespace for deterministic hashing."""
    without_markup = re.sub(r"<[^>]*>", " ", html.unescape(value))
    normalized = unicodedata.normalize("NFKC", without_markup)
    return " ".join(normalized.split())


def digest_content(value: str) -> str:
    """Hash normalized text, independent of harmless case and formatting."""
    canonical = normalize_news_text(value).casefold()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _require_text(values: tuple[str, ...], reason: str) -> None:
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ProviderError(reason)


def _require_provider_text(values: tuple[str, ...]) -> None:
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ProviderPolicyError("provider_registration_incomplete")


def _validate_provider_endpoint(endpoint_url: str, is_test_fixture: bool) -> None:
    parts = urlsplit(endpoint_url)
    if parts.scheme != "https" or not parts.hostname:
        raise ProviderPolicyError("provider_endpoint_must_use_https")
    if parts.username or parts.password:
        raise ProviderPolicyError("provider_endpoint_credentials_forbidden")
    if any(key.lower() in SENSITIVE_QUERY_KEYS for key, _ in parse_qsl(parts.query)):
        raise ProviderPolicyError("provider_endpoint_must_not_contain_credentials")
    if is_test_fixture and not parts.hostname.endswith(".invalid"):
        raise ProviderPolicyError("test_fixture_endpoint_must_use_invalid_domain")


def _validate_provider_identity(provider_id: str, data_kind: str) -> None:
    if not PROVIDER_ID_PATTERN.fullmatch(provider_id):
        raise ProviderPolicyError("invalid_provider_id")
    if data_kind not in {"news", "market_analytics"}:
        raise ProviderPolicyError("unsupported_provider_kind")


def _validate_provider_status(
    provider_id: str,
    license_status: str,
    active: bool,
    is_test_fixture: bool,
    algorithmic_non_display_approved: bool,
) -> None:
    if not is_test_fixture and license_status == "test_only":
        raise ProviderPolicyError("test_only_provider_cannot_be_runtime")
    if is_test_fixture and license_status != "approved":
        raise ProviderPolicyError("fixture_license_must_be_test_approved")
    is_blocked_tradingview = (
        "tradingview" in provider_id.casefold()
        and active
        and not algorithmic_non_display_approved
    )
    if is_blocked_tradingview:
        raise ProviderPolicyError("tradingview_non_display_blocked")


def _assert_provider_approved(
    provider_id: str,
    license_status: str,
    active: bool,
    algorithmic_non_display_approved: bool,
) -> None:
    if not active:
        raise ProviderPolicyError("provider_inactive")
    if license_status != "approved":
        raise ProviderPolicyError("provider_license_not_approved")
    if "tradingview" in provider_id.casefold() and not algorithmic_non_display_approved:
        raise ProviderPolicyError("tradingview_non_display_blocked")


@dataclass(frozen=True, slots=True)
class ProviderRegistration:
    """A provider's reviewed scope and access policy."""

    provider_id: str
    data_kind: str
    endpoint_url: str
    license_status: str
    rate_limit_policy: str
    retention_policy: str
    timestamp_semantics: str
    owner: str
    active: bool = False
    is_test_fixture: bool = False
    algorithmic_non_display_approved: bool = False

    def __post_init__(self) -> None:
        _validate_provider_identity(self.provider_id, self.data_kind)
        _require_provider_text(
            (
                self.endpoint_url,
                self.license_status,
                self.rate_limit_policy,
                self.retention_policy,
                self.timestamp_semantics,
                self.owner,
            )
        )
        _validate_provider_endpoint(self.endpoint_url, self.is_test_fixture)
        _validate_provider_status(
            self.provider_id,
            self.license_status,
            self.active,
            self.is_test_fixture,
            self.algorithmic_non_display_approved,
        )

    def assert_ingestion_allowed(self) -> None:
        """Reject before transport access unless source approval is explicit."""
        _assert_provider_approved(
            self.provider_id,
            self.license_status,
            self.active,
            self.algorithmic_non_display_approved,
        )


@dataclass(frozen=True, slots=True)
class ConnectorResponse:
    """A response returned by an injected test or approved transport."""

    status_code: int
    body: str
    received_at: datetime
    rate_limit_status: str = RateLimitStatus.OK


@dataclass(frozen=True, slots=True)
class NewsEvent:
    """Normalized source text before local NLP inference."""

    event_id: str
    provider_id: str
    source_id: str
    published_at: datetime
    ingested_at: datetime
    available_at: datetime | None
    revision: str
    asset_mapping: tuple[str, ...]
    license_status: str
    content_hash: str
    text: str
    title: str = ""
    processed_at: datetime | None = None
    schema_version: str = NEWS_RECORD_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class MarketObservation:
    """A normalized numeric observation from a market-analytics provider."""

    event_id: str
    provider_id: str
    source_id: str
    market: str
    event_timestamp: datetime
    ingested_at: datetime
    available_at: datetime | None
    revision: str
    value: float
    unit: str
    quote_currency: str
    license_status: str
    rate_limit_status: str
    content_hash: str
    schema_version: str = EXTERNAL_RECORD_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class NewsFeature:
    """Auditable NLP output, available no earlier than inference completion."""

    event_id: str
    provider_id: str
    source_id: str
    published_at: datetime
    ingested_at: datetime
    available_at: datetime
    processed_at: datetime
    revision: str
    asset_mapping: tuple[str, ...]
    license_status: str
    content_hash: str
    sentiment_label: str
    sentiment_score: float
    confidence: float
    event_type: str
    model_id: str
    model_revision: str
    calibration_status: str = "uncalibrated"
    schema_version: str = NLP_RESULT_SCHEMA_VERSION


def _validate_news_schema_and_rights(event: NewsEvent) -> None:
    _require_text(
        (
            event.event_id,
            event.provider_id,
            event.source_id,
            event.revision,
            event.license_status,
            event.content_hash,
            event.text,
        ),
        "news_record_required_field_missing",
    )
    if event.schema_version != NEWS_RECORD_SCHEMA_VERSION:
        raise ProviderError("news_schema_version_mismatch")
    if event.license_status != "approved":
        raise ProviderPolicyError("source_not_approved")
    if not HASH_PATTERN.fullmatch(event.content_hash):
        raise ProviderError("news_content_hash_invalid")


def _validate_news_timeline(event: NewsEvent) -> None:
    published = parse_timestamp(event.published_at, "published_at")
    ingested = parse_timestamp(event.ingested_at, "ingested_at")
    available = parse_timestamp(event.available_at, "available_at")
    if published > ingested:
        raise ProviderError("invalid_timestamp_order:published_after_ingest")
    if ingested > available:
        raise ProviderError("invalid_timestamp_order:ingest_after_available")
    if event.processed_at is not None:
        processed = parse_timestamp(event.processed_at, "processed_at")
        if processed < available:
            raise ProviderError("invalid_timestamp_order:processed_before_available")


def _validate_asset_mapping(assets: tuple[str, ...], reason: str) -> None:
    if any(not isinstance(asset, str) or not asset.strip() for asset in assets):
        raise ProviderError(reason)


def validate_news_event(event: NewsEvent) -> None:
    """Validate source rights, schema and temporal provenance."""
    _validate_news_schema_and_rights(event)
    _validate_news_timeline(event)
    _validate_asset_mapping(event.asset_mapping, "news_asset_mapping_invalid")


def _validate_market_fields(observation: MarketObservation) -> None:
    _require_text(
        (
            observation.event_id,
            observation.provider_id,
            observation.source_id,
            observation.market,
            observation.revision,
            observation.unit,
            observation.quote_currency,
            observation.license_status,
            observation.rate_limit_status,
            observation.content_hash,
        ),
        "market_record_required_field_missing",
    )
    if observation.schema_version != EXTERNAL_RECORD_SCHEMA_VERSION:
        raise ProviderError("market_schema_version_mismatch")
    value = observation.value
    valid_value = (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )
    if not valid_value:
        raise ProviderError("market_value_non_finite_or_invalid")
    if not HASH_PATTERN.fullmatch(observation.content_hash):
        raise ProviderError("market_content_hash_invalid")


def _validate_market_provenance(
    observation: MarketObservation,
) -> tuple[datetime, datetime]:
    event_time = parse_timestamp(observation.event_timestamp, "event_timestamp")
    ingested = parse_timestamp(observation.ingested_at, "ingested_at")
    available = parse_timestamp(observation.available_at, "available_at")
    if ingested > available:
        raise ProviderError("invalid_timestamp_order:ingest_after_available")
    if observation.license_status != "approved":
        raise ProviderPolicyError("source_not_approved")
    if observation.rate_limit_status != RateLimitStatus.OK:
        raise ProviderPolicyError("rate_limit_policy_not_approved")
    return event_time, available


def _run_market_quality_gate(
    observation: MarketObservation,
    event_time: datetime,
    available: datetime,
) -> None:
    quality = validate_external_record(
        {
            "schema_version": observation.schema_version,
            "provider_id": observation.provider_id,
            "source_id": observation.source_id,
            "event_id": observation.event_id,
            "event_timestamp": event_time.isoformat(),
            "ingested_at": parse_timestamp(
                observation.ingested_at, "ingested_at"
            ).isoformat(),
            "available_at": available.isoformat(),
            "unit": observation.unit,
            "quote_currency": observation.quote_currency,
            "revision": observation.revision,
            "license_status": observation.license_status,
            "rate_limit_status": observation.rate_limit_status,
            "content_hash": observation.content_hash,
        },
        as_of_timestamp=available,
    )
    if quality.passed and quality.status is QualityStatus.VALID:
        return
    reasons = ",".join(quality.reasons) or "unknown_quality_failure"
    raise ProviderError(f"market_record_quality_rejected:{reasons}")


def validate_market_observation(observation: MarketObservation) -> None:
    """Validate a numeric record against provider rules and the shared DQ gate."""
    _validate_market_fields(observation)
    event_time, available = _validate_market_provenance(observation)
    _run_market_quality_gate(observation, event_time, available)


def _validate_news_feature_fields(feature: NewsFeature) -> None:
    _require_text(
        (
            feature.event_id,
            feature.provider_id,
            feature.source_id,
            feature.revision,
            feature.license_status,
            feature.content_hash,
            feature.model_id,
            feature.model_revision,
            feature.event_type,
        ),
        "nlp_feature_required_field_missing",
    )
    if feature.schema_version != NLP_RESULT_SCHEMA_VERSION:
        raise ProviderError("nlp_schema_version_mismatch")
    if feature.license_status != "approved":
        raise ProviderPolicyError("source_not_approved")
    if not HASH_PATTERN.fullmatch(feature.content_hash):
        raise ProviderError("nlp_content_hash_invalid")
    if feature.sentiment_label not in ALLOWED_SENTIMENTS:
        raise ProviderError("nlp_sentiment_label_invalid")


def _validate_numeric_range(
    name: str, value: float, lower: float, upper: float
) -> None:
    valid_number = (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and lower <= value <= upper
    )
    if not valid_number:
        raise ProviderError(f"nlp_{name}_invalid")


def _validate_nlp_timeline(feature: NewsFeature) -> None:
    published = parse_timestamp(feature.published_at, "published_at")
    ingested = parse_timestamp(feature.ingested_at, "ingested_at")
    available = parse_timestamp(feature.available_at, "available_at")
    processed = parse_timestamp(feature.processed_at, "processed_at")
    if published > ingested or ingested > processed or processed > available:
        raise ProviderError("invalid_timestamp_order:nlp_feature")


def validate_news_feature(feature: NewsFeature) -> None:
    """Reject non-auditable NLP features before they are stored or joined."""
    _validate_news_feature_fields(feature)
    _validate_numeric_range("sentiment_score", feature.sentiment_score, -1.0, 1.0)
    _validate_numeric_range("confidence", feature.confidence, 0.0, 1.0)
    _validate_nlp_timeline(feature)
    if feature.calibration_status not in {"uncalibrated", "calibrated"}:
        raise ProviderError("nlp_calibration_status_invalid")
    _validate_asset_mapping(feature.asset_mapping, "nlp_asset_mapping_invalid")
