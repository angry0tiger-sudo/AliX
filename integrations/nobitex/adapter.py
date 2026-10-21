"""Novaryn-derived Nobitex contract for AliX Phase 1.

This module intentionally supplies no concrete HTTP transport and no order-write
path. Production transport/authentication requires a later, explicitly approved
integration gate. Unit tests inject transport doubles only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol


class NobitexAdapterError(RuntimeError):
    """Base exception for the AliX Nobitex integration boundary."""


class LiveExecutionDisabled(NobitexAdapterError):
    """Raised whenever order submission is attempted in this phase."""


class ReadOnlyTransportUnavailable(NobitexAdapterError):
    """No injected read-only transport is configured."""


class MarketMappingError(ValueError):
    """Exchange market metadata cannot be normalized safely."""


class UnknownOrderStatus(ValueError):
    """Nobitex returned an unmapped state; reconciliation is required."""


class ReadOnlyTransport(Protocol):
    """Minimal read-only HTTP boundary; no write method is intentionally defined."""

    def get_json(
        self, path: str, params: Mapping[str, str] | None = None
    ) -> Mapping[str, Any]:
        """Return a decoded JSON object for a GET request."""
        ...


@dataclass(frozen=True, slots=True)
class SpotMarketMetadata:
    exchange_symbol: str
    normalized_symbol: str
    base: str
    quote: str
    settlement: str
    market_type: str
    price_precision: int | None
    amount_precision: int | None
    min_amount: Decimal | None
    max_amount: Decimal | None
    min_price: Decimal | None
    max_price: Decimal | None
    supported_order_types: tuple[str, ...]
    supported_sides: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_market_identity(
            self.exchange_symbol,
            self.normalized_symbol,
            self.base,
            self.quote,
            self.settlement,
            self.market_type,
            self.supported_order_types,
            self.supported_sides,
        )
        _validate_market_limits(
            (self.min_amount, self.max_amount, self.min_price, self.max_price),
            (self.price_precision, self.amount_precision),
        )


def _validate_market_identity(
    exchange_symbol: str,
    normalized_symbol: str,
    base: str,
    quote: str,
    settlement: str,
    market_type: str,
    order_types: tuple[str, ...],
    sides: tuple[str, ...],
) -> None:
    required = (
        exchange_symbol,
        normalized_symbol,
        base,
        quote,
        settlement,
        market_type,
    )
    if not all(value.strip() for value in required):
        raise MarketMappingError("required market identity is missing")
    if market_type.lower() != "spot":
        raise MarketMappingError("only spot market metadata is supported")
    if base == quote:
        raise MarketMappingError("market base and quote must differ")
    if not order_types or not sides:
        raise MarketMappingError("supported order types and sides are required")


def _validate_market_limits(
    limits: tuple[Decimal | None, ...],
    precisions: tuple[int | None, ...],
) -> None:
    for value in limits:
        if value is not None and not _valid_nonnegative_decimal(value):
            raise MarketMappingError("market limits must be finite and non-negative")
    for precision in precisions:
        if precision is not None and precision < 0:
            raise MarketMappingError("precision must be non-negative")


def _valid_nonnegative_decimal(value: Decimal) -> bool:
    return value.is_finite() and value >= 0


class OrderStatus:
    SUBMITTED = "SUBMITTED"
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


_STATUS_MAP = {
    "NEW": OrderStatus.SUBMITTED,
    "ACTIVE": OrderStatus.OPEN,
    "DONE": OrderStatus.FILLED,
    "CANCELED": OrderStatus.CANCELLED,
    "CANCELLED": OrderStatus.CANCELLED,
    "INACTIVE": OrderStatus.REJECTED,
}


def _text(value: object) -> str:
    return str(value or "").strip()


def _currency(value: object) -> str:
    currency = _text(value).upper()
    if currency in {"RLS", "IRR"}:
        return "IRT"
    return currency


def _exchange_market_symbol(value: object) -> str:
    """Normalize separators used by Nobitex market keys without merging underscores."""
    symbol = _text(value).upper()
    if not symbol:
        raise MarketMappingError("empty Nobitex market symbol")
    return symbol.replace("-", "").replace("/", "")


def _compact_market_symbol(value: object) -> str:
    """Build the canonical separator-free key used for market-resolution lookups."""
    symbol = _text(value).upper()
    if not symbol:
        raise MarketMappingError("empty market symbol")
    return symbol.replace("-", "").replace("/", "").replace("_", "")


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise MarketMappingError("invalid precision metadata")
    try:
        result = int(value)
    except ValueError as exc:
        raise MarketMappingError("invalid precision metadata") from exc
    if result < 0:
        raise MarketMappingError("precision must be non-negative")
    return result


def _optional_decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise MarketMappingError("invalid decimal market metadata") from exc
    if not result.is_finite() or result < 0:
        raise MarketMappingError("market limits must be finite and non-negative")
    return result


def _market_identity(
    payload: Mapping[str, Any],
) -> tuple[str, str, str]:
    exchange_symbol = _text(payload.get("symbol") or payload.get("market")).upper()
    base = _currency(payload.get("base") or payload.get("srcCurrency"))
    quote = _currency(payload.get("quote") or payload.get("dstCurrency"))
    if not exchange_symbol or not base or not quote:
        raise MarketMappingError("exchange symbol, base and quote are required")
    return exchange_symbol, base, quote


def _market_capabilities(
    payload: Mapping[str, Any],
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    market_type = _text(
        payload.get("marketType") or payload.get("type") or "spot"
    ).lower()
    if market_type not in {"spot", "spotmarket"}:
        raise MarketMappingError(f"unsupported market type: {market_type}")
    order_types = _normalize_strings(
        payload.get("supportedOrderTypes") or payload.get("orderTypes")
    )
    sides = _normalize_strings(payload.get("supportedSides") or payload.get("sides"))
    if not order_types or not sides:
        raise MarketMappingError("missing supported order types or sides")
    return "spot", order_types, sides


def normalize_spot_market(payload: Mapping[str, Any]) -> SpotMarketMetadata:
    """Normalize exchange metadata following Novaryn's strict spot-market contract."""
    exchange_symbol, base, quote = _market_identity(payload)
    market_type, order_types, sides = _market_capabilities(payload)
    return SpotMarketMetadata(
        exchange_symbol=exchange_symbol,
        normalized_symbol=f"{base}/{quote}",
        base=base,
        quote=quote,
        settlement=quote,
        market_type=market_type,
        price_precision=_optional_int(payload.get("pricePrecision")),
        amount_precision=_optional_int(payload.get("amountPrecision")),
        min_amount=_optional_decimal(payload.get("minAmount")),
        max_amount=_optional_decimal(payload.get("maxAmount")),
        min_price=_optional_decimal(payload.get("minPrice")),
        max_price=_optional_decimal(payload.get("maxPrice")),
        supported_order_types=order_types,
        supported_sides=sides,
    )


def _normalize_strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(
        sorted({str(item).strip().lower() for item in value if str(item).strip()})
    )


def normalize_spot_pair(pair: str) -> str:
    """Accept only canonical BASE/USDT spot symbols, as specified by Novaryn."""
    parts = pair.strip().upper().split("/")
    if len(parts) != 2:
        raise ValueError("pair must use BASE/USDT notation")
    base, quote = parts
    if not base or quote != "USDT" or base == quote:
        raise ValueError(
            "Nobitex adapter contract is restricted to BASE/USDT spot pairs"
        )
    if not all(part.isalnum() for part in (base, quote)):
        raise ValueError("pair contains unsupported characters")
    return f"{base}/{quote}"


def _decimal(value: object, field_name: str) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a decimal number") from exc
    if not number.is_finite() or number <= 0:
        raise ValueError(f"{field_name} must be positive and finite")
    return number


def map_order_status(
    status: str, *, matched_amount: object | None = None, amount: object | None = None
) -> str:
    """Map known states and distinguish partial fills; ambiguity fails closed."""
    normalized = status.strip().upper()
    try:
        mapped = _STATUS_MAP[normalized]
    except KeyError as exc:
        raise UnknownOrderStatus(f"unknown Nobitex order status: {status}") from exc

    if matched_amount is None or amount is None:
        return mapped
    try:
        matched = Decimal(str(matched_amount))
        total = Decimal(str(amount))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise UnknownOrderStatus("invalid matched or total order amount") from exc
    if (
        not matched.is_finite()
        or not total.is_finite()
        or matched < 0
        or total <= 0
        or matched > total
    ):
        raise UnknownOrderStatus("matched order amount is outside valid bounds")
    if mapped in {OrderStatus.SUBMITTED, OrderStatus.OPEN}:
        if matched == total:
            return OrderStatus.FILLED
        if matched > 0:
            return OrderStatus.PARTIALLY_FILLED
    return mapped


class NobitexAdapter:
    """Read-only adapter facade plus pure order-payload validation helpers."""

    def __init__(self, transport: ReadOnlyTransport | None = None) -> None:
        self._transport = transport

    def get_exchange_market_symbols(self) -> dict[str, str]:
        """Resolve canonical USDT pairs from the Novaryn public market-stats contract."""
        if self._transport is None:
            raise ReadOnlyTransportUnavailable(
                "no read-only transport is installed; Phase 1 never opens a live connection"
            )
        payload = self._transport.get_json("/market/stats")
        raw_stats = payload.get("stats")
        if not isinstance(raw_stats, Mapping):
            raise NobitexAdapterError(
                "invalid market stats response while resolving symbols"
            )

        resolved: dict[str, str] = {}
        for raw_symbol in raw_stats:
            exchange_symbol = _exchange_market_symbol(raw_symbol)
            canonical = _compact_market_symbol(exchange_symbol)
            if not canonical.endswith("USDT"):
                continue
            previous = resolved.get(canonical)
            if previous is not None and previous != exchange_symbol:
                raise MarketMappingError(
                    f"ambiguous Nobitex market symbol mapping for {canonical}"
                )
            resolved[canonical] = exchange_symbol
        if not resolved:
            raise NobitexAdapterError(
                "no USDT Nobitex markets available for symbol resolution"
            )
        return dict(sorted(resolved.items()))

    def market_stats(self, pair: str) -> Mapping[str, Any]:
        normalized = normalize_spot_pair(pair)
        if self._transport is None:
            raise ReadOnlyTransportUnavailable(
                "no read-only transport is installed; Phase 1 never opens a live connection"
            )
        base, quote = normalized.split("/")
        result = self._transport.get_json(
            "/market/stats",
            {"srcCurrency": base.lower(), "dstCurrency": quote.lower()},
        )
        if not isinstance(result, Mapping):
            raise NobitexAdapterError("Nobitex market stats response must be an object")
        return result

    @staticmethod
    def build_order_payload(
        *,
        pair: str,
        side: str,
        order_type: str,
        amount: object,
        client_order_id: str,
        price: object | None = None,
    ) -> dict[str, str]:
        """Build and validate the Novaryn-style Spot/USDT payload without sending it."""
        normalized_pair = normalize_spot_pair(pair)
        base, quote = normalized_pair.split("/")
        normalized_side = side.strip().lower()
        normalized_type = order_type.strip().lower()
        if normalized_side not in {"buy", "sell"}:
            raise ValueError("side must be buy or sell")
        if normalized_type not in {"limit", "market"}:
            raise ValueError("order_type must be limit or market")
        normalized_id = client_order_id.strip()
        if not normalized_id or len(normalized_id) > 32:
            raise ValueError("client_order_id must contain 1..32 characters")
        payload = {
            "type": normalized_side,
            "execution": normalized_type,
            "srcCurrency": base.lower(),
            "dstCurrency": quote.lower(),
            "amount": str(_decimal(amount, "amount")),
            "clientOrderId": normalized_id,
            "tradeType": "spot",
        }
        if price is not None:
            payload["price"] = str(_decimal(price, "price"))
        if normalized_type == "limit" and price is None:
            raise ValueError("limit order requires a positive price")
        return payload

    def submit_order(self, **_: object) -> None:
        """Order writes are intentionally impossible in Phase 1."""
        raise LiveExecutionDisabled(
            "live/order-write transport is not part of Phase 1; reconcile and authorize in a later gate"
        )
