"""Pure Nobitex Spot/USDT response contracts; this module performs no I/O.

The models normalize injected fixtures and adapter responses. They do not prove
that the provider's current live schema matches these contracts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from integrations.nobitex.adapter import (
    MarketMappingError,
    SpotMarketMetadata,
    map_order_status,
    normalize_spot_pair,
)


@dataclass(frozen=True, slots=True)
class SpotBalanceSnapshot:
    currency: str
    total: Decimal
    blocked: Decimal

    @property
    def available(self) -> Decimal:
        return self.total - self.blocked


@dataclass(frozen=True, slots=True)
class SpotOrderSnapshot:
    exchange_order_id: str
    status: str
    side: str
    pair: str
    amount: Decimal
    matched_amount: Decimal
    price: Decimal | None
    client_order_id: str | None
    execution: str
    average_price: Decimal | None
    total_price: Decimal
    fee_cost: Decimal
    fee_currency: str | None


@dataclass(frozen=True, slots=True)
class PaginationRequest:
    page: int = 1
    page_size: int = 100

    def __post_init__(self) -> None:
        if self.page < 1:
            raise ValueError("page must be positive")
        if self.page_size < 1 or self.page_size > 100:
            raise ValueError("page_size must be within 1..100")


@dataclass(frozen=True, slots=True)
class ReadRetryPolicy:
    """Retry policy for idempotent read operations only, never order mutations."""

    max_retries: int = 2
    retryable_status_codes: frozenset[int] = frozenset({408, 429})

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if any(code < 100 or code > 599 for code in self.retryable_status_codes):
            raise ValueError("retryable HTTP status codes must be within 100..599")

    def allows_retry(self, status_code: int, retries_used: int) -> bool:
        if status_code < 100 or status_code > 599:
            raise ValueError("status_code must be within 100..599")
        if retries_used < 0:
            raise ValueError("retries_used cannot be negative")
        if retries_used >= self.max_retries:
            return False
        return status_code in self.retryable_status_codes or status_code >= 500


def _decimal(value: object, field_name: str, *, allow_zero: bool) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise MarketMappingError(f"invalid decimal in {field_name}") from exc
    if not result.is_finite() or result < 0 or (not allow_zero and result == 0):
        qualifier = "non-negative finite" if allow_zero else "positive finite"
        raise MarketMappingError(f"{field_name} must be {qualifier}")
    return result


def _currency(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise MarketMappingError("currency is required")
    aliases = {
        "BITCOIN": "BTC",
        "TETHER": "USDT",
        "ETHEREUM": "ETH",
        "DOGECOIN": "DOGE",
        "TRON": "TRX",
        "LITECOIN": "LTC",
        "RIPPLE": "XRP",
        "RLS": "IRT",
        "IRR": "IRT",
        "﷼": "IRT",
    }
    return aliases.get(raw.upper(), raw.upper())


def normalize_spot_balance(
    payload: Mapping[str, Any], currency: str
) -> SpotBalanceSnapshot:
    """Normalize one wallet from a read-only balance response."""
    normalized_currency = _currency(currency)
    wallets = payload.get("wallets")
    if not isinstance(wallets, Mapping):
        raise MarketMappingError("wallet response must contain a wallets object")

    wallet: Mapping[str, Any] | None = None
    for raw_currency, raw_wallet in wallets.items():
        if _currency(raw_currency) == normalized_currency:
            if not isinstance(raw_wallet, Mapping):
                raise MarketMappingError("wallet entry must be an object")
            wallet = raw_wallet
            break
    if wallet is None:
        raise MarketMappingError(f"wallet was not returned: {normalized_currency}")

    total = _decimal(wallet.get("balance"), "wallet.balance", allow_zero=True)
    blocked = _decimal(wallet.get("blocked"), "wallet.blocked", allow_zero=True)
    if blocked > total:
        raise MarketMappingError("blocked wallet amount exceeds total balance")
    return SpotBalanceSnapshot(normalized_currency, total, blocked)


def _order_pair(order: Mapping[str, Any]) -> str:
    source = order.get("srcCurrency")
    destination = order.get("dstCurrency")
    if source and destination:
        return normalize_spot_pair(f"{_currency(source)}/{_currency(destination)}")

    market = str(order.get("market") or "").strip().upper()
    if "/" in market:
        parts = market.split("/")
    elif "-" in market:
        parts = market.split("-", maxsplit=1)
    elif market.endswith("USDT") and len(market) > len("USDT"):
        parts = [market[:-4], "USDT"]
    else:
        raise MarketMappingError("order must identify a BASE/USDT market")
    if len(parts) != 2:
        raise MarketMappingError("order market symbol is ambiguous")
    return normalize_spot_pair(f"{_currency(parts[0])}/{_currency(parts[1])}")


def normalize_spot_order(
    payload: Mapping[str, Any], *, fallback_order_id: str | None = None
) -> SpotOrderSnapshot:
    """Normalize a Nobitex-like order response and fail closed on ambiguity."""
    raw_order: object = payload.get("order", payload)
    if not isinstance(raw_order, Mapping):
        raise MarketMappingError("order response must contain an order object")
    order = raw_order

    order_id = str(order.get("id") or fallback_order_id or "").strip()
    if not order_id or not order_id.isdigit():
        raise MarketMappingError("order ID must be numeric and non-empty")
    status = str(order.get("status") or "").strip()
    if not status:
        raise MarketMappingError("order status is required")

    side = str(order.get("type") or "").strip().lower()
    if side not in {"buy", "sell"}:
        raise MarketMappingError("order side must be buy or sell")
    pair = _order_pair(order)
    amount = _decimal(order.get("amount"), "order.amount", allow_zero=False)
    matched_amount = _decimal(
        order.get("matchedAmount", "0"),
        "order.matchedAmount",
        allow_zero=True,
    )
    if matched_amount > amount:
        raise MarketMappingError("matched order amount exceeds order amount")
    canonical_status = map_order_status(
        status, matched_amount=matched_amount, amount=amount
    )

    raw_price = order.get("price")
    price: Decimal | None
    if raw_price is None or str(raw_price).strip().lower() in {"", "market"}:
        price = None
    else:
        price = _decimal(raw_price, "order.price", allow_zero=False)

    raw_average = order.get("averagePrice", "0")
    average = _decimal(raw_average, "order.averagePrice", allow_zero=True)
    average_price = average if average > 0 else None
    total_price = _decimal(
        order.get("totalPrice", "0"), "order.totalPrice", allow_zero=True
    )
    fee_cost = _decimal(order.get("fee", "0"), "order.fee", allow_zero=True)

    raw_client_id = order.get("clientOrderId")
    client_order_id = (
        str(raw_client_id).strip() if raw_client_id is not None else None
    )
    if client_order_id == "":
        client_order_id = None
    if client_order_id is not None and len(client_order_id) > 32:
        raise MarketMappingError("client order ID exceeds 32 characters")

    raw_fee_currency = order.get("feeCurrency")
    fee_currency = _currency(raw_fee_currency) if raw_fee_currency else None
    execution = str(order.get("execution") or "").strip().lower()
    if execution not in {"limit", "market"}:
        raise MarketMappingError("order execution type is missing or unsupported")

    return SpotOrderSnapshot(
        exchange_order_id=order_id,
        status=canonical_status,
        side=side,
        pair=pair,
        amount=amount,
        matched_amount=matched_amount,
        price=price,
        client_order_id=client_order_id,
        execution=execution,
        average_price=average_price,
        total_price=total_price,
        fee_cost=fee_cost,
        fee_currency=fee_currency,
    )


def _decimal_places(value: Decimal) -> int:
    return max(0, -value.normalize().as_tuple().exponent)


def validate_order_against_market(
    market: SpotMarketMetadata,
    *,
    pair: str,
    side: str,
    order_type: str,
    amount: object,
    price: object | None = None,
) -> None:
    """Validate Spot/USDT order values against normalized market constraints.

    This performs no network or order-write operation; it validates a candidate.
    """
    normalized_pair = normalize_spot_pair(pair)
    if market.market_type.lower() != "spot" or market.quote.upper() != "USDT":
        raise MarketMappingError("only normalized Spot/USDT metadata is accepted")
    if normalize_spot_pair(market.normalized_symbol) != normalized_pair:
        raise MarketMappingError("order pair does not match market metadata")

    normalized_side = side.strip().lower()
    if normalized_side not in {value.lower() for value in market.supported_sides}:
        raise MarketMappingError("order side is not supported by the market")
    normalized_type = order_type.strip().lower()
    if normalized_type not in {
        value.lower() for value in market.supported_order_types
    }:
        raise MarketMappingError("order type is not supported by the market")

    normalized_amount = _decimal(amount, "order.amount", allow_zero=False)
    if market.min_amount is not None and normalized_amount < market.min_amount:
        raise MarketMappingError("order amount is below market minimum")
    if market.max_amount is not None and normalized_amount > market.max_amount:
        raise MarketMappingError("order amount exceeds market maximum")
    if (
        market.amount_precision is not None
        and _decimal_places(normalized_amount) > market.amount_precision
    ):
        raise MarketMappingError("order amount exceeds market precision")

    if normalized_type == "limit" and price is None:
        raise MarketMappingError("limit order requires a price")
    if price is not None:
        normalized_price = _decimal(price, "order.price", allow_zero=False)
        if market.min_price is not None and normalized_price < market.min_price:
            raise MarketMappingError("order price is below market minimum")
        if market.max_price is not None and normalized_price > market.max_price:
            raise MarketMappingError("order price exceeds market maximum")
        if (
            market.price_precision is not None
            and _decimal_places(normalized_price) > market.price_precision
        ):
            raise MarketMappingError("order price exceeds market precision")
