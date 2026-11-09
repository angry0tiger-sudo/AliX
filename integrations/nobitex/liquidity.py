"""Mock-only Nobitex order-book liquidity gate and passive slice planner.

There is deliberately no HTTP implementation or order-write method here.
The GET path is represented by an in-memory mock transport for contract tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from integrations.nobitex.adapter import normalize_spot_pair

ORDERBOOK_PATH_PREFIX = "/v3/orderbook/"
DEFAULT_PRICE_BAND = Decimal("0.005")
DEFAULT_MAX_BOOK_AGE_SECONDS = 5.0
DEFAULT_MAX_PARTICIPATION = Decimal("0.10")


class OrderBookError(ValueError):
    """The mock order-book snapshot violates its market-data contract."""


class LiquidityGateRejected(RuntimeError):
    """The intended quote notional exceeds visible depth inside the price band."""


class InsufficientMakerDepth(RuntimeError):
    """Passive-side book depth cannot support the requested child-order plan."""


@dataclass(frozen=True, slots=True)
class BookLevel:
    price: Decimal
    amount: Decimal

    @property
    def quote_notional(self) -> Decimal:
        return self.price * self.amount


@dataclass(frozen=True, slots=True)
class OrderBookSnapshot:
    pair: str
    symbol: str
    last_update: datetime
    received_at: datetime
    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]

    @property
    def mid_price(self) -> Decimal:
        return (self.bids[0].price + self.asks[0].price) / Decimal("2")

    @property
    def age_seconds(self) -> float:
        return (self.received_at - self.last_update).total_seconds()


@dataclass(frozen=True, slots=True)
class LiquidityGateResult:
    approved: bool
    side: str
    reason: str
    mid_price: Decimal
    intended_quote_notional: Decimal
    available_quote_depth: Decimal
    price_band_fraction: Decimal


@dataclass(frozen=True, slots=True)
class MockMakerSlice:
    side: str
    limit_price: Decimal
    base_quantity: Decimal
    quote_notional: Decimal
    post_only_required: bool = True
    exchange_enforced_post_only_confirmed: bool = False
    execution_mode: str = "MOCK_ONLY"


@dataclass(frozen=True, slots=True)
class MockMakerOrderPlan:
    pair: str
    side: str
    intended_quote_notional: Decimal
    liquidity: LiquidityGateResult
    child_slices: tuple[MockMakerSlice, ...]
    execution_permission: bool = False


@dataclass(slots=True)
class MockOrderBookTransport:
    """In-memory fake for the documented GET endpoint; never performs network I/O."""

    responses: Mapping[str, Mapping[str, Any]]
    calls: list[str] = field(default_factory=list)

    def get_json(self, path: str) -> Mapping[str, Any]:
        if not path.startswith(ORDERBOOK_PATH_PREFIX):
            raise OrderBookError("mock_orderbook_path_not_allowed")
        symbol = path.removeprefix(ORDERBOOK_PATH_PREFIX)
        if not symbol or "/" in symbol or "?" in symbol:
            raise OrderBookError("mock_orderbook_symbol_invalid")
        self.calls.append(path)
        response = self.responses.get(symbol)
        if response is None:
            raise OrderBookError("mock_orderbook_fixture_missing")
        return response


def _positive_decimal(value: object, field_name: str) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        raise OrderBookError(f"orderbook_{field_name}_invalid") from None
    if not number.is_finite() or number <= 0:
        raise OrderBookError(f"orderbook_{field_name}_must_be_positive")
    return number


def _utc_now(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise OrderBookError("orderbook_now_timezone_required")
    return value.astimezone(timezone.utc)


def _last_update(value: object) -> datetime:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise OrderBookError("orderbook_last_update_invalid")
    rendered = str(value)
    if not rendered.isdigit():
        raise OrderBookError("orderbook_last_update_invalid")
    epoch_ms = int(rendered)
    if epoch_ms < 1_000_000_000_000:
        raise OrderBookError("orderbook_last_update_must_be_epoch_milliseconds")
    try:
        return datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        raise OrderBookError("orderbook_last_update_invalid") from None


def _parse_book_level(item: object, side: str) -> BookLevel:
    if not isinstance(item, (list, tuple)):
        raise OrderBookError(f"orderbook_{side}_level_schema_invalid")
    if len(item) != 2:
        raise OrderBookError(f"orderbook_{side}_level_schema_invalid")
    return BookLevel(
        price=_positive_decimal(item[0], f"{side}_price"),
        amount=_positive_decimal(item[1], f"{side}_amount"),
    )


def _validate_bid_price_order(prices: list[Decimal]) -> None:
    if any(left <= right for left, right in zip(prices, prices[1:])):
        raise OrderBookError("orderbook_bids_must_be_strictly_descending")


def _validate_ask_price_order(prices: list[Decimal]) -> None:
    if any(left >= right for left, right in zip(prices, prices[1:])):
        raise OrderBookError("orderbook_asks_must_be_strictly_ascending")


def _parse_levels(value: object, side: str) -> tuple[BookLevel, ...]:
    if side not in {"bids", "asks"}:
        raise OrderBookError("orderbook_side_invalid")
    if not isinstance(value, list) or not value:
        raise OrderBookError(f"orderbook_{side}_levels_missing")
    parsed = tuple(_parse_book_level(item, side) for item in value)
    prices = [level.price for level in parsed]
    if side == "bids":
        _validate_bid_price_order(prices)
    else:
        _validate_ask_price_order(prices)
    return parsed


def _validate_band(value: Decimal) -> Decimal:
    if not value.is_finite() or value < 0 or value >= 1:
        raise ValueError("price_band_fraction_out_of_range")
    return value


def _validate_side(side: str) -> str:
    normalized = side.strip().lower()
    if normalized not in {"buy", "sell"}:
        raise ValueError("order_side_must_be_buy_or_sell")
    return normalized


def _depth_in_band(snapshot: OrderBookSnapshot, side: str, band: Decimal) -> Decimal:
    mid = snapshot.mid_price
    if side == "buy":
        limit = mid * (Decimal("1") + band)
        levels = (level for level in snapshot.asks if level.price <= limit)
    else:
        limit = mid * (Decimal("1") - band)
        levels = (level for level in snapshot.bids if level.price >= limit)
    return sum((level.quote_notional for level in levels), Decimal("0"))


def _maker_price_would_cross(
    side: str, price: Decimal, best_opposite: Decimal
) -> bool:
    if side == "buy":
        return price >= best_opposite
    return price <= best_opposite


def _create_passive_slice(
    level: BookLevel,
    *,
    side: str,
    best_opposite: Decimal,
    remaining: Decimal,
    participation: Decimal,
) -> tuple[MockMakerSlice | None, Decimal]:
    child_notional = min(remaining, level.quote_notional * participation)
    if child_notional <= 0:
        return None, remaining
    if _maker_price_would_cross(side, level.price, best_opposite):
        reason = (
            "maker_buy_price_would_cross"
            if side == "buy"
            else "maker_sell_price_would_cross"
        )
        raise OrderBookError(reason)
    item = MockMakerSlice(
        side=side,
        limit_price=level.price,
        base_quantity=child_notional / level.price,
        quote_notional=child_notional,
    )
    return item, remaining - child_notional


def _build_passive_slices(
    levels: tuple[BookLevel, ...],
    *,
    side: str,
    best_opposite: Decimal,
    intended_quote_notional: Decimal,
    participation: Decimal,
) -> tuple[MockMakerSlice, ...]:
    remaining = intended_quote_notional
    slices: list[MockMakerSlice] = []
    for level in levels:
        item, remaining = _create_passive_slice(
            level,
            side=side,
            best_opposite=best_opposite,
            remaining=remaining,
            participation=participation,
        )
        if item is not None:
            slices.append(item)
        if remaining <= Decimal("1e-18"):
            break
    if remaining > Decimal("1e-18"):
        raise InsufficientMakerDepth("maker_side_depth_below_participation_cap")
    return tuple(slices)


class NobitexMockOrderBookAdapter:
    """Validates mock snapshots and plans passive child slices; it cannot submit orders."""

    def __init__(
        self,
        transport: MockOrderBookTransport,
        *,
        max_book_age_seconds: float = DEFAULT_MAX_BOOK_AGE_SECONDS,
    ) -> None:
        if not isinstance(transport, MockOrderBookTransport):
            raise TypeError("mock_orderbook_transport_required")
        if (
            isinstance(max_book_age_seconds, bool)
            or not isinstance(max_book_age_seconds, (int, float))
            or not 0 < float(max_book_age_seconds) <= 60
        ):
            raise ValueError("max_book_age_seconds_out_of_range")
        self._transport = transport
        self.max_book_age_seconds = float(max_book_age_seconds)

    def fetch_order_book(self, pair: str, *, now: datetime) -> OrderBookSnapshot:
        normalized = normalize_spot_pair(pair)
        symbol = normalized.replace("/", "")
        received_at = _utc_now(now)
        response = self._transport.get_json(f"{ORDERBOOK_PATH_PREFIX}{symbol}")
        if response.get("status") != "ok":
            raise OrderBookError("orderbook_status_not_ok")
        last_update = _last_update(response.get("lastUpdate"))
        age = (received_at - last_update).total_seconds()
        if age < 0 or age > self.max_book_age_seconds:
            raise OrderBookError("orderbook_snapshot_stale_or_from_future")
        bids = _parse_levels(response.get("bids"), "bids")
        asks = _parse_levels(response.get("asks"), "asks")
        if bids[0].price >= asks[0].price:
            raise OrderBookError("orderbook_best_prices_crossed")
        return OrderBookSnapshot(
            pair=normalized,
            symbol=symbol,
            last_update=last_update,
            received_at=received_at,
            bids=bids,
            asks=asks,
        )

    @staticmethod
    def liquidity_gate(
        snapshot: OrderBookSnapshot,
        *,
        side: str,
        intended_quote_notional: Decimal,
        price_band_fraction: Decimal = DEFAULT_PRICE_BAND,
    ) -> LiquidityGateResult:
        normalized_side = _validate_side(side)
        intended = _positive_decimal(intended_quote_notional, "intended_quote_notional")
        band = _validate_band(price_band_fraction)
        depth = _depth_in_band(snapshot, normalized_side, band)
        approved = depth >= intended
        return LiquidityGateResult(
            approved=approved,
            side=normalized_side,
            reason="PASS" if approved else "INSUFFICIENT_DEPTH_WITHIN_PRICE_BAND",
            mid_price=snapshot.mid_price,
            intended_quote_notional=intended,
            available_quote_depth=depth,
            price_band_fraction=band,
        )

    @staticmethod
    def _passive_levels(
        snapshot: OrderBookSnapshot, side: str, band: Decimal
    ) -> tuple[BookLevel, ...]:
        mid = snapshot.mid_price
        if side == "buy":
            minimum = mid * (Decimal("1") - band)
            return tuple(level for level in snapshot.bids if level.price >= minimum)
        maximum = mid * (Decimal("1") + band)
        return tuple(level for level in snapshot.asks if level.price <= maximum)

    @classmethod
    def plan_maker_slices(
        cls,
        snapshot: OrderBookSnapshot,
        *,
        side: str,
        intended_quote_notional: Decimal,
        price_band_fraction: Decimal = DEFAULT_PRICE_BAND,
        max_participation: Decimal = DEFAULT_MAX_PARTICIPATION,
    ) -> MockMakerOrderPlan:
        normalized_side = _validate_side(side)
        participation = _validate_band(max_participation)
        if participation <= 0:
            raise ValueError("max_participation_must_be_positive")
        gate = cls.liquidity_gate(
            snapshot,
            side=normalized_side,
            intended_quote_notional=intended_quote_notional,
            price_band_fraction=price_band_fraction,
        )
        if not gate.approved:
            raise LiquidityGateRejected(gate.reason)

        passive_levels = cls._passive_levels(
            snapshot, normalized_side, price_band_fraction
        )
        best_opposite = (
            snapshot.asks[0].price
            if normalized_side == "buy"
            else snapshot.bids[0].price
        )
        slices = _build_passive_slices(
            passive_levels,
            side=normalized_side,
            best_opposite=best_opposite,
            intended_quote_notional=gate.intended_quote_notional,
            participation=participation,
        )
        return MockMakerOrderPlan(
            pair=snapshot.pair,
            side=normalized_side,
            intended_quote_notional=gate.intended_quote_notional,
            liquidity=gate,
            child_slices=slices,
        )

        cls,
        snapshot: OrderBookSnapshot,
        *,
        side: str,
        intended_quote_notional: Decimal,
        price_band_fraction: Decimal = DEFAULT_PRICE_BAND,
        max_participation: Decimal = DEFAULT_MAX_PARTICIPATION,
    ) -> MockMakerOrderPlan:
        normalized_side = _validate_side(side)
        participation = _validate_band(max_participation)
        if participation <= 0:
            raise ValueError("max_participation_must_be_positive")
        gate = cls.liquidity_gate(
            snapshot,
            side=normalized_side,
            intended_quote_notional=intended_quote_notional,
            price_band_fraction=price_band_fraction,
        )
        if not gate.approved:
            raise LiquidityGateRejected(gate.reason)

        passive_levels = cls._passive_levels(
            snapshot, normalized_side, price_band_fraction
        )
        best_opposite = (
            snapshot.asks[0].price
            if normalized_side == "buy"
            else snapshot.bids[0].price
        )
        remaining = gate.intended_quote_notional
        slices: list[MockMakerSlice] = []
        for level in passive_levels:
            capacity = level.quote_notional * participation
            child_notional = min(remaining, capacity)
            if child_notional <= 0:
                continue
            if normalized_side == "buy" and level.price >= best_opposite:
                raise OrderBookError("maker_buy_price_would_cross")
            if normalized_side == "sell" and level.price <= best_opposite:
                raise OrderBookError("maker_sell_price_would_cross")
            slices.append(
                MockMakerSlice(
                    side=normalized_side,
                    limit_price=level.price,
                    base_quantity=child_notional / level.price,
                    quote_notional=child_notional,
                )
            )
            remaining -= child_notional
            if remaining <= Decimal("1e-18"):
                break
        if remaining > Decimal("1e-18"):
            raise InsufficientMakerDepth("maker_side_depth_below_participation_cap")
        return MockMakerOrderPlan(
            pair=snapshot.pair,
            side=normalized_side,
            intended_quote_notional=gate.intended_quote_notional,
            liquidity=gate,
            child_slices=tuple(slices),
        )
