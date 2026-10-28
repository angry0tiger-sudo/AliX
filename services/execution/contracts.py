"""Typed contracts for persistent order protection and startup reconciliation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
import hashlib
from typing import Protocol

PROTECTION_DATA_KEY = "alix_crash_recovery_protection"
PROTECTION_SCHEMA_VERSION = "alix.exchange-protection.v1"
CLIENT_ORDER_PREFIX = "alixr-"
DEFAULT_TAKE_PROFIT_FRACTION = Decimal("0.015")
DEFAULT_STOP_LOSS_FRACTION = Decimal("0.01")
ORDER_ID_TOLERANCE = Decimal("0.00000001")

OrderMapping = Mapping[str, object]
ProtectionRecord = Mapping[str, object]


@dataclass(frozen=True, slots=True)
class LocalPosition:
    """A position reconstructed from the durable local Freqtrade database."""

    trade_id: str
    pair: str
    base_asset: str
    quantity: Decimal
    entry_price: Decimal
    entry_timestamp_ms: int
    entry_order_id: str
    pending_exit: bool = False

    def __post_init__(self) -> None:
        if (
            not self.trade_id.strip()
            or not self.pair.strip()
            or not self.base_asset.strip()
        ):
            raise ValueError("position_identity_required")
        if not self.quantity.is_finite() or self.quantity <= 0:
            raise ValueError("position_quantity_must_be_positive")
        if not self.entry_price.is_finite() or self.entry_price <= 0:
            raise ValueError("position_entry_price_must_be_positive")
        if self.entry_timestamp_ms < 0:
            raise ValueError("position_timestamp_invalid")


@dataclass(frozen=True, slots=True)
class ProtectionPlan:
    """Immutable protection plan with exchange-enforced idempotency identifiers."""

    trade_id: str
    pair: str
    base_asset: str
    quantity: Decimal
    entry_price: Decimal
    stop_price: Decimal
    take_profit_price: Decimal
    entry_order_id: str
    list_client_order_id: str
    take_profit_client_order_id: str
    stop_loss_client_order_id: str

    def __post_init__(self) -> None:
        if not (self.stop_price < self.entry_price < self.take_profit_price):
            raise ValueError("protection_prices_not_ordered")
        if not self.quantity.is_finite() or self.quantity <= 0:
            raise ValueError("protection_quantity_must_be_positive")


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    """Outcome from one startup or periodic reconciliation pass."""

    halted: bool
    simulated: bool
    positions_checked: int
    positions_protected: int
    positions_closed: int
    actions: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()


class ExchangeGateway(Protocol):
    """Minimal exchange surface used by the coordinator and mock-only tests."""

    is_simulated: bool

    def fetch_balances(self) -> Mapping[str, Decimal]: ...

    def fetch_open_orders(self) -> Sequence[OrderMapping]: ...

    def fetch_my_trades(self, pair: str, since_ms: int) -> Sequence[OrderMapping]: ...

    def fetch_oco(
        self, pair: str, list_client_order_id: str
    ) -> OrderMapping | None: ...

    def place_oco(self, plan: ProtectionPlan) -> OrderMapping: ...

    def cancel_oco(self, pair: str, list_client_order_id: str) -> OrderMapping: ...


class RecoveryStore(Protocol):
    """Durable state store. Production implementation delegates to Freqtrade DB."""

    def open_positions(self) -> Sequence[LocalPosition]: ...

    def get_protection(self, trade_id: str) -> ProtectionRecord | None: ...

    def save_protection(self, trade_id: str, record: Mapping[str, object]) -> None: ...

    def mark_externally_closed(
        self,
        trade_id: str,
        order: OrderMapping,
        protection_record: Mapping[str, object],
    ) -> None: ...


def client_order_ids(position: LocalPosition) -> tuple[str, str, str]:
    """Build stable list/leg IDs from immutable trade and entry-order identity."""
    identity = "|".join(
        (position.trade_id, position.pair, position.entry_order_id)
    ).encode("utf-8")
    token = hashlib.sha256(identity).hexdigest()[:16]
    stem = f"{CLIENT_ORDER_PREFIX}{token}"
    return f"{stem}-list", f"{stem}-tp", f"{stem}-sl"


def build_protection_plan(
    position: LocalPosition,
    *,
    stop_loss_fraction: Decimal = DEFAULT_STOP_LOSS_FRACTION,
    take_profit_fraction: Decimal = DEFAULT_TAKE_PROFIT_FRACTION,
) -> ProtectionPlan:
    """Build fixed, validated protective levels from the actual filled position."""
    if (
        not stop_loss_fraction.is_finite()
        or not take_profit_fraction.is_finite()
        or stop_loss_fraction <= 0
        or stop_loss_fraction >= 1
        or take_profit_fraction <= 0
    ):
        raise ValueError("protection_fraction_invalid")
    list_id, take_profit_id, stop_loss_id = client_order_ids(position)
    stop_price = position.entry_price * (Decimal("1") - stop_loss_fraction)
    target_price = position.entry_price * (Decimal("1") + take_profit_fraction)
    return ProtectionPlan(
        trade_id=position.trade_id,
        pair=position.pair,
        base_asset=position.base_asset,
        quantity=position.quantity,
        entry_price=position.entry_price,
        stop_price=stop_price,
        take_profit_price=target_price,
        entry_order_id=position.entry_order_id,
        list_client_order_id=list_id,
        take_profit_client_order_id=take_profit_id,
        stop_loss_client_order_id=stop_loss_id,
    )


def record_for_plan(
    plan: ProtectionPlan,
    *,
    status: str,
    simulated: bool,
    error: str | None = None,
    exchange_order_list_id: str | None = None,
) -> dict[str, object]:
    """Return a JSON-serializable record for the Freqtrade custom-data column."""
    return {
        "schema_version": PROTECTION_SCHEMA_VERSION,
        "trade_id": plan.trade_id,
        "pair": plan.pair,
        "base_asset": plan.base_asset,
        "quantity": str(plan.quantity),
        "entry_price": str(plan.entry_price),
        "stop_price": str(plan.stop_price),
        "take_profit_price": str(plan.take_profit_price),
        "entry_order_id": plan.entry_order_id,
        "list_client_order_id": plan.list_client_order_id,
        "take_profit_client_order_id": plan.take_profit_client_order_id,
        "stop_loss_client_order_id": plan.stop_loss_client_order_id,
        "status": status,
        "simulated": simulated,
        "error": error,
        "exchange_order_list_id": exchange_order_list_id,
    }


def plan_from_record(
    position: LocalPosition, record: ProtectionRecord
) -> ProtectionPlan:
    """Validate persisted protection metadata against the local trade identity."""
    if record.get("schema_version") != PROTECTION_SCHEMA_VERSION:
        raise ValueError("protection_schema_mismatch")
    if (
        record.get("trade_id") != position.trade_id
        or record.get("pair") != position.pair
    ):
        raise ValueError("protection_trade_identity_mismatch")
    try:
        quantity = Decimal(str(record["quantity"]))
        entry_price = Decimal(str(record["entry_price"]))
        stop_price = Decimal(str(record["stop_price"]))
        target_price = Decimal(str(record["take_profit_price"]))
    except (KeyError, ValueError, ArithmeticError) as exc:
        raise ValueError("protection_record_numeric_field_invalid") from exc
    if not all(
        value.is_finite() for value in (quantity, entry_price, stop_price, target_price)
    ):
        raise ValueError("protection_record_non_finite")
    if quantity != position.quantity:
        raise ValueError("protection_quantity_differs_from_local_trade")
    if entry_price != position.entry_price:
        raise ValueError("protection_entry_price_differs_from_local_trade")
    expected_ids = client_order_ids(position)
    stored_ids = (
        str(record.get("list_client_order_id", "")),
        str(record.get("take_profit_client_order_id", "")),
        str(record.get("stop_loss_client_order_id", "")),
    )
    if stored_ids != expected_ids:
        raise ValueError("protection_client_order_id_mismatch")
    return ProtectionPlan(
        trade_id=position.trade_id,
        pair=position.pair,
        base_asset=position.base_asset,
        quantity=quantity,
        entry_price=entry_price,
        stop_price=stop_price,
        take_profit_price=target_price,
        entry_order_id=position.entry_order_id,
        list_client_order_id=stored_ids[0],
        take_profit_client_order_id=stored_ids[1],
        stop_loss_client_order_id=stored_ids[2],
    )
