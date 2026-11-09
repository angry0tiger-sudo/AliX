"""Fail-closed Nobitex MVP: Donchian-20, ATR-14, 2-ATR trailing stop.

This module supports deterministic paper simulation and a read-only public
order-book client. Live order submission is intentionally unavailable because
the published Nobitex Spot order contract does not document exchange-enforced
post-only execution.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import ssl
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from http.client import HTTPSConnection, HTTPException
from pathlib import Path
from typing import Any, Callable, Mapping

API_HOST = "apiv2.nobitex.ir"
ORDERBOOK_PATH_PREFIX = "/v3/orderbook/"
PRICE_BAND = Decimal("0.005")
MAX_PARTICIPATION = Decimal("0.10")
MAX_BOOK_AGE_SECONDS = 5.0
PUBLIC_REQUESTS_PER_MINUTE = 60
PRIVATE_ORDER_REQUESTS_PER_10_MINUTES = 240
DONCHIAN_LOOKBACK = 20
ATR_PERIOD = 14
TRAILING_ATR_MULTIPLIER = 2.0
DEFAULT_TIMEFRAME_MINUTES = 30
LIVE_ORDER_SUBMISSION_SUPPORTED = False


class MvpError(RuntimeError):
    """Base exception for MVP execution failures."""


class LiveExecutionBlocked(MvpError):
    """Live order submission is blocked by an unresolved exchange contract."""


class OrderBookError(MvpError):
    """Nobitex order-book data is invalid, stale, or outside the accepted contract."""


class LiquidityRejected(MvpError):
    """The requested size exceeds approved depth or passive maker capacity."""


@dataclass(frozen=True, slots=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def validate(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("candle_timestamp_must_be_timezone_aware")
        values = (self.open, self.high, self.low, self.close, self.volume)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("candle_values_must_be_finite")
        if min(self.open, self.high, self.low, self.close) <= 0.0 or self.volume < 0.0:
            raise ValueError("candle_ohlcv_geometry_invalid")
        if self.high < max(self.open, self.close) or self.low > min(
            self.open, self.close
        ):
            raise ValueError("candle_ohlc_geometry_invalid")


@dataclass(frozen=True, slots=True)
class BookLevel:
    price: Decimal
    amount: Decimal

    @property
    def quote_notional(self) -> Decimal:
        return self.price * self.amount


@dataclass(frozen=True, slots=True)
class OrderBookSnapshot:
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
    side: str
    approved: bool
    mid_price: Decimal
    intended_quote_notional: Decimal
    available_quote_depth: Decimal
    band_fraction: Decimal
    reason: str


@dataclass(frozen=True, slots=True)
class MakerSlice:
    side: str
    limit_price: Decimal
    base_quantity: Decimal
    quote_notional: Decimal
    post_only_required: bool = True
    exchange_enforced_post_only_confirmed: bool = False


@dataclass(frozen=True, slots=True)
class MakerOrderPlan:
    symbol: str
    side: str
    intended_quote_notional: Decimal
    liquidity: LiquidityGateResult
    child_slices: tuple[MakerSlice, ...]
    execution_permission: bool = False


@dataclass(frozen=True, slots=True)
class StrategyDecision:
    action: str
    reference_price: float
    atr: float
    stop_price: float | None
    reason: str
    bar_high: float


@dataclass(frozen=True, slots=True)
class PaperFill:
    side: str
    base_quantity: Decimal
    quote_notional: Decimal
    average_price: Decimal
    child_order_count: int
    simulated: bool = True


@dataclass(frozen=True, slots=True)
class MvpConfig:
    dry_run: bool = True
    allow_live_trading: bool = False
    symbol: str = "BTCUSDT"
    quote_equity: Decimal = Decimal("1000")
    risk_per_trade_fraction: Decimal = Decimal("0.005")
    max_position_fraction: Decimal = Decimal("0.10")
    timeframe_minutes: int = DEFAULT_TIMEFRAME_MINUTES

    def validate(self) -> None:
        if not self.symbol or not re.fullmatch(r"[A-Z0-9]{5,20}", self.symbol):
            raise ValueError("nobitex_symbol_invalid")
        if not self.quote_equity.is_finite() or self.quote_equity <= 0:
            raise ValueError("paper_quote_equity_must_be_positive")
        if not self.risk_per_trade_fraction.is_finite() or not (
            Decimal("0") < self.risk_per_trade_fraction <= Decimal("0.02")
        ):
            raise ValueError("risk_per_trade_fraction_out_of_range")
        if not self.max_position_fraction.is_finite() or not (
            Decimal("0") < self.max_position_fraction <= Decimal("0.25")
        ):
            raise ValueError("max_position_fraction_out_of_range")
        if self.timeframe_minutes not in {15, 30, 60, 240}:
            raise ValueError("timeframe_not_allowed")
        if not self.dry_run:
            raise LiveExecutionBlocked(
                "live_mode_blocked_until_nobitex_post_only_is_officially_supported"
            )
        if self.allow_live_trading:
            raise LiveExecutionBlocked(
                "ALIX_ALLOW_LIVE_TRADING_must_remain_false_for_this_release"
            )

    @classmethod
    def from_environment(cls, values: Mapping[str, str]) -> MvpConfig:
        def read_bool(name: str, default: str) -> bool:
            raw = values.get(name, default).strip().lower()
            if raw not in {"true", "false"}:
                raise ValueError(f"{name}_must_be_true_or_false")
            return raw == "true"

        try:
            quote_equity = Decimal(values.get("MVP_PAPER_EQUITY_QUOTE", "1000"))
            risk_fraction = Decimal(values.get("MVP_RISK_PER_TRADE", "0.005"))
            position_fraction = Decimal(values.get("MVP_MAX_POSITION_FRACTION", "0.10"))
            timeframe = int(values.get("MVP_TIMEFRAME_MINUTES", "30"))
        except (InvalidOperation, ValueError):
            raise ValueError("mvp_numeric_environment_value_invalid") from None
        config = cls(
            dry_run=read_bool("ALIX_DRY_RUN", "true"),
            allow_live_trading=read_bool("ALIX_ALLOW_LIVE_TRADING", "false"),
            symbol=values.get("NOBITEX_SYMBOL", "BTCUSDT").strip().upper(),
            quote_equity=quote_equity,
            risk_per_trade_fraction=risk_fraction,
            max_position_fraction=position_fraction,
            timeframe_minutes=timeframe,
        )
        config.validate()
        return config


def load_env_file(
    path: Path, inherited: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Load simple KEY=VALUE lines; process environment values take precedence."""
    values = dict(os.environ if inherited is None else inherited)
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError("env_file_line_must_be_key_value")
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError("env_file_key_invalid")
        values.setdefault(key, value.strip().strip("\"'"))
    return values


class TokenBucket:
    """Token bucket with injectable time functions for deterministic tests."""

    def __init__(
        self,
        capacity: float,
        refill_per_second: float,
        *,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        if not math.isfinite(capacity) or capacity < 1:
            raise ValueError("token_bucket_capacity_invalid")
        if not math.isfinite(refill_per_second) or refill_per_second <= 0:
            raise ValueError("token_bucket_refill_invalid")
        self.capacity = capacity
        self.refill_per_second = refill_per_second
        self._clock = clock or time.monotonic
        self._sleep = sleeper or time.sleep
        self._tokens = capacity
        self._updated_at = self._clock()

    def acquire(self) -> None:
        while True:
            now = self._clock()
            elapsed = max(0.0, now - self._updated_at)
            self._tokens = min(
                self.capacity, self._tokens + elapsed * self.refill_per_second
            )
            self._updated_at = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            self._sleep((1.0 - self._tokens) / self.refill_per_second)


def public_rate_limiter() -> TokenBucket:
    """Conservative local cap: 60 public requests per minute."""
    return TokenBucket(
        capacity=PUBLIC_REQUESTS_PER_MINUTE,
        refill_per_second=PUBLIC_REQUESTS_PER_MINUTE / 60.0,
    )


def private_order_rate_limiter() -> TokenBucket:
    """Conservative placeholder cap below the documented shared 300/10-minute limit."""
    return TokenBucket(
        capacity=PRIVATE_ORDER_REQUESTS_PER_10_MINUTES,
        refill_per_second=PRIVATE_ORDER_REQUESTS_PER_10_MINUTES / 600.0,
    )


def _positive_decimal(value: object, field_name: str) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        raise OrderBookError(f"orderbook_{field_name}_invalid") from None
    if not number.is_finite() or number <= 0:
        raise OrderBookError(f"orderbook_{field_name}_must_be_positive")
    return number


def _parse_book_levels(value: object, side: str) -> tuple[BookLevel, ...]:
    if not isinstance(value, list) or not value:
        raise OrderBookError(f"orderbook_{side}_missing")
    levels: list[BookLevel] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise OrderBookError(f"orderbook_{side}_level_invalid")
        levels.append(
            BookLevel(
                price=_positive_decimal(item[0], f"{side}_price"),
                amount=_positive_decimal(item[1], f"{side}_amount"),
            )
        )
    prices = [level.price for level in levels]
    if side == "bids" and any(a <= b for a, b in zip(prices, prices[1:])):
        raise OrderBookError("orderbook_bids_not_strictly_descending")
    if side == "asks" and any(a >= b for a, b in zip(prices, prices[1:])):
        raise OrderBookError("orderbook_asks_not_strictly_ascending")
    return tuple(levels)


def parse_order_book(
    payload: Mapping[str, Any],
    symbol: str,
    *,
    received_at: datetime,
    max_age_seconds: float = MAX_BOOK_AGE_SECONDS,
) -> OrderBookSnapshot:
    if payload.get("status") != "ok":
        raise OrderBookError("orderbook_status_not_ok")
    raw_update = payload.get("lastUpdate")
    if isinstance(raw_update, bool) or not isinstance(raw_update, (int, str)):
        raise OrderBookError("orderbook_last_update_invalid")
    rendered_update = str(raw_update)
    if not rendered_update.isdigit() or int(rendered_update) < 1_000_000_000_000:
        raise OrderBookError("orderbook_last_update_must_be_epoch_milliseconds")
    if received_at.tzinfo is None or received_at.utcoffset() is None:
        raise OrderBookError("orderbook_received_at_timezone_required")
    if not math.isfinite(max_age_seconds) or max_age_seconds <= 0:
        raise ValueError("orderbook_max_age_out_of_range")
    last_update = datetime.fromtimestamp(int(rendered_update) / 1000, tz=timezone.utc)
    received_utc = received_at.astimezone(timezone.utc)
    age = (received_utc - last_update).total_seconds()
    if age < 0 or age > max_age_seconds:
        raise OrderBookError("orderbook_snapshot_stale_or_from_future")
    bids = _parse_book_levels(payload.get("bids"), "bids")
    asks = _parse_book_levels(payload.get("asks"), "asks")
    if bids[0].price >= asks[0].price:
        raise OrderBookError("orderbook_best_prices_crossed")
    return OrderBookSnapshot(
        symbol=symbol,
        last_update=last_update,
        received_at=received_utc,
        bids=bids,
        asks=asks,
    )


class NobitexPublicOrderBookClient:
    """GET-only public order-book client; no authentication and no order-write method."""

    def __init__(
        self,
        *,
        host: str = API_HOST,
        timeout_seconds: float = 10.0,
        rate_limiter: TokenBucket | None = None,
        dry_run: bool = True,
        owner_approved: bool = False,
    ) -> None:
        if host != API_HOST:
            raise ValueError("nobitex_api_host_not_in_allowlist")
        if not 0 < timeout_seconds <= 30:
            raise ValueError("nobitex_http_timeout_out_of_range")
        self.host = host
        self.timeout_seconds = timeout_seconds
        self.rate_limiter = rate_limiter or public_rate_limiter()
        self.dry_run = dry_run
        self.owner_approved = owner_approved

    def fetch_order_book(
        self, symbol: str, *, now: datetime | None = None
    ) -> OrderBookSnapshot:
        if self.dry_run or not self.owner_approved:
            raise LiveExecutionBlocked(
                "public_api_calls_require_dry_run_false_and_explicit_owner_approval"
            )
        normalized = symbol.upper()
        if not re.fullmatch(r"[A-Z0-9]{5,20}", normalized):
            raise ValueError("nobitex_symbol_invalid")
        self.rate_limiter.acquire()
        path = f"{ORDERBOOK_PATH_PREFIX}{normalized}"
        connection = HTTPSConnection(
            self.host,
            timeout=self.timeout_seconds,
            context=ssl.create_default_context(),
        )
        try:
            connection.request(
                "GET",
                path,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "AliX-Nobitex-MVP/1.0",
                },
            )
            response = connection.getresponse()
            body = response.read()
            if response.status != 200:
                raise OrderBookError(f"orderbook_http_status_{response.status}")
            payload = json.loads(body.decode("utf-8"))
            if not isinstance(payload, dict):
                raise OrderBookError("orderbook_payload_not_object")
        except (
            HTTPException,
            TimeoutError,
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ):
            raise OrderBookError("orderbook_public_get_failed") from None
        finally:
            connection.close()
        return parse_order_book(
            payload,
            normalized,
            received_at=now or datetime.now(timezone.utc),
        )


def _depth_within_band(
    snapshot: OrderBookSnapshot, side: str, band: Decimal
) -> Decimal:
    mid = snapshot.mid_price
    if side == "buy":
        boundary = mid * (Decimal("1") + band)
        levels = (level for level in snapshot.asks if level.price <= boundary)
    elif side == "sell":
        boundary = mid * (Decimal("1") - band)
        levels = (level for level in snapshot.bids if level.price >= boundary)
    else:
        raise ValueError("order_side_must_be_buy_or_sell")
    return sum((level.quote_notional for level in levels), Decimal("0"))


def liquidity_gate(
    snapshot: OrderBookSnapshot,
    *,
    side: str,
    intended_quote_notional: Decimal,
    band: Decimal = PRICE_BAND,
) -> LiquidityGateResult:
    if not band.is_finite() or not Decimal("0") <= band < Decimal("1"):
        raise ValueError("liquidity_band_out_of_range")
    intended = _positive_decimal(intended_quote_notional, "intended_quote_notional")
    depth = _depth_within_band(snapshot, side, band)
    approved = depth >= intended
    return LiquidityGateResult(
        side=side,
        approved=approved,
        mid_price=snapshot.mid_price,
        intended_quote_notional=intended,
        available_quote_depth=depth,
        band_fraction=band,
        reason="PASS" if approved else "INSUFFICIENT_DEPTH_WITHIN_0_5_PERCENT",
    )


def plan_maker_slices(
    snapshot: OrderBookSnapshot,
    *,
    side: str,
    intended_quote_notional: Decimal,
    band: Decimal = PRICE_BAND,
    max_participation: Decimal = MAX_PARTICIPATION,
) -> MakerOrderPlan:
    if side not in {"buy", "sell"}:
        raise ValueError("order_side_must_be_buy_or_sell")
    if not Decimal("0") < max_participation <= Decimal("0.25"):
        raise ValueError("maker_participation_out_of_range")
    gate = liquidity_gate(
        snapshot, side=side, intended_quote_notional=intended_quote_notional, band=band
    )
    if not gate.approved:
        raise LiquidityRejected(gate.reason)
    mid = snapshot.mid_price
    if side == "buy":
        passive_levels = tuple(
            level
            for level in snapshot.bids
            if level.price >= mid * (Decimal("1") - band)
        )
        best_opposite = snapshot.asks[0].price
    else:
        passive_levels = tuple(
            level
            for level in snapshot.asks
            if level.price <= mid * (Decimal("1") + band)
        )
        best_opposite = snapshot.bids[0].price
    remaining = gate.intended_quote_notional
    slices: list[MakerSlice] = []
    for level in passive_levels:
        if (side == "buy" and level.price >= best_opposite) or (
            side == "sell" and level.price <= best_opposite
        ):
            raise OrderBookError("passive_limit_would_cross_opposite_best")
        child_notional = min(remaining, level.quote_notional * max_participation)
        if child_notional <= 0:
            continue
        slices.append(
            MakerSlice(
                side=side,
                limit_price=level.price,
                base_quantity=child_notional / level.price,
                quote_notional=child_notional,
            )
        )
        remaining -= child_notional
        if remaining <= Decimal("1e-18"):
            break
    if remaining > Decimal("1e-18"):
        raise LiquidityRejected("PASSIVE_SIDE_DEPTH_BELOW_PARTICIPATION_CAP")
    if (
        sum((item.quote_notional for item in slices), Decimal("0"))
        != gate.intended_quote_notional
    ):
        raise LiquidityRejected("CHILD_SLICE_TOTAL_MISMATCH")
    return MakerOrderPlan(
        symbol=snapshot.symbol,
        side=side,
        intended_quote_notional=gate.intended_quote_notional,
        liquidity=gate,
        child_slices=tuple(slices),
        execution_permission=False,
    )


class DonchianAtrStrategy:
    """Long-only Donchian-20 close breakout with a ratcheting 2-ATR stop."""

    def __init__(
        self,
        lookback: int = DONCHIAN_LOOKBACK,
        atr_period: int = ATR_PERIOD,
        trailing_atr_multiplier: float = TRAILING_ATR_MULTIPLIER,
    ) -> None:
        if lookback != DONCHIAN_LOOKBACK or atr_period != ATR_PERIOD:
            raise ValueError("mvp_strategy_geometry_is_fixed")
        if not math.isfinite(trailing_atr_multiplier) or trailing_atr_multiplier != 2.0:
            raise ValueError("mvp_trailing_stop_must_be_two_atr")
        self.lookback = lookback
        self.atr_period = atr_period
        self.trailing_atr_multiplier = trailing_atr_multiplier
        self.history: list[Candle] = []
        self._true_ranges: list[float] = []
        self._atr: float | None = None
        self.position_open = False
        self.high_water_mark: float | None = None
        self.trailing_stop: float | None = None
        self.last_entry_atr: float | None = None

    def _update_atr(self, candle: Candle) -> float | None:
        if self.history:
            previous_close = self.history[-1].close
            true_range = max(
                candle.high - candle.low,
                abs(candle.high - previous_close),
                abs(candle.low - previous_close),
            )
        else:
            true_range = candle.high - candle.low
        self._true_ranges.append(true_range)
        if len(self._true_ranges) == self.atr_period:
            self._atr = sum(self._true_ranges) / self.atr_period
        elif len(self._true_ranges) > self.atr_period and self._atr is not None:
            self._atr = (
                self._atr * (self.atr_period - 1) + true_range
            ) / self.atr_period
        return self._atr

    def on_bar(self, candle: Candle) -> StrategyDecision | None:
        candle.validate()
        if self.history and candle.timestamp <= self.history[-1].timestamp:
            raise ValueError("strategy_candles_must_be_strictly_increasing")
        previous_bars = self.history[-self.lookback :]
        prior_high = (
            max(item.high for item in previous_bars)
            if len(previous_bars) == self.lookback
            else None
        )
        atr = self._update_atr(candle)
        decision: StrategyDecision | None = None

        if self.position_open:
            if self.trailing_stop is None or self.high_water_mark is None:
                raise RuntimeError("strategy_open_position_state_incomplete")
            if candle.low <= self.trailing_stop:
                exit_reference = min(candle.open, self.trailing_stop)
                decision = StrategyDecision(
                    action="SELL",
                    reference_price=exit_reference,
                    atr=atr or self.last_entry_atr or 0.0,
                    stop_price=self.trailing_stop,
                    reason="TRAILING_STOP_2_ATR",
                    bar_high=candle.high,
                )
            else:
                self.high_water_mark = max(self.high_water_mark, candle.high)
                if atr is not None:
                    ratcheted = (
                        self.high_water_mark - self.trailing_atr_multiplier * atr
                    )
                    self.trailing_stop = max(self.trailing_stop, ratcheted)
        elif atr is not None and prior_high is not None and candle.close > prior_high:
            decision = StrategyDecision(
                action="BUY",
                reference_price=candle.close,
                atr=atr,
                stop_price=candle.close - self.trailing_atr_multiplier * atr,
                reason="DONCHIAN_20_CLOSE_BREAKOUT",
                bar_high=candle.high,
            )

        self.history.append(candle)
        return decision

    def register_entry_fill(self, price: float, atr: float, bar_high: float) -> None:
        if self.position_open or not all(
            math.isfinite(value) and value > 0 for value in (price, atr, bar_high)
        ):
            raise ValueError("strategy_entry_fill_invalid")
        self.position_open = True
        self.high_water_mark = max(price, bar_high)
        self.trailing_stop = price - self.trailing_atr_multiplier * atr
        self.last_entry_atr = atr

    def register_exit_fill(self) -> None:
        if not self.position_open:
            raise ValueError("strategy_exit_without_position")
        self.position_open = False
        self.high_water_mark = None
        self.trailing_stop = None
        self.last_entry_atr = None


class RiskSizer:
    """Risk-budget sizing with a hard equity fraction cap; Spot and long-only."""

    def __init__(
        self,
        equity: Decimal,
        risk_fraction: Decimal,
        max_position_fraction: Decimal,
    ) -> None:
        if not equity.is_finite() or equity <= 0:
            raise ValueError("risk_sizer_equity_must_be_positive")
        if not risk_fraction.is_finite() or not (
            Decimal("0") < risk_fraction <= Decimal("0.02")
        ):
            raise ValueError("risk_sizer_risk_fraction_invalid")
        if not max_position_fraction.is_finite() or not (
            Decimal("0") < max_position_fraction <= Decimal("0.25")
        ):
            raise ValueError("risk_sizer_position_fraction_invalid")
        self.equity = equity
        self.risk_fraction = risk_fraction
        self.max_position_fraction = max_position_fraction

    def entry_quantity(self, price: float, atr: float) -> Decimal:
        if not math.isfinite(price) or not math.isfinite(atr) or min(price, atr) <= 0:
            raise ValueError("risk_sizer_market_inputs_invalid")
        risk_budget = float(self.equity * self.risk_fraction)
        stop_distance = TRAILING_ATR_MULTIPLIER * atr
        risk_quantity = risk_budget / stop_distance
        quote_cap = float(self.equity * self.max_position_fraction)
        cap_quantity = quote_cap / price
        quantity = min(risk_quantity, cap_quantity)
        return Decimal(str(quantity))

    def quote_notional(self, price: float, atr: float) -> Decimal:
        return self.entry_quantity(price, atr) * Decimal(str(price))


class SimulatedPaperExecutor:
    """Deterministic fill simulator. It never has an HTTP or exchange transport."""

    @staticmethod
    def fill(plan: MakerOrderPlan) -> PaperFill:
        quantity = sum((item.base_quantity for item in plan.child_slices), Decimal("0"))
        quote = sum((item.quote_notional for item in plan.child_slices), Decimal("0"))
        if quantity <= 0 or quote <= 0:
            raise ValueError("paper_fill_plan_empty")
        return PaperFill(
            side=plan.side,
            base_quantity=quantity,
            quote_notional=quote,
            average_price=quote / quantity,
            child_order_count=len(plan.child_slices),
        )


def synthetic_order_book(reference_price: float, symbol: str) -> OrderBookSnapshot:
    if not math.isfinite(reference_price) or reference_price <= 0:
        raise ValueError("synthetic_book_reference_price_invalid")
    mid = Decimal(str(reference_price))
    bids = tuple(
        BookLevel(
            price=mid * (Decimal("1") - Decimal(str(0.0005 * (index + 1)))),
            amount=Decimal("100"),
        )
        for index in range(10)
    )
    asks = tuple(
        BookLevel(
            price=mid * (Decimal("1") + Decimal(str(0.0005 * (index + 1)))),
            amount=Decimal("100"),
        )
        for index in range(10)
    )
    now = datetime.now(timezone.utc)
    return OrderBookSnapshot(
        symbol=symbol,
        last_update=now,
        received_at=now,
        bids=bids,
        asks=asks,
    )


def synthetic_candles(
    hours: int,
    *,
    timeframe_minutes: int = DEFAULT_TIMEFRAME_MINUTES,
    seed: int = 17,
) -> tuple[Candle, ...]:
    if not 1 <= hours <= 24 * 365:
        raise ValueError("simulation_hours_out_of_range")
    if timeframe_minutes not in {15, 30, 60, 240}:
        raise ValueError("simulation_timeframe_not_allowed")
    count = hours * 60 // timeframe_minutes
    if count < DONCHIAN_LOOKBACK + ATR_PERIOD + 2:
        raise ValueError("simulation_window_too_short_for_strategy_warmup")
    generator = random.Random(seed)
    result: list[Candle] = []
    price = 100.0
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index in range(count):
        phase = index % 32
        if phase < 20:
            drift = 0.00005 * math.sin(index)
        elif phase < 24:
            drift = 0.0035
        else:
            drift = -0.0015
        change = drift + generator.gauss(0.0, 0.0005)
        open_price = price
        close = max(1.0, price * (1.0 + change))
        high = max(open_price, close) * (1.0 + generator.uniform(0.0003, 0.0010))
        low = min(open_price, close) * (1.0 - generator.uniform(0.0003, 0.0010))
        result.append(
            Candle(
                timestamp=start + timedelta(minutes=index * timeframe_minutes),
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=generator.uniform(10.0, 100.0),
            )
        )
        price = close
    return tuple(result)


def run_paper_simulation(
    config: MvpConfig,
    *,
    hours: int = 24,
    seed: int = 17,
) -> dict[str, Any]:
    config.validate()
    strategy = DonchianAtrStrategy()
    risk_sizer = RiskSizer(
        config.quote_equity,
        config.risk_per_trade_fraction,
        config.max_position_fraction,
    )
    executor = SimulatedPaperExecutor()
    held_quantity = Decimal("0")
    events: list[dict[str, Any]] = []
    counts = {
        "bars_processed": 0,
        "buy_signals": 0,
        "sell_signals": 0,
        "simulated_buy_fills": 0,
        "simulated_sell_fills": 0,
        "simulated_child_orders": 0,
        "liquidity_rejections": 0,
    }
    for candle in synthetic_candles(
        hours, timeframe_minutes=config.timeframe_minutes, seed=seed
    ):
        counts["bars_processed"] += 1
        decision = strategy.on_bar(candle)
        if decision is None:
            continue
        event: dict[str, Any] = {
            "timestamp": candle.timestamp.isoformat(),
            "event": "SIGNAL",
            "action": decision.action,
            "reference_price": decision.reference_price,
            "atr": decision.atr,
            "stop_price": decision.stop_price,
            "reason": decision.reason,
        }
        if decision.action == "BUY":
            counts["buy_signals"] += 1
            notional = risk_sizer.quote_notional(decision.reference_price, decision.atr)
        else:
            counts["sell_signals"] += 1
            if held_quantity <= 0:
                raise RuntimeError("paper_exit_without_held_quantity")
            notional = held_quantity * Decimal(str(decision.reference_price))
        event["intended_quote_notional"] = str(notional)
        book = synthetic_order_book(decision.reference_price, config.symbol)
        try:
            plan = plan_maker_slices(
                book,
                side=decision.action.lower(),
                intended_quote_notional=notional,
            )
        except LiquidityRejected as error:
            counts["liquidity_rejections"] += 1
            event["status"] = "LIQUIDITY_REJECTED"
            event["rejection_reason"] = str(error)
            events.append(event)
            continue
        event["liquidity_gate"] = {
            "approved": plan.liquidity.approved,
            "available_quote_depth": str(plan.liquidity.available_quote_depth),
            "band_fraction": str(plan.liquidity.band_fraction),
        }
        event["child_orders"] = [
            {
                "side": child.side,
                "limit_price": str(child.limit_price),
                "base_quantity": str(child.base_quantity),
                "quote_notional": str(child.quote_notional),
                "post_only_required": child.post_only_required,
                "exchange_enforced_post_only_confirmed": (
                    child.exchange_enforced_post_only_confirmed
                ),
            }
            for child in plan.child_slices
        ]
        fill = executor.fill(plan)
        counts["simulated_child_orders"] += fill.child_order_count
        if fill.side == "buy":
            held_quantity += fill.base_quantity
            strategy.register_entry_fill(
                float(fill.average_price), decision.atr, decision.bar_high
            )
            counts["simulated_buy_fills"] += 1
        else:
            held_quantity = Decimal("0")
            strategy.register_exit_fill()
            counts["simulated_sell_fills"] += 1
        event["status"] = "SIMULATED_FILL"
        event["simulated_fill"] = {
            "base_quantity": str(fill.base_quantity),
            "quote_notional": str(fill.quote_notional),
            "average_price": str(fill.average_price),
            "child_order_count": fill.child_order_count,
            "simulated": fill.simulated,
        }
        events.append(event)
    return {
        "schema_version": "alix.nobitex-mvp.paper-report.v1",
        "mode": "PAPER_SIMULATION",
        "simulated_hours": hours,
        "timeframe_minutes": config.timeframe_minutes,
        "symbol": config.symbol,
        "strategy": {
            "entry": "CLOSE_GREATER_THAN_PRIOR_20_COMPLETED_BAR_HIGHS",
            "donchian_lookback": DONCHIAN_LOOKBACK,
            "atr_period": ATR_PERIOD,
            "trailing_stop": "HIGHEST_HIGH_MINUS_2_ATR",
            "long_only": True,
            "leverage": False,
        },
        "execution": {
            "order_type": "LIMIT_MAKER_INTENT",
            "market_orders_allowed": False,
            "price_band_fraction": float(PRICE_BAND),
            "max_level_participation": float(MAX_PARTICIPATION),
            "exchange_enforced_post_only_confirmed": False,
            "actual_exchange_orders": 0,
            "actual_exchange_fills": 0,
            "api_calls": 0,
            "fee_model": "NOT_MODELED; CURRENT_NOBITEX_FEES_REQUIRED_FOR_ANY_FUTURE_LIVE_RUN",
        },
        "events": events,
        "safety": {
            "dry_run": True,
            "allow_live_trading": False,
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "private_api_calls": 0,
            "real_order_attempts": 0,
            "nobitex_api_calls": 0,
        },
        "counts": counts,
        "profitability_claim": False,
    }


def submit_live_maker_order(_: MakerOrderPlan) -> None:
    """No live order path exists until Nobitex documents exchange-enforced post-only."""
    raise LiveExecutionBlocked(
        "Nobitex published Spot order schema documents execution=limit but no "
        "postOnly flag; a non-crossing price check cannot guarantee maker-only "
        "execution during order-book races. No order request was sent."
    )
