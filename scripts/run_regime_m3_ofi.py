from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from bisect import bisect_right
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

CONFIG_PATH = Path("config/regime_m3/ofi_preregistration.json")
CONFIG_SHA256 = "c665dc6e3c4320b2315a308d65f198c22b4d47a41402af7d5e1d689b22b856c4"
FEATURE_NAMES = (
    "best_level_ofi_5m",
    "mlofi_level_2_5m",
    "mlofi_level_3_5m",
    "mlofi_level_4_5m",
    "mlofi_level_5_5m",
    "microprice_deviation_bps",
    "signed_trade_volume_ratio_5m",
    "order_book_imbalance_top5",
)
ARTIFACTS = {
    1: ("45553679888", "15016992581", "aafe15ed9da1d6f6e8badf905627665695acfd605e926b242856f43888980ce8"),
    2: ("45620061415", "15054886094", "c25c202e5d556912bc40e6a9fdcf78adb5a23db32f3f760ad3a1024d79cf8961"),
    3: ("45683066882", "15077454919", "c7b7ec592cda5fec117976a422d2257a908c2b260d89b4b9b89ef87ab4f20a62"),
    4: ("45735110603", "15095790023", "ec4befa0ad6fc96cf1874cec4ba15b2a2dc5ca27e58adf82c33eb62f18393332"),
    5: ("45775145557", "15121226078", "c52cb93bfc345e1ba66104f7f0e4f8ea3664687e87d3dd89e44ea9236279db88"),
}
HORIZONS = (1, 4)
COST = 0.009
EPSILON = 1e-12
UTC = timezone.utc
OfiVector = tuple[float, float, float, float, float]


@dataclass(frozen=True)
class Gap:
    start: datetime
    end: datetime
    reason: str


@dataclass(frozen=True)
class Book:
    time: datetime
    source: str
    session: int
    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]

    @property
    def mid(self) -> float:
        return (self.bids[0][0] + self.asks[0][0]) / 2.0


@dataclass(frozen=True)
class BookEvent:
    time: datetime
    ofi: OfiVector


@dataclass(frozen=True)
class TradeEvent:
    time: datetime
    side: str
    volume: float


@dataclass(frozen=True)
class EventSeries:
    times: list[datetime]
    prefix: list[OfiVector]


@dataclass(frozen=True)
class TradeSeries:
    times: list[datetime]
    buy: list[float]
    sell: list[float]
    total: list[float]
    count: list[int]


@dataclass(frozen=True)
class Candidate:
    time: datetime
    features: tuple[float, ...]
    entry_mid: float
    exit_mid: float
    label: int


@dataclass
class Segment:
    number: int
    run_id: str
    started: datetime
    finished: datetime
    gaps: list[Gap]
    gap_ends: list[datetime]
    books: list[Book]
    book_times: list[datetime]
    book_events: dict[int, EventSeries]
    trade_events: dict[int, TradeSeries]
    exclusions: Counter[str]
    trade_type_counts: Counter[str]
    rest_seed_count: int

    def session_at(self, timestamp: datetime) -> int:
        return bisect_right(self.gap_ends, timestamp)

    def intersects_gap(self, start: datetime, end: datetime) -> bool:
        return any(item.start <= end and item.end >= start for item in self.gaps)

    def state_at(self, timestamp: datetime) -> Book | None:
        index = bisect_right(self.book_times, timestamp) - 1
        if index < 0:
            return None
        state = self.books[index]
        return state if state.session == self.session_at(timestamp) else None


@dataclass(frozen=True)
class Trade:
    entry_time: datetime
    exit_time: datetime
    notional: float
    entry_mid: float
    gross_return: float
    net_return: float
    pnl: float


@dataclass
class Portfolio:
    trades: list[Trade]
    nav: list[float]
    pnl: float
    net_pf: float | None
    max_dd: float
    turnover: float
    max_concurrent: int


@dataclass
class FoldOutcome:
    report: dict[str, Any]
    model: Portfolio
    always_long: Portfolio
    majority: Portfolio
    eligible: bool
    stable: bool


def _parse_time(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field}_INVALID_UTC") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field}_TIMEZONE_REQUIRED")
    return parsed.astimezone(UTC)


def _number(value: str, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}_NOT_NUMERIC") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field}_NOT_FINITE")
    return result


def _read_csv(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CAPTURE_FILE_MISSING:{path.name}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != fields:
            raise ValueError(f"CAPTURE_HEADER_MISMATCH:{path.name}")
        return [dict(row) for row in reader]


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _verify_manifest_safety(manifest: dict[str, Any]) -> None:
    expected = {
        "authorization_header_sent": False,
        "websocket_connection_token_sent": False,
        "private_api_calls": 0,
        "holdout_bytes_read": False,
        "posthack_data_read": False,
        "real_order_attempts": 0,
        "paper_orders": 0,
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"CAPTURE_SAFETY_INVARIANT_FAILED:{key}")


def _verify_capture_files(root: Path, manifest: dict[str, Any]) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("CAPTURE_MANIFEST_FILES_INVALID")
    for name in ("orderbook.csv", "trades.csv", "gaps.csv"):
        metadata = files.get(name)
        path = root / name
        if not isinstance(metadata, dict) or not path.is_file():
            raise ValueError(f"CAPTURE_METADATA_MISSING:{name}")
        payload = path.read_bytes()
        if _digest(payload) != metadata.get("sha256"):
            raise ValueError(f"CAPTURE_FILE_SHA256_MISMATCH:{name}")
        if len(payload) != metadata.get("bytes"):
            raise ValueError(f"CAPTURE_FILE_SIZE_MISMATCH:{name}")


def _verify_capture(root: Path, expected_run: str) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError("CAPTURE_MANIFEST_MISSING")
    payload = path.read_bytes()
    manifest = json.loads(payload)
    if not isinstance(manifest, dict):
        raise ValueError("CAPTURE_MANIFEST_INVALID")
    checksum_path = root / "manifest.sha256"
    checksums = checksum_path.read_text(encoding="utf-8").split() if checksum_path.is_file() else []
    if not checksums or checksums[0] != _digest(payload):
        raise ValueError("CAPTURE_MANIFEST_SHA256_MISMATCH")
    if str(manifest.get("workflow_run_id")) != expected_run:
        raise ValueError("CAPTURE_WORKFLOW_RUN_MISMATCH")
    if manifest.get("market") != "BTCIRT":
        raise ValueError("CAPTURE_MARKET_MISMATCH")
    _verify_manifest_safety(manifest)
    _verify_capture_files(root, manifest)
    return manifest


def _parse_levels(raw: str, side: str) -> tuple[tuple[float, float], ...]:
    values = json.loads(raw)
    if not isinstance(values, list) or len(values) < 5:
        raise ValueError(f"{side.upper()}_DEPTH_BELOW_FIVE")
    levels: list[tuple[float, float]] = []
    for item in values[:5]:
        if not isinstance(item, list) or len(item) < 2:
            raise ValueError(f"{side.upper()}_LEVEL_INVALID")
        price = _number(str(item[0]), f"{side}_price")
        quantity = _number(str(item[1]), f"{side}_quantity")
        if price <= 0.0 or quantity < 0.0:
            raise ValueError(f"{side.upper()}_PRICE_OR_QUANTITY_INVALID")
        levels.append((price, quantity))
    return tuple(levels)


def _book_from_row(row: dict[str, str]) -> Book:
    source = row["source"]
    allowed = {"websocket", "rest_initial_snapshot", "rest_reconnect_snapshot"}
    if source not in allowed:
        raise ValueError("ORDERBOOK_SOURCE_INVALID")
    bids = _parse_levels(row["bids_top_json"], "bid")
    asks = _parse_levels(row["asks_top_json"], "ask")
    if bids[0][0] >= asks[0][0]:
        raise ValueError("CROSSED_OR_LOCKED_BOOK")
    for index in range(1, 5):
        if bids[index][0] >= bids[index - 1][0]:
            raise ValueError("BID_RANK_ORDER_INVALID")
        if asks[index][0] <= asks[index - 1][0]:
            raise ValueError("ASK_RANK_ORDER_INVALID")
    return Book(_parse_time(row["captured_at_utc"], "book_time"), source, 0, bids, asks)


def _load_gaps(rows: list[dict[str, str]]) -> list[Gap]:
    gaps: list[Gap] = []
    for row in rows:
        if not row["gap_start_utc"] or not row["gap_end_utc"]:
            raise ValueError("GAP_BOUNDARY_MISSING")
        start = _parse_time(row["gap_start_utc"], "gap_start")
        end = _parse_time(row["gap_end_utc"], "gap_end")
        if end < start:
            raise ValueError("GAP_INTERVAL_REVERSED")
        gaps.append(Gap(start, end, row["reason"]))
    return sorted(gaps, key=lambda item: (item.start, item.end))


def _event_ofi(previous: Book, current: Book) -> OfiVector:
    values: list[float] = []
    for index in range(5):
        old_bid_price, old_bid_qty = previous.bids[index]
        bid_price, bid_qty = current.bids[index]
        old_ask_price, old_ask_qty = previous.asks[index]
        ask_price, ask_qty = current.asks[index]
        bid_event = (
            bid_qty
            if bid_price > old_bid_price
            else bid_qty - old_bid_qty
            if bid_price == old_bid_price
            else -old_bid_qty
        )
        ask_event = (
            -ask_qty
            if ask_price > old_ask_price
            else ask_qty - old_ask_qty
            if ask_price == old_ask_price
            else old_ask_qty
        )
        depth = max(EPSILON, (old_bid_qty + old_ask_qty + bid_qty + ask_qty) / 2.0)
        values.append((bid_event - ask_event) / depth)
    return cast(OfiVector, tuple(values))


def _make_book_series(events: list[BookEvent]) -> EventSeries:
    ordered = sorted(events, key=lambda item: item.time)
    times: list[datetime] = []
    prefix: list[OfiVector] = [(0.0, 0.0, 0.0, 0.0, 0.0)]
    for item in ordered:
        times.append(item.time)
        previous = prefix[-1]
        values = tuple(previous[i] + item.ofi[i] for i in range(5))
        prefix.append(cast(OfiVector, values))
    return EventSeries(times, prefix)


def _make_trade_series(events: list[TradeEvent]) -> TradeSeries:
    ordered = sorted(events, key=lambda item: item.time)
    times: list[datetime] = []
    buy = [0.0]
    sell = [0.0]
    total = [0.0]
    count = [0]
    for item in ordered:
        times.append(item.time)
        buy.append(buy[-1] + (item.volume if item.side == "buy" else 0.0))
        sell.append(sell[-1] + (item.volume if item.side == "sell" else 0.0))
        total.append(total[-1] + item.volume)
        count.append(count[-1] + 1)
    return TradeSeries(times, buy, sell, total, count)


def _read_capture_tables(root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    orderbook = _read_csv(
        root / "orderbook.csv",
        (
            "captured_at_utc", "source", "exchange_timestamp_ms", "offset",
            "best_bid_price", "best_bid_size", "best_ask_price", "best_ask_size",
            "spread", "mid_price", "bids_top_json", "asks_top_json",
        ),
    )
    trades = _read_csv(
        root / "trades.csv",
        ("captured_at_utc", "source", "exchange_timestamp_ms", "side", "price", "volume", "offset"),
    )
    gaps = _read_csv(
        root / "gaps.csv",
        (
            "stream", "gap_start_utc", "gap_end_utc", "gap_start_ms", "gap_end_ms",
            "reason", "previous_offset", "next_offset", "replay_attempted", "replay_available",
        ),
    )
    return orderbook, trades, gaps


def _load_book_states(
    rows: list[dict[str, str]], gaps: list[Gap]
) -> tuple[list[Book], list[Gap], dict[int, EventSeries], Counter[str]]:
    exclusions: Counter[str] = Counter()
    raw_books: list[Book] = []
    invalid_gaps: list[Gap] = []
    for row in rows:
        try:
            raw_books.append(_book_from_row(row))
        except (ValueError, json.JSONDecodeError) as exc:
            timestamp = _parse_time(row["captured_at_utc"], "invalid_book_time")
            exclusions[str(exc)] += 1
            invalid_gaps.append(Gap(timestamp, timestamp, f"invalid_book:{exc}"))
    all_gaps = sorted(gaps + invalid_gaps, key=lambda item: (item.start, item.end))
    gap_ends = sorted({item.end for item in all_gaps})
    raw_books.sort(
        key=lambda item: (
            item.time,
            item.source != "rest_initial_snapshot",
            item.source != "rest_reconnect_snapshot",
        )
    )
    previous: dict[int, Book] = {}
    books: list[Book] = []
    event_rows: dict[int, list[BookEvent]] = {}
    for raw in raw_books:
        session = bisect_right(gap_ends, raw.time)
        current = Book(raw.time, raw.source, session, raw.bids, raw.asks)
        prior = previous.get(session)
        if current.source == "websocket" and prior is not None:
            event = BookEvent(current.time, _event_ofi(prior, current))
            event_rows.setdefault(session, []).append(event)
        previous[session] = current
        books.append(current)
    books.sort(key=lambda item: item.time)
    series = {key: _make_book_series(value) for key, value in event_rows.items()}
    return books, all_gaps, series, exclusions


def _load_trade_events(
    rows: list[dict[str, str]],
    gaps: list[Gap],
    gap_ends: list[datetime],
    exclusions: Counter[str],
) -> tuple[dict[int, TradeSeries], Counter[str], int]:
    event_rows: dict[int, list[TradeEvent]] = {}
    type_counts: Counter[str] = Counter()
    rest_seed_count = 0
    for row in rows:
        if row["source"] == "rest_recent_seed":
            rest_seed_count += 1
            continue
        if row["source"] != "websocket":
            raise ValueError("TRADE_SOURCE_INVALID")
        side = row["side"]
        if side not in {"buy", "sell"}:
            raise ValueError(f"UNKNOWN_PUBLIC_TRADE_TYPE:{side}")
        timestamp = _parse_time(row["captured_at_utc"], "trade_time")
        volume = _number(row["volume"], "trade_volume")
        if volume < 0.0:
            raise ValueError("TRADE_VOLUME_NEGATIVE")
        if any(gap.start <= timestamp <= gap.end for gap in gaps):
            exclusions["trade_event_inside_gap"] += 1
            continue
        session = bisect_right(gap_ends, timestamp)
        type_counts[side] += 1
        event_rows.setdefault(session, []).append(TradeEvent(timestamp, side, volume))
    series = {key: _make_trade_series(value) for key, value in event_rows.items()}
    return series, type_counts, rest_seed_count


def _load_segment(number: int, root: Path, run_id: str) -> Segment:
    manifest = _verify_capture(root, run_id)
    started = _parse_time(str(manifest["started_at_utc"]), "segment_start")
    finished = _parse_time(str(manifest["finished_at_utc"]), "segment_end")
    if finished <= started:
        raise ValueError("SEGMENT_TIME_RANGE_INVALID")
    orderbook_rows, trade_rows, gap_rows = _read_capture_tables(root)
    gaps = _load_gaps(gap_rows)
    books, all_gaps, book_events, exclusions = _load_book_states(orderbook_rows, gaps)
    gap_ends = sorted({item.end for item in all_gaps})
    trade_events, type_counts, rest_seed_count = _load_trade_events(
        trade_rows, all_gaps, gap_ends, exclusions
    )
    if not type_counts:
        raise ValueError("SIGNED_TRADE_TYPE_PREFLIGHT_NO_WEBSOCKET_EVENTS")
    return Segment(
        number, run_id, started, finished, all_gaps, gap_ends, books,
        [item.time for item in books], book_events, trade_events,
        exclusions, type_counts, rest_seed_count,
    )


def _minute_grid(start: datetime, end: datetime) -> list[datetime]:
    first = start.replace(second=0, microsecond=0)
    if first < start:
        first += timedelta(minutes=1)
    last = end.replace(second=0, microsecond=0)
    if last < first:
        return []
    count = int((last - first).total_seconds() // 60) + 1
    return [first + timedelta(minutes=index) for index in range(count)]


def _window_ofi(
    series: EventSeries | None, start: datetime, end: datetime
) -> OfiVector:
    if series is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    left = bisect_right(series.times, start)
    right = bisect_right(series.times, end)
    values = tuple(
        series.prefix[right][index] - series.prefix[left][index]
        for index in range(5)
    )
    return cast(OfiVector, values)


def _signed_volume_ratio(
    series: TradeSeries | None, start: datetime, end: datetime
) -> float:
    if series is None:
        return 0.0
    left = bisect_right(series.times, start)
    right = bisect_right(series.times, end)
    volume = series.total[right] - series.total[left]
    count = series.count[right] - series.count[left]
    if count == 0:
        return 0.0
    if volume <= EPSILON:
        raise ValueError("TRADE_WINDOW_NONEMPTY_ZERO_VOLUME")
    signed = (
        series.buy[right] - series.buy[left]
        - series.sell[right] + series.sell[left]
    )
    return signed / volume


def _book_features(book: Book) -> tuple[float, float]:
    bid_price, bid_qty = book.bids[0]
    ask_price, ask_qty = book.asks[0]
    midpoint = book.mid
    top_qty = bid_qty + ask_qty
    bid_depth = sum(item[1] for item in book.bids)
    ask_depth = sum(item[1] for item in book.asks)
    if midpoint <= 0.0 or top_qty <= EPSILON or bid_depth + ask_depth <= EPSILON:
        raise ValueError("BOOK_FEATURE_DENOMINATOR_INVALID")
    microprice = (ask_price * bid_qty + bid_price * ask_qty) / top_qty
    deviation = 10_000.0 * (microprice - midpoint) / midpoint
    imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
    return deviation, imbalance


def _features_at(segment: Segment, timestamp: datetime) -> tuple[float, ...]:
    book = segment.state_at(timestamp)
    if book is None:
        raise ValueError("BOOK_STATE_MISSING_OR_CROSSES_SESSION")
    start = timestamp - timedelta(minutes=5)
    ofi = _window_ofi(segment.book_events.get(book.session), start, timestamp)
    signed_volume = _signed_volume_ratio(
        segment.trade_events.get(book.session), start, timestamp
    )
    microprice, imbalance = _book_features(book)
    values = ofi + (microprice, signed_volume, imbalance)
    if len(values) != len(FEATURE_NAMES):
        raise ValueError("FEATURE_VECTOR_LENGTH_MISMATCH")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("FEATURE_VECTOR_NON_FINITE")
    return values


def _build_candidates(
    segment: Segment, horizon: int
) -> tuple[list[Candidate], Counter[str]]:
    rows: list[Candidate] = []
    exclusions: Counter[str] = Counter()
    lookback = timedelta(minutes=5)
    forward = timedelta(hours=horizon)
    for timestamp in _minute_grid(segment.started, segment.finished):
        feature_start = timestamp - lookback
        target_time = timestamp + forward
        if feature_start < segment.started:
            exclusions["insufficient_initial_lookback"] += 1
            continue
        if segment.intersects_gap(feature_start, timestamp):
            exclusions["feature_window_intersects_gap"] += 1
            continue
        if segment.intersects_gap(timestamp, target_time):
            exclusions["target_window_intersects_gap"] += 1
            continue
        current = segment.state_at(timestamp)
        future = segment.state_at(target_time)
        if current is None:
            exclusions["book_state_missing_or_new_session"] += 1
            continue
        if future is None or future.session != current.session:
            exclusions["target_state_missing_or_new_session"] += 1
            continue
        try:
            features = _features_at(segment, timestamp)
        except ValueError as exc:
            exclusions[str(exc)] += 1
            continue
        label = int(math.log(future.mid / current.mid) > 0.0)
        rows.append(Candidate(timestamp, features, current.mid, future.mid, label))
    return rows, exclusions


def _make_pipeline(config: dict[str, Any]) -> Any:
    params = config["model"]["parameters"]
    return Pipeline(
        steps=[
            ("standard_scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    penalty=str(params["penalty"]),
                    C=float(params["C"]),
                    solver=str(params["solver"]),
                    fit_intercept=bool(params["fit_intercept"]),
                    class_weight=params["class_weight"],
                    max_iter=int(params["max_iter"]),
                    tol=float(params["tol"]),
                    random_state=int(params["random_state"]),
                ),
            ),
        ]
    )


def _net_profit_factor(trades: Sequence[Trade]) -> float | None:
    positive = sum(item.pnl for item in trades if item.pnl > 0.0)
    negative = abs(sum(item.pnl for item in trades if item.pnl < 0.0))
    return positive / negative if negative > EPSILON else None


def _max_drawdown(nav: Sequence[float]) -> float:
    peak = 1.0
    maximum = 0.0
    for value in nav:
        peak = max(peak, value)
        if peak > EPSILON:
            maximum = max(maximum, (peak - value) / peak)
    return maximum


def _make_trades(
    candidates: Sequence[Candidate], signals: Sequence[bool], horizon: int
) -> list[Trade]:
    if len(candidates) != len(signals):
        raise ValueError("PORTFOLIO_SIGNAL_LENGTH_MISMATCH")
    holding = horizon * 60
    notional = 1.0 / float(holding)
    return [
        Trade(
            row.time,
            row.time + timedelta(minutes=holding),
            notional,
            row.entry_mid,
            row.exit_mid / row.entry_mid - 1.0,
            row.exit_mid / row.entry_mid - 1.0 - COST,
            notional * (row.exit_mid / row.entry_mid - 1.0 - COST),
        )
        for row, active in zip(candidates, signals)
        if active
    ]


def _portfolio_nav(
    segment: Segment, trades: Sequence[Trade], horizon: int
) -> tuple[list[float], int]:
    nav = [1.0]
    realized = 0.0
    active: list[Trade] = []
    cursor = 0
    maximum = 0
    holding = horizon * 60
    for timestamp in _minute_grid(segment.started, segment.finished):
        if segment.intersects_gap(timestamp, timestamp):
            continue
        mark = segment.state_at(timestamp)
        if mark is None:
            continue
        while cursor < len(trades) and trades[cursor].entry_time <= timestamp:
            active.append(trades[cursor])
            cursor += 1
        still_open: list[Trade] = []
        for trade in active:
            if trade.exit_time <= timestamp:
                realized += trade.pnl
            else:
                still_open.append(trade)
        active = still_open
        maximum = max(maximum, len(active))
        unrealized = sum(
            item.notional * (mark.mid / item.entry_mid - 1.0)
            - item.notional * COST
            for item in active
        )
        nav.append(1.0 + realized + unrealized)
    if trades and (active or cursor != len(trades)):
        raise ValueError("TRADE_NOT_CLOSED_INSIDE_TEST_SEGMENT")
    if maximum > holding:
        raise ValueError("GROSS_EXPOSURE_CAP_BREACHED")
    return nav, maximum


def _portfolio(
    segment: Segment,
    candidates: Sequence[Candidate],
    signals: Sequence[bool],
    horizon: int,
) -> Portfolio:
    trades = _make_trades(candidates, signals, horizon)
    nav, maximum = _portfolio_nav(segment, trades, horizon)
    return Portfolio(
        trades=trades,
        nav=nav,
        pnl=sum(item.pnl for item in trades),
        net_pf=_net_profit_factor(trades),
        max_dd=_max_drawdown(nav),
        turnover=sum(item.notional for item in trades),
        max_concurrent=maximum,
    )


def _portfolio_metrics(result: Portfolio) -> dict[str, Any]:
    return {
        "completed_oos_trades": len(result.trades),
        "total_net_pnl_fraction_of_initial_equity": result.pnl,
        "net_profit_factor": result.net_pf,
        "max_drawdown": result.max_dd,
        "turnover_fraction_of_initial_equity": result.turnover,
        "maximum_concurrent_cohorts": result.max_concurrent,
        "trade_records": [asdict(item) for item in result.trades],
        "minute_nav_points": len(result.nav),
    }


def _aggregate(folds: Sequence[Portfolio]) -> dict[str, Any]:
    trades = [trade for fold in folds for trade in fold.trades]
    nav: list[float] = []
    carry = 0.0
    for fold in folds:
        nav.extend(value + carry for value in fold.nav)
        carry += fold.pnl
    return {
        "completed_oos_trades": len(trades),
        "total_net_pnl_fraction_of_initial_equity": sum(item.pnl for item in trades),
        "net_profit_factor": _net_profit_factor(trades),
        "max_drawdown": _max_drawdown(nav),
        "turnover_fraction_of_initial_equity": sum(item.notional for item in trades),
        "maximum_concurrent_cohorts": max(
            (item.max_concurrent for item in folds), default=0
        ),
    }


def _stable_fold(result: Portfolio) -> bool:
    return (
        len(result.trades) >= 20
        and result.pnl > 0.0
        and result.net_pf is not None
        and result.net_pf > 1.0
    )


def _verdict(
    metrics: dict[str, Any], stable: int, eligible: int
) -> tuple[str, dict[str, bool]]:
    trades = int(metrics["completed_oos_trades"])
    pf = metrics["net_profit_factor"]
    dd = float(metrics["max_drawdown"])
    gates = {
        "net_profit_factor_gte_1_10": pf is not None and float(pf) >= 1.10,
        "completed_oos_trades_gte_100": trades >= 100,
        "max_drawdown_lte_15_percent": dd <= 0.15,
        "at_least_3_stable_folds": stable >= 3,
    }
    if trades < 100 or eligible < 3:
        return "INSUFFICIENT_SAMPLE / NO-GO", gates
    return ("PASS" if all(gates.values()) else "NO-GO"), gates


def _predictive_metrics(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    if not labels:
        return {
            "accuracy": None,
            "balanced_accuracy": None,
            "confusion_matrix_0_1": [[0, 0], [0, 0]],
        }
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "confusion_matrix_0_1": matrix.astype(int).tolist(),
    }


def _training_rows(
    fold: dict[str, Any], rows: dict[int, list[Candidate]]
) -> tuple[list[Candidate], list[Candidate]]:
    train = [
        row
        for number in fold["train_segments"]
        for row in rows[int(number)]
    ]
    return train, rows[int(fold["test_segment"])]


def _class_counts(train: Sequence[Candidate]) -> Counter[int]:
    return Counter(row.label for row in train)


def _training_sufficient(
    train: Sequence[Candidate], counts: Counter[int], rules: dict[str, Any]
) -> bool:
    minimum_rows = int(rules["minimum_training_rows_per_horizon"])
    minimum_class = int(rules["minimum_training_rows_per_class"])
    return (
        len(train) >= minimum_rows
        and counts.get(0, 0) >= minimum_class
        and counts.get(1, 0) >= minimum_class
    )


def _baseline_portfolios(
    segment: Segment,
    test: list[Candidate],
    majority_long: bool,
    horizon: int,
) -> tuple[Portfolio, Portfolio]:
    always = _portfolio(segment, test, [True] * len(test), horizon)
    majority = _portfolio(segment, test, [majority_long] * len(test), horizon)
    return always, majority


def _fold_base_report(
    fold: dict[str, Any],
    train: list[Candidate],
    test: list[Candidate],
    counts: Counter[int],
    sufficient: bool,
    majority_long: bool,
    exclusions: Counter[str],
) -> dict[str, Any]:
    return {
        "fold_id": fold["fold_id"],
        "train_segments": fold["train_segments"],
        "test_segment": fold["test_segment"],
        "training_rows": len(train),
        "training_class_counts": {
            "class_0": counts.get(0, 0),
            "class_1": counts.get(1, 0),
        },
        "test_rows": len(test),
        "test_sample_exclusions": dict(exclusions),
        "training_preflight": "PASS" if sufficient else "INSUFFICIENT_SAMPLE",
        "training_majority_direction": "LONG" if majority_long else "CASH",
        "model": None,
        "predictive_metrics": None,
        "portfolio": None,
        "stable_fold": False,
    }


def _empty_fold_outcome(
    report: dict[str, Any],
    segment: Segment,
    horizon: int,
    always: Portfolio,
    majority: Portfolio,
) -> FoldOutcome:
    empty = _portfolio(segment, [], [], horizon)
    report["fold_status"] = "INSUFFICIENT_SAMPLE"
    return FoldOutcome(report, empty, always, majority, False, False)


def _fit_test_fold(
    report: dict[str, Any],
    train: list[Candidate],
    test: list[Candidate],
    segment: Segment,
    horizon: int,
    config: dict[str, Any],
) -> Portfolio:
    x_train = np.asarray([row.features for row in train], dtype=float)
    y_train = np.asarray([row.label for row in train], dtype=int)
    x_test = np.asarray([row.features for row in test], dtype=float)
    y_test = [row.label for row in test]
    model = _make_pipeline(config)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = [int(value >= 0.5) for value in probabilities]
    result = _portfolio(
        segment, test, [value >= 0.5 for value in probabilities], horizon
    )
    eligible = len(result.trades) >= 20
    stable = _stable_fold(result)
    report.update(
        {
            "fold_status": "EVALUATED" if eligible else "INSUFFICIENT_SAMPLE",
            "model": {
                "pipeline": ["StandardScaler", "LogisticRegression"],
                "parameters": config["model"]["parameters"],
                "signal_threshold": 0.5,
                "training_rows": len(train),
            },
            "predictive_metrics": _predictive_metrics(y_test, predictions),
            "portfolio": _portfolio_metrics(result),
            "stable_fold": stable,
        }
    )
    return result


def _evaluate_fold(
    fold: dict[str, Any],
    rows: dict[int, list[Candidate]],
    segments: dict[int, Segment],
    exclusions: dict[int, Counter[str]],
    config: dict[str, Any],
    horizon: int,
) -> FoldOutcome:
    train, test = _training_rows(fold, rows)
    counts = _class_counts(train)
    rules = config["walk_forward_protocol"]["training_rules"]
    sufficient = _training_sufficient(train, counts, rules)
    majority_long = counts.get(1, 0) > counts.get(0, 0)
    test_segment = int(fold["test_segment"])
    segment = segments[test_segment]
    always, majority = _baseline_portfolios(segment, test, majority_long, horizon)
    report = _fold_base_report(
        fold, train, test, counts, sufficient, majority_long,
        exclusions[test_segment],
    )
    if not sufficient or not test:
        return _empty_fold_outcome(
            report, segment, horizon, always, majority
        )
    result = _fit_test_fold(report, train, test, segment, horizon, config)
    return FoldOutcome(
        report, result, always, majority,
        len(result.trades) >= 20, _stable_fold(result),
    )


def _evaluate_horizon(
    horizon: int,
    segments: dict[int, Segment],
    rows: dict[int, list[Candidate]],
    exclusions: dict[int, Counter[str]],
    config: dict[str, Any],
) -> dict[str, Any]:
    outcomes = [
        _evaluate_fold(fold, rows, segments, exclusions, config, horizon)
        for fold in config["walk_forward_protocol"]["folds"]
    ]
    models = [item.model for item in outcomes]
    always = [item.always_long for item in outcomes]
    majorities = [item.majority for item in outcomes]
    stable = sum(item.stable for item in outcomes)
    eligible = sum(item.eligible for item in outcomes)
    pooled = _aggregate(models)
    verdict, gates = _verdict(pooled, stable, eligible)
    return {
        "horizon_hours": horizon,
        "logical_trial_id": f"NOBITEX-M3-OFI-{horizon}h-LOGISTIC-BASELINE-01",
        "folds": [item.report for item in outcomes],
        "pooled_oos": pooled,
        "fold_stability": {
            "stable_folds": stable,
            "eligible_folds_with_at_least_20_trades": eligible,
            "required_stable_folds": 3,
        },
        "acceptance_gates": gates,
        "verdict": verdict,
        "baselines": {
            "always_long_same_horizon_same_cost": _aggregate(always),
            "no_trade_cash_baseline": {
                "completed_oos_trades": 0,
                "total_net_pnl_fraction_of_initial_equity": 0.0,
                "net_profit_factor": None,
                "max_drawdown": 0.0,
                "turnover_fraction_of_initial_equity": 0.0,
            },
            "training_fold_majority_direction": _aggregate(majorities),
        },
        "sample_exclusions_by_segment": {
            str(number): dict(value) for number, value in exclusions.items()
        },
    }


def _validate_config(config: dict[str, Any]) -> None:
    preregistration = config.get("preregistration")
    if not isinstance(preregistration, dict):
        raise ValueError("LOCKED_PREREGISTRATION_SECTION_MISSING")
    if preregistration.get("id") != "ALIX-M3-OFI-G1":
        raise ValueError("LOCKED_PREREGISTRATION_ID_MISMATCH")
    if preregistration.get("status") != "PREREGISTERED_LOCKED":
        raise ValueError("LOCKED_PREREGISTRATION_STATUS_MISMATCH")
    if tuple(config["feature_specification"]["feature_names"]) != FEATURE_NAMES:
        raise ValueError("LOCKED_FEATURE_SET_MISMATCH")
    if tuple(config["targets"]["horizons"]) != HORIZONS:
        raise ValueError("LOCKED_HORIZONS_MISMATCH")
    if len(config["walk_forward_protocol"]["folds"]) != 4:
        raise ValueError("LOCKED_FOLD_COUNT_MISMATCH")
    if config["portfolio_evaluation"]["round_trip_cost_bps"] != 90:
        raise ValueError("LOCKED_COST_MISMATCH")
    if config["model"]["estimator"] != "sklearn.linear_model.LogisticRegression":
        raise ValueError("LOCKED_ESTIMATOR_MISMATCH")
    if config["model"]["signal"]["long_entry_probability_threshold"] != 0.5:
        raise ValueError("LOCKED_THRESHOLD_MISMATCH")


def _load_config() -> tuple[dict[str, Any], str]:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError("LOCKED_PREREGISTRATION_MISSING")
    payload = CONFIG_PATH.read_bytes()
    digest = _digest(payload)
    if digest != CONFIG_SHA256:
        raise ValueError("LOCKED_PREREGISTRATION_SHA256_MISMATCH")
    config = json.loads(payload)
    if not isinstance(config, dict):
        raise ValueError("LOCKED_PREREGISTRATION_ROOT_INVALID")
    _validate_config(config)
    manifest = config.get("dataset_manifest")
    if not isinstance(manifest, list) or len(manifest) != 5:
        raise ValueError("REGISTERED_DATASET_MANIFEST_INVALID")
    for item in manifest:
        number = int(item["segment"])
        if number not in ARTIFACTS:
            raise ValueError("UNREGISTERED_DATASET_SEGMENT")
        expected = ARTIFACTS[number]
        actual = (
            str(item["workflow_run_id"]),
            str(item["artifact_id"]),
            str(item["zip_sha256"]),
        )
        if actual != expected:
            raise ValueError(f"REGISTERED_ARTIFACT_MANIFEST_MISMATCH:{number}")
    return config, digest


def _load_segments(
    input_root: Path, config: dict[str, Any]
) -> dict[int, Segment]:
    result: dict[int, Segment] = {}
    for item in config["dataset_manifest"]:
        number = int(item["segment"])
        directory = input_root / f"segment_{number}"
        if not directory.is_dir():
            raise FileNotFoundError(f"REGISTERED_SEGMENT_MISSING:{number}")
        result[number] = _load_segment(number, directory, ARTIFACTS[number][0])
    return result


def _segment_summaries(segments: dict[int, Segment]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for number, segment in segments.items():
        result[str(number)] = {
            "workflow_run_id": segment.run_id,
            "capture_start_utc": segment.started,
            "capture_end_utc": segment.finished,
            "capture_seconds": (segment.finished - segment.started).total_seconds(),
            "valid_book_rows": len(segment.books),
            "websocket_trade_type_counts": dict(segment.trade_type_counts),
            "rest_recent_seed_rows_excluded": segment.rest_seed_count,
            "gap_intervals": len(segment.gaps),
            "data_quality_exclusions": dict(segment.exclusions),
        }
    return result


def _build_horizon_inputs(
    segments: dict[int, Segment], horizon: int
) -> tuple[dict[int, list[Candidate]], dict[int, Counter[str]]]:
    rows: dict[int, list[Candidate]] = {}
    exclusions: dict[int, Counter[str]] = {}
    for number, segment in segments.items():
        rows[number], exclusions[number] = _build_candidates(segment, horizon)
        exclusions[number].update(segment.exclusions)
    return rows, exclusions


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _run_horizons(
    segments: dict[int, Segment], config: dict[str, Any]
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for horizon in HORIZONS:
        rows, exclusions = _build_horizon_inputs(segments, horizon)
        results[str(horizon)] = _evaluate_horizon(
            horizon, segments, rows, exclusions, config
        )
    return results


def _trade_type_preflight(segments: dict[int, Segment]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for segment in segments.values():
        counts.update(segment.trade_type_counts)
    if set(counts) != {"buy", "sell"}:
        raise ValueError("SIGNED_TRADE_TYPE_PREFLIGHT_REQUIRES_BUY_AND_SELL")
    return counts


def _overall_verdict(horizons: dict[str, Any]) -> str:
    verdicts = [item["verdict"] for item in horizons.values()]
    if all(item == "PASS" for item in verdicts):
        return "PASS"
    if any("INSUFFICIENT_SAMPLE" in item for item in verdicts):
        return "INSUFFICIENT_SAMPLE / NO-GO"
    return "NO-GO"


def _make_report(
    config_hash: str,
    segments: dict[int, Segment],
    horizons: dict[str, Any],
    trade_types: Counter[str],
) -> dict[str, Any]:
    summaries = _segment_summaries(segments)
    return {
        "schema_version": "alix.regime-m3-ofi-evaluation.v1",
        "phase": "M3.5",
        "preregistration_id": "ALIX-M3-OFI-G1",
        "preregistration_sha256": config_hash,
        "branch": os.environ.get("GITHUB_REF_NAME"),
        "commit_sha": os.environ.get("GITHUB_SHA"),
        "dataset_segments": summaries,
        "total_capture_seconds": sum(
            float(item["capture_seconds"]) for item in summaries.values()
        ),
        "inter_segment_gaps_preserved": True,
        "feature_names": list(FEATURE_NAMES),
        "trade_type_semantics_preflight": {
            "result": "PASS_PUBLIC_API_TYPE_PROXY_ONLY",
            "documented_values": ["buy", "sell"],
            "observed_counts": dict(trade_types),
            "mapping": "buy=+volume; sell=-volume",
            "aggressor_side_interpretation_claimed": False,
        },
        "rest_recent_trade_seed_rows_excluded_total": sum(
            segment.rest_seed_count for segment in segments.values()
        ),
        "horizons": horizons,
        "overall_verdict": _overall_verdict(horizons),
        "safety_invariants": {
            "Champion": None,
            "order_permission": False,
            "trading_authorization": "NO-GO",
            "private_api_calls": 0,
            "real_order_attempts": 0,
            "paper_orders": 0,
            "holdout_bytes_read": False,
            "posthack_data_read": False,
            "external_data_requests": 0,
            "local_execution": False,
        },
        "interpretation": (
            "Midprice research simulation only. Overlapping cohorts are not "
            "independent observations. No executable-fill claim or trading authorization."
        ),
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False)
    path.write_text(content + "\n", encoding="utf-8")


def run_evaluation(input_root: Path, output: Path) -> dict[str, Any]:
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise RuntimeError("M3_5_REQUIRES_GITHUB_ACTIONS")
    if os.environ.get("ALIX_M3_OFI_OWNER_DIRECTIVE") != "2026-11-27":
        raise RuntimeError("M3_5_OWNER_DIRECTIVE_GUARD_FAILED")
    if os.environ.get("ALIX_DRY_RUN") != "true":
        raise RuntimeError("M3_5_DRY_RUN_GUARD_FAILED")
    if os.environ.get("ALIX_ALLOW_LIVE_TRADING") != "false":
        raise RuntimeError("M3_5_LIVE_TRADING_GUARD_FAILED")
    config, config_hash = _load_config()
    segments = _load_segments(input_root, config)
    trade_types = _trade_type_preflight(segments)
    horizons = _run_horizons(segments, config)
    report = _make_report(config_hash, segments, horizons, trade_types)
    _write_report(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regime M3.5 locked OFI walk-forward evaluation"
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_evaluation(args.input_root, args.output)
    summary = {
        "overall_verdict": report["overall_verdict"],
        "horizons": {
            key: {
                "verdict": item["verdict"],
                "pooled_oos": item["pooled_oos"],
            }
            for key, item in report["horizons"].items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
