"""Bounded, unauthenticated Nobitex L2 and public-trade capture.

This collector records only observed messages. It never interpolates, forward-fills,
or otherwise fabricates events across disconnects or publisher-offset gaps.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import ssl
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Callable, Protocol, cast
from urllib.request import Request, urlopen

REST_BASE = "https://apiv2.nobitex.ir"
WS_URL = "wss://ws.nobitex.ir/connection/websocket"
BOOK_FIELDS = (
    "captured_at_utc",
    "source",
    "exchange_timestamp_ms",
    "offset",
    "best_bid_price",
    "best_bid_size",
    "best_ask_price",
    "best_ask_size",
    "spread",
    "mid_price",
    "bids_top_json",
    "asks_top_json",
)
TRADE_FIELDS = (
    "captured_at_utc",
    "source",
    "exchange_timestamp_ms",
    "side",
    "price",
    "volume",
    "offset",
)
GAP_FIELDS = (
    "stream",
    "gap_start_utc",
    "gap_end_utc",
    "gap_start_ms",
    "gap_end_ms",
    "reason",
    "previous_offset",
    "next_offset",
    "replay_attempted",
    "replay_available",
)


class SocketLike(Protocol):
    def send(self, payload: str) -> None: ...
    def recv(self) -> str | bytes: ...
    def close(self) -> None: ...


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def decimal_value(value: object, field: str, *, allow_zero: bool = False) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field}_NOT_NUMERIC") from exc
    if not result.is_finite() or result < 0 or (result == 0 and not allow_zero):
        raise ValueError(f"{field}_OUT_OF_RANGE")
    return result


def _levels(value: object, field: str, limit: int) -> list[list[str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field}_MISSING")
    result: list[list[str]] = []
    for item in value[:limit]:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"{field}_LEVEL_INVALID")
        price = decimal_value(item[0], f"{field}_PRICE")
        size = decimal_value(item[1], f"{field}_SIZE")
        result.append([str(price), str(size)])
    return result


def make_book_row(
    payload: dict[str, object],
    captured_at: str,
    source: str,
    offset: int | None,
    levels: int = 10,
) -> dict[str, object]:
    bids = _levels(payload.get("bids"), "bids", levels)
    asks = _levels(payload.get("asks"), "asks", levels)
    timestamp = payload.get("lastUpdate")
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp <= 0:
        raise ValueError("BOOK_TIMESTAMP_INVALID")
    best_bid = decimal_value(bids[0][0], "best_bid")
    best_ask = decimal_value(asks[0][0], "best_ask")
    spread = best_ask - best_bid
    mid = (best_ask + best_bid) / Decimal(2)
    return {
        "captured_at_utc": captured_at,
        "source": source,
        "exchange_timestamp_ms": str(timestamp),
        "offset": "" if offset is None else str(offset),
        "best_bid_price": str(best_bid),
        "best_bid_size": bids[0][1],
        "best_ask_price": str(best_ask),
        "best_ask_size": asks[0][1],
        "spread": str(spread),
        "mid_price": str(mid),
        "bids_top_json": json.dumps(bids, separators=(",", ":")),
        "asks_top_json": json.dumps(asks, separators=(",", ":")),
    }


def make_trade_row(
    payload: dict[str, object],
    captured_at: str,
    source: str,
    offset: int | None,
) -> dict[str, object]:
    timestamp = payload.get("time")
    side = payload.get("type")
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp <= 0:
        raise ValueError("TRADE_TIMESTAMP_INVALID")
    if not isinstance(side, str) or side not in {"buy", "sell"}:
        raise ValueError("TRADE_SIDE_INVALID")
    price = decimal_value(payload.get("price"), "trade_price")
    volume = decimal_value(payload.get("volume"), "trade_volume", allow_zero=True)
    return {
        "captured_at_utc": captured_at,
        "source": source,
        "exchange_timestamp_ms": str(timestamp),
        "side": str(side),
        "price": str(price),
        "volume": str(volume),
        "offset": "" if offset is None else str(offset),
    }


def decode_messages(raw: str | bytes) -> list[dict[str, object]]:
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    messages: list[dict[str, object]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            messages.append(cast(dict[str, object], value))
        elif isinstance(value, list):
            messages.extend(
                cast(dict[str, object], item)
                for item in value
                if isinstance(item, dict)
            )
        else:
            raise ValueError("WEBSOCKET_MESSAGE_NOT_OBJECT")
    return messages


def gap_row(
    stream: str,
    start_utc: str,
    end_utc: str,
    reason: str,
    start_ms: int | None = None,
    end_ms: int | None = None,
    previous_offset: int | None = None,
    next_offset: int | None = None,
) -> dict[str, object]:
    return {
        "stream": stream,
        "gap_start_utc": start_utc,
        "gap_end_utc": end_utc,
        "gap_start_ms": "" if start_ms is None else str(start_ms),
        "gap_end_ms": "" if end_ms is None else str(end_ms),
        "reason": reason,
        "previous_offset": "" if previous_offset is None else str(previous_offset),
        "next_offset": "" if next_offset is None else str(next_offset),
        "replay_attempted": "false",
        "replay_available": "false",
    }


class OffsetTracker:
    """Detect publisher offset skips without attempting synthetic repair."""

    def __init__(self) -> None:
        self._last: dict[str, tuple[int, int]] = {}

    def observe(
        self, stream: str, offset: int, event_ms: int
    ) -> dict[str, object] | None:
        if offset < 0 or event_ms <= 0:
            raise ValueError("PUBLISHER_OFFSET_OR_TIMESTAMP_INVALID")
        previous = self._last.get(stream)
        if previous is None:
            self._last[stream] = (offset, event_ms)
            return None
        previous_offset, previous_ms = previous
        if offset <= previous_offset:
            return gap_row(
                stream,
                utc_now(),
                utc_now(),
                "non_monotonic_publisher_offset",
                previous_ms,
                event_ms,
                previous_offset,
                offset,
            )
        self._last[stream] = (offset, event_ms)
        if offset > previous_offset + 1:
            return gap_row(
                stream,
                utc_now(),
                utc_now(),
                "publisher_offset_gap",
                previous_ms,
                event_ms,
                previous_offset,
                offset,
            )
        return None


def fetch_public_json(path: str, timeout: float = 10.0) -> dict[str, object]:
    request = Request(
        f"{REST_BASE}{path}",
        headers={
            "Accept": "application/json",
            "User-Agent": "AliX-Regime-M-Capture/1.0",
        },
        method="GET",
    )
    if (
        "Authorization" in request.headers
        or "Authorization" in request.unredirected_hdrs
    ):
        raise RuntimeError("PRIVATE_AUTH_HEADER_FORBIDDEN")
    with urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"PUBLIC_API_HTTP_{response.status}")
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict) or value.get("status") != "ok":
        raise RuntimeError("PUBLIC_API_RESPONSE_INVALID")
    return cast(dict[str, object], value)


def _create_socket() -> tuple[SocketLike, type[BaseException]]:
    module = importlib.import_module("websocket")
    create_connection = cast(
        Callable[..., SocketLike], getattr(module, "create_connection")
    )
    timeout_type = cast(
        type[BaseException], getattr(module, "WebSocketTimeoutException")
    )
    socket = create_connection(
        WS_URL,
        timeout=5,
        enable_multithread=True,
        sslopt={"cert_reqs": ssl.CERT_REQUIRED},
    )
    return socket, timeout_type


def _receive(
    socket: SocketLike,
    timeout_type: type[BaseException],
) -> list[dict[str, object]]:
    try:
        raw = socket.recv()
    except Exception as exc:
        if isinstance(exc, timeout_type):
            return []
        raise
    return decode_messages(raw)


def _await_ack(
    socket: SocketLike,
    timeout_type: type[BaseException],
    command_id: int,
    response_key: str,
    pending: list[dict[str, object]],
) -> None:
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        for message in _receive(socket, timeout_type):
            if message == {}:
                socket.send("{}")
                continue
            if message.get("id") == command_id:
                response = message.get(response_key)
                if not isinstance(response, dict) or "error" in response:
                    raise RuntimeError(f"WEBSOCKET_{response_key.upper()}_REJECTED")
                return
            if isinstance(message.get("push"), dict):
                pending.append(message)
    raise TimeoutError(f"WEBSOCKET_{response_key.upper()}_ACK_TIMEOUT")


def connect_and_subscribe() -> tuple[
    SocketLike, type[BaseException], list[dict[str, object]]
]:
    socket, timeout_type = _create_socket()
    pending: list[dict[str, object]] = []
    socket.send(json.dumps({"connect": {}, "id": 1}, separators=(",", ":")))
    _await_ack(socket, timeout_type, 1, "connect", pending)
    for command_id, channel in (
        (2, "public:orderbook-BTCIRT"),
        (3, "public:trades-BTCIRT"),
    ):
        socket.send(
            json.dumps(
                {"id": command_id, "subscribe": {"channel": channel}},
                separators=(",", ":"),
            )
        )
        _await_ack(socket, timeout_type, command_id, "subscribe", pending)
    return socket, timeout_type, pending


def _trade_key(row: dict[str, object]) -> tuple[str, str, str, str]:
    return (
        str(row["exchange_timestamp_ms"]),
        str(row["price"]),
        str(row["volume"]),
        str(row["side"]),
    )


def _process_pushes(
    messages: list[dict[str, object]],
    book_rows: list[dict[str, object]],
    trade_rows: list[dict[str, object]],
    gaps: list[dict[str, object]],
    offsets: OffsetTracker,
    seen_trades: set[tuple[str, str, str, str]],
    snapshot_cutoff_ms: int,
) -> int:
    duplicates = 0
    for message in messages:
        if message == {}:
            continue
        push_value = message.get("push")
        if not isinstance(push_value, dict):
            continue
        push = cast(dict[str, object], push_value)
        channel_value = push.get("channel")
        publication_value = push.get("pub")
        if not isinstance(channel_value, str) or not isinstance(
            publication_value, dict
        ):
            continue
        publication = cast(dict[str, object], publication_value)
        raw_data = publication.get("data")
        if isinstance(raw_data, str):
            payload_value = json.loads(raw_data)
        else:
            payload_value = raw_data
        if not isinstance(payload_value, dict):
            raise ValueError("PUBLICATION_DATA_INVALID")
        payload = cast(dict[str, object], payload_value)
        offset_value = publication.get("offset")
        offset = (
            offset_value
            if isinstance(offset_value, int) and not isinstance(offset_value, bool)
            else None
        )
        captured_at = utc_now()
        if channel_value == "public:orderbook-BTCIRT":
            timestamp = payload.get("lastUpdate")
            if not isinstance(timestamp, int):
                gaps.append(
                    gap_row(
                        "orderbook",
                        captured_at,
                        captured_at,
                        "invalid_or_missing_book_timestamp",
                    )
                )
                continue
            gap: dict[str, object] | None = None
            if offset is None:
                gaps.append(
                    gap_row(
                        "orderbook",
                        captured_at,
                        captured_at,
                        "publisher_offset_missing",
                        end_ms=timestamp,
                    )
                )
            else:
                gap = offsets.observe("orderbook", offset, timestamp)
                if gap is not None:
                    gaps.append(gap)
            if gap is not None and gap["reason"] == "non_monotonic_publisher_offset":
                continue
            if timestamp > snapshot_cutoff_ms:
                book_rows.append(
                    make_book_row(payload, captured_at, "websocket", offset)
                )
        elif channel_value == "public:trades-BTCIRT":
            timestamp = payload.get("time")
            if not isinstance(timestamp, int):
                gaps.append(
                    gap_row(
                        "trades",
                        captured_at,
                        captured_at,
                        "invalid_or_missing_trade_timestamp",
                    )
                )
                continue
            if offset is None:
                gaps.append(
                    gap_row(
                        "trades",
                        captured_at,
                        captured_at,
                        "publisher_offset_missing",
                        end_ms=timestamp,
                    )
                )
            else:
                gap = offsets.observe("trades", offset, timestamp)
                if gap is not None:
                    gaps.append(gap)
                    if gap["reason"] == "non_monotonic_publisher_offset":
                        continue
            row = make_trade_row(payload, captured_at, "websocket", offset)
            key = _trade_key(row)
            if key in seen_trades:
                duplicates += 1
            else:
                seen_trades.add(key)
                trade_rows.append(row)
    return duplicates


def _write_csv(
    path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_capture_artifacts(
    output_dir: Path,
    book_rows: list[dict[str, object]],
    trade_rows: list[dict[str, object]],
    gaps: list[dict[str, object]],
    started_at: str,
    finished_at: str,
    duration_minutes: int,
    duplicate_trade_count: int,
    errors: list[str],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    book_path = output_dir / "orderbook.csv"
    trade_path = output_dir / "trades.csv"
    gaps_path = output_dir / "gaps.csv"
    _write_csv(book_path, BOOK_FIELDS, book_rows)
    _write_csv(trade_path, TRADE_FIELDS, trade_rows)
    _write_csv(gaps_path, GAP_FIELDS, gaps)
    files: dict[str, object] = {}
    for path, count in (
        (book_path, len(book_rows)),
        (trade_path, len(trade_rows)),
        (gaps_path, len(gaps)),
    ):
        files[path.name] = {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "records": count,
        }

    def time_range(rows: list[dict[str, object]]) -> dict[str, str | None]:
        timestamps = [int(str(row["exchange_timestamp_ms"])) for row in rows]
        if not timestamps:
            return {
                "first_exchange_timestamp_ms": None,
                "last_exchange_timestamp_ms": None,
            }
        return {
            "first_exchange_timestamp_ms": str(min(timestamps)),
            "last_exchange_timestamp_ms": str(max(timestamps)),
        }

    websocket_book_count = sum(row["source"] == "websocket" for row in book_rows)
    websocket_trade_count = sum(row["source"] == "websocket" for row in trade_rows)
    rest_trade_seed_count = sum(
        row["source"] == "rest_recent_seed" for row in trade_rows
    )
    manifest: dict[str, object] = {
        "schema_version": "alix.regime-m.capture.v1",
        "code_commit_sha": os.environ.get("GITHUB_SHA", "LOCAL_UNSET"),
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID", "LOCAL_UNSET"),
        "market": "BTCIRT",
        "source": "NOBITEX_PUBLIC_REST_AND_WEBSOCKET",
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
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "requested_duration_minutes": duration_minutes,
        "time_ranges": {
            "orderbook": time_range(book_rows),
            "trades": time_range(trade_rows),
        },
        "websocket_orderbook_records": websocket_book_count,
        "websocket_trade_records": websocket_trade_count,
        "rest_trade_seed_records": rest_trade_seed_count,
        "record_counts": {
            "orderbook": len(book_rows),
            "trades": len(trade_rows),
            "gaps": len(gaps),
            "duplicate_trades_dropped": duplicate_trade_count,
        },
        "gap_count": len(gaps),
        "capture_status": "GAPPED"
        if gaps
        else ("DEGRADED" if errors else "NO_GAPS_OBSERVED"),
        "errors": errors,
        "files": files,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "manifest.sha256").write_text(
        f"{_sha256(manifest_path)}  manifest.json\n",
        encoding="utf-8",
    )
    return manifest


def run_capture(duration_minutes: int, output_dir: Path) -> dict[str, object]:
    if duration_minutes < 1 or duration_minutes > 60:
        raise ValueError("DURATION_MINUTES_MUST_BE_BETWEEN_1_AND_60")
    started_at = utc_now()
    book_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    gaps: list[dict[str, object]] = []
    errors: list[str] = []
    seen_trades: set[tuple[str, str, str, str]] = set()
    offsets = OffsetTracker()
    duplicate_count = 0
    socket: SocketLike | None = None
    deadline = 0.0
    disconnect_start: str | None = None
    snapshot_cutoff_ms = 0
    try:
        socket, timeout_type, pending = connect_and_subscribe()
        deadline = time.monotonic() + duration_minutes * 60
        snapshot = fetch_public_json("/v3/orderbook/BTCIRT")
        snapshot_row = make_book_row(snapshot, utc_now(), "rest_initial_snapshot", None)
        snapshot_cutoff_ms = int(str(snapshot_row["exchange_timestamp_ms"]))
        book_rows.append(snapshot_row)
        try:
            recent = fetch_public_json("/v2/trades/BTCIRT")
            trades_value = recent.get("trades")
            if isinstance(trades_value, list):
                for item in trades_value:
                    if isinstance(item, dict):
                        row = make_trade_row(
                            cast(dict[str, object], item),
                            utc_now(),
                            "rest_recent_seed",
                            None,
                        )
                        key = _trade_key(row)
                        if key not in seen_trades:
                            seen_trades.add(key)
                            trade_rows.append(row)
        except Exception as exc:
            errors.append(f"recent_trade_seed:{type(exc).__name__}")
        duplicate_count += _process_pushes(
            pending,
            book_rows,
            trade_rows,
            gaps,
            offsets,
            seen_trades,
            snapshot_cutoff_ms,
        )
        while time.monotonic() < deadline:
            if socket is None:
                raise RuntimeError("WEBSOCKET_SOCKET_UNAVAILABLE")
            try:
                messages = _receive(socket, timeout_type)
                for message in messages:
                    if message == {}:
                        socket.send("{}")
                duplicate_count += _process_pushes(
                    messages,
                    book_rows,
                    trade_rows,
                    gaps,
                    offsets,
                    seen_trades,
                    snapshot_cutoff_ms,
                )
            except Exception as exc:
                disconnect_start = utc_now()
                errors.append(f"websocket_disconnect:{type(exc).__name__}")
                socket.close()
                socket = None
                while time.monotonic() < deadline and socket is None:
                    time.sleep(min(5.0, max(0.0, deadline - time.monotonic())))
                    if time.monotonic() >= deadline:
                        break
                    try:
                        socket, timeout_type, pending = connect_and_subscribe()
                        reconnect_at = utc_now()
                        for stream in ("orderbook", "trades"):
                            gaps.append(
                                gap_row(
                                    stream,
                                    disconnect_start,
                                    reconnect_at,
                                    "websocket_disconnect_no_replay",
                                )
                            )
                        snapshot = fetch_public_json("/v3/orderbook/BTCIRT")
                        snapshot_row = make_book_row(
                            snapshot, reconnect_at, "rest_reconnect_snapshot", None
                        )
                        snapshot_cutoff_ms = int(
                            str(snapshot_row["exchange_timestamp_ms"])
                        )
                        book_rows.append(snapshot_row)
                        duplicate_count += _process_pushes(
                            pending,
                            book_rows,
                            trade_rows,
                            gaps,
                            offsets,
                            seen_trades,
                            snapshot_cutoff_ms,
                        )
                    except Exception as reconnect_exc:
                        errors.append(
                            f"reconnect_failed:{type(reconnect_exc).__name__}"
                        )
                        if socket is not None:
                            socket.close()
                            socket = None
                if socket is None:
                    reconnect_end = utc_now()
                    for stream in ("orderbook", "trades"):
                        gaps.append(
                            gap_row(
                                stream,
                                disconnect_start,
                                reconnect_end,
                                "reconnect_unavailable_no_replay",
                            )
                        )
                disconnect_start = None
    except Exception as exc:
        errors.append(f"capture_failed:{type(exc).__name__}:{exc}")
        if disconnect_start is not None:
            for stream in ("orderbook", "trades"):
                gaps.append(
                    gap_row(
                        stream, disconnect_start, utc_now(), "reconnect_unavailable"
                    )
                )
    finally:
        if socket is not None:
            socket.close()
    finished_at = utc_now()
    manifest = write_capture_artifacts(
        output_dir,
        book_rows,
        trade_rows,
        gaps,
        started_at,
        finished_at,
        duration_minutes,
        duplicate_count,
        errors,
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture public Nobitex L2 and trade flow."
    )
    parser.add_argument("--duration-minutes", type=int, default=60)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("microstructure-capture")
    )
    args = parser.parse_args()
    manifest = run_capture(args.duration_minutes, args.output_dir)
    counts = cast(dict[str, object], manifest["record_counts"])
    print(
        json.dumps(
            {"capture_status": manifest["capture_status"], "record_counts": counts},
            sort_keys=True,
        )
    )
    return 0 if counts["orderbook"] and counts["trades"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
