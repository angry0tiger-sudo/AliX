from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.capture_nobitex_microstructure import (
    CaptureState,
    OffsetTracker,
    SocketLike,
    _connect_with_fresh_offset_baseline,
    _process_pushes,
    _record_disconnect_gaps,
    decode_messages,
    make_book_row,
    make_trade_row,
    run_capture,
    write_capture_artifacts,
)


def valid_book() -> dict[str, object]:
    return {
        "lastUpdate": 1_700_000_000_000,
        "bids": [["100.0", "2.5"], ["99.5", "3.0"], ["99.0", "4.0"]],
        "asks": [["101.0", "1.5"], ["101.5", "2.0"], ["102.0", "3.0"]],
    }


def valid_trade() -> dict[str, object]:
    return {
        "time": 1_700_000_000_100,
        "price": "100.5",
        "volume": "0.25",
        "type": "buy",
    }


def test_book_schema_extracts_top_levels_and_best_prices() -> None:
    row = make_book_row(valid_book(), "2026-11-18T00:00:00Z", "mock", 7, levels=2)

    assert row["best_bid_price"] == "100.0"
    assert row["best_ask_price"] == "101.0"
    assert row["spread"] == "1.0"
    assert row["mid_price"] == "100.5"
    assert json.loads(str(row["bids_top_json"])) == [["100.0", "2.5"], ["99.5", "3.0"]]
    assert row["offset"] == "7"


def test_book_schema_rejects_missing_side() -> None:
    payload = valid_book()
    payload["asks"] = []

    with pytest.raises(ValueError, match="asks_MISSING"):
        make_book_row(payload, "2026-11-18T00:00:00Z", "mock", None)


def test_trade_schema_validates_and_preserves_side() -> None:
    row = make_trade_row(valid_trade(), "2026-11-18T00:00:00Z", "mock", 9)

    assert row["side"] == "buy"
    assert row["price"] == "100.5"
    assert row["volume"] == "0.25"
    assert row["offset"] == "9"


def test_trade_schema_rejects_invalid_side() -> None:
    payload = valid_trade()
    payload["type"] = "unknown"

    with pytest.raises(ValueError, match="TRADE_SIDE_INVALID"):
        make_trade_row(payload, "2026-11-18T00:00:00Z", "mock", None)


def test_decode_messages_splits_newline_delimited_server_frames() -> None:
    messages = decode_messages(
        b'{"id":1,"connect":{}}\n{"push":{"channel":"public:trades-BTCIRT"}}\n'
    )

    assert len(messages) == 2
    assert messages[0]["id"] == 1
    assert isinstance(messages[1]["push"], dict)


def test_offset_tracker_records_skips_and_non_monotonic_offsets() -> None:
    tracker = OffsetTracker()

    assert tracker.observe("orderbook", 10, 1_000) is None
    skipped = tracker.observe("orderbook", 12, 1_200)
    assert skipped is not None
    assert skipped["reason"] == "publisher_offset_gap"
    assert skipped["previous_offset"] == "10"
    assert skipped["next_offset"] == "12"

    repeated = tracker.observe("orderbook", 11, 1_300)
    assert repeated is not None
    assert repeated["reason"] == "non_monotonic_publisher_offset"


def test_manifest_contains_sha256_record_counts_and_safety(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALIX_SCHEDULE_INTERVAL_MINUTES", "60")
    book = make_book_row(valid_book(), "2026-11-18T00:00:00Z", "mock", None)
    trade = make_trade_row(valid_trade(), "2026-11-18T00:00:01Z", "mock", None)
    gap: dict[str, object] = {
        "stream": "trades",
        "gap_start_utc": "2026-11-18T00:00:02Z",
        "gap_end_utc": "2026-11-18T00:00:03Z",
        "gap_start_ms": "",
        "gap_end_ms": "",
        "reason": "mock_disconnect",
        "previous_offset": "",
        "next_offset": "",
        "replay_attempted": "false",
        "replay_available": "false",
    }

    manifest = write_capture_artifacts(
        tmp_path,
        [book],
        [trade],
        [gap],
        "2026-11-18T00:00:00Z",
        "2026-11-18T00:01:00Z",
        1,
        [],
    )

    assert manifest["capture_status"] == "GAPPED"
    assert manifest["gap_count"] == 1
    counts = manifest["record_counts"]
    assert isinstance(counts, dict)
    assert counts["orderbook"] == 1
    assert counts["trades"] == 1
    assert counts["duplicate_trades_dropped"] == 0
    policy = manifest["trade_deduplication_policy"]
    assert isinstance(policy, str)
    assert policy.startswith("none; preserve")
    files = manifest["files"]
    assert isinstance(files, dict)
    assert files["orderbook.csv"]["records"] == 1
    digest = hashlib.sha256((tmp_path / "trades.csv").read_bytes()).hexdigest()
    assert files["trades.csv"]["sha256"] == digest
    manifest_digest = hashlib.sha256(
        (tmp_path / "manifest.json").read_bytes()
    ).hexdigest()
    assert (
        (tmp_path / "manifest.sha256")
        .read_text(encoding="utf-8")
        .startswith(manifest_digest)
    )
    assert manifest["authorization_header_sent"] is False
    assert manifest["private_api_calls"] == 0
    assert manifest["holdout_bytes_read"] is False
    assert manifest["posthack_data_read"] is False
    assert manifest["Champion"] is None
    assert manifest["order_permission"] is False
    assert manifest["trading_authorization"] == "NO-GO"
    assert manifest["reconnect_attempts"] == 0
    assert manifest["reconnect_successes"] == 0
    assert manifest["reconnect_failures"] == 0


def test_scheduled_segment_declares_missing_inter_run_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    monkeypatch.setenv("ALIX_SCHEDULE_INTERVAL_MINUTES", "60")

    manifest = write_capture_artifacts(
        tmp_path,
        [],
        [],
        [],
        "2026-11-18T00:00:00Z",
        "2026-11-18T00:55:00Z",
        55,
        [],
    )

    coverage = manifest["coverage_segment"]
    assert isinstance(coverage, dict)
    assert coverage["inter_segment_coverage_status"] == "MISSING_COVERAGE"
    assert coverage["schedule_interval_minutes"] == 60
    assert coverage["nominal_inter_segment_missing_minutes"] == 5
    assert coverage["exact_inter_segment_gap_boundaries"].startswith("DERIVE_FROM_")
    assert coverage["capture_segments_must_not_be_joined_as_continuous"] is True


def test_scheduled_segment_rejects_a_window_without_missing_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    monkeypatch.setenv("ALIX_SCHEDULE_INTERVAL_MINUTES", "55")

    with pytest.raises(
        ValueError, match="SCHEDULED_CAPTURE_CADENCE_OR_DURATION_INVALID"
    ):
        write_capture_artifacts(
            tmp_path,
            [],
            [],
            [],
            "2026-11-18T00:00:00Z",
            "2026-11-18T00:55:00Z",
            55,
            [],
        )


def test_manual_long_soak_segment_declares_ten_minute_coverage_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("ALIX_CAPTURE_SEGMENT_INTERVAL_MINUTES", "360")

    manifest = write_capture_artifacts(
        tmp_path,
        [],
        [],
        [],
        "2026-11-23T00:00:00Z",
        "2026-11-23T05:50:00Z",
        350,
        [],
        reconnect_attempts=2,
        reconnect_successes=1,
        reconnect_failures=1,
    )

    coverage = manifest["coverage_segment"]
    assert isinstance(coverage, dict)
    assert coverage["kind"] == "MANUAL_SOAK_SEGMENT"
    assert coverage["inter_segment_coverage_status"] == "MISSING_COVERAGE"
    assert coverage["segment_interval_minutes"] == 360
    assert coverage["nominal_inter_segment_missing_minutes"] == 10
    assert coverage["capture_segments_must_not_be_joined_as_continuous"] is True
    assert manifest["reconnect_attempts"] == 2
    assert manifest["reconnect_successes"] == 1
    assert manifest["reconnect_failures"] == 1


def test_manual_long_soak_segment_rejects_interval_not_longer_than_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("ALIX_CAPTURE_SEGMENT_INTERVAL_MINUTES", "350")

    with pytest.raises(
        ValueError, match="MANUAL_SOAK_SEGMENT_INTERVAL_MUST_EXCEED_DURATION"
    ):
        write_capture_artifacts(
            tmp_path,
            [],
            [],
            [],
            "2026-11-23T00:00:00Z",
            "2026-11-23T05:50:00Z",
            350,
            [],
        )


def test_run_capture_rejects_duration_above_long_soak_limit(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="DURATION_MINUTES_MUST_BE_BETWEEN_1_AND_350"):
        run_capture(351, tmp_path)


def test_push_processor_marks_offset_gap_and_drops_non_monotonic_event() -> None:
    tracker = OffsetTracker()
    book_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    gaps: list[dict[str, object]] = []
    first: dict[str, object] = {
        "push": {
            "channel": "public:orderbook-BTCIRT",
            "pub": {"offset": 10, "data": valid_book()},
        }
    }
    skipped: dict[str, object] = {
        "push": {
            "channel": "public:orderbook-BTCIRT",
            "pub": {
                "offset": 12,
                "data": {
                    **valid_book(),
                    "lastUpdate": 1_700_000_000_200,
                },
            },
        }
    }
    repeated: dict[str, object] = {
        "push": {
            "channel": "public:orderbook-BTCIRT",
            "pub": {
                "offset": 12,
                "data": {
                    **valid_book(),
                    "lastUpdate": 1_700_000_000_300,
                },
            },
        }
    }

    _process_pushes(
        [first, skipped, repeated],
        book_rows,
        trade_rows,
        gaps,
        tracker,
        0,
    )

    assert len(book_rows) == 2
    assert len(gaps) == 2
    assert [gap["reason"] for gap in gaps] == [
        "publisher_offset_gap",
        "non_monotonic_publisher_offset",
    ]


class _FakeSocket(SocketLike):
    def send(self, payload: str) -> None:
        return None

    def recv(self) -> str:
        return "{}"

    def close(self) -> None:
        return None


def test_reconnect_resets_both_offset_baselines_and_records_disconnect_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = CaptureState()
    initial_book = {
        "push": {
            "channel": "public:orderbook-BTCIRT",
            "pub": {"offset": 1000, "data": valid_book()},
        }
    }
    initial_trade = {
        "push": {
            "channel": "public:trades-BTCIRT",
            "pub": {"offset": 1000, "data": valid_trade()},
        }
    }
    _process_pushes(
        [initial_book, initial_trade],
        state.book_rows,
        state.trade_rows,
        state.gaps,
        state.offsets,
        0,
    )
    assert [row["offset"] for row in state.book_rows] == ["1000"]
    assert [row["offset"] for row in state.trade_rows] == ["1000"]

    _record_disconnect_gaps(
        state,
        "2026-11-25T00:00:00+00:00",
        "2026-11-25T00:00:05+00:00",
        "websocket_disconnect_no_replay",
    )

    socket = _FakeSocket()
    monkeypatch.setattr(
        "scripts.capture_nobitex_microstructure.connect_and_subscribe",
        lambda: (socket, TimeoutError, []),
    )
    connected_socket, timeout_type, pending = _connect_with_fresh_offset_baseline(state)

    assert connected_socket is socket
    assert timeout_type is TimeoutError
    assert pending == []

    post_reconnect: list[dict[str, object]] = []
    for offset in range(1, 5):
        post_reconnect.append(
            {
                "push": {
                    "channel": "public:orderbook-BTCIRT",
                    "pub": {
                        "offset": offset,
                        "data": {
                            **valid_book(),
                            "lastUpdate": 1_700_000_001_000 + offset,
                        },
                    },
                }
            }
        )
        post_reconnect.append(
            {
                "push": {
                    "channel": "public:trades-BTCIRT",
                    "pub": {
                        "offset": offset,
                        "data": {
                            **valid_trade(),
                            "time": 1_700_000_002_000 + offset,
                        },
                    },
                }
            }
        )

    _process_pushes(
        post_reconnect,
        state.book_rows,
        state.trade_rows,
        state.gaps,
        state.offsets,
        0,
    )

    assert [row["offset"] for row in state.book_rows] == [
        "1000",
        "1",
        "2",
        "3",
        "4",
    ]
    assert [row["offset"] for row in state.trade_rows] == [
        "1000",
        "1",
        "2",
        "3",
        "4",
    ]
    assert len(state.gaps) == 2
    assert {gap["stream"] for gap in state.gaps} == {"orderbook", "trades"}
    assert {gap["reason"] for gap in state.gaps} == {"websocket_disconnect_no_replay"}


def test_offset_tracker_reset_can_be_scoped_to_one_stream() -> None:
    tracker = OffsetTracker()
    assert tracker.observe("orderbook", 1000, 1_700_000_000_000) is None
    assert tracker.observe("trades", 2000, 1_700_000_000_100) is None

    tracker.reset_baseline("orderbook")

    assert tracker.observe("orderbook", 1, 1_700_000_000_200) is None
    assert tracker.observe("trades", 2001, 1_700_000_000_300) is None


def test_trade_stream_preserves_identical_public_observations() -> None:
    tracker = OffsetTracker()
    book_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    gaps: list[dict[str, object]] = []
    first: dict[str, object] = {
        "push": {
            "channel": "public:trades-BTCIRT",
            "pub": {"offset": 31, "data": valid_trade()},
        }
    }
    second: dict[str, object] = {
        "push": {
            "channel": "public:trades-BTCIRT",
            "pub": {"offset": 32, "data": valid_trade()},
        }
    }

    _process_pushes(
        [first, second],
        book_rows,
        trade_rows,
        gaps,
        tracker,
        0,
    )

    assert len(trade_rows) == 2
    assert trade_rows[0]["price"] == trade_rows[1]["price"]
    assert (
        trade_rows[0]["exchange_timestamp_ms"] == trade_rows[1]["exchange_timestamp_ms"]
    )
    assert gaps == []
