from __future__ import annotations

import pytest

from scripts.aggregate_m8_2_data import (
    AggregateError,
    inter_segment_gap_rows,
    stable_timestamp_sort,
)


def test_stable_timestamp_sort_preserves_ties_and_all_rows() -> None:
    rows = [
        {"exchange_timestamp_ms": "20", "row": "late"},
        {"exchange_timestamp_ms": "10", "row": "first"},
        {"exchange_timestamp_ms": "10", "row": "second"},
    ]
    result = stable_timestamp_sort(rows)
    assert [item["row"] for item in result] == ["first", "second", "late"]


def test_inter_segment_gap_rows_records_exact_missing_boundaries() -> None:
    previous = {"finished_at_utc": "2026-11-28T00:00:00Z"}
    following = {"started_at_utc": "2026-11-28T00:10:00Z"}
    rows = inter_segment_gap_rows(previous, following)
    assert len(rows) == 2
    assert {item["stream"] for item in rows} == {"orderbook", "trades"}
    assert rows[0]["gap_start_ms"] == "1795824000000"
    assert rows[0]["gap_end_ms"] == "1795824600000"
    assert rows[0]["reason"] == "m8_2_inter_segment_missing_coverage"


def test_inter_segment_gap_rejects_overlapping_segments() -> None:
    previous = {"finished_at_utc": "2026-11-28T00:11:00Z"}
    following = {"started_at_utc": "2026-11-28T00:10:00Z"}
    with pytest.raises(AggregateError, match="NONPOSITIVE_GAP"):
        inter_segment_gap_rows(previous, following)
