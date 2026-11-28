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


def test_artifact_download_uses_separate_unsigned_https_redirect_request() -> None:
    from unittest.mock import patch

    import httpx

    from scripts.aggregate_m8_2_data import _download_artifact_zip

    signed_url = "https://actions.blob.core.windows.net/container/signed-artifact"
    api_response = httpx.Response(
        302,
        headers={"location": signed_url},
        request=httpx.Request(
            "GET",
            "https://api.github.com/repos/owner/repo/actions/artifacts/123/zip",
        ),
    )
    blob_response = httpx.Response(
        200,
        content=b"zip-payload",
        request=httpx.Request("GET", signed_url),
    )

    with patch(
        "scripts.aggregate_m8_2_data.httpx.get",
        side_effect=[api_response, blob_response],
    ) as get:
        assert _download_artifact_zip("owner/repo", 123, "gh-token") == b"zip-payload"

    assert get.call_count == 2
    first_call, second_call = get.call_args_list
    assert first_call.kwargs["headers"]["Authorization"] == "Bearer gh-token"
    assert "headers" not in second_call.kwargs
    assert first_call.kwargs["verify"] is True
    assert second_call.kwargs["verify"] is True


def test_artifact_download_rejects_http_signed_redirect() -> None:
    from unittest.mock import patch

    import httpx

    from scripts.aggregate_m8_2_data import (
        AggregateError,
        _download_artifact_zip,
    )

    api_response = httpx.Response(
        302,
        headers={"location": "http://insecure.example/artifact"},
        request=httpx.Request(
            "GET",
            "https://api.github.com/repos/owner/repo/actions/artifacts/123/zip",
        ),
    )

    with patch(
        "scripts.aggregate_m8_2_data.httpx.get",
        return_value=api_response,
    ) as get:
        with pytest.raises(AggregateError, match="MUST_USE_HTTPS"):
            _download_artifact_zip("owner/repo", 123, "gh-token")

    assert get.call_count == 1
