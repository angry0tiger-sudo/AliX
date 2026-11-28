"""Download, verify and merge all 14 M8.2 public BTCUSDT artifacts.

Rows are preserved and stably sorted by exchange timestamp. Exact inter-segment gaps
are materialized in gaps.csv so forward-return windows cannot bridge missing coverage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.audit_microstructure_capture import audit_capture

ARTIFACT_PREFIX = "alix-m8-2-btcusdt-"
RAW_FILES = {"manifest.json", "manifest.sha256", "orderbook.csv", "trades.csv", "gaps.csv"}
CSV_SCHEMAS = {
    "orderbook.csv": (
        "captured_at_utc", "source", "exchange_timestamp_ms", "offset",
        "best_bid_price", "best_bid_size", "best_ask_price", "best_ask_size",
        "spread", "mid_price", "bids_top_json", "asks_top_json",
    ),
    "trades.csv": (
        "captured_at_utc", "source", "exchange_timestamp_ms", "side",
        "price", "volume", "offset",
    ),
    "gaps.csv": (
        "stream", "gap_start_utc", "gap_end_utc", "gap_start_ms",
        "gap_end_ms", "reason", "previous_offset", "next_offset",
        "replay_attempted", "replay_available",
    ),
}


class AggregateError(RuntimeError):
    """Raised when an M8.2 artifact fails provenance or coverage validation."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise AggregateError("NAIVE_CAPTURE_TIMESTAMP")
    return parsed.astimezone(timezone.utc)


def to_epoch_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def stable_timestamp_sort(
    rows: list[dict[str, str]], timestamp_field: str = "exchange_timestamp_ms"
) -> list[dict[str, str]]:
    """Stable sort keeps original row order for equal exchange timestamps."""
    return sorted(rows, key=lambda row: int(row[timestamp_field]))


def inter_segment_gap_rows(
    previous: dict[str, Any], following: dict[str, Any]
) -> list[dict[str, str]]:
    previous_end = parse_time(str(previous["finished_at_utc"]))
    next_start = parse_time(str(following["started_at_utc"]))
    if next_start <= previous_end:
        raise AggregateError("SEGMENTS_OVERLAP_OR_HAVE_NONPOSITIVE_GAP")
    start_ms = to_epoch_ms(previous_end)
    end_ms = to_epoch_ms(next_start)
    start_text = previous_end.isoformat().replace("+00:00", "Z")
    end_text = next_start.isoformat().replace("+00:00", "Z")
    return [
        {
            "stream": stream,
            "gap_start_utc": start_text,
            "gap_end_utc": end_text,
            "gap_start_ms": str(start_ms),
            "gap_end_ms": str(end_ms),
            "reason": "m8_2_inter_segment_missing_coverage",
            "previous_offset": "",
            "next_offset": "",
            "replay_attempted": "false",
            "replay_available": "false",
        }
        for stream in ("orderbook", "trades")
    ]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None


def _api_get(url: str, token: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "AliX-M8.2-Free-GitHub-Actions",
        },
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def _download_artifact_zip(repo: str, artifact_id: int, token: str) -> bytes:
    url = f"https://api.github.com/repos/{repo}/actions/artifacts/{artifact_id}/zip"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "AliX-M8.2-Free-GitHub-Actions",
        },
    )
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=90) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code not in {301, 302, 303, 307, 308}:
            raise
        location = exc.headers.get("Location")
        if not location:
            raise AggregateError("ARTIFACT_REDIRECT_WITHOUT_LOCATION") from exc
    # The signed redirect is fetched without forwarding the GitHub token.
    with urllib.request.urlopen(location, timeout=180) as response:
        return response.read()


def _list_artifacts(repo: str, token: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for page in range(1, 11):
        url = (
            f"https://api.github.com/repos/{repo}/actions/artifacts"
            f"?per_page=100&page={page}"
        )
        payload = json.loads(_api_get(url, token).decode("utf-8"))
        batch = payload.get("artifacts")
        if not isinstance(batch, list):
            raise AggregateError("GITHUB_ARTIFACT_LIST_INVALID")
        result.extend(item for item in batch if isinstance(item, dict))
        if len(batch) < 100:
            break
    else:
        raise AggregateError("GITHUB_ARTIFACT_LIST_EXCEEDS_1000")
    return result


def _extract_raw_artifact(data: bytes, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if set(archive.namelist()) != RAW_FILES:
                raise AggregateError("RAW_ARTIFACT_FILE_SET_MISMATCH")
            for name in sorted(RAW_FILES):
                (destination / name).write_bytes(archive.read(name))
    except zipfile.BadZipFile as exc:
        raise AggregateError("DOWNLOADED_ARTIFACT_NOT_ZIP") from exc
    audit_capture(destination)


def _load_segment(
    artifact: dict[str, Any],
    segment_number: int,
    destination: Path,
    repo: str,
    token: str,
    soak_id: str,
    expected_sha: str,
) -> dict[str, Any]:
    artifact_id = int(artifact["id"])
    content = _download_artifact_zip(repo, artifact_id, token)
    downloaded_sha = sha256(content)
    api_digest = str(artifact.get("digest") or "")
    if api_digest.startswith("sha256:") and api_digest.removeprefix("sha256:") != downloaded_sha:
        raise AggregateError(f"ARTIFACT_API_DIGEST_MISMATCH:{segment_number}")
    _extract_raw_artifact(content, destination)
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    expected_fields = {
        "market": "BTCUSDT",
        "soak_id": soak_id,
        "segment_number": segment_number,
        "expected_segment_count": 14,
        "code_commit_sha": expected_sha,
        "requested_duration_minutes": 110,
    }
    for key, expected in expected_fields.items():
        if manifest.get(key) != expected:
            raise AggregateError(f"MANIFEST_{key.upper()}_MISMATCH:{segment_number}")
    if int(manifest.get("workflow_run_id", 0)) != int(artifact.get("workflow_run_id", -1)):
        raise AggregateError(f"WORKFLOW_RUN_ID_MISMATCH:{segment_number}")
    started = parse_time(str(manifest["started_at_utc"]))
    finished = parse_time(str(manifest["finished_at_utc"]))
    captured_seconds = (finished - started).total_seconds()
    if not 6600 <= captured_seconds <= 6900:
        raise AggregateError(f"CAPTURE_DURATION_OUT_OF_RANGE:{segment_number}")
    coverage = manifest.get("coverage_segment", {})
    if coverage.get("segment_interval_minutes") != 120:
        raise AggregateError(f"SEGMENT_CADENCE_MISMATCH:{segment_number}")
    if coverage.get("nominal_inter_segment_missing_minutes") != 10:
        raise AggregateError(f"SEGMENT_GAP_DECLARATION_MISMATCH:{segment_number}")
    return {
        "segment_number": segment_number,
        "run_id": int(manifest["workflow_run_id"]),
        "artifact_id": artifact_id,
        "artifact_name": str(artifact["name"]),
        "artifact_sha256": downloaded_sha,
        "manifest_sha256": sha256((destination / "manifest.json").read_bytes()),
        "started_at_utc": manifest["started_at_utc"],
        "finished_at_utc": manifest["finished_at_utc"],
        "capture_seconds": round(captured_seconds, 3),
        "orderbook_records": int(manifest["record_counts"]["orderbook"]),
        "trade_records": int(manifest["record_counts"]["trades"]),
        "gap_records": int(manifest["record_counts"]["gaps"]),
        "capture_status": manifest["capture_status"],
    }


def _read_rows(path: Path, filename: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CSV_SCHEMAS[filename]:
            raise AggregateError(f"CSV_SCHEMA_MISMATCH:{filename}")
        return list(reader)


def _write_csv(path: Path, filename: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_SCHEMAS[filename])
        writer.writeheader()
        writer.writerows(rows)
    raw = path.read_bytes()
    return {"sha256": sha256(raw), "bytes": len(raw), "records": len(rows)}


def aggregate(
    soak_id: str,
    expected_segments: int,
    output_dir: Path,
    repo: str,
    token: str,
    expected_sha: str,
    final_run_id: int,
) -> dict[str, Any]:
    if expected_segments != 14:
        raise AggregateError("M8_2_REQUIRES_14_SEGMENTS_FOR_AT_LEAST_24H_CAPTURE")
    artifacts = _list_artifacts(repo, token)
    prefix = f"{ARTIFACT_PREFIX}{soak_id}-segment-"
    candidates = [
        item for item in artifacts
        if str(item.get("name", "")).startswith(prefix)
        and item.get("expired") is False
    ]
    by_segment: dict[int, dict[str, Any]] = {}
    for item in candidates:
        match = re.fullmatch(
            re.escape(prefix) + r"(\d+)-run-(\d+)",
            str(item.get("name", "")),
        )
        if not match:
            raise AggregateError(f"UNEXPECTED_SOAK_ARTIFACT_NAME:{item.get('name')}")
        number, run_id = int(match.group(1)), int(match.group(2))
        if int(item.get("workflow_run_id", -1)) != run_id:
            raise AggregateError(f"ARTIFACT_NAME_RUN_ID_MISMATCH:{number}")
        if number in by_segment:
            raise AggregateError(f"DUPLICATE_SEGMENT_ARTIFACT:{number}")
        by_segment[number] = item
    expected_numbers = set(range(1, expected_segments + 1))
    if set(by_segment) != expected_numbers:
        raise AggregateError(
            f"SOAK_ARTIFACT_SET_MISMATCH:expected={sorted(expected_numbers)} "
            f"actual={sorted(by_segment)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_root = output_dir / "raw_segments"
    raw_root.mkdir(parents=True, exist_ok=True)
    segment_meta: list[dict[str, Any]] = []
    segment_dirs: list[Path] = []
    for number in range(1, expected_segments + 1):
        segment_dir = raw_root / f"segment_{number:02d}"
        segment_meta.append(
            _load_segment(
                by_segment[number],
                number,
                segment_dir,
                repo,
                token,
                soak_id,
                expected_sha,
            )
        )
        segment_dirs.append(segment_dir)

    inter_gaps: list[dict[str, Any]] = []
    for previous, following in zip(segment_meta, segment_meta[1:]):
        generated = inter_segment_gap_rows(previous, following)
        inter_gaps.append({
            "after_segment": previous["segment_number"],
            "before_segment": following["segment_number"],
            "start_utc": generated[0]["gap_start_utc"],
            "end_utc": generated[0]["gap_end_utc"],
            "missing_seconds": (
                parse_time(generated[0]["gap_end_utc"])
                - parse_time(generated[0]["gap_start_utc"])
            ).total_seconds(),
            "nominal_missing_seconds": 600,
        })

    merged_rows: dict[str, list[dict[str, str]]] = {}
    for filename in ("orderbook.csv", "trades.csv"):
        combined: list[dict[str, str]] = []
        for segment_dir in segment_dirs:
            combined.extend(_read_rows(segment_dir / filename, filename))
        merged_rows[filename] = stable_timestamp_sort(combined)

    gaps: list[dict[str, str]] = []
    for segment_dir in segment_dirs:
        gaps.extend(_read_rows(segment_dir / "gaps.csv", "gaps.csv"))
    for previous, following in zip(segment_meta, segment_meta[1:]):
        gaps.extend(inter_segment_gap_rows(previous, following))
    gaps.sort(
        key=lambda row: (
            int(row["gap_start_ms"] or 0),
            row["stream"],
            row["reason"],
        )
    )

    file_meta: dict[str, Any] = {}
    for filename, rows in (
        ("orderbook.csv", merged_rows["orderbook.csv"]),
        ("trades.csv", merged_rows["trades.csv"]),
        ("gaps.csv", gaps),
    ):
        file_meta[filename] = _write_csv(output_dir / filename, filename, rows)

    actual_seconds = round(sum(item["capture_seconds"] for item in segment_meta), 3)
    if actual_seconds < 24 * 60 * 60:
        raise AggregateError("LESS_THAN_24_HOURS_OF_ACTUAL_CAPTURE")
    first_started = segment_meta[0]["started_at_utc"]
    last_finished = segment_meta[-1]["finished_at_utc"]
    total_orderbook = len(merged_rows["orderbook.csv"])
    total_trades = len(merged_rows["trades.csv"])
    total_gaps = len(gaps)
    aggregate_manifest: dict[str, Any] = {
        "schema_version": "alix.m8_2.aggregate.v1",
        "study": "M8.2_PUBLIC_BTCUSDT_24H_CAPTURE_AGGREGATE",
        "soak_id": soak_id,
        "market": "BTCUSDT",
        "workflow_run_id": str(final_run_id),
        "code_commit_sha": expected_sha,
        "expected_segment_count": expected_segments,
        "actual_segment_count": len(segment_meta),
        "target_cadence_minutes": 120,
        "requested_capture_minutes_per_segment": 110,
        "total_requested_capture_minutes": 110 * expected_segments,
        "total_actual_capture_seconds": actual_seconds,
        "started_at_utc": first_started,
        "finished_at_utc": last_finished,
        "record_counts": {
            "orderbook": total_orderbook,
            "trades": total_trades,
            "gaps": total_gaps,
            "duplicate_trades_dropped": 0,
        },
        "gap_count": total_gaps,
        "files": file_meta,
        "coverage": {
            "captured_minutes_target": 1440,
            "captured_minutes_actual": round(actual_seconds / 60, 3),
            "inter_segment_gap_count": len(inter_gaps),
            "inter_segment_gaps": inter_gaps,
            "coverage_status": "DISCONTINUOUS_BY_DESIGN",
            "rows_deduplicated": False,
            "all_rows_sorted_by_exchange_timestamp_ms": True,
        },
        "source_segments": segment_meta,
        "safety": {
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
        },
        "interpretation": {
            "tfi_side_field": "Nobitex public trade type buy/sell",
            "tfi_aggressor_semantics": "PROXY_NOT_EXPLICITLY_DEFINED_BY_OFFICIAL_SCHEMA",
        },
    }
    manifest_bytes = (json.dumps(aggregate_manifest, indent=2, sort_keys=True) + "\n").encode()
    (output_dir / "manifest.json").write_bytes(manifest_bytes)
    manifest_digest = sha256(manifest_bytes)
    (output_dir / "manifest.sha256").write_text(
        f"{manifest_digest}  manifest.json\n", encoding="utf-8"
    )

    archive_path = output_dir / "analysis_archive.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in (
            "manifest.json", "manifest.sha256", "orderbook.csv", "trades.csv", "gaps.csv"
        ):
            archive.write(output_dir / filename, arcname=filename)
    archive_digest = sha256(archive_path.read_bytes())
    inventory = {
        "schema_version": 1,
        "collection": "M8.2 aggregate BTCUSDT at least 24h public microstructure capture",
        "dataset_market": "BTCUSDT",
        "expected_archive_count": 1,
        "artifacts_by_sha256": {
            archive_digest: {
                "run_id": final_run_id,
                "artifact_id": None,
                "market": "BTCUSDT",
                "orderbook_rows": total_orderbook,
                "source_segment_count": expected_segments,
            }
        },
    }
    (output_dir / "analysis_inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "soak_id": soak_id,
        "segment_count": len(segment_meta),
        "actual_capture_minutes": round(actual_seconds / 60, 3),
        "wall_clock_start_utc": first_started,
        "wall_clock_end_utc": last_finished,
        "orderbook_records": total_orderbook,
        "trade_records": total_trades,
        "gap_records": total_gaps,
        "inter_segment_gap_count": len(inter_gaps),
        "merged_orderbook_sha256": file_meta["orderbook.csv"]["sha256"],
        "merged_trades_sha256": file_meta["trades.csv"]["sha256"],
        "merged_gaps_sha256": file_meta["gaps.csv"]["sha256"],
        "aggregate_manifest_sha256": manifest_digest,
        "analysis_archive_sha256": archive_digest,
        "analysis_inventory": str(output_dir / "analysis_inventory.json"),
    }
    (output_dir / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soak-id", required=True)
    parser.add_argument("--expected-segments", type=int, default=14)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import os

    token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    expected_sha = os.environ.get("EXPECTED_CODE_SHA", "")
    final_run_id = int(os.environ.get("GITHUB_RUN_ID", "0"))
    if not token or not repo or not expected_sha or final_run_id <= 0:
        raise SystemExit("GITHUB_ACTIONS_CONTEXT_MISSING")
    result = aggregate(
        args.soak_id, args.expected_segments, args.output_dir,
        repo, token, expected_sha, final_run_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
