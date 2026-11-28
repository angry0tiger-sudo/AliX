"""Fail-closed M8 BTCUSDT microstructure correlation study.

The evaluator accepts only hash-pinned M8.2 aggregate artifacts. Explicit inter-segment
gaps remain in gaps.csv and any forward-return window crossing a gap is rejected.
It never reads Holdout/Post-Hack, fits a model, or places orders.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import io
import json
import math
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence, cast

DEFAULT_INVENTORY = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "research"
    / "m1_6_artifact_inventory.json"
)
BOOK_FIELDS = {
    "exchange_timestamp_ms",
    "best_bid_price",
    "best_bid_size",
    "best_ask_price",
    "best_ask_size",
    "spread",
    "mid_price",
    "bids_top_json",
    "asks_top_json",
}
FEATURES = ("obi_top5", "microprice_edge_bps", "trade_flow_imbalance_60s")
HORIZONS_MS = {"5m": 5 * 60_000, "15m": 15 * 60_000}
STALE_MAX_MS = 5_000
FUTURE_TOLERANCE_MS = 30_000
ALPHA = 0.05


class InputError(ValueError):
    """Raised when the raw input or its provenance is invalid."""


@dataclass(frozen=True)
class Snapshot:
    timestamp_ms: int
    mid: float
    features: dict[str, float]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_utc_ms(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise InputError("GAP_TIMESTAMP_MUST_BE_TIMEZONE_AWARE")
    return int(parsed.timestamp() * 1000)


def _parse_level(item: Any, field: str) -> tuple[float, float]:
    if not isinstance(item, list) or len(item) != 2:
        raise InputError(f"{field}_LEVEL_INVALID")
    try:
        price, size = float(item[0]), float(item[1])
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputError(f"{field}_LEVEL_INVALID") from exc
    if not (math.isfinite(price) and math.isfinite(size) and price > 0 and size >= 0):
        raise InputError(f"{field}_LEVEL_OUT_OF_RANGE")
    return price, size


def _levels(raw: str, field: str) -> list[tuple[float, float]]:
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InputError(f"{field}_JSON_INVALID") from exc
    if not isinstance(values, list) or len(values) < 5:
        raise InputError(f"{field}_MUST_HAVE_AT_LEAST_FIVE_LEVELS")
    return [_parse_level(item, field) for item in values[:5]]


def _parse_top_of_book(
    row: dict[str, str],
) -> tuple[float, float, float, float, float, float]:
    try:
        values = (
            float(row["best_bid_price"]),
            float(row["best_ask_price"]),
            float(row["best_bid_size"]),
            float(row["best_ask_size"]),
            float(row["mid_price"]),
            float(row["spread"]),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise InputError("TOP_OF_BOOK_NUMERIC_FIELD_INVALID") from exc
    if not all(math.isfinite(value) for value in values):
        raise InputError("TOP_OF_BOOK_NONFINITE")
    bid, ask, bid_size, ask_size, mid, spread = values
    if bid <= 0 or ask <= 0 or mid <= 0 or ask < bid or bid_size < 0 or ask_size < 0:
        raise InputError("TOP_OF_BOOK_OUT_OF_RANGE")
    return values


def _top_five_depths(row: dict[str, str]) -> tuple[float, float]:
    bids = _levels(row["bids_top_json"], "bids_top_json")
    asks = _levels(row["asks_top_json"], "asks_top_json")
    return sum(size for _, size in bids), sum(size for _, size in asks)


def compute_features(row: dict[str, str]) -> tuple[float, dict[str, float]]:
    """Compute top-five OBI, microprice edge, relative spread, and depth."""
    bid, ask, bid_size, ask_size, mid, spread = _parse_top_of_book(row)
    bid_depth, ask_depth = _top_five_depths(row)
    total_depth = bid_depth + ask_depth
    if total_depth <= 0:
        raise InputError("TOP5_DEPTH_NONPOSITIVE")
    top_size = bid_size + ask_size
    if top_size <= 0:
        raise InputError("TOP_OF_BOOK_SIZE_NONPOSITIVE")
    microprice = (ask * bid_size + bid * ask_size) / top_size
    return mid, {
        "obi_top5": (bid_depth - ask_depth) / total_depth,
        "microprice_edge_bps": (microprice / mid - 1.0) * 10_000.0,
        "spread_bps": (spread / mid) * 10_000.0,
        "depth_top5": total_depth,
    }


def _normalize_run_id(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _validate_manifest_identity(
    manifest: dict[str, Any], entry: dict[str, Any], path: Path
) -> None:
    if manifest.get("market") != entry.get("market"):
        raise InputError(f"MANIFEST_MARKET_INVENTORY_MISMATCH:{path.name}")
    manifest_run_id = _normalize_run_id(manifest.get("workflow_run_id"))
    inventory_run_id = _normalize_run_id(entry.get("run_id"))
    if manifest_run_id is None or manifest_run_id != inventory_run_id:
        raise InputError(f"MANIFEST_RUN_ID_MISMATCH:{path.name}")


def _validate_archive_layout(archive: zipfile.ZipFile, path: Path) -> None:
    required = {
        "manifest.json",
        "manifest.sha256",
        "orderbook.csv",
        "trades.csv",
        "gaps.csv",
    }
    if set(archive.namelist()) != required:
        raise InputError(f"ARCHIVE_FILE_SET_MISMATCH:{path.name}")


def _read_manifest(archive: zipfile.ZipFile, path: Path) -> dict[str, Any]:
    manifest_bytes = archive.read("manifest.json")
    checksum_line = archive.read("manifest.sha256").decode("utf-8").strip().split()
    if checksum_line != [_sha256(manifest_bytes), "manifest.json"]:
        raise InputError(f"MANIFEST_SHA256_MISMATCH:{path.name}")
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError(f"MANIFEST_JSON_INVALID:{path.name}") from exc
    if not isinstance(manifest, dict):
        raise InputError(f"MANIFEST_OBJECT_REQUIRED:{path.name}")
    return manifest


def _read_csv_rows(
    archive: zipfile.ZipFile, files_meta: dict[str, Any], filename: str
) -> tuple[list[str] | None, list[dict[str, str]]]:
    contents = archive.read(filename)
    meta = files_meta.get(filename)
    if not isinstance(meta, dict):
        raise InputError(f"MANIFEST_FILE_METADATA_MISSING:{filename}")
    if meta.get("sha256") != _sha256(contents) or meta.get("bytes") != len(contents):
        raise InputError(f"CSV_INTEGRITY_MISMATCH:{filename}")
    with io.TextIOWrapper(io.BytesIO(contents), encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = cast(list[dict[str, str]], list(reader))
        fieldnames = list(reader.fieldnames) if reader.fieldnames is not None else None
    if meta.get("records") != len(rows):
        raise InputError(f"CSV_RECORD_COUNT_MISMATCH:{filename}")
    return fieldnames, rows


def _parse_orderbook_rows(
    rows: list[dict[str, str]], fieldnames: list[str] | None, path: Path
) -> list[Snapshot]:
    if not fieldnames or not BOOK_FIELDS.issubset(set(fieldnames)):
        raise InputError("ORDERBOOK_SCHEMA_MISMATCH")
    snapshots: list[Snapshot] = []
    for row in rows:
        try:
            timestamp = int(row["exchange_timestamp_ms"])
        except (KeyError, ValueError, TypeError) as exc:
            raise InputError("ORDERBOOK_TIMESTAMP_INVALID") from exc
        mid, features = compute_features(row)
        snapshots.append(Snapshot(timestamp, mid, features))
    _validate_orderbook_timestamps(snapshots, path)
    return snapshots


def _attach_trade_flow_imbalance(
    snapshots: list[Snapshot], trade_rows: list[dict[str, str]]
) -> None:
    """Attach trailing-60s signed-volume imbalance from WebSocket trades only.

    REST seed observations are excluded because the public capture has no stable
    trade identifier with which to prove that seed rows do not duplicate WebSocket rows.
    """
    trades: list[tuple[int, str, float]] = []
    for row in trade_rows:
        if row.get("source") != "websocket":
            continue
        try:
            timestamp = int(row["exchange_timestamp_ms"])
            side = row["side"]
            volume = float(row["volume"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise InputError("TRADE_FLOW_ROW_INVALID") from exc
        if side not in {"buy", "sell"}:
            raise InputError("TRADE_FLOW_SIDE_INVALID")
        if timestamp <= 0 or not math.isfinite(volume) or volume < 0:
            raise InputError("TRADE_FLOW_VALUE_OUT_OF_RANGE")
        trades.append((timestamp, side, volume))

    trades.sort(key=lambda item: item[0])
    left = 0
    right = 0
    buy_volume = 0.0
    sell_volume = 0.0
    for snapshot in snapshots:
        while right < len(trades) and trades[right][0] <= snapshot.timestamp_ms:
            _, side, volume = trades[right]
            if side == "buy":
                buy_volume += volume
            else:
                sell_volume += volume
            right += 1

        cutoff = snapshot.timestamp_ms - 60_000
        while left < right and trades[left][0] <= cutoff:
            _, side, volume = trades[left]
            if side == "buy":
                buy_volume -= volume
            else:
                sell_volume -= volume
            left += 1

        total_volume = buy_volume + sell_volume
        snapshot.features["trade_flow_imbalance_60s"] = (
            (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0.0
        )


def _validate_orderbook_timestamps(snapshots: list[Snapshot], path: Path) -> None:
    if not snapshots:
        raise InputError(f"ORDERBOOK_EMPTY:{path.name}")
    timestamps = [item.timestamp_ms for item in snapshots]
    if timestamps != sorted(timestamps) or len(set(timestamps)) != len(timestamps):
        raise InputError(f"ORDERBOOK_TIMESTAMPS_NOT_STRICTLY_INCREASING:{path.name}")


def _gap_timestamp(row: dict[str, str], ms_field: str, utc_field: str) -> int:
    if row.get(ms_field):
        return int(row[ms_field])
    return _parse_utc_ms(row[utc_field])


def _parse_gap_rows(rows: list[dict[str, str]]) -> list[tuple[int, int]]:
    gaps: list[tuple[int, int]] = []
    for row in rows:
        if row.get("stream") != "orderbook":
            continue
        start = _gap_timestamp(row, "gap_start_ms", "gap_start_utc")
        end = _gap_timestamp(row, "gap_end_ms", "gap_end_utc")
        if end < start:
            raise InputError("GAP_INTERVAL_REVERSED")
        gaps.append((start, end))
    return gaps


def _read_verified_archive(
    path: Path, inventory: dict[str, Any], expected_market: str
) -> tuple[dict[str, Any], list[Snapshot], list[tuple[int, int]]]:
    raw_archive = path.read_bytes()
    archive_sha = _sha256(raw_archive)
    entry = inventory.get("artifacts_by_sha256", {}).get(archive_sha)
    if not isinstance(entry, dict):
        raise InputError(f"UNREGISTERED_ARCHIVE_SHA256:{path.name}")
    try:
        with zipfile.ZipFile(io.BytesIO(raw_archive)) as archive:
            _validate_archive_layout(archive, path)
            manifest = _read_manifest(archive, path)
            _validate_manifest_identity(manifest, entry, path)
            if manifest.get("market") != expected_market:
                return manifest, [], []
            files_meta = manifest.get("files")
            if not isinstance(files_meta, dict):
                raise InputError(f"MANIFEST_FILES_MISSING:{path.name}")
            trade_fields, trade_rows = _read_csv_rows(
                archive, files_meta, "trades.csv"
            )
            if not trade_fields or not {
                "source", "exchange_timestamp_ms", "side", "volume"
            }.issubset(set(trade_fields)):
                raise InputError("TRADE_SCHEMA_MISMATCH")
            book_fields, book_rows = _read_csv_rows(
                archive, files_meta, "orderbook.csv"
            )
            _, gap_rows = _read_csv_rows(archive, files_meta, "gaps.csv")
            snapshots = _parse_orderbook_rows(book_rows, book_fields, path)
            _attach_trade_flow_imbalance(snapshots, trade_rows)
            gaps = _parse_gap_rows(gap_rows)
    except zipfile.BadZipFile as exc:
        raise InputError(f"ARCHIVE_NOT_VALID_ZIP:{path.name}") from exc
    return manifest, snapshots, gaps


def _window_crosses_gap(
    start_ms: int, end_ms: int, gaps: Sequence[tuple[int, int]]
) -> bool:
    return any(
        gap_start <= end_ms and gap_end >= start_ms for gap_start, gap_end in gaps
    )


EXCLUSION_REASONS = (
    "no_past_snapshot",
    "stale_feature_snapshot",
    "no_future_snapshot",
    "future_lag",
    "gap_crossing",
)


def _past_snapshot(
    timestamps: Sequence[int], snapshots: Sequence[Snapshot], grid: int
) -> tuple[Snapshot | None, str | None]:
    index = bisect.bisect_right(timestamps, grid) - 1
    if index < 0:
        return None, "no_past_snapshot"
    past = snapshots[index]
    if grid - past.timestamp_ms > STALE_MAX_MS:
        return None, "stale_feature_snapshot"
    return past, None


def _future_snapshot(
    timestamps: Sequence[int], snapshots: Sequence[Snapshot], target_time: int
) -> tuple[Snapshot | None, str | None]:
    index = bisect.bisect_left(timestamps, target_time)
    if index >= len(snapshots):
        return None, "no_future_snapshot"
    future = snapshots[index]
    if future.timestamp_ms - target_time > FUTURE_TOLERANCE_MS:
        return None, "future_lag"
    return future, None


def _candidate_sample(
    timestamps: Sequence[int],
    snapshots: Sequence[Snapshot],
    gaps: Sequence[tuple[int, int]],
    grid: int,
    horizon_ms: int,
) -> tuple[Snapshot | None, Snapshot | None, str | None]:
    past, reason = _past_snapshot(timestamps, snapshots, grid)
    if reason is not None:
        return None, None, reason
    future, reason = _future_snapshot(timestamps, snapshots, grid + horizon_ms)
    if reason is not None:
        return None, None, reason
    assert past is not None and future is not None
    if _window_crosses_gap(past.timestamp_ms, future.timestamp_ms, gaps):
        return None, None, "gap_crossing"
    return past, future, None


def _samples_for_horizon(
    snapshots: Sequence[Snapshot],
    gaps: Sequence[tuple[int, int]],
    horizon_ms: int,
) -> tuple[dict[str, list[float]], list[float], dict[str, int]]:
    timestamps = [item.timestamp_ms for item in snapshots]
    first, last = timestamps[0], timestamps[-1]
    grid = ((first + horizon_ms - 1) // horizon_ms) * horizon_ms
    pairs: dict[str, list[float]] = {name: [] for name in FEATURES}
    targets: list[float] = []
    excluded = {reason: 0 for reason in EXCLUSION_REASONS}
    while grid + horizon_ms <= last + FUTURE_TOLERANCE_MS:
        past, future, reason = _candidate_sample(
            timestamps, snapshots, gaps, grid, horizon_ms
        )
        if reason is not None:
            excluded[reason] += 1
        else:
            assert past is not None and future is not None
            target = math.log(future.mid / past.mid) * 10_000.0
            if not math.isfinite(target):
                raise InputError("NONFINITE_FORWARD_RETURN")
            targets.append(target)
            for name in FEATURES:
                pairs[name].append(past.features[name])
        grid += horizon_ms
    return pairs, targets, excluded


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 3:
        return None
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    dx = [value - mean_x for value in x]
    dy = [value - mean_y for value in y]
    denominator = math.sqrt(
        sum(value * value for value in dx) * sum(value * value for value in dy)
    )
    if denominator == 0:
        return None
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(dx, dy)) / denominator))


def _betacf(a: float, b: float, x: float) -> float:
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return h


def _regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _correlation_p_value(r: float | None, n: int) -> float | None:
    if r is None or n < 3:
        return None
    if abs(r) >= 1.0:
        return 0.0
    degrees = n - 2
    t_squared = (r * r * degrees) / (1.0 - r * r)
    x = degrees / (degrees + t_squared)
    return min(1.0, max(0.0, _regularized_beta(x, degrees / 2.0, 0.5)))


def correlation_summary(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    n = len(x)
    pearson = _pearson(x, y)
    spearman = _pearson(_rank(x), _rank(y)) if n >= 3 else None
    return {
        "n": n,
        "pearson_r": pearson,
        "pearson_p_two_sided": _correlation_p_value(pearson, n),
        "spearman_rho": spearman,
        "spearman_p_two_sided": _correlation_p_value(spearman, n),
    }


def _collect_verified_segments(
    archive_paths: Sequence[Path],
    inventory: dict[str, Any],
    expected_market: str,
) -> tuple[
    list[dict[str, Any]],
    set[str],
    list[tuple[dict[str, Any], list[Snapshot], list[tuple[int, int]]]],
]:
    segments: list[dict[str, Any]] = []
    actual_markets: set[str] = set()
    loaded: list[tuple[dict[str, Any], list[Snapshot], list[tuple[int, int]]]] = []
    seen_hashes: set[str] = set()
    for path in archive_paths:
        manifest, snapshots, gaps = _read_verified_archive(
            path, inventory, expected_market
        )
        archive_hash = _sha256(path.read_bytes())
        if archive_hash in seen_hashes:
            raise InputError("DUPLICATE_ARCHIVE")
        seen_hashes.add(archive_hash)
        actual_market = str(manifest.get("market", "UNKNOWN"))
        actual_markets.add(actual_market)
        entry = inventory["artifacts_by_sha256"][archive_hash]
        segments.append(
            {
                "run_id": manifest.get("workflow_run_id"),
                "artifact_id": entry["artifact_id"],
                "market": actual_market,
                "capture_status": manifest.get("capture_status"),
                "orderbook_rows": manifest.get("record_counts", {}).get("orderbook"),
                "gap_records": manifest.get("gap_count"),
                "archive_sha256": archive_hash,
            }
        )
        loaded.append((manifest, snapshots, gaps))
    if len(seen_hashes) != len(inventory.get("artifacts_by_sha256", {})):
        raise InputError("ARCHIVE_INVENTORY_SET_MISMATCH")
    return segments, actual_markets, loaded


def _base_report(
    expected_market: str, segments: list[dict[str, Any]], actual_markets: set[str]
) -> dict[str, Any]:
    return {
        "study": "M8_ORDER_BOOK_MICROSTRUCTURE_CORRELATION",
        "expected_market": expected_market,
        "actual_markets": sorted(actual_markets),
        "segments": segments,
        "segment_count": len(segments),
        "total_orderbook_rows": sum(
            int(item["orderbook_rows"] or 0) for item in segments
        ),
        "total_gap_records": sum(int(item["gap_records"] or 0) for item in segments),
        "features": list(FEATURES),
        "feature_definitions": {
            "obi_top5": "Top-five bid depth minus ask depth divided by total top-five depth.",
            "microprice_edge_bps": "Queue-size-weighted microprice deviation from bid/ask midpoint, in basis points.",
            "trade_flow_imbalance_60s": "Trailing-60-second (buy volume minus sell volume) divided by total volume, using WebSocket trade rows only.",
        },
        "horizons": ["5m", "15m"],
        "safety": {
            "holdout_bytes_read": False,
            "posthack_data_read": False,
            "private_api_calls": 0,
            "paper_orders": 0,
            "real_order_attempts": 0,
            "model_fitting": False,
            "trading_authorization": "NO-GO",
        },
    }


def _aggregate_segment_samples(
    loaded: list[tuple[dict[str, Any], list[Snapshot], list[tuple[int, int]]]],
) -> tuple[
    dict[str, dict[str, list[float]]],
    dict[str, list[float]],
    dict[str, dict[str, int]],
    list[dict[str, Any]],
]:
    pooled: dict[str, dict[str, list[float]]] = {
        horizon: {feature: [] for feature in FEATURES} for horizon in HORIZONS_MS
    }
    pooled_targets: dict[str, list[float]] = {horizon: [] for horizon in HORIZONS_MS}
    excluded_by_horizon = {
        horizon: {reason: 0 for reason in EXCLUSION_REASONS} for horizon in HORIZONS_MS
    }
    per_segment: list[dict[str, Any]] = []
    for manifest, snapshots, gaps in loaded:
        segment_report: dict[str, Any] = {
            "run_id": manifest.get("workflow_run_id"),
            "samples": {},
        }
        for horizon_name, horizon_ms in HORIZONS_MS.items():
            feature_pairs, targets, excluded = _samples_for_horizon(
                snapshots, gaps, horizon_ms
            )
            segment_report["samples"][horizon_name] = {
                "n": len(targets),
                "excluded": excluded,
            }
            for feature in FEATURES:
                pooled[horizon_name][feature].extend(feature_pairs[feature])
            pooled_targets[horizon_name].extend(targets)
            for reason, count in excluded.items():
                excluded_by_horizon[horizon_name][reason] += count
        per_segment.append(segment_report)
    return pooled, pooled_targets, excluded_by_horizon, per_segment


def _feature_correlation_result(
    feature_values: list[float], targets: list[float], n_tests: int
) -> dict[str, Any]:
    result = correlation_summary(feature_values, targets)
    for method, p_key in (
        ("pearson", "pearson_p_two_sided"),
        ("spearman", "spearman_p_two_sided"),
    ):
        raw_p = result[p_key]
        result[f"{method}_p_bonferroni_{n_tests}_tests"] = (
            min(1.0, raw_p * n_tests) if raw_p is not None else None
        )
        adjusted_p = result[f"{method}_p_bonferroni_{n_tests}_tests"]
        result[f"{method}_significant_after_bonferroni"] = bool(
            adjusted_p is not None and adjusted_p < ALPHA
        )
    return result


def _all_correlations(
    pooled: dict[str, dict[str, list[float]]],
    pooled_targets: dict[str, list[float]],
) -> dict[str, Any]:
    correlations: dict[str, Any] = {}
    # One family: 3 features x 2 horizons x 2 statistics = 12 tests.
    n_tests = len(FEATURES) * len(HORIZONS_MS) * 2
    for horizon_name in HORIZONS_MS:
        correlations[horizon_name] = {}
        for feature in FEATURES:
            correlations[horizon_name][feature] = _feature_correlation_result(
                pooled[horizon_name][feature], pooled_targets[horizon_name], n_tests
            )
    return correlations


def _microstructure_signal_found(correlations: dict[str, Any]) -> bool:
    for horizon in HORIZONS_MS:
        for feature in FEATURES:
            item = correlations[horizon][feature]
            if (
                item.get("pearson_significant_after_bonferroni")
                or item.get("spearman_significant_after_bonferroni")
            ):
                return True
    return False


def _blocked_market_report(
    report: dict[str, Any], actual_markets: set[str], expected_market: str
) -> dict[str, Any]:
    report.update(
        {
            "status": "BLOCKED_MARKET_MISMATCH",
            "correlation_status": "NOT_COMPUTED",
            "correlations": {},
            "decision": "NEW_BTCUSDT_CAPTURE_REQUIRED",
            "decision_note": (
                f"Input market(s) {sorted(actual_markets)} do not match {expected_market}; "
                "no cross-market inference or correlation calculation was performed."
            ),
        }
    )
    return report


def _completed_market_report(
    report: dict[str, Any],
    correlations: dict[str, Any],
    per_segment: list[dict[str, Any]],
    excluded: dict[str, dict[str, int]],
) -> dict[str, Any]:
    signal_found = _microstructure_signal_found(correlations)
    report.update(
        {
            "status": "ANALYZED",
            "correlation_status": "COMPUTED",
            "sampling": {
                "grid": "non-overlapping UTC epoch grid at each target horizon",
                "feature_snapshot_max_staleness_ms": STALE_MAX_MS,
                "future_snapshot_max_lag_ms": FUTURE_TOLERANCE_MS,
                "gap_policy": "reject any sample whose feature-to-target interval intersects an explicit orderbook gap",
                "segment_policy": "process each artifact separately; never stitch capture segments",
                "returns": "10,000 * ln(mid[t+h] / mid[t])",
                "multiple_testing": "Joint Bonferroni correction across 12 tests (3 features x 2 horizons x Pearson/Spearman), family-wise alpha=0.05.",
            },
            "per_segment": per_segment,
            "excluded_samples": excluded,
            "correlations": correlations,
            "decision": (
                "MICROSTRUCTURE_FEATURE_SIGNAL_FOUND"
                if signal_found
                else "NO_SIGNIFICANT_MICROSTRUCTURE_CORRELATION"
            ),
            "decision_note": (
                "At least one of OBI, microprice deviation, or 60-second trade-flow imbalance is significant after the joint 12-test Bonferroni correction."
                if signal_found
                else "None of OBI, microprice deviation, or 60-second trade-flow imbalance is significant after the joint 12-test Bonferroni correction."
            ),
            "caveat": "Exploratory correlation is not evidence of tradable edge, causality, or post-cost profitability. TFI uses public trade type buy/sell as a side proxy; the official schema does not explicitly define it as aggressor/taker side. REST seed rows are excluded to avoid possible duplicates. Serial dependence is not removed by Bonferroni, so effective sample size may be lower than raw N.",
        }
    )
    return report


def analyze_archives(
    archive_paths: Sequence[Path],
    inventory_path: Path = DEFAULT_INVENTORY,
    expected_market: str = "BTCUSDT",
) -> dict[str, Any]:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    expected_count = inventory.get("expected_archive_count")
    if len(archive_paths) != expected_count:
        raise InputError(f"EXPECTED_{expected_count}_ARCHIVES_GOT_{len(archive_paths)}")
    segments, actual_markets, loaded = _collect_verified_segments(
        archive_paths, inventory, expected_market
    )
    report = _base_report(expected_market, segments, actual_markets)
    if actual_markets != {expected_market}:
        return _blocked_market_report(report, actual_markets, expected_market)
    pooled, pooled_targets, excluded, per_segment = _aggregate_segment_samples(loaded)
    correlations = _all_correlations(pooled, pooled_targets)
    return _completed_market_report(report, correlations, per_segment, excluded)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the hash-pinned M8 L2 microstructure correlation diagnostic."
    )
    parser.add_argument(
        "--archives",
        nargs="+",
        type=Path,
        required=True,
        help="Hash-registered M8.2 aggregate ZIP archive(s).",
    )
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--expected-market", default="BTCUSDT")
    parser.add_argument(
        "--output", type=Path, default=Path("m8_microstructure_report.json")
    )
    args = parser.parse_args()

    report = analyze_archives(args.archives, args.inventory, args.expected_market)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] == "BLOCKED_MARKET_MISMATCH":
        print(
            "M8_PREFLIGHT_BLOCKED: BTCUSDT data required; correlations were NOT computed."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
