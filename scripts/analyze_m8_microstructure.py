"""Fail-closed M8 order-book microstructure correlation study.

The evaluator accepts only the exact, hash-pinned M1.6 artifact inventory.
Each capture segment is analysed independently. It never joins segment boundaries,
interpolates gaps, reads Holdout/Post-Hack, performs model fitting, or places orders.

Expected raw CSV schema is the schema emitted by scripts.capture_nobitex_microstructure.
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
from typing import Any, Sequence

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
FEATURES = ("obi_top5", "microprice_edge_bps", "spread_bps", "depth_top5")
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


def _levels(raw: str, field: str) -> list[tuple[float, float]]:
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InputError(f"{field}_JSON_INVALID") from exc
    if not isinstance(values, list) or len(values) < 5:
        raise InputError(f"{field}_MUST_HAVE_AT_LEAST_FIVE_LEVELS")
    parsed: list[tuple[float, float]] = []
    for item in values[:5]:
        if not isinstance(item, list) or len(item) != 2:
            raise InputError(f"{field}_LEVEL_INVALID")
        price, size = float(item[0]), float(item[1])
        if not (
            math.isfinite(price) and math.isfinite(size) and price > 0 and size >= 0
        ):
            raise InputError(f"{field}_LEVEL_OUT_OF_RANGE")
        parsed.append((price, size))
    return parsed


def compute_features(row: dict[str, str]) -> tuple[float, dict[str, float]]:
    """Compute top-five OBI, microprice edge, relative spread, and depth."""
    try:
        bid = float(row["best_bid_price"])
        ask = float(row["best_ask_price"])
        bid_size = float(row["best_bid_size"])
        ask_size = float(row["best_ask_size"])
        mid = float(row["mid_price"])
        spread = float(row["spread"])
    except (KeyError, ValueError) as exc:
        raise InputError("TOP_OF_BOOK_NUMERIC_FIELD_INVALID") from exc

    if not all(math.isfinite(x) for x in (bid, ask, bid_size, ask_size, mid, spread)):
        raise InputError("TOP_OF_BOOK_NONFINITE")
    if bid <= 0 or ask <= 0 or mid <= 0 or ask < bid or bid_size < 0 or ask_size < 0:
        raise InputError("TOP_OF_BOOK_OUT_OF_RANGE")

    bids = _levels(row["bids_top_json"], "bids_top_json")
    asks = _levels(row["asks_top_json"], "asks_top_json")
    bid_depth = sum(size for _, size in bids)
    ask_depth = sum(size for _, size in asks)
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


def _read_verified_archive(
    path: Path, inventory: dict[str, Any], expected_market: str
) -> tuple[dict[str, Any], list[Snapshot], list[tuple[int, int]]]:
    raw_archive = path.read_bytes()
    archive_sha = _sha256(raw_archive)
    entry = inventory.get("artifacts_by_sha256", {}).get(archive_sha)
    if not isinstance(entry, dict):
        raise InputError(f"UNREGISTERED_ARCHIVE_SHA256:{path.name}")

    try:
        archive = zipfile.ZipFile(io.BytesIO(raw_archive))
    except zipfile.BadZipFile as exc:
        raise InputError(f"ARCHIVE_NOT_VALID_ZIP:{path.name}") from exc

    required = {
        "manifest.json",
        "manifest.sha256",
        "orderbook.csv",
        "trades.csv",
        "gaps.csv",
    }
    if set(archive.namelist()) != required:
        raise InputError(f"ARCHIVE_FILE_SET_MISMATCH:{path.name}")

    manifest_bytes = archive.read("manifest.json")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    checksum_line = archive.read("manifest.sha256").decode("utf-8").strip().split()
    if checksum_line != [_sha256(manifest_bytes), "manifest.json"]:
        raise InputError(f"MANIFEST_SHA256_MISMATCH:{path.name}")

    if manifest.get("market") != entry.get("market"):
        raise InputError(f"MANIFEST_MARKET_INVENTORY_MISMATCH:{path.name}")
    try:
        run_id_matches = int(manifest.get("workflow_run_id")) == int(
            entry.get("run_id")
        )
    except (TypeError, ValueError):
        run_id_matches = False
    if not run_id_matches:
        raise InputError(f"MANIFEST_RUN_ID_MISMATCH:{path.name}")

    if manifest.get("market") != expected_market:
        return manifest, [], []

    files_meta = manifest.get("files")
    if not isinstance(files_meta, dict):
        raise InputError(f"MANIFEST_FILES_MISSING:{path.name}")

    snapshots: list[Snapshot] = []
    gaps: list[tuple[int, int]] = []
    for filename in ("orderbook.csv", "trades.csv", "gaps.csv"):
        contents = archive.read(filename)
        meta = files_meta.get(filename)
        if not isinstance(meta, dict):
            raise InputError(f"MANIFEST_FILE_METADATA_MISSING:{filename}")
        if meta.get("sha256") != _sha256(contents) or meta.get("bytes") != len(
            contents
        ):
            raise InputError(f"CSV_INTEGRITY_MISMATCH:{filename}")
        with io.TextIOWrapper(
            io.BytesIO(contents), encoding="utf-8", newline=""
        ) as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            if meta.get("records") != len(rows):
                raise InputError(f"CSV_RECORD_COUNT_MISMATCH:{filename}")
            if filename == "orderbook.csv":
                if not reader.fieldnames or not BOOK_FIELDS.issubset(
                    set(reader.fieldnames)
                ):
                    raise InputError("ORDERBOOK_SCHEMA_MISMATCH")
                for row in rows:
                    timestamp = int(row["exchange_timestamp_ms"])
                    mid, features = compute_features(row)
                    snapshots.append(Snapshot(timestamp, mid, features))
            elif filename == "gaps.csv":
                for row in rows:
                    if row.get("stream") != "orderbook":
                        continue
                    start = (
                        int(row["gap_start_ms"])
                        if row.get("gap_start_ms")
                        else _parse_utc_ms(row["gap_start_utc"])
                    )
                    end = (
                        int(row["gap_end_ms"])
                        if row.get("gap_end_ms")
                        else _parse_utc_ms(row["gap_end_utc"])
                    )
                    if end < start:
                        raise InputError("GAP_INTERVAL_REVERSED")
                    gaps.append((start, end))

    if not snapshots:
        raise InputError(f"ORDERBOOK_EMPTY:{path.name}")
    timestamps = [item.timestamp_ms for item in snapshots]
    if timestamps != sorted(timestamps) or len(set(timestamps)) != len(timestamps):
        raise InputError(f"ORDERBOOK_TIMESTAMPS_NOT_STRICTLY_INCREASING:{path.name}")
    return manifest, snapshots, gaps


def _window_crosses_gap(
    start_ms: int, end_ms: int, gaps: Sequence[tuple[int, int]]
) -> bool:
    return any(
        gap_start <= end_ms and gap_end >= start_ms for gap_start, gap_end in gaps
    )


def _samples_for_horizon(
    snapshots: Sequence[Snapshot],
    gaps: Sequence[tuple[int, int]],
    horizon_ms: int,
) -> tuple[dict[str, list[float]], list[float], dict[str, int]]:
    timestamps = [item.timestamp_ms for item in snapshots]
    first = timestamps[0]
    last = timestamps[-1]
    grid = ((first + horizon_ms - 1) // horizon_ms) * horizon_ms
    pairs = {name: [] for name in FEATURES}
    targets: list[float] = []
    excluded = {
        "no_past_snapshot": 0,
        "stale_feature_snapshot": 0,
        "no_future_snapshot": 0,
        "future_lag": 0,
        "gap_crossing": 0,
    }

    while grid + horizon_ms <= last + FUTURE_TOLERANCE_MS:
        past_index = bisect.bisect_right(timestamps, grid) - 1
        if past_index < 0:
            excluded["no_past_snapshot"] += 1
            grid += horizon_ms
            continue
        past = snapshots[past_index]
        if grid - past.timestamp_ms > STALE_MAX_MS:
            excluded["stale_feature_snapshot"] += 1
            grid += horizon_ms
            continue

        target_time = grid + horizon_ms
        future_index = bisect.bisect_left(timestamps, target_time)
        if future_index >= len(snapshots):
            excluded["no_future_snapshot"] += 1
            grid += horizon_ms
            continue
        future = snapshots[future_index]
        if future.timestamp_ms - target_time > FUTURE_TOLERANCE_MS:
            excluded["future_lag"] += 1
            grid += horizon_ms
            continue
        if _window_crosses_gap(past.timestamp_ms, future.timestamp_ms, gaps):
            excluded["gap_crossing"] += 1
            grid += horizon_ms
            continue

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


def analyze_archives(
    archive_paths: Sequence[Path],
    inventory_path: Path = DEFAULT_INVENTORY,
    expected_market: str = "BTCUSDT",
) -> dict[str, Any]:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    expected = inventory.get("expected_archive_count")
    if len(archive_paths) != expected:
        raise InputError(f"EXPECTED_{expected}_ARCHIVES_GOT_{len(archive_paths)}")

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
        segments.append(
            {
                "run_id": manifest.get("workflow_run_id"),
                "artifact_id": inventory["artifacts_by_sha256"][archive_hash][
                    "artifact_id"
                ],
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

    base_report: dict[str, Any] = {
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

    if actual_markets != {expected_market}:
        base_report.update(
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
        return base_report

    pooled: dict[str, dict[str, list[float]]] = {
        horizon: {feature: [] for feature in FEATURES} for horizon in HORIZONS_MS
    }
    pooled_targets: dict[str, list[float]] = {horizon: [] for horizon in HORIZONS_MS}
    excluded_by_horizon: dict[str, dict[str, int]] = {}
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
            if horizon_name not in excluded_by_horizon:
                excluded_by_horizon[horizon_name] = {key: 0 for key in excluded}
            for key, value in excluded.items():
                excluded_by_horizon[horizon_name][key] += value
        per_segment.append(segment_report)

    correlations: dict[str, Any] = {}
    n_tests = len(FEATURES) * len(HORIZONS_MS)
    for horizon_name in HORIZONS_MS:
        correlations[horizon_name] = {}
        for feature in FEATURES:
            result = correlation_summary(
                pooled[horizon_name][feature], pooled_targets[horizon_name]
            )
            for method, p_key in (
                ("pearson", "pearson_p_two_sided"),
                ("spearman", "spearman_p_two_sided"),
            ):
                raw_p = result[p_key]
                result[f"{method}_p_bonferroni_8_tests"] = (
                    min(1.0, raw_p * n_tests) if raw_p is not None else None
                )
            adj_p = result["pearson_p_bonferroni_8_tests"]
            result["pearson_significant_after_bonferroni"] = bool(
                adj_p is not None and adj_p < ALPHA
            )
            correlations[horizon_name][feature] = result

    obi_results = {
        horizon: correlations[horizon]["obi_top5"] for horizon in HORIZONS_MS
    }
    any_obi_significant = any(
        item["pearson_significant_after_bonferroni"]
        or (
            item["spearman_p_bonferroni_8_tests"] is not None
            and item["spearman_p_bonferroni_8_tests"] < ALPHA
        )
        for item in obi_results.values()
    )
    base_report.update(
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
                "multiple_testing": f"Bonferroni correction across {n_tests} feature-horizon tests separately for Pearson and Spearman",
            },
            "per_segment": per_segment,
            "excluded_samples": excluded_by_horizon,
            "correlations": correlations,
            "decision": (
                "OBI_CORRELATION_SIGNAL_FOUND"
                if any_obi_significant
                else "NO_SIGNIFICANT_OBI_CORRELATION"
            ),
            "decision_note": (
                "At least one OBI horizon is significant after the family-wise Bonferroni correction."
                if any_obi_significant
                else "Neither OBI horizon is significant after the family-wise Bonferroni correction."
            ),
            "caveat": "Exploratory correlation is not evidence of tradable edge, causality, or post-cost profitability.",
        }
    )
    return base_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the hash-pinned M8 L2 microstructure correlation diagnostic."
    )
    parser.add_argument(
        "--archives",
        nargs="+",
        type=Path,
        required=True,
        help="Exactly the five M1.6 ZIP archives.",
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
