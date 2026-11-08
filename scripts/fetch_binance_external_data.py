"""Fetch only public Binance USD-M Funding Rate history for AI-6."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from external_data.binance_futures import (
    ALLOWED_SYMBOLS,
    BinanceFundingSnapshot,
    BinancePublicHttpsTransport,
    BinanceUsdMFuturesConnector,
    snapshot_manifest,
)

FUNDING_COLUMNS = (
    "symbol",
    "funding_time",
    "funding_rate",
    "mark_price",
    "available_at",
    "content_hash",
)


def _funding_frame(snapshot: BinanceFundingSnapshot) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": item.symbol,
                "funding_time": item.funding_time.isoformat(),
                "funding_rate": item.funding_rate,
                "mark_price": item.mark_price,
                "available_at": item.available_at.isoformat(),
                "content_hash": item.content_hash,
            }
            for item in snapshot.funding_rates
        ],
        columns=FUNDING_COLUMNS,
    )


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _write_snapshot(
    output_dir: Path,
    snapshot: BinanceFundingSnapshot,
    *,
    commit_sha: str,
    requested_days: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"binance-{snapshot.symbol.lower()}"
    funding_path = output_dir / f"{stem}-funding.csv"
    _funding_frame(snapshot).to_csv(funding_path, index=False)
    manifest = snapshot_manifest(
        snapshot,
        commit_sha=commit_sha,
        requested_days=requested_days,
    )
    manifest["funding_csv"] = funding_path.name
    manifest["funding_csv_sha256"] = _sha256(funding_path)
    manifest_path = output_dir / f"{stem}-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _validate_fetch_request(
    symbols: tuple[str, ...],
    requested_days: int,
    commit_sha: str,
    start_utc: str,
    end_utc: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("ai6_symbol_selection_invalid")
    if any(symbol not in ALLOWED_SYMBOLS for symbol in symbols):
        raise ValueError("ai6_symbol_not_allowed")
    if requested_days != 1825:
        raise ValueError("ai6_requested_days_must_be_1825")
    if len(commit_sha) != 40 or any(
        character not in "0123456789abcdef" for character in commit_sha
    ):
        raise ValueError("ai6_commit_sha_invalid")
    start = pd.Timestamp(start_utc)
    end = pd.Timestamp(end_utc)
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("ai6_funding_window_invalid")
    expected = pd.Timedelta(days=requested_days) - pd.Timedelta(minutes=1)
    if end - start != expected or start.second or end.second:
        raise ValueError("ai6_funding_window_must_match_1825_days")
    return start.tz_convert("UTC"), end.tz_convert("UTC")


def fetch_public_history(
    *,
    symbols: tuple[str, ...],
    requested_days: int,
    commit_sha: str,
    start_utc: str,
    end_utc: str,
    output_dir: Path,
    connector: BinanceUsdMFuturesConnector | None = None,
) -> list[dict[str, Any]]:
    """Fetch the same exact 1825-day window used by the Spot baseline manifests."""
    start, end = _validate_fetch_request(
        symbols, requested_days, commit_sha, start_utc, end_utc
    )
    active_connector = connector or BinanceUsdMFuturesConnector(
        BinancePublicHttpsTransport()
    )
    manifests: list[dict[str, Any]] = []
    for symbol in symbols:
        snapshot = active_connector.fetch(
            symbol,
            start=start.to_pydatetime(),
            end=end.to_pydatetime(),
        )
        manifest = _write_snapshot(
            output_dir,
            snapshot,
            commit_sha=commit_sha,
            requested_days=requested_days,
        )
        manifests.append(manifest)
        funding = manifest["funding"]
        print(
            f"{symbol}: funding_rows={funding['row_count']} "
            f"span_days={funding['observed_span_days']:.3f} "
            f"max_gap_hours={funding['max_event_gap_hours']:.2f} "
            f"status={manifest['research_status']}"
        )
    return manifests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols", nargs="+", choices=ALLOWED_SYMBOLS, default=list(ALLOWED_SYMBOLS)
    )
    parser.add_argument("--days", type=int, choices=(1825,), default=1825)
    parser.add_argument("--start-utc", required=True)
    parser.add_argument("--end-utc", required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifests = fetch_public_history(
        symbols=tuple(args.symbols),
        requested_days=args.days,
        commit_sha=args.commit_sha,
        start_utc=args.start_utc,
        end_utc=args.end_utc,
        output_dir=args.output_dir,
    )
    blocked = any(
        item["research_status"] != "READY_FOR_FUNDING_FEATURE_JOIN"
        for item in manifests
    )
    print(
        "AI6_STATUS="
        + (
            "BLOCKED_INSUFFICIENT_FUNDING_HISTORY"
            if blocked
            else "READY_FOR_FUNDING_ONLY_SWEEP"
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
