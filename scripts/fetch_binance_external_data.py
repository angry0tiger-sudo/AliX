"""Fetch public Binance USD-M funding and bounded historical open-interest data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from external_data.binance_futures import (
    ALLOWED_SYMBOLS,
    BinanceFuturesSnapshot,
    BinanceUsdMFuturesConnector,
    UrllibPublicJsonTransport,
    snapshot_manifest,
)


def _funding_frame(snapshot: BinanceFuturesSnapshot) -> pd.DataFrame:
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
        columns=(
            "symbol",
            "funding_time",
            "funding_rate",
            "mark_price",
            "available_at",
            "content_hash",
        ),
    )


def _open_interest_frame(snapshot: BinanceFuturesSnapshot) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": item.symbol,
                "period": item.period,
                "timestamp": item.timestamp.isoformat(),
                "sum_open_interest": item.sum_open_interest,
                "sum_open_interest_value": item.sum_open_interest_value,
                "available_at": item.available_at.isoformat(),
                "content_hash": item.content_hash,
            }
            for item in snapshot.open_interest
        ],
        columns=(
            "symbol",
            "period",
            "timestamp",
            "sum_open_interest",
            "sum_open_interest_value",
            "available_at",
            "content_hash",
        ),
    )


def _write_snapshot(
    output_dir: Path,
    snapshot: BinanceFuturesSnapshot,
    *,
    commit_sha: str,
    requested_days: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"binance-{snapshot.symbol.lower()}"
    funding_path = output_dir / f"{stem}-funding.csv"
    oi_path = output_dir / f"{stem}-open-interest.csv"
    _funding_frame(snapshot).to_csv(funding_path, index=False)
    _open_interest_frame(snapshot).to_csv(oi_path, index=False)
    manifest = snapshot_manifest(
        snapshot,
        commit_sha=commit_sha,
        requested_days=requested_days,
    )
    manifest["funding_csv"] = funding_path.name
    manifest["funding_csv_sha256"] = _sha256(funding_path)
    manifest["open_interest_csv"] = oi_path.name
    manifest["open_interest_csv_sha256"] = _sha256(oi_path)
    manifest_path = output_dir / f"{stem}-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _sha256(path: Path) -> str:
    import hashlib

    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fetch_public_history(
    *,
    symbols: tuple[str, ...],
    requested_days: int,
    commit_sha: str,
    oi_period: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Fetch the declared window without credentials and persist per-symbol manifests."""
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("ai6_symbol_selection_invalid")
    if any(symbol not in ALLOWED_SYMBOLS for symbol in symbols):
        raise ValueError("ai6_symbol_not_allowed")
    if requested_days != 1825:
        raise ValueError("ai6_requested_days_must_be_1825")
    if len(commit_sha) != 40 or any(character not in "0123456789abcdef" for character in commit_sha):
        raise ValueError("ai6_commit_sha_invalid")
    end = pd.Timestamp.now(tz="UTC").floor("min")
    start = end - pd.Timedelta(days=requested_days) + pd.Timedelta(minutes=1)
    connector = BinanceUsdMFuturesConnector(UrllibPublicJsonTransport())
    manifests: list[dict[str, Any]] = []
    for symbol in symbols:
        snapshot = connector.fetch(
            symbol,
            start=start.to_pydatetime(),
            end=end.to_pydatetime(),
            oi_period=oi_period,
        )
        manifest = _write_snapshot(
            output_dir,
            snapshot,
            commit_sha=commit_sha,
            requested_days=requested_days,
        )
        manifests.append(manifest)
        print(
            f"{symbol}: funding_rows={manifest['funding']['row_count']} "
            f"oi_rows={manifest['open_interest']['row_count']} "
            f"oi_span_days={manifest['open_interest']['observed_span_days']:.3f} "
            f"status={manifest['research_status']}"
        )
    return manifests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", choices=ALLOWED_SYMBOLS, default=list(ALLOWED_SYMBOLS))
    parser.add_argument("--days", type=int, choices=(1825,), default=1825)
    parser.add_argument("--oi-period", choices=("4h",), default="4h")
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifests = fetch_public_history(
        symbols=tuple(args.symbols),
        requested_days=args.days,
        commit_sha=args.commit_sha,
        oi_period=args.oi_period,
        output_dir=args.output_dir,
    )
    if any(item["research_status"] != "READY_FOR_FULL_WINDOW_FEATURE_JOIN" for item in manifests):
        print("AI6_STATUS=BLOCKED_INSUFFICIENT_OI_HISTORY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
