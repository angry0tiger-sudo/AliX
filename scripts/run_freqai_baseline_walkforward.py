"""Run deterministic offline walk-forward evaluation for the Phase 4 baseline.

No network calls, exchange clients, order writes, or notification integrations live here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from services.freqai_baseline.walkforward import WalkForwardConfig, run_walk_forward

TIMEFRAME_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "1d": 86400,
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--synthetic",
        action="store_true",
        help="use seeded synthetic OHLCV (mechanics validation only, not economic evidence)",
    )
    source.add_argument(
        "--input",
        type=Path,
        help="owner-supplied historical OHLCV CSV with timezone-aware date and OHLCV columns",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="path to JSON report"
    )
    parser.add_argument("--pair", default="SYNTHETIC/USDT")
    parser.add_argument(
        "--dataset-kind",
        choices=("owner_supplied_historical", "public_historical_market_data"),
        default="owner_supplied_historical",
        help="provenance label for --input",
    )
    parser.add_argument("--timeframe", choices=sorted(TIMEFRAME_SECONDS), default="1h")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--min-train-size", type=int, default=300)
    parser.add_argument(
        "--fee-bps", type=float, default=10.0, help="fee per side in basis points"
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=5.0,
        help="slippage per side in basis points",
    )
    parser.add_argument(
        "--commit-sha",
        default=os.environ.get("GITHUB_SHA", "UNKNOWN"),
        help="source commit recorded in the report",
    )
    return parser.parse_args()


def _synthetic_ohlcv(rows: int = 1_800, seed: int = 20261022) -> DataFrame:
    if rows < 400:
        raise ValueError("synthetic dataset must contain at least 400 candles")
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01T00:00:00Z", periods=rows, freq="h")
    close = np.empty(rows, dtype=float)
    opened = np.empty(rows, dtype=float)
    high = np.empty(rows, dtype=float)
    low = np.empty(rows, dtype=float)
    volume = np.empty(rows, dtype=float)
    close[0] = 100.0
    opened[0] = 100.0
    for index in range(rows):
        if index > 0:
            opened[index] = close[index - 1]
            regime_vol = 0.0025 if (index // 240) % 2 == 0 else 0.009
            drift = 0.00005 if (index // 360) % 2 == 0 else -0.00003
            close[index] = opened[index] * np.exp(
                drift + regime_vol * rng.standard_normal()
            )
        candle_spread = abs(rng.normal(0.002, 0.001))
        high[index] = max(opened[index], close[index]) * (1.0 + candle_spread)
        low[index] = min(opened[index], close[index]) * max(0.01, 1.0 - candle_spread)
        volume[index] = float(rng.lognormal(mean=4.0, sigma=0.5))
    return DataFrame(
        {
            "date": dates,
            "open": opened,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _load_owner_csv(path: Path) -> tuple[DataFrame, str]:
    raw_bytes = path.read_bytes()
    frame = pd.read_csv(path)
    if "date" not in frame:
        raise ValueError("input CSV must contain a date column")
    parsed: list[pd.Timestamp] = []
    for row_number, value in enumerate(frame["date"], start=2):
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid date timestamp on CSV line {row_number}"
            ) from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError(
                f"naive date timestamp on CSV line {row_number}; timezone/offset is required"
            )
        parsed.append(timestamp)
    frame["date"] = pd.to_datetime(parsed, utc=True)
    return frame, hashlib.sha256(raw_bytes).hexdigest()


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main() -> int:
    args = _arguments()
    timeframe_seconds = TIMEFRAME_SECONDS[args.timeframe]
    config = WalkForwardConfig(
        horizon=args.horizon,
        n_splits=args.folds,
        min_train_size=args.min_train_size,
        fee_bps_per_side=args.fee_bps,
        slippage_bps_per_side=args.slippage_bps,
    )
    if args.synthetic:
        dataframe = _synthetic_ohlcv()
        dataset_kind = "synthetic"
        serialized = dataframe.to_csv(
            index=False, date_format="%Y-%m-%dT%H:%M:%S%z"
        ).encode()
        dataset_sha256 = hashlib.sha256(serialized).hexdigest()
        pair = "SYNTHETIC/USDT"
    else:
        if args.input is None:
            raise ValueError("input path is required unless --synthetic is selected")
        dataframe, dataset_sha256 = _load_owner_csv(args.input)
        dataset_kind = args.dataset_kind
        pair = args.pair

    report = run_walk_forward(
        dataframe,
        pair=pair,
        timeframe=args.timeframe,
        timeframe_seconds=timeframe_seconds,
        config=config,
        dataset_kind=dataset_kind,
        dataset_sha256=dataset_sha256,
        commit_sha=args.commit_sha,
    )
    _write_report(args.output, report)
    economic = report["out_of_sample"]["costed_fixed_horizon_research_rule"]
    summary = {
        "report": str(args.output),
        "evaluation_status": report["evaluation_status"],
        "dataset_sha256": report["dataset"]["sha256"],
        "oos_samples": report["out_of_sample"]["sample_count"],
        "net_return": economic["net_return"],
        "max_drawdown": economic["max_drawdown"],
        "trade_count": economic["trade_count"],
        "champion": report["champion"],
        "promotion_status": report["promotion_status"],
    }
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
