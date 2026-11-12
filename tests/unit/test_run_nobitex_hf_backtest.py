from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from scripts import fetch_nobitex_hf_ohlcv as proxy_fetcher
from scripts import run_nobitex_hf_backtest as backtest


def _write_proxy_fixture(
    tmp_path: Path, interval: str = "15m"
) -> tuple[Path, Path, str]:
    seconds = proxy_fetcher.interval_seconds(interval)
    start = (1_700_000_000 // seconds) * seconds
    count = 40
    end = start + (count - 1) * seconds
    dates = pd.date_range(
        datetime.fromtimestamp(start, timezone.utc),
        periods=count,
        freq=f"{seconds}s",
    )
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0] * count,
            "high": [101.0] * count,
            "low": [99.0] * count,
            "close": [100.5] * count,
            "volume": [1.0] * count,
        }
    )
    csv_path = tmp_path / f"PROXY_BINANCE_{interval.upper()}.csv"
    frame.to_csv(csv_path, index=False)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / f"PROXY_BINANCE_{interval.upper()}.manifest.json"
    manifest = {
        "dataset_id": f"PROXY_BINANCE_{interval.upper()}",
        "dataset_role": "RESEARCH_PROXY_ONLY",
        "provider": "Binance Spot public market-data REST API",
        "endpoint": proxy_fetcher.BINANCE_KLINES_ENDPOINT,
        "venue": "BINANCE_SPOT",
        "market_type": "SPOT",
        "symbol": "BTCUSDT",
        "execution_venue": "NOBITEX",
        "interval": interval,
        "native_interval": True,
        "dataset_window": "2021_11_TO_2025_06_17_PROXY",
        "source_policy": "BINANCE_PROXY_RESEARCH_ONLY",
        "proxy_not_execution_truth": True,
        "start_timestamp": start,
        "end_timestamp": end,
        "requested_start_timestamp": start,
        "requested_end_timestamp": end,
        "start_utc": dates[0].isoformat().replace("+00:00", "Z"),
        "end_utc": dates[-1].isoformat().replace("+00:00", "Z"),
        "row_count": count,
        "expected_candles": count,
        "missing_intervals": 0,
        "csv_sha256": digest,
        "commit_sha": "a" * 40,
        "public_api_only": True,
        "market_data_only_endpoint": True,
        "public_api_requests": 1,
        "credentials_used": False,
        "private_api_calls": 0,
        "real_order_attempts": 0,
        "Champion": None,
        "order_permission": False,
        "trading_authorization": "NO-GO",
        "no_resampling": True,
        "no_interpolation": True,
        "not_nobitex_backfill": True,
        "data_authority": "THIRD_PARTY_PUBLIC_MARKET_DATA_NOT_NOBITEX_EXECUTION_TRUTH",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return csv_path, manifest_path, start.__str__()


def test_proxy_manifest_range_requires_exact_registered_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seconds = 900
    start = (1_700_000_000 // seconds) * seconds
    end = start + 40 * seconds
    monkeypatch.setattr(backtest, "PROXY_START_TIMESTAMP", start)
    monkeypatch.setattr(backtest, "PROXY_END_TIMESTAMP", end)
    manifest = {"start_timestamp": start, "end_timestamp": end}
    assert backtest._intraday_manifest_range(manifest, "15m") == (start, end, 41)
    with pytest.raises(ValueError, match="proxy_manifest_range_invalid"):
        backtest._intraday_manifest_range(
            {"start_timestamp": start + seconds, "end_timestamp": end}, "15m"
        )


def test_backtest_loads_only_proxy_binance_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path, manifest_path, start_string = _write_proxy_fixture(tmp_path)
    start = int(start_string)
    end = start + 39 * 900
    monkeypatch.setattr(backtest, "PROXY_START_TIMESTAMP", start)
    monkeypatch.setattr(backtest, "PROXY_END_TIMESTAMP", end)
    frame, manifest = backtest._load_intraday_dataset(
        csv_path, manifest_path, "15m", "a" * 40
    )
    assert len(frame) == 40
    assert manifest["dataset_id"] == "PROXY_BINANCE_15M"
    assert frame["date"].iloc[0] == pd.Timestamp(manifest["start_utc"])


def test_backtest_rejects_nobitex_or_mislabeled_proxy_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path, manifest_path, start_string = _write_proxy_fixture(tmp_path)
    start = int(start_string)
    end = start + 39 * 900
    monkeypatch.setattr(backtest, "PROXY_START_TIMESTAMP", start)
    monkeypatch.setattr(backtest, "PROXY_END_TIMESTAMP", end)
    manifest = json.loads(manifest_path.read_text())
    manifest["dataset_id"] = "NOBITEX_NATIVE_15M"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest_mismatch:15m:dataset_id"):
        backtest._load_intraday_dataset(csv_path, manifest_path, "15m", "a" * 40)


def test_backtest_rejects_proxy_manifest_without_public_api_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path, manifest_path, start_string = _write_proxy_fixture(tmp_path)
    start = int(start_string)
    end = start + 39 * 900
    monkeypatch.setattr(backtest, "PROXY_START_TIMESTAMP", start)
    monkeypatch.setattr(backtest, "PROXY_END_TIMESTAMP", end)
    manifest = json.loads(manifest_path.read_text())
    manifest["public_api_requests"] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="public_api_requests_invalid:15m"):
        backtest._load_intraday_dataset(csv_path, manifest_path, "15m", "a" * 40)


def test_backtest_rejects_corrupted_proxy_csv_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path, manifest_path, start_string = _write_proxy_fixture(tmp_path)
    start = int(start_string)
    end = start + 39 * 900
    monkeypatch.setattr(backtest, "PROXY_START_TIMESTAMP", start)
    monkeypatch.setattr(backtest, "PROXY_END_TIMESTAMP", end)
    csv_path.write_text(csv_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(
        ValueError, match="hf_intraday_manifest_mismatch:15m:csv_sha256"
    ):
        backtest._load_intraday_dataset(csv_path, manifest_path, "15m", "a" * 40)
